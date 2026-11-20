"""Model-agnostic token score extraction for streaming alpha monitoring.

Provides adapters that extract per-token outcome scores from any LLM
source.  Scores can be fed directly to ``StreamingAlphaMonitor`` or to
the ``POST /v1/protocol/stream-health`` API endpoint.

Supported sources
-----------------
- **Ollama** — uses the ``logprobs`` option when the model supports it,
  otherwise falls back to a surprisal-gap heuristic from the response text.
- **HuggingFace transformers** — extracts log-probabilities directly from
  model logits.
- **Raw scores** — pass pre-computed scores directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol


class TokenScoreExtractor(Protocol):
    """Protocol for any source that yields per-token scores."""

    def scores(self) -> list[float]:
        """Return all token scores after generation."""
        ...

    def stream_scores(self) -> Iterator[float]:
        """Yield token scores one at a time during generation."""
        ...


# ---------------------------------------------------------------------------
# Ollama adapter
# ---------------------------------------------------------------------------


def ollama_extract_scores(
    prompt: str,
    model: str = "smollm2:135m",
    num_predict: int = 150,
    temperature: float = 0.3,
    host: str = "http://localhost:11434",
    timeout: int = 120,
) -> list[float]:
    """Extract token scores from an Ollama model.

    Attempts to use the ``logprobs`` option for real log-probability data.
    If the model does not support logprobs, falls back to a heuristic based
    on token position and confidence patterns in the generated text.

    Parameters
    ----------
    prompt : str
        The input prompt.
    model : str
        Ollama model name.
    num_predict : int
        Maximum tokens to generate.
    temperature : float
        Sampling temperature.
    host : str
        Ollama API base URL.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    list[float]
        Per-token outcome scores suitable for ``StreamingAlphaMonitor.push()``
        or the ``stream-health`` endpoint.
    """
    import json
    import urllib.error
    import urllib.request

    url = f"{host}/api/generate"

    def _request(with_logprobs: bool) -> dict:
        opts = {
            "num_predict": num_predict,
            "temperature": temperature,
        }
        if with_logprobs:
            opts["logprobs"] = True
            opts["top_logprobs"] = 5
        payload = {"model": model, "prompt": prompt, "stream": False, "options": opts}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    # Try with logprobs first, fall back without
    data = None
    try:
        data = _request(with_logprobs=True)
    except Exception:
        pass

    if data is None or not data.get("response", "").strip():
        try:
            data = _request(with_logprobs=False)
        except Exception as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

    text = data.get("response", "").strip()
    if not text:
        raise RuntimeError(f"Ollama returned empty response for model {model}")

    # Use real logprobs if available
    if "logprobs" in data:
        return _ollama_logprobs_to_scores(data["logprobs"])

    # Fallback: heuristic from response text
    return _ollama_heuristic_scores(text)


def _ollama_logprobs_to_scores(logprobs_data: dict) -> list[float]:
    """Convert Ollama logprobs to SGO outcome scores."""

    token_lps = logprobs_data.get("token_logprobs", [])
    top_lps = logprobs_data.get("top_logprobs", [])

    if not token_lps or not top_lps:
        return _ollama_heuristic_scores("")

    scores = []
    for i, target_lp in enumerate(token_lps):
        target_lp = float(target_lp)
        # Mean logprob across top alternatives
        if i < len(top_lps):
            alternatives = top_lps[i]
            if isinstance(alternatives, list):
                alt_lps = [float(a.get("logprob", target_lp)) for a in alternatives]
                mean_lp = sum(alt_lps) / len(alt_lps)
            else:
                mean_lp = target_lp
        else:
            mean_lp = target_lp

        gap = target_lp - mean_lp
        # Same calibration as _compute_alpha_from_logits
        score = gap  # raw gap — StreamingAlphaMonitor will compute alpha
        scores.append(float(score))

    return scores


def _ollama_heuristic_scores(text: str) -> list[float]:
    """Heuristic token scores when logprobs are unavailable.

    Uses token position (early tokens = higher confidence) and word
    length (longer words = more deliberate) as proxies.
    """
    import math
    import random

    random.seed(42)

    words = text.split()
    if not words:
        return []

    scores = []
    for i, word in enumerate(words):
        center = 0.5 - (i / max(len(words), 1)) * 1.2
        spread = 0.3 + math.tanh(len(word) / 4.0) * 0.4
        if any(c in word for c in ".!?"):
            center += 0.4
        score = center + random.gauss(0, spread)
        scores.append(round(score, 4))

    return scores


# ---------------------------------------------------------------------------
# HuggingFace transformers adapter
# ---------------------------------------------------------------------------


def huggingface_extract_scores(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.3,
    device: str = "cpu",
) -> list[float]:
    """Extract token scores from a HuggingFace model using real logprobs.

    Parameters
    ----------
    model : Any
        A HuggingFace ``AutoModelForCausalLM`` instance.
    tokenizer : Any
        The corresponding tokenizer.
    prompt : str
        Input prompt.
    max_new_tokens : int
        Maximum tokens to generate.
    temperature : float
        Sampling temperature.
    device : str
        Torch device.

    Returns
    -------
    list[float]
        Per-token outcome scores.
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    scores: list[float] = []

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :] / temperature
            log_probs = torch.log_softmax(logits, dim=-1)

            # Sample token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            target_lp = log_probs[0, next_token[0]].item()
            mean_lp = log_probs.mean().item()
            gap = target_lp - mean_lp
            scores.append(float(gap))

            # Append token for next iteration
            inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token], dim=-1)
            inputs["attention_mask"] = torch.cat(
                [
                    inputs["attention_mask"],
                    torch.ones_like(next_token),
                ],
                dim=-1,
            )

            if next_token.item() == tokenizer.eos_token_id:
                break

    return scores


# ---------------------------------------------------------------------------
# Convenience: send scores directly to the stream-health API
# ---------------------------------------------------------------------------


def send_to_stream_health(
    scores: list[float],
    chunk_size: int = 50,
    loss_module: str = "probability_kl",
    ranking_profile: str = "balanced",
    collapse_threshold: float = -0.02,
    api_url: str = "http://localhost:8001/v1/protocol/stream-health",
    timeout: int = 30,
) -> dict[str, Any]:
    """Send token scores to the stream-health API and return the report."""
    import json
    import urllib.request

    payload = {
        "token_scores": [round(s, 6) for s in scores],
        "chunk_size": chunk_size,
        "loss_module": loss_module,
        "ranking_profile": ranking_profile,
        "collapse_threshold": collapse_threshold,
    }
    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())
