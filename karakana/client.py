"""Python SDK for the karakana Cloud API.

Usage::

    from karakana.client import KarakanaClient

    client = KarakanaClient("http://localhost:8001")
    client.login("user@example.com", "password")

    # Structural evaluation
    result = client.evaluate([10.0, 5.0], [-2.0, -1.0])

    # Financial RAG
    resp = client.financial_generate("What is Sharpe ratio?")

    # Streaming alpha
    report = client.stream_health([0.5, 0.4, 0.3, 0.2, 0.1])
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class KarakanaClient:
    """Client for the supported public karakana Cloud REST API surface."""

    def __init__(self, base_url: str = "http://localhost:8001", api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self._token: str | None = None
        self._api_key: str | None = api_key

    # ------------------------------------------------------------------
    # Auth & API Keys
    # ------------------------------------------------------------------

    def login(self, email: str, password: str) -> str:
        """Authenticate and store the JWT token."""
        body = self._post_form(
            "/token",
            {
                "username": email,
                "password": password,
            },
        )
        self._token = body["access_token"]
        self._api_key = None
        return self._token

    def register(self, email: str, password: str) -> dict[str, Any]:
        """POST /register — support quick trial registrations."""
        return self._post("/register", {"email": email, "password": password})

    def create_api_key(
        self, name: str, expires_in_days: int | None = None, scopes: list[str] | None = None
    ) -> dict[str, Any]:
        """POST /v1/api-keys — programmatic key provisioning."""
        payload: dict[str, Any] = {"name": name}
        if expires_in_days:
            payload["expires_in_days"] = expires_in_days
        if scopes:
            payload["scopes"] = scopes
        return self._post("/v1/api-keys", payload)

    def list_api_keys(self) -> list[dict[str, Any]]:
        """GET /v1/api-keys — list active keys."""
        return self._get("/v1/api-keys")

    def delete_api_key(self, prefix: str) -> dict[str, Any]:
        """DELETE /v1/api-keys/{prefix} — revoke a key."""
        return self._request("DELETE", f"/v1/api-keys/{prefix}")

    def set_api_key(self, api_key: str) -> None:
        """Use an API key for subsequent requests."""
        self._api_key = api_key

    def clear_api_key(self) -> None:
        """Stop sending the configured API key."""
        self._api_key = None

    @property
    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["X-API-Key"] = self._api_key
        elif self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    # ------------------------------------------------------------------
    # Health & Usage
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """GET / — service health check."""
        return self._get("/")

    def model_status(self) -> dict[str, Any]:
        """GET /v1/models/status — ML model load status."""
        return self._get("/v1/models/status")

    def get_usage_stats(self) -> dict[str, Any]:
        """GET /v1/usage/stats — tier-based usage telemetry."""
        return self._get("/v1/usage/stats")

    def get_usage_incidents(self) -> list[dict[str, Any]]:
        """GET /v1/usage/incidents — active system incidents."""
        return self._get("/v1/usage/incidents")

    def get_dashboard_overview(self) -> dict[str, Any]:
        """GET /v1/dashboard/overview — dashboard summary with radar and incidents."""
        return self._get("/v1/dashboard/overview")

    # ------------------------------------------------------------------
    # Structural Evaluation & Formulas
    # ------------------------------------------------------------------

    def evaluate(
        self,
        positive_outcomes: list[float],
        negative_outcomes: list[float],
        *,
        utility_power: float = 1.1,
        ranking_profile: str = "balanced",
        regime_aware: bool = False,
        positive_alpha_weight: float | None = None,
    ) -> dict[str, Any]:
        """POST /evaluate — structural geometry audit."""
        payload: dict[str, Any] = {
            "positive_outcomes": positive_outcomes,
            "negative_outcomes": negative_outcomes,
            "utility_power": utility_power,
            "ranking_profile": ranking_profile,
            "regime_aware": regime_aware,
        }
        if positive_alpha_weight is not None:
            payload["positive_alpha_weight"] = positive_alpha_weight
        return self._post("/evaluate", payload)

    def get_evaluations(self) -> list[dict[str, Any]]:
        """GET /evaluations — list past evaluations."""
        return self._get("/evaluations")

    def get_evaluation(self, evaluation_id: str) -> dict[str, Any]:
        """GET /evaluations/{id} — get specific evaluation."""
        return self._get(f"/evaluations/{evaluation_id}")

    def delete_evaluation(self, evaluation_id: str) -> dict[str, Any]:
        """DELETE /evaluations/{id} — remove evaluation."""
        return self._request("DELETE", f"/evaluations/{evaluation_id}")

    def verify_formula(self, text: str) -> dict[str, Any]:
        """POST /v1/formula/verify — audit formulas in text."""
        return self._post("/v1/formula/verify", {"text": text})

    # ------------------------------------------------------------------
    # Agent Tracking Lifecycle
    # ------------------------------------------------------------------

    def register_agent(self, agent_id: str, name: str, domain: str = "general") -> dict[str, Any]:
        """POST /v1/agents — register a new agent for tracking."""
        return self._post("/v1/agents", {"agent_id": agent_id, "name": name, "domain": domain})

    def list_agents(self) -> list[dict[str, Any]]:
        """GET /v1/agents — list all registered agents."""
        return self._get("/v1/agents")

    def get_agent_history(self, agent_id: str) -> dict[str, Any]:
        """GET /v1/agents/{id}/history — get agent health history."""
        return self._get(f"/v1/agents/{agent_id}/history")

    def report_agent_health(self, agent_id: str, outcomes: list[float]) -> dict[str, Any]:
        """POST /v1/agents/{id}/health — report outcome stream for an agent."""
        return self._post(f"/v1/agents/{agent_id}/health", {"outcomes": outcomes})

    # ------------------------------------------------------------------
    # Trading Collapse Early-Warning
    # ------------------------------------------------------------------

    def audit_trading_history(
        self,
        trades: list[dict[str, Any]],
        window_size: int = 500,
        window_step: int = 50,
        utility_power: float = 1.1,
        ema_weight: float = 0.3,
        v_alpha_panic_threshold: float = -0.02,
        alpha_safety_floor: float = 0.10,
    ) -> dict[str, Any]:
        """POST /v1/trading/audit — structural early-warning audit for trading strategies."""
        return self._post(
            "/v1/trading/audit",
            {
                "trades": trades,
                "window_size": window_size,
                "window_step": window_step,
                "utility_power": utility_power,
                "ema_weight": ema_weight,
                "v_alpha_panic_threshold": v_alpha_panic_threshold,
                "alpha_safety_floor": alpha_safety_floor,
            },
        )

    def audit_capital_guard(self, trades: list[dict[str, Any]], **options: Any) -> dict[str, Any]:
        """POST /v1/capital-guard/audit — split long/short structural audit."""
        return self._post("/v1/capital-guard/audit", {"trades": trades, **options})

    def report_capital_guard(self, trades: list[dict[str, Any]], **options: Any) -> dict[str, Any]:
        """POST /v1/capital-guard/audit/report — deterministic report payload."""
        return self._post("/v1/capital-guard/audit/report", {"trades": trades, **options})

    def report_hyperliquid_vault_capital_guard(
        self,
        vault_address: str,
        fills: list[dict[str, Any]] | None = None,
        aggregate_by_time: bool = True,
    ) -> dict[str, Any]:
        """Fetch or submit vault fills and return a CapitalGuard report."""
        return self._post(
            f"/v1/hyperliquid/vaults/{vault_address}/capital-guard/report",
            {"fills": fills or [], "aggregate_by_time": aggregate_by_time},
        )

    def audit_hyperliquid_vault_capital_guard(
        self,
        vault_address: str,
        fills: list[dict[str, Any]] | None = None,
        aggregate_by_time: bool = True,
    ) -> dict[str, Any]:
        """Audit public or submitted Hyperliquid realized fills."""
        return self._post(
            f"/v1/hyperliquid/vaults/{vault_address}/capital-guard",
            {"fills": fills or [], "aggregate_by_time": aggregate_by_time},
        )

    # ------------------------------------------------------------------
    # Hyperliquid Vaults & Advisory Allocator
    # ------------------------------------------------------------------

    def list_hyperliquid_vaults(self) -> dict[str, Any]:
        return self._get("/v1/hyperliquid/vaults")

    def refresh_hyperliquid_vaults(
        self,
        vault_addresses: list[str],
        *,
        user: str | None = None,
        include_portfolio: bool = True,
        timeframe: str = "allTime",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "vault_addresses": vault_addresses,
            "include_portfolio": include_portfolio,
            "timeframe": timeframe,
        }
        if user is not None:
            payload["user"] = user
        return self._post("/v1/hyperliquid/vaults/refresh", payload)

    def get_hyperliquid_vault(self, vault_address: str) -> dict[str, Any]:
        return self._get(f"/v1/hyperliquid/vaults/{vault_address}")

    def delete_hyperliquid_vault(self, vault_address: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/hyperliquid/vaults/{vault_address}")

    def score_hyperliquid_vault(self, vault_address: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._post(f"/v1/hyperliquid/vaults/{vault_address}/score", payload or {})

    def promote_hyperliquid_vault_to_monitor(
        self, vault_address: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self._post(
            f"/v1/hyperliquid/vaults/{vault_address}/monitor",
            payload or {"seed_history": True},
        )

    def plan_hyperliquid_allocation(
        self,
        *,
        vaults: list[dict[str, Any]] | None = None,
        holdings: list[dict[str, Any]] | None = None,
        total_capital: float | None = None,
        max_allocation_per_vault_pct: float = 0.25,
        min_horizon_chunks: float = 3.0,
        config: dict[str, Any] | None = None,
        include_capital_guard: bool = False,
        capital_guard_fills: dict[str, list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "vaults": vaults or [],
            "holdings": holdings or [],
            "max_allocation_per_vault_pct": max_allocation_per_vault_pct,
            "min_horizon_chunks": min_horizon_chunks,
            "config": config or {},
            "include_capital_guard": include_capital_guard,
            "capital_guard_fills": capital_guard_fills or {},
        }
        if total_capital is not None:
            payload["total_capital"] = total_capital
        return self._post("/v1/hyperliquid/allocation/plan", payload)

    def rebalance_hyperliquid_allocation(
        self,
        *,
        vaults: list[dict[str, Any]] | None = None,
        holdings: list[dict[str, Any]] | None = None,
        total_capital: float | None = None,
        max_allocation_per_vault_pct: float = 0.25,
        min_horizon_chunks: float = 3.0,
        config: dict[str, Any] | None = None,
        include_capital_guard: bool = False,
        capital_guard_fills: dict[str, list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        """Build an advisory rebalance with an optional non-binding overlay."""
        payload = self._hyperliquid_allocation_payload(
            vaults=vaults,
            holdings=holdings,
            total_capital=total_capital,
            max_allocation_per_vault_pct=max_allocation_per_vault_pct,
            min_horizon_chunks=min_horizon_chunks,
            config=config,
            include_capital_guard=include_capital_guard,
            capital_guard_fills=capital_guard_fills,
        )
        return self._post("/v1/hyperliquid/allocation/rebalance", payload)

    def _hyperliquid_allocation_payload(self, **kwargs: Any) -> dict[str, Any]:
        """Resolve allocator keyword arguments without issuing a request."""
        payload: dict[str, Any] = {
            "vaults": kwargs.get("vaults") or [],
            "holdings": kwargs.get("holdings") or [],
            "max_allocation_per_vault_pct": kwargs.get("max_allocation_per_vault_pct", 0.25),
            "min_horizon_chunks": kwargs.get("min_horizon_chunks", 3.0),
            "config": kwargs.get("config") or {},
            "include_capital_guard": kwargs.get("include_capital_guard", False),
            "capital_guard_fills": kwargs.get("capital_guard_fills") or {},
        }
        if kwargs.get("total_capital") is not None:
            payload["total_capital"] = kwargs["total_capital"]
        return payload

    def get_current_hyperliquid_allocation(self) -> dict[str, Any]:
        return self._get("/v1/hyperliquid/allocation/current")

    def list_hyperliquid_allocation_runs(self) -> list[dict[str, Any]]:
        return self._get("/v1/hyperliquid/allocation/runs")

    def get_hyperliquid_allocation_run(self, run_id: str) -> dict[str, Any]:
        return self._get(f"/v1/hyperliquid/allocation/runs/{run_id}")

    def get_hyperliquid_vault_equities(self, user_address: str) -> dict[str, Any]:
        return self._get(f"/v1/hyperliquid/users/{user_address}/vault-equities")

    def get_hyperliquid_portfolio(self, user_address: str) -> dict[str, Any]:
        return self._get(f"/v1/hyperliquid/users/{user_address}/portfolio")

    # ------------------------------------------------------------------
    # Persistent Stream Monitor
    # ------------------------------------------------------------------

    def get_stream_monitor_presets(self) -> dict[str, Any]:
        return self._get("/v1/stream-monitor/presets")

    def list_stream_monitor_sessions(self) -> list[dict[str, Any]]:
        return self._get("/v1/stream-monitor/sessions")

    def create_stream_monitor_session(
        self,
        name: str,
        *,
        preset_id: str = "trading-open-trade",
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "/v1/stream-monitor/sessions",
            {
                "name": name,
                "preset_id": preset_id,
                "config": config or {},
                "metadata": metadata or {},
            },
        )

    def get_stream_monitor_session(self, session_id: str) -> dict[str, Any]:
        return self._get(f"/v1/stream-monitor/sessions/{session_id}")

    def delete_stream_monitor_session(self, session_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/stream-monitor/sessions/{session_id}")

    def append_stream_monitor_event(
        self,
        session_id: str,
        value: float,
        *,
        event_type: str = "score",
        timestamp: str | None = None,
        metadata: dict[str, Any] | None = None,
        token_id: int | None = None,
        entropy: float | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "value": value,
            "event_type": event_type,
            "metadata": metadata or {},
        }
        if timestamp is not None:
            payload["timestamp"] = timestamp
        if token_id is not None:
            payload["token_id"] = token_id
        if entropy is not None:
            payload["entropy"] = entropy
        return self._post(f"/v1/stream-monitor/sessions/{session_id}/events", payload)

    def replay_stream_monitor_session(self, session_id: str) -> dict[str, Any]:
        return self._post(f"/v1/stream-monitor/sessions/{session_id}/replay", {})

    def get_stream_monitor_report(self, session_id: str) -> dict[str, Any]:
        return self._get(f"/v1/stream-monitor/sessions/{session_id}/report")

    # ------------------------------------------------------------------
    # Device Sessions (Edge / IoT)
    # ------------------------------------------------------------------

    def create_device_session(self, request: dict[str, Any]) -> dict[str, Any]:
        """POST /v1/devices/sessions — create an edge device monitoring session."""
        return self._post("/v1/devices/sessions", request)

    def get_device_session(self, device_session_id: str) -> dict[str, Any]:
        """GET /v1/devices/sessions/{id} — get edge device session details."""
        return self._get(f"/v1/devices/sessions/{device_session_id}")

    def append_device_telemetry(self, device_session_id: str, value: float, **options: Any) -> dict[str, Any]:
        """POST /v1/devices/sessions/{id}/telemetry — push a sensor/control score."""
        payload: dict[str, Any] = {"value": value}
        if "channel" in options:
            payload["channel"] = options["channel"]
        if "timestamp" in options:
            payload["timestamp"] = options["timestamp"]
        if "metadata" in options:
            payload["metadata"] = options["metadata"]
        return self._post(f"/v1/devices/sessions/{device_session_id}/telemetry", payload)

    # ------------------------------------------------------------------
    # LLM Collapse Telemetry
    # ------------------------------------------------------------------

    def audit_llm_collapse(
        self,
        generated_text: str,
        token_scores: list[float] | None = None,
        target_log_probs: list[float] | None = None,
        mean_log_probs: list[float] | None = None,
        chunk_size: int = 50,
        collapse_threshold: float = -0.02,
        loss_module: str = "probability_kl",
        ranking_profile: str = "conservative",
        scoring_mode: str = "auto",
    ) -> dict[str, Any]:
        """POST /v1/llm/collapse-audit — universal model collapse telemetry."""
        payload: dict[str, Any] = {
            "generated_text": generated_text,
            "chunk_size": chunk_size,
            "collapse_threshold": collapse_threshold,
            "loss_module": loss_module,
            "ranking_profile": ranking_profile,
            "scoring_mode": scoring_mode,
        }
        if token_scores:
            payload["token_scores"] = token_scores
        if target_log_probs:
            payload["target_log_probs"] = target_log_probs
        if mean_log_probs:
            payload["mean_log_probs"] = mean_log_probs
        return self._post("/v1/llm/collapse-audit", payload)

    def audit_openai_compatible_collapse(
        self,
        base_url: str,
        model: str,
        prompt: str,
        api_key: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        timeout_seconds: int = 120,
        request_logprobs: bool = True,
        chunk_size: int = 50,
        collapse_threshold: float = -0.02,
        loss_module: str = "probability_kl",
        ranking_profile: str = "conservative",
        stream: bool = False,
    ) -> dict[str, Any]:
        """POST /v1/llm/openai-compatible/collapse-audit — audit remote OpenAI-compatible API."""
        return self._post(
            "/v1/llm/openai-compatible/collapse-audit",
            {
                "base_url": base_url,
                "model": model,
                "prompt": prompt,
                "api_key": api_key,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "timeout_seconds": timeout_seconds,
                "request_logprobs": request_logprobs,
                "chunk_size": chunk_size,
                "collapse_threshold": collapse_threshold,
                "loss_module": loss_module,
                "ranking_profile": ranking_profile,
                "stream": stream,
            },
        )

    def list_ollama_models(self, base_url: str = "http://localhost:11434/v1") -> dict[str, Any]:
        """GET /v1/llm/ollama/models — discover locally hosted Ollama models via the API."""
        return self._get(f"/v1/llm/ollama/models?base_url={urllib.parse.quote(base_url, safe='')}")

    def audit_local_hf_collapse(
        self,
        model_id: str,
        prompt: str,
        max_new_tokens: int = 128,
        max_input_tokens: int = 1024,
        torch_dtype: str = "auto",
        chunk_size: int = 50,
        collapse_threshold: float = -0.02,
        loss_module: str = "probability_kl",
        ranking_profile: str = "conservative",
    ) -> dict[str, Any]:
        """POST /v1/llm/local-hf/collapse-audit — audit local Hugging Face model."""
        return self._post(
            "/v1/llm/local-hf/collapse-audit",
            {
                "model_id": model_id,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "max_input_tokens": max_input_tokens,
                "torch_dtype": torch_dtype,
                "chunk_size": chunk_size,
                "collapse_threshold": collapse_threshold,
                "loss_module": loss_module,
                "ranking_profile": ranking_profile,
            },
        )

    # ------------------------------------------------------------------
    # Managed SGO Training Platform
    # ------------------------------------------------------------------

    def get_training_defaults(self) -> dict[str, Any]:
        """GET /v1/training/defaults — environment-specific training defaults."""
        return self._get("/v1/training/defaults")

    def get_job_artifacts(self, job_id: str) -> dict[str, Any]:
        """GET /v1/training/jobs/{id}/artifacts — artifact manifest for a training job."""
        return self._get(f"/v1/training/jobs/{job_id}/artifacts")

    def download_job_artifact_file(self, job_id: str, filename: str, output_path: str):
        """GET /v1/training/jobs/{id}/artifacts/{filename} — download a dataset artifact."""
        url = f"{self.base_url}/v1/training/jobs/{job_id}/artifacts/{filename}"
        req = urllib.request.Request(url, headers=self._headers)
        with urllib.request.urlopen(req) as resp, open(output_path, "wb") as f:
            f.write(resp.read())

    def download_job_profile_artifact(self, job_id: str, filename: str, output_path: str):
        """GET /v1/training/jobs/{id}/profile-artifacts/{filename} — download profile artifact."""
        url = f"{self.base_url}/v1/training/jobs/{job_id}/profile-artifacts/{filename}"
        req = urllib.request.Request(url, headers=self._headers)
        with urllib.request.urlopen(req) as resp, open(output_path, "wb") as f:
            f.write(resp.read())

    def get_job_model_card(self, job_id: str) -> str:
        """GET /v1/training/jobs/{id}/model-card — download model card markdown text."""
        url = f"{self.base_url}/v1/training/jobs/{job_id}/model-card"
        req = urllib.request.Request(url, headers=self._headers)
        with urllib.request.urlopen(req) as resp:
            return resp.read().decode("utf-8")

    def export_job_onnx(self, job_id: str, checkpoint_type: str = "best") -> dict[str, Any]:
        """POST /v1/training/jobs/{id}/exports/onnx — export a policy to ONNX format."""
        return self._post(
            f"/v1/training/jobs/{job_id}/exports/onnx",
            {"checkpoint_type": checkpoint_type},
        )

    def download_job_onnx(self, job_id: str, output_path: str, checkpoint_type: str = "best"):
        """GET /v1/training/jobs/{id}/exports/onnx/download — download ONNX export binary."""
        url = f"{self.base_url}/v1/training/jobs/{job_id}/exports/onnx/download?checkpoint_type={checkpoint_type}"
        req = urllib.request.Request(url, headers=self._headers)
        with urllib.request.urlopen(req) as resp, open(output_path, "wb") as f:
            f.write(resp.read())

    def create_training_job(
        self,
        trainer_type: str = "spg",
        env_name: str | None = "CartPole-v1",
        episodes: int = 1000,
        lambda_sgo: float = 0.75,
        learning_rate: float = 0.001,
        loss_module: str = "moment",
        ranking_profile: str = "balanced",
        steering_enabled: bool = True,
        entropy_coef: float = 0.01,
        tail_risk_coef: float = 0.0,
        async_run: bool = True,
        dataset_csv_b64: str | None = None,
        target_column: str | None = None,
        task_type: str = "classification",
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """POST /v1/training/jobs — spawn a managed SGO training job."""
        return self._post(
            "/v1/training/jobs",
            {
                "trainer_type": trainer_type,
                "env_name": env_name,
                "episodes": episodes,
                "lambda_sgo": lambda_sgo,
                "learning_rate": learning_rate,
                "loss_module": loss_module,
                "ranking_profile": ranking_profile,
                "steering_enabled": steering_enabled,
                "entropy_coef": entropy_coef,
                "tail_risk_coef": tail_risk_coef,
                "async_run": async_run,
                "dataset_csv_b64": dataset_csv_b64,
                "target_column": target_column,
                "task_type": task_type,
                "random_seed": random_seed,
            },
        )

    def inspect_training_dataset(
        self,
        dataset_csv_b64: str,
        target_column: str,
        task_type: str = "classification",
    ) -> dict[str, Any]:
        """Inspect an uploaded CSV before creating a dataset training job."""
        return self._post(
            "/v1/training/datasets/inspect",
            {
                "dataset_csv_b64": dataset_csv_b64,
                "target_column": target_column,
                "task_type": task_type,
            },
        )

    def list_training_jobs(self) -> list[dict[str, Any]]:
        """GET /v1/training/jobs — list all training jobs."""
        return self._get("/v1/training/jobs")

    def get_training_job(self, job_id: str) -> dict[str, Any]:
        """GET /v1/training/jobs/{id} — get job status and metrics."""
        return self._get(f"/v1/training/jobs/{job_id}")

    def cancel_training_job(self, job_id: str) -> dict[str, Any]:
        """POST /v1/training/jobs/{id}/cancel — stop a running job."""
        return self._request("POST", f"/v1/training/jobs/{job_id}/cancel")

    def deploy_training_job(self, job_id: str) -> dict[str, Any]:
        """POST /v1/training/jobs/{id}/deploy — deploy model to Stream Monitor."""
        return self._request("POST", f"/v1/training/jobs/{job_id}/deploy")

    def delete_training_job(self, job_id: str) -> dict[str, Any]:
        """DELETE /v1/training/jobs/{id} — remove job record."""
        return self._request("DELETE", f"/v1/training/jobs/{job_id}")

    def download_checkpoint(self, job_id: str, output_path: str):
        """GET /v1/training/jobs/{id}/download — download PyTorch checkpoint."""
        url = f"{self.base_url}/v1/training/jobs/{job_id}/download"
        req = urllib.request.Request(url, headers=self._headers)
        with urllib.request.urlopen(req) as resp, open(output_path, "wb") as f:
            f.write(resp.read())

    def download_report(self, job_id: str, output_path: str):
        """GET /v1/training/jobs/{id}/report — download structural audit PDF."""
        url = f"{self.base_url}/v1/training/jobs/{job_id}/report"
        req = urllib.request.Request(url, headers=self._headers)
        with urllib.request.urlopen(req) as resp, open(output_path, "wb") as f:
            f.write(resp.read())

    def predict_training_dataset(
        self,
        job_id: str,
        records: list[dict[str, Any]],
        ignore_extra_fields: bool = False,
    ) -> dict[str, Any]:
        """Run hosted inference using a completed portable dataset model."""
        return self._post(
            f"/v1/training/jobs/{job_id}/predict",
            {
                "records": records,
                "ignore_extra_fields": ignore_extra_fields,
            },
        )

    def predict_training_job(
        self,
        job_id: str,
        records: list[dict[str, Any]],
        ignore_extra_fields: bool = False,
    ) -> dict[str, Any]:
        """Alias for hosted dataset prediction."""
        return self.predict_training_dataset(job_id, records, ignore_extra_fields)

    def get_training_model_manifest(self, job_id: str) -> dict[str, Any]:
        """Fetch the portable dataset model manifest."""
        return self._get(f"/v1/training/jobs/{job_id}/artifacts")

    def get_training_integration_examples(self, job_id: str) -> dict[str, Any]:
        """Return generated Python and curl examples for a dataset model."""
        return self._get(f"/v1/training/jobs/{job_id}/integration-examples")

    def download_training_bundle(self, job_id: str, output_path: str):
        """Download a completed dataset model's portable ZIP bundle."""
        url = f"{self.base_url}/v1/training/jobs/{job_id}/bundle"
        req = urllib.request.Request(url, headers=self._headers)
        with urllib.request.urlopen(req) as resp, open(output_path, "wb") as f:
            f.write(resp.read())

    # ------------------------------------------------------------------
    # Calibration & Model Profiles
    # ------------------------------------------------------------------

    def list_profile_cases(self, profile_id: str) -> list[dict[str, Any]]:
        """GET /v1/model-profiles/{id}/cases — list calibration cases for a profile."""
        return self._get(f"/v1/model-profiles/{profile_id}/cases")

    def create_profile_case(self, profile_id: str, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """POST /v1/model-profiles/{id}/cases — bulk create calibration cases."""
        return self._post(
            f"/v1/model-profiles/{profile_id}/cases",
            {"cases": cases},
        )

    def delete_profile_case(self, profile_id: str, case_id: str) -> dict[str, Any]:
        """DELETE /v1/model-profiles/{id}/cases/{case_id} — delete a calibration case."""
        return self._request("DELETE", f"/v1/model-profiles/{profile_id}/cases/{case_id}")

    def get_profile_sweep_metadata(self, profile_id: str) -> dict[str, Any]:
        """GET /v1/model-profiles/{id}/sweep-metadata — discover sweep parameters."""
        return self._get(f"/v1/model-profiles/{profile_id}/sweep-metadata")

    def run_calibration_sweep(self, sweep_request: dict[str, Any]) -> dict[str, Any]:
        """POST /v1/calibration/sweep — sweep candidate configs against a profile."""
        return self._post("/v1/calibration/sweep", sweep_request)

    def list_model_profiles(self) -> list[dict[str, Any]]:
        """GET /v1/model-profiles — list registered model profiles."""
        return self._get("/v1/model-profiles")

    def create_model_profile(self, profile_data: dict[str, Any]) -> dict[str, Any]:
        """POST /v1/model-profiles — create custom model profile."""
        return self._post("/v1/model-profiles", profile_data)

    def patch_model_profile(self, profile_id: str, profile_patch: dict[str, Any]) -> dict[str, Any]:
        """PATCH /v1/model-profiles/{id} — partial update profile."""
        return self._request("PATCH", f"/v1/model-profiles/{profile_id}", profile_patch)

    def list_calibration_suites(self) -> list[dict[str, Any]]:
        """GET /v1/calibration/suites — list benchmark suites."""
        return self._get("/v1/calibration/suites")

    def run_calibration(
        self, profile_id: str, suite_id: str = "manual", cases: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """POST /v1/calibration/run — execute calibration benchmark."""
        payload: dict[str, Any] = {"profile_id": profile_id, "suite_id": suite_id}
        if cases:
            payload["cases"] = cases
        return self._post("/v1/calibration/run", payload)

    def list_calibration_runs(self) -> list[dict[str, Any]]:
        """GET /v1/calibration/runs — list past calibration executions."""
        return self._get("/v1/calibration/runs")

    def get_calibration_run(self, run_id: str) -> dict[str, Any]:
        """GET /v1/calibration/runs/{id} — get calibration results."""
        return self._get(f"/v1/calibration/runs/{run_id}")

    def apply_calibration(self, run_id: str, change_paths: list[str] | None = None) -> dict[str, Any]:
        """POST /v1/calibration/apply — persist recommended config deltas."""
        payload: dict[str, Any] = {"run_id": run_id}
        if change_paths:
            payload["change_paths"] = change_paths
        return self._post("/v1/calibration/apply", payload)

    def list_calibration_deltas(self) -> list[dict[str, Any]]:
        """GET /v1/calibration/deltas — history of applied deltas."""
        return self._get("/v1/calibration/deltas")

    def rollback_calibration(self, delta_id: str) -> dict[str, Any]:
        """POST /v1/calibration/rollback — revert a specific delta."""
        return self._post("/v1/calibration/rollback", {"delta_id": delta_id})

    # ------------------------------------------------------------------
    # Humanize (AI-text-to-Human)
    # ------------------------------------------------------------------

    def humanize_text(
        self,
        text: str,
        *,
        provider: str = "local",
        model_id: str | None = None,
        target_signature: str = "high-variance-human",
        max_iterations: int = 3,
        temperature: float = 0.85,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        """POST /v1/humanize/text — rewrite AI-generated text to appear human-written."""
        payload: dict[str, Any] = {
            "text": text,
            "provider": provider,
            "target_signature": target_signature,
            "max_iterations": max_iterations,
            "temperature": temperature,
        }
        if model_id is not None:
            payload["model_id"] = model_id
        if api_key is not None:
            payload["api_key"] = api_key
        if base_url is not None:
            payload["base_url"] = base_url
        return self._post("/v1/humanize/text", payload)

    def humanize_document(self, file_path: str, **options: Any) -> dict[str, Any]:
        """POST /v1/humanize/document — extract and humanize text from a document file."""
        import mimetypes
        import os

        filename = os.path.basename(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or "application/octet-stream"

        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        with open(file_path, "rb") as f:
            file_data = f.read()

        fields: list[tuple[str, str]] = [("provider", options.get("provider", "local"))]
        if options.get("model_id"):
            fields.append(("model_id", options["model_id"]))
        fields.append(("target_signature", options.get("target_signature", "high-variance-human")))
        fields.append(("max_iterations", str(options.get("max_iterations", 3))))
        fields.append(("temperature", str(options.get("temperature", 0.85))))
        if options.get("api_key"):
            fields.append(("api_key", options["api_key"]))
        if options.get("base_url"):
            fields.append(("base_url", options["base_url"]))

        body_parts = []
        for field_name, field_value in fields:
            body_parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{field_name}"\r\n\r\n{field_value}\r\n'.encode()
            )
        body_parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: {mime_type}\r\n\r\n'.encode()
        )
        body_parts.append(file_data)
        body_parts.append(f"\r\n--{boundary}--\r\n".encode())
        body = b"".join(body_parts)

        headers = dict(self._headers)
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        headers["Content-Length"] = str(len(body))

        url = f"{self.base_url}/v1/humanize/document"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
                if not raw:
                    return {}
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace") if exc.fp else str(exc)
            raise KarakanaAPIError(exc.code, detail) from exc

    def audit_document_experimental(self, file_path: str) -> dict[str, Any]:
        """POST /v1/audit/document/experimental — research-calibrated document detector."""
        import mimetypes
        import os

        filename = os.path.basename(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or "application/octet-stream"

        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        with open(file_path, "rb") as f:
            file_data = f.read()

        body = (
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                f"Content-Type: {mime_type}\r\n\r\n"
            ).encode()
            + file_data
            + f"\r\n--{boundary}--\r\n".encode()
        )

        headers = dict(self._headers)
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        headers["Content-Length"] = str(len(body))

        url = f"{self.base_url}/v1/audit/document/experimental"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
                if not raw:
                    return {}
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace") if exc.fp else str(exc)
            raise KarakanaAPIError(exc.code, detail) from exc

    # ------------------------------------------------------------------
    # Legacy / Core
    # ------------------------------------------------------------------

    def research_audit(
        self,
        study_name: str,
        outcomes_csv_b64: str,
        *,
        domain: str = "meta-science",
        magnitude_column: str = "magnitude",
        label_column: str = "outcome",
        success_label: str = "success",
    ) -> dict[str, Any]:
        """POST /research/audit — meta-science bias audit."""
        return self._post(
            "/research/audit",
            {
                "study_name": study_name,
                "outcomes_csv_base64": outcomes_csv_b64,
                "domain": domain,
                "magnitude_column": magnitude_column,
                "label_column": label_column,
                "success_label": success_label,
            },
        )

    def feasibility_report(self) -> dict[str, Any]:
        """GET /reports/feasibility — latest feasibility report."""
        return self._get("/reports/feasibility")

    def comparison_report(self) -> dict[str, Any]:
        """GET /reports/comparison — latest comparison report."""
        return self._get("/reports/comparison")

    def sgo_stream(self, session_id: str, magnitude: float) -> dict[str, Any]:
        """POST /sgo/stream — stream a magnitude for live SGO tracking."""
        return self._post(
            "/sgo/stream",
            {
                "session_id": session_id,
                "magnitude": magnitude,
            },
        )

    def lambda_suggest(
        self,
        session_id: str,
        current_lambda: float,
        structural_error_history: list[float],
        velocity: float,
        momentum: float,
    ) -> dict[str, Any]:
        """POST /sgo/lambda/suggest — get lambda recommendation."""
        return self._post(
            "/sgo/lambda/suggest",
            {
                "session_id": session_id,
                "current_lambda": current_lambda,
                "structural_error_history": structural_error_history,
                "velocity": velocity,
                "momentum": momentum,
            },
        )

    def create_monitor(
        self,
        name: str,
        benchmark_id: str,
        *,
        threshold_metric: str = "normalized_mse_cofactors",
        threshold_value: float = 0.5,
        alert_email: str | None = None,
    ) -> dict[str, Any]:
        """POST /monitors — create a structural monitor."""
        payload: dict[str, Any] = {
            "name": name,
            "benchmark_id": benchmark_id,
            "threshold_metric": threshold_metric,
            "threshold_value": threshold_value,
        }
        if alert_email:
            payload["alert_email"] = alert_email
        return self._post("/monitors", payload)

    def financial_generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 150,
        stream_scores: bool = False,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """POST /v1/financial/generate — structurally-anchored financial answer."""
        payload: dict[str, Any] = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "stream_scores": stream_scores,
        }
        if history:
            payload["history"] = history
        return self._post("/v1/financial/generate", payload)

    def audit_authenticity(self, text: str) -> dict[str, Any]:
        """POST /v1/audit/authenticity — deepfake / AI-generated text detection."""
        return self._post("/v1/audit/authenticity", {"text": text})

    def audit_document(self, file_path: str) -> dict[str, Any]:
        """POST /v1/audit/document — extract and audit text from PDF, DOCX, or TXT."""
        import mimetypes
        import os

        filename = os.path.basename(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or "application/octet-stream"

        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        with open(file_path, "rb") as f:
            file_data = f.read()

        body = (
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                f"Content-Type: {mime_type}\r\n\r\n"
            ).encode()
            + file_data
            + f"\r\n--{boundary}--\r\n".encode()
        )

        headers = dict(self._headers)
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        headers["Content-Length"] = str(len(body))

        url = f"{self.base_url}/v1/audit/document"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
                if not raw:
                    return {}
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace") if exc.fp else str(exc)
            raise KarakanaAPIError(exc.code, detail) from exc

    def agent_health(self, agent_id: str, outcomes: list[float]) -> dict[str, Any]:
        """POST /v1/protocol/agent-health — audit an agent's outcome stream."""
        return self._post(
            "/v1/protocol/agent-health",
            {
                "agent_id": agent_id,
                "outcomes": outcomes,
            },
        )

    def stream_health(
        self,
        token_scores: list[float],
        *,
        chunk_size: int = 50,
        collapse_threshold: float = -0.02,
        loss_module: str = "moment",
        ranking_profile: str = "balanced",
    ) -> dict[str, Any]:
        """POST /v1/protocol/stream-health — streaming alpha audit."""
        return self._post(
            "/v1/protocol/stream-health",
            {
                "token_scores": token_scores,
                "chunk_size": chunk_size,
                "collapse_threshold": collapse_threshold,
                "loss_module": loss_module,
                "ranking_profile": ranking_profile,
            },
        )

    def rag_integrity(self, query: str, retrieved_context: str) -> dict[str, Any]:
        """POST /v1/audit/rag-integrity — audit RAG query-vs-context consistency."""
        return self._post(
            "/v1/audit/rag-integrity",
            {
                "query": query,
                "retrieved_context": retrieved_context,
            },
        )

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str) -> Any:
        return self._request("GET", path)

    def _post(self, path: str, data: dict[str, Any]) -> Any:
        return self._request("POST", path, data)

    def _post_form(self, path: str, data: dict[str, Any]) -> Any:
        return self._request("POST", path, data, content_type="application/x-www-form-urlencoded")

    def _request(
        self,
        method: str,
        path: str,
        data: dict[str, Any] | None = None,
        *,
        content_type: str = "application/json",
    ) -> Any:
        url = f"{self.base_url}{path}"
        if not data:
            body_bytes = None
        elif content_type == "application/x-www-form-urlencoded":
            body_bytes = urllib.parse.urlencode(data).encode()
        else:
            body_bytes = json.dumps(data).encode()
        headers = dict(self._headers)
        headers["Content-Type"] = content_type
        req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
                if not raw:
                    return {}
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace") if exc.fp else str(exc)
            raise KarakanaAPIError(exc.code, detail) from exc


class KarakanaAPIError(Exception):
    """Raised when the API returns an error response."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")
