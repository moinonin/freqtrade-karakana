"""Gymnasium benchmark runner for karakana."""

from __future__ import annotations

import argparse
import json
from typing import Any

import gymnasium as gym
import numpy as np

from .common import build_benchmark_record, summarize_benchmark_records, write_benchmark_outputs


def _policy_action(observation: np.ndarray, weights: np.ndarray) -> int:
    if weights.ndim == 1:
        return 1 if float(np.dot(observation, weights)) > 0 else 0
    scores = weights @ observation
    return int(np.argmax(scores))


def _run_chunk(
    env_name: str,
    chunk_id: int,
    weight_scale: float,
    episodes: int,
    success_threshold: float,
    max_steps: int | None,
    seed: int,
    ranking_profile: str = "conservative",
) -> dict[str, Any]:
    env = gym.make(env_name)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    obs_size = int(np.asarray(obs).shape[0])
    action_space = getattr(env, "action_space", None)
    action_count = int(getattr(action_space, "n", 0) or 0)
    if action_count <= 0:
        raise ValueError(f"Unsupported action space for benchmark runner: {action_space!r}")
    if action_count == 2:
        weights = rng.uniform(-1.0, 1.0, size=obs_size) * weight_scale
    else:
        weights = rng.uniform(-1.0, 1.0, size=(action_count, obs_size)) * weight_scale

    positives: list[float] = []
    negatives: list[float] = []
    episode_returns: list[float] = []
    episode_lengths: list[int] = []

    for episode_idx in range(episodes):
        observation, _ = env.reset(seed=seed + episode_idx + 1)
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        while not (terminated or truncated):
            action = _policy_action(observation, weights)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break

        episode_returns.append(total_reward)
        episode_lengths.append(steps)
        if total_reward >= success_threshold:
            positives.append(total_reward)
        else:
            negatives.append(total_reward - success_threshold)

    env.close()

    return build_benchmark_record(
        benchmark="gymnasium",
        benchmark_id=env_name,
        subject=f"{env_name}-chunk-{chunk_id}",
        positives=np.asarray(positives, dtype=float),
        negatives=np.asarray(negatives, dtype=float),
        metadata={
            "chunk_id": chunk_id,
            "episodes": episodes,
            "success_threshold": success_threshold,
            "weight_scale": weight_scale,
            "action_count": action_count,
            "seed": seed,
            "mean_episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        },
        ranking_profile=ranking_profile,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark karakana on Gymnasium environments.")
    parser.add_argument("--env", default="CartPole-v1", help="Gymnasium environment id")
    parser.add_argument("--chunks", type=int, default=5, help="Number of benchmark chunks")
    parser.add_argument("--episodes-per-chunk", type=int, default=50, help="Episodes per chunk")
    parser.add_argument(
        "--success-threshold", type=float, default=100.0, help="Reward threshold for a positive outcome"
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max steps per episode")
    parser.add_argument("--ranking-profile", default="balanced", help="karakana ranking profile")
    parser.add_argument("--api-key", default=None, help="karakana Cloud API Key for remote logging")
    parser.add_argument("--base-url", default="https://api.karakana.ai", help="karakana Cloud API Base URL")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--output-dir", default="benchmark_runs/gymnasium", help="Directory for benchmark outputs")
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    if args.chunks <= 0:
        raise SystemExit("--chunks must be > 0")
    for chunk_id in range(1, args.chunks + 1):
        weight_scale = float(chunk_id) / float(args.chunks)
        record = _run_chunk(
            env_name=args.env,
            chunk_id=chunk_id,
            weight_scale=weight_scale,
            episodes=args.episodes_per_chunk,
            success_threshold=args.success_threshold,
            max_steps=args.max_steps,
            seed=args.seed + chunk_id,
            ranking_profile=args.ranking_profile,
        )
        records.append(record)

    summary = summarize_benchmark_records(records)
    outputs = write_benchmark_outputs(args.output_dir, records, summary)

    # Optional Cloud Logging
    if args.api_key:
        from karakana.cloud import KarakanaCloudClient

        print("Logging benchmark to karakana Cloud...")
        client = KarakanaCloudClient(api_key=args.api_key, base_url=args.base_url)
        for record in records:
            try:
                # Send the positives and negatives from the run
                client.evaluate(
                    positives=[record["metrics"]["avg_success_magnitude"]],
                    negatives=[-record["metrics"]["avg_failure_magnitude"]],
                    ranking_profile=args.ranking_profile,
                )
            except Exception as e:
                print(f"Cloud logging failed for subject {record.get('subject')}: {e}")

    print(json.dumps({"summary": summary, "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
