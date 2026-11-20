import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from karakana.pytorch import calculate_rho_pn_sq
from karakana.trainers.spg import SPGTrainer


def run_lambda_sweep():
    print("Starting SGO Autoresearch Loop...")
    lambdas = [0.3067]
    episodes = 100

    os.makedirs("user_data", exist_ok=True)
    tsv_path = "user_data/autoresearch_sgo.tsv"

    # Initialize or clear the TSV file
    with open(tsv_path, "w") as f:
        f.write("Lambda\tSuccess_Rate\tRho_PN_Sq\n")

    all_weights = []
    labels = []

    for lam in lambdas:
        print(f"\n--- Training SPG with lambda_sgo = {lam} ---")
        trainer = SPGTrainer("CartPole-v1", lambda_sgo=lam)

        rewards = []
        for ep in range(episodes):
            ep_reward, _ = trainer.train_episode()
            rewards.append(ep_reward)
            if ep % 25 == 0:
                print(f"Episode {ep}: Reward = {ep_reward}")

        # Calculate Structural Error over the last 20 episodes
        recent_outcomes = torch.tensor(rewards[-20:], dtype=torch.float32)
        rho_pn_sq = calculate_rho_pn_sq(recent_outcomes).item()

        # Calculate Success Rate (e.g. > 100 is success in CartPole)
        successes = sum(1 for r in rewards[-20:] if r > 100)
        success_rate = successes / 20.0

        # Log to TSV
        with open(tsv_path, "a") as f:
            f.write(f"{lam}\t{success_rate}\t{rho_pn_sq}\n")

        print(f"Result for lambda={lam}: Success Rate={success_rate:.2f}, Rho_PN_Sq={rho_pn_sq:.4f}")

        # Extract weights
        weights = []
        for param in trainer.policy.parameters():
            weights.append(param.data.view(-1).cpu().numpy())
        weight_vector = np.concatenate(weights)

        all_weights.append(weight_vector)
        labels.append(lam)

    print("\nTraining complete. Running t-SNE Vector-Field Projection...")

    # t-SNE Projection
    # Note: Perplexity must be less than n_samples, so we set it to min(5, len-1)
    X = np.array(all_weights)
    tsne = TSNE(n_components=2, perplexity=min(5, len(X) - 1), random_state=42)
    X_2d = tsne.fit_transform(X)

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="viridis", s=200, edgecolors="k")
    plt.colorbar(scatter, label="Lambda (\u03bb)")

    # Annotate points
    for i, lam in enumerate(labels):
        plt.annotate(f"\u03bb={lam}", (X_2d[i, 0], X_2d[i, 1]), xytext=(10, 10), textcoords="offset points")

    plt.title("SGO Vector-Field Steering (t-SNE of Policy Manifolds)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, linestyle="--", alpha=0.7)

    plot_path = "user_data/tsne_experiment_lambda_sweep.png"
    plt.savefig(plot_path)
    print(f"Visual Proof saved to: {plot_path}")
    print("Autoresearch Loop Complete.")


if __name__ == "__main__":
    run_lambda_sweep()
