"""Differentiable Dual-Cosine Structural Geometry Optimization (SGO) loss."""

import torch
import torch.nn as nn

EPS = 1e-12


class DualCosineSGOLoss(nn.Module):
    """Differentiable Dual-Cosine SGO loss.

    Combines two scale-invariant trigonometric identities:
    1. Macro Alignment Angle (cos^2 psi_{a,d}) -> normalized_alpha_s_sil
    2. Trajectory Coupling Ratio (cos^2 theta_{P,N}) -> rho_pn_sq

    Rule of Thumb for Standalone Training:
        Keep ``w_coupling <= 0.1 * w_alpha`` (default ``w_alpha=1.0, w_coupling=0.1``).
        Because ``rho_pn_sq`` is symmetric in P_u and N_u, it reaches 0 when chunks
        are EITHER all wins (N_u = 0) OR all losses (P_u = 0). If an untrained policy
        starts with more losses than wins (N_u > P_u) and ``w_coupling`` is too large,
        the coupling gradient initially pushes P_u -> 0 (all losses). Keeping
        ``w_coupling <= 0.1 * w_alpha`` ensures the directional win-maximizing gradient
        (``w_alpha``) always dominates early training.

    Chunk size (validated on the grpo_sgo benchmark, 10 seeds, 2,000 steps;
    see the upstream grg repo ``grpo_sgo_k_sweep.py``): small k wins. Default
    ``k=2`` gives the most robust profile while keeping both cosine terms
    active; ``k=4`` peaks mean accuracy with a worse floor. Large k is the
    worst regime: the coupling term is already satisfied at (k-1)/k wins per
    chunk, and the legacy ``k=10`` was the worst tested config. Keep batch
    size divisible by k (tail truncation).
    """

    def __init__(
        self,
        k: int = 2,
        p: float = 1.1,
        w_alpha: float = 1.0,
        # Rule of thumb: keep w_coupling <= 0.1 * w_alpha so the directional
        # win-maximizing term (w_alpha) dominates the symmetric P_u*N_u decoupling
        # term early in training when N_u > P_u.
        w_coupling: float = 0.1,
        ideal_alpha: float = -1.0,
        eps: float = EPS,
    ):
        super().__init__()
        self.k = k
        self.p = p
        self.w_alpha = w_alpha
        self.w_coupling = w_coupling
        self.ideal_alpha = ideal_alpha
        self.eps = eps

    def forward(self, outcomes: torch.Tensor, base_loss: torch.Tensor | None = None) -> torch.Tensor:
        outcomes_flat = outcomes.reshape(-1)
        n = outcomes_flat.numel()
        m = n // self.k
        if m == 0:
            raise ValueError(f"Outcomes size ({n}) must be at least chunk size k ({self.k}).")

        valid_outcomes = outcomes_flat[: m * self.k]
        transformed = torch.sign(valid_outcomes) * (torch.abs(valid_outcomes) ** self.p)
        reshaped = transformed.reshape(m, self.k)

        P_u = torch.clamp(reshaped, min=0.0).sum(dim=1)
        N_u = torch.clamp(-reshaped, min=0.0).sum(dim=1)

        mu_P = P_u.mean()
        mu_N = N_u.mean()
        s_observed = (mu_P - mu_N) / (mu_P + mu_N + self.eps)
        alpha_s_sil = ((self.ideal_alpha + s_observed) ** 2) / (self.ideal_alpha**2 + s_observed**2 + self.eps)
        normalized_alpha_s_sil = alpha_s_sil / 2.0

        E_P2 = (P_u**2).mean()
        E_N2 = (N_u**2).mean()
        E_PN = (P_u * N_u).mean()
        denom = torch.clamp(E_P2 * E_N2, min=self.eps)
        rho_pn_sq = (E_PN**2) / (denom + self.eps)

        combined_loss = (self.w_alpha * normalized_alpha_s_sil) + (self.w_coupling * rho_pn_sq)
        if base_loss is not None:
            return base_loss + combined_loss
        return combined_loss


class AnchoredDualCosineSGOLoss(DualCosineSGOLoss):
    """Dual-Cosine SGO + magnitude anchor (hinge margin) — the dead-zone fix.

    The pure ``DualCosineSGOLoss`` is homogeneous of degree 0 (scale-invariant):
    it sees only the win/loss PATTERN of outcomes, never their magnitude. Once
    nearly all chunks are all-wins, both cosine terms vanish and their gradient
    dies (the single-sign dead zone), stalling training short of the optimum.
    Measured on the grpo_sgo benchmark suite (upstream grg repo,
    ``grpo_sgo_rigor_suite.py``): |grad| collapses ~1e-1 -> ~1e-5 as win rate
    reaches ~0.94 and ~90% of chunks are all-wins, leaving 2/10 seeds with
    degraded accuracy.

    The anchor adds a scale-sensitive hinge, silent only at convergence:

        L_total = L_DualCosine(outcomes) + w_hinge * relu(margin - outcomes).mean()

    - wrong samples (outcome < 0): gradient pushes a sign flip,
    - correct but low-margin samples: gradient keeps pushing confidence,
    - gradient is alive while ANY outcome is below ``margin``.

    Directional-only: the hinge pushes outcomes up, so it cannot trigger the
    symmetric P_u -> 0 collapse mode; it composes safely with the dominance
    rule ``w_coupling <= 0.1 * w_alpha``.

    Validated on the grpo_sgo benchmark suite (upstream grg repo,
    ``grpo_sgo_margin_sweep.py``, 10 seeds, 2,000 steps): margin=0.0 (pure
    hinge) peaks accuracy (0.995±0.010, beats PPO 9W/1T acc and 10W/0T reward,
    zero bad seeds); margin=0.3 matches PPO's tail stability; w_hinge=1.0.

    Note: this variant deliberately breaks scale invariance — that is the
    purpose of the anchor (pure-loss scale invariance is asserted by tests and
    relied on by callers; keep the pure class unchanged).

    The ``base_loss`` composition hook is preserved: passing ``base_loss``
    yields ``base_loss + L_DualCosine + w_hinge * hinge``.
    """

    def __init__(self, margin: float = 0.0, w_hinge: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.w_hinge = w_hinge

    def forward(self, outcomes: torch.Tensor, base_loss: torch.Tensor | None = None) -> torch.Tensor:
        base = super().forward(outcomes, base_loss)
        hinge = torch.relu(self.margin - outcomes.reshape(-1)).mean()
        return base + self.w_hinge * hinge
