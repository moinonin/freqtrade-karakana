import numpy as np


class ScientificMagnitudeNormalizer:
    """
    Utility for scaling scientific effect sizes and affinities into
    comparable magnitudes for structural auditing.
    """

    @staticmethod
    def from_ic50(ic50_values: list[float] | np.ndarray, baseline: float = 1000.0) -> np.ndarray:
        """
        Convert IC50 (affinity) to log-scale magnitude.
        Higher affinity (lower IC50) becomes higher magnitude.
        """
        vals = np.asarray(ic50_values, dtype=float)
        # magnitude = -log10(ic50 / baseline)
        return -np.log10(vals / (baseline + 1e-12))

    @staticmethod
    def from_cohens_d(d_values: list[float] | np.ndarray) -> np.ndarray:
        """
        Standardize Cohen's d effect sizes.
        Basically identity, as d is already standardized.
        """
        return np.asarray(d_values, dtype=float)

    @staticmethod
    def from_hazard_ratio(hr_values: list[float] | np.ndarray) -> np.ndarray:
        """
        Convert Hazard Ratios to log-hazard (standard effect size).
        HR < 1 (improvement) becomes positive magnitude if we invert.
        """
        vals = np.asarray(hr_values, dtype=float)
        # Log-hazard ratio; invert so 'improvement' is positive
        return -np.log(vals + 1e-12)

    @staticmethod
    def normalize_to_unit_energy(magnitudes: np.ndarray) -> np.ndarray:
        """Scale magnitudes such that the total absolute energy is 1.0."""
        total_energy = np.sum(np.abs(magnitudes))
        if total_energy < 1e-12:
            return magnitudes
        return magnitudes / total_energy
