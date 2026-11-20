"""Citation and certification engine for karakana Research Audits."""

from datetime import datetime
from typing import Any


class EvidenceCertificationEngine:
    """
    Generates publication-ready structural evidence certificates and citations.
    """

    @staticmethod
    def generate_certificate_markdown(audit_results: dict[str, Any]) -> str:
        """
        Create a Markdown certificate for inclusion in scientific publications.
        """
        audit_id = audit_results.get("audit_id", "N/A")
        indices = audit_results.get("bias_indices", {})
        recommendation = audit_results.get("recommendation", "N/A")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md = f"""# karakana Structural Evidence Certificate
**Audit ID:** `{audit_id}`
**Generated:** {timestamp}

## Executive Summary
{recommendation}

## Structural Geometry Metrics
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Fragility Index (NMSE)** | {indices.get("fragility_index", 0.0):.4f} | {EvidenceCertificationEngine._interpret_fragility(indices.get("fragility_index", 0.0))} |
| **Outcome Imbalance Gap** | {indices.get("outcome_imbalance_gap", 0.0):.4f} | {EvidenceCertificationEngine._interpret_gap(indices.get("outcome_imbalance_gap", 0.0))} |
| **Energy Deficit** | {indices.get("energy_deficit", 0.0):.4f} | {EvidenceCertificationEngine._interpret_energy(indices.get("energy_deficit", 0.0))} |
| **Publication Bias Risk** | {indices.get("publication_bias_risk", "Unknown")} | Qualitative assessment based on geometric symmetry. |

## Citation
> karakana Cloud (2026). *Structural Evidence Audit Report {audit_id}*.
> Evaluated using SGO-Matrix Geometry (v0.1.0).
> https://api.karakana.ai/audits/{audit_id}

---
*This certificate validates the internal geometric consistency of the provided experimental outcomes. It does not replace independent peer review or meta-analysis.*
"""
        return md

    @staticmethod
    def _interpret_fragility(val: float) -> str:
        if val < 0.2:
            return "Excellent stability"
        if val < 0.5:
            return "Acceptable stability"
        if val < 0.8:
            return "Moderate fragility detected"
        return "High structural fragility; results may be outlier-driven"

    @staticmethod
    def _interpret_gap(val: float) -> str:
        if abs(val) < 1.0:
            return "Well-balanced outcome magnitudes"
        if abs(val) < 5.0:
            return "Minor magnitude imbalance"
        return "Significant magnitude imbalance detected"

    @staticmethod
    def _interpret_energy(val: float) -> str:
        if val < 0.1:
            return "Full energy capture"
        if val < 0.3:
            return "Minor energy deficit"
        return "Significant passive/null outcome energy detected"
