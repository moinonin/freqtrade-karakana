import base64
from pathlib import Path
from typing import Any

import httpx


class KarakanaCloudClient:
    """
    Client for interacting with karakana Cloud API.

    This client is intentionally network-only. It does not import or execute
    the local metric engine, so evaluation and reporting require a live API.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.karakana.ai"):
        if not api_key:
            raise ValueError("KarakanaCloudClient requires an API key")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-API-Key": self.api_key}

    def evaluate(
        self,
        positives: list[float],
        negatives: list[float],
        utility_power: float = 1.1,
        ranking_profile: str = "conservative",
        regime_aware: bool = False,
        positive_alpha_weight: float | None = None,
    ) -> dict[str, Any]:
        """
        Perform remote structural evaluation.
        """
        url = f"{self.base_url}/evaluate"
        payload = {
            "positive_outcomes": positives,
            "negative_outcomes": negatives,
            "utility_power": utility_power,
            "ranking_profile": ranking_profile,
            "regime_aware": regime_aware,
            "positive_alpha_weight": positive_alpha_weight,
        }
        with httpx.Client(headers=self.headers) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    def stream_outcome(self, session_id: str, magnitude: float) -> dict[str, Any]:
        """
        Stream a single outcome to the cloud and get current penalty.
        """
        url = f"{self.base_url}/sgo/stream"
        payload = {"session_id": session_id, "magnitude": magnitude}
        with httpx.Client(headers=self.headers) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    def get_history(self, limit: int = 10, offset: int = 0) -> list[dict[str, Any]]:
        """
        Get historical evaluations.
        """
        url = f"{self.base_url}/evaluations"
        params = {"limit": limit, "offset": offset}
        with httpx.Client(headers=self.headers) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    def create_monitor(
        self,
        name: str,
        benchmark_id: str,
        threshold_metric: str = "normalized_mse_cofactors",
        threshold_value: float = 0.5,
        alert_email: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a cloud monitor for a benchmark or production stream.
        """
        url = f"{self.base_url}/monitors"
        payload = {
            "name": name,
            "benchmark_id": benchmark_id,
            "threshold_metric": threshold_metric,
            "threshold_value": threshold_value,
            "alert_email": alert_email,
        }
        with httpx.Client(headers=self.headers) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    def research_audit(
        self,
        study_name: str,
        csv_data: str | bytes | Path,
        domain: str = "meta-science",
        magnitude_column: str = "magnitude",
        label_column: str = "outcome",
        success_label: str = "success",
    ) -> dict[str, Any]:
        """
        Upload a CSV payload to the research audit endpoint.

        `csv_data` may be raw CSV text, CSV bytes, or a filesystem path.
        """
        url = f"{self.base_url}/research/audit"
        raw_csv = self._read_csv_payload(csv_data)
        payload = {
            "study_name": study_name,
            "outcomes_csv_base64": base64.b64encode(raw_csv).decode("ascii"),
            "domain": domain,
            "magnitude_column": magnitude_column,
            "label_column": label_column,
            "success_label": success_label,
        }
        with httpx.Client(headers=self.headers) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    def get_feasibility_report(
        self,
        report_dir: str = "feasibility_report_2026_04_29",
    ) -> dict[str, Any]:
        """
        Fetch a generated feasibility report JSON artifact from the API.
        """
        url = f"{self.base_url}/reports/feasibility"
        with httpx.Client(headers=self.headers) as client:
            response = client.get(url, params={"report_dir": report_dir})
            response.raise_for_status()
            return response.json()

    def get_comparison_report(
        self,
        report_dir: str = "public_openml_10_vs_full",
    ) -> dict[str, Any]:
        """
        Fetch a generated benchmark comparison report JSON artifact from the API.
        """
        url = f"{self.base_url}/reports/comparison"
        with httpx.Client(headers=self.headers) as client:
            response = client.get(url, params={"report_dir": report_dir})
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _read_csv_payload(csv_data: str | bytes | Path) -> bytes:
        if isinstance(csv_data, bytes):
            return csv_data
        if isinstance(csv_data, Path):
            return csv_data.read_bytes()
        if "\n" in csv_data or "\r" in csv_data:
            return csv_data.encode("utf-8")
        path = Path(csv_data)
        if path.exists():
            return path.read_bytes()
        return csv_data.encode("utf-8")
