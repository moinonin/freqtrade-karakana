import base64
from pathlib import Path

from karakana.cloud.client import KarakanaCloudClient
from karakana.cloud.streaming import AsyncSGOStreamer


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _ClientDouble:
    calls = []

    def __init__(self, headers=None):
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json):
        self.calls.append(("POST", url, json, self.headers))
        if url.endswith("/evaluate"):
            return _Response({"ranking_score": 0.1, "metrics": {"alpha_s_sil": 0.2}})
        if url.endswith("/sgo/stream"):
            return _Response({"session_id": json["session_id"], "current_penalty": 0.3})
        if url.endswith("/monitors"):
            return _Response({"monitor_id": "mon-1", "status": "active"})
        if url.endswith("/research/audit"):
            return _Response({"audit_id": "audit-1", "recommendation": "ok"})
        return _Response({})

    def get(self, url, params=None):
        self.calls.append(("GET", url, params or {}, self.headers))
        if url.endswith("/evaluations"):
            return _Response([{"id": 1, "ranking_score": 0.1}])
        if url.endswith("/reports/feasibility"):
            return _Response({"record_count": 2})
        if url.endswith("/reports/comparison"):
            return _Response({"metric_deltas": []})
        return _Response({})


def test_cloud_client_covers_current_api_surface(monkeypatch):
    import karakana.cloud.client as client_module

    _ClientDouble.calls = []
    monkeypatch.setattr(client_module.httpx, "Client", _ClientDouble)

    client = KarakanaCloudClient(api_key="test_key", base_url="http://api.local/")

    evaluation = client.evaluate(
        [1.0, 2.0],
        [-1.0],
        regime_aware=True,
        positive_alpha_weight=1.5,
    )
    stream = client.stream_outcome("sess-1", 1.5)
    history = client.get_history(limit=5, offset=10)
    monitor = client.create_monitor("guard", "bench-1", threshold_value=0.25)
    audit = client.research_audit("study", "outcome,magnitude\nsuccess,1.0\nfailure,-0.5\n")
    feasibility = client.get_feasibility_report("sample_feasibility")
    comparison = client.get_comparison_report("sample_comparison")

    assert evaluation["metrics"]["alpha_s_sil"] == 0.2
    assert stream["current_penalty"] == 0.3
    assert history[0]["id"] == 1
    assert monitor["status"] == "active"
    assert audit["audit_id"] == "audit-1"
    assert feasibility["record_count"] == 2
    assert comparison["metric_deltas"] == []

    evaluate_call = _ClientDouble.calls[0]
    assert evaluate_call[2]["regime_aware"] is True
    assert evaluate_call[2]["positive_alpha_weight"] == 1.5
    assert evaluate_call[3] == {"X-API-Key": "test_key"}

    history_call = [call for call in _ClientDouble.calls if call[1].endswith("/evaluations")][0]
    assert history_call[2] == {"limit": 5, "offset": 10}

    audit_call = [call for call in _ClientDouble.calls if call[1].endswith("/research/audit")][0]
    decoded = base64.b64decode(audit_call[2]["outcomes_csv_base64"]).decode("utf-8")
    assert "success,1.0" in decoded


def test_cloud_client_requires_api_key():
    try:
        KarakanaCloudClient(api_key="")
    except ValueError as exc:
        assert "API key" in str(exc)
    else:
        raise AssertionError("Expected KarakanaCloudClient to require an API key")


def test_streamer_flushes_remaining_rewards_on_stop(monkeypatch):
    import karakana.cloud.streaming as streaming_module

    _ClientDouble.calls = []
    monkeypatch.setattr(streaming_module.httpx, "Client", _ClientDouble)

    streamer = AsyncSGOStreamer(
        api_key="test_key",
        session_id="sess-1",
        base_url="http://api.local",
        batch_size=10,
        flush_interval=60.0,
    )
    streamer.push_reward(2.5)
    streamer.stop()

    stream_calls = [call for call in _ClientDouble.calls if call[1].endswith("/sgo/stream")]
    assert len(stream_calls) == 1
    assert stream_calls[0][2] == {"session_id": "sess-1", "magnitude": 2.5}
    assert streamer.get_latest_penalty() == 0.3


def test_cloud_sdk_does_not_import_local_metric_engine():
    cloud_dir = Path(__file__).resolve().parents[1]
    sources = [path for path in cloud_dir.glob("*.py") if path.name not in {"normalizers.py", "__init__.py"}]

    for path in sources:
        source = path.read_text()
        assert "karakana.metrics" not in source
        assert "evaluate_structural_geometry" not in source
