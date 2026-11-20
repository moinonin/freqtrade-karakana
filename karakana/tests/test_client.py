import json
import urllib.parse

from karakana.client import KarakanaClient


class _FakeHTTPResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


def test_login_posts_oauth_form_data(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["data"] = request.data.decode()
        captured["timeout"] = timeout
        return _FakeHTTPResponse({"access_token": "token-123", "token_type": "bearer"})

    monkeypatch.setattr("karakana.client.urllib.request.urlopen", fake_urlopen)

    client = KarakanaClient("http://api.local/")
    token = client.login("user@example.com", "secret pass")

    assert token == "token-123"
    assert client._token == "token-123"
    assert captured["url"] == "http://api.local/token"
    assert captured["headers"]["Content-type"] == "application/x-www-form-urlencoded"
    assert urllib.parse.parse_qs(captured["data"]) == {
        "username": ["user@example.com"],
        "password": ["secret pass"],
    }
