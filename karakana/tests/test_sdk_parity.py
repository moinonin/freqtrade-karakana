import unittest
from unittest.mock import MagicMock, patch

from karakana.client import KarakanaClient


class TestKarakanaClientParity(unittest.TestCase):
    def setUp(self):
        self.client = KarakanaClient("http://testserver")

    @patch("urllib.request.urlopen")
    def test_login(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"access_token": "test-token"}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        token = self.client.login("user@example.com", "password")
        self.assertEqual(token, "test-token")
        self.assertEqual(self.client._token, "test-token")

    @patch("urllib.request.urlopen")
    def test_api_key_lifecycle(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"key": "nl_test_key", "prefix": "nl_test"}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        resp = self.client.create_api_key("Test Key")
        self.assertEqual(resp["key"], "nl_test_key")

        mock_resp.read.return_value = b'[{"prefix": "nl_test"}]'
        keys = self.client.list_api_keys()
        self.assertEqual(len(keys), 1)

    @patch("urllib.request.urlopen")
    def test_training_jobs(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"job_id": "job-123", "status": "queued"}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        job = self.client.create_training_job("spg", "CartPole-v1")
        self.assertEqual(job["job_id"], "job-123")

    @patch("urllib.request.urlopen")
    def test_dataset_model_builder_methods(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"ready_to_train": true}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        inspection = self.client.inspect_training_dataset("YQ==", "target")
        self.assertTrue(inspection["ready_to_train"])

        mock_resp.read.return_value = b'{"predictions": [{"label": "ok"}]}'
        prediction = self.client.predict_training_dataset("job-123", [{"feature_a": 1.0}])
        self.assertEqual(prediction["predictions"][0]["label"], "ok")

    @patch("urllib.request.urlopen")
    def test_trading_audit(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"structural_alpha": 1.5, "velocity": -0.01}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        audit = self.client.audit_trading_history([{"magnitude": 10.0}, {"magnitude": -2.0}])
        self.assertEqual(audit["structural_alpha"], 1.5)

    @patch("urllib.request.urlopen")
    def test_calibration_workflow(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"run_id": "run-456", "status": "completed"}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        run = self.client.run_calibration("finance_rag")
        self.assertEqual(run["run_id"], "run-456")


if __name__ == "__main__":
    unittest.main()
