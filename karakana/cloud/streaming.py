import queue
import threading
import time

import httpx


class AsyncSGOStreamer:
    """
    High-performance asynchronous streamer for karakana Cloud.
    Buffers outcomes locally and flushes in batches to minimize training latency.
    """

    def __init__(
        self,
        api_key: str,
        session_id: str,
        base_url: str = "https://api.karakana.ai",
        batch_size: int = 10,
        flush_interval: float = 1.0,
    ):
        self.api_key = api_key
        self.session_id = session_id
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self.queue = queue.Queue()
        self.current_penalty = 0.0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def push_reward(self, reward: float):
        """
        Non-blocking push of a reward to the buffer.
        """
        self.queue.put(reward)

    def get_latest_penalty(self) -> float:
        """
        Get the last calculated structural penalty from the cloud.
        """
        return self.current_penalty

    def _worker(self):
        """
        Background worker to flush rewards to the API.
        """
        buffer = []
        last_flush = time.time()

        with httpx.Client(headers={"X-API-Key": self.api_key}) as client:
            while not self.stop_event.is_set():
                try:
                    # Wait for reward with timeout to allow interval check
                    reward = self.queue.get(timeout=0.1)
                    buffer.append(reward)
                except queue.Empty:
                    pass

                now = time.time()
                if len(buffer) >= self.batch_size or (buffer and now - last_flush >= self.flush_interval):
                    self._flush(client, buffer)
                    buffer = []
                    last_flush = now

            if buffer:
                self._flush(client, buffer)

    def _flush(self, client: httpx.Client, batch: list[float]):
        """
        Send a batch of rewards to the cloud.
        """
        url = f"{self.base_url}/sgo/stream"
        for reward in batch:
            try:
                # In a real production system, we'd add a /batch endpoint
                # For now, we iterate over the batch or use the last one for penalty
                payload = {"session_id": self.session_id, "magnitude": reward}
                response = client.post(url, json=payload)
                if response.status_code == 200:
                    self.current_penalty = response.json().get("current_penalty", 0.0)
            except Exception as e:
                print(f"Karakana SDK error: {e}")

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)
