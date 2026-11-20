import uuid

from .streaming import AsyncSGOStreamer


class SGOTrainer:
    """
    A 1-line wrapper for RL training loops to integrate SGO regularization.
    """

    def __init__(
        self,
        api_key: str,
        session_id: str | None = None,
        base_url: str = "https://api.karakana.ai",
        batch_size: int = 20,
        flush_interval: float = 2.0,
    ):
        self.session_id = session_id or f"sess-{uuid.uuid4().hex[:8]}"
        self.streamer = AsyncSGOStreamer(
            api_key=api_key,
            session_id=self.session_id,
            base_url=base_url,
            batch_size=batch_size,
            flush_interval=flush_interval,
        )
        print(f"karakana Cloud: SGO Session active - {self.session_id}")

    def record_outcome(self, magnitude: float):
        """
        Record an outcome (reward/payoff) in the cloud.
        """
        self.streamer.push_reward(magnitude)

    def get_penalty(self) -> float:
        """
        Get the latest structural penalty from the cloud.
        Add this to your policy loss.
        """
        return self.streamer.get_latest_penalty()

    def close(self):
        """
        Stop the background streamer.
        """
        self.streamer.stop()


# Helper for Stable-Baselines3 Callback
try:
    from stable_baselines3.common.callbacks import BaseCallback

    class SGOCallback(BaseCallback):
        """
        Stable-Baselines3 callback for karakana Cloud SGO integration.
        """

        def __init__(self, api_key: str, verbose: int = 0):
            super().__init__(verbose)
            self.trainer = SGOTrainer(api_key=api_key)

        def _on_step(self) -> bool:
            # rewards is a numpy array of rewards for the step
            rewards = self.locals.get("rewards")
            if rewards is not None:
                for r in rewards:
                    self.trainer.record_outcome(float(r))
            return True

        def _on_training_end(self) -> None:
            self.trainer.close()

except ImportError:

    class SGOCallback:
        pass
