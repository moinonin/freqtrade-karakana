import unittest

from karakana.streaming import StreamingAlphaMonitor


class TestContentAwareSigmoid(unittest.TestCase):
    def test_repetition_penalty(self):
        """Verify that repetitive tokens reduce alpha more than diverse ones."""
        # 1. Diverse scores
        diverse_scores = [1.0, 2.0, 3.0, 4.0, 5.0] * 10  # 50 tokens
        monitor_diverse = StreamingAlphaMonitor(chunk_size=50)
        for s in diverse_scores:
            monitor_diverse.push(s, token_id=diverse_scores.index(s) % 5)
        alpha_diverse = monitor_diverse.current_alpha

        # 2. Repetitive scores (same scores, but same token IDs)
        rep_scores = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
        monitor_rep = StreamingAlphaMonitor(chunk_size=50)
        for s in rep_scores:
            monitor_rep.push(s, token_id=1)  # Same token ID every time
        alpha_rep = monitor_rep.current_alpha

        print(f"Alpha Diverse: {alpha_diverse:.4f}, Alpha Repetitive: {alpha_rep:.4f}")
        self.assertLess(alpha_rep, alpha_diverse, "Repetitive tokens should have lower alpha")

    def test_entropy_gate(self):
        """Verify that low-entropy (highly certain) loops trigger heavier penalties."""
        # 1. High repetition with HIGH entropy (less penalized)
        monitor_high_ent = StreamingAlphaMonitor(chunk_size=50)
        for s in [1.0] * 50:
            monitor_high_ent.push(s, token_id=1, entropy=2.0)
        alpha_high_ent = monitor_high_ent.current_alpha

        # 2. High repetition with LOW entropy (more penalized)
        monitor_low_ent = StreamingAlphaMonitor(chunk_size=50)
        for s in [1.0] * 50:
            monitor_low_ent.push(s, token_id=1, entropy=0.01)
        alpha_low_ent = monitor_low_ent.current_alpha

        print(f"Alpha High Entropy Rep: {alpha_high_ent:.4f}, Alpha Low Entropy Rep: {alpha_low_ent:.4f}")
        self.assertLess(alpha_low_ent, alpha_high_ent, "Low-entropy repetition should be penalized more severely")


if __name__ == "__main__":
    unittest.main()
