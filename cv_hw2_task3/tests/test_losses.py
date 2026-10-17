import unittest

import torch

from src.losses import CombinedLoss, DiceLoss


class DiceLossTests(unittest.TestCase):
    def test_perfect_prediction_is_near_zero(self):
        target = torch.tensor(
            [[[0, 1], [2, 1]]],
            dtype=torch.long,
        )
        logits = torch.full((1, 3, 2, 2), -12.0)
        for cls in range(3):
            logits[0, cls][target[0] == cls] = 12.0

        loss = DiceLoss(num_classes=3)(logits, target)
        self.assertLess(loss.item(), 1e-4)

    def test_combined_loss_backpropagates(self):
        logits = torch.randn(2, 3, 8, 8, requires_grad=True)
        target = torch.randint(0, 3, (2, 8, 8), dtype=torch.long)

        loss = CombinedLoss(num_classes=3, alpha=1.0, beta=1.0)(logits, target)
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all().item())


if __name__ == "__main__":
    unittest.main()
