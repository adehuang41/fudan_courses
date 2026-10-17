import unittest

import torch

from src.unet import UNet


class UNetTests(unittest.TestCase):
    def test_forward_shape_matches_segmentation_contract(self):
        model = UNet(in_channels=3, num_classes=3, base_ch=64)
        x = torch.randn(2, 3, 256, 256)

        y = model(x)

        self.assertEqual(y.shape, (2, 3, 256, 256))
        self.assertTrue(torch.isfinite(y).all().item())


if __name__ == "__main__":
    unittest.main()
