import importlib.util
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('cv2'), 'Local tracking tests require the geometry/SAM environment')
class LocalTrackingTests(unittest.TestCase):
    def test_known_image_translation_and_unknown_after_mask_loss(self):
        import cv2
        from scripts.stage3_tracks import track_sequence
        # Keep features away from the border even at the coarsest pyramid level.
        image = cv2.GaussianBlur(np.random.default_rng(12).integers(0, 256, (256, 256), dtype=np.uint8), (5, 5), 1)
        images = [cv2.warpAffine(image, np.float32([[1, 0, i*2], [0, 1, i]]), (256, 256)) for i in range(4)]
        seeds = np.float32([[100, 100], [140, 100], [100, 140], [140, 140]])
        masks = [np.ones((256, 256), bool) for _ in images]
        uv, valid, _ = track_sequence(images, masks, 0, seeds)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(uv[-1], seeds+[6, 3], atol=.15)
        masks[2][:] = False
        uv, valid, _ = track_sequence(images, masks, 0, seeds)
        self.assertTrue(valid[1].all())
        self.assertFalse(valid[2:].any())
        self.assertTrue(np.isnan(uv[2:]).all())


if __name__ == '__main__':
    unittest.main()
