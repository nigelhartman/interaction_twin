"""Fail-closed fresh-body mask gates; no model calls."""
import unittest
import numpy as np
from charger_mask_refit_20261103_v1 import quality


class ChargerMaskRefitTests(unittest.TestCase):
    def test_thin_lost_cable_mask_cannot_become_a_rigid_adapter_pose(self):
        mask = np.zeros((960, 718), bool); mask[300:350, 350:352] = True
        self.assertIn('too_few_visible_adapter_body_pixels', quality(mask))

    def test_fresh_model_body_box_rejects_leakage_and_tiny_wrong_region(self):
        mask = np.zeros((960, 718), bool); mask[300:360, 350:400] = True
        self.assertEqual(quality(mask, [345., 295., 405., 365.]), [])
        self.assertIn('mask_leaks_outside_fresh_model_body_box', quality(mask, [20., 20., 120., 120.]))
        tiny = np.zeros_like(mask); tiny[300:320, 350:365] = True
        self.assertIn('mask_does_not_cover_localized_adapter_body', quality(tiny, [300., 250., 500., 500.]))

    def test_fragmented_and_full_frame_masks_remain_unknown(self):
        mask = np.zeros((960, 718), bool); mask[100:120, 100:120] = True; mask[500:520, 500:520] = True
        self.assertIn('fragmented_adapter_mask', quality(mask))
        self.assertIn('implausibly_large_single_adapter_mask', quality(np.ones_like(mask)))


if __name__ == '__main__': unittest.main()
