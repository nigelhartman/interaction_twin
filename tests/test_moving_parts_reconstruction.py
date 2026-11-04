from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import moving_parts_reconstruction as reconstruction


class MovingPartsReconstructionTests(unittest.TestCase):
    def test_real_mask_inputs_are_actual_sam_candidates(self):
        root = Path(__file__).resolve().parents[1]
        parts = reconstruction.load_parts(root)
        self.assertEqual([p['proposal_id'] for p in parts], ['case_base', 'case_lid'])
        for part in parts:
            self.assertEqual(part['source_frame']['timestamp_s'], 22.001933)
            self.assertEqual(part['source_frame']['segmentation_audit_status'], 'not_model_audited')
            rgb, mask, image, metadata = reconstruction.prepare_input(root / part['image_path'], root / part['mask_path'])
            self.assertLessEqual(max(image.size), 512)
            self.assertFalse(metadata['inpainting'])
            self.assertFalse(metadata['manual_pixel_or_geometry_edits'])

    def test_crop_padding_is_deterministic_and_keeps_unseen_pixels_gray(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            rgb = np.full((100, 120, 3), [10, 20, 30], np.uint8)
            mask = np.zeros((100, 120), np.uint8)
            mask[20:80, 30:90] = 255
            mask[45:50, 50:55] = 0
            Image.fromarray(rgb).save(path / 'rgb.png')
            Image.fromarray(mask).save(path / 'mask.png')
            _, _, input_image, metadata = reconstruction.prepare_input(path / 'rgb.png', path / 'mask.png')
            self.assertEqual(metadata['source_bbox_xyxy'], [30, 20, 90, 80])
            ox, oy = metadata['crop_offset_xy_in_model_input']
            pixels = np.asarray(input_image)
            np.testing.assert_array_equal(pixels[25 + oy, 20 + ox], [128, 128, 128])
            np.testing.assert_array_equal(pixels[oy, ox], [10, 20, 30])
            self.assertFalse(metadata['mask_holes_inpainted'])

    def test_soft_or_empty_masks_fail_no_geometry_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            Image.new('RGB', (20, 20)).save(path / 'rgb.png')
            for value in [0, 255, 127]:
                Image.new('L', (20, 20), value).save(path / 'mask.png')
                with self.assertRaises(ValueError):
                    reconstruction.prepare_input(path / 'rgb.png', path / 'mask.png')


if __name__ == '__main__':
    unittest.main()
