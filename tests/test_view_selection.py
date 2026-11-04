import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location("view_selection", Path(__file__).parents[1] / "scripts/stage3_select_views.py")
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


class ViewSelectionTests(unittest.TestCase):
    def test_no_silent_mask_resize_and_empty_rejected(self):
        rgb = np.zeros((80, 100, 3), np.uint8)
        with self.assertRaises(ValueError):
            selection.mask_metrics(rgb, np.zeros((40, 50), bool))
        with self.assertRaises(ValueError):
            selection.mask_metrics(rgb, np.zeros((80, 100), bool))

    def test_clipping_and_blur_reduce_image_quality(self):
        rng = np.random.default_rng(9)
        rgb = rng.integers(0, 256, (80, 100, 3), dtype=np.uint8)
        centered = np.zeros((80, 100), bool)
        centered[20:60, 25:75] = True
        clipped = np.zeros_like(centered)
        clipped[:40, :50] = True
        center_metrics = selection.mask_metrics(rgb, centered)
        clipped_metrics = selection.mask_metrics(rgb, clipped)
        self.assertEqual(center_metrics["frame_sides_touched"], 0)
        self.assertEqual(clipped_metrics["frame_sides_touched"], 2)
        self.assertGreater(clipped_metrics["edge_contact_fraction"], 0)
        self.assertGreater(center_metrics["interior_laplacian_variance"],
                           selection.mask_metrics(np.full_like(rgb, 128), centered)["interior_laplacian_variance"])

    def test_temporal_scope_and_selection_diversity(self):
        def candidate(t, area=100, scope="body"):
            return {"timestamp_s": t, "sample_index": int(t*10), "mask_scope": scope,
                    "metrics": {"area_pixels": area, "bbox_mask_coverage": .7,
                                "interior_laplacian_variance": 20., "edge_contact_fraction": 0,
                                "frame_sides_touched": 0}}
        c = candidate(1)
        self.assertEqual(selection.temporal_consistency(c, [c, candidate(1.1, 1000, "assembly")]), .5)
        self.assertAlmostEqual(selection.temporal_consistency(c, [c, candidate(1.1)]), 1.)
        ranked = selection.score_candidates([candidate(1), candidate(1.1), candidate(2.5)])
        distinct = selection.select_diverse(ranked, 5, 1)
        self.assertEqual(len(distinct), 2)
        self.assertGreaterEqual(abs(distinct[0]["timestamp_s"]-distinct[1]["timestamp_s"]), 1.)

    def test_export_preserves_original_rgb_and_mask_pixel_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb = np.random.default_rng(6).integers(0, 256, (80, 100, 3), dtype=np.uint8)
            mask = np.zeros((80, 100), np.uint8)
            mask[20:50, 40:70] = 255
            Image.fromarray(rgb).save(root / "input.png")
            Image.fromarray(mask).save(root / "mask.png")
            output = root / "export"
            output.mkdir()
            candidate = {"sample_index": 5, "image_file": "input.png", "mask_file": "mask.png",
                         "metrics": {"bbox_xyxy": [40, 20, 70, 50]}}
            exported = selection.export_selected(root, output, [candidate])[0]["exports"]
            self.assertTrue(np.array_equal(rgb, np.asarray(Image.open(output / exported["rgb"]))))
            rgba = np.asarray(Image.open(output / exported["rgba"]))
            self.assertTrue(np.array_equal(rgba[:, :, :3], rgb))
            self.assertTrue(np.array_equal(rgba[:, :, 3], mask))
            self.assertFalse(exported["resized"])


if __name__ == "__main__":
    unittest.main()
