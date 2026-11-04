"""Execute the actual saved-mask controller with asynchronous DOM/image doubles."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('node'), 'Node is required for the actual JavaScript controller checks')
class ComponentMaskOverlayTests(unittest.TestCase):
    def test_exact_pts_playback_stale_loads_dimensions_and_real_mask_provenance(self):
        result = subprocess.run([shutil.which('node'), 'tests/masks_runtime.cjs'], cwd=ROOT,
                                check=True, capture_output=True, text=True)
        report = json.loads(result.stdout)
        self.assertEqual(report['status'], 'passed')
        self.assertGreaterEqual(report['checks'], 40)
        self.assertTrue(report['actual_shipped_controller'])
        self.assertFalse(report['browser_pixel_validation'])


if __name__ == '__main__':
    unittest.main()
