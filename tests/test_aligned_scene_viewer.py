"""Run the actual scene contract's projection, timing and iframe trust checks."""
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]


class AlignedSceneViewerTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Existing Node runtime required; no installation')
    def test_real_contract_projection_timing_scale_and_message_boundaries(self):
        result=subprocess.run([shutil.which('node'),'--test',str(ROOT/'tests/test_aligned_scene_contract.cjs')],
                              capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    @unittest.skipUnless(shutil.which('node'), 'Existing Node runtime required; no installation')
    def test_actual_viewer_deferred_manifest_native_load_and_parent_message_lifecycle(self):
        result=subprocess.run([shutil.which('node'),'--test',str(ROOT/'tests/test_aligned_scene_lifecycle.cjs')],
                              capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
