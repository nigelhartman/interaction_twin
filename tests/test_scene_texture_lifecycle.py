"""Actual texture callbacks cannot report old-case failures after disposal."""
from pathlib import Path
import shutil,subprocess,unittest
ROOT=Path(__file__).resolve().parents[1];NODE=shutil.which('node')


@unittest.skipUnless(NODE,'Existing Node runtime required; no install')
class SceneTextureLifecycleTests(unittest.TestCase):
    def test_disposed_old_error_is_ignored_and_current_error_remains_visible(self):
        r=subprocess.run([NODE,str(ROOT/'tests/scene_texture_lifecycle_harness.cjs'),'late_error'],capture_output=True,text=True,timeout=10)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)

    def test_disposed_old_image_load_cannot_upload_or_draw_but_current_can(self):
        r=subprocess.run([NODE,str(ROOT/'tests/scene_texture_lifecycle_harness.cjs'),'late_load'],capture_output=True,text=True,timeout=10)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)

    def test_error_regression_detects_original_missing_guard_without_editing_source(self):
        r=subprocess.run([NODE,str(ROOT/'tests/scene_texture_lifecycle_harness.cjs'),'late_error','drop_disposed_error_guard'],capture_output=True,text=True,timeout=10)
        self.assertNotEqual(r.returncode,0,'Negative control with original callback must fail')
        self.assertIn('DisposedA error must not mark currentB',r.stderr)


if __name__=='__main__':unittest.main()
