"""Real app/showcase async selection regressions with synthetic UI doubles."""
import shutil,subprocess,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];NODE=shutil.which('node')


@unittest.skipUnless(NODE,'Existing Node runtime required; no installs')
class LazySelectionTests(unittest.TestCase):
    def scenario(self,surface,name):
        result=subprocess.run([NODE,str(ROOT/'tests/lazy_selection_harness.cjs'),surface,name],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_workspace_late_initial_result_cannot_replace_newer_selection(self):self.scenario('workspace','late_initial')
    def test_showcase_late_initial_result_cannot_replace_newer_selection(self):self.scenario('showcase','late_initial')
    def test_workspace_failed_fetch_keeps_old_source_explicit_and_can_recover(self):self.scenario('workspace','failed_fetch')
    def test_showcase_failed_native_fetch_clears_previous_model_and_can_recover(self):self.scenario('showcase','failed_fetch')
    def test_workspace_hash_failure_keeps_old_source_explicit_and_can_recover(self):self.scenario('workspace','failed_hash')
    def test_showcase_native_hash_failure_clears_previous_model_and_can_recover(self):self.scenario('showcase','failed_hash')
    def test_workspace_initial_failure_never_displays_previous_geometry(self):self.scenario('workspace','initial_failure')
    def test_showcase_initial_native_failure_never_displays_previous_geometry(self):self.scenario('showcase','initial_failure')
    def test_workspace_cached_late_completion_is_ignored_after_new_selection(self):self.scenario('workspace','cached_late_completion')
    def test_showcase_cached_late_completion_is_ignored_after_new_selection(self):self.scenario('showcase','cached_late_completion')

    def test_delayed_result_regression_detects_missing_workspace_generation_guard(self):
        result=subprocess.run([NODE,str(ROOT/'tests/lazy_selection_harness.cjs'),'workspace','late_initial','drop_generation_guards'],capture_output=True,text=True,timeout=10)
        self.assertNotEqual(result.returncode,0,'The test must fail if the actual controller generation guards are removed in memory')
        self.assertIn('AssertionError',result.stderr)

    def test_delayed_result_regression_detects_missing_showcase_generation_guard(self):
        result=subprocess.run([NODE,str(ROOT/'tests/lazy_selection_harness.cjs'),'showcase','late_initial','drop_generation_guards'],capture_output=True,text=True,timeout=10)
        self.assertNotEqual(result.returncode,0,'The test must fail if the actual controller generation guards are removed in memory')
        self.assertIn('AssertionError',result.stderr)


if __name__=='__main__':unittest.main()
