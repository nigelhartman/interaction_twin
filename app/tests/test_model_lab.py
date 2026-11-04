"""The actual upgrade controller must preserve selection and local-only boundaries."""
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which('node')


@unittest.skipUnless(NODE, 'Uses the existing Node.js runtime; installs nothing')
class ModelLabTests(unittest.TestCase):
    def run_scenario(self, scenario):
        result = subprocess.run([NODE, str(ROOT / 'tests/model_lab_harness.cjs'), scenario], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_connected_controller_does_not_start_work_automatically(self):
        self.run_scenario('connected')

    def test_saved_file_mode_makes_no_requests(self):
        self.run_scenario('offline')

    def test_late_plan_cannot_enable_a_different_component(self):
        self.run_scenario('stale_plan')

    def test_selected_mode_and_original_hash_reach_local_runner(self):
        self.run_scenario('run_and_mode')

    def test_explicit_cpu_mode_reaches_the_same_guarded_local_runner(self):
        self.run_scenario('cpu_mode')

    def test_late_result_scene_cannot_replace_another_project(self):
        self.run_scenario('result_race')

    def test_matching_result_renders_separately_with_actual_denominators(self):
        self.run_scenario('result_scene')

    def test_invalid_jobs_fixtures_masks_and_oversized_uploads_stay_out(self):
        self.run_scenario('imports_and_jobs')

    def test_mask_job_requires_prompt_and_errors_remain_plain_text(self):
        self.run_scenario('prompt_and_error')

    def test_experimental_uv_handoff_is_explicit_and_separate_from_mask_job(self):
        self.run_scenario('appearance_handoff')

    def test_resume_requires_identical_saved_request_and_original_hash(self):
        self.run_scenario('resume')

    def test_different_source_sampling_is_disclosed_before_running(self):
        self.run_scenario('sampling_note')

    def test_unsupported_native_imports_remain_inspectable_without_enabling_alignment(self):
        self.run_scenario('native_only_import')

    def test_bounded_responses_links_and_exact_import_binding(self):
        self.run_scenario('helper_contracts')


if __name__ == '__main__':
    unittest.main()
