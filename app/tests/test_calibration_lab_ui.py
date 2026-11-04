"""Actual shipped JS with adversarial synthetic response ordering; no inference."""
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which('node')


@unittest.skipUnless(NODE, 'Existing Node runtime required; no installation')
class CalibrationLabControllerTests(unittest.TestCase):
    def scenario(self, name, mutation=None):
        command = [NODE, str(ROOT / 'tests/calibration_lab_harness.cjs'), name]
        if mutation: command.append(mutation)
        return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20)

    def verify(self, name):
        result = self.scenario(name)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_older_job_metadata_cannot_replace_newer_selection(self): self.verify('stale_metadata')
    def test_older_report_cannot_replace_newer_selection(self): self.verify('stale_report')
    def test_older_active_report_cannot_restart_old_poll(self): self.verify('stale_poll')
    def test_older_completion_refresh_cannot_reset_newer_selector(self): self.verify('stale_refresh')
    def test_new_selection_immediately_removes_old_source_result(self): self.verify('clear_pending')
    def test_wrong_job_response_is_not_displayed(self): self.verify('wrong_job')
    def test_wrong_source_report_is_rejected(self): self.verify('wrong_source')
    def test_changed_report_digest_is_rejected(self): self.verify('wrong_hash')
    def test_failed_linked_camera_verification_removes_partial_pass(self): self.verify('partial_camera_failure')
    def test_camera_detections_and_overlay_cannot_cross_attempt_boundaries(self): self.verify('cross_attempt_artifacts')
    def test_valid_camera_and_gallery_software_fixture_render(self): self.verify('successful_camera_gallery_fixture')
    def test_result_stays_loading_until_all_camera_and_gallery_evidence_verifies(self): self.verify('staged_camera_gallery_commit')
    def test_path_guards_reject_external_prefix_query_and_traversal_before_fetch(self): self.verify('path_confinement')
    def test_source_text_is_rendered_as_literal_text(self): self.verify('safe_text')

    def test_poll_negative_control_detects_removed_real_guard(self):
        result = self.scenario('stale_poll', 'drop_postrender_guard')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Stale poll continuation', result.stderr)

    def test_refresh_negative_control_detects_removed_real_guard(self):
        result = self.scenario('stale_refresh', 'drop_refresh_guard')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Stale refresh', result.stderr)

    def test_loading_negative_control_detects_removed_immediate_clear(self):
        result = self.scenario('clear_pending', 'drop_immediate_clear')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Old source result remains visible', result.stderr)


if __name__ == '__main__': unittest.main()
