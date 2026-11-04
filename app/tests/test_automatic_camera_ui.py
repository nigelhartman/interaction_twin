"""Read-only camera viewer contracts against actual shipped JS; no inference."""
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which('node')

@unittest.skipUnless(NODE, 'Existing Node runtime required; no installation')
class AutomaticCameraControllerTests(unittest.TestCase):
    def verify(self, scenario):
        result = subprocess.run([NODE, str(ROOT / 'tests/automatic_camera_harness.cjs'), scenario], cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless((ROOT / 'data/automatic-camera-20261101/camera-body-display-v1/ui-report.json').is_file(), 'Saved real comparison data absent from this source-only checkout')
    def test_saved_actual_comparison_and_frame_hashes(self): self.verify('actual_record')

    def test_inspection_returns_only_verified_geometry_and_binding(self): self.verify('inspect')
    def test_inspection_rejects_changed_viewer(self): self.verify('inspect_wrong_hash')
    def test_inspection_verifies_actual_source_pixels(self): self.verify('inspect_changed_source')
    def test_inspection_rejects_another_hypothesis_camera(self): self.verify('inspect_wrong_camera')
    def test_inspection_rejects_source_timestamp_drift(self): self.verify('inspect_wrong_source_metadata')
    def test_inspection_rejects_physical_claims(self): self.verify('inspect_wrong_physical_scope')
    def test_pending_viewer_cannot_escape_after_source_switch(self): self.verify('inspect_stale_viewer')
    def test_pending_source_validation_cannot_escape_after_source_switch(self): self.verify('inspect_stale_frame')
    def test_original_report_frame_hashes_bind_the_viewer_sidecar(self): self.verify('original_source_frames')

    def test_permalink_resolver_returns_verified_payload(self): self.verify('resolve')
    def test_permalink_resolver_requires_exact_source_binding(self): self.verify('resolve_wrong_binding')
    def test_permalink_resolver_does_not_substitute_unknown_hypothesis(self): self.verify('resolve_unknown_hypothesis')
    def test_permalink_resolver_rejects_pathlike_hypothesis_before_fetch(self): self.verify('resolve_bad_id')
    def test_permalink_resolver_revalidates_actual_source_pixels(self): self.verify('resolve_changed_pixels')

    def test_study_scope_is_explicit_in_shipped_renderer(self): self.verify('study_scope')
    def test_async_frame_callback_error_is_displayed_without_rejection(self): self.verify('frame_callback_error')
    def test_old_frame_callback_error_cannot_cross_a_source_switch(self): self.verify('frame_callback_stale')

    def test_valid_report_and_exact_source_pixels(self): self.verify('valid')
    def test_wrong_source_is_not_substituted(self): self.verify('unsupported_source')
    def test_wrong_component_is_not_substituted(self): self.verify('unsupported_component')
    def test_invalid_binding_fetches_nothing(self): self.verify('invalid_binding')
    def test_report_must_match_selected_source(self): self.verify('wrong_report_source')
    def test_report_bytes_are_verified(self): self.verify('wrong_report_hash')
    def test_raw_comparison_is_verified_before_display(self): self.verify('wrong_raw_hash')
    def test_source_frame_bytes_are_verified(self): self.verify('wrong_frame_hash')
    def test_switch_clears_old_result_immediately(self): self.verify('clear_pending')
    def test_late_report_cannot_replace_newer_source(self): self.verify('stale_report')
    def test_late_provenance_cannot_replace_newer_source(self): self.verify('stale_original')
    def test_late_frame_cannot_cross_sources(self): self.verify('stale_frame')
    def test_frame_reselection_clears_old_pixels(self): self.verify('frame_cleared_pending')
    def test_disposal_blocks_pending_commit(self): self.verify('dispose')
    def test_physical_claim_flags_are_rejected(self): self.verify('claim_scope')
    def test_no_empty_score_is_fabricated(self): self.verify('empty_metric')
    def test_paths_cannot_escape_local_evidence(self): self.verify('path_confinement')
    def test_conflicting_registry_does_not_choose_arbitrarily(self): self.verify('duplicate_registry')

if __name__ == '__main__': unittest.main()
