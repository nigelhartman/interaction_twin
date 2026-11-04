"""Run actual app upload-navigation logic against browser-shaped test doubles."""
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NODE=shutil.which('node')


@unittest.skipUnless(NODE,'Existing Node.js runtime is required; no installation is attempted')
class WorkspaceImportTests(unittest.TestCase):
    def scenario(self,name):
        result=subprocess.run([NODE,str(ROOT/'tests/workspace_import_harness.cjs'),name],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_upload_deep_link_waits_for_processor_and_resets_consent(self):
        self.scenario('delayed_deep_link')

    def test_native_default_never_loads_or_renders_historical_geometry_or_pose_coverage(self):
        self.scenario('native_default')

    def test_upload_profiles_show_only_native_sam3d_unless_legacy_is_explicit(self):
        self.scenario('native_profile_filter')

    def test_saved_file_review_does_not_claim_processing_or_make_network_calls(self):
        self.scenario('offline_file')

    def test_missing_profiles_keep_upload_disabled_with_reason(self):
        self.scenario('missing_profiles')

    def test_user_navigation_opens_import_without_starting_a_run(self):
        self.scenario('navigation')

    def test_invalid_selected_profile_blocks_upload(self):
        self.scenario('invalid_profile')

    def test_offline_processor_shows_guidance_without_uploading(self):
        self.scenario('offline_http')

    def test_component_jump_uses_only_the_selected_enabled_part(self):
        self.scenario('component_jump')

    def test_portable_export_defaults_to_no_video_and_keeps_project_context(self):
        self.scenario('portable_export')

    def test_portable_export_rejects_inconsistent_output_links(self):
        self.scenario('portable_bad_response')


if __name__=='__main__':unittest.main()
