"""Run production Meta insights UI against offline browser-shaped test doubles."""
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


@unittest.skipUnless(NODE, "Existing Node.js runtime required; no installation attempted")
class InsightsRuntimeTests(unittest.TestCase):
    def scenario(self, name):
        result = subprocess.run([NODE, str(ROOT / "tests/insights_runtime.cjs"), name],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cached_response_cannot_replace_new_project_or_export_its_data(self):
        self.scenario("stale_cached_project")

    def test_old_submission_cannot_poll_new_project(self):
        self.scenario("stale_submit_project")

    def test_duplicate_form_submission_cannot_start_two_paid_requests(self):
        self.scenario("duplicate_paid_submit")

    def test_saved_question_history_uses_immutable_contract_and_does_not_mutate_it(self):
        self.scenario("immutable_question_history")

    def test_citations_use_exact_source_ids_and_unrounded_timestamps(self):
        self.scenario("exact_citations")

    def test_mask_filters_distinguish_plausible_unknown_rejected_and_missing(self):
        self.scenario("mask_filters")

    def test_running_job_and_failures_do_not_duplicate_or_retry_paid_work(self):
        self.scenario("resume_and_failure")

    def test_cached_answer_uses_canonical_question_without_new_poll(self):
        self.scenario("cached_question")

    def test_interrupted_and_cancelled_jobs_are_honest_terminal_states_without_retry(self):
        self.scenario("interrupted_and_cancelled")

    def test_complete_job_without_requested_answer_does_not_show_another_history_answer_as_success(self):
        self.scenario("missing_requested_answer")

    def test_missing_cached_answer_is_reported_without_new_paid_request(self):
        self.scenario("missing_cached_answer")

    def test_omission_disclosure_stays_visible_after_completed_question_and_cached_replay(self):
        self.scenario("persistent_omission_disclosure")


if __name__ == "__main__":
    unittest.main()
