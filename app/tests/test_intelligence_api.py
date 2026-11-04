"""Published-source boundaries and restart/cache behavior; no live inference."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import intelligence_api as I
import stage5
import task_intelligence as T
try:
    from .test_revisions import CASE, published_fixture
except ImportError:
    # The existing shared fixture imports sibling test_stage5 by its discovery
    # name. Support both unittest's package and `discover -s tests` execution.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_revisions import CASE, published_fixture


class IntelligenceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.source, self.doc = published_fixture(self.root)
        self.digest = stage5.sha(self.source)
        self.body = {'case_id': CASE, 'mode': 'analysis', 'hosted_reasoning_consent': True}

    def tearDown(self):
        self.temp.cleanup()

    def saved_job(self, status='running', number='1', case=CASE):
        job_id = 'intel-' + number * 24
        directory = self.root / 'data/task-intelligence-jobs' / job_id
        directory.mkdir(parents=True)
        path = directory / 'job.json'
        value = {'id': job_id, 'case_id': case, 'interaction_sha256': self.digest,
                 'mode': 'analysis', 'status': status, 'created_at': '2026-10-30T10:00:00+00:00'}
        path.write_text(json.dumps(value))
        return path, value

    def test_browser_can_only_select_published_source_with_explicit_consent(self):
        request = I.canonical_request(self.body, root=self.root)
        self.assertEqual(request['interaction_sha256'], self.digest)
        self.assertEqual(request['interaction_path'], 'data/case/interaction.json')
        for change in ({'command': 'anything'}, {'path': '/private/file'}, {'model': 'unpriced'},
                       {'hosted_reasoning_consent': False}, {'case_id': '../../private'}, {'question': 'not analysis data'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                I.canonical_request(self.body | change, root=self.root)

    def test_question_is_bounded_and_exact_trimmed_question_is_cache_identity(self):
        request = I.canonical_request(self.body | {'mode': 'ask', 'question': '  What moved?  '}, root=self.root)
        self.assertEqual(request['question'], 'What moved?')
        for question in (' ', 'a' * 1501, {'command': 'anything'}):
            with self.subTest(question=type(question).__name__), self.assertRaises(ValueError):
                I.canonical_request(self.body | {'mode': 'ask', 'question': question}, root=self.root)

    def test_record_selects_current_exact_contract_instead_of_hash_sort_order(self):
        actual = T.plan_analysis(self.source, root=self.root, max_images=12, max_output_tokens=12000)
        canonical = {'request_key': actual['request_key'], 'contract': {'mode': 'analysis'}, 'result': {'title': 'Canonical12k'}}
        other = {'request_key': 'f' * 64, 'contract': {'mode': 'analysis'}, 'result': {'title': 'Other8k'}}
        answer = {'request_key': 'e' * 64, 'contract': {'mode': 'ask', 'question': 'What moved?'}, 'result': {'answer': 'Fixture'}}
        with patch.object(I, 'cache_rows', return_value=[canonical, answer, other]):
            record = I.record(CASE, root=self.root)
        self.assertEqual(record['analysis'], canonical)
        self.assertEqual(record['answers'], [answer])
        self.assertFalse(record['api_calls_on_get'])
        self.assertFalse(record['readiness']['simulation_ready'])
        with patch.object(I, 'cache_rows', return_value=[other]):
            self.assertIsNone(I.record(CASE, root=self.root)['analysis'])

    def test_get_displays_exact_verified_sixteen_k_cli_record_without_relabel_or_inference(self):
        alternate = T.plan_analysis(self.source, root=self.root, max_images=12, max_output_tokens=16000)
        wrong_images = T.plan_analysis(self.source, root=self.root, max_images=2, max_output_tokens=16000)
        saved = {'request_key': alternate['request_key'], 'contract': alternate['contract']}
        unrelated = {'request_key': wrong_images['request_key'], 'contract': wrong_images['contract']}
        original = json.dumps(saved, sort_keys=True)
        with patch.object(T, 'run_analysis', side_effect=AssertionError('GET cannot infer')):
            with patch.object(I, 'cache_rows', return_value=[saved]):
                result = I.record(CASE, root=self.root)
            with patch.object(I, 'cache_rows', return_value=[unrelated]):
                rejected = I.record(CASE, root=self.root)
        self.assertEqual(result['analysis'], saved)
        self.assertEqual(result['analysis_selection']['kind'], 'current_exact_16000_output_contract')
        self.assertEqual(result['analysis_selection']['served_contract']['max_output_tokens'], 16000)
        self.assertFalse(result['analysis_selection']['new_api_call'])
        self.assertFalse(result['analysis_selection']['immutable_record_modified'])
        self.assertEqual(json.dumps(saved, sort_keys=True), original)
        self.assertIsNone(rejected['analysis'])

    def test_restart_view_unblocks_stale_job_without_mutating_original_or_retrying(self):
        path, job = self.saved_job()
        raw = path.read_bytes()
        with patch.object(T, 'run_analysis', side_effect=AssertionError('GET must never infer')):
            after_restart = I.record(CASE, root=self.root, active_job_id=None)
            offline = I.record(CASE, root=self.root)
        self.assertEqual(after_restart['job']['status'], 'interrupted')
        self.assertEqual(after_restart['job']['saved_status'], 'running')
        self.assertFalse(after_restart['job']['automatic_retry'])
        self.assertEqual(offline['job']['status'], 'running')
        self.assertEqual(path.read_bytes(), raw)

    def test_matching_live_worker_is_not_marked_interrupted_and_other_case_is_ignored(self):
        _, job = self.saved_job(status='queued')
        self.saved_job(status='running', number='2', case='other-published-case')
        result = I.record(CASE, root=self.root, active_job_id=job['id'])
        self.assertEqual(result['job']['status'], 'queued')
        self.assertNotIn('saved_status', result['job'])

    def test_terminal_job_is_not_rewritten_and_invalid_jobs_do_not_break_saved_review(self):
        path, job = self.saved_job(status='failed')
        malformed = path.parent.parent / ('intel-' + '2' * 24)
        malformed.mkdir()
        (malformed / 'job.json').write_text('{not json')
        wrong_id = path.parent.parent / ('intel-' + '3' * 24)
        wrong_id.mkdir()
        (wrong_id / 'job.json').write_text(json.dumps(job))
        external = self.root / 'external-job.json'
        external.write_text(json.dumps(job))
        symlinked = path.parent.parent / ('intel-' + '4' * 24)
        symlinked.mkdir()
        (symlinked / 'job.json').symlink_to(external)
        result = I.record(CASE, root=self.root, active_job_id=None)
        self.assertEqual(result['job'], job)
        self.assertEqual(json.loads(path.read_text()), job)

    def test_worker_revalidates_published_sha_before_one_bounded_service_call(self):
        request = I.canonical_request(self.body, root=self.root)
        runner = I.IntelligenceRunner(request, root=self.root)
        with patch.object(T, 'run_analysis', return_value={'request_key': 'testkey'}) as run:
            runner.execute()
        self.assertEqual(runner.state['status'], 'complete')
        run.assert_called_once()
        options = run.call_args.kwargs
        self.assertTrue(options['allow_paid_api'])
        self.assertEqual(options['max_output_tokens'], 12000)
        self.assertEqual(options['timeout'], 600)
        self.assertEqual(options['mode'], 'analysis')

    def test_source_change_and_prestart_cancel_do_not_call_paid_service(self):
        request = I.canonical_request(self.body, root=self.root)
        runner = I.IntelligenceRunner(request, root=self.root)
        self.doc['summary'] = 'Changed after publication.'
        self.source.write_text(json.dumps(self.doc))
        with patch.object(T, 'run_analysis') as run:
            runner.execute()
            run.assert_not_called()
        self.assertEqual(runner.state['status'], 'failed')
        # A queued cancellation is also strictly pre-inference.
        runner2 = I.IntelligenceRunner(request, root=self.root)
        runner2.cancel()
        with patch.object(T, 'run_analysis') as run:
            runner2.execute()
            run.assert_not_called()
        self.assertEqual(runner2.state['status'], 'cancelled')

    def test_cache_match_uses_same_canonical_twelve_k_contract_as_worker(self):
        request = I.canonical_request(self.body, root=self.root)
        plan = T.plan_analysis(self.source, root=self.root, max_images=12, max_output_tokens=12000)
        record = {'request_key': plan['request_key'], 'contract': {'mode': 'analysis'}}
        with patch.object(I, 'cache_rows', return_value=[record]):
            self.assertEqual(I.cached_match(request, root=self.root), record)
        with patch.object(I, 'cache_rows', return_value=[dict(record, request_key='other')]):
            self.assertIsNone(I.cached_match(request, root=self.root))

    def test_api_cache_rows_preserve_immutable_record_keys_and_hash(self):
        record = {'mode': 'ask', 'contract': {'question': 'What moved?'},
                  'result': {'answer': 'Explicit synthetic fixture.'}}
        record['record_sha256'] = T._hash(record)
        original_bytes = T._bytes(record)
        with patch.object(T, 'cached_analyses', return_value=[{'path': 'data/saved/analysis.json', 'record': record}]):
            returned = I.cache_rows(self.source, root=self.root)
        self.assertEqual(T._bytes(returned[0]), original_bytes)
        self.assertEqual(returned[0]['record_sha256'], T._hash({k: v for k, v in returned[0].items() if k != 'record_sha256'}))
        self.assertNotIn('question', returned[0])
        self.assertEqual(returned[0]['contract']['question'], 'What moved?')

    def test_verified_historical_fallback_matches_current_image_and_token_settings_and_is_marked(self):
        plan = T.plan_analysis(self.source, root=self.root, max_images=12, max_output_tokens=12000)
        historical = {'request_key': 'a' * 64, 'contract': dict(plan['contract'], code_sha256='old-reviewed-code'),
                      'result': {'title': 'Historical valid fixture'}}
        wrong_bound = {'request_key': 'z' * 64, 'contract': dict(historical['contract'], max_output_tokens=8000),
                       'result': {'title': 'Wrong settings'}}
        with patch.object(I, 'cache_rows', return_value=[historical, wrong_bound]):
            record = I.record(CASE, root=self.root)
        self.assertEqual(record['analysis'], historical)
        self.assertEqual(record['analysis_selection']['kind'], 'verified_historical_contract')
        self.assertEqual(record['analysis_selection']['served_contract'], historical['contract'])
        self.assertFalse(record['analysis_selection']['immutable_record_modified'])
        canonical = dict(historical, request_key=plan['request_key'], contract=plan['contract'])
        with patch.object(I, 'cache_rows', return_value=[canonical, historical]):
            record = I.record(CASE, root=self.root)
        self.assertEqual(record['analysis'], canonical)
        self.assertEqual(record['analysis_selection']['kind'], 'current_exact_contract')

    def test_identical_inference_with_new_loader_code_reuses_verified_saved_result(self):
        request = I.canonical_request(self.body, root=self.root)
        plan = T.plan_analysis(self.source, root=self.root, max_images=12, max_output_tokens=12000)
        historical = {'request_key': 'a' * 64, 'contract': dict(plan['contract'], code_sha256='previous-reviewed-loader')}
        with patch.object(I, 'cache_rows', return_value=[historical]):
            self.assertEqual(I.cached_match(request, root=self.root), historical)
        changed = dict(historical, contract=dict(historical['contract'], prompt_sha256='different-prompt'))
        with patch.object(I, 'cache_rows', return_value=[changed]):
            self.assertIsNone(I.cached_match(request, root=self.root))

    def test_completion_during_cache_read_cannot_show_terminal_job_without_its_saved_answer(self):
        path, job = self.saved_job(status='running')
        answer = {'request_key': 'saved-answer', 'contract': {'mode': 'ask', 'question': 'What moved?'},
                  'result': {'answer': 'Explicit synthetic answer.'}}

        def complete_during_old_cache_snapshot(*args, **kwargs):
            # Reproduce the exact race: the cache iterator captured no answer;
            # then the worker saves its answer and terminal job before returning.
            terminal = dict(job, status='complete', request_key=answer['request_key'])
            path.write_text(json.dumps(terminal))
            return []

        with patch.object(I, 'cache_rows', side_effect=complete_during_old_cache_snapshot):
            first = I.record(CASE, root=self.root, active_job_id=job['id'])
        self.assertEqual(first['answers'], [])
        self.assertEqual(first['job']['status'], 'running')
        self.assertFalse(first['job']['status'] == 'complete' and not first['answers'])
        with patch.object(I, 'cache_rows', return_value=[answer]):
            second = I.record(CASE, root=self.root, active_job_id=job['id'])
        self.assertEqual(second['job']['status'], 'complete')
        self.assertEqual(second['answers'], [answer])


if __name__ == '__main__':
    unittest.main()
