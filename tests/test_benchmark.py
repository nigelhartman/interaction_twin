import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout

import benchmark as B


def document():
    return {'tracklets': [{'id': 'part', 'label': 'Part', 'source_sample_range': [1, 4], 'identity_status': 'unverified'},
                          {'id': 'other_observation', 'label': 'Another tracklet', 'source_sample_range': [2, 3], 'identity_status': 'unverified'}],
            'assets': [{'track_id': 'part'}],
            'evidence': [{'id': f'source_{i}', 'source_index': i, 'timestamp_s': float(i),
                          'observations': [{'track_id': 'part', 'audit_status': 'plausible_visible_mask', 'gate_status': 'candidate_geometric_gate_only'},
                                           {'track_id': 'other_observation', 'audit_status': 'reject_wrong_or_lost_mask', 'gate_status': 'withheld_geometric_gate'}]}
                         for i in range(1, 5)],
            'motion': [{'track_id': 'part', 'source_sample_index': i, 'timestamp_s': float(i),
                        'status': status, 'camera_from_native_mesh': matrix}
                       for i, status, matrix in [(1, B.REFERENCE, [[1]]), (2, B.SUPPORTED, [[1]]), (3, 'unsupported', None)]],
            'steps': [{'id': 'remove'}], 'assertions': [{'id': 'outside'}], 'relationships': [{'id': 'removable'}],
            'reasoning': {'human_reviewed': True}}


class BenchmarkTests(unittest.TestCase):
    def test_conditioned_coverage_excludes_reference_and_counts_absent_samples(self):
        metrics = B.metrics(document())
        part = metrics['parts'][0]
        self.assertEqual(part['reference_samples'], 1)
        self.assertEqual(part['supported_nonreference_samples'], 1)
        self.assertEqual(part['full_source_coverage'], {'numerator': 1, 'denominator': 3, 'fraction': 1/3})
        self.assertEqual(part['tracklet_range_coverage'], {'numerator': 1, 'denominator': 3, 'fraction': 1/3})
        self.assertEqual(part['unprocessed_or_absent_in_tracklet'], 1)
        self.assertEqual(part['withheld_motion_samples'], 1)
        self.assertIsNone(part['physical_pose_accuracy'])

    def test_observation_tracklets_are_never_reported_as_physical_parts(self):
        metrics = B.metrics(document())
        self.assertEqual(metrics['mask_observed_tracklets'], 2)
        self.assertEqual(metrics['model_audited_plausible_tracklets'], 1)
        self.assertEqual(metrics['generated_tracklets'], 1)
        self.assertIsNone(metrics['physical_distinct_part_count'])
        self.assertFalse(metrics['physical_accuracy_verified'])

    def test_native_asset_with_no_motion_is_zero_coverage_not_success(self):
        doc = document(); doc['motion'] = []
        part = B.metrics(doc)['parts'][0]
        self.assertEqual(part['full_source_coverage']['fraction'], 0)
        self.assertEqual(part['unprocessed_or_absent_in_tracklet'], 4)
        doc['assets'] = []
        self.assertEqual(B.metrics(doc)['parts'], [])

    def test_zero_denominator_is_unknown_not_perfect_or_nan(self):
        self.assertEqual(B.ratio(0, 0), {'numerator': 0, 'denominator': 0, 'fraction': None})

    def test_duplicate_unlinked_or_out_of_range_motion_rejected(self):
        for mode in ['duplicate', 'unknown_track', 'wrong_time', 'outside_range']:
            doc = document()
            if mode == 'duplicate': doc['motion'].append(copy.deepcopy(doc['motion'][0]))
            if mode == 'unknown_track': doc['motion'][0]['track_id'] = 'invented'
            if mode == 'wrong_time': doc['motion'][0]['timestamp_s'] = 99.
            if mode == 'outside_range': doc['tracklets'][0]['source_sample_range'] = [3, 4]
            with self.subTest(mode=mode), self.assertRaises(ValueError): B.metrics(doc)

    def test_different_experiment_cadence_joins_original_pts_without_mutating_indices(self):
        doc = document()
        expected = B.metrics(doc)
        # A separate experiment's sample IDs do not index document evidence.
        for row, index in zip(doc['motion'], [11, 21, 31]):
            row['source_sample_index'] = index
        original = copy.deepcopy(doc)
        self.assertEqual(B.metrics(doc), expected)
        self.assertEqual(doc, original)

    def test_original_pts_join_allows_only_rounding_not_a_nearby_frame(self):
        doc = document()
        doc['motion'][1]['timestamp_s'] += B.SOURCE_TIME_TOLERANCE_S / 2
        self.assertEqual(B.metrics(doc)['parts'][0]['supported_nonreference_samples'], 1)
        doc['motion'][1]['timestamp_s'] = 2. + 2 * B.SOURCE_TIME_TOLERANCE_S
        with self.assertRaisesRegex(ValueError, 'one exact source sample'):
            B.metrics(doc)

    def test_ambiguous_or_duplicate_pts_mapping_is_rejected(self):
        for mode in ['ambiguous_evidence', 'duplicate_different_experiment_index']:
            doc = document()
            if mode == 'ambiguous_evidence':
                extra = copy.deepcopy(doc['evidence'][0])
                extra.update(id='nearby', source_index=5, timestamp_s=1. + B.SOURCE_TIME_TOLERANCE_S / 2)
                doc['evidence'].append(extra)
            else:
                extra = copy.deepcopy(doc['motion'][0])
                extra['source_sample_index'] = 99
                doc['motion'].append(extra)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                B.metrics(doc)

    def test_human_review_pending_despite_model_human_reviewed_flag(self):
        result = B.human_scores(document(), None, 'abc')
        self.assertEqual(result['statement_correctness']['status'], 'pending_human_review')
        self.assertEqual(result['statement_correctness']['unreviewed'], 3)
        self.assertIsNone(result['statement_correctness']['accuracy_on_decided_only']['fraction'])
        self.assertIsNone(result['tracklet_identity_correctness']['accuracy_on_decided_only']['fraction'])

    def test_explicit_human_review_counts_only_decided_items(self):
        review = {'source': 'human', 'reviewer': 'Test reviewer', 'interaction_sha256': 'abc',
                  'statements': [{'id': 'step:remove', 'verdict': 'correct', 'note': 'Compared source boundary frames'},
                                 {'id': 'assertion:outside', 'verdict': 'unknown', 'note': 'Hidden by a hand'}],
                  'identities': [{'id': 'part', 'verdict': 'incorrect', 'note': 'Switch after occlusion'}]}
        result = B.human_scores(document(), review, 'abc')
        self.assertEqual(result['statement_correctness']['accuracy_on_decided_only']['denominator'], 1)
        self.assertEqual(result['statement_correctness']['unreviewed'], 1)
        self.assertEqual(result['statement_correctness']['explicitly_unknown'], 1)
        self.assertEqual(result['tracklet_identity_correctness']['incorrect'], 1)

    def test_stale_model_or_invalid_review_rejected(self):
        base = {'source': 'human', 'reviewer': 'Reviewer', 'interaction_sha256': 'abc'}
        for patch in [{'source': 'model'}, {'reviewer': ''}, {'interaction_sha256': 'stale'},
                      {'physical_part_count': True}, {'physical_part_count': -1},
                      {'statements': [{'id': 'step:invented', 'verdict': 'correct', 'note': 'x'}]},
                      {'statements': [{'id': 'step:remove', 'verdict': 'correct', 'note': ''}]}]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                B.human_scores(document(), dict(base, **patch), 'abc')

    def test_sensitive_traversal_absolute_and_symlink_paths_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'data').mkdir(); (root/'.secrets').mkdir()
            (root/'data'/'disguised.json').symlink_to(root/'.secrets'/'meta.env')
            for value in ['.secrets/meta.env', '../outside', '/etc/passwd', 'data/credential.key', 'data/disguised.json']:
                with self.subTest(path=value), self.assertRaises(ValueError): B.path_in(root, value)

    def test_missing_and_changed_artifacts_do_not_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'frame.jpg').write_bytes(b'original')
            expected = B.digest(root/'frame.jpg')
            self.assertEqual(B.artifact(root, 'frame.jpg', expected)['status'], 'verified')
            (root/'frame.jpg').write_bytes(b'changed')
            self.assertEqual(B.artifact(root, 'frame.jpg', expected)['status'], 'changed')
            self.assertEqual(B.artifact(root, 'missing.jpg', expected)['status'], 'missing')

    def test_runtime_requires_matching_exact_program_and_inference_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = {'id': 'r', 'profile': {'mode': 'local_inference'}, 'status': 'complete',
                   'artifacts': {'interaction': 'program.json'}, 'created_at': '2026-01-01T00:00:00+00:00',
                   'completed_at': '2026-01-01T00:00:12+00:00',
                   'phases': [{'name': 'procedure', 'status': 'complete', 'files': [{'path': 'program.json', 'sha256': 'abc'}]}]}
            def measure():
                (root/'run.json').write_text(json.dumps(run))
                return B.runtime(root, {'run': 'run.json'}, 'program.json', 'abc')
            self.assertEqual(measure()['elapsed_s'], 12)
            run['profile']['mode'] = 'verified_evidence'
            self.assertIsNone(measure()['elapsed_s'])
            run['profile']['mode'] = 'local_inference'
            run['phases'][0]['files'][0]['sha256'] = 'old'
            self.assertIsNone(measure()['elapsed_s'])
            run['phases'][0]['files'][0]['sha256'] = 'abc'
            run['completed_at'] = None
            self.assertIsNone(measure()['elapsed_s'])

    def test_missing_case_and_preserved_failure_are_reviewable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {'version': '1.0', 'cases': [{'id': 'missing', 'interaction': 'missing.json'}], 'failure_runs': ['failed.json']}
            (root/'suite.json').write_text(json.dumps(config))
            (root/'failed.json').write_text(json.dumps({'id': 'failed', 'status': 'failed', 'error': 'Mask audit unavailable', 'phases': []}))
            result = B.build(root, 'suite.json')
            self.assertEqual(result['cases'][0]['status'], 'failed')
            self.assertEqual(result['preserved_failures'][0]['status'], 'failed')
            self.assertFalse(result['policy']['new_inference'])
            page = B.render(result)
            self.assertIn('Mask audit unavailable', page)
            self.assertIn('pending until a human', page)

    def test_matching_bytes_cannot_replace_failed_upstream_source_pose_join(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ['source.mp4', 'frame.jpg', 'mask.png', 'part.glb']:
                (root/name).write_bytes(name.encode())
            doc = document()
            doc['source'] = {'path': 'source.mp4', 'sha256': B.digest(root/'source.mp4')}
            doc['inputs'] = []
            doc['assets'][0].update(path='part.glb', sha256=B.digest(root/'part.glb'))
            for e in doc['evidence']:
                e.update(frame_path='frame.jpg', frame_sha256=B.digest(root/'frame.jpg'))
                for o in e['observations']:
                    o.update(mask_path='mask.png', mask_sha256=B.digest(root/'mask.png'))
            (root/'program.json').write_text(json.dumps(doc))
            def validate(value, root, verify_files):
                if verify_files: raise ValueError('Motion transform differs from upstream experiment')
                return True
            with patch('stage5.validate_interaction', side_effect=validate):
                result = B.case_result(root, {'id': 'mismatch', 'interaction': 'program.json'})
            self.assertEqual(result['structural_validation']['status'], 'valid')
            self.assertEqual(result['source_linkage']['changed_or_missing'], 0)
            self.assertEqual(result['upstream_join_validation']['status'], 'invalid')
            self.assertEqual(result['source_linkage']['status'], 'needs_review')
            self.assertEqual(result['status'], 'needs_review')

    def test_html_escapes_untrusted_titles_and_errors(self):
        report = {'suite': '<script>bad</script>', 'created_at': '', 'cases': [
            {'title': '<img src=x onerror=bad()>', 'status': 'failed', 'error': '<script>oops</script>'}], 'preserved_failures': []}
        page = B.render(report)
        self.assertNotIn('<script>', page)
        self.assertIn('&lt;script&gt;', page)

    def test_cli_regenerates_workspace_mirror_and_same_manifest(self):
        script = Path(__file__).resolve().parents[1]/'scripts/run_benchmark.py'
        spec = importlib.util.spec_from_file_location('benchmark_cli_test', script)
        cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
        report = {'suite': 'new-suite', 'created_at': 'now', 'cases': [], 'preserved_failures': []}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'workspace/benchmark').mkdir(parents=True)
            (root/'workspace/benchmark/index.html').write_text('stale content')
            with patch.object(cli, 'ROOT', root), patch.object(B, 'build', return_value=report), \
                 patch('sys.argv', ['run_benchmark.py']), redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(), 0)
            self.assertNotIn('stale content', (root/'workspace/benchmark/index.html').read_text())
            self.assertIn('new-suite', (root/'workspace/benchmark/index.html').read_text())
            self.assertEqual((root/'workspace/benchmark/report.json').read_bytes(),
                             (root/'output/benchmark-local-v1/report.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
