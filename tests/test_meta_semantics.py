"""Omission-only derived semantics; no model/API calls or fabricated evidence."""
import copy
import unittest

import meta_semantics as M
import stage5


class SemanticScopeContracts(unittest.TestCase):
    def setUp(self):
        self.rows = [{'index': i, 'timestamp_s': float(i - 1)} for i in range(1, 5)]
        self.tracks = [{'id': 'part', 'sample_range': [1, 3]}, {'id': 'holder', 'sample_range': [1, 4]}]
        self.proposal = {'summary': 'Model-proposed visible handling.', 'relationships': [],
            'variables': [{'id': 'part_state', 'subject_track_id': 'part', 'name': 'Part visual relation',
                           'values': ['unknown', 'outside', 'inside']},
                          {'id': 'holder_state', 'subject_track_id': 'holder', 'name': 'Holder visual state',
                           'values': ['unknown', 'open']}],
            'assertions': [{'id': 'part_outside', 'variable_id': 'part_state', 'value': 'outside',
                            'evidence_sample': 2, 'status': 'visual_model_proposal', 'rationale': 'Model interpretation.'},
                           {'id': 'holder_open', 'variable_id': 'holder_state', 'value': 'open',
                            'evidence_sample': 4, 'status': 'visual_model_proposal', 'rationale': 'Model interpretation.'}],
            'steps': [{'id': 'handle_parts', 'title': 'Handle the observed components', 'description': 'Model interpretation.',
                       'participant_track_ids': ['part', 'holder'], 'evidence_samples': [2, 4],
                       'preconditions': [{'variable_id': 'part_state', 'value': 'outside', 'evidence_sample': 2}],
                       'postconditions': [{'variable_id': 'part_state', 'value': 'inside', 'evidence_sample': 4},
                                          {'variable_id': 'holder_state', 'value': 'open', 'evidence_sample': 4}],
                       'limitations': ['Contact and seating are unknown.']}], 'limitations': ['Coarse source observations.']}

    def test_omit_only_outside_condition_keep_original_and_trace_every_field(self):
        before = copy.deepcopy(self.proposal)
        with self.assertRaisesRegex(ValueError, 'tracklet range'):
            stage5.validate_proposal(self.proposal, self.rows, self.tracks)
        result, trace = M.normalize_proposal(self.proposal, self.rows, self.tracks)
        self.assertEqual(self.proposal, before)
        self.assertTrue(stage5.validate_proposal(result, self.rows, self.tracks))
        self.assertEqual(result['assertions'], before['assertions'])
        self.assertEqual(result['steps'][0]['preconditions'], before['steps'][0]['preconditions'])
        self.assertEqual(result['steps'][0]['postconditions'], [before['steps'][0]['postconditions'][1]])
        self.assertEqual(result['steps'][0]['evidence_samples'], [2, 4])
        self.assertEqual(result['steps'][0]['limitations'], before['steps'][0]['limitations'] + [M.STEP_DISCLOSURE])
        self.assertEqual(trace['removed_count'], 1)
        self.assertEqual(trace['removed'][0]['original_item'], before['steps'][0]['postconditions'][0])
        self.assertEqual(trace['removed'][0]['subject_tracklet_sample_range'], [1, 3])
        self.assertFalse(trace['source_samples_retargeted'])
        self.assertFalse(trace['manual_semantic_story_used'])

    def test_outside_assertion_and_matching_condition_can_only_be_omitted(self):
        self.proposal['assertions'].append({'id': 'unsupported_inside', 'variable_id': 'part_state', 'value': 'inside',
                                           'evidence_sample': 4, 'status': 'inferred', 'rationale': 'Unsupported model claim.'})
        result, trace = M.normalize_proposal(self.proposal, self.rows, self.tracks)
        self.assertEqual(trace['removed_assertions'], 1)
        self.assertEqual(trace['removed_conditions'], 1)
        self.assertEqual(len(result['assertions']), 2)
        self.assertIn(M.ASSERTION_DISCLOSURE, result['limitations'])

    def test_valid_proposal_is_unchanged_and_deterministic(self):
        self.proposal['steps'][0]['postconditions'].pop(0)
        result, trace = M.normalize_proposal(self.proposal, self.rows, self.tracks)
        self.assertEqual(result, self.proposal)
        self.assertEqual(trace['removed'], [])
        self.assertEqual(trace['original_proposal_sha256'], trace['normalized_proposal_sha256'])
        self.assertEqual((result, trace), M.normalize_proposal(self.proposal, self.rows, self.tracks))
        self.assertEqual(trace['status'], 'unchanged_valid_proposal')

    def test_unknown_ids_illegal_values_invented_samples_and_bad_boundaries_are_rejected(self):
        for kind in ['variable', 'sample', 'value', 'participant', 'boundary', 'subject', 'duplicate_condition']:
            bad = copy.deepcopy(self.proposal)
            condition = bad['steps'][0]['postconditions'][0]
            if kind == 'variable': condition['variable_id'] = 'unknown_variable'
            if kind == 'sample': condition['evidence_sample'] = 999
            if kind == 'value': condition['value'] = 'electrically_verified'
            if kind == 'participant': bad['steps'][0]['participant_track_ids'] = ['holder']
            if kind == 'boundary': condition['evidence_sample'] = 3
            if kind == 'subject': bad['variables'][0]['subject_track_id'] = 'invented_track'
            if kind == 'duplicate_condition': bad['steps'][0]['postconditions'].append(copy.deepcopy(condition))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                M.normalize_proposal(bad, self.rows, self.tracks)

    def test_contradictions_are_rejected_even_when_claims_would_be_outside_scope(self):
        self.proposal['assertions'].append({'id': 'unsupported_outside', 'variable_id': 'part_state', 'value': 'outside',
                                           'evidence_sample': 4, 'status': 'visual_model_proposal', 'rationale': 'Conflicting model claim.'})
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            M.normalize_proposal(self.proposal, self.rows, self.tracks)
        self.proposal['steps'][0]['postconditions'][0]['value'] = 'outside'
        self.proposal['assertions'].append({'id': 'duplicate_point', 'variable_id': 'part_state', 'value': 'outside',
                                           'evidence_sample': 4, 'status': 'visual_model_proposal', 'rationale': 'Duplicate model claim.'})
        with self.assertRaisesRegex(ValueError, 'Duplicate or contradictory'):
            M.normalize_proposal(self.proposal, self.rows, self.tracks)

    def test_other_immutable_validator_failures_are_not_repaired(self):
        for kind in ['unknown_relation', 'cycle', 'step_time', 'unknown_status', 'duplicate_id']:
            bad = copy.deepcopy(self.proposal)
            if kind in {'unknown_relation', 'cycle'}:
                bad['relationships'] = [{'id': 'relation_a', 'subject_track_id': 'part', 'predicate': 'component_of',
                                         'object_track_id': 'invented' if kind == 'unknown_relation' else 'holder',
                                         'status': 'inferred', 'evidence_samples': [2], 'rationale': 'Model hypothesis.', 'limitations': []}]
                if kind == 'cycle':
                    inverse = copy.deepcopy(bad['relationships'][0])
                    inverse.update(id='relation_b', subject_track_id='holder', object_track_id='part')
                    bad['relationships'].append(inverse)
            if kind == 'step_time':
                earlier = copy.deepcopy(bad['steps'][0])
                earlier.update(id='out_of_order', evidence_samples=[1], preconditions=[], postconditions=[])
                bad['steps'].append(earlier)
            if kind == 'unknown_status': bad['assertions'][0]['status'] = 'unknown'
            if kind == 'duplicate_id': bad['assertions'].append(copy.deepcopy(bad['assertions'][0]))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                M.normalize_proposal(bad, self.rows, self.tracks)


if __name__ == '__main__':
    unittest.main()
