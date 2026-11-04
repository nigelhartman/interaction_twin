"""Offline mechanism classification/coordinate/one-attempt guards."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import mechanism_reasoning as M


def context():
    return {'source_sha256': 'a'*64, 'interaction_sha256': 'b'*64, 'interaction_path': 'data/example/interaction.json',
            'tracklets': [{'id': x, 'source_sample_range': [1, 9]} for x in ['body', 'lid', 'insert']],
            'evidence': [{'evidence_id': 'frame_a', 'source_index': 2, 'timestamp_s': 1.,
                          'observations': [{'track_id': x} for x in ['body', 'lid', 'insert']]}],
            'images': [{'role': 'source_rgb', 'evidence_id': 'frame_a', 'width': 540, 'height': 960, 'path': 'data/source.jpg'}]}


def result():
    return {'title': 'Possible mechanism', 'joints': [{'id': 'lid_joint', 'parent_id': 'body', 'child_id': 'lid',
             'joint_type': 'revolute', 'attachment': 'persistent', 'confidence': 'medium', 'evidence_ids': ['frame_a'],
             'state_hints': [{'evidence_id': 'frame_a', 'state': 'open'}],
             'hinge_hints': [{'evidence_id': 'frame_a', 'endpoint_a_normalized': [.25, .5], 'endpoint_b_normalized': [.75, .5],
                              'uncertainty_normalized': .02, 'confidence': 'low'}],
             'rationale': 'Visible seam and persistent attachment candidate.', 'limitations': ['Not physically verified.']}],
            'limitations': ['Depth is unknown.'], 'physical_accuracy_verified': False}


class MechanismReasoningTests(unittest.TestCase):
    def test_normalized_uv_is_deterministically_grounded_to_exact_source_pixels(self):
        value = M.export_proposals(result(), context())
        hint = value['joints'][0]['hinge_hints'][0]
        self.assertEqual(hint['source_sample_index'], 2)
        self.assertEqual(hint['timestamp_s'], 1.)
        self.assertEqual(hint['endpoints_px'], [[134.75, 479.5], [404.25, 479.5]])
        self.assertEqual(hint['uncertainty_px'], 19.2)
        self.assertFalse(value['physical_accuracy_verified'])
        self.assertFalse(value['simulation_ready'])
        self.assertNotIn('axis', value['joints'][0])
        self.assertNotIn('limits_rad', value['joints'][0])

    def test_rejects_unseen_evidence_unknown_parts_self_joint_and_invented_physics(self):
        for mutate in [lambda r: r['joints'][0].update(evidence_ids=['unseen']),
                       lambda r: r['joints'][0].update(child_id='invented'),
                       lambda r: r['joints'][0].update(child_id='body'),
                       lambda r: r.update(physical_accuracy_verified=True),
                       lambda r: r['joints'][0].update(axis=[0, 1, 0]),
                       lambda r: r['joints'][0]['hinge_hints'][0].update(endpoint_a_normalized=[1.1, .5])]:
            r = result(); mutate(r)
            with self.subTest(value=r), self.assertRaises(Exception):
                M.validate_result(r, context())

    def test_removable_is_not_hinge_or_persistent_attachment(self):
        r = result(); j = r['joints'][0]
        j.update(child_id='insert', joint_type='removable', attachment='removable', hinge_hints=[])
        self.assertTrue(M.validate_result(r, context()))
        j['attachment'] = 'persistent'
        with self.assertRaises(ValueError): M.validate_result(r, context())
        j['attachment'] = 'removable'; j['hinge_hints'] = result()['joints'][0]['hinge_hints']
        with self.assertRaises(ValueError): M.validate_result(r, context())

    def test_rejects_cycles_and_multiple_persistent_parents(self):
        r = result(); second = copy.deepcopy(r['joints'][0]); second.update(id='other_joint', parent_id='lid', child_id='body')
        r['joints'].append(second)
        with self.assertRaises(ValueError): M.validate_result(r, context())
        second.update(parent_id='insert', child_id='lid')
        with self.assertRaises(ValueError): M.validate_result(r, context())

    def test_wire_schema_remains_strict_bounded_local_schema_retained(self):
        full = M.response_schema(); wire = M.provider_schema_projection(full)
        self.assertFalse(wire['additionalProperties'])
        self.assertEqual(set(wire['required']), set(wire['properties']))
        self.assertIn('maxItems', full['properties']['joints'])
        self.assertNotIn('maxItems', wire['properties']['joints'])
        self.assertTrue(M.schema_diagnostics(wire)['within_published_numeric_limits'])

    def test_prepared_call_is_off_by_default_and_never_retried_after_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); folder = root/'data/prepared'; folder.mkdir(parents=True)
            p = {'request_key': 'x', 'contract': {'max_images': 16, 'max_output_tokens': 10000,
                 'prompt_sha256': __import__('hashlib').sha256(b'prompt').hexdigest()}, 'context': context(),
                 'schema': M.response_schema(), 'provider_schema': M.provider_schema_projection(M.response_schema()), 'prompt': 'prompt'}
            for name, value in [('plan.json', {k:v for k,v in p.items() if k not in {'schema','provider_schema','prompt'}}),
                                ('schema.json', p['schema']), ('provider-schema.json', p['provider_schema'])]:
                (folder/name).write_text(json.dumps(value))
            (folder/'prompt.txt').write_text('prompt')
            class Fake:
                calls = 0
                def json_result(self, *args, **kwargs):
                    self.calls += 1
                    raise RuntimeError('offline test transport failed')
            client = Fake()
            with patch.object(M, 'plan', return_value=p):
                with self.assertRaisesRegex(ValueError, 'not authorized'):
                    M.run_prepared('data/prepared', root=root, client=client)
                self.assertEqual(client.calls, 0)
                self.assertFalse((folder/'attempt').exists())
                with self.assertRaises(RuntimeError):
                    M.run_prepared('data/prepared', root=root, client=client, allow_paid_api=True)
                with self.assertRaisesRegex(ValueError, 'no automatic paid retry'):
                    M.run_prepared('data/prepared', root=root, client=client, allow_paid_api=True)
                self.assertEqual(client.calls, 1)
                self.assertTrue((folder/'attempt/failure.json').is_file())

    def test_five_dollar_task_envelope_admission_happens_before_transport(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = Path(temp)/'budget.json'
            plan.write_text(json.dumps({'additional_estimated_api_ceiling_usd': '5', 'budget_baseline': {'settled_usd': '1', 'reserved_usd': '0'}}))
            client = object.__new__(M.MechanismMetaClient)
            client.task_budget_plan = plan
            class Ledger:
                def status(self): return {'settled_usd': '5', 'reserved_usd': '0', 'per_request_cap_usd': '2'}
            client.ledger = Ledger()
            with patch.object(M.IncrementMetaClient, 'request') as transport:
                with self.assertRaisesRegex(M.MetaError, 'limit_reached'):
                    client.request({})
                transport.assert_not_called()


if __name__ == '__main__': unittest.main()
