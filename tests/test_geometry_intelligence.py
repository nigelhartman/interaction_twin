import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import geometry_intelligence as intelligence


def context_fixture():
    return {'components': {'camera_body': {'case_id': 'camera'}, 'charging_case': {'case_id': 'jabra'}},
            'input_frames': [{'id': 'camera::frame_a', 'case_id': 'camera', 'component_ids': ['camera_body'],
                              'image_url': '/files/data/a.jpg', 'sha256': '', 'timestamp_s': 1.},
                             {'id': 'jabra::frame_b', 'case_id': 'jabra', 'component_ids': ['charging_case'],
                              'image_url': '/files/data/b.jpg', 'sha256': '', 'timestamp_s': 2.}],
            'source_bindings': {}}


def answer_fixture(context):
    return {'summary': 'Separate observed surfaces and moving rigid part candidates.',
            'components': [{'id': component, 'case_id': item['case_id'], 'label': component,
                            'motion_type': 'held_rigid_motion', 'reason': 'Visible held motion candidate.',
                            'evidence_ids': [next(f['id'] for f in context['input_frames'] if component in f['component_ids'])],
                            'reconstruction_strategy': 'Use more observed views.', 'missing_evidence': ['Unseen back remains unknown.']}
                           for component, item in context['components'].items()],
            'proposed_components': [{'id': 'candidate_lid', 'label': 'case lid', 'case_id': 'jabra',
                                     'parent_track_id': 'charging_case', 'motion_type': 'hinge_like_candidate',
                                     'sam_prompt': 'case lid', 'evidence_ids': ['jabra::frame_b'],
                                     'identity_status': 'model_proposal', 'reason': 'Candidate split requires further evidence.'}],
            'hole_advice': [{'component_id': 'camera_body', 'region': 'unknown back',
                             'advice': 'Keep hidden material absent.', 'status': 'retain_unknown',
                             'evidence_ids': ['camera::frame_a']}],
            'capture_actions': [{'case_id': 'jabra', 'component_ids': ['charging_case'],
                                 'action': 'Show lid motion without fingers covering the join.',
                                 'evidence_ids': ['jabra::frame_b']}],
            'metric_geometry_established': False, 'joint_axis_established': False,
            'physical_contact_established': False}


class GeometryIntelligenceTests(unittest.TestCase):
    def setUp(self):
        self.context = context_fixture()
        self.answer = answer_fixture(self.context)

    def test_valid_bounded_advisory_has_no_geometry(self):
        self.assertIs(intelligence.validate_result(self.answer, self.context), self.answer)

    def test_invented_frame_id_rejected(self):
        self.answer['components'][0]['evidence_ids'] = ['camera::unknown']
        with self.assertRaises(Exception):
            intelligence.validate_result(self.answer, self.context)

    def test_cross_case_citation_rejected(self):
        self.answer['components'][0]['evidence_ids'] = ['jabra::frame_b']
        with self.assertRaisesRegex(ValueError, 'cross-case'):
            intelligence.validate_result(self.answer, self.context)

    def test_component_without_saved_observation_rejected(self):
        self.context['input_frames'][0]['component_ids'] = []
        with self.assertRaisesRegex(ValueError, 'observation support'):
            intelligence.validate_result(self.answer, self.context)

    def test_proposed_part_parent_must_exist(self):
        self.answer['proposed_components'][0]['parent_track_id'] = 'invented_parent'
        with self.assertRaises(Exception):
            intelligence.validate_result(self.answer, self.context)

    def test_proposed_part_must_not_replace_existing_identity(self):
        self.answer['proposed_components'][0]['id'] = 'charging_case'
        with self.assertRaisesRegex(ValueError, 'collides'):
            intelligence.validate_result(self.answer, self.context)

    def test_proposed_part_parent_cross_case_rejected(self):
        self.answer['proposed_components'][0]['parent_track_id'] = 'camera_body'
        with self.assertRaisesRegex(ValueError, 'parent case'):
            intelligence.validate_result(self.answer, self.context)

    def test_physical_dimensions_and_certification_rejected(self):
        for unsupported in ['The width is 24 mm.', 'This is a physically accurate shape.', 'This is a simulator-ready asset.']:
            answer = copy.deepcopy(self.answer)
            answer['summary'] = unsupported
            with self.assertRaises(ValueError):
                intelligence.validate_result(answer, self.context)

    def test_negative_uncertainty_language_is_permitted(self):
        self.answer['summary'] = 'This is not a simulator-ready model; hidden contacts remain unknown.'
        intelligence.validate_result(self.answer, self.context)

    def test_code_and_invented_fields_rejected(self):
        self.answer['summary'] = '```python\nimport os\n```'
        with self.assertRaises(ValueError):
            intelligence.validate_result(self.answer, self.context)
        self.answer = answer_fixture(self.context)
        self.answer['geometry'] = [[0, 0, 0]]
        with self.assertRaises(Exception):
            intelligence.validate_result(self.answer, self.context)

    def test_missing_existing_component_not_silently_accepted(self):
        self.answer['components'].pop()
        with self.assertRaisesRegex(ValueError, 'exactly'):
            intelligence.validate_result(self.answer, self.context)

    def fixture_directory(self, root):
        (root / 'data').mkdir()
        for name in ['a', 'b']:
            Image.new('RGB', (16, 12), '#678abc').save(root / 'data' / (name + '.jpg'))
        for frame in self.context['input_frames']:
            frame['sha256'] = intelligence.sha256(root / frame['image_url'].removeprefix('/files/'))
        return root / 'data' / 'fresh'

    def fake_client(self, answer, calls):
        class Fake:
            def __init__(self, *, timeout):
                self.timeout = timeout

            def json_result(self, prompt, schema, images, **kwargs):
                calls.append({'timeout': self.timeout, **kwargs, 'image_count': len(images)})
                receipt = {'request_id': 'mock-only-request', 'status': 'completed',
                           'estimated_cost_usd': '0.01', 'usage': {'input_tokens': 1, 'output_tokens': 1}}
                intelligence.write(Path(kwargs['output']) / 'meta-request.json', receipt)
                return answer, receipt
        return Fake

    def test_mock_success_exactly_one_call_and_second_attempt_forbidden(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            output = self.fixture_directory(root)
            with patch.object(intelligence, 'build_context', return_value=self.context):
                intelligence.prepare(root, output)
            calls = []
            result = intelligence.run(root, output, client_factory=self.fake_client(self.answer, calls))
            self.assertEqual(result['status'], 'model_advisory_not_geometry_truth')
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]['timeout'], 240)
            self.assertEqual(calls[0]['max_output_tokens'], 16000)
            with self.assertRaisesRegex(ValueError, 'retry forbidden'):
                intelligence.run(root, output, client_factory=self.fake_client(self.answer, calls))
            self.assertEqual(len(calls), 1)

    def test_mock_bad_advisory_preserves_raw_paid_answer_and_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            output = self.fixture_directory(root)
            with patch.object(intelligence, 'build_context', return_value=self.context):
                intelligence.prepare(root, output)
            calls = []
            self.answer['metric_geometry_established'] = True
            with self.assertRaises(Exception):
                intelligence.run(root, output, client_factory=self.fake_client(self.answer, calls))
            self.assertEqual(len(calls), 1)
            self.assertTrue((output / 'raw-advisory.json').exists())
            self.assertTrue((output / 'meta-request.json').exists())
            self.assertFalse((output / 'review.json').exists())
            self.assertEqual(intelligence.read(output / 'metadata.json')['status'], 'failed_no_retry')

    def test_changed_input_is_blocked_before_any_client(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            output = self.fixture_directory(root)
            with patch.object(intelligence, 'build_context', return_value=self.context):
                intelligence.prepare(root, output)
            (root / 'data/a.jpg').write_bytes(b'changed')
            calls = []
            with self.assertRaisesRegex(ValueError, 'source image changed'):
                intelligence.run(root, output, client_factory=self.fake_client(self.answer, calls))
            self.assertFalse(calls)

    def test_uniform_jabra_frame_selection_is_deterministic(self):
        program = {'evidence': [{'id': i} for i in range(34)]}
        self.assertEqual([e['id'] for e in intelligence.select_case_frames(program, 'jabra')], [0, 6, 11, 16, 22, 28, 33])

    def test_answer_retention_discards_hidden_reasoning_provider_ids_and_error_text(self):
        document = {'id': 'provider-private-id', 'error': 'arbitrary provider text',
                    'choices': [{'message': {'content': '{"summary":"answer"}',
                                             'reasoning_content': 'hidden reasoning must not be retained',
                                             'tool_calls': [{'id': 'private-tool-id'}]},
                                 'finish_reason': 'stop', 'provider_trace': 'private'}],
                    'usage': {'prompt_tokens': 3, 'completion_tokens': 4,
                              'completion_tokens_details': {'reasoning_tokens': 2, 'reasoning_text': 'hidden'},
                              'arbitrary': 'untrusted text'}}
        snapshot = intelligence.safe_answer_snapshot(document)
        self.assertEqual(snapshot['choices'], [{'message': {'content': '{"summary":"answer"}'}, 'finish_reason': 'stop'}])
        retained = json.dumps(snapshot)
        for denied in ['private', 'hidden reasoning must', 'reasoning_text', 'arbitrary', 'provider_trace']:
            self.assertNotIn(denied, retained)
        self.assertEqual(snapshot['usage']['completion_tokens_details'], {'reasoning_tokens': 2})


if __name__ == '__main__':
    unittest.main()
