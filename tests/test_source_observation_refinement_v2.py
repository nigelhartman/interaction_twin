import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

import mechanism_reasoning as M
import object_program as O
import source_observation_refinement_v2 as R
from tests.test_source_observation_refinement_v1 import fixture


class SourceObservationMechanismReview(unittest.TestCase):
    def _model_fixture(self, plan):
        _, result = fixture()
        result['format'] = R.FORMAT
        result['source_sha256'] = plan['context']['source_sha256']
        for frame, image in zip(result['frames'], plan['context']['images']):
            frame['timestamp_s'] = image['timestamp_s']
        result['title'] = 'Synthetic fixture only'
        result['joints'] = [{'id': 'j_laptop_base_laptop_lid', 'parent_id': 'laptop_base', 'child_id': 'laptop_lid',
                             'joint_type': 'revolute', 'attachment': 'persistent', 'confidence': 'low',
                             'evidence_ids': [i['evidence_id'] for i in plan['context']['images']],
                             'state_hints': [{'evidence_id': i['evidence_id'], 'state': 'closed'} for i in plan['context']['images']],
                             'hinge_hints': [], 'rationale': 'synthetic fixture', 'limitations': ['synthetic fixture']}]
        return result

    def test_preparation_is_source_only_and_native_export_contract_complete(self):
        with patch('meta_api.MetaClient.request', side_effect=AssertionError('offline only')):
            plan = R.build_plan()
        self.assertEqual(plan['contract']['context_sha256'], R.digest(plan['context']))
        self.assertEqual(plan['contract']['schema_sha256'], R.digest(plan['schema']))
        self.assertEqual(len(plan['context']['images']), 8)
        self.assertTrue(all(i['role'] == 'source_rgb' for i in plan['context']['images']))
        self.assertEqual(plan['context']['tracklets'], O.read_json(R.ROOT / plan['context']['interaction_path'])['tracklets'])
        self.assertEqual(plan['schema']['properties']['joints']['maxItems'], 1)
        self.assertEqual(plan['schema']['properties']['joints']['items']['properties']['child_id']['const'], 'laptop_lid')
        self.assertLess(float(plan['maximum_estimated_reservation_usd']), 2)

    def test_full_observations_round_trip_mechanism_without_claiming_new_inference(self):
        plan = R.build_plan()
        result = self._model_fixture(plan)
        R.validate_result(result, plan)
        projection = {key: result[key] for key in M.response_schema()['properties']}
        proposal = M.export_proposals(projection, plan['context'])
        self.assertEqual(len(proposal['joints'][0]['state_hints']), 8)
        self.assertEqual(proposal['joints'][0]['state_hints'][0]['source_sample_index'], 1)
        self.assertEqual(proposal['interaction_sha256'], plan['context']['interaction_sha256'])
        bad = copy.deepcopy(result)
        bad['joints'][0]['state_hints'][0]['state'] = 'open'
        with self.assertRaisesRegex(ValueError, 'Frame state'):
            R.validate_result(bad, plan)

    def test_existing_native_program_reader_accepts_exact_extended_response(self):
        plan = R.build_plan()
        result = self._model_fixture(plan)
        projection = {key: result[key] for key in M.response_schema()['properties']}
        proposal = M.export_proposals(projection, plan['context'])
        program = O.read_json(R.ROOT / plan['context']['interaction_path'])
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            folder = root / 'data/synthetic-source-review'
            folder.mkdir(parents=True)
            for image in plan['context']['images']:
                path = root / image['path']
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((R.ROOT / image['path']).read_bytes())
            R.write_new(folder / 'plan.json', plan)
            R.write_new(folder / 'schema.json', plan['schema'])
            R.write_new(folder / 'model-result.json', result)
            usage = {'prompt_tokens': 1, 'completion_tokens': 1}
            receipt = {'status': 'complete_accounted', 'model': R.MODEL, 'usage': usage, 'estimated_cost_usd': '0.000001'}
            R.write_new(folder / 'receipt.json', receipt)
            proposal['reasoning'] = {'provider': 'meta_api', 'model': R.MODEL, 'geometric_joint_fit_performed': False,
                'plan_path': 'data/synthetic-source-review/plan.json', 'plan_sha256': R.sha(folder / 'plan.json'),
                'model_result_path': 'data/synthetic-source-review/model-result.json', 'model_result_sha256': R.sha(folder / 'model-result.json'),
                'request_receipt': 'data/synthetic-source-review/receipt.json', 'usage': usage, 'estimated_cost_usd': '0.000001'}
            O._verify_proposal(proposal, program, root)
            modified = copy.deepcopy(proposal)
            modified['joints'][0]['state_hints'][0]['state'] = 'open'
            with self.assertRaisesRegex(ValueError, 'differs from saved model'):
                O._verify_proposal(modified, program, root)


if __name__ == '__main__':
    unittest.main()
