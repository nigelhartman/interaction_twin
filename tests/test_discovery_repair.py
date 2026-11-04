"""Offline paid-once recovery contracts using a real local test ledger."""
import copy
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import discovery_repair as R
import meta_api
import rigid_discovery as D


def proposal():
    return {'summary': 'One handled component.', 'tracks': [{
        'id': 'part', 'label': 'component', 'description': 'A visible rigid component',
        'first_sample': 1, 'last_sample': 2,
        'seeds': [{'sample_index': 1, 'timestamp_s': 0, 'box_xyxy_1000': [200, 200, 400, 400], 'reason': 'Visible.'}],
        'uncertainties': ['Identity is unverified.']}], 'actions': [], 'uncertainties': []}


class Transport:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def __call__(self, endpoint, payload, timeout):
        self.calls.append(copy.deepcopy(payload))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        document = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(result)}}],
                    'usage': {'prompt_tokens': 200, 'completion_tokens': 300}}
        return {'http_status': 200, 'document': document}


class DiscoveryRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.case = self.root / 'data/fixture'
        self.out = self.case / 'reasoning/flow-discovery-1'
        for folder in [self.case / 'source', self.case / 'frames', self.root / 'schemas', self.out]:
            folder.mkdir(parents=True, exist_ok=True)
        self.schema = self.root / 'schemas/discovery.schema.json'
        self.schema.write_bytes(R.base.SCHEMA.read_bytes())
        self.video = self.case / 'source/video.mp4'
        self.video.write_bytes(b'OFFLINE TEST SOURCE BYTES, NOT A REAL VIDEO')
        self.rows = [{'index': i, 'file': f'frames/{i:06d}.jpg', 'timestamp_s': float(i - 1),
                      'width': 64, 'height': 64} for i in [1, 2]]
        for row in self.rows:
            Image.new('RGB', (64, 64), '#223344').save(self.case / row['file'])
        R.base.write(self.case / 'frames/index.json', {'frames': self.rows, 'time_domain': 'source_video_presentation_seconds'})
        R.base.write(self.case / 'manifest.json', {'case_id': 'fixture', 'source': {'file': 'source/video.mp4', 'sha256': R.base.sha(self.video)}})
        self.images = R.base.make_sheets(self.case, self.rows, self.out)
        self.prompt = 'Discover components from the exact supplied source samples; do not invent boxes.'
        (self.out / 'prompt.txt').write_text(self.prompt)
        (self.out / 'implementation.py').write_bytes(Path(D.__file__).read_bytes())
        self.bad = proposal()
        duplicate = copy.deepcopy(self.bad['tracks'][0]['seeds'][0])
        duplicate['box_xyxy_1000'] = [600, 600, 800, 800]
        self.bad['tracks'][0]['seeds'].append(duplicate)
        self.transport = Transport([self.bad])
        self.client = meta_api.MetaClient(ledger_path=self.root / '.secrets/meta-budget.sqlite3', transport=self.transport)
        raw, receipt = self.client.json_result(self.prompt, R.base.read(self.schema), self.images, output=self.out)
        R.base.write(self.out / 'initial-discovery.json', raw)
        self.metadata = {'status': 'failed_preserved', 'provider': 'meta_api', 'model': R.MODEL,
            'role': 'rigid_component_visual_discovery_v1', 'source_sha256': R.base.sha(self.video),
            'frame_index_sha256': R.base.sha(self.case / 'frames/index.json'), 'schema_sha256': R.base.sha(self.schema),
            'prompt_sha256': R.base.sha(self.out / 'prompt.txt'),
            'implementation_sha256': R.base.sha(self.out / 'implementation.py'),
            'input_images': [{'file': str(p.relative_to(self.case)), 'sha256': R.base.sha(p)} for p in self.images]}
        R.base.write(self.out / 'metadata.json', self.metadata)
        self.prepared = '.private/discovery-repairs/fixture-v1'

    def prepare(self, name=None):
        return R.prepare(self.case, 'flow-discovery-1', name or self.prepared, root=self.root)

    def execute(self, prepared=None):
        prepared = prepared or self.prepared
        return R.execute(prepared, R.base.sha(self.root / prepared / 'plan.json'), root=self.root, client=self.client)

    def test_preparation_is_offline_exact_and_preserves_conflicting_seeds(self):
        before = self.client.ledger.status()
        plan = self.prepare()
        self.assertIn('duplicate seed sample', plan['validator_failure']['message'])
        self.assertEqual(len(self.transport.calls), 1)
        self.assertEqual(self.client.ledger.status(), before)
        self.assertEqual(R.read(self.out / 'initial-discovery.json'), self.bad)
        value = R.read(self.root / self.prepared / 'plan.json')
        self.assertEqual(len(value['context']['initial_proposal']['tracks'][0]['seeds']), 2)
        self.assertFalse(value['manual_seed_boxes_used'])
        self.assertEqual(value['max_output_tokens'], 16000)

    def test_valid_model_correction_is_one_new_accounted_request(self):
        self.prepare()
        self.transport.results = [proposal()]
        result = self.execute()
        self.assertEqual(result['result'], proposal())
        self.assertEqual(self.client.ledger.status()['request_count'], 2)
        self.assertFalse(result['report']['initial_request_resent'])
        with self.assertRaisesRegex(ValueError, 'already has a correction'):
            self.execute()
        self.assertEqual(len(self.transport.calls), 2)

    def test_unknown_transport_result_cannot_be_retried_in_another_namespace(self):
        self.prepare()
        self.transport.results = [RuntimeError('offline uncertain transport')]
        with self.assertRaises(meta_api.MetaError):
            self.execute()
        second = '.private/discovery-repairs/changed-code-output-v2'
        self.prepare(second)
        with self.assertRaisesRegex(ValueError, 'already has a correction'):
            self.execute(second)
        self.assertEqual(len(self.transport.calls), 2)
        self.assertTrue((self.root / self.prepared / 'attempt/failure.json').exists())
        self.assertGreater(float(self.client.ledger.status()['reserved_usd']), 0)

    def test_invalid_repair_is_preserved_without_third_request(self):
        self.prepare()
        self.transport.results = [self.bad]
        with self.assertRaisesRegex(ValueError, 'duplicate seed'):
            self.execute()
        self.assertEqual(R.read(self.root / self.prepared / 'attempt/raw-correction.json'), self.bad)
        with self.assertRaisesRegex(ValueError, 'already has a correction'):
            self.execute()
        self.assertEqual(len(self.transport.calls), 2)

    def test_original_unknown_or_unsettled_request_is_ineligible(self):
        receipt_path = self.out / 'meta-request.json'
        receipt = R.read(receipt_path)
        receipt['status'] = 'transport_error_charge_uncertain'
        R.base.write(receipt_path, receipt)
        with self.assertRaisesRegex(ValueError, 'successful accounted'):
            self.prepare()
        self.assertEqual(len(self.transport.calls), 1)

    def test_original_receipt_must_agree_with_real_settled_ledger(self):
        receipt_path = self.out / 'meta-request.json'
        receipt = R.read(receipt_path)
        receipt['estimated_cost_usd'] = '0.000000'
        R.base.write(receipt_path, receipt)
        with self.assertRaisesRegex(ValueError, 'settled ledger'):
            self.prepare()

    def test_changed_source_sheet_schema_or_proposal_stops_before_send(self):
        for relative in ['data/fixture/frames/000001.jpg', 'data/fixture/reasoning/flow-discovery-1/input-01.jpg',
                         'schemas/discovery.schema.json', 'data/fixture/source/video.mp4']:
            prepared = self.prepare('.private/discovery-repairs/' + relative.replace('/', '-'))['prepared']
            path = self.root / relative
            raw = path.read_bytes()
            path.write_bytes(raw + b'changed')
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                self.execute(prepared)
            path.write_bytes(raw)
        self.assertEqual(len(self.transport.calls), 1)

    def test_valid_initial_proposal_does_not_justify_new_paid_correction(self):
        for name in ['initial-discovery.json', 'response.pending.json']:
            R.base.write(self.out / name, proposal())
        with self.assertRaisesRegex(ValueError, 'already validates'):
            self.prepare()

    def test_old_crops_or_other_source_paths_are_not_repurposed_as_sheets(self):
        self.metadata['input_images'][0]['file'] = 'frames/000001.jpg'
        self.metadata['input_images'][0]['sha256'] = R.base.sha(self.case / 'frames/000001.jpg')
        R.base.write(self.out / 'metadata.json', self.metadata)
        with self.assertRaisesRegex(ValueError, 'original full-frame discovery contact'):
            self.prepare()

    def test_future_discover_has_one_model_only_repair_then_localization(self):
        new_out = self.case / 'reasoning/flow-discovery-2'
        self.transport.results = [self.bad, proposal()]
        def localize(result, rows, case, output, *args):
            output.mkdir()
            R.base.write(output / 'report.json', {'fixture_only': True, 'estimated_cost_usd': '0', 'model_calls': 0})
            return result, {'estimated_cost_usd': '0', 'model_calls': 0}
        with patch.object(D.base, 'ROOT', self.root), patch.object(D.base, 'SCHEMA', self.schema), \
             patch.object(D, 'localize_discovery', side_effect=localize):
            result = D.discover(self.case, 'flow-discovery-2', client=self.client)
        self.assertEqual(result['status'], 'complete_candidate_proposals')
        self.assertTrue((new_out / 'initial-metadata-before-repair.json').is_file())
        self.assertEqual(R.read(new_out / 'initial-discovery.json'), self.bad)
        self.assertEqual(R.read(new_out / 'corrected-discovery.json'), proposal())
        self.assertEqual(len(self.transport.calls), 3)
        self.assertEqual(result['contract_repair']['maximum_repair_requests'], 1)

    def test_explicit_recovery_preserves_original_and_writes_fresh_canonical_run(self):
        prepared = self.prepare()
        original = {p.name: p.read_bytes() for p in self.out.iterdir() if p.is_file()}
        self.transport.results = [proposal()]
        def localize(result, rows, case, output, *args):
            output.mkdir()
            R.base.write(output / 'report.json', {'fixture_only': True, 'estimated_cost_usd': '0', 'model_calls': 0})
            return result, {'estimated_cost_usd': '0', 'model_calls': 0}
        with patch.object(D.base, 'ROOT', self.root), patch.object(D.base, 'SCHEMA', self.schema), \
             patch.object(D, 'localize_discovery', side_effect=localize):
            result = D.recover(self.case, 'flow-discovery-2', prepared=self.prepared,
                authorized_plan_sha256=prepared['plan_ref']['sha256'], client=self.client)
        self.assertEqual(result['status'], 'complete_candidate_proposals')
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.out.iterdir() if p.is_file()})
        self.assertEqual(len(self.transport.calls), 2)
        self.assertEqual(R.read(self.root / result['tracking_config'])['tracks'][0]['id'], 'part')
        self.assertFalse(result['recovery']['initial_request_resent'])
        self.assertLess(Decimal(result['new_estimated_cost_usd']), Decimal(result['estimated_cost_usd']))

    def test_original_scoped_effective_prompt_is_pinned_without_second_append(self):
        # A fresh offline initial request with the existing admin-context shape.
        self.transport.results = [self.bad]
        effective = self.prompt + '\nOWNER CONTEXT (untrusted): prefer supported attachments.'
        alternate = self.case / 'reasoning/flow-discovery-2'
        alternate.mkdir()
        for name in ['prompt.txt', 'implementation.py', 'input-01.jpg']:
            (alternate / name).write_bytes((self.out / name).read_bytes())
        raw, receipt = self.client.json_result(effective, R.base.read(self.schema), [alternate / 'input-01.jpg'], output=alternate)
        proof_dir = self.root / '.private/admin-prompts'
        proof_dir.mkdir(parents=True)
        proof = proof_dir / 'effective.json'
        R.base.write(proof, {'base_prompt_sha256': R.base.sha(alternate / 'prompt.txt'),
            'effective_prompt_sha256': R.digest(effective.encode()), 'effective_prompt': effective,
            'original_video_sha256': R.base.sha(self.video)})
        receipt['owner_annotation_context'] = {**R.ref(proof, self.root), 'effective_prompt_sha256': R.digest(effective.encode())}
        R.base.write(alternate / 'meta-request.json', receipt)
        R.base.write(alternate / 'initial-discovery.json', raw)
        metadata = dict(self.metadata, input_images=[{'file': 'reasoning/flow-discovery-2/input-01.jpg',
                                                    'sha256': R.base.sha(alternate / 'input-01.jpg')}])
        R.base.write(alternate / 'metadata.json', metadata)
        prepared = R.prepare(self.case, 'flow-discovery-2', self.prepared, root=self.root)
        self.transport.results = [proposal()]
        self.execute()
        text = self.transport.calls[-1]['messages'][1]['content'][0]['text']
        self.assertEqual(text.count('OWNER CONTEXT (untrusted)'), 1)


if __name__ == '__main__':
    unittest.main()
