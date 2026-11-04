"""Offline real-adapter checks with model responses explicitly replaced by fixtures."""
import copy
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import pipeline
import stage5


def module(name, script):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / script)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


D = module('tested_meta_discover', 'stage2_meta_reason.py')
P = module('tested_meta_propose', 'stage5_meta_propose.py')
V = module('tested_meta_select', 'stage3_meta_select_track_views.py')
C = module('tested_preserved_compiler', 'stage5_compile.py')


class MetaReasoningContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'data')
        self.root = Path(self.temp.name)
        self.case = self.root / 'data/clip'
        self.rows = []
        for index in [1, 2, 3]:
            image = self.case / f'frames/{index}.png'
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((120, 80, 3), 80 + index * 30, np.uint8)).save(image)
            self.rows.append({'index': index, 'timestamp_s': index - .75,
                              'width': 80, 'height': 120, 'file': f'frames/{index}.png'})
        D.write(self.case / 'frames/index.json', {'frames': self.rows})
        self.source_hash = 'a' * 64
        D.write(self.case / 'manifest.json', {'case_id': 'clip', 'source': {'sha256': self.source_hash},
                                            'sampling': {'fps': 1}})
        self.discovery = {'summary': 'Fixture visible part.', 'tracks': [
            {'id': 'part', 'label': 'rigid part', 'description': 'One separately observed component.',
             'first_sample': 1, 'last_sample': 3,
             'seeds': [{'sample_index': 1, 'timestamp_s': .25, 'box_xyxy_1000': [187.5, 166.666667, 562.5, 500],
                        'reason': 'Fixture clear view.'}], 'uncertainties': ['Physical identity unverified.']}],
            'actions': [{'description': 'Visible movement.', 'participant_track_ids': ['part'],
                         'evidence_samples': [1, 2], 'uncertainties': ['Contact unknown.']}], 'uncertainties': []}
        self.calls = []
        self.localization_calls = []

    def tearDown(self):
        self.temp.cleanup()

    def args(self, run='discovery'):
        return types.SimpleNamespace(case='clip', run=run, timeout_s=30,
                                     model='muse-spark-1.3', max_output_tokens=16000)

    def client(self, value):
        calls = self.calls

        class Client:
            def __init__(self, **kwargs):
                self.options = kwargs

            def json_result(self, prompt, schema, images, **kwargs):
                calls.append({'prompt': prompt, 'schema': schema, 'images': list(images), **kwargs})
                receipt = {'usage': {'input_tokens': 100, 'output_tokens': 200}, 'estimated_cost_usd': '0.002000'}
                D.write(Path(kwargs['output']) / 'response.pending.json', value)
                D.write(Path(kwargs['output']) / 'meta-request.json', receipt)
                return copy.deepcopy(value), receipt

        return Client

    def discover(self, value=None, run='discovery', localized_value=None):
        def localize(raw, rows, case, output, client, model, max_output_tokens):
            self.localization_calls.append({'initial': copy.deepcopy(raw), 'rows': rows,
                                             'case': case, 'output': output, 'model': model,
                                             'max_output_tokens': max_output_tokens})
            report = {'estimated_cost_usd': '0.001000', 'model_calls': 1,
                      'explicit_synthetic_localization_fixture': True, 'manual_boxes_used': False}
            D.write(output / 'report.json', report)
            return copy.deepcopy(raw if localized_value is None else localized_value), report
        with patch.object(D, 'ROOT', self.root), patch.object(D, 'MetaClient', self.client(value or self.discovery)), \
             patch.object(D, 'localize_discovery', side_effect=localize), redirect_stdout(io.StringIO()):
            D.discover(self.args(run))
        out = self.case / 'reasoning' / run
        record = D.read(out / 'discovery.json')
        config_path = self.root / D.read(out / 'metadata.json')['tracking_config']
        return record, config_path

    def segment_fixture(self, *, withheld=False):
        record, config = self.discover()
        out = self.case / 'segmentation/run'
        cfg = D.read(config)
        D.write(out / 'annotations.json', cfg)
        D.write(out / 'metadata.json', {'status': 'complete', 'source_sha256': self.source_hash,
                                      'model': 'sam-3.1', 'device': 'hosted_Meta_Model_API',
                                      'checkpoint_sha256': None, 'checkpoint_identity': 'provider_managed_not_disclosed',
                                      'seed_origin': 'meta_api_model_proposal'})
        frames = []
        for row in self.rows:
            mask = np.zeros((120, 80), np.uint8)
            mask[20:60, 15:45] = 255
            name = f'masks/{row["index"]}.png'
            (out / name).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(out / name)
            frames.append({'sample_index': row['index'], 'timestamp_s': row['timestamp_s'], 'raw_mask_file': name,
                           'raw_area_pixels': 1200, 'raw_bbox_xyxy': [15, 20, 45, 60],
                           'status': 'withheld_due_to_occlusion_or_ambiguity' if withheld or row['index'] == 3 else 'candidate_visible'})
        D.write(out / 'tracks.json', {'tracks': [{'track_id': 'part', 'frames': frames}]})
        D.write(out / 'frames.json', {'frames': [{'sample_index': row['index'], 'tracks': [
            {'track_id': 'part', 'status': 'candidate_geometric_gate_only'}]} for row in self.rows]})
        return out, config

    def test_discovery_saves_actual_meta_identity_exact_seeds_and_cost_receipt(self):
        record, path = self.discover()
        self.assertEqual(record['provider'], 'meta_api')
        self.assertEqual(record['inference_location'], 'hosted_Meta_Model_API')
        self.assertEqual(record['estimated_cost_usd'], '0.003000')
        self.assertEqual(record['result'], self.discovery)
        cfg = D.read(path)
        self.assertEqual(cfg, D.tracking_config(self.discovery, self.rows, 'clip', D.sha(self.case / 'reasoning/discovery/discovery.json')))
        self.assertFalse(cfg['manual_seed_boxes_used'])
        self.assertEqual(self.calls[0]['model'], 'muse-spark-1.3')
        self.assertEqual(self.calls[0]['max_output_tokens'], 16000)
        self.assertEqual(len(self.localization_calls), 1)
        self.assertEqual(self.localization_calls[0]['initial'], self.discovery)
        self.assertEqual(record['localization']['model_calls'], 1)

    def test_independent_localization_output_is_authoritative_and_report_hash_bound(self):
        localized = copy.deepcopy(self.discovery)
        localized['tracks'][0]['seeds'][0]['box_xyxy_1000'] = [200, 200, 600, 550]
        localized['tracks'][0]['seeds'][0]['reason'] = 'Independent full-frame fixture localization.'
        record, config = self.discover(localized_value=localized)
        self.assertEqual(record['result'], localized)
        self.assertEqual(D.read(self.case / 'reasoning/discovery/initial-discovery.json'), self.discovery)
        cfg = D.read(config)
        self.assertEqual(cfg['tracks'][0]['prompts'][0]['box_xyxy'], [16, 24, 48, 66])
        report = self.case / 'reasoning/discovery' / record['localization']['report']
        self.assertEqual(D.sha(report), record['localization']['report_sha256'])
        self.assertFalse(record['localization']['manual_boxes_used'])
        self.assertEqual(record['estimated_cost_usd'], '0.003000')

    def test_invalid_discovery_never_exports_tracking_configuration(self):
        invalid = copy.deepcopy(self.discovery)
        invalid['tracks'][0]['seeds'][0]['timestamp_s'] = 999
        with self.assertRaisesRegex(ValueError, 'timestamp'):
            self.discover(invalid)
        out = self.case / 'reasoning/discovery'
        self.assertFalse((out / 'discovery.json').exists())
        self.assertEqual(D.read(out / 'metadata.json')['status'], 'failed')
        self.assertFalse((self.root / 'annotations').exists())

    def test_audit_covers_only_associated_visible_sam2_compatible_status(self):
        out, _ = self.segment_fixture()
        answer = {'summary': 'Fixture visual audit.', 'observations': [
            {'track_id': 'part', 'sample_index': index, 'verdict': 'plausible_visible_mask', 'reason': 'Fixture.'}
            for index in [1, 2]], 'limitations': ['Not physical accuracy.']}
        self.calls.clear()
        with patch.object(D, 'ROOT', self.root), patch.object(D, 'MetaClient', self.client(answer)), redirect_stdout(io.StringIO()):
            D.audit(self.args('run'))
        evidence = D.read(out / 'model-audit-v1/evidence.json')
        self.assertEqual([row['sample_index'] for row in evidence], [1, 2])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(D.read(out / 'model-audit-v1/audit.json')['provider'], 'meta_api')

    def test_empty_audit_never_constructs_paid_client(self):
        out, _ = self.segment_fixture(withheld=True)
        with patch.object(D, 'ROOT', self.root), patch.object(D, 'MetaClient', side_effect=AssertionError('No paid request')), redirect_stdout(io.StringIO()):
            D.audit(self.args('run'))
        result = D.read(out / 'model-audit-v1/audit.json')
        self.assertFalse(result['model_call_made'])
        self.assertEqual(result['estimated_cost_usd'], '0.000000')
        self.assertEqual(result['result']['observations'], [])

    def test_review_keeps_withheld_associations_out_of_candidate_masks(self):
        out, _ = self.segment_fixture()
        with patch.object(D, 'ROOT', self.root), patch.object(D.subprocess, 'run'), redirect_stdout(io.StringIO()):
            D.review(self.args('run'))
        frames = D.read(out / 'frames.json')['frames']
        self.assertEqual(frames[0]['tracks'][0]['status'], 'candidate_geometric_gate_only')
        self.assertIn('source_association_withheld', frames[2]['tracks'][0]['reasons'])
        self.assertIsNone(frames[2]['tracks'][0]['candidate_mask_file'])

    def test_meta_selector_verifies_full_discovery_configuration_and_native_model_identity(self):
        out, _ = self.segment_fixture()
        candidates, _, _, automatic = V.collect_track(self.case, 'run', 'part')
        self.assertTrue(automatic)
        self.assertEqual(len(candidates), 2)
        provenance = candidates[0]['mask_provenance']
        self.assertFalse(provenance['manual_seed_boxes_used'])
        self.assertEqual(provenance['prompt_provider'], 'hosted_Meta_Model_API')
        self.assertEqual(provenance['model'], 'sam-3.1')
        self.assertIsNone(provenance['checkpoint_sha256'])
        cfg = D.read(out / 'annotations.json')
        cfg['tracks'][0]['prompts'][0]['box_xyxy'][0] += 1
        D.write(out / 'annotations.json', cfg)
        with self.assertRaisesRegex(ValueError, 'exact Meta discovery'):
            V.collect_track(self.case, 'run', 'part')

    def test_meta_selector_rejects_discovery_hash_or_provider_mismatch(self):
        out, _ = self.segment_fixture()
        path = self.case / 'reasoning/discovery/discovery.json'
        original = path.read_text()
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'hash-matched'):
            V.collect_track(self.case, 'run', 'part')
        path.write_text(original)
        record = D.read(path)
        record['provider'] = 'codex_cli'
        D.write(path, record)
        cfg = D.read(out / 'annotations.json')
        cfg['discovery_record_sha256'] = D.sha(path)
        D.write(out / 'annotations.json', cfg)
        with self.assertRaisesRegex(ValueError, 'provider/source/frame'):
            V.collect_track(self.case, 'run', 'part')

    def proposal(self):
        return {'summary': 'Fixture observed movement.', 'relationships': [],
                'variables': [{'id': 'visible_state', 'subject_track_id': 'part', 'name': 'Visual state', 'values': ['unknown', 'held']}],
                'assertions': [{'id': 'held_view', 'variable_id': 'visible_state', 'value': 'held', 'evidence_sample': 2,
                                'status': 'visual_model_proposal', 'rationale': 'Fixture visible interpretation.'}],
                'steps': [{'id': 'movement', 'title': 'Observed movement', 'description': 'Fixture movement.',
                           'participant_track_ids': ['part'], 'evidence_samples': [1, 2], 'preconditions': [], 'postconditions': [],
                           'limitations': ['No physical contact conclusion.']}], 'limitations': ['Point observations only.']}

    def propose(self, value):
        _, config = self.discover()
        args = types.SimpleNamespace(case=self.case, discovery=self.case / 'reasoning/discovery/discovery.json',
                                     annotations=config, out=self.case / 'procedure-proposal', timeout_s=30,
                                     model='muse-spark-1.3', max_output_tokens=16000)
        with patch.object(P, 'ROOT', ROOT), patch.object(P, 'MetaClient', self.client(value)), redirect_stdout(io.StringIO()):
            P.propose(args)
        return args

    def test_procedure_uses_real_validator_and_exports_provider_cost_and_evidence(self):
        args = self.propose(self.proposal())
        record = D.read(args.out / 'proposal.json')
        self.assertEqual(record['provider'], 'meta_api')
        self.assertEqual(record['status'], 'complete_validated_model_proposal')
        self.assertEqual(record['estimated_cost_usd'], '0.002000')
        self.assertTrue(stage5.validate_proposal(record['result'], self.rows, D.read(args.annotations)['tracks']))
        self.assertEqual(record['frame_index_sha256'], D.sha(self.case / 'frames/index.json'))

    def test_procedure_rejects_invented_evidence_instead_of_exporting_fallback(self):
        invalid = self.proposal()
        invalid['steps'][0]['evidence_samples'] = [100]
        with self.assertRaisesRegex(ValueError, 'evidence'):
            self.propose(invalid)
        out = self.case / 'procedure-proposal'
        self.assertFalse((out / 'proposal.json').exists())
        self.assertEqual(D.read(out / 'metadata.json')['status'], 'failed_no_compiled_procedure')

    def test_generated_meta_viewer_disclosure_changes_only_new_output_not_compiler(self):
        from tests.test_stage5 import fixture
        doc = fixture()
        doc['reasoning'].update(provider='meta_api', model='muse-spark-1.3',
                                inference_location='hosted_Meta_Model_API', role='visual_procedure_reasoning')
        out = self.root / 'viewer'
        out.mkdir()
        source_hash = pipeline.sha(ROOT / 'scripts/stage5_compile.py')
        C.build_review(doc, out)
        before = (out / 'index.html').read_text()
        self.assertIn('temporary Meta reasoning substitute', before)
        self.assertTrue(pipeline.label_generated_meta_viewer(out / 'index.html', doc))
        after = (out / 'index.html').read_text()
        self.assertIn('official Meta Model API inference', after)
        self.assertNotIn('temporary Meta reasoning substitute', after)
        self.assertEqual(source_hash, pipeline.sha(ROOT / 'scripts/stage5_compile.py'))
        self.assertEqual(before.replace('; temporary Meta reasoning substitute.', '; official Meta Model API inference.'), after)
        doc['reasoning']['provider'] = 'codex_cli'
        self.assertFalse(pipeline.label_generated_meta_viewer(out / 'index.html', doc))


if __name__ == '__main__':
    unittest.main()
