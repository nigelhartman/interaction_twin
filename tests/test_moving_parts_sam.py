import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import geometry_intelligence as gi
import moving_parts_sam as moving


class MovingPartsSamTests(unittest.TestCase):
    def frame_and_native(self, root, count=2):
        (root / 'data').mkdir(exist_ok=True)
        image = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
        source = root / 'data/source.png'
        Image.fromarray(image).save(source)
        directory = root / 'data/job'
        (directory / 'inference').mkdir(parents=True)
        records = []
        for i in range(count):
            mask = np.zeros((20, 30), np.uint8)
            mask[4:12, 3 + i * 10:9 + i * 10] = 255
            path = directory / 'inference' / f'instance-{i + 1:06d}-mask.png'
            Image.fromarray(mask).save(path)
            records.append({'instance_id': str(7 + i), 'mask': path.name, 'mask_sha256': gi.sha256(path),
                            'mask_foreground_pixels': 48, 'confidence': .8,
                            'box_xyxy_half_open': [3 + i * 10, 4, 9 + i * 10, 12]})
        result = {'status': 'completed' if count else 'completed_zero_matches', 'instances': records,
                  'raw_response_sha256': 'a' * 64}
        gi.write(directory / 'inference/metadata.json', result)
        job = {'frame': {'id': 'jabra::sample', 'source_evidence_id': 'sample', 'timestamp_s': 1.25,
                         'image_url': '/files/data/source.png', 'sha256': gi.sha256(source), 'width': 30, 'height': 20}}
        return directory, job, result, image

    def test_all_native_instances_retained_without_temporal_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            directory, job, result, image = self.frame_and_native(root)
            before = (root / 'data/source.png').read_bytes()
            frame = moving.export_frame(root, directory, job, result)
            self.assertEqual([i['native_instance_id'] for i in frame['instances']], ['7', '8'])
            self.assertTrue(all(i['identity_scope'] == 'this_image_response_only' for i in frame['instances']))
            self.assertFalse(frame['cross_frame_identity_established'])
            self.assertEqual(frame['timestamp_s'], 1.25)
            self.assertEqual(frame['segmentation_audit_status'], 'not_model_audited')
            self.assertEqual((root / 'data/source.png').read_bytes(), before)
            with Image.open(directory / 'overlay.png') as opened:
                self.assertEqual(opened.size, (30, 20))

    def test_zero_native_instances_remains_zero_without_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            directory, job, result, image = self.frame_and_native(root, 0)
            frame = moving.export_frame(root, directory, job, result)
            self.assertEqual(frame['instances'], [])
            with Image.open(directory / 'overlay.png') as opened:
                np.testing.assert_array_equal(np.asarray(opened), image)

    def test_native_mask_tampering_and_resizing_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            directory, job, result, _ = self.frame_and_native(root, 1)
            mask_path = directory / 'inference' / result['instances'][0]['mask']
            Image.new('L', (29, 20), 255).save(mask_path)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                moving.export_frame(root, directory, job, result)
            result['instances'][0]['mask_sha256'] = gi.sha256(mask_path)
            with self.assertRaisesRegex(ValueError, 'source-size'):
                moving.export_frame(root, directory, job, result)

    def test_actual_plan_is_bounded_and_preserves_advisory_prompts(self):
        root = Path(__file__).resolve().parents[1]
        plan = moving.build_plan(root)
        self.assertEqual(len(plan['jobs']), 8)
        self.assertEqual(sum(j['group'] == 'proposed_part' for j in plan['jobs']), 6)
        prompts = {p['id']: p['sam_prompt'] for p in plan['proposals']}
        for job in plan['jobs']:
            if job['group'] == 'proposed_part':
                self.assertEqual(job['concept'], prompts[job['proposal_id']])
            else:
                self.assertEqual(job['frame']['timestamp_s'], 66.9669)
                self.assertEqual(job['frame']['source_evidence_id'], 'dense_source_000024')
        self.assertFalse(plan['manual_boxes_or_crops'])

    def test_uncertain_budget_stops_before_another_request_without_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root, calls = Path(d), []
            (root / 'data').mkdir()
            output = root / 'data/batch'
            output.mkdir()
            jobs = []
            for n in range(2):
                directory = output / str(n)
                directory.mkdir()
                gi.write(directory / 'input.json', {})
                jobs.append({'group': 'proposed_part', 'proposal_id': 'part', 'concept': 'lid', 'prompt_origin': 'model',
                             'relative_output': str(n), 'frame': {'id': 'frame'},
                             'input_manifest_sha256': gi.sha256(directory / 'input.json')})
            plan = {'source_bindings': {moving.REVIEW: 'a' * 64}, 'jobs': jobs, 'proposals': [{'id': 'part'}],
                    'proposal_review_url': '/files/data/review.json'}
            # A fixture binding to a real local placeholder; no credential/network.
            source = root / moving.REVIEW
            source.parent.mkdir(parents=True)
            source.write_text('{}')
            plan['source_bindings'][moving.REVIEW] = gi.sha256(source)
            gi.write(output / 'plan.json', plan)
            (output / 'implementation.py').write_bytes(Path(moving.__file__).read_bytes())
            gi.write(output / 'metadata.json', {'status': 'prepared_not_sent',
                     'plan_sha256': gi.sha256(output / 'plan.json'),
                     'implementation_sha256': gi.sha256(moving.__file__),
                     'sam_adapter_sha256': gi.sha256(moving.meta_sam_api.__file__)})
            class FakeClient:
                def __init__(self, timeout):
                    self.blocked = False
                    self.ledger = self
                def status(self):
                    return {'blocked': self.blocked}
                def sam_callback(self, directory, images):
                    def callback(payload):
                        calls.append(payload)
                        self.blocked = True
                        raise RuntimeError('Explicit uncertain transport fixture')
                    return callback
            def fake_segmenter(path, concept, output, request, **kwargs):
                return request({'fixture': True})
            result = moving.run(root, output, client_factory=FakeClient, segmenter=fake_segmenter)
            self.assertEqual(len(calls), 1)
            self.assertEqual(result['requests_attempted'], 1)
            self.assertEqual(result['status'], 'halted_budget_or_uncertain_usage')
            with self.assertRaisesRegex(ValueError, 'no automatic retry'):
                moving.run(root, output, client_factory=FakeClient, segmenter=fake_segmenter)
            self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
