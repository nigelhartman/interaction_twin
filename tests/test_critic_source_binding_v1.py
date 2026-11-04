"""Actual encoded critic packets cannot be relabelled as another source/review."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import critic_source_binding_v1 as V
import source_observation_refinement_v3 as T


class CriticSourceBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'data').mkdir()
        self.counter = 0
        self.image = self.root / 'data/frame.png'
        Image.new('RGB', (12, 9), '#456789').save(self.image)
        image = self.ref(self.image)
        self.program = {'evidence': [{'source_index': 1, 'timestamp_s': 0.0,
                         'frame_path': image['path'], 'frame_sha256': image['sha256'],
                         'observations': [{'track_id': 'bud', 'mask_path': image['path'],
                                           'mask_sha256': image['sha256']}]}]}
        p = self.write(self.program)
        identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
        asset = {'path': 'data/unread-native.glb', 'sha256': 'a' * 64, 'bytes': 1}
        self.scene = {'original_program': p, 'source': {'sha256': 'b' * 64},
                      'components': [{'id': 'bud', 'asset': asset, 'poses': [
                          {'source_sample_index': 1, 'camera_from_native_mesh': identity}]}]}
        scene = self.write(self.scene)
        self.packet = {'format': 'interaction-twin-jabra-earbud-source-orientation-plan',
                       'max_output_tokens': 16000, 'maximum_new_requests': 1,
                       'automatic_retries': False, 'model': 'muse-spark-1.3',
                       'prompt': 'Compare this source to its automatic candidate.',
                       'schema': {'type': 'object', 'properties': {}, 'additionalProperties': False},
                       'images': [image], 'context': {'source_scene': scene, 'component_id': 'bud',
                           'native_asset': asset, 'source_samples': [{
                               'source_sample_index': 1, 'timestamp_s': 0.0, 'source_rgb': image,
                               'baseline_pose': identity, 'source_mask': image, 'source_crop': image}]}}
        wire = T.wire(self.packet['prompt'], self.packet['schema'], self.packet['images'], root=self.root)
        self.receipt = {'status': 'complete_accounted', 'model': 'muse-spark-1.3',
                        'image_inputs': wire['image_inputs'], 'payload_sha256': wire['payload_sha256']}
        self.document_core = {'evidence_program': p, 'baseline_scene_ref': scene,
                              'source': self.scene['source']}

    def ref(self, path):
        raw = path.read_bytes()
        return {'path': path.relative_to(self.root).as_posix(), 'sha256': hashlib.sha256(raw).hexdigest(),
                'bytes': len(raw)}

    def write(self, value):
        self.counter += 1
        path = self.root / f'data/object-{self.counter}.json'
        path.write_text(json.dumps(value, sort_keys=True))
        return self.ref(path)

    def document(self, packet=None, receipt=None):
        review = {'plan': self.write(self.packet if packet is None else packet),
                  'receipt': self.write(self.receipt if receipt is None else receipt),
                  'model_result': self.write({'physical_accuracy_verified': False}),
                  'automatic_approval': False}
        return {**self.document_core, 'visual_review_refs': [self.write(review)]}

    def test_real_image_and_payload_reproduction_passes_without_client_or_transport(self):
        result = V.validate_files(self.document(), root=self.root)
        self.assertTrue(result[0]['accounted_payload_reproduced'])
        self.assertTrue(result[0]['source_and_baseline_bound'])

    def test_changed_prompt_cannot_reuse_an_accounted_image_receipt(self):
        packet = copy.deepcopy(self.packet)
        packet['prompt'] = 'Different question about another object.'
        with self.assertRaisesRegex(ValueError, 'wire cannot be reproduced'):
            V.validate_files(self.document(packet), root=self.root)

    def test_copied_image_receipt_hash_is_rejected(self):
        receipt = copy.deepcopy(self.receipt)
        receipt['image_inputs'][0]['source_sha256'] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'image order/hash differs'):
            V.validate_files(self.document(receipt=receipt), root=self.root)

    def test_source_pts_cannot_be_relabelled(self):
        packet = copy.deepcopy(self.packet)
        packet['context']['source_samples'][0]['timestamp_s'] = 0.5
        with self.assertRaisesRegex(ValueError, 'sample/PTS'):
            V.validate_files(self.document(packet), root=self.root)

    def test_private_or_unregistered_output_packets_rejected_before_read(self):
        read = V._reader(self.root)
        for path in ['.secrets/meta.env', 'data/../.secrets/meta.env', 'output/other/plan.json']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                read({'path': path, 'bytes': 10, 'sha256': 'a' * 64})


if __name__ == '__main__':
    unittest.main()
