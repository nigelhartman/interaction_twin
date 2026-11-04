"""Offline SAM-video wrapper checks; masks are explicitly synthetic fixtures."""
import copy
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts import stage2_meta_video_segment as W


class Client:
    def __init__(self):
        self.requests = []

    def sam_callback(self, directory, *, images=0, video_frames=0):
        self.requests.append({'directory': str(directory), 'images': images, 'video_frames': video_frames})
        return lambda payload: {'explicit_synthetic_transport_fixture': True}


class VideoSegmentContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.case = Path(self.temp.name) / 'video-fixture'
        (self.case / 'frames').mkdir(parents=True)
        self.rows = []
        for index in range(1, 5):
            path = self.case / f'frames/{index:06d}.png'
            pixels = np.zeros((30, 40, 3), dtype=np.uint8)
            pixels[:, :, 0] = index * 35
            pixels[4:12, 3 + index:11 + index, 1:] = 190
            Image.fromarray(pixels).save(path)
            self.rows.append({'index': index, 'file': str(path.relative_to(self.case)),
                              'timestamp_s': (index - 1) * 1.000083, 'width': 40, 'height': 30})
        W.write(self.case / 'frames/index.json', {'time_domain': W.sam.TIME_DOMAIN, 'frames': self.rows})
        W.write(self.case / 'manifest.json', {'source': {'sha256': 'a' * 64}})
        track = {'id': 'part_a', 'label': 'component', 'description': 'Fixture model observation',
                 'sample_range': [1, 4], 'candidate_entity_ids': ['part_candidate'],
                 'identity_status': 'model_proposed_tracklet_physical_identity_unverified',
                 'prompts': [{'sample_index': 2, 'box_xyxy': [3.2, 4.3, 11.7, 12.4],
                              'origin': 'Meta_API_model_proposal', 'source_timestamp_s': 1.000083}],
                 'withhold_ranges': [[4, 4]], 'uncertainties': ['Synthetic fixture only.']}
        second = copy.deepcopy(track)
        second.update(id='part_b', sample_range=[2, 3], candidate_entity_ids=['second_candidate'], withhold_ranges=[])
        self.cfg = {'seed_origin': 'Meta_API_model_proposal', 'manual_seed_boxes_used': False,
                    'manual_geometry_used': False, 'tracks': [track, second]}
        self.config = self.case / 'config.json'
        self.save()
        self.segment_calls = []

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        W.write(self.config, self.cfg)

    def preparation_fixture(self, case, rows, directory, sources):
        directory.mkdir(parents=True)
        path = directory / 'sampled-1fps.mp4'
        path.write_bytes(b'Synthetic encoded-container fixture; no API accepted this file.')
        record = {'video_sha256': W.sha(path), 'fixture_only': True}
        W.write(directory / 'encoding.json', record)
        return path, record

    def segmenter(self, *, wrong_source=False, duplicate=False):
        def fake(video, objects, output, request, **kwargs):
            self.segment_calls.append({'video': video, 'objects': objects, **kwargs})
            request({'fixture_only': True})
            output.mkdir(parents=True)
            mask = np.zeros((30, 40), np.uint8)
            mask[4:12, 3:11] = 255
            result = {'instances': [], 'explicit_synthetic_masks': True}
            # id 0 is missing on API frame 2; id 1 is also returned outside its
            # declared tracklet. Unknown id 99 is preserved but never reassigned.
            pairs = [(0, 0), (0, 1), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (99, 1)]
            if duplicate:
                pairs.append((0, 0))
            for number, (identifier, frame) in enumerate(pairs):
                path = output / f'instance-{number:06d}.png'
                Image.fromarray(mask).save(path)
                source = copy.deepcopy(kwargs['frame_sources'][frame])
                if wrong_source:
                    source['timestamp_s'] += 1
                result['instances'].append({'instance_id': str(identifier), 'frame_index': frame,
                                             'source': source, 'mask': path.name, 'mask_sha256': W.sha(path)})
            W.write(output / 'metadata.json', result)
            return result
        return fake

    def run_fixture(self, name='video-test', **kwargs):
        client = Client()
        metadata = W.run_case(self.case, self.config, name, client=client,
                              segmenter=self.segmenter(**kwargs), video_preparer=self.preparation_fixture)
        return metadata, client, self.case / 'segmentation' / name

    def test_one_video_call_preserves_model_boxes_exact_pts_ids_and_native_masks(self):
        metadata, client, out = self.run_fixture()
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0]['video_frames'], 4)
        self.assertEqual(client.requests[0]['images'], 0)
        self.assertEqual((out / 'annotations.json').read_bytes(), self.config.read_bytes())
        self.assertTrue(metadata['sam_video_propagation_executed'])
        self.assertEqual(metadata['hosted_image_request_count'], 0)
        self.assertEqual(metadata['native_instance_count'], 8)
        self.assertEqual(metadata['unmapped_native_instance_count'], 1)
        call = self.segment_calls[0]
        self.assertEqual(call['objects'][0], {'id': 0, 'frame_index': 1, 'box_xyxy': [3.2, 4.3, 11.7, 12.4]})
        self.assertEqual(call['objects'][1]['id'], 1)
        self.assertEqual(call['propagation_direction'], 'both')
        self.assertEqual([s['timestamp_s'] for s in call['frame_sources']], [r['timestamp_s'] for r in self.rows])
        tracks = W.load(out / 'tracks.json')['tracks']
        self.assertEqual([f['status'] for f in tracks[0]['frames']], ['candidate_visible', 'candidate_visible',
                         'withheld_due_to_occlusion_or_ambiguity', 'withheld_due_to_occlusion_or_ambiguity'])
        for frame in tracks[0]['frames'][2:]:
            self.assertTrue(frame['zero_placeholder_for_missing_or_withheld'])
            self.assertEqual(frame['raw_area_pixels'], 0)
            self.assertFalse(np.asarray(Image.open(out / frame['raw_mask_file'])).any())
        self.assertIsNotNone(tracks[0]['frames'][3]['api_instance'])
        self.assertTrue((out / tracks[0]['frames'][3]['api_instance']['mask_file']).is_file())
        self.assertEqual([f['sample_index'] for f in tracks[1]['frames']], [2, 3])

    def test_wrong_source_and_duplicate_results_fail_without_synthetic_success(self):
        for name, kwargs in [('wrong-source', {'wrong_source': True}), ('duplicate', {'duplicate': True})]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.run_fixture(name, **kwargs)
            out = self.case / 'segmentation' / name
            self.assertEqual(W.load(out / 'metadata.json')['status'], 'failed')
            self.assertFalse((out / 'tracks.json').exists())
            self.assertTrue((out / 'api/result/metadata.json').exists())

    def test_empty_model_discovery_makes_no_video_or_paid_request(self):
        self.cfg['tracks'] = []
        self.save()
        def forbidden(*args, **kwargs):
            raise AssertionError('No video preparation, client or segmentation allowed')
        metadata = W.run_case(self.case, self.config, 'no-tracks', client=None,
                              segmenter=forbidden, video_preparer=forbidden)
        self.assertEqual(metadata['hosted_video_request_count'], 0)
        self.assertFalse(metadata['learned_inference_executed'])

    def test_source_mapping_rejects_changed_dimensions_order_and_oversized_plan(self):
        document = W.load(self.case / 'frames/index.json')
        manifest = W.load(self.case / 'manifest.json')
        for kind in ['time', 'dimension', 'index', 'count']:
            rows = copy.deepcopy(self.rows)
            if kind == 'time': rows[2]['timestamp_s'] = rows[1]['timestamp_s']
            if kind == 'dimension': rows[0]['width'] = 41
            if kind == 'index': rows[1]['index'] = 99
            if kind == 'count': rows = rows * 31
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                W.verify_rows(self.case, rows, document, manifest)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'Local encoder unavailable')
    def test_actual_h264_encode_decode_has_exact_ordered_yuv_frame_hashes(self):
        document = W.load(self.case / 'frames/index.json')
        manifest = W.load(self.case / 'manifest.json')
        width, height, sources = W.verify_rows(self.case, self.rows, document, manifest)
        video, record = W.prepare_video(self.case, self.rows, self.case / 'encoding-check', sources)
        self.assertTrue(video.is_file())
        self.assertEqual((record['width'], record['height'], record['frame_count']), (40, 30, 4))
        self.assertFalse(record['resize_performed'])
        self.assertFalse(record['original_pts_replaced_by_encoded_pts'])
        self.assertEqual(len({r['decoded_yuv420_pixels_sha256'] for r in record['frames']}), 4)
        for index, row in enumerate(record['frames']):
            self.assertEqual(row['submitted_frame_index'], index)
            self.assertEqual(row['source']['timestamp_s'], self.rows[index]['timestamp_s'])
            self.assertEqual(row['reference_yuv420_pixels_sha256'], row['decoded_yuv420_pixels_sha256'])
        self.assertEqual(record['video_sha256'], hashlib.sha256(video.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
