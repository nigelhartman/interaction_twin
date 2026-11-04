import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import source_observation_refinement_v1 as R


def fixture():
    images = [{'sample_index': n, 'timestamp_s': float(n)} for n in R.LAPTOP_SAMPLES]
    plan = {'context': {'images': images, 'source_sha256': 'a' * 64}}
    frames = []
    for image in images:
        frames.append({**image, 'laptop_state': 'closed', 'state_confidence': .9,
                       'lid_visible_face': 'occluded', 'base_visible_face': 'underside',
                       'visible_laptop_panel': 'laptop_base', 'visible_panel_identity_confidence': .8,
                       'parts': [{'track_id': tid, 'visibility': 'visible' if tid == 'laptop_base' else 'occluded',
                                  'observed_bbox_xyxy_normalized': [.1, .1, .9, .8] if tid == 'laptop_base' else [],
                                  'confidence': .8, 'hidden_part_pose_observable': False, 'reason': 'fixture'} for tid in R.TRACK_IDS],
                       'hinge_seam_visible': False, 'hinge_seam_uv': [], 'occlusion_notes': 'fixture'})
    result = {'format': R.FORMAT, 'source_sha256': 'a' * 64, 'persistent_attachment': 'supported',
              'frames': frames, 'limitations': ['synthetic fixture'], 'physical_accuracy_verified': False}
    return plan, result


class SourceObservationReview(unittest.TestCase):
    def test_closed_underside_identity_is_independent_of_old_lid_label(self):
        plan, result = fixture()
        self.assertIs(R.validate_result(result, plan), result)
        self.assertEqual(result['frames'][0]['visible_laptop_panel'], 'laptop_base')
        self.assertEqual(result['frames'][0]['parts'][1]['visibility'], 'occluded')

    def test_hidden_box_invented_pose_or_changed_pts_rejected(self):
        plan, result = fixture()
        for change in ['hidden_box', 'hidden_pose', 'pts', 'duplicate', 'seam']:
            bad = copy.deepcopy(result)
            if change == 'hidden_box':
                bad['frames'][0]['parts'][1]['observed_bbox_xyxy_normalized'] = [.1, .1, .9, .8]
            elif change == 'hidden_pose':
                bad['frames'][0]['parts'][1]['hidden_part_pose_observable'] = True
            elif change == 'pts':
                bad['frames'][0]['timestamp_s'] += .1
            elif change == 'duplicate':
                bad['frames'][0]['parts'][1]['track_id'] = 'laptop_base'
            else:
                bad['frames'][0]['hinge_seam_uv'] = [[.1, .2], [.5, .2]]
            with self.subTest(change=change), self.assertRaises(ValueError):
                R.validate_result(bad, plan)

    def test_original_source_packet_preparation_makes_no_request(self):
        with patch('meta_api.MetaClient.request', side_effect=AssertionError('offline only')):
            doc = R.build_plan()
        self.assertEqual([i['sample_index'] for i in doc['context']['images']], list(R.LAPTOP_SAMPLES))
        self.assertEqual(len(doc['wire_provenance']['image_inputs']), 8)
        self.assertTrue(doc['context']['source_only'])
        self.assertFalse(doc['context']['3d_renders_supplied'])
        self.assertFalse(doc['context']['mask_images_sent'])
        self.assertEqual(doc['maximum_new_requests'], 1)
        self.assertLess(float(doc['maximum_estimated_reservation_usd']), 2)

    def test_uncertain_attempt_cannot_be_resent(self):
        client = Mock()
        client.json_result.side_effect = TimeoutError('synthetic fixture')
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            folder = root / 'data/fresh-packet'
            folder.mkdir(parents=True)
            (folder / 'plan.json').write_text('{}')
            doc = {'context': {'images': []}}
            extras = {'prompt': 'fixture', 'provider_schema': {}}
            with patch.object(R, 'verified_plan', return_value=(doc, extras, folder)):
                with self.assertRaises(TimeoutError):
                    R.run(root=root, authorized_plan_sha256=R.sha(folder / 'plan.json'), allow_paid_api=True, client=client)
                self.assertEqual(json.loads((folder / 'attempt/failure.json').read_text())['status'], 'preserved_no_automatic_retry')
                with self.assertRaisesRegex(ValueError, 'Once-only'):
                    R.run(root=root, authorized_plan_sha256=R.sha(folder / 'plan.json'), allow_paid_api=True, client=client)
        self.assertEqual(client.json_result.call_count, 1)


if __name__ == '__main__':
    unittest.main()
