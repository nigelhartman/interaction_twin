"""Recovery packet contract tests; no paid requests or original-data mutation."""
import copy
import unittest
import jsonschema

import charger_tracking_recovery_20261103_v1 as C


def result():
    return {'observations': [{'sample_index': i, 'visibility': 'visible', 'box_xyxy_normalized': [.1, .2, .3, .6],
        'attachment_state': 'unknown', 'confidence': .8, 'reason': 'test fixture only'} for i in C.SAMPLES],
        'identity_gap_merge': False, 'physical_accuracy_verified': False, 'limitations': ['fixture; not measured']}


class ChargerTrackingRecoveryTests(unittest.TestCase):
    def test_muse_boxes_map_to_original_source_subset_without_reindexing_source_ids(self):
        objects = C.objects_from_localization(result(), 718, 960)
        self.assertEqual([x['frame_index'] for x in objects], [0, 6, 12, 16, 24, 36])
        self.assertTrue(all(x['id'] == 0 for x in objects))
        self.assertEqual(objects[0]['box_xyxy'], [71.8, 192., 215.4, 576.])

    def test_absent_and_low_confidence_are_not_prompted_or_filled(self):
        value = result(); value['observations'][0].update(visibility='absent', box_xyxy_normalized=[])
        value['observations'][1]['confidence'] = .49
        self.assertEqual([x['frame_index'] for x in C.objects_from_localization(value, 718, 960)], [12, 16, 24, 36])

    def test_invalid_box_identity_merge_reordered_or_duplicate_source_is_rejected(self):
        for change in ['outside', 'reversed', 'nan', 'absent_coordinates', 'duplicate', 'reordered', 'merge']:
            with self.subTest(change=change):
                value = result()
                if change == 'outside': value['observations'][0]['box_xyxy_normalized'] = [-.1, .2, .3, .6]
                elif change == 'reversed': value['observations'][0]['box_xyxy_normalized'] = [.8, .2, .3, .6]
                elif change == 'nan': value['observations'][0]['box_xyxy_normalized'] = [.1, .2, float('nan'), .6]
                elif change == 'absent_coordinates': value['observations'][0]['visibility'] = 'absent'
                elif change == 'duplicate': value['observations'][1] = copy.deepcopy(value['observations'][0])
                elif change == 'reordered': value['observations'].reverse()
                elif change == 'merge': value['identity_gap_merge'] = True
                with self.assertRaises((ValueError, jsonschema.ValidationError)):
                    C.objects_from_localization(value, 718, 960)

    def test_preparing_schema_and_source_free_plan_never_constructs_client(self):
        import jsonschema
        from unittest.mock import patch
        with patch.object(C, 'client', side_effect=AssertionError('must not construct a paid client')):
            jsonschema.Draft202012Validator.check_schema(C.schema())
            C.validate_localization(result())


if __name__ == '__main__':
    unittest.main()
