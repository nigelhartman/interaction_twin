import copy
import unittest

import numpy as np

import localized_moving_parts as localized
import meta_sam_api


class LocalizedMovingPartsTests(unittest.TestCase):
    def setUp(self):
        self.raw = {'localizations': [
            {'component_id': 'case_base', 'frame_id': localized.FRAME_ID, 'visibility': 'visible',
             'box_xyxy_1000': [100, 200, 400, 700], 'reason': 'Synthetic test fixture.'},
            {'component_id': 'case_lid', 'frame_id': localized.FRAME_ID, 'visibility': 'uncertain',
             'box_xyxy_1000': None, 'reason': 'Synthetic uncertainty fixture.'}]}

    def test_visible_and_withheld_parts_preserved(self):
        self.assertIs(localized.validate_localization(self.raw), self.raw)

    def test_invalid_outside_reversed_nonfinite_boxes_never_clipped(self):
        for box in [[-1, 10, 50, 60], [10, 20, 1001, 70], [40, 20, 30, 80], [0, 0, float('nan'), 90]]:
            raw = copy.deepcopy(self.raw)
            raw['localizations'][0]['box_xyxy_1000'] = box
            with self.assertRaises(ValueError):
                localized.validate_localization(raw)

    def test_uncertain_part_cannot_supply_box(self):
        self.raw['localizations'][1]['box_xyxy_1000'] = [1, 2, 3, 4]
        with self.assertRaisesRegex(ValueError, 'null box'):
            localized.validate_localization(self.raw)

    def test_wrong_frame_or_duplicate_component_fails(self):
        self.raw['localizations'][1]['component_id'] = 'case_base'
        with self.assertRaisesRegex(ValueError, 'two requested'):
            localized.validate_localization(self.raw)
        self.raw['localizations'][1]['component_id'] = 'case_lid'
        self.raw['localizations'][1]['frame_id'] = 'jabra::wrong'
        with self.assertRaises(Exception):
            localized.validate_localization(self.raw)

    def test_source_pixel_conversion_and_official_rounding(self):
        normalized = [100.3, 200.2, 400.4, 700.1]
        pixel = localized.source_pixel_box(normalized, 540, 960)
        np.testing.assert_allclose(pixel, [54.162, 192.192, 216.216, 672.096])
        prompt, records = meta_sam_api._model_box_prompt([{'id': 0, 'box_xyxy': pixel}], width=540, height=960,
                     frame_count=1, prompt_origin='model_generated_fullframe_visual_box_localization', media='image')
        self.assertEqual(records[0]['rounded_box_xyxy_half_open'], [54, 192, 217, 673])
        self.assertEqual(records[0]['wire_box_xyxy_inclusive'], [54, 192, 216, 672])


if __name__ == '__main__':
    unittest.main()
