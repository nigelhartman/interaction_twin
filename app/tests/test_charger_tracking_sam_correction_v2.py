"""Exercise actual SAM prompt contract with a stub run, no client or network."""
import unittest
from unittest.mock import patch
from pathlib import Path

import charger_tracking_sam_correction_v2 as C
import meta_sam_api as SAM


class ChargerSAMCorrectionTests(unittest.TestCase):
    def test_actual_adapter_rejects_old_provenance_before_run_or_callback(self):
        with patch.object(SAM, '_run', side_effect=AssertionError('must not reach run')) as run:
            with self.assertRaisesRegex(SAM.SamAPIError, 'explicit model-generated provenance'):
                SAM.segment_video_boxes(Path('unused.mp4'), [{'id': 0, 'frame_index': 0, 'box_xyxy': [1, 2, 20, 30]}], Path('unused'),
                    lambda _: self.fail('must not transport'), width=718, height=960,
                    frame_sources=[{'video_sha256': 'a'*64, 'sample_index': 30, 'timestamp_s': 20.2, 'time_domain': SAM.TIME_DOMAIN}],
                    prompt_origin='fresh six-source-image Muse localization of rigid adapter body; not old lost mask')
            run.assert_not_called()

    def test_actual_adapter_accepts_corrected_origin_and_preserves_sources_with_stub_run(self):
        source = {'video_sha256': 'a'*64, 'sample_index': 30, 'timestamp_s': 20.2, 'time_domain': SAM.TIME_DOMAIN}
        with patch.object(SAM, '_data_url', return_value='data:video/mp4;base64,AA=='), patch.object(SAM, 'sha256', return_value='b'*64), patch.object(SAM, '_run', return_value={'stub_only': True}) as run:
            result = SAM.segment_video_boxes(Path('unused.mp4'), [{'id': 0, 'frame_index': 0, 'box_xyxy': [1, 2, 20, 30]}], Path('unused'),
                lambda _: self.fail('must not transport'), width=718, height=960, frame_sources=[source], prompt_origin=C.PROMPT_ORIGIN)
            self.assertEqual(result, {'stub_only': True}); run.assert_called_once()
            args, kw = run.call_args; self.assertEqual(kw['sources'], [source]); self.assertEqual(args[0]['model'], 'sam-3.1')
            self.assertEqual(kw['provenance']['box_prompts'][0]['frame_index'], 0)
            self.assertEqual(kw['provenance']['box_prompts'][0]['wire_box_xyxy_inclusive'], [1, 2, 19, 29])


if __name__ == '__main__': unittest.main()
