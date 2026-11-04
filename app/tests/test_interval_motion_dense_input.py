from decimal import Decimal
import unittest
from interval_motion_dense_input import select_native_frames,select_episode, POLICY


class NativeDensitySamplingTests(unittest.TestCase):
    def pts(self):return [str(Decimal(i)/30) for i in range(61)]

    def test_exact_native_samples_common_frames_forced_and_no_interpolation(self):
        pts=self.pts();common=[{'sample_index':i+1,'timestamp_s':float(Decimal(i)/30)} for i in range(0,61,10)]
        selected,mapping=select_native_frames(pts,common,0,2)
        self.assertEqual(len(selected),25)
        self.assertTrue(set(range(0,61,10)).issubset(selected))
        self.assertEqual(set(mapping),set(range(0,61,10)))
        self.assertTrue(all(i in range(61) for i in selected))

    def test_source_pts_mismatch_and_collapsed_old_ids_rejected(self):
        with self.assertRaises(ValueError):select_native_frames(self.pts(),[{'sample_index':1,'timestamp_s':.03}],0,2)
        with self.assertRaises(ValueError):select_native_frames(self.pts(),[{'sample_index':1,'timestamp_s':0},{'sample_index':2,'timestamp_s':.000001}],0,2)

    def test_density_bound_rejects_long_interval(self):
        pts=[str(Decimal(i)/30) for i in range(301)]
        with self.assertRaises(ValueError):select_native_frames(pts,[],0,10)

    def test_timestamp_rounding_allowance_is_not_frame_interpolation(self):
        selected,mapping=select_native_frames(self.pts(),[{'sample_index':1,'timestamp_s':.333334}],0,2)
        self.assertIn(10,selected);self.assertIn(10,mapping)
        self.assertEqual(POLICY['maximum_program_pts_rounding_difference_s'],.000002)

    def test_all_episode_selection_preserves_frozen_reference_and_chronology(self):
        plan={'episodes':[{'id':'late','start_s':10,'reference_sample_index':62},
            {'id':'early','start_s':7,'reference_sample_index':53}]}
        self.assertEqual(select_episode(plan)['reference_sample_index'],53)
        self.assertEqual(select_episode(plan,'late')['reference_sample_index'],62)
        self.assertEqual(plan['episodes'][0]['id'],'late')
        with self.assertRaises(ValueError):select_episode(plan,'new-reference')


if __name__=='__main__':unittest.main()
