import copy
import tempfile
from pathlib import Path
import unittest
import numpy as np
import interval_motion as M
from interval_motion_density import make_variants,validate_plan_manifest,same_point_arrays,compare


class DensityComparisonTests(unittest.TestCase):
    def frame(self,i,common=True,reference=False,status=None):
        return {'sample_index':700+i,'timestamp_s':i/12,'source_image_sha256':str(i),
            'mask_sha256':str(i+100),'reference':reference,'status':status or M.SUPPORTED,
            'native_decoded_frame_index_zero_based':699+i,'native_pts_exact_s':str(i/12),
            'original_program_sample_index':i if common else None,
            'evaluation_timestamp_reserved_before_reference_selection':i%3==0,'reasons':[]}

    def fixture(self):
        frames=[self.frame(i,common=i%4==0,reference=i==4) for i in range(9)]
        identity=dict(track_id='lid',source_video_sha256='s',mesh_sha256='m',interaction_sha256='i',
            fixed_reference_fit={'automatic':True},fixed_intrinsics={'matrix':[[1,0,0],[0,1,0],[0,0,1]]},
            fixed_original_query_uv_source=[[1,2]],fixed_original_crop_xyxy=[0,0,5,5])
        plan={**identity,'episodes':[{'id':'original','reference_sample_index':704,'frames':frames}]}
        return plan,{**identity,'frames':copy.deepcopy(frames)}

    def test_variants_share_exact_pixels_masks_pts_and_reference(self):
        plan,manifest=self.fixture();validate_plan_manifest(plan,manifest)
        a,b=make_variants(plan)
        self.assertEqual(len(a['frames']),3);self.assertEqual(len(b['frames']),9)
        self.assertEqual([r['sample_index'] for r in a['frames']],[700,704,708])
        self.assertEqual(a['reference_sample_index'],704)
        self.assertEqual(plan['episodes'][0]['id'],'original')
        report=compare(a,b);self.assertEqual(report['matched_nonreference_count'],2)
        self.assertEqual(report['control_supported'],2)
        self.assertFalse(report['physical_accuracy_measured'])

    def test_source_identity_or_rows_cannot_change(self):
        for key in ['track_id','source_video_sha256','mesh_sha256','interaction_sha256']:
            plan,manifest=self.fixture();plan[key]='different'
            with self.assertRaises(ValueError):validate_plan_manifest(plan,manifest)
        plan,manifest=self.fixture();plan['episodes'][0]['frames'][0]['mask_sha256']='changed'
        with self.assertRaises(ValueError):validate_plan_manifest(plan,manifest)
        for key in ['fixed_reference_fit','fixed_intrinsics','fixed_original_query_uv_source','fixed_original_crop_xyxy']:
            plan,manifest=self.fixture();plan[key]='different'
            with self.assertRaises(ValueError):validate_plan_manifest(plan,manifest)

    def test_comparison_refuses_pts_image_mask_or_reference_mismatch(self):
        for key,value in [('timestamp_s',999),('source_image_sha256','x'),('mask_sha256','x'),('reference',True)]:
            plan,_=self.fixture();a,b=make_variants(plan);b['frames'][0][key]=value
            with self.assertRaises(ValueError):compare(a,b)

    def test_reference_never_counts_as_success(self):
        plan,_=self.fixture();a,b=make_variants(plan)
        for row in a['frames']:
            if not row['reference']:row['status']=M.WITHHELD
        result=compare(a,b)
        self.assertEqual(result['control_supported'],0);self.assertEqual(result['dense_supported'],2)
        self.assertEqual(result['matched_nonreference_count'],2)

    def test_point_identity_requires_exact_ids_and_ray_intersection(self):
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'a.npz';b=Path(folder)/'b.npz'
            values=dict(query_uv=np.array([[1.,2.],[3.,4.]]),reference_ray_hits=np.array([True,False]),
                training_point_ids=np.array([0]),heldout_point_ids=np.array([1]),canonical_xyz=np.array([[1.,2.,3.],[np.nan,np.nan,np.nan]]))
            np.savez(a,**values);np.savez(b,**values)
            checks,delta=same_point_arrays(a,b);self.assertTrue(all(checks.values()));self.assertEqual(delta,0.)
            values['query_uv']=values['query_uv']+1;np.savez(b,**values)
            checks,_=same_point_arrays(a,b);self.assertFalse(checks['query_uv'])


if __name__=='__main__':unittest.main()
