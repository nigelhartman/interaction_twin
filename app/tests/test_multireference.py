import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from multireference_tracking import choose_references,evaluation_frame,merge_branches,scope_veto,contiguous_nonempty_segment


class MultiReferenceTests(unittest.TestCase):
    def test_empty_masks_split_without_a_fallback_crop(self):
        self.assertEqual(contiguous_nonempty_segment([True,False,True,True],3),[2,3])
        with self.assertRaises(ValueError):contiguous_nonempty_segment([True,False],1)
    def test_reference_selection_excludes_reserved_times_and_bad_scope(self):
        rows=[{'timestamp_s':t,'quality_score':s,'scope_safe':safe,'mask_verdict':'plausible_visible_mask'} for t,s,safe in [(5.,100,True),(6.2,10,False),(8.2,8,True),(9.2,9,True),(13.2,7,True)]]
        chosen=choose_references(rows,2.,2,2.)
        self.assertEqual([r['timestamp_s'] for r in chosen],[9.2,13.2])
        self.assertTrue(evaluation_frame(12.001))
        self.assertTrue(evaluation_frame(20.001756))
        self.assertFalse(evaluation_frame(12.2))

    def branch(self,name,t=0.):
        return {'branch_id':name,'reference_timestamp_s':t,'provenance':{'mesh_sha256':'mesh','source_video_sha256':'vid'},'intrinsics':{'matrix':np.eye(3).tolist()},'frames':[{'sample_index':1,'timestamp_s':1.,'status':'supported_conditional_motion_candidate','reasons':[],'candidate_rotation':np.eye(3).tolist(),'candidate_translation':[0,0,3]}]}

    def test_disagreement_withheld_and_no_averaging(self):
        a,b=self.branch('a'),self.branch('b',2.)
        b['frames'][0]['candidate_rotation']=np.diag([-1,-1,1]).tolist()
        rec=merge_branches([a,b])[0]
        self.assertIn('unsupported',rec['status'])
        self.assertNotIn('candidate_rotation',rec)
        self.assertEqual(len(rec['branch_conflicts']),1)

    def test_nearest_reference_not_residual_selects(self):
        a,b=self.branch('a',0.),self.branch('b',1.5)
        a['frames'][0]['heldout_median_px']=.1
        b['frames'][0]['heldout_median_px']=2.
        self.assertEqual(merge_branches([a,b])[0]['selected_branch_id'],'b')

    def test_scope_veto_and_lineage_mismatch(self):
        self.assertEqual(scope_veto({'single_component_visible':True,'component_boundary_plausible':True,'other_components_inside_mask':True}),['component_scope_model_rejected_mask'])
        a,b=self.branch('a'),self.branch('b');b['provenance']['mesh_sha256']='other'
        with self.assertRaises(ValueError):merge_branches([a,b])
        b=self.branch('b');b['provenance']['source_video_sha256']='different-video'
        with self.assertRaises(ValueError):merge_branches([a,b])
        b=self.branch('b');b['intrinsics']['matrix'][0][0]=2.
        with self.assertRaisesRegex(ValueError,'camera matrix'):merge_branches([a,b])

    def test_audit_cannot_transfer_by_sample_number_alone(self):
        from scripts.stage4_multireference import load_audits
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            (p/'audit.json').write_text(json.dumps({'result':{'observations':[{'track_id':'x','sample_index':2,'verdict':'plausible_visible_mask'}]}}))
            (p/'evidence.json').write_text(json.dumps([{'track_id':'x','sample_index':2,'mask_sha256':'old-mask'}]))
            with self.assertRaisesRegex(ValueError,'different mask'):
                load_audits([p/'audit.json'],{('x',2):'new-mask'})
            verdicts,_,_=load_audits([p/'audit.json'],{('x',2):'old-mask'})
            self.assertEqual(verdicts[('x',2)],'plausible_visible_mask')
            with self.assertRaisesRegex(ValueError,'different source image'):
                load_audits([p/'audit.json'],{('x',2):'old-mask'},{('x',2):'expected-image'})

    def test_reference_is_not_temporal_evaluation(self):
        b=self.branch('reference',1.)
        b['frames'][0]['status']='reference_candidate_unverified_geometry_and_pose'
        rec=merge_branches([b])[0]
        self.assertFalse(rec['evaluation_timestamp_reserved_before_reference_selection'])

    def test_unknown_gaps_do_not_interpolate_and_pts_cannot_mix(self):
        a=self.branch('a');middle=copy.deepcopy(a['frames'][0]);middle.update(sample_index=2,timestamp_s=1.2,status='unsupported',reasons=['visibility_loss'])
        middle.pop('candidate_rotation');middle.pop('candidate_translation');last=copy.deepcopy(a['frames'][0]);last.update(sample_index=3,timestamp_s=1.4)
        a['frames']=[a['frames'][0],middle,last]
        records=merge_branches([a]);self.assertEqual(records[1]['status'],'unsupported');self.assertNotIn('candidate_rotation',records[1])
        b=self.branch('b');b['frames'][0]['timestamp_s']=1.1
        with self.assertRaisesRegex(ValueError,'presentation timestamps'):merge_branches([a,b])


if __name__=='__main__':unittest.main()
