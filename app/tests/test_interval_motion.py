import copy
import fcntl
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import interval_motion as M


class IntervalMotionTests(unittest.TestCase):
    def rows(self):
        return [{'sample_index':i+1,'timestamp_s':i/3,'available':True,
            'audit_verdict':'plausible_visible_mask','mask_area_pixels':400,
            'mask_centroid_xy':[40,40],'mask_extent_px':20.,'quality_score':100+i,
            'temporal_split':'evaluation' if M.evaluation_time(i/3) else 'reference_selection'} for i in range(25)]

    def test_no_evaluation_reference_and_bounded_nonoverlapping_episodes(self):
        rows=self.rows();rows[12]['quality_score']=1e12
        episodes=M.choose_episodes(rows)
        self.assertEqual(len(episodes),2)
        for episode in episodes:
            self.assertFalse(M.evaluation_time(episode['reference_timestamp_s']))
            self.assertLessEqual(len(episode['frames']),M.POLICY['maximum_frames_per_episode'])
            self.assertLessEqual(episode['end_s']-episode['start_s'],2.001)
        first={r['sample_index'] for r in episodes[0]['frames']};second={r['sample_index'] for r in episodes[1]['frames']}
        self.assertFalse(first&second)

    def test_missing_mask_and_known_bad_audit_split_episode(self):
        rows=self.rows();rows[12]['available']=False;rows[12]['quality_score']=1e12
        episodes=M.choose_episodes(rows)
        for episode in episodes:
            ids=[r['sample_index'] for r in episode['frames']]
            self.assertNotIn(13,ids)
            self.assertFalse(min(ids)<13<max(ids))
        rows[12]['available']=True;rows[12]['audit_verdict']='uncertain'
        self.assertTrue(all(e['reference_sample_index']!=13 for e in M.choose_episodes(rows)))

    def test_area_or_centroid_jump_is_not_continuity(self):
        a,b=self.rows()[:2]
        b['mask_area_pixels']=10_000
        self.assertFalse(M.stable_pair(a,b))
        b['mask_area_pixels']=400;b['mask_centroid_xy']=[100,100]
        self.assertFalse(M.stable_pair(a,b))

    def test_only_training_masks_expand_crop(self):
        rows=[{'temporal_split':'reference_selection'},{'temporal_split':'evaluation'}]
        small=np.zeros((200,200),bool);small[80:100,80:100]=True
        full=np.ones((200,200),bool)
        self.assertEqual(M.episode_crop(rows,[small,full]),[64,64,116,116])

    def test_unprotected_model_worker_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);directory=root/'data/flow-runs';directory.mkdir(parents=True)
            path=directory/'.model-worker.lock';path.touch()
            with self.assertRaises(RuntimeError):M.verify_worker_lock(root)
            with path.open('a+') as owner:
                fcntl.flock(owner.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                M.verify_worker_lock(root)

    def test_saved_file_url_resolves_inside_project_data_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'data').mkdir();path=root/'data/frame.png';path.touch()
            self.assertEqual(M.file_path(root,'/files/data/frame.png'),path.resolve())
            self.assertEqual(M.file_path(root,'data/frame.png'),path.resolve())
            with self.assertRaises(ValueError):M.file_path(root,'/files/data/../../outside')

    def test_reassessment_preserves_original_and_uses_shared_audit(self):
        record={'status':'unsupported_candidate_projection_retained','candidate_rotation':np.eye(3).tolist(),'candidate_translation':[0,0,2],
            'training_visible_point_ids':list(range(12)),'training_inlier_point_ids':list(range(10)),
            'heldout_residual_px':[1.]*6,'raw_full_projection_iou_with_sam':.8,'source_mask_coverage':.9}
        before=copy.deepcopy(record);row={'gate_status':'candidate_geometric_gate_only'}
        assessment={'verdict':'plausible_visible_mask','scope_safe':True}
        result=M.reassess_baseline(record,row,assessment)
        self.assertEqual(result['baseline_reassessed_status'],M.SUPPORTED)
        self.assertFalse(result['baseline_original_supported']);self.assertEqual(record,before)
        assessment['scope_safe']=False
        self.assertEqual(M.reassess_baseline(record,row,assessment)['baseline_reassessed_status'],M.WITHHELD)

    def test_reference_never_becomes_supported_nonreference(self):
        record={'status':'reference_candidate_unverified_geometry_and_pose'}
        result=M.reassess_baseline(record,{},{});self.assertEqual(result['baseline_reassessed_status'],M.REFERENCE)
        self.assertFalse(result['baseline_original_supported'])

    def test_bad_or_too_few_disjoint_scores_fail(self):
        self.assertTrue(M.pose_reasons(20,18,[.1]*5,.9,.9,[]))
        self.assertTrue(M.pose_reasons(20,18,[.1]*5+[float('inf')],.9,.9,[]))
        self.assertTrue(M.pose_reasons(20,18,[.1]*6,.9,.9,['source_mask_not_plausible_in_model_audit']))
        self.assertFalse(M.pose_reasons(20,18,[.1]*6,.9,.9,[]))

    def test_audit_exact_plan_and_per_target_hash_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);directory=root/'data';directory.mkdir()
            plan_path=directory/'plan.json';plan_path.write_text('{}')
            target={'sample_index':1,'source_image_sha256':'image','mask_sha256':'mask','audit_verdict':'unassessed'}
            plan={'track_id':'part','interaction_sha256':'program','source_video_sha256':'video','episodes':[{'frames':[target]}]}
            observation={'track_id':'part','sample_index':1,'source_image_sha256':'image','mask_sha256':'mask',
                'verdict':'plausible_visible_mask','single_component_visible':True,'component_boundary_plausible':True,'other_components_inside_mask':False}
            audit={'plan_sha256':M.sha(plan_path),'interaction_sha256':'program','source_video_sha256':'video','result':{'observations':[observation]}}
            path=directory/'audit.json';path.write_text(json.dumps(audit))
            result,_=M.load_audit(plan,plan_path,path,root);self.assertTrue(result[1]['scope_safe'])
            observation['mask_sha256']='different';path.write_text(json.dumps(audit))
            with self.assertRaises(ValueError):M.load_audit(plan,plan_path,path,root)

    def test_unassessed_masks_fail_closed_and_withheld_has_no_display_pose(self):
        row={'sample_index':1,'timestamp_s':.333,'temporal_split':'reference_selection','gate_status':'candidate_geometric_gate_only'}
        assessment={'verdict':'unassessed','scope_safe':True}
        self.assertTrue(M.audit_reasons(row,assessment))
        rec=M.frame_record(row,assessment,2)
        self.assertIsNone(rec['candidate_rotation']);self.assertIsNone(rec['candidate_translation'])
        self.assertEqual(rec['status'],M.WITHHELD);self.assertFalse(rec['physical_pose_accuracy_measured'])

    def test_scope_failure_cuts_sequence_and_baseline_cannot_bridge_it(self):
        rows=self.rows()[:7]
        for row in rows:row['gate_status']='candidate_geometric_gate_only'
        assessment={r['sample_index']:{'verdict':'plausible_visible_mask','scope_safe':True} for r in rows}
        assessment[5]['scope_safe']=False
        self.assertEqual(M.contiguous_scope_segment(rows,assessment,2),[0,1,2,3])
        self.assertEqual(M.contiguous_scope_segment(rows,assessment,5),[5,6])
        self.assertTrue(M.baseline_barrier_reasons(1,7,rows,assessment))
        self.assertFalse(M.baseline_barrier_reasons(1,4,rows,assessment))

    def test_unassessed_intermediate_mask_is_not_a_motion_bridge(self):
        rows=self.rows()[:4]
        for row in rows:row['gate_status']='candidate_geometric_gate_only'
        assessment={r['sample_index']:{'verdict':'plausible_visible_mask','scope_safe':True} for r in rows}
        assessment[2]['verdict']='unassessed'
        self.assertEqual(M.contiguous_scope_segment(rows,assessment,0),[0])


if __name__=='__main__':unittest.main()
