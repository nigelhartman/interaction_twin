"""Temporary synthetic source-domain contracts, never published as model output."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

try:
    import numpy as np
except ImportError:
    np=None

from tests import test_motion_lab as fixture_helpers
from motion_lab import sha
from motion_density_lab import normalize
save=fixture_helpers.save


@unittest.skipUnless(np is not None,'Uses the prepared local NumPy test environment')
class DensityDisplayContracts(unittest.TestCase):
    def fixture(self):
        root,parent_path,parent=fixture_helpers.MotionLabContracts.fixture(self);data=root/'data';episode=parent['episodes'][0]
        query=[[1,1],[2,2],[3,3],[4,4]]
        arrays={'query_uv':np.asarray(query,dtype=np.float32),'reference_ray_hits':np.ones(4,dtype=bool),
                'training_point_ids':np.array([1,2]),'heldout_point_ids':np.array([0,3]),'canonical_xyz':np.arange(12,dtype=float).reshape(4,3)}
        parent_tracks=data/'parent-tracks.npz';np.savez(parent_tracks,**arrays)
        episode['queries'].update(automatic_query_uv_source=query,reference_mesh_ray_hits=[True]*4)
        episode.update(crop_xyxy=[0,0,8,8],tracks_url='/files/data/parent-tracks.npz',tracks_sha256=sha(parent_tracks))
        save(parent_path,parent)
        plan=data/'plan.json';audit=data/'audit.json';save(plan,{'synthetic':True});save(audit,{'synthetic':True})
        identity={k:deepcopy(parent[k])for k in ['case_id','component_id','track_id','source','source_video_sha256','asset','mesh_sha256','interaction_path','interaction_sha256']}
        identity.update(label='Synthetic component',new_api_calls=0,manual_geometry_or_poses=False,
                        physical_pose_accuracy_measured=False,usable_physical_pose_recovered=False,
                        common_assembly_established=False,original_semantic_program_extended=False,
                        inherited_automatic_reference_pose_used=True)
        parent_binding={'path':'data/interval-motion.json','sha256':sha(parent_path)}
        native=[]
        for sample,time,old in [(11,1,1),(16,1.25,None),(21,1.5,2),(26,1.75,None),(31,2,3)]:
            native.append({'sample_index':sample,'timestamp_s':time,'native_decoded_frame_index_zero_based':sample-1,
                'native_pts_exact_s':str(time),'source_image_url':'/files/data/source.jpg','source_image_sha256':sha(data/'source.jpg'),
                'mask_url':'/files/data/mask.png','mask_sha256':sha(data/'mask.png'),
                'native_sam2_mask_url':'/files/data/mask.png','native_sam2_mask_sha256':sha(data/'mask.png'),
                'original_program_sample_index':old,'original_3fps_jpeg_copied_byte_identical':old is not None,
                'temporal_split':'evaluation'if old in {2,3}else'reference_selection'})
        extension={k:deepcopy(identity[k])for k in ['case_id','track_id','interaction_sha256','source_video_sha256','mesh_sha256']}
        extension.update(format='short-interval-density-input',version=1,separate_sampling_domain=True,
            published_interaction_program=False,original_program_evidence_was_extended_or_overwritten=False,
            parent_interval_result=parent_binding,selected_episode_id='episode_01',original_reference_sample_index=1,
            fixed_intrinsics=episode['intrinsics'],fixed_original_query_uv_source=query,fixed_original_crop_xyxy=[0,0,8,8],
            reference_sample_index=11,selected_interval_s=[1,2],frames=native,frame_count=5)
        extension_path=data/'dense-input.json';save(extension_path,extension)
        identity.update(source_extension={'path':'data/dense-input.json','sha256':sha(extension_path)},
            plan={'path':'data/plan.json','sha256':sha(plan)},audit_binding={'path':'data/audit.json','sha256':sha(audit)},
            parent_interval_result=parent_binding,motion_policy={'synthetic_fixture':True})
        group={**deepcopy(identity),'format':'short-interval-density-experiment','version':1,'status':'complete_with_gaps','variants':[],
               'limitations':['Synthetic fixture; no learned inference or physical truth.']}
        for name,fps in [('control-3fps',3),('density-12fps',12)]:
            frames=[]
            for original in native:
                if fps==3 and original['original_program_sample_index']is None:continue
                frame=deepcopy(original);i=frame['sample_index'];reference=i==11
                status='reference_unverified'if reference else'withheld'if i==31 else'supported_nonreference_candidate'
                frame.update(status=status,reference=reference,reasons=[],
                    evaluation_timestamp_reserved_before_reference_selection=frame['temporal_split']=='evaluation',
                    candidate_rotation=deepcopy(parent['episodes'][0]['frames'][0]['candidate_rotation'])if status!='withheld'else None,
                    candidate_translation=[0,0,3]if status!='withheld'else None,
                    training_visible_point_ids=[1,2],heldout_visible_point_ids=[0,3])
                frames.append(frame)
            tracks=data/(name+'-tracks.npz');np.savez(tracks,**arrays)
            branch={**deepcopy(identity),'format':'short-interval-density-run','version':1,'status':'complete_with_gaps',
                'context_requested_fps':fps,'coordinate_convention':parent['coordinate_convention'],
                'episodes':[{'id':name,'reference_sample_index':11,'reference_timestamp_s':1,'start_s':1,'end_s':2,
                    'intrinsics':episode['intrinsics'],'crop_xyxy':[0,0,8,8],'queries':deepcopy(episode['queries']),
                    'frames':frames,'tracks_url':'/files/data/'+tracks.name,'tracks_sha256':sha(tracks)}]}
            target=data/(name+'.json');save(target,branch)
            group['variants'].append({'id':name,'path':'data/'+target.name,'sha256':sha(target)})
        common=[]
        for row in native:
            if row['original_program_sample_index']is None:continue
            common.append({k:row[k]for k in ['sample_index','timestamp_s','native_pts_exact_s','source_image_sha256','mask_sha256','original_program_sample_index']})
            status='reference_unverified'if row['sample_index']==11 else'withheld'if row['sample_index']==31 else'supported_nonreference_candidate'
            common[-1].update(reference=row['sample_index']==11,control_status=status,dense_status=status,
                              evaluation_timestamp_reserved_before_reference_selection=row['temporal_split']=='evaluation',
                              control_supported=status=='supported_nonreference_candidate',dense_supported=status=='supported_nonreference_candidate')
        group['comparison']={'exact_common_pts_rows':common,'matched_nonreference_count':2,'control_supported':1,'dense_supported':1,
            'reserved_evaluation_count':2,'control_reserved_evaluation_supported':1,'dense_reserved_evaluation_supported':1}
        target=data/'density-experiment.json';save(target,group);return root,target,group

    def test_extra_native_samples_remain_separate_and_common_counts_recomputed(self):
        root,target,group=self.fixture();before={p:sha(p)for p in (root/'data').iterdir()if p.is_file()}
        result=normalize(root,target)
        self.assertEqual(result['comparison_kind'],'density_control')
        self.assertEqual(result['comparison_counts'],{'fair_same_pts_samples':2,'candidate_supported':1,'baseline_supported':1,'both_supported':1,'references_excluded':1})
        self.assertEqual(len(result['episodes'][0]['frames']),5)
        self.assertIsNone(result['episodes'][0]['frames'][1]['baseline'])
        self.assertEqual(result['display_labels']['baseline'],'3 fps control')
        self.assertEqual(before,{p:sha(p)for p in (root/'data').iterdir()if p.is_file()})

    def test_invented_counts_semantics_reference_points_or_frames_fail_closed(self):
        for field in ['counts','program_extension','reference','canonical_points','native_frame','mask_identity']:
            with self.subTest(field=field):
                root,target,group=self.fixture()
                if field=='counts':group['comparison']['dense_supported']=2
                elif field=='program_extension':group['original_semantic_program_extended']=True
                else:
                    binding=group['variants'][1];branch_path=root/binding['path'];branch=json.loads(branch_path.read_text());episode=branch['episodes'][0]
                    if field=='reference':episode['frames'][0]['candidate_translation']=[0,0,4]
                    elif field=='native_frame':episode['frames'][1]['sample_index']=999
                    elif field=='mask_identity':episode['frames'][1]['mask_sha256']='0'*64
                    else:
                        tracks=root/episode['tracks_url'].removeprefix('/files/')
                        with np.load(tracks,allow_pickle=False)as archive:arrays={k:archive[k].copy()for k in archive.files}
                        arrays['canonical_xyz'][0,0]+=1;np.savez(tracks,**arrays);episode['tracks_sha256']=sha(tracks)
                    save(branch_path,branch);binding['sha256']=sha(branch_path)
                save(target,group)
                with self.assertRaises(ValueError):normalize(root,target)

