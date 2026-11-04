import copy
import hashlib
import json
import math
import unittest
from pathlib import Path

import motion_hypothesis_contract_v1 as C


def fixture():
    files={}
    def raw(name,value):
        body=value if isinstance(value,bytes)else json.dumps(value,sort_keys=True).encode()
        pointer={'path':'data/temporal-fixture/'+name,'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body)}
        files[pointer['path']]=body
        return pointer
    video=raw('video.mp4',b'fixture-only-codec-bytes')
    images=[raw(f'{i}.jpg',f'source{i}'.encode())for i in range(3)]
    asset=raw('native.glb',b'fixture-native-geometry')
    root=[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,1.],[0,0,0,1]]
    identity=[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0,0,0,1]]
    child=copy.deepcopy(root);child[0][3]=.1
    offset=copy.deepcopy(identity);offset[0][3]=.1
    evidence=[{'source_index':i+1,'timestamp_s':float(i),'frame_path':im['path'],'frame_sha256':im['sha256']}for i,im in enumerate(images)]
    program={'case_id':'future-program-internal-id','source':video,'evidence':evidence}
    def poses(matrix):return [{'source_sample_index':i+1,'timestamp_s':float(i),'status':'unknown'if i==1 else'estimated','camera_from_native_mesh':None if i==1 else copy.deepcopy(matrix)}for i in range(3)]
    joint={'id':'hinge','parent_id':'body','child_id':'lid','axis':[1.,0.,0.],'parent_from_joint':identity,'joint_from_child_zero':offset,
        'samples':[{'source_sample_index':i+1,'timestamp_s':float(i),'status':'unknown'if i==1 else'estimated','q_rad':None if i==1 else 0.}for i in range(3)]}
    scene={'source':video,'camera':{'K':[[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]],'width':100,'height':100},'components':[{'id':'body','asset':asset,'poses':poses(root)},{'id':'lid','asset':asset,'poses':poses(child)}],'mechanics':{'joints':[joint]}}
    pr=raw('program.json',program);scene['original_program']=pr;sr=raw('scene.json',scene)
    rgb={'image_ref':images[1],'timestamp_s':1.,'decoded_gray_sha256':'a'*64,'tracking_width':100,'tracking_height':100}
    points=[[x,y,0.]for y in [-.12,0.,.12]for x in [-.18,-.06,.06,.18]]
    pixels=[[p[0]*100+50,p[1]*100+50]for p in points]
    fitted={'status':'estimated','camera_from_native_mesh':root,'PnP_inliers':12,'inlier_fraction':1.,'median_reprojection_px':0.,'feature_coverage':.5,'native_points':points,'target_pixels':pixels}
    proof={'id':'gap@1','timestamp_s':1.,'target_RGB':rgb,'driver_before':'body','driver_after':'body','past':fitted,'future':copy.deepcopy(fitted),'agreement':{'accepted':True,'rotation_rad':0.,'center_extent_fraction':0.,'depth_ratio':1.}}
    fr=raw('proofs.json',{'format':'interaction-twin-temporal-source-flow-proofs','version':1,'rows':[proof]})
    state={'id':'hinge','q_rad':0.,'method':'matching_neighboring_saved_joint_states_only','hidden_part_observed':False}
    def anchor(i):return {'component_id':'body','source_sample_index':i+1,'timestamp_s':float(i),'camera_from_native_mesh':copy.deepcopy(root),'source_mask_ref':None,'source_RGB_ref':images[i]}
    prediction={'timestamp_s':1.,'source_sample_index':2,'source_RGB':rgb,'status':'estimated','component_poses':{'body':root,'lid':child},'joint_states':[state],'original_unknown_row_preserved':True,'flow_proof_id':'gap@1','support':'past_and_future_RGB_agree','view_segment_id':'gap:same_source_view'}
    episode={'id':'gap','anchor_before':anchor(0),'anchor_after':anchor(2),'driver_before':anchor(0),'driver_after':anchor(2),'view_change_break':False,'original_unknown_indices':[2],'constant_joint_states':[state],'predictions':[prediction]}
    policy={'maximum_anchor_gap_s':4.,'maximum_viewer_interpolation_gap_s':.22,'maximum_one_sided_distance_s':1.,'maximum_constant_joint_state_difference_rad':.05,'minimum_PnP_inliers':12,'minimum_PnP_inlier_fraction':.45,'maximum_median_PnP_reprojection_px':1.1,'maximum_PnP_reprojection_px':2.,'maximum_forward_backward_error_px':1.25,'minimum_reference_feature_coverage':.07,'original_unknown_poses_or_barriers_changed':False,'physical_accuracy_verified':False}
    document={'format':C.FORMAT,'version':1,'twin_id':'legacy-public-record','source_ref':video,'evidence_program_ref':pr,'baseline_scene_ref':sr,'flow_proofs_ref':fr,
        'catalog_ref':{'path':'showcase/public/library-catalog.json','sha256':'b'*64,'bytes':100},'implementation_ref':{'path':'source_temporal_recovery_v1.py','sha256':'c'*64,'bytes':100},'nonroutable_fingerprints':['catalog_ref','implementation_ref'],
        'policy':policy,'camera':{'K':[[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]],'width':100,'height':100},'groups':[{'root_component_id':'body','component_ids':['body','lid'],'episodes':[episode]}],
        'source_unknowns_preserved':True,'native_geometry_modified':False,'physical_accuracy_verified':False,'simulation_ready':False,'new_API_calls':0,'new_GPU_calls':0}
    def read(pointer,maximum=None):return files[pointer['path']]
    return document,files,read


def conditional_fixture():
    d,files,read=fixture()
    def put(name,value):
        body=value if isinstance(value,bytes)else json.dumps(value,sort_keys=True).encode()
        pointer={'path':'data/temporal-fixture/'+name,'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body)}
        files[pointer['path']]=body;return pointer
    plan=put('sam-plan.json',{'format':'interaction-twin-bidirectional-local-SAM-temporal-plan','episode_id':'gap','device':'cpu','model':'Meta SAM2.1 Tiny','manual_points_boxes_masks_or_poses':False,'minimum_forward_backward_mask_IoU':.72})
    episode=d['groups'][0]['episodes'][0];p=episode['predictions'][0]
    rows=[]
    for i in range(3):
        image=episode['anchor_before']['source_RGB_ref']if i==0 else episode['anchor_after']['source_RGB_ref']if i==2 else p['source_RGB']['image_ref']
        mask=put(f'{i}.png',f'mask{i}'.encode())
        rgb=copy.deepcopy(p['source_RGB']);rgb['timestamp_s']=float(i);rgb['image_ref']=image
        rows.append({'timestamp_s':float(i),'source_RGB':rgb,'rgb_ref':image,'past_mask_ref':mask,'future_mask_ref':mask,'consensus_mask_ref':mask,
            'status':'estimated','forward_backward_mask_IoU':.95,'consensus':'intersection_of_independent_source_mask_propagations','model_observation_not_ground_truth':True})
    report=put('sam-observations.json',{'format':'interaction-twin-bidirectional-local-SAM-temporal-observations','version':1,'plan_ref':plan,'episode_id':'gap',
        **{k:d[k]for k in ['source_ref','evidence_program_ref','baseline_scene_ref']},'rows':rows,'actual_local_SAM2_CPU_states':2,'new_hosted_API_or_remote_GPU_calls':0,'physical_accuracy_verified':False,'target_masks_will_be_fitting_inputs_not_independent_tests':True})
    d['source_SAM_temporal_observations_ref']=report
    # A separate, checked source-feature result supplies the near temporal prior.
    # Its synthetic codec references are test fixtures, never saved observations.
    prior=copy.deepcopy(d);prior.pop('source_SAM_temporal_observations_ref')
    prior_episode=prior['groups'][0]['episodes'][0];prior_episode['predictions']=[]
    original_flow=json.loads(files[d['flow_proofs_ref']['path']]);prior_proofs=[]
    for timestamp in [.5,1.5]:
        item=copy.deepcopy(p);item.update(timestamp_s=timestamp,source_sample_index=None,flow_proof_id=f'prior@{timestamp}')
        item['source_RGB']={'video_ref':d['source_ref'],'decoded_frame_index':int(timestamp*30),'timestamp_s':timestamp,
            'decoded_gray_sha256':'a'*64,'tracking_width':100,'tracking_height':100}
        prior_episode['predictions'].append(item)
        proof=copy.deepcopy(original_flow['rows'][0]);proof.update(id=item['flow_proof_id'],timestamp_s=timestamp,target_RGB=item['source_RGB'])
        prior_proofs.append(proof)
    prior['flow_proofs_ref']=put('prior-proofs.json',{'format':'interaction-twin-temporal-source-flow-proofs','version':1,'rows':prior_proofs})
    C.validate(prior,read)
    prior_ref=put('source-confirmed-prior.json',prior)
    flow=json.loads(files[d['flow_proofs_ref']['path']]);flow['rows'][0]['conditional_SAM_mask_fit']={
        'observations_ref':report,'consensus_mask_ref':rows[1]['consensus_mask_ref'],'source_RGB_ref':rows[1]['rgb_ref'],
        'target_mask_used_for_fitting':True,'independent_validation':False,'same_source_view_segment':True,
        'full_triangle_conditional_mask_iou':.9,'conditional_foreground_coverage':.95,'forward_backward_mask_IoU':.95,
        'rotation_delta_from_neighbor_prior_rad':0.,'maximum_rotation_search_rad':.3,'fitted_root_matrix':p['component_poses']['body'],'native_uniform_scale_unchanged':1.,
        'prior_source_motion_ref':prior_ref,'prior_anchor_proof_ids':['prior@0.5','prior@1.5'],
        'prior_anchor_timestamps_s':[.5,1.5],'maximum_source_confirmed_interpolation_gap_s':1.5}
    d['flow_proofs_ref']=put('conditional-proofs.json',flow)
    p['support']='source_SAM_mask_conditioned_pose';p['evidence_kind']='conditional_model_mask_fit'
    return d,files,read


def rewrite(files,pointer,value):
    body=json.dumps(value,sort_keys=True).encode();files[pointer['path']]=body
    pointer.update(sha256=hashlib.sha256(body).hexdigest(),bytes=len(body))


def continuity_fixture(*,withheld=False):
    """A deliberately sparse interval must stay split, even with valid poses."""
    d,files,read=fixture();parent=copy.deepcopy(d)
    body=json.dumps(parent,sort_keys=True).encode();name='data/temporal-fixture/parent.json';files[name]=body
    d['parent_motion_hypothesis_ref']={'path':name,'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body)}
    episode=d['groups'][0]['episodes'][0];p=episode['predictions'][0]
    if withheld:p.update(status='withheld',component_poses={},joint_states=[])
    else:p['view_segment_id']='gap:continuous-1'
    episode['anchor_before_view_segment_id']='gap:continuous-0';episode['anchor_after_view_segment_id']='gap:continuous-2'
    asset=json.loads(files[d['baseline_scene_ref']['path']])['components'][0]['asset']
    nodes=[{'timestamp_s':float(i),'segment_id':f'gap:continuous-{i}','has_pose':not(withheld and i==1),'kind':'prediction'if i==1 else'anchor_before'if i==0 else'anchor_after'}for i in range(3)]
    edges=[{'from_s':float(i),'to_s':float(i+1),'passed':False,
        'source_delta_basis':'individually_verified_RGB_pixel_correspondences',
        **({}if withheld else{'rotation_rad':0.,'depth_ratio':1.,'scale_ratio':1.,'source_relative_center_delta_error':None})}for i in range(2)]
    entry={'episode_id':'gap','root_component_id':'body','native_asset_ref':asset,'native_probe_points':[[x,y,z]for x in [-.1,0.,.1]for y in [-.1,0.,.1]for z in [-.1,0.,.1]],
        'source_rows':[],'nodes':nodes,'edges':edges,'all_edges_pass':False,'pose_or_joint_edits':False}
    report={'format':'interaction-twin-source-checked-display-continuity','version':1,'input_hypothesis_ref':d['parent_motion_hypothesis_ref'],
        **{k:d[k]for k in ['source_ref','evidence_program_ref','baseline_scene_ref']},
        'limits':{'maximum_adjacent_rotation_rad':math.radians(12),'maximum_adjacent_depth_ratio':1.12,'maximum_source_relative_center_delta_error':.12,
            'maximum_adjacent_scale_ratio':1.000001,'maximum_interpolation_gap_s':.22},
        'episodes':[entry],'source_original_unknowns_preserved':True,'poses_angles_geometry_or_source_masks_edited':False,'physical_accuracy_verified':False}
    body=json.dumps(report,sort_keys=True).encode();name='data/temporal-fixture/continuity.json';files[name]=body
    d['display_continuity_ref']={'path':name,'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body)}
    return d,files,read


class TemporalContractTests(unittest.TestCase):
    def test_checked_segments_cannot_cross_sparse_or_unknown_intervals(self):
        for withheld in [False,True]:
            d,files,read=continuity_fixture(withheld=withheld);C.validate(d,read)
            pointer=d['display_continuity_ref'];report=json.loads(files[pointer['path']])
            report['episodes'][0]['edges'][0]['passed']=True;rewrite(files,pointer,report)
            with self.subTest(withheld=withheld),self.assertRaisesRegex(ValueError,'decision was fabricated'):C.validate(d,read)

    def test_rejected_continuity_edge_cannot_be_hidden_by_a_segment_name(self):
        d,files,read=continuity_fixture();episode=d['groups'][0]['episodes'][0]
        episode['predictions'][0]['view_segment_id']='gap:continuous-0'
        with self.assertRaisesRegex(ValueError,'checked continuity segment'):C.validate(d,read)
        d,files,read=continuity_fixture();d['groups'][0]['episodes'][0]['anchor_after_view_segment_id']='gap:continuous-0'
        with self.assertRaisesRegex(ValueError,'endpoint or complete-clip'):C.validate(d,read)

    def test_unbound_continuity_episode_is_rejected(self):
        d,files,read=continuity_fixture();pointer=d['display_continuity_ref'];report=json.loads(files[pointer['path']])
        extra=copy.deepcopy(report['episodes'][0]);extra['episode_id']='unselected-source';report['episodes'].append(extra);rewrite(files,pointer,report)
        with self.assertRaisesRegex(ValueError,'Unbound temporal continuity'):C.validate(d,read)

    def test_conditional_mask_needs_exact_near_source_confirmed_prior(self):
        for mutation in ['missing','unknown_prior','wrong_clock','invented_original_anchor']:
            d,files,read=conditional_fixture();pointer=d['flow_proofs_ref'];flow=json.loads(files[pointer['path']]);fit=flow['rows'][0]['conditional_SAM_mask_fit']
            if mutation=='missing':fit.pop('prior_source_motion_ref')
            elif mutation=='unknown_prior':fit['prior_anchor_proof_ids'][0]='unknown'
            elif mutation=='wrong_clock':fit['prior_anchor_timestamps_s'][0]=.6
            else:
                fit.pop('prior_source_motion_ref');episode=d['groups'][0]['episodes'][0]
                fit['original_prior_anchor_rows']=[episode['anchor_before'],episode['anchor_after']]
            rewrite(files,pointer,flow)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):C.validate(d,read)

    def test_another_program_of_the_same_video_cannot_reuse_old_scene(self):
        d,files,read=fixture();old=d['evidence_program_ref'];program=json.loads(files[old['path']]);program['case_id']='same-video-different-revision'
        body=json.dumps(program,sort_keys=True).encode();name='data/temporal-fixture/other-program.json';files[name]=body
        d['evidence_program_ref']={'path':name,'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body)}
        with self.assertRaisesRegex(ValueError,'different exact source program'):C.validate(d,read)
    def test_forged_tracking_intrinsics_cannot_relabel_native_pixel_proof(self):
        for row,column in [(0,0),(0,2),(1,1),(1,2),(2,2)]:
            d,files,read=fixture();d['camera']['K'][row][column]+=.25
            with self.subTest(row=row,column=column),self.assertRaises(ValueError):C.validate(d,read)

    def test_structural_schema_and_deep_reader_accept_both_real_evidence_classes(self):
        import jsonschema
        schema=json.loads((Path(__file__).parents[1]/'schemas/motion-hypothesis.schema.json').read_text())
        for factory in [fixture,conditional_fixture]:
            d,files,read=factory();jsonschema.validate(d,schema);C.validate(d,read)

    def test_conditional_model_masks_cannot_claim_independent_validation(self):
        d,files,read=conditional_fixture();pointer=d['flow_proofs_ref'];flow=json.loads(files[pointer['path']])
        flow['rows'][0]['conditional_SAM_mask_fit']['independent_validation']=True
        body=json.dumps(flow,sort_keys=True).encode();files[pointer['path']]=body;pointer['sha256']=hashlib.sha256(body).hexdigest();pointer['bytes']=len(body)
        with self.assertRaises(ValueError):C.validate(d,read)

    def test_bidirectional_model_mask_disagreement_cannot_be_promoted(self):
        d,files,read=conditional_fixture();pointer=d['source_SAM_temporal_observations_ref'];report=json.loads(files[pointer['path']])
        report['rows'][1]['forward_backward_mask_IoU']=.5
        body=json.dumps(report,sort_keys=True).encode();files[pointer['path']]=body;pointer['sha256']=hashlib.sha256(body).hexdigest();pointer['bytes']=len(body)
        with self.assertRaises(ValueError):C.validate(d,read)

    def test_exact_source_features_and_hidden_child_joint_validate_without_CV(self):
        document,files,read=fixture();before=copy.deepcopy(document)
        result=C.validate(document,read)
        self.assertEqual(document,before)
        self.assertTrue(result['original_unknowns_preserved'])
        self.assertFalse(result['physical_accuracy_verified'])
        self.assertNotIn(document['catalog_ref'],result['artifacts'])
        self.assertNotIn(document['implementation_ref'],result['artifacts'])
        self.assertEqual(result['source_scene']['components'][0]['poses'][1]['status'],'unknown')

    def test_source_or_program_hash_mismatch_rejected(self):
        for key in ['source_ref','evidence_program_ref','baseline_scene_ref','flow_proofs_ref']:
            document,files,read=fixture();document[key]['sha256']='0'*64
            with self.subTest(key=key),self.assertRaises(ValueError):C.validate(document,read)

    def test_original_unknown_clock_or_anchor_matrix_cannot_be_fabricated(self):
        mutations=[lambda e:e['anchor_before']['camera_from_native_mesh'][0].__setitem__(3,.1),
            lambda e:e['original_unknown_indices'].clear(),
            lambda e:e['predictions'][0].__setitem__('timestamp_s',1.1)]
        for mutate in mutations:
            d,_,r=fixture();mutate(d['groups'][0]['episodes'][0])
            with self.assertRaises(ValueError):C.validate(d,r)

    def test_child_cannot_detach_and_joint_angle_cannot_be_invented(self):
        d,_,r=fixture();d['groups'][0]['episodes'][0]['predictions'][0]['component_poses']['lid'][1][3]+=.02
        with self.assertRaises(ValueError):C.validate(d,r)
        d,_,r=fixture();d['groups'][0]['episodes'][0]['constant_joint_states'][0]['q_rad']=.3
        with self.assertRaises(ValueError):C.validate(d,r)

    def test_prediction_cannot_ignore_real_feature_correspondences(self):
        d,_,r=fixture();p=d['groups'][0]['episodes'][0]['predictions'][0]
        for v in p['component_poses'].values():v[0][3]+=.2
        with self.assertRaises(ValueError):C.validate(d,r)

    def test_view_change_or_weak_policy_cannot_be_removed(self):
        d,_,r=fixture();d['groups'][0]['episodes'][0]['view_change_break']=True
        with self.assertRaises(ValueError):C.validate(d,r)
        for key,value in [('maximum_anchor_gap_s',5.),('maximum_PnP_reprojection_px',4.),('minimum_PnP_inliers',5)]:
            d,_,r=fixture();d['policy'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):C.validate(d,r)

    def test_unknown_never_contains_a_matrix_and_private_routes_are_rejected(self):
        d,_,r=fixture();d['groups'][0]['episodes'][0]['predictions'][0]['status']='withheld'
        with self.assertRaises(ValueError):C.validate(d,r)
        d,_,r=fixture();d['flow_proofs_ref']['path']='data/../.secrets/key.env'
        with self.assertRaises(ValueError):C.validate(d,r)

    def test_dense_grid_points_cannot_masquerade_as_independent_feature_evidence(self):
        d,files,r=fixture();pointer=d['flow_proofs_ref'];flow=json.loads(files[pointer['path']])
        flow['rows'][0]['past']['method']='dense_bidirectional_ECC_native_plane_pose_hypothesis'
        body=json.dumps(flow,sort_keys=True).encode();files[pointer['path']]=body
        pointer['sha256']=hashlib.sha256(body).hexdigest();pointer['bytes']=len(body)
        with self.assertRaises(ValueError):C.validate(d,r)


if __name__=='__main__':unittest.main()
