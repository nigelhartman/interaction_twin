"""Small read-only contract for explicitly predicted, source-conditioned motion.

No CV/model/runtime dependencies are imported. The caller must admit every data
artifact through an exact public/curated allowlist. Producer/catalog fingerprints
are provenance only, never arbitrary executable/download routes. Physical and
original-observation claims cannot be upgraded by this separate prediction layer.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
from statistics import median

FORMAT='interaction-twin-source-conditioned-motion-hypotheses'
MAX_JSON=16*1024*1024


def require(value,message):
    if not value:raise ValueError(message)


def finite(value):return type(value)in {int,float}and math.isfinite(value)


def reference(value,maximum=256*1024*1024):
    require(isinstance(value,dict)and set(value)=={'path','sha256','bytes'},'Exact temporal artifact reference required')
    p=value['path'];require(isinstance(p,str)and p.startswith('data/')and re.fullmatch(r'[A-Za-z0-9_./-]+',p)and str(PurePosixPath(p))==p and not any(x in {'','.','..'}or x.startswith('.')for x in p.split('/')),'Ordinary exact temporal data path required')
    require(isinstance(value['sha256'],str)and re.fullmatch('[a-f0-9]{64}',value['sha256'])and type(value['bytes'])is int and 0<value['bytes']<=maximum,'Bounded temporal hash and length required')
    return value


def matrix(value):
    require(isinstance(value,list)and len(value)==4 and all(isinstance(r,list)and len(r)==4 and all(finite(v)for v in r)for r in value)and value[3]==[0,0,0,1],'Finite temporal native matrix required')
    a=value;det=a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0])
    require(det>0,'Positive temporal native scale required');scale=det**(1/3)
    require(all(abs(sum(a[k][i]*a[k][j]for k in range(3))-(scale*scale if i==j else 0))<2e-5 for i in range(3)for j in range(3)),'Temporal prediction must preserve a proper native similarity')
    return value


def multiply(a,b):return [[sum(a[i][k]*b[k][j]for k in range(4))for j in range(4)]for i in range(4)]


def rotation(axis,q):
    x,y,z=axis;c,s=math.cos(q),math.sin(q);v=1-c
    return [[x*x*v+c,x*y*v-z*s,x*z*v+y*s,0],[y*x*v+z*s,y*y*v+c,y*z*v-x*s,0],[z*x*v-y*s,z*y*v+x*s,z*z*v+c,0],[0,0,0,1]]


def close(a,b,tolerance=2e-6):return max(abs(a[i][j]-b[i][j])for i in range(4)for j in range(4))<=tolerance


def reprojection(native,pixels,pose,k):
    errors=[]
    for p,uv in zip(native,pixels):
        x=[sum(pose[i][j]*p[j]for j in range(3))+pose[i][3]for i in range(3)]
        require(x[2]>0,'Tracked native feature behind estimated camera')
        h=[sum(k[i][j]*x[j]for j in range(3))for i in range(3)]
        errors.append(math.hypot(h[0]/h[2]-uv[0],h[1]/h[2]-uv[1]))
    return median(errors)


def projected_mean(points,pose,k):
    values=[]
    for p in points:
        x=[sum(pose[i][j]*p[j]for j in range(3))+pose[i][3]for i in range(3)]
        h=[sum(k[i][j]*x[j]for j in range(3))for i in range(3)]
        require(h[2]>0,'Continuity probe behind camera');values.append([h[0]/h[2],h[1]/h[2]])
    return [sum(p[i]for p in values)/len(values)for i in range(2)]


def adjacent_motion(a,b):
    sa=math.sqrt(sum(a[i][0]**2 for i in range(3)));sb=math.sqrt(sum(b[i][0]**2 for i in range(3)))
    trace=sum(a[i][j]*b[i][j]/sa/sb for i in range(3)for j in range(3))
    return math.acos(max(-1.,min(1.,(trace-1)/2))),max(a[2][3]/b[2][3],b[2][3]/a[2][3]),max(sa/sb,sb/sa)


def _validate(document,read_bytes):
    """Return checked source documents and exact data artifacts for transport."""
    require(callable(read_bytes),'Curated exact temporal artifact reader required')
    require(isinstance(document,dict)and document.get('format')==FORMAT and document.get('version')==1,'Unknown temporal hypothesis format')
    require(len(json.dumps(document,allow_nan=False))<=MAX_JSON,'Bounded temporal document required')
    require(document.get('physical_accuracy_verified')is False and document.get('simulation_ready')is False and document.get('source_unknowns_preserved')is True and document.get('native_geometry_modified')is False and document.get('new_API_calls')==document.get('new_GPU_calls')==0,'Prediction layer cannot upgrade physical or original observations')
    require(document.get('nonroutable_fingerprints')==['catalog_ref','implementation_ref'],'Exact nonroutable provenance declaration required')
    for key,path in [('catalog_ref','showcase/public/library-catalog.json'),('implementation_ref','source_temporal_recovery_v1.py')]:
        f=document.get(key);require(isinstance(f,dict)and set(f)=={'path','sha256','bytes'}and f['path']==path and re.fullmatch('[a-f0-9]{64}',f['sha256'])and type(f['bytes'])is int and 0<f['bytes']<MAX_JSON,'Unknown temporal producer/catalog fingerprint')
    cache={};artifacts={}
    def raw(pointer,maximum=MAX_JSON):
        reference(pointer,maximum);key=(pointer['path'],pointer['sha256'],pointer['bytes'])
        if key not in cache:
            body=read_bytes(pointer,maximum=maximum)
            require(isinstance(body,(bytes,bytearray))and len(body)==pointer['bytes']and hashlib.sha256(body).hexdigest()==pointer['sha256'],'Temporal admitted bytes changed')
            cache[key]=bytes(body);artifacts[key]=pointer
        return cache[key]
    def read(pointer):
        value=json.loads(raw(pointer),parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite temporal JSON')))
        require(isinstance(value,dict),'Temporal JSON object required');return value
    program=read(document['evidence_program_ref']);scene=read(document['baseline_scene_ref']);flow=read(document['flow_proofs_ref'])
    if document.get('parent_motion_hypothesis_ref'):
        parent=read(document['parent_motion_hypothesis_ref'])
        require(parent.get('format')==FORMAT and parent.get('version')==1 and all(parent.get(k)==document.get(k)for k in ['twin_id','source_ref','evidence_program_ref','baseline_scene_ref']),'Temporal derivative source lineage changed')
    for fingerprint in document.get('additional_implementation_fingerprints',[]):
        require(set(fingerprint)=={'path','sha256','bytes'}and fingerprint['path']in {'source_temporal_selection_v1.py','temporal_sam_recovery_v1.py','source_temporal_continuity_v1.py'}and re.fullmatch('[a-f0-9]{64}',fingerprint['sha256'])and type(fingerprint['bytes'])is int and 0<fingerprint['bytes']<MAX_JSON,'Unknown nonroutable temporal derivative producer')
    if document.get('selection_report_ref'):
        selection=read(document['selection_report_ref'])
        require(selection.get('format')=='interaction-twin-temporal-source-agreement-selection'and selection.get('version')==1 and selection.get('no_original_observations_or_assets_modified')is True and selection.get('physical_accuracy_verified')is False,'Unknown temporal source-agreement selection')
    sam=None;sam_rows={}
    if document.get('source_SAM_temporal_observations_ref'):
        sam=read(document['source_SAM_temporal_observations_ref'])
        require(sam.get('format')=='interaction-twin-bidirectional-local-SAM-temporal-observations'and sam.get('version')==1 and all(sam.get(k)==document.get(k)for k in ['source_ref','evidence_program_ref','baseline_scene_ref'])and sam.get('actual_local_SAM2_CPU_states')==2 and sam.get('new_hosted_API_or_remote_GPU_calls')==0 and sam.get('physical_accuracy_verified')is False and sam.get('target_masks_will_be_fitting_inputs_not_independent_tests')is True,'SAM conditional motion must preserve exact CPU/source/conditional scope')
        plan=read(sam['plan_ref']);require(plan.get('format')=='interaction-twin-bidirectional-local-SAM-temporal-plan'and plan.get('episode_id')==sam.get('episode_id')and plan.get('device')=='cpu'and plan.get('model')=='Meta SAM2.1 Tiny'and plan.get('manual_points_boxes_masks_or_poses')is False and plan.get('minimum_forward_backward_mask_IoU')>=.72,'Exact automatic bounded CPU SAM plan required')
        require(isinstance(sam.get('rows'),list)and 3<=len(sam['rows'])<=96,'Bounded independent SAM clip required')
        for row in sam['rows']:
            require(row['timestamp_s']not in sam_rows and row.get('model_observation_not_ground_truth')is True,'Unique fallible SAM source rows required')
            sam_rows[row['timestamp_s']]=row
            for field in ['rgb_ref','past_mask_ref','future_mask_ref']:
                raw(row[field])
            if row['status']=='estimated':
                require(row['forward_backward_mask_IoU']>=.72 and row.get('consensus')=='intersection_of_independent_source_mask_propagations','SAM opposite-direction masks disagree')
                raw(row['consensus_mask_ref'])
    source=document['source_ref'];raw(source,maximum=256*1024*1024)
    require(source['sha256']==program['source']['sha256']==scene['source']['sha256']and source['path']==program['source']['path'],'Temporal source/program/scene mismatch')
    require(scene.get('original_program')==document['evidence_program_ref'],'Temporal scene belongs to a different exact source program revision')
    require(isinstance(document.get('twin_id'),str)and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,159}',document['twin_id']),'Bounded public temporal recording ID required')
    parts={c['id']:c for c in scene['components']};evidence={r['source_index']:r for r in program['evidence']};joints={j['id']:j for j in scene.get('mechanics',{}).get('joints',[])}
    require(flow.get('format')=='interaction-twin-temporal-source-flow-proofs'and flow.get('version')==1 and isinstance(flow.get('rows'),list)and len(flow['rows'])<=10000,'Bounded source flow proof inventory required')
    proofs={p['id']:p for p in flow['rows']};require(len(proofs)==len(flow['rows']),'Unique temporal flow proof IDs required')
    policy=document['policy']
    require(0<policy['maximum_anchor_gap_s']<=4 and 0<policy['maximum_viewer_interpolation_gap_s']<=.22 and 0<policy['maximum_one_sided_distance_s']<=1.0 and 0<policy['maximum_constant_joint_state_difference_rad']<=math.radians(3)+1e-10,'Temporal recovery windows exceed bounded policy')
    require(policy['minimum_PnP_inliers']>=12 and policy['minimum_PnP_inlier_fraction']>=.45 and policy['maximum_median_PnP_reprojection_px']<=1.1 and policy['maximum_PnP_reprojection_px']<=2 and policy['maximum_forward_backward_error_px']<=1.25,'Temporal image evidence thresholds were weakened')
    require(policy.get('minimum_target_mask_iou',.48)>=.48 and policy.get('minimum_target_mask_coverage',.62)>=.62,'Temporal independent target-mask checks were weakened')
    require(policy['original_unknown_poses_or_barriers_changed']is False and policy['physical_accuracy_verified']is False,'Temporal policy cannot change original ledger')
    camera=document['camera'];k=camera['K'];require(len(k)==3 and all(len(r)==3 and all(finite(x)for x in r)for r in k),'Finite temporal tracking intrinsics required')
    require(type(camera['width'])is int and type(camera['height'])is int and 16<=min(camera['width'],camera['height'])<=max(camera['width'],camera['height'])<=1280,'Bounded temporal tracking raster required')
    original_camera=scene['camera'];expected_k=[r[:]for r in original_camera['K']]
    for axis,scale in enumerate([camera['width']/original_camera['width'],camera['height']/original_camera['height']]):
        expected_k[axis]=[v*scale for v in expected_k[axis]];expected_k[axis][2]+=(scale-1)/2
    require(all(abs(k[i][j]-expected_k[i][j])<=1e-8 for i in range(3)for j in range(3)),'Temporal tracking camera differs from exact scaled source-scene intrinsics')
    continuity={}
    if document.get('display_continuity_ref'):
        report=read(document['display_continuity_ref'])
        require(report.get('format')=='interaction-twin-source-checked-display-continuity'and report.get('version')==1 and report.get('input_hypothesis_ref')==document.get('parent_motion_hypothesis_ref')and all(report.get(key)==document.get(key)for key in ['source_ref','evidence_program_ref','baseline_scene_ref'])and report.get('source_original_unknowns_preserved')is True and report.get('poses_angles_geometry_or_source_masks_edited')is False and report.get('physical_accuracy_verified')is False,'Unknown or upgraded temporal display continuity report')
        limits=report['limits'];require(0<limits['maximum_adjacent_rotation_rad']<=math.radians(12)+1e-9 and 1<=limits['maximum_adjacent_depth_ratio']<=1.12 and 0<limits['maximum_source_relative_center_delta_error']<=.12 and 1<=limits['maximum_adjacent_scale_ratio']<=1.000001 and 0<limits['maximum_interpolation_gap_s']<=.22,'Temporal continuity gates were weakened')
        for entry in report['episodes']:
            require(entry['episode_id']not in continuity,'Unique continuity episode required');continuity[entry['episode_id']]=entry
    def anchor(value):
        require(value['component_id']in parts,'Unknown temporal source component')
        pose=next((r for r in parts[value['component_id']]['poses']if r['source_sample_index']==value['source_sample_index']),None)
        require(pose is not None and pose['status']=='estimated'and pose.get('camera_from_native_mesh')==value['camera_from_native_mesh']and pose['timestamp_s']==value['timestamp_s'],'Temporal anchor differs from original known scene pose')
        require(value.get('source_mask_ref')==pose.get('source_mask'),'Temporal anchor source mask changed')
        if value.get('source_mask_ref'):raw(value['source_mask_ref'])
        e=evidence[value['source_sample_index']];require(e['timestamp_s']==value['timestamp_s'],'Temporal anchor RGB clock differs')
        raw(value['source_RGB_ref']);require(value['source_RGB_ref']['path']==e['frame_path']and value['source_RGB_ref']['sha256']==e['frame_sha256'],'Temporal anchor source RGB changed')
        return pose
    def image(value,timestamp):
        require(value['timestamp_s']==timestamp and isinstance(value.get('decoded_gray_sha256'),str)and re.fullmatch('[a-f0-9]{64}',value['decoded_gray_sha256']),'Temporal RGB clock/hash required')
        require(value['tracking_width']==camera['width']and value['tracking_height']==camera['height'],'Temporal RGB raster differs')
        if value.get('image_ref'):
            raw(value['image_ref']);e=next((r for r in program['evidence']if r['timestamp_s']==timestamp),None)
            require(e is not None and value['image_ref']['path']==e['frame_path']and value['image_ref']['sha256']==e['frame_sha256'],'Temporal saved RGB belongs to another observation')
        else:require(value.get('video_ref')==source and type(value.get('decoded_frame_index'))is int and 0<=value['decoded_frame_index']<=20000,'Temporal dense RGB must bind exact source video frame')
    used=set();group_ids=set();used_continuity=set()
    require(isinstance(document['groups'],list)and len(document['groups'])<=32,'Bounded temporal groups required')
    for group in document['groups']:
        root=group['root_component_id'];require(root in parts and root not in group_ids,'Unique temporal root required');group_ids.add(root)
        related=[j for j in joints.values()if j['parent_id']==root]
        ids=[root]+[j['child_id']for j in related];require(group['component_ids']==ids,'Temporal rigid descendants changed')
        for cid in ids:raw(parts[cid]['asset'],maximum=32*1024*1024)
        require(len(group['episodes'])<=256,'Bounded temporal episodes required')
        for episode in group['episodes']:
            a,b=episode['anchor_before'],episode['anchor_after'];anchor(a);anchor(b)
            require(a['component_id']==b['component_id']==root and a['timestamp_s']<b['timestamp_s'],'Temporal episode roots/time order changed')
            old=[r['source_sample_index']for r in parts[root]['poses']if a['timestamp_s']<r['timestamp_s']<b['timestamp_s']and r['status']=='unknown']
            require(episode['original_unknown_indices']==old,'Original unknown interval identity changed')
            if not episode['predictions']:continue
            require(b['timestamp_s']-a['timestamp_s']<=policy['maximum_anchor_gap_s']+1e-9,'Temporal prediction exceeds anchor gap')
            da,db=episode['driver_before'],episode['driver_after'];anchor(da);anchor(db)
            require(da['component_id']in ids and db['component_id']in ids and da['timestamp_s']==a['timestamp_s']and db['timestamp_s']==b['timestamp_s'],'Temporal source-visible driver changed')
            require(episode['view_change_break']==(da['component_id']!=db['component_id']),'Cross-face motion break was removed')
            if episode['id']in continuity:
                used_continuity.add(episode['id'])
                checked=continuity[episode['id']];require(not episode['view_change_break']and checked['root_component_id']==root and checked['native_asset_ref']==parts[root]['asset']and checked.get('pose_or_joint_edits')is False,'Cannot normalize a different native or cross-face episode')
                probes=checked['native_probe_points'];require(len(probes)==27 and all(len(p)==3 and all(finite(x)for x in p)for p in probes),'Exact bounded native continuity probes required')
                source_rows={r['timestamp_s']:r for r in checked['source_rows']}
                require(len(source_rows)==len(checked['source_rows']),'Unique source continuity observations required')
                for t,row in source_rows.items():
                    require(t in sam_rows and row['mask_ref']==sam_rows[t]['consensus_mask_ref']and len(row['centroid_px'])==2 and all(finite(x)for x in row['centroid_px'])and finite(row['diagonal_px'])and row['diagonal_px']>0,'Continuity source delta must bind actual admitted SAM observation');raw(row['mask_ref'])
                nodes=[(a['timestamp_s'],a['camera_from_native_mesh'],None)]+[(p['timestamp_s'],p['component_poses'].get(root)if p['status']=='estimated'else None,p)for p in episode['predictions']]+[(b['timestamp_s'],b['camera_from_native_mesh'],None)]
                require(len(checked['nodes'])==len(nodes)and len(checked['edges'])==len(nodes)-1,'Continuity ledger omitted a source/unknown row')
                counter=0;all_pass=True
                for i,(t,pose,prediction)in enumerate(nodes):
                    saved=checked['nodes'][i];require(saved['timestamp_s']==t and saved['has_pose']==(pose is not None),'Continuity node identity changed')
                    if i:
                        pt,previous_pose,_=nodes[i-1];edge=checked['edges'][i-1];require(edge['from_s']==pt and edge['to_s']==t,'Continuity edge clock changed')
                        passed=False
                        if pose is not None and previous_pose is not None:
                            angle,depth,scale=adjacent_motion(previous_pose,pose)
                            require(abs(edge['rotation_rad']-angle)<1e-7 and abs(edge['depth_ratio']-depth)<1e-8 and abs(edge['scale_ratio']-scale)<1e-8,'Continuity 3D jump diagnostic does not reproduce')
                            error=None
                            if pt in source_rows and t in source_rows:
                                first,last=source_rows[pt],source_rows[t];ca=projected_mean(probes,previous_pose,k);cb=projected_mean(probes,pose,k)
                                error=math.hypot(*[(cb[j]-ca[j])-(last['centroid_px'][j]-first['centroid_px'][j])for j in range(2)])/max(1.,(first['diagonal_px']+last['diagonal_px'])/2)
                                require(edge.get('source_delta_basis')=='actual_bidirectional_SAM_source_mask_centroid_delta'and abs(edge['source_relative_center_delta_error']-error)<1e-8,'Continuity actual-source 2D delta differs')
                            else:require(edge.get('source_relative_center_delta_error')is None,'Unobserved source delta cannot be invented')
                            passed=angle<=limits['maximum_adjacent_rotation_rad']and depth<=limits['maximum_adjacent_depth_ratio']and scale<=limits['maximum_adjacent_scale_ratio']and t-pt<=limits['maximum_interpolation_gap_s']and(error is None or error<=limits['maximum_source_relative_center_delta_error'])
                        require(edge['passed']==passed,'Continuity gate decision was fabricated')
                        if not passed:counter+=1;all_pass=False
                    segment=episode['id']+f':continuous-{counter}';require(saved['segment_id']==segment,'Continuity segment illegally crosses a rejected edge')
                    if prediction is not None and pose is not None:require(prediction.get('view_segment_id')==segment,'Estimated frame differs from checked continuity segment')
                require(checked['all_edges_pass']==all_pass and episode['anchor_before_view_segment_id']==checked['nodes'][0]['segment_id']and episode['anchor_after_view_segment_id']==checked['nodes'][-1]['segment_id'],'Continuity endpoint or complete-clip decision differs')
            states=episode['constant_joint_states'];require({s['id']for s in states}=={j['id']for j in related},'Hidden child requires exact coupled joint state')
            for state in states:
                j=joints[state['id']];ja=next(r for r in j['samples']if r['source_sample_index']==a['source_sample_index']);jb=next(r for r in j['samples']if r['source_sample_index']==b['source_sample_index'])
                require(ja['status']==jb['status']=='estimated'and abs(ja['q_rad']-jb['q_rad'])<=policy['maximum_constant_joint_state_difference_rad']and abs(state['q_rad']-(ja['q_rad']+jb['q_rad'])/2)<1e-10 and state['hidden_part_observed']is False,'Joint angle was fabricated rather than inherited from matching neighbours')
            previous=a['timestamp_s']
            for prediction in episode['predictions']:
                t=prediction['timestamp_s'];require(finite(t)and previous<t<b['timestamp_s'],'Strict temporal source time required');previous=t
                require(prediction['original_unknown_row_preserved']is True,'Prediction cannot relabel original unknown')
                if prediction.get('source_sample_index')is not None:
                    i=prediction['source_sample_index'];require(i in old and evidence[i]['timestamp_s']==t,'Recovered original row must retain exact unknown ID/clock')
                image(prediction['source_RGB'],t);pid=prediction['flow_proof_id'];require(pid in proofs and pid not in used,'Exact unique temporal flow proof required');used.add(pid);proof=proofs[pid]
                require(proof['timestamp_s']==t and proof['target_RGB']==prediction['source_RGB']and proof['driver_before']==da['component_id']and proof['driver_after']==db['component_id'],'Temporal proof source/driver mismatch')
                check=prediction.get('independent_target_mask_check')
                if check:
                    require(prediction.get('source_sample_index')in evidence and check.get('target_mask_used_for_fitting')is False,'Target-mask diagnostics must retain their unfitted original source row')
                    allowed={(o['mask_path'],o['mask_sha256'])for o in evidence[prediction['source_sample_index']].get('observations',[])if o.get('mask_path')and o['track_id']in ids}
                    for pointer in check.get('mask_refs',[]):
                        require((pointer['path'],pointer['sha256'])in allowed,'Independent target mask belongs to another component/source');raw(pointer)
                    if check['status']in {'passed','failed'}:
                        require(check.get('SAM_mask_is_ground_truth')is False and check.get('full_native_triangle_render')is True,'Target mask diagnostics must preserve fallible SAM/full-native scope')
                        passed=check['raw_full_triangle_union_iou']>=policy['minimum_target_mask_iou']and check['source_foreground_coverage']>=policy['minimum_target_mask_coverage']
                        require(passed==(check['status']=='passed'),'Independent target mask gate decision changed')
                    if prediction['status']=='estimated':require(check['status']in {'passed','unavailable'},'Failed independent mask cannot become an estimated pose')
                if prediction['status']=='withheld':require(prediction['component_poses']=={}and prediction['joint_states']==[],'Unknown temporal hypothesis must not carry a pose or angle');continue
                require(prediction['status']=='estimated'and set(prediction['component_poses'])==set(ids)and prediction['joint_states']==states,'Estimated temporal assembly must retain coupled components')
                for pose in prediction['component_poses'].values():matrix(pose)
                for j in related:
                    q=next(s['q_rad']for s in states if s['id']==j['id']);expected=multiply(multiply(multiply(prediction['component_poses'][root],j['parent_from_joint']),rotation(j['axis'],q)),j['joint_from_child_zero'])
                    require(close(expected,prediction['component_poses'][j['child_id']]),'Predicted child detached from saved joint FK')
                if prediction.get('support')=='source_SAM_mask_conditioned_pose':
                    fit=proof.get('conditional_SAM_mask_fit');observation=sam_rows.get(t)
                    require(sam is not None and observation is not None and observation['status']=='estimated'and not episode['view_change_break']and sam['episode_id']==episode['id'],'Conditional SAM pose crossed a surface-view break or lacks source mask observation')
                    require(isinstance(fit,dict)and fit['observations_ref']==document['source_SAM_temporal_observations_ref']and fit['consensus_mask_ref']==observation['consensus_mask_ref']and fit['source_RGB_ref']==observation['rgb_ref']and observation['source_RGB']==prediction['source_RGB'],'Conditional SAM fit/source refs differ')
                    if fit.get('prior_source_motion_ref'):
                        prior_document=read(fit['prior_source_motion_ref'])
                        require(prior_document.get('format')==FORMAT and all(prior_document.get(key)==document.get(key)for key in ['source_ref','evidence_program_ref','baseline_scene_ref']),'Conditional pose prior belongs to another source')
                        prior_rows={p['flow_proof_id']:p for g in prior_document['groups']for e in g['episodes']if e['id']==episode['id']for p in e['predictions']}
                        prior_ids=fit['prior_anchor_proof_ids'];require(len(prior_ids)==2 and all(i in prior_rows for i in prior_ids),'Exact two source-confirmed conditional prior anchors required')
                        first,last=[prior_rows[i]for i in prior_ids]
                        require(all(p['status']=='estimated'and p['support']in {'past_and_future_RGB_agree','one_sided_near_anchor_RGB'}for p in [first,last])and first['timestamp_s']<t<last['timestamp_s']and last['timestamp_s']-first['timestamp_s']<=fit['maximum_source_confirmed_interpolation_gap_s']<=1.5 and fit['prior_anchor_timestamps_s']==[first['timestamp_s'],last['timestamp_s']],'Conditional prior cannot cross missing/conditional states or exceed its source-confirmed bracket')
                    elif fit.get('original_prior_anchor_rows'):
                        angle,depth,_=adjacent_motion(a['camera_from_native_mesh'],b['camera_from_native_mesh'])
                        require(len(ids)==1 and angle<=math.radians(15)+1e-9 and depth<=1.35 and fit['original_prior_anchor_rows']==[a,b]and b['timestamp_s']-a['timestamp_s']<=fit['maximum_source_confirmed_interpolation_gap_s']<=1.5 and fit['prior_anchor_timestamps_s']==[a['timestamp_s'],b['timestamp_s']],'Conditional original-scene bracket changed')
                    else:raise ValueError('Conditional source-mask pose needs an exact source-confirmed prior bracket')
                    require(fit.get('target_mask_used_for_fitting')is True and fit.get('independent_validation')is False and fit.get('same_source_view_segment')is True and fit['full_triangle_conditional_mask_iou']>=.60 and fit['conditional_foreground_coverage']>=.75 and fit['forward_backward_mask_IoU']>=.72,'Conditional mask fit cannot masquerade as an independent or weak test')
                    require(0<=fit['rotation_delta_from_neighbor_prior_rad']<=fit['maximum_rotation_search_rad']+1e-9 and fit['maximum_rotation_search_rad']<=math.radians(18)+1e-9 and close(fit['fitted_root_matrix'],prediction['component_poses'][root]),'Conditional pose exceeds bounded source-mask search')
                    before_scale=math.sqrt(sum(a['camera_from_native_mesh'][i][0]**2 for i in range(3)));after_scale=math.sqrt(sum(prediction['component_poses'][root][i][0]**2 for i in range(3)))
                    require(abs(before_scale-after_scale)<1e-6 and abs(fit['native_uniform_scale_unchanged']-before_scale)<1e-6,'Conditional mask pose changed native scale')
                    require(prediction.get('evidence_kind')=='conditional_model_mask_fit'and isinstance(prediction.get('view_segment_id'),str)and(prediction['view_segment_id']==episode['id']+':same_source_view'or episode['id']in continuity),'Explicit same-view conditional estimate required')
                    continue
                valid=[]
                for label,driver in [('past',da),('future',db)]:
                    result=proof[label]
                    if result['status']!='estimated':continue
                    matrix(result['camera_from_native_mesh']);require(result['PnP_inliers']>=policy['minimum_PnP_inliers']and result['inlier_fraction']>=policy['minimum_PnP_inlier_fraction']and result['median_reprojection_px']<=policy['maximum_median_PnP_reprojection_px']and result['feature_coverage']>=policy['minimum_reference_feature_coverage'],'Weak source feature proof cannot support a temporal pose')
                    if result.get('method')=='dense_bidirectional_ECC_native_plane_pose_hypothesis':
                        require(result.get('PnP_grid_points_are_independent_feature_matches')is False and result.get('neighbour_prior_used_only_for_initialization')is True and result['dense_correlation']>=.93 and result['reverse_dense_correlation']>=.93 and result['median_warp_cycle_px']<=1.25 and result['dense_photometric_pixels']>=64,'Dense warp needs actual photometric and reverse-warp evidence, never invented independent features')
                    points,pixels=result['native_points'],result['target_pixels'];require(12<=len(points)==len(pixels)<=20 and all(len(p)==3 and all(finite(x)for x in p)for p in points)and all(len(p)==2 and all(finite(x)for x in p)for p in pixels),'Bounded actual source feature correspondences required')
                    require(reprojection(points,pixels,result['camera_from_native_mesh'],k)<=2.,'Source PnP correspondences do not reproduce')
                    error=reprojection(points,pixels,prediction['component_poses'][driver['component_id']],k)
                    valid.append((label,error,abs(t-driver['timestamp_s'])))
                if prediction['support']=='past_and_future_RGB_agree':
                    require(not episode['view_change_break']and len(valid)==2 and proof.get('agreement',{}).get('accepted')is True,'Cross-face or missing future motion cannot be blended')
                    require(max(v[1]for v in valid)<=5.,'Blended pose does not follow both source feature sets')
                else:
                    require(prediction['support']=='one_sided_near_anchor_RGB'and any(error<=2. and distance<=policy['maximum_one_sided_distance_s']+1e-9 for _,error,distance in valid),'Unbounded one-sided temporal extrapolation')
                require(isinstance(prediction.get('view_segment_id'),str)and prediction['view_segment_id'].startswith(episode['id']+':'),'Explicit temporal view segment required')
    require(used==set(proofs),'Unbound temporal proof rows present')
    require(used_continuity==set(continuity),'Unbound temporal continuity episodes present')
    return {'document':document,'artifacts':list(artifacts.values()),'source_program':program,'source_scene':scene,'flow_proofs':flow,
        'original_unknowns_preserved':True,'physical_accuracy_verified':False}


def validate(document,read_bytes):
    """Fail malformed hypotheses closed without importing any CV dependencies.

    A curated caller additionally binds ``twin_id``, ``evidence_program_ref`` and
    ``baseline_scene_ref`` to its selected record. Public legacy IDs intentionally
    need not equal the newer approved program's internal case ID.
    """
    try:return _validate(document,read_bytes)
    except (KeyError,TypeError,IndexError,AttributeError,OverflowError,ZeroDivisionError)as error:
        raise ValueError('Malformed bounded temporal hypothesis')from error
