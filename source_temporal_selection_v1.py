"""Conservative new derivative selecting only reproducible source predictions."""
from pathlib import Path
import copy
import json
import numpy as np

import source_temporal_recovery_v1 as T
import sam3d_mechanisms as M
import motion_hypothesis_contract_v1 as C


def filter_source_agreement(hypothesis_ref,output,*,root=T.ROOT):
    root=Path(root).resolve();out=root/output;T.require(not out.exists(),'New checked temporal selection required');out.mkdir(parents=True)
    implementation=T.ref(Path(__file__),root);d=T.read_checked(hypothesis_ref,root);scene=T.read_checked(d['baseline_scene_ref'],root)
    program=T.read_checked(d['evidence_program_ref'],root);proofs=T.read_checked(d['flow_proofs_ref'],root)
    by_proof={p['id']:p for p in proofs['rows']};parts={c['id']:c for c in scene['components']};ev={e['source_index']:e for e in program['evidence']};meshes={}
    changes=[];k=np.asarray(d['camera']['K']);size=(d['camera']['width'],d['camera']['height'])
    for g in d['groups']:
        related=[j for j in scene.get('mechanics',{}).get('joints',[])if j['parent_id']==g['root_component_id']]
        for episode in g['episodes']:
            for p in episode['predictions']:
                if p['status']!='estimated':continue
                proof=by_proof[p['flow_proof_id']];errors=[]
                for label,driver in [('past',episode['driver_before']),('future',episode['driver_after'])]:
                    result=proof[label]
                    if result['status']=='estimated':
                        error=C.reprojection(result['native_points'],result['target_pixels'],p['component_poses'][driver['component_id']],d['camera']['K'])
                        errors.append((label,driver,error,abs(p['timestamp_s']-driver['timestamp_s'])))
                valid=(len(errors)==2 and max(e[2]for e in errors)<=5.)if p['support']=='past_and_future_RGB_agree'else any(e[2]<=2. and e[3]<=d['policy']['maximum_one_sided_distance_s']for e in errors)
                if valid:continue
                original_support=p['support'];selected=None
                for label,driver,_,distance in sorted(errors,key=lambda e:e[3]):
                    if distance>d['policy']['maximum_one_sided_distance_s']:continue
                    root_matrix=np.asarray(proof[label]['camera_from_native_mesh'])
                    if driver['component_id']!=g['root_component_id']:
                        joint=next(j for j in related if j['child_id']==driver['component_id']);q=next(s['q_rad']for s in episode['constant_joint_states']if s['id']==joint['id'])
                        relative=M.forward(np.eye(4),joint['parent_from_joint'],joint['axis'],q,joint['joint_from_child_zero'])
                        root_matrix=root_matrix@np.linalg.inv(relative)
                    poses={g['root_component_id']:root_matrix.tolist()}
                    for joint in related:
                        q=next(s['q_rad']for s in episode['constant_joint_states']if s['id']==joint['id'])
                        poses[joint['child_id']]=M.forward(root_matrix,joint['parent_from_joint'],joint['axis'],q,joint['joint_from_child_zero']).tolist()
                    trial=copy.deepcopy(p);trial['component_poses']=poses
                    if p.get('source_sample_index')is not None:
                        check=T.source_mask_check(trial,None,ev[p['source_sample_index']],parts,meshes,k,size,root=root,policy=d['policy'])
                        if check['status']=='failed':continue
                        trial['independent_target_mask_check']=check
                    trial['support']='one_sided_near_anchor_RGB';trial['reason']=None
                    trial['selection_method']='nearest_source_track_that_reproduces_exact_feature_pixels_and_passes_independent_existing_mask'
                    trial['view_segment_id']=episode['id']+(':before'if label=='past'else':after')if episode['view_change_break']else episode['id']+':same_source_view'
                    selected=trial;break
                changes.append({'proof_id':p['flow_proof_id'],'old_support':original_support,'feature_reprojection_errors':[{'side':a,'median_px':e,'anchor_distance_s':t}for a,_,e,t in errors],
                    'decision':'selected_near_source_anchor'if selected else'withheld'})
                if selected:p.update(selected)
                else:p.update(status='withheld',reason='fused_or_one_sided_pose_does_not_reproduce_source_features',component_poses={},joint_states=[])
    d['parent_motion_hypothesis_ref']=hypothesis_ref;d['additional_implementation_fingerprints']=[implementation]
    def read(pointer,maximum=None):return (root/pointer['path']).read_bytes()
    C.validate(d,read)
    T.write_new(out/'selection-report.json',{'format':'interaction-twin-temporal-source-agreement-selection','version':1,'input':hypothesis_ref,'changes':changes,
        'no_original_observations_or_assets_modified':True,'physical_accuracy_verified':False,'new_API_or_GPU_calls':0})
    d['selection_report_ref']=T.ref(out/'selection-report.json',root)
    T.write_new(out/'hypotheses.json',d)
    return {'hypotheses':T.ref(out/'hypotheses.json',root),'selection_report':d['selection_report_ref'],'changed':len(changes),
        'estimated':sum(p['status']=='estimated'for g in d['groups']for e in g['episodes']for p in e['predictions'])}


def check_original_targets(hypothesis_ref,output,*,root=T.ROOT):
    """Evaluate conditional SAM2 poses against untouched existing target masks."""
    root=Path(root).resolve();out=root/output;T.require(not out.exists(),'New original-target temporal evaluation required');out.mkdir(parents=True)
    implementation=T.ref(Path(__file__),root);d=T.read_checked(hypothesis_ref,root);scene=T.read_checked(d['baseline_scene_ref'],root)
    program=T.read_checked(d['evidence_program_ref'],root);parts={c['id']:c for c in scene['components']};ev={e['source_index']:e for e in program['evidence']}
    meshes={};k=np.asarray(d['camera']['K']);size=(d['camera']['width'],d['camera']['height']);checks=[]
    for g in d['groups']:
        for episode in g['episodes']:
            for p in episode['predictions']:
                if p['status']!='estimated'or p.get('support')!='source_SAM_mask_conditioned_pose'or p.get('source_sample_index')is None:continue
                check=T.source_mask_check(p,None,ev[p['source_sample_index']],parts,meshes,k,size,root=root,policy=d['policy'])
                p['independent_target_mask_check']=check
                checks.append({'proof_id':p['flow_proof_id'],'source_sample_index':p['source_sample_index'],'check':check})
                if check['status']=='failed':p.update(status='withheld',reason='untouched_original_target_mask_disagrees_with_conditional_pose',component_poses={},joint_states=[])
    T.write_new(out/'selection-report.json',{'format':'interaction-twin-temporal-source-agreement-selection','version':1,'input':hypothesis_ref,'checks':checks,
        'no_original_observations_or_assets_modified':True,'physical_accuracy_verified':False,'new_API_or_GPU_calls':0,
        'target_old_masks_used_for_new_SAM2_or_pose_fit':False})
    d['parent_motion_hypothesis_ref']=hypothesis_ref
    d['additional_implementation_fingerprints']=[*d.get('additional_implementation_fingerprints',[]),implementation]
    d['selection_report_ref']=T.ref(out/'selection-report.json',root)
    C.validate(d,lambda r,maximum=None:(root/r['path']).read_bytes())
    T.write_new(out/'hypotheses.json',d)
    return {'hypotheses':T.ref(out/'hypotheses.json',root),'checks':checks}
