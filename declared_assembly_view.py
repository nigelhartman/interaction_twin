"""Fast read-only view of an already approved geometric-derivative snapshot.

The publication path runs refined_assembly_contract.validate, including actual
mesh reproduction and all source-mask projections. This reader intentionally
does no fitting, image/mesh decoding or inference. Its callback MUST enforce the
owner-curated exact artifact allowlist and hash/length checks. It then verifies
the approved JSON lineage, clocks, selected assets, fixed joint and conservative
unknown/facing policy without importing model or computer-vision dependencies.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import PurePosixPath
import re
from statistics import median

FORMAT = 'interaction-twin-declared-derived-assembly'
MAX_JSON = 8*1024*1024
POLICY = {
    'maximum_parent_rotation_component_rad': .1745,
    'maximum_parent_translation_extent_fraction': .18,
    'maximum_parent_depth_fraction': .18,
    'maximum_fit_evaluations': 450,
    'sparse_vertex_target': 1400,
    'angle_grid_count': 65,
    'minimum_child_convex_mask_iou': .55,
    'minimum_union_convex_mask_iou': .60,
    'maximum_child_iou_regression': .12,
    'minimum_retained_samples': 12,
    'minimum_retained_fraction_of_original_supported': .50,
    'minimum_supported_child_median_iou': .75,
    'minimum_supported_union_median_iou': .65,
    'maximum_joint_speed_rad_per_s': 12.,
    'cavity_facing_ambiguity_gate': 'withhold_bracketed_open_when_parent_cavity_faces_camera_and_child_cavity_faces_away',
    'minimum_parent_cavity_facing_cosine': .15,
    'minimum_child_cavity_facing_cosine_for_bracketed_open': .05,
    'fill_original_unknowns': False,
    'mask_clipping_or_mesh_hiding': False,
    'physical_accuracy_verified': False,
}
CLAIMS = {'original_native_assets_preserved': True, 'declared_derivative_geometry_used': True,
          'manual_geometry_or_poses_used': False, 'conditional_convex_source_mask_check_performed': True,
          'physical_accuracy_verified': False, 'simulation_ready': False, 'physical_robot_execution_available': False}
JOINT_FIELDS = ['id','type','parent_id','child_id','status','axis','parent_from_joint','joint_from_child_zero',
                'parent_anchor_native','child_anchor_native','parent_anchor_vertex_id','child_anchor_vertex_id',
                'parent_native_scale','child_native_scale','limits_rad','limits_kind','physical_limits_rad',
                'zero_reference','angle_quality','evidence_ids','matrix_convention','physical_accuracy_verified','simulation_ready']


def require(value, message):
    if not value:
        raise ValueError(message)


def reference(value, maximum=32*1024*1024):
    require(isinstance(value,dict) and set(value)=={'path','sha256','bytes'}, 'Exact declared artifact reference required')
    name=value['path']
    require(isinstance(name,str) and name.startswith('data/') and '\\' not in name and
            re.fullmatch(r'[A-Za-z0-9_./-]+',name) and not any(p in {'','.','..'} or p.startswith('.') for p in name.split('/')) and
            str(PurePosixPath(name))==name, 'Only ordinary declared data paths are allowed')
    require(isinstance(value['sha256'],str) and re.fullmatch(r'[a-f0-9]{64}',value['sha256']) and
            type(value['bytes']) is int and 0<value['bytes']<=maximum, 'Bounded declared hash/length required')
    return value


def number(value):
    return type(value) in {int,float} and math.isfinite(value)


def matrix(value):
    require(isinstance(value,list) and len(value)==4 and all(isinstance(r,list) and len(r)==4 and all(number(v) for v in r) for r in value) and
            value[3]==[0,0,0,1], 'Finite homogeneous joint matrix required')
    return value


def _facing(transform, normal):
    matrix(transform)
    require(isinstance(normal,list) and len(normal)==3 and all(number(v) for v in normal), 'Finite native cavity normal required')
    n=[sum(transform[i][k]*normal[k] for k in range(3)) for i in range(3)]
    view=[-transform[i][3] for i in range(3)]
    a,b=math.sqrt(sum(v*v for v in n)),math.sqrt(sum(v*v for v in view))
    require(min(a,b)>1e-9, 'Nondegenerate cavity-facing frame required')
    return sum(x*y for x,y in zip(n,view))/a/b


def _bracket(index,hints):
    lower=[i for i in hints if i<=index];upper=[i for i in hints if i>=index]
    if not lower or not upper:return {'state':None,'source_sample_indices':[]}
    first,last=max(lower),min(upper)
    return {'state':hints[first] if hints[first]==hints[last] else None,'source_sample_indices':[first,last]}


def _index(rows,key,maximum=120):
    require(isinstance(rows,list) and 0<len(rows)<=maximum, 'Bounded nonempty declared inventory required')
    result={}
    for row in rows:
        require(isinstance(row,dict) and key in row and row[key] not in result, 'Duplicate or missing declared ID')
        result[row[key]]=row
    return result


def _project_joint(joint):
    return {**{key:joint[key] for key in JOINT_FIELDS},'axis_frame':'joint_frame_after_parent_from_joint',
            'samples':[{key:r[key] for key in ['source_sample_index','timestamp_s','status','q_rad']} for r in joint['samples']]}


def validate(document, read_bytes):
    """Validate a hash-approved read view; publication retains pixel validation."""
    if isinstance(document,dict) and document.get('format')=='interaction-twin-declared-geometry-derivative':
        import declared_geometry_view_v1
        declared_geometry_view_v1.validate(document,read_bytes)
        return document
    require(callable(read_bytes), 'Exact curated artifact callback required')
    json.dumps(document,allow_nan=False)
    require(isinstance(document,dict) and document.get('format')==FORMAT and document.get('version')==1 and
            document.get('physical_accuracy_verified') is False and document.get('simulation_ready') is False and
            document.get('claims')==CLAIMS, 'Unknown or physically upgraded declared assembly')
    cache={}
    def read(pointer,maximum=MAX_JSON):
        reference(pointer,maximum)
        key=(pointer['path'],pointer['sha256'],pointer['bytes'])
        if key not in cache:
            raw=read_bytes(pointer,maximum=maximum)
            require(isinstance(raw,(bytes,bytearray)) and len(raw)==pointer['bytes'] and hashlib.sha256(raw).hexdigest()==pointer['sha256'],
                    'Curated JSON hash or byte length changed')
            # The callback performs approved-allowlist AND hash validation.
            # This parse rejects nonstandard floating values before comparisons.
            value=json.loads(raw,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite declared JSON')))
            require(isinstance(value,dict), 'Declared JSON object required')
            cache[key]=value
        return cache[key]
    base=read(document['base_object_program_ref'])
    report=read(document['refinement_report_ref'])
    scene=read(document['scene_ref'])
    audit=read(document['source_alignment_audit_ref'])
    require(base.get('format')=='interaction-twin-object-program' and base.get('version')=='0.2' and
            base.get('physical_accuracy_verified') is False and base.get('simulation_ready') is False, 'Exact native descriptive baseline required')
    require(report.get('format')=='interaction-twin-native-assembly-refinement' and report.get('version')==1 and
            report.get('status')=='accepted_geometric_derivative' and report.get('physical_accuracy_verified') is False and
            report.get('manual_geometry_or_poses_used') is False and report.get('new_model_calls')==0, 'Declared mathematical refinement provenance changed')
    require(audit.get('format')=='interaction-twin-derived-assembly-source-audit' and audit.get('version')==1 and
            audit.get('policy')==POLICY and audit.get('new_api_calls')==audit.get('new_gpu_calls')==0 and
            audit.get('manual_geometry_or_poses_used') is False and audit.get('physical_accuracy_verified') is False,
            'Source audit lacks the approved facing/unknown policy')
    require(audit['base_object_program_ref']==document['base_object_program_ref'] and
            audit['refinement_report_ref']==document['refinement_report_ref'] and
            base['constrained_scene']==report['original_scene'] and report['original_program']==base['evidence_program']==document['evidence_program']==scene['original_program'],
            'Declared baseline/refinement/procedure lineage changed')
    require(document['source']==base['source']==scene['source'] and report['source_sha256']==base['source']['sha256'] and
            scene.get('physical_accuracy_verified') is False and scene.get('simulation_ready') is False and
            scene.get('manual_geometry_or_poses_used') is False, 'Declared source or research scope changed')
    require(scene.get('assembly_refinement',{}).get('source_audit')==document['source_alignment_audit_ref'], 'Scene does not cite the exact approved audit')
    original=read(report['original_scene']);unfit=read(report['scene_ref']);proposal=read(base['proposal'])
    require(original['source']==unfit['source']==scene['source'] and original['original_program']==unfit['original_program']==document['evidence_program'] and
            scene['camera']==unfit['camera']==original['camera'], 'Source frame or camera changed')
    parts=_index(scene['components'],'id',32);old_parts=_index(unfit['components'],'id',32)
    native_assets=_index(base['component_assets'],'component_id',32);declared_assets=_index(document['component_assets'],'component_id',32);report_assets=_index(report['assets'],'id',32)
    require(set(parts)==set(old_parts)==set(native_assets)==set(declared_assets)==set(report_assets), 'Declared/native component inventory changed')
    joint=scene['mechanics']['joints'][0]
    require(len(scene['mechanics']['joints'])==1 and joint['id']==report['joint']['id'] and
            joint['parent_id']==report['joint']['parent_id'] and joint['child_id']==report['joint']['child_id'], 'One exact declared joint required')
    pid,cid=joint['parent_id'],joint['child_id'];fixed=report['joint']['refined']
    require(pid in parts and cid in parts and pid!=cid, 'Unknown declared joint component')
    for field in ['id','type','parent_id','child_id','axis','parent_from_joint','joint_from_child_zero','limits_rad','parent_native_scale','child_native_scale']:
        require(joint[field]==fixed[field], 'Fixed geometric joint changed')
    matrix(joint['parent_from_joint']);matrix(joint['joint_from_child_zero'])
    require(len(joint['axis'])==3 and all(number(v) for v in joint['axis']) and abs(sum(v*v for v in joint['axis'])-1)<1e-7,
            'Finite unit joint axis required')
    require(document['joints']==[_project_joint(joint)] and document['removable_relationships']==base['removable_relationships'] and
            document['unresolved_connections']==base['unresolved_connections'], 'Declared semantic/joint projection changed')
    for part in parts:
        component,old,asset,native,reported=parts[part],old_parts[part],declared_assets[part],native_assets[part],report_assets[part]
        for field in ['native_glb','original_native_glb','native_metadata']:reference(asset[field])
        require(asset['native_glb']==component['asset']==old['asset']==reported['asset'] and
                asset['original_native_glb']==native['native_glb']==reported['original_asset'] and
                asset['native_metadata']==component['native_metadata']==old['native_metadata']==native['native_metadata'] and
                asset['native_uniform_scale']==component['native_uniform_scale']==old['native_uniform_scale'], 'Declared geometry/metadata bindings changed')
        changed=part==cid
        require(reported['geometry_modified'] is changed and
                asset['geometry_origin']==('automatic_bounded_native_rim_derivative' if changed else 'unchanged_SAM_3D_inference') and
                (asset['native_glb']!=asset['original_native_glb']) is changed, 'Derived mesh was relabelled as unchanged native generation')
        if part not in {pid,cid}:require(component==old, 'Source fit moved an unrelated component')
    rows=_index(audit['rows'],'source_sample_index')
    samples=_index(joint['samples'],'source_sample_index')
    poses=_index(parts[cid]['poses'],'source_sample_index');old_poses=_index(old_parts[cid]['poses'],'source_sample_index')
    parent_poses=_index(parts[pid]['poses'],'source_sample_index');old_parents=_index(old_parts[pid]['poses'],'source_sample_index')
    require(set(samples)==set(poses)==set(old_poses)==set(parent_poses)==set(old_parents), 'Original source sample inventory changed')
    original_supported={index for index,row in old_poses.items() if row['camera_from_native_mesh'] is not None}
    require(set(rows)==original_supported, 'Original unknown samples were filled or supported audit rows omitted')
    proposed=next(j for j in proposal['joints'] if j['id']==joint['id'])
    hints={row['source_sample_index']:row['state'] for row in proposed.get('state_hints',[])}
    cavity=report['joint']['original']['cavity_mating_prior'];retained=[];previous=None
    for index,pose in poses.items():
        old=old_poses[index];sample=samples[index]
        require(type(index) is int and index>0 and number(pose['timestamp_s']) and pose['timestamp_s']==old['timestamp_s']==sample['timestamp_s'] and
                pose.get('source_mask')==old.get('source_mask'), 'Source clock or mask binding changed')
        if index not in rows:
            require(pose['camera_from_native_mesh'] is None and pose['status']==sample['status']=='unknown' and sample['q_rad'] is None, 'An original unknown is now visible')
            continue
        row=rows[index]
        require(row['timestamp_s']==pose['timestamp_s'] and row['parent_mask']==old_parents[index]['source_mask'] and row['child_mask']==old['source_mask'], 'Audit source mask/sample binding changed')
        reference(row['parent_mask']);reference(row['child_mask'])
        p,c=row['parent_camera_from_native_mesh'],row['child_camera_from_native_mesh'];matrix(p);matrix(c)
        q=row['q_rad'];require(number(q) and joint['limits_rad'][0]<=q<=joint['limits_rad'][1], 'Estimated angle exceeds approved range')
        bracket=_bracket(index,hints)
        facing={'parent':_facing(p,cavity['parent']['normal']),'child':_facing(c,cavity['child']['normal'])}
        rejected=bool(bracket['state']=='open' and facing['parent']>POLICY['minimum_parent_cavity_facing_cosine'] and
                      facing['child']<POLICY['minimum_child_cavity_facing_cosine_for_bracketed_open'])
        require(row['source_state_bracket']==bracket and row['facing_ambiguity_rejected'] is rejected and
                all(number(row['cavity_facing_cosine'][key]) and abs(row['cavity_facing_cosine'][key]-facing[key])<1e-12 for key in facing), 'Cavity-facing ambiguity check changed')
        scores=row['mask_iou'];require(set(scores)=={'parent','child','union'} and all(number(v) and 0<=v<=1 for v in scores.values()) and
                number(row['original_child_convex_mask_iou']) and 0<=row['original_child_convex_mask_iou']<=1, 'Invalid approved pixel diagnostic')
        keep=bool(scores['child']>=POLICY['minimum_child_convex_mask_iou'] and scores['union']>=POLICY['minimum_union_convex_mask_iou'] and
                  scores['child']>=row['original_child_convex_mask_iou']-POLICY['maximum_child_iou_regression'] and not rejected)
        require(row['retained'] is keep, 'An ineligible/ambiguous pose bypassed the visibility gate')
        if keep:
            require(pose['status']==sample['status']=='estimated' and pose['camera_from_native_mesh']==c and
                    parent_poses[index]['camera_from_native_mesh']==p and sample['q_rad']==q, 'Visible scene differs from the approved audit')
            if previous:
                require(pose['timestamp_s']>previous[0] and abs(q-previous[1])/(pose['timestamp_s']-previous[0])<=POLICY['maximum_joint_speed_rad_per_s'], 'Joint curve exceeds its bounded source-time rate')
            previous=(pose['timestamp_s'],q);retained.append(row)
        else:
            require(pose['camera_from_native_mesh'] is None and pose['status']==sample['status']=='unknown' and sample['q_rad'] is None and
                    parent_poses[index]==old_parents[index], 'Rejected pose remained visible or moved its parent')
    child_median=median([r['mask_iou']['child'] for r in retained]) if retained else 0.
    union_median=median([r['mask_iou']['union'] for r in retained]) if retained else 0.
    eligible=bool(len(retained)>=POLICY['minimum_retained_samples'] and len(retained)>=len(rows)*POLICY['minimum_retained_fraction_of_original_supported'] and
                  child_median>=POLICY['minimum_supported_child_median_iou'] and union_median>=POLICY['minimum_supported_union_median_iou'])
    expected={'original_supported_samples':len(rows),'retained_samples':len(retained),'newly_withheld_samples':len(rows)-len(retained),
              'supported_child_median_convex_mask_iou':child_median,'supported_union_median_convex_mask_iou':union_median,
              'default_eligible':eligible,'physical_accuracy_verified':False,'original_unknown_samples':len(poses)-len(rows),
              'total_unknown_samples':len(poses)-len(retained)}
    require(audit['summary']==document['source_audit_summary']==expected and eligible, 'Declared eligibility/counts differ from approved rows')
    closed=[r for r in retained if r['semantic_state_hint']=='closed' and r['q_rad']==0.]
    require(closed and joint['zero_reference']['state']=='estimated_geometric_closed' and
            any(r['source_sample_index']==joint['zero_reference']['source_sample_index'] and r['timestamp_s']==joint['zero_reference']['timestamp_s'] for r in closed),
            'Source-cited geometric closed reference missing')
    return document
