"""Lightweight exact-source transport fence for source-visible depth surfaces.

No numpy, model runtime, token, checkpoint or inference dependency is imported.
Only explicitly enumerated data artifacts can enter the returned inventory.
"""
from pathlib import Path,PurePosixPath
import hashlib,json,re
ROOT=Path(__file__).resolve().parent
FORMAT='interaction-twin-source-visible-flexure-series'
SHA=re.compile(r'^[a-f0-9]{64}$')
def require(value,message):
    if not value:raise ValueError(message)
def reference(value):
    require(isinstance(value,dict),'Exact reference object required');r={k:value[k]for k in ['path','sha256','bytes']if k in value}
    require(set(r)=={'path','sha256','bytes'},'Incomplete reference');p=r['path'];require(isinstance(p,str)and p.startswith('data/')and re.fullmatch(r'[A-Za-z0-9_./-]+',p)and all(x and not x.startswith('.')for x in p.split('/')),'Only safe explicit data paths are admitted')
    require(SHA.fullmatch(r['sha256'])is not None and isinstance(r['bytes'],int)and not isinstance(r['bytes'],bool)and 0<r['bytes']<=64*1024*1024,'Invalid reference fingerprint/bound');return r
def checked(r,read_bytes):
    r=reference(r);raw=read_bytes(r);require(isinstance(raw,bytes)and len(raw)==r['bytes']and hashlib.sha256(raw).hexdigest()==r['sha256'],'Exact artifact bytes differ: '+r['path']);return raw
def validate(document,read_bytes,_derivation_depth=0):
    d=document;require(isinstance(d,dict)and d.get('format')==FORMAT and d.get('version')==1 and d.get('kind')=='source_observed_monocular_depth_prior','Unknown observed surface contract')
    require(d.get('physical_accuracy_verified')is False and d.get('simulation_ready')is False and d.get('manual_geometry_or_poses_used')is False and d.get('native_SAM_contract_compatible')is False,'Observed surface scope has been changed')
    scene=json.loads(checked(d['source_scene'],read_bytes));program=json.loads(checked(d['source_program'],read_bytes));require(d['source_program']==scene['original_program'],'Observed surface program identity differs');require(d['source']==scene['source']and d['camera']==scene['camera'],'Observed surface camera/source differs');require(program['source']['sha256']==d['source']['sha256'],'Original video identity differs')
    require(d['policy']['maximum_source_time_error_s']==.18 and d['policy']['timing']=='nearest_source_keyframe_within_0.18s_only_no_geometric_interpolation','Surface timing limit widened');require(d['policy']['hidden_surfaces']=='absent_not_inpainted'and d['policy']['numerical_fold_parameters']=='unknown_not_identifiably_recovered','Unknown surfaces or joints cannot be completed')
    ids=d['replacement_component_ids'];require(isinstance(ids,list)and ids and len(set(ids))==len(ids)and set(ids)<={c['id']for c in scene['components']},'Replacement component identity differs')
    inventory={}
    def admit(r):
        r=reference(r);old=inventory.get(r['path']);require(old is None or old==r,'Conflicting exact artifact reference');inventory[r['path']]=r;return r
    for key in ['source_scene','source_program','protocol']:admit(d[key])
    protocol=json.loads(checked(d['protocol'],read_bytes));require(protocol.get('source_scene')==d['source_scene']and protocol.get('source_program')==d['source_program']and protocol.get('source_video_sha256')==d['source']['sha256'],'Protocol binding differs')
    require(protocol.get('physical_accuracy_verified')is False and protocol.get('network_access_during_inference')is False and protocol.get('new_GPU_calls')==0 and protocol.get('depth_model',{}).get('device')=='cpu','Depth-prior scope differs')
    require('path'not in protocol['depth_model']and SHA.fullmatch(protocol['depth_model']['checkpoint_sha256'])is not None,'Checkpoint/runtime paths are not an artifact route')
    proof_version=d.get('source_footprint_proof_version',1);require(type(proof_version)is int and proof_version in [1,2],'Unknown source footprint proof version')
    parent_rows={}
    if proof_version==2:
        require(_derivation_depth==0,'Cached depth derivation chains are bounded to one original descriptor')
        require(d['policy'].get('triangle_support')=='all_unit_pixel_squares_intersecting_closed_triangle_inside_original_2px_SAM_core','Conservative source-pixel footprint policy missing')
        parent_ref=admit(d['cached_depth_parent']);require(protocol.get('cached_depth_parent')==parent_ref,'Cached-depth protocol parent differs')
        parent_doc=json.loads(checked(parent_ref,read_bytes));require(parent_doc.get('source_footprint_proof_version',1)==1,'Cached depth parent must use the original proof')
        parent_result=validate(parent_doc,read_bytes,_derivation_depth+1);parent=parent_result['document']
        for key in ['source_scene','source_program','source','camera','replacement_component_ids','unaffected_component_ids','closed_native_assets']:
            require(d.get(key)==parent.get(key),'Cached depth parent binding differs: '+key)
        # Closed exact dependency inventory: a downloaded derivative can
        # revalidate its parent without an unlisted filesystem dependency.
        for artifact in parent_result['artifacts']:admit(artifact)
        parent_rows={r['source_sample_index']:r for r in parent['samples']}
    else:
        require('cached_depth_parent'not in d,'Original footprint proof cannot silently carry a cached-depth derivative')
    expected_native=sorted([reference(c['asset'])for c in scene['components']if c['id']in ids],key=lambda r:r['path']);actual_native=sorted([reference(r)for r in d.get('closed_native_assets',[])],key=lambda r:r['path']);require(actual_native==expected_native,'Original surface assets differ from exact replaced components')
    rows=d['samples'];evidence=program['evidence'];require(len(rows)==len(evidence)and 0<len(rows)<=20000,'Sample ledger differs');native={c['id']:c for c in scene['components']};old_unknown={r['source_sample_index']for r in native[ids[0]]['poses']if r['status']=='unknown'};supported=[];unknown=[]
    for row,e in zip(rows,evidence):
        index=e['source_index'];require(row['source_sample_index']==index and abs(row['timestamp_s']-e['timestamp_s'])<=1e-6,'Source sample/time differs');require(row['status']in ['unknown','source_observed_prior'],'Invalid surface state')
        if proof_version==2:
            require(index in parent_rows,'Cached-depth parent source is absent')
            if parent_rows[index]['status']=='unknown':require(row['status']=='unknown','Cached-depth parent unknown barrier was overwritten')
        state=row.get('qualitative_state',{}).get('wallet_state')
        if index in old_unknown or state in ['occluded','uncertain']:require(row['status']=='unknown','Original or explicitly occluded unknown barrier was overwritten')
        if row['status']=='unknown':require('asset'not in row,'Unknown sample exposes a surface');unknown.append(index);continue
        supported.append(index);require(row.get('physical_accuracy_verified')is False and row.get('source_visible_only')is True and row.get('triangle_source_footprint_verified')is True,'Source surface proof fence missing');require(row.get('hidden_backs_culled')is True and row.get('unlit_exact_vertex_RGB')is True,'Hidden/back appearance policy differs')
        require(row['source_RGB']['path']==e['frame_path']and row['source_RGB']['sha256']==e['frame_sha256'],'Surface RGB source differs');admit(row['source_RGB'])
        observations={o['track_id']:o for o in e['observations']}
        for r in row['source_masks']:
            require(r['track_id']in observations,'Surface mask track differs');o=observations[r['track_id']];require(r['path']==o['mask_path']and r['sha256']==o['mask_sha256'],'Surface mask identity differs');admit(r)
        for key in ['asset','source_pixel_map','provenance']:
            r=admit(row[key]);require(r['path'].startswith(str(PurePosixPath(d['protocol']['path']).parent)+'/source-'),'Surface-owned artifact escaped package')
        require(row['vertices']<=1800 and row['triangles']>0,'Surface mesh budget invalid');require(row['camera_from_asset']==[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]],'Observed source-camera surface was repositioned')
        flex=row['virtual_flexure'];require(flex.get('status')=='unknown'and flex.get('angle_rad')is None and flex.get('axis_camera')is None and flex.get('origin_camera')is None,'Numerical physical joint was invented')
        require('path'not in row['depth_model']and SHA.fullmatch(row['depth_model']['checkpoint_sha256'])is not None,'Surface depth model must be a nonroutable fingerprint')
        if proof_version==2:
            previous=parent_rows[index];require(row.get('source_footprint_proof_version')==2,'Per-frame footprint proof differs')
            require(row.get('topology_refined_from_cached_depth')is True and row.get('geometry_unchanged_from_footprint_verified_parent')is not True,'Adaptive geometry must be declared as retessellated, not unchanged')
            require(row['source_RGB']==previous['source_RGB']and row['source_masks']==previous['source_masks']and row['depth_model']==previous['depth_model'],'Cached-depth source evidence or model differs')
            require(row.get('cached_depth_parent_map')==previous['source_pixel_map'],'Cached-depth map is not the exact matching original sample');admit(row['cached_depth_parent_map'])
            sampling=row.get('depth_sampling',{});stride=previous['source_grid_stride_px'];require(sampling.get('method')=='cached_monocular_prior_barycentric_or_bounded_nearest_not_measured_depth'and sampling.get('parent_grid_stride_px')==stride,'Cached-depth sampling method differs')
            # The heavy verifier checks all weights, source coordinates and
            # conservative support, rather than trusting these declarations.
            import math
            require(sampling.get('maximum_nearest_sampling_distance_px')==float(math.ceil(math.sqrt(2)*(stride+2)))and sampling.get('maximum_interpolation_parent_edge_px')==float(stride*2.5),'Cached-depth sampling bounds differ')
        else:require('cached_depth_parent_map'not in row and row.get('source_footprint_proof_version',1)==1,'Original source-pixel proof was relabelled')
    return {'document':d,'artifacts':list(inventory.values()),'supported_sources':supported,'unknown_sources':unknown,'source_scene':scene,'source_program':program,'protocol':protocol,'new_inference_calls':0}
def validate_descriptor(reference_value,root=ROOT):
    root=Path(root).resolve()
    def read(r):
        p=root/r['path'];require(not p.is_symlink()and p.resolve().is_relative_to(root),'Artifact escaped data root');return p.read_bytes()
    document=json.loads(checked(reference_value,read));result=validate(document,read);result['artifacts']=[reference(reference_value),*result['artifacts']];return result
