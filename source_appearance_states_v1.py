"""Read-only, source-observed vertex RGB states. No electronic state simulation.

Compact patches change display RGB only. They never change original/derived
GLBs, geometry, poses or alpha. Native appearance is restored outside explicitly
supported saved-source intervals; that fallback is not an observed screen state.
"""
from __future__ import annotations
import hashlib,json,math,struct
from pathlib import Path

FORMAT='interaction-twin-source-appearance-states'
MAX_BYTES=8*1024*1024
POLICY={'representation':'exact_source_RGB_vertex_patch_with_native_fallback',
 'hidden_vertices':'unchanged_approved_native_appearance',
 'patch_interpolation':'none','unknown_rows_are_barriers':True,
 'maximum_nearest_observation_offset_s':.36,
 'state_transition_gap':'unknown_native_fallback_not_electronic_state',
 'restore_native_before_each_patch':True,'electronics_simulation':False}

def require(value,message):
    if not value:raise ValueError(message)

def decode_patch(raw):
    require(isinstance(raw,bytes)and 16<=len(raw)<=MAX_BYTES,'Bounded compact RGB patch required')
    magic,version,total,count=struct.unpack_from('<4sIII',raw)
    require(magic==b'ITCP'and version==1 and 0<count<=total<=1000000 and len(raw)==16+count*11,'Invalid compact RGB header or record count')
    rows=list(struct.iter_unpack('<IHHBBB',raw[16:]));last=-1
    for i,x,y,r,g,b in rows:
        require(last<i<total,'Sorted unique native vertex indices required');last=i
    return {'vertex_count':total,'records':rows}

def validate(document,read_bytes):
    require(document.get('format')==FORMAT and document.get('version')==1 and document.get('display_policy')==POLICY,'Unknown source appearance contract/policy')
    require(document.get('physical_accuracy_verified')is False and document.get('simulation_ready')is False and document.get('electronic_state_verified')is False,'Appearance cannot claim physical/electronic validation')
    def checked(ref,decode=True):
        require(isinstance(ref,dict)and set(ref)=={'path','sha256','bytes'}and isinstance(ref['path'],str)and ref['path'].startswith('data/')and '..'not in ref['path'].split('/'),'Exact ordinary data artifact required')
        raw=read_bytes(ref,maximum=MAX_BYTES)
        require(isinstance(raw,bytes)and 0<len(raw)<=MAX_BYTES and len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256'],'Appearance artifact hash/size differs')
        return json.loads(raw)if decode else raw
    previous=checked(document['previous_derived_assembly_ref']);scene=checked(document['scene_ref']);program=checked(document['evidence_program'])
    require(document['scene_ref']==previous['scene_ref']and document['evidence_program']==previous['evidence_program']and document['source']==previous['source']==scene['source']and all(document['source'].get(k)==v for k,v in program['source'].items()),'Appearance source/approved scene differs')
    identifier=document['component_id'];binding=next((c for c in previous['component_assets']if c['component_id']==identifier),None);component=next((c for c in scene['components']if c['id']==identifier),None)
    require(binding is not None and component is not None and document['baseline_asset_ref']==binding['native_glb']==component['asset'],'Appearance native asset binding differs')
    require(1<=document['vertex_count']<=1000000,'Bounded vertex count required');vertex_count=document['vertex_count']
    poses={p['source_sample_index']:p for p in component['poses']};evidence={p['source_index']:p for p in program['evidence']}
    keys={}
    for key in document['keyframes']:
        require(key['id']not in keys,'Duplicate appearance keyframe');keys[key['id']]=key
        n=key['source_sample_index'];p=poses.get(n);e=evidence.get(n)
        require(p and e and p['status']=='estimated'and key['timestamp_s']==p['timestamp_s']==e['timestamp_s'],'Keyframe must have exact supported source clock')
        require(key['source_rgb_ref']['path']==e['frame_path']and key['source_rgb_ref']['sha256']==e['frame_sha256']and key['source_mask_ref']==p['source_mask'],'Keyframe RGB/SAM differs from exact source pose')
        checked(key['source_rgb_ref'],False);checked(key['source_mask_ref'],False)
        require(key['camera_from_native_mesh']==p['camera_from_native_mesh']and key['source_camera']==scene['camera'],'Appearance projection camera/pose changed')
        patch=decode_patch(checked(key['patch_ref'],False));require(patch['vertex_count']==vertex_count and len(patch['records'])==key['painted_vertex_count'],'Patch vertex count differs')
        require(all(x<document['source']['width']and y<document['source']['height']for _,x,y,*_ in patch['records']),'Patch source pixel outside RGB')
        require(key['native_fallback_vertex_count']==vertex_count-key['painted_vertex_count'],'Native fallback count differs')
    require(1<=len(keys)<=8,'Bounded actual source keyframes required')
    rows=document['observations'];require(len(rows)==len(component['poses'])and len(rows)<=240,'All source rows including unknowns required')
    for row,p in zip(rows,component['poses']):
        e=evidence[p['source_sample_index']]
        require(row['source_sample_index']==p['source_sample_index']and row['timestamp_s']==p['timestamp_s']==e['timestamp_s'],'Appearance observation source clock differs')
        if 'source_rgb_ref'in row or 'source_mask_ref'in row:
            require(row.get('source_rgb_ref',{}).get('path')==e['frame_path']and row['source_rgb_ref']['sha256']==e['frame_sha256']and row.get('source_mask_ref')==p['source_mask'],'Optional unknown-row source RGB/mask differs')
            checked(row['source_rgb_ref'],False);checked(row['source_mask_ref'],False)
        if row['status']=='unknown':require(row['keyframe_id']is None,'Unknown appearance cannot carry a patch');continue
        require(row['status']=='source_observed_appearance_estimate'and p['status']=='estimated'and row['keyframe_id']in keys,'Appearance is unsupported by known source geometry')
        key=keys[row['keyframe_id']];require(row['appearance_label']==key['appearance_label'],'Observation/keyframe appearance labels differ')
        require(row['source_rgb_ref']['path']==e['frame_path']and row['source_rgb_ref']['sha256']==e['frame_sha256']and row['source_mask_ref']==p['source_mask'],'Observation does not bind actual source RGB/mask')
        checked(row['source_rgb_ref'],False);checked(row['source_mask_ref'],False)
    return {'document':document,'previous_derived_assembly':previous,'scene':scene,'keyframes':keys}

def at(document,time_s):
    require(type(time_s)in(int,float)and math.isfinite(time_s),'Finite source time required')
    rows=document['observations'];nearest=min(rows,key=lambda p:abs(p['timestamp_s']-time_s))
    if abs(nearest['timestamp_s']-time_s)>POLICY['maximum_nearest_observation_offset_s']or nearest['status']=='unknown':return None
    before=next((r for r in reversed(rows)if r['timestamp_s']<=time_s),None);after=next((r for r in rows if r['timestamp_s']>=time_s),None)
    if before is None or after is None:return None
    if before['status']=='unknown'or after['status']=='unknown'or before['appearance_label']!=after['appearance_label']:return None
    return next(k for k in document['keyframes']if k['id']==nearest['keyframe_id'])

def validate_files(document,*,root):
    root=Path(root).resolve()
    def read(ref,*,maximum=MAX_BYTES):
        path=root/ref['path'];require(path.resolve().is_relative_to(root/'data')and not any(p.is_symlink()for p in[path,*path.parents]),'Ordinary data path required');raw=path.read_bytes();require(len(raw)<=maximum,'Appearance artifact exceeds byte budget');return raw
    return validate(document,read)
