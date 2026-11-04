"""Saved 2D cable observations, explicitly separate from rigid 3D replay.

The stdlib runtime reader verifies hashes and source/clock/coordinate contracts.
Only the explicit offline builder imports and recomputes skeleton geometry.
No depth, invisible span, endpoint identity, interpolation or rod simulation.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
FORMAT='interaction-twin-nonrigid-observations'
MAX_BYTES=8*1024*1024


def require(v,m):
    if not v:raise ValueError(m)


def reference(path,root=ROOT):
    root=Path(root).resolve();path=Path(path);path=path if path.is_absolute()else root/path
    require(path.resolve()==path and path.is_relative_to(root/'data') and not any(p.is_symlink()for p in[path,*path.parents]),'Ordinary data observation artifact required')
    raw=path.read_bytes();require(0<len(raw)<=MAX_BYTES,'Bounded observation JSON required')
    return {'path':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def validate(document, read_bytes):
    require(isinstance(document,dict)and document.get('format')==FORMAT and document.get('version')==1,'Unknown nonrigid observation contract')
    require(document.get('depth')=='undefined' and document.get('physical_accuracy_verified')is False and document.get('simulation_ready')is False,'2D observations cannot claim depth or simulation')
    policy=document.get('display_policy',{})
    require(policy=={'rigid_native_pose_replay':'withheld_for_known_nonrigid_component','native_static_asset_inspection_retained':True,
        'viewer_representation':'source_pixel_polyline_overlay_video_camera_only','temporal_interpolation':'none',
        'maximum_nearest_sample_offset_s':.36,'unknown_rows_are_barriers':True,'display_between_samples':'nearest_saved_annotation_estimate_not_motion_measurement'},'Nonrigid display policy changed')
    def checked(ref):
        require(isinstance(ref,dict)and set(ref)=={'path','sha256','bytes'} and ref['path'].startswith('data/'),'Exact data reference required')
        raw=read_bytes(ref,maximum=MAX_BYTES);require(isinstance(raw,bytes)and len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256'],'Observation artifact changed')
        return json.loads(raw)
    program=checked(document['evidence_program']);trace=checked(document['trace_ref'])
    require(document['source']==program['source'] and trace['program_ref']==document['evidence_program'] and trace['source_sha256']==program['source']['sha256'],'Nonrigid source lineage differs')
    track=next((r for r in program['tracklets']if r['id']==document.get('component_id')),None)
    require(track is not None and trace['track_id']==track['id'] and any('deform' in str(x).lower()for x in track.get('limitations',[])),'Explicit model-described nonrigid track required')
    require(trace.get('format')=='interaction-twin-cable-trace-2d' and trace.get('version')==1,'Saved cable trace format required')
    rows=trace['samples'];require(isinstance(rows,list)and 1<=len(rows)<=90,'Bounded cable samples required')
    evidence={e['source_index']:e for e in program['evidence']};last=-1.
    for row in rows:
        index=row['source_index'];source=evidence.get(index)
        require(source and row['timestamp_s']==source['timestamp_s'] and row['evidence_id']==source['id'] and last<row['timestamp_s'],'Original source sample/PTS changed')
        last=row['timestamp_s'];require(row['frame_ref']['path']==source['frame_path']and row['frame_ref']['sha256']==source['frame_sha256'],'Source RGB identity differs')
        observation=next((x for x in source['observations']if x['track_id']==track['id']),None)
        if row.get('mask_ref')is not None:require(observation and row['mask_ref']['path']==observation['mask_path']and row['mask_ref']['sha256']==observation['mask_sha256'],'Source SAM mask identity differs')
        t=row.get('trace')
        if t is None:continue
        require(t['coordinate_system']=='original_binary_mask_pixels_top_left_xy'and t['endpoint_identity']=='unassigned_lexicographic_order_not_connector_identity','2D coordinate/endpoint scope differs')
        width,height=t['mask_size_px'];points=t.get('points_px',[])
        require(isinstance(points,list)and 0<=len(points)<=12000 and all(isinstance(p,list)and len(p)==2 and all(type(x)in(int,float)and math.isfinite(x)for x in p)and 0<=p[0]<width and 0<=p[1]<height for p in points),'Bounded source-pixel path required')
    require(document['sample_count']==len(rows)and document['source_time_domain']=='source_video_presentation_seconds','Source clock scope differs')
    return trace


def build(capture,*,root=ROOT):
    root=Path(root).resolve();require(capture in{'charger','laptop'},'Explicit existing whole-cable capture required')
    import cable_trace_2d as T
    original=next((root/'output/new-captures-20261103/semantic-review-v1'/capture).glob('cable-trace-2d-*.json'))
    raw=original.read_bytes();trace=json.loads(raw);T.validate(trace,root=root)
    program=json.loads((root/trace['program_ref']['path']).read_bytes());folder=root/'data/nonrigid-observations'/f'{capture}-20261104-v1';folder.mkdir(parents=True,exist_ok=False)
    (folder/'source-cable-trace.json').write_bytes(raw)
    document={'format':FORMAT,'version':1,'evidence_program':trace['program_ref'],'source':program['source'],'source_time_domain':trace['source_ref']and program['source']['time_domain'],
        'component_id':trace['track_id'],'trace_ref':reference(folder/'source-cable-trace.json',root),'sample_count':len(trace['samples']),
        'depth':'undefined','endpoint_physical_identity':'unknown','invisible_spans_completed':False,'physical_accuracy_verified':False,'simulation_ready':False,
        'display_policy':{'rigid_native_pose_replay':'withheld_for_known_nonrigid_component','native_static_asset_inspection_retained':True,
            'viewer_representation':'source_pixel_polyline_overlay_video_camera_only','temporal_interpolation':'none',
            'maximum_nearest_sample_offset_s':.36,'unknown_rows_are_barriers':True,'display_between_samples':'nearest_saved_annotation_estimate_not_motion_measurement'},
        'limitations':trace['limitations']+['Only one selected visible skeleton path may be present when the source mask has branches/disconnected spans; physical endpoints and depth remain unknown.'],
        'original_trace_sha256':hashlib.sha256(raw).hexdigest(),'fresh_skeleton_recomputation_verified':True,'new_api_calls':0,'new_gpu_calls':0}
    def read_bytes(ref,*,maximum=MAX_BYTES):
        raw=(root/ref['path']).read_bytes();require(len(raw)<=maximum,'Observation artifact exceeds the runtime size bound');return raw
    validate(document,read_bytes)
    path=folder/'nonrigid-observations.json';path.write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')
    return reference(path,root)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',choices=['charger','laptop']);a=p.parse_args();print(json.dumps(build(a.capture)),flush=True)
