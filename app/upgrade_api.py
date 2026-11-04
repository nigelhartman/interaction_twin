"""Local application boundary for saved-evidence model upgrades.

The browser chooses published IDs, never commands, credentials or output paths.
Model execution lives in the separately serialized upgrade runner.
"""
from __future__ import annotations

import json
import importlib.util
import math
from pathlib import Path
import re
import statistics

from twin_catalog import TwinCatalog

ROOT = Path(__file__).resolve().parent
RUN_ID = re.compile(r'upgrade-[a-z0-9-]{1,90}\Z')
JOB_ID = re.compile(r'job-[a-f0-9]{32}\Z')


def local_url(path, root=ROOT):
    from urllib.parse import quote
    root=Path(root).resolve();target=Path(path).resolve()
    if not target.is_relative_to(root/'data') or not target.is_file():
        raise ValueError('Upgrade link must identify an existing local data artifact')
    return '/files/'+quote(str(target.relative_to(root)),safe='/')


def metrics(doc, track_id):
    rows=[r for r in doc['motion'] if r['track_id']==track_id]
    return {'conditional_supported':sum(r['status']=='supported_conditional_motion_candidate' for r in rows),
            'reference_samples':sum(r['status'].startswith('reference_') for r in rows),
            'total_samples':len(doc['evidence']),
            'silhouette_iou_median':None,'residual_px_median':None,
            'scope':'Internal source-frame diagnostics; no physical pose truth'}


def catalog(*,root=ROOT):
    root=Path(root).resolve();store=TwinCatalog(root);projects=[];errors=[]
    for candidate in store.entries():
        try:
            row,doc,digest=store.load(candidate['id'])
            components=[]
            for track in doc['tracklets']:
                tid=track['id'];asset=next((a for a in doc['assets'] if a['track_id']==tid),None)
                source=None;meta=None
                if asset:
                    for item in doc['inputs']:
                        if item['kind']!='learned_asset_metadata':continue
                        path=store.data_path(item['path']);value=json.loads(path.read_text())
                        if value.get('object_id')==tid:
                            meta=value;break
                    if meta:
                        t=meta.get('source',{}).get('timestamp_s')
                        source=next((e for e in doc['evidence'] if isinstance(t,(float,int)) and abs(e['timestamp_s']-t)<1e-7),None)
                source=source or next((e for e in doc['evidence'] if any(o['track_id']==tid for o in e['observations'])),None)
                diagnostic=metrics(doc,tid)
                for item in doc['inputs']:
                    if item['kind']!='model_conditional_motion_experiment':continue
                    pack=json.loads(store.data_path(item['path']).read_text())
                    if pack.get('object_id')!=tid:continue
                    supported=[r for r in pack.get('frames',[]) if r['status']=='supported_conditional_motion_candidate']
                    for target,key in [('silhouette_iou_median','raw_full_projection_iou_with_sam'),('residual_px_median','heldout_median_px')]:
                        values=[r[key] for r in supported if isinstance(r.get(key),(float,int)) and not isinstance(r.get(key),bool) and math.isfinite(r[key])]
                        diagnostic[target]=statistics.median(values) if values else None
                components.append({'track_id':tid,'label':track['label'],'has_asset':bool(asset),
                    'backend':asset['backend'] if asset else None,'mesh_sha256':asset['sha256'] if asset else None,
                    'source_frame_url':local_url(root/source['frame_path'],root) if source else None,
                    'source_timestamp_s':source['timestamp_s'] if source else None,'baseline_metrics':diagnostic})
            projects.append({'id':row['id'],'title':row['title'],'base_sha256':digest,
                'video_url':local_url(root/doc['source']['path'],root),'duration_s':doc['source']['duration_s'],
                'poster_url':local_url(root/doc['evidence'][0]['frame_path'],root),'components':components})
        except (ValueError,KeyError,OSError,TypeError) as exc:
            errors.append({'id':candidate.get('id'),'error':str(exc)})
    return {'projects':projects,'errors':errors,'backends':[
        {'id':'refit_existing','label':'Refit current mesh','kind':'local','requires':'Existing Apple MPS runtime','available':True},
        {'id':'local_triposr','label':'Local reconstruction comparison','kind':'local','requires':'Prepared TripoSR / MPS','available':True},
        {'id':'refit_native_cpu','label':'Native-material refit on CPU','kind':'local','requires':'Prepared CPU CoTracker runtime; saved geometry and reasoning','available':True},
        {'id':'sam3d_objects','label':'SAM 3D Objects','kind':'external_result','available':False,'status':'Input bundle and returned result workflow; no GPU inference yet'},
        {'id':'sam31_image','label':'SAM 3.1 image comparison','kind':'external_result','available':False,'status':'Single-image mask comparison; not video identity or propagation'}],
        'policy':{'local_only':True,'hosted_reasoning':False,'Meta_API_calls':False,'cluster_contact':False,
                  'credentials_read':False,'physical_accuracy_verified':False,'simulation_ready':False}}


def canonical_request(body):
    allowed={'twin_id','track_id','base_sha256','mode','imported_result_id'}
    if not isinstance(body,dict) or set(body)-allowed:raise ValueError('Unknown model-upgrade request fields')
    if body.get('mode') not in {'refit_existing','local_triposr','imported_sam3d','refit_native_cpu'}:raise ValueError('Choose a supported upgrade mode')
    for key in ['twin_id','track_id']:
        if not isinstance(body.get(key),str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,180}',body[key]):raise ValueError('Invalid '+key)
    if not isinstance(body.get('base_sha256'),str) or not re.fullmatch(r'[a-f0-9]{64}',body['base_sha256']):raise ValueError('A saved program hash is required')
    imported=body.get('imported_result_id')
    if body['mode']=='imported_sam3d':
        if not isinstance(imported,str) or not re.fullmatch(r'import-[a-f0-9]{32}',imported):raise ValueError('Choose a verified imported SAM 3D result')
    elif imported is not None:raise ValueError('An imported result belongs only to the SAM 3D mode')
    return {k:v for k,v in body.items() if v is not None}


def plan(body,*,root=ROOT):
    from upgrade_runner import plan as build_plan
    return build_plan(canonical_request(body),root=Path(root).resolve())


def scene(twin_id,*,root=ROOT):
    """Rebuild one source-validated scene from the real GLB and pose artifacts."""
    root=Path(root).resolve();row,_,_=TwinCatalog(root).load(twin_id)
    spec=importlib.util.spec_from_file_location('upgrade_workspace_scene',ROOT/'scripts/build_workspace.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    builder.ROOT=root
    result=builder.payload(root/row['interaction_path'],row.get('title'))
    return {'case':result}


def run_record(identifier,*,root=ROOT):
    if not isinstance(identifier,str) or not RUN_ID.fullmatch(identifier):raise ValueError('Invalid upgrade run ID')
    root=Path(root).resolve();base=root/'data/model-upgrades/runs';path=base/identifier/'run.json'
    if path.resolve()!=path or not path.is_file():raise ValueError('Upgrade run is unavailable')
    value=json.loads(path.read_text())
    if value.get('id')!=identifier:raise ValueError('Upgrade receipt ID differs from its directory')
    return value


def runs(*,root=ROOT):
    base=Path(root).resolve()/'data/model-upgrades/runs';rows=[]
    for path in sorted(base.glob('upgrade-*/run.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        try:rows.append(run_record(path.parent.name,root=root))
        except (ValueError,OSError,KeyError):continue
    return {'runs':sorted(rows,key=lambda r:str(r.get('created_at','')),reverse=True)[:60]}


def job_view(value,*,root=ROOT):
    """The browser receives local links; CLI-only absolute paths stay internal."""
    result={k:v for k,v in value.items() if k not in {'directory','job_path','zip_path'}}
    result['id']=value.get('job_id')
    if value.get('status')!='invalid_saved_job':
        result['download_url']=local_url(Path(root)/value['zip_relative_path'],root)
        result['job_url']=local_url(Path(root)/value['job_relative_path'],root)
        result['source_frame_url']=local_url(Path(root)/value['image_path'],root)
        result['source_mask_url']=local_url(Path(root)/value['mask_path'],root)
    return result


def import_view(value,*,root=ROOT):
    result={k:v for k,v in value.items() if k!='directory'}
    result['id']=value.get('import_id')
    if value.get('comparison_path'):
        result['comparison_url']=local_url(Path(root)/value['comparison_path'],root)
    if value.get('native_asset_path'):
        result['asset_url']=local_url(Path(root)/value['native_asset_path'],root)
    return result
