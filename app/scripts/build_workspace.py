#!/usr/bin/env python3
"""Build an offline review payload from actual validated interaction artifacts."""
import argparse
import os
import tempfile
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import stage5
from stage4_generated_observer import source_projection,scene_layout


def read(p):return json.loads(Path(p).read_text())

def atomic_text(path,content):
    """A live browser must see either complete catalog generation, never a prefix."""
    path=Path(path)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                         prefix='.'+path.name+'.',delete=False) as stream:
            temporary=Path(stream.name)
            stream.write(content)
            stream.flush()
        os.replace(temporary,path)
        temporary=None
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
def relative(p):return str(Path(p).resolve().relative_to(ROOT))
def mesh_reader():
    spec=importlib.util.spec_from_file_location('workspace_mesh',ROOT/'scripts/stage3_reconstruction_review.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def static_sam3d_assets(doc):
    """Native GLBs are inspected separately until fresh motion is supported.

    Do not decode/duplicate high-resolution native meshes into a huge aggregate
    JSON scene just to inspect static independent candidates. Original program
    asset references, exports and per-component verified native viewers remain.
    Legacy geometry and accepted native motion keep their existing scene path.
    """
    assets=doc.get('assets',[])
    if not assets or not all(a.get('backend')=='Meta SAM 3D Objects' for a in assets):return False
    tracks={a['track_id'] for a in assets}
    return not any(m.get('track_id') in tracks and m.get('status')=='supported_conditional_motion_candidate'
                   and m.get('camera_from_native_mesh') is not None for m in doc.get('motion',[]))


def generic_scene(doc):
    """Independent assets retain their own motion without claiming assembly."""
    if not doc['assets']:return None
    if static_sam3d_assets(doc):return None
    reader=mesh_reader();objects=[];transformed=[];timeline=set()
    for asset in doc['assets']:
        primitives,stats=reader.mesh_primitives(ROOT/asset['path'])
        vertices=np.concatenate([np.asarray(p['vertices']) for p in primitives]);lo,hi=vertices.min(0),vertices.max(0);span=float(np.linalg.norm(hi-lo))
        motion=sorted([m for m in doc['motion'] if m['track_id']==asset['track_id']],key=lambda m:m['timestamp_s'])
        rows=[];component_xyz=[]
        for m in motion:
            allowed=m['status'] in stage5.ALLOWED_POSES and m['camera_from_native_mesh'] is not None
            row={'sample_index':m['source_sample_index'],'timestamp_s':m['timestamp_s'],'status':m['status'],
                 'visible':allowed,'reference':m['status'].startswith('reference_'),'reasons':m['reasons'],
                 'camera_from_native_mesh':m['camera_from_native_mesh'] if allowed else None}
            rows.append(row);timeline.add(row['timestamp_s'])
            if allowed:
                matrix=np.array(m['camera_from_native_mesh']);xyz=vertices@matrix[:3,:3].T+matrix[:3,3]
                transformed.append(xyz);component_xyz.append(xyz)
        period=float(np.median(np.diff([r['timestamp_s'] for r in rows]))) if len(rows)>1 else 0.
        label=next((t['label'] for t in doc['tracklets'] if t['id']==asset['track_id']),asset['track_id'])
        motion_layout=None;projection=None
        if component_xyz:
            xyz=np.concatenate(component_xyz);ml,mh=xyz.min(0),xyz.max(0);ms=float(np.linalg.norm(mh-ml))
            motion_layout={'target':((ml+mh)/2).tolist(),'span':ms,'initial_distance':ms*1.8}
            # Each independent reconstruction has its own native scale and
            # conditional camera. Match the experiment to this exact asset.
            for source in doc['inputs']:
                if source['kind']!='model_conditional_motion_experiment':continue
                pack=read(ROOT/source['path'])
                if pack.get('object_id')!=asset['track_id'] or pack.get('provenance',{}).get('mesh_sha256')!=asset['sha256']:continue
                frame_source=next(i for i in doc['inputs'] if i['kind']=='source_frame_index')
                frame=read(ROOT/frame_source['path'])['frames'][0]
                if ml[2]>0:
                    projection=source_projection(pack['intrinsics']['matrix'],frame['width'],frame['height'],max(.00001,float(ml[2])*.05),float(mh[2]+ms*8)).tolist()
                break
        objects.append({'object_id':asset['track_id'],'label':label,'primitives':primitives,'mesh_stats':stats,'frames':rows,
                        'native_scene':{'target':((lo+hi)/2).tolist(),'span':span,'initial_distance':span*1.8},
                        'motion_scene':motion_layout,'source_projection_rows':projection,
                        'max_hold_s':period*1.05,'fixed_depth_gauge_scale':1.,'provenance':{'mesh_file':asset['path'],'mesh_sha256':asset['sha256']}})
    # Independent native units must never be silently assembled together.
    native_only=len(objects)!=1 or not transformed
    if transformed and not native_only:
        xyz=np.concatenate(transformed);lo,hi=xyz.min(0),xyz.max(0);span=float(np.linalg.norm(hi-lo))
        layout={'target':((lo+hi)/2).tolist(),'span':span,'initial_distance':span*1.8}
    else:layout=objects[0]['native_scene']
    frame_index=next(i for i in doc['inputs'] if i['kind']=='source_frame_index');frame=read(ROOT/frame_index['path'])['frames'][0]
    projection=objects[0]['source_projection_rows'] if not native_only else None
    return {'objects':objects,'scene':layout,'native_only':native_only,'timeline':sorted(timeline),
            'initial_source_time_s':next((r['timestamp_s'] for o in objects for r in o['frames'] if r['visible']),doc['evidence'][0]['timestamp_s']),
            'source_projection_rows':projection,'image_size':[frame['width'],frame['height']],
            'source_duration_s':doc['source']['duration_s'],'physical_assembly_verified':False}


def scene_for(doc):
    if static_sam3d_assets(doc):return None
    path=doc.get('links',{}).get('scene_viewer')
    if path:
        contract=(ROOT/path).parent/'scene.json'
        if contract.exists():
            scene=read(contract)
            if scene_lineage_valid(scene,doc):
                # Re-embed the actual GLB, never geometry asserted by a JSON label.
                reader=mesh_reader();assets={a['track_id']:a for a in doc['assets']}
                for obj in scene['objects']:
                    obj['primitives'],obj['mesh_stats']=reader.mesh_primitives(ROOT/assets[obj['object_id']]['path'])
                return scene
    return generic_scene(doc)


def scene_lineage_valid(scene,doc):
    """Same-track asset and source/pose joins before accepting a saved scene."""
    try:
        if len(scene['objects'])>1 and not scene.get('depth_gauge',{}).get('selected_scales'):return False
        assets={a['track_id']:a for a in doc['assets']}
        experiments={str((ROOT/i['path']).resolve()):i['sha256'] for i in doc['inputs'] if i['kind']=='model_conditional_motion_experiment'}
        seen=set()
        for obj in scene['objects']:
            tid=obj['object_id'];prov=obj['provenance']
            if tid in seen or tid not in assets:return False
            seen.add(tid);asset=assets[tid]
            mesh=Path(prov['mesh_file']);mesh=mesh if mesh.is_absolute() else ROOT/mesh
            if mesh.resolve()!=(ROOT/asset['path']).resolve() or prov['mesh_sha256']!=asset['sha256']:return False
            exp=Path(prov['experiment_file']).resolve()
            if experiments.get(str(exp))!=prov['experiment_sha256'] or stage5.sha(exp)!=prov['experiment_sha256']:return False
            pack=read(exp)
            if pack['object_id']!=tid or pack['provenance']['source_video_sha256']!=doc['source']['sha256']:return False
            if pack['provenance']['mesh_sha256']!=asset['sha256']:return False
            rows={f['sample_index']:f for f in pack['frames']};scale=float(obj['fixed_depth_gauge_scale'])
            if not np.isfinite(scale) or scale<=0:return False
            if len(scene['objects'])>1 and scene['depth_gauge']['selected_scales'].get(tid)!=scale:return False
            for f in obj['frames']:
                upstream=rows.get(f['sample_index'])
                if upstream is None or f['timestamp_s']!=upstream['timestamp_s'] or f['status']!=upstream['status']:return False
                visible=upstream['status'] in stage5.ALLOWED_POSES
                if f['visible']!=visible:return False
                if not visible:
                    if f['camera_from_native_mesh'] is not None:return False
                    continue
                expected=np.eye(4);expected[:3,:3]=np.asarray(upstream['candidate_rotation'])*scale;expected[:3,3]=np.asarray(upstream['candidate_translation'])*scale
                if not np.allclose(f['camera_from_native_mesh'],expected,rtol=0,atol=1e-7):return False
        return bool(seen)
    except (KeyError,ValueError,OSError,TypeError):return False


def payload(path,title=None):
    doc=read(path);stage5.validate_interaction(doc)
    exports=[{'title':'Interaction program · JSON','path':relative(path),'filename':'interaction.json'},
             {'title':'Readable procedure · text','path':relative(path.parent/'procedure.txt'),'filename':'procedure.txt'},
             {'title':'Replay and condition checks · JSON','path':relative(path.parent/'replay.json'),'filename':'replay.json'}]
    exports += [{'title':a['track_id']+' · generated GLB','path':a['path'],'filename':a['track_id']+'.glb'} for a in doc['assets']]
    masks=[]
    if doc['source']['sha256']=='7474ebd9d06c55b63fc073f632e71e5125353ace2538fa7bcf632cfdc1c3d0f2':
        mask_path=ROOT/'data/jabra-local-v2/occlusion-v1/review/review-data.json'
        if mask_path.exists():
            pack=read(mask_path)
            for r in pack.get('frames',[]):
                # Review data is self-contained, source-time keyed evidence.
                file=r.get('hand_overlay') or r.get('hand_overlay_url') or r.get('hand')
                if file:
                    target=(mask_path.parent/file).resolve()
                    if target.is_file() and target.is_relative_to(ROOT):masks.append({'timestamp_s':r['timestamp_s'],'path':relative(target),'audit':r.get('hand_audit')})
    evidence=doc['evidence'];scene=scene_for(doc)
    identifier=path.parent.parent.name+'--'+path.parent.name
    parts=path.resolve().relative_to(ROOT).parts
    if len(parts)>5 and parts[:3]==('data','model-upgrades','runs') and parts[3].startswith('upgrade-'):
        identifier=parts[3]+'--'+path.parent.parent.name+'--'+path.parent.name
    return {'id':identifier,'case_id':doc['case_id'],
            'title':title or doc['case_id'].replace('-',' '),'description':doc['summary'],
            'video_path':doc['source']['path'],'poster_path':evidence[0]['frame_path'],'duration_s':doc['source']['duration_s'],
            'interaction_path':relative(path),'original_sha256':stage5.sha(path),'interaction':doc,'scene':scene,'mask_frames':masks,
            'exports':[e for e in exports if (ROOT/e['path']).is_file()],
            'labels':{'source_samples':len(evidence),'assets':len(doc['assets']),'steps':len(doc['steps']),
                      'conditional_poses':sum(m['status']=='supported_conditional_motion_candidate' for m in doc['motion'])}}


def registered_candidates(root=None):
    """Explicitly publish additional validated studies without scanning raw data."""
    root=Path(root or ROOT).resolve();registry=root/'workspace/registered-cases.json'
    if not registry.exists():return [],[]
    candidates=[];errors=[]
    try:
        document=read(registry)
        rows=document['cases']
        if not isinstance(rows,list) or len(rows)>50:raise ValueError('At most fifty additional published cases')
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return [],[{'path':'workspace/registered-cases.json','error':str(exc)}]
    for row in rows:
        try:
            path=row['interaction'];title=row['title']
            if (not isinstance(path,str) or not path.startswith('data/') or not path.endswith('/interaction.json')
                    or any(part.startswith('.') for part in Path(path).parts)):
                raise ValueError('Registered program must be a non-hidden data/ interaction.json')
            target=(root/path).resolve()
            if not target.is_relative_to(root/'data') or not target.is_file():
                raise ValueError('Registered program is missing or outside data/')
            if not isinstance(title,str) or not 1<=len(title.strip())<=80:
                raise ValueError('Registered title must contain one to eighty characters')
            candidates.append((target,title.strip()))
        except (ValueError,KeyError,TypeError) as exc:
            errors.append({'path':str(row.get('interaction','')) if isinstance(row,dict) else 'invalid_registry_entry','error':str(exc)})
    return candidates,errors


def lazy_payload(data, out):
    """Display derivatives only; the full published catalog remains unchanged."""
    out = Path(out)
    directory = out / 'scenes'
    directory.mkdir(exist_ok=True)
    rows = []
    for case in data['cases']:
        row = dict(case)
        scene = row.pop('scene', None)
        row['scene'] = None
        if scene is not None:
            wrapper = {'format': 'workspace-case-scene', 'version': 1,
                       'case_id': case['id'], 'interaction_sha256': case['original_sha256'],
                       'scene': scene}
            raw = (json.dumps(wrapper, separators=(',', ':'), allow_nan=False) + '\n').encode()
            digest = hashlib.sha256(raw).hexdigest()
            target = directory / (digest + '.json')
            if target.exists() and target.read_bytes() != raw:
                raise ValueError('Preserve immutable case-scene derivative')
            if not target.exists():
                target.write_bytes(raw)
            row['scene_ref'] = {'path': 'scenes/' + target.name, 'sha256': digest, 'bytes': len(raw)}
        rows.append(row)
    index = dict(data, cases=rows, lazy_scenes=True)
    encoded = json.dumps(index, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    atomic_text(out / 'cases-index.js','window.TWIN_CASES=' + encoded + ';\n')
    atomic_text(out / 'cases-index.json',encoded + '\n')
    return {'index_bytes': len(encoded.encode()), 'scene_files': sum('scene_ref' in row for row in rows),
            'full_catalog_preserved': True}


def build(out=None, *, publish_replays=True):
    out=Path(out or ROOT/'workspace');out.mkdir(parents=True,exist_ok=True)
    wallet=ROOT/'data/wallet-stage6-v1/procedure-fresh-v1/interaction.json'
    if not wallet.exists():wallet=ROOT/'data/wallet-stage6-v1/procedure-v1/interaction.json'
    preferred=[(ROOT/'data/jabra-stage6-v1/procedure-v1/interaction.json','Earbuds & charging case'),
               (wallet,'Wallet & coin')]
    fresh=sorted((ROOT/'data').glob('*/flow-procedure-*/interaction.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    registered,errors=registered_candidates()
    candidates=preferred+registered+[(p,'Pipeline · '+p.parent.parent.name) for p in fresh]
    cases=[];seen=set()
    for path,title in candidates:
        if not path.is_file():continue
        if path.resolve() in seen:continue
        seen.add(path.resolve())
        try:cases.append(payload(path,title))
        except (ValueError,KeyError,OSError,StopIteration) as exc:errors.append({'path':relative(path),'error':str(exc)})
    data={'version':'0.1','cases':cases,'build_errors':errors,'network_dependencies':False,'physical_accuracy_verified':False}
    encoded=json.dumps(data,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    atomic_text(out/'cases.js','window.TWIN_CASES='+encoded+';\n')
    atomic_text(out/'cases.json',encoded+'\n')
    replay_reports=[]
    if publish_replays and out.resolve()==(ROOT/'workspace').resolve():
        from scene_replay_publication import publish_estimated_replays
        replay_reports=publish_estimated_replays(cases,root=ROOT)
    atomic_text(out/'build-report.json',json.dumps({'cases':len(cases),'errors':errors,'payload_bytes':len(encoded),
        'estimated_scene_replays':replay_reports},indent=2)+'\n')
    lazy_payload(data, out)
    print(json.dumps({'output':str(out/'index.html'),'cases':len(cases),'errors':errors,'payload_bytes':len(encoded)},indent=2))
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path)
    p.add_argument('--no-replays',action='store_true',help='Private owner review: do not update any scene publication registry')
    a=p.parse_args();build(a.out,publish_replays=not a.no_replays)
