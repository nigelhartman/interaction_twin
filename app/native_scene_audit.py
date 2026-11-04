"""Immutable, source-bound full-native-triangle motion/render diagnostics.

No inference, repair, publication or mesh modification. A predeclared motion
holdout uses only neighboring training poses/joint coordinates for prediction;
its target SAM pixels are read only AFTER that prediction. Historical scenes
fitted on every frame are explicitly retrospective diagnostics, never heldout
validation. SAM masks and generated geometry remain fallible model outputs.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation,Slerp

import upgrade_bundles as U
import sam3d_common_scene as S
import sam3d_mechanisms as M
from native_appearance import load_native_glb,render_native_mesh
from native_audit_protocol import make_protocol

ROOT=Path(__file__).resolve().parent
FORMAT='interaction-twin-native-render-audit'
VERSION=1
MAX_FRAMES=24
MAX_COMPONENTS=12
COMPONENT_ID=re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z')
CODE=['native_scene_audit.py','native_audit_protocol.py','native_appearance.py','alignment_render.py','sam3d_common_scene.py','sam3d_mechanisms.py','part_ledger.py']


def require(value,message):
    if not value:raise ValueError(message)


def checked(reference,root):
    require(isinstance(reference,dict),'Exact ordinary-data artifact reference required')
    path=U._local(reference.get('path',''),root)
    require(S.ref(path,root)==reference,'Audit input bytes changed: '+path.name)
    return path


def _indices(rows,limit):
    if not rows:return []
    return [rows[i]for i in sorted(set(np.rint(np.linspace(0,len(rows)-1,min(limit,len(rows)))).astype(int).tolist()))]


def select_samples(scene,max_frames=12):
    require(type(max_frames)is int and 4<=max_frames<=MAX_FRAMES,'Audit frame budget must be between4 and24')
    rows=scene['components'][0]['poses'];ids=[r['source_sample_index']for r in rows]
    require(len(ids)==len(set(ids)) and 2<=len(ids)<=1200,'Unique bounded scene samples required')
    protocol=scene.get('evaluation_protocol',{})
    held=set(protocol.get('held_out_sample_indices',[])) if protocol.get('mode')=='predeclared_mask_holdout' else set()
    if held:
        development=_indices([i for i in ids if i not in held],max_frames//2)
        final=_indices([i for i in ids if i in held],max_frames-len(development))
    else:
        chosen=_indices(ids,max_frames);development=chosen[::2];final=chosen[1::2]
    return [(i,'development')for i in development]+[(i,'final_test')for i in final]


def _neighbors(rows,index,held,max_gap):
    ordered=sorted(rows,key=lambda r:r['source_sample_index'])
    left=[r for r in ordered if r['source_sample_index']<index and r['source_sample_index']not in held]
    right=[r for r in ordered if r['source_sample_index']>index and r['source_sample_index']not in held]
    if not left or not right:return None
    a,b=left[-1],right[0]
    if a['status']!='estimated' or b['status']!='estimated' or not 0<b['timestamp_s']-a['timestamp_s']<=max_gap:return None
    if any(r['source_sample_index']not in held and r['status']!='estimated' for r in ordered if a['source_sample_index']<r['source_sample_index']<b['source_sample_index']):return None
    return a,b


def interpolate_similarity(a,b,fraction):
    ra,ta,sa=M.rotation_pose(a);rb,tb,sb=M.rotation_pose(b)
    require(np.isclose(sa,sb,rtol=1e-6),'Heldout prediction cannot change native scale')
    result=np.eye(4);result[:3,:3]=sa*Slerp([0.,1.],Rotation.from_matrix([ra,rb]))(fraction).as_matrix()
    result[:3,3]=ta*(1-fraction)+tb*fraction
    return result


def predict_poses(scene,sample_index):
    """Only training transforms/scalars; never source RGB/masks or heldout pose."""
    components={c['id']:c for c in scene['components']};mechanics=scene.get('mechanics',{})
    joints={j['child_id']:j for j in mechanics.get('joints',[])}
    unresolved={j['child_id']for j in mechanics.get('unresolved_children',[])}
    held=set(scene.get('evaluation_protocol',{}).get('held_out_sample_indices',[]))
    max_gap=min(scene.get('timing',{}).get('max_interpolation_gap_s',.8),1.)
    rows=components[next(iter(components))]['poses'];target=next(r for r in rows if r['source_sample_index']==sample_index);timestamp=target['timestamp_s']
    cache={};visiting=set()
    def pose(identifier):
        if identifier in cache:return cache[identifier]
        require(identifier not in visiting,'Cyclic audit mechanism graph')
        visiting.add(identifier);component=components[identifier]
        result={'matrix':None,'method':'unsupported','neighbors':[],'reason':'source_estimate_unavailable'}
        if identifier in unresolved:result['reason']='persistent_joint_unresolved'
        elif sample_index not in held:
            row=next(r for r in component['poses']if r['source_sample_index']==sample_index)
            if row['status']=='estimated' and row['camera_from_native_mesh']is not None:
                result={'matrix':np.asarray(row['camera_from_native_mesh'],float),'method':'saved_postfit_estimate','neighbors':[sample_index],'reason':None}
        elif identifier in joints:
            joint=joints[identifier];parent=pose(joint['parent_id']);pair=_neighbors(joint['samples'],sample_index,held,max_gap)
            if parent['matrix']is not None and pair and all(r.get('q_rad')is not None for r in pair):
                a,b=pair;f=(timestamp-a['timestamp_s'])/(b['timestamp_s']-a['timestamp_s']);q=a['q_rad']*(1-f)+b['q_rad']*f
                result={'matrix':M.forward(parent['matrix'],joint['parent_from_joint'],joint['axis'],q,joint['joint_from_child_zero']),
                        'method':'heldout_training_neighbor_joint_FK','neighbors':[a['source_sample_index'],b['source_sample_index']],
                        'q_rad':float(q),'reason':None}
            else:result['reason']='no_bounded_training_parent_and_joint_neighbors'
        else:
            pair=_neighbors(component['poses'],sample_index,held,max_gap)
            if pair:
                a,b=pair;f=(timestamp-a['timestamp_s'])/(b['timestamp_s']-a['timestamp_s'])
                if a.get('camera_from_native_mesh')is not None and b.get('camera_from_native_mesh')is not None:
                    result={'matrix':interpolate_similarity(a['camera_from_native_mesh'],b['camera_from_native_mesh'],f),
                            'method':'heldout_training_neighbor_LERP_SLERP','neighbors':[a['source_sample_index'],b['source_sample_index']],'reason':None}
            else:result['reason']='no_bounded_training_pose_neighbors'
        visiting.remove(identifier);cache[identifier]=result;return result
    for identifier in components:pose(identifier)
    return timestamp,cache


def mask_metrics(observed,predicted,*,other_observed=None,amodal=None):
    observed=np.asarray(observed,bool);predicted=np.asarray(predicted,bool)
    require(observed.shape==predicted.shape,'Audit masks must have identical raster coordinates')
    other=np.zeros_like(observed)if other_observed is None else np.asarray(other_observed,bool)
    amodal=predicted if amodal is None else np.asarray(amodal,bool)
    require(other.shape==observed.shape and amodal.shape==observed.shape,'Occlusion masks differ')
    intersection=int(np.count_nonzero(observed&predicted));union=int(np.count_nonzero(observed|predicted))
    area=int(np.count_nonzero(observed));rendered=int(np.count_nonzero(predicted))
    return {'mask_iou':intersection/union if union else None,'visible_source_coverage':intersection/area if area else None,
        'render_precision':intersection/rendered if rendered else None,'source_mask_pixels':area,'render_mask_pixels':rendered,
        'intersection_pixels':intersection,'union_pixels':union,
        'known_other_source_overlap_pixels':int(np.count_nonzero(predicted&other)),
        'unexplained_render_pixels':int(np.count_nonzero(predicted&~observed&~other)),
        'estimated_self_occluded_render_pixels':int(np.count_nonzero(amodal&~predicted)),
        'unmodeled_hand_occlusion_possible':True}


def _image(path):
    with Image.open(path)as image:
        require(0<image.width*image.height<=16_000_000,'Bounded source image required')
        return np.asarray(image.convert('RGB'))


def _mask(path,shape):
    with Image.open(path)as image:
        require(image.size==(shape[1],shape[0]),'Source mask and RGB dimensions differ')
        raw=np.asarray(image)
    require(raw.ndim==2 and set(np.unique(raw).tolist())<={0,1,255},'Original binary SAM mask required')
    return raw>0


def _save(path,array):
    require(not path.exists(),'Audit output already exists')
    Image.fromarray(array).save(path)


def comparison_panel(source,observed,rendered,visible,component,timestamp,role):
    h,w=observed.shape
    mask_rgb=np.full((h,w,3),242,np.uint8);mask_rgb[observed]=[55,102,237]
    native=np.full((h,w,3),242,np.uint8);native[visible]=rendered[visible]
    difference=np.full((h,w,3),242,np.uint8)
    difference[observed&visible]=[70,153,117];difference[observed&~visible]=[55,102,237];difference[visible&~observed]=[220,101,79]
    canvas=Image.new('RGB',(w*4,h+56),'#f7f8fb');draw=ImageDraw.Draw(canvas)
    for i,(label,rgb)in enumerate(zip(['Source RGB','SAM visible mask','Full native triangles','Overlap / disagreement'],[source,mask_rgb,native,difference])):
        canvas.paste(Image.fromarray(rgb),(i*w,34));draw.text((i*w+7,7),label,fill='#172033')
    draw.text((7,h+39),f'{component} | {timestamp:.3f}s | {role} | blue: source-only; coral: render-only; green: overlap',fill='#626b7b')
    return np.asarray(canvas)


def _checked_scene(scene_path,root):
    path=U._local(scene_path,root);scene=U.read(path)
    require(scene.get('format')==S.FORMAT and scene.get('physical_accuracy_verified')is False and scene.get('accepted_motion_poses')==0,'Automatic estimated native scene required')
    require(1<=len(scene.get('components',[]))<=MAX_COMPONENTS,'Bounded native component set required')
    program_path=checked(scene['original_program'],root);program=U.read(program_path)
    require(program['source']['sha256']==scene['source']['sha256'] and scene['source']['time_domain']==U.TIME_DOMAIN,'Audit scene/program source differs')
    evidence={e['source_index']:e for e in program['evidence']}
    require(2<=len(evidence)<=1200,'Bounded original source evidence required')
    ids=[r['source_sample_index']for r in scene['components'][0]['poses']]
    for component in scene['components']:
        require(COMPONENT_ID.fullmatch(str(component.get('id',''))),'Ordinary bounded component IDs required')
        checked(component['asset'],root)
        require(component['id']in {t['id']for t in program['tracklets']},'Scene component absent from evidence program')
        require([r['source_sample_index']for r in component['poses']]==ids,'Scene components use different source samples')
        for row in component['poses']:
            require(row['source_sample_index']in evidence and abs(row['timestamp_s']-evidence[row['source_sample_index']]['timestamp_s'])<1e-9,'Scene estimate uses another source timestamp')
    protocol=scene.get('evaluation_protocol')
    if protocol:
        expected=make_protocol(scene['source']['sha256'],ids,protocol['held_out_sample_indices'],protected_sample_indices=protocol['protected_native_anchor_indices'])
        require(protocol==expected,'Evaluation protocol differs from its declared source samples')
        held=set(protocol['held_out_sample_indices'])
        for component in scene['components']:
            require(all(r['status']=='unknown' and r['camera_from_native_mesh']is None for r in component['poses']if r['source_sample_index']in held),'A declared heldout pose was fitted or filled')
        for joint in scene.get('mechanics',{}).get('joints',[]):
            require(all(r['status']=='unknown' and r['q_rad']is None for r in joint['samples']if r['source_sample_index']in held),'A declared heldout joint was fitted')
    return path,scene,program,evidence


def validate_manifest(path,*,root=ROOT):
    root=Path(root).resolve();path=U._local(path,root);doc=U.read(path)
    require(doc.get('format')==FORMAT and doc.get('version')==VERSION and doc.get('no_independent_physical_validation')is True and doc.get('new_api_calls')==doc.get('new_gpu_calls')==0,'Unknown native audit contract')
    program=U.read(checked(doc['interaction_ref'],root));scene=U.read(checked(doc['scene_ref'],root))
    require(doc['source']['sha256']==program['source']['sha256']==scene['source']['sha256'],'Audit source/program differs')
    require(doc['interaction_ref']==scene['original_program'],'Audit scene uses a different evidence program')
    for item in [*doc['inputs'],*doc['outputs']]:checked(item,root)
    inputs={r['path']:r for r in doc['inputs']};outputs={r['path']:r for r in doc['outputs']}
    require(len(inputs)==len(doc['inputs'])and len(outputs)==len(doc['outputs']),'Repeated audit artifact paths')
    for item in doc.get('implementation_snapshots',[]):
        checked(item,root);require(outputs.get(item['path'])==item,'Implementation snapshot absent from audit outputs')
        require(any(item['sha256']==code['sha256'] and Path(item['path']).name==code['path']for code in doc['implementation']),'Audit source snapshot differs')
    require(len(doc['samples'])<=MAX_FRAMES*MAX_COMPONENTS,'Audit sample count exceeds bounds')
    evidence={e['source_index']:e for e in program['evidence']};parts={c['id']for c in scene['components']};seen=set()
    for sample in doc['samples']:
        identifier=sample['component_id'];index=sample['source_index'];key=sample['sample_id']
        require(identifier in parts and index in evidence and key==f'{identifier}:{index}' and key not in seen,'Unknown or repeated audit sample')
        seen.add(key);frame=evidence[index]
        require(sample['role']in {'development','final_test'} and sample['timestamp_s']==frame['timestamp_s'],'Audit role or source timestamp differs')
        rgb=sample['source_rgb_ref'];require(rgb['path']==frame['frame_path']and rgb['sha256']==frame['frame_sha256']and inputs.get(rgb['path'])==rgb,'Audit RGB is not the exact evidence frame')
        mask=sample.get('source_mask_ref')
        if mask is not None:
            observation=next((o for o in frame['observations']if o['track_id']==identifier),None)
            require(observation and mask['path']==observation['mask_path']and mask['sha256']==observation['mask_sha256']and inputs.get(mask['path'])==mask,'Audit SAM mask belongs to another source/part')
        for name in ['render_mask_ref','comparison_panel_ref']:
            value=sample[name];require(outputs.get(value['path'])==value and (root/value['path']).is_relative_to(path.parent),'Audit panel is not a frozen output of this packet')
        require(sample['status']in {'diagnostic_available','needs_visual_review','insufficient_pose_prediction','insufficient_source_observation'},'Unknown native audit status')
        require(sample.get('original_native_mesh_unchanged')is True,'Audit cannot claim to modify native geometry')
        for metric in ['mask_iou','visible_source_coverage','render_precision']:
            value=sample['metrics'][metric];require(value is None or type(value)in (int,float)and np.isfinite(value)and 0<=value<=1,'Invalid audit ratio')
    return doc


def build_audit(scene_path,output,*,root=ROOT,max_frames=12,max_dimension=384):
    root=Path(root).resolve();start=time.monotonic();path,scene,program,evidence=_checked_scene(scene_path,root)
    require(type(max_dimension)is int and 128<=max_dimension<=768,'Audit raster must be between128 and768 pixels')
    out=Path(output);out=out if out.is_absolute()else root/out
    require(out.is_relative_to(root/'data') and out.resolve()==out and not any(p.is_symlink()for p in [out,*out.parents]),'Ordinary fresh data audit output required')
    selected=select_samples(scene,max_frames);code=[S.source_ref(root/name,root)for name in CODE]
    config={'max_frames':max_frames,'max_dimension':max_dimension,'selected_samples':selected,'triangle_policy':'all_original_native_faces_no_decimation_or_convex_hull'}
    signature=U.digest(U.encoded({'scene':S.ref(path,root),'config':config,'implementation':code}))
    if out.exists():
        doc=validate_manifest(out/'audit.json',root=root);require(doc['signature']==signature,'Audit configuration or code changed; choose a new output')
        return {'manifest':S.ref(out/'audit.json',root),'reused_verified_cache':True,'summary':doc['summary'],'development_summary':doc.get('development_summary')}
    meshes={c['id']:load_native_glb(checked(c['asset'],root))for c in scene['components']}
    original_h,original_w=int(scene['source']['height']),int(scene['source']['width']);factor=min(1.,max_dimension/max(original_h,original_w));w=max(1,round(original_w*factor));h=max(1,round(original_h*factor))
    k=np.asarray(scene['camera']['K'],float).copy();k[0]*=w/original_w;k[1]*=h/original_h
    # Preserve the renderer's integer pixel-centre convention under downsampling.
    k[0,2]+=(w/original_w-1)/2;k[1,2]+=(h/original_h-1)/2
    out.mkdir(parents=True);inputs={r['path']:r for r in [S.ref(path,root),scene['original_program'],*[c['asset']for c in scene['components']]]};outputs=[];samples=[]
    snapshots=[];(out/'implementation').mkdir()
    for implementation in code:
        snapshot=out/'implementation'/implementation['path'];snapshot.write_bytes((root/implementation['path']).read_bytes());snapshot_ref=S.ref(snapshot,root)
        require(snapshot_ref['sha256']==implementation['sha256'],'Builder source changed while freezing audit implementation')
        snapshots.append(snapshot_ref);outputs.append(snapshot_ref)
    if scene.get('part_ledger'):checked(scene['part_ledger'],root);inputs[scene['part_ledger']['path']]=scene['part_ledger']
    held=set(scene.get('evaluation_protocol',{}).get('held_out_sample_indices',[]))
    for index,role in sorted(selected):
        timestamp,poses=predict_poses(scene,index) # Deliberately before reading any target image/mask pixels.
        source=evidence[index];rgb_ref={'path':source['frame_path'],'sha256':source['frame_sha256'],'bytes':U._local(source['frame_path'],root).stat().st_size}
        rgb=_image(checked(rgb_ref,root));require(rgb.shape[:2]==(original_h,original_w),'Source raster differs from scene camera')
        small=cv2.resize(rgb,(w,h),interpolation=cv2.INTER_AREA);inputs[rgb_ref['path']]=rgb_ref
        observations={o['track_id']:o for o in source.get('observations',[])};source_masks={};source_refs={};usable={}
        for component in scene['components']:
            identifier=component['id'];observation=observations.get(identifier)
            if observation:
                mask_ref={'path':observation['mask_path'],'sha256':observation['mask_sha256'],'bytes':U._local(observation['mask_path'],root).stat().st_size}
                raw=_mask(checked(mask_ref,root),(original_h,original_w));source_masks[identifier]=cv2.resize(raw.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0;source_refs[identifier]=mask_ref;inputs[mask_ref['path']]=mask_ref
                usable[identifier]=observation.get('gate_status')=='candidate_geometric_gate_only' and observation.get('audit_status')not in {'reject','rejected','withheld'} and np.count_nonzero(raw)>=32
            else:source_masks[identifier]=np.zeros((h,w),bool);source_refs[identifier]=None;usable[identifier]=False
        renders={}
        for identifier,prediction in poses.items():
            if prediction['matrix']is not None:
                rotation,translation,scale=M.rotation_pose(prediction['matrix']);renders[identifier]=render_native_mesh(meshes[identifier],k,rotation,translation,w,h,scale=scale)
        depth=np.full((h,w),np.inf);owner=np.full((h,w),-1,int);identifiers=list(poses)
        for n,identifier in enumerate(identifiers):
            if identifier not in renders:continue
            rendered=renders[identifier];closer=rendered['mask']&(rendered['depth']<depth);depth[closer]=rendered['depth'][closer];owner[closer]=n
        for n,identifier in enumerate(identifiers):
            prediction=poses[identifier];observed=source_masks[identifier];rendered=renders.get(identifier);visible=owner==n
            native_rgb=rendered['rgb']if rendered else np.zeros((h,w,3),np.uint8)
            other=np.zeros((h,w),bool)
            for cid,mask in source_masks.items():
                if cid!=identifier:other|=mask
            metrics=mask_metrics(observed,visible,other_observed=other,amodal=rendered['mask']if rendered else visible)
            reason=[]
            if rendered is None:status='insufficient_pose_prediction';reason.append(prediction['reason'])
            elif not usable[identifier]:status='insufficient_source_observation';reason.append('SAM observation missing, too small or withheld; no hidden mask completion')
            else:status='needs_visual_review'if (metrics['mask_iou']or 0)<.45 else 'diagnostic_available'
            sample_id=f'{identifier}:{index}';stem=f'{index:06d}-{identifier}';mask_path=out/(stem+'-render-mask.png');panel_path=out/(stem+'-comparison.png')
            _save(mask_path,visible.astype(np.uint8)*255);_save(panel_path,comparison_panel(small,observed,native_rgb,visible,identifier,timestamp,role))
            mask_ref,panel_ref=S.ref(mask_path,root),S.ref(panel_path,root);outputs.extend([mask_ref,panel_ref])
            samples.append({'sample_id':sample_id,'source_index':index,'timestamp_s':timestamp,'component_id':identifier,'role':role,
                'source_rgb_ref':rgb_ref,'source_mask_ref':source_refs[identifier],'render_mask_ref':mask_ref,'comparison_panel_ref':panel_ref,
                'metrics':metrics,'status':status,'limitation_reasons':reason+['SAM masks and estimated occlusion are not independently verified; hand occlusion is unmodeled'],
                'prediction':{key:value for key,value in prediction.items()if key!='matrix'},'target_mask_withheld_from_motion_fit':index in held,
                'original_native_mesh_unchanged':True,'rendered_triangle_count':len(meshes[identifier].faces)})
    metrics=[s['metrics']['mask_iou']for s in samples if s['status']in {'needs_visual_review','diagnostic_available'} and s['metrics']['mask_iou']is not None]
    summary={'source_frames':len(selected),'component_samples':len(samples),'components':len(meshes),
        'usable_comparisons':len(metrics),'needs_visual_review':sum(s['status']=='needs_visual_review'for s in samples),
        'insufficient_samples':sum(s['status'].startswith('insufficient')for s in samples),'median_native_triangle_mask_iou':float(np.median(metrics))if metrics else None,
        'physical_accuracy_verified':False,'simulation_ready':False,'scope':'all audit samples; never use this aggregate for repair selection'}
    development=[s for s in samples if s['role']=='development'];development_usable=[s for s in development if s['status']in {'needs_visual_review','diagnostic_available'}]
    development_metrics=[s['metrics']['mask_iou']for s in development_usable if s['metrics']['mask_iou']is not None]
    development_summary={'component_samples':len(development),'usable_comparisons':len(development_metrics),
        'needs_visual_review':sum(s['status']=='needs_visual_review'for s in development),
        'median_native_triangle_mask_iou':float(np.median(development_metrics))if development_metrics else None,
        'scope':'development samples only; final_test evidence and scores excluded from repair selection'}
    program_assets={a['track_id']:a for a in program.get('assets',[])}
    inventory={'scene_component_ids':[c['id']for c in scene['components']],'compiled_asset_component_ids':sorted(program_assets),
        'scene_components_without_compiled_assets':[c['id']for c in scene['components']if c['id']not in program_assets],
        'native_hash_mismatches':[c['id']for c in scene['components']if c['id']in program_assets and c['asset']['sha256']!=program_assets[c['id']]['sha256']]}
    inventory['consistent']=not inventory['scene_components_without_compiled_assets']and not inventory['native_hash_mismatches']
    protocol=scene.get('evaluation_protocol')or {'mode':'retrospective_diagnostic','held_out_sample_indices':[],
        'scope':'Saved poses were fitted on the source masks; final_test is reviewer isolation only, not motion holdout validation'}
    doc={'format':FORMAT,'version':VERSION,'signature':signature,'source':program['source'],'interaction_ref':scene['original_program'],'scene_ref':S.ref(path,root),
        'evaluation_protocol':protocol,'no_independent_physical_validation':True,'config':config,'implementation':code,'implementation_snapshots':snapshots,'component_inventory':inventory,
        'inputs':[inputs[key]for key in sorted(inputs)],'outputs':outputs,'samples':samples,'summary':summary,'development_summary':development_summary,
        'new_api_calls':0,'new_gpu_calls':0,'published':False,'original_mesh_bytes_modified':False,'wall_seconds':time.monotonic()-start,
        'limitations':['Full original triangles and native base colours are projected with estimated camera/poses; no convex-hull score is used as validation.',
            'Inter-part z-buffer occlusion follows the estimated geometry; hands and missing parts do not become automatic occlusion excuses.',
            'Development panels may guide repairs; final_test panels must never be used for repair selection.',
            'A mask holdout tests temporal motion prediction conditional on generated geometry and topology; it does not validate metric scale, identity, contact or physics.']}
    S.write(out/'audit.json',doc);validate_manifest(out/'audit.json',root=root)
    return {'manifest':S.ref(out/'audit.json',root),'reused_verified_cache':False,'summary':summary,'development_summary':development_summary,'wall_seconds':doc['wall_seconds']}
