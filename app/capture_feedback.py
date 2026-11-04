"""Read-only source-pixel capture feedback; no inference or pose selection.

Metrics are inspection proxies, not diagnoses of a physical failure. Saved
model judgments retain their citations and fallibility. Feedback never changes
reference selection, training/evaluation splits, masks, assets or poses.
"""
from __future__ import annotations
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image

import revisions

POLICY = {'version':'capture-feedback-v1','feedback_bin_seconds':3.0,
    'minimum_mask_pixels':160,'minimum_bbox_side_px':16,
    'minimum_detail_core_pixels':64,'detail_core_erosion_pixels':2,
    'absolute_low_laplacian_variance':25.0,'relative_low_detail_fraction_of_component_p75':.25,
    'maximum_adjacent_gap_cadences':1.6,'area_change_ratio':2.5,
    'minimum_translation_aligned_mask_iou':.25,'other_mask_overlap_fraction':.15,
    'image_border_margin_px':1,'used_for_reconstruction_selection_or_fitting':False}


def definition(title, semantics, action, priority, capture=False):
    return {'title':title,'semantics':semantics,'action':action,'priority':priority,
        'requires':{'additional_capture':capture,'api':False,'gpu':False},'cause_verified':False}


DEFINITIONS = {
 'mask_unusable':definition('Target mask is missing or empty','measured_mask_availability_not_physical_absence',
    'Inspect these frames. Record the target continuously visible through the movement if that portion is needed.',100,True),
 'saved_audit_rejected':definition('Saved model audit rejects this mask','saved_model_judgment_not_measured_accuracy',
    'Review the cited source/mask pair before using this sample; do not propagate its pose through the interval.',95),
 'small_visible_target':definition('Few source pixels describe this part','measured_visible_mask_size_not_full_object_size',
    'Record a closer view with the part surface in focus; keep the complete visible part inside the frame.',85,True),
 'saved_model_hand_occlusion':definition('Saved audit describes hand/finger occlusion','saved_model_judgment_not_new_occlusion_measurement',
    'Review the cited audit and frame. Capture an angle that keeps the target surface visible around the hand.',80,True),
 'target_at_image_border':definition('Target mask touches the source image border','measured_mask_border_contact_not_proven_object_clipping',
    'Keep the component farther from the image border during a repeat capture.',75,True),
 'low_detail_proxy':definition('Low high-frequency image detail','measured_image_detail_proxy_not_confirmed_blur',
    'Check focus and motion blur at these seconds. Move slowly and keep stable visible texture; a smooth surface can also produce this signal.',70,True),
 'saved_model_nonrigidity':definition('Saved description warns of deformation','saved_model_description_not_measured_deformation',
    'Treat this component as potentially deformable. Use a short near-rigid segment or a deformable representation.',65),
 'mask_visibility_change':definition('Mask shape or visible area changes sharply','measured_2d_mask_change_not_proven_tracking_error_or_nonrigidity',
    'Inspect the neighboring source frames for real rotation, changing visibility, deformation or a mask mistake before trusting continuity.',60),
 'overlapping_model_masks':definition('Two model masks overlap','measured_mask_overlap_not_depth_order_or_occlusion',
    'Inspect the overlapping masks. Part/whole labels and segmentation spill can cause overlap; do not infer contact or depth order.',55),
 'saved_audit_uncertain':definition('Saved mask audit is uncertain','saved_model_judgment_not_measured_accuracy',
    'Review the cited raw source and mask; the cached model audit did not establish a plausible visible mask.',50),
 'conditional_pose_gap':definition('Baseline has no supported non-reference pose here','recorded_pipeline_status_not_measured_failure_cause',
    'Inspect the original motion failure reasons. Extra image motion alone does not establish useful 3D viewpoints.',35),
}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def data_path(root, value):
    if not isinstance(value,str): raise ValueError('Expected project data path')
    value=value.removeprefix('/files/')
    if not value.startswith('data/') or any(c in value for c in ['\\','\0','?','#']):
        raise ValueError('Only project data artifacts may be inspected')
    path=(Path(root)/value).resolve()
    if not path.is_relative_to(Path(root).resolve()/'data'): raise ValueError('Data path escaped project')
    return path


def checked(root, value, expected=None):
    path=data_path(root,value)
    if not path.is_file() or path.stat().st_size>128*1024*1024: raise ValueError('Input is missing or too large')
    if expected and sha(path)!=expected: raise ValueError('Source artifact changed: '+value)
    return path


def published(root, interaction=None, case_id=None):
    root=Path(root).resolve()
    if interaction is not None:
        path=data_path(root,interaction)
        rows=read(root/'workspace/cases.json')['cases']
        matches=[r for r in rows if r.get('interaction_path')==path.relative_to(root).as_posix()]
        if len(matches)!=1: raise ValueError('Interaction must be a unique registered workspace publication')
        if case_id and matches[0]['id']!=case_id: raise ValueError('Case/interaction identity differs')
        case_id=matches[0]['id']
    path,program,digest=revisions.published_source(root,case_id)
    if program.get('format')!='interaction_twin' and not all(k in program for k in ['source','evidence','tracklets','motion']):
        raise ValueError('Unsupported interaction program')
    return path,program,digest,case_id


def mask_metrics(gray, mask):
    """Original-pixel metrics. No resizing, segmentation, geometry or poses."""
    if gray.ndim!=2 or mask.shape!=gray.shape: raise ValueError('Image/mask dimensions differ')
    yy,xx=np.nonzero(mask)
    row={'mask_area_pixels':int(len(xx)),'frame_area_pixels':int(mask.size),
         'mask_fraction':float(len(xx)/mask.size),'bbox_xyxy':None,'minimum_bbox_side_px':0,
         'centroid_xy':None,'detail_core_pixels':0,'laplacian_variance':None,'gray_std':None,
         'border_contact':False}
    if not len(xx): return row
    row.update(bbox_xyxy=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
        minimum_bbox_side_px=int(min(xx.max()-xx.min()+1,yy.max()-yy.min()+1)),
        centroid_xy=[float(xx.mean()),float(yy.mean())],
        border_contact=bool(xx.min()<=POLICY['image_border_margin_px'] or yy.min()<=POLICY['image_border_margin_px']
            or xx.max()>=mask.shape[1]-1-POLICY['image_border_margin_px'] or yy.max()>=mask.shape[0]-1-POLICY['image_border_margin_px']))
    core=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8),iterations=POLICY['detail_core_erosion_pixels'])>0
    row['detail_core_pixels']=int(core.sum())
    if core.sum()>=POLICY['minimum_detail_core_pixels']:
        detail=cv2.Laplacian(gray.astype(np.float32),cv2.CV_32F,ksize=3)
        row['laplacian_variance']=float(np.var(detail[core]))
        row['gray_std']=float(np.std(gray[core]))
    return row


def translated_mask_iou(first,second):
    if first.shape!=second.shape: raise ValueError('Temporal masks differ in size')
    ay,ax=np.nonzero(first);by,bx=np.nonzero(second)
    if not len(ax) or not len(bx): return None
    shift=np.rint([ax.mean()-bx.mean(),ay.mean()-by.mean()]).astype(int)
    aligned=cv2.warpAffine(second.astype(np.uint8),np.array([[1.,0.,shift[0]],[0.,1.,shift[1]]]),
        (second.shape[1],second.shape[0]),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0
    union=(first|aligned).sum()
    return float((first&aligned).sum()/union) if union else None


def model_hand_occlusion(reason):
    text=str(reason).lower()
    if re.search(r'\b(no|without|not)\b.{0,20}\b(occlu|hand|finger)',text): return False
    return bool(re.search(r'(hand|finger).{0,45}occlu|occlu.{0,45}(hand|finger)',text))


def baseline_summary(program, track):
    rows=[r for r in program['motion'] if r['track_id']==track]
    supported=[r for r in rows if r['status']=='supported_conditional_motion_candidate']
    references=[r for r in rows if r['status']=='reference_candidate_unverified_geometry_and_pose']
    return {'attempted':bool(rows),'sample_count':len(rows),'status_counts':dict(Counter(r['status'] for r in rows)),
        'supported_nonreference_samples':len(supported),'reference_samples':len(references),
        'supported_source_indices':[r['source_sample_index'] for r in supported],
        'reference_source_indices':[r['source_sample_index'] for r in references],
        'physical_pose_accuracy_measured':False,'viewpoint_sufficiency_established':False}


def interval_row(component, samples, identifier, *, kind='feedback_display_bin', episode_id=None):
    codes=sorted({c for r in samples for c in r['diagnostic_codes']},key=lambda c:(-DEFINITIONS[c]['priority'],c))
    diagnostics=[]
    for code in codes:
        affected=[r for r in samples if code in r['diagnostic_codes']]
        row=copy.deepcopy(DEFINITIONS[code]);row['code']=code
        row['evidence_ids']=[r['evidence_id'] for r in affected]
        row['source_sample_ids']=[r['source_index'] for r in affected]
        row['measurement']={'affected_samples':len(affected),'interval_samples':len(samples),
            'examples':[{'source_index':r['source_index'],'timestamp_s':r['timestamp_s'],
                'mask_area_pixels':r['metrics']['mask_area_pixels'],'minimum_bbox_side_px':r['metrics']['minimum_bbox_side_px'],
                'laplacian_variance':r['metrics']['laplacian_variance'],
                'translation_aligned_iou':r['metrics'].get('translation_aligned_iou'),
                'area_change_ratio':r['metrics'].get('area_change_ratio'),
                'model_audit':r.get('model_audit')} for r in affected[:5]]}
        diagnostics.append(row)
    return {'id':identifier,'episode_id':episode_id,'component_id':component,'kind':kind,
        'start_s':samples[0]['timestamp_s'],'end_s':samples[-1]['timestamp_s'],
        'source_sample_ids':[r['source_index'] for r in samples],
        'source_evidence_ids':[r['evidence_id'] for r in samples],
        'diagnostics':diagnostics,'action_priority':diagnostics[0]['priority'] if diagnostics else 0,
        'cause_verified':False,'used_for_tracking_selection_or_fitting':False}


def build(root, *, interaction=None, case_id=None):
    root=Path(root).resolve();path,program,digest,case_id=published(root,interaction,case_id)
    inputs={path.relative_to(root).as_posix():digest}
    video=checked(root,program['source']['path'],program['source']['sha256']);inputs[program['source']['path']]=sha(video)
    audit={};audit_file=None
    references={r['kind']:r for r in program['inputs']}
    if 'model_mask_plausibility_audit' in references:
        entry=references['model_mask_plausibility_audit'];audit_file=checked(root,entry['path'],entry['sha256'])
        document=read(audit_file)
        tracks=references['actual_sam_masks'];checked(root,tracks['path'],tracks['sha256'])
        inputs.update({entry['path']:entry['sha256'],tracks['path']:tracks['sha256']})
        if document['source_sha256']!=program['source']['sha256'] or document['source_track_index_sha256']!=tracks['sha256'] or document.get('manual_corrections_used') is not False:
            raise ValueError('Saved audit source/provenance differs')
        for item in document['result']['observations']:
            key=(item['track_id'],item['sample_index'])
            if key in audit: raise ValueError('Duplicate saved mask-audit observation')
            audit[key]=dict(item,audit_path=entry['path'],audit_sha256=entry['sha256'],
                provider=document.get('provider'),model=document.get('model'),judgment_accuracy_measured=False)
    evidence=program['evidence'];indices=[r['source_index'] for r in evidence];times=[r['timestamp_s'] for r in evidence]
    if len(set(indices))!=len(indices) or any(not math.isfinite(t) for t in times) or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Source sample IDs/timestamps must be unique, finite and increasing')
    cadence=float(np.median(np.diff(times))) if len(times)>1 else None
    tracks={t['id']:t for t in program['tracklets']}
    samples={key:[] for key in tracks};previous={}
    baseline={(r['track_id'],r['source_sample_index']):r for r in program['motion']}
    for frame in evidence:
        image=checked(root,frame['frame_path'],frame['frame_sha256']);inputs[frame['frame_path']]=frame['frame_sha256']
        with Image.open(image) as decoded:
            if decoded.width*decoded.height>16_777_216: raise ValueError('Source image pixel budget exceeded')
            gray=np.asarray(decoded.convert('L'))
        masks={};observations={}
        for observation in frame['observations']:
            key=observation['track_id']
            if key in observations or key not in tracks: raise ValueError('Duplicate or unknown source component')
            mask_path=checked(root,observation['mask_path'],observation['mask_sha256'])
            with Image.open(mask_path) as decoded: mask=np.asarray(decoded.convert('L'))>127
            if mask.shape!=gray.shape: raise ValueError('Raw source mask/image dimensions differ')
            masks[key]=mask;observations[key]=observation;inputs[observation['mask_path']]=observation['mask_sha256']
        for key,track in tracks.items():
            scope=track.get('source_sample_range',[min(indices),max(indices)])
            if not scope[0]<=frame['source_index']<=scope[1]: continue
            observation=observations.get(key);mask=masks.get(key,np.zeros(gray.shape,bool))
            metrics=mask_metrics(gray,mask);codes=[];saved=audit.get((key,frame['source_index']))
            if not metrics['mask_area_pixels']: codes.append('mask_unusable')
            elif metrics['mask_area_pixels']<POLICY['minimum_mask_pixels'] or metrics['minimum_bbox_side_px']<POLICY['minimum_bbox_side_px']:
                codes.append('small_visible_target')
            if metrics['border_contact']: codes.append('target_at_image_border')
            overlaps=[]
            for other,other_mask in masks.items():
                if other==key or not metrics['mask_area_pixels']: continue
                intersection=int((mask&other_mask).sum());fraction=intersection/metrics['mask_area_pixels']
                if intersection: overlaps.append({'component_id':other,'intersection_pixels':intersection,'target_fraction':fraction})
                if fraction>=POLICY['other_mask_overlap_fraction'] and 'overlapping_model_masks' not in codes: codes.append('overlapping_model_masks')
            metrics['other_model_mask_overlap']=overlaps
            old=previous.get(key)
            if old and cadence and frame['timestamp_s']-old[0]<=POLICY['maximum_adjacent_gap_cadences']*cadence and mask.any() and old[1].any():
                ratio=max(mask.sum()/old[1].sum(),old[1].sum()/mask.sum())
                iou=translated_mask_iou(old[1],mask)
                metrics.update(area_change_ratio=float(ratio),translation_aligned_iou=iou,previous_source_index=old[2])
                if ratio>POLICY['area_change_ratio'] or iou<POLICY['minimum_translation_aligned_mask_iou']: codes.append('mask_visibility_change')
            previous[key]=(frame['timestamp_s'],mask,frame['source_index'])
            if saved:
                if saved['verdict']=='reject_wrong_or_lost_mask': codes.append('saved_audit_rejected')
                elif saved['verdict']=='uncertain': codes.append('saved_audit_uncertain')
                if model_hand_occlusion(saved.get('reason','')): codes.append('saved_model_hand_occlusion')
            limitation_quotes=[text for text in track.get('limitations',[]) if re.search(r'\b(deformable|not rigid|non.rigid|bending|flexible)\b',text.lower())]
            if limitation_quotes: codes.append('saved_model_nonrigidity')
            status=baseline.get((key,frame['source_index']),{}).get('status')
            if status and status not in {'supported_conditional_motion_candidate','reference_candidate_unverified_geometry_and_pose'}: codes.append('conditional_pose_gap')
            samples[key].append({'evidence_id':frame['id'],'source_index':frame['source_index'],'timestamp_s':frame['timestamp_s'],
                'image_path':frame['frame_path'],'image_sha256':frame['frame_sha256'],
                'mask_path':observation['mask_path'] if observation else None,'mask_sha256':observation['mask_sha256'] if observation else None,
                'gate_status':observation.get('gate_status') if observation else 'no_saved_observation',
                'model_audit':saved,'model_description_quotes':limitation_quotes,
                'baseline_status':status,'metrics':metrics,'diagnostic_codes':codes})
    components=[]
    for key,rows in samples.items():
        values=[r['metrics']['laplacian_variance'] for r in rows if r['metrics']['laplacian_variance'] is not None]
        p75=float(np.percentile(values,75)) if values else None
        for row in rows:
            value=row['metrics']['laplacian_variance']
            if value is not None and (value<POLICY['absolute_low_laplacian_variance'] or value<POLICY['relative_low_detail_fraction_of_component_p75']*p75):
                row['diagnostic_codes'].append('low_detail_proxy')
        bins={}
        for row in rows: bins.setdefault(int(row['timestamp_s']//POLICY['feedback_bin_seconds']),[]).append(row)
        intervals=[interval_row(key,group,f'{key}-feedback-{bucket:04d}') for bucket,group in bins.items()]
        counts=Counter(c for row in rows for c in row['diagnostic_codes'])
        components.append({'id':key,'label':tracks[key]['label'],'identity_status':tracks[key].get('identity_status'),
            'summary':{'source_samples':len(rows),'diagnostic_sample_counts':dict(counts),'detail_variance_p75':p75,
                'baseline':baseline_summary(program,key),'capture_causes_verified':False},
            'samples':rows,'intervals':intervals})
    candidates=[(d['priority'],d['measurement']['affected_samples'],interval,d) for component in components
                for interval in component['intervals'] for d in interval['diagnostics']]
    candidates.sort(key=lambda x:(-x[0],-x[1],x[2]['start_s'],x[2]['component_id']))
    recommended=None
    if candidates:
        _,_,interval,d=candidates[0]
        recommended={k:copy.deepcopy(d[k]) for k in ['code','title','action','requires','semantics','cause_verified']}
        recommended.update(component_id=interval['component_id'],start_s=interval['start_s'],end_s=interval['end_s'],
            source_sample_ids=d['source_sample_ids'],evidence_ids=d['evidence_ids'])
    return {'format':'source-capture-feedback','version':1,'case_id':case_id,'program_case_id':program['case_id'],
        'interaction_path':path.relative_to(root).as_posix(),'interaction_sha256':digest,
        'source':program['source'],'time_domain':'source_video_presentation_seconds','policy':POLICY,
        'diagnostic_definitions':DEFINITIONS,'components':components,'episodes':[],
        'recommended_next_action':recommended,'input_sha256':inputs,
        'purpose':'read_only_capture_review_not_reference_or_interval_selection',
        'viewpoint_assessment':{'status':'not_established','reason':'2D displacement and mask change do not establish 3D viewpoint diversity or rigidity.',
            'action':'For accurate geometry, capture sharp overlapping views with calibrated intrinsics and a scale reference; keep component surfaces visible.'},
        'new_api_calls':0,'new_model_inference_calls':0,'mps_used':False,'network_access':False,
        'manual_geometry_or_poses_used':False,'physical_accuracy_verified':False,'simulation_ready':False,
        'limitations':['Laplacian variance measures image detail; smooth texture, exposure, focus and motion can all affect it. Blur is not diagnosed.',
            'A model mask is fallible. Empty masks do not establish absence; mask overlap does not establish depth order or physical occlusion.',
            'Saved model descriptions/audits are cited judgments, not new physical labels or measured accuracy.',
            'Fixed feedback display bins are unrelated to reconstruction episode selection; no training/evaluation choices are changed.']}


def save(root, report, output):
    root=Path(root).resolve();path=data_path(root,output)
    if not path.name.startswith('capture-feedback') or path.suffix!='.json' or path.exists():
        raise ValueError('Use a fresh capture-feedback*.json artifact')
    path.parent.mkdir(parents=True,exist_ok=True)
    snapshot=path.with_suffix('.implementation.py')
    with snapshot.open('xb') as stream: stream.write(Path(__file__).read_bytes())
    report=copy.deepcopy(report)
    report['implementation']={'path':snapshot.relative_to(root).as_posix(),'sha256':sha(snapshot)}
    with path.open('x') as stream: json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    return {'case_id':report['case_id'],'path':path.relative_to(root).as_posix(),'sha256':sha(path),
        'interaction_path':report['interaction_path'],'interaction_sha256':report['interaction_sha256'],
        'component_count':len(report['components']),'episode_count':len(report['episodes'])}


def attach_episodes(root, feedback, plans, *, scope_audits=()):
    """Create a derivative report using exact frozen engine intervals only."""
    root=Path(root).resolve();base=checked(root,feedback);report=copy.deepcopy(read(base));episodes=[];bindings=[]
    if report.get('format')!='source-capture-feedback': raise ValueError('Unsupported base feedback')
    _,_,digest,_=published(root,report['interaction_path'],report['case_id'])
    if digest!=report['interaction_sha256']: raise ValueError('Feedback publication changed')
    for relative_input,expected in report['input_sha256'].items(): checked(root,relative_input,expected)
    scope_by_plan={};scope_bindings=[]
    for relative_audit in scope_audits:
        audit_path=checked(root,relative_audit);audit=read(audit_path)
        if (audit.get('format')!='short-interval-mask-scope-audit' or audit.get('manual_corrections') is not False
                or audit.get('interaction_sha256')!=report['interaction_sha256']
                or audit.get('source_video_sha256')!=report['source']['sha256']):
            raise ValueError('Supplemental scope audit has different source/provenance')
        request_path=audit_path.parent/'request.json'
        checked(root,request_path.relative_to(root).as_posix(),audit['request_sha256']);request=read(request_path)
        plan_path=checked(root,request['plan_path'],audit['plan_sha256'])
        if request['plan_sha256']!=audit['plan_sha256'] or request['interaction_sha256']!=report['interaction_sha256']:
            raise ValueError('Scope request/plan binding differs')
        raw_path=checked(root,(audit_path.parent/audit['raw_response']).relative_to(root).as_posix());raw=read(raw_path)
        raw_rows={r['sample_index']:r for r in raw['observations']}
        targets={r['sample_index']:r for r in request['targets']}
        if len(raw_rows)!=len(raw['observations']) or len(targets)!=len(request['targets']):
            raise ValueError('Duplicate supplemental scope source IDs')
        rows={}
        for observation in audit['result']['observations']:
            index=observation['sample_index'];target=targets.get(index);original=raw_rows.get(index)
            if index in rows or not target or not original: raise ValueError('Supplemental scope source IDs differ')
            for key in ['verdict','component_boundary_plausible','single_component_visible','other_components_inside_mask','notes']:
                if observation.get(key)!=original.get(key): raise ValueError('Saved model scope judgment was changed')
            if (observation['source_image_sha256']!=target['source_image_sha256']
                    or observation['mask_sha256']!=target['mask_sha256'] or observation['track_id']!=request['track_id']):
                raise ValueError('Supplemental scope source identities differ')
            rows[index]=dict(observation,audit_path=audit_path.relative_to(root).as_posix(),audit_sha256=sha(audit_path),
                provider=audit.get('provider'),model=audit.get('model'),judgment_accuracy_measured=False,
                scope_definition=audit.get('scope_definition','single_component_visible_in_presented_crop_v1'),
                strict_motion_scope_gate_passed=(observation['verdict']=='plausible_visible_mask'
                    and observation['component_boundary_plausible'] is True
                    and observation['single_component_visible'] is True
                    and observation['other_components_inside_mask'] is False))
        if set(rows)!=set(targets): raise ValueError('Incomplete supplemental scope audit')
        key=plan_path.relative_to(root).as_posix()
        if key in scope_by_plan: raise ValueError('Duplicate supplemental audit for one plan')
        scope_by_plan[key]=rows
        paths=[audit_path,request_path,raw_path]
        paths.extend(checked(root,r['path'],r['sha256']) for r in audit.get('image_inputs',[]))
        scope_bindings.append({'path':relative_audit,'sha256':sha(audit_path),'plan_path':key,
            'plan_sha256':sha(plan_path),'scope_definition':audit.get('scope_definition','single_component_visible_in_presented_crop_v1'),
            'artifacts':[{'path':p.relative_to(root).as_posix(),'sha256':sha(p)} for p in paths]})
    for relative_plan in plans:
        path=checked(root,relative_plan);plan=read(path)
        if plan.get('format')!='short-interval-motion-plan' or plan['interaction_path']!=report['interaction_path'] or plan['interaction_sha256']!=report['interaction_sha256'] or plan['source_video_sha256']!=report['source']['sha256']:
            raise ValueError('Plan is not bound to the feedback publication/source')
        component=next((c for c in report['components'] if c['id']==plan['component_id']),None)
        if not component: raise ValueError('Unknown plan component')
        by_index={row['source_index']:row for row in component['samples']}
        for episode in plan['episodes']:
            selected=[]
            for frame in episode['frames']:
                row=by_index[frame['sample_index']]
                if abs(row['timestamp_s']-frame['timestamp_s'])>1e-7 or row['image_sha256']!=frame['source_image_sha256'] or row['mask_sha256']!=frame['mask_sha256']:
                    raise ValueError('Plan frame is not the exact source/mask/PTS')
                copied=copy.deepcopy(row)
                supplemental=scope_by_plan.get(path.relative_to(root).as_posix(),{}).get(frame['sample_index'])
                if supplemental:
                    if (supplemental['source_image_sha256']!=row['image_sha256'] or supplemental['mask_sha256']!=row['mask_sha256']):
                        raise ValueError('Supplemental scope audit differs from exact feedback source')
                    copied['supplemental_model_scope_audit']=supplemental
                selected.append(copied)
            if not selected or any(b['timestamp_s']<=a['timestamp_s'] for a,b in zip(selected,selected[1:])):
                raise ValueError('Plan samples must be nonempty and increasing')
            row=interval_row(component['id'],selected,component['id']+'--'+episode['id'],kind='frozen_reconstruction_episode',episode_id=episode['id'])
            reference=next((r for r in selected if r['source_index']==episode.get('reference_sample_index')),None)
            scope=reference.get('supplemental_model_scope_audit') if reference else None
            row.update(plan_path=path.relative_to(root).as_posix(),plan_sha256=sha(path),
                reference_sample_index=episode.get('reference_sample_index'),
                reference_scope_admission=('passed_saved_model_scope_gate' if scope['strict_motion_scope_gate_passed']
                    else 'withheld_by_saved_model_scope_gate') if scope else 'no_supplemental_scope_audit_attached',
                reference_scope_definition=scope.get('scope_definition') if scope else None,
                supplemental_scope_observations=[r['supplemental_model_scope_audit'] for r in selected if r.get('supplemental_model_scope_audit')])
            if scope and not scope['strict_motion_scope_gate_passed']:
                row['scope_review_action']={'title':'This reference did not pass the strict single-component scope gate',
                    'semantics':'cached_model_judgment_not_measured_mask_accuracy_or_physical_occlusion',
                    'action':'Review the cited crop. For a new capture, show the selected component clearly in an uncluttered close view. A correct visible mask can still fail this stricter motion experiment scope rule.',
                    'reference_source_index':reference['source_index'],'reference_timestamp_s':reference['timestamp_s'],
                    'audit':scope,'requires':{'additional_capture':True,'api':False,'gpu':False},'cause_verified':False}
            episodes.append(row)
        bindings.append({'path':path.relative_to(root).as_posix(),'sha256':sha(path)})
    if set(scope_by_plan)-{r['path'] for r in bindings}: raise ValueError('Scope audit needs its exact frozen plan attached')
    report.update(version=report['version']+1,episodes=episodes,derived_from={'path':base.relative_to(root).as_posix(),'sha256':sha(base)},
        episode_plans=bindings,supplemental_scope_audits=scope_bindings)
    return report
