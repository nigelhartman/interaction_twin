"""Hash-checked read-only display adapter for independent interval motion.

Only exact saved samples are displayable. No pose interpolation, alignment,
reference recovery or reconstruction is performed by this module.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import struct

LOADED_SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

STATUSES={'supported_nonreference_candidate','reference_unverified','withheld'}
CONVENTION='camera_from_native_mesh; OpenCV x-right,y-down,z-forward; native mesh @ R.T+t'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)

def path(root,value):
    root=Path(root).resolve()
    if not isinstance(value,str) or any(c in value for c in ['\\','?','#']):raise ValueError('Missing local display artifact')
    if value.startswith('/files/'):value=value[7:]
    candidate=Path(value) if Path(value).is_absolute() else root/value
    if '..' in candidate.parts or any(p.startswith('.') for p in candidate.parts if p not in {'/'}):
        raise ValueError('Hidden or traversing artifact path')
    if candidate.resolve()!=candidate or not candidate.is_file() or not candidate.is_relative_to(root/'data'):
        raise ValueError('Motion artifact must be an existing local data file without symlinks')
    return candidate


def verified(root,value,digest):
    if not isinstance(digest,str) or not re.fullmatch('[a-f0-9]{64}',digest):raise ValueError('Missing motion artifact identity')
    target=path(root,value)
    if sha(target)!=digest:raise ValueError('Motion source artifact bytes changed')
    return target


def url(root,target):return '/files/'+str(Path(target).relative_to(Path(root).resolve()))

def vector(v,n):return isinstance(v,list) and len(v)==n and all(finite(x)for x in v)

def rotation(R):
    if not isinstance(R,list) or len(R)!=3 or not all(vector(r,3)for r in R):return False
    dot=lambda a,b:sum(x*y for x,y in zip(a,b))
    if max(abs(dot(a,b)-(1 if i==j else 0))for i,a in enumerate(R)for j,b in enumerate(R))>1e-4:return False
    det=R[0][0]*(R[1][1]*R[2][2]-R[1][2]*R[2][1])-R[0][1]*(R[1][0]*R[2][2]-R[1][2]*R[2][0])+R[0][2]*(R[1][0]*R[2][1]-R[1][1]*R[2][0])
    return abs(det-1)<1e-4


def intrinsics(value):
    K=value.get('matrix')
    if not isinstance(K,list)or len(K)!=3 or not all(vector(r,3)for r in K)or K[0][0]<=0 or K[1][1]<=0 or K[2]!=[0,0,1]:
        raise ValueError('Invalid saved pinhole intrinsics')
    if not all(isinstance(value.get(k),int)and not isinstance(value[k],bool)and value[k]>0 for k in ['width','height']):
        raise ValueError('Missing original source dimensions')
    if value.get('calibrated',False) is not False:raise ValueError('This diagnostic viewer requires uncalibrated camera disclosure')
    return {'matrix':K,'width':value['width'],'height':value['height'],'calibrated':False}


def pose(row):
    R,t=row.get('candidate_rotation'),row.get('candidate_translation')
    if not rotation(R)or not vector(t,3):raise ValueError('Displayable saved pose is not a finite proper rigid transform')
    return {'R':R,'t':t}


def baseline_frame(row):
    status=('supported_nonreference_candidate'if row.get('status')=='supported_conditional_motion_candidate'else
            'reference_unverified'if row.get('status')=='reference_candidate_unverified_geometry_and_pose'else'withheld')
    result={'status':status,'reasons':row.get('reasons',[]),'reference':status=='reference_unverified'}
    result['pose']=pose(row)if status!='withheld'else None
    for key in ['heldout_median_px','heldout_p90_px','silhouette_iou','mask_coverage']:
        value=row.get(key)
        if finite(value):result[key]=value
    result['raw_status']=row.get('status')
    return result


def capture_feedback(root,feedback_path,doc,episodes):
    """Read a separately saved coach report; never use advice to admit poses."""
    target=path(root,str(feedback_path));report=read(target)
    if (report.get('format')!='source-capture-feedback' or report.get('version')not in {2,3} or
            report.get('interaction_sha256')!=doc['interaction_sha256'] or
            report.get('program_case_id')!=doc['case_id'] or
            report.get('source',{}).get('sha256')!=doc['source']['sha256']):
        raise ValueError('Capture feedback belongs to another source program')
    verified(root,report['interaction_path'],report['interaction_sha256'])
    rows=[]
    for episode in episodes:
        matches=[e for e in report.get('episodes',[])if e.get('component_id')==doc['component_id']and e.get('episode_id')==episode['id']]
        if not matches:continue
        if len(matches)!=1:raise ValueError('Ambiguous capture feedback episode')
        item=matches[0]
        if (abs(item['start_s']-episode['start_s'])>1e-7 or abs(item['end_s']-episode['end_s'])>1e-7 or
                item.get('reference_sample_index')!=episode['reference_sample_index']):
            raise ValueError('Capture advice episode differs from frozen motion plan')
        frames={f['sample_index']:f for f in episode['frames']}
        observations=[]
        for observation in item.get('supplemental_scope_observations',[]):
            frame=frames.get(observation.get('sample_index'))
            if frame is None or (observation.get('source_image_sha256')!=frame['image_sha256'] or
                    observation.get('mask_sha256')!=frame['mask_sha256'] or observation.get('track_id')!=doc['component_id']):
                raise ValueError('Capture scope note source/mask identity mismatch')
            verified(root,observation['audit_path'],observation['audit_sha256'])
            observations.append(observation)
        rows.append({'episode_id':episode['id'],'diagnostics':item.get('diagnostics',[]),
                     'scope_review_action':item.get('scope_review_action'),
                     'reference_scope_definition':item.get('reference_scope_definition'),
                     'scope_observations':observations,'cause_verified':False})
    return {'url':url(root,target),'sha256':sha(target),'version':report['version'],
            'episodes':rows,'used_for_tracking_selection_or_fitting':False,
            'meaning':'Read-only capture advice and fallible model scope notes; not physical or mask truth.'}


def normalize(root,input_path,feedback_path=None):
    root=Path(root).resolve();input_path=path(root,str(input_path));doc=read(input_path)
    if doc.get('format')!='short-interval-motion'or doc.get('version')!=1:
        raise ValueError('Not a completed versioned interval-motion result')
    if doc.get('status')not in {'complete_with_gaps','unsupported'}:
        raise ValueError('Running and failed partial attempts must remain raw diagnostics')
    if doc.get('physical_pose_accuracy_measured')is not False or doc.get('common_assembly_established')is not False:
        raise ValueError('Unsupported physical accuracy or assembly claim')
    if (type(doc.get('new_api_calls'))is not int or doc.get('new_api_calls')!=0 or doc.get('usable_physical_pose_recovered')is not False or
            doc.get('old_pose_initialization_used')is not False or doc.get('manual_geometry_or_poses')is not False):
        raise ValueError('Require explicit local automatic independent-reference provenance')
    if doc.get('track_id')!=doc.get('component_id')or doc.get('source_video_sha256')!=doc.get('source',{}).get('sha256')or doc.get('mesh_sha256')!=doc.get('asset',{}).get('sha256'):
        raise ValueError('Interval identity aliases disagree')
    if doc.get('coordinate_convention')!=CONVENTION:raise ValueError('Unknown native pose convention')
    source=verified(root,doc['source']['path'],doc['source']['sha256'])
    interaction=verified(root,doc['interaction_path'],doc['interaction_sha256'])
    program=read(interaction)
    if (program.get('case_id')!=doc.get('case_id')or program.get('source',{}).get('sha256')!=doc['source']['sha256']or
            doc['component_id']not in {t['id']for t in program.get('tracklets',[])}):
        raise ValueError('Interval component/program/source identity mismatch')
    program_evidence={}
    for evidence in program.get('evidence',[]):
        index=evidence.get('source_index')
        if index in program_evidence:raise ValueError('Duplicate registered program source index')
        program_evidence[index]=evidence
    asset=verified(root,doc['asset']['path'],doc['asset']['sha256'])
    if asset.suffix.lower()!='.glb'or doc['asset'].get('physical_scale_known')is not False:raise ValueError('Require unchanged native GLB and unknown scale')
    asset_bytes=asset.read_bytes()
    if len(asset_bytes)<20 or asset_bytes[:4]!=b'glTF' or struct.unpack_from('<II',asset_bytes,4)!=(2,len(asset_bytes)):
        raise ValueError('Malformed native GLB header')
    json_size,json_type=struct.unpack_from('<II',asset_bytes,12)
    if json_type!=0x4e4f534a:raise ValueError('Native GLB has no standard JSON scene')
    gltf=json.loads(asset_bytes[20:20+json_size])
    if any(isinstance(r.get('uri'),str)and not r['uri'].startswith('data:')for r in gltf.get('buffers',[])+gltf.get('images',[])):
        raise ValueError('External resources are not allowed in the native GLB inspector')
    baseline_path=verified(root,doc['baseline']['path'],doc['baseline']['sha256']);baseline=read(baseline_path)
    provenance=baseline.get('provenance',{})
    if provenance.get('source_video_sha256')!=doc['source']['sha256']or provenance.get('mesh_sha256')!=doc['asset']['sha256']:
        raise ValueError('Baseline and interval candidate must use identical source and native asset bytes')
    if doc['comparison'].get('exact_same_mesh')is not True:raise ValueError('Same-mesh comparison is required')
    reassessment=doc.get('baseline_reassessment_policy')
    if not isinstance(reassessment,dict):raise ValueError('Explicit same-audit baseline reassessment policy is required')
    if (reassessment.get('uses_saved_numeric_candidates_only')is not True or
            reassessment.get('uses_same_current_audit_and_original_geometric_gates')is not True or
            reassessment.get('new_inference_or_pose_refit')is not False or
            reassessment.get('original_outputs_overwritten')is not False):
        raise ValueError('Baseline reassessment must preserve saved poses and shared audit rules')
    baseline_K=intrinsics(baseline['intrinsics'])
    base_rows={}
    for row in baseline['frames']:
        key=round(row['timestamp_s'],7)
        if key in base_rows:raise ValueError('Duplicate preserved baseline timestamp')
        base_rows[key]=row
    comparison_rows={}
    for row in doc['comparison']['exact_pts_rows']:
        key=(row['episode_id'],row['sample_index'])
        if key in comparison_rows:raise ValueError('Duplicate exact-PTS comparison row')
        comparison_rows[key]=row
    episodes=[];ids=set();receipts={input_path,source,asset,baseline_path,interaction}
    for key in ['plan','audit_binding']:
        if isinstance(doc.get(key),dict):
            binding=doc[key]
            receipts.add(verified(root,binding['path'],binding['sha256']))
    counts={'fair_same_pts_samples':0,'candidate_supported':0,'baseline_supported':0,'both_supported':0,'references_excluded':0}
    for episode in doc['episodes']:
        eid=episode['id']
        if not isinstance(eid,str)or not re.fullmatch('[a-zA-Z0-9_-]{1,100}',eid)or eid in ids:raise ValueError('Duplicate/invalid interval identifier')
        ids.add(eid);K=intrinsics(episode['intrinsics'])
        if episode.get('gauge')!='independent_reference_conditioned_candidate':raise ValueError('Unknown independent-interval gauge')
        train=set(episode.get('queries',{}).get('training_point_ids',[]));held=set(episode.get('queries',{}).get('heldout_point_ids',[]))
        if (train&held or not all(type(i)is int and i>=0 and i%3!=0 for i in train)or
                not all(type(i)is int and i>=0 and i%3==0 for i in held)):
            raise ValueError('Training and held-out point IDs must retain the frozen modulo-three split')
        start,end=episode['start_s'],episode['end_s']
        if not finite(start)or not finite(end)or not 0<=start<=end:raise ValueError('Invalid source interval')
        frames=[];previous=-1;seen=set()
        for frame in episode['frames']:
            status=frame['status'];time=frame['timestamp_s'];sample=frame['sample_index']
            if status not in STATUSES or not finite(time)or not start-1e-7<=time<=end+1e-7 or time<=previous or sample in seen:
                raise ValueError('Unknown support status, duplicate sample or nonmonotonic source PTS')
            previous=time;seen.add(sample)
            evidence=program_evidence.get(sample)
            observations=[o for o in (evidence or {}).get('observations',[])if o.get('track_id')==doc['component_id']]
            if (not evidence or len(observations)!=1 or abs(evidence['timestamp_s']-time)>1e-7 or
                    evidence.get('frame_sha256')!=frame['source_image_sha256'] or
                    observations[0].get('mask_sha256')!=frame['mask_sha256']):
                raise ValueError('Motion frame must match exact registered source evidence and component mask')
            image=verified(root,frame['source_image_url'],frame['source_image_sha256'])
            mask=verified(root,frame['mask_url'],frame['mask_sha256']);receipts.update([image,mask])
            if (verified(root,evidence['frame_path'],evidence['frame_sha256'])!=image or
                    verified(root,observations[0]['mask_path'],observations[0]['mask_sha256'])!=mask):
                raise ValueError('Motion paths differ from registered exact evidence artifacts')
            from PIL import Image
            with Image.open(image)as im,Image.open(mask)as ma:
                if im.size!=(K['width'],K['height'])or ma.size!=im.size:
                    raise ValueError('Source image/mask dimensions disagree with saved camera')
            record={'sample_index':sample,'timestamp_s':time,'image_url':url(root,image),'image_sha256':frame['source_image_sha256'],
                    'mask_url':url(root,mask),'mask_sha256':frame['mask_sha256'],'status':status,
                    'reference':frame.get('reference')is True,'reasons':frame.get('reasons',[]),
                    'pose':pose(frame)if status!='withheld'else None,
                    'temporal_split':frame.get('temporal_split'),
                    'evaluation_timestamp_reserved_before_reference_selection':frame.get('evaluation_timestamp_reserved_before_reference_selection')is True}
            if status=='withheld'and(frame.get('candidate_rotation')is not None or frame.get('candidate_translation')is not None):
                raise ValueError('Rejected pose must be separated under raw diagnostics, not display candidate fields')
            if status=='reference_unverified'and not record['reference']:
                raise ValueError('Unverified reference pose must retain its original reference flag')
            if status=='supported_nonreference_candidate'and record['reference']:
                raise ValueError('Reference initialization cannot count as supported motion')
            split=frame.get('temporal_split');reserved=frame.get('evaluation_timestamp_reserved_before_reference_selection')
            if (split not in {'evaluation','reference_selection'}or type(reserved)is not bool or
                    reserved!=(split=='evaluation')or (record['reference']and reserved)):
                raise ValueError('Reserved temporal evaluation identity is inconsistent')
            visible_train=set(frame.get('training_visible_point_ids',[]));visible_held=set(frame.get('heldout_visible_point_ids',[]))
            if visible_train&visible_held or not visible_train<=train or not visible_held<=held:
                raise ValueError('Visible point IDs must remain within their frozen query partitions')
            for key in ['train_count','inlier_count','heldout_median_px','heldout_p90_px','silhouette_iou','mask_coverage']:
                if finite(frame.get(key)):record[key]=frame[key]
            base=base_rows.get(round(time,7))
            if base is not None and abs(base['timestamp_s']-time)<=1e-7:
                if base.get('source_frame_sha256')!=frame['source_image_sha256']or base.get('sam_mask_sha256')!=frame['mask_sha256']:
                    raise ValueError('Matched PTS baseline source/mask bytes differ')
                comparison=comparison_rows.get((eid,sample))
                if comparison is None:raise ValueError('Missing explicit exact-PTS comparison row')
                reassessed=comparison.get('baseline_reassessed_status')
                if reassessed not in STATUSES:raise ValueError('Missing explicit reassessed baseline status')
                original=baseline_frame(base)
                if (comparison.get('baseline_status')!=base.get('status') or
                        comparison.get('baseline_original_supported')!=(original['status']=='supported_nonreference_candidate')):
                    raise ValueError('Original preserved baseline status changed in comparison')
                if (reassessed=='reference_unverified')!=original['reference']:
                    raise ValueError('Baseline reference cannot be promoted or reassigned')
                record['baseline']={**original,'status':reassessed,
                    'original_status':base['status'],'original_supported':comparison['baseline_original_supported'],
                    'reasons':comparison.get('baseline_reassessed_reasons',[]),
                    'pose':pose(base)if reassessed!='withheld'else None}
                expected={'candidate_supported':status=='supported_nonreference_candidate',
                          'baseline_supported':record['baseline']['status']=='supported_nonreference_candidate',
                          'candidate_reference':record['reference'],'baseline_reference':record['baseline']['reference']}
                if abs(comparison['timestamp_s']-time)>1e-7 or any(comparison.get(k)!=v for k,v in expected.items()):
                    raise ValueError('Comparison flags or PTS do not match actual saved numeric records')
                fair=not(record['reference']or record['baseline']['reference'])and record['evaluation_timestamp_reserved_before_reference_selection']
                record['fair_comparison']=fair
                if fair:
                    counts['fair_same_pts_samples']+=1
                    counts['candidate_supported']+=expected['candidate_supported'];counts['baseline_supported']+=expected['baseline_supported']
                    counts['both_supported']+=expected['candidate_supported']and expected['baseline_supported']
                elif record['reference']or record['baseline']['reference']:counts['references_excluded']+=1
            else:record['baseline']=None;record['fair_comparison']=False
            frames.append(record)
        references=[r for r in frames if r['reference']]
        if (len(references)!=1 or references[0]['sample_index']!=episode['reference_sample_index']or
                abs(references[0]['timestamp_s']-episode['reference_timestamp_s'])>1e-7):
            raise ValueError('Interval reference identity or exact timestamp changed')
        episodes.append({'id':eid,'start_s':start,'end_s':end,'reference_sample_index':episode['reference_sample_index'],
                         'reference_timestamp_s':episode['reference_timestamp_s'],'intrinsics':K,
                         'gauge':episode['gauge'],'status':episode.get('status'),'reason':episode.get('reason'),
                         'frames':frames})
    adapter_hash=LOADED_SOURCE_SHA256
    identity={'input_sha256':sha(input_path),'adapter_sha256':adapter_hash,
              'feedback_sha256':sha(path(root,str(feedback_path)))if feedback_path else None}
    result={'format':'interval-motion-viewer','version':1,'id':'motion-'+hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20],
            'case_id':doc.get('case_id'),'component_id':doc['component_id'],'label':doc.get('label',doc['component_id']),
            'status':doc['status'],'raw_url':url(root,input_path),'input_sha256':sha(input_path),'adapter_sha256':adapter_hash,
            'source':{'url':url(root,source),'sha256':doc['source']['sha256'],'original_sha256':doc['source'].get('original_sha256')},
            'asset':{'url':url(root,asset),'sha256':doc['asset']['sha256'],'backend':doc['asset'].get('backend'),'physical_scale_known':False},
            'coordinate_convention':CONVENTION,'episodes':episodes,'baseline':{'url':url(root,baseline_path),'sha256':doc['baseline']['sha256'],'intrinsics':baseline_K},
            'comparison_counts':counts,'physical_pose_accuracy_measured':False,'common_assembly_established':False,
            'interaction_path':doc.get('interaction_path'),'interaction_sha256':doc.get('interaction_sha256'),
            'baseline_reassessment_policy':reassessment,
            'limitations':doc.get('limitations',[])+['Only saved exact samples are displayed. Unsupported poses and times between samples remain hidden.',
                'Each interval has its own reference-conditioned branch. Do not join intervals into a continuous physical trajectory.',
                'Reference candidates are unverified initializations and excluded from matched motion-support counts.',
                'Headline counts use exact matched evaluation timestamps reserved before reference selection. The two methods use different reference hypotheses and point sets; these counts do not measure physical accuracy.',
                'Orbiting changes the inspection camera only. Metric scale, physical shape and actual object motion remain unverified.'],
            'artifact_receipts':[{'path':str(p.relative_to(root)),'sha256':sha(p)}for p in sorted(receipts)]}
    if feedback_path:result['capture_feedback']=capture_feedback(root,feedback_path,doc,episodes)
    return result
