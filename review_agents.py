"""Four bounded automatic reviewer roles; structural checks, not independent AIs.

No inference or repair is launched here. Optional hosted reviews must arrive via
an explicitly consented, fixed trusted callback and retain provider provenance.
"""
from __future__ import annotations
import json
from pathlib import Path
import re
import struct

import stage5
from twin_catalog import CODE_INPUTS

ROLES=('evidence','geometry','mechanics','semantics')
HOSTED_ERROR_CODES={'prior_attempt_preserved_no_retry','budget_admission_blocked',
    'transport_failed_charge_uncertain','incomplete_or_invalid_model_response',
    'review_adapter_disabled','provider_request_rejected','hosted_review_unavailable',
    'structural_review_blocked','consent_not_granted','adapter_factory_unavailable',
    'invalid_or_unverifiable_report'}


def hosted_error(role,error=None,*,code=None):
    """A bounded diagnostic label, never a raw exception/provider message."""
    if role not in ROLES:raise ValueError('Unknown hosted-review role')
    if code is None:
        message=str(error).lower()
        if 'already attempted'in message or 'no paid resend'in message or 'already_attempted'in message:
            code='prior_attempt_preserved_no_retry'
        elif any(s in message for s in ['limit_reached','budget','admission','ledger']):code='budget_admission_blocked'
        elif any(s in message for s in ['transport','charge_uncertain']):code='transport_failed_charge_uncertain'
        elif 'disabled'in message:code='review_adapter_disabled'
        elif 'provider_rejected'in message:code='provider_request_rejected'
        elif any(s in message for s in ['incomplete','structured','schema','unknown saved id']):code='incomplete_or_invalid_model_response'
        elif any(s in message for s in ['hosted review','hosted reviewer','private hosted review']):code='invalid_or_unverifiable_report'
        else:code='hosted_review_unavailable'
    if code not in HOSTED_ERROR_CODES:raise ValueError('Unknown safe hosted-review error code')
    name=type(error).__name__ if error is not None else 'Unavailable'
    if not re.fullmatch('[A-Za-z_][A-Za-z0-9_]{0,79}',name):name='Exception'
    return {'format':'interaction-twin-hosted-review-error','role':role,'code':code,'error_type':name,
        'status':'unavailable','independent_model_review':False,'automatic_retry':False,
        'completed_pipeline_outputs_preserved':True}


def checked_hosted_error(value,role):
    """Allow a trusted factory to replay a saved failure without transport."""
    if (not isinstance(value,dict)or value.get('format')!='interaction-twin-hosted-review-error'or
        value.get('role')!=role or value.get('code')not in HOSTED_ERROR_CODES or
        not isinstance(value.get('error_type'),str)or not re.fullmatch('[A-Za-z_][A-Za-z0-9_]{0,79}',value['error_type'])):
        raise ValueError('Invalid saved hosted-review error')
    return {**hosted_error(role,code=value['code']),'error_type':value['error_type']}


def data_path(root,relative):
    root=Path(root).resolve()
    if not isinstance(relative,str) or not relative.startswith('data/') or any(p.startswith('.') for p in Path(relative).parts):
        raise ValueError('Review artifacts must use ordinary data paths')
    path=root/relative
    if path.resolve()!=path or any(p.is_symlink() for p in [path,*path.parents]):raise ValueError('Review artifact path is not confined')
    return path


def verified_ref(root,reference,maximum=512*1024*1024):
    if not isinstance(reference,dict) or set(reference)!={'path','sha256','bytes'}:raise ValueError('Exact reviewed artifact reference required')
    if not isinstance(reference['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',reference['sha256']) or type(reference['bytes'])is not int or not 0<reference['bytes']<=maximum:
        raise ValueError('Invalid reviewed artifact hash/size')
    path=data_path(root,reference['path'])
    import object_program
    if not path.is_file() or path.stat().st_size!=reference['bytes'] or object_program.sha(path)!=reference['sha256']:
        raise ValueError('Reviewed artifact is missing or changed')
    return path


def read_interaction(root,reference):
    path=verified_ref(root,reference,6*1024*1024)
    doc=json.loads(path.read_text());json.dumps(doc,allow_nan=False)
    # Confinement precedes the legacy validator's hashing, including compiler
    # paths; no submitted document can make a credential file an input.
    for item in [doc['source'],*doc['assets']]:data_path(root,item['path'])
    for item in doc['inputs']:
        if (item['kind'],item['path'])in CODE_INPUTS:
            p=Path(root)/item['path']
            if p.resolve()!=p or p.is_symlink():raise ValueError('Compiler review input escapes project')
        elif item['kind']=='model_tracklet_proposals' and item['path'].startswith('annotations/'):
            p=Path(root)/item['path']
            if p.resolve()!=p or p.is_symlink() or p.suffix!='.json':raise ValueError('Invalid model annotation input')
        else:data_path(root,item['path'])
    for frame in doc['evidence']:
        data_path(root,frame['frame_path'])
        for o in frame['observations']:data_path(root,o['mask_path'])
    for link in doc.get('links',{}).values():
        if link:data_path(root,link)
    stage5.validate_interaction(doc,root=root)
    return doc


def role(name,status,checks,issues):
    return {'role':name,'status':status,'mode':'deterministic_structural_review',
            'independent_model_review':False,'checks':checks,'issues':issues,'physical_accuracy_verified':False}


def review_revision(manifest,*,root,hosted_reviewer=None,processing_consent=False):
    root=Path(root).resolve();reviews=[];doc=None;object_doc=None;proposal=None
    try:
        verified_ref(root,manifest['source_ref'])
        artifacts=manifest.get('artifacts',[])
        if not isinstance(artifacts,list)or len(artifacts)>2048:raise ValueError('Invalid bounded revision artifacts')
        for item in artifacts:verified_ref(root,item)
        for key in ['object_program_ref','scene_ref']:
            if manifest.get(key):verified_ref(root,manifest[key],6*1024*1024)
        doc=read_interaction(root,manifest['interaction_ref'])
        capture=manifest['source_ref']['sha256']
        if capture not in {doc['source']['sha256'],doc['source']['original_sha256']}:
            raise ValueError('Uploaded recording does not match the evidence program source')
        reviews.append(role('evidence','pass',['Full saved program/source/artifact hashes and source sample times validated.'],[]))
    except Exception as exc:
        # Error text is local structural diagnostics only, never provider bodies.
        reviews.append(role('evidence','fail',[],[{'code':'invalid_source_evidence_contract','severity':'blocking','message':type(exc).__name__+': '+str(exc)[:400]}]))
    if doc is not None:
        try:
            issues=[]
            for asset in doc['assets']:
                path=data_path(root,asset['path'])
                with path.open('rb') as stream:header=stream.read(12)
                if len(header)!=12 or header[:4]!=b'glTF' or struct.unpack_from('<I',header,4)[0]!=2 or struct.unpack_from('<I',header,8)[0]!=path.stat().st_size:
                    raise ValueError('Generated asset is not a complete GLB2 container')
                from portable_twin import decode_glb_bytes
                decode_glb_bytes(path.read_bytes())
            issues.append({'code':'geometry_shape_not_automatically_verified','severity':'warning',
                           'message':'Hashes/container checks do not establish faithful geometry, texture, metric scale or clean contact surfaces.'})
            if not doc['assets']:issues.append({'code':'no_selected_geometry','severity':'warning','message':'This revision has descriptive evidence but no selected 3D assets.'})
            reviews.append(role('geometry','warning',['Selected asset hashes/GLB2 containers checked.'],issues))
        except (ValueError,OSError,KeyError,TypeError) as exc:
            reviews.append(role('geometry','fail',[],[{'code':'invalid_generated_asset','severity':'blocking','message':str(exc)[:400]}]))
        if manifest.get('object_program_ref'):
            try:
                import object_program
                path=verified_ref(root,manifest['object_program_ref'],object_program.MAX_JSON)
                object_doc=object_program.read_json(path);object_program.validate(object_doc,root=root)
                if object_doc['evidence_program']!=manifest['interaction_ref']:raise ValueError('Kinematics points to another source program')
                if manifest.get('scene_ref') and object_doc['constrained_scene']!=manifest['scene_ref']:raise ValueError('Scene and object-program companion differ')
                proposal=json.loads(verified_ref(root,object_doc['proposal'],6*1024*1024).read_text())
                issues=[{'code':'physics_unverified','severity':'warning','message':'Estimated joints and observed configuration ranges are not physical parameters or simulator acceptance.'}]
                unknown=sum(s['status']=='unknown' for j in object_doc['joints'] for s in j['samples'])
                if unknown:issues.append({'code':'unknown_joint_samples','severity':'warning','message':str(unknown)+' joint samples remain unknown and must stay absent.'})
                if object_doc['unresolved_connections']:issues.append({'code':'unresolved_persistent_connections','severity':'warning','message':'Unsupported persistent children remain withheld.'})
                if any(not j['angle_quality']['angle_visual_consistency'] for j in object_doc['joints']):
                    issues.append({'code':'opening_angle_unresolved','severity':'warning','message':'A fitted scalar is a configuration parameter, not a supported opening angle.'})
                reviews.append(role('mechanics','warning',['Source/asset-bound frames, actual native anchors, scale-once FK, angle evidence and unknown gaps validated.'],issues))
            except Exception as exc:
                reviews.append(role('mechanics','fail',[],[{'code':'invalid_kinematics','severity':'blocking','message':type(exc).__name__+': '+str(exc)[:400]}]))
        else:
            reviews.append(role('mechanics','warning',[],[{'code':'descriptive_only_no_kinematics','severity':'warning','message':'No verified kinematics companion; no physical joint or insertion claims allowed.'}]))
        reviews.append(role('semantics','warning',['Exact saved IDs, relation cycles, state/step citations and source-time ranges checked by evidence compiler.'],
            [{'code':'model_semantics_fallible','severity':'warning','message':'Visual meaning/identity/contact remain fallible proposals; human inspection required.'}]))
    else:
        for name in ['geometry','mechanics','semantics']:
            reviews.append(role(name,'fail',[],[{'code':'blocked_by_invalid_evidence','severity':'blocking','message':'Cannot review a source-invalid revision.'}]))
    hosted=[];hosted_errors=[];enabled=hosted_reviewer is not None
    structural_blocked=any(r['status']=='fail'for r in reviews)
    if enabled and not structural_blocked and processing_consent is True:
        for name in ROLES:
            try:
                result=hosted_reviewer(name,manifest)
                if isinstance(result,dict)and result.get('format')=='interaction-twin-hosted-review-error':
                    hosted_errors.append(checked_hosted_error(result,name));continue
                if not isinstance(result,dict) or result.get('role')!=name or result.get('physical_accuracy_verified')is not False or len(json.dumps(result))>16000:
                    raise ValueError('Invalid bounded hosted role-review result')
                if result.get('provider')!='meta_api' or not result.get('model') or not result.get('request_receipt'):
                    raise ValueError('Hosted role review needs actual provider/model/request provenance')
                result={**result,'mode':'hosted_role_review','independent_model_review':False}
                from admin_muse_reviews import verify_report
                verify_report(result,manifest,root=root)
                hosted.append(result)
            except Exception as exc:
                hosted_errors.append(hosted_error(name,exc))
    elif enabled:
        code='structural_review_blocked'if structural_blocked else'consent_not_granted'
        hosted_errors=[hosted_error(name,code=code)for name in ROLES]
    hosted_status=('complete'if len(hosted)==len(ROLES)else'partial'if hosted else
        'withheld_structural_failure'if structural_blocked and enabled else'unavailable'if enabled else'not_requested')
    from task_taxonomy import derive_task_classes
    return {'format':'interaction-twin-multireview','version':1,'roles':reviews,'hosted_roles':hosted,
            'hosted_roles_completed':len(hosted),'hosted_roles_expected':len(ROLES),'hosted_errors':hosted_errors,
            'hosted_review_enabled':enabled,'hosted_review_status':hosted_status,
            'hosted_warnings':([]if len(hosted)==len(ROLES)else[{'code':'optional_hosted_reviews_incomplete',
                'message':str(len(hosted))+' of '+str(len(ROLES))+' optional text-only Muse roles completed. Missing reviews do not certify the result; owner inspection and exact approval remain required.'}]),
            'role_count':len(reviews),'all_required_roles_completed':len(reviews)==4,
            'blocking_failures':sum(r['status']=='fail' for r in reviews)+sum(r['verdict']=='blocked' or any(i['severity']=='blocking'for i in r['issues'])for r in hosted),
            'task_classes':derive_task_classes(doc or {},object_program=object_doc,mechanism_proposals=proposal),
            'nigel_approval_required':True,'automatic_publication':False,'training_performed':False,
            'physical_accuracy_verified':False,'simulation_ready':False}
