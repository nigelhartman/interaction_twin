"""Versioned, gated local reconstruction of a registered saved component.

Stage receipts are append-only. Unsupported geometry stops downstream work;
cached sparse geometry is reused only by explicit identifier and verified source
binding. Inferred faces never enter the observed mesh or production catalog.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import statistics

VERSION = 'gated-local-v1'
BASE = 'data/observed-app-runs-20261031'
CACHE_ROOTS = ['data/observed-geometry-experiment-20261031',
               'data/observed-cotracker-experiment-20261101', BASE]
CACHE_ID = re.compile(r'sparse-cache-[0-9a-f]{16}\Z')
STAGES = ['sparse', 'dense', 'validation', 'completion', 'exports']


class StageUnavailable(RuntimeError):
    """A requested prepared capability is unavailable; never silently fallback."""
    def __init__(self,message,*,result=None):
        super().__init__(message)
        self.result=result


CAMERA_DISAGREEMENT='TRAIN-frame camera priors disagree; no camera assumption is returned'


def camera_disagreement_result(root,folder,error):
    """Recognize only a complete, unchanged, known unsupported camera result.

    Arbitrary exceptions and malformed receipts remain failures. This does not
    change a camera, rerun a model, relax a threshold or manufacture geometry.
    """
    if not isinstance(error,ValueError) or str(error)!=CAMERA_DISAGREEMENT:
        return None
    root,folder=Path(root).resolve(),Path(folder)
    try:
        if folder.resolve()!=folder or not folder.is_relative_to(root/'data'):
            return None
        def regular(name):
            path=folder/name
            if path.resolve()!=path or not path.is_file() or path.stat().st_size>16*1024*1024:
                raise ValueError('Invalid preserved camera diagnostic')
            return path
        report_path=regular('camera-prior/camera-prior.json');report=read(report_path)
        engine_protocol=read(regular('protocol.json'));prior=read(regular('camera-prior/protocol.json'))
        if not all(isinstance(row,dict) for row in [report,engine_protocol,prior]):return None
        provider=engine_protocol.get('camera_provider')
        if not isinstance(provider,dict) or not isinstance(provider.get('declaration'),dict):return None
        policy=provider['declaration']
        if (report.get('format')!='ordinary-video-camera-prior' or report.get('version')!=1
                or report.get('status')!='unsupported' or report.get('reason')!=CAMERA_DISAGREEMENT
                or report.get('K') is not None or report.get('policy')!=policy
                or policy.get('version')!='moge2-train-camera-prior-v1'
                or prior.get('format')!='ordinary-video-camera-prior-protocol' or prior.get('version')!=1
                or prior.get('policy')!=policy or prior.get('frozen_before_inference') is not True
                or prior.get('heldout_images_used') is not False):return None
        for row in [report,policy]:
            if row.get('new_api_calls')!=0:return None
            if any(row.get(flag) is not False for flag in ['camera_calibrated','physical_scale_known','physical_accuracy_verified']):return None
        if any(policy.get(flag) is not False for flag in ['heldout_images_used','depth_or_point_maps_used','downloads_allowed']):return None
        integrity=report.get('input_integrity')
        if not isinstance(integrity,dict) or integrity.get('unchanged') is not True:return None
        inputs=prior.get('input_views');source=engine_protocol.get('input_views')
        if (not isinstance(inputs,list) or not 2<=len(inputs)<=3 or not all(isinstance(row,dict) for row in inputs)
                or not isinstance(source,list) or not all(isinstance(row,dict) for row in source)
                or type(report.get('actual_model_calls')) is not int or type(report.get('completed_model_calls')) is not int
                or report['actual_model_calls']!=len(inputs) or report['completed_model_calls']!=len(inputs)):return None
        files,buffers=integrity.get('files'),integrity.get('decoded_source_buffers')
        if (not isinstance(files,list) or not files or not isinstance(buffers,list) or len(buffers)!=len(inputs)
                or not all(isinstance(row,dict) and row.get('unchanged') is True for row in files+buffers)):return None
        minimum,maximum=policy.get('minimum_train_images'),policy.get('maximum_train_images')
        if type(minimum) is not int or type(maximum) is not int or not 2<=minimum<=maximum<=3 or not minimum<=len(inputs)<=maximum:return None
        training=sorted([row for row in source if row.get('split')=='train'],key=lambda row:(row['timestamp_s'],row['id']))
        if len(training)<minimum or [row['id'] for row in inputs]!=[training[i]['id'] for i in sorted({0,len(training)//2,len(training)-1})]:return None
        integrity_hashes={row['path']:row['sha256_before'] for row in files}
        adapter_hash=sha(regular('camera-prior/implementation.py'))
        if adapter_hash!=prior.get('implementation_sha256') or not any(integrity_hashes.get(name)==adapter_hash for name in ['automatic_camera_prior.py','adapter_implementation']):return None
        known={row['id']:row for row in source if row.get('split')=='train'}
        if len({row['id'] for row in inputs})!=len(inputs) or [row.get('frame_id') for row in buffers]!=[row['id'] for row in inputs]:return None
        size=prior.get('native_size')
        if not isinstance(size,list) or len(size)!=2 or any(type(x) is not int or x<=0 for x in size):return None
        cameras=[];predictions=[]
        for i,row in enumerate(inputs,1):
            if row.get('id') not in known or any(row.get(k)!=known[row['id']].get(k) for k in ['id','timestamp_s','image_url','image_sha256','split']):return None
            path=root/row['image_url'].removeprefix('/files/')
            if (path.resolve()!=path or not path.is_relative_to(root/'data') or sha(path)!=row['image_sha256']
                    or integrity_hashes.get(str(path.relative_to(root)))!=row['image_sha256']):return None
            call_path=regular(f'camera-prior/call-{i}-start.json');prediction_path=regular(f'camera-prior/prediction-{i}.json')
            call,prediction=read(call_path),read(prediction_path)
            if not isinstance(call,dict) or not isinstance(prediction,dict):return None
            factor=min(1.,policy['maximum_input_long_side']/max(size));model_size=[max(1,round(x*factor)) for x in size]
            if (call.get('call_number')!=i or call.get('frame_id')!=row['id'] or call.get('split')!='train'
                    or call.get('input_size')!=model_size or call.get('status')!='inference_invocation_start_recorded'
                    or prediction.get('frame_id')!=row['id'] or prediction.get('input_size')!=model_size):return None
            k=prediction.get('normalized_K')
            if (not isinstance(k,list) or len(k)!=3 or any(not isinstance(r,list) or len(r)!=3 for r in k)
                    or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for r in k for x in r)
                    or k[2]!=[0.,0.,1.] or k[0][0]<=0 or k[1][1]<=0 or abs(k[0][1])>1e-10 or abs(k[1][0])>1e-10):return None
            native=[k[0][0]*size[0],k[1][1]*size[1],k[0][2]*size[0]-.5,k[1][2]*size[1]-.5]
            if (any(not policy['minimum_focal_over_long_side']<=native[j]/max(size)<=policy['maximum_focal_over_long_side'] for j in [0,1])
                    or not -.1*size[0]<=native[2]<=1.1*size[0] or not -.1*size[1]<=native[3]<=1.1*size[1]):return None
            cameras.append(native)
            predictions.append({'frame_id':row['id'],'report_url':'/files/'+str(prediction_path.relative_to(root)),
                'sha256':sha(prediction_path),'call_start':{'report_url':'/files/'+str(call_path.relative_to(root)),
                    'sha256':sha(call_path)}})
        if len(list((folder/'camera-prior').glob('call-*-start.json')))!=len(inputs) or len(list((folder/'camera-prior').glob('prediction-*.json')))!=len(inputs):return None
        median=[statistics.median([k[j] for k in cameras]) for j in range(4)]
        spread=max((max(k[j] for k in cameras)-min(k[j] for k in cameras))/median[j] for j in [0,1])
        center=max(max(k[j] for k in cameras)-min(k[j] for k in cameras) for j in [2,3])/max(size)
        if not (spread>policy['maximum_relative_focal_spread'] or center>policy['maximum_principal_point_spread_over_long_side']):return None
        return {'camera_report_url':'/files/'+str(report_path.relative_to(root)),'camera_report_sha256':sha(report_path),
            'camera_status':'unsupported_inconsistent_train_camera_priors','camera_reason':CAMERA_DISAGREEMENT,
            'actual_model_calls':len(inputs),'completed_model_calls':len(inputs),'new_api_calls':0,
            'camera_calibrated':False,'physical_scale_known':False,'model_fallback_ran':False,'automatic_retry':False,
            'camera_diagnostics':{'protocol':{'report_url':'/files/'+str((folder/'camera-prior/protocol.json').relative_to(root)),
                'sha256':sha(folder/'camera-prior/protocol.json')},'predictions':predictions,
                'relative_focal_spread':spread,'principal_point_spread_over_long_side':center}}
    except (OSError,ValueError,KeyError,TypeError,AttributeError):
        return None


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def data_path(root, value):
    if not isinstance(value, str) or not value.startswith('data/') or any(c in value for c in ['\\', '?', '#', '\0']):
        raise ValueError('Only registered project data artifacts are permitted')
    path = (Path(root) / value).resolve()
    if not path.is_relative_to(Path(root).resolve() / 'data'):
        raise ValueError('Artifact escaped project data')
    return path


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, allow_nan=False).encode() + b'\n'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def _candidate_binding(root, folder):
    protocol = read(folder / 'protocol.json')
    if not all(protocol.get(flag) is False for flag in ['manual_coordinates', 'generated_meshes_read', 'depth_maps_read', 'old_poses_read']):
        raise ValueError('Forbidden-input exclusion is not established')
    if sha(folder / 'implementation.py') != protocol['implementation_sha256']:
        raise ValueError('Sparse implementation snapshot mismatch')
    verify_correspondence_provider(folder, protocol)
    program_path = data_path(root, protocol['program_path'])
    if sha(program_path) != protocol['program_sha256']:
        raise ValueError('Sparse input program changed')
    program = read(program_path)
    binding_artifacts = {protocol['program_path']: sha(program_path)}
    if program.get('format') == 'observed_geometry_input':
        provenance = program['provenance']
        metadata_path = data_path(root, provenance['metadata_path'])
        metadata = read(metadata_path)
        binding_artifacts[provenance['metadata_path']] = sha(metadata_path)
        if metadata['geometry_input']['sha256'] != sha(program_path) or metadata.get('no_manual_geometry_or_poses') is not True:
            raise ValueError('Derived dense-mask input binding is not established')
        binding = metadata['source_program']
        source_path = data_path(root, binding['path'])
        if sha(source_path) != binding['sha256']:
            raise ValueError('Derived input original program changed')
        binding_artifacts[binding['path']] = sha(source_path)
    else:
        binding = {'path': protocol['program_path'], 'sha256': protocol['program_sha256']}
    viewer = read(folder / 'viewer.json')
    if viewer['input_protocol_sha256'] != sha(folder / 'protocol.json') or viewer.get('prior_geometry_or_depth_used') is not False:
        raise ValueError('Sparse review provenance mismatch')
    camera_artifacts=verify_camera_provider(root,folder,protocol,viewer,program)
    binding_artifacts.update(camera_artifacts)
    binding = dict(binding, artifact_sha256=binding_artifacts)
    if camera_artifacts:
        binding['camera_artifact_sha256']=camera_artifacts
    return protocol, viewer, binding


def verify_camera_provider(root,folder,protocol,viewer,program=None):
    """Recheck the frozen local TRAIN-camera receipt chain without inference."""
    provider=protocol.get('camera_provider')
    if provider is None:
        if viewer.get('camera_assumption') is not None or viewer.get('learned_camera_prior_used') is True or protocol.get('learned_camera_prior_used') is True:
            raise ValueError('Learned camera metadata has no frozen provider declaration')
        return {}
    if not isinstance(provider,dict):
        raise ValueError('Saved camera provider must be a record')
    root,folder=Path(root).resolve(),Path(folder)
    if folder.resolve()!=folder or not folder.is_relative_to(root/'data'):
        raise ValueError('Camera cache must be a regular confined data directory')
    def regular(path):
        path=Path(path)
        if path.resolve()!=path or not path.is_relative_to(folder) or not path.is_file():
            raise ValueError('Camera cache artifacts must be regular files strictly inside their result')
        if path.stat().st_size>16*1024*1024:
            raise ValueError('Camera metadata artifact exceeds its bounded receipt size')
        return path
    declaration=provider.get('declaration')
    if not isinstance(declaration,dict) or protocol.get('learned_camera_prior_used') is not True:
        raise ValueError('Camera cache lacks its frozen declaration')
    if declaration.get('version')!='moge2-train-camera-prior-v1':
        raise ValueError('Unsupported saved camera provider')
    for flag in ['heldout_images_used','depth_or_point_maps_used','camera_calibrated','physical_scale_known',
                 'physical_accuracy_verified','downloads_allowed','manual_coordinates','device_fallback_allowed']:
        if declaration.get(flag) is not False:
            raise ValueError('Camera declaration exceeds local unverified TRAIN scope')
    if declaration.get('new_api_calls')!=0 or provider.get('estimation_precedes_geometry') is not True or provider.get('heldout_images_supplied_to_provider') is not False:
        raise ValueError('Camera declaration does not precede new TRAIN-only geometry')
    result_path=regular(folder/'camera-assumption.json')
    result_name=str(result_path.relative_to(root))
    if provider.get('result_record_path')!=result_name:
        raise ValueError('Camera result differs from its predeclared destination')
    record=viewer.get('camera_assumption')
    if not isinstance(record,dict) or viewer.get('learned_camera_prior_used') is not True or record.get('result_record')!={'path':result_name,'sha256':sha(result_path)}:
        raise ValueError('Viewer camera result identity differs')
    assumption=read(result_path)
    if not isinstance(assumption,dict):
        raise ValueError('Saved camera assumption must be a record')
    if {k:v for k,v in record.items() if k!='result_record'}!=assumption:
        raise ValueError('Viewer camera values differ from the immutable result')
    prior=folder/'camera-prior'
    report_path=regular(prior/'camera-prior.json');report_name=str(report_path.relative_to(root))
    if (assumption.get('report_path')!=report_name or assumption.get('report_sha256')!=sha(report_path)
            or assumption.get('report')!={'path':report_name,'sha256':sha(report_path)}):
        raise ValueError('Camera report identity differs from the camera result')
    report=read(report_path)
    if not isinstance(report,dict):
        raise ValueError('Saved camera inference report must be a record')
    integrity=report.get('input_integrity')
    if not isinstance(integrity,dict):
        raise ValueError('Camera input integrity must be a record')
    if (report.get('format')!='ordinary-video-camera-prior' or report.get('version')!=1
            or report.get('status')!='unvalidated_learned_camera_assumption'
            or assumption.get('kind')!=report['status'] or report.get('policy')!=declaration
            or assumption.get('policy')!=declaration):
        raise ValueError('Camera report status or policy differs')
    for row in [assumption,report]:
        for flag in ['camera_calibrated','physical_scale_known','physical_accuracy_verified',
                     'prior_geometry_or_poses_used','learned_depth_or_point_maps_used']:
            if row.get(flag) is not False:
                raise ValueError('Camera receipt claims unsupported geometry or physical accuracy')
        if row.get('new_api_calls')!=0:
            raise ValueError('Camera receipt contains paid calls')
    if report.get('geometry_must_be_reconstructed_with_this_assumption') is not True or assumption.get('geometry_reconstructed_with_this_assumption') is not True:
        raise ValueError('Camera result is not bound to fresh geometry')
    program=program if program is not None else read(data_path(root,protocol['program_path']))
    if protocol.get('source')!=program.get('source'):
        raise ValueError('Camera source identity differs from its pinned program')
    if assumption.get('source_binding')!={'case_id':program.get('case_id'),'source':protocol['source'],
            'track_id':protocol['track_id'],'program_path':protocol['program_path'],'program_sha256':protocol['program_sha256']}:
        raise ValueError('Camera result belongs to another source program or component')
    size=assumption.get('native_size')
    if not isinstance(size,list) or len(size)!=2 or any(type(x) is not int or x<=0 for x in size) or report.get('native_size')!=size:
        raise ValueError('Camera native dimensions differ')
    def camera_K(value):
        if (not isinstance(value,list) or len(value)!=3 or any(not isinstance(row,list) or len(row)!=3 for row in value)
                or any(isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) for row in value for x in row)
                or value[2]!=[0.,0.,1.] or value[0][0]<=0 or value[1][1]<=0
                or abs(value[0][1])>1e-10 or abs(value[1][0])>1e-10):
            raise ValueError('Invalid full camera intrinsics in cache')
        return value
    K=camera_K(assumption.get('K'))
    if camera_K(report.get('K'))!=K:
        raise ValueError('Camera K differs from report')
    saved_cameras=viewer.get('cameras',[]);saved_frames=viewer.get('frames',[])
    if (not isinstance(saved_cameras,list) or not all(isinstance(row,dict) for row in saved_cameras)
            or not isinstance(saved_frames,list) or not all(isinstance(row,dict) for row in saved_frames)):
        raise ValueError('Saved geometry cameras and source frames must be record lists')
    for camera in saved_cameras:
        if camera.get('K')!=K:
            raise ValueError('Saved geometry camera uses a different K')
    if any([frame.get('width'),frame.get('height')]!=size for frame in saved_frames):
        raise ValueError('Saved geometry uses another native image size')
    prior_protocol_path=regular(prior/'protocol.json');prior_protocol=read(prior_protocol_path)
    if not isinstance(prior_protocol,dict):
        raise ValueError('Saved camera inference protocol must be a record')
    if (report.get('protocol_sha256')!=sha(prior_protocol_path)
            or prior_protocol.get('format')!='ordinary-video-camera-prior-protocol'
            or prior_protocol.get('version')!=1 or prior_protocol.get('policy')!=declaration
            or prior_protocol.get('native_size')!=size or prior_protocol.get('frozen_before_inference') is not True
            or prior_protocol.get('heldout_images_used') is not False):
        raise ValueError('Camera inference protocol differs from its result')
    if sha(regular(prior/'implementation.py'))!=prior_protocol.get('implementation_sha256'):
        raise ValueError('Camera adapter snapshot changed')
    provenance_path=regular(prior/'model-provenance.json')
    provenance=read(provenance_path)
    if not isinstance(provenance,dict) or sha(provenance_path)!=prior_protocol.get('model_provenance_sha256') or provenance.get('checkpoint_sha256')!=prior_protocol.get('checkpoint_sha256'):
        raise ValueError('Camera model provenance snapshot changed')
    for field,name in [('model_code','model-code.json'),('model_code_integrity','model-code-integrity.json')]:
        path=regular(prior/name)
        if report.get(field)!={'path':str(path.relative_to(root)),'sha256':sha(path)}:
            raise ValueError('Camera model implementation receipt changed')
    model_code=read(prior/'model-code.json');model_integrity=read(prior/'model-code-integrity.json')
    if not isinstance(model_code,dict) or not isinstance(model_integrity,dict):
        raise ValueError('Camera model code and integrity must be records')
    model_sha=sha(regular(prior/'model-implementation.py'))
    if (model_code.get('module_sha256')!=model_sha or model_code.get('checkpoint_sha256')!=prior_protocol.get('checkpoint_sha256')
            or model_integrity.get('unchanged') is not True or model_integrity.get('module_sha256_before')!=model_sha
            or model_integrity.get('module_sha256_after')!=model_sha):
        raise ValueError('Camera model code snapshot or integrity differs')
    inputs=report.get('input_views')
    minimum,maximum=declaration.get('minimum_train_images'),declaration.get('maximum_train_images')
    if (type(minimum) is not int or type(maximum) is not int or not 2<=minimum<=maximum<=3
            or not isinstance(inputs,list) or not minimum<=len(inputs)<=maximum or not all(isinstance(row,dict) for row in inputs)
            or assumption.get('input_views')!=inputs or prior_protocol.get('input_views')!=inputs):
        raise ValueError('Camera TRAIN source count or bindings differ')
    if len({row.get('timestamp_s') for row in inputs})<minimum or len({row.get('image_url') for row in inputs})!=len(inputs):
        raise ValueError('Camera TRAIN source times and images must be distinct')
    train=sorted([f for f in protocol['input_views'] if f['split']=='train'],key=lambda f:(f['timestamp_s'],f['id']))
    if len(train)<minimum:
        raise ValueError('Camera source lacks distinct TRAIN views')
    expected=[train[i] for i in sorted({0,len(train)//2,len(train)-1})]
    if [r.get('id') for r in inputs]!=[r['id'] for r in expected]:
        raise ValueError('Camera TRAIN selection differs from the predeclared temporal policy')
    predictions=report.get('predictions')
    if not isinstance(predictions,list) or len(predictions)!=len(inputs) or not all(isinstance(row,dict) for row in predictions):
        raise ValueError('Camera prediction count differs')
    if (type(report.get('actual_model_calls')) is not int or type(report.get('completed_model_calls')) is not int
            or report['actual_model_calls']!=len(inputs) or report['completed_model_calls']!=len(inputs)
            or integrity.get('unchanged') is not True):
        raise ValueError('Camera invocation counters or input integrity differ')
    files_integrity=integrity.get('files',[]);buffers_integrity=integrity.get('decoded_source_buffers',[])
    if (not isinstance(files_integrity,list) or not files_integrity or not isinstance(buffers_integrity,list)
            or not all(isinstance(row,dict) for row in files_integrity+buffers_integrity)
            or len(buffers_integrity)!=len(inputs) or any(row.get('unchanged') is not True for row in files_integrity+buffers_integrity)
            or [row.get('frame_id') for row in buffers_integrity]!=[row['id'] for row in inputs]):
        raise ValueError('Camera source/code integrity records are incomplete')
    integrity_by_path={row.get('path'):row.get('sha256_before') for row in files_integrity}
    if (integrity_by_path.get('models/moge-2-vits-normal/model.pt')!=prior_protocol['checkpoint_sha256']
            or integrity_by_path.get('models/moge-2-vits-normal/provenance.json')!=prior_protocol['model_provenance_sha256']
            or not any(integrity_by_path.get(name)==prior_protocol['implementation_sha256']
                       for name in ['automatic_camera_prior.py','adapter_implementation'])):
        raise ValueError('Camera model/adapter integrity hashes differ from frozen snapshots')
    try:
        import cv2
    except ImportError as error:
        raise ValueError('The prepared native source decoder is unavailable for cache verification') from error
    native_cameras=[]
    for i,(row,frozen,prediction) in enumerate(zip(inputs,expected,predictions),1):
        if any(row.get(k)!=frozen[k] for k in ['id','timestamp_s','image_url','image_sha256','split']) or row.get('native_size')!=size:
            raise ValueError('Camera input differs from an exact native TRAIN frame')
        if integrity_by_path.get(row['image_url'].removeprefix('/files/'))!=row['image_sha256']:
            raise ValueError('Camera input-integrity hash differs from its exact TRAIN source')
        image_path=root/row['image_url'].removeprefix('/files/')
        data_path(root,row['image_url'].removeprefix('/files/'))
        if image_path.resolve()!=image_path or not image_path.is_file() or sha(image_path)!=row['image_sha256']:
            raise ValueError('Camera TRAIN source image changed')
        decoded=cv2.imread(str(image_path))
        if decoded is None or [decoded.shape[1],decoded.shape[0]]!=size or hashlib.sha256(decoded.tobytes()).hexdigest()!=row.get('decoded_bgr_sha256'):
            raise ValueError('Camera decoded native source pixels changed')
        factor=min(1.,declaration['maximum_input_long_side']/max(size))
        model_size=[max(1,round(x*factor)) for x in size]
        call=read(regular(prior/f'call-{i}-start.json'))
        saved_prediction=read(regular(prior/f'prediction-{i}.json'))
        if (call.get('call_number')!=i or call.get('frame_id')!=row['id'] or call.get('split')!='train'
                or call.get('input_size')!=model_size or call.get('status')!='inference_invocation_start_recorded'
                or saved_prediction!=prediction or prediction.get('frame_id')!=row['id'] or prediction.get('input_size')!=model_size):
            raise ValueError('Camera inference invocation or prediction receipt differs')
        normalized=camera_K(prediction.get('normalized_K'))
        native=[[normalized[0][0]*size[0],0.,normalized[0][2]*size[0]-.5],
                [0.,normalized[1][1]*size[1],normalized[1][2]*size[1]-.5],[0.,0.,1.]]
        if (any(not declaration['minimum_focal_over_long_side']<=native[j][j]/max(size)<=declaration['maximum_focal_over_long_side'] for j in [0,1])
                or not -.1*size[0]<=native[0][2]<=1.1*size[0] or not -.1*size[1]<=native[1][2]<=1.1*size[1]):
            raise ValueError('Camera prediction exceeds the frozen intrinsic bounds')
        native_cameras.append(native)
    combined=[[statistics.median([k[i][j] for k in native_cameras]) for j in range(3)] for i in range(3)]
    if combined!=K:
        raise ValueError('Saved camera K is not the declared median of its TRAIN predictions')
    spread=max((max(k[j][j] for k in native_cameras)-min(k[j][j] for k in native_cameras))/K[j][j] for j in [0,1])
    center=max(max(k[j][2] for k in native_cameras)-min(k[j][2] for k in native_cameras) for j in [0,1])/max(size)
    metrics=report.get('metrics',{})
    if (not isinstance(metrics,dict) or spread>declaration['maximum_relative_focal_spread'] or center>declaration['maximum_principal_point_spread_over_long_side']
            or metrics.get('train_image_count')!=len(inputs)
            or metrics.get('agreement_measures_physical_accuracy') is not False
            or not math.isclose(metrics.get('relative_focal_spread',math.inf),spread,rel_tol=1e-12,abs_tol=1e-12)
            or not math.isclose(metrics.get('principal_point_spread_over_long_side',math.inf),center,rel_tol=1e-12,abs_tol=1e-12)):
        raise ValueError('Camera prediction agreement is inconsistent with the frozen policy')
    files={result_name:sha(result_path)}
    artifacts=list(prior.rglob('*'))
    if len(artifacts)>64 or len(list(prior.glob('call-*-start.json')))!=len(inputs) or len(list(prior.glob('prediction-*.json')))!=len(inputs):
        raise ValueError('Unexpected camera receipt count')
    for path in artifacts:
        if path.resolve()!=path or not path.is_relative_to(prior) or path.is_symlink():
            raise ValueError('Camera receipt paths must not traverse symlinks')
        if path.is_file():
            files[str(path.relative_to(root))]=sha(regular(path))
        elif not path.is_dir():
            raise ValueError('Camera receipt contains a nonregular artifact')
    return files


def verify_correspondence_provider(folder, protocol):
    provider = protocol.get('correspondence_provider')
    if provider is None:
        return
    if provider['protocol']['feature_provider'] not in {'cotracker3_mask_core_v1', 'cotracker3_mask_core_v2'}:
        raise ValueError('Unsupported saved correspondence provider')
    if sha(folder / 'correspondence-implementation.py') != provider['provider_implementation_sha256'] or sha(folder / 'cotracker-predictor.py') != provider['predictor_implementation_sha256']:
        raise ValueError('Correspondence implementation snapshots changed')
    if read(folder / 'cotracker-model-provenance.json') != provider['model_provenance'] or provider['model_provenance']['checkpoint_sha256'] != provider['checkpoint_sha256']:
        raise ValueError('Correspondence checkpoint provenance differs')
    tracking = read(folder / 'tracking.json'); audit = read(folder / 'correspondence-audit.json')
    if sha(folder / 'tracking.json') != audit['tracking_json_sha256']:
        raise ValueError('Correspondence tracking receipt changed')
    if audit['provider_implementation_sha256'] != provider['provider_implementation_sha256'] or audit['checkpoint_sha256'] != provider['checkpoint_sha256']:
        raise ValueError('Correspondence audit differs from frozen provider')
    if tracking['actual_model_calls'] != len(tracking['calls']) or tracking['actual_model_calls'] != audit['actual_model_calls'] or audit['actual_model_calls'] > provider['maximum_model_calls']:
        raise ValueError('Correspondence call accounting differs')
    if tracking.get('status') != 'complete' or sum(c['scope'].startswith('train_geometry') for c in tracking['calls']) != audit['training_model_calls'] or sum(c['scope'].startswith('heldout_') for c in tracking['calls']) != audit['heldout_model_calls']:
        raise ValueError('Correspondence call scope accounting differs')
    train = [f['id'] for f in protocol['input_views'] if f['split'] == 'train']
    heldout = [f['id'] for f in protocol['input_views'] if f['split'] == 'heldout']
    if provider['training_source_frame_ids'] != train or provider['heldout_source_frame_ids'] != heldout:
        raise ValueError('Correspondence source split differs')
    if provider['manual_queries'] is not False or provider['heldout_images_used_for_geometry'] is not False or tracking['manual_queries'] is not False:
        raise ValueError('Manual queries or held-out geometry input is forbidden')
    if audit['geometry_used_only_training_images'] is not True or audit['old_poses_or_generated_geometry_read'] is not False or tracking['learned_3d_or_depth_calls'] != 0 or tracking['new_api_calls'] != 0:
        raise ValueError('Correspondence source scope is not established')
    for call in tracking['calls']:
        ids = call['source_frame_ids']
        if call['scope'].startswith('train_geometry') and any(frame not in train for frame in ids):
            raise ValueError('Held-out source image entered geometry tracking')
        if call['scope'].startswith('heldout_') and (len(ids) != 2 or sum(frame in heldout for frame in ids) != 1 or sum(frame in train for frame in ids) != 1):
            raise ValueError('Held-out tracking is not isolated to reference plus test image')


def sparse_caches(root, *, interaction_sha256=None, track_id=None):
    """Identifier catalog only; exact pixel hashes are rechecked on execution."""
    root = Path(root).resolve()
    rows = []
    for base in CACHE_ROOTS:
        for protocol_file in sorted((root / base).glob('*/protocol.json')):
            folder = protocol_file.parent
            if not all((folder / name).is_file() for name in ['viewer.json', 'geometry.json', 'implementation.py']):
                continue
            try:
                protocol, viewer, binding = _candidate_binding(root, folder)
                if interaction_sha256 and binding['sha256'] != interaction_sha256:
                    continue
                if track_id and protocol['track_id'] != track_id:
                    continue
                relative = folder.relative_to(root).as_posix()
                frozen_files = ['protocol.json', 'viewer.json', 'geometry.json', 'implementation.py']
                if protocol.get('correspondence_provider'):
                    frozen_files += ['correspondence-implementation.py', 'cotracker-predictor.py',
                        'cotracker-model-provenance.json', 'tracking.json', 'correspondence-audit.json']
                    frozen_files += [p.name for p in sorted(folder.glob('*.npz'))]
                    if (folder / 'automatic-queries.json').is_file(): frozen_files.append('automatic-queries.json')
                frozen_files += [str(Path(path).relative_to(folder.relative_to(root)))
                    for path in binding.get('camera_artifact_sha256',{})]
                rows.append({'id': 'sparse-cache-' + hashlib.sha256(relative.encode()).hexdigest()[:16],
                    'label': folder.name, 'path': relative, 'track_id': protocol['track_id'],
                    'interaction_sha256': binding['sha256'], 'source_program_path': binding['path'],
                    'binding_artifact_sha256': binding['artifact_sha256'],
                    'status': viewer['status'], 'metrics': viewer['metrics'],
                    'verification': 'source_binding_and_snapshot_checked_pixel_hashes_rechecked_on_run',
                    'files': {name: sha(folder / name) for name in frozen_files}})
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return rows


def workflow_options(body, request, *, root):
    allowed = {'case_id', 'track_id', 'workflow', 'feature_provider', 'completion', 'reuse_sparse_id', 'camera_mode'}
    if set(body) - allowed or body.get('workflow') != VERSION:
        raise ValueError('Choose the explicit gated-local-v1 workflow and supported local options')
    provider = body.get('feature_provider', 'sift')
    completion = body.get('completion', 'none')
    if provider not in {'sift', 'cotracker', 'reuse_verified'} or completion not in {'none', 'smooth_gaps', 'moge'}:
        raise ValueError('Unsupported local feature provider or completion mode')
    options = {'workflow': VERSION, 'feature_provider': provider, 'completion': completion}
    camera_mode = body.get('camera_mode', 'assumed')
    if camera_mode not in {'assumed', 'moge_prior'}:
        raise ValueError('Choose assumed or local learned-camera mode')
    if camera_mode == 'moge_prior' and provider == 'reuse_verified':
        raise ValueError('A new camera assumption requires fresh reconstruction; saved geometry keeps its original camera')
    if 'camera_mode' in body:
        options['camera_mode'] = camera_mode
    cache_id = body.get('reuse_sparse_id')
    if provider == 'reuse_verified':
        if not isinstance(cache_id, str) or not CACHE_ID.fullmatch(cache_id):
            raise ValueError('Select an explicitly listed saved sparse result')
        rows = sparse_caches(root, interaction_sha256=request['interaction_sha256'], track_id=request['track_id'])
        selected = next((row for row in rows if row['id'] == cache_id), None)
        if not selected:
            raise ValueError('Saved sparse result is not bound to this registered component')
        options.update(reuse_sparse_id=cache_id, sparse_cache=selected)
    elif cache_id is not None:
        raise ValueError('A saved sparse identifier requires reuse_verified mode')
    return options


def sparse_gate(viewer):
    metrics = viewer.get('metrics', {})
    reasons = []
    if metrics.get('recovered_points', 0) < 40:
        reasons.append('Fewer than forty recovered source-correspondence points')
    if metrics.get('registered_train_views', 0) < 4:
        reasons.append('Fewer than four registered TRAIN views for a pair and two additional support views')
    if viewer.get('prior_geometry_or_depth_used') is not False:
        reasons.append('Observed-only sparse provenance is not established')
    if 'linear_or_degenerate' in viewer.get('status', ''):
        reasons.append('Recovered points are degenerate')
    return {'eligible': not reasons, 'reasons': reasons,
        'interpretation': 'Eligibility permits diagnostic stereo; it does not validate camera calibration or shape.'}


def capture_diagnostics(sparse, dense=None, validation=None):
    if not sparse:
        return [{'code': 'reconstruction_not_run', 'evidence': 'No sparse reconstruction result was produced.',
            'action': 'Inspect the recorded stage/runtime status before changing the capture.',
            'scope': 'deterministic_runtime_diagnostic_not_measured_capture_failure'}]
    metrics = sparse.get('metrics', {})
    rows = []
    if metrics.get('feature_tracks', 0) == 0:
        rows.append({'code': 'no_stable_tracks', 'evidence': 'No stable feature tracks were recovered.',
            'action': 'Record closer, sharp, overlapping views of one component; move slowly and keep its textured surface visible.'})
    if metrics.get('registered_train_views', 0) < metrics.get('training_views', 0):
        rows.append({'code': 'unregistered_views', 'evidence': 'Some source views could not be registered.',
            'action': 'Use continuous overlapping views with less abrupt motion and less hand occlusion.'})
    if metrics.get('heldout_views_passing', 0) < metrics.get('heldout_views', 0):
        rows.append({'code': 'limited_test_coverage', 'evidence': 'Some withheld views lack passing conditional projection checks.',
            'action': 'Capture wider overlapping viewpoints; inspect missing poses rather than treating missing errors as zero.'})
    if dense and not dense.get('surface'):
        rows.append({'code': 'no_supported_dense_patch', 'evidence': 'No supported observed surface patch was produced.',
            'action': 'Improve visible texture, focus and moderate viewpoint diversity; smooth reflective regions may remain unsupported.'})
    if validation is not None and validation.get('metrics', {}).get('eligible_photo_checks', 0) == 0:
        rows.append({'code': 'no_eligible_dense_test_checks',
            'evidence': 'The held-out diagnostic produced zero eligible dense photometric comparisons; no agreement fraction is available.',
            'action': 'Inspect skipped coverage/anchor/pose reasons and record more overlapping, textured viewpoints before interpreting the dense shape.'})
    rows.append({'code': 'uncalibrated', 'evidence': 'Camera intrinsics and physical scale remain unmeasured.',
        'action': 'Inspect automatic camera hypotheses on the existing footage. Visual reconstruction can continue while exact dimensions remain unknown.'})
    for row in rows:
        row['scope'] = 'deterministic_capture_suggestion_not_measured_failure_cause'
    return rows


class Workflow:
    def __init__(self, root, request, output):
        self.root = Path(root).resolve(); self.request = request; self.output = Path(output).resolve()
        if not self.output.is_relative_to(self.root / BASE) or self.output.exists():
            raise ValueError('Use a fresh observed workflow data directory')
        self.output.mkdir(parents=True)
        self.prefix = self.output.name.removesuffix('-workflow')
        self.manifest = {'format': 'observed-local-workflow', 'version': 1, 'id': self.output.name,
            'workflow': VERSION, 'request': request, 'status': 'running', 'created_at': now(),
            'stages': [], 'capture_diagnostics': [], 'exports': {}, 'result_url': None,
            'new_api_calls': 0, 'production_assets_changed': False, 'physics_ready': False,
            'implementation_sha256': sha(__file__)}
        write_once(self.output / 'contract.json', {'request': request, 'implementation_sha256': sha(__file__),
            'code_files': {name: sha(self.root / name) for name in ['observed_geometry.py', 'observed_stereo.py',
                'observed_surface_validation.py', 'observed_surface_completion.py', 'observed_depth_completion.py',
                'observed_surface_export.py', 'observed_completion_export.py']}})
        if request.get('camera_mode') == 'moge_prior':
            self.manifest['camera_mode'] = 'moge_prior'
            self.manifest['camera_is_unvalidated_learned_assumption'] = True
            write_once(self.output / 'camera-option.json', {'camera_mode': 'moge_prior',
                'implementation_sha256': sha(self.root / 'automatic_camera_prior.py'),
                'fresh_reconstruction_required': True, 'physical_scale_known': False})
        (self.output / 'implementation.py').write_bytes(Path(__file__).read_bytes())
        self.save()

    def relative(self, path):
        return Path(path).resolve().relative_to(self.root).as_posix()

    def url(self, path):
        return '/files/' + self.relative(path)

    def location(self, stage):
        return self.root / BASE / (self.prefix + '-' + stage)

    def save(self):
        from observed_jobs import _write
        self.manifest['updated_at'] = now()
        _write(self.output / 'workflow.json', self.manifest)
        print(json.dumps({'phase': 'workflow_progress', 'stage': self.manifest.get('phase'),
            'workflow_manifest_url': self.url(self.output / 'workflow.json')}), flush=True)

    def stage(self, name, callback=None, *, status=None, reason=None):
        started = time.monotonic()
        directory = self.output / 'stages' / name
        start = {'stage': name, 'started_at': now(), 'workflow_contract_sha256': sha(self.output / 'contract.json'),
                 'upstream_results_sha256': {row['stage']: sha(self.output / 'stages' / row['stage'] / 'result.json') for row in self.manifest['stages']},
                 'requested_options': {k: self.request.get(k) for k in ['feature_provider', 'completion', 'camera_mode']}}
        write_once(directory / 'start.json', start)
        self.manifest['phase'] = name; self.save()
        row = dict(start)
        try:
            payload = callback() if callback else {}
            row.update(status=status or 'completed', result=payload)
            if reason:
                row['reason'] = reason
        except Exception as error:
            row.update(status='unsupported' if isinstance(error, StageUnavailable) else 'failed',
                failure_type=type(error).__name__, reason=str(error) if isinstance(error, StageUnavailable) else
                'Stage failed without retry; saved partial artifacts remain preserved.')
            if isinstance(error,StageUnavailable) and error.result is not None:
                row['result']=error.result
            write_once(directory / 'failure.json', {'type': type(error).__name__, 'source_artifacts_preserved': True})
            print(json.dumps({'phase': 'stage_failed', 'stage': name, 'failure_type': type(error).__name__}), flush=True)
        artifacts = {}
        artifact_url_keys = {'viewer_url', 'report_url', 'glb_url', 'zip_url', 'metadata_url', 'readme_url',
                             'inferred_only_url', 'observed_version_url', 'inferred_version_url','camera_report_url'}
        def collect(value, key=None):
            if isinstance(value, str) and key in artifact_url_keys and value.startswith('/files/data/'):
                path = data_path(self.root, value.removeprefix('/files/'))
                if path.is_file():
                    artifacts[self.relative(path)] = sha(path)
            elif isinstance(value, dict):
                for child_key, item in value.items(): collect(item, child_key)
            elif isinstance(value, list):
                for item in value: collect(item)
        collect(row.get('result', {}))
        folders = [self.location(name)]
        if name == 'exports':
            folders = [self.location('export'), self.location('inferred-export')]
        for folder in folders:
            if folder.is_dir():
                for path in folder.iterdir():
                    if path.is_file() and not path.is_symlink():
                        artifacts[self.relative(path)] = sha(path)
        row['artifact_sha256'] = artifacts
        row.update(finished_at=now(), elapsed_s=round(time.monotonic() - started, 6))
        write_once(directory / 'result.json', row)
        self.manifest['stages'].append(row); self.save()
        return row

    def sparse(self):
        request = self.request
        if request['feature_provider'] == 'reuse_verified':
            row = request['sparse_cache']; folder = data_path(self.root, row['path'])
            for name, expected in row['files'].items():
                if sha(folder / name) != expected:
                    raise ValueError('Selected immutable sparse result changed')
            for path, expected in row.get('binding_artifact_sha256', {}).items():
                if sha(data_path(self.root, path)) != expected:
                    raise ValueError('Selected sparse source-binding record changed')
            protocol, viewer, binding = _candidate_binding(self.root, folder)
            if binding['sha256'] != request['interaction_sha256'] or protocol['track_id'] != request['track_id']:
                raise ValueError('Selected cache is not bound to registered component')
            for frame in protocol['input_views']:
                for field, hash_field in [('image_url', 'image_sha256'), ('mask_url', 'mask_sha256')]:
                    if sha(data_path(self.root, frame[field].removeprefix('/files/'))) != frame[hash_field]:
                        raise ValueError('Sparse source pixels changed')
            return folder, viewer
        folder = self.location('sparse')
        program = data_path(self.root, request['interaction_path'])
        if sha(program) != request['interaction_sha256']:
            raise ValueError('Registered program changed')
        camera_kwargs = {}
        if request.get('camera_mode') == 'moge_prior':
            import torch
            if not torch.backends.mps.is_available():
                raise StageUnavailable('Prepared local camera estimation is unavailable; no model fallback or camera substitution ran.')
            option = read(self.output / 'camera-option.json')
            if sha(self.root / 'automatic_camera_prior.py') != option['implementation_sha256']:
                raise ValueError('Camera implementation changed after workflow admission')
            from automatic_camera_prior import CameraPrior
            camera_kwargs['camera_provider'] = CameraPrior(self.root)
        try:
            if request['feature_provider'] == 'sift':
                import observed_geometry
                viewer = observed_geometry.run(program, request['track_id'], folder, root=self.root,
                    max_views=35, sensitivity=False, max_nfev=70, **camera_kwargs)
            else:
                import torch
                if not torch.backends.mps.is_available():
                    raise StageUnavailable('Prepared Apple MPS execution is unavailable in this process; no CoTracker inference or fallback ran.')
                import observed_cotracker
                viewer = observed_cotracker.run(program, request['track_id'], folder, root=self.root,
                    max_views=35, sensitivity=False, max_nfev=70, device='mps', **camera_kwargs)
                verify_correspondence_provider(folder, read(folder / 'protocol.json'))
        except ValueError as error:
            diagnostic=camera_disagreement_result(self.root,folder,error) if camera_kwargs else None
            if diagnostic is not None:
                raise StageUnavailable('The local camera model disagrees across the TRAIN frames; a shared lens assumption is unsupported. All estimates are preserved. No camera fallback or automatic retry ran.',result=diagnostic) from error
            raise
        return folder, viewer

    def completion(self, dense, mode, excluded):
        output = self.location('completion')
        if mode == 'moge':
            import torch
            if not torch.backends.mps.is_available():
                raise StageUnavailable('Prepared Apple MPS execution is unavailable in this process; no MoGe inference or fallback ran. The observed surface remains available.')
            import observed_depth_completion as engine
            engine.run(self.root, self.relative(dense), self.relative(output), excluded_mask_urls=excluded)
        else:
            import observed_surface_completion as engine
            engine.complete(self.root, self.relative(dense), self.relative(output), excluded_mask_urls=excluded)
        # The legacy completion modules label every target camera_body. Preserve
        # their raw records and bind metadata to the actual registered component.
        viewer_path = output / 'viewer.json'
        viewer = read(viewer_path)
        raw = viewer_path.read_bytes()
        write_once(output / 'component-binding.json', {'version': 'registered-component-binding-v1',
            'registered_case_id': self.request['case_id'], 'component_id': self.request['track_id'],
            'original_component_label': viewer.get('component_id'), 'original_viewer_sha256': hashlib.sha256(raw).hexdigest(),
            'geometry_changed': False, 'depth_or_pose_changed': False, 'human_annotation': False,
            'binding_source': self.request['interaction_path'], 'binding_source_sha256': self.request['interaction_sha256']})
        (output / 'viewer-legacy-label.json').write_bytes(raw)
        viewer['component_id'] = self.request['track_id']
        if viewer.get('inferred_surface'):
            viewer['inferred_surface']['component_id'] = self.request['track_id']
        viewer['component_binding_url'] = self.url(output / 'component-binding.json')
        viewer_path.write_bytes(json.dumps(viewer, separators=(',', ':'), allow_nan=False).encode() + b'\n')
        return output, viewer

    def version_record(self, path, sparse_folder, dense_folder, export, validation, *, inferred=False, completion=None):
        # Use the exact sparse input contract already verified by the dense stage.
        protocol = read(sparse_folder / 'protocol.json')
        frames = protocol.get('input_views', [])
        artifacts = {'glb': {'path': export['glb_url'].removeprefix('/files/'), 'sha256': export['glb_sha256'], 'mime': 'model/gltf-binary'},
            'quality': {'path': self.relative(validation / 'report.json'), 'sha256': sha(validation / 'report.json'), 'mime': 'application/json'}}
        record = {'format': 'interaction-reconstruction-version', 'version': 1,
            'version_id': self.prefix + ('-inferred-v1' if inferred else '-observed-v1'),
            'label': self.request['label'] + (' · inferred local completion' if inferred else ' · observed source surface'),
            'case_id': self.request['case_id'], 'interaction_path': self.request['interaction_path'],
            'interaction_sha256': self.request['interaction_sha256'], 'source_video_sha256': protocol['source']['sha256'],
            'component_id': self.request['track_id'], 'kind': 'inferred' if inferred else 'observed',
            'artifacts': artifacts, 'source_evidence': [{'id': frame['id'], 'timestamp_s': frame['timestamp_s'],
                'image_path': frame['image_url'].removeprefix('/files/'), 'image_sha256': frame['image_sha256'],
                'mask_path': frame['mask_url'].removeprefix('/files/'), 'mask_sha256': frame['mask_sha256'],
                'source_video_sha256': protocol['source']['sha256']} for frame in frames],
            'gauge': {'metric': False, 'units': 'arbitrary'}, 'acceptance': 'unreviewed',
            'physical_confidence': 'unknown', 'physics_ready': False, 'complete_object': False,
            'physical_accuracy_verified': False, 'simulation_ready': False,
            'workflow_manifest_path': self.relative(self.output / 'result.json'),
            'workflow_contract_sha256': sha(self.output / 'contract.json'),
            'sparse_protocol_path': self.relative(sparse_folder / 'protocol.json'),
            'sparse_protocol_sha256': sha(sparse_folder / 'protocol.json'),
            'surface_protocol_path': self.relative(dense_folder / 'protocol.json'),
            'surface_protocol_sha256': sha(dense_folder / 'protocol.json'),
            'limitations': ['Diagnostic reconstruction only; no measured shape, intrinsics, scale, joints or contact.',
                'Source images and masks remain fallible observations; photo agreement is not physical validation.']}
        if inferred:
            artifacts['combined_preview'] = artifacts['glb']
            artifacts['glb'] = {'path': export['inferred_only_url'].removeprefix('/files/'),
                'sha256': export['inferred_only_sha256'], 'mime': 'model/gltf-binary'}
            record.update(observed_version_id=self.prefix + '-observed-v1', all_faces_inferred=True,
                contains_observed_layer=False, inferred_node_name='Inferred depth prior' if self.request['completion'] == 'moge' else 'Inferred smooth gaps',
                inference_metadata_path=self.relative(completion / 'protocol.json'),
                component_binding_path=self.relative(completion / 'component-binding.json'))
        write_once(path, record)
        return self.url(path)

    def run(self):
        from observed_stereo import run as dense_run, derive_display_review
        from observed_surface_validation import run as validate
        sparse_folder = sparse_viewer = dense_folder = dense_viewer = validation_folder = validation_report = completion_folder = completion_viewer = None
        def sparse_stage():
            nonlocal sparse_folder, sparse_viewer
            sparse_folder, sparse_viewer = self.sparse()
            self.manifest.update(result_url=self.url(sparse_folder / 'viewer.json'), result_kind='observed_sparse_geometry',
                result_status=sparse_viewer['status'], metrics=sparse_viewer['metrics'])
            return {'viewer_url': self.url(sparse_folder / 'viewer.json'), 'status': sparse_viewer['status'],
                'metrics': sparse_viewer['metrics'], 'gate': sparse_gate(sparse_viewer),
                'reused': self.request['feature_provider'] == 'reuse_verified',
                **({'camera': {'report_url': '/files/' + sparse_viewer['camera_assumption']['report_path'],
                     'kind': sparse_viewer['camera_assumption']['kind'],
                     'camera_calibrated': False, 'physical_scale_known': False}}
                   if sparse_viewer.get('camera_assumption') else {})}
        first = self.stage('sparse', sparse_stage, status='reused' if self.request['feature_provider'] == 'reuse_verified' else None)
        gate = sparse_gate(sparse_viewer) if sparse_viewer else {'eligible': False, 'reasons': [first.get('reason', 'Sparse stage did not produce geometry')]}
        if first['status'] == 'failed' or not gate['eligible']:
            for stage in STAGES[1:]:
                self.stage(stage, status='blocked', reason='; '.join(gate['reasons']))
            self.manifest['capture_diagnostics'] = capture_diagnostics(sparse_viewer or {})
            return self.finish('unsupported' if first['status'] != 'failed' else 'failed')
        def dense_stage():
            nonlocal dense_folder, dense_viewer
            dense_folder = self.location('dense')
            dense_viewer = dense_run(sparse_folder, dense_folder, root=self.root)
            if dense_viewer.get('surface'):
                derive_display_review(dense_folder, root=self.root)
                self.manifest.update(result_url=self.url(dense_folder / 'viewer-display.json'), result_kind='observed_surface',
                    result_status=dense_viewer['status'], metrics=dense_viewer['metrics'])
            return {'viewer_url': self.url(dense_folder / ('viewer-display.json' if dense_viewer.get('surface') else 'viewer.json')),
                'status': dense_viewer['status'], 'metrics': dense_viewer['metrics'], 'surface_present': bool(dense_viewer.get('surface'))}
        second = self.stage('dense', dense_stage)
        if second['status'] == 'failed' or not dense_viewer or not dense_viewer.get('surface'):
            for stage in STAGES[2:]:
                self.stage(stage, status='blocked', reason='No supported source-derived dense surface exists')
            self.manifest['capture_diagnostics'] = capture_diagnostics(sparse_viewer, dense_viewer)
            return self.finish('unsupported' if second['status'] != 'failed' else 'failed')
        def validation_stage():
            nonlocal validation_folder, validation_report
            validation_folder = self.location('validation')
            report = validate(dense_folder, validation_folder, root=self.root)
            validation_report = report
            return {'report_url': self.url(validation_folder / 'report.json'), 'metrics': report['metrics'],
                    'status': 'diagnostic_heldout_photometric_check' if report['metrics']['eligible_photo_checks'] else 'no_eligible_heldout_photo_checks',
                    'allows_optional_completion_hypothesis': report['metrics']['eligible_photo_checks'] > 0,
                'physical_geometry_validated': False, 'scope': 'conditional_heldout_photometric_diagnostic'}
        third = self.stage('validation', validation_stage)
        if self.request['completion'] == 'none':
            self.stage('completion', status='not_requested', reason='Optional inference was not requested')
        elif third['status'] == 'failed':
            self.stage('completion', status='blocked', reason='Conditional validation report could not be produced')
        elif not third['result']['allows_optional_completion_hypothesis']:
            self.stage('completion', status='blocked', reason='Zero eligible held-out dense photo checks; optional completion is unsupported for this observed patch. The observed diagnostic export remains available.')
        else:
            def completion_stage():
                nonlocal completion_folder, completion_viewer
                source_frame = dense_viewer['surface']['source_frame_id']
                policy = read(dense_folder / 'protocol.json')
                source_row = next(frame for frame in policy['input_views'] if frame['id'] == source_frame)
                excluded = ['/files/' + row['mask_path'] for row in source_row.get('subtracted_components', [])]
                completion_folder, completion_viewer = self.completion(dense_folder, self.request['completion'], excluded)
                return {'viewer_url': self.url(completion_folder / 'viewer.json'), 'status': completion_viewer['status'],
                    'metrics': completion_viewer['metrics'], 'inferred_surface_present': bool(completion_viewer.get('inferred_surface')),
                    'all_faces_inferred': True, 'physical_confidence': 'unknown'}
            self.stage('completion', completion_stage)
        def export_stage():
            import observed_surface_export
            import observed_completion_export
            exported = observed_surface_export.export_run(self.root, self.relative(dense_folder), self.location('export'))
            result = {'observed': exported}
            if validation_folder and (validation_folder / 'report.json').is_file():
                result['observed_version_url'] = self.version_record(self.output / 'reconstruction-version.json', sparse_folder,
                    dense_folder, exported, validation_folder)
            # Preserve the successful observed download even if the optional
            # separate inferred-layer export subsequently fails.
            self.manifest['exports'] = dict(result)
            self.save()
            if completion_viewer and completion_viewer.get('inferred_surface'):
                inferred = observed_completion_export.export_completion(self.root, self.relative(completion_folder), self.location('inferred-export'),
                    observed_source=self.relative(dense_folder), observed_export_path=self.relative(self.location('export')))
                result['inferred'] = inferred
                if validation_folder and (validation_folder / 'report.json').is_file():
                    result['inferred_version_url'] = self.version_record(self.output / 'inferred-version.json', sparse_folder,
                        dense_folder, inferred, validation_folder, inferred=True, completion=completion_folder)
            self.manifest['exports'] = result
            return result
        self.stage('exports', export_stage)
        self.manifest['capture_diagnostics'] = capture_diagnostics(sparse_viewer, dense_viewer, validation_report)
        return self.finish('completed_with_stage_failures' if any(row['status'] == 'failed' for row in self.manifest['stages']) else
            'completed_with_unsupported_stages' if any(row['status'] in {'unsupported', 'blocked'} for row in self.manifest['stages']) else 'completed')

    def finish(self, status):
        self.manifest.update(status=status, phase='complete', finished_at=now())
        self.save()
        write_once(self.output / 'result.json', self.manifest)
        print(json.dumps({'phase': 'workflow_complete', 'status': status,
            'workflow_manifest_url': self.url(self.output / 'workflow.json'), 'result_url': self.manifest.get('result_url')}), flush=True)
        return self.manifest


def run(root, request, output):
    # Revalidate the original publication and every browser-controlled option.
    from observed_jobs import canonical_request
    body = {key: request[key] for key in ['case_id', 'track_id', 'workflow', 'feature_provider', 'completion']}
    if 'camera_mode' in request:
        body['camera_mode'] = request['camera_mode']
    if request.get('reuse_sparse_id'):
        body['reuse_sparse_id'] = request['reuse_sparse_id']
    checked = canonical_request(body, root=root)
    if checked != request:
        raise ValueError('Registered workflow contract changed before execution')
    return Workflow(root, checked, output).run()
