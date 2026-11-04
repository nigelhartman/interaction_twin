"""Read-only source-bound chunks from an already completed V1 GPU attempt.

No dispatch, retry, allocation command, credential, cache or file write exists
here. The client first obtains a complete SHA-pinned inventory from V1, then
supplies its exact completion/native-metadata refs. Those small proof documents
are re-hashed for every chunk; full large-file integrity remains checked at the
client before any native import. File sizes/stat values are never integrity proof.
"""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import re
import sys

TASK=Path('/mnt/vast-kisski/home/nigel.hartman01/u29467/.local/state/ai-training/sam3d-20261101-v1')
NATIVE={'metadata.json','object.glb','object-splat.ply','layout-and-pointmap.npz','inputs/manifest.json',
        'inputs/image-source.png','inputs/mask-source.png','inputs/rgb.png','inputs/mask.png'}
EVIDENCE={'launch.json','supervisor-result.json','prepare.json','plan.json','command-spec.json','execution.json',
          'offline-preflight.json','offline-run.json','offline-manifest.json'}
PROOFS={'evidence/launch.json','evidence/supervisor-result.json','evidence/execution.json','native/metadata.json'}
MAX_FILE=256*2**20
MAX_CHUNK=4*2**20
SHA=re.compile(r'[a-f0-9]{64}\Z')
JOB=re.compile(r'upload-[a-z0-9-]{1,64}\Z')


def require(value,message):
    if not value:raise ValueError(message)


def source_files(identifier):
    require(isinstance(identifier,str)and JOB.fullmatch(identifier),'ExactExistingUploadJobRequired')
    folder=TASK/'upload-bridge-jobs'/identifier;output=TASK/'outputs'/identifier
    files={'native/'+name:output/name for name in NATIVE}
    for name in EVIDENCE:
        if name in {'launch.json','supervisor-result.json'}:path=folder/name
        elif name=='offline-manifest.json':path=TASK/('offline-job-'+identifier+'.json')
        else:path=TASK/'component-jobs'/identifier/name
        files['evidence/'+name]=path
    return files


def checked_reference(value,files,*,maximum=MAX_FILE,read=False):
    require(isinstance(value,dict)and set(value)=={'path','sha256','bytes'},'ExactInventoryReferenceRequired')
    require(value.get('path')in files and isinstance(value.get('sha256'),str)and SHA.fullmatch(value['sha256'])and
            type(value.get('bytes'))is int and 0<value['bytes']<=maximum,'BoundedWhitelistedReferenceRequired')
    path=files[value['path']]
    require(path.resolve()==path and path.is_relative_to(TASK)and not any(p.is_symlink()for p in [path,*path.parents])and
            path.is_file()and path.stat().st_size==value['bytes'],'OrdinaryCompletedFileLengthDiffers')
    if not read:return path
    raw=path.read_bytes()
    require(len(raw)==value['bytes']and hashlib.sha256(raw).hexdigest()==value['sha256'],'CompletionProofHashChanged')
    value=json.loads(raw)
    require(isinstance(value,dict),'CompletionProofObjectRequired')
    return value


def completion(request,files):
    refs=request.get('completion_refs')
    require(isinstance(refs,list)and len(refs)==len(PROOFS)and {r.get('path')for r in refs}==PROOFS,'ExactCompletionProofInventoryRequired')
    docs={r['path']:checked_reference(r,files,maximum=256*1024,read=True)for r in refs}
    launch,done,execution,metadata=[docs[k]for k in ['evidence/launch.json','evidence/supervisor-result.json','evidence/execution.json','native/metadata.json']]
    identifier=request['job_id'];helper=request.get('generation_helper_sha256')
    require(isinstance(helper,str)and SHA.fullmatch(helper)and launch.get('job_id')==identifier and
            launch.get('format')=='sam3d-upload-bridge-launch'and launch.get('helper_sha256')==helper and
            launch.get('maximum_model_commands')==1 and launch.get('automatic_retry')is False and
            launch.get('parent_allocation_modified')is False,'OriginalGenerationLaunchProofDiffers')
    require(done.get('job_id')==identifier and done.get('returncode')==0 and done.get('automatic_retry')is False,
            'OriginalSupervisorDidNotComplete')
    require(execution.get('job_id')==identifier and execution.get('status')=='native_output_ready_for_independent_import_review'and
            execution.get('model_commands_attempted')==1 and execution.get('model_inference_executed')is True and
            execution.get('automatic_retry')is False and execution.get('parent_allocation_modified')is False,
            'ExactlyOneCompletedOriginalModelAttemptRequired')
    metadata_ref=next(r for r in refs if r['path']=='native/metadata.json')
    require(execution.get('output_metadata_sha256')==metadata_ref['sha256'],'OriginalNativeMetadataExecutionBindingDiffers')
    require(metadata.get('learned_inference_executed')is True and metadata.get('status')=='candidate_generated_pending_visual_review',
            'OriginalNativeMetadataNotComplete')
    return metadata


def expected_native(reference,metadata):
    name=reference['path']
    if not name.startswith('native/'):return
    relative=name.removeprefix('native/')
    if relative=='metadata.json':return
    outputs={r['file']:r for r in metadata.get('outputs',{}).get('files',[])}
    if relative in {'object.glb','object-splat.ply','layout-and-pointmap.npz'}:
        expected=outputs.get(relative,{})
        require(expected.get('sha256')==reference['sha256']and expected.get('bytes')==reference['bytes'],
                'OriginalModelOutputHashDiffers')
    elif relative=='inputs/manifest.json':
        require(metadata['inputs'].get('manifest_sha256')==reference['sha256'],'OriginalInputManifestHashDiffers')
    elif relative in {'inputs/image-source.png','inputs/mask-source.png'}:
        key='image'if relative=='inputs/image-source.png'else'mask';expected=metadata['inputs']['files'][key]
        require(expected['sha256']==reference['sha256']and expected['bytes']==reference['bytes'],'OriginalSourceInputHashDiffers')
    else:
        require(metadata['model_input_sha256'][Path(relative).name]==reference['sha256'],'OriginalEncodedModelInputHashDiffers')


def serve(request):
    require(isinstance(request,dict)and set(request)=={'action','job_id','path','offset','file_ref','completion_refs',
        'generation_helper_sha256','inventory_sha256'},'ExactReadOnlyChunkProtocolRequired')
    require(request.get('action')=='chunk_v2'and isinstance(request.get('inventory_sha256'),str)and SHA.fullmatch(request['inventory_sha256']),
            'ReadOnlyCompletedChunkActionRequired')
    files=source_files(request['job_id']);metadata=completion(request,files)
    reference=request['file_ref'];path=checked_reference(reference,files)
    require(request.get('path')==reference['path']and type(request.get('offset'))is int and 0<=request['offset']<reference['bytes'],
            'OriginalInventoryPathOffsetDiffers')
    expected_native(reference,metadata)
    with path.open('rb')as stream:
        stream.seek(request['offset']);raw=stream.read(min(MAX_CHUNK,reference['bytes']-request['offset']))
    require(len(raw)==min(MAX_CHUNK,reference['bytes']-request['offset']),'CompletedChunkLengthChanged')
    return {'path':reference['path'],'offset':request['offset'],'base64':base64.b64encode(raw).decode(),
        'chunk_sha256':hashlib.sha256(raw).hexdigest(),'file_sha256':reference['sha256'],'inventory_sha256':request['inventory_sha256'],
        'protocol':'sam3d-source-bound-transfer-v2','read_only':True,'large_file_hash_not_repeated':True}


if __name__=='__main__':
    try:
        raw=sys.stdin.buffer.read(32769);require(len(raw)<=32768,'BoundedTransferRequestRequired')
        print(json.dumps(serve(json.loads(raw)),allow_nan=False))
    except Exception as error:
        print(json.dumps({'status':'transfer_rejected','error_type':type(error).__name__,'reason':str(error)[:200]}))
        sys.exit(1)
