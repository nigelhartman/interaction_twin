"""Public-only robot kit/task transport and deterministic in-memory starter ZIP.

No model, scheduler, filesystem write, private catalog discovery or robot
execution is available. Canonical task interpretation belongs to robot_task_program.
"""
from collections import OrderedDict
import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import threading
import zipfile

from curated_twin_catalog import CuratedTwinCatalog

ROOT=Path(__file__).resolve().parent
MAX_PROGRAM=256*1024
MAX_BUNDLE=96*1024*1024
ID=re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z')
FIXED_FILES={
 'controller':('showcase/robot-controller.js','text/javascript'),
 'task-controller':('showcase/robot-task-plan.js','text/javascript'),
 'recorder':('showcase/robot-recorder.js','text/javascript'),
 'workbench':('showcase/robot-workbench.html','text/html'),
 'workbench-runtime':('showcase/robot-workbench.js','text/javascript'),
 'workbench-style':('showcase/robot-workbench.css','text/css'),
 'workbench-navigation':('showcase/robot-workbench-ui-v2.js','text/javascript'),
 'scene-contract':('showcase/aligned-scene-contract.js','text/javascript'),
 'mark':('showcase/mark.svg','image/svg+xml'),
 'brand-mark-v2':('showcase/brand/v2/mark.svg','image/svg+xml'),
 'three':('workspace/robot-vendor/three.module.js','text/javascript'),
 'three-core':('workspace/robot-vendor/three.core.js','text/javascript'),
 'orbit-controls':('workspace/robot-vendor/OrbitControls.js','text/javascript'),
 'gltf-loader':('workspace/robot-vendor/GLTFLoader.js','text/javascript'),
 'geometry-utils':('workspace/robot-vendor/BufferGeometryUtils.js','text/javascript'),
 'three-license':('workspace/robot-vendor/LICENSE','text/plain'),
 'starter-server':('robot_starter/preview_server.py','text/x-python'),
}


def encoded(value):
    try:raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
    except (TypeError,ValueError,RecursionError):raise ValueError('Finite bounded JSON object required') from None
    if len(raw)>MAX_PROGRAM:raise ValueError('Robot task/kit exceeds its bounded JSON transport size')
    return raw


def descriptor(raw,uri,*,mime='application/json',url=None):
    return {'uri':uri,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'mime_type':mime,**({'url':url}if url else{})}


class RobotTaskCatalog:
    """The robotics surface always uses the curated snapshot, including stdio."""
    def __init__(self,root=ROOT):
        self.root=Path(root).resolve();self.catalog=CuratedTwinCatalog(self.root)
        self._bundles=OrderedDict();self._lock=threading.RLock();self._build_lock=threading.Lock()

    def _canonical_kit(self,kit_id=None):
        import robot_task_program as R
        if kit_id is not None and(not isinstance(kit_id,str)or not ID.fullmatch(kit_id)):raise ValueError('Invalid declared robot kit ID')
        return R.get_robot_kit(root=self.root,**({'kit_id':kit_id}if kit_id is not None else{}))

    def _fixed(self,artifact_id):
        if artifact_id not in FIXED_FILES:raise ValueError('Unknown declared robot kit artifact')
        name,mime=FIXED_FILES[artifact_id];path=self.root/name
        if path.resolve()!=path or any(p.is_symlink()for p in [path,*path.parents])or not path.is_file()or not 0<path.stat().st_size<=5*1024*1024:
            raise ValueError('Declared robot starter file is unavailable')
        raw=path.read_bytes()
        if len(raw)>5*1024*1024:raise ValueError('Robot starter file changed beyond its size bound')
        return name,mime,raw

    def list_kits(self):
        import robot_task_program as R
        return R.list_robot_kits(root=self.root)

    def get_kit(self,kit_id=None):
        import robot_task_program as R
        kit=self._canonical_kit(kit_id);identifier=kit_id or R.KIT_ID
        raw=encoded(kit);resources={'kit':descriptor(raw,f'twin://robot-kits/{identifier}/artifacts/kit',url='/api/robotics/kits/'+identifier+'/artifacts/kit')}
        resources['schema']=descriptor(encoded(R.task_schema()),f'twin://robot-kits/{identifier}/artifacts/schema',url='/api/robotics/kits/'+identifier+'/artifacts/schema')
        missing=[]
        for artifact_id in FIXED_FILES:
            try:name,mime,body=self._fixed(artifact_id)
            except ValueError:missing.append(artifact_id);continue
            resources[artifact_id]=descriptor(body,f'twin://robot-kits/{identifier}/artifacts/{artifact_id}',mime=mime,url='/api/robotics/kits/'+identifier+'/artifacts/'+artifact_id)
        return {'kit':kit,'resources':resources,'starter_files_complete':not missing,'missing_starter_files':missing,
                'scope':'authored_kinematic_browser_kit','read_only':True,'api_calls':0,'robot_execution_available':False}

    def kit_artifact(self,kit_id,artifact_id):
        self._canonical_kit(kit_id)
        if artifact_id=='kit':return encoded(self._canonical_kit(kit_id))
        if artifact_id=='schema':
            import robot_task_program as R
            return encoded(R.task_schema())
        return self._fixed(artifact_id)[2]

    def plan(self,twin_id,instruction,kit_id=None):
        import robot_task_program as R
        if not isinstance(instruction,str)or not 1<=len(instruction.strip())<=2000:raise ValueError('A bounded robot task instruction is required')
        value=R.plan_robot_task(twin_id,instruction,root=self.root,catalog=self.catalog,**({'kit_id':kit_id}if kit_id is not None else{}))
        encoded(value)
        return value

    def validate(self,task_program):
        import robot_task_program as R
        if not isinstance(task_program,dict):raise ValueError('A typed robot task object is required')
        encoded(task_program)
        return R.validate_robot_task(task_program,root=self.root,catalog=self.catalog)

    def _build_bundle(self,task_program):
        import robot_task_program as R
        validation=self.validate(task_program)
        if validation.get('valid')is not True or task_program.get('status')!='ready':raise ValueError('Only an exact valid ready task can be packaged; missing/unsupported requirements must be resolved first')
        kit=self._canonical_kit(task_program['kit_id']);files={}
        for artifact_id in FIXED_FILES:
            name,_,raw=self._fixed(artifact_id)
            files['preview_server.py'if artifact_id=='starter-server'else name]=raw
        for reference in R.task_bindings(task_program):
            # Every source/mesh byte is gated by the CURRENT exact curated
            # allowlist, never a working-catalog path or a planner supplied URL.
            raw=self.catalog._bytes(reference,maximum=16*1024*1024)
            name=reference['path']
            if name in files and files[name]!=raw:raise ValueError('Conflicting declared bundle paths')
            files[name]=raw
        files['task-program.json']=encoded(task_program);files['robot-kit.json']=encoded(kit);files['robot-task.schema.json']=encoded(R.task_schema())
        kit_response=self.get_kit(task_program['kit_id'])
        for artifact_id,ref in kit_response['resources'].items():
            if artifact_id=='kit':name='robot-kit.json'
            elif artifact_id=='schema':name='robot-task.schema.json'
            else:
                name=FIXED_FILES[artifact_id][0];name='preview_server.py'if artifact_id=='starter-server'else name
            if len(files[name])!=ref['bytes']or hashlib.sha256(files[name]).hexdigest()!=ref['sha256']:
                raise ValueError('Starter code changed during package preparation; retry a fresh export')
            # This envelope travels with the ZIP, so its HTTP URLs must name
            # actual included files rather than the live application's routes.
            # Preserve the original MCP URI solely as explicit provenance.
            ref['url']='/'+name
        kit_response['resource_url_scope']='packaged_manifest_files'
        kit_response['mcp_resources_available']=False
        kit_response['mcp_uri_provenance']='Original live catalog resource identifiers; the standalone server does not implement MCP.'
        files['kit-response.json']=encoded(kit_response)
        files['README.txt']=(
            'INTERACTION TWIN — AUTHORED ROBOT TASK STARTER\n\n'
            'Extract completely. Run: python3 -I -S preview_server.py --port 0\n'
            'Open the printed localhost URL; Python3.10+ and no packages or internet are required.\n'
            'The browser uses exact included native models and an authored joint-limited kinematic robot.\n'
            'Use the browser JSON editor for task variations; the included source/kit bindings stay fixed.\n'
            'On-disk files are integrity-bound. A developer fork needs a fresh manifest.\n'
            'For agent edits: validate_robot_task(edited_task), then export_robot_task_bundle(edited_task).\n'
            'Read the new declared MCP ZIP resource and verify its hash before extraction.\n'
            'No physical scale, collision/contact validation,\n'
            'robot training, hardware control, hosted model calls or approved execution is included.\n'
            'Original source/program provenance pointers are retained; only files in bundle-manifest.json\n'
            'are included. No owner login, uploads, private feedback or credentials are packaged.\n'
            'Keep the vendored Three.js LICENSE with its source. Original workbench code is supplied\n'
            'as a reusable project starter. Nothing is deployed by this archive.\n').encode()
        total=sum(map(len,files.values()))
        if len(files)>96 or total>MAX_BUNDLE:raise ValueError('Robot starter exceeds its bounded package size')
        manifest={'format':'interaction-twin-robot-starter-bundle','version':1,'task_program_sha256':hashlib.sha256(files['task-program.json']).hexdigest(),
            'kit_id':task_program['kit_id'],'curated_twin_id':task_program['twin_id'],
            'files':[{'path':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}for name,raw in sorted(files.items())],
            'physical_accuracy_verified':False,'simulation_ready':False,'hardware_execution_authorized':False,'private_data_included':False,'inference_available':False}
        files['bundle-manifest.json']=encoded(manifest)
        stream=BytesIO()
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3)as archive:
            for name,raw in sorted(files.items()):
                info=zipfile.ZipInfo('Interaction-Twin-Robot-Starter/'+name,date_time=(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
                archive.writestr(info,raw,compress_type=zipfile.ZIP_DEFLATED,compresslevel=3)
        body=stream.getvalue()
        if len(body)>MAX_BUNDLE:raise ValueError('Compressed robot starter exceeds its bounded package size')
        return body,manifest

    def export_bundle(self,task_program):
        raw=encoded(task_program);snapshot=json.loads(raw)
        with self._build_lock:body,manifest=self._build_bundle(snapshot)
        identifier=hashlib.sha256(body).hexdigest()
        with self._lock:
            self._bundles[identifier]={'program':json.loads(raw),'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body)};self._bundles.move_to_end(identifier)
            while len(self._bundles)>8:self._bundles.popitem(last=False)
        return {'bundle_id':identifier,'artifact':descriptor(body,'twin://robot-task-bundles/'+identifier,mime='application/zip',url='/api/robotics/bundles/'+identifier),
            'manifest':manifest,'read_only':True,'new_api_calls':0,'new_gpu_calls':0,'task_program_unchanged':True,'task_program_sha256':hashlib.sha256(raw).hexdigest(),'scope':'current_curated_assets_and_authored_kit_only'}

    def bundle_artifact(self,bundle_id):
        if not isinstance(bundle_id,str)or not re.fullmatch('[a-f0-9]{64}',bundle_id):raise ValueError('Exact robot bundle ID required')
        with self._lock:
            if bundle_id not in self._bundles:raise ValueError('Bundle descriptor expired; export the same task again')
            saved=copy.deepcopy(self._bundles[bundle_id])
        with self._build_lock:body,_=self._build_bundle(saved['program'])
        if len(body)!=saved['bytes']or hashlib.sha256(body).hexdigest()!=saved['sha256']:
            raise ValueError('Starter dependencies changed; export the task again for an exact fresh descriptor')
        return body
