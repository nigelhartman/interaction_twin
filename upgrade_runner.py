"""Local, hash-guarded geometry upgrades using unchanged cached reasoning.

This orchestrator never calls a reasoning provider, downloads weights or contacts
a cluster. Descriptive media/procedure text is data and never executed.
"""
from __future__ import annotations

import copy
import ast
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import threading
import time
import uuid

from twin_catalog import TwinCatalog

ROOT = Path(__file__).resolve().parent
LOADED_SOURCE = Path(__file__).read_bytes()
LOADED_SOURCE_SHA256 = hashlib.sha256(LOADED_SOURCE).hexdigest()
MODES = {'refit_existing', 'refit_native_cpu', 'local_triposr', 'imported_sam3d'}
RUN_ID = re.compile(r'upgrade-[a-zA-Z0-9_-]{1,80}')
STAGES = [('cached_inputs', 'Verify saved reasoning and masks'),
          ('reconstruct', 'Prepare component geometry'), ('pose', 'Fit the reference view'),
          ('motion', 'Track conditional component motion'),
          ('procedure', 'Recompile the unchanged procedure'),
          ('comparison', 'Compare matching source frames'), ('publish', 'Open the updated twin')]


def stage_source_identities(source=LOADED_SOURCE):
    """Hash frozen executable source per stage, without presentation files.

    A comparison implementation edit must invalidate comparison; an execution
    or phase/cache-contract edit invalidates every worker. A viewer/CSS edit
    never enters these identities. The complete original source is separately
    frozen in the run for audit, including code outside these stage scopes.
    """
    text=source.decode('utf-8'); tree=ast.parse(text)
    functions={node.name:ast.get_source_segment(text,node) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    runner=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='UpgradeRunner')
    methods={node.name:ast.get_source_segment(text,node) for node in runner.body if isinstance(node,ast.FunctionDef)}
    execute=next(node for node in runner.body if isinstance(node,ast.FunctionDef) and node.name=='_execute')
    callbacks={node.name:ast.get_source_segment(text,node) for node in execute.body if isinstance(node,ast.FunctionDef)}
    # Include the orchestration statements for the specific self.phase call,
    # and dependency/result setup outside other stage callback bodies.
    orchestration='\n'.join(ast.get_source_segment(text,node) for node in execute.body if not isinstance(node,ast.FunctionDef))
    common='\n'.join([functions[name] for name in ['now','read','sha','fingerprint','write','data_path','file_receipts','verify_receipts','request_value','stage_source_identities']]
                     +[methods[name] for name in ['__init__','save','cancel','_terminate','command','phase','execute']])
    callback_names={'cached_inputs':'cached','reconstruct':'reconstruct','pose':'pose','motion':'motion','procedure':'procedure','comparison':'comparison','publish':'publish'}
    result={}
    for key,_ in STAGES:
        parts=[common,orchestration,callbacks[callback_names[key]],callbacks['cli']]
        if key=='cached_inputs': parts += [functions['_context'],functions['plan']]
        if key=='comparison': parts += [functions['compare_motion']]
        if key=='motion': parts += [functions['cotracker_runtime_identity']]
        if key in {'pose','motion'}: parts += [methods['_adapt_backend'],functions['alignment_configuration']]
        result[key]=hashlib.sha256('\n'.join(parts).encode()).hexdigest()
    return result


def alignment_configuration(metadata, imported=None):
    """Choose a versioned fitter without changing an existing vertex baseline.

    A resolved historical import may advertise a freshly validated capability
    separately from its immutable import-time metadata.
    """
    declared=(imported or {}).get('alignment_adapter') or metadata.get('fit_contract',{}).get('alignment_adapter')
    if declared not in (None,'legacy-vertex-colour-v1','native-base-color-v1'):
        raise ValueError('Unsupported native alignment adapter contract')
    native=declared=='native-base-color-v1'
    pose='stage4_native_pose.py' if native else 'stage4_generated_pose.py'
    motion='stage4_native_track.py' if native else 'stage4_track_generated.py'
    shared=['alignment_fit.py','alignment_render.py']
    if native: shared += ['native_appearance.py','native_alignment.py','portable_twin.py','stage5.py']
    return {'native':native,'adapter':declared or 'legacy-vertex-colour-v1',
            'pose_script':pose,'motion_script':motion,
            'pose_dependencies':['scripts/'+pose,*shared,*(['scripts/stage4_generated_pose.py'] if native else [])],
            'motion_dependencies':['scripts/'+motion,*shared,*(['scripts/stage4_track_generated.py'] if native else [])]}


def cotracker_runtime_identity(root=ROOT):
    """Read fixed installed runtime/code identities; perform no inference.

    Source files are restricted to the prepared CoTracker package, plus pinned
    package metadata and the main PyTorch extension. No environment/credential
    files, network, checkpoints or model execution enter this read-only probe.
    """
    python=Path(root)/'.venv-sam/bin/python'
    probe=r'''
import hashlib,json,sys
from pathlib import Path
from importlib.metadata import distribution
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
cot=distribution('cotracker'); package=Path(cot.locate_file('cotracker')).resolve()
if not package.is_dir(): raise ValueError('Prepared CoTracker source package unavailable')
code=[{'file':str(p.relative_to(package)),'sha256':sha(p)} for p in sorted(package.rglob('*.py'))]
if not code: raise ValueError('No CoTracker source files found')
versions={}
for name in ['cotracker','torch','numpy','opencv-python-headless','scipy','trimesh','Pillow']:
 d=distribution(name); meta=d.read_text('METADATA') or ''
 versions[name]={'version':d.version,'metadata_sha256':hashlib.sha256(meta.encode()).hexdigest()}
torch=distribution('torch'); extensions=sorted(Path(torch.locate_file('torch')).glob('_C*.so'))
result={'python_version':sys.version,'python_binary_sha256':sha(sys.executable),'cotracker_direct_url':json.loads(cot.read_text('direct_url.json') or '{}'),'cotracker_source':code,'packages':versions,'torch_main_extensions':[{'file':p.name,'sha256':sha(p)} for p in extensions], 'scope':'CoTracker Python source bytes, installed dependency versions/metadata, Python binary, main torch extension; not every transitive system-library byte'}
print(json.dumps(result,sort_keys=True))
'''
    result=subprocess.run([str(python),'-I','-c',probe],cwd=Path(root),stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=30,
                          env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1'))
    if result.returncode: raise ValueError('Prepared CoTracker runtime identity could not be verified: '+result.stderr[-1200:])
    return json.loads(result.stdout)


def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''): digest.update(block)
    return digest.hexdigest()
def fingerprint(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp-'+uuid.uuid4().hex[:8])
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def data_path(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or not relative.startswith('data/') or any(p.startswith('.') for p in Path(relative).parts):
        raise ValueError('Only nonhidden declared data artifacts are accepted')
    path = (root/relative).resolve()
    if not path.is_relative_to(root/'data') or path != root/relative:
        raise ValueError('Symlinks or paths outside data are unavailable')
    return path


def file_receipts(paths, root):
    rows=[]
    for path in sorted(set(Path(p).resolve() for p in paths)):
        if not path.is_file() or not path.is_relative_to(root): raise ValueError('Dependency file unavailable')
        relative = str(path.relative_to(root))
        if any(p.startswith('.') for p in Path(relative).parts): raise ValueError('Hidden dependencies are unavailable')
        rows.append({'path': relative, 'sha256': sha(path), 'bytes': path.stat().st_size})
    return rows


def verify_receipts(rows, root):
    if not rows: return False
    try:
        for row in rows:
            path = root/row['path']
            if path.resolve() != path or not path.is_file() or sha(path) != row['sha256']: return False
        return True
    except (OSError, KeyError, ValueError): return False


def request_value(value):
    if not isinstance(value, dict) or set(value)-{'twin_id','track_id','base_sha256','mode','imported_result_id'}:
        raise ValueError('Unexpected upgrade request fields')
    result = {key:value.get(key) for key in ['twin_id','track_id','base_sha256','mode']}
    if not all(isinstance(result[k], str) and 0<len(result[k])<=180 for k in ['twin_id','track_id']): raise ValueError('Choose a published twin and component')
    if not isinstance(result['base_sha256'],str) or not re.fullmatch(r'[a-f0-9]{64}',result['base_sha256']): raise ValueError('Current original SHA256 is required')
    if not isinstance(result['mode'],str) or result['mode'] not in MODES: raise ValueError('Unsupported upgrade mode')
    if result['mode']=='imported_sam3d':
        if not isinstance(value.get('imported_result_id'),str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,120}',value['imported_result_id']): raise ValueError('Choose a validated SAM 3D import')
        result['imported_result_id']=value['imported_result_id']
    elif value.get('imported_result_id') is not None: raise ValueError('An import belongs only to imported_sam3d mode')
    return result


def _context(request, root):
    root=Path(root).resolve(); request=request_value(request)
    catalog=TwinCatalog(root); row,document,digest=catalog.load(request['twin_id'])
    if digest!=request['base_sha256']: raise ValueError('The original program changed; refresh and plan again')
    asset=next((a for a in document['assets'] if a['track_id']==request['track_id']),None)
    if not asset: raise ValueError('Choose a component with a saved learned GLB')
    inputs={}
    for item in document['inputs']: inputs.setdefault(item['kind'],[]).append(item)
    def single(kind):
        rows=inputs.get(kind,[])
        if len(rows)!=1: raise ValueError('Exactly one saved '+kind+' is required')
        return data_path(root,rows[0]['path'])
    case=single('source_manifest').parent
    discovery=single('model_discovery'); annotations=single('model_tracklet_proposals'); proposal=single('model_semantic_proposal')
    segmentation=single('actual_sam_masks').parent
    if single('geometric_mask_gates').parent!=segmentation: raise ValueError('Mask gates and tracks have different origins')
    audit=single('model_mask_plausibility_audit')
    audit_record=read(audit)
    if audit_record.get('source_sha256')!=document['source']['sha256'] or audit_record.get('source_track_index_sha256')!=sha(segmentation/'tracks.json'):
        raise ValueError('Cached mask audit no longer matches source and mask track index')
    record=read(proposal)
    if record.get('status')!='complete_validated_model_proposal' or record.get('source_sha256')!=document['source']['sha256']:
        raise ValueError('A successful source-matched cached semantic proposal is required')
    for key,path in [('discovery_sha256',discovery),('annotation_sha256',annotations),('frame_index_sha256',case/'frames/index.json')]:
        if record.get(key)!=sha(path): raise ValueError('Cached reasoning dependency changed: '+key)
    models=[]; selected=None
    for item in inputs.get('learned_asset_metadata',[]):
        path=data_path(root,item['path']); model=read(path); models.append(path)
        if model.get('object_id')==request['track_id']: selected=path
    if selected is None: raise ValueError('No same-track model metadata accompanies this asset')
    model=read(selected)
    if model.get('source',{}).get('video_sha256')!=document['source']['sha256'] or model.get('manual_geometry_fallback') is not False or model.get('geometry_reconstructed_by_model') is not True:
        raise ValueError('An unchanged learned source-matched model is required')
    glb=(selected.parent/model['outputs']['glb_path']).resolve()
    if glb!=data_path(root,asset['path']) or sha(glb)!=asset['sha256']: raise ValueError('Model and published GLB do not join')
    model_pixels=[]
    for role,hashkey in [('image','image_sha256'),('mask','mask_sha256')]:
        path=(selected.parent/model['source'][role+'_path']).resolve()
        if not path.is_relative_to(root/'data') or not path.is_file() or sha(path)!=model['source'].get(hashkey):
            raise ValueError('Saved model reference pixels changed: '+role)
        model_pixels.append(path)
    selection=Path(model['source']['selection_manifest_path']).resolve()
    if not selection.is_relative_to(root/'data') or not selection.is_file() or sha(selection)!=model['source']['selection_manifest_sha256']:
        raise ValueError('Original automatic selection is unavailable or changed')
    selection_doc=read(selection)
    if selection_doc.get('object_id')!=request['track_id'] or selection_doc.get('selection_method',{}).get('agent_visual_choice') is not False:
        raise ValueError('Source selection must retain its algorithmic component identity')
    chosen=selection_doc['selected'][0]
    if chosen['selection_rank']!=1: raise ValueError('Original rank-one selection required')
    selection_pixels=[]
    for role in ['rgb','mask']:
        path=(selection.parent/chosen['exports'][role]).resolve()
        if not path.is_relative_to(root/'data') or sha(path)!=chosen['exports'][role+'_sha256']: raise ValueError('Selected input pixels changed')
        selection_pixels.append(path)
    origins=[p for p in selection.parents if (p/'manifest.json').is_file() and read(p/'manifest.json').get('source',{}).get('sha256')==document['source']['sha256']]
    if not origins: raise ValueError('Original selection has no source-matched case ancestor')
    tracking_case=origins[0]; tracking_seg=tracking_case/'segmentation'/segmentation.name
    # A saved geometry may come from a coarser source-PTS sampling. Retain that
    # exact sampling and disclose it instead of relabelling its old anchor.
    if not (tracking_seg/'tracks.json').is_file(): raise ValueError('Original selected-view tracking masks are unavailable')
    cfg=read(tracking_seg/'annotations.json')
    if cfg.get('manual_seed_boxes_used') is not False or not any(t['id']==request['track_id'] for t in cfg['tracks']): raise ValueError('Model-generated same-track prompt lineage is required')
    tracking_audit=tracking_seg/'model-audit-v1/audit.json'
    if not tracking_audit.is_file(): raise ValueError('Original tracking mask audit is unavailable')
    old_audit=read(tracking_audit)
    if old_audit.get('source_sha256')!=document['source']['sha256'] or old_audit.get('source_track_index_sha256')!=sha(tracking_seg/'tracks.json'):
        raise ValueError('Original tracking mask audit changed')
    motion_paths=[data_path(root,i['path']) for i in inputs.get('model_conditional_motion_experiment',[])]
    baseline_motion=next((p for p in motion_paths if read(p).get('object_id')==request['track_id']),None)
    adapter=single('dense_semantic_input_adapter') if inputs.get('dense_semantic_input_adapter') else None
    if adapter and record.get('adapter_sha256')!=sha(adapter): raise ValueError('Dense semantic adapter changed')
    dependencies=[data_path(root,i['path']) for i in document['inputs'] if i['path'].startswith('data/')]
    dependencies += [data_path(root,document['source']['path']),data_path(root,asset['path']),selection,*selection_pixels,*model_pixels,tracking_audit,
                     tracking_case/'manifest.json',tracking_case/'frames/index.json',tracking_seg/'annotations.json',tracking_seg/'tracks.json',tracking_seg/'frames.json',tracking_seg/'quality.json']
    # Record every frame and mask consumed by the model worker, not just the
    # JSON indices. Reasoning itself will not be repeated.
    tracking_rows={r['index']:r for r in read(tracking_case/'frames/index.json')['frames']}
    track=next(t for t in read(tracking_seg/'tracks.json')['tracks'] if t['track_id']==request['track_id'])
    dependencies += [tracking_case/tracking_rows[f['sample_index']]['file'] for f in track['frames']]
    dependencies += [tracking_seg/f['raw_mask_file'] for f in track['frames']]
    native_cpu_preflight=None
    if request['mode']=='refit_native_cpu':
        from native_appearance import load_native_glb
        native_cpu_preflight=load_native_glb(glb).receipt
    imported=None
    if request['mode']=='imported_sam3d':
        import upgrade_bundles
        imported=upgrade_bundles.resolve_import(request['imported_result_id'],root=root)
        if imported.get('backend')!='sam3d_objects' or imported.get('status')!='validated_candidate_pending_alignment':
            raise ValueError('This upgrade needs a validated SAM 3D geometry import; SAM 3.1 anchor comparisons are not a replacement 3D asset')
        if imported.get('alignment_eligible') is not True:
            raise ValueError('Native SAM 3D asset retained for inspection, but the current generic alignment cannot consume it: '+str(imported.get('alignment_ineligible_reason') or 'the native material/geometry contract is unsupported'))
        for key in ['twin_id','track_id','base_sha256']:
            if imported.get(key)!=request[key]: raise ValueError('SAM 3D import belongs to a different original/component')
        imported_model=(root/imported['model_metadata_path']).resolve()
        if not imported_model.is_relative_to(root/'data'): raise ValueError('Imported model outside data')
        dependencies += [imported_model,imported_model.parent/'asset.glb']
    return dict(request=request,row=row,document=document,digest=digest,case=case,discovery=discovery,annotations=annotations,
                proposal=proposal,segmentation=segmentation,audit=audit,models=models,selected_model=selected,selection=selection,
                motion_paths=motion_paths,baseline_motion=baseline_motion,adapter=adapter,tracking_case=tracking_case,
                tracking_segmentation=tracking_seg,dependencies=dependencies,imported=imported,native_cpu_preflight=native_cpu_preflight)


def plan(request, root=ROOT):
    root=Path(root).resolve(); context=_context(request,root); request=context['request']
    stages=[]
    for key,label in STAGES:
        reuse=key=='cached_inputs' or (key=='reconstruct' and request['mode'] in {'refit_existing','refit_native_cpu'})
        reason={'cached_inputs':'Reuse exact saved discovery, mask audit and procedure proposal; no new reasoning.',
                'reconstruct':'Keep the original mesh unchanged.' if reuse else ('Use a validated genuine SAM 3D result.' if request['mode']=='imported_sam3d' else 'Run the cached local TripoSR model on the original selected view.'),
                'pose':('CPU native-material silhouette/appearance initialization using unchanged original GLB; no hand-set pose.' if request['mode']=='refit_native_cpu' else 'Bounded automatic silhouette/appearance initialization; no hand-set pose.'),
                'motion':('Explicit CPU CoTracker; maximum model side256pixels, aspect follows source with stride rounding; original sampling, unsupported poses withheld.' if request['mode']=='refit_native_cpu' else 'CoTracker on the original selected-view sampling, with unsupported poses withheld.'),
                'procedure':'Use identical cached semantics; replace only this component’s assets/motion.',
                'comparison':'Compare common source PTS, excluding both reference anchors.',
                'publish':'Register a separate result; preserve the preferred original.'}[key]
        stages.append({'id':key,'label':label,'action':'reuse' if reuse else 'run','reason':reason})
    blockers=[]
    if not (root/'.venv-sam/bin/python').is_file(): blockers.append('Prepared local .venv-sam runtime unavailable')
    return {'format':'interaction-twin-upgrade-plan','version':'0.1','can_run':not blockers,'request':request,
            'stages':stages,'blockers':blockers,'inputs':file_receipts(context['dependencies'],root),
            'cached_reasoning':True,'local_only':True,'tracking_source_case':str(context['tracking_case'].relative_to(root)),
            'source_sampling_may_differ_from_baseline':context['tracking_case']!=context['case'],
            'coordinate_policy':'independent_native_component_gauges; no physical assembly',
            'physical_accuracy_verified':False,
            **({'execution_device':'cpu','alignment_adapter':'native-base-color-v1','tracking_model_maximum_dimension':256,
                'new_reconstruction_inference':False,'new_reasoning_inference':False,'native_preflight':context['native_cpu_preflight']} if request['mode']=='refit_native_cpu' else {})}


def compare_motion(baseline, candidate):
    """Exact PTS matching; reference and fitting-anchor observations excluded."""
    baseline=baseline or {'frames':[]}; candidate=candidate or {'frames':[]}
    left={f['timestamp_s']:f for f in baseline.get('frames',[])}; right={f['timestamp_s']:f for f in candidate.get('frames',[])}
    anchors={t for rows in [left,right] for t,f in rows.items() if f.get('status','').startswith('reference_')}
    times=sorted((set(left)&set(right))-anchors)
    def score(rows):
        supported=[rows[t] for t in times if rows[t].get('status')=='supported_conditional_motion_candidate']
        def median(key):
            values=[r[key] for r in supported if isinstance(r.get(key),(int,float)) and not isinstance(r[key],bool) and math.isfinite(r[key])]
            return statistics.median(values) if values else None
        return {'conditional_supported':len(supported),'total_samples':len(times),'unsupported':len(times)-len(supported),
                'silhouette_iou_median':median('raw_full_projection_iou_with_sam'),'residual_px_median':median('heldout_median_px')}
    a,b=score(left),score(right)
    return {'scope':'Internal conditional source-frame diagnostics; masks enter fitting/gates and whole-clip temporal context is used. Not physical accuracy or independent generalization.',
            'matched_sample_count':len(times),'timestamps_s':times,'excluded_reference_timestamps_s':sorted(anchors),
            'baseline':a,'candidate':b,'baseline_all_sample_count':len(left),'candidate_all_sample_count':len(right),
            'metrics':[{'label':label,'before':a[key],'after':b[key],'unit':unit,'denominator':len(times)} for key,label,unit in
                       [('conditional_supported','Supported non-reference samples','samples'),('silhouette_iou_median','Median supported silhouette IoU','IoU'),('residual_px_median','Median supported held-out residual','px')]],
            'metric_population_note':'IoU/residual use supported samples on each side and may represent different subsets; coverage is shown with the common denominator.',
            'automatic_baseline_replacement':False,'physical_accuracy_verified':False,'unknown_between_samples':True}


class Cancelled(RuntimeError): pass


class UpgradeRunner:
    def __init__(self, request=None, *, root=ROOT, run_id=None):
        self.root=Path(root).resolve(); self.base=data_path(self.root,'data/model-upgrades/runs'); self.base.mkdir(parents=True,exist_ok=True)
        self.id=run_id or 'upgrade-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
        if not RUN_ID.fullmatch(self.id): raise ValueError('Invalid upgrade run ID')
        self.directory=data_path(self.root,'data/model-upgrades/runs/'+self.id); self.directory.mkdir(exist_ok=True)
        self.path=self.directory/'run.json'; self.cancel_event=threading.Event(); self.process=None; self.termination_lock=threading.Lock()
        self.code_snapshot=self.directory/('upgrade-runner-source-'+LOADED_SOURCE_SHA256+'.py')
        if self.code_snapshot.exists():
            if sha(self.code_snapshot)!=LOADED_SOURCE_SHA256: raise ValueError('Frozen upgrade source snapshot changed')
        else: self.code_snapshot.write_bytes(LOADED_SOURCE)
        self.implementation_identities=stage_source_identities()
        self.state=read(self.path) if self.path.exists() else {'id':self.id,'created_at':now(),'status':'created','phase':None,'progress':0,'phases':[],
                  'logs_url':'/files/'+str((self.directory/'pipeline.log').relative_to(self.root)),'local_only':True,'cached_reasoning':True}
        self.request=request_value(request) if request is not None else self.state.get('request')
        source={'path':str(self.code_snapshot.relative_to(self.root)),'sha256':LOADED_SOURCE_SHA256}
        self.state['orchestrator_source']=source
        if source not in self.state.setdefault('orchestrator_source_history',[]): self.state['orchestrator_source_history'].append(source)
        if not (self.directory/'pipeline.log').exists(): (self.directory/'pipeline.log').write_text('Local-only geometry upgrade. No new hosted reasoning, cluster access, credentials or downloads.\n')
        self.save()

    def save(self): self.state['updated_at']=now(); write(self.path,self.state)
    def cancel(self): self.cancel_event.set(); self._terminate()
    def _terminate(self):
        with self.termination_lock:
            process=self.process
            if process and process.poll() is None:
                try:
                    os.killpg(process.pid,signal.SIGTERM); process.wait(timeout=4)
                except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); process.wait(timeout=4)
                except ProcessLookupError: pass
    def command(self,args,timeout=600):
        if self.cancel_event.is_set(): raise Cancelled('Cancelled before subprocess')
        with (self.directory/'pipeline.log').open('a') as log:
            log.write('\n'+now()+' '+json.dumps([str(x) for x in args])+'\n'); log.flush()
            environment=os.environ.copy(); environment.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
            self.process=subprocess.Popen([str(x) for x in args],cwd=self.root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=environment)
            start=time.monotonic()
            while self.process.poll() is None:
                if self.cancel_event.wait(.2): self._terminate(); raise Cancelled('Cancelled; subprocess group terminated')
                if time.monotonic()-start>timeout: self._terminate(); raise TimeoutError('Upgrade stage exceeded its local time budget')
            code=self.process.returncode; self.process=None
            if self.cancel_event.is_set(): raise Cancelled('Cancelled; subprocess group terminated')
            if code: raise RuntimeError('Local stage exited '+str(code)+'; inspect the preserved log')

    def phase(self,name,dependencies,action,*,parameters=None,optional=False):
        if self.cancel_event.is_set(): raise Cancelled('Cancelled by user')
        old=next((p for p in reversed(self.state['phases']) if p['name']==name),None)
        inputs=file_receipts(dependencies,self.root)
        implementation={'stage_sha256':self.implementation_identities[name],'source_path':str(self.code_snapshot.relative_to(self.root)),'source_sha256':LOADED_SOURCE_SHA256}
        signature=fingerprint({'contract':'upgrade-stage-v2','name':name,'inputs':inputs,'parameters':parameters,'implementation_sha256':implementation['stage_sha256']})
        self.state.update(phase=name,progress=next(i for i,s in enumerate(STAGES) if s[0]==name)/len(STAGES))
        if old and old.get('signature')==signature and old.get('status') in {'complete','complete_with_gaps','reused_verified'} and verify_receipts(old.get('files'),self.root):
            old.update(status='reused_verified',reverified_at=now()); self.save(); return old['result']
        attempt=1+max([p.get('attempt',0) for p in self.state['phases'] if p['name']==name] or [0])
        directory=self.directory/(name+'-attempt-'+str(attempt)); directory.mkdir()
        receipt={'name':name,'status':'running','attempt':attempt,'signature':signature,'implementation':implementation,'parameters':parameters,'inputs':inputs,'directory':str(directory.relative_to(self.root)),'started_at':now()}
        self.state['phases'].append(receipt); self.save()
        try:
            result,files=action(directory)
            receipt.update(status='complete_with_gaps' if result.get('has_gaps') else 'complete',result=result,files=file_receipts(files,self.root),completed_at=now()); self.save(); return result
        except (Cancelled,KeyboardInterrupt): receipt.update(status='cancelled',completed_at=now()); self.save(); raise
        except Exception as exc:
            receipt.update(status='failed',error=str(exc),completed_at=now()); self.save()
            if not optional: raise
            return {'has_gaps':True,'unavailable':str(exc)}

    def execute(self,request=None):
        request=request_value(request or self.request)
        if self.state.get('request') and self.state['request']!=request: raise ValueError('Resume requires the identical component/mode/original request')
        self.state.update(request=request,status='running',error=None,result=None); self.save()
        try:
            context=_context(request,self.root); proposed=plan(request,self.root)
            frozen=self.directory/'plan.json'
            if frozen.exists():
                old=read(frozen)
                if old['request']!=proposed['request'] or old['inputs']!=proposed['inputs']: raise ValueError('Original cached inputs changed; start a separate upgrade')
            else: write(frozen,proposed)
            if proposed['blockers']: raise ValueError('; '.join(proposed['blockers']))
            lock_path=data_path(self.root,'data/flow-runs')/'.model-worker.lock'; lock_path.parent.mkdir(parents=True,exist_ok=True)
            with lock_path.open('a+') as lock:
                try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError: raise RuntimeError('Another local pipeline owns the model worker; retry after it finishes')
                self._execute(context)
            latest={p['name']:p for p in self.state['phases']}
            self.state.update(status='complete_with_gaps' if any(p.get('result',{}).get('has_gaps') or p['status']=='failed' for p in latest.values()) else 'complete',phase=None,progress=1,completed_at=now())
        except (Cancelled,KeyboardInterrupt) as exc: self.state.update(status='cancelled',error=str(exc),completed_at=now())
        except Exception as exc: self.state.update(status='failed',error=str(exc),completed_at=now())
        finally: self._terminate(); self.save()
        return self.state

    def _execute(self,c):
        root=self.root; python=root/'.venv-sam/bin/python'; request=c['request']; tid=request['track_id']; mode=request['mode']
        relative=lambda path:str(Path(path).resolve().relative_to(root))
        def cli(script,*args): self.command([python,root/'scripts'/script,*args])
        def cached(directory):
            output=directory/'receipt.json'; write(output,{'status':'verified_exact_cache','reasoning_inference_executed':False,'inputs':file_receipts(c['dependencies'],root)})
            return {'cached_reasoning':True,'reasoning_inference_executed':False},[output]
        self.phase('cached_inputs',c['dependencies'],cached)
        def reconstruct(directory):
            if mode in {'refit_existing','refit_native_cpu'}: path=c['selected_model']
            elif mode=='imported_sam3d': path=(root/c['imported']['model_metadata_path']).resolve()
            else:
                output=directory/'model'; cli('stage3_reconstruct_local.py','--selection',c['selection'],'--output',output,'--resolution','96'); path=output/'metadata.json'
            model=read(path)
            if model['object_id']!=tid or model['source']['video_sha256']!=c['document']['source']['sha256']: raise ValueError('Reconstructed model source/component mismatch')
            mesh=path.parent/model['outputs']['glb_path']; expected=next(x['sha256'] for x in model['artifact_files'] if x['file']=='asset.glb')
            if sha(mesh)!=expected: raise ValueError('Reconstructed GLB hash mismatch')
            result={'model':relative(path),'backend':model['backend'],'native_asset':relative(mesh),'physical_shape_verified':False,'reconstruction_inference_executed':mode=='local_triposr'}
            receipt=directory/'receipt.json'; write(receipt,result)
            return result,[receipt,path,mesh,path.parent/model['source']['image_path'],path.parent/model['source']['mask_path']]
        reconstruct_deps=[c['selected_model'],c['selection'],*[(c['selection'].parent/read(c['selection'])['selected'][0]['exports'][role]).resolve() for role in ['rgb','mask']]]
        if mode=='local_triposr':
            reconstruct_deps += [root/'scripts/stage3_reconstruct_local.py',root/'models/triposr-checkpoint/model.ckpt',root/'models/triposr-checkpoint/config.yaml',root/'models/triposr-checkpoint/provenance.json']
            reconstruct_deps += sorted((root/'models/triposr-code/tsr').rglob('*.py'))
        if mode=='imported_sam3d':
            imported_metadata=(root/c['imported']['model_metadata_path']).resolve()
            reconstruct_deps += [imported_metadata,imported_metadata.parent/'asset.glb']
        geometry=self.phase('reconstruct',reconstruct_deps,reconstruct,parameters={'mode':mode,'resolution':96})
        metadata=root/geometry['model']; mesh=root/geometry['native_asset']
        alignment=alignment_configuration(read(metadata),{'alignment_adapter':'native-base-color-v1'} if mode=='refit_native_cpu' else (c['imported'] if mode=='imported_sam3d' else None))
        native=alignment['native']
        def pose(directory):
            output=directory/'fit'; cli(alignment['pose_script'],'--manifest',metadata,'--out',output,'--search-width','128','--refine-count','2','--max-evaluations','60','--appearance-weight','.25')
            path=output/'experiment.json'; raw=read(path)
            if raw.get('object_id')!=tid or not raw.get('candidate'): raise ValueError('No component reference candidate was produced')
            if mode=='imported_sam3d' and not native: self._adapt_backend(path,metadata)
            return {'reference':relative(path),'has_gaps':not raw.get('usable_pose_gate',{}).get('accepted',False),'physical_pose_verified':False},[path,*([output/'raw-experiment.json'] if mode=='imported_sam3d' and not native else [])]
        pose_parameters={'search_width':128,'starts':2,'evaluations':60,'appearance_weight':.25}
        if native: pose_parameters.update(alignment_adapter=alignment['adapter'],installed_runtime=cotracker_runtime_identity(root))
        fitted=self.phase('pose',[metadata,mesh,*[root/p for p in alignment['pose_dependencies']]],pose,parameters=pose_parameters,optional=True)
        def motion(directory):
            if not fitted.get('reference'): raise ValueError('No reference fit; motion remains unknown')
            output=directory/'tracking'
            options=['--device','cpu','--model-max-dimension','256'] if mode=='refit_native_cpu' else []
            cli(alignment['motion_script'],'--reference',root/fitted['reference'],'--segmentation-run',c['tracking_segmentation'].name,'--output',output,*options)
            path=output/'experiment.json'; record=read(path)
            if record.get('status')!='complete_conditional_tracking_experiment' or record.get('object_id')!=tid: raise ValueError('Tracking did not complete; native mesh retained')
            if mode=='imported_sam3d' and not native: self._adapt_backend(path,metadata)
            return {'experiment':relative(path),'supported_nonreference_frames':record.get('supported_nonreference_frames',0),'has_gaps':record.get('supported_nonreference_frames',0)==0,'physical_motion_verified':False},[path,output/'tracks.npz',output/'point-support.json',*([output/'raw-experiment.json'] if mode=='imported_sam3d' and not native else [])]
        motion_deps=[metadata,mesh,*[root/p for p in alignment['motion_dependencies']],root/'models/cotracker3/scaled_offline.pth',root/'models/cotracker3/provenance.json',*c['dependencies']]
        if fitted.get('reference'): motion_deps.append(root/fitted['reference'])
        motion_parameters={'installed_runtime':cotracker_runtime_identity(root)}
        if mode=='refit_native_cpu': motion_parameters.update(device='cpu',model_maximum_dimension=256,alignment_adapter=alignment['adapter'])
        tracked=self.phase('motion',motion_deps,motion,parameters=motion_parameters,optional=True)
        model_paths=[metadata if read(p)['object_id']==tid else p for p in c['models']]
        motion_paths=[p for p in c['motion_paths'] if read(p)['object_id']!=tid]
        if tracked.get('experiment'): motion_paths.append(root/tracked['experiment'])
        def procedure(directory):
            output=directory/'procedure'
            args=['build','--proposal',c['proposal'],'--segmentation',c['segmentation'],'--out',output]
            if c['adapter']: script='stage5_dense_compile.py'; args += ['--adapter',c['adapter']]
            else: script='stage5_compile.py'; args += ['--case',c['case'],'--discovery',c['discovery'],'--annotations',c['annotations']]
            for path in model_paths: args += ['--model-metadata',path]
            for path in motion_paths: args += ['--motion',path]
            cli(script,*args); cli('stage5_compile.py','validate',output/'interaction.json')
            document=read(output/'interaction.json')
            # These are structural equality checks against the compiler output,
            # not a claim that the cached statements are physically correct.
            for key in ['steps','relationships','variables','assertions','tracklets']:
                if document[key]!=c['document'][key]: raise ValueError('Cached semantic section changed unexpectedly: '+key)
            return {'interaction':relative(output/'interaction.json'),'has_gaps':not bool(tracked.get('experiment')),'cached_semantics_unchanged':True},[output/'interaction.json',output/'replay.json',output/'procedure.txt',output/'validation.json']
        compiled=self.phase('procedure',[c['proposal'],c['discovery'],c['annotations'],*model_paths,*motion_paths,root/'stage5.py',root/'scripts/stage5_compile.py',root/'schemas/interaction.schema.json',*([c['adapter'],root/'scripts/stage5_dense_compile.py'] if c['adapter'] else [])],procedure)
        def comparison(directory):
            value=compare_motion(read(c['baseline_motion']) if c['baseline_motion'] else None,read(root/tracked['experiment']) if tracked.get('experiment') else None)
            value.update(track_id=tid,baseline_twin_id=request['twin_id'],candidate_interaction=compiled['interaction'],baseline_asset=next(a for a in c['document']['assets'] if a['track_id']==tid),candidate_asset=next(a for a in read(root/compiled['interaction'])['assets'] if a['track_id']==tid))
            path=directory/'comparison.json'; write(path,value); return {'comparison':value,'comparison_path':relative(path)},[path]
        comparison_deps=[root/compiled['interaction'],*([c['baseline_motion']] if c['baseline_motion'] else []),*([root/tracked['experiment']] if tracked.get('experiment') else [])]
        compared=self.phase('comparison',comparison_deps,comparison,parameters={'policy':'exact_source_pts_exclude_union_of_reference_anchors_v1'})
        def publish(directory):
            interaction=compiled['interaction']; registry=root/'workspace/registered-cases.json'; current=read(registry) if registry.exists() else {'cases':[]}
            if not isinstance(current.get('cases'),list): raise ValueError('Invalid workspace registry')
            title=c['row'].get('title','Interaction twin')[:45]+' · '+tid[:16]+' upgrade'
            if not any(row.get('interaction')==interaction for row in current['cases']): current['cases'].append({'interaction':interaction,'title':title}); write(registry,current)
            cli('build_workspace.py')
            row=next((r for r in read(root/'workspace/cases.json')['cases'] if r['interaction_path']==interaction),None)
            if not row: raise ValueError('Compiled candidate did not publish; original remains available')
            value={'workspace_case_id':row['id'],'result_case_id':row['id'],'workspace_url':'/workspace/index.html?case='+row['id'],
                   'comparison_url':'/files/'+compared['comparison_path'],'comparison':compared['comparison'],
                   'candidate_interaction':interaction,'baseline_twin_id':request['twin_id'],'original_remains_preferred':True,
                   'cached_reasoning':True,'new_reasoning_calls':0,'physical_assembly_verified':False}
            path=directory/'published.json'; write(path,value); return value,[path,root/interaction]
        published=self.phase('publish',[root/compiled['interaction'],root/compared['comparison_path'],root/'scripts/build_workspace.py'],publish)
        self.state['result']=published; self.save()

    def _adapt_backend(self,path,metadata):
        """Correct an old TripoSR-only label on a NEW derived record.

        The immutable script's raw receipt remains byte preserved alongside it.
        Geometry, pose matrices, gates and numerical results are not changed.
        """
        raw=path.read_bytes(); original=path.with_name('raw-experiment.json')
        if original.exists(): raise ValueError('Raw adapter receipt already exists')
        original.write_bytes(raw); record=json.loads(raw); model=read(metadata)
        record.setdefault('provenance',{}).update(sam3d_output=True,reconstruction_backend=model['backend'],appearance=model.get('appearance','model-generated appearance'))
        record['backend_label_adapter']={'format':'interaction-twin-backend-label-adapter','raw_path':str(original.relative_to(self.root)),
            'raw_sha256':sha(original),'model_metadata_sha256':sha(metadata),'reason':'Unchanged generic legacy fitter contains a TripoSR-only display label; actual input is validated SAM 3D.',
            'geometry_or_numeric_results_modified':False}
        write(path,record)
        html=path.parent/'index.html'
        if html.is_file():
            html.with_name('raw-index.html').write_bytes(html.read_bytes()); html.write_text(html.read_text().replace('TripoSR', 'SAM 3D Objects (generic legacy fit)'))


def get_run(run_id, root=ROOT):
    if not isinstance(run_id,str) or not RUN_ID.fullmatch(run_id): raise ValueError('Invalid upgrade run ID')
    path=data_path(Path(root).resolve(),'data/model-upgrades/runs/'+run_id+'/run.json')
    if not path.is_file(): raise ValueError('Unknown upgrade run')
    return read(path)


def list_runs(root=ROOT):
    base=data_path(Path(root).resolve(),'data/model-upgrades/runs')
    if not base.exists(): return []
    return sorted([get_run(p.parent.name,root) for p in base.glob('*/run.json') if RUN_ID.fullmatch(p.parent.name)],key=lambda r:r['created_at'],reverse=True)
