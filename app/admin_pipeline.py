"""Fixed owner-upload/repair executor. Feedback never becomes executable code.

Repairs adopt only hash-verified completed predecessor outputs, then create new
numbered attempts. A GPU-unavailable result remains a gap; it is not replaced.
"""
from __future__ import annotations
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import threading

import pipeline as P
from agent_memory import private_directory, write_immutable
from review_agents import verified_ref

ROOT=P.ROOT
PAID_PHASES={'discover','segment','audit','procedure_proposal','mechanism_topology'}
SUCCESS={'complete','complete_with_gaps','reused_verified','reused_verified_with_gaps','adopted_verified','adopted_verified_with_gaps'}
ATTEMPT_DIRECTORY=re.compile(r'flow-(?:discovery|components|inputs|portable-input|model(?:-receipts)?|pose(?:-receipts)?|motion(?:-receipts)?|procedure-proposal|procedure)-([1-9][0-9]{0,6})(?:-[0-9]+)?\Z')
RESERVATION_FILE=re.compile(r'attempt-([1-9][0-9]{0,6})\.json\Z')


def reserve_repair_attempt(base_state,run_id,phase,minimum,*,root=ROOT):
    """Reserve a fresh number inside one verified source case, never a retry.

    The bounded scan includes pre-reservation legacy failures. Atomic immutable
    receipts and the case lock serialize forked repairs, including reservations
    whose process died before making its output directory. No global run scan,
    output deletion, paid request inspection or inference occurs here.
    """
    root=Path(root).resolve()
    if phase not in P.PHASES or not P.SLUG.fullmatch(run_id) or type(minimum)is not int or not 1<=minimum<=9999999:
        raise ValueError('Bounded trusted repair attempt required')
    if not isinstance(base_state.get('phases'),list)or len(base_state['phases'])>64:
        raise ValueError('Bounded predecessor phase manifest required')
    prepare=next((p for p in base_state.get('phases',[])if p.get('name')=='prepare'),None)
    case_rel=(prepare or{}).get('result',{}).get('case')
    if not isinstance(case_rel,str):raise ValueError('Repair requires the exact predecessor source case')
    case=root/case_rel
    if case.resolve()!=case or case.parent!=root/'data' or case.is_symlink()or not case.is_dir():
        raise ValueError('Repair source case must remain an ordinary direct data directory')
    manifest=case/'manifest.json'
    if manifest.is_symlink()or not manifest.is_file():raise ValueError('Repair source manifest must remain ordinary')
    reference=next((r for r in prepare.get('files',[])if r.get('path')==(manifest.relative_to(root).as_posix())),None)
    if not reference or not P.verify_receipts([reference]):raise ValueError('Repair source manifest changed')
    source_sha=base_state.get('source',{}).get('sha256')
    source=P.read(manifest).get('source',{})
    if not isinstance(source_sha,str)or not re.fullmatch('[0-9a-f]{64}',source_sha)or source_sha not in {source.get('sha256'),source.get('original_sha256')}:
        raise ValueError('Repair case differs from the exact predecessor input')
    binding={'case':case_rel,'source_sha256':source_sha,'source_manifest':reference}
    identity=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    folder=private_directory(root,'admin-attempt-reservations')/identity
    if folder.is_symlink():raise ValueError('Repair reservation namespace cannot be a symlink')
    folder.mkdir(mode=0o700,exist_ok=True)

    def bounded_entries(directory):
        if directory.is_symlink()or directory.resolve()!=directory:raise ValueError('Repair attempt scan cannot follow symlinks')
        result=[]
        with os.scandir(directory)as entries:
            for entry in entries:
                if len(result)>=4096:raise ValueError('Repair source-case scan budget exceeded')
                result.append(entry)
        return result

    descriptor=os.open(folder/'.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    with os.fdopen(descriptor,'a+')as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        highest=max([minimum-1,*[int(r.get('attempt',0))for r in base_state.get('phases',[])if type(r.get('attempt',0))is int]])
        for directory in [case,case/'reasoning',case/'segmentation']:
            if directory.is_symlink():raise ValueError('Repair attempt scan cannot follow symlinks')
            if not directory.exists():continue
            for entry in bounded_entries(directory):
                match=ATTEMPT_DIRECTORY.fullmatch(entry.name)
                if match:
                    if entry.is_symlink()or not entry.is_dir(follow_symlinks=False):raise ValueError('Invalid preserved repair attempt directory')
                    highest=max(highest,int(match[1]))
        for entry in bounded_entries(folder):
            if entry.name=='.lock':continue
            match=RESERVATION_FILE.fullmatch(entry.name)
            if not match or entry.is_symlink()or not entry.is_file(follow_symlinks=False)or entry.stat(follow_symlinks=False).st_size>16384:
                raise ValueError('Invalid immutable repair reservation')
            previous=json.loads(Path(entry.path).read_text())
            if previous.get('format')!='interaction-twin-repair-attempt-reservation'or previous.get('binding')!=binding or previous.get('attempt')!=int(match[1]):
                raise ValueError('Repair reservation belongs to another source or number')
            highest=max(highest,int(match[1]))
        attempt=highest+1
        if attempt>9999999:raise ValueError('Repair attempt namespace exhausted')
        value={'format':'interaction-twin-repair-attempt-reservation','version':1,'binding':binding,
               'attempt':attempt,'phase':phase,'run_id':run_id,'base_run_id':base_state['id'],'created_utc':P.now(),
               'existing_outputs_modified':False,'reservation_authorizes_paid_retry':False}
        path=folder/f'attempt-{attempt}.json';raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
        fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        directory_fd=os.open(folder,os.O_RDONLY)
        try:os.fsync(directory_fd)
        finally:os.close(directory_fd)
        return attempt,{'path':path.relative_to(root).as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def adoption_inputs(receipts,*,repair=False):
    """A repair can adopt historical output without executing changed code."""
    differences=[]
    for receipt in receipts or []:
        path=receipt.get('path','');parts=Path(path).parts
        code=(len(parts)==1 and (path.endswith(('.py','.js')) or path.startswith('requirements-')) or
              len(parts)==2 and parts[0]=='scripts' and path.endswith('.py') or
              len(parts)==2 and parts[0]=='schemas' and path.endswith('.schema.json') or
              path in {'workspace/scene.js','workspace/case-data.js','workspace/robot-vendor.js'})
        if repair and code:
            current=P.confined(ROOT/path)
            if current.is_symlink() or not current.is_file():raise ValueError('Historical code input is unavailable')
            actual=P.sha(current)
            if actual!=receipt['sha256']:
                differences.append({'path':path,'historical_sha256':receipt['sha256'],'current_sha256':actual,
                                    'executed_current_code':False})
        elif not P.verify_receipts([receipt]):raise ValueError('Historical evidence input changed: '+path)
    if not receipts:raise ValueError('Historical phase inputs unavailable')
    return differences


def memory_snapshot(context,root=ROOT):
    from admin_memory_context import validate_context
    validate_context(context)
    folder=private_directory(root,'pipeline-memory')
    identifier=write_immutable(folder,'context',context)
    return (folder/(identifier+'.json')).relative_to(root).as_posix()


def repair_boundary(targets):
    if not targets or not set(targets)<={'reasoning','tracking','reconstruction','mechanics'}:
        raise ValueError('Unsupported trusted repair target')
    if 'tracking' in targets:return 'discover'
    if 'reconstruction' in targets:return 'select'
    return 'procedure_proposal'


class AdminRunner(P.Runner):
    """Explicit lineage adoption, separate from ordinary same-signature caching."""
    def __init__(self,run_id,*,base_state=None,boundary=None,repair_number=0,execution_lease=None,machine_review_ref=None):
        super().__init__(run_id)
        self.base_state=copy.deepcopy(base_state)
        self.boundary=boundary
        self.repair_number=repair_number
        self.execution_lease=execution_lease
        self.machine_review_ref=machine_review_ref
        prior_guidance=self.state.get('admin_machine_review_ref')
        if prior_guidance!=machine_review_ref and prior_guidance is not None:
            raise ValueError('This run already belongs to another immutable machine review context')
        if machine_review_ref is not None:
            self.state['admin_machine_review_ref']=machine_review_ref
            self.save()
        self.adoptions=[]
        if base_state and boundary and not self.state.get('phases'):
            # New outputs get a larger attempt number in the same immutable case.
            self.state['phases']=[{'name':r['name'],'status':'repair_pending','attempt':r.get('attempt',0)}
                                  for r in base_state.get('phases',[])]
            self.save()

    def phase(self,name,dependencies,action,*,optional=False):
        if getattr(self,'execution_lease',None):self.execution_lease.guard('phase_'+name)
        base=next((r for r in (self.base_state or {}).get('phases',[]) if r['name']==name),None)
        adopt=base and (self.boundary is None or P.PHASES.index(name)<P.PHASES.index(self.boundary))
        if adopt:
            if base.get('status') in SUCCESS and P.verify_receipts(base.get('files')):
                # Completed snapshots are rehashed; no model or old pose is
                # invented here. Native artifacts retain their original lineage.
                code_differences=adoption_inputs(base.get('inputs'),repair=self.boundary is not None)
                row=copy.deepcopy(base)
                row.update(status='adopted_verified_with_gaps' if base.get('result',{}).get('has_gaps') else 'adopted_verified',
                           adopted_from_run=self.base_state['id'],adopted_at=P.now(),new_inference=False,
                           historical_code_changes=code_differences)
                self.state['phases']=[r for r in self.state['phases'] if r['name']!=name]+[row]
                self.adoptions.append(name);self.save();return row['result']
            if name in PAID_PHASES:
                raise ValueError('Prior paid attempt is uncertain or changed at '+name+'; no automatic resend. Inspect its preserved request receipt.')
            if base.get('status') in SUCCESS:
                raise ValueError('Predecessor output changed at '+name+'; repair cannot adopt it')
        dependencies=list(dependencies)
        if getattr(self,'machine_review_ref',None):
            dependencies.extend([ROOT/self.machine_review_ref['path'],ROOT/'admin_machine_context.py'])
        if self.base_state and self.boundary is not None:
            previous=copy.deepcopy(next((r for r in self.state['phases']if r.get('name')==name),None))
            if name in PAID_PHASES and previous and previous.get('status')not in SUCCESS|{'repair_pending'}:
                raise ValueError('Prior paid attempt in this repair is uncertain at '+name+'; no automatic resend. Preserve its request receipt.')
            original_action=action
            def action(suggested_attempt):
                if name in PAID_PHASES and previous and previous.get('status')!='repair_pending':
                    raise ValueError('Prior paid attempt in this repair is uncertain or no longer reusable at '+name+'; no automatic resend. Preserve its request receipt.')
                attempt,reservation=reserve_repair_attempt(self.base_state,self.id,name,suggested_attempt,root=ROOT)
                current=next(r for r in self.state['phases']if r['name']==name)
                current.update(attempt=attempt,attempt_reservation=reservation)
                self.save()
                result,artifacts=original_action(attempt)
                return result,[*artifacts,ROOT/reservation['path']]
            dependencies.append(ROOT/'admin_pipeline.py')
        return super().phase(name,dependencies,action,optional=optional)

    def command(self,args,timeout):
        if getattr(self,'execution_lease',None):self.execution_lease.guard('dispatch_subprocess')
        args=list(args)
        if getattr(self,'machine_review_ref',None):
            for index,value in enumerate(args):
                if Path(str(value)).name=='run_admin_stage.py':
                    args[index+1:index+1]=['--machine-review-ref',json.dumps(self.machine_review_ref,separators=(',',':'))]
                    break
        # Reconstruction feedback requests another already-audited view, not
        # another random model on exactly the same pixels. Maximum two repairs.
        if self.boundary=='select' and any(Path(str(a)).name=='stage3_rigid_select_track_views.py' for a in args):
            args+=['--candidate-offset',str(self.repair_number)]
        result=super().command(args,timeout)
        if getattr(self,'execution_lease',None):self.execution_lease.guard()
        return result


def state_result(state,*,root=ROOT,repair_execution=()):
    root=Path(root);art=state.get('artifacts',{})
    if state.get('status') not in {'complete','complete_with_gaps'} or not art.get('interaction'):
        return {'status':'pending' if state.get('status')!='cancelled' else 'failed','run_id':state['id'],
                'diagnostic':state.get('error') or 'The source run has no compiled interaction yet; preserved outputs require inspection.',
                'repair_execution':list(repair_execution)}
    import object_program
    interaction=object_program.ref(root/art['interaction'],root)
    document=json.loads((root/art['interaction']).read_text())
    paths={a['path'] for a in document['assets']}
    paths.update(a['frame_path'] for a in document['evidence'])
    for key in ['interaction','object_program','constrained_scene','part_ledger']:
        if art.get(key):paths.add(art[key])
    # No mutable run logs, indexes, request attempts or private memory in the
    # revision artifact list. Source/program validation checks full references.
    references=[object_program.ref(root/path,root) for path in sorted(paths)]
    return {'status':'complete_with_gaps','run_id':state['id'],'interaction_ref':interaction,
            'object_program_ref':object_program.ref(root/art['object_program'],root) if art.get('object_program') else None,
            'scene_ref':object_program.ref(root/art['constrained_scene'],root) if art.get('constrained_scene') else None,
            'part_ledger_ref':object_program.ref(root/art['part_ledger'],root) if art.get('part_ledger') else None,
            'artifacts':references,'repair_execution':list(repair_execution),
            'diagnostic':'Candidate requires owner inspection. Generated shape, joint parameters, metric scale and physical contact remain unverified.'}


class FixedExecutor:
    def __init__(self,*,root=ROOT,cancel_event=None):
        self.root=Path(root).resolve();self.cancel_event=cancel_event or threading.Event();self.runner=None

    def cancel(self):
        self.cancel_event.set()
        if self.runner:self.runner.cancel()

    def __call__(self,job):
        if self.root!=ROOT:raise ValueError('Production pipeline uses the fixed project root')
        if job.get('processing_consent') is not True:raise ValueError('Owner processing consent required')
        lease=job.get('execution_lease')
        if lease:lease.guard('pipeline_preflight')
        policy=job.get('improvement_policy')
        if policy:
            from admin_execution_policy import validate_policy
            validate_policy(policy,self.root)
        if job.get('evaluation_plan_ref'):
            from admin_evaluation_plan import load
            load(job['evaluation_plan_ref'],root=self.root)
        verified_ref(self.root,job['source_ref'])
        if self.cancel_event.is_set():return {'status':'failed','diagnostic':'Cancelled before model dispatch'}
        context_path=memory_snapshot(job['memory_context'],self.root)
        if job['kind']=='review':
            base=job['base_revision']['manifest']
            for reference in [base['source_ref'],base['interaction_ref'],*base['artifacts']]:verified_ref(self.root,reference)
            return {'status':'complete_with_gaps','run_id':base['run_id'],
                **{k:base.get(k) for k in ['interaction_ref','object_program_ref','scene_ref','part_ledger_ref','artifacts']},
                'repair_execution':[{'strategy':'explicit_hosted_review_only','original_artifacts_modified':False,
                                     'new_gpu_calls':0,'new_geometry_generated':False}],
                'diagnostic':'New review revision of the same unchanged artifacts; no geometry regenerated.'}
        import_recipe=private_directory(self.root,'admin-imports')/(job['run_id']+'.json')
        if job['kind']=='pipeline' and import_recipe.exists():
            # Server-created exact current public selection, never browser data.
            if import_recipe.is_symlink():raise ValueError('Invalid existing-example recipe')
            recipe=json.loads(import_recipe.read_text())
            if recipe['source_ref']!=job['source_ref']:raise ValueError('Existing example source changed')
            for ref in recipe['result']['artifacts']:verified_ref(self.root,ref)
            return recipe['result']
        repairs=[];base_state=None;boundary=None;number=0
        run_id=job['run_id'];input_path=self.root/job['input_path']
        if job['kind']=='repair':
            revision=job['base_revision']['manifest'];plan=job['repair_plan']
            base_run=revision['run_id']
            path=self.root/'data/flow-runs'/base_run/'run.json'
            if not path.is_file():
                return {'status':'pending','diagnostic':'This imported example has no resumable pipeline receipt. Upload its original recording for a fresh automatic run.'}
            base_state=json.loads(path.read_text());boundary=repair_boundary(plan['targets']);number=plan['attempt_number']
            if base_state.get('artifacts',{}).get('interaction')!=revision['interaction_ref']['path']:
                return {'status':'pending','diagnostic':'Saved run no longer selects this exact base revision; no paid repair was launched.'}
            base_program=json.loads(verified_ref(self.root,revision['interaction_ref']).read_text())
            if job['source_ref']['sha256'] not in {base_program['source']['sha256'],base_program['source']['original_sha256']}:
                raise ValueError('Repair input and base revision source differ')
            original=Path(base_state['request']['input_path'])
            if not original.is_file() or P.sha(original)!=base_state['source']['sha256']:
                return {'status':'pending','diagnostic':'Original base-run input is unavailable or changed; preserved geometry is not adopted onto another recording.'}
            input_path=original
            run_id='run-admin-repair-'+job['id'].removeprefix('job_')[:24]
            repairs=[{'targets':plan['targets'],'strategy':'verified_predecessors_then_fresh_downstream_attempts',
                      'restart_phase':boundary,'base_run_id':base_run,'model_regeneration_requested':'reconstruction'in plan['targets'],
                      'alternative_audited_view_offset':number if boundary=='select' else None,
                      'original_artifacts_modified':False,'success_not_guaranteed':True}]
        else:
            prior=self.root/'data/flow-runs'/run_id/'run.json'
            if prior.exists():base_state=json.loads(prior.read_text())
        machine_ref=None
        if job['kind']=='repair'and job['repair_plan'].get('machine_authored'):
            from admin_machine_context import snapshot
            machine_ref=snapshot(job['repair_plan']['assessment_ref'],source_sha256=job['source_ref']['sha256'],root=self.root)
        self.runner=AdminRunner(run_id,base_state=base_state,boundary=boundary,repair_number=number,execution_lease=lease,machine_review_ref=machine_ref)
        if self.cancel_event.is_set():self.runner.cancel()
        options={'overnight_task_budget_plan':policy['task_budget_plan']}if policy else{}
        state=self.runner.execute('meta_sam3d_default',input_path=input_path,
                                  memory_context_path=context_path,publication_mode='admin_review',**options)
        if repairs:repairs[0]['adopted_phases']=self.runner.adoptions
        return state_result(state,root=self.root,repair_execution=repairs)


class QueueRunner:
    """One serialized owner queue sharing the legacy worker lifecycle."""
    def __init__(self,root=ROOT):
        self.root=Path(root);self.id='admin-queue';self.state={'status':'queued'}
        self.executor=FixedExecutor(root=root);self.outcomes=[]
    def cancel(self):self.executor.cancel()
    def execute(self):
        from admin_workflow import AdminWorkflow
        from admin_muse_reviews import build_reviewer
        self.state['status']='running';workflow=AdminWorkflow(self.root)
        try:
            while not self.executor.cancel_event.is_set():
                # The factory binds each job's actual scoped context. It never
                # runs when browsing and incomplete hosted attempts never retry.
                def review_factory(job):
                    client=None
                    if job.get('improvement_policy'):
                        from overnight_budget import OvernightMetaClient
                        client=OvernightMetaClient(root=self.root,dispatch_guard=job['execution_lease'].guard)
                    return build_reviewer(job['memory_context'],root=self.root,allow_paid_api=True,client=client,
                        allow_protocol_upgrade=job.get('allow_review_protocol_upgrade',False))
                def visual_factory(job):
                    from admin_visual_audit import build_reviewer
                    return build_reviewer(job,root=self.root)
                def budget_checker(job):
                    from overnight_budget import budget_status
                    return budget_status(root=self.root)['can_admit']is True
                result=workflow.worker_once(self.executor,hosted_reviewer_factory=review_factory,
                    visual_reviewer_factory=visual_factory,budget_checker=budget_checker)
                if result['status']=='idle':
                    self.state['waiting_for_execution_lease']=result.get('reason')=='another_execution_lease_is_active'
                    break
                self.outcomes.append(result)
            self.state['status']='complete'
        except Exception as exc:self.state.update(status='failed',error=type(exc).__name__+': '+str(exc)[:500])
