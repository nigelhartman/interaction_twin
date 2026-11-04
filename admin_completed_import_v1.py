"""Explicit local adoption of a completed private pipeline into owner review.

This module has no inference/SSH/publication adapter. It does not weaken the
curated-only HTTP import endpoint. Preparation is read-only. Registration uses
an exact, non-runnable pending-consent job and leases only that job for four
deterministic structural checks; it never drains the general owner queue.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import re
import time

from agent_memory import digest,private_directory
from review_agents import data_path,verified_ref,read_interaction,review_revision

ROOT=Path(__file__).resolve().parent
FORMAT='interaction-twin-completed-private-import'
RUN=re.compile(r'run-[A-Za-z0-9_-]{1,116}\Z')


def ref(path,root=ROOT):
    root=Path(root).resolve();path=Path(path)
    if not path.is_absolute():path=root/path
    if (path.resolve()!=path or not path.is_relative_to(root)
        or not path.relative_to(root).parts or path.relative_to(root).parts[0]not in {'data','output'}
        or any(p.startswith('.')for p in path.relative_to(root).parts)
        or any(p.is_symlink()for p in [path,*path.parents])):
        raise ValueError('Only ordinary data/output evidence may be hashed')
    import object_program
    return object_program.ref(path,root)


def build(state_ref,*,root=ROOT):
    """Return an exact import plan without constructing/writing a workflow."""
    root=Path(root).resolve();path=verified_ref(root,state_ref,16*1024*1024)
    state=json.loads(path.read_text())
    run=state.get('id')
    if not isinstance(run,str)or not RUN.fullmatch(run)or state_ref['path']!=f'data/flow-runs/{run}/run.json':
        raise ValueError('Choose one exact completed pipeline run receipt')
    if state.get('status')not in {'complete','complete_with_gaps'}:
        raise ValueError('The pipeline must finish before private import; no work is resumed here')
    if any(p.get('status')in {'running','queued','repair_pending','unknown'}for p in state.get('phases',[])):
        raise ValueError('An unresolved pipeline attempt cannot be adopted')
    artifacts=state.get('artifacts',{})
    if not isinstance(artifacts,dict)or not isinstance(artifacts.get('interaction'),str):
        raise ValueError('Completed run has no compiled evidence program')
    interaction_ref=ref(data_path(root,artifacts['interaction']),root)
    doc=read_interaction(root,interaction_ref)
    source_ref=ref(data_path(root,doc['source']['path']),root)
    if source_ref['sha256']!=doc['source']['sha256']:
        raise ValueError('Normalized program recording changed')
    # Prefer a durable preserved original when the actual completed run records
    # one. A Downloads input is never copied or moved implicitly by this helper.
    original=state.get('source',{})
    if not isinstance(original,dict):raise ValueError('Completed run source receipt required')
    if isinstance(original,dict)and isinstance(original.get('path'),str):
        raw_path=Path(original['path']);absolute=raw_path if raw_path.is_absolute()else root/raw_path
        if absolute.is_relative_to(root/'data'):
            candidate=ref(data_path(root,absolute.relative_to(root).as_posix()),root)
            if candidate['sha256']!=original.get('sha256')or candidate['sha256']not in {doc['source']['sha256'],doc['source']['original_sha256']}:
                raise ValueError('Completed run source differs from the compiled program')
            source_ref=candidate
    if original.get('sha256')not in {doc['source']['sha256'],doc['source']['original_sha256']}:
        raise ValueError('Completed run and evidence program identify different recordings')
    paths={source_ref['path'],interaction_ref['path']}
    for asset in doc['assets']:
        p=data_path(root,asset['path']);actual=ref(p,root)
        if actual['sha256']!=asset['sha256']:raise ValueError('Compiled native asset changed')
        paths.add(asset['path'])
    for frame in doc['evidence']:
        paths.add(frame['frame_path'])
        paths.update(o['mask_path']for o in frame['observations'])
    companion=None;scene=None;ledger=None
    for key in ['object_program','constrained_scene','part_ledger']:
        if artifacts.get(key):
            actual=ref(data_path(root,artifacts[key]),root);paths.add(actual['path'])
            if key=='object_program':companion=actual
            elif key=='constrained_scene':scene=actual
            else:ledger=actual
    if scene:
        scene_doc=json.loads(verified_ref(root,scene,6*1024*1024).read_text())
        if scene_doc.get('original_program')!=interaction_ref or scene_doc.get('source',{}).get('sha256')!=doc['source']['sha256']:
            raise ValueError('Connected scene is bound to another exact program/source')
        for part in scene_doc.get('components',[]):
            verified_ref(root,part['asset']);paths.add(part['asset']['path'])
    if ledger:
        import part_ledger
        part_ledger.validate(json.loads(verified_ref(root,ledger,6*1024*1024).read_text()),
                             program_path=verified_ref(root,interaction_ref),root=root)
    refs=[ref(data_path(root,p),root)for p in sorted(paths)]
    if len(refs)>2048:raise ValueError('Import exceeds the fixed private artifact bound')
    result={'status':'complete_with_gaps','run_id':run,'interaction_ref':interaction_ref,'object_program_ref':companion,
            'scene_ref':scene,'part_ledger_ref':ledger,'artifacts':refs,'new_api_calls':0,'new_gpu_calls':0,
            'repair_execution':[{'strategy':'explicit_completed_private_pipeline_adoption',
                                 'completed_run_ref':state_ref,'original_artifacts_modified':False,
                                 'new_geometry_generated':False,'new_api_calls':0,'new_gpu_calls':0}],
            'diagnostic':'Completed private pipeline adopted for owner review. Four local structural checks; no hosted inference, regeneration, approval or publication.'}
    binding={**{k:result[k]for k in ['interaction_ref','object_program_ref','scene_ref','artifacts']},'source_ref':source_ref}
    review=review_revision(binding,root=root)
    if not review['all_required_roles_completed']or review['role_count']!=4 or review['blocking_failures']:
        raise ValueError('Completed evidence failed structural preflight; preserve outputs and inspect the report')
    identity={'source_ref':source_ref,'interaction_ref':interaction_ref,'object_program_ref':companion,'scene_ref':scene,'part_ledger_ref':ledger,'artifacts':refs}
    return {'format':FORMAT,'version':1,'completed_run_ref':state_ref,'candidate_run_id':'run-admin-finished-'+digest(identity)[:24],
            'source_ref':source_ref,'result':result,'structural_preflight':review,'new_api_calls':0,'new_gpu_calls':0,
            'automatic_approval':False,'automatic_publication':False,'requires_explicit_local_registration':True}


def verify(plan,*,root=ROOT):
    if not isinstance(plan,dict)or plan.get('format')!=FORMAT or plan.get('version')!=1:
        raise ValueError('Wrong completed import contract')
    current=build(plan['completed_run_ref'],root=root)
    if current!=plan:raise ValueError('Prepared completed import changed; no mutation performed')
    return current


def register(plan,*,root=ROOT,actor='nigel'):
    """Explicit owner CLI action; no HTTP handler, queue runner or model client."""
    from admin_workflow import AdminWorkflow,owner,identifier,now
    owner(actor);root=Path(root).resolve();plan=verify(plan,root=root)
    workflow=AdminWorkflow(root)
    # Write only a new exact cached-output recipe. FixedExecutor can later use
    # it after a distinct explicit owner action; this function never invokes it.
    recipe={'source_ref':plan['source_ref'],'result':plan['result']}
    raw=json.dumps(recipe,sort_keys=True,indent=2)+'\n'
    folder=private_directory(root,'admin-imports');path=folder/(plan['candidate_run_id']+'.json')
    if path.exists():
        if path.is_symlink()or path.read_text()!=raw:raise ValueError('Existing private import recipe changed')
    else:
        with path.open('x')as stream:stream.write(raw)
        path.chmod(0o600)
    submitted=workflow.submit_upload(run_id=plan['candidate_run_id'],input_path=plan['source_ref']['path'],
                                     source_ref=plan['source_ref'],processing_consent=False,actor=actor)
    candidate=submitted['candidate'];job_row=submitted['job']
    detail=workflow.detail(candidate['id'])
    if candidate['current_revision_id']:
        revision=next(r for r in detail['revisions']if r['id']==candidate['current_revision_id'])
        manifest=revision['manifest']
        if any(manifest.get(k)!=plan['result'].get(k)for k in ['interaction_ref','object_program_ref','scene_ref','part_ledger_ref','artifacts'])or manifest['source_ref']!=plan['source_ref']:
            raise ValueError('Existing candidate contains another artifact binding')
        return {'status':'reused_exact_private_revision','candidate_id':candidate['id'],'revision_id':revision['id'],
                'new_api_calls':0,'new_gpu_calls':0,'automatic_publication':False}
    token=identifier('lease')
    with workflow.db()as db:
        db.execute('BEGIN IMMEDIATE')
        live=workflow._row(db,'jobs',job_row['id'])
        if live['candidate_id']!=candidate['id']or live['kind']!='pipeline'or live['state']!='pending_consent':
            raise ValueError('Prior import is not an unattempted local-review job; no automatic retry')
        if db.execute("SELECT 1 FROM jobs WHERE state='running' AND lease_until>?",(time.time(),)).fetchone():
            raise ValueError('Another owner job is running; private import remains pending without inference')
        payload=json.loads(live['payload_json'])
        if payload.get('improvement_policy')or payload.get('repair_plan')or payload.get('source_ref')!=plan['source_ref']or payload.get('run_id')!=plan['candidate_run_id']:
            raise ValueError('Local import job gained an execution policy or foreign source')
        db.execute("UPDATE jobs SET state='running',lease_token=?,lease_until=?,updated_utc=? WHERE id=? AND state='pending_consent'",
                   (token,time.time()+300,now(),live['id']))
        db.execute("UPDATE candidates SET status='processing',updated_utc=? WHERE id=?",(now(),candidate['id']))
        job={'id':live['id'],'kind':live['kind'],'candidate_id':candidate['id'],**payload}
    try:
        # Actual model/hosted-review code is absent. The existing finalizer
        # revalidates all bytes and produces the normal four structural roles.
        summary=workflow._finalize_revision(job,copy.deepcopy(plan['result']),token,hosted_reviewer=None)
    except Exception as error:
        with workflow.db()as db:
            changed=db.execute("UPDATE jobs SET state='failed',diagnostic=?,lease_token=NULL,lease_until=NULL,updated_utc=? WHERE id=? AND state='running' AND lease_token=?",
                ('Local completed-import validation stopped: '+type(error).__name__,now(),job['id'],token)).rowcount
            if changed:db.execute("UPDATE candidates SET status='needs_attention',updated_utc=? WHERE id=?",(now(),candidate['id']))
        raise
    return {'status':'registered_private_review','candidate_id':candidate['id'],'revision_id':summary['revision_id'],
            'candidate_status':summary['candidate_status'],'structural_review':summary['review'],
            'new_api_calls':0,'new_gpu_calls':0,'automatic_approval':False,'automatic_publication':False}
