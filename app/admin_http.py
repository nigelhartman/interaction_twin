"""Authenticated owner routes. Authentication is enforced by serve_workspace.

All model work is submitted to the one serialized worker. GET is read-only.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import uuid

from admin_workflow import AdminWorkflow
from agent_memory import private_directory
from review_agents import verified_ref

ROOT=Path(__file__).resolve().parent
ID=r'[a-z]+_[a-f0-9]{32}'


def fields(body,allowed,required=()):
    if not isinstance(body,dict) or set(body)-set(allowed) or not set(required)<=set(body):
        raise ValueError('Invalid administrator request fields')


def schedule(server,root=ROOT):
    """Queueing is explicit; status polling never submits or retries jobs."""
    from admin_pipeline import QueueRunner
    with server.lock:
        pending=getattr(server,'admin_queue_pending',None)
        if pending and pending.state.get('status') in {'created','queued','running'}:
            return {'scheduled':True,'worker_id':pending.id}
        active=server.active
        waiting=bool(active and active.state.get('status') in {'created','queued','running'})
        runner=QueueRunner(root);server.admin_queue_pending=runner
        def execute():
            with server.lock:server.active=runner
            try:runner.execute()
            finally:
                with server.lock:server.admin_queue_pending=None
                # Jobs authorized while this worker was finishing remain queued;
                # failed/uncertain attempts are never selected by this wakeup.
                if not runner.state.get('waiting_for_execution_lease')and any(j['state']=='queued' for j in AdminWorkflow(root).snapshot()['jobs']):schedule(server,root)
        server.executor.submit(execute)
        return {'scheduled':True,'waiting_for_worker':waiting,'worker_id':runner.id}


def memory_status(workflow):
    entries=workflow.memory.entries();revoked=[]
    from agent_memory import digest
    for path in workflow.memory.folder.glob('revoke_*.json'):
        if path.is_symlink() or path.stat().st_size>8000:raise ValueError('Invalid memory revocation')
        value=json.loads(path.read_text())
        if path.stem!='revoke_'+digest(value):raise ValueError('Memory revocation changed')
        revoked.append(value['feedback_id'])
    return {'entries':entries,'revoked_ids':sorted(set(revoked)),'training_performed':False}


def candidate_labels(workflow,candidates):
    """Readable owner labels, derived from saved metadata rather than ID slugs."""
    catalog_path=ROOT/'showcase/public/library-catalog.json'
    catalog=json.loads(catalog_path.read_text()) if catalog_path.is_file() else {'records':[]}
    known={r['interaction']['sha256']:r.get('public_title') or r['title'] for r in catalog['records']}
    with workflow.db() as db:
        for candidate in candidates:
            row=db.execute('SELECT input_path FROM candidates WHERE id=?',(candidate['id'],)).fetchone()
            candidate['title']=Path(row['input_path']).name if row else 'Recording '+candidate['id'][-6:]
            if candidate.get('current_revision_id'):
                revision=workflow._row(db,'revisions',candidate['current_revision_id'])
                manifest=workflow._revision_row(revision)['manifest'];ref=manifest['interaction_ref']
                if ref['sha256']in known:candidate['title']=known[ref['sha256']]
                else:
                    program=json.loads(verified_ref(ROOT,ref).read_text())
                    labels=list(dict.fromkeys(t.get('label',t['id']) for t in program.get('tracklets',[])))
                    if labels:candidate['title']=' + '.join(labels[:4])[:160]
    return candidates


def existing_example(workflow,interaction_path,root=ROOT):
    """Adopt only an exact currently curated example, without running models."""
    import curated_publication as C
    import object_program
    from pipeline import list_runs
    root=Path(root);catalog=json.loads((root/'showcase/public/library-catalog.json').read_text())
    rows=[r for r in catalog['records'] if r['interaction']['path']==interaction_path]
    if len(rows)!=1:raise ValueError('Choose one exact currently curated interaction program')
    record=rows[0];program_path=verified_ref(root,record['interaction'])
    doc=json.loads(program_path.read_text());source=object_program.ref(C.local(root,doc['source']['path']),root)
    if source['sha256']!=doc['source']['sha256']:raise ValueError('Existing source recording changed')
    # Prefer the preserved original for source-scoped feedback and subsequent
    # repair. Normalized video remains the public playback/evidence source.
    for ancestor in program_path.parents:
        if not ancestor.is_relative_to(root/'data'):break
        manifest_path=ancestor/'manifest.json'
        if manifest_path.is_file():
            source_meta=json.loads(manifest_path.read_text())['source']
            original=ancestor/source_meta.get('original_file',source_meta['file'])
            original_ref=object_program.ref(C.local(root,original.relative_to(root).as_posix()),root)
            if original_ref['sha256']==doc['source']['original_sha256']:source=original_ref
            break
    run_id='run-admin-import-'+record['interaction']['sha256'][:24]
    known=next((r for r in list_runs() if r.get('artifacts',{}).get('interaction')==interaction_path),None)
    artifacts=[record['interaction'],source]
    for asset in doc['assets']:
        reference=object_program.ref(C.local(root,asset['path']),root)
        if reference['sha256']!=asset['sha256']:raise ValueError('Existing curated geometry changed')
        artifacts.append(reference)
    scene=None;companion=None
    scenes=json.loads((root/'showcase/public/aligned-scenes.json').read_text()).get('scenes',[])
    selected=next((s for s in scenes if s.get('source_sha256')==doc['source']['sha256'] and
                   s.get('original_program',{}).get('sha256',record['interaction']['sha256'])==record['interaction']['sha256']),None)
    if selected:
        copy_path=root/'showcase/public'/selected['scene_ref']['path'] if selected.get('scene_ref') else None
        if copy_path and copy_path.is_file():
            wrapper=json.loads(copy_path.read_text());scene=wrapper.get('manifest');companion=wrapper.get('object_program')
        else:
            scene=selected.get('manifest');companion=selected.get('object_program')
        for reference in [scene,companion]:
            if reference:verified_ref(root,reference);artifacts.append(reference)
    if scene:
        scene_doc=json.loads(verified_ref(root,scene).read_text())
        if scene_doc.get('original_program')!=record['interaction']:
            raise ValueError('Curated scene does not match this exact program')
    result={'status':'complete_with_gaps','run_id':known['id'] if known else run_id,
            'interaction_ref':record['interaction'],'scene_ref':scene,'object_program_ref':companion,
            'artifacts':list({r['path']:r for r in artifacts}.values()),'new_api_calls':0,
            'diagnostic':'Existing showcased example imported for private review; no regeneration performed.'}
    recipe={'source_ref':source,'result':result}
    path=private_directory(root,'admin-imports')/(run_id+'.json')
    raw=json.dumps(recipe,sort_keys=True,indent=2)+'\n'
    if path.exists():
        if path.is_symlink() or path.read_text()!=raw:raise ValueError('Existing-example import changed')
    else:
        path.write_text(raw);path.chmod(0o600)
    return workflow.submit_upload(run_id=run_id,input_path=source['path'],source_ref=source,processing_consent=False)


def dispatch(handler,route,method,body):
    if not route.startswith('/api/admin/'):return False
    workflow=AdminWorkflow(ROOT);actor=getattr(handler,'admin_actor','nigel')
    if method=='GET':
        if route=='/api/admin/snapshot':
            from scripts.serve_workspace import meta_budget_status,sam3d_runtime_status
            from task_taxonomy import TASK_CLASSES
            value=workflow.snapshot();candidate_labels(workflow,value['candidates']);active=handler.server.active
            value.update(budget=meta_budget_status(ROOT),runtime=sam3d_runtime_status(ROOT),
                task_classes=TASK_CLASSES,worker={'id':active.id,'state':active.state} if active else None)
        elif route=='/api/admin/memory':value=memory_status(workflow)
        elif (match:=re.fullmatch('/api/admin/candidates/('+ID+')',route)):
            value=workflow.detail(match[1])
            candidate_labels(workflow,[value['candidate']])
            for revision in value['revisions']:
                program=json.loads(verified_ref(ROOT,revision['manifest']['interaction_ref']).read_text())
                rows=json.loads((ROOT/'workspace/cases-index.json').read_text())['cases']
                case=next((c for c in rows if c.get('interaction_path')==revision['manifest']['interaction_ref']['path'] and
                           c.get('original_sha256')==revision['manifest']['interaction_ref']['sha256']),None)
                revision['preview_case_id']=case['id'] if case else None
                revision['video_url']='/files/'+program['source']['path']
        else:return False
        handler.json(200,value);return True
    if method!='POST':return False
    code=200;queued=False
    if route=='/api/admin/candidates':
        fields(body,{'input_path','processing_consent','object_family'},{'input_path','processing_consent'})
        if not isinstance(body['input_path'],str) or not body['input_path'].startswith('data/uploads/'):
            raise ValueError('Choose a recording from this owner upload endpoint')
        # A repeated explicit POST after a lost response reuses this exact
        # uploaded file's queue identity rather than purchasing another run.
        run='run-admin-'+hashlib.sha256(body['input_path'].encode()).hexdigest()[:24]
        from admin_execution_policy import improvement_policy
        # Fixed owner-authorized policy, never a browser-supplied cost or code plan.
        value=workflow.submit_upload(run_id=run,actor=actor,improvement_policy=improvement_policy(ROOT),**body);code=202;queued=True
    elif route=='/api/admin/import-existing':
        fields(body,{'interaction_path','processing_consent'},{'interaction_path'})
        if 'processing_consent'in body and type(body['processing_consent'])is not bool:raise ValueError('Explicit boolean consent required')
        value=existing_example(workflow,body['interaction_path'])
        if body.get('processing_consent') is True and value['candidate']['status']=='pending_consent':
            value=workflow.authorize_processing(value['candidate']['id'],actor=actor);queued=True
        code=202
    elif route=='/api/admin/process-queue':
        fields(body,set());value={'queue':schedule(handler.server)};handler.json(202,value);return True
    elif (match:=re.fullmatch('/api/admin/candidates/('+ID+')/(approve|reject|repair|process|resume|review)',route)):
        cid,action=match.groups()
        if action=='approve':
            fields(body,{'revision_id','expected_fingerprint','review_override_reason'},{'revision_id','expected_fingerprint'})
            value=workflow.approve(cid,actor=actor,**body)
            import curated_publication
            value['public_snapshot']=curated_publication.build(root=ROOT)
        elif action=='reject':
            fields(body,{'revision_id','feedback','scopes','object_family','repair_targets'},{'revision_id','feedback'})
            value=workflow.reject(cid,actor=actor,**body)
        elif action=='repair':
            fields(body,{'revision_id','plan_id','processing_consent'},{'revision_id','plan_id','processing_consent'})
            value=workflow.request_repair(cid,actor=actor,**body);code=202;queued=True
        elif action=='review':
            fields(body,{'revision_id','processing_consent'},{'revision_id','processing_consent'})
            value=workflow.request_review(cid,actor=actor,**body);code=202;queued=True
        else:
            fields(body,{'processing_consent'},{'processing_consent'})
            if body['processing_consent'] is not True:raise ValueError('Explicit processing consent required')
            value=workflow.resume_initial(cid,actor=actor,processing_consent=True) if action=='resume' else workflow.authorize_processing(cid,actor=actor)
            code=202;queued=True
    elif (match:=re.fullmatch(r'/api/admin/memory/(feedback_[a-f0-9]{64})/revoke',route)):
        fields(body,{'reason'},{'reason'});value=workflow.memory.revoke(match[1],body['reason'],actor=actor)
    else:return False
    if queued:value['queue']=schedule(handler.server)
    handler.json(code,value);return True
