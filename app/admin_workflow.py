"""Durable owner-only candidate, review, repair and approval workflow.

HTTP authentication/CSRF are the caller's responsibility. This module is private
application state, never an MCP/public mutation surface. A worker invokes only a
trusted fixed executor; no feedback/model text is interpreted as code. Original
pipeline artifacts remain untouched and publication requires a current approval.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
import uuid

from agent_memory import FeedbackMemory, digest, private_directory, write_immutable
from review_agents import review_revision, verified_ref, data_path

ROOT=Path(__file__).resolve().parent
TARGETS=('reasoning','tracking','reconstruction','mechanics')


def now():return datetime.now(timezone.utc).isoformat()
def identifier(prefix):return prefix+'_'+uuid.uuid4().hex

def owner(actor):
    if actor!='nigel':raise ValueError('Authenticated owner identity required')


def override_reason(value):
    if value is None:return None
    if not isinstance(value,str)or len(value)>2000:raise ValueError('Review override reason must be at most2000characters')
    return value.strip()or None


def file_ref(path,root):
    import object_program
    return object_program.ref(Path(path),Path(root))


class AdminWorkflow:
    timestamp=staticmethod(now)
    def __init__(self,root=ROOT):
        self.root=Path(root).resolve();self.folder=private_directory(self.root,'admin-workflow')
        self.path=self.folder/'state.sqlite3'
        if self.path.is_symlink():raise ValueError('Private workflow database cannot use a symlink')
        self.memory=FeedbackMemory(self.root)
        with self.db() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS candidates(
              id TEXT PRIMARY KEY,run_id TEXT UNIQUE NOT NULL,input_path TEXT NOT NULL,source_json TEXT NOT NULL,
              object_family TEXT,status TEXT NOT NULL,processing_consent INTEGER NOT NULL,current_revision_id TEXT,
              created_utc TEXT NOT NULL,updated_utc TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs(
              id TEXT PRIMARY KEY,candidate_id TEXT NOT NULL,kind TEXT NOT NULL,state TEXT NOT NULL,
              payload_json TEXT NOT NULL,result_json TEXT,diagnostic TEXT,lease_token TEXT,lease_until REAL,
              created_utc TEXT NOT NULL,updated_utc TEXT NOT NULL,
              FOREIGN KEY(candidate_id) REFERENCES candidates(id));
            CREATE TABLE IF NOT EXISTS revisions(
              id TEXT PRIMARY KEY,candidate_id TEXT NOT NULL,number INTEGER NOT NULL,manifest_path TEXT NOT NULL,
              manifest_sha256 TEXT NOT NULL,fingerprint TEXT NOT NULL,created_utc TEXT NOT NULL,
              UNIQUE(candidate_id,number),FOREIGN KEY(candidate_id) REFERENCES candidates(id));
            CREATE TABLE IF NOT EXISTS decisions(
              id TEXT PRIMARY KEY,candidate_id TEXT NOT NULL,revision_id TEXT NOT NULL,action TEXT NOT NULL,
              record_path TEXT NOT NULL,created_utc TEXT NOT NULL,
              FOREIGN KEY(candidate_id) REFERENCES candidates(id),FOREIGN KEY(revision_id) REFERENCES revisions(id));
            CREATE TABLE IF NOT EXISTS repair_plans(
              id TEXT PRIMARY KEY,candidate_id TEXT NOT NULL,revision_id TEXT NOT NULL,record_path TEXT NOT NULL,
              created_utc TEXT NOT NULL,FOREIGN KEY(candidate_id) REFERENCES candidates(id));
            CREATE TABLE IF NOT EXISTS improvement_rounds(
              job_id TEXT PRIMARY KEY,candidate_id TEXT NOT NULL,revision_id TEXT NOT NULL,
              round_number INTEGER NOT NULL,record_path TEXT NOT NULL,created_utc TEXT NOT NULL,
              FOREIGN KEY(candidate_id) REFERENCES candidates(id),FOREIGN KEY(revision_id) REFERENCES revisions(id));
            CREATE TABLE IF NOT EXISTS improvement_stops(
              job_id TEXT PRIMARY KEY,candidate_id TEXT NOT NULL,record_path TEXT NOT NULL,created_utc TEXT NOT NULL,
              FOREIGN KEY(candidate_id) REFERENCES candidates(id));
            ''')
            columns={r['name'] for r in db.execute('PRAGMA table_info(jobs)')}
            for column in ['heartbeat_utc','progress_json']:
                if column not in columns:db.execute('ALTER TABLE jobs ADD COLUMN '+column+' TEXT')
        self.path.chmod(0o600)

    @contextmanager
    def db(self):
        connection=sqlite3.connect(self.path,timeout=10)
        connection.row_factory=sqlite3.Row
        connection.execute('PRAGMA foreign_keys=ON');connection.execute('PRAGMA journal_mode=WAL')
        try:
            yield connection;connection.commit()
        except BaseException:
            connection.rollback();raise
        finally:connection.close()

    @staticmethod
    def _candidate(row):
        return {'id':row['id'],'run_id':row['run_id'],'status':row['status'],'source_sha256':json.loads(row['source_json'])['sha256'],
            'object_family':row['object_family'],'processing_consent':bool(row['processing_consent']),
            'current_revision_id':row['current_revision_id'],'created_utc':row['created_utc'],'updated_utc':row['updated_utc']}

    @staticmethod
    def _job(row):
        return {'id':row['id'],'candidate_id':row['candidate_id'],'kind':row['kind'],'state':row['state'],
            'created_utc':row['created_utc'],'updated_utc':row['updated_utc'],'diagnostic':row['diagnostic'],
            'result':json.loads(row['result_json']) if row['result_json'] else None,
            'heartbeat_utc':row['heartbeat_utc'],
            'progress':json.loads(row['progress_json']) if row['progress_json'] else None}

    def _private_record(self,path):
        p=self.root/path
        if p.resolve()!=p or not p.is_relative_to(self.folder) or p.is_symlink() or not p.is_file() or p.stat().st_size>6*1024*1024:
            raise ValueError('Private workflow record unavailable')
        value=json.loads(p.read_text())
        checksum=p.stem.rsplit('_',1)[-1]
        if not re.fullmatch('[0-9a-f]{64}',checksum) or digest(value)!=checksum:raise ValueError('Immutable workflow record changed')
        return value

    def _save_record(self,prefix,value):
        record=write_immutable(self.folder,prefix,value)
        return str((self.folder/(record+'.json')).relative_to(self.root))

    def _row(self,db,table,record_id):
        if table not in {'candidates','revisions','repair_plans','jobs'}:raise ValueError('Invalid private record table')
        row=db.execute('SELECT * FROM '+table+' WHERE id=?',(record_id,)).fetchone()
        if row is None:raise ValueError('Unknown workflow '+table.rstrip('s'))
        return row

    def _enqueue(self,db,candidate,kind,payload,consent):
        job_id=identifier('job');stamp=now();state='queued' if consent else 'pending_consent'
        db.execute('INSERT INTO jobs(id,candidate_id,kind,state,payload_json,created_utc,updated_utc) VALUES(?,?,?,?,?,?,?)',
            (job_id,candidate['id'],kind,state,json.dumps(payload),stamp,stamp))
        return self._job(self._row(db,'jobs',job_id))

    def submit_upload(self,*,run_id,input_path,source_ref=None,processing_consent=False,actor='nigel',object_family=None,improvement_policy=None):
        owner(actor)
        if not isinstance(run_id,str) or not re.fullmatch('[A-Za-z0-9_-]{1,120}',run_id):raise ValueError('Invalid pipeline run ID')
        if type(processing_consent)is not bool:raise ValueError('Explicit boolean processing consent required')
        if improvement_policy is not None:
            from admin_execution_policy import validate_policy
            validate_policy(improvement_policy,self.root)
        from task_taxonomy import TASK_CLASSES
        if object_family is not None and object_family not in TASK_CLASSES:raise ValueError('Unknown object family')
        p=data_path(self.root,input_path)
        if not p.is_file() or not 0<p.stat().st_size<=512*1024*1024:raise ValueError('Uploaded recording unavailable or oversized')
        actual=file_ref(p,self.root)
        if source_ref is not None and source_ref!=actual:raise ValueError('Uploaded recording reference changed')
        candidate_id=identifier('candidate');stamp=now()
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            existing=db.execute('SELECT * FROM candidates WHERE run_id=?',(run_id,)).fetchone()
            if existing:
                if json.loads(existing['source_json'])!=actual:raise ValueError('Run ID already belongs to another recording')
                job=db.execute('SELECT * FROM jobs WHERE candidate_id=? ORDER BY created_utc LIMIT 1',(existing['id'],)).fetchone()
                return {'candidate':self._candidate(existing),'job':self._job(job)}
            db.execute('INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?)',
                (candidate_id,run_id,input_path,json.dumps(actual),object_family,'queued' if processing_consent else 'pending_consent',
                 int(processing_consent),None,stamp,stamp))
            candidate=self._row(db,'candidates',candidate_id)
            payload={'run_id':run_id,'input_path':input_path,'source_ref':actual,'processing_consent':processing_consent,
                     'base_revision':None,'repair_plan':None,'memory_context':self.memory.retrieve(source_sha256=actual['sha256'],object_family=object_family)}
            if improvement_policy is not None:payload.update(improvement_policy=improvement_policy,improvement_round=0)
            job=self._enqueue(db,candidate,'pipeline',payload,processing_consent)
            return {'candidate':self._candidate(candidate),'job':job}

    def authorize_processing(self,candidate_id,*,actor='nigel'):
        owner(actor)
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE');candidate=self._row(db,'candidates',candidate_id)
            job=db.execute("SELECT * FROM jobs WHERE candidate_id=? AND state='pending_consent' ORDER BY created_utc LIMIT 1",(candidate_id,)).fetchone()
            if job is None:raise ValueError('No pending consent job for this candidate')
            payload=json.loads(job['payload_json']);payload['processing_consent']=True
            payload['memory_context']=self.memory.retrieve(source_sha256=json.loads(candidate['source_json'])['sha256'],object_family=candidate['object_family'])
            db.execute("UPDATE jobs SET state='queued',payload_json=?,updated_utc=? WHERE id=?",(json.dumps(payload),now(),job['id']))
            db.execute("UPDATE candidates SET status='queued',processing_consent=1,updated_utc=? WHERE id=?",(now(),candidate_id))
            return {'candidate':self._candidate(self._row(db,'candidates',candidate_id)),'job':self._job(self._row(db,'jobs',job['id']))}

    def snapshot(self):
        with self.db() as db:
            candidates=[self._candidate(r) for r in db.execute('SELECT * FROM candidates ORDER BY created_utc DESC LIMIT 200')]
            jobs=[self._job(r) for r in db.execute('SELECT * FROM jobs ORDER BY created_utc DESC LIMIT 200')]
        return {'format':'interaction-twin-private-workflow','version':1,'candidates':candidates,'jobs':jobs,
            'publication_requires_nigel_approval':True,'feedback_is_private':True,'training_performed':False}

    def detail(self,candidate_id):
        with self.db() as db:
            candidate=self._candidate(self._row(db,'candidates',candidate_id))
            revisions=[self._revision_row(r) for r in db.execute('SELECT * FROM revisions WHERE candidate_id=? ORDER BY number DESC',(candidate_id,))]
            jobs=[self._job(r) for r in db.execute('SELECT * FROM jobs WHERE candidate_id=? ORDER BY created_utc DESC',(candidate_id,))]
            plans=[{'id':r['id'],**self._private_record(r['record_path'])} for r in db.execute('SELECT * FROM repair_plans WHERE candidate_id=? ORDER BY created_utc DESC',(candidate_id,))]
            decisions=[self._private_record(r['record_path']) for r in db.execute('SELECT * FROM decisions WHERE candidate_id=? ORDER BY created_utc',(candidate_id,))]
            rounds=[self._private_record(r['record_path']) for r in db.execute('SELECT * FROM improvement_rounds WHERE candidate_id=? ORDER BY round_number,created_utc',(candidate_id,))]
            stops=[self._private_record(r['record_path']) for r in db.execute('SELECT * FROM improvement_stops WHERE candidate_id=? ORDER BY created_utc',(candidate_id,))]
        return {'candidate':candidate,'revisions':revisions,'jobs':jobs,'repair_plans':plans,'decisions':decisions,
            'feedback':[e for e in self.memory.entries() if e['candidate_id']==candidate_id], 'improvement_rounds':rounds,
            'best_revision_id':rounds[-1]['best_revision_id'] if rounds else candidate['current_revision_id'], 'improvement_stops':stops}

    def _revision_row(self,row):
        record=self._private_record(row['manifest_path'])
        if digest(record['manifest'])!=row['fingerprint'] or record.get('fingerprint')!=row['fingerprint']:
            raise ValueError('Private revision fingerprint changed')
        path=self.root/row['manifest_path']
        if file_ref(path,self.root)['sha256']!=row['manifest_sha256']:raise ValueError('Private revision manifest changed')
        return {'id':row['id'],'number':row['number'],'fingerprint':row['fingerprint'],
                'manifest_ref':file_ref(path,self.root),'manifest':record['manifest'],'created_utc':row['created_utc']}

    def _verify_revision(self,row,*,allow_hosted_override=False):
        if type(allow_hosted_override)is not bool:raise ValueError('Explicit hosted finding override required')
        revision=self._revision_row(row);manifest=revision['manifest']
        for item in [manifest['source_ref'],manifest['interaction_ref'],*manifest['artifacts']]:verified_ref(self.root,item)
        for key in ['object_program_ref','scene_ref','part_ledger_ref']:
            if manifest.get(key):verified_ref(self.root,manifest[key],6*1024*1024)
        original=manifest['review'];roles=original.get('roles',[])
        if manifest.get('visual_review'):
            from admin_execution_policy import review_assessment
            review_assessment(manifest['visual_review'],manifest,root=self.root)
        from review_agents import ROLES
        if (original.get('all_required_roles_completed')is not True or len(roles)!=len(ROLES)or
            {r.get('role')for r in roles}!=set(ROLES)or any(r.get('status')not in {'pass','warning','fail'}for r in roles)):
            raise ValueError('Original automatic reviewer roles did not complete successfully')
        structural_failures=sum(r['status']=='fail'for r in roles)
        if structural_failures:raise ValueError('Structural review failures cannot be overridden')
        hosted_failures=0
        for hosted in original.get('hosted_roles',[]):
            from admin_muse_reviews import verify_report
            verify_report(hosted,manifest,root=self.root)
            hosted_failures+=hosted['verdict']=='blocked'or any(i['severity']=='blocking'for i in hosted['issues'])
        if original.get('blocking_failures')!=structural_failures+hosted_failures:
            raise ValueError('Saved review failure counts do not match actual role findings')
        if hosted_failures and not allow_hosted_override:
            raise ValueError('Hosted review findings require an explicit owner override reason')
        review=review_revision(manifest,root=self.root)
        if review['blocking_failures']:raise ValueError('Revision has blocking structural review failures')
        revision['hosted_blocking_findings']=hosted_failures
        return revision

    def _finalize_revision(self,job,result,lease_token,hosted_reviewer=None,visual_review=None):
        if not result.get('interaction_ref'):raise ValueError('Completed pipeline needs an exact interaction program reference')
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            live=self._row(db,'jobs',job['id'])
            if live['state']!='running' or live['lease_token']!=lease_token or (live['lease_until']or 0)<=time.time():raise ValueError('Worker lease changed; revision not accepted')
            candidate=self._row(db,'candidates',job['candidate_id'])
            base=job.get('base_revision')
            if base and candidate['current_revision_id']!=base['id']:raise ValueError('Repair result is based on a stale revision')
            number=db.execute('SELECT COALESCE(MAX(number),0)+1 FROM revisions WHERE candidate_id=?',(candidate['id'],)).fetchone()[0]
            revision_id=identifier('revision')
            artifacts=result.get('artifacts',[])
            if not isinstance(artifacts,list) or len(artifacts)>2048:raise ValueError('Invalid bounded revision artifacts')
            manifest={'format':'interaction-twin-private-candidate-revision','version':1,'candidate_id':candidate['id'],
                'revision_id':revision_id,'number':number,'run_id':result.get('run_id',candidate['run_id']),
                'source_ref':json.loads(candidate['source_json']),'interaction_ref':result['interaction_ref'],
                'object_program_ref':result.get('object_program_ref'),'scene_ref':result.get('scene_ref'),
                'part_ledger_ref':result.get('part_ledger_ref'),
                'artifacts':artifacts,'base_revision_id':base['id'] if base else None,'job_id':job['id'],
                'memory_context_sha256':job['memory_context']['context_sha256'],
                'human_guidance_used':bool(job['memory_context']['lessons']),
                'training_performed':False,'memory_only_not_weight_training':True,
                'repair_execution':result.get('repair_execution',[]),'pipeline_status':result['status'],
                'created_utc':now(),'physical_accuracy_verified':False,'simulation_ready':False}
            reviews=review_revision(manifest,root=self.root,hosted_reviewer=hosted_reviewer,
                processing_consent=job.get('processing_consent')is True)
            if visual_review is not None:
                from admin_execution_policy import review_assessment
                review_assessment(visual_review,manifest,root=self.root)
                manifest['visual_review']=visual_review
            manifest['review']=reviews;manifest['task_classes']=reviews['task_classes']
            fingerprint=digest(manifest);path=self._save_record('revision',{'manifest':manifest,'fingerprint':fingerprint})
            db.execute('INSERT INTO revisions VALUES(?,?,?,?,?,?,?)',
                (revision_id,candidate['id'],number,path,file_ref(self.root/path,self.root)['sha256'],fingerprint,manifest['created_utc']))
            status='blocked_review' if reviews['blocking_failures'] else 'awaiting_review'
            db.execute('UPDATE candidates SET status=?,current_revision_id=?,updated_utc=? WHERE id=?',(status,revision_id,now(),candidate['id']))
            summary={'revision_id':revision_id,'fingerprint':fingerprint,'candidate_status':status,'review':reviews}
            db.execute("UPDATE jobs SET state='complete',result_json=?,diagnostic=?,lease_token=NULL,lease_until=NULL,updated_utc=? WHERE id=?",
                (json.dumps({**result,'revision':summary}),str(result.get('diagnostic',''))[:2000],now(),job['id']))
            return summary

    def worker_once(self,executor,*,hosted_reviewer=None,hosted_reviewer_factory=None,visual_reviewer_factory=None,budget_checker=None):
        """Claim one durable job and invoke only the caller's trusted fixed adapter."""
        if not callable(executor):raise ValueError('A trusted fixed executor is required')
        if hosted_reviewer is not None and hosted_reviewer_factory is not None:raise ValueError('Choose one fixed hosted-review adapter')
        if hosted_reviewer_factory is not None and not callable(hosted_reviewer_factory):raise ValueError('Trusted hosted-review factory required')
        if visual_reviewer_factory is not None and not callable(visual_reviewer_factory):raise ValueError('Trusted visual-review factory required')
        self.reconcile_improvements(budget_checker=budget_checker)
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT 1 FROM jobs WHERE state='running' AND lease_until>?",(time.time(),)).fetchone():
                return {'status':'idle','reason':'another_execution_lease_is_active'}
            row=db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY created_utc,id LIMIT 1").fetchone()
            if row is None:return {'status':'idle'}
            token=identifier('lease');payload=json.loads(row['payload_json'])
            if payload.get('processing_consent')is not True:raise ValueError('Queued processing lacks explicit admin consent')
            db.execute("UPDATE jobs SET state='running',lease_token=?,lease_until=?,updated_utc=? WHERE id=?",
                (token,time.time()+300,now(),row['id']))
            db.execute("UPDATE candidates SET status='processing',updated_utc=? WHERE id=?",(now(),row['candidate_id']))
            job={'id':row['id'],'kind':row['kind'],'candidate_id':row['candidate_id'],**payload}
        lease=None
        try:
            from admin_execution_policy import WorkerLease,validate_policy
            lease=WorkerLease(self,job['id'],token,cancel=getattr(executor,'cancel',None),cancel_event=getattr(executor,'cancel_event',None)).start()
            job['execution_lease']=lease
            if job.get('improvement_policy'):validate_policy(job['improvement_policy'],self.root)
            lease.guard('verify_source')
            verified_ref(self.root,job['source_ref'])
            result=executor(job)
            if not isinstance(result,dict) or result.get('status')not in {'completed','complete_with_gaps','pending','failed'}:
                raise ValueError('Trusted executor returned an invalid workflow result')
            revision=None
            if result['status']in {'completed','complete_with_gaps'}:
                artifacts=result.get('artifacts',[])
                if not isinstance(artifacts,list)or len(artifacts)>2048:raise ValueError('Invalid bounded revision artifacts')
                if hosted_reviewer_factory is not None:
                    try:hosted_reviewer=hosted_reviewer_factory(job)
                    except Exception as exc:
                        from review_agents import ROLES,hosted_error
                        factory_errors={role:hosted_error(role,exc,code='adapter_factory_unavailable')for role in ROLES}
                        def hosted_reviewer(role,manifest):return factory_errors[role]
                reviewed_callback=None
                if hosted_reviewer is not None:
                    # Hosted transport must not hold the SQLite writer lock.
                    # Binding uses immutable artifact refs, never random job or
                    # revision IDs, so interrupted recovery reuses paid reports.
                    binding={'source_ref':job['source_ref'],'artifacts':artifacts,
                        **{k:result.get(k)for k in ['interaction_ref','object_program_ref','scene_ref']}}
                    adapter=hosted_reviewer
                    def fenced_review(role,manifest):
                        lease.guard('review_'+role)
                        value=adapter(role,manifest)
                        lease.guard()
                        return value
                    preflight=review_revision(binding,root=self.root,hosted_reviewer=fenced_review,processing_consent=True)
                    reports={r['role']:r for r in preflight['hosted_roles']}
                    errors={r['role']:r for r in preflight['hosted_errors']}
                    def reviewed_callback(role,manifest):
                        from admin_muse_reviews import artifact_binding
                        if artifact_binding(manifest)!=artifact_binding(binding):raise ValueError('Hosted preflight artifact binding changed')
                        return reports[role]if role in reports else errors[role]
                visual=None
                if job.get('improvement_policy'):
                    lease.guard('visual_review')
                    binding={'source_ref':job['source_ref'],'artifacts':artifacts,
                        **{k:result.get(k)for k in ['interaction_ref','object_program_ref','scene_ref']}}
                    structural=review_revision(binding,root=self.root)
                    if structural['blocking_failures']:
                        visual={'status':'unavailable','diagnostic':'Structural evidence review failed before visual dispatch'}
                    elif hosted_reviewer is not None and preflight['hosted_review_status']!='complete':
                        visual={'status':'unavailable','diagnostic':'Hosted text reviews incomplete; no visual dispatch'}
                    elif visual_reviewer_factory is not None:
                        try:visual=visual_reviewer_factory(job)(binding)
                        except Exception as exc:
                            visual={'status':'unavailable','diagnostic':type(exc).__name__}
                    else:visual={'status':'unavailable','diagnostic':'No authorized visual packet adapter'}
                lease.guard('finalize_revision')
                lease.close()  # No heartbeat may mistake our own completed commit for lease loss.
                revision=self._finalize_revision(job,result,token,hosted_reviewer=reviewed_callback,visual_review=visual)
                if job.get('improvement_policy'):
                    self.advance_improvement(job['id'],budget_checker=budget_checker)
                return {'status':'complete','job_id':job['id'],'candidate_id':job['candidate_id'],'result':result,'revision':revision}
            state='complete' if revision else ('pending' if result['status']=='pending' else 'failed')
            lease.guard('preserve_terminal_output');lease.close()
            with self.db() as db:
                changed=db.execute('UPDATE jobs SET state=?,result_json=?,diagnostic=?,lease_token=NULL,lease_until=NULL,updated_utc=? WHERE id=? AND lease_token=? AND state=?',
                    (state,json.dumps({**result,**({'revision':revision} if revision else {})}),str(result.get('diagnostic',''))[:2000],now(),job['id'],token,'running')).rowcount
                if changed!=1:raise ValueError('Worker lease changed; result not accepted')
                if not revision:db.execute('UPDATE candidates SET status=?,updated_utc=? WHERE id=?',('needs_attention',now(),job['candidate_id']))
            if job.get('improvement_policy'):self.record_improvement_stop(job['id'],state)
            return {'status':state,'job_id':job['id'],'candidate_id':job['candidate_id'],'result':result,'revision':revision}
        except Exception as error:
            with self.db() as db:
                changed=db.execute("UPDATE jobs SET state='failed',diagnostic=?,lease_token=NULL,lease_until=NULL,updated_utc=? WHERE id=? AND lease_token=?",
                    (type(error).__name__+': '+str(error)[:1500],now(),job['id'],token)).rowcount
                if changed:db.execute("UPDATE candidates SET status='needs_attention',updated_utc=? WHERE id=?",(now(),job['candidate_id']))
            if changed and job.get('improvement_policy'):self.record_improvement_stop(job['id'],'failed')
            return {'status':'failed','job_id':job['id'],'candidate_id':job['candidate_id'],'error_type':type(error).__name__}
        finally:
            if lease:lease.close()

    def reconcile_improvements(self,*,budget_checker=None):
        """Finish an interrupted post-review decision without repeating inference."""
        with self.db() as db:
            rows=list(db.execute("SELECT id,payload_json FROM jobs WHERE state='complete' AND id NOT IN(SELECT job_id FROM improvement_rounds)"))
        for row in rows:
            if json.loads(row['payload_json']).get('improvement_policy'):
                try:self.advance_improvement(row['id'],budget_checker=budget_checker)
                except Exception as exc:
                    # An invalid private candidate must not wedge every other
                    # authorized upload. Preserve its completed artifacts and
                    # stop only this candidate without retrying any inference.
                    with self.db() as db:
                        job=self._row(db,'jobs',row['id'])
                        db.execute("UPDATE jobs SET state='failed',diagnostic=?,updated_utc=? WHERE id=? AND state='complete'",
                            ('Improvement reconciliation stopped: '+type(exc).__name__+'; completed artifacts preserved, no automatic retry.',now(),row['id']))
                        db.execute("UPDATE candidates SET status='needs_attention',updated_utc=? WHERE id=?",(now(),job['candidate_id']))
                    self.record_improvement_stop(row['id'],'failed')
        with self.db() as db:
            stopped=list(db.execute("SELECT id,payload_json,state FROM jobs WHERE state IN('failed','pending','interrupted') AND id NOT IN(SELECT job_id FROM improvement_stops)"))
        for row in stopped:
            if json.loads(row['payload_json']).get('improvement_policy'):self.record_improvement_stop(row['id'],row['state'])

    def record_improvement_stop(self,job_id,state):
        """Immutable failed/unknown attempt record, without a fabricated revision."""
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            prior=db.execute('SELECT * FROM improvement_stops WHERE job_id=?',(job_id,)).fetchone()
            if prior:return self._private_record(prior['record_path'])
            row=self._row(db,'jobs',job_id);job=json.loads(row['payload_json'])
            if row['state']not in {'failed','pending','interrupted'}:raise ValueError('Only a terminal private attempt may be recorded')
            candidate=self._row(db,'candidates',row['candidate_id'])
            latest=db.execute('SELECT * FROM improvement_rounds WHERE candidate_id=? ORDER BY round_number DESC LIMIT 1',(candidate['id'],)).fetchone()
            best=self._private_record(latest['record_path'])['best_revision_id']if latest else candidate['current_revision_id']
            value={'format':'interaction-twin-immutable-improvement-stop','version':1,'job_id':job_id,'candidate_id':candidate['id'],
                'round_number':job.get('improvement_round',0),'policy':job.get('improvement_policy'),'state':state,
                'diagnostic':str(row['diagnostic']or'')[:1500],'best_revision_id':best,'no_automatic_retry':True,
                'automatic_approval':False,'automatic_publication':False,'created_utc':now()}
            path=self._save_record('stop',value)
            db.execute('INSERT INTO improvement_stops VALUES(?,?,?,?)',(job_id,candidate['id'],path,now()))
            return value

    def advance_improvement(self,job_id,*,budget_checker=None):
        """Persist one measured decision and at most one allowlisted repair.

        Machine findings never call reject(), author owner feedback, approve(),
        or publish. The last approved public revision remains untouched.
        """
        from admin_execution_policy import validate_policy,review_assessment,improves
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            prior=db.execute('SELECT * FROM improvement_rounds WHERE job_id=?',(job_id,)).fetchone()
            if prior:return self._private_record(prior['record_path'])
            row=self._row(db,'jobs',job_id);job=json.loads(row['payload_json'])
            policy=job['improvement_policy'];policy_current=True
            try:validate_policy(policy,self.root)
            except ValueError:policy_current=False
            outcome=json.loads(row['result_json']or'{}');summary=outcome.get('revision')
            if row['state']!='complete' or not summary:raise ValueError('Improvement needs an exact completed reviewed revision')
            revision=self._revision_row(self._row(db,'revisions',summary['revision_id']))
            manifest=revision['manifest'];candidate=self._row(db,'candidates',row['candidate_id'])
            if manifest['job_id']!=job_id:raise ValueError('Automatic improvement job/revision binding differs')
            superseded=candidate['current_revision_id']!=revision['id']
            external=superseded or candidate['status']not in {'awaiting_review','blocked_review'} or bool(db.execute(
                "SELECT 1 FROM jobs WHERE candidate_id=? AND state IN('queued','running')",(candidate['id'],)).fetchone())
            rounds=[self._private_record(r['record_path']) for r in db.execute('SELECT * FROM improvement_rounds WHERE candidate_id=? ORDER BY round_number',(candidate['id'],))]
            best=rounds[-1]['best_revision_id'] if rounds else candidate['current_revision_id']
            best_metrics=rounds[-1].get('best_metrics') if rounds else None
            best_comparison_key=rounds[-1].get('best_comparison_key') if rounds else None
            assessment=review_assessment(manifest.get('visual_review'),manifest,root=self.root)
            reason='unknown_visual_quality';accepted=False
            if assessment and not external and policy_current:
                accepted=best_metrics is None or improves(assessment['metrics'],best_metrics,policy['minimum_improvement'])
                if best_comparison_key is not None and best_comparison_key!=assessment['comparison_key']:accepted=False
                if rounds:
                    from admin_muse_reviews import artifact_binding
                    best_manifest=self._revision_row(self._row(db,'revisions',best))['manifest']
                    if artifact_binding(best_manifest)==artifact_binding(manifest):accepted=False
                if accepted:best=revision['id'];best_metrics=assessment['metrics'];best_comparison_key=assessment['comparison_key']
                reason='repair_candidate' if accepted else 'plateau_or_regression'
            review=manifest['review'];repair_count=db.execute("SELECT COUNT(*) FROM jobs WHERE candidate_id=? AND kind='repair'",(candidate['id'],)).fetchone()[0]
            if not policy_current:reason='authorization_policy_changed'
            elif external:reason='human_or_external_decision_preempted'
            elif any(r['status']=='fail'for r in review['roles']):reason='structural_review_failed'
            elif review.get('hosted_review_status')!='complete':reason='review_incomplete'
            elif assessment and accepted:
                if not assessment['automatic_repair_safe']:reason='repair_requires_human_inspection'
                elif not assessment['targets']:reason='no_supported_repair_needed'
                elif repair_count>=policy['maximum_repairs']:reason='round_limit_reached'
                elif budget_checker is None:reason='budget_admission_unavailable'
                else:
                    try:
                        if budget_checker(job) is not True:reason='budget_exhausted'
                    except Exception:reason='budget_or_usage_unknown'
            next_job=None
            if reason=='repair_candidate':
                plan={'format':'interaction-twin-bounded-repair-plan','version':2,'candidate_id':candidate['id'],
                    'revision_id':revision['id'],'base_fingerprint':revision['fingerprint'],
                    'machine_authored':True,'human_feedback_authored':False,'targets':assessment['targets'],
                    'assessment_ref':assessment['report_ref'],'attempt_number':repair_count+1,'maximum_attempts':2,
                    'existing_spending_guards_required':True,'created_utc':now()}
                plan_id=identifier('repairplan');path=self._save_record('repairplan',plan)
                db.execute('INSERT INTO repair_plans VALUES(?,?,?,?,?)',(plan_id,candidate['id'],revision['id'],path,now()))
                payload={k:job[k]for k in ['run_id','input_path','source_ref','memory_context','improvement_policy']}
                payload.update(processing_consent=True,base_revision=revision,repair_plan={'id':plan_id,**plan},
                    improvement_round=job.get('improvement_round',0)+1)
                if assessment.get('evaluation_plan_ref'):payload['evaluation_plan_ref']=assessment['evaluation_plan_ref']
                next_job=self._enqueue(db,candidate,'repair',payload,True)
                db.execute("UPDATE candidates SET status='queued_repair',updated_utc=? WHERE id=?",(now(),candidate['id']))
            elif not external and policy_current:
                best_row=self._row(db,'revisions',best);best_review=self._revision_row(best_row)['manifest']['review']
                db.execute('UPDATE candidates SET current_revision_id=?,status=?,updated_utc=? WHERE id=?',
                    (best,'blocked_review'if best_review['blocking_failures']else'awaiting_review',now(),candidate['id']))
            record={'format':'interaction-twin-immutable-improvement-round','version':1,'job_id':job_id,
                'candidate_id':candidate['id'],'revision_id':revision['id'],'round_number':job.get('improvement_round',0),
                'policy':policy,'assessment':assessment,'decision':reason,'best_revision_id':best,'best_metrics':best_metrics,'best_comparison_key':best_comparison_key,
                'measured_improvement':accepted,'next_job_id':next_job['id']if next_job else None,
                'human_feedback_authored':False,'automatic_approval':False,'automatic_publication':False,'created_utc':now()}
            path=self._save_record('round',record)
            db.execute('INSERT INTO improvement_rounds VALUES(?,?,?,?,?,?)',(job_id,candidate['id'],revision['id'],record['round_number'],path,now()))
            return record

    def recover_expired_jobs(self):
        """Interrupted work stays pending review; never silently repeat a paid phase."""
        with self.db() as db:
            rows=list(db.execute("SELECT * FROM jobs WHERE state='running' AND lease_until<?",(time.time(),)))
            for row in rows:
                db.execute("UPDATE jobs SET state='interrupted',diagnostic='Worker lease expired; inspect preserved pipeline caches before explicit recovery.',lease_token=NULL,lease_until=NULL,updated_utc=? WHERE id=?",(now(),row['id']))
                db.execute("UPDATE candidates SET status='needs_attention',updated_utc=? WHERE id=?",(now(),row['candidate_id']))
        return len(rows)

    def resume_initial(self,candidate_id,*,processing_consent=True,actor='nigel'):
        """Explicit owner recovery of the SAME initial run and paid-cache identity."""
        owner(actor)
        if processing_consent is not True:raise ValueError('Initial recovery requires explicit processing consent')
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE');candidate=self._row(db,'candidates',candidate_id)
            if candidate['current_revision_id'] is not None:raise ValueError('A reviewed candidate must use a bounded revision repair plan')
            if db.execute("SELECT 1 FROM jobs WHERE candidate_id=? AND state IN('queued','running')",(candidate_id,)).fetchone():
                raise ValueError('Candidate already has an active job')
            jobs=list(db.execute("SELECT * FROM jobs WHERE candidate_id=? AND kind='pipeline' ORDER BY created_utc,id",(candidate_id,)))
            if not jobs or jobs[-1]['state']not in {'pending','failed','interrupted'}:raise ValueError('No failed/pending initial run to resume')
            if len(jobs)>=3:raise ValueError('Initial recovery limit reached; inspect preserved run manually')
            previous=jobs[-1];payload=json.loads(previous['payload_json'])
            if payload['run_id']!=candidate['run_id'] or payload['source_ref']!=json.loads(candidate['source_json']):
                raise ValueError('Initial recovery source/run identity changed')
            # Preserve exact prior reasoning context and all run/cache paths;
            # new global feedback cannot invalidate completed paid phases here.
            payload.update(processing_consent=True,resume_of_job_id=previous['id'],
                resume_policy='same_run_preserve_completed_paid_caches_stop_on_ambiguous_attempt')
            job=self._enqueue(db,candidate,'pipeline',payload,True)
            db.execute("UPDATE candidates SET status='queued_resume',processing_consent=1,updated_utc=? WHERE id=?",(now(),candidate_id))
            return {'candidate':self._candidate(self._row(db,'candidates',candidate_id)),'job':job}

    def request_review(self,candidate_id,revision_id,*,processing_consent=False,allow_protocol_upgrade=False,actor='nigel'):
        """Explicit new review revision; never regenerate assets or hide a retry."""
        owner(actor)
        if processing_consent is not True or type(allow_protocol_upgrade)is not bool:
            raise ValueError('Explicit owner consent required for a new hosted review')
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE');candidate=self._row(db,'candidates',candidate_id)
            revision=self._row(db,'revisions',revision_id)
            if revision['candidate_id']!=candidate_id or candidate['current_revision_id']!=revision_id:
                raise ValueError('New reviews require the exact current revision')
            if db.execute("SELECT 1 FROM jobs WHERE candidate_id=? AND state IN('queued','running')",(candidate_id,)).fetchone():
                raise ValueError('Candidate already has an active job')
            if db.execute("SELECT COUNT(*) FROM jobs WHERE candidate_id=? AND kind='review'",(candidate_id,)).fetchone()[0]>=2:
                raise ValueError('Review-only attempt limit reached')
            source=json.loads(candidate['source_json'])
            base=self._revision_row(revision)
            original_job=self._row(db,'jobs',base['manifest']['job_id'])
            frozen=json.loads(original_job['payload_json'])['memory_context']
            from admin_memory_context import validate_context
            validate_context(frozen)
            if frozen['context_sha256']!=base['manifest']['memory_context_sha256']:
                raise ValueError('Review-only context differs from the original immutable revision')
            payload={'run_id':candidate['run_id'],'input_path':candidate['input_path'],'source_ref':source,
                'processing_consent':True,'base_revision':base,'repair_plan':None,
                'memory_context':frozen,
                'allow_review_protocol_upgrade':allow_protocol_upgrade}
            job=self._enqueue(db,candidate,'review',payload,True)
            db.execute("UPDATE candidates SET status='queued_review',updated_utc=? WHERE id=?",(now(),candidate_id))
            return {'candidate':self._candidate(self._row(db,'candidates',candidate_id)),'job':job}

    def approve(self,candidate_id,revision_id,expected_fingerprint,*,actor='nigel',review_override_reason=None):
        owner(actor)
        reason=override_reason(review_override_reason)
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE');candidate=self._row(db,'candidates',candidate_id);row=self._row(db,'revisions',revision_id)
            if row['candidate_id']!=candidate_id or candidate['current_revision_id']!=revision_id:raise ValueError('Only the current candidate revision may be approved')
            if db.execute("SELECT 1 FROM jobs WHERE candidate_id=? AND state IN('queued','running')",(candidate_id,)).fetchone():
                raise ValueError('Wait for the active candidate job before human approval')
            if expected_fingerprint!=row['fingerprint']:raise ValueError('Approval fingerprint is stale')
            revision=self._verify_revision(row,allow_hosted_override=reason is not None)
            overridden=bool(reason and revision['hosted_blocking_findings'])
            decision={'id':identifier('decision'),'candidate_id':candidate_id,'revision_id':revision_id,'action':'approve',
                'fingerprint':expected_fingerprint,'actor':actor,'created_utc':now(),
                'human_review_override':overridden,'review_override_reason':reason if overridden else None,
                'human_review_kind':'nigel_curatorial_approval','physical_accuracy_verified':False,'simulation_ready':False}
            path=self._save_record('decision',decision)
            db.execute('INSERT INTO decisions VALUES(?,?,?,?,?,?)',(decision['id'],candidate_id,revision_id,'approve',path,decision['created_utc']))
            db.execute("UPDATE candidates SET status='approved',updated_utc=? WHERE id=?",(now(),candidate_id))
            publication={'candidate_id':candidate_id,'revision_id':revision_id,'fingerprint':expected_fingerprint,'manifest_ref':revision['manifest_ref']}
            return {'candidate':self._candidate(self._row(db,'candidates',candidate_id)),'decision':decision,'publication':publication}

    def reject(self,candidate_id,revision_id,feedback,*,scopes=None,object_family=None,repair_targets=None,actor='nigel'):
        owner(actor)
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE');candidate=self._row(db,'candidates',candidate_id);row=self._row(db,'revisions',revision_id)
            if row['candidate_id']!=candidate_id or candidate['current_revision_id']!=revision_id:raise ValueError('Feedback must target the current revision')
            revision=self._revision_row(row)
            note=self.memory.add(candidate_id=candidate_id,revision_id=revision_id,source_sha256=json.loads(candidate['source_json'])['sha256'],
                feedback=feedback,scopes=scopes,object_family=object_family,actor=actor)
            if repair_targets is None:
                text=feedback.lower();targets={'reasoning'}
                if any(w in text for w in ['geometry','shape','texture','mesh','reconstruct','cavity','regenerat']):targets.add('reconstruction')
                if any(w in text for w in ['mask','track','identity','segment','occlu','missing','swap']):targets.add('tracking')
                if any(w in text for w in ['hinge','joint','axis','pivot','closed','closing','insertion','prismatic','motion']):targets.add('mechanics')
            else:
                if not isinstance(repair_targets,list) or not repair_targets or not set(repair_targets)<=set(TARGETS):raise ValueError('Unsupported repair target')
                targets=set(repair_targets)
            count=db.execute("SELECT COUNT(*) FROM jobs WHERE candidate_id=? AND kind='repair'",(candidate_id,)).fetchone()[0]
            plan={'format':'interaction-twin-bounded-repair-plan','version':1,'candidate_id':candidate_id,'revision_id':revision_id,
                'base_fingerprint':row['fingerprint'],'feedback_id':note['id'],'targets':[t for t in TARGETS if t in targets],
                'attempt_number':count+1,'maximum_attempts':2,'status':'ready_for_explicit_processing_consent' if count<2 else 'attempt_limit_reached',
                'execution_policy':'Fixed trusted stage adapters only; preserve original programs/models/paid caches; never execute feedback or model code.',
                'new_api_authorization':False,'existing_spending_guards_required':True,'created_utc':now()}
            plan_id=identifier('repairplan');path=self._save_record('repairplan',plan)
            db.execute('INSERT INTO repair_plans VALUES(?,?,?,?,?)',(plan_id,candidate_id,revision_id,path,plan['created_utc']))
            decision={'id':identifier('decision'),'candidate_id':candidate_id,'revision_id':revision_id,'action':'reject',
                'fingerprint':row['fingerprint'],'feedback_id':note['id'],'actor':actor,'created_utc':now()}
            decision_path=self._save_record('decision',decision)
            db.execute('INSERT INTO decisions VALUES(?,?,?,?,?,?)',(decision['id'],candidate_id,revision_id,'reject',decision_path,decision['created_utc']))
            db.execute("UPDATE candidates SET status='rejected',updated_utc=? WHERE id=?",(now(),candidate_id))
            return {'candidate':self._candidate(self._row(db,'candidates',candidate_id)),'feedback':note,'repair_plan':{'id':plan_id,**plan}}

    def request_repair(self,candidate_id,revision_id,plan_id,*,processing_consent=False,actor='nigel'):
        owner(actor)
        if processing_consent is not True:raise ValueError('Repair processing requires explicit owner consent')
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE');candidate=self._row(db,'candidates',candidate_id);revision=self._row(db,'revisions',revision_id);p=self._row(db,'repair_plans',plan_id)
            if p['candidate_id']!=candidate_id or p['revision_id']!=revision_id or candidate['current_revision_id']!=revision_id:
                raise ValueError('Repair plan targets another or stale revision')
            plan=self._private_record(p['record_path'])
            if plan['base_fingerprint']!=revision['fingerprint']:raise ValueError('Repair plan fingerprint changed')
            active=db.execute("SELECT * FROM jobs WHERE candidate_id=? AND state IN('queued','running')",(candidate_id,)).fetchone()
            if active:raise ValueError('Candidate already has an active job')
            count=db.execute("SELECT COUNT(*) FROM jobs WHERE candidate_id=? AND kind='repair'",(candidate_id,)).fetchone()[0]
            if count>=2 or plan['attempt_number']!=count+1:raise ValueError('Repair attempt limit reached or plan is stale')
            source=json.loads(candidate['source_json'])
            payload={'run_id':candidate['run_id'],'input_path':candidate['input_path'],'source_ref':source,'processing_consent':True,
                'base_revision':self._revision_row(revision),'repair_plan':{'id':plan_id,**plan},
                'memory_context':self.memory.retrieve(source_sha256=source['sha256'],object_family=candidate['object_family'])}
            job=self._enqueue(db,candidate,'repair',payload,True)
            db.execute("UPDATE candidates SET status='queued_repair',updated_utc=? WHERE id=?",(now(),candidate_id))
            return {'candidate':self._candidate(self._row(db,'candidates',candidate_id)),'job':job}

    def public_index(self):
        """Keep the last explicit approval until a replacement is approved.

        Reject is private revision feedback, not a publication withdrawal. A
        newer queued, failed or rejected repair cannot remove the prior approved
        immutable revision. Changed artifacts are still withheld on every read.
        Private manifests here are internal packager inputs, never public URLs.
        """
        records=[];blocked=[]
        with self.db() as db:
            for candidate in db.execute('SELECT * FROM candidates ORDER BY created_utc,id'):
                approvals=db.execute("SELECT * FROM decisions WHERE candidate_id=? AND action='approve' ORDER BY created_utc DESC,id DESC",(candidate['id'],)).fetchall()
                for approval in approvals:
                    try:
                        row=self._row(db,'revisions',approval['revision_id'])
                        decision=self._private_record(approval['record_path'])
                        if (row['candidate_id']!=candidate['id'] or decision.get('candidate_id')!=candidate['id'] or
                            decision.get('revision_id')!=row['id'] or decision.get('action')!='approve' or
                            decision.get('actor')!='nigel' or decision.get('fingerprint')!=row['fingerprint']):
                            raise ValueError('Approved revision decision binding changed')
                        overridden=decision.get('human_review_override',False)
                        reason=override_reason(decision.get('review_override_reason'))
                        if type(overridden)is not bool or overridden!=(reason is not None):raise ValueError('Approved hosted override reason is not bound')
                        revision=self._verify_revision(row,allow_hosted_override=overridden)
                        if overridden!=bool(revision['hosted_blocking_findings']):raise ValueError('Approved hosted override findings differ')
                        records.append({'candidate_id':candidate['id'],'revision_id':row['id'],'fingerprint':row['fingerprint'],
                            'manifest_ref':revision['manifest_ref'],'approval_kind':'nigel_curatorial_approval',
                            'human_review_override':overridden,
                            'task_classes':revision['manifest']['task_classes'],'physical_accuracy_verified':False,'simulation_ready':False})
                        break
                    except (ValueError,OSError,KeyError,TypeError):blocked.append(candidate['id'])
        return {'format':'interaction-twin-approved-revision-selection','version':1,'records':records,'blocked_changed_candidates':sorted(set(blocked)),
                'contains_private_feedback':False,'training_performed':False,'selection_only_not_public_payload':True}
