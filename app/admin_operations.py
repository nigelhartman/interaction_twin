"""Private owner operations projections. GET reads local records, never models/SSH.

No credentials, prompts, raw request payloads, workflow feedback or remote command
output are exposed. Explicit GPU refresh queries only the fixed existing parent
allocation through its existing SSH master; it never starts a compute step.
HTTP owner authentication and CSRF are the caller's responsibility.
"""
from collections import Counter,defaultdict
from contextlib import contextmanager
from datetime import datetime,timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import sqlite3
import stat
import subprocess
import tempfile
import threading
import time

import meta_budget as MB
ROOT=Path(__file__).resolve().parent
LEDGER='.secrets/meta-budget.sqlite3'
WORKFLOW='.private/admin-workflow/state.sqlite3'
MARKETING='.private/marketing-studio/studio.sqlite3'
GPU_CACHE='.private/admin-operations/gpu-status.json'
PARENT='16889584'
STALE_SECONDS=120
MAX_ROWS=50000
USAGE_FIELDS=('input_tokens','output_tokens','cached_input_tokens','reasoning_tokens','billable_output_tokens','images','video_frames','billable_video_frames')
REQUEST_STATES={'reserved','in_flight','uncertain','bounded_unknown','settled','released'}
HELD={'reserved','in_flight','uncertain','bounded_unknown'}
JOB_STATES={'pending_consent','queued','running','pending','failed','interrupted','complete','cancelled'}
CANDIDATE_STATES={'pending_consent','queued','processing','queued_repair','awaiting_review','blocked_review','approved','rejected','needs_attention','failed','pending'}
MARKETING_STATES={'planned','rendering','review_ready','error'}
SLURM_STATES={'PENDING','RUNNING','COMPLETING','COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','SUSPENDED','PREEMPTED','BOOT_FAIL','DEADLINE','REQUEUED','REQUEUE_FED','RESV_DEL_HOLD','SPECIAL_EXIT','STOPPED','CONFIGURING','RESIZING','REVOKED','SIGNALING','STAGE_OUT'}
GPU_ERRORS={'configuration_missing','configuration_disabled','configuration_unavailable','master_unavailable','master_timeout','query_timeout','query_unavailable','scheduler_unavailable','scheduler_partial','ambiguous_allocation','invalid_response','cache_unavailable'}
_LOCK=threading.Lock()


def now():return datetime.now(timezone.utc).isoformat()
def _time(value):
    try:
        result=datetime.fromisoformat(value.replace('Z','+00:00'))
        return result.replace(tzinfo=timezone.utc)if result.tzinfo is None else result
    except (ValueError,TypeError,AttributeError):return None

def _age(value,clock=None):
    parsed=_time(value)
    return max(0,int((clock or time.time())-parsed.timestamp()))if parsed else None


def _file(root,relative,maximum=128*1024*1024):
    root=Path(root).resolve();path=root/relative
    if path.resolve()!=path or any(p.is_symlink()for p in [path,*path.parents]):raise ValueError('Unavailable ordinary private status file')
    info=path.lstat()
    if not stat.S_ISREG(info.st_mode)or info.st_uid!=os.getuid()or not 0<info.st_size<=maximum:raise ValueError('Unavailable bounded private status file')
    return path


@contextmanager
def _readonly(root,relative):
    path=_file(root,relative)
    db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=2)
    try:
        db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
        yield db
    finally:db.close()


def _known(value,allowed):return value if value in allowed else 'other'
def _model(value):
    return value if isinstance(value,str)and re.fullmatch(r'(?:muse[-_.][A-Za-z0-9_.-]{1,64}|sam[-_.]?[A-Za-z0-9_.-]{0,64})',value)else'other_model'
def _endpoint(value):return value if value in {'/v1/chat/completions','/v1/responses','/v1/segment'}else'other_endpoint'
def _id(value):return value if isinstance(value,str)and re.fullmatch(r'(?:job|candidate)_[a-f0-9]{32}',value)else None
def _date(value):
    parsed=_time(value)if isinstance(value,str)and len(value)<=40 else None
    return parsed.astimezone(timezone.utc).isoformat()if parsed else None

def _usage(value):
    if not isinstance(value,str)or len(value)>8192:return None
    try:return MB._clean_usage(json.loads(value))
    except (ValueError,TypeError,MB.BudgetError):return None


def _new_counts():return {'sums':Counter(),'known':Counter(),'request_count':0,'requests_without_reported_usage':0,'settled':0,'held':0}
def _add(group,row,usage):
    group['request_count']+=1
    if not usage:group['requests_without_reported_usage']+=1
    else:
        for key,value in usage.items():group['sums'][key]+=value;group['known'][key]+=1
    if row['state']=='settled'and type(row['actual'])is int and row['actual']>=0:group['settled']+=row['actual']
    if row['state']in HELD and type(row['reserved'])is int and row['reserved']>=0:group['held']+=row['reserved']
def _counts(group):return {key:group['sums'][key]if group['known'][key]else None for key in USAGE_FIELDS}


def budget_and_usage(root=ROOT):
    """Same safe retained-hold projection as read_budget_status, one RO snapshot."""
    unavailable={'status':'unavailable','currency':'USD','scope':'local_ledger_only_not_account_balance','blocked':True,
        'actual_meta_credit_balance_verified':False,'provider_balance_usd':None,'task_envelopes':[]}
    empty_usage={'status':'unavailable','totals':{k:None for k in USAGE_FIELDS},'counts_by_state':{},'models':[],'recent_requests':[],'requests_without_reported_usage':None,'scope':'reported_local_receipt_counts_only'}
    try:
        with _readonly(root,LEDGER)as db:
            budget={'status':'available',**MB._status_from_db(db),'actual_meta_credit_balance_verified':False,'provider_balance_usd':None}
            for field in ['cap_usd','per_request_cap_usd','settled_usd','reserved_usd','remaining_estimated_usd','over_cap_usd']:MB._decimal(budget[field])
            rows=db.execute('SELECT request_id,model,endpoint,kind,state,reserved,actual,usage_json,created,updated FROM requests ORDER BY updated DESC,request_id LIMIT ?',(MAX_ROWS+1,)).fetchall()
        truncated=len(rows)>MAX_ROWS;rows=rows[:MAX_ROWS];groups={};total=_new_counts();states=Counter();recent=[]
        for row in rows:
            usage=_usage(row['usage_json']);model=_model(row['model']);kind=_known(row['kind'],{'text','sam'});state=_known(row['state'],REQUEST_STATES)
            key=(model,kind);group=groups.setdefault(key,_new_counts());_add(group,row,usage);_add(total,row,usage);states[state]+=1
            if len(recent)<20:
                recent.append({'request_id':'request_'+hashlib.sha256(str(row['request_id']).encode()).hexdigest()[:12],
                    'model':model,'kind':kind,'endpoint':_endpoint(row['endpoint']),'state':state,
                    'settled_usd':MB.usd(row['actual'])if row['state']=='settled'and type(row['actual'])is int and row['actual']>=0 else None,
                    'reserved_usd':MB.usd(row['reserved'])if row['state']in HELD and type(row['reserved'])is int and row['reserved']>=0 else '0.000000',
                    'usage':usage,'usage_reported':usage is not None,'created_at':_date(row['created']),'updated_at':_date(row['updated']),
                    'uncertain_charge':row['state']in {'uncertain','bounded_unknown'}})
        usage={'status':'partial'if truncated else'available','scope':'reported_local_receipt_counts_only','model_scope':'Meta API ledger only; local and cluster model activity is not measured here',
            'request_count':budget['request_count'],'processed_requests':len(rows),'truncated':truncated,'totals':_counts(total),
            'requests_without_reported_usage':total['requests_without_reported_usage'],'counts_by_state':{key:states[key]for key in sorted(REQUEST_STATES|set(states))},
            'models':[{'model':model,'kind':kind,'request_count':g['request_count'],'settled_usd':MB.usd(g['settled']),
                'reserved_usd':MB.usd(g['held']),'totals':_counts(g),'requests_without_reported_usage':g['requests_without_reported_usage']}for(model,kind),g in sorted(groups.items())],
            'recent_requests':recent,'reasoning_tokens_may_be_subset_of_output_tokens':True,'estimates_are_not_actual_provider_balance':True}
        budget['task_envelopes']=_envelopes(root,budget)
        return budget,usage
    except (OSError,ValueError,sqlite3.Error,KeyError,TypeError,OverflowError,MB.BudgetError):
        if not(Path(root)/LEDGER).exists():unavailable['status']='not_started'
        return unavailable,empty_usage


def _envelopes(root,budget):
    result=[]
    try:
        from overnight_budget import budget_status
        value=budget_status(root=root,ledger_status=budget)
        remaining=Decimal(value['task_remaining_estimated_usd'])
        result.append({'id':'overnight_pass','status':'available','ceiling_usd':value['task_ceiling_usd'],
            'remaining_estimated_usd':format(max(Decimal(0),remaining),'.6f'),'over_ceiling_usd':format(max(Decimal(0),-remaining),'.6f'),
            'can_admit':value['can_admit'],'scope':'existing_approved_task_not_extra_credit'})
    except Exception:result.append({'id':'overnight_pass','status':'unavailable','can_admit':False})
    try:
        from mechanism_pass_budget import load
        from increment_budget import remaining_increment
        value=load(root);remaining=remaining_increment(value,budget)
        result.append({'id':'mechanism_subtask','status':'available','ceiling_usd':value['additional_estimated_api_ceiling_usd'],
            'remaining_estimated_usd':format(max(Decimal(0),remaining),'.6f'),'over_ceiling_usd':format(max(Decimal(0),-remaining),'.6f'),
            'can_admit':remaining>=Decimal(budget['per_request_cap_usd'])and not budget['blocked'],'scope':'nested_within_overnight_pass_not_extra_credit'})
    except Exception:result.append({'id':'mechanism_subtask','status':'unavailable','can_admit':False})
    return result


def processing_summary(root=ROOT):
    result={'status':'unavailable','counts_by_state':{},'counts_by_candidate_state':{},'active_jobs':[],'recent_jobs':[],
        'worker_liveness_verified':False,'state_is_saved_record_not_external_execution_proof':True}
    try:
        with _readonly(root,WORKFLOW)as db:
            states=Counter({key:0 for key in JOB_STATES});candidates=Counter({key:0 for key in CANDIDATE_STATES})
            for r in db.execute('SELECT state,COUNT(*) n FROM jobs GROUP BY state'):states[_known(r['state'],JOB_STATES)]+=r['n']
            for r in db.execute('SELECT status,COUNT(*) n FROM candidates GROUP BY status'):candidates[_known(r['status'],CANDIDATE_STATES)]+=r['n']
            columns={r['name']for r in db.execute('PRAGMA table_info(jobs)')}
            fields='id,candidate_id,kind,state,lease_until,created_utc,updated_utc'+(',heartbeat_utc'if'heartbeat_utc'in columns else',NULL heartbeat_utc')+(',progress_json'if'progress_json'in columns else',NULL progress_json')
            rows=db.execute('SELECT '+fields+' FROM jobs ORDER BY updated_utc DESC,id LIMIT 20').fetchall()
            active_rows=db.execute("SELECT "+fields+" FROM jobs WHERE state IN('queued','running','pending') ORDER BY updated_utc DESC,id LIMIT 20").fetchall()
        stamp=time.time()
        def project(row):
            lease=row['lease_until'];lease=lease if type(lease)in(int,float)and math.isfinite(lease)else None
            stage=None
            if isinstance(row['progress_json'],str)and len(row['progress_json'])<=4096:
                try:
                    value=json.loads(row['progress_json']).get('stage')
                    if isinstance(value,str)and re.fullmatch('[a-z][a-z0-9_]{0,119}',value):stage=value
                except (ValueError,TypeError,AttributeError):pass
            return {'id':_id(row['id']),'candidate_id':_id(row['candidate_id']),'kind':_known(row['kind'],{'pipeline','repair'}),
                'state':_known(row['state'],JOB_STATES),'stage':stage,'created_at':_date(row['created_utc']),'updated_at':_date(row['updated_utc']),
                'heartbeat_at':_date(row['heartbeat_utc']),'heartbeat_age_s':_age(row['heartbeat_utc'],stamp),
                'lease_status':('expired'if lease<=stamp else'current')if row['state']=='running'and lease is not None else'not_verified'}
        result.update(status='available',counts_by_state=dict(sorted(states.items())),counts_by_candidate_state=dict(sorted(candidates.items())),recent_jobs=[project(r)for r in rows],
            active_jobs=[project(r)for r in active_rows],active_list_limit=20,recent_list_limit=20)
    except (OSError,ValueError,sqlite3.Error,TypeError,KeyError):
        if not(Path(root)/WORKFLOW).exists():result['status']='not_started'
    return result


def marketing_processing_summary(root=ROOT):
    """Project saved local render status without constructing/migrating a studio.

    IDs join only this fixed private database. No immutable campaign record,
    title, brief, feedback, worker token, error or render-result body is read.
    A saved rendering state does not prove that its worker is still alive.
    """
    result={'status':'unavailable','counts_by_state':{},'active_jobs':[],'recent_jobs':[],
        'active_list_limit':20,'recent_list_limit':20,'omitted_invalid_identity_rows':0,
        'worker_liveness_verified':False,'state_is_saved_record_not_external_execution_proof':True}
    try:
        with _readonly(root,MARKETING)as db:
            counts=Counter({key:0 for key in MARKETING_STATES})
            for row in db.execute('SELECT status,COUNT(*) n FROM jobs GROUP BY status'):
                counts[_known(row['status'],MARKETING_STATES)]+=row['n']
            fields='id,campaign_id,revision_id,format_id,status,progress,heartbeat,created'
            recent=db.execute('SELECT '+fields+' FROM jobs ORDER BY created DESC,id LIMIT 20').fetchall()
            active=db.execute("SELECT "+fields+" FROM jobs WHERE status IN('planned','rendering') ORDER BY created DESC,id LIMIT 20").fetchall()
        stamp=time.time();invalid=set()
        def project(row):
            for name,prefix in [('id','job'),('campaign_id','campaign'),('revision_id','revision')]:
                if not isinstance(row[name],str)or not re.fullmatch(prefix+r'_[a-f0-9]{64}',row[name]):
                    invalid.add(str(row['id']));return None
            progress=row['progress'];progress=progress if type(progress)in(int,float)and math.isfinite(progress)and 0<=progress<=1 else None
            return {'id':row['id'],'campaign_id':row['campaign_id'],'revision_id':row['revision_id'],
                'format_id':_known(row['format_id'],{'landscape','portrait','square'})if row['format_id']in {'landscape','portrait','square'}else None,
                'state':_known(row['status'],MARKETING_STATES),'progress':progress,
                'created_at':_date(row['created']),'heartbeat_at':_date(row['heartbeat']),
                'heartbeat_age_s':_age(row['heartbeat'],stamp)}
        def projected(rows):return [item for row in rows if(item:=project(row))is not None]
        result.update(status='available',counts_by_state=dict(sorted(counts.items())),active_jobs=projected(active),recent_jobs=projected(recent),omitted_invalid_identity_rows=len(invalid))
    except (OSError,ValueError,sqlite3.Error,TypeError,KeyError):
        if not(Path(root)/MARKETING).exists():result['status']='not_started'
    return result


def _configuration(root):
    from sam3d_cluster_bridge import configuration
    if not(Path(root)/'config/sam3d-cluster.json').exists():return None,'missing'
    try:
        value=configuration(root)
        return(value,'enabled')if value else(None,'disabled')
    except Exception:return None,'unavailable'


def _gpu_base(root):
    config,state=_configuration(root)
    return config,{'status':'not_checked','configuration_status':state,'configured':config is not None,'allocation_id':PARENT,
        'maximum_parallel_workers':config['maximum_parallel_workers']if config else None,'checked_at':None,'connection_verified_at':None,
        'stale':True,'age_seconds':None,'allocation':None,'error_code':None,'gpu_utilization':None,'gpu_memory':None,
        'device_measurements_status':'not_queried','worker_readiness':'not_probed','allocation_or_holder_modified':False}


def _allocation(value):
    if not isinstance(value,dict)or value.get('job_id')!=PARENT:return None
    state=value.get('state');state=state.split()[0].rstrip('+*')if isinstance(state,str)and state else'UNKNOWN'
    state=state if state in SLURM_STATES else'UNKNOWN'
    def text(key,pattern,maximum=240):
        v=value.get(key);return v if isinstance(v,str)and len(v)<=maximum and re.fullmatch(pattern,v)else None
    tres=text('allocated_tres',r'[A-Za-z0-9_.:/=,+-]*');gpus=None
    if tres:
        match=re.search(r'(?:^|,)gres/gpu=(\d{1,3})(?:,|$)',tres)
        if match and int(match.group(1))<=64:gpus=int(match.group(1))
    return {'job_id':PARENT,'state':state,'gpus':gpus,'node_list':text('node_list',r'[A-Za-z0-9_,.\[\]-]*',180),
        'start_time':text('start_time',r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:Z|[+-]\d\d:\d\d)?',40),
        'end_time':text('end_time',r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:Z|[+-]\d\d:\d\d)?',40),
        'elapsed':text('elapsed',r'[0-9:.+-]*',40),'time_limit':text('time_limit',r'[0-9:.+-]*',40),
        'allocated_tres':tres,'scheduler_time_zone':'cluster_reported_unspecified','start_is_scheduler_estimate':state=='PENDING'}


def gpu_status(root=ROOT):
    _,result=_gpu_base(root)
    try:
        value=json.loads(_file(root,GPU_CACHE,32768).read_text())
        if not isinstance(value,dict)or value.get('format')!='interaction-twin-private-gpu-status'or type(value.get('version'))is not int or value.get('version')!=1 or value.get('allocation_id')!=PARENT:raise ValueError('Invalid cached GPU status')
        stamp=_date(value.get('checked_at'));age=_age(stamp)
        if stamp is None or _time(stamp).timestamp()>time.time()+5:raise ValueError('Current GPU check time required')
        status=value.get('status');status=status if status in {'checked','not_found','unavailable'}else'unavailable'
        allocation=_allocation(value.get('allocation'))
        if status=='checked'and allocation is None:raise ValueError('Exact cached allocation required')
        result.update(status=status,checked_at=stamp,age_seconds=age,stale=age is None or age>STALE_SECONDS,
            connection_verified_at=stamp if value.get('connection_verified')is True else None,
            allocation=allocation,error_code=value.get('error_code')if value.get('error_code')in GPU_ERRORS else None)
    except (OSError,ValueError,TypeError,KeyError):pass
    result['refresh_in_progress']=_LOCK.locked()
    return result


# Executed by the already-installed frozen Python over an EXISTING SSH master.
# These are the only two remote subprocess argument vectors. No srun/submit/cancel,
# nvidia-smi, holder changes, filesystem writes, downloads or model calls exist.
GPU_QUERY_CODE=r'''import json,subprocess
result={'version':1,'squeue':None,'sacct':None}
commands={'squeue':['squeue','--noheader','--jobs','16889584','--format','%i|%T|%S|%e|%N'],
'sacct':['sacct','--noheader','--parsable2','--allocations','--jobs','16889584','--format','JobIDRaw,State,Start,End,AllocTRES,Elapsed,Timelimit,NodeList']}
for key,argv in commands.items():
 try:
  p=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=6)
  result[key]={'ok':p.returncode==0,'rows':p.stdout.splitlines()[:12]if p.returncode==0 and len(p.stdout)<=8192 else[]}
 except (OSError,subprocess.TimeoutExpired):result[key]={'ok':False,'rows':[]}
print(json.dumps(result,separators=(',',':')))
'''


def _query_gpu(config):
    from sam3d_cluster_bridge import REMOTE_PYTHON
    options=['ssh','-o','BatchMode=yes','-o','NumberOfPasswordPrompts=0','-o','ConnectionAttempts=1','-o','ConnectTimeout=5',
        '-o','ControlMaster=no','-o','ProxyCommand=false','-S',config['ssh_control_path']]
    try:check=subprocess.run(options+['-O','check','KISSKI'],stdin=subprocess.DEVNULL,capture_output=True,timeout=6)
    except subprocess.TimeoutExpired:return {'status':'unavailable','error_code':'master_timeout','connection_verified':False}
    except OSError:return {'status':'unavailable','error_code':'master_unavailable','connection_verified':False}
    if check.returncode:return {'status':'unavailable','error_code':'master_unavailable','connection_verified':False}
    try:
        p=subprocess.run(options+['KISSKI',shlex.join([REMOTE_PYTHON,'-I','-c',GPU_QUERY_CODE])],stdin=subprocess.DEVNULL,capture_output=True,timeout=22)
    except subprocess.TimeoutExpired:return {'status':'unavailable','error_code':'query_timeout','connection_verified':False}
    except OSError:return {'status':'unavailable','error_code':'query_unavailable','connection_verified':False}
    if p.returncode or len(p.stdout)>32768:return {'status':'unavailable','error_code':'query_unavailable','connection_verified':False}
    try:value=json.loads(next(line for line in reversed(p.stdout.splitlines())if line.startswith(b'{')))
    except (ValueError,StopIteration):return {'status':'unavailable','error_code':'invalid_response','connection_verified':False}
    if not isinstance(value,dict)or type(value.get('version'))is not int or value.get('version')!=1:return {'status':'unavailable','error_code':'invalid_response','connection_verified':False}
    queue=value.get('squeue',{});account=value.get('sacct',{})
    if not isinstance(queue,dict)or not isinstance(account,dict)or not all(type(x.get('ok'))is bool and isinstance(x.get('rows'),list)and len(x['rows'])<=12 and all(isinstance(line,str)and len(line)<=8192 for line in x['rows'])for x in [queue,account]):
        return {'status':'unavailable','error_code':'invalid_response','connection_verified':False}
    row=None;account_rows=[];queue_rows=[]
    for line in account['rows']if account['ok']else[]:
        parts=line.strip().split('|')
        if not line.strip():continue
        if len(parts)<8 or parts[0]!=PARENT:return {'status':'unavailable','error_code':'invalid_response','connection_verified':True}
        if len(parts)>=8 and parts[0]==PARENT:
            account_rows.append(dict(zip(['job_id','state','start_time','end_time','allocated_tres','elapsed','time_limit','node_list'],parts[:8])))
    for line in queue['rows']if queue['ok']else[]:
        parts=line.strip().split('|')
        if not line.strip():continue
        if len(parts)!=5 or parts[0]!=PARENT:return {'status':'unavailable','error_code':'invalid_response','connection_verified':True}
        if len(parts)==5 and parts[0]==PARENT:
            queue_rows.append(dict(zip(['job_id','state','start_time','end_time','node_list'],parts)))
    if len(account_rows)>1 or len(queue_rows)>1:return {'status':'unavailable','error_code':'ambiguous_allocation','connection_verified':True}
    if account_rows:row=account_rows[0]
    if queue_rows:row={**(row or{}),**queue_rows[0]}
    if row:return {'status':'checked','allocation':_allocation(row),'connection_verified':True,'error_code':None}
    if queue['ok']and account['ok']:return {'status':'not_found','allocation':None,'connection_verified':True,'error_code':None}
    return {'status':'unavailable','connection_verified':True,'error_code':'scheduler_partial'if queue['ok']or account['ok']else'scheduler_unavailable'}


def _save_gpu(root,value):
    from agent_memory import private_directory
    folder=private_directory(root,'admin-operations');target=folder/'gpu-status.json'
    if target.is_symlink():raise ValueError('GPU cache symlink refused')
    fd,temporary=tempfile.mkstemp(prefix='.gpu-status-',suffix='.tmp',dir=folder)
    try:
        with os.fdopen(fd,'w')as stream:json.dump(value,stream,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,target)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def refresh_gpu(root=ROOT):
    """Explicit owner action only: fixed read-only scheduler probe + local cache."""
    root=Path(root).resolve()
    if not _LOCK.acquire(blocking=False):return snapshot(root)
    try:
        config,state=_configuration(root)
        value=(_query_gpu(config)if config else{'status':'unavailable','connection_verified':False,'error_code':'configuration_'+state})
        value={key:value.get(key)for key in ['status','allocation','connection_verified','error_code']}
        value.update(format='interaction-twin-private-gpu-status',version=1,checked_at=now(),allocation_id=PARENT)
        try:_save_gpu(root,value)
        except (OSError,ValueError):
            result=snapshot(root);result['gpu'].update(status='unavailable',error_code='cache_unavailable');return result
    finally:_LOCK.release()
    return snapshot(root)


def snapshot(root=ROOT):
    root=Path(root).resolve();budget,usage=budget_and_usage(root);processing=processing_summary(root);marketing=marketing_processing_summary(root);gpu=gpu_status(root)
    return {'format':'interaction-twin-private-operations','version':1,'checked_at':now(),'budget':budget,'usage':usage,'processing':processing,'marketing':marketing,'gpu':gpu,
        'health':{'status':'ok'if budget['status']=='available'and processing['status']=='available'else'partial',
            'snapshot_service':'responding','local_records_only':True,'worker_liveness_verified':False,'provider_balance_verified':False},
        'limitations':['Costs are local conservative estimates, not the provider account balance.','Uncertain request holds remain reserved; missing reported token counts stay unknown.',
            'GPU connection/allocation status is verified only at the recorded explicit refresh time.','Device utilization and memory are not measured; GET never contacts SSH or changes jobs.']}
