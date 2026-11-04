#!/usr/bin/env python3
"""One reviewed batch continuation: GPU priority, then private semantic stages."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,subprocess,sys,time,os,fcntl
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_capture_gpu_priority_20261103 as B
OUT=ROOT/'output/new-captures-20261103'
PYTHON=ROOT/'.venv-sam/bin/python'
COMMANDS=[('compact',ROOT/'scripts/recover_capture_compact_sam_20261103.py',['--allow-paid-api']),('reconcile',ROOT/'scripts/reconcile_capture_gpu_priority_20261103_v2.py',['--reconciliation-id','reconnect1'])]
def now():return datetime.now(timezone.utc).isoformat()
def save(value):B.P.write(OUT/'controller-progress.json',value)
def command(label,script,args):
 marker=OUT/('controller-'+label+'-intent.json')
 B.immutable(marker,{'label':label,'script':B.reference(script),'args':args,'created_utc':now(),'automatic_retry':False,'publication':False})
 with(OUT/('controller-'+label+'.log')).open('ab')as log:
  result=subprocess.run([str(PYTHON),str(script),*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 record={'label':label,'exit_code':result.returncode,'finished_utc':now(),'log':str((OUT/('controller-'+label+'.log')).relative_to(ROOT))}
 B.immutable(OUT/('controller-'+label+'-exit.json'),record)
 return record
def main():
 if(OUT/'controller-started.json').exists():raise ValueError('Controller already started; inspect existing process instead of duplicating')
 B.immutable(OUT/'controller-started.json',{'pid':os.getpid(),'started_utc':now(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'automatic_publication':False})
 save({'status':'waiting_for_initial_gpu_batch','updated_utc':now()})
 deadline=time.monotonic()+5400
 while not(OUT/'batch-gpu-first-result.json').is_file():
  if time.monotonic()>deadline:raise TimeoutError('Initial GPU batch did not reach its durable terminal receipt')
  time.sleep(3)
 with(ROOT/'.state/capture-gpu-priority-20261103.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  fcntl.flock(lock,fcntl.LOCK_UN)
 outcomes=[]
 for label,script,args in COMMANDS:
  save({'status':'running','stage':label,'outcomes':outcomes,'updated_utc':now()})
  outcomes.append(command(label,script,args))
 # Finish only exact successful GPU checkpoints. No blocked finish receipt is
 # created for a case that still needs an explicit reviewed recovery.
 doc,ref=B.load_manifest(ROOT/B.MANIFEST)
 for row in doc['captures']:
  path=ROOT/'data/flow-runs'/row['run_id']/'run.json'
  state=B.P.read(path)
  if(state.get('status')=='paused_after_reconstruct'and B.verified_checkpoint(state)
     and B.gpu_outcome(state)['all_selected_native_models_ready']):
   label='finish-'+row['id'];save({'status':'running','stage':label,'outcomes':outcomes,'updated_utc':now()})
   outcomes.append(command(label,ROOT/'scripts/finish_capture_gpu_priority_20261103_v2.py',['--capture',row['id']]))
  else:outcomes.append({'label':'finish-'+row['id'],'status':'deferred_incomplete_gpu_checkpoint','run_status':state.get('status')})
 save({'status':'controller_finished','outcomes':outcomes,'completed_utc':now(),'automatic_publication':False,'parent_allocation_modified':False})
 return 0
if __name__=='__main__':raise SystemExit(main())
