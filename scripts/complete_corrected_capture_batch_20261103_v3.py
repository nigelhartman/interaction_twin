#!/usr/bin/env python3
"""Finish genuine captures only after all prioritized geometry/review work."""
from pathlib import Path
import fcntl,json,sys,time,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import complete_capture_batch_20261103 as C
OUT=C.OUT;B=C.B
BOTTLE_RUN='run-capture-bottle-consistency-20261103-v2'
BOTTLE_SCRIPT=ROOT/'scripts/run_bottle_component_consistency_20261103_v2.py'
def ready(run_id):
 p=ROOT/'data/flow-runs'/run_id/'run.json'
 if not p.is_file():return False
 d=B.P.read(p)
 return(d.get('status')=='paused_after_reconstruct'and B.verified_checkpoint(d)and B.gpu_outcome(d)['all_selected_native_models_ready'])
def save(v):B.P.write(OUT/'corrected-controller-v3-progress.json',v)
def main():
 B.immutable(OUT/'corrected-controller-v3-started.json',{'pid':os.getpid(),'started_utc':C.now(),'driver':B.reference(Path(__file__)),'automatic_publication':False})
 save({'status':'waiting_for_phone_and_charger_gpu_queue','updated_utc':C.now()})
 deadline=time.monotonic()+5400
 while True:
  q=OUT/'gpu-queue-v2-progress.json'
  if q.is_file()and B.P.read(q).get('status')=='gpu_queue_continuations_finished':break
  if time.monotonic()>deadline:raise TimeoutError('Prior GPU queue did not reach durable terminal marker')
  time.sleep(2)
 with(ROOT/'.state/capture-gpu-priority-20261103.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);fcntl.flock(lock,fcntl.LOCK_UN)
 outcomes=[]
 save({'status':'running','stage':'bottle_full_frame_consistency_gpu','outcomes':outcomes,'updated_utc':C.now()})
 outcomes.append(C.command('bottle-consistency-gpu-initial',BOTTLE_SCRIPT,['--mode','gpu-first','--allow-paid-api']))
 doc,ref=B.load_manifest(ROOT/B.MANIFEST)
 for row in doc['captures']:
  if row['id']=='bottle':continue
  if ready(row['run_id']):
   label='finish-'+row['id'];save({'status':'running','stage':label,'outcomes':outcomes,'updated_utc':C.now()})
   outcomes.append(C.command(label,ROOT/'scripts/finish_capture_gpu_priority_20261103_v2.py',['--capture',row['id']]))
  else:outcomes.append({'label':'finish-'+row['id'],'status':'deferred_pending_or_failed_gpu','run_id':row['run_id']})
 if ready(BOTTLE_RUN):
  save({'status':'running','stage':'finish-corrected-bottle','outcomes':outcomes,'updated_utc':C.now()})
  outcomes.append(C.command('finish-corrected-bottle',BOTTLE_SCRIPT,['--mode','finish','--attempt-id','finish1','--allow-paid-api']))
 else:outcomes.append({'label':'finish-corrected-bottle','status':'deferred_pending_or_failed_gpu','run_id':BOTTLE_RUN})
 save({'status':'corrected_controller_finished','outcomes':outcomes,'completed_utc':C.now(),'automatic_publication':False,'parent_allocation_modified':False})
 return 0
if __name__=='__main__':raise SystemExit(main())
