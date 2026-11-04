#!/usr/bin/env python3
"""Keep GPU work ahead of descriptions; no model/publication/allowance reset."""
from pathlib import Path
import fcntl,json,sys,time,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import complete_capture_batch_20261103 as C
OUT=C.OUT
B=C.B
def main():
 B.immutable(OUT/'gpu-queue-v2-started.json',{'pid':os.getpid(),'started_utc':C.now(),'driver':B.reference(Path(__file__)),'automatic_publication':False})
 B.P.write(OUT/'gpu-queue-v2-progress.json',{'status':'waiting_for_initial_batch','updated_utc':C.now()})
 deadline=time.monotonic()+4200
 while not(OUT/'batch-gpu-first-result.json').is_file():
  if time.monotonic()>deadline:raise TimeoutError('Initial batch still active; no duplicate submitted')
  time.sleep(2)
 with(ROOT/'.state/capture-gpu-priority-20261103.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);fcntl.flock(lock,fcntl.LOCK_UN)
 outcomes=[]
 for label,script,args in [C.COMMANDS[1],C.COMMANDS[0]]:
  B.P.write(OUT/'gpu-queue-v2-progress.json',{'status':'running','stage':label,'outcomes':outcomes,'updated_utc':C.now()})
  outcomes.append(C.command(label,script,args))
 B.P.write(OUT/'gpu-queue-v2-progress.json',{'status':'gpu_queue_continuations_finished','outcomes':outcomes,'completed_utc':C.now(),'next':'Root-reviewed bottle consistency branch before any descriptions','automatic_publication':False,'parent_allocation_modified':False})
 return 0
if __name__=='__main__':raise SystemExit(main())
