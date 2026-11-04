#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from observed_track_refinement import run
if __name__=='__main__':
    p=argparse.ArgumentParser(description='CPU image refinement of frozen CoTracker source tracks; no model/API calls')
    p.add_argument('--program',required=True);p.add_argument('--track',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--max-views',type=int,default=49);p.add_argument('--max-ba-evaluations',type=int,default=70)
    a=p.parse_args();run(ROOT/a.program,a.track,ROOT/a.output,root=ROOT,cache=ROOT/a.cache,max_views=a.max_views,max_nfev=a.max_ba_evaluations)
