#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from observed_correspondence_review import run
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Read-only source-pixel, inference-scope and projection audit; no model calls')
    p.add_argument('--result',action='append',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();report=run([ROOT/path for path in a.result],ROOT/a.output,root=ROOT)
    print(json.dumps({'passed':report['passed'],'results':[{'component':r['component_id'],'passed':r['passed'],'counts':r['counts'],'errors':r['errors']} for r in report['results']]}))
