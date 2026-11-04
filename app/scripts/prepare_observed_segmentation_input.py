#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from observed_segmentation_input import build
if __name__=='__main__':
    p=argparse.ArgumentParser(description='Bind existing model-only SAM observations and fresh audits into geometry input')
    p.add_argument('--case',required=True);p.add_argument('--segmentation',required=True)
    p.add_argument('--annotations',required=True);p.add_argument('--output',required=True);p.add_argument('--audit')
    a=p.parse_args();doc,record=build(ROOT,ROOT/a.case,ROOT/a.segmentation,ROOT/a.annotations,ROOT/a.output,ROOT/a.audit if a.audit else None)
    print(json.dumps({'tracks':len(doc['tracklets']),'frames':len(doc['evidence']),
                      'observations':record['included_observations'],'fresh_audit_bound':record['fresh_visual_audit_bound']}))
