#!/usr/bin/env python3
"""Bounded read-only relative-scale compatibility; no inference or pose edits."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assembly_scale_feasibility import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', nargs=2, action='append', required=True, metavar=('FIRST', 'SECOND'))
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if not 1 <= len(args.pair) <= 2:
        parser.error('One or two predeclared pairs only')
    result = run(args.pair, ROOT / args.out, root=ROOT)
    print(json.dumps({'status': result['status'], 'accepted_conditional_hypotheses': result['accepted_conditional_hypotheses'],
                      'pairs': [{'components': [r['component_a'], r['component_b']],
                                 'episodes': [r['a_episode_id'], r['b_episode_id']], 'status': r['status'],
                                 'reasons': r['reasons']} for r in result['pairs']]}))
