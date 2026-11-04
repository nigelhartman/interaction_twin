#!/usr/bin/env python3
"""Gauge-invariant source-bound assembly compatibility; no inference."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assembly_scale_feasibility_v2 import run
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair', nargs=2, action='append', required=True)
    p.add_argument('--out', required=True); a = p.parse_args()
    if not 1 <= len(a.pair) <= 3: p.error('At most three prospectively fixed saved pairs')
    value = run(a.pair, ROOT / a.out, root=ROOT)
    print(json.dumps({'status': value['status'], 'accepted_conditional_hypotheses': value['accepted_conditional_hypotheses'],
                      'pairs': [{'components': [r['component_a'], r['component_b']], 'status': r['status'],
                                 'reasons': r['reasons']} for r in value['pairs']]}))
