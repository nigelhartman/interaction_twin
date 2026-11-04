#!/usr/bin/env python3
"""One fixed same-frame component context; orchestration owns model flock."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simultaneous_pair_motion import run
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True); p.add_argument('--audit', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    value, compatibility = run(a.plan, a.audit, ROOT / a.out, root=ROOT)
    print(json.dumps({'status': value['status'], 'supported_nonreference_frames': value['supported_nonreference_frames'],
                      'actual_model_calls': value['actual_cotracker_model_calls'], 'wall_seconds': value['wall_seconds'],
                      'accepted_conditional_hypotheses': compatibility['accepted_conditional_hypotheses']}))
