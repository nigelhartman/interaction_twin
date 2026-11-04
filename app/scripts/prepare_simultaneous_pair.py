#!/usr/bin/env python3
"""Prepare fixed paired source evidence; model-worker lock belongs to caller."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simultaneous_pair_input import prepare
from interval_motion_dense_input import propagate

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'propagate'])
    p.add_argument('--parent-plan'); p.add_argument('--parent-result')
    p.add_argument('--peer-input'); p.add_argument('--peer-result'); p.add_argument('--episode')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if a.action == 'prepare':
        if not all([a.parent_plan, a.parent_result, a.peer_input, a.peer_result, a.episode]):
            p.error('Explicit frozen parent/peer paths and parent episode required')
        result = prepare(a.parent_plan, a.parent_result, a.peer_input, a.peer_result, ROOT / a.out, root=ROOT, episode_id=a.episode)
    else:
        result = propagate(ROOT / a.out, root=ROOT)
    print(json.dumps({'status': result['status'], 'frame_count': result['frame_count'], 'track_id': result['track_id'],
                      'reference_native_frame': result['reference_sample_index']}))
