#!/usr/bin/env python3
"""Optional private 2D traces from exact saved SAM cable masks; no models."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cable_trace_2d as C


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program', required=True)
    parser.add_argument('--track-id', required=True)
    parser.add_argument('--evidence-id', action='append')
    parser.add_argument('--max-frames', type=int, default=90)
    parser.add_argument('--max-long-edge', type=int, default=960)
    parser.add_argument('--prune-spur-px', type=int, default=6)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    value = C.build(args.program, args.track_id, evidence_ids=args.evidence_id,
                    max_frames=args.max_frames, max_long_edge=args.max_long_edge,
                    prune_spur_px=args.prune_spur_px)
    reference = C.write(value, args.output_dir)
    counts = {}
    for sample in value['samples']:
        counts[sample['status']] = counts.get(sample['status'], 0) + 1
    print(json.dumps({'artifact': reference, 'samples_by_status': counts,
                      'two_dimensional_only': True, 'new_model_calls': 0,
                      'public_library_changed': False}, indent=2))


if __name__ == '__main__':
    main()
