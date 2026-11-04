#!/usr/bin/env python3
"""Explicit, bounded future-upload mechanism phases; never invoked by readers."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mechanism_pipeline as M


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['adopt', 'topology', 'scene', 'export'])
    parser.add_argument('--program')
    parser.add_argument('--proposal-ref')
    parser.add_argument('--proposal-path')
    parser.add_argument('--scene-ref')
    parser.add_argument('--part-ledger',help='Exact compiled part-ledger companion; disables review discovery')
    parser.add_argument('--metadata-path',action='append',default=None,help='Explicit native metadata path; repeat in authoritative ledger order')
    parser.add_argument('--held-out-sample',action='append',type=int,default=[],help='Predeclared source sample excluded from pose/joint mask fitting')
    parser.add_argument('--report', required=True)
    parser.add_argument('--task-budget-plan', default=M.BUDGET_PLAN)
    parser.add_argument('--overnight-task-budget-plan',help='Exact outer plan; new suballocation also requires the inherited guarded client')
    parser.add_argument('--max-images', type=int, default=16)
    parser.add_argument('--max-output-tokens', type=int, default=10000)
    parser.add_argument('--allow-paid-api', action='store_true')
    parser.add_argument('--private', action='store_true', help='Export without publishing; owner review must approve first')
    args = parser.parse_args()
    report = M.local(ROOT, args.report, exists=False)
    if args.phase == 'adopt':
        value = M.adopt_verified_topology(args.program, args.proposal_path)
    elif args.phase == 'topology':
        value = M.topology(args.program, allow_paid_api=args.allow_paid_api, task_budget_plan=args.task_budget_plan,
                           max_images=args.max_images, max_output_tokens=args.max_output_tokens,
                           overnight_task_budget_plan=args.overnight_task_budget_plan)
    elif args.phase == 'scene':
        value = M.constrained_scene(args.program, json.loads(args.proposal_ref),part_ledger_path=args.part_ledger,metadata_paths=args.metadata_path,held_out_sample_indices=args.held_out_sample)
    else:
        value = M.export_scene(json.loads(args.scene_ref), publish=not args.private)
    M.write(report, value)
    print(json.dumps({k:v for k,v in value.items() if k != 'artifacts'}, indent=2))


if __name__ == '__main__':
    main()
