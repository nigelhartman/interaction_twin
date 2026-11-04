#!/usr/bin/env python3
"""Run a fixed stage inside the approved overnight estimate envelope.

Only this child substitutes clients before target modules are imported. Existing
increment, mechanism-task, global ledger and paid-once checks remain in force.
"""
import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ALLOWED = {'discover_rigid_components.py', 'stage2_meta_reason.py', 'stage2_meta_video_segment.py',
           'stage2_meta_segment.py', 'stage3_meta_select_track_views.py', 'stage5_meta_propose.py',
           'run_mechanism_pipeline.py', 'run_admin_stage.py'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-budget-plan', required=True)
    parser.add_argument('script', choices=sorted(ALLOWED))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from overnight_budget import OvernightMetaClient, PLAN, plan
    candidate = Path(args.task_budget_plan)
    if candidate.is_absolute():
        candidate = candidate.relative_to(ROOT)
    if candidate.as_posix() != PLAN:
        raise ValueError('Only the fixed approved overnight budget is executable')
    plan(ROOT)
    import meta_api
    import increment_budget
    # Frozen classes imported by the target inherit this stricter client. Never
    # patch an already constructed production client or change its key/origin.
    meta_api.MetaClient = OvernightMetaClient
    increment_budget.IncrementMetaClient = OvernightMetaClient
    target = ROOT / 'scripts' / args.script
    sys.argv = [str(target), *args.arguments]
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
