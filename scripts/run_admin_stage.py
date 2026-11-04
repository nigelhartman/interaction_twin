#!/usr/bin/env python3
"""Execute one fixed private Meta stage with scoped, untrusted owner feedback."""
import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from admin_memory_context import ALLOWED, contextual_client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-context-ref', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--machine-review-ref')
    parser.add_argument('script', choices=sorted(ALLOWED))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import json
    import meta_api
    import increment_budget
    base_class=None
    from overnight_budget import OvernightMetaClient
    if issubclass(meta_api.MetaClient,OvernightMetaClient):
        from mechanism_pass_budget import scoped_client_class
        base_class=scoped_client_class(meta_api.MetaClient,root=ROOT)
    client = contextual_client(json.loads(args.memory_context_ref), source_sha256=args.source_sha256, script=args.script,
        machine_review_ref=json.loads(args.machine_review_ref)if args.machine_review_ref else None,base_class=base_class)
    # Both import styles occur in the fixed stage adapters. This substitution
    # exists only in this child process; the parent's clients remain unchanged.
    meta_api.MetaClient = client
    increment_budget.IncrementMetaClient = client
    sys.argv = [str(ROOT / 'scripts' / args.script), *args.arguments]
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
