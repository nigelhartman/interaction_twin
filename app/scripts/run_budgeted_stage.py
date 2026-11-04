#!/usr/bin/env python3
"""Run one allowlisted stage with the additional approved estimate envelope.

The existing MetaClient transport, credential loading and global ledger do not
change. Only this child process substitutes the stricter subclass before loading
the selected stage; no environment credential or arbitrary command is accepted.
"""
import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ALLOWED = {'discover_rigid_components.py', 'stage2_meta_reason.py', 'stage2_meta_video_segment.py',
           'stage2_meta_segment.py', 'stage3_meta_select_track_views.py', 'stage5_meta_propose.py'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('script', choices=sorted(ALLOWED))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from increment_budget import IncrementMetaClient
    import meta_api
    meta_api.MetaClient = IncrementMetaClient
    sys.argv = [str(ROOT / 'scripts' / args.script), *args.arguments]
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
