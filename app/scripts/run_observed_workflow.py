#!/usr/bin/env python3
"""Run one locally registered, fixed-option observed reconstruction workflow."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from observed_pipeline import run, data_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    request = data_path(ROOT, args.request)
    if request.name != 'workflow-request.json' or not request.is_relative_to(ROOT / 'data/observed-app-runs-20261031'):
        raise ValueError('Only an explicit local job workflow request is accepted')
    run(ROOT, json.loads(request.read_text()), data_path(ROOT, args.output))
