#!/usr/bin/env python3
"""Explicit offline omission-only topology derivative, followed by optional CPU fit."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import laptop_hinge_omission_recovery_v1 as R

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('action', choices=['recover', 'verify', 'scene'])
args = parser.parse_args()
value = {'recover': R.recover, 'verify': R.verify, 'scene': R.build_scene}[args.action]()
print(json.dumps(value), flush=True)
