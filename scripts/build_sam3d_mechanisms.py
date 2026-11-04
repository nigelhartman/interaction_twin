#!/usr/bin/env python3
"""Fit source-cited, mesh-anchored persistent SAM mechanisms without inference."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sam3d_mechanisms import build
from sam3d_common_scene import publish_registry

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', required=True, type=Path)
    p.add_argument('--proposals', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--publish-index', action='store_true')
    a = p.parse_args(); result = build(a.scene, a.proposals, a.output)
    if a.publish_index:
        result['registry'] = publish_registry(result)
    print(json.dumps(result, indent=2))
