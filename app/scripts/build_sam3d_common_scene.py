#!/usr/bin/env python3
"""Build an automatic estimated SAM 3D scene from existing models and masks."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sam3d_common_scene import build, publish_registry

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--program', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--publish-index', action='store_true')
    a = p.parse_args(); result = build(a.program, a.output)
    if a.publish_index:
        result['registry'] = publish_registry(result)
    print(json.dumps(result, indent=2))
