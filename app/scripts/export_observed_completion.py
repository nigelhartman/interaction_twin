#!/usr/bin/env python3
"""Package existing optional inferred layers; never generate new geometry."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from observed_completion_export import export_completions

if __name__ == '__main__':
    print(json.dumps(export_completions(ROOT), indent=2))
