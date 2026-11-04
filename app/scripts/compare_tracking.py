#!/usr/bin/env python3
"""Create source-time comparison evidence for a denser tracking experiment."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking_comparison import compare, sha, svg

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--candidate', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
if a.out.exists():
    raise ValueError('Preserve prior comparisons; use a fresh output directory')
record = compare(json.loads(a.baseline.read_text()), json.loads(a.candidate.read_text()))
record['input_files'] = {'baseline': str(a.baseline.resolve()), 'baseline_sha256': sha(a.baseline),
                         'candidate': str(a.candidate.resolve()), 'candidate_sha256': sha(a.candidate)}
a.out.mkdir(parents=True)
(a.out/'comparison.json').write_text(json.dumps(record, indent=2)+'\n')
(a.out/'comparison.svg').write_text(svg(record))
(a.out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Tracking comparison</title>'
    '<style>body{font:17px system-ui;background:#111c29;color:#edf4fb;margin:28px}img{max-width:100%}a{color:#77e3c7}pre{white-space:pre-wrap}</style>'
    '<h1>Baseline and dense tracking</h1><p>Compare the same source interval and generated mesh. These are internal support checks, not measured 3D accuracy.</p>'
    '<img src="comparison.svg" alt="Source-time tracking observations"><pre>'
    + json.dumps(record['nonreference_matched_time_transitions'], indent=2)
    + '</pre><p><a href="comparison.json">All samples, timing and limitations</a></p>')
print(json.dumps({'output': str(a.out), 'matched_time_changes': record['nonreference_matched_time_transitions']}, indent=2))
