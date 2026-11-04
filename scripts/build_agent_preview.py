#!/usr/bin/env python3
"""Snapshot real validated catalog responses for the local browser preview.

No model inference, MCP transport, installation, credentials or network access.
The stdio server's SDK transport is tested separately from this static preview.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twin_catalog import TwinCatalog


def between_sample_time(document, exact):
    """Choose a real gap, never a nearby observation disguised as a state."""
    samples = sorted({0., float(document['source']['duration_s']),
                      *[float(row['timestamp_s']) for row in document['evidence']],
                      *[float(row['timestamp_s']) for row in document['motion']]})
    gaps = [(a, b) for a, b in zip(samples, samples[1:]) if b-a > 1e-5]
    if not gaps:
        return None
    left, right = min(gaps, key=lambda pair: (min(abs(pair[0]-exact), abs(pair[1]-exact)), pair[0]))
    return (left+right)/2


def call(name, arguments, response):
    return {'request': {'name': name, 'arguments': arguments}, 'response': response}


def build(root=ROOT, out=None, max_cases=25):
    root = Path(root).resolve()
    out = Path(out or root/'workspace').resolve()
    if not out.is_relative_to(root/'workspace'):
        raise ValueError('Preview output must stay inside the local workspace')
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or not 1 <= max_cases <= 25:
        raise ValueError('Preview case budget must be between one and 25')
    catalog = TwinCatalog(root)
    rows, errors = [], []
    for entry in catalog.entries()[:max_cases]:
        twin_id = entry['id']
        try:
            _, document, digest = catalog.load(twin_id)
            evidence = {row['id']: row for row in document['evidence']}
            assertion = next((row for row in document['assertions'] if row['value'] != 'unknown'), None)
            exact = evidence[assertion['evidence_id']]['timestamp_s'] if assertion else document['evidence'][0]['timestamp_s']
            gap = between_sample_time(document, exact)
            step_id = document['steps'][0]['id'] if document['steps'] else None
            arguments = {'twin_id': twin_id}
            procedure_arguments = arguments | ({'step_id': step_id} if step_id else {})
            exact_response = catalog.state(twin_id, exact)
            gap_response = catalog.state(twin_id, gap) if gap is not None else None
            if gap_response is not None and (any(row['value'] != 'unknown' for row in gap_response['states'].values()) or gap_response['conditional_poses']):
                raise ValueError('Between-sample preview unexpectedly contains state or motion')
            outputs = {
                'describe_twin': call('describe_twin', arguments, catalog.describe(twin_id)),
                'get_procedure': call('get_procedure', procedure_arguments, catalog.procedure(twin_id, step_id)),
                'query_state_exact': call('query_state', arguments | {'timestamp_s': exact}, exact_response),
                'get_simulation_manifest': call('get_simulation_manifest', arguments, catalog.simulation_manifest(twin_id)),
            }
            if gap_response is not None:
                outputs['query_state_gap'] = call('query_state', arguments | {'timestamp_s': gap}, gap_response)
            # Recheck the original program identity before publishing the snapshot.
            _, _, final_digest = catalog.load(twin_id)
            if digest != final_digest:
                raise ValueError('Program changed during preview generation')
            rows.append({'id': twin_id, 'title': entry['title'], 'summary': document['summary'],
                         'program_sha256': digest, 'asset_count': len(document['assets']),
                         'step_count': len(document['steps']), 'relationship_count': len(document['relationships']),
                         'exact_timestamp_s': exact, 'exact_has_nonunknown_assertion': assertion is not None,
                         'gap_timestamp_s': gap, 'first_step_id': step_id,
                         'thumbnail_path': evidence[assertion['evidence_id']]['frame_path'] if assertion else document['evidence'][0]['frame_path'],
                         'assets': [catalog.asset_descriptor(twin_id, asset) for asset in document['assets']],
                         'outputs': outputs})
        except (ValueError, KeyError, OSError, StopIteration) as exc:
            errors.append({'twin_id': twin_id, 'error': str(exc)})
    setup = {'mcpServers': {'interaction-twin': {
        'command': str(root/'.venv-mcp/bin/python'),
        'args': [str(root/'scripts/serve_twin_mcp.py')],
    }}}
    document = {'format': 'interaction-twin-saved-agent-preview', 'version': '0.1',
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'method': 'actual TwinCatalog Python calls over hash-validated saved artifacts',
                'browser_executes_mcp': False, 'browser_executes_models': False,
                'stdio_transport_tested_by_this_builder': False, 'network_requests_by_builder': False,
                'catalog_code_sha256': hashlib.sha256((root/'twin_catalog.py').read_bytes()).hexdigest(),
                'setup': setup, 'twins': rows, 'build_errors': errors}
    out.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    (out/'agent-preview.json').write_text(encoded+'\n')
    (out/'agent-preview.js').write_text('window.TWIN_AGENT_PREVIEW='+encoded+';\n')
    print(json.dumps({'page': str(out/'agents.html'), 'valid_twins': len(rows), 'build_errors': errors,
                      'snapshot_bytes': len(encoded.encode()), 'fresh_model_inference': False}))
    return document


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--max-cases', type=int, default=25)
    args = parser.parse_args()
    build(out=args.out, max_cases=args.max_cases)
