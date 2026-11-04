"""Select one exact historical HTML route after archived JavaScript V2.

No fallback/extraction/writer/server runs here. The current public HTML must
continue to fail its old pin; only an explicit immutable archive selection may
serve the identical original bytes under the original URL.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
from scripts.launch_capture_review import checked, require, ROOT
from scripts.restore_capture_review_runtime_20261104_v2 import restore_spec as restore_javascript

MANIFEST = 'output/new-captures-20261104/capture-review-runtime-archive-v3-1/manifest.json'
MANIFEST_SHA = '327d087881fb2d1ed7cb5b52dbf3659612a2fb43c2718bfbd029491313d55a10'
MANIFEST_BYTES = 5467
ALLOWED_URLS = {'/showcase/aligned-scene.html'}
ARCHIVE_PREFIX = 'output/new-captures-20261104/capture-review-runtime-archive-v3-1/runtime/'


def restore_spec(spec, *, root=ROOT):
    root = Path(root).resolve()
    selected = restore_javascript(spec, root=root)
    try:
        raw = checked({'path': MANIFEST, 'sha256': MANIFEST_SHA,
                       'bytes': MANIFEST_BYTES}, root)
    except (ValueError, OSError):
        raise ValueError('Archived runtime V3 HTML selection manifest changed') from None
    manifest = json.loads(raw)
    require(manifest.get('format') == 'interaction-twin-private-capture-runtime-archive'
            and manifest.get('version') == 3, 'Exact archived runtime V3 HTML manifest required')
    from scripts import restore_capture_review_runtime_20261104_v2 as base
    require(manifest.get('base_runtime_archive_ref') ==
            {'path': base.MANIFEST, 'sha256': base.MANIFEST_SHA, 'bytes': base.MANIFEST_BYTES},
            'HTML archive must retain its exact prior JavaScript selection')
    records = manifest.get('snapshots')
    require(isinstance(records, list) and len(records) == 2,
            'Exact two distinct historical HTML snapshot bindings required')
    matches = [record for record in records if selected.get('snapshot_ref') in
               [record['source_snapshot_ref'], record['restored_snapshot_ref']]]
    if not matches:return copy.deepcopy(selected)
    require(len(matches) == 1, 'Ambiguous archived HTML selection')
    record = matches[0]
    require({key: selected.get(key) for key in ['id','kind','port','prefix']} == record['viewer_identity'],
            'Archived snapshot viewer identity changed')
    source = json.loads(checked(record['source_snapshot_ref'], root))
    restored = json.loads(checked(record['restored_snapshot_ref'], root))
    require(set(record['runtime_substitutions']) == ALLOWED_URLS,
            'Only the exact archived HTML route may be substituted')
    expected = copy.deepcopy(source)
    for url, pair in record['runtime_substitutions'].items():
        require(set(pair) == {'original_ref','archived_ref'}, 'Exact HTML substitution pair required')
        original, archived = pair['original_ref'], pair['archived_ref']
        require(source['routes'].get(url) == original, 'Historical HTML reference differs from its archive binding')
        require(set(archived) == {'path','sha256','bytes'}
                and archived['path'].startswith(ARCHIVE_PREFIX)
                and archived['sha256'] == original['sha256'] and archived['bytes'] == original['bytes'],
                'Archived HTML must preserve exact historical bytes')
        checked(archived, root)
        expected['routes'][url] = archived
    require(restored == expected, 'Restored snapshot changed more than the archived HTML filesystem path')
    result = copy.deepcopy(selected)
    result['snapshot_ref'] = copy.deepcopy(record['restored_snapshot_ref'])
    return result
