"""Select exact archived runtime bytes for immutable V1/V2/V3 capture reviews.

No extraction, fallback, download or writer runs here. Only three preverified
JavaScript route references change; source videos, meshes, manifests and URLs
remain identical. Unmapped selections are returned unchanged and fail closed
under the original launcher's ordinary file/hash validation.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path

from scripts.launch_capture_review import checked, require, ROOT

MANIFEST = 'output/new-captures-20261104/capture-review-runtime-archive-v2/manifest.json'
MANIFEST_SHA = '631381b53e2134b31a16072ec3be611a3dfacb76330737f92cf769b70d5b8b2b'
MANIFEST_BYTES = 11607
ALLOWED_URLS = {'/showcase/aligned-scene.js', '/showcase/aligned-scene-contract.js',
                '/showcase/aligned-scene-client.js'}
ARCHIVE_PREFIX = 'output/new-captures-20261104/capture-review-runtime-archive-v2/runtime/'


def restore_spec(spec, *, root=ROOT):
    root = Path(root).resolve()
    try:
        raw = checked({'path': MANIFEST, 'sha256': MANIFEST_SHA,
                       'bytes': MANIFEST_BYTES}, root)
    except (ValueError, OSError):
        raise ValueError('Archived runtime V2 selection manifest changed') from None
    manifest = json.loads(raw)
    require(manifest.get('format') == 'interaction-twin-private-capture-runtime-archive'
            and manifest.get('version') == 2, 'Exact archived runtime V2 manifest required')
    records = manifest.get('snapshots')
    require(isinstance(records, list) and len(records) == 4,
            'Exact four historical snapshot bindings required')
    matches = [r for r in records if r['source_snapshot_ref'] == spec.get('snapshot_ref')]
    if not matches:
        return copy.deepcopy(spec)
    require(len(matches) == 1, 'Ambiguous archived runtime selection')
    record = matches[0]
    require({k: spec.get(k) for k in ['id', 'kind', 'port', 'prefix']} == record['viewer_identity'],
            'Archived snapshot viewer identity changed')
    source = json.loads(checked(record['source_snapshot_ref'], root))
    restored = json.loads(checked(record['restored_snapshot_ref'], root))
    require(set(record['runtime_substitutions']) == ALLOWED_URLS,
            'Only the exact three runtime routes may be substituted')
    expected = copy.deepcopy(source)
    for url, pair in record['runtime_substitutions'].items():
        require(set(pair) == {'original_ref', 'archived_ref'}, 'Exact runtime substitution pair required')
        original, archived = pair['original_ref'], pair['archived_ref']
        require(source['routes'].get(url) == original,
                'Historical runtime reference differs from its exact archive binding')
        require(set(archived) == {'path', 'sha256', 'bytes'}
                and archived['path'].startswith(ARCHIVE_PREFIX)
                and archived['sha256'] == original['sha256']
                and archived['bytes'] == original['bytes'],
                'Archived runtime must preserve exact historical bytes')
        checked(archived, root)
        expected['routes'][url] = archived
    require(restored == expected,
            'Restored snapshot changed more than the three archived runtime paths')
    result = copy.deepcopy(spec)
    result['snapshot_ref'] = copy.deepcopy(record['restored_snapshot_ref'])
    return result
