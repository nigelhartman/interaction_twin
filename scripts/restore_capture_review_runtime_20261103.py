"""Explicit archived-runtime selection for historical private capture snapshots.

No fallback reads, downloads, writes or runtime repair. Existing V1/V2 selections
remain immutable and fail closed if their mutable source files changed. This
helper substitutes only exact reviewed archived runtime references in two newly
versioned snapshots; all source, model, manifest and URL identities stay intact.
"""
from __future__ import annotations
import copy,json
from pathlib import Path
from scripts.launch_capture_review import checked,require,ROOT

MANIFEST='output/new-captures-20261103/capture-review-runtime-archive-v1/manifest.json'
MANIFEST_SHA='2ad7891be34e581146e69d1e55c01c68000b6e81c290fe6bde77e5eadf269e87'
MANIFEST_BYTES=5432
ALLOWED_URLS={'/showcase/aligned-scene-contract.js','/showcase/aligned-scene-client.js'}
ARCHIVE_PREFIX='output/new-captures-20261103/capture-review-runtime-archive-v1/runtime/'


def restore_spec(spec,*,root=ROOT):
    root=Path(root).resolve()
    try:
        raw=checked({'path':MANIFEST,'sha256':MANIFEST_SHA,'bytes':MANIFEST_BYTES},root)
    except (ValueError,OSError):
        raise ValueError('Archived runtime selection manifest changed') from None
    manifest=json.loads(raw)
    require(manifest.get('format')=='interaction-twin-private-capture-runtime-archive'and manifest.get('version')==1,
        'Exact archived runtime manifest required')
    records=manifest.get('snapshots')
    require(isinstance(records,list)and len(records)==2,'Exact two archived laptop snapshot variants required')
    matches=[r for r in records if r['source_snapshot_ref']==spec.get('snapshot_ref')]
    if not matches:return copy.deepcopy(spec)
    require(len(matches)==1,'Ambiguous archived runtime selection')
    record=matches[0]
    require({k:spec.get(k)for k in ['id','kind','port','prefix']}==record['viewer_identity'],'Archived snapshot viewer identity changed')
    source=json.loads(checked(record['source_snapshot_ref'],root))
    restored=json.loads(checked(record['restored_snapshot_ref'],root))
    expected=copy.deepcopy(source)
    require(set(record['runtime_substitutions'])==ALLOWED_URLS,'Only exact two runtime routes may be substituted')
    for url,pair in record['runtime_substitutions'].items():
        require(set(pair)=={'original_ref','archived_ref'},'Exact runtime substitution pair required')
        original,archived=pair['original_ref'],pair['archived_ref']
        require(source['routes'].get(url)==original,'Historical runtime reference differs from archive selection')
        require(set(archived)=={'path','sha256','bytes'}and archived['path'].startswith(ARCHIVE_PREFIX)
            and archived['sha256']==original['sha256']and archived['bytes']==original['bytes'],
            'Archived runtime must preserve exact historical bytes')
        checked(archived,root)
        expected['routes'][url]=archived
    require(restored==expected,'Restored snapshot changed more than archived runtime filesystem references')
    result=copy.deepcopy(spec);result['snapshot_ref']=copy.deepcopy(record['restored_snapshot_ref'])
    return result
