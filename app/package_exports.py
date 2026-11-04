"""Bounded local package export requests for published original programs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import uuid

from twin_catalog import TwinCatalog

ROOT = Path(__file__).resolve().parent


class ExportConflict(ValueError):
    pass


def create_export(twin_id, request, *, root=ROOT, exporter=None):
    """Export only a hash-matched, published original. No model is executed."""
    root = Path(root).resolve()
    if not isinstance(request, dict) or set(request) - {'base_sha256', 'include_video'}:
        raise ValueError('Expected only base_sha256 and optional include_video')
    if not isinstance(request.get('base_sha256'), str) or not re.fullmatch(r'[a-f0-9]{64}', request['base_sha256']):
        raise ValueError('A source program SHA256 is required')
    include_video = request.get('include_video', False)
    if not isinstance(include_video, bool):
        raise ValueError('include_video must be a boolean')
    _, _, digest = TwinCatalog(root).load(twin_id)
    if request['base_sha256'] != digest:
        raise ExportConflict('The original program changed; reload before exporting')
    export_root = root / 'data/portable-exports'
    if export_root.resolve() != export_root:
        raise ValueError('Export storage must not use symlinks')
    export_root.mkdir(parents=True, exist_ok=True)
    name = 'export-' + uuid.uuid4().hex
    directory = export_root / name
    if exporter is None:
        from portable_twin import export_case
        exporter = export_case
    started = datetime.now(timezone.utc).isoformat()
    try:
        result = exporter(twin_id, directory, root=root, include_video=include_video)
        required = {'index': directory / 'index.html', 'manifest': directory / 'manifest.json',
                    'zip': Path(str(directory) + '.zip')}
        for field, expected in required.items():
            actual = Path(result[field]).resolve()
            if actual != expected or not actual.is_file():
                raise ValueError('Package exporter did not produce its declared ' + field)
        def public(path):
            return '/files/' + str(path.relative_to(root))
        return {'status': 'complete', 'twin_id': twin_id, 'base_sha256': digest,
                'viewer_url': public(required['index']), 'manifest_url': public(required['manifest']),
                'zip_url': public(required['zip']), 'source_video_included': include_video,
                'manifest_sha256': result['manifest_sha256'], 'physical_simulation_ready': False}
    except Exception as exc:
        # Preserve partial artifacts with a truthful receipt; do not present a
        # failed or incomplete package as a downloadable successful export.
        receipt = {'status': 'failed', 'twin_id': twin_id, 'base_sha256': digest,
                   'started_at': started, 'error': type(exc).__name__ + ': ' + str(exc)}
        (export_root / (name + '-failure.json')).write_text(json.dumps(receipt, indent=2)+'\n')
        raise
