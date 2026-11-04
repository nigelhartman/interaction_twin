"""Read-only discovery of saved constrained replays during workspace refresh.

A refresh never fits an independent scene, contacts a model, or starts a paid
request. Only a fully verified descriptive object-program companion can update
the local index; a missing/failed mechanism phase remains explicitly pending.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
import tempfile


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _local(root, relative):
    if not isinstance(relative, str) or not relative.startswith('data/'):
        raise ValueError('A replay must reference a saved data/ artifact')
    if any(p.startswith('.') for p in Path(relative).parts):
        raise ValueError('Hidden or traversing replay inputs are unsupported')
    target = (root / relative).resolve()
    if not target.is_relative_to(root / 'data') or not target.is_file():
        raise ValueError('Saved replay input is unavailable')
    return target


def publish_constrained_result(companion, program, *, root):
    """Atomically index a validated companion, never an independent baseline."""
    import object_program
    root = Path(root).resolve()
    program = _local(root, str(Path(program).relative_to(root)) if Path(program).is_absolute() else str(program))
    document = companion['document']
    object_program.validate(document, root=root)
    artifact = object_program.checked_ref(companion['artifact'], root)
    if json.loads(artifact.read_text()) != document:
        raise ValueError('Object-program companion changed before publication')
    expected = {'path': program.relative_to(root).as_posix(), 'sha256': _sha(program), 'bytes': program.stat().st_size}
    if document['evidence_program'] != expected:
        raise ValueError('Constrained replay belongs to a different original program')
    manifest = document['constrained_scene']
    object_program.checked_ref(manifest, root)
    source = document['source']['sha256']
    folder = root / 'showcase'; folder.mkdir(exist_ok=True)
    path = folder / 'aligned-scenes.json'
    with (folder / '.aligned-scenes.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = json.loads(path.read_text()) if path.exists() else {'format': 'sam3d-aligned-scene-index', 'version': 1, 'scenes': []}
        if current.get('format') != 'sam3d-aligned-scene-index' or current.get('version') != 1 or not isinstance(current.get('scenes'), list):
            raise ValueError('Scene index contract differs')
        rows = [row for row in current['scenes'] if row['source_sha256'] != source]
        rows.append({'source_sha256': source, 'manifest': manifest, 'object_program': companion['artifact']})
        if len(rows) > 100:
            raise ValueError('Scene index budget exceeded')
        encoded = json.dumps({'format': 'sam3d-aligned-scene-index', 'version': 1, 'scenes': rows}, indent=2, allow_nan=False) + '\n'
        if not path.exists() or path.read_text() != encoded:
            with tempfile.NamedTemporaryFile('w', dir=folder, prefix='.aligned-scenes-', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name); stream.write(encoded)
            temporary.replace(path)
    return manifest


def publish_estimated_replays(cases, *, root, timeout_s=40, maximum_new=1):
    """Compatibility API: reuse verified constrained exports; perform zero work.

    The old timeout/new-count arguments remain bounded for existing callers,
    but no fitter, API or GPU is launched, even when a companion is missing.
    """
    import object_program
    root = Path(root).resolve()
    if type(maximum_new) is not int or not 0 <= maximum_new <= 3 or not 1 <= timeout_s <= 90:
        raise ValueError('Invalid read-only replay refresh bounds')
    candidates = []
    for case in cases:
        assets = case.get('interaction', {}).get('assets', [])
        if not assets or not all(a.get('backend') == 'Meta SAM 3D Objects' for a in assets):
            continue
        try:
            program = _local(root, case.get('interaction_path'))
            if _sha(program) != case.get('original_sha256'):
                raise ValueError('Saved program bytes changed')
            candidates.append((program.stat().st_mtime_ns, case['id'], case, program))
        except (OSError, ValueError, KeyError):
            continue
    seen, results = set(), []
    for _, _, case, program in sorted(candidates, key=lambda row: (row[0], row[1]), reverse=True):
        source = case['interaction'].get('source', {}).get('sha256')
        if source in seen:
            continue
        seen.add(source)
        row = {'case_id': case['id'], 'source_sha256': source, 'display_derivative_only': True,
               'accepted_motion_changed': False, 'physical_accuracy_verified': False, 'simulation_ready': False,
               'new_model_calls': 0, 'new_api_calls': 0, 'new_cpu_fits': 0, 'independent_scene_fallback': False}
        try:
            companion = object_program.discover(root, case['original_sha256'], source)
            if companion is None:
                row.update(status='pending_constrained_mechanism_pipeline')
            else:
                manifest = publish_constrained_result(companion, program, root=root)
                row.update(status='estimated_constrained_replay_ready', reused=True,
                           manifest=manifest, object_program=companion['artifact'])
        except (ValueError, OSError, KeyError, TypeError) as exc:
            row.update(status='constrained_replay_unavailable', diagnostic=str(exc)[:1000])
        results.append(row)
    return results
