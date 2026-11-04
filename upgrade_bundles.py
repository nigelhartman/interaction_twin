"""Local, provenance-bound handoff to already provisioned GPU adapters.

This module never contacts a GPU, reads credentials, downloads code/weights or
runs learned inference. Returned metadata is an attestation, not proof of model
execution or physical accuracy. Existing programs/assets are never overwritten.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import uuid
import zipfile

import numpy as np
from PIL import Image

from twin_catalog import TwinCatalog

ROOT = Path(__file__).resolve().parent
HEX64 = re.compile(r'[0-9a-f]{64}\Z')
JOB_ID = re.compile(r'job-[0-9a-f]{32}\Z')
IMPORT_ID = re.compile(r'import-[0-9a-f]{32}\Z')
TIME_DOMAIN = 'source_video_presentation_seconds'
SAM3D_COMMIT = 'f91db411c50efee93d8db7aeb323885650f6f722'
SAM31_COMMIT = 'f1647e8d0c1d280991f4d1b7d974c2704537fe66'
BACKENDS = {
    'sam3d_objects': {'label': 'SAM 3D Objects', 'adapter': 'scripts/stage3_sam3d.py',
                      'commit': SAM3D_COMMIT, 'model': 'facebook/sam-3d-objects',
                      'scope': 'one source RGB + saved SAM mask → native 3D candidate'},
    'sam31_image': {'label': 'SAM 3.1 · one image', 'adapter': 'scripts/sam31_segment.py',
                    'commit': SAM31_COMMIT, 'model': 'facebook/sam3.1',
                    'scope': 'text-segment one anchor image; no video propagation or physical identity'},
}
MAX_FILE = 256 * 1024 * 1024
MAX_TOTAL = 512 * 1024 * 1024
MAX_FILES = 512


class Conflict(ValueError):
    """The original program changed after the user selected it."""


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(Path(path).read_bytes(), parse_constant=reject)


def write(path, value):
    Path(path).write_bytes(encoded(value))


def _relative(path, root):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def _local(path, root):
    """Only nonhidden data artifacts, with no symbolic path segment."""
    root = Path(root).resolve()
    value = str(path)
    if Path(value).is_absolute():
        try:
            value = Path(value).relative_to(root).as_posix()
        except ValueError:
            raise ValueError('Artifact is outside this workspace') from None
    pure = PurePosixPath(value)
    if not value.startswith('data/') or '\\' in value or any(p.startswith('.') for p in pure.parts):
        raise ValueError('Only ordinary data artifacts are permitted')
    target = root / pure
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Symlink artifacts are not permitted')
    if not target.is_file() or target.resolve() != target:
        raise ValueError('Artifact unavailable')
    return target


def _directory(root, name):
    root = Path(root).resolve()
    for p in (root / 'data', root / 'data' / name):
        if p.is_symlink():
            raise ValueError('Storage directory must not be a symlink')
    path = root / 'data' / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checked_file(path, expected):
    if not HEX64.fullmatch(str(expected or '')) or sha(path) != expected:
        raise ValueError('Declared artifact hash differs: ' + Path(path).name)


def _pixels(image, mask):
    with Image.open(image) as opened:
        if opened.width * opened.height > 16_000_000:
            raise ValueError('Source image exceeds the 16-megapixel local inspection budget')
        rgb = np.asarray(opened.convert('RGB'))
    with Image.open(mask) as opened:
        if opened.size != (rgb.shape[1], rgb.shape[0]):
            raise ValueError('Mask dimensions differ from the source image')
        binary = np.asarray(opened)
    if binary.ndim != 2 or binary.shape != rgb.shape[:2] or not set(np.unique(binary).tolist()).issubset({0, 1, 255}):
        raise ValueError('RGB/mask must retain identical full-frame pixels and a binary mask')
    if not np.any(binary) or np.all(binary):
        raise ValueError('Mask must distinguish visible object and background')
    return rgb, binary > 0


def context(twin_id, track_id, *, root=ROOT):
    """Resolve a catalog-bound source anchor without following supplied paths."""
    root = Path(root).resolve()
    catalog = TwinCatalog(root)
    row, doc, base_sha = catalog.load(twin_id)
    track = next((t for t in doc['tracklets'] if t['id'] == track_id), None)
    asset = next((a for a in doc['assets'] if a['track_id'] == track_id), None)
    if track is None or asset is None:
        raise ValueError('Choose a published tracklet with a saved learned asset')
    asset_path = _local(asset['path'], root)
    matches = []
    for item in doc['inputs']:
        if item['kind'] != 'learned_asset_metadata':
            continue
        metadata_path = _local(item['path'], root)
        _checked_file(metadata_path, item['sha256'])
        metadata = read(metadata_path)
        glb = (metadata_path.parent / metadata.get('outputs', {}).get('glb_path', '')).resolve()
        if metadata.get('object_id') == track_id and glb == asset_path:
            matches.append((metadata_path, metadata))
    if len(matches) != 1:
        raise ValueError('Exactly one matching source reconstruction metadata record is required')
    metadata_path, metadata = matches[0]
    source = metadata['source']
    if source['video_sha256'] != doc['source']['sha256']:
        raise ValueError('Reconstruction and program source videos differ')
    selection_path = _local(source['selection_manifest_path'], root)
    _checked_file(selection_path, source['selection_manifest_sha256'])
    selection = read(selection_path)
    if selection.get('object_id') != track_id or selection.get('source', {}).get('sha256') != source['video_sha256']:
        raise ValueError('Selection lineage does not identify this track/source')
    selected = next((x for x in selection.get('selected', [])
                     if x['sample_index'] == source['sample_index'] and x['timestamp_s'] == source['timestamp_s']), None)
    if selected is None or selection.get('selection_method', {}).get('agent_visual_choice') is not False:
        raise ValueError('Saved automatic source selection is unavailable')
    image_path = _local(metadata_path.parent / source['image_path'], root)
    mask_path = _local(metadata_path.parent / source['mask_path'], root)
    _checked_file(image_path, source['image_sha256'])
    _checked_file(mask_path, source['mask_sha256'])
    rgb, mask = _pixels(image_path, mask_path)
    if source.get('time_domain') != TIME_DOMAIN or not math.isfinite(source['timestamp_s']) or source['timestamp_s'] < 0:
        raise ValueError('Invalid source PTS domain')
    provenance = copy.deepcopy(source['mask_provenance'])
    # Older selector provenance has explicit seed flags rather than this field.
    provenance['manual_intervention'] = provenance.get('manual_intervention', provenance.get('manual_seed_boxes_used', False))
    provenance.setdefault('backend', provenance.get('model', 'saved SAM backend'))
    provenance.setdefault('prompt_origin', 'saved_model_generated_prompt')
    if not isinstance(provenance['manual_intervention'], bool):
        raise ValueError('Manual assistance must be explicit')
    return {'twin_id': twin_id, 'track_id': track_id, 'title': row['title'],
            'track_label': track['label'], 'base_sha256': base_sha,
            'model_metadata_path': _relative(metadata_path, root), 'model_metadata_sha256': sha(metadata_path),
            'selection_path': _relative(selection_path, root), 'selection_sha256': sha(selection_path),
            'image_path': _relative(image_path, root), 'image_sha256': sha(image_path),
            'mask_path': _relative(mask_path, root), 'mask_sha256': sha(mask_path),
            'baseline_asset_path': asset['path'], 'baseline_asset_sha256': asset['sha256'],
            'source': {'video_sha256': source['video_sha256'], 'sample_index': source['sample_index'],
                       'timestamp_s': source['timestamp_s'], 'time_domain': TIME_DOMAIN,
                       'width': int(rgb.shape[1]), 'height': int(rgb.shape[0]),
                       'rgb_array_sha256': digest(rgb.tobytes()), 'mask_array_sha256': digest(mask.tobytes())},
            'mask_provenance': provenance, 'mask_scope': source.get('mask_scope', ''),
            'selection_rank': source.get('selection_rank', 1), 'selection_score': source.get('selection_score'),
            'physical_accuracy_verified': False}


def _job_read(job_id, root):
    if not JOB_ID.fullmatch(str(job_id)):
        raise ValueError('Invalid job ID')
    path = _local(f'data/upgrade-jobs/{job_id}/job.json', root)
    job = read(path)
    if job.get('format') != 'interaction-twin-upgrade-job' or job.get('version') != '1' or job.get('job_id') != job_id:
        raise ValueError('Invalid saved upgrade job')
    current = context(job['twin_id'], job['track_id'], root=root)
    if current != job['input_binding']:
        raise Conflict('Saved original or selected input changed; create a fresh handoff')
    for item in job['files']:
        file = _local(path.parent / item['path'], root)
        _checked_file(file, item['sha256'])
        if file.stat().st_size != item['bytes']:
            raise ValueError('Saved job file size differs')
    archive_path = _local(str(path.parent) + '.zip', root)
    archive_files = _zip_files(archive_path)
    expected = {item['path']: item for item in job['files']}
    if set(archive_files) != set(expected) | {'job.json'} or archive_files['job.json'] != path.read_bytes():
        raise ValueError('Prepared ZIP differs from the saved job')
    if any(digest(archive_files[name]) != row['sha256'] or len(archive_files[name]) != row['bytes'] for name, row in expected.items()):
        raise ValueError('Prepared ZIP input/script bytes differ')
    return path, job


def prepare_job(body, *, root=ROOT):
    root = Path(root).resolve()
    if not isinstance(body, dict) or set(body) - {'twin_id', 'track_id', 'backend', 'base_sha256', 'prompt', 'appearance'}:
        raise ValueError('Unexpected handoff request fields')
    backend = body.get('backend')
    if backend not in BACKENDS:
        raise ValueError('Choose sam3d_objects or sam31_image')
    appearance = body.get('appearance', 'vertex-color')
    if appearance not in {'vertex-color', 'uv-baked'} or ('appearance' in body and backend != 'sam3d_objects'):
        raise ValueError('Choose vertex-color or uv-baked appearance for SAM 3D only')
    info = context(body.get('twin_id'), body.get('track_id'), root=root)
    if body.get('base_sha256') != info['base_sha256']:
        raise Conflict('Original program changed; refresh before preparing a handoff')
    supplied_prompt = body.get('prompt')
    if supplied_prompt is not None and (not isinstance(supplied_prompt, str) or not supplied_prompt.strip() or len(supplied_prompt) > 200 or any(ord(c) < 32 for c in supplied_prompt)):
        raise ValueError('A text concept must contain 1–200 ordinary characters')
    if supplied_prompt is not None and backend != 'sam31_image':
        raise ValueError('SAM 3D uses the saved mask, not a text prompt')
    prompt = supplied_prompt.strip() if supplied_prompt is not None else info['track_label']
    job_id = 'job-' + uuid.uuid4().hex
    directory = _directory(root, 'upgrade-jobs') / job_id
    directory.mkdir()
    (directory / 'input').mkdir()
    (directory / 'scripts').mkdir()
    for role in ('image', 'mask'):
        shutil.copy2(_local(info[role + '_path'], root), directory / 'input' / ('rgb.png' if role == 'image' else 'mask.png'))
    shutil.copy2(_local(info['selection_path'], root), directory / 'input/selection-original.json')
    manifest = {'schema_version': '0.1', 'object_id': info['track_id'],
                'image': 'rgb.png', 'mask': 'mask.png', 'image_sha256': info['image_sha256'],
                'mask_sha256': info['mask_sha256'], 'source': info['source'],
                'mask_provenance': info['mask_provenance'],
                'selection_source_sha256': info['selection_sha256']}
    write(directory / 'input/input.json', manifest)
    adapter = BACKENDS[backend]['adapter']
    for relative in (adapter, 'scripts/pack_upgrade_result.py', 'scripts/run_upgrade_job.py'):
        # The roots in tests can deliberately supply reviewed local script fixtures.
        script = root / relative
        if script.is_symlink() or not script.is_file():
            raise ValueError('Reviewed adapter is unavailable')
        shutil.copy2(script, directory / relative)
    pack_command = 'python scripts/pack_upgrade_result.py --job job.json --adapter-output output --zip returned-result.zip'
    if backend == 'sam3d_objects':
        command = ['python', adapter, 'run', '--input-manifest', 'input/input.json', '--repo', '/REPLACE/pinned/sam-3d-objects',
                   '--config', '/REPLACE/approved/pipeline.yaml', '--expected-code-commit', SAM3D_COMMIT,
                   '--appearance', appearance, '--output', 'output']
    else:
        command = ['python', adapter, 'run', '--input-manifest', 'input/input.json', '--repo', '/REPLACE/pinned/sam3',
                   '--checkpoint-manifest', '/REPLACE/approved/checkpoint-provenance.json',
                   '--expected-code-commit', SAM31_COMMIT, '--prompt', prompt,
                   '--prompt-origin', 'user_text_concept' if supplied_prompt is not None else 'cached_model_track_label', '--output', 'output']
    # Commands are argument arrays. No untrusted text enters shell expansion.
    write(directory / 'command.json', {'argv': command, 'pack_result_argv': pack_command.split()})
    (directory / 'README.txt').write_text(
        'INTERACTION TWIN · GPU HANDOFF\n\n'
        + BACKENDS[backend]['scope'] + '\n\n'
        'This is a private source-image bundle. No full video, credentials or weights are included.\n'
        'Nothing has been submitted, downloaded or inferred by preparing it.\n'
        'Use only inside a real allocated GPU session with the official runtime and approved local weights.\n'
        + ('First check: python scripts/run_upgrade_job.py preflight --repo /PATH/sam-3d-objects --config /PATH/pipeline.yaml\n'
           'Then run: python scripts/run_upgrade_job.py run --repo /PATH/sam-3d-objects --config /PATH/pipeline.yaml\n'
           if backend == 'sam3d_objects' else
           'First check: python scripts/run_upgrade_job.py preflight --repo /PATH/sam3 --checkpoint-manifest /PATH/checkpoint-provenance.json\n'
           'Then run: python scripts/run_upgrade_job.py run --repo /PATH/sam3 --checkpoint-manifest /PATH/checkpoint-provenance.json\n')
        + '\nThe runner checks every shipped input/script hash before invoking the fixed adapter.\n'
        'Replace the /REPLACE paths in command.json. Review argv; run with Python subprocess.run(argv, check=True), not shell=True.\n'
        'For a non-inferencing readiness check replace argv[2] "run" with "preflight" and choose a fresh report filename for --output.\n'
        'After a successful run, package the untouched output:\n' + pack_command + '\n\n'
        'Return returned-result.zip through the local app. Import validates contract, file hashes and source binding.\n'
        'Adapter receipts attest execution; they do not independently prove it or establish physical accuracy.\n'
        'SAM 3.1 results remain anchor-image comparison instances; no automatic video ID or old audit is inherited.\n'
        'SAM 3D results remain native model candidates. No existing physical transform or insertion is assumed.\n'
        + ('UV baking is experimental model-generated appearance from Gaussian renders, not photographic texture recovered from the original video.\n'
           if backend == 'sam3d_objects' and appearance == 'uv-baked' else '')
    )
    files = [{'path': p.relative_to(directory).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in sorted(directory.rglob('*')) if p.is_file()]
    job = {'format': 'interaction-twin-upgrade-job', 'version': '1', 'job_id': job_id,
           'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'prepared_not_submitted',
           'twin_id': info['twin_id'], 'track_id': info['track_id'], 'base_sha256': info['base_sha256'],
           'backend': backend, 'backend_label': BACKENDS[backend]['label'],
           'expected_code_commit': BACKENDS[backend]['commit'], 'model_repository': BACKENDS[backend]['model'],
           'input_binding': info, 'input_manifest_sha256': sha(directory / 'input/input.json'),
           'input_manifest': manifest,
           'files': files, 'full_video_included': False, 'learned_inference_executed': False,
           'fixture': False, 'command_argv': command,
           'readiness': 'GPU/runtime/approved local weights still required; no remote availability checked',
           'physical_accuracy_verified': False}
    if backend == 'sam3d_objects':
        job['appearance_mode'] = appearance
    write(directory / 'job.json', job)
    zip_path = Path(str(directory) + '.zip')
    with zipfile.ZipFile(zip_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(directory.rglob('*')):
            if p.is_file():
                archive.write(p, p.relative_to(directory).as_posix())
    return _job_summary(directory / 'job.json', job, root)


def _job_summary(path, job, root):
    base = _relative(path.parent, root)
    return {k: job[k] for k in ('job_id', 'twin_id', 'track_id', 'base_sha256', 'backend', 'backend_label', 'status', 'created_at_utc', 'readiness')} | {
        'directory': str(path.parent), 'job_path': str(path), 'zip_path': str(path.parent) + '.zip',
        'job_relative_path': base + '/job.json', 'zip_relative_path': base + '.zip',
        'image_path': base + '/input/rgb.png', 'mask_path': base + '/input/mask.png',
        'source': job['input_binding']['source'], 'full_video_included': False,
        'learned_inference_executed': False,
        'appearance_mode': job.get('appearance_mode', 'vertex-color') if job['backend'] == 'sam3d_objects' else None}


def list_jobs(*, root=ROOT):
    base = Path(root) / 'data/upgrade-jobs'
    if not base.exists():
        return []
    rows = []
    for path in sorted(base.glob('job-*/job.json'), reverse=True):
        try:
            valid, job = _job_read(path.parent.name, root)
            rows.append(_job_summary(valid, job, root))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            rows.append({'job_id': path.parent.name, 'status': 'invalid_saved_job', 'error': str(exc)})
    return rows


def _safe_name(name):
    if not isinstance(name, str) or not name or '\\' in name or '\x00' in name:
        raise ValueError('Invalid archive path')
    p = PurePosixPath(name)
    if p.is_absolute() or name != p.as_posix() or any(part.startswith('.') or ':' in part for part in p.parts):
        raise ValueError('Archive contains hidden, absolute or traversing paths')
    return name


def _read_zip_files(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_TOTAL:
        raise ValueError('Invalid or oversized result ZIP')
    files, total = {}, 0
    with zipfile.ZipFile(path) as archive:
        items = archive.infolist()
        if len(items) > MAX_FILES:
            raise ValueError('Too many result ZIP members')
        for item in items:
            name = _safe_name(item.filename.rstrip('/') if item.is_dir() else item.filename)
            mode = item.external_attr >> 16
            if stat.S_ISLNK(mode) or (mode and stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError('Archive symlinks/special files are forbidden')
            if item.is_dir():
                continue
            if name in files or item.flag_bits & 1 or item.file_size > MAX_FILE:
                raise ValueError('Duplicate, encrypted or oversized result ZIP member')
            total += item.file_size
            if total > MAX_TOTAL:
                raise ValueError('Expanded result ZIP exceeds budget')
            files[name] = archive.read(item)
    return files


def _zip_files(path):
    try:
        return _read_zip_files(path)
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        raise ValueError('Invalid or unsupported ZIP archive: ' + type(exc).__name__) from None


def _json_bytes(raw):
    return json.loads(raw, parse_constant=lambda s: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def _bound_metadata(job, files):
    raw = files.get('raw/metadata.json')
    if raw is None or len(raw) > 4 * 1024 * 1024:
        raise ValueError('Missing or oversized adapter metadata')
    metadata = _json_bytes(raw)
    if metadata.get('fixture') is not None and metadata.get('fixture') is not False:
        raise ValueError('Fixture metadata cannot be imported as a model result')
    if metadata.get('learned_inference_executed') is not True:
        raise ValueError('Adapter does not attest completed learned inference')
    if metadata.get('model_repository') != job['model_repository']:
        raise ValueError('Wrong model repository')
    code = metadata.get('code', {})
    official = 'https://github.com/facebookresearch/sam-3d-objects' if job['backend'] == 'sam3d_objects' else 'https://github.com/facebookresearch/sam3'
    if code.get('commit') != job['expected_code_commit'] or code.get('tracked_changes', '') or code.get('origin') != official:
        raise ValueError('Adapter code pin/cleanliness differs from handoff')
    if metadata.get('errors'):
        raise ValueError('Adapter reported preflight errors')
    inputs = metadata.get('inputs', {})
    if inputs.get('manifest_sha256') != job['input_manifest_sha256'] or inputs.get('manifest') != job['input_manifest']:
        raise ValueError('Adapter input manifest differs from handoff')
    info = job['input_binding']
    source = inputs.get('manifest', {}).get('source', {})
    if source != info['source'] or inputs.get('manifest', {}).get('object_id') != info['track_id']:
        raise ValueError('Wrong original source PTS, track or video')
    for role, label in (('image', 'image'), ('mask' if job['backend'] == 'sam3d_objects' else 'reference_mask', 'mask')):
        if inputs.get('files', {}).get(role, {}).get('sha256') != info[label + '_sha256']:
            raise ValueError('Adapter source image/mask hash differs')
    runtime = metadata.get('runtime', {})
    if runtime.get('platform') != 'Linux' or runtime.get('device_requested', runtime.get('device')) != 'cuda' or runtime.get('cuda_available') is not True:
        raise ValueError('Official GPU adapter runtime attestation is incomplete')
    return metadata


def _validate_result_files(job, receipt, files):
    """Apply the same bounded allowlist to uploads and saved import revalidation."""
    if receipt.get('format') != 'interaction-twin-upgrade-result' or receipt.get('version') != '1':
        raise ValueError('Unsupported result format')
    if receipt.get('fixture') is not False:
        raise ValueError('Fixtures cannot enter production model imports')
    if receipt.get('job_id') != job['job_id'] or receipt.get('backend') != job['backend'] or receipt.get('input_binding') != job['input_binding']:
        raise ValueError('Returned result is not bound to this exact saved job')
    fixed = {'raw/metadata.json', 'raw/source-rgb.png', 'raw/overlay.png'} if job['backend'] == 'sam31_image' else {
        'raw/metadata.json', 'raw/object.glb', 'raw/object-splat.ply', 'raw/layout-and-pointmap.npz',
        'raw/inputs/manifest.json', 'raw/inputs/rgb.png', 'raw/inputs/mask.png',
        'raw/inputs/image-source.png', 'raw/inputs/mask-source.png'}
    records = receipt.get('files', [])
    declared = set()
    if not isinstance(records, list):
        raise ValueError('Invalid result file list')
    for item in records:
        name = _safe_name(item.get('path'))
        allowed = name in fixed or (job['backend'] == 'sam31_image' and re.fullmatch(r'raw/instance-[0-9]{6}-(mask|rgba)\.png', name))
        if not allowed or name in declared or name not in files or item.get('bytes') != len(files[name]) or item.get('sha256') != digest(files[name]):
            raise ValueError('Result file list, hash or size differs')
        declared.add(name)
    if set(files) != declared | {'result.json'}:
        raise ValueError('Undeclared files in result ZIP')


def _safe_layout(raw):
    """Check the nested compressed archive before loading only three tiny arrays."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names, total = set(), 0
        for item in archive.infolist():
            if item.filename not in {'rotation.npy', 'translation.npy', 'scale.npy', 'pointmap.npy', 'pointmap_colors.npy'} or item.filename in names or item.file_size > 64 * 1024 * 1024:
                raise ValueError('Unexpected or oversized native-layout array')
            total += item.file_size
            names.add(item.filename)
        if total > 128 * 1024 * 1024 or not {'rotation.npy', 'translation.npy', 'scale.npy'} <= names:
            raise ValueError('Invalid native layout array budget')
        for name in ('rotation.npy', 'translation.npy', 'scale.npy'):
            raw_array = archive.read(name)
            # Genuine decoded layouts are four or three floats, batched at most once.
            if len(raw_array) > 8192:
                raise ValueError('Decoded layout transform is not one object')
    return np.load(io.BytesIO(raw), allow_pickle=False)


def _validate_glb(raw):
    # Full accessor/buffer validation uses the same bounded CPU decoder as export.
    from portable_twin import decode_glb_bytes
    return decode_glb_bytes(raw)


def _fit_contract_v1(raw, appearance_mode, decoded):
    """Inspect the actual GLB using the immutable fitting stage's loader.

    TextureVisuals is deliberately not converted into invented vertex colours.
    Imported UV geometry remains a useful native asset, but cannot enter the
    existing pose/tracking code until a real textured fitting adapter exists.
    """
    contract = {'version': '1', 'raw_glb_sha256': digest(raw),
                'current_adapter_requires': 'trimesh ColorVisuals.vertex_colors',
                'appearance_mode': appearance_mode, 'alignment_eligible': False,
                'alignment_ineligible_reason': None, 'visual_kind': None,
                'appearance_conversion_performed': False}
    if appearance_mode != 'vertex-color':
        contract['alignment_ineligible_reason'] = 'UV-textured native asset; current local pose/tracking adapters require vertex colours.'
        return contract
    if not decoded['primitives'] or any(not p.get('colors') or p.get('texture') for p in decoded['primitives']):
        contract['alignment_ineligible_reason'] = 'Native GLB lacks exclusively vertex-coloured primitives required by the current fitting adapters.'
        return contract
    try:
        import trimesh
        mesh = trimesh.load(io.BytesIO(raw), file_type='glb', force='mesh', process=False)
        visual = mesh.visual
        contract['visual_kind'] = getattr(visual, 'kind', None)
        colors = np.asarray(visual.vertex_colors)
        vertices = np.asarray(mesh.vertices)
        supported = (contract['visual_kind'] == 'vertex' and vertices.ndim == 2 and vertices.shape[1] == 3
                     and len(vertices) > 0 and colors.shape in {(len(vertices), 3), (len(vertices), 4)}
                     and np.isfinite(vertices).all() and np.isfinite(colors).all())
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
        contract['alignment_ineligible_reason'] = 'Current generic GLB loader cannot expose usable vertex colours (' + type(exc).__name__ + ').'
        return contract
    contract['alignment_eligible'] = bool(supported)
    if not supported:
        contract['alignment_ineligible_reason'] = 'Actual loaded mesh visual data does not satisfy the current vertex-colour fitting contract.'
    return contract


def _fit_contract(raw, appearance_mode, decoded):
    """Bind declared model appearance to a verified, versioned native adapter.

    Version one remains available to verify historical import receipts. Current
    capability is checked separately when resolving one of those imports; no
    historical receipt or original GLB is rewritten to grant a new capability.
    """
    from native_appearance import alignment_contract
    contract = {'version': '2', 'raw_glb_sha256': digest(raw),
                'current_adapter_requires': 'verified opaque native base-colour appearance',
                'appearance_mode': appearance_mode, 'alignment_eligible': False,
                'alignment_ineligible_reason': None, 'visual_kind': None,
                'alignment_adapter': None, 'appearance_conversion_performed': False}
    primitives = decoded.get('primitives', [])
    if appearance_mode == 'vertex-color':
        if not primitives or any(not p.get('colors') or p.get('texture') for p in primitives):
            contract['alignment_ineligible_reason'] = 'Declared vertex-colour output does not contain exclusively vertex-coloured primitives without textures.'
            return contract
    elif appearance_mode == 'uv-baked':
        if not any(p.get('texture') and p.get('uv') for p in primitives):
            contract['alignment_ineligible_reason'] = 'Declared UV output has no actual embedded texture and UV coordinates.'
            return contract
    else:
        contract['alignment_ineligible_reason'] = 'Unknown declared appearance mode.'
        return contract
    capability = alignment_contract(raw, 'uv_texture' if appearance_mode == 'uv-baked' else 'vertex_rgb')
    contract['native_appearance'] = capability.get('receipt')
    if not capability.get('eligible'):
        contract['alignment_ineligible_reason'] = capability.get('reason') or 'Native appearance is unsupported by the current alignment adapter.'
        return contract
    # The unchanged legacy route is kept for genuinely compatible vertex data.
    # Every other supported primitive layout uses the new native renderer; UVs
    # are sampled per pixel rather than converted into invented vertex colours.
    legacy = _fit_contract_v1(raw, appearance_mode, decoded)
    use_legacy = bool(legacy['alignment_eligible'] and
                      (capability.get('receipt') or {}).get('legacy_compatible') is True)
    contract.update(alignment_eligible=True,
                    visual_kind=legacy['visual_kind'] if use_legacy else ('texture' if appearance_mode == 'uv-baked' else 'native'),
                    alignment_adapter='legacy-vertex-colour-v1' if use_legacy else 'native-base-color-v1')
    return contract


def _validate_sam3d(job, metadata, files, root, *, fit_version='2'):
    if metadata.get('status') != 'candidate_generated_pending_visual_review' or metadata.get('backend') != 'Meta SAM 3D Objects':
        raise ValueError('SAM 3D adapter did not finish a candidate')
    outputs = metadata.get('outputs', {})
    if outputs.get('geometry_reconstructed_by_model') is not True or outputs.get('manual_geometry_fallback') is not False:
        raise ValueError('Model geometry provenance is incomplete')
    if outputs.get('metric_scale_verified') is not False or outputs.get('appearance_mode') not in ('vertex-color', 'uv-baked'):
        raise ValueError('Invalid scale/appearance provenance')
    if 'appearance_mode' in job and outputs.get('appearance_mode') != job['appearance_mode']:
        raise ValueError('Returned appearance differs from the requested GPU job')
    checkpoint = metadata.get('checkpoint', {})
    if checkpoint.get('weights_hashed') is not True or not any(HEX64.fullmatch(str(v.get('sha256', ''))) and Path(v.get('path', '')).suffix in {'.ckpt', '.pt', '.pth', '.safetensors', '.bin'} for v in checkpoint.get('files', [])):
        raise ValueError('Missing hashed local checkpoint attestation')
    required = {'object.glb', 'object-splat.ply', 'layout-and-pointmap.npz'}
    records = outputs.get('files', [])
    if {r.get('file') for r in records} != required:
        raise ValueError('SAM 3D must preserve mesh, splat and native layout')
    for item in records:
        raw = files.get('raw/' + item['file'])
        if raw is None or len(raw) != item.get('bytes') or digest(raw) != item.get('sha256'):
            raise ValueError('Raw SAM 3D artifact hash/size differs')
    decoded = _validate_glb(files['raw/object.glb'])
    if outputs['appearance_mode'] == 'vertex-color' and not any(p.get('colors') for p in decoded['primitives']):
        raise ValueError('Vertex-colour mode did not preserve model vertex colours')
    if outputs['appearance_mode'] == 'uv-baked' and not any(p.get('texture') and p.get('uv') for p in decoded['primitives']):
        raise ValueError('UV-baked mode did not preserve embedded texture/UVs')
    expected_input = _local(f"data/upgrade-jobs/{job['job_id']}/input/input.json", root).read_bytes()
    if files.get('raw/inputs/manifest.json') != expected_input:
        raise ValueError('Original adapter input manifest was not preserved')
    info = job['input_binding']
    for name, field in (('image-source.png', 'image_sha256'), ('mask-source.png', 'mask_sha256')):
        if digest(files.get('raw/inputs/' + name, b'')) != info[field]:
            raise ValueError('Original source file bytes were not preserved')
    for name in ('rgb.png', 'mask.png'):
        if digest(files.get('raw/inputs/' + name, b'')) != metadata.get('model_input_sha256', {}).get(name):
            raise ValueError('Actual model input image bytes differ')
    rgb, mask = _pixels(io.BytesIO(files['raw/inputs/rgb.png']), io.BytesIO(files['raw/inputs/mask.png']))
    if digest(rgb.tobytes()) != info['source']['rgb_array_sha256'] or digest(mask.tobytes()) != info['source']['mask_array_sha256']:
        raise ValueError('Model input arrays do not preserve the exact source pixels')
    if metadata.get('model_input_array_sha256') != {'rgb_uint8': digest(rgb.tobytes()), 'mask_bool': digest(mask.tobytes())}:
        raise ValueError('Model input array attestation differs')
    # Parse NPZ safely; arbitrary pickle payloads are never loaded.
    from sam3d_coordinates import native_layout_to_cv, DECODED_SCALE_SEMANTICS
    with _safe_layout(files['raw/layout-and-pointmap.npz']) as layout:
        transform = native_layout_to_cv(rotation_wxyz=layout['rotation'], translation=layout['translation'],
                                       scale_xyz=layout['scale'], source_commit=SAM3D_COMMIT,
                                       scale_semantics=DECODED_SCALE_SEMANTICS)
        coordinates = transform.metadata()
    coordinates.update({'convention_verified': False,
                        'source_video_pose_validation': 'not_performed',
                        'use_for_initialization_only': True})
    if fit_version not in {'1', '2'}:
        raise ValueError('Unknown saved alignment contract version')
    fit = _fit_contract_v1 if fit_version == '1' else _fit_contract
    return coordinates, fit(files['raw/object.glb'], outputs['appearance_mode'], decoded)


def _validate_sam31(job, metadata, files):
    if metadata.get('status') not in ('completed_candidates', 'completed_no_instances') or metadata.get('backend') != 'Meta SAM 3.1 Object Multiplex':
        raise ValueError('SAM 3.1 adapter did not finish its image comparison')
    checkpoint = metadata.get('checkpoint', {})
    checkpoint_manifest = checkpoint.get('manifest', {})
    if (not HEX64.fullmatch(str(checkpoint.get('sha256', ''))) or checkpoint_manifest.get('model_repository') != 'facebook/sam3.1'
            or not re.fullmatch(r'[0-9a-f]{40}', str(checkpoint_manifest.get('model_revision', '')))
            or checkpoint_manifest.get('checkpoint_file') != 'sam3.1_multiplex.pt'
            or checkpoint_manifest.get('checkpoint_sha256') != checkpoint['sha256']):
        raise ValueError('SAM 3.1 checkpoint provenance missing')
    if metadata.get('checkpoint_coverage', {}).get('strict_key_and_shape_coverage') is not True:
        raise ValueError('SAM 3.1 checkpoint coverage was not verified')
    argv = job['command_argv']
    expected_prompt = {'text': argv[argv.index('--prompt') + 1], 'origin': argv[argv.index('--prompt-origin') + 1], 'reference_mask_used_as_prompt': False}
    if metadata.get('prompt') != expected_prompt:
        raise ValueError('SAM 3.1 text concept/provenance differs from handoff')
    expected_size = (job['input_binding']['source']['width'], job['input_binding']['source']['height'])
    with Image.open(io.BytesIO(files['raw/source-rgb.png'])) as opened:
        if opened.size != expected_size:
            raise ValueError('SAM 3.1 source image dimensions differ')
        rgb = np.asarray(opened.convert('RGB'))
    if digest(rgb.tobytes()) != job['input_binding']['source']['rgb_array_sha256']:
        raise ValueError('SAM 3.1 source pixels differ')
    records = metadata.get('outputs', {}).get('instances', [])
    if len(records) > 256 or metadata.get('outputs', {}).get('instance_count') != len(records):
        raise ValueError('Invalid SAM 3.1 instance count')
    expected_status = 'completed_candidates' if records else 'completed_no_instances'
    if metadata['status'] != expected_status:
        raise ValueError('SAM 3.1 completion status contradicts its instance count')
    seen = set()
    for index, record in enumerate(records, 1):
        identifier = record.get('model_instance_id')
        if isinstance(identifier, bool) or not isinstance(identifier, int) or identifier in seen:
            raise ValueError('SAM 3.1 model instance IDs must remain unique integers')
        seen.add(identifier)
        name = record.get('mask_file')
        if name != f'instance-{index:06d}-mask.png' or record.get('rgba_file') != f'instance-{index:06d}-rgba.png':
            raise ValueError('Unexpected SAM 3.1 mask instance path')
        mask_bytes = files.get('raw/' + name)
        if mask_bytes is None or digest(mask_bytes) != record.get('mask_sha256'):
            raise ValueError('SAM 3.1 instance mask hash differs')
        with Image.open(io.BytesIO(mask_bytes)) as opened:
            if opened.size != expected_size:
                raise ValueError('SAM 3.1 instance mask dimensions differ')
            mask = np.asarray(opened)
        if mask.ndim != 2 or mask.shape != rgb.shape[:2] or not set(np.unique(mask).tolist()).issubset({0, 255}):
            raise ValueError('SAM 3.1 mask dimensions/values differ')
        if int((mask > 0).sum()) != record.get('mask_pixels') or record.get('physical_identity_verified') is not False:
            raise ValueError('SAM 3.1 visible-pixel/identity provenance differs')
        score = record.get('score')
        box = record.get('box_xywh_normalized', [])
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1 or len(box) != 4 or any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in box) or box[0] + box[2] > 1.0001 or box[1] + box[3] > 1.0001:
            raise ValueError('SAM 3.1 scores/boxes invalid')
        with Image.open(io.BytesIO(files['raw/' + record['rgba_file']])) as opened:
            if opened.size != expected_size:
                raise ValueError('SAM 3.1 cutout dimensions differ')
            rgba = np.asarray(opened.convert('RGBA'))
        if rgba.shape != (*rgb.shape[:2], 4) or not np.array_equal(rgba[:, :, :3], rgb) or not np.array_equal(rgba[:, :, 3], mask):
            raise ValueError('SAM 3.1 cutout pixels differ from source/mask')
    return records


def _normalized_model(job, metadata, directory, coordinates, fit_contract, root):
    info = job['input_binding']
    for source_name, destination in [('raw/object.glb', 'asset.glb')]:
        shutil.copy2(directory / source_name, directory / destination)
    for role in ('image', 'mask'):
        shutil.copy2(_local(info[role + '_path'], root), directory / ('source-rgb.png' if role == 'image' else 'source-mask.png'))
    source_metadata = read(_local(info['model_metadata_path'], root))['source']
    source = copy.deepcopy(source_metadata)
    source.update(image_path='source-rgb.png', mask_path='source-mask.png',
                  selection_manifest_path=str(_local(info['selection_path'], root)),
                  selection_manifest_sha256=info['selection_sha256'])
    model = {'schema_version': '0.1', 'status': 'imported_model_candidate_pending_alignment',
             'object_id': job['track_id'], 'backend': 'Meta SAM 3D Objects', 'provider': 'Meta',
             'sam3d_output': True, 'sam3d_milestone_complete': False, 'learned_inference_executed': True,
             'source': source, 'outputs': {'glb_path': 'asset.glb'},
             'geometry_reconstructed_by_model': True, 'manual_geometry_fallback': False,
             'manual_pose_used': False, 'appearance': metadata['outputs']['appearance'],
             'appearance_mode': metadata['outputs']['appearance_mode'],
             'hidden_surfaces': 'model-inferred, not observed or measured',
             'metric_scale_verified': False, 'camera_pose_estimated': False,
             'coordinates': 'Unchanged official SAM 3D GLB native coordinates; no video pose or shared metric gauge',
             'validation_status': 'pending_independent_source_projection_review',
             'model_provenance': {'model_repository': job['model_repository'], 'code': metadata['code'],
                                  'checkpoint': metadata['checkpoint'],
                                  'adapter_metadata_path': 'raw/metadata.json',
                                  'adapter_metadata_sha256': sha(directory / 'raw/metadata.json')},
             'runtime': metadata['runtime'], 'native_layout_candidate': coordinates,
             'fit_contract': fit_contract,
             'upgrade_lineage': {'job_id': job['job_id'], 'base_sha256': job['base_sha256'],
                                 'original_metadata_path': info['model_metadata_path'],
                                 'original_metadata_sha256': info['model_metadata_sha256'],
                                 'original_asset_path': info['baseline_asset_path'],
                                 'original_asset_sha256': info['baseline_asset_sha256'],
                                 'raw_asset_sha256': sha(directory / 'raw/object.glb'),
                                 'geometry_bytes_modified': False, 'execution_independently_verified': False},
             'limitations': ['Imported adapter execution attestation is contract-checked, not cryptographic proof.',
                             'One source view; hidden shape/appearance are model predictions.',
                             'Native layout conversion is a candidate initialization, not a validated pose.',
                             'No shared component transform, metric scale, contact or seating is established.']}
    model['artifact_files'] = [{'file': name, 'sha256': sha(directory / name), 'bytes': (directory / name).stat().st_size}
                               for name in ('asset.glb', 'source-rgb.png', 'source-mask.png')]
    write(directory / 'metadata.json', model)
    return directory / 'metadata.json'


def _import_result(job_id, zip_path, *, root=ROOT):
    root = Path(root).resolve()
    job_path, job = _job_read(job_id, root)
    files = _zip_files(zip_path)
    if 'result.json' not in files or len(files['result.json']) > 1024 * 1024:
        raise ValueError('Missing or oversized result receipt')
    receipt = _json_bytes(files['result.json'])
    if receipt.get('job_sha256') != sha(job_path):
        raise ValueError('Returned result is not bound to this exact saved job')
    _validate_result_files(job, receipt, files)
    metadata = _bound_metadata(job, files)
    if job['backend'] == 'sam3d_objects':
        coordinates, fit_contract = _validate_sam3d(job, metadata, files, root)
    else:
        instances = _validate_sam31(job, metadata, files)
    result_id = 'import-' + uuid.uuid4().hex
    directory = _directory(root, 'upgrade-results') / result_id
    directory.mkdir()
    for name, raw in files.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    result = {'format': 'interaction-twin-upgrade-import', 'version': '1', 'import_id': result_id,
              'job_id': job_id, 'twin_id': job['twin_id'], 'track_id': job['track_id'],
              'base_sha256': job['base_sha256'], 'backend': job['backend'],
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'validated_candidate_pending_alignment' if job['backend'] == 'sam3d_objects' else 'validated_anchor_comparison_only',
              'fixture': False, 'source': job['input_binding']['source'],
              'result_zip_sha256': sha(zip_path), 'physical_accuracy_verified': False,
              'execution_independently_verified': False,
              'original_assets_unchanged': True, 'source_program_unchanged': True}
    if job['backend'] == 'sam3d_objects':
        model_path = _normalized_model(job, metadata, directory, coordinates, fit_contract, root)
        result['model_metadata_path'] = _relative(model_path, root)
        result['native_layout_path'] = _relative(directory / 'raw/layout-and-pointmap.npz', root)
        result['native_asset_path'] = _relative(directory / 'asset.glb', root)
        result['alignment_eligible'] = fit_contract['alignment_eligible']
        result['alignment_ineligible_reason'] = fit_contract['alignment_ineligible_reason']
        result['alignment_adapter'] = fit_contract.get('alignment_adapter')
    else:
        reference = np.asarray(Image.open(_local(job['input_binding']['mask_path'], root))) > 0
        comparisons = []
        for instance in instances:
            mask = np.asarray(Image.open(directory / 'raw' / instance['mask_file'])) > 0
            comparisons.append({'model_instance_id': instance['model_instance_id'],
                                'mask_path': _relative(directory / 'raw' / instance['mask_file'], root),
                                'mask_sha256': instance['mask_sha256'], 'visible_pixels': int(mask.sum()),
                                'iou_with_saved_sam_mask_not_accuracy': float((mask & reference).sum() / max(1, (mask | reference).sum())),
                                'physical_identity_verified': False})
        comparison = {'scope': 'one source anchor image only', 'instances': comparisons,
                      'track_id_is_reference_only': job['track_id'], 'cross_frame_identity_inherited': False,
                      'old_sam2_audit_reused': False, 'promoted_to_video_segmentation': False,
                      'overlap_is_accuracy': False, 'source': result['source']}
        write(directory / 'anchor-comparison.json', comparison)
        result['comparison_path'] = _relative(directory / 'anchor-comparison.json', root)
        result['instance_count'] = len(comparisons)
    result['files'] = [{'path': p.relative_to(directory).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size}
                       for p in sorted(directory.rglob('*')) if p.is_file()]
    write(directory / 'import.json', result)
    return _import_summary(directory / 'import.json', result, root)


def import_result(job_id, zip_path, *, root=ROOT):
    try:
        return _import_result(job_id, zip_path, root=root)
    except (KeyError, TypeError, IndexError, zipfile.BadZipFile, NotImplementedError) as exc:
        raise ValueError('Malformed or unsupported adapter result: ' + type(exc).__name__) from None


def _import_summary(path, result, root):
    return copy.deepcopy(result) | {'directory': str(path.parent), 'import_path': _relative(path, root)}


def resolve_import(result_id, *, root=ROOT):
    if not IMPORT_ID.fullmatch(str(result_id)):
        raise ValueError('Invalid import ID')
    path = _local(f'data/upgrade-results/{result_id}/import.json', root)
    result = read(path)
    if result.get('format') != 'interaction-twin-upgrade-import' or result.get('import_id') != result_id or result.get('fixture') is not False:
        raise ValueError('Invalid saved import')
    job_path, job = _job_read(result['job_id'], root)
    if any(result.get(k) != job[k] for k in ('twin_id', 'track_id', 'base_sha256', 'backend')):
        raise ValueError('Saved import is bound to a different job')
    for item in result['files']:
        file = _local(path.parent / _safe_name(item['path']), root)
        _checked_file(file, item['sha256'])
        if file.stat().st_size != item['bytes']:
            raise ValueError('Saved imported file size differs')
    # The derivative import.json is not a substitute for the bound returned receipt.
    raw_files = {}
    for p in (path.parent / 'raw').rglob('*'):
        if p.is_symlink():
            raise ValueError('Saved import contains an undeclared symlink')
        if p.is_file():
            safe = _local(p, root)
            raw_files[safe.relative_to(path.parent).as_posix()] = safe.read_bytes()
    raw_files['result.json'] = (path.parent / 'result.json').read_bytes()
    receipt = _json_bytes(raw_files['result.json'])
    if receipt.get('job_sha256') != sha(job_path):
        raise ValueError('Saved result is bound to a different job version')
    _validate_result_files(job, receipt, raw_files)
    metadata = _bound_metadata(job, raw_files)
    if result['backend'] == 'sam3d_objects':
        model = read(_local(result['model_metadata_path'], root))
        version = model.get('fit_contract', {}).get('version')
        _, fit_contract = _validate_sam3d(job, metadata, raw_files, root, fit_version=version)
        expected_source = read(_local(job['input_binding']['model_metadata_path'], root))['source']
        expected_source = copy.deepcopy(expected_source)
        expected_source.update(image_path='source-rgb.png', mask_path='source-mask.png',
                               selection_manifest_path=str(_local(job['input_binding']['selection_path'], root)),
                               selection_manifest_sha256=job['input_binding']['selection_sha256'])
        if (model['upgrade_lineage']['job_id'] != result['job_id'] or model['sam3d_output'] is not True
                or model['backend'] != 'Meta SAM 3D Objects' or model['source'] != expected_source
                or model['appearance'] != metadata['outputs']['appearance']
                or model.get('fit_contract') != fit_contract
                or result.get('alignment_eligible') != fit_contract['alignment_eligible']
                or result.get('alignment_ineligible_reason') != fit_contract['alignment_ineligible_reason']
                or result.get('alignment_adapter') != fit_contract.get('alignment_adapter')):
            raise ValueError('Normalized model provenance differs')
        _checked_file(_local(Path(result['model_metadata_path']).parent / 'asset.glb', root), model['upgrade_lineage']['raw_asset_sha256'])
        # Import-time lineage is immutable. New code may gain or lose support
        # without altering that record; report its current capability explicitly.
        current = _fit_contract(raw_files['raw/object.glb'], metadata['outputs']['appearance_mode'],
                                _validate_glb(raw_files['raw/object.glb']))
        result = copy.deepcopy(result)
        result['import_time_alignment_eligible'] = result.get('alignment_eligible')
        result['import_time_alignment_adapter'] = result.get('alignment_adapter')
        result['adapter_capability'] = current
        result['alignment_eligible'] = current['alignment_eligible']
        result['alignment_ineligible_reason'] = current['alignment_ineligible_reason']
        result['alignment_adapter'] = current.get('alignment_adapter')
    else:
        _validate_sam31(job, metadata, raw_files)
    return _import_summary(path, result, root)


def get_import(result_id, *, root=ROOT):
    return resolve_import(result_id, root=root)


def list_imports(*, root=ROOT):
    base = Path(root) / 'data/upgrade-results'
    if not base.exists():
        return []
    rows = []
    for path in sorted(base.glob('import-*/import.json'), reverse=True):
        try:
            rows.append(resolve_import(path.parent.name, root=root))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            rows.append({'import_id': path.parent.name, 'status': 'invalid_saved_import', 'error': str(exc)})
    return rows
