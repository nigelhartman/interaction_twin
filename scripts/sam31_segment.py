#!/usr/bin/env python3
"""Fail-closed local SAM 3.1 text segmentation of one timestamped source image.

Uses the official multiplex predictor's supported single-image resource path,
not the SAM 3-only image builder. Preflight imports no upstream SAM code and
performs no download, authentication, credential lookup, or learned inference.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np
from PIL import Image

OFFICIAL_CODE = 'https://github.com/facebookresearch/sam3'
MODEL_REPOSITORY = 'facebook/sam3.1'
PINNED_CODE_COMMIT = 'f1647e8d0c1d280991f4d1b7d974c2704537fe66'
TIME_DOMAIN = 'source_video_presentation_seconds'
HEX64 = re.compile(r'[0-9a-f]{64}\Z')
HEX40 = re.compile(r'[0-9a-f]{40}\Z')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_input(path):
    raw = path.read_bytes()
    manifest = json.loads(raw)
    source = manifest.get('source', {})
    if not isinstance(manifest.get('object_id'), str) or not manifest['object_id'].strip():
        raise ValueError('object_id is required; it is a concept label, not proved physical identity')
    if not HEX64.fullmatch(source.get('video_sha256', '')):
        raise ValueError('A source video SHA256 is required')
    pts = source.get('timestamp_s')
    if isinstance(pts, bool) or not isinstance(pts, (int, float)) or not math.isfinite(pts) or pts < 0:
        raise ValueError('timestamp_s must preserve the finite nonnegative original source PTS')
    if source.get('time_domain') != TIME_DOMAIN:
        raise ValueError('time_domain must be source_video_presentation_seconds')
    if isinstance(source.get('sample_index'), bool) or not isinstance(source.get('sample_index'), int) or source['sample_index'] < 1:
        raise ValueError('sample_index must be the positive source sampling index')
    image_path = (path.parent / manifest['image']).resolve()
    if image_path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}:
        raise ValueError('Image extension must match an official supported single-image resource')
    if not HEX64.fullmatch(manifest.get('image_sha256', '')) or sha256(image_path) != manifest['image_sha256']:
        raise ValueError('Image SHA256 mismatch or missing expected hash')
    with Image.open(image_path) as opened:
        if opened.getexif().get(274, 1) not in (None, 1):
            raise ValueError('Normalize image orientation first; EXIF rotation must not silently change source coordinates')
        image = np.asarray(opened.convert('RGB'))
    files = {'image': {'path': str(image_path), 'sha256': sha256(image_path)}}
    mask = None
    if manifest.get('mask') is not None:
        mask_path = (path.parent / manifest['mask']).resolve()
        if not HEX64.fullmatch(manifest.get('mask_sha256', '')) or sha256(mask_path) != manifest['mask_sha256']:
            raise ValueError('Reference mask SHA256 mismatch or missing expected hash')
        mask = np.asarray(Image.open(mask_path))
        if mask.ndim != 2 or mask.shape != image.shape[:2] or not set(np.unique(mask).tolist()).issubset({0, 1, 255, False, True}):
            raise ValueError('Reference mask must be binary and share unchanged full-frame RGB coordinates')
        if not np.any(mask) or np.all(mask):
            raise ValueError('Reference mask must distinguish visible object pixels from background')
        provenance = manifest.get('mask_provenance', {})
        if not isinstance(provenance.get('prompt_origin'), str) or not provenance['prompt_origin'].strip():
            raise ValueError('Reference mask prompt_origin provenance is required')
        manual = provenance.get('manual_intervention', provenance.get('manual_seed_boxes_used'))
        if not isinstance(manual, bool):
            raise ValueError('Reference mask manual_intervention or manual_seed_boxes_used must be explicitly boolean')
        files['reference_mask'] = {'path': str(mask_path), 'sha256': sha256(mask_path)}
        mask = mask > 0
    return {'manifest': manifest, 'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'files': files,
            'width': image.shape[1], 'height': image.shape[0],
            'reference_mask_passed_to_model': False,
            'pixel_policy': 'original full-frame RGB; upstream 1008-square internal resize; returned masks must restore original dimensions'}, image, mask


def inspect_code(repo, expected_commit):
    errors = []
    record = {'path': str(repo), 'expected_commit': expected_commit, 'official_repository': OFFICIAL_CODE}
    def git(*args):
        return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()
    try:
        commit, origin = git('rev-parse', 'HEAD'), git('remote', 'get-url', 'origin')
        record['commit'] = commit
        official = origin.rstrip('/').removesuffix('.git') == OFFICIAL_CODE
        record['origin'] = OFFICIAL_CODE if official else 'non-official origin (redacted)'
        if not official:
            errors.append('Code origin must be the public official HTTPS SAM 3 repository')
        if not HEX40.fullmatch(expected_commit or '') or commit != expected_commit:
            errors.append('Official code HEAD must match the supplied full reviewed commit SHA')
        if git('status', '--porcelain', '--untracked-files=normal'):
            errors.append('Official checkout must be clean, including nonignored untracked files')
    except (OSError, subprocess.CalledProcessError):
        errors.append('No inspectable pinned official SAM 3 Git checkout is available')
    required = ['sam3/model_builder.py', 'sam3/model/sam3_base_predictor.py',
                'sam3/model/sam3_multiplex_video_predictor.py', 'sam3/model/io_utils.py',
                'sam3/assets/bpe_simple_vocab_16e6.txt.gz']
    record['files'] = []
    for name in required:
        path = repo / name
        if path.is_file():
            record['files'].append({'file': name, 'sha256': sha256(path)})
        else:
            errors.append(f'Missing official API/tokenizer file: {name}')
    return record, errors


def inspect_checkpoint(path):
    if path is None or not path.is_file():
        return {}, ['Provision an approved local SAM 3.1 checkpoint and its provenance manifest; no download is attempted']
    manifest = json.loads(path.read_text())
    errors = []
    if manifest.get('model_repository') != MODEL_REPOSITORY:
        errors.append('Checkpoint provenance must identify facebook/sam3.1, not SAM 3 or SAM 2')
    if not HEX40.fullmatch(manifest.get('model_revision', '')):
        errors.append('Checkpoint model_revision must be the resolved full Hugging Face commit SHA')
    if manifest.get('checkpoint_file') != 'sam3.1_multiplex.pt':
        errors.append('This reviewed adapter supports the official sam3.1_multiplex.pt checkpoint')
    checkpoint = (path.parent / manifest.get('checkpoint_file', '')).resolve()
    expected = manifest.get('checkpoint_sha256', '')
    record = {'manifest': manifest, 'manifest_sha256': sha256(path), 'path': str(checkpoint),
              'origin_status': 'provided download provenance; no online account or repository lookup performed'}
    if not checkpoint.is_file():
        errors.append('Local SAM 3.1 checkpoint file is missing')
    else:
        actual = sha256(checkpoint)
        record.update(sha256=actual, bytes=checkpoint.stat().st_size)
        if not HEX64.fullmatch(expected) or actual != expected:
            errors.append('Local SAM 3.1 checkpoint SHA256 differs from download provenance')
    return record, errors


def version_tuple(text):
    match = re.match(r'^(\d+)\.(\d+)', str(text))
    return tuple(map(int, match.groups())) if match else (0, 0)


def inspect_runtime():
    errors = []
    record = {'platform': platform.system(), 'python': platform.python_version(), 'device_requested': 'cuda',
              'supported_adapter_target': 'Linux CUDA; no MPS/CPU fallback', 'package_checks_are_not_a_model_smoke_test': True}
    if platform.system() != 'Linux':
        errors.append('This adapter targets the Linux CUDA cluster; no Mac MPS or CPU fallback is implemented')
    if sys.version_info < (3, 12):
        errors.append('Official README prerequisites require Python >=3.12')
    if not (1, 26) <= version_tuple(np.__version__) < (2, 0):
        errors.append('Official package declares numpy>=1.26,<2; use a separate compatible environment')
    record['numpy'] = np.__version__
    names = ['torch', 'torchvision', 'timm', 'ftfy', 'regex', 'iopath', 'huggingface_hub', 'einops', 'pkg_resources', 'PIL']
    record['dependency_presence'] = {name: importlib.util.find_spec(name) is not None for name in names}
    missing = [name for name, exists in record['dependency_presence'].items() if not exists]
    if missing:
        errors.append('Missing runtime dependencies: ' + ', '.join(missing))
    try:
        import torch
        record.update(torch=torch.__version__, torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
        if version_tuple(torch.__version__) < (2, 7):
            errors.append('Official prerequisites require PyTorch >=2.7')
        if version_tuple(torch.version.cuda) < (12, 6):
            errors.append('Official prerequisites require CUDA >=12.6')
        if not torch.cuda.is_available():
            errors.append('No CUDA GPU is visible; run inside an allocated GPU session')
        else:
            gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
            record.update(gpu_name=gpu.name, gpu_memory_gib=gpu.total_memory / 2**30,
                          bfloat16_supported=torch.cuda.is_bf16_supported())
            if not record['bfloat16_supported']:
                errors.append('The reviewed official multiplex predictor uses CUDA bfloat16 autocast')
    except ImportError:
        errors.append('PyTorch is unavailable')
    return record, errors


def preflight(args):
    errors = []
    image, mask = None, None
    try:
        inputs, image, mask = read_input(args.input_manifest)
    except (ValueError, TypeError, OSError, KeyError) as exc:
        inputs = {'manifest_path': str(args.input_manifest)}
        errors.append(f'Invalid timestamped image input: {exc}')
    for value, name in [(args.prompt, 'prompt'), (args.prompt_origin, 'prompt_origin')]:
        if not isinstance(value, str) or not value.strip():
            errors.append(f'{name} must be a nonempty explicit string')
    code, code_errors = inspect_code(args.repo.resolve(), args.expected_code_commit)
    try:
        checkpoint, checkpoint_errors = inspect_checkpoint(args.checkpoint_manifest)
    except (OSError, ValueError, TypeError) as exc:
        checkpoint, checkpoint_errors = {}, [f'Invalid local checkpoint provenance: {exc}']
    runtime, runtime_errors = inspect_runtime()
    errors.extend(code_errors + checkpoint_errors + runtime_errors)
    return {'schema_version': '0.1', 'status': 'prerequisites_missing' if errors else 'ready_for_first_model_smoke_test',
            'backend': 'Meta SAM 3.1 Object Multiplex', 'model_repository': MODEL_REPOSITORY,
            'learned_inference_executed': False, 'scope': 'one source image as an official single-frame video session',
            'prompt': {'text': args.prompt, 'origin': args.prompt_origin, 'reference_mask_used_as_prompt': False},
            'inputs': inputs, 'code': code, 'checkpoint': checkpoint, 'runtime': runtime,
            'network_or_credential_access_attempted': False, 'errors': errors}, image, mask


def verify_checkpoint_coverage(model_state, checkpoint_state):
    """Upstream loads non-strictly; reject partial/random-weight models here."""
    if not isinstance(checkpoint_state, dict) or not model_state or not checkpoint_state:
        raise RuntimeError('Checkpoint/model state must be nonempty; no inference or fallback allowed')
    if 'model' in checkpoint_state and isinstance(checkpoint_state['model'], dict):
        checkpoint_state = checkpoint_state['model']
    if not isinstance(checkpoint_state, dict):
        raise RuntimeError('Checkpoint is not a state dictionary; no inference or fallback allowed')
    missing = set(model_state) - set(checkpoint_state)
    unexpected = set(checkpoint_state) - set(model_state)
    wrong_shape = [key for key in set(model_state) & set(checkpoint_state)
                   if not hasattr(checkpoint_state[key], 'shape') or tuple(model_state[key].shape) != tuple(checkpoint_state[key].shape)]
    if missing or unexpected or wrong_shape:
        raise RuntimeError(f'Checkpoint coverage rejected: {len(missing)} missing, {len(unexpected)} unexpected, {len(wrong_shape)} shape mismatches; no inference or fallback allowed')
    return {'strict_key_and_shape_coverage': True, 'tensor_count': len(model_state),
            'legacy_checkpoint_key_remapping_supported': False}


def export_prediction(output, image, out, reference_mask=None):
    required = ['out_obj_ids', 'out_probs', 'out_boxes_xywh', 'out_binary_masks']
    if not isinstance(output, dict) or any(key not in output for key in required):
        raise RuntimeError('Missing official model outputs; no synthetic mask or fallback is allowed')
    arrays = {}
    for key in required:
        value = output[key]
        arrays[key] = np.asarray(value.detach().cpu().numpy() if hasattr(value, 'detach') else value)
    masks, ids = arrays['out_binary_masks'], arrays['out_obj_ids']
    count = len(ids)
    if masks.dtype != bool or masks.shape != (count, *image.shape[:2]):
        raise RuntimeError('Official masks must be boolean NxHxW in original full-frame coordinates')
    if ids.shape != (count,) or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != count:
        raise RuntimeError('Official model instance IDs must be unique integers')
    scores, boxes = arrays['out_probs'], arrays['out_boxes_xywh']
    if scores.shape != (count,) or boxes.shape != (count, 4) or not np.isfinite(scores).all() or not np.isfinite(boxes).all():
        raise RuntimeError('Official scores/boxes have invalid dimensions or nonfinite values')
    if np.any((scores < 0) | (scores > 1)):
        raise RuntimeError('Official probability scores are outside [0,1]')
    if np.any(boxes < -1e-4) or np.any(boxes > 1 + 1e-4) or np.any(boxes[:, :2] + boxes[:, 2:] > 1 + 1e-4):
        raise RuntimeError('Official normalized xywh boxes exceed source coordinates')
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out / 'source-rgb.png')
    overlay = image.copy()
    records = []
    for index, (instance_id, score, box, mask) in enumerate(zip(ids, scores, boxes, masks)):
        path = out / f'instance-{index + 1:06d}-mask.png'
        Image.fromarray((mask * 255).astype(np.uint8)).save(path)
        rgba = np.dstack([image, (mask * 255).astype(np.uint8)])
        rgba_path = out / f'instance-{index + 1:06d}-rgba.png'
        Image.fromarray(rgba).save(rgba_path)
        color = np.array([(67 + index * 73) % 200 + 30, (149 + index * 37) % 200 + 30, (201 + index * 53) % 200 + 30])
        overlay[mask] = np.rint(.6 * image[mask] + .4 * color).astype(np.uint8)
        record = {'model_instance_id': int(instance_id), 'physical_identity_verified': False,
                  'score': float(score), 'score_is_calibrated_accuracy': False,
                  'box_xywh_normalized': box.tolist(), 'mask_pixels': int(mask.sum()),
                  'mask_file': path.name, 'mask_sha256': sha256(path), 'rgba_file': rgba_path.name}
        if reference_mask is not None:
            record['iou_with_earlier_reference_mask_not_accuracy'] = float((mask & reference_mask).sum() / max(1, (mask | reference_mask).sum()))
        records.append(record)
    Image.fromarray(overlay).save(out / 'overlay.png')
    return {'instances': records, 'instance_count': count, 'no_instance_result_is_legitimate': True,
            'reference_mask_passed_to_model': False, 'handcrafted_mask_fallback': False}


def run(args, report, image, reference_mask):
    if report['errors']:
        raise RuntimeError('Preflight failed; refusing model construction or fallback')
    if args.output.exists():
        raise ValueError('Choose a fresh output directory to preserve prior evidence')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(args.repo.resolve()))
    import torch
    import sam3
    import sam3.model_builder as builder
    if Path(sam3.__file__).resolve() != args.repo.resolve() / 'sam3/__init__.py':
        raise RuntimeError('Imported SAM code differs from the reviewed checkout')
    def deny_download(*unused_args, **unused_kwargs):
        raise RuntimeError('Checkpoint auto-download is disabled in this offline local adapter')
    builder.download_ckpt_from_hf = deny_download
    builder.hf_hub_download = deny_download
    predictor = builder.build_sam3_predictor(checkpoint_path=report['checkpoint']['path'],
        bpe_path=str(args.repo.resolve() / 'sam3/assets/bpe_simple_vocab_16e6.txt.gz'),
        version='sam3.1', compile=False, warm_up=False, use_fa3=False,
        use_rope_real=True, async_loading_frames=False)
    loaded = torch.load(report['checkpoint']['path'], map_location='cpu', weights_only=True)
    coverage = verify_checkpoint_coverage(predictor.model.state_dict(), loaded)
    del loaded
    session_id = None
    started = time.perf_counter()
    try:
        session = predictor.handle_request({'type': 'start_session', 'resource_path': report['inputs']['files']['image']['path']})
        session_id = session['session_id']
        response = predictor.handle_request({'type': 'add_prompt', 'session_id': session_id,
            'frame_index': 0, 'text': args.prompt, 'output_prob_thresh': args.probability_threshold})
        if response.get('frame_index') != 0:
            raise RuntimeError('Official response refers to the wrong single-frame session index')
        outputs = export_prediction(response.get('outputs'), image, args.output, reference_mask)
    finally:
        if session_id is not None:
            predictor.handle_request({'type': 'close_session', 'session_id': session_id})
    report.update(status='completed_candidates' if outputs['instance_count'] else 'completed_no_instances',
                  learned_inference_executed=True, outputs=outputs, checkpoint_coverage=coverage,
                  inference_session_seconds=time.perf_counter() - started,
                  probability_threshold=args.probability_threshold,
                  model_inference_scope='one image; no temporal propagation or cross-frame identity inferred',
                  reference_mask_provenance_does_not_describe_new_text_prompt=True)
    (args.output / 'metadata.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'run'])
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--expected-code-commit', default=PINNED_CODE_COMMIT)
    parser.add_argument('--checkpoint-manifest', type=Path)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--prompt-origin', required=True)
    parser.add_argument('--probability-threshold', type=float, default=.5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.probability_threshold) or not 0 < args.probability_threshold < 1:
        parser.error('probability-threshold must be finite and inside (0,1)')
    if args.output.exists():
        parser.error('Output already exists; choose a fresh path')
    report, image, reference_mask = preflight(args)
    if args.command == 'preflight':
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    elif report['errors']:
        print(json.dumps({'status': report['status'], 'errors': report['errors']}))
        return 2
    else:
        report = run(args, report, image, reference_mask)
    print(json.dumps({'status': report['status'], 'learned_inference_executed': report['learned_inference_executed'],
                      'errors': report['errors'], 'output': str(args.output)}))
    return 2 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
