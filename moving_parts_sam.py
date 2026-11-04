"""Eight one-attempt SAM image checks, with native per-response identities.

No pixel seeds, crops, manual geometry, pose, temporal identity or motion fit.
The six Jabra prompts are copied verbatim from the saved Meta model advisory;
the two camera prompts are separately authorized generic occluder/target nouns.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import geometry_intelligence as intelligence
import meta_sam_api


MAX_REQUESTS = 8
REVIEW = 'data/geometry-intelligence-20261031/reconstruction-review-v3/review.json'
JABRA = intelligence.SOURCE_PROGRAMS['jabra']
CAMERA = intelligence.SOURCE_PROGRAMS['camera']
CAMERA_PROTOCOL = 'data/observed-geometry-experiment-20261031/camera-body-core-v5/protocol.json'
CAPTION = 'Independent SAM3.1 masks, motion/identity unverified'


def _url(root, path):
    return '/files/' + str(Path(path).relative_to(root))


def build_plan(root):
    root = Path(root).resolve()
    review_path = intelligence.local_path(root, REVIEW)
    review = intelligence.read(review_path)
    context = intelligence.read(review_path.with_name('input.json'))
    intelligence.validate_result(intelligence.read(review_path.with_name('raw-advisory.json')), context)
    if review['status'] != 'model_advisory_not_geometry_truth':
        raise ValueError('A locally validated saved model advisory is required')
    proposals = [p for p in review['proposed_components'] if p['id'] in {'case_base', 'case_lid'}]
    if {p['id'] for p in proposals} != {'case_base', 'case_lid'} or len(proposals) != 2:
        raise ValueError('Bounded scope requires the two authorized model-proposed case parts')
    frames = {f['id']: f for f in review['input_frames']}
    programs = {case: intelligence.read(intelligence.local_path(root, path))
                for case, path in [('jabra', JABRA), ('camera', CAMERA)]}
    source_bindings = {path: intelligence.sha256(intelligence.local_path(root, path))
                       for path in [REVIEW, JABRA, CAMERA, CAMERA_PROTOCOL]}
    for case, program in programs.items():
        relative = JABRA if case == 'jabra' else CAMERA
        if source_bindings[relative] != context['source_bindings'][relative]:
            raise ValueError('Source program changed after model advisory')
        video = intelligence.local_path(root, program['source']['path'])
        if intelligence.sha256(video) != program['source']['sha256']:
            raise ValueError('Source video hash mismatch')
    jobs = []
    required_frames = {'jabra::source_000001', 'jabra::source_000007', 'jabra::source_000023'}
    for proposal in sorted(proposals, key=lambda p: p['id']):
        if proposal['case_id'] != 'jabra' or proposal['parent_track_id'] != 'charging_case':
            raise ValueError('Part proposal case/parent mismatch')
        if set(proposal['evidence_ids']) != required_frames:
            raise ValueError('Expected exact cited source frames for bounded experiment')
        for frame_id in sorted(required_frames):
            frame = frames[frame_id]
            native = next(e for e in programs['jabra']['evidence'] if e['id'] == frame['source_evidence_id'])
            jobs.append({'group': 'proposed_part', 'proposal_id': proposal['id'], 'concept': proposal['sam_prompt'],
                         'prompt_origin': 'muse_spark_model_proposed_component_sam_prompt',
                         'frame': frame, 'sample_index': native['source_index'],
                         'video_sha256': programs['jabra']['source']['sha256'],
                         'relative_output': proposal['id'] + '/' + frame['source_evidence_id']})
    camera_frame = frames['camera::dense_source_000024']
    original_row = next(f for f in intelligence.read(intelligence.local_path(root, CAMERA_PROTOCOL))['input_views']
                        if f['id'] == 'dense_source_000024')
    if original_row['image_sha256'] != camera_frame['sha256'] or original_row['timestamp_s'] != camera_frame['timestamp_s']:
        raise ValueError('Camera exact-frame identity mismatch')
    for concept in ['hand', 'camera']:
        jobs.append({'group': 'camera_occlusion', 'concept': concept,
                     'prompt_origin': 'explicitly_authorized_generic_occluder_and_target_concept',
                     'frame': camera_frame, 'sample_index': original_row['source_index'],
                     'video_sha256': programs['camera']['source']['sha256'],
                     'relative_output': 'camera_occlusion/' + concept + '/dense_source_000024'})
    for job in jobs:
        meta_sam_api._concept(job['concept'])  # The reviewed adapter's local constraint.
        image = intelligence.local_path(root, job['frame']['image_url'])
        if intelligence.sha256(image) != job['frame']['sha256']:
            raise ValueError('Exact cited source image changed')
    if len(jobs) != MAX_REQUESTS:
        raise ValueError('Unexpected bounded experiment request count')
    return {'format': 'moving-parts-sam-image-plan', 'version': 1, 'jobs': jobs, 'proposals': proposals,
            'source_bindings': source_bindings, 'proposal_review_url': '/files/' + REVIEW,
            'maximum_requests': MAX_REQUESTS, 'estimated_total_usd': '0.020000',
            'caption': CAPTION, 'manual_boxes_or_crops': False, 'temporal_identity_fitting': False}


def export_frame(root, job_directory, job, result):
    """Preserve every native mask; add presentation-only source-size overlay."""
    frame = job['frame']
    image_path = intelligence.local_path(root, frame['image_url'])
    if intelligence.sha256(image_path) != frame['sha256']:
        raise ValueError('Image changed before overlay')
    with Image.open(image_path) as opened:
        image = np.array(opened.convert('RGB'))
    if image.shape[:2] != (frame['height'], frame['width']):
        raise ValueError('Mask/source coordinate dimensions differ')
    overlay = image.copy()
    instances = []
    for i, native in enumerate(result['instances']):
        mask_path = job_directory / 'inference' / native['mask']
        if intelligence.sha256(mask_path) != native['mask_sha256']:
            raise ValueError('Native mask hash mismatch')
        with Image.open(mask_path) as opened:
            mask = np.array(opened)
        if mask.shape != image.shape[:2] or not set(np.unique(mask)).issubset({0, 255}):
            raise ValueError('Native mask must remain source-size binary pixels')
        mask = mask > 0
        if int(mask.sum()) != native['mask_foreground_pixels']:
            raise ValueError('Native mask count mismatch')
        colour = np.array([(47 + i * 79) % 210 + 25, (137 + i * 43) % 210 + 25, (173 + i * 61) % 210 + 25])
        overlay[mask] = np.rint(.6 * image[mask] + .4 * colour).astype(np.uint8)
        instances.append({'native_instance_id': native['instance_id'],
                          'identity_scope': 'this_image_response_only', 'physical_identity_verified': False,
                          'mask_url': _url(root, mask_path), 'mask_sha256': native['mask_sha256'],
                          'mask_pixels': native['mask_foreground_pixels'], 'confidence': native['confidence'],
                          'confidence_is_accuracy': False, 'native_box_xyxy_half_open': native['box_xyxy_half_open']})
    presentation = Image.fromarray(overlay)
    draw = ImageDraw.Draw(presentation)
    for i, native in enumerate(result['instances']):
        x, y = native['box_xyxy_half_open'][:2]
        draw.text((x, max(0, y - 12)), 'native ' + str(native['instance_id']), fill='white', stroke_width=1, stroke_fill='black')
    overlay_path = job_directory / 'overlay.png'
    presentation.save(overlay_path)
    return {'id': frame['source_evidence_id'], 'evidence_id': frame['id'], 'timestamp_s': frame['timestamp_s'],
            'image_url': frame['image_url'], 'image_sha256': frame['sha256'],
            'width': frame['width'], 'height': frame['height'], 'overlay_url': _url(root, overlay_path),
            'overlay_sha256': intelligence.sha256(overlay_path), 'overlay_presentation_only': True,
            'status': result['status'], 'instance_count': len(instances), 'instances': instances,
            'segmentation_audit_status': 'not_model_audited', 'provider': 'Meta Model API', 'model': 'sam-3.1',
            'native_metadata_url': _url(root, job_directory / 'inference' / 'metadata.json'),
            'native_metadata_sha256': intelligence.sha256(job_directory / 'inference' / 'metadata.json'),
            'raw_response_url': _url(root, job_directory / 'inference' / 'response.json'),
            'raw_response_sha256': result['raw_response_sha256'],
            'physical_motion_recovered': False, 'cross_frame_identity_established': False}


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root / 'data') or output.exists():
        raise ValueError('Use a fresh output directory inside data')
    meta_sam_api.parser_preflight()
    plan = build_plan(root)
    output.mkdir(parents=True)
    for job in plan['jobs']:
        directory = output / job['relative_output']
        directory.mkdir(parents=True)
        image = intelligence.local_path(root, job['frame']['image_url'])
        manifest = {'schema_version': '0.1', 'object_id': job.get('proposal_id', job['concept']) + '_concept_only',
                    'image': os.path.relpath(image, directory), 'image_sha256': job['frame']['sha256'],
                    'source': {'video_sha256': job['video_sha256'], 'timestamp_s': job['frame']['timestamp_s'],
                               'sample_index': job['sample_index'], 'time_domain': 'source_video_presentation_seconds'}}
        intelligence.write(directory / 'input.json', manifest)
        job['input_manifest_sha256'] = intelligence.sha256(directory / 'input.json')
    intelligence.write(output / 'plan.json', plan)
    (output / 'implementation.py').write_bytes(Path(__file__).read_bytes())
    (output / 'implementation.py').chmod(0o444)
    metadata = {'status': 'prepared_not_sent', 'maximum_requests': MAX_REQUESTS,
                'plan_sha256': intelligence.sha256(output / 'plan.json'),
                'implementation_sha256': intelligence.sha256(output / 'implementation.py'),
                'sam_adapter_sha256': intelligence.sha256(Path(meta_sam_api.__file__)),
                'budget_cap_usd': '100', 'per_request_cap_usd': '2', 'automatic_retries': False}
    intelligence.write(output / 'metadata.json', metadata)
    return metadata


def run(root, output, *, client_factory=None, segmenter=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    metadata = intelligence.read(output / 'metadata.json')
    if metadata['status'] != 'prepared_not_sent' or (output / 'manifest.json').exists():
        raise ValueError('Batch already attempted; no automatic retry')
    for path, expected in [(output / 'plan.json', metadata['plan_sha256']),
                           (output / 'implementation.py', metadata['implementation_sha256']),
                           (Path(__file__), metadata['implementation_sha256']),
                           (Path(meta_sam_api.__file__), metadata['sam_adapter_sha256'])]:
        if intelligence.sha256(path) != expected:
            raise ValueError('Frozen implementation or plan changed')
    plan = intelligence.read(output / 'plan.json')
    for relative, expected in plan['source_bindings'].items():
        if intelligence.sha256(intelligence.local_path(root, relative)) != expected:
            raise ValueError('Bound source artifact changed')
    if not 1 <= len(plan['jobs']) <= MAX_REQUESTS:
        raise ValueError('Request limit exceeded')
    if client_factory is None:
        from meta_api import MetaClient
        client_factory = MetaClient
    segmenter = segmenter or meta_sam_api.segment_image
    client = client_factory(timeout=240)
    manifest = {'format': 'moving-parts-sam-image-review', 'version': 1, 'status': 'running',
                'caption': CAPTION, 'proposal_review_url': plan['proposal_review_url'],
                'proposal_review_sha256': plan['source_bindings'][REVIEW],
                'provider': 'Meta Model API', 'model': 'sam-3.1', 'request_limit': MAX_REQUESTS,
                'proposals': [dict(p, frames=[]) for p in plan['proposals']], 'camera_occlusion_review': [],
                'failures': [], 'requests_attempted': 0,
                'limitations': ['These independent image masks do not measure motion or cross-frame physical identity.',
                                'Every returned native instance is preserved; model confidence is not measured accuracy.',
                                'Masks are unaudited and may include wrong parts, hands, background or multiple objects.',
                                'An empty hand result does not establish absence of occlusion.',
                                'No 3D geometry, joints, axes, poses, physical contact or successful insertion is recovered.']}
    metadata['status'] = 'single_batch_starting'
    intelligence.write(output / 'metadata.json', metadata)
    intelligence.write(output / 'manifest.json', manifest)
    halted = False
    for job in plan['jobs']:
        if client.ledger.status()['blocked']:
            halted = True
            break
        directory = output / job['relative_output']
        if intelligence.sha256(directory / 'input.json') != job['input_manifest_sha256']:
            raise ValueError('Prepared input manifest changed')
        manifest['requests_attempted'] += 1
        intelligence.write(output / 'manifest.json', manifest)
        try:
            result = segmenter(directory / 'input.json', job['concept'], directory / 'inference',
                               client.sam_callback(directory / 'accounting', images=1),
                               prompt_origin=job['prompt_origin'])
            frame = export_frame(root, directory, job, result)
            receipt_path = directory / 'accounting' / 'meta-request.json'
            receipt = intelligence.read(receipt_path)
            frame.update(receipt_url=_url(root, receipt_path), receipt_sha256=intelligence.sha256(receipt_path),
                         estimated_cost_usd=receipt.get('estimated_cost_usd'), request_id=receipt['request_id'])
            if job['group'] == 'proposed_part':
                next(p for p in manifest['proposals'] if p['id'] == job['proposal_id'])['frames'].append(frame)
            else:
                manifest['camera_occlusion_review'].append({'concept': job['concept'], 'frame': frame})
        except Exception as exc:
            failure = {'concept': job['concept'], 'proposal_id': job.get('proposal_id'),
                       'frame_id': job['frame']['id'], 'failure_type': type(exc).__name__,
                       'automatic_retry': False, 'relative_output': job['relative_output']}
            receipt_path = directory / 'accounting' / 'meta-request.json'
            if receipt_path.exists():
                receipt = intelligence.read(receipt_path)
                failure.update(receipt_url=_url(root, receipt_path), status=receipt['status'],
                               estimated_cost_usd=receipt.get('estimated_cost_usd'))
            manifest['failures'].append(failure)
            if client.ledger.status()['blocked']:
                halted = True
                break
        finally:
            intelligence.write(output / 'manifest.json', manifest)
    manifest['status'] = 'halted_budget_or_uncertain_usage' if halted else 'complete_with_failures' if manifest['failures'] else 'complete_independent_image_masks'
    manifest['budget_at_completion'] = client.ledger.status()
    metadata['status'] = manifest['status']
    metadata['manifest_sha256'] = None
    intelligence.write(output / 'manifest.json', manifest)
    metadata['manifest_sha256'] = intelligence.sha256(output / 'manifest.json')
    intelligence.write(output / 'metadata.json', metadata)
    return manifest
