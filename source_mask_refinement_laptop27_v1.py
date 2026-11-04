"""One source-image SAM refresh from a completed automatic Muse observation.

No geometry/pose, boxes or masks are hand authored. The previous source27 masks
stay intact. A real-adapter offline payload preflight precedes the one explicit
paid request, which shares all existing budget envelopes and cannot auto-retry.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

import meta_sam_api as S
import source_observation_refinement_v3 as R

ROOT = Path(__file__).resolve().parent
OUTPUT = 'data/source-mask-refinements/laptop-opening-27-20261103-v1'
FORMAT = 'interaction-twin-model-source-mask-refresh-v1'
SAMPLE = 27
ORIGIN = 'muse_spark_model_generated_visible_part_boxes_source_observation_v3'
IDS = {'laptop_base': 0, 'laptop_lid': 1}


def source_review(*, root=ROOT):
    root = Path(root).resolve()
    plan, _, folder = R.verified_plan(root=root)
    result_path = R.local(root, R.DEFAULT_OUTPUT + '/attempt/model-result.json')
    receipt_path = R.local(root, R.DEFAULT_OUTPUT + '/attempt/request/meta-request.json')
    result, receipt = R.read(result_path), R.read(receipt_path)
    R.validate_result(result, plan)
    R.require(receipt.get('status') == 'complete_accounted'
              and receipt.get('payload_sha256') == plan['wire_provenance']['payload_sha256']
              and receipt.get('image_inputs') == plan['wire_provenance']['image_inputs'], 'Completed exact Muse source observation required')
    frame = next(row for row in result['frames'] if row['sample_index'] == SAMPLE)
    image = next(row for row in plan['context']['images'] if row['sample_index'] == SAMPLE)
    R.require(frame['laptop_state'] == 'partially_open' and frame['visible_laptop_panel'] == 'both', 'Source27 must show both opening panels')
    objects = []
    for tid, identifier in IDS.items():
        part = next(row for row in frame['parts'] if row['track_id'] == tid)
        R.require(part['visibility'] in {'visible', 'partly_visible'} and part['confidence'] >= .7,
                  'Both automatic visible-part boxes need sufficient model confidence')
        box = part['observed_bbox_xyxy_normalized']
        R.require(len(box) == 4, 'Observed model box required')
        objects.append({'id': identifier, 'box_xyxy': [box[0] * image['width'], box[1] * image['height'],
                                                     box[2] * image['width'], box[3] * image['height']]})
    source = {'video_sha256': plan['context']['source_sha256'], 'sample_index': SAMPLE,
              'timestamp_s': image['timestamp_s'], 'time_domain': 'source_video_presentation_seconds'}
    return {'plan': R.ref(root, folder / 'plan.json'), 'model_result': R.ref(root, result_path),
            'accounting': R.ref(root, receipt_path), 'source_image': image, 'source': source,
            'frame_observation': frame, 'objects': objects}


def manifest_for(root, folder, review):
    image = R.local(root, review['source_image']['path'])
    return {'object_id': 'laptop_panels_source_000027',
            'image': os.path.relpath(image, folder), 'image_sha256': review['source_image']['sha256'],
            'source': review['source'], 'mask': None,
            'automatic_prompt_provenance': {'origin': ORIGIN, 'model_result': review['model_result'],
                                             'manual_seed_boxes_used': False}}


class _PayloadCaptured(RuntimeError):
    pass


def adapter_preflight(input_manifest, objects, *, root=ROOT):
    """Exercise the actual adapter through its callback, without any transport."""
    captured = []
    def offline(payload):
        captured.append(payload)
        raise _PayloadCaptured()
    with tempfile.TemporaryDirectory(prefix='laptop27-adapter-preflight-') as name:
        try:
            S.segment_image_boxes(input_manifest, objects, Path(name) / 'result', offline, prompt_origin=ORIGIN)
        except _PayloadCaptured:
            pass
    R.require(len(captured) == 1, 'Exactly one real-adapter offline callback required')
    payload = captured[0]
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    R.require(payload['model'] == 'sam-3.1' and payload['stream'] is False, 'Exact bounded SAM payload required')
    _, normalized = S._model_box_prompt(objects, width=722, height=960, frame_count=1, prompt_origin=ORIGIN, media='image')
    return {'status': 'real_adapter_payload_captured_without_network', 'new_model_requests': 0,
            'sam_adapter': R.ref(root, Path(S.__file__)), 'parser': S.parser_preflight(),
            'payload_sha256': hashlib.sha256(raw).hexdigest(), 'payload_bytes': len(raw),
            'normalized_box_prompts': normalized, 'prompt_origin': ORIGIN,
            'manual_seed_boxes_used': False, 'reference_masks_sent': False}


def build_plan(*, root=ROOT, output=OUTPUT):
    root = Path(root).resolve()
    folder = R.local(root, output, exists=False)
    review = source_review(root=root)
    input_path = folder / 'input.json'
    expected_manifest = manifest_for(root, folder, review)
    R.require(R.read(input_path) == expected_manifest, 'Exact portable source input changed')
    preflight = adapter_preflight(input_path, review['objects'], root=root)
    return {'format': FORMAT, 'version': 1, 'implementation': R.ref(root, Path(__file__)),
            'source_review': review, 'input_manifest': R.ref(root, input_path),
            'object_registry': [{'id': identifier, 'track_id': tid} for tid, identifier in IDS.items()],
            'adapter_preflight': preflight, 'maximum_new_requests': 1, 'automatic_retries': False,
            'maximum_estimated_reservation_usd': '0.002500', 'model': 'sam-3.1', 'media_scope': 'one_image_two_model_boxes',
            'new_gpu_requests': 0, 'automatic_publication': False, 'manual_geometry_or_pose_used': False,
            'limitations': ['Independent source image masks; no cross-frame identity, metric geometry or physical joint accuracy is established.',
                            'These fresh masks may still include hands or the wrong panel; preserve all results and apply source/fitting gates.']}


def prepare(*, root=ROOT, output=OUTPUT):
    root = Path(root).resolve()
    folder = R.local(root, output, exists=False)
    R.require(not folder.exists(), 'Preserve prior source-mask attempt')
    review = source_review(root=root)
    folder.mkdir(parents=True)
    R.write_new(folder / 'input.json', manifest_for(root, folder, review))
    doc = build_plan(root=root, output=output)
    R.write_new(folder / 'plan.json', doc)
    return R.ref(root, folder / 'plan.json')


def verified_plan(*, root=ROOT, output=OUTPUT):
    root = Path(root).resolve()
    folder = R.local(root, output + '/plan.json').parent
    doc = R.read(folder / 'plan.json')
    R.require(doc == build_plan(root=root, output=output), 'Prepared mask refresh source/code/payload changed')
    return doc, folder


def run(*, authorized_plan_sha256, allow_paid_api=False, root=ROOT, output=OUTPUT, client=None):
    root = Path(root).resolve()
    plan, folder = verified_plan(root=root, output=output)
    R.require(allow_paid_api and R.sha(folder / 'plan.json') == authorized_plan_sha256, 'Exact reviewed SAM packet authorization required')
    attempt = folder / 'attempt'
    R.require(not attempt.exists(), 'One mask-refresh attempt only; no automatic retry')
    attempt.mkdir()
    R.write_new(attempt / 'intent.json', {'plan': R.ref(root, folder / 'plan.json'), 'maximum_new_requests': 1, 'automatic_retries': False})
    if client is None:
        from overnight_budget import OvernightMetaClient
        from mechanism_pass_budget import scoped_client_class
        client = scoped_client_class(OvernightMetaClient, root=root)(timeout=300)
    try:
        callback = client.sam_callback(attempt / 'accounting', images=1)
        def exact_payload(payload):
            R.require(R.digest(payload) == plan['adapter_preflight']['payload_sha256'], 'Paid payload differs from offline adapter preflight')
            return callback(payload)
        result = S.segment_image_boxes(folder / 'input.json', plan['source_review']['objects'], attempt / 'segmentation',
                                       exact_payload, prompt_origin=ORIGIN)
        receipt = R.read(attempt / 'accounting/meta-request.json')
        R.require(receipt.get('status') == 'complete_accounted'
                  and receipt.get('payload_sha256') == plan['adapter_preflight']['payload_sha256'], 'Exact accounted SAM result required')
        instances = []
        for item in result['instances']:
            path = attempt / 'segmentation' / item['mask']
            with Image.open(path) as image:
                pixels = np.asarray(image)
                R.require(image.size == (722, 960) and pixels.ndim == 2, 'Original source mask dimensions required')
                R.require(set(np.unique(pixels).tolist()).issubset({0, 1, 255}), 'Binary SAM result required')
                bounds = image.getbbox()
            track = next((tid for tid, identifier in IDS.items() if str(identifier) == str(item['instance_id'])), None)
            R.require(track is not None and item['source'] == plan['source_review']['source'], 'Unknown model/source association')
            instances.append({'track_id': track, 'source': item['source'], 'native_instance_id': item['instance_id'],
                              'mask': R.ref(root, path), 'area_pixels': int(np.count_nonzero(pixels)),
                              'bbox_xyxy': list(bounds) if bounds else None, 'confidence': item.get('confidence'),
                              'status': 'candidate_mask_requires_fresh_scope_and_pose_gates'})
        summary = {'format': FORMAT, 'status': 'complete_accounted_candidate_masks',
                   'plan': R.ref(root, folder / 'plan.json'), 'instances': instances,
                   'native_result': R.ref(root, attempt / 'segmentation/metadata.json'),
                   'receipt': R.ref(root, attempt / 'accounting/meta-request.json'),
                   'new_api_calls': 1, 'estimated_cost_usd': receipt['estimated_cost_usd'],
                   'automatic_publication': False, 'geometry_or_pose_fit_performed': False}
        R.write_new(attempt / 'summary.json', summary)
        return summary
    except BaseException as error:
        R.write_new(attempt / 'failure.json', {'status': 'preserved_no_automatic_retry', 'error_type': type(error).__name__})
        raise
