"""Private, automatic unchanged-native charger refit from fresh SAM masks.

This is a display estimate with matched full-triangle diagnostics. Original
program, scene, 3D files and saved failure history are never overwritten. Unknown
or inadequate masks remain unknown; no physical or identity claim is promoted.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

import charger_tracking_recovery_20261103_v1 as C
import sam3d_common_scene as S
import sam3d_mechanisms as M
from native_appearance import load_native_glb, render_native_mask, render_native_mesh

ROOT = Path(__file__).resolve().parent
BASE_SCENE = 'data/mechanism-scenes/61c175165c5a08a1da6341add682a97baf3e6d7c5b52f9aed3b401a66c57930a/constrained/scene.json'
OUTPUT = 'data/tracking-recoveries/charger-20261103-v1/mask-refit-v1'
SIDE = 288


def quality(mask, box=None):
    """Fail closed on tiny masks, frame leakage or prompt/body mismatch."""
    mask = np.asarray(mask, bool); area = int(mask.sum()); reasons = []
    if area < 256: reasons.append('too_few_visible_adapter_body_pixels')
    if area > mask.size * .25: reasons.append('implausibly_large_single_adapter_mask')
    if area:
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
        if largest < area * .8: reasons.append('fragmented_adapter_mask')
    if box is not None and area:
        left, top, right, bottom = box; rectangle = np.zeros_like(mask)
        rectangle[max(0, math.floor(top)):min(mask.shape[0], math.ceil(bottom)), max(0, math.floor(left)):min(mask.shape[1], math.ceil(right))] = True
        overlap = int(np.count_nonzero(mask & rectangle))
        if overlap / area < .8: reasons.append('mask_leaks_outside_fresh_model_body_box')
        if area / max(1., (right-left)*(bottom-top)) < .12: reasons.append('mask_does_not_cover_localized_adapter_body')
    return reasons


def raw_metrics(mesh, matrix, camera, mask, width, height):
    rotation, translation, scale = M.rotation_pose(matrix)
    factor = min(1., SIDE / max(width, height)); w, h = round(width * factor), round(height * factor)
    k = np.asarray(camera, float).copy(); k[0] *= w / width; k[1] *= h / height
    k[0, 2] += (w / width - 1) / 2; k[1, 2] += (h / height - 1) / 2
    target = cv2.resize(np.asarray(mask, np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    rendered = render_native_mask(mesh, k, rotation, translation, w, h, scale=scale)['mask']
    intersection = np.count_nonzero(target & rendered); union = np.count_nonzero(target | rendered)
    return {'full_triangle_raw_mask_iou': float(intersection / max(1, union)),
        'visible_source_coverage': float(intersection / max(1, target.sum())), 'comparison_width': w, 'comparison_height': h,
        'full_triangle_count': len(mesh.faces), 'geometry_decimated': False, 'target_same_as_fitting_mask': True}


def build(root=ROOT):
    root = Path(root).resolve(); started = time.monotonic(); recovery = C.local(C.OUTPUT, root)
    complete_ref = C.ref(recovery / 'sam-correction-v2/attempt/complete.json', root); complete = C.read(recovery / 'sam-correction-v2/attempt/complete.json')
    for key in ['plan', 'result', 'request']: C.require(C.ref(complete[key]['path'], root) == complete[key], 'Fresh SAM lineage changed')
    request = C.read(C.local(complete['request']['path'], root)); C.require(request.get('status') == 'complete_accounted' and request.get('model') == 'sam-3.1', 'Fresh accounted SAM3.1 required')
    sam_plan = C.read(C.local(complete['plan']['path'], root)); result = C.read(C.local(complete['result']['path'], root))
    sources = sam_plan['frame_sources']; C.require(len(sources) == 37 and [r['sample_index'] for r in sources] == C.TRACK_SAMPLES, 'Exact original source clock subset required')
    detections = {}
    result_dir = C.local(complete['result']['path'], root).parent
    for item in result['instances']:
        frame = item.get('frame_index'); C.require(type(frame) is int and 0 <= frame < len(sources) and item.get('source') == sources[frame] and str(item.get('instance_id')) == '0', 'Exact numeric SAM body/source mapping required')
        index = sources[frame]['sample_index']; C.require(index not in detections, 'Duplicate SAM body/source record')
        path = result_dir / item['mask']; reference = C.ref(path, root); C.require(reference['sha256'] == item['mask_sha256'], 'Fresh returned SAM pixels changed')
        with Image.open(path) as im:
            pixels = np.asarray(im)
        C.require(pixels.shape == (sam_plan['height'], sam_plan['width']) and set(np.unique(pixels)) <= {0, 1, 255}, 'Binary original-coordinate SAM mask required')
        detections[index] = {'mask': pixels > 0, 'reference': reference}
    baseline_ref = C.ref(BASE_SCENE, root); baseline = C.read(C.local(BASE_SCENE, root)); program_ref = C.ref(C.PROGRAM, root)
    C.require(baseline['original_program'] == program_ref and sam_plan['program'] == program_ref, 'Unchanged original program required')
    C.require(not baseline['mechanics']['joints'], 'Recovery cannot alter a connected rigid mechanism')
    scene = copy.deepcopy(baseline); component = next(c for c in scene['components'] if c['id'] == 'adapter_late')
    old_component = next(c for c in baseline['components'] if c['id'] == 'adapter_late')
    mesh = load_native_glb(C.local(component['asset']['path'], root)); C.require(C.ref(component['asset']['path'], root) == component['asset'], 'Original native mesh changed')
    vertices = mesh.vertices[np.linspace(0, len(mesh.vertices)-1, min(2400, len(mesh.vertices))).astype(int)]
    model = S.load_model(C.local(component['native_metadata']['path'], root), scene['source']['sha256'], root)
    initial_rotation = model['layout'].rotation_cv_from_glb; scale = component['native_uniform_scale']
    gauge = component['depth_gauge']['scale']; initial_depth = float(model['layout'].translation_cv[2] * gauge)
    camera = np.asarray(scene['camera']['K']); width, height = scene['source']['width'], scene['source']['height']
    rows = {p['source_sample_index']: p for p in component['poses']}; old_rows = {p['source_sample_index']: p for p in old_component['poses']}
    boxes = {sources[o['frame_index']]['sample_index']: o['box_xyxy'] for o in sam_plan['objects']}
    anchor = component['source_anchor']['sample_index']; comparison = []; inputs = [baseline_ref, program_ref, complete_ref, *[complete[k] for k in ['plan', 'result', 'request']]]
    for sequence in [list(range(anchor, max(C.TRACK_SAMPLES)+1)), list(range(anchor-1, min(C.TRACK_SAMPLES)-1, -1))]:
        rotation, depth, previous_area, previous_time = initial_rotation, initial_depth, None, None
        for index in sequence:
            row = rows[index]; baseline_row = old_rows[index]; entry = detections.get(index)
            diagnostic = {'source_sample_index': index, 'timestamp_s': row['timestamp_s'], 'baseline_status': baseline_row['status'], 'candidate_status': 'unknown'}
            if entry is None:
                diagnostic['reason'] = 'no_returned_fresh_SAM_mask'; comparison.append(diagnostic); continue
            mask = entry['mask']; inputs.append(entry['reference']); reasons = quality(mask, boxes.get(index))
            diagnostic.update(fresh_mask=entry['reference'], mask_area_pixels=int(mask.sum()), mask_gate_reasons=reasons)
            if reasons:
                diagnostic['reason'] = 'fresh_adapter_body_mask_failed_gate'; comparison.append(diagnostic); continue
            area = int(mask.sum())
            if previous_area is not None and max(area/previous_area, previous_area/area) > 3.5 and abs(row['timestamp_s']-previous_time) < .8:
                diagnostic['reason'] = 'abrupt_mask_area_change_possible_occlusion'; comparison.append(diagnostic); continue
            fit = S.estimate_pose(vertices, initial_rotation, scale, camera, mask, depth, previous_rotation=rotation)
            if not .4 * initial_depth <= fit['depth'] <= 2.5 * initial_depth or fit['convex_silhouette_iou'] < .35:
                diagnostic['reason'] = 'strict_native_depth_or_convex_fit_gate_failed'; comparison.append(diagnostic); continue
            candidate = raw_metrics(mesh, fit['matrix'], camera, mask, width, height)
            original = raw_metrics(mesh, baseline_row['camera_from_native_mesh'], camera, mask, width, height) if baseline_row['status'] == 'estimated' else None
            diagnostic.update(candidate_full_native=candidate, baseline_full_native=original, convex_silhouette_fit_iou=fit['convex_silhouette_iou'])
            # Same-mask comparison is conditional fitting, not held-out accuracy.
            # Never hide raw geometric disagreement behind an occluder discount.
            if candidate['full_triangle_raw_mask_iou'] < .35 or candidate['visible_source_coverage'] < .5 or (original is not None and candidate['full_triangle_raw_mask_iou'] < original['full_triangle_raw_mask_iou']-.025):
                diagnostic['reason'] = 'full_native_mask_gate_failed_or_existing_pose_better'; comparison.append(diagnostic); continue
            row.update(status='estimated', camera_from_native_mesh=fit['matrix'].tolist(), source_mask=entry['reference'],
                reason='fresh_Muse_body_localization_and_SAM31_mask_native_display_refit', method='fresh_SAM31_mask_strict_native_prior_temporal_fit',
                convex_silhouette_fit_iou=fit['convex_silhouette_iou'], full_triangle_raw_mask_iou=candidate['full_triangle_raw_mask_iou'],
                native_uniform_scale=scale, mask_area_pixels=area, accepted_motion_evidence=False)
            diagnostic.update(candidate_status='estimated', reason='strict_fresh_mask_native_fit_retained'); comparison.append(diagnostic)
            rotation, depth, previous_area, previous_time = fit['rotation'], fit['depth'], area, row['timestamp_s']
    comparison.sort(key=lambda r: r['source_sample_index'])
    # Existing estimates outside or rejected within the recovery interval remain
    # byte-identical. A failed new hypothesis never clears a valid baseline row.
    scene['fitter_version'] = 'fresh_sam31_adapter_body_refit_v1'
    scene['research_derivation'] = {'kind': 'source_bound_component_mask_tracking_recovery', 'component_id': 'adapter_late',
        'baseline_scene': baseline_ref, 'fresh_SAM_completion': complete_ref, 'identity_gap_merged': False,
        'native_geometry_or_scale_modified': False, 'fitting_evidence': 'same fresh SAM masks, not independent heldout accuracy',
        'full_triangle_gate_min_iou': .35, 'full_triangle_gate_min_visible_source_coverage': .5,
        'maximum_existing_raw_iou_drop': .025, 'physical_accuracy_verified': False}
    scene['inputs'] = list({r['path']: r for r in [*scene['inputs'], *inputs]}.values())
    scene['limitations'] += ['Only the late rigid adapter mask track was resegmented. Thin cable deformation, connector seating, outlet contact, shape and metric scale remain unverified.']
    output = C.local(OUTPUT, root); C.require(not output.exists(), 'Preserve previous refit output'); output.mkdir()
    for name in ['charger_mask_refit_20261103_v1.py', 'charger_tracking_recovery_20261103_v1.py']:
        target = output / (name + '.txt'); target.write_bytes((root/name).read_bytes()); scene['inputs'].append(C.ref(target, root))
    scene['input_signature'] = C.digest({'inputs': scene['inputs'], 'research_derivation': scene['research_derivation']})
    proposal = baseline['mechanics']['proposal']
    scene.pop('mechanics')
    C.write_new(output/'initialized-scene.json', scene)
    # Re-run the unchanged zero-joint mechanics compiler so descriptive exports
    # bind this exact new source scene, never the previous independent estimates.
    fitted = M.build(output/'initialized-scene.json', C.local(proposal['path'], root), output/'constrained', root=root)
    gained = [r['source_sample_index'] for r in comparison if r['baseline_status'] != 'estimated' and r['candidate_status'] == 'estimated']
    retained = [r['source_sample_index'] for r in comparison if r['candidate_status'] == 'estimated']
    report = {'format': 'interaction-twin-fresh-SAM-charger-tracking-refit', 'version': 1, 'baseline': baseline_ref,
        'candidate_scene': fitted['manifest'], 'fresh_SAM_completion': complete_ref, 'comparison': comparison,
        'baseline_estimated_count': sum(r['status'] == 'estimated' for r in old_component['poses']),
        'candidate_estimated_count': sum(r['status'] == 'estimated' for r in component['poses']),
        'newly_supported_source_samples': gained, 'fresh_refit_retained_samples': retained,
        'new_api_calls_this_fit': 0, 'new_gpu_calls': 0, 'manual_geometry_or_poses_used': False,
        'original_scene_or_assets_modified': False, 'physical_accuracy_verified': False, 'automatic_publication': False,
        'same_mask_conditional_fit_not_independent_accuracy': True, 'wall_seconds': time.monotonic()-started}
    C.write_new(output/'report.json', report)
    return {'report': C.ref(output/'report.json', root), 'scene': report['candidate_scene'], 'newly_supported': gained,
        'baseline_estimated': report['baseline_estimated_count'], 'candidate_estimated': report['candidate_estimated_count']}


if __name__ == '__main__':
    print(json.dumps(build()), flush=True)
