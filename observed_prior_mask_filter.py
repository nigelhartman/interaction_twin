"""Derive a conservative model-prior layer from exact-frame native SAM masks.

No inference occurs here. Native detections and the original geometry remain
unchanged. Filtering is a model-based hypothesis, never proof of clean shape.
"""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path

import cv2
import numpy as np

import observed_surface_completion as C


PROTOCOL = {
    'version': 'native-camera-hand-model-prior-filter-v1',
    'camera_selection': 'unique maximum native-instance IoU against original exact-frame target mask',
    'minimum_camera_iou': 0.30, 'minimum_runner_up_iou_margin': 0.05,
    'camera_mask_erosion_radius_px': 1, 'hand_union_dilation_radius_px': 2,
    'hand_selection': 'union of every returned native hand instance; no manual selection',
    'require_at_least_one_native_hand_instance': True,
    'maximum_edge_sample_spacing_px': 0.5,
    'sample_membership': 'all floor/ceil neighbouring pixels inside refined domain',
    'triangle_checks': ['vertices', 'dense_edges', 'centroid', 'rasterized_interior'],
    'original_geometry_changed': False, 'new_vertices_or_faces_created': False,
    'physical_confidence': 'unknown', 'clean_segmentation_established': False,
    'new_api_calls': 0, 'new_model_inference_calls': 0,
}


def choose_camera(instances, original):
    original = np.asarray(original) > 0
    candidates = []
    for instance_id, mask in instances:
        mask = np.asarray(mask) > 0
        if mask.shape != original.shape:
            raise ValueError('Native camera masks must match exact source dimensions')
        union = int((mask | original).sum())
        iou = float((mask & original).sum()/union) if union else 0.
        candidates.append({'native_instance_id': str(instance_id), 'iou_with_original_target': iou,
                           'native_area_px': int(mask.sum())})
    candidates.sort(key=lambda row: (-row['iou_with_original_target'], row['native_instance_id']))
    if not candidates or candidates[0]['iou_with_original_target'] < PROTOCOL['minimum_camera_iou']:
        return None, {'status': 'no_eligible_native_camera', 'candidates': candidates}
    margin = candidates[0]['iou_with_original_target'] - (candidates[1]['iou_with_original_target'] if len(candidates) > 1 else 0.)
    if len(candidates) > 1 and margin < PROTOCOL['minimum_runner_up_iou_margin']:
        return None, {'status': 'ambiguous_native_camera_instances', 'candidates': candidates, 'winner_margin': margin}
    return candidates[0]['native_instance_id'], {'status': 'unique_native_camera_selected', 'candidates': candidates, 'winner_margin': margin}


def refined_domain(camera_mask, hand_masks):
    camera = np.asarray(camera_mask) > 0
    if camera.ndim != 2:
        raise ValueError('A two-dimensional native camera mask is required')
    union = np.zeros(camera.shape, np.uint8)
    for hand in hand_masks:
        hand = np.asarray(hand) > 0
        if hand.shape != camera.shape:
            raise ValueError('All native hands must match the exact source dimensions')
        union |= hand.astype(np.uint8)
    erosion = PROTOCOL['camera_mask_erosion_radius_px']
    guard = PROTOCOL['hand_union_dilation_radius_px']
    inner = cv2.erode(camera.astype(np.uint8), np.ones((2*erosion+1, 2*erosion+1), np.uint8)) > 0
    guarded_hands = cv2.dilate(union, np.ones((2*guard+1, 2*guard+1), np.uint8)) > 0
    return inner & ~guarded_hands, union > 0, guarded_hands


def strict_mask_membership(mask, pixels):
    mask, pixels = np.asarray(mask, bool), np.asarray(pixels, float).reshape(-1, 2)
    if not np.isfinite(pixels).all():
        return False
    low, high = np.floor(pixels).astype(int), np.ceil(pixels).astype(int)
    h, w = mask.shape
    if (low < 0).any() or (high[:, 0] >= w).any() or (high[:, 1] >= h).any():
        return False
    return bool(mask[low[:, 1], low[:, 0]].all() and mask[low[:, 1], high[:, 0]].all()
                and mask[high[:, 1], low[:, 0]].all() and mask[high[:, 1], high[:, 0]].all())


def triangle_allowed(mask, triangle):
    triangle = np.asarray(triangle, float).reshape(3, 2)
    if not strict_mask_membership(mask, triangle):
        return False, 'vertex_outside_refined_domain'
    samples = [triangle.mean(axis=0)]
    for a, b in ((0, 1), (1, 2), (2, 0)):
        count = max(1, int(np.ceil(np.linalg.norm(triangle[a]-triangle[b])/PROTOCOL['maximum_edge_sample_spacing_px'])))
        samples.extend(np.linspace(triangle[a], triangle[b], count+1))
    if not strict_mask_membership(mask, samples):
        return False, 'edge_or_centroid_outside_refined_domain'
    # Source vertices are integer lattice centres, but rasterize conservatively
    # at 1/16 px for generic subpixel triangles and check every covered pixel.
    low = np.floor(triangle.min(axis=0)).astype(int)
    high = np.ceil(triangle.max(axis=0)).astype(int)
    tile = np.zeros((high[1]-low[1]+1, high[0]-low[0]+1), np.uint8)
    cv2.fillConvexPoly(tile, np.rint((triangle-low)*16).astype(np.int32), 1, shift=4)
    region = np.asarray(mask, bool)[low[1]:high[1]+1, low[0]:high[0]+1]
    if not region[tile > 0].all():
        return False, 'triangle_interior_outside_refined_domain'
    return True, None


def filter_surface(surface, domain):
    if not surface or surface.get('kind') != 'model_prior_source_visible_patch':
        raise ValueError('Expected a separate saved model-prior surface')
    pixels = np.asarray(surface['vertex_source_pixels'], float)
    vertices = np.asarray(surface['vertices'], float)
    triangles = np.asarray(surface['triangles'], int).reshape(-1, 3)
    if pixels.shape != (len(vertices), 2) or not np.isfinite(pixels).all() or not np.isfinite(vertices).all():
        raise ValueError('Source pixels and vertices must be finite and paired')
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise ValueError('Invalid saved model-prior triangle index')
    kept, rejected = [], Counter()
    for face_id, triangle in enumerate(triangles):
        keep, reason = triangle_allowed(domain, pixels[triangle])
        if keep:
            kept.append(face_id)
        else:
            rejected[reason] += 1
    report = {'original_vertices': len(vertices), 'original_triangles': len(triangles),
              'retained_triangles': len(kept), 'dropped_triangles': len(triangles)-len(kept),
              'dropped_by_reason': dict(rejected), 'new_vertices_or_faces_created': 0}
    if not kept:
        report['retained_vertices'] = 0
        return None, report
    retained_faces = triangles[kept]
    used = sorted(set(retained_faces.ravel().tolist()))
    mapping = {old: new for new, old in enumerate(used)}
    result = copy.deepcopy(surface)
    for key in ['vertices', 'colors', 'uv', 'vertex_source_pixels', 'vertex_provenance']:
        if key in surface:
            if len(surface[key]) != len(vertices):
                raise ValueError('Saved per-vertex attribute has invalid length')
            result[key] = [copy.deepcopy(surface[key][index]) for index in used]
    for key in ['face_provenance', 'face_confidence']:
        if key in surface:
            if len(surface[key]) != len(triangles):
                raise ValueError('Saved per-face attribute has invalid length')
            result[key] = [surface[key][index] for index in kept]
    result['triangles'] = [[mapping[int(index)] for index in face] for face in retained_faces]
    result['original_vertex_ids'] = used
    result['original_face_ids'] = kept
    result['inferred_face_count'] = len(kept)
    result['mask_filtering'] = 'native_camera_unique_iou_minus_union_all_native_hands_with_guard'
    result['clean_segmentation_established'] = False
    result['physical_confidence'] = 'unknown'
    report['retained_vertices'] = len(used)
    return result, report


def _read_native_review(root, manifest_path, source_frame, source_image_sha256):
    records = json.loads(manifest_path.read_text())['camera_occlusion_review']
    groups = {}
    files = [manifest_path]
    for concept in ['camera', 'hand']:
        matches = [record for record in records if record.get('concept') == concept]
        if len(matches) != 1:
            raise ValueError('Expected one exact-frame native review for '+concept)
        record = matches[0]
        frame = record['frame']
        if frame.get('model') != 'sam-3.1' or frame.get('provider') != 'Meta Model API':
            raise ValueError('Expected preserved native SAM 3.1 provider outputs')
        if str(frame['id']) != str(source_frame['id']) or frame['image_url'] != source_frame['image_url'] or abs(float(frame['timestamp_s'])-float(source_frame['timestamp_s'])) > 1e-8:
            raise ValueError('Native segmentation does not describe the exact source frame')
        if frame.get('image_sha256') != source_image_sha256:
            raise ValueError('Native segmentation source image identity is not bound')
        for artifact in ['native_metadata', 'raw_response', 'receipt']:
            path = C._local(root, frame[artifact+'_url'])
            if C.sha256(path) != frame[artifact+'_sha256']:
                raise ValueError('Native provider provenance identity mismatch')
            files.append(path)
        instances = []
        for instance in frame['instances']:
            path = C._local(root, instance['mask_url'])
            if C.sha256(path) != instance['mask_sha256']:
                raise ValueError('Native instance mask identity mismatch')
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (source_frame['height'], source_frame['width']):
                raise ValueError('Native mask dimensions do not match exact source pixels')
            instances.append((str(instance['native_instance_id']), mask))
            files.append(path)
        groups[concept] = instances
    return groups, files


def run(root, source_directory, native_manifest_path, output_directory):
    root = Path(root).resolve()
    source, output, manifest = (root/source_directory).resolve(), (root/output_directory).resolve(), (root/native_manifest_path).resolve()
    if not all(path.is_relative_to(root/'data') for path in [source, output, manifest]):
        raise ValueError('All inputs and output must remain inside project data')
    output.mkdir(parents=True, exist_ok=False)
    original_path = source/'viewer.json'
    view = json.loads(original_path.read_text())
    layer = view.get('inferred_surface')
    if not layer:
        raise ValueError('Source has no eligible saved model-prior surface')
    frame = next(f for f in view['frames'] if f['id'] == view['source_frame_id'])
    image_path, original_mask_path = C._local(root, frame['image_url']), C._local(root, frame['mask_url'])
    groups, native_files = _read_native_review(root, manifest, frame, C.sha256(image_path))
    prediction_path = source/'prediction.npz'
    originals = [original_path, source/'protocol.json', image_path, original_mask_path, prediction_path, *native_files]
    hashes = {str(path.relative_to(root)): C.sha256(path) for path in originals}
    implementation = Path(__file__).resolve()
    frozen = {'protocol': copy.deepcopy(PROTOCOL), 'source_directory': str(source.relative_to(root)),
              'native_manifest_path': str(manifest.relative_to(root)), 'source_frame': frame,
              'input_sha256': hashes, 'implementation_sha256': C.sha256(implementation),
              'frozen_before_filtering': True, 'manual_native_instance_selection': False}
    (output/'protocol.json').write_text(json.dumps(frozen, indent=2)+'\n')
    (output/'implementation.py').write_bytes(implementation.read_bytes())
    original_mask = cv2.imread(str(original_mask_path), cv2.IMREAD_GRAYSCALE)
    selected, selection = choose_camera(groups['camera'], original_mask)
    report = {'camera_selection': selection, 'native_hand_instance_count': len(groups['hand']),
              'new_api_calls': 0, 'new_model_inference_calls': 0, 'physical_confidence': 'unknown',
              'clean_segmentation_established': False}
    filtered, domain = None, np.zeros(original_mask.shape, bool)
    hand_union = np.zeros(original_mask.shape, bool)
    if selected is None:
        report['status'] = selection['status']
    elif not groups['hand']:
        report['status'] = 'no_native_hands_returned_filtering_unverified'
    else:
        camera_mask = next(mask for instance_id, mask in groups['camera'] if instance_id == selected)
        domain, hand_union, hand_guard = refined_domain(camera_mask, [mask for _, mask in groups['hand']])
        cv2.imwrite(str(output/'native-camera-mask.png'), (camera_mask > 0).astype(np.uint8)*255)
        cv2.imwrite(str(output/'native-hand-union.png'), hand_union.astype(np.uint8)*255)
        cv2.imwrite(str(output/'native-hand-guard.png'), hand_guard.astype(np.uint8)*255)
        cv2.imwrite(str(output/'refined-domain.png'), domain.astype(np.uint8)*255)
        filtered, geometry = filter_surface(layer, domain)
        report.update(status='model_filtered_prior_candidate_requires_review' if filtered else 'all_model_prior_faces_rejected',
                      geometry=geometry, selected_camera_native_instance_id=selected,
                      refined_domain_pixels=int(domain.sum()), hand_union_pixels=int(hand_union.sum()), hand_guard_pixels=int(hand_guard.sum()))
    # Preview and measurements concern the surviving triangle footprint, not
    # the entire native mask or all original model predictions.
    image = cv2.imread(str(image_path))
    footprint = np.zeros(domain.shape, np.uint8)
    original_footprint = np.zeros(domain.shape, np.uint8)
    for surface, target in [(layer, original_footprint), (filtered, footprint)]:
        if surface:
            pixels = np.asarray(surface['vertex_source_pixels'], float)
            for face in surface['triangles']:
                cv2.fillConvexPoly(target, np.rint(pixels[face]*16).astype(np.int32), 1, shift=4)
    with np.load(prediction_path) as prediction:
        original_inferred = prediction['inferred_source_mask']
    if original_inferred.shape != domain.shape:
        raise ValueError('Original inferred source mask dimensions mismatch')
    retained_inferred = (footprint > 0) & original_inferred
    report.update(retained_inferred_source_pixels=int(retained_inferred.sum()),
                  original_native_hand_overlap_footprint_pixels=int(((original_footprint > 0) & hand_union).sum()),
                  retained_native_hand_overlap_footprint_pixels=int(((footprint > 0) & hand_union).sum()),
                  original_face_footprint_pixels=int(original_footprint.sum()),
                  retained_face_footprint_pixels=int(footprint.sum()))
    limits = ['This is a derived model-filtered hypothesis, not proof of clean segmentation or correct shape.',
              'The camera instance is selected automatically by unique IoU; every native hand mask is subtracted with a guard.',
              'Native masks can miss or misclassify hands and camera parts. All filtered faces remain learned depth priors with unknown confidence.',
              'Original observed and model-prior geometries remain unchanged; the derivative only removes original inferred faces.',
              'No hidden backs, thickness, cavities, metric scale or physical simulation properties are established.']
    report['limitations'] = limits
    derived = copy.deepcopy(view)
    derived.update(id=output.name, title='Optional model prior · native camera/hand filtered',
                   status=report['status'], reason='Cached model-prior faces filtered by exact-frame native camera and hand masks; still an inferred hypothesis.',
                   inferred_surface=filtered, mask_filter_report=report,
                   original_model_prior_url='/files/'+str(original_path.relative_to(root)),
                   limitations=list(dict.fromkeys(view['limitations']+limits)), new_api_calls=0,
                   new_model_inference_calls=0)
    derived['metrics'] = copy.deepcopy(view['metrics'])
    derived['metrics'].update(inferred_vertices=len(filtered['vertices']) if filtered else 0,
                              inferred_triangles=len(filtered['triangles']) if filtered else 0,
                              mask_filter_dropped_triangles=report.get('geometry',{}).get('dropped_triangles'),
                              mask_filter_original_triangles=len(layer['triangles']),
                              native_hand_instances=len(groups['hand']),
                              original_model_inferred_source_pixels=view['metrics']['model_inferred_source_pixels'],
                              model_inferred_source_pixels=int(retained_inferred.sum()),
                              filter_new_model_inferences=0)
    (output/'viewer.json').write_text(json.dumps(derived, separators=(',', ':'), allow_nan=False)+'\n')
    (output/'filter-report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    preview = image.copy()
    preview[footprint > 0] = (.45*preview[footprint > 0]+.55*np.array([45,165,245])).astype(np.uint8)
    cv2.imwrite(str(output/'source-filtered-model-prior.png'), preview)
    cv2.imwrite(str(output/'retained-inferred-footprint.png'), footprint*255)
    for path, digest in hashes.items():
        if C.sha256(root/path) != digest:
            raise ValueError('A native mask or original geometry changed')
    (output/'input-integrity.json').write_text(json.dumps({'unchanged': True, 'input_sha256': hashes}, indent=2)+'\n')
    return report
