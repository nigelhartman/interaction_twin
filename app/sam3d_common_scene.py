"""Automatic, explicitly estimated SAM 3D scene replay from existing video masks.

This is a display estimate, not accepted motion, physical assembly or a simulator.
Native meshes and decoded SAM instance scales remain unchanged. One inferred K
and a camera-relative coordinate frame are shared by all parts. Background RGB
matches and inferred pointmaps estimate inter-anchor depth-gauge scale; per-frame
mask convex silhouettes estimate translation and unconstrained coarse rotation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

import upgrade_bundles as U
from sam3d_coordinates import native_layout_to_cv, SAM3D_COMMIT, DECODED_SCALE_SEMANTICS
from sam3d_source_projection import recover_estimated_camera
from native_appearance import load_native_glb

ROOT = Path(__file__).resolve().parent
FORMAT = 'sam3d-common-video-scene'
MAX_PARTS = 12
MAX_ROWS = 1200
SAMPLE_VERTICES = 1000
DEPENDENCIES = ['sam3d_common_scene.py', 'sam3d_coordinates.py', 'sam3d_source_projection.py',
                'native_appearance.py', 'sam3d_review_catalog.py', 'native_audit_protocol.py', 'part_ledger.py']


def require(value, message):
    if not value:
        raise ValueError(message)


def ref(path, root):
    path = U._local(path, root)
    return {'path': path.relative_to(root).as_posix(), 'sha256': U.sha(path), 'bytes': path.stat().st_size}


def source_ref(path, root):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.parent == root, 'Builder source unavailable')
    return {'path': path.name, 'sha256': U.sha(path), 'bytes': path.stat().st_size}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.new')
    temporary.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')
    temporary.replace(path)


def segmentation_for_program(program_path, *, root=ROOT):
    """Resolve exactly the compiled SAM attempt, never a directory-name default.

    The compiler's mask tracks and geometric gate references must be unique,
    hashed and in one ordinary segmentation directory inside the same case.
    The case manifest and source-frame index are independently source-bound.
    """
    root=Path(root).resolve();program_path=U._local(program_path,root);program=U.read(program_path)
    inputs=program.get('inputs',[])
    require(isinstance(inputs,list) and len(inputs)<=256 and all(isinstance(row,dict)for row in inputs),'Bounded compiled input references required')
    def compiled(identifier,expected=None):
        rows=[r for r in inputs if r.get('id')==identifier]
        require(len(rows)==1,'One exact compiled '+identifier+' input is required')
        row=rows[0];path=U._local(row.get('path',''),root)
        require(U.HEX64.fullmatch(str(row.get('sha256',''))) and U.sha(path)==row['sha256'],'Compiled '+identifier+' input changed')
        if expected is not None:require(path==expected,'Compiled '+identifier+' belongs to a different case or SAM attempt')
        return path
    manifest_path=compiled('source_manifest')
    require(manifest_path.name=='manifest.json','Compiled source manifest requires its canonical case filename')
    case=manifest_path.parent
    index_path=compiled('source_frame_index',case/'frames/index.json')
    manifest,index=U.read(manifest_path),U.read(index_path)
    source=program.get('source',{})
    require(source.get('time_domain')==U.TIME_DOMAIN and U.HEX64.fullmatch(str(source.get('sha256',''))) and manifest.get('source',{}).get('sha256')==source['sha256'] and index.get('time_domain')==U.TIME_DOMAIN,
            'Compiled SAM source recording or presentation-time index differs')
    tracks_path=compiled('mask_tracks');segment=tracks_path.parent
    require(tracks_path.name=='tracks.json' and segment.parent==case/'segmentation' and not segment.name.startswith('.'),
            'Compiled SAM tracks belong to a different case or segmentation directory')
    frames_path=compiled('mask_quality',segment/'frames.json')
    quality_path=U._local(segment/'quality.json',root)
    return {'case':case,'directory':segment,'tracks':tracks_path,'frames':frames_path,'quality':quality_path,
            'manifest':manifest_path,'index':index_path,'source_sha256':source['sha256']}


def fit_similarity(source, target):
    """Proper Sim(3), estimated from background correspondences only."""
    x, y = np.asarray(source, float), np.asarray(target, float)
    require(x.shape == y.shape and x.ndim == 2 and x.shape[1] == 3 and len(x) >= 3, 'Three matched 3D points required')
    xc, yc = x - x.mean(0), y - y.mean(0)
    require(np.linalg.matrix_rank(xc, tol=1e-9) >= 2, 'Background geometry is collinear')
    u, singular, vt = np.linalg.svd(yc.T @ xc / len(x))
    sign = np.ones(3); sign[-1] = np.linalg.det(u @ vt)
    r = u @ np.diag(sign) @ vt
    scale = float(np.sum(singular * sign) / np.mean(np.sum(xc * xc, axis=1)))
    require(np.isfinite(scale) and scale > 0, 'Positive background similarity required')
    return scale, r, y.mean(0) - scale * r @ x.mean(0)


def background_scale(image, pointmap, allowed, reference, reference_map, reference_allowed):
    """Frozen train/held-out ORB matches constrain inferred depth gauge, not physics."""
    feature = cv2.ORB_create(nfeatures=1800, edgeThreshold=15, fastThreshold=12)
    first, a = feature.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), allowed.astype(np.uint8) * 255)
    second, b = feature.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY), reference_allowed.astype(np.uint8) * 255)
    if a is None or b is None:
        return {'status': 'unknown_background_gauge', 'scale': 1., 'reason': 'insufficient_background_features'}
    matches = [pair[0] for pair in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(a, b, k=2)
               if len(pair) == 2 and pair[0].distance < .72 * pair[1].distance]
    pairs = []; source_pixels = []; target_pixels = []
    for match in matches:
        x, y = np.rint(first[match.queryIdx].pt).astype(int)
        u, v = np.rint(second[match.trainIdx].pt).astype(int)
        if min(x, y, u, v) < 0 or y >= len(pointmap) or x >= pointmap.shape[1] or v >= len(reference_map) or u >= reference_map.shape[1]:
            continue
        p, q = pointmap[y, x] * [-1, -1, 1], reference_map[v, u] * [-1, -1, 1]
        if np.isfinite(p).all() and np.isfinite(q).all() and min(p[2], q[2]) > 1e-5:
            pairs.append((p, q))
            source_pixels.append([x, y]); target_pixels.append([u, v])
    if len(pairs) < 18:
        return {'status': 'unknown_background_gauge', 'scale': 1., 'matches': len(pairs), 'reason': 'fewer_than_18_finite_background_matches'}
    source, target = np.asarray(pairs).transpose(1, 0, 2)
    ids = np.arange(len(source)); train, held = ids[ids % 3 != 0], ids[ids % 3 == 0]
    threshold = float(np.median(target[train, 2]) * .025)
    rng = np.random.default_rng(42); best = np.empty(0, int)
    try:
        for _ in range(240):
            sample = rng.choice(train, 3, replace=False)
            try:
                scale, rotation, translation = fit_similarity(source[sample], target[sample])
            except (ValueError, np.linalg.LinAlgError):
                continue
            errors = np.linalg.norm(scale * source[train] @ rotation.T + translation - target[train], axis=1)
            selected = train[errors < threshold]
            if len(selected) > len(best):
                best = selected
        require(len(best) >= 12, 'Insufficient background consensus')
        scale, rotation, translation = fit_similarity(source[best], target[best])
        errors = np.linalg.norm(scale * source[held] @ rotation.T + translation - target[held], axis=1)
        require(.25 <= scale <= 4 and np.median(errors) <= threshold * 1.5, 'Background held-out consistency failed')
        return {'status': 'estimated_background_pointmap_gauge', 'scale': scale, 'matches': len(source),
                'training_inliers': len(best), 'heldout_count': len(held), 'heldout_median_native_units': float(np.median(errors)),
                'heldout_p90_native_units': float(np.percentile(errors, 90)), 'native_threshold': threshold,
                'rotation_target_from_source': rotation.tolist(), 'translation_target_from_source': translation.tolist(),
                'source_inlier_bbox_image_fraction': float(np.prod(np.ptp(np.asarray(source_pixels)[best], axis=0)) / (image.shape[0] * image.shape[1])),
                'target_inlier_bbox_image_fraction': float(np.prod(np.ptp(np.asarray(target_pixels)[best], axis=0)) / (reference.shape[0] * reference.shape[1])),
                'physical_camera_calibration': False, 'method': 'ORB background matches + proper Sim3 RANSAC; fixed modulo3 heldout'}
    except (ValueError, np.linalg.LinAlgError) as error:
        return {'status': 'unknown_background_gauge', 'scale': 1., 'matches': len(source),
                'training_inliers': len(best), 'heldout_count': len(held), 'native_threshold': threshold,
                'reason': str(error), 'scale_origin': 'unchanged_native_SAM_prior_not_registered_background'}


def mask_shape(mask):
    yy, xx = np.where(mask)
    require(len(xx) >= 32, 'Mask has fewer than32 pixels')
    points = np.c_[xx, yy].astype(float)
    low, high = np.percentile(points, [1, 99], axis=0)
    require(np.all(high - low >= 3), 'Mask extent too small')
    center = points.mean(0)
    covariance = np.cov(points.T)
    hull = cv2.convexHull(points.astype(np.float32))
    return {'center': center, 'low': low, 'high': high, 'extent': high - low, 'area': len(xx),
            'covariance': covariance, 'contour': hull[::max(1, math.ceil(len(hull) / 96))]}


def projected_pose(vertices, rotation, scale, k, target, depth):
    """Fit translation and depth to mask extent; object scale remains fixed."""
    transformed = vertices @ rotation.T * scale
    center3 = np.mean(transformed, axis=0)
    # A positive depth keeps all representative native vertices before the camera.
    z = max(float(depth), float(np.ptp(transformed[:, 2])) * 1.5 + .02)
    for _ in range(2):
        translation = np.array([(target['center'][0] - k[0, 2]) * z / k[0, 0],
                                (target['center'][1] - k[1, 2]) * z / k[1, 1], z]) - center3
        camera = transformed + translation
        if np.min(camera[:, 2]) <= 1e-6:
            return None
        uvw = camera @ k.T; uv = uvw[:, :2] / uvw[:, 2:]
        low, high = np.percentile(uv, [1, 99], axis=0)
        ratio = np.sqrt(np.prod(np.maximum(high - low, 1) / target['extent']))
        z *= float(np.clip(ratio, .35, 3))
    translation = np.array([(target['center'][0] - k[0, 2]) * z / k[0, 0],
                            (target['center'][1] - k[1, 2]) * z / k[1, 1], z]) - center3
    camera = transformed + translation
    if np.min(camera[:, 2]) <= 1e-6:
        return None
    uvw = camera @ k.T; uv = uvw[:, :2] / uvw[:, 2:]
    # Correct projected centroid, without changing the mesh origin or scale.
    shift = target['center'] - uv.mean(0)
    translation[:2] += shift * z / np.diag(k)[:2]
    camera = transformed + translation; uvw = camera @ k.T; uv = uvw[:, :2] / uvw[:, 2:]
    return translation, uv, z


def silhouette_score(uv, target):
    """Cheap convex-silhouette fitting loss, explicitly not a raw mesh IoU gate."""
    center, extent = target['center'], target['extent']
    a = ((uv - center) / extent * 48 + 48).astype(np.float32)
    b = ((target['contour'].reshape(-1, 2) - center) / extent * 48 + 48).astype(np.float32)
    if np.max(np.abs(a)) > 1e5:
        return 2., 0.
    ma = np.zeros((96, 96), np.uint8); mb = ma.copy()
    cv2.fillConvexPoly(ma, np.rint(cv2.convexHull(a)).astype(np.int32), 1)
    cv2.fillConvexPoly(mb, np.rint(b).astype(np.int32), 1)
    intersection = int(np.count_nonzero(ma & mb)); union = int(np.count_nonzero(ma | mb))
    iou = intersection / max(1, union)
    return 1 - iou, iou


def estimate_pose(vertices, initial_rotation, scale, k, mask, initial_depth, previous_rotation=None, search=True):
    target = mask_shape(mask)
    initial = initial_rotation if previous_rotation is None else previous_rotation
    candidates = [initial, initial_rotation]
    # No manually selected hinge axis: all three camera-frame axes are tested.
    if search:
        for axis in np.eye(3):
            for degrees in [-22, -9, 9, 22]:
                candidates.append(Rotation.from_rotvec(axis * math.radians(degrees)).as_matrix() @ initial)
    best = None
    for rotation in candidates:
        distance = Rotation.from_matrix(rotation @ initial.T).magnitude()
        if previous_rotation is not None and distance > math.radians(45):
            continue
        fit = projected_pose(vertices, rotation, scale, k, target, initial_depth)
        if fit is None:
            continue
        translation, uv, depth = fit
        loss, iou = silhouette_score(uv, target)
        # Do not chase small convex-silhouette fluctuations with a 9/22degree
        # orientation jump. This remains an estimate, never accepted evidence.
        objective = loss + .14 * distance
        if best is None or objective < best['objective']:
            best = {'rotation': rotation, 'translation': translation, 'depth': depth, 'objective': objective,
                    'convex_silhouette_iou': iou, 'mask_area': target['area']}
    require(best is not None, 'No finite camera-relative estimate')
    matrix = np.eye(4); matrix[:3, :3] = scale * best['rotation']; matrix[:3, 3] = best['translation']
    best['matrix'] = matrix
    return best


def metadata_for_program(program, root):
    from sam3d_review_catalog import discover_reviews, match_reviews
    record = {'source': program['source'], 'components': [{'id': t['id']} for t in program['tracklets']], 'assets': program.get('assets', [])}
    rows = match_reviews(discover_reviews(root), record); result = []
    for candidate in rows:
        manifest = U.read(root / candidate['review_manifest']['path'])
        receipt = U._local(manifest['import_receipt']['path'], root)
        result.append(receipt.parent / 'metadata.json')
    return result


def interpolate_short_missing(poses, maximum_gap_s=1.):
    """Fill missing-mask rows only between two estimates of this SAME mesh.

    No extrapolation, skipped rejection, scale change, observed-status promotion
    or invented canonical geometry. Both source sample anchors are retained.
    """
    output = [dict(p) for p in poses]
    index = 0
    while index < len(output):
        if output[index]['status'] != 'unknown':
            index += 1; continue
        end = index
        while end < len(output) and output[end]['status'] == 'unknown':
            end += 1
        if index > 0 and end < len(output):
            left, right = output[index - 1], output[end]
            gap = right['timestamp_s'] - left['timestamp_s']
            missing = output[index:end]
            if 0 < gap <= maximum_gap_s and all(p['reason'] == 'no_usable_source_mask' for p in missing):
                a, b = np.asarray(left['camera_from_native_mesh']), np.asarray(right['camera_from_native_mesh'])
                sa, sb = np.cbrt(np.linalg.det(a[:3, :3])), np.cbrt(np.linalg.det(b[:3, :3]))
                require(sa > 0 and np.isclose(sa, sb, rtol=1e-6), 'Interpolation requires the same fixed native scale')
                slerp = Slerp([0., 1.], Rotation.from_matrix([a[:3, :3] / sa, b[:3, :3] / sb]))
                for p in missing:
                    fraction = (p['timestamp_s'] - left['timestamp_s']) / gap
                    matrix = np.eye(4); matrix[:3, :3] = sa * slerp(fraction).as_matrix()
                    matrix[:3, 3] = (1 - fraction) * a[:3, 3] + fraction * b[:3, 3]
                    p.update(status='estimated', camera_from_native_mesh=matrix.tolist(),
                             reason='short_missing_mask_gap_between_past_and_future_estimates', method='bounded_bidirectional_LERP_SLERP',
                             interpolation_neighbors=[left['source_sample_index'], right['source_sample_index']],
                             interpolation_gap_s=gap, native_uniform_scale=float(sa), accepted_motion_evidence=False)
        index = end
    return output


def unavailable_mask_reason(row, withheld_samples=()):
    if row is None:
        return 'no_usable_source_mask'
    if row['sample_index'] in withheld_samples:
        return 'source_mask_failed_saved_quality_gate'
    if row.get('zero_placeholder_for_missing_or_withheld') is True or row.get('status') == 'withheld_due_to_occlusion_or_ambiguity':
        return 'source_mask_withheld_or_ambiguous'
    return None


def load_model(path, source, root):
    path = U._local(path, root); model = U.read(path)
    require(model.get('backend') == 'Meta SAM 3D Objects' and model.get('sam3d_output') is True
            and model.get('learned_inference_executed') is True and model.get('manual_geometry_fallback') is False
            and model.get('manual_pose_used') is False and model.get('source', {}).get('video_sha256') == source,
            'A genuine automatic source-matched SAM 3D asset is required')
    asset = U._local(path.parent / 'asset.glb', root)
    artifact = next(row for row in model['artifact_files'] if row['file'] == 'asset.glb')
    require(U.sha(asset) == artifact['sha256'], 'Native asset changed')
    raw = U.read(U._local(path.parent / 'raw/metadata.json', root))
    layout = U._local(path.parent / 'raw/layout-and-pointmap.npz', root)
    require(next(row['sha256'] for row in raw['outputs']['files'] if row['file'] == 'layout-and-pointmap.npz') == U.sha(layout), 'Raw layout changed')
    with U._safe_layout(layout.read_bytes()) as arrays:
        pointmap = arrays['pointmap'].copy()
        decoded = native_layout_to_cv(rotation_wxyz=arrays['rotation'], translation=arrays['translation'], scale_xyz=arrays['scale'],
                                       source_commit=SAM3D_COMMIT, scale_semantics=DECODED_SCALE_SEMANTICS)
    image_path = U._local(path.parent / model['source']['image_path'], root)
    mask_path = U._local(path.parent / model['source']['mask_path'], root)
    require(U.sha(image_path) == model['source']['image_sha256'] and U.sha(mask_path) == model['source']['mask_sha256'], 'Native source RGB/mask changed')
    image = np.asarray(Image.open(image_path).convert('RGB')); mask = np.asarray(Image.open(mask_path)) > 0
    require(pointmap.shape == (*mask.shape, 3) and image.shape[:2] == mask.shape, 'Exact full-frame pointmap required')
    return {'metadata': model, 'metadata_ref': ref(path, root), 'asset': ref(asset, root), 'layout': decoded,
            'pointmap': pointmap, 'image': image, 'source_mask': mask, 'layout_ref': ref(layout, root),
            'image_ref': ref(image_path, root), 'mask_ref': ref(mask_path, root), 'camera': recover_estimated_camera(pointmap)}


def build(program_path, output, *, root=ROOT, metadata_paths=None, part_ledger_path=None, held_out_sample_indices=()):
    root = Path(root).resolve(); start = time.monotonic()
    program_path = U._local(program_path, root); program = U.read(program_path)
    require(program.get('source', {}).get('time_domain') == U.TIME_DOMAIN, 'Source presentation-time program required')
    out = Path(output); out = out if out.is_absolute() else root / out
    require(out.is_relative_to(root / 'data') and out.resolve() == out and not any(p.is_symlink() for p in [out, *out.parents]), 'Ordinary new data output required')
    ledger_ref=ref(part_ledger_path,root) if part_ledger_path is not None else None
    if ledger_ref is not None:
        import part_ledger
        authoritative=part_ledger.metadata_paths(part_ledger_path,program_path=program_path,root=root)
        if metadata_paths is not None:
            require([U._local(p,root)for p in metadata_paths]==authoritative,'Explicit native metadata paths differ from the authoritative part ledger')
        model_paths=authoritative
    else:
        model_paths=[U._local(p,root)for p in (metadata_paths if metadata_paths is not None else metadata_for_program(program,root))]
    requested_holdouts=list(held_out_sample_indices)
    require(all(type(i)is int and i>0 for i in requested_holdouts) and len(set(requested_holdouts))==len(requested_holdouts),'Unique held-out source samples required')
    dependencies = [source_ref(root / name, root) for name in DEPENDENCIES]
    if (out / 'scene.json').exists():
        previous = U.read(out / 'scene.json')
        built = U.read(out / 'build.json')
        require(built.get('format') == 'sam3d-common-scene-build' and built.get('manifest') == ref(out / 'scene.json', root), 'Cached scene output changed')
        require(previous.get('part_ledger')==ledger_ref and previous.get('evaluation_protocol',{}).get('held_out_sample_indices',[])==sorted(requested_holdouts),'Existing scene ledger or evaluation split changed; choose a fresh version')
        require(previous.get('format') == FORMAT and previous.get('original_program') == ref(program_path, root)
                and previous.get('dependencies') == dependencies and previous.get('model_metadata_paths') == [p.relative_to(root).as_posix() for p in model_paths],
                'Existing scene source or implementation changed; choose a fresh version')
        payload={'inputs':previous['inputs'],'dependencies':dependencies}
        if previous.get('fitter_version')=='compiled-segmentation-v2':payload.update(evaluation_protocol=previous['evaluation_protocol'],part_ledger=previous.get('part_ledger'))
        require(previous.get('input_signature')==U.digest(U.encoded(payload)),'Cached input signature differs')
        for item in previous['inputs']:
            require(ref(item['path'], root) == item, 'Existing scene frozen source/model/mask changed; choose a fresh version')
        return {'status': 'reused_verified_estimated_scene', 'manifest': ref(out / 'scene.json', root),
                'source_sha256': program['source']['sha256'], 'components': len(previous['components']),
                'estimated_pose_count': sum(p['status'] == 'estimated' for c in previous['components'] for p in c['poses']),
                'unknown_pose_count': sum(p['status'] == 'unknown' for c in previous['components'] for p in c['poses']), 'accepted_motion_poses': 0}
    models = [load_model(p, program['source']['sha256'], root) for p in model_paths]
    require(1 <= len(models) <= MAX_PARTS and len({m['metadata']['object_id'] for m in models}) == len(models), 'One to12 unique source-bound components required')
    binding=segmentation_for_program(program_path,root=root);case=binding['case']
    manifest = U.read(binding['manifest']); index = U.read(binding['index'])
    require(manifest['source']['sha256'] == program['source']['sha256'] and index['time_domain'] == U.TIME_DOMAIN, 'Source case/time domain differs')
    video = U._local(case / manifest['source']['file'], root)
    require(U.sha(video) == program['source']['sha256'], 'Original video changed')
    segment=binding['directory'];track_path=binding['tracks'];tracks=U.read(track_path)
    quality_path=binding['quality'];quality=U.read(quality_path)
    frame_rows = index['frames']; require(2 <= len(frame_rows) <= MAX_ROWS, 'Bounded saved source frames required')
    source_rows = {row['index']: row for row in frame_rows}
    from native_audit_protocol import make_protocol
    protocol=make_protocol(program['source']['sha256'],list(source_rows),requested_holdouts,protected_sample_indices=[m['metadata']['source']['sample_index'] for m in models])
    held_out=set(protocol['held_out_sample_indices'])
    source_tracks = {row['track_id']: {r['sample_index']: r for r in row['frames']} for row in tracks['tracks']}
    inputs = [ref(program_path, root), ref(case / 'manifest.json', root), ref(case / 'frames/index.json', root), ref(track_path, root), ref(binding['frames'],root), ref(quality_path, root), ref(video, root)]
    if ledger_ref:inputs.append(ledger_ref)
    for model in models:
        inputs.extend(model[key] for key in ['metadata_ref', 'asset', 'layout_ref', 'image_ref', 'mask_ref'])
    pixel_inputs = {}
    for track in tracks['tracks']:
        for row in track['frames']:
            path = U._local(segment / row['raw_mask_file'], root); item = ref(path, root)
            require(item['sha256'] == row['raw_mask_sha256'], 'Source mask changed')
            pixel_inputs[item['path']] = item
            frame = U._local(case / source_rows[row['sample_index']]['file'], root); item = ref(frame, root)
            require(item['sha256'] == row['source_frame_sha256'], 'Source pixels changed')
            pixel_inputs[item['path']] = item
    inputs += [pixel_inputs[key] for key in sorted(pixel_inputs)]
    signature = U.digest(U.encoded({'inputs': inputs, 'dependencies': dependencies,'evaluation_protocol':protocol,'part_ledger':ledger_ref}))
    require(not out.exists(), 'Preserve previous scene version')
    # Largest available automatic anchor is a deterministic camera reference.
    reference = max(models, key=lambda m: int(m['source_mask'].sum()))
    k = np.asarray(reference['camera']['K_estimated']); height, width = reference['source_mask'].shape
    reference_index = reference['metadata']['source']['sample_index']
    mask_cache = {}
    def mask_at(component, sample):
        key = (component, sample)
        if sample in held_out:return (None,None)
        if key in mask_cache:
            return mask_cache[key]
        row = source_tracks.get(component, {}).get(sample)
        withheld = quality.get('tracks', {}).get(component, {}).get('withheld_samples', [])
        if unavailable_mask_reason(row, withheld) is not None:
            mask_cache[key] = (None, None); return mask_cache[key]
        path = U._local(segment / row['raw_mask_file'], root)
        require(U.sha(path) == row['raw_mask_sha256'], 'Saved source mask changed')
        image = np.asarray(Image.open(path)) > 0
        require(image.shape == (height, width), 'Source mask dimensions differ')
        mask_cache[key] = (image, ref(path, root)); return mask_cache[key]
    def background(sample):
        excluded = np.zeros((height, width), np.uint8)
        for component in source_tracks:
            mask, _ = mask_at(component, sample)
            if mask is not None:
                excluded |= mask.astype(np.uint8)
        return cv2.dilate(excluded, np.ones((21, 21), np.uint8)) == 0
    reference_background = background(reference_index)
    components = []; observations = []
    for model in models:
        identifier = model['metadata']['object_id']; require(identifier in source_tracks, 'No source masks for native component')
        layout = model['layout']; anchor = model['metadata']['source']['sample_index']
        require(anchor in source_rows and source_rows[anchor]['timestamp_s'] == model['metadata']['source']['timestamp_s'], 'Native anchor source PTS differs')
        frame_path = U._local(case / source_rows[anchor]['file'], root)
        require(np.array_equal(np.asarray(Image.open(frame_path).convert('RGB')), model['image']), 'Native anchor pixels differ from source frame')
        if model is reference:
            gauge = {'status': 'reference_background_gauge', 'scale': 1., 'source_anchor': anchor}
        else:
            gauge = background_scale(model['image'], model['pointmap'], background(anchor), reference['image'], reference['pointmap'], reference_background)
        scale = layout.scale * gauge['scale']
        mesh = load_native_glb(root / model['asset']['path'])
        # A representative vertex subset is used only for pose estimation. The
        # full unchanged GLB is the asset rendered and exported by the viewer.
        vertices = mesh.vertices[np.linspace(0, len(mesh.vertices) - 1, min(SAMPLE_VERTICES, len(mesh.vertices)), dtype=int)]
        initial_rotation = layout.rotation_cv_from_glb
        initial_depth = float(layout.translation_cv[2] * gauge['scale'])
        poses = {r['index']: {'timestamp_s': r['timestamp_s'], 'source_sample_index': r['index'], 'status': 'unknown',
                             'camera_from_native_mesh': None, 'reason': 'no_usable_source_mask', 'method': 'unknown'} for r in frame_rows}
        candidates = []
        for row in frame_rows:
            if row['index'] in held_out:
                poses[row['index']]['reason']='held_out_for_native_render_audit';continue
            mask, mask_ref = mask_at(identifier, row['index'])
            if mask is not None and np.count_nonzero(mask) >= 32:
                candidates.append((row, mask, mask_ref))
            else:
                reason = unavailable_mask_reason(source_tracks.get(identifier, {}).get(row['index']),
                                                 quality.get('tracks', {}).get(identifier, {}).get('withheld_samples', []))
                poses[row['index']]['reason'] = reason or 'source_mask_too_small_for_pose_estimation'
        # Walk outward from the exact automatic native anchor; no hand-authored
        # states, hinge axis or previous backend poses participate.
        left = sorted((c for c in candidates if c[0]['index'] < anchor), key=lambda c: -c[0]['index'])
        right = sorted((c for c in candidates if c[0]['index'] >= anchor), key=lambda c: c[0]['index'])
        for sequence in [right, left]:
            rotation = initial_rotation; depth = initial_depth; previous_area = None; previous_time = None
            for row, mask, mask_ref in sequence:
                current = poses[row['index']]; area = int(mask.sum())
                # Abrupt partial silhouettes cannot establish apparent depth.
                abrupt = previous_area is not None and max(area / previous_area, previous_area / area) > 3.5 and abs(row['timestamp_s'] - previous_time) < .8
                if abrupt:
                    current['reason'] = 'abrupt_mask_area_change_possible_occlusion'; continue
                fit = estimate_pose(vertices, initial_rotation, scale, k, mask, depth, previous_rotation=rotation)
                if not .4 * initial_depth <= fit['depth'] <= 2.5 * initial_depth:
                    current.update(reason='mask_depth_ambiguous_outside_native_prior_envelope', source_mask=mask_ref,
                                   rejected_estimated_depth=fit['depth'], native_depth_prior=initial_depth)
                    continue
                if fit['convex_silhouette_iou'] < .35:
                    current.update(reason='convex_silhouette_fit_inadequate', source_mask=mask_ref,
                                   convex_silhouette_fit_iou=fit['convex_silhouette_iou']); continue
                current.update(status='estimated', camera_from_native_mesh=fit['matrix'].tolist(),
                               reason='automatic_mask_and_SAM_native_prior_display_estimate_not_validated_6DoF',
                               method='coarse_all_axes_rotation_and_bbox_depth_translation_fit', source_mask=mask_ref,
                               convex_silhouette_fit_iou=fit['convex_silhouette_iou'], mask_area_pixels=area,
                               native_uniform_scale=scale, accepted_motion_evidence=False)
                rotation, depth, previous_area, previous_time = fit['rotation'], fit['depth'], area, row['timestamp_s']
        track = next(t for t in program['tracklets'] if t['id'] == identifier)
        pose_rows = interpolate_short_missing([poses[r['index']] for r in frame_rows])
        components.append({'id': identifier, 'label': track.get('label', identifier), 'asset': model['asset'],
                           'native_metadata': model['metadata_ref'], 'source_anchor': model['metadata']['source'],
                           'native_uniform_scale': scale, 'predicted_SAM_instance_scale': layout.scale, 'depth_gauge': gauge,
                           'sampled_vertices_for_estimation_only': len(vertices), 'full_mesh_geometry_modified': False,
                           'poses': pose_rows, 'physical_identity_verified': False,
                           'physical_accuracy_verified': False, 'source_video_alignment_verified': False})
        observations.append({'id': identifier, 'estimated': sum(r['status'] == 'estimated' for r in pose_rows),
                             'interpolated_short_missing': sum(r.get('method') == 'bounded_bidirectional_LERP_SLERP' for r in pose_rows),
                             'unknown': sum(r['status'] == 'unknown' for r in pose_rows), 'depth_gauge': gauge})
        known = [r for r in pose_rows if r['status'] == 'estimated']
        depths = [float(np.asarray(r['camera_from_native_mesh'])[2, 3]) for r in known]
        steps = []
        for previous, current in zip(pose_rows, pose_rows[1:]):
            if previous['status'] == current['status'] == 'estimated':
                a = np.asarray(previous['camera_from_native_mesh'])[:3, :3] / scale
                b = np.asarray(current['camera_from_native_mesh'])[:3, :3] / scale
                steps.append(float(np.rad2deg(Rotation.from_matrix(b @ a.T).magnitude())))
        observations[-1].update(native_depth_prior=initial_depth,
                                displayed_translation_z_range=[min(depths), max(depths)] if depths else None,
                                adjacent_rotation_step_degrees_median=float(np.median(steps)) if steps else None,
                                adjacent_rotation_step_degrees_p90=float(np.percentile(steps, 90)) if steps else None,
                                depth_rejections=sum(r['reason'] == 'mask_depth_ambiguous_outside_native_prior_envelope' for r in pose_rows),
                                orientation_scope='temporally_regularized_coarse_silhouette_estimate_not_recovered_joint_motion')
    scene = {'format': FORMAT, 'version': 1, 'input_signature': signature, 'created_utc': datetime.now(timezone.utc).isoformat(),
             'source': {**program['source'], 'width': width, 'height': height}, 'original_program': ref(program_path, root),
             'camera': {'K': k.tolist(), 'width': width, 'height': height, 'world_from_camera': np.eye(4).tolist(),
                        'calibrated': False, 'method': 'fixed estimated intrinsics from native anchor inferred pointmap',
                        'reference_component': reference['metadata']['object_id'], 'reference_sample_index': reference_index,
                        'per_component_inferred_cameras': {m['metadata']['object_id']: m['camera']['K_estimated'] for m in models}},
             'common_frame': {'id': 'current_source_camera_cv', 'coordinate_convention': '+Xright+Ydown+Zforward',
                              'metric_scale_verified': False, 'assembly_verified': False, 'camera_motion_removed': False,
                              'scope': 'simultaneous camera-relative display estimates; not a reconstructed stationary world'},
             'timing': {'max_hold_s': 0, 'max_interpolation_gap_s': .8, 'max_estimated_gap_s': 1.0,
                        'interpolation': 'viewer may interpolate adjacent valid estimates; no crossing unknown rows'},
             'components': components, 'inputs': inputs, 'dependencies': dependencies,
             'model_metadata_paths': [p.relative_to(root).as_posix() for p in model_paths], 'part_ledger':ledger_ref,'evaluation_protocol':protocol,'fitter_version':'compiled-segmentation-v2', 'accepted_motion_poses': 0,
             'physical_accuracy_verified': False, 'simulation_ready': False, 'manual_geometry_or_poses_used': False,
             'old_backend_geometry_or_poses_read': False, 'learned_tracker_run': False, 'model_or_api_calls': 0,
             'diagnostics': observations, 'wall_seconds': time.monotonic() - start,
             'limitations': ['All displayed transforms are automatic estimates, not accepted observed 6DoF motion.',
                             'One inferred shared pinhole camera is approximate; source intrinsics are not calibrated.',
                             'Convex silhouettes omit concavities and contact. Fitting IoU is not an independent raw-mesh validation.',
                             'Out-of-plane orientation, hidden geometry, depth and physical identity remain ambiguous.',
                             'The camera-relative frame contains camera motion; it is not a static world reconstruction.',
                             'Background gauge fitting uses inferred pointmaps, not measured depth or metric scale.',
                             'No joint axis, correct assembly, insertion, seating, collision, contact or simulation is established.']}
    out.mkdir(parents=True, exist_ok=False); write(out / 'scene.json', scene)
    write(out / 'build.json', {'format': 'sam3d-common-scene-build', 'version': 1,
                             'manifest': ref(out / 'scene.json', root), 'input_signature': signature,
                             'model_or_api_calls': 0, 'accepted_motion_poses': 0})
    return {'status': 'estimated_common_scene_built', 'manifest': ref(out / 'scene.json', root),
            'source_sha256': program['source']['sha256'], 'components': len(components),
            'estimated_pose_count': sum(r['estimated'] for r in observations), 'unknown_pose_count': sum(r['unknown'] for r in observations),
            'accepted_motion_poses': 0, 'wall_seconds': scene['wall_seconds']}


def publish_registry(result, *, root=ROOT):
    root = Path(root).resolve(); record = ref(result['manifest']['path'], root)
    require(record == result['manifest'], 'Scene bytes changed before local indexing')
    scene = U.read(root / record['path'])
    require(scene.get('format') == FORMAT and scene.get('version') == 1 and scene.get('physical_accuracy_verified') is False
            and scene.get('simulation_ready') is False and scene['source']['sha256'] == result['source_sha256'], 'Scene scope differs')
    path = root / 'showcase/aligned-scenes.json'
    current = U.read(path) if path.exists() else {'format': 'sam3d-aligned-scene-index', 'version': 1, 'scenes': []}
    require(current.get('format') == 'sam3d-aligned-scene-index' and current.get('version') == 1, 'Scene index format differs')
    rows = [r for r in current['scenes'] if r['source_sha256'] != result['source_sha256']]
    rows.append({'source_sha256': result['source_sha256'], 'manifest': record})
    require(len(rows) <= 100, 'Scene index budget exceeded')
    write(path, {'format': 'sam3d-aligned-scene-index', 'version': 1, 'scenes': rows})
    return {'status': 'local_estimated_scene_indexed', 'scenes': len(rows)}
