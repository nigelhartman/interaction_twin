#!/usr/bin/env python3
"""Conditional generated-mesh motion from generic queries and actual CoTracker.

No manual corners, dimensions, old poses, hidden seating, or depth are used.
Reference geometry, intrinsics and pose remain unverified model assumptions.
"""
import argparse
import hashlib
import html
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignment_fit import solve_pose
from alignment_render import ray_mesh_points, render_mesh


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def font(size):
    try:
        return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', size)
    except OSError:
        return ImageFont.load_default(size=size)


def query_points(image, mask, maximum=96):
    """Deterministic interior Shi-Tomasi corners plus an adaptive mask grid."""
    if image.shape[:2] != mask.shape:
        raise ValueError('Image and mask must share full-frame pixels')
    interior = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if not interior.any():
        return np.empty((0, 2), np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = cv2.goodFeaturesToTrack(gray, maxCorners=maximum//2, qualityLevel=.003,
                                     minDistance=4, mask=interior.astype(np.uint8)*255, blockSize=3)
    seeds = [] if features is None else [p.tolist() for p in features[:, 0]]
    spacing = max(3, int(math.ceil(math.sqrt(interior.sum()/maximum))))
    ys, xs = np.where(interior)
    for y in range(int(ys.min()), int(ys.max())+1, spacing):
        for x in range(int(xs.min()), int(xs.max())+1, spacing):
            if interior[y, x] and (not seeds or np.min(np.linalg.norm(np.array(seeds)-[x, y], axis=1)) >= 3):
                seeds.append([float(x), float(y)])
            if len(seeds) >= maximum:
                break
        if len(seeds) >= maximum:
            break
    # Stable IDs assigned by pixel location, never by fitted residuals.
    return np.asarray(sorted(seeds, key=lambda p: (p[1], p[0])), dtype=np.float32).reshape(-1, 2)


def support_masks(raw_uv, model_visible, masks, anchor):
    """Separate raw evidence from absorbing loss, independently each direction."""
    raw_uv = np.asarray(raw_uv)
    model_visible = np.asarray(model_visible, bool)
    if raw_uv.ndim != 3 or raw_uv.shape[2] != 2 or model_visible.shape != raw_uv.shape[:2]:
        raise ValueError('Expected frame-by-point UV and visibility arrays')
    bounded = np.isfinite(raw_uv).all(2)
    membership = np.zeros_like(model_visible)
    for j, mask in enumerate(masks):
        h, w = mask.shape
        bounded[j] &= (raw_uv[j, :, 0] >= 0) & (raw_uv[j, :, 0] <= w-1) & (raw_uv[j, :, 1] >= 0) & (raw_uv[j, :, 1] <= h-1)
        ids = np.flatnonzero(bounded[j])
        pixel = np.rint(raw_uv[j, ids]).astype(int)
        membership[j, ids] = mask[pixel[:, 1], pixel[:, 0]]
    instantaneous = bounded & membership & model_visible
    valid = instantaneous.copy()
    for direction in [1, -1]:
        previous = anchor
        for j in range(anchor+direction, len(masks) if direction == 1 else -1, direction):
            valid[j] &= valid[previous]
            previous = j
    return {'image_bounds': bounded, 'sam_membership': membership,
            'instantaneous_support': instantaneous, 'valid_no_reacquisition': valid}


def split_ids(reference_valid):
    ids = np.arange(len(reference_valid))
    return ids[reference_valid & (ids % 3 != 0)], ids[reference_valid & (ids % 3 == 0)]


def project(xyz, k, rotation, translation):
    camera = np.asarray(xyz) @ rotation.T + translation
    pixels = camera @ k.T
    return pixels[:, :2]/pixels[:, 2:]


def pose_gate(train_count, inlier_count, heldout_errors, silhouette_iou, coverage, mask_verdict):
    errors = np.asarray(heldout_errors, float)
    reasons = []
    if train_count < 12:
        reasons.append('fewer_than_12_training_points')
    if inlier_count < 8 or inlier_count/max(1, train_count) < .65:
        reasons.append('insufficient_training_consensus')
    if len(errors) < 6:
        reasons.append('fewer_than_6_disjoint_heldout_points')
    if not len(errors) or not np.isfinite(errors).all() or np.median(errors) > 3 or np.percentile(errors, 90) > 8:
        reasons.append('heldout_error_above_3px_median_or_8px_p90')
    if silhouette_iou < .45 or coverage < .6:
        reasons.append('raw_full_mesh_silhouette_gate_failed')
    if mask_verdict != 'plausible_visible_mask':
        reasons.append('source_mask_not_plausible_in_model_audit')
    return reasons


def cotracker(images, seeds, anchor, device):
    import torch
    from cotracker.predictor import CoTrackerPredictor
    from importlib.metadata import distribution
    if device != 'mps' or not torch.backends.mps.is_available():
        raise RuntimeError('Actual MPS required; no CPU inference fallback')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('CPU operator fallback must remain disabled')
    checkpoint = ROOT/'models/cotracker3/scaled_offline.pth'
    provenance = read(checkpoint.parent/'provenance.json')
    if sha(checkpoint) != provenance['checkpoint_sha256']:
        raise ValueError('CoTracker checkpoint hash changed')
    package = json.loads(distribution('cotracker').read_text('direct_url.json'))
    if package['vcs_info']['commit_id'] != provenance['code_commit']:
        raise ValueError('CoTracker installed code commit differs from provenance')
    model = CoTrackerPredictor(checkpoint=str(checkpoint)).to(device).eval()
    model.interp_shape = (384, 216)  # Explicit 9:16 portrait development resolution.
    frames = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)[None].float().to(device)
    queries = torch.from_numpy(np.c_[np.full(len(seeds), anchor), seeds].astype(np.float32))[None].to(device)
    start = time.monotonic()
    with torch.inference_mode():
        prediction, visibility = model(frames, queries=queries, backward_tracking=True)
        torch.mps.synchronize()
        uv = prediction[0].cpu().numpy()
        visible = visibility[0].cpu().numpy().astype(bool)
    return uv, visible, {'backend': 'Meta CoTracker3 offline', 'role': 'learned 2D tracks, not TAPIP3D or measured 3D',
                         'device': device, 'model_input_hw': [384, 216], 'native_input_hw': list(images[0].shape[:2]),
                         'input_aspect_ratio_preserved': True, 'cpu_operator_fallback_enabled': False,
                         'checkpoint_download_performed': False, 'model_provenance': provenance,
                         'torch': torch.__version__, 'inference_seconds': round(time.monotonic()-start, 3)}


def make_review(out, images, frames, masks, renders, raw_uv, support, heldout_ids):
    (out/'overlays').mkdir()
    (out/'source').mkdir()
    (out/'details').mkdir()
    (out/'full_projection_masks').mkdir()
    (out/'mesh_rgb').mkdir()
    tiles = []
    for j, (image, rec) in enumerate(zip(images, frames)):
        sample = rec['sample_index']
        source = Image.fromarray(image)
        source.save(out/'source'/f'{sample:06d}.jpg', quality=95)
        overlay = image.astype(float).copy()
        render = renders[j]
        if render is not None:
            raw = render['mask']
            Image.fromarray(raw.astype(np.uint8)*255).save(out/'full_projection_masks'/f'{sample:06d}.png')
            Image.fromarray(render['rgb']).save(out/'mesh_rgb'/f'{sample:06d}.png')
            rec['raw_full_projection_mask_file'] = f'full_projection_masks/{sample:06d}.png'
            rec['raw_generated_rgb_file'] = f'mesh_rgb/{sample:06d}.png'
            overlay[raw] = image[raw]*.35+render['rgb'][raw]*.65
        overlay = Image.fromarray(overlay.astype(np.uint8))
        draw = ImageDraw.Draw(overlay)
        for point_id in np.flatnonzero(support[j]):
            x, y = raw_uv[j, point_id]
            color = '#36ff8e' if point_id in heldout_ids else '#ffd54a'
            draw.ellipse((x-1.5, y-1.5, x+1.5, y+1.5), fill=color)
        draw.rectangle((0, 0, image.shape[1], 100), fill='#142d3d')
        draw.text((12, 12), f"#{sample} / {rec['timestamp_s']:.6f}s", font=font(22), fill='white')
        draw.text((12, 42), rec['status'], font=font(19), fill='#bfe9ff')
        draw.text((12, 70), 'yellow=train / green=held out; raw full mesh', font=font(16), fill='white')
        overlay.save(out/'overlays'/f'{sample:06d}.jpg', quality=95)
        target = masks[j]
        visible = target.copy()
        if render is not None:
            visible |= render['mask']
        ys, xs = np.where(visible)
        crop = (max(0, int(xs.min())-24), max(0, int(ys.min())-24), min(image.shape[1], int(xs.max())+25), min(image.shape[0], int(ys.max())+25)) if len(xs) else (0, 0, image.shape[1], image.shape[0])
        detail = Image.new('RGB', (960, 400), '#eff2f5')
        for k, im in enumerate([source, overlay]):
            thumb = ImageOps.contain(im.crop(crop), (460, 340))
            detail.paste(thumb, (10+k*480+(460-thumb.width)//2, 48+(340-thumb.height)//2))
        d = ImageDraw.Draw(detail)
        d.text((12, 8), f"sample {sample} / source LEFT / candidate raw projection RIGHT", fill='#173244', font=font(22))
        detail.save(out/'details'/f'{sample:06d}.jpg', quality=95)
        rec['preview_detail_crop_xyxy'] = list(crop)
        tiles.append(detail)
    contact = Image.new('RGB', (960, len(tiles)*250+55), '#eff2f5')
    d = ImageDraw.Draw(contact)
    d.text((12, 12), 'Conditional generated-earbud motion / failures retained', fill='#173244', font=font(23))
    for j, tile in enumerate(tiles):
        thumb = ImageOps.contain(tile, (960, 245))
        contact.paste(thumb, (0, 55+j*250))
    contact.save(out/'contact-sheet.jpg', quality=94)
    rows = [{'index': r['sample_index'], 'timestamp_s': r['timestamp_s'], 'status': r['status'], 'reasons': r['reasons']} for r in frames]
    page = r'''<!doctype html><meta charset="utf-8"><title>Generated mesh motion pilot</title>
<style>body{font:16px system-ui;background:#eef2f5;color:#153244;margin:24px}h1{font-size:26px}main{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{max-width:100%;max-height:66vh;object-fit:contain}figure{margin:0;text-align:center;background:white;padding:12px;border-radius:12px}input{width:70%}pre{white-space:pre-wrap}#detail{max-height:400px}button{padding:8px}</style>
<h1>Generated mesh → tracked video: conditional motion experiment</h1>
<p>Real TripoSR mesh + CoTracker3 MPS + model-seeded SAM. Reference pose, focal length and hidden shape are unverified. No physical pose accuracy or seating is claimed.</p>
<button id="prev">←</button> <input id="slider" type="range" min="0" value="0"> <button id="next">→</button>
<p id="status"></p><main><figure><figcaption>Source, original pixels and PTS</figcaption><img id="source"></figure><figure><figcaption>Raw mesh projection + disjoint point support; rejected fits retained</figcaption><img id="overlay"></figure></main>
<figure><img id="detail"></figure><pre id="reason"></pre>
<p>Unsupported frames have no exported usable pose. No interpolation or reacquisition. Green points never enter PnP fitting. Detail crops only enlarge the display.</p>
<p><a href="experiment.json">Evidence and provenance</a> · <a href="point-support.json">Raw point support</a></p>
<script>const rows=ROWS;let j=0;const slider=document.getElementById('slider');slider.max=rows.length-1;function show(){let r=rows[j],f=String(r.index).padStart(6,'0')+'.jpg';slider.value=j;document.getElementById('source').src='source/'+f;document.getElementById('overlay').src='overlays/'+f;document.getElementById('detail').src='details/'+f;document.getElementById('status').textContent=`sample ${r.index} / source ${r.timestamp_s.toFixed(6)} s / ${r.status}`;document.getElementById('reason').textContent=r.reasons.join('\n')||'Reference initialization / internal conditional gates only';}slider.oninput=()=>{j=Number(slider.value);show()};document.getElementById('prev').onclick=()=>{j=Math.max(0,j-1);show()};document.getElementById('next').onclick=()=>{j=Math.min(rows.length-1,j+1);show()};document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft')document.getElementById('prev').click();if(e.key==='ArrowRight')document.getElementById('next').click()});show();</script>'''
    (out/'observer.html').write_text(page.replace('ROWS', json.dumps(rows).replace('</', '<\\/')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--segmentation-run', default='automatic-components-v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['mps'], default='mps')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise ValueError('Fresh output required; preserve previous experiments')
    reference = read(args.reference)
    manifest_path = Path(reference['inputs']['manifest'])
    if sha(manifest_path) != reference['inputs']['manifest_sha256']:
        raise ValueError('Generated reconstruction metadata changed after reference fit')
    metadata = read(manifest_path)
    selection = Path(metadata['source']['selection_manifest_path'])
    case = next((p for p in selection.parents if (p/'manifest.json').is_file() and read(p/'manifest.json')['source']['sha256'] == reference['source_video_sha256']), None)
    if case is None:
        raise ValueError('No source case matches reference hash')
    tid = reference['object_id']
    segmentation = case/'segmentation'/args.segmentation_run
    config = read(segmentation/'annotations.json')
    if config.get('manual_seed_boxes_used') is not False:
        raise ValueError('This new-video experiment requires explicitly model-generated seeds')
    track = next(t for t in read(segmentation/'tracks.json')['tracks'] if t['track_id'] == tid)
    all_rows = read(case/'frames/index.json')['frames']
    rows = [all_rows[r['sample_index']-1] for r in track['frames']]
    anchor = next(j for j, r in enumerate(rows) if r['index'] == metadata['source']['sample_index'])
    images = [np.asarray(Image.open(case/r['file']).convert('RGB')) for r in rows]
    mask_paths = [segmentation/r['raw_mask_file'] for r in track['frames']]
    masks = [np.asarray(Image.open(p)) > 0 for p in mask_paths]
    if sha(case/rows[anchor]['file']) != sha(manifest_path.parent/metadata['source']['image_path']):
        # Selection PNG can be a lossless copy of decoded JPEG; compare actual pixels.
        if not np.array_equal(images[anchor], np.asarray(Image.open(manifest_path.parent/metadata['source']['image_path']).convert('RGB'))):
            raise ValueError('Reference source pixels disagree with case frame')
    if not np.array_equal(masks[anchor], np.asarray(Image.open(manifest_path.parent/metadata['source']['mask_path'])) > 0):
        raise ValueError('Reference mask differs from tracked source mask')
    mesh_path = manifest_path.parent/metadata['outputs']['glb_path']
    if sha(mesh_path) != reference['inputs']['mesh_sha256']:
        raise ValueError('Generated mesh hash changed')
    import trimesh
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    colors = np.asarray(mesh.visual.vertex_colors)[:, :3]
    k = np.asarray(reference['intrinsics']['matrix'])
    rotation = np.asarray(reference['candidate']['rotation'])
    translation = np.asarray(reference['candidate']['translation'])
    seeds = query_points(images[anchor], masks[anchor])
    canonical, hits = ray_mesh_points(seeds, vertices, faces, k, rotation, translation)
    train, held = split_ids(hits)
    out.mkdir(parents=True)
    start = time.monotonic()
    provenance = {'reference_file': str(args.reference.resolve()), 'reference_sha256': sha(args.reference),
                  'reference_status': reference['status'], 'model_metadata_file': str(manifest_path), 'model_metadata_sha256': sha(manifest_path),
                  'mesh_file': str(mesh_path), 'mesh_sha256': sha(mesh_path), 'source_video_sha256': reference['source_video_sha256'],
                  'original_source_sha256': read(case/'manifest.json')['source']['original_sha256'],
                  'annotations_sha256': sha(segmentation/'annotations.json'), 'track_index_sha256': sha(segmentation/'tracks.json'),
                  'manual_corners_used': False, 'manual_geometry_used': False, 'manual_pose_used': False,
                  'manual_seed_boxes_used': False, 'old_camera_or_battery_poses_read': False, 'depth_inference_used': False}
    base = {'schema_version': '0.1', 'status': 'running', 'object_id': tid, 'provenance': provenance,
            'reference_sample_index': rows[anchor]['index'], 'coordinates': '+X right,+Y down,+Z forward; native mesh @ R.T+t',
            'fixed_native_mesh_scale': 1., 'physical_scale_known': False, 'intrinsics': reference['intrinsics'],
            'queries': {'method': 'interior Shi-Tomasi corners plus adaptive grid; sorted y,x IDs', 'count': len(seeds),
                        'reference_mesh_ray_hits': int(hits.sum()), 'training_point_ids': train.tolist(), 'heldout_point_ids': held.tolist(),
                        'split': 'point_id modulo3; remainder0 held out, split before any fit'},
            'physical_pose_accuracy_measured': False, 'usable_physical_pose_recovered': False}
    write(out/'experiment.json', base)
    if len(train) < 12 or len(held) < 6:
        base.update(status='failed_insufficient_reference_ray_hit_queries', reason='No fallback seeds or manual intervention', wall_seconds=time.monotonic()-start)
        write(out/'experiment.json', base)
        print(json.dumps(base, indent=2))
        return
    try:
        raw_uv, model_visible, runtime = cotracker(images, seeds, anchor, args.device)
        evidence = support_masks(raw_uv, model_visible, masks, anchor)
        valid = evidence['valid_no_reacquisition'] & hits[None]
        audit_path = segmentation/'model-audit-v1/audit.json'
        audit = read(audit_path)
        verdicts = {(r['track_id'], r['sample_index']): r['verdict'] for r in audit['result']['observations']}
        quality = read(segmentation/'quality.json')
        withheld = quality['tracks'][tid]['withheld_samples']
        records, renders, support_rows = [], [], []
        for j, row in enumerate(rows):
            sample = row['index']
            present = train[valid[j, train]]
            checks = held[valid[j, held]]
            rec = {'sample_index': sample, 'timestamp_s': row['timestamp_s'], 'status': 'unsupported', 'reasons': [],
                   'source_frame_sha256': sha(case/row['file']), 'sam_mask_sha256': sha(mask_paths[j]),
                   'training_visible_point_ids': present.tolist(), 'heldout_visible_point_ids': checks.tolist(),
                   'mask_model_audit_verdict': verdicts.get((tid, sample), 'unassessed'), 'usable_pose_exported': False}
            fit = solve_pose(canonical[present], raw_uv[j, present], k, rotation, translation) if len(present) >= 6 else {'success': False, 'reason': 'fewer_than_6_valid_training_correspondences'}
            render = None
            if fit['success']:
                r, t = fit['rotation'], fit['translation']
                errors = np.linalg.norm(project(canonical[checks], k, r, t)-raw_uv[j, checks], axis=1)
                render = render_mesh(vertices, faces, k, r, t, row['width'], row['height'], vertex_colors=colors)
                intersection = int((render['mask'] & masks[j]).sum())
                iou = intersection/max(1, int((render['mask'] | masks[j]).sum()))
                coverage = intersection/max(1, int(masks[j].sum()))
                rec.update(candidate_rotation=r.tolist(), candidate_translation=t.tolist(),
                           training_inlier_point_ids=present[fit['inlier_indices']].tolist(),
                           training_residual_px=[float(x) if np.isfinite(x) else None for x in fit['residual_px']],
                           heldout_residual_px=errors.tolist(), heldout_median_px=float(np.median(errors)) if len(errors) else None,
                           heldout_p90_px=float(np.percentile(errors, 90)) if len(errors) else None,
                           raw_full_projection_iou_with_sam=iou, source_mask_coverage=coverage)
                rec['reasons'] = pose_gate(len(present), len(fit['inlier_indices']), errors, iou, coverage, rec['mask_model_audit_verdict'])
                if sample in withheld:
                    rec['reasons'].append('source_mask_failed_geometric_gate')
                if j == anchor:
                    rec['status'] = 'reference_candidate_unverified_geometry_and_pose'
                elif not rec['reasons']:
                    rec['status'] = 'supported_conditional_motion_candidate'
                    rec['conditional_pose_exported'] = True
                else:
                    rec['status'] = 'unsupported_candidate_projection_retained'
            else:
                rec['reasons'] = [fit['reason']]
            records.append(rec)
            renders.append(render)
            support_rows.append({'sample_index': sample, 'timestamp_s': row['timestamp_s'], 'points': [
                {'point_id': p, 'raw_uv': raw_uv[j, p].tolist() if np.isfinite(raw_uv[j, p]).all() else None,
                 'model_visible': bool(model_visible[j, p]), 'inside_image': bool(evidence['image_bounds'][j, p]),
                 'inside_raw_sam': bool(evidence['sam_membership'][j, p]), 'instantaneous_support': bool(evidence['instantaneous_support'][j, p]),
                 'valid_without_reacquisition': bool(valid[j, p]), 'reference_mesh_ray_hit': bool(hits[p]),
                 'split': 'heldout' if p % 3 == 0 else 'training'} for p in range(len(seeds))]})
        np.savez_compressed(out/'tracks.npz', raw_model_uv=raw_uv, model_visibility=model_visible, query_uv=seeds,
                            canonical_xyz=canonical, reference_ray_hits=hits, valid_without_reacquisition=valid,
                            sam_membership=evidence['sam_membership'], image_bounds=evidence['image_bounds'],
                            point_ids=np.arange(len(seeds)), training_point_ids=train, heldout_point_ids=held,
                            sample_indices=np.array([r['index'] for r in rows]), timestamps_s=np.array([r['timestamp_s'] for r in rows]))
        write(out/'point-support.json', {'frames': support_rows})
        make_review(out, images, records, masks, renders, raw_uv, valid, set(held.tolist()))
        supported = [r for r in records if r['status'] == 'supported_conditional_motion_candidate']
        unsupported = [r for j, r in enumerate(records) if j != anchor and r['status'] != 'supported_conditional_motion_candidate']
        base.update(status='complete_conditional_tracking_experiment', runtime=runtime, frames=records,
                    audit_sha256=sha(audit_path), frame_count=len(records), supported_nonreference_frames=len(supported),
                    unsupported_nonreference_frames=len(unsupported), reference_frames=1,
                    minimum_valid_points=int(valid.sum(1).min()), maximum_valid_points=int(valid.sum(1).max()),
                    supported_sample_indices=[r['sample_index'] for r in supported], unsupported_sample_indices=[r['sample_index'] for r in unsupported],
                    reference_pose_independently_verified=False, source_shape_independently_verified=False,
                    policy='Once model visibility, image bounds or SAM membership fails, that point stays lost in that time direction. No interpolation/reidentification.',
                    evaluation='PnP training point IDs disjoint from heldout IDs. Heldout CoTracker observations are correlated learned 2D checks conditional on the same reference mesh/pose/camera, not physical 6D truth.',
                    raw_projections_clipped_to_sam=False, wall_seconds=round(time.monotonic()-start, 3),
                    limitations=['Unknown metric scale, uncalibrated 60deg assumed focal length, unverified reference orientation and generated hidden surfaces.',
                                 'Partial occlusion, similar texture and small image support can create plausible wrong correspondences.',
                                 'Newly visible surfaces have no reference 3D correspondences. No contact, seating, joint or assembled-scene recovery.',
                                 'Masks and semantic audits are model proposals; no human-corrected points/corners/masks enter this experiment.'])
        write(out/'experiment.json', base)
        print(json.dumps({k:base[k] for k in ['status','frame_count','supported_nonreference_frames','unsupported_nonreference_frames','supported_sample_indices','unsupported_sample_indices','runtime']}, indent=2))
    except Exception as exc:
        base.update(status='failed', error=str(exc), wall_seconds=round(time.monotonic()-start, 3))
        write(out/'experiment.json', base)
        raise


if __name__ == '__main__':
    main()
