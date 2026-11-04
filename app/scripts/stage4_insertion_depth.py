#!/usr/bin/env python3
"""Probe insertion depth with real MoGe-2; preserve the 2D-only baseline.

Depth is calibrated only to the current camera's nominal proxy gauge. It is
independent monocular evidence, never measured metric geometry or contact.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignment_render import render_mesh
from geometry import lift_pixels, rigid_fit
from alignment_fit import solve_pose


def read(p): return json.loads(p.read_text())
def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')


def joint_depth_probe(base, out):
    """Fixed-scale joint 2D/Z fitting on training IDs; held IDs only assess it."""
    analysis = read(out / 'analysis.json')
    baseline = read(base / 'battery-poses-v1.json')
    tracklet = next(t for t in baseline['tracklets'] if t['observation_id'] == 'battery_track_00007')
    k = np.array(baseline['camera']['intrinsics']); s = baseline['asset_scale']
    tracks = np.load(base / 'battery-inference-v1/battery_track_00007/tracks.npz')
    pack = np.load(base / 'battery-inference-v1/battery_track_00007/canonical.npz')
    canonical = pack['points']; train = pack['train_ids']; held = pack['heldout_ids']
    records = []
    for evidence in analysis['frames']:
        i = evidence['sample_index']; row = tracklet['frames'][i - 1]
        raw = np.load(out / f'{i:06d}.npz')
        depth = cv2.resize(raw['depth'], (960, 540), interpolation=cv2.INTER_LINEAR) * evidence['body_depth_gauge_factor']
        uv = tracks['raw_model_uv'][i - 1]
        xyz = lift_pixels(uv, depth, k)
        available = tracks['valid'][i - 1] & np.isfinite(xyz).all(axis=1)
        ids = train[available[train]]; checks = held[available[held]]
        if len(ids) < 10 or 'rotation' not in row:
            records.append({'sample_index': i, 'status': 'insufficient_support', 'pose_promoted': False}); continue
        initial_r = np.array(row['rotation']); initial_t = np.array(row['translation'])
        fit2d = solve_pose(canonical[ids] * s, uv[ids], k, initial_r, initial_t)
        if not fit2d['success']:
            records.append({'sample_index': i, 'status': 'no_2D_consensus', 'pose_promoted': False}); continue
        ids = ids[fit2d['inlier_indices']]
        sigma_z = max(.25 * s, evidence['depth_plane_residual_median_diameters'] * s * 1.4826)
        def residual(p):
            r = cv2.Rodrigues(p[:3])[0]; camera_xyz = canonical[ids] * s @ r.T + p[3:]
            projected = camera_xyz @ k.T; projected = projected[:, :2] / projected[:, 2:]
            return np.r_[((projected - uv[ids]) / 2).ravel(), (camera_xyz[:, 2] - xyz[ids, 2]) / sigma_z]
        seeds = [np.r_[cv2.Rodrigues(initial_r)[0].ravel(), initial_t]]
        direct = evidence['rigid_depth_fit']
        if 'rotation' in direct:
            seeds.append(np.r_[cv2.Rodrigues(np.array(direct['rotation']))[0].ravel(), direct['translation']])
        proposals = [least_squares(residual, seed, loss='soft_l1', f_scale=1, max_nfev=100) for seed in seeds]
        selected = min(proposals, key=lambda proposal: proposal.cost)
        r = cv2.Rodrigues(selected.x[:3])[0]; t = selected.x[3:]
        predicted = canonical[checks] * s @ r.T + t
        projected = predicted @ k.T; projected = projected[:, :2] / projected[:, 2:]
        errors = np.linalg.norm(projected - uv[checks], axis=1)
        zerror = np.abs(predicted[:, 2] - xyz[checks, 2]) / s
        med = float(np.median(errors)) if len(errors) else None
        p90 = float(np.percentile(errors, 90)) if len(errors) else None
        depthmed = float(np.median(zerror)) if len(zerror) else None
        passed = len(checks) >= 5 and med <= 3 and p90 <= 8 and depthmed <= .2
        records.append({'sample_index': i, 'timestamp_s': row['timestamp_s'], 'status': 'diagnostic_only', 'training_consensus_count': len(ids), 'heldout_count': len(checks), 'heldout_median_reprojection_px': med, 'heldout_p90_reprojection_px': p90, 'heldout_median_depth_error_diameters': depthmed, 'depth_sigma_diameters': sigma_z / s, 'joint_gate_pass': bool(passed), 'rotation': r.tolist(), 'translation': t.tolist(), 'asset_scale': s, 'pose_promoted': False})
    write(out / 'joint-2d-depth.json', {'method': 'Robust training-ID 2D reprojection plus visible Z-depth; two optimizer seeds ranked by training cost; disjoint held-out IDs only assess outputs.', 'status': 'diagnostic_only_no_replacement', 'passing_frame_count': sum(r.get('joint_gate_pass', False) for r in records), 'limits': ['Depth derives from the same image and an approximate camera-body gauge, not measured geometry.', 'Held-out CoTracker point IDs are an internal consistency check.', 'Passing a single frame does not establish a consistent new trajectory or resolve identity.'], 'frames': records})


def main():
    import torch
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable; no CPU fallback')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1': raise RuntimeError('Operator fallback disabled')
    os.environ['HF_HOME'] = str(ROOT / 'models/hf-cache')
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    from moge.model.v2 import MoGeModel
    base = ROOT / 'data/camera-battery-full-scene'
    out = base / 'battery-depth-refinement-v1'
    out.mkdir(exist_ok=True)
    case = ROOT / 'data/camera-battery-insertion-10fps'
    rows = read(case / 'frames/index.json')['frames']
    selected = [6, 8, 10, 12, 13, 14, 16]
    checkpoint = ROOT / 'models/moge-2-vits-normal/model.pt'
    provenance = read(checkpoint.parent / 'provenance.json')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != provenance['checkpoint_sha256']: raise ValueError('Checkpoint changed')
    model = MoGeModel.from_pretrained(str(checkpoint)).to('mps').eval()
    started = time.monotonic()
    inference = []
    for i in selected:
        row = rows[i - 1]
        dest = out / f'{i:06d}.npz'
        if not dest.exists():
            image = Image.open(case / row['file']).convert('RGB').resize((640, 360), Image.Resampling.LANCZOS)
            tensor = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float().to('mps') / 255
            with torch.inference_mode():
                result = model.infer(tensor, num_tokens=1200, use_fp16=False, apply_mask=True)
                torch.mps.synchronize()
                np.savez_compressed(dest, depth=result['depth'].cpu().numpy().astype(np.float32), points=result['points'].cpu().numpy().astype(np.float32), intrinsics_normalized=result['intrinsics'].cpu().numpy().astype(np.float32), valid=result['mask'].cpu().numpy().astype(bool))
        inference.append({'source_sample_index': i, 'timestamp_s': row['timestamp_s'], 'file': dest.name, 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest(), 'image_sha256': hashlib.sha256((case / row['file']).read_bytes()).hexdigest()})
        print(f'MoGe insertion sample{i} done', flush=True)
    del model
    torch.mps.empty_cache()
    write(out / 'inference.json', {'status': 'complete', 'provider': 'MoGe-2 Small', 'model_provenance': provenance, 'device': 'mps', 'cpu_operator_fallback_enabled': False, 'input_hw': [360, 640], 'num_tokens': 1200, 'inference_precision': 'float32', 'source_case': case.name, 'frames': inference, 'inference_total_s': time.monotonic() - started})
    battery = read(base / 'battery-poses-v1.json')
    camera = read(base / 'camera-poses-v1.json')
    camera_mesh = trimesh.load(base / 'geometry/camera-v1/camera.ply', force='mesh')
    k = np.array(battery['camera']['intrinsics'])
    s = battery['asset_scale']
    track = next(t for t in battery['tracklets'] if t['observation_id'] == 'battery_track_00007')
    tracks = np.load(base / 'battery-inference-v1/battery_track_00007/tracks.npz')
    canonical_pack = np.load(base / 'battery-inference-v1/battery_track_00007/canonical.npz')
    canonical = canonical_pack['points']
    oldcase = ROOT / 'data/camera-battery-action-5fps'
    oldrows = read(oldcase / 'frames/index.json')['frames']
    records = []
    for i in selected:
        row = rows[i - 1]; raw = np.load(out / f'{i:06d}.npz')
        depth = cv2.resize(raw['depth'], (960, 540), interpolation=cv2.INTER_LINEAR)
        valid = cv2.resize(raw['valid'].astype(np.uint8), (960, 540), interpolation=cv2.INTER_NEAREST) > 0
        mask = np.array(Image.open(case / 'segmentation/battery-v1/raw_masks/in_first' / f'{i:06d}.png')) > 0
        mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        cp = min(camera['frames'], key=lambda r: abs(r['timestamp_s'] - row['timestamp_s']))
        rendered = render_mesh(np.asarray(camera_mesh.vertices), np.asarray(camera_mesh.faces), k, np.array(cp['rotation']), np.array(cp['translation']), 960, 540, scale=s)
        oldrow = min(oldrows, key=lambda r: abs(r['timestamp_s'] - row['timestamp_s']))
        body_mask = np.array(Image.open(oldcase / 'segmentation/components-v2/resolved_masks/camera' / f"{oldrow['index']:06d}.png")) > 0
        body_pixels = body_mask & rendered['mask'] & valid & ~cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        body_pixels &= np.isfinite(depth) & (depth > 0)
        ratio_values = rendered['depth'][body_pixels] / depth[body_pixels]
        factor = float(np.median(ratio_values)) if len(ratio_values) else 1.
        corrected = depth * factor
        y, x = np.where(mask & valid & np.isfinite(corrected) & (corrected > 0)); uv = np.c_[x, y]
        center = np.median(uv, axis=0); _, _, basis = np.linalg.svd(uv - center, full_matrices=False); axis = basis[0]
        if axis[0] < 0: axis *= -1
        a = (uv - center) @ axis
        across = (uv - center) @ np.array([-axis[1], axis[0]])
        z = corrected[y, x]
        design = np.c_[a, across, np.ones(len(a))]
        start = np.linalg.lstsq(design, z, rcond=None)[0]
        fit = least_squares(lambda p: design @ p - z, start, loss='soft_l1', f_scale=.005, max_nfev=60)
        span = np.percentile(a, 90) - np.percentile(a, 10)
        gradient = float(fit.x[0] * span)
        residual = np.abs(design @ fit.x - z)
        current = track['frames'][i - 1]
        lifted = lift_pixels(tracks['raw_model_uv'][i - 1], corrected, k)
        usable = tracks['valid'][i - 1] & np.isfinite(canonical).all(axis=1) & np.isfinite(lifted).all(axis=1)
        rigid = {'status': 'insufficient_correspondences'}
        if usable.sum() >= 6:
            try:
                transform, errors = rigid_fit(canonical[usable] * s, lifted[usable])
                relative = float(np.median(errors) / s)
                predicted = canonical[usable] * s @ transform[:3, :3].T + transform[:3, 3]
                projected = predicted @ k.T; projected = projected[:, :2] / projected[:, 2:]
                reproject = np.linalg.norm(projected - tracks['raw_model_uv'][i - 1, usable], axis=1)
                rigid = {'status': 'diagnostic_only', 'rotation': transform[:3, :3].tolist(), 'translation': transform[:3, 3].tolist(), 'median_fixed_scale_3D_residual_diameters': relative, 'median_reprojection_px': float(np.median(reproject)), 'p90_reprojection_px': float(np.percentile(reproject, 90)), 'pose_promoted': False}
            except ValueError as exc: rigid = {'status': 'degenerate', 'reason': str(exc)}
        rec = {'sample_index': i, 'timestamp_s': row['timestamp_s'], 'source_baseline_status': current['status'], 'body_depth_gauge_factor': factor, 'body_pixels': int(body_pixels.sum()), 'body_mask_source_delta_s': abs(oldrow['timestamp_s'] - row['timestamp_s']), 'body_depth_factor_p10_p90': np.percentile(ratio_values, [10, 90]).tolist() if len(ratio_values) else None, 'battery_depth_median_nominal': float(np.median(z)), 'battery_visible_pixels': len(z), 'positive_imageX_axis': axis.tolist(), 'axial_depth_change_over_visible_p10_p90_span': gradient, 'axial_depth_change_in_battery_diameters': gradient / s, 'axial_depth_sign': 'farther_at_positive_imageX_end' if gradient > 0 else 'nearer_at_positive_imageX_end', 'depth_plane_residual_median_diameters': float(np.median(residual) / s), 'lifted_track_count': int(usable.sum()), 'rigid_depth_fit': rigid}
        records.append(rec)
        print(json.dumps({k: rec[k] for k in ['timestamp_s', 'axial_depth_sign', 'axial_depth_change_in_battery_diameters', 'depth_plane_residual_median_diameters']}), flush=True)
    accepted = [r for r in records if r['rigid_depth_fit'].get('median_fixed_scale_3D_residual_diameters', 99) <= .2 and r['rigid_depth_fit'].get('median_reprojection_px', 99) <= 3 and r['rigid_depth_fit'].get('p90_reprojection_px', 99) <= 8]
    write(out / 'analysis.json', {'status': 'diagnostic_depth_evidence_only', 'camera_pose_sha256': hashlib.sha256((base / 'camera-poses-v1.json').read_bytes()).hexdigest(), 'baseline_battery_pose_sha256': hashlib.sha256((base / 'battery-poses-v1.json').read_bytes()).hexdigest(), 'shared_intrinsics': k.tolist(), 'asset_scale_constant': s, 'physical_scale_verified': False, 'new_pose_exported': False, 'decision': 'No meaningful depth-refined trajectory recovered; retain the baseline as an uncertain 2D-supported candidate.', 'passing_direct_depth_fit_count': len(accepted), 'direct_depth_quality_gate': {'maximum_median_3D_residual_diameters': .2, 'maximum_median_reprojection_px': 3, 'maximum_p90_reprojection_px': 8}, 'depth_axis_sign_summary': {'negative_Z_at_positive_imageX_count': sum(r['axial_depth_change_in_battery_diameters'] < 0 for r in records), 'assessed_frames': len(records), 'interpretation': 'Suggests alternate axial tilt, but sign is inconsistent and visible-surface noise/scale conflict prevent promotion.'}, 'limitations': ['Monocular depth is independent per frame and not measured geometry.', 'Camera proxy visible depth sets a nominal gauge, not independent physical calibration.', 'SAM body-mask overlap and hidden camera geometry can bias the gauge.', 'The visible cylinder is small, partly occluded and may not yield a reliable axial depth gradient.', 'Fixed-scale rigid fit residuals/reprojection must support a meaningful refinement before any pose is promoted.'], 'frames': records})
    joint_depth_probe(base, out)


if __name__ == '__main__': main()
