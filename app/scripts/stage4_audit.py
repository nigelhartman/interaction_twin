#!/usr/bin/env python3
"""Audit a saved battery alignment run against its source and frozen evidence."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stage4_align import project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='battery-pilot-v2')
    args = parser.parse_args()
    case = ROOT / 'data/camera-battery-withdrawal-15fps'
    out = case / 'alignment' / args.run
    report = json.loads((out / 'alignment.json').read_text())
    parent = case / 'geometry/local-battery-v3'
    pack = json.loads((parent / 'experiment.json').read_text())
    tracks = np.load(parent / 'motion-cotracker/tracks.npz')
    poses = np.load(out / 'poses.npz')
    mesh = trimesh.load(case / 'geometry/battery-fitted-v1/battery.ply', force='mesh')
    train, heldout = poses['train_point_ids'], poses['heldout_point_ids']
    assert not set(train) & set(heldout)
    assert len(report['frames']) == len(pack['samples']) == len(poses['rotations'])
    assert poses['asset_scale'].shape == ()
    assert float(poses['asset_scale']) == report['asset_scale']
    assert report['source_sha256'] == pack['source_sha256']
    assert hashlib.sha256((parent / 'motion-cotracker/tracks.npz').read_bytes()).hexdigest() == report['tracks_sha256']
    k = np.array(report['camera']['intrinsics'])
    for j, (row, record) in enumerate(zip(pack['samples'], report['frames'])):
        i = row['sample_index']
        assert row['timestamp_s'] == record['timestamp_s'] == poses['timestamps_s'][j]
        assert i == record['sample_index'] == poses['sample_indices'][j]
        assert record['asset_scale'] == report['asset_scale']
        r, t = poses['rotations'][j], poses['translations'][j]
        assert np.isfinite(r).all() and np.isfinite(t).all()
        assert np.allclose(r.T @ r, np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(r), 1)
        assert ((mesh.vertices * report['asset_scale']) @ r.T + t)[:, 2].min() > 0
        assert np.allclose(record['rotation'], r) and np.allclose(record['translation'], t)
        path = case / 'occlusion/hands-v1/foreground_masks' / f'{i:06d}.png'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == report['occlusion']['foreground_masks_sha256'][f'{i:06d}']
        hand = np.array(Image.open(path)) > 0
        full = np.array(Image.open(out / 'full_masks' / f'{i:06d}.png')) > 0
        visible = np.array(Image.open(out / 'visible_masks' / f'{i:06d}.png')) > 0
        depth = np.load(out / 'depth' / f'{i:06d}.npy')
        assert full.shape == visible.shape == depth.shape == (540, 960)
        assert np.array_equal(visible, full & ~hand)
        assert np.array_equal(full, np.isfinite(depth))
        checks = heldout[tracks['valid'][j, heldout]]
        errors = np.linalg.norm(project(poses['canonical_points'][checks], k, r, t, report['asset_scale']) - tracks['uv'][j, checks], axis=1)
        assert np.isclose(np.median(errors), record['heldout_reprojection_px']['median'])
        assert np.isclose(np.percentile(errors, 90), record['heldout_reprojection_px']['p90'])
        if not tracks['xyz_usable'][j].any():
            assert record['status'] == 'unsupported'
    result = {'status': 'pass', 'run': args.run, 'samples_checked': len(report['frames']),
              'checks': ['source timestamps', 'constant scalar scale', 'disjoint fit/check IDs',
                         'proper positive-depth poses', 'foreground-mask hashes', 'unclipped projections and separate occlusion',
                         'depth/mask agreement', 'recomputed held-out errors', 'rejected source-depth support'],
              'verified_physical_pose_accuracy': False}
    (out / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
