#!/usr/bin/env python3
"""Private source-bound CPU render and dense-FK checks for a new research fit."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import diagnose_laptop_orientation_20261103_v1 as F
import native_orientation_diagnostic_v1 as D
import native_scene_audit as A
import sam3d_mechanisms as M
import object_program as O
from native_appearance import load_native_glb


def build():
    start = time.monotonic(); folder = ROOT / F.OUTPUT / 'appearance-refit-v1'
    result = json.loads((folder / 'result.json').read_text())
    D.require(result.get('joint_native_anchors_recomputed') is True and result.get('fit_result', {}).get('joints') == 1, 'A complete new native-anchor hinge fit is required; unknown stays unknown')
    scene_path = A.checked(result['fit_result']['manifest'], ROOT); scene = json.loads(scene_path.read_text()); O.validate(json.loads(A.checked(result['descriptive_export']['object_program'], ROOT).read_text()), root=ROOT)
    D.require(scene['original_program']['sha256'] == 'ad13657bba9544af8a4b0b11611ef38fe9d5da1520303354d87906003aca25ce', 'Exact laptop source program required')
    output = folder / 'verification-v1'; D.require(not output.exists(), 'Preserve prior candidate verification'); output.mkdir()
    code = r'''const fs=require('node:fs');const C=require('./showcase/aligned-scene-contract.js');const s=JSON.parse(fs.readFileSync(process.argv[1]));C.validateScene(s,s.source.sha256);const ts=[...new Set([...Array.from({length:1001},(_,i)=>s.source.duration_s*i/1000),...s.components[0].poses.map(r=>r.timestamp_s)])].sort((a,b)=>a-b);process.stdout.write(JSON.stringify(ts.map(t=>({t,...C.sceneAt(s,t,true)}))));'''
    raw = subprocess.run(['node', '-e', code, str(scene_path)], cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, timeout=30, check=True).stdout
    frames = json.loads(raw); parts = {c['id']: c for c in scene['components']}; joint = scene['mechanics']['joints'][0]; parent = parts[joint['parent_id']]

    def interpolate(rows, clock, key):
        exact = next((r for r in rows if abs(r['timestamp_s'] - clock) <= 1e-6), None)
        if exact is not None: return exact.get(key) if exact['status'] == 'estimated' else None
        after = next((i for i, row in enumerate(rows) if row['timestamp_s'] > clock), len(rows))
        if after in (0, len(rows)): return None
        a, b = rows[after - 1], rows[after]; gap = b['timestamp_s'] - a['timestamp_s']
        if a['status'] != 'estimated' or b['status'] != 'estimated' or gap > min(scene['timing']['max_interpolation_gap_s'], scene['timing']['max_estimated_gap_s']): return None
        fraction = (clock - a['timestamp_s']) / gap
        if key == 'q_rad': return a[key] * (1 - fraction) + b[key] * fraction
        ra, ta, sa = M.rotation_pose(a[key]); rb, tb, sb = M.rotation_pose(b[key]); value = np.eye(4)
        value[:3, :3] = Slerp([0, 1], Rotation.from_matrix([ra, rb]))([fraction]).as_matrix()[0] * (sa * (1 - fraction) + sb * fraction)
        value[:3, 3] = ta * (1 - fraction) + tb * fraction
        return value

    matrix_errors = []; anchor_gaps = []; known = unknown = 0
    for frame in frames:
        poses = {p['id']: p for p in frame['parts']}; p = interpolate(parent['poses'], frame['t'], 'camera_from_native_mesh'); q = interpolate(joint['samples'], frame['t'], 'q_rad')
        if p is None or q is None:
            D.require(all(row['camera_from_native_mesh'] is None for row in poses.values()), 'Unknown connected interval leaked a mesh pose'); unknown += 1; continue
        actual = np.asarray(poses[joint['child_id']]['camera_from_native_mesh']); expected = M.forward(p, joint['parent_from_joint'], joint['axis'], q, joint['joint_from_child_zero'])
        matrix_errors.append(float(np.max(np.abs(actual - expected)))); anchor_gaps.append(float(np.linalg.norm(p @ np.r_[joint['parent_anchor_native'], 1] - actual @ np.r_[joint['child_anchor_native'], 1]))); known += 1
    D.require(known > 100 and max(matrix_errors) < 1e-8 and max(anchor_gaps) < 1e-8, 'Dense FK failed')
    meshes = {c['id']: load_native_glb(A.checked(c['asset'], ROOT)) for c in scene['components']}
    for component_id, field in [(joint['parent_id'], 'parent'), (joint['child_id'], 'child')]:
        D.require(np.allclose(meshes[component_id].vertices[joint[field + '_anchor_vertex_id']], joint[field + '_anchor_native'], atol=1e-12), 'Fitted anchor is not an exact unchanged native vertex')
    F.write_new(output / 'dense-fk-frames.json', frames)
    dense = {'clock_count': len(frames), 'known_connected_count': known, 'withheld_connected_count': unknown,
             'maximum_matrix_error': max(matrix_errors), 'maximum_native_anchor_gap': max(anchor_gaps), 'exact_native_vertex_anchors': True,
             'angle_interpretation': joint['angle_quality']['angle_interpretation'], 'frames_ref': F.ref(output / 'dense-fk-frames.json')}
    # Same exact old render protocol; ONLY development metrics may be compared.
    audit_result = A.build_audit(scene_path, output / 'native-audit', root=ROOT, max_frames=12, max_dimension=384)
    baseline = json.loads((ROOT / 'data/mechanism-hinge-relocalizations/laptop-20261103-v2/local-fit-v1/visual-audit-v1/audit.json').read_text())
    audit = json.loads((ROOT / audit_result['manifest']['path']).read_text())
    def development(doc): return {row['sample_id']: row for row in doc['samples'] if row['role'] == 'development'}
    before, after = development(baseline), development(audit); D.require(set(before) == set(after), 'Development comparison sample basis changed')
    diagnostic = json.loads(A.checked(result['diagnostic_ref'], ROOT).read_text()); frame_appearance = []
    width, height = diagnostic['raster']['width'], diagnostic['raster']['height']; k = np.asarray(diagnostic['raster']['K'])
    for component in diagnostic['components']:
        part = parts[component['id']]; mesh = meshes[part['id']]
        for frame in component['frame_results']:
            row = next(r for r in part['poses'] if r['source_sample_index'] == frame['source_sample_index'])
            if row['status'] != 'estimated':
                frame_appearance.append({'component_id': part['id'], 'source_sample_index': frame['source_sample_index'], 'status': 'withheld_after_joint_refit'}); continue
            with Image.open(A.checked(frame['source_rgb_ref'], ROOT)) as im: source = cv2.resize(np.asarray(im.convert('RGB')), (width, height), interpolation=cv2.INTER_AREA)
            with Image.open(A.checked(frame['source_mask_ref'], ROOT)) as im: observed = cv2.resize((np.asarray(im) > 0).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
            rendered = F.render(mesh, np.asarray(row['camera_from_native_mesh']), k, width, height)
            metrics = D.appearance_metrics(source, observed, rendered['rgb'], rendered['mask'])
            proof = F.proof(source, observed, [('New source-hinge refit', rendered['rgb'], rendered['mask'])], f'{part["id"]} | source {frame["source_sample_index"]} | appearance-initialized native-anchor FK, unverified angle', output / f'{part["id"]}-{frame["source_sample_index"]:06d}-proof.png')
            frame_appearance.append({'component_id': part['id'], 'source_sample_index': frame['source_sample_index'], 'before_saved_pose_metrics': frame['original_saved_pose_metrics'], 'after_joint_refit_metrics': metrics, 'proof': proof})
    document = {'format': 'interaction-twin-appearance-hinge-research-verification', 'version': 1, 'scene_ref': result['fit_result']['manifest'], 'result_ref': F.ref(folder / 'result.json'),
                'dense_fk': dense, 'native_audit_ref': audit_result['manifest'], 'development_mask_comparison': [{'sample_id': key, 'before': before[key]['metrics'], 'after': after[key]['metrics']} for key in sorted(before)],
                'frame_appearance': frame_appearance, 'all_final_test_values_excluded_from_comparison': True, 'evaluation_mode': 'retrospective_diagnostic_not_physical_validation',
                'new_api_calls': 0, 'new_gpu_calls': 0, 'geometry_modified': False, 'published': False, 'physical_accuracy_verified': False, 'wall_seconds': round(time.monotonic() - start, 3)}
    F.write_new(output / 'verification.json', document)
    return {'verification_ref': F.ref(output / 'verification.json'), 'dense_fk': dense, 'development_summary': audit_result['development_summary'], 'new_api_calls': 0, 'new_gpu_calls': 0}


if __name__ == '__main__':
    print(json.dumps(build()), flush=True)
