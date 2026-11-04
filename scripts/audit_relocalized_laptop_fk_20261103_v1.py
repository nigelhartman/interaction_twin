#!/usr/bin/env python3
"""Dense browser-contract versus independent NumPy/SciPy laptop FK check."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import laptop_hinge_relocalization_v2 as L
import laptop_hinge_omission_recovery_v1 as R
import sam3d_mechanisms as M
import object_program as O
from native_appearance import load_native_glb


def build():
    folder = ROOT / L.OUTPUT / 'local-fit-v1'
    result = R.read(folder / 'result.json')
    scene_path = ROOT / result['scene']['manifest']['path']
    scene = R.read(scene_path)
    O.validate(R.read(ROOT / result['descriptive_export']['object_program']['path']), root=ROOT)
    code = r'''const fs=require('node:fs');const C=require('./showcase/aligned-scene-contract.js');const s=JSON.parse(fs.readFileSync(process.argv[1]));C.validateScene(s,s.source.sha256);const ts=[...new Set([...Array.from({length:1001},(_,i)=>s.source.duration_s*i/1000),...s.components[0].poses.map(r=>r.timestamp_s)])].sort((a,b)=>a-b);process.stdout.write(JSON.stringify(ts.map(t=>({t,...C.sceneAt(s,t,true)}))));'''
    raw = subprocess.run(['node', '-e', code, str(scene_path)], cwd=ROOT, stdin=subprocess.DEVNULL,
                         capture_output=True, timeout=30, check=True).stdout
    frames = json.loads(raw)
    components = {c['id']: c for c in scene['components']}
    joint = scene['mechanics']['joints'][0]
    parent = components[joint['parent_id']]
    known, unknown, matrix_error, anchor_error = 0, 0, [], []

    def interpolate(rows, time, key):
        exact = next((r for r in rows if abs(r['timestamp_s'] - time) <= 1e-6), None)
        if exact:
            return exact.get(key) if exact['status'] == 'estimated' else None
        after = next((i for i, r in enumerate(rows) if r['timestamp_s'] > time), len(rows))
        if after == 0 or after == len(rows):
            return None
        a, b = rows[after - 1], rows[after]
        gap = b['timestamp_s'] - a['timestamp_s']
        if a['status'] != 'estimated' or b['status'] != 'estimated' or gap > min(scene['timing']['max_interpolation_gap_s'], scene['timing']['max_estimated_gap_s']):
            return None
        f = (time - a['timestamp_s']) / gap
        if key == 'q_rad':
            return a[key] * (1 - f) + b[key] * f
        A, B = np.asarray(a[key]), np.asarray(b[key])
        sa = np.linalg.norm(A[:3, 0]); sb = np.linalg.norm(B[:3, 0])
        rot = Slerp([0, 1], Rotation.from_matrix([A[:3, :3] / sa, B[:3, :3] / sb]))([f]).as_matrix()[0]
        value = np.eye(4)
        value[:3, :3] = rot * (sa * (1 - f) + sb * f)
        value[:3, 3] = A[:3, 3] * (1 - f) + B[:3, 3] * f
        return value

    for frame in frames:
        t = frame['t']
        poses = {p['id']: p for p in frame['parts']}
        p = interpolate(parent['poses'], t, 'camera_from_native_mesh')
        q = interpolate(joint['samples'], t, 'q_rad')
        if p is None or q is None:
            R.require(all(x['camera_from_native_mesh'] is None for x in poses.values()), 'Connected unknown windows leaked geometry')
            unknown += 1
            continue
        expected = M.forward(p, joint['parent_from_joint'], joint['axis'], q, joint['joint_from_child_zero'])
        actual = np.asarray(poses[joint['child_id']]['camera_from_native_mesh'])
        matrix_error.append(float(np.max(np.abs(actual - expected))))
        hp = p @ np.r_[joint['parent_anchor_native'], 1]
        hc = actual @ np.r_[joint['child_anchor_native'], 1]
        anchor_error.append(float(np.linalg.norm(hp - hc)))
        known += 1
    R.require(known > 100 and max(matrix_error) < 1e-8 and max(anchor_error) < 1e-8, 'Dense FK matrix/anchor verification failed')
    for component_id, field in [(joint['parent_id'], 'parent'), (joint['child_id'], 'child')]:
        mesh = load_native_glb(ROOT / components[component_id]['asset']['path'])
        R.require(np.allclose(mesh.vertices[joint[field + '_anchor_vertex_id']], joint[field + '_anchor_native'], atol=1e-12), 'Anchor is not the exact unchanged native vertex')
    output = folder / 'dense-fk-frames.json'
    R.write_new(output, frames)
    summary = {'format': 'interaction-twin-private-laptop-dense-fk-audit', 'version': 1,
               'scene': R.ref(scene_path, ROOT), 'browser_contract': R.ref(ROOT / 'showcase/aligned-scene-contract.js', ROOT),
               'frames': R.ref(output, ROOT), 'dense_clock_count': len(frames), 'known_connected_count': known,
               'withheld_connected_count': unknown, 'maximum_matrix_error': max(matrix_error),
               'maximum_anchor_gap_scene_units': max(anchor_error), 'actual_native_vertex_anchors_verified': True,
               'physical_accuracy_verified': False, 'angle_interpretation': joint['angle_quality']['angle_interpretation'],
               'new_api_calls': 0, 'new_gpu_calls': 0, 'published': False}
    R.write_new(folder / 'dense-fk-audit.json', summary)
    return summary


if __name__ == '__main__':
    print(json.dumps(build()), flush=True)
