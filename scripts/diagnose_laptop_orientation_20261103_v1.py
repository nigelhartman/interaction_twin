#!/usr/bin/env python3
"""Local-only, immutable appearance diagnostic; never edits the laptop scene."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import native_orientation_diagnostic_v1 as D
import native_scene_audit as A
import sam3d_common_scene as S
import sam3d_mechanisms as M
from native_appearance import load_native_glb, render_native_mesh

SCENE = 'data/mechanism-scenes/1e7771cdbc91174f8e57900b42b1b95789556255aaa73911cc09909f50db8422/constrained/scene.json'
SCENE_SHA = 'c8389b1405850bf35314952125eb1d70324a2cf65ee9d6b4e17cc84d6568f2d4'
OUTPUT = 'data/native-orientation-diagnostics/laptop-20261103-v1-attempt2'
SAMPLES = [30, 36, 44, 48, 52]
PRIMARY = 44
DIMENSION = 320
CODE = ['native_orientation_diagnostic_v1.py', 'scripts/diagnose_laptop_orientation_20261103_v1.py', 'native_appearance.py', 'alignment_render.py', 'sam3d_common_scene.py', 'sam3d_mechanisms.py', 'native_scene_audit.py']


def write_new(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(document, indent=2, allow_nan=False) + '\n')


def ref(path):
    path = Path(path)
    D.require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT), 'Ordinary checked project artifact required')
    raw = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def render(mesh, matrix, k, width, height):
    rotation, translation, scale = M.rotation_pose(matrix)
    return render_native_mesh(mesh, k, rotation, translation, width, height, scale=scale)


def proof(source, mask, images, title, path):
    yy, xx = np.where(mask)
    padding = 12
    box = [max(0, int(xx.min()) - padding), max(0, int(yy.min()) - padding), min(source.shape[1], int(xx.max()) + padding + 1), min(source.shape[0], int(yy.max()) + padding + 1)]
    width, height = 320, 290
    canvas = Image.new('RGB', (width * (len(images) + 1), height + 62), '#f7f8fb'); draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, fill='#151923')
    for column, (label, rgb, silhouette) in enumerate([('Exact source / SAM crop', source, mask), *images]):
        rgb = rgb.copy()
        if column:
            rgb[~silhouette] = [240, 242, 247]
        crop = Image.fromarray(rgb).crop(tuple(box)); crop.thumbnail((width - 20, height - 28))
        canvas.paste(crop, (column * width + (width - crop.width) // 2, 58 + (height - 28 - crop.height) // 2))
        draw.text((column * width + 10, 38), label, fill='#3766ed')
    canvas.save(path)
    return {'path': ref(path), 'editorial_crop_source_pixel_box_at_raster': box}


def build():
    start = time.monotonic()
    scene_path, scene, program, evidence = A._checked_scene(ROOT / SCENE, ROOT)
    D.require(ref(scene_path)['sha256'] == SCENE_SHA, 'Exact approved private source scene required')
    prior_audit = json.loads((ROOT / 'data/mechanism-hinge-relocalizations/laptop-20261103-v2/local-fit-v1/visual-audit-v1/audit.json').read_text())
    excluded = sorted({r['source_index'] for r in prior_audit['samples'] if r['role'] == 'final_test'})
    D.require(not set(SAMPLES) & set(excluded), 'Previously designated final-test IDs cannot select appearance hypotheses')
    output = ROOT / OUTPUT
    D.require(not output.exists(), 'Immutable diagnostic output already exists; never replace a prior diagnostic')
    output.mkdir(parents=True)
    implementations = [ref(ROOT / name) for name in CODE]
    for item in implementations:
        target = output / 'implementation' / item['path']; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes((ROOT / item['path']).read_bytes())
    protocol = {'mode': 'retrospective_appearance_diagnostic', 'primary_sample_index': PRIMARY, 'additional_predeclared_sample_indices': [i for i in SAMPLES if i != PRIMARY],
                'prior_final_test_sample_indices_excluded': excluded, 'sampling_chosen_before_scores': True,
                'no_independent_pose_validation': True, 'reason': 'Native generation anchor44 supports same-appearance comparison; other RGB views are diagnostics and may show changing screen content.'}
    write_new(output / 'intent.json', {'format': D.FORMAT, 'version': D.VERSION, 'scene_ref': ref(scene_path), 'source_sha256': scene['source']['sha256'],
                                    'protocol': protocol, 'config': {'maximum_dimension': DIMENSION, 'candidate_count': 24, 'native_triangles': 'all_original', 'scale': 'unchanged_per_component'}, 'implementation': implementations})
    width = round(scene['source']['width'] * DIMENSION / max(scene['source']['width'], scene['source']['height']))
    height = round(scene['source']['height'] * DIMENSION / max(scene['source']['width'], scene['source']['height']))
    k = np.asarray(scene['camera']['K'], float).copy(); k[0] *= width / scene['source']['width']; k[1] *= height / scene['source']['height']
    k[0, 2] += (width / scene['source']['width'] - 1) / 2; k[1, 2] += (height / scene['source']['height'] - 1) / 2
    input_refs = {scene['original_program']['path']: scene['original_program'], scene_path.relative_to(ROOT).as_posix(): ref(scene_path)}
    components = []
    for component in scene['components']:
        component_id = component['id']; asset_path = A.checked(component['asset'], ROOT); input_refs[component['asset']['path']] = component['asset']
        mesh = load_native_glb(asset_path); center = mesh.vertices.mean(0)
        rotations = D.pca_rotations(mesh.vertices)
        representative = mesh.vertices[np.linspace(0, len(mesh.vertices) - 1, min(2000, len(mesh.vertices))).astype(int)]
        frame_rows = []; saved_primary = {}; source_primary = mask_primary = None
        for sample_index in SAMPLES:
            source = evidence[sample_index]
            source_ref = {'path': source['frame_path'], 'sha256': source['frame_sha256'], 'bytes': (ROOT / source['frame_path']).stat().st_size}
            observation = next(o for o in source['observations'] if o['track_id'] == component_id)
            mask_ref = {'path': observation['mask_path'], 'sha256': observation['mask_sha256'], 'bytes': (ROOT / observation['mask_path']).stat().st_size}
            A.checked(source_ref, ROOT); A.checked(mask_ref, ROOT); input_refs[source_ref['path']] = source_ref; input_refs[mask_ref['path']] = mask_ref
            D.require(observation.get('gate_status') == 'candidate_geometric_gate_only' and observation.get('audit_status') not in {'reject', 'rejected', 'withheld'}, 'Selected evidence observation is not usable')
            with Image.open(ROOT / source_ref['path']) as im: rgb = np.asarray(im.convert('RGB'))
            with Image.open(ROOT / mask_ref['path']) as im: mask = np.asarray(im) > 0
            small = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA); observed = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
            original_pose = next(row for row in component['poses'] if row['source_sample_index'] == sample_index)
            D.require(original_pose['status'] == 'estimated' and original_pose['camera_from_native_mesh'] is not None, 'No supported pose for selected appearance diagnostic')
            original_matrix = np.asarray(original_pose['camera_from_native_mesh'], float)
            original_render = render(mesh, original_matrix, k, width, height)
            original_metrics = D.appearance_metrics(small, observed, original_render['rgb'], original_render['mask'])
            target = S.mask_shape(observed); scale = M.rotation_pose(original_matrix)[2]
            candidate_rows = []; visuals = {}
            for hypothesis in rotations:
                local = D.centered_transform(hypothesis['rotation'], center); matrix = original_matrix @ local
                rotation = M.rotation_pose(matrix)[0]
                fit = S.projected_pose(representative, rotation, scale, k, target, float(original_matrix[2, 3]))
                if fit is None:
                    candidate_rows.append({'id': hypothesis['id'], 'metrics': [{'status': 'unknown', 'reason': 'nonfinite_projection', 'objective': None}]}); continue
                translation, _, _ = fit; matrix[:3, 3] = translation
                rendered = render(mesh, matrix, k, width, height)
                metrics = D.appearance_metrics(small, observed, rendered['rgb'], rendered['mask'])
                candidate_rows.append({'id': hypothesis['id'], 'metrics': [metrics], 'camera_from_native_mesh_hypothesis': matrix.tolist()})
                visuals[hypothesis['id']] = rendered
            ranking = D.choose_candidate(candidate_rows)
            frame_rows.append({'source_sample_index': sample_index, 'timestamp_s': source['timestamp_s'], 'source_rgb_ref': source_ref, 'source_mask_ref': mask_ref,
                               'original_saved_pose_metrics': original_metrics, 'hypotheses': candidate_rows, 'ranking': ranking})
            order = [r['id'] for r in ranking['ranked'][:3]]
            if 'identity' not in order: order.insert(0, 'identity')
            labels = [('Saved connected pose', original_render['rgb'], original_render['mask'])]
            for identifier in order[:4]:
                if identifier in visuals:
                    score = next(r['metrics'][0].get('objective') for r in candidate_rows if r['id'] == identifier)
                    labels.append((identifier + (' / ' + str(round(score, 3)) if score is not None else ' / low texture'), visuals[identifier]['rgb'], visuals[identifier]['mask']))
            proof_ref = proof(small, observed, labels, f'{component_id} | exact source {sample_index} / {source["timestamp_s"]:.3f}s | independent appearance hypotheses, not a joint scene', output / f'{component_id}-{sample_index:06d}-proof.png')
            frame_rows[-1]['proof'] = proof_ref
            print(json.dumps({'part': component_id, 'sample': sample_index, 'ranking': ranking}), flush=True)
        anchor = next(row for row in frame_rows if row['source_sample_index'] == PRIMARY)
        anchor_selected = anchor['ranking']['selected_id']
        matching = [row['source_sample_index'] for row in frame_rows if row['ranking']['selected_id'] == anchor_selected and anchor_selected is not None]
        components.append({'id': component_id, 'asset_ref': component['asset'], 'native_appearance_receipt': mesh.receipt, 'pca_hypotheses': rotations,
                           'native_center': center.tolist(), 'primary_ranking': anchor['ranking'], 'same_winner_source_indices': matching,
                           'frame_results': frame_rows, 'connected_joint_candidate_exported': False})
    # Verify that no source input, original scene, or implementation changed while rendering.
    for item in input_refs.values(): A.checked(item, ROOT)
    for item in implementations: D.require(ref(ROOT / item['path']) == item, 'Implementation changed while rendering')
    document = {'format': D.FORMAT, 'version': D.VERSION, 'scene_ref': ref(scene_path), 'source_sha256': scene['source']['sha256'], 'protocol': protocol,
                'inputs': list(input_refs.values()), 'implementation': implementations, 'raster': {'width': width, 'height': height, 'K': k.tolist()},
                'components': components, 'wall_seconds': round(time.monotonic() - start, 3),
                'limitations': ['A native appearance state may differ from later source RGB, especially a changing display.',
                               'Repeated or hallucinated texture on both mesh sides cannot establish the physical back side.',
                               'Independent appearance hypotheses must be reanchored and refitted through the joint model before use in a connected scene.',
                               'No model has changed; no orientation candidate is accepted, published, or a verified physical pose.'],
                'new_api_calls': 0, 'new_gpu_calls': 0, 'source_scene_modified': False, 'geometry_modified': False, 'published': False, 'physical_accuracy_verified': False}
    write_new(output / 'diagnostic.json', document)
    return {'diagnostic_ref': ref(output / 'diagnostic.json'), 'components': [{'id': row['id'], 'primary_ranking': row['primary_ranking'], 'same_winner_source_indices': row['same_winner_source_indices']} for row in components], 'wall_seconds': document['wall_seconds']}


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(build()), flush=True)
