#!/usr/bin/env python3
"""Conditional, private CPU hinge refit after strong appearance agreement.

No rotations, pivots, angles or mesh edits are manually selected. A diagnostic
must first have the same proper native rotation supported by at least three
predeclared RGB/SAM frames for EACH part, with a meaningful score margin.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import diagnose_laptop_orientation_20261103_v1 as F
import native_orientation_diagnostic_v1 as D
import native_scene_audit as A
import sam3d_common_scene as S
import sam3d_mechanisms as M
import mechanism_pipeline as P
from native_appearance import load_native_glb


def qualifying_support(components, expected_samples=F.SAMPLES):
    """Conservative local admission; unique predeclared frames, never votes."""
    support = {}; rotations = {}
    D.require(isinstance(components, list) and len(components) == 2 and {c['id'] for c in components} == {'laptop_base', 'laptop_lid'}, 'Exact laptop base/lid diagnostic required')
    for component in components:
        D.require([row['source_sample_index'] for row in component['frame_results']] == list(expected_samples), 'One exact result per predeclared sample is required')
        selected = component['primary_ranking']['selected_id']
        qualified = []
        for row in component['frame_results']:
            ranking = row['ranking']
            candidate = next((r for r in ranking['ranked'] if r['id'] == selected), None)
            if selected is not None and ranking['selected_id'] == selected and ranking.get('top_two_margin', 0) >= .025 and candidate and candidate['median_mask_iou'] >= .6:
                qualified.append(row['source_sample_index'])
        support[component['id']] = {'selected_id': selected, 'qualified_predeclared_samples': qualified, 'required_samples': 3, 'minimum_top_two_margin': .025, 'minimum_mask_iou': .6}
        if len(qualified) >= 3:
            rotation = next(row['rotation'] for row in component['pca_hypotheses'] if row['id'] == selected)
            D.centered_transform(rotation, np.zeros(3))  # No reflections or scale hidden in an imported result.
            rotations[component['id']] = rotation
    return support, rotations


def build():
    start = time.monotonic()
    diagnostic_path = ROOT / F.OUTPUT / 'diagnostic.json'
    diagnostic_ref = F.ref(diagnostic_path); diagnostic = json.loads(diagnostic_path.read_text())
    D.require(diagnostic.get('format') == D.FORMAT and diagnostic.get('version') == D.VERSION and diagnostic.get('source_scene_modified') is False and diagnostic.get('geometry_modified') is False, 'Exact immutable appearance diagnostic required')
    D.require(diagnostic['protocol']['primary_sample_index'] == F.PRIMARY and diagnostic['protocol']['additional_predeclared_sample_indices'] == [i for i in F.SAMPLES if i != F.PRIMARY], 'Predeclared diagnostic sample basis changed')
    for item in diagnostic['inputs']: A.checked(item, ROOT)
    for item in diagnostic['implementation']: D.require(F.ref(ROOT / item['path']) == item, 'Diagnostic implementation changed')
    current_path = A.checked(diagnostic['scene_ref'], ROOT); current = json.loads(current_path.read_text())
    D.require(diagnostic['scene_ref']['sha256'] == F.SCENE_SHA, 'Exact untouched baseline scene required')
    original_run = F.ref(ROOT / 'data/flow-runs/run-capture-laptop-20261103-v1/run.json')
    output = ROOT / F.OUTPUT / 'appearance-refit-v1'; D.require(not output.exists(), 'Preserve prior appearance refit')
    output.mkdir()
    support, rotations = qualifying_support(diagnostic['components'])
    decision = {'format': 'interaction-twin-appearance-initialized-hinge-research', 'version': 1, 'diagnostic_ref': diagnostic_ref, 'baseline_scene_ref': diagnostic['scene_ref'],
                'support': support, 'new_api_calls': 0, 'new_gpu_calls': 0, 'original_scene_modified': False, 'geometry_modified': False, 'physical_accuracy_verified': False, 'published': False}
    if len(rotations) != len(diagnostic['components']):
        decision.update(status='ambiguous_appearance_no_hinge_refit', reason='At least one part lacks three same-winner, clearly separated, well-overlapping predeclared frames.')
        F.write_new(output / 'result.json', decision)
        return decision
    proposal_ref = current['mechanics']['proposal']; A.checked(proposal_ref, ROOT)
    source_program = json.loads(A.checked(current['original_program'], ROOT).read_text()); evidence = {row['source_index']: row for row in source_program['evidence']}
    # A NEW research intermediate, not an original common-scene build receipt.
    research = copy.deepcopy(current); research.pop('mechanics', None)
    research['fitter_version'] = 'source-rgb-appearance-initialized-research-v1'
    research['model_or_api_calls'] = 0; research['accepted_motion_poses'] = 0
    research['research_derivation'] = {'kind': 'independent_appearance_initialization_before_full_native_anchor_refit', 'diagnostic_ref': diagnostic_ref,
                                     'baseline_scene_ref': diagnostic['scene_ref'], 'support': support, 'conjugation_only_joint_patch': False,
                                     'source_joint_hints_changed': False, 'manual_geometry_or_poses_used': False}
    research['limitations'].append('Proper native orientation hypotheses are initialized by fallible retrospective RGB texture agreement, then hinges must be independently refitted to actual unchanged native vertices.')
    for component in research['components']:
        mesh = load_native_glb(A.checked(component['asset'], ROOT)); local = D.centered_transform(rotations[component['id']], mesh.vertices.mean(0))
        vertices = mesh.vertices[np.linspace(0, len(mesh.vertices) - 1, min(2000, len(mesh.vertices))).astype(int)]
        for row in component['poses']:
            if row['status'] != 'estimated' or row['camera_from_native_mesh'] is None:
                continue
            source = evidence[row['source_sample_index']]; observation = next(o for o in source['observations'] if o['track_id'] == component['id'])
            mask_ref = {'path': observation['mask_path'], 'sha256': observation['mask_sha256'], 'bytes': (ROOT / observation['mask_path']).stat().st_size}
            A.checked(mask_ref, ROOT)
            if observation.get('gate_status') != 'candidate_geometric_gate_only' or observation.get('audit_status') in {'reject', 'rejected', 'withheld'}:
                row.update(status='unknown', camera_from_native_mesh=None, reason='appearance_research_source_mask_not_usable'); continue
            with Image.open(ROOT / mask_ref['path']) as image: mask = np.asarray(image) > 0
            matrix = np.asarray(row['camera_from_native_mesh'], float) @ local; rotation, _, scale = M.rotation_pose(matrix)
            fit = S.projected_pose(vertices, rotation, scale, np.asarray(research['camera']['K']), S.mask_shape(mask), float(matrix[2, 3]))
            if fit is None:
                row.update(status='unknown', camera_from_native_mesh=None, reason='appearance_research_projection_unknown'); continue
            matrix[:3, 3] = fit[0]
            row.update(camera_from_native_mesh=matrix.tolist(), method='automatic_PCA_appearance_hypothesis_and_source_bbox_refit', reason='independent_source_texture_initialization_for_new_joint_refit', accepted_motion_evidence=False)
    snapshots = []
    for name in ['scripts/refit_laptop_appearance_hinge_20261103_v1.py', 'native_orientation_diagnostic_v1.py']:
        target = output / 'implementation' / (Path(name).name + '.txt'); target.parent.mkdir(exist_ok=True); target.write_bytes((ROOT / name).read_bytes()); snapshots.append(F.ref(target))
    by_path = {r['path']: r for r in [*research['inputs'], diagnostic_ref, diagnostic['scene_ref'], *snapshots]}
    research['inputs'] = [by_path[path] for path in sorted(by_path)]
    research['input_signature'] = hashlib.sha256(json.dumps({'inputs': research['inputs'], 'research_derivation': research['research_derivation']}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    intermediate = output / 'appearance-initialized-scene.json'; F.write_new(intermediate, research)
    fitted = M.build(intermediate, ROOT / proposal_ref['path'], output / 'constrained', root=ROOT)
    descriptive = P.export_scene(fitted['manifest'], root=ROOT, publish=False)
    D.require(F.ref(ROOT / original_run['path']) == original_run and F.ref(current_path) == diagnostic['scene_ref'], 'Original run or baseline scene changed')
    decision.update(status=fitted['status'], initialized_scene_ref=F.ref(intermediate), source_proposal_ref=proposal_ref, fit_result=fitted,
                    descriptive_export=descriptive, original_main_run_ref=original_run, original_main_run_changed=False, wall_seconds=round(time.monotonic() - start, 3),
                    joint_native_anchor_refit_attempted=True, joint_native_anchors_recomputed=bool(fitted.get('joints')), independent_mesh_pose_fallback=False)
    F.write_new(output / 'result.json', decision)
    return decision


if __name__ == '__main__':
    print(json.dumps(build()), flush=True)
