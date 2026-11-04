"""Prepare two new phone SAM3D anchors without inference or publication.

Uses the saved model masks and unchanged full-frame RGB. The colour preference
is specific to the existing model-labelled blue phone and is a disclosed
development heuristic, not a new visual model or an accuracy measurement.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import initial_sam3d_handoff as H
from scripts.stage3_sam3d import read_inputs

ROOT = Path(__file__).resolve().parent
CASE = ROOT / 'data/flow-capture-phone-20261103-v1'
OUTPUT = ROOT / 'output/phone-source-regeneration-20261104-v1'
PRIOR_SELECTION = ROOT / ('data/sam3d-upload-jobs/'
    '5cf65bb285d005739c08cdb9ad3a970947a8b9bdf4965475a528e9254edc6ded/'
    'candidate/initial-handoff/provenance/selection.json')


def ref(path):
    path = Path(path)
    return {'path': path.relative_to(ROOT).as_posix(),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def score(rgb, mask):
    """Reject border cuts/thin views; prefer visible unreflective blue back."""
    ys, xs = np.nonzero(mask)
    H.require(len(xs) > 0, 'Nonempty saved SAM mask required')
    rect = cv2.minAreaRect(np.column_stack((xs, ys)).astype(np.float32))
    aspect = min(rect[1]) / max(rect[1])
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    solidity = min(1., len(xs) / max(1., cv2.contourArea(cv2.convexHull(
        max(contours, key=cv2.contourArea)))))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    blue = (hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 125) & \
        (hsv[:, :, 1] >= 60) & (hsv[:, :, 2] >= 45)
    blue_fraction = float(np.mean(blue[mask]))
    sharpness = float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                                   cv2.CV_64F)[mask].var())
    touches = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    eligible = not touches and aspect >= .35 and len(xs) >= 10000
    value = len(xs) * blue_fraction ** 2 * min(sharpness / 100, 1) * solidity if eligible else 0.
    return {'area_pixels': len(xs), 'blue_fraction': blue_fraction,
            'sharpness_laplacian_variance': sharpness, 'minrect_aspect': float(aspect),
            'solidity': float(solidity), 'touches_frame_border': touches,
            'eligible': eligible, 'score': float(value)}


def prepare():
    destination = CASE / 'reconstruction-inputs/phone-back-regeneration-20261104-v1'
    H.require(not destination.exists(), 'Fresh private selection directory required')
    program_path = CASE / 'flow-procedure-1/interaction.json'
    program = json.loads(program_path.read_text())
    prior = json.loads(PRIOR_SELECTION.read_text())
    observations = []
    for evidence in program['evidence']:
        obs = next((o for o in evidence['observations'] if o['track_id'] == 'phone_blue'), None)
        if obs is None:
            continue
        image_path, mask_path = ROOT/evidence['frame_path'], ROOT/obs['mask_path']
        H.require(ref(image_path)['sha256'] == evidence['frame_sha256'] and
                  ref(mask_path)['sha256'] == obs['mask_sha256'], 'Exact saved source required')
        rgb = np.asarray(Image.open(image_path).convert('RGB'))
        mask = np.asarray(Image.open(mask_path)) > 0
        observations.append({'source_index': evidence['source_index'],
            'timestamp_s': evidence['timestamp_s'], 'image': ref(image_path),
            'mask': ref(mask_path), 'metrics': score(rgb, mask)})
    ranked = sorted(observations, key=lambda r: (-r['metrics']['score'], r['source_index']))
    first = ranked[0]
    # Five seconds avoids adjacent redundant frames from the same back view.
    second = next(r for r in ranked if r['metrics']['eligible'] and
                  abs(r['timestamp_s'] - first['timestamp_s']) >= 5)
    chosen = [first, second]
    destination.mkdir(parents=True)
    selection = deepcopy(prior)
    selection.update(created_at_utc='2026-11-04T00:00:00+00:00', selected=[],
        status='ranked_private_retry_inputs_not_reconstruction',
        selection_method={'type': 'source_RGB_SAM_mask_visible_blue_back_ranking_v1',
            'agent_visual_choice': False, 'reads_geometry_or_corner_annotations': False,
            'candidate_count': len(observations),
            'rule': 'no_frame_cut,min_aspect0.35,min_area10000;area*blue_fraction^2*min(sharpness/100,1)*solidity;second_anchor_at_least5s_apart',
            'blue_HSV_bounds': {'h': [85,125], 's_min':60, 'v_min':45},
            'rule_authored_after_source_development_views_were_seen': True,
            'not_independent_accuracy_measurement': True})
    for rank, item in enumerate(chosen, 1):
        index = item['source_index']
        row = deepcopy(prior['selected'][0])
        relative = f'rank-{rank:02d}-source-{index:06d}'
        export = destination/relative
        export.mkdir()
        Image.open(ROOT/item['image']['path']).convert('RGB').save(export/'image.png')
        (export/'mask.png').write_bytes((ROOT/item['mask']['path']).read_bytes())
        row.update(sample_index=index, source_sample_index=index, timestamp_s=item['timestamp_s'],
            image_file=item['image']['path'].removeprefix('data/'+CASE.name+'/'),
            mask_file=item['mask']['path'].removeprefix('data/'+CASE.name+'/'),
            image_sha256=item['image']['sha256'], mask_sha256=item['mask']['sha256'],
            metrics=item['metrics'], score=item['metrics']['score'], selection_rank=rank,
            occlusion_candidate=False, segmentation_accuracy_measured=False,
            score_components=item['metrics'], occlusion_candidate_penalty=0,
            exports={'directory':relative, 'rgb':relative+'/image.png', 'mask':relative+'/mask.png',
                     'rgb_mask_dimensions':[720,960], 'resized':False, 'background_invented':False,
                     'rgb_sha256':ref(export/'image.png')['sha256'],
                     'mask_sha256':ref(export/'mask.png')['sha256']})
        # False here means the heuristic cannot establish occlusion, not a clear-view certificate.
        row['occlusion_status'] = 'unknown_not_measured'
        selection['selected'].append(row)
    selection_path = destination/'selection.json'
    save(selection_path, selection)
    prepared=[]
    for row in selection['selected']:
        rank=row['selection_rank']; export=destination/row['exports']['directory']
        manifest={'schema_version':'0.1','object_id':'phone_blue','image':'image.png','mask':'mask.png',
            'image_sha256':ref(export/'image.png')['sha256'],'mask_sha256':ref(export/'mask.png')['sha256'],
            'source':{'video_sha256':selection['source']['sha256'],'sample_index':row['sample_index'],
                'timestamp_s':row['timestamp_s'],'source_case':CASE.name,'source_sample_index':row['source_sample_index'],
                'time_domain':'source_video_presentation_seconds'},
            'mask_provenance':dict(row['mask_provenance'],backend=row['model_provider'],
                prompt_origin=row['mask_provenance']['prompt_origin'],manual_intervention=False,scope=row['mask_scope']),
            'selection':{'method':'automatic deterministic quality ranking','rank':rank,'score':row['score'],
                'selection_sha256':ref(selection_path)['sha256'],'agent_visual_selection':False,
                'manual_geometry_used':False,'manual_camera_corners_used':False},
            'limitations':['New source anchor; no geometry repair or accepted reconstruction exists yet.',
                'Blue back ranking cannot establish hidden geometry, physical thickness, pose or mask accuracy.']}
        save(export/'input.json', manifest)
        read_inputs(export/'input.json')
        handoff=ROOT/f'data/initial-sam3d-handoffs/phone-back-regeneration-20261104-v1-rank{rank}'
        result=H.prepare(export/'input.json',selection_path,handoff)
        verified=H.inspect(handoff/'handoff.json')
        prepared.append({'source_index':row['source_sample_index'],'rank':rank,'prepared':result,'verified':verified})
    report={'format':'interaction-twin-phone-new-source-regeneration-plan','version':1,
        'implementation':ref(Path(__file__)),'program':ref(program_path),'prior_selection':ref(PRIOR_SELECTION),
        'selection':ref(selection_path),'all_observations':observations,'anchors':prepared,
        'inference_commands_executed':False,'hosted_API_calls':0,'GPU_calls':0,
        'native_models_modified':False,'publication_changed':False,'physical_accuracy_verified':False,
        'local_runtime':{'sam3d_package_available':False,'inference_backend_required':'existing_reviewed_CUDA_cluster_runtime'},
        'generation_policy':{'maximum_new_inferences':2,'maximum_parallel':2,'automatic_retry':False,
             'parent_allocation_must_not_be_changed':'16889584','seed':42,'appearance':'vertex-color'},
        'validation_policy':{'no_automatic_promotion':True,'mandatory_prior_failed_source5_gate_retained':True,
            'mandatory_prior_midpoint4_gate_retained':True,'full_mesh_all_views_required':True,
            'thin_profile_diagnostic_only_not_metric_thickness':True,
            'reject_duplicate_or_perpendicular_slabs':True,'reject_open_holes_or_body_discontinuity':True,
            'heldout_minimum_IoU':.65,'heldout_maximum_baseline_regression':.05,
            'new_camera_pose_fits_required':True,'source_observed_RGB_preserved':True,
            'existing_source_views_are_development_evidence_not_unseen_test':True},
        'next_step':'Check existing SSH/allocation availability, then dispatch each exact handoff through unchanged bridge. Import privately; do not reuse old phone poses or approve by front silhouette alone.'}
    save(OUTPUT/'regeneration-plan.json',report)
    return {'plan':ref(OUTPUT/'regeneration-plan.json'),'source_indices':[r['source_index']for r in prepared],
            'handoffs':[r['prepared']['handoff']for r in prepared]}


if __name__ == '__main__':
    print(json.dumps(prepare(),indent=2))
