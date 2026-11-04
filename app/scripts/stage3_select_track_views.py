#!/usr/bin/env python3
"""Rank model-discovered track views at arbitrary image sizes; preserve seed lineage."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twin import case_path
from stage3_select_views import sha256, mask_metrics, score_candidates, select_diverse, export_selected, contact_sheet


def read(path):
    return json.loads(path.read_text())


def collect_track(case, run, track_id, audit_run=None):
    manifest = read(case / 'manifest.json')
    segmentation = case / 'segmentation' / run
    meta = read(segmentation / 'metadata.json')
    if meta.get('status') != 'complete' or meta['source_sha256'] != manifest['source']['sha256']:
        raise ValueError('Only complete segmentation from the identical source is accepted')
    cfg = read(segmentation / 'annotations.json')
    spec = next(t for t in cfg['tracks'] if t['id'] == track_id)
    track = next(t for t in read(segmentation / 'tracks.json')['tracks'] if t['track_id'] == track_id)
    rows = {r['index']: r for r in read(case / 'frames/index.json')['frames']}
    # Verify lineage before assigning the canonical model_generated_prompt origin.
    model_origin = cfg.get('seed_origin') == 'codex_cli_hosted_OpenAI_model_proposal'
    model_origin = model_origin and cfg.get('manual_seed_boxes_used') is False
    discoveries = list((case / 'reasoning').glob('*/discovery.json'))
    discovery = next((p for p in discoveries if sha256(p) == cfg.get('discovery_record_sha256')), None)
    if model_origin and discovery is None:
        raise ValueError('Model-generated seed claim has no hash-matched discovery record')
    origin = 'model_generated_prompt' if model_origin else meta.get('seed_origin', 'unknown')
    provenance = {'prompt_origin': origin, 'original_seed_origin': meta.get('seed_origin'),
                  'model': meta['model'], 'device': meta['device'],
                  'checkpoint_sha256': meta['checkpoint_sha256'],
                  'metadata_file': str((segmentation / 'metadata.json').relative_to(case)),
                  'metadata_sha256': sha256(segmentation / 'metadata.json'),
                  'annotations_file': str((segmentation / 'annotations.json').relative_to(case)),
                  'annotations_sha256': sha256(segmentation / 'annotations.json'),
                  'prompts': spec['prompts'], 'human_confirmed': False,
                  'manual_seed_boxes_used': not model_origin, 'accuracy_measured': False}
    if discovery:
        provenance.update(discovery_file=str(discovery.relative_to(case)), discovery_sha256=sha256(discovery),
                          prompt_provider='hosted_OpenAI_via_local_Codex_CLI_temporary_Meta_substitute')
    quality = read(segmentation / 'frames.json')
    accepted = {(r['sample_index'], t['track_id']): t for r in quality['frames'] for t in r['tracks']
                if t.get('status') == 'candidate_geometric_gate_only'}
    audited = evidence = None
    if audit_run is not None:
        audit_path = segmentation / audit_run / 'audit.json'
        audit = read(audit_path)
        if (audit.get('status') != 'complete_candidate_visual_audit'
                or audit['source_sha256'] != manifest['source']['sha256']
                or audit['source_track_index_sha256'] != sha256(segmentation / 'tracks.json')):
            raise ValueError('Model audit must match the current source and track index')
        audited = {(r['sample_index'], r['track_id']) for r in audit['result']['observations']
                   if r['verdict'] == 'plausible_visible_mask'}
        evidence = {(r['sample_index'], r['track_id']): r
                    for r in read(segmentation / audit_run / 'evidence.json')}
        provenance.update(mask_plausibility_audit_file=str(audit_path.relative_to(case)),
                          mask_plausibility_audit_sha256=sha256(audit_path),
                          mask_plausibility_is_accuracy_measurement=False)
    candidates = []
    for observation in track['frames']:
        index = observation['sample_index']
        row = rows[index]
        if abs(row['timestamp_s'] - observation['timestamp_s']) > 1e-6:
            raise ValueError('Mask/source PTS mismatch')
        if (index, track_id) not in accepted or observation.get('status') != 'candidate_visible':
            continue
        if audited is not None and (index, track_id) not in audited:
            continue
        image_path = case / row['file']
        mask_path = segmentation / observation['raw_mask_file']
        if evidence is not None:
            checked = evidence.get((index, track_id), {})
            if (checked.get('source_frame_sha256') != sha256(image_path)
                    or checked.get('raw_mask_sha256') != sha256(mask_path)
                    or abs(checked.get('timestamp_s', -1) - row['timestamp_s']) > 1e-6):
                raise ValueError('Audited RGB/mask pixels or PTS changed')
        image = np.asarray(Image.open(image_path).convert('RGB'))
        mask = np.asarray(Image.open(mask_path).convert('L')) > 0
        metrics = mask_metrics(image, mask)
        if image.shape[:2] != (row['height'], row['width']):
            raise ValueError('Recorded and actual pixel dimensions disagree')
        # Conservative resolution gate, not a semantic mask-accuracy test.
        x0, y0, x1, y1 = metrics['bbox_xyxy']
        if metrics['area_pixels'] < 400 or min(x1-x0, y1-y0) < 16:
            continue
        candidates.append({'sample_index': index, 'timestamp_s': row['timestamp_s'],
                           'source_case': case.name, 'source_sample_index': index,
                           'source_track_id': track_id, 'object_id': track_id,
                           'physical_identity_status': spec['identity_status'],
                           'image_file': row['file'], 'mask_file': str(mask_path.relative_to(case)),
                           'image_sha256': sha256(image_path), 'mask_sha256': sha256(mask_path),
                           'mask_scope': spec.get('description', spec.get('label', track_id)),
                           'model_provider': meta['model'], 'mask_provenance': provenance,
                           'metrics': metrics, 'occlusion_candidate': {'status': 'not_assessed', 'hand_proximity_fraction': None},
                           'segmentation_accuracy_measured': False})
    return candidates, manifest, spec, model_origin


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True)
    p.add_argument('--run', default='automatic-components-v1')
    p.add_argument('--output', default='automatic-reconstruction-inputs-v1')
    p.add_argument('--track-id')
    p.add_argument('--audit-run', default='model-audit-v1')
    p.add_argument('--top-k', type=int, default=3)
    a = p.parse_args()
    import re
    if any(not re.fullmatch(r'[a-z0-9][a-z0-9-]*', v) for v in [a.run, a.output, a.audit_run]) or not 1 <= a.top_k <= 10:
        p.error('Invalid run/output slug or top-k')
    case = case_path(a.case)
    out = case / a.output
    if out.exists():
        raise FileExistsError('Output exists; choose a new output slug')
    cfg = read(case / 'segmentation' / a.run / 'annotations.json')
    ids = [t['id'] for t in cfg['tracks'] if a.track_id is None or t['id'] == a.track_id]
    if not ids:
        raise ValueError('No requested model track exists')
    out.mkdir()
    records = []
    for tid in ids:
        candidates, manifest, spec, automatic = collect_track(case, a.run, tid, a.audit_run)
        ranked = score_candidates(candidates)
        selected = select_diverse(ranked, a.top_k, 1)
        if not selected:
            records.append({'track_id': tid, 'status': 'no_candidate_passed_model_audit_resolution_and_geometric_gates'})
            continue
        directory = out / tid
        directory.mkdir()
        selected = export_selected(case, directory, selected)
        if sha256(case / manifest['source']['file']) != manifest['source']['sha256']:
            raise ValueError('Source video hash mismatch')
        record = {'schema_version': '0.1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                  'status': 'ranked_inputs_not_reconstruction', 'object_id': tid,
                  'object_choice_origin': 'generic_model_discovery' if automatic else 'existing_prompted_track',
                  'source': manifest['source'], 'registry_object': {'id': tid, 'label': spec.get('label'),
                                                                       'physical_identity_verified': False},
                  'source_sha256_verified_against_local_video': True,
                  'selection_method': {'type': 'existing_deterministic_image_mask_quality_heuristics',
                                       'agent_visual_choice': False, 'reads_geometry_or_corner_annotations': False,
                                       'candidate_count': len(candidates)},
                  'input_contract': {'rgb_mask_same_pts': True,
                                     'dimensions': selected[0]['exports']['rgb_mask_dimensions'],
                                     'time_domain': 'source_video_presentation_seconds', 'resize_performed': False},
                  'assistance_disclosure': {'manual_seed_boxes_used': not automatic,
                                            'manual_geometry_used': False, 'manual_camera_corners_used': False,
                                            'semantic_mask_accuracy_measured': False,
                                            'fully_automatic_video_to_twin_verified': False},
                  'limitations': ['Geometric gates and view ranking can retain a semantically wrong or occluded mask.',
                                  'Model-proposed tracklets do not establish physical identity or complete 3D shape.'],
                  'selected': selected, 'all_ranked_candidates': ranked}
        (directory / 'selection.json').write_text(json.dumps(record, indent=2) + '\n')
        contact_sheet(directory, selected)
        records.append({'track_id': tid, 'label': spec.get('label'), 'status': 'ranked_inputs_not_reconstruction',
                        'selection': str((directory / 'selection.json').relative_to(case)),
                        'candidate_count': len(candidates), 'top_sample': selected[0]['sample_index'],
                        'top_timestamp_s': selected[0]['timestamp_s'], 'manual_seeds_used': not automatic})
    (out / 'index.json').write_text(json.dumps({'case': a.case, 'tracks': records}, indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
