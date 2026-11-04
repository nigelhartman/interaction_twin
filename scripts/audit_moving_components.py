#!/usr/bin/env python3
"""One bounded saved-pixel geometry attempt per separately moving Jabra part.

No model inference, old asset/pose/depth inputs, shared part coordinates or
manual geometry enter these experiments. Failures remain first-class results.
"""
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import observed_geometry

OUTPUT = ROOT / 'data/observed-moving-parts-20261031'
PROGRAM = ROOT / 'data/jabra-local-v2/procedure-v1/interaction.json'
MASK_BASE = ROOT / 'data/recording-144929-1fps-v3/segmentation/automatic-components-v1'
PARTS = [('container_lid', 'jabra-lid-source-pixels-v1', 'Jabra moving lid'),
         ('small_object_a', 'jabra-earbud-a-source-pixels-v1', 'First separately handled earbud')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    if OUTPUT.exists():
        raise ValueError('Preserve prior attempts; this fixed two-attempt experiment already exists')
    OUTPUT.mkdir()
    program = json.loads(PROGRAM.read_text())
    annotations = json.loads((MASK_BASE / 'annotations.json').read_text())
    mask_metadata = json.loads((MASK_BASE / 'metadata.json').read_text())
    script_hash = sha(__file__)
    engine_hash = sha(observed_geometry.__file__)
    entries = []
    for track_id, identifier, label in PARTS:
        started = time.monotonic()
        with (OUTPUT / (identifier + '.log')).open('w') as log:
            with contextlib.redirect_stdout(log):
                result = observed_geometry.run(PROGRAM, track_id, OUTPUT / identifier,
                    root=ROOT, max_views=35, sensitivity=False, max_nfev=70)
        duration = time.monotonic() - started
        protocol = json.loads((OUTPUT / identifier / 'protocol.json').read_text())
        # Independently recompute which sibling observations must be subtracted.
        source_evidence = {e['id']: e for e in program['evidence']}
        checks = []
        for frame in protocol['input_views']:
            source = source_evidence[frame['id']]
            expected = [{'track_id': o['track_id'], 'mask_path': o['mask_path'], 'mask_sha256': o['mask_sha256']}
                for o in source['observations'] if o['track_id'] != track_id and
                o.get('audit_status') not in {'reject_wrong_or_lost_mask', 'uncertain'}]
            checks.append(frame['subtracted_components'] == expected)
        audit = {
            'input_views': len(checks), 'all_expected_sibling_masks_subtracted': all(checks),
            'original_program_sha256_matches': protocol['program_sha256'] == sha(PROGRAM),
            'engine_snapshot_sha256_matches': protocol['implementation_sha256'] == engine_hash and
                sha(OUTPUT / identifier / 'implementation.py') == engine_hash,
            'old_generated_meshes_read': protocol['generated_meshes_read'],
            'old_depth_maps_read': protocol['depth_maps_read'], 'old_poses_read': protocol['old_poses_read'],
            'manual_coordinates': protocol['manual_coordinates'],
            'part_coordinates': 'independent arbitrary gauge; not aligned or fused across components',
            'new_api_calls': 0, 'new_inference_calls': 0,
        }
        assert audit['all_expected_sibling_masks_subtracted'] and audit['engine_snapshot_sha256_matches']
        assert not any(audit[k] for k in ['old_generated_meshes_read', 'old_depth_maps_read', 'old_poses_read', 'manual_coordinates'])
        write(OUTPUT / identifier / 'input-audit.json', audit)
        entry = {'id': identifier, 'track_id': track_id, 'label': label,
            'viewer_url': '/files/' + (OUTPUT / identifier / 'viewer.json').relative_to(ROOT).as_posix(),
            'protocol_url': '/files/' + (OUTPUT / identifier / 'protocol.json').relative_to(ROOT).as_posix(),
            'input_audit_url': '/files/' + (OUTPUT / identifier / 'input-audit.json').relative_to(ROOT).as_posix(),
            'status': result['status'], 'reason': result['reason'], 'metrics': result['metrics'],
            'program_url': '/files/' + PROGRAM.relative_to(ROOT).as_posix(),
            'mask_provider': mask_metadata['model'] + ' / ' + mask_metadata['device'],
            'mask_seed_origin': annotations['seed_origin'],
            'manual_seed_boxes_used': annotations['manual_seed_boxes_used'],
            'manual_geometry_used': annotations['manual_geometry_used'],
            'identity_status': 'model_proposed_tracklet_physical_identity_unverified',
            'coordinates': 'independent_arbitrary_component_coordinates', 'elapsed_s': round(duration, 3),
            'source_pts_s': [f['timestamp_s'] for f in protocol['input_views']],
            'sensitivity_tested': False, 'calibration_measured': False, 'new_api_calls': 0,
            'new_inference_calls': 0, 'prior_geometry_or_depth_used': False}
        entries.append(entry)
        print(json.dumps({'part': track_id, 'status': result['status'], 'metrics': result['metrics'], 'elapsed_s': duration}), flush=True)
    result = {'format': 'observed-moving-component-experiments', 'version': 1,
        'reason': 'source_pixel_geometry_for_each_separately_moving_rigid_component',
        'components': entries, 'source_program_url': '/files/' + PROGRAM.relative_to(ROOT).as_posix(),
        'source_program_sha256': sha(PROGRAM), 'engine_sha256': engine_hash, 'orchestration_sha256': script_hash,
        'source_mask_metadata_sha256': sha(MASK_BASE / 'metadata.json'),
        'source_annotations_sha256': sha(MASK_BASE / 'annotations.json'),
        'limits': {'max_views': 35, 'max_ba_evaluations': 70, 'sensitivity_enabled': False,
                   'attempts_per_component': 1, 'components_fused': False},
        'new_api_calls': 0, 'new_inference_calls': 0, 'old_pose_or_learned_geometry_used': False,
        'limitations': ['This is a source-pixel feasibility test, separate from saved generated-asset replay.',
            'Part identities and masks remain model assumptions; mask plausibility is not measured accuracy.',
            'Per-component coordinates are independent and uncalibrated; no shared assembly or hinge is inferred.',
            'No complete object, hidden surface, physical scale, collision/contact or simulation is established.',
            'The bounded initial diagnostic does not evaluate focal sensitivity or perform parameter search.']}
    write(OUTPUT / 'manifest.json', result)
    (OUTPUT / 'orchestration.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    main()
