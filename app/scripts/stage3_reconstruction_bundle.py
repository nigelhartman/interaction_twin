#!/usr/bin/env python3
"""Make a portable model input from an algorithmically ranked RGB/SAM view."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rank', type=int, default=1)
    args = parser.parse_args()
    selection_path = args.selection.resolve()
    selection = json.loads(selection_path.read_text())
    row = next(r for r in selection['selected'] if r['selection_rank'] == args.rank)
    if args.output.exists():
        raise FileExistsError('Preserve previous inputs; choose a new output directory')
    args.output.mkdir(parents=True)
    for key, name in [('rgb', 'image.png'), ('mask', 'mask.png')]:
        source = selection_path.parent / row['exports'][key]
        expected = row['exports'][key + '_sha256']
        if digest(source) != expected:
            raise ValueError('Selected input hash mismatch: ' + key)
        shutil.copy2(source, args.output / name)
    provenance = row['mask_provenance']
    origin = provenance.get('prompt_origin', 'unknown')
    # Conservative: unknown assistance never becomes an automatic-input claim.
    manual = origin not in ['automatic_mask_generation', 'model_generated_prompt']
    manifest = {
        'schema_version': '0.1', 'object_id': selection['object_id'],
        'image': 'image.png', 'mask': 'mask.png',
        'image_sha256': digest(args.output / 'image.png'),
        'mask_sha256': digest(args.output / 'mask.png'),
        'source': {'video_sha256': selection['source']['sha256'],
                   'sample_index': row['sample_index'], 'timestamp_s': row['timestamp_s'],
                   'source_case': row['source_case'], 'source_sample_index': row['source_sample_index'],
                   'time_domain': 'source_video_presentation_seconds'},
        'mask_provenance': {**provenance, 'backend': row['model_provider'],
                            'prompt_origin': origin, 'manual_intervention': manual,
                            'scope': row['mask_scope']},
        'selection': {'method': 'automatic deterministic quality ranking', 'rank': args.rank,
                      'score': row['score'], 'selection_sha256': digest(selection_path),
                      'agent_visual_selection': False, 'manual_geometry_used': False,
                      'manual_camera_corners_used': False},
        'limitations': ['Inherited segmentation prompts/corrections are disclosed; this does not prove automatic object discovery.',
                       'Masked visible pixels do not establish hidden geometry or a verified physical scale.'],
    }
    from stage3_sam3d import read_inputs
    path = args.output / 'input.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    read_inputs(path)
    print(json.dumps({'manifest': str(path.resolve()), 'object_id': manifest['object_id'],
                      'timestamp_s': row['timestamp_s'], 'manual_segmentation_inherited': manual}, indent=2))


if __name__ == '__main__':
    main()
