#!/usr/bin/env python3
"""Preserve SAM hand masks and a disclosed conservative thumb occlusion candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from sam2.build_sam import build_sam2_video_predictor

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='camera-battery-withdrawal-15fps')
    parser.add_argument('--config', type=Path, default=ROOT/'annotations/stage4-hands-v1.json')
    parser.add_argument('--run', default='hands-v1')
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error('MPS unavailable; no silent CPU fallback')
    case = ROOT/'data'/args.case
    rows = json.loads((case/'frames/index.json').read_text())['frames']
    config = json.loads(args.config.read_text())
    output = case/'occlusion'/args.run
    if output.exists():
        parser.error('Output already exists; preserve it and use a fresh run')
    output.mkdir(parents=True)
    write(output/'annotations.json', config)
    checkpoint = ROOT/'models/sam2.1_hiera_tiny.pt'
    start = time.monotonic()
    metadata = {
        'status': 'running', 'scope': 'stage_4_battery_pilot_hand_occlusion_candidates',
        'model': 'Meta SAM 2.1 Hiera Tiny', 'device': 'mps',
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'source_sha256': json.loads((case/'manifest.json').read_text())['source']['sha256'],
        'torch': torch.__version__, 'human_confirmed': False,
        'prompt_origin': 'assistant_visual_annotation', 'cpu_storage_offload': True,
        'cpu_operator_fallback_enabled': False,
        'target_mask_used_to_construct_occluder': False,
        'mask_meaning': 'Visible hand pixels; not a depth ordering or ground truth mask',
        'foreground_mask_meaning': 'Conservative right thumb subset: raw SAM right hand intersected with assistant annotated thumb corridor, eroded by 2 pixels; corridor interpolated between reviewed seed frames. No target silhouette clipping.',
        'depth_ordering': 'Right thumb visibly crosses in front of selected battery. Other fingers can be behind or below it; raw hand union must not be treated as universally in front.',
        'accuracy_measured': False,
        'quality_limits': ['No ground-truth masks or measured IoU', 'SAM boundary leakage is possible', 'Foreground corridor is a manual prior and does not infer hidden hand geometry', 'Fast blurred sample 52 is unsupported for occlusion-dependent fit validation'],
    }
    write(output/'metadata.json', metadata)
    predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml', str(checkpoint), device='mps', apply_postprocessing=False)
    try:
        with tempfile.TemporaryDirectory(prefix='.sam-frames-', dir=output) as temp:
            frame_dir = Path(temp)
            for offset, row in enumerate(rows):
                os.symlink((case/row['file']).resolve(), frame_dir/f'{offset:06d}.jpg')
            for track in config['tracks']:
                track_start = time.monotonic()
                target = output/'raw_masks'/track['id']
                target.mkdir(parents=True)
                print('Tracking ' + track['id'], flush=True)
                with torch.inference_mode():
                    state = predictor.init_state(str(frame_dir), offload_video_to_cpu=True, offload_state_to_cpu=True)
                    for prompt in track['prompts']:
                        predictor.add_new_points_or_box(
                            state, frame_idx=prompt['sample_index']-1, obj_id=1,
                            box=np.array(prompt['box_xyxy'], dtype=np.float32),
                            points=np.array(prompt['points'], dtype=np.float32),
                            labels=np.array(prompt['labels'], dtype=np.int32),
                        )
                    first = min(prompt['sample_index'] for prompt in track['prompts'])-1
                    def save(index, logits):
                        mask = (logits[0, 0] > 0).cpu().numpy()
                        Image.fromarray(mask.astype(np.uint8)*255).save(target/f'{index+1:06d}.png')
                    for index, ids, logits in predictor.propagate_in_video(state, start_frame_idx=first):
                        save(index, logits)
                    if first:
                        for index, ids, logits in predictor.propagate_in_video(state, start_frame_idx=first, reverse=True):
                            save(index, logits)
                    torch.mps.synchronize()
                    del state
                print(f'{track["id"]} done in {time.monotonic()-track_start:.2f}s', flush=True)
        gate_seeds = config['foreground_thumb_corridor_seeds']
        seed_indices = np.array([seed['sample_index'] for seed in gate_seeds])
        seed_bounds = np.array([seed['box_xyxy'] for seed in gate_seeds])
        for directory in ['masks', 'foreground_masks', 'review']:
            (output/directory).mkdir()
        frame_records = []
        tiles = []
        for row in rows:
            index = row['index']
            raw_masks = [np.array(Image.open(output/'raw_masks'/track['id']/f'{index:06d}.png')) > 0 for track in config['tracks']]
            union = np.logical_or.reduce(raw_masks)
            Image.fromarray(union.astype(np.uint8)*255).save(output/'masks'/f'{index:06d}.png')
            right = np.array(Image.open(output/'raw_masks/right_hand'/f'{index:06d}.png')) > 0
            bounds = np.array([np.interp(index, seed_indices, seed_bounds[:, k]) for k in range(4)]).round().astype(int)
            x0, y0, x1, y1 = bounds
            corridor = np.zeros_like(right)
            corridor[y0:y1, x0:x1] = True
            foreground = cv2.erode((right & corridor).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            Image.fromarray(foreground.astype(np.uint8)*255).save(output/'foreground_masks'/f'{index:06d}.png')
            target_mask_path = case/'segmentation/battery-v1/raw_masks/out_left'/f'{index:06d}.png'
            battery = np.array(Image.open(target_mask_path)) > 0
            frame_records.append({
                'sample_index': index, 'timestamp_s': row['timestamp_s'],
                'raw_hand_union': f'masks/{index:06d}.png',
                'foreground_thumb_candidate': f'foreground_masks/{index:06d}.png',
                'raw_union_area_pixels': int(union.sum()),
                'foreground_area_pixels': int(foreground.sum()),
                'foreground_thumb_corridor_xyxy': bounds.tolist(),
                'hand_union_intersection_target_sam_pixels': int((union & battery).sum()),
                'foreground_intersection_target_sam_pixels': int((foreground & battery).sum()),
                'intersection_interpretation': 'Diagnostic boundary disagreement, not measured segmentation error; target mask is not used in construction',
                'status': 'unsupported_motion_blur' if index == 52 else 'assistant_reviewed_occlusion_candidate',
            })
            original = np.array(Image.open(case/row['file']).convert('RGB'))
            overlay = original.copy().astype(float)
            for mask, color in zip(raw_masks, [[32, 180, 255], [230, 80, 210]]):
                overlay[mask] = overlay[mask]*0.70 + np.array(color)*0.30
            overlay[foreground] = overlay[foreground]*0.35 + np.array([255, 180, 40])*0.65
            rendered = Image.fromarray(overlay.astype(np.uint8))
            draw = ImageDraw.Draw(rendered)
            draw.rectangle((4, 4, 590, 40), fill='black')
            draw.text((12, 10), f'Sample {index:02d} / {row["timestamp_s"]:.3f}s; cyan/right, magenta/left, gold/front-thumb candidate', fill='white')
            rendered.save(output/'review'/f'{index:06d}.jpg', quality=91)
            tile = rendered.resize((480, 270))
            tiles.append(tile)
        for sheet_index in range(3):
            subset = tiles[sheet_index*18:(sheet_index+1)*18]
            sheet = Image.new('RGB', (480*3, 270*6), (20, 24, 32))
            for position, tile in enumerate(subset):
                sheet.paste(tile, (position % 3*480, position // 3*270))
            sheet.save(output/f'contact-{sheet_index+1:02d}.jpg', quality=92)
        write(output/'frames.json', {'time_domain': 'source_video_presentation_seconds', 'frames': frame_records})
        metadata.update(status='complete', wall_time_s=round(time.monotonic()-start, 3), raw_mask_count=len(rows)*len(config['tracks']), frame_count=len(rows), review_status='pending_assistant_review')
        write(output/'metadata.json', metadata)
        print(json.dumps(metadata, indent=2))
    except Exception as error:
        metadata.update(status='failed', error=str(error), wall_time_s=round(time.monotonic()-start, 3))
        write(output/'metadata.json', metadata)
        raise


if __name__ == '__main__':
    main()
