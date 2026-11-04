#!/usr/bin/env python3
"""Hosted SAM 3.1 video propagation from automatic model boxes, with exact PTS joins.

The submitted derivative has one ordered sampled image per encoded second. Its
timestamps are not substituted for the original source presentation timestamps.
H264 is lossless after the disclosed RGB-to-YUV420 conversion; decoded YUV frame
hashes must exactly match the ordered encoding reference before any paid call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stage2_meta_segment import SLUG, bbox, confined, load, sha, validate_config, write
import meta_sam_api as sam

MAX_FRAMES = 120
MAX_DIMENSION = 960
MAX_VIDEO_BYTES = 20 * 1024 * 1024


def verify_rows(case, rows, frame_document, manifest):
    if frame_document.get('time_domain') != sam.TIME_DOMAIN:
        raise ValueError('Original source presentation time domain is required')
    if not 1 <= len(rows) <= MAX_FRAMES:
        raise ValueError('SAM video input requires one to 120 actual sampled source frames')
    source_hash = manifest['source']['sha256']
    width, height = rows[0]['width'], rows[0]['height']
    if max(width, height) > MAX_DIMENSION or width % 2 or height % 2:
        raise ValueError('Video input requires even native dimensions no larger than 960; no implicit resize')
    previous = -1
    sources = []
    for index, row in enumerate(rows, 1):
        if row['index'] != index or (row['width'], row['height']) != (width, height):
            raise ValueError('Frame rows must retain ordered indices and one native coordinate system')
        source = {'video_sha256': source_hash, 'sample_index': index, 'timestamp_s': row['timestamp_s'],
                  'time_domain': sam.TIME_DOMAIN, 'source_case': case.name, 'source_sample_index': index}
        sam._source(source)
        if source['timestamp_s'] <= previous:
            raise ValueError('Original source timestamps must strictly increase')
        previous = source['timestamp_s']
        path = confined(case / row['file'], case)
        with Image.open(path) as image:
            if image.size != (width, height) or image.getexif().get(274, 1) not in (None, 1):
                raise ValueError('Frame pixels/orientation differ from the source coordinate record')
        sources.append(source)
    return width, height, sources


def command(argv, timeout=180):
    return subprocess.run(list(map(str, argv)), check=True, capture_output=True, timeout=timeout)


def prepare_video(case, rows, directory, sources):
    """Encode and independently decode one bounded H264 video; no network."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    width, height = rows[0]['width'], rows[0]['height']
    video = directory / 'sampled-1fps.mp4'
    frame_records = []
    with tempfile.TemporaryDirectory(prefix='encoding-work-', dir=directory) as temporary:
        work = Path(temporary)
        for offset, (row, source) in enumerate(zip(rows, sources)):
            path = confined(case / row['file'], case)
            with Image.open(path) as image:
                rgb = image.convert('RGB')
                pixels = np.asarray(rgb)
                rgb.save(work / f'{offset:06d}.png')
            frame_records.append({'submitted_frame_index': offset, 'submitted_timestamp_s': float(offset),
                                  'source': source, 'source_image_file': row['file'],
                                  'source_image_sha256': sha(path),
                                  'source_rgb_pixels_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()})
        reference, decoded = work / 'reference.yuv', work / 'decoded.yuv'
        command(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-framerate', '1', '-start_number', '0',
                 '-i', work / '%06d.png', '-frames:v', len(rows), '-pix_fmt', 'yuv420p',
                 '-f', 'rawvideo', reference])
        command(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pixel_format', 'yuv420p',
                 '-video_size', f'{width}x{height}', '-framerate', '1', '-i', reference, '-frames:v', len(rows),
                 '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-qp', '0', '-pix_fmt', 'yuv420p',
                 '-movflags', '+faststart', video])
        if not 0 < video.stat().st_size <= MAX_VIDEO_BYTES:
            raise ValueError('Lossless sampled video exceeds the bounded 20 MiB upload allowance; no paid request made')
        probe = json.loads(command(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                                    '-show_streams', '-show_frames', '-show_entries',
                                    'stream=codec_name,width,height,nb_read_frames,avg_frame_rate:frame=best_effort_timestamp_time',
                                    '-of', 'json', video]).stdout)
        streams, frames = probe.get('streams', []), probe.get('frames', [])
        if (len(streams) != 1 or streams[0].get('codec_name') != 'h264'
                or (streams[0].get('width'), streams[0].get('height')) != (width, height)
                or int(streams[0].get('nb_read_frames', -1)) != len(rows) or len(frames) != len(rows)
                or streams[0].get('avg_frame_rate') != '1/1'):
            raise ValueError('Encoded video codec, dimensions, frame count or frame rate differs from the plan')
        for offset, frame in enumerate(frames):
            if abs(float(frame.get('best_effort_timestamp_time', -1)) - offset) > 1e-6:
                raise ValueError('Encoded frame order/timestamps differ from the explicit source mapping')
        command(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', video, '-frames:v', len(rows),
                 '-pix_fmt', 'yuv420p', '-f', 'rawvideo', decoded])
        bytes_per_frame = width * height * 3 // 2
        if reference.stat().st_size != len(rows) * bytes_per_frame or decoded.stat().st_size != reference.stat().st_size:
            raise ValueError('Decoded frame plane count differs from the encoding reference')
        with reference.open('rb') as expected, decoded.open('rb') as actual:
            for record in frame_records:
                a, b = expected.read(bytes_per_frame), actual.read(bytes_per_frame)
                if a != b:
                    raise ValueError('Decoded video frame pixels/order differ from the lossless encoding reference')
                record['reference_yuv420_pixels_sha256'] = hashlib.sha256(a).hexdigest()
                record['decoded_yuv420_pixels_sha256'] = hashlib.sha256(b).hexdigest()
    manifest = {'status': 'verified_exact_decoded_frame_order_and_YUV_pixels',
                'video': video.name, 'video_sha256': sha(video), 'video_bytes': video.stat().st_size,
                'codec': 'h264', 'codec_settings': 'libx264 veryfast qp=0 yuv420p', 'fps': 1,
                'width': width, 'height': height, 'frame_count': len(rows), 'frames': frame_records,
                'original_pts_replaced_by_encoded_pts': False, 'resize_performed': False,
                'colour_provenance': 'Original saved RGB pixels converted to YUV420 by ffmpeg; chroma subsampling/colour conversion is not lossless RGB. H264 preserves that YUV reference exactly.',
                'source_mapping': 'Submitted frame N maps only to frames[N].source original PTS; no interpolation or new observations.'}
    write(directory / 'encoding.json', manifest)
    return video, manifest


def run_case(case, config_path, run, *, client=None, segmenter=None, video_preparer=None):
    """Run one multi-object video request; injection supports offline fixtures."""
    case = Path(case).resolve()
    config_path = Path(config_path).resolve()
    if not SLUG.fullmatch(case.name) or not SLUG.fullmatch(run):
        raise ValueError('Invalid case/run slug')
    frame_document, manifest, cfg = load(case / 'frames/index.json'), load(case / 'manifest.json'), load(config_path)
    rows = frame_document['frames']
    validate_config(cfg, rows)
    width, height, sources = verify_rows(case, rows, frame_document, manifest)
    out = case / 'segmentation' / run
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, out / 'annotations.json')
    started = time.monotonic()
    registry = [{'sam_object_id': number, 'track_id': track['id']} for number, track in enumerate(cfg['tracks'])]
    objects = [{'id': number, 'frame_index': prompt['sample_index'] - 1, 'box_xyxy': prompt['box_xyxy']}
               for number, track in enumerate(cfg['tracks']) for prompt in track['prompts']]
    write(out / 'object-registry.json', {'objects': registry, 'prompts': objects,
                                        'ids_scope': 'SAM object IDs bound to the unchanged model tracklets; physical identity unverified'})
    metadata = {'status': 'running', 'model': 'sam-3.1', 'device': 'hosted_Meta_Model_API', 'provider': 'Meta Model API',
                'source_sha256': manifest['source']['sha256'], 'seed_origin': cfg['seed_origin'],
                'checkpoint_sha256': None, 'checkpoint_identity': 'provider_managed_not_disclosed',
                'human_confirmed': False, 'manual_seed_boxes_used': False, 'manual_geometry_used': False,
                'config_sha256': sha(config_path), 'scope': 'SAM_video_propagation_from_model_generated_boxes',
                'sam_video_propagation_executed': False, 'accuracy_measured': False,
                'hosted_image_request_count': 0, 'hosted_video_request_count': 0, 'old_sam2_audits_reused': False,
                'review_status': 'pending_fresh_Meta_mask_audit'}
    write(out / 'metadata.json', metadata)
    try:
        if not objects:
            write(out / 'tracks.json', {'tracks': [], 'outside_track_ranges': 'not_assessed_not_evidence_of_absence'})
            metadata.update(status='complete', outcome='completed_zero_model_proposed_tracks', track_count=0,
                            mask_count=0, native_instance_count=0, candidate_visible_count=0, model_call_made=False,
                            learned_inference_executed=False, estimated_cost_usd='0.000000', wall_time_s=round(time.monotonic() - started, 3))
            write(out / 'metadata.json', metadata)
            return metadata
        if segmenter is None:
            sam.parser_preflight()
            segmenter = sam.segment_video_boxes
        video, encoding = (video_preparer or prepare_video)(case, rows, out / 'video-input', sources)
        if client is None:
            from meta_api import MetaClient
            client = MetaClient()
        result_directory = out / 'api/result'
        metadata.update(hosted_video_request_count=1, encoding_manifest='video-input/encoding.json')
        write(out / 'metadata.json', metadata)
        result = segmenter(video, objects, result_directory,
                           client.sam_callback(out / 'api/accounting', video_frames=len(rows)),
                           width=width, height=height, frame_sources=sources,
                           prompt_origin=cfg['seed_origin'] + '; model box observations', propagation_direction='both')
        detections, unmapped = {}, []
        identifiers = {str(item['sam_object_id']) for item in registry}
        for item in result['instances']:
            frame = item.get('frame_index')
            identifier = str(item.get('instance_id'))
            if type(frame) is not int or not 0 <= frame < len(rows) or item.get('source') != sources[frame]:
                raise ValueError('Returned SAM frame has no exact original-source mapping')
            if identifier not in identifiers:
                unmapped.append({'instance_id': identifier, 'frame_index': frame})
                continue
            key = (identifier, frame)
            if key in detections:
                raise ValueError('Duplicate SAM object/frame result cannot be silently merged')
            path = confined(result_directory / item['mask'], result_directory)
            pixels = np.asarray(Image.open(path).convert('L'))
            if pixels.shape != (height, width) or not set(np.unique(pixels).tolist()).issubset({0, 1, 255}):
                raise ValueError('Returned SAM mask is not binary in the original coordinate dimensions')
            if item.get('mask_sha256') and item['mask_sha256'] != sha(path):
                raise ValueError('Returned SAM mask changed after decoding')
            detections[key] = (item, path, pixels > 0)
        tracks = []
        for binding, track in zip(registry, cfg['tracks']):
            records = []
            lo, hi = track['sample_range']
            for index in range(lo, hi + 1):
                row = rows[index - 1]
                returned = detections.get((str(binding['sam_object_id']), index - 1))
                withheld = any(low <= index <= high for low, high in track.get('withhold_ranges', []))
                reasons = []
                if returned is None:
                    reasons.append('no_returned_SAM_video_mask_at_this_source_sample')
                elif not returned[2].any():
                    reasons.append('returned_empty_SAM_video_mask')
                if withheld:
                    reasons.append('explicit_model_withhold_range')
                supported = not reasons
                mask = returned[2] if supported else np.zeros((height, width), dtype=bool)
                path = out / 'raw_masks' / track['id'] / f'{index:06d}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                if supported:
                    shutil.copyfile(returned[1], path)
                else:
                    Image.fromarray(mask.astype(np.uint8)).save(path)
                native = None if returned is None else {'instance_id': str(returned[0]['instance_id']),
                                                       'frame_index': index - 1,
                                                       'mask_file': str(returned[1].relative_to(out)),
                                                       'instance_id_scope': 'SAM_video_response_track_not_verified_physical_identity'}
                records.append({'sample_index': index, 'timestamp_s': row['timestamp_s'], 'raw_mask_file': str(path.relative_to(out)),
                                'raw_area_pixels': int(mask.sum()), 'raw_bbox_xyxy': bbox(mask),
                                'raw_mask_sha256': sha(path), 'source_frame_sha256': sha(case / row['file']),
                                'status': 'candidate_visible' if supported else 'withheld_due_to_occlusion_or_ambiguity',
                                'identity_status': track['identity_status'], 'api_instance': native,
                                'association': {'method': 'explicit_SAM_video_object_id_to_model_tracklet',
                                                'accepted': supported, 'reasons': reasons, 'physical_identity_verified': False},
                                'zero_placeholder_for_missing_or_withheld': not supported,
                                'native_returned_instance_preserved': returned is not None})
            record = {'track_id': track['id'], 'candidate_entity_ids': track['candidate_entity_ids'],
                      'sample_range': [lo, hi], 'frames': records, 'sam_object_id': binding['sam_object_id'],
                      'physical_identity_verified': False}
            write(out / 'tracks' / f'{track["id"]}.json', record)
            tracks.append(record)
        write(out / 'tracks.json', {'tracks': tracks, 'outside_track_ranges': 'not_assessed_not_evidence_of_absence'})
        write(out / 'unmapped-returned-ids.json', {'instances': unmapped, 'all_raw_results_preserved': True})
        metadata.update(status='complete', sam_video_propagation_executed=True, learned_inference_executed=True,
                        track_count=len(tracks), mask_count=sum(len(item['frames']) for item in tracks),
                        native_instance_count=len(result['instances']), unmapped_native_instance_count=len(unmapped),
                        candidate_visible_count=sum(row['status'] == 'candidate_visible' for item in tracks for row in item['frames']),
                        submitted_frame_count=len(rows), submitted_video_sha256=encoding['video_sha256'],
                        wall_time_s=round(time.monotonic() - started, 3),
                        limitations=['Numeric SAM IDs are observation tracks, not verified physical identity across occlusion.',
                                     'Missing/withheld outputs are explicit zero placeholders; no interpolation, old masks or hidden completion.',
                                     '2D segmentation does not establish 3D shape, contact or seating.'])
        write(out / 'metadata.json', metadata)
        return metadata
    except Exception as exc:
        metadata.update(status='failed', error_type=type(exc).__name__, wall_time_s=round(time.monotonic() - started, 3),
                        retry_policy='no_automatic_retry_preserve_request_and_encoding_receipts')
        write(out / 'metadata.json', metadata)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run', default='flow-components-1')
    args = parser.parse_args()
    if not SLUG.fullmatch(args.case) or not SLUG.fullmatch(args.run):
        parser.error('Invalid case/run slug')
    print(json.dumps(run_case(ROOT / 'data' / args.case, args.config, args.run), indent=2))


if __name__ == '__main__':
    main()
