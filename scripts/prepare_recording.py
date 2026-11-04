#!/usr/bin/env python3
"""Preserve a phone recording and make an upright SDR derivative with verified PTS."""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twin import case_path, sha256, write_json


def call(args):
    return subprocess.run([str(x) for x in args], check=True, capture_output=True, text=True)


def probe(path):
    return json.loads(call(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', path]).stdout)


def times(path):
    data = json.loads(call(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
                            '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', path]).stdout)
    return [float(f['best_effort_timestamp_time']) for f in data['frames']]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True)
    p.add_argument('--file', type=Path, required=True)
    p.add_argument('--fps', type=float, default=1)
    p.add_argument('--max-dimension', type=int, default=960)
    p.add_argument('--ffmpeg', type=Path, default=Path(shutil.which('ffmpeg') or 'ffmpeg'))
    a = p.parse_args()
    source = a.file.resolve()
    if not source.is_file() or not 128 <= a.max_dimension <= 1920 or not 0 < a.fps <= 30:
        p.error('Valid file, 128–1920 max dimension and 0–30 review fps required')
    case = case_path(a.case)
    if case.exists():
        p.error('Case already exists; choose a fresh name to preserve evidence')
    out = case / 'source'
    out.mkdir(parents=True)
    original = out / ('original' + source.suffix.lower())
    shutil.copy2(source, original)
    source_hash = sha256(source)
    if sha256(original) != source_hash:
        raise RuntimeError('Original copy hash mismatch')
    info = probe(original)
    stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
    hdr = stream.get('color_transfer') in {'arib-std-b67', 'smpte2084'}
    filters = [f'scale={a.max_dimension}:{a.max_dimension}:force_original_aspect_ratio=decrease:force_divisible_by=2']
    if hdr:
        filters += ['zscale=t=linear:npl=100', 'format=gbrpf32le',
                    'tonemap=tonemap=hable:desat=0', 'zscale=p=bt709:t=bt709:m=bt709:r=tv']
    filters += ['format=yuv420p', 'setsar=1']
    video = out / 'video.mp4'
    if hdr and 'zscale' not in call([a.ffmpeg, '-hide_banner', '-filters']).stdout:
        raise RuntimeError('HDR preparation requires an explicit FFmpeg executable with zscale')
    command = [a.ffmpeg, '-hide_banner', '-nostdin', '-n', '-i', original, '-map', '0:v:0',
               '-an', '-sn', '-dn', '-vf', ','.join(filters), '-c:v', 'libx264',
               '-crf', '18', '-preset', 'fast', '-fps_mode', 'passthrough', '-enc_time_base:v', '1:90000',
               '-video_track_timescale', '90000', '-movflags', '+faststart',
               '-metadata:s:v:0', 'rotate=0']
    if hdr:
        command += ['-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709']
    command += [video]
    try:
        log = call(command)
    except subprocess.CalledProcessError as exc:
        (case / 'normalization-error.log').write_text(exc.stderr)
        raise
    (case / 'normalization.log').write_text(log.stderr)
    before, after = times(original), times(video)
    if not before or len(before) != len(after):
        raise RuntimeError('Frame count changed during normalization; do not use this derivative')
    max_error = max(abs(x-y) for x,y in zip(before, after))
    if max_error > 0.000012:
        raise RuntimeError(f'Source PTS changed by {max_error}s; do not use this derivative')
    derived_info = probe(video)
    record = {'schema_version': '0.1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'original_file': str(original.relative_to(case)), 'original_download_file': str(source),
              'original_sha256': source_hash, 'original_probe': info,
              'derivative_file': str(video.relative_to(case)), 'derivative_sha256': sha256(video),
              'derivative_probe': derived_info, 'ffmpeg_command': [str(x) for x in command],
              'ffmpeg_version': call([a.ffmpeg, '-version']).stdout.splitlines()[0],
              'upright_rotation': 'FFmpeg default display-matrix autorotation; derivative rotation cleared',
              'appearance': 'HLG/PQ to SDR BT.709, Hable tone mapping' if hdr else 'SDR resize, no HDR tone mapping',
              'hdr_tone_mapped': hdr, 'geometric_warp': False,
              'audio': 'original audio retained in original file; diagnostic derivative video only',
              'source_pts_verified': True, 'native_video_frames': len(before),
              'max_pts_difference_s': max_error, 'source_pts_s': before,
              'rights_provenance': 'user says they recorded this clip; no distribution permission inferred'}
    write_json(case / 'normalization.json', record)
    call([sys.executable, ROOT / 'twin.py', 'prepare', '--case', a.case, '--file', video,
          '--fps', a.fps, '--max-width', a.max_dimension])
    manifest_path = case / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['source']['original_sha256'] = source_hash
    manifest['source']['original_file'] = str(original.relative_to(case))
    manifest['source']['normalization'] = 'normalization.json'
    manifest['source']['rights_status'] = 'user_recorded_private_test_no_distribution_authorized'
    write_json(manifest_path, manifest)
    print(json.dumps({'case': str(case), 'duration_s': manifest['source']['video']['duration_s'],
                      'sample_count': manifest['frame_count'], 'frame_pts_max_error_s': max_error,
                      'hdr_tone_mapped': hdr, 'review': str(case / 'review.html')}, indent=2))


if __name__ == '__main__':
    main()
