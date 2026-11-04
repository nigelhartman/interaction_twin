#!/usr/bin/env python3
"""Version 3: genuine SAM3D milestone, local 58-second film from actual saved outputs. No inference or publication.

Editorial cards, camera orbits and the authored robot rehearsal are labelled.
Original GLBs, poses, source frames and programs are never modified.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/trailer-20261102'
MEDIA = ROOT / 'showcase/media'
PREFIX = 'interaction-library-trailer-20261102'
PROGRAM = 'data/flow-rigid-parts-jabra-20261031-v1/flow-procedure-1/interaction.json'
PROGRAM_SHA = '31708efdb18a645052d03760eaa9c8f9d03efea60c340d0859510cf38024bad5'
W, H, FPS = 1920, 1080, 24
BG, INK, MUTED = '#f4f2e9', '#152d24', '#607065'
LIME, LINE, SOFT, AMBER = '#dcf3a1', '#d7dfd3', '#e8ede2', '#c78c32'
PART_COLORS = {'case_base': '#34a89e', 'case_lid': '#e49756', 'earbud_a': '#7a8fce', 'earbud_b': '#bb7cad'}
CHAPTERS = [
    dict(seconds=5, title='A demonstration becomes an asset',
         narration='Turn a demonstration into reusable assets for robot task authoring.'),
    dict(seconds=7, title='Separate visible components',
         narration='Start with an ordinary recording. Meta Sam three point one tracks the visible components.'),
    dict(seconds=7, title='Reason about the procedure',
         narration='Muse reasons about relationships and steps, and cites the frames behind each proposal.'),
    dict(seconds=8, title='The first SAM 3D part, back in its source',
         narration='Sam three D builds our first earbud candidate. Its own layout already projects back into the source frame.'),
    dict(seconds=8, title='Keep an inspectable interaction program',
         narration='Each interaction carries a descriptive program: components, actions, evidence, and the gaps still unknown.'),
    dict(seconds=8, title='Discover and retrieve through MCP',
         narration='Through M C P, agents discover the library, inspect procedures, and retrieve models and source frames.'),
    dict(seconds=7, title='Author a robot task',
         narration='Stage robot tasks with those assets. This rehearsal is illustrative. Physical contact still needs validation.'),
    dict(seconds=8, title='One recording. A reusable interaction.',
         narration='One recording. Parts, actions, and evidence. A reusable starting point for people and agents building robot tasks.'),
]
DURATION = sum(c['seconds'] for c in CHAPTERS)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def local(value):
    if not isinstance(value, str) or any(x in value for x in ('\\', '\0', ':', '?', '#')):
        raise ValueError('Only declared project-local artifacts are eligible')
    path = (ROOT / value.removeprefix('/files/')).absolute()
    path.relative_to(ROOT)
    if '..' in path.relative_to(ROOT).parts or any(p.startswith('.') for p in path.relative_to(ROOT).parts):
        raise ValueError('Unsafe artifact path')
    if path.resolve() != path or not path.is_file():
        raise ValueError('Artifact must be an existing regular nonsymlink file')
    return path


def record(path, kind, **extra):
    path = Path(path).absolute()
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), kind=kind, **extra)


def fresh(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def checked(row):
    path = local(row['path'])
    if sha(path) != row['sha256']:
        raise ValueError('Frozen artifact changed: ' + row['path'])
    return path


@lru_cache(maxsize=128)
def font(size, bold=False, mono=False):
    name = '/System/Library/Fonts/Menlo.ttc' if mono else '/System/Library/Fonts/Supplemental/Arial' + (' Bold' if bold else '') + '.ttf'
    return ImageFont.truetype(name, size)


def text(image, xy, value, size=32, color=INK, bold=False, spacing=10, mono=False):
    ImageDraw.Draw(image).multiline_text(xy, value, fill=color, font=font(size, bold, mono), spacing=spacing)


def chip(image, xy, value, color=INK, fill=LIME, size=20):
    x, y = xy
    width = round(font(size, True).getlength(value)) + 32
    ImageDraw.Draw(image).rounded_rectangle((x, y, x + width, y + 40), 12, fill=fill)
    text(image, (x + 16, y + 8), value, size, color, True)
    return width


def line(image, a, b, color=LINE, width=2):
    ImageDraw.Draw(image).line((*a, *b), fill=color, width=width)


def card(image, source, box, background=SOFT, cover=False, radius=24, border=None):
    x, y, w, h = map(round, box)
    pane = Image.new('RGB', (w, h), background)
    source = source.convert('RGB')
    crop = (ImageOps.fit if cover else ImageOps.contain)(source, (w, h), Image.Resampling.LANCZOS)
    pane.paste(crop, ((w - crop.width) // 2, (h - crop.height) // 2))
    mask = Image.new('L', (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius, fill=255)
    image.paste(pane, (x, y), mask)
    if border:
        ImageDraw.Draw(image).rounded_rectangle((x, y, x + w - 1, y + h - 1), radius, outline=border, width=2)


def base(kicker=None, title=None, subtitle=None, dark=False):
    color = BG if dark else INK
    canvas = Image.new('RGB', (W, H), INK if dark else BG)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((66, 50, 99, 83), 10, fill=LIME if dark else INK)
    draw.line((74, 67, 91, 67), fill=INK if dark else LIME, width=3)
    draw.ellipse((72, 63, 80, 71), fill=INK if dark else LIME)
    draw.ellipse((86, 63, 94, 71), fill=INK if dark else LIME)
    text(canvas, (117, 51), 'INTERACTION TWIN', 25, color, True)
    if kicker:
        text(canvas, (68, 122), kicker.upper(), 21, '#b8ccad' if dark else MUTED, True)
    if title:
        text(canvas, (62, 160), title, 68, color, True, spacing=4)
    if subtitle:
        text(canvas, (68, 253), subtitle, 27, '#b8ccad' if dark else MUTED)
    return canvas


def footer(image, message, dark=False):
    line(image, (68, 1000), (1852, 1000), '#3d5547' if dark else LINE)
    text(image, (68, 1020), message, 21, '#bfceba' if dark else MUTED)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    assets = OUT / 'assets'
    assets.mkdir()
    program_path = local(PROGRAM)
    if sha(program_path) != PROGRAM_SHA:
        raise ValueError('Actual saved program changed')
    program = read(program_path)
    source = local(program['source']['path'])
    if sha(source) != program['source']['sha256']:
        raise ValueError('Actual source video changed')
    rows = []
    all_boxes = []
    for evidence in program['evidence']:
        if not 11.99 <= evidence['timestamp_s'] <= 19.01:
            continue
        source_path = local(evidence['frame_path'])
        if sha(source_path) != evidence['frame_sha256']:
            raise ValueError('Cited source frame changed')
        pixels = np.array(Image.open(source_path).convert('RGB'))
        masks = []
        for observation in evidence['observations']:
            path = local(observation['mask_path'])
            if sha(path) != observation['mask_sha256']:
                raise ValueError('Cited SAM mask changed')
            mask = np.asarray(Image.open(path).convert('L')) > 127
            ys, xs = np.nonzero(mask)
            if len(xs):
                all_boxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
            color = np.array(tuple(int(PART_COLORS[observation['track_id']][j:j + 2], 16) for j in (1, 3, 5)))
            pixels[mask] = np.rint(.59 * pixels[mask] + .41 * color).astype(np.uint8)
            edge = cv2.morphologyEx(mask.astype('uint8'), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
            pixels[edge] = color
            masks.append(record(path, 'actual_saved_SAM31_video_mask', component=observation['track_id']))
        output = assets / (evidence['id'] + '-overlay.jpg')
        Image.fromarray(pixels).save(output, quality=97)
        rows.append(dict(id=evidence['id'], timestamp_s=evidence['timestamp_s'],
                         source=record(source_path, 'actual_exact_saved_source_frame'), masks=masks,
                         overlay=record(output, 'mask_tint_and_edges_only')))
    boxes = np.asarray(all_boxes)
    crop = [max(0, int(boxes[:, 0].min()) - 45), max(0, int(boxes[:, 1].min()) - 60),
            min(pixels.shape[1], int(boxes[:, 2].max()) + 45), min(pixels.shape[0], int(boxes[:, 3].max()) + 60)]
    inherited_path = ROOT / 'output/showcase-studio-20261031/source-manifest.json'
    inherited = read(inherited_path)
    for row in inherited['inputs'] + inherited['native_assets'] + inherited['rendered_asset_inputs']:
        checked(row)
    for asset in inherited['native_assets']:
        checked(asset['source_image'])
        checked(asset['source_mask'])
    robot_dir = OUT / 'captures/robot-v2'
    robot_receipt = read(robot_dir / 'capture-receipt.json')
    if len(robot_receipt['frames']) != 32 or PROGRAM.split('/')[1] not in robot_receipt['url']:
        raise ValueError('Actual same-program robot UI capture required')
    robot = [record(Path(row['path']), 'actual_UI_authored_rehearsal_screenshot',
                    authored_progress=row['authored_progress']) for row in robot_receipt['frames']]
    handoff_path = ROOT / 'data/mcp-fresh-jabra-motion-handoff-20261101-v1/sdk-trace.json'
    handoff = read(handoff_path)
    if handoff['source_program_sha256'] != PROGRAM_SHA or len(handoff['calls']) != 36:
        raise ValueError('Actual same-program 36-call MCP trace required')
    program_context = []
    for evidence_id in ('source_000021', 'source_000033', 'source_000053', 'source_000058'):
        row = next(r for r in program['evidence'] if r['id'] == evidence_id)
        path = local(row['frame_path'])
        if sha(path) != row['frame_sha256']:
            raise ValueError('Program citation changed')
        program_context.append(dict(id=evidence_id, timestamp_s=row['timestamp_s'], image=record(path, 'actual_program_citation')))
    snapshot = OUT / 'builder-at-prepare.py'
    shutil.copyfile(Path(__file__), snapshot)
    fresh(OUT / 'preparation.json', dict(format='interaction-library-trailer-preparation', version=1,
        program=record(program_path, 'actual_schema_validated_interaction_program'), source=record(source, 'actual_upright_recording'),
        sequence=rows, fixed_editorial_crop_xyxy=crop, context=program_context,
        inherited_render_manifest=record(inherited_path, 'frozen_native_render_provenance'),
        inherited_render_manifest_content=inherited, robot=robot,
        robot_capture_receipt=record(robot_dir / 'capture-receipt.json', 'actual_CUA_capture_receipt'),
        mcp_trace=record(handoff_path, 'actual_scripted_SDK_run_not_LLM'), builder=record(snapshot, 'builder_at_prepare'),
        chapters=CHAPTERS, duration_s=DURATION, new_api_calls=0, model_inference_calls=0,
        policy={'source_and_GLBS_unchanged': True, 'native_gauges_independent': True,
                'camera_orbits_editorial': True, 'motion_between_observations_not_invented': True,
                'robot_rehearsal_authored_not_physics': True, 'physical_accuracy_verified': False,
                'SAM3D_result_included': False, 'private_local_only': True}))
    print(json.dumps({'status': 'prepared', 'source_samples': len(rows), 'duration_s': DURATION, 'crop': crop}), flush=True)


def speech(take='speech-v2'):
    if not take.replace('-', '').isalnum():
        raise ValueError('Simple versioned local speech take required')
    directory = OUT / take
    directory.mkdir()
    rows = []
    for i, chapter in enumerate(CHAPTERS):
        path = directory / f'{i + 1:02d}.txt'
        path.write_text(chapter['narration'] + '\n')
        output = directory / f'{i + 1:02d}.aiff'
        subprocess.run(['say', '-v', 'Samantha', '-r', '172', '-f', str(path), '-o', str(output)], check=True)
        wav = output.with_suffix('.wav')
        subprocess.run(['ffmpeg', '-hide_banner', '-v', 'error', '-i', str(output), '-ar', '48000', '-ac', '1', str(wav)], check=True)
        probe = read_probe(wav)
        seconds = float(probe.get('format', {}).get('duration', 0))
        if seconds < .2:
            raise ValueError('Bundled local speech returned empty audio; keep failed attempt')
        # Tiny explicit pitch-preserving tempo adjustment only when needed.
        usable = chapter['seconds'] - .8
        tempo = max(1., seconds / usable)
        if tempo > 1.3:
            raise ValueError('Narration is too dense; revise a new take')
        target = wav
        if tempo > 1.001:
            target = directory / f'{i + 1:02d}-tempo.wav'
            subprocess.run(['ffmpeg', '-hide_banner', '-v', 'error', '-i', str(wav), '-af', f'atempo={tempo:.9f}', str(target)], check=True)
        rows.append(dict(chapter=i + 1, text=record(path, 'exact_local_speech_text'),
                         original=record(output, 'bundled_macOS_voice'), audio=record(target, 'local_voice_PCM'),
                         requested_rate=172, original_duration_s=seconds, tempo=tempo, lead_s=.35))
        print(json.dumps({'speech': i + 1, 'seconds': seconds, 'tempo': tempo}), flush=True)
    fresh(OUT / 'speech-v5.json', dict(voice='Samantha', local_only=True, rows=rows, remote_voice_service=False))


def native_render():
    """New display renders of this exact program's original three native GLBs."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'scripts'))
    from portable_twin import decode_glb_bytes
    from build_showcase_media import centered_mesh
    program_path = local(PROGRAM)
    if sha(program_path) != PROGRAM_SHA:
        raise ValueError('Exact current program required')
    directory = OUT / 'native-renders-v1'
    directory.mkdir()
    rows = []
    for asset in read(program_path)['assets']:
        path = checked(asset)
        decoded = decode_glb_bytes(path.read_bytes())
        target = directory / asset['track_id']
        target.mkdir()
        frames = []
        for n in range(48):
            image = centered_mesh(decoded['primitives'], (620, 510), -35 + n * 360 / 48)
            pane = Image.new('RGB', image.size, '#e9efeb')
            pane.paste(image, (0, 0), image)
            output = target / f'{n:03d}.jpg'
            pane.save(output, quality=96)
            frames.append(record(output, 'editorial_native_camera_render'))
        rows.append(dict(component=asset['track_id'], asset=record(path, 'unchanged_current_program_TripoSR_asset'), frames=frames))
        print(json.dumps({'native_display_render': asset['track_id'], 'frames': 48}), flush=True)
    fresh(directory / 'manifest.json', dict(program=record(program_path, 'actual_current_program'), components=rows,
        renderer=record(ROOT / 'scripts/build_showcase_media.py', 'existing_painter_display_only'),
        decoder=record(ROOT / 'portable_twin.py', 'existing_native_GLb_decoder'),
        camera_recipe={'euler_xyz_degrees': [-18, '-35 + n*360/48', 12], 'editorial_camera': True},
        native_geometry_changed=False, common_scale_or_assembly_established=False, model_inference_calls=0))


def read_probe(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)], text=True))


def original_score(path):
    """Original deterministic instrumental from oscillators; no external samples."""
    sr, beat = 48000, .625
    count = int(DURATION * sr)
    mix = np.zeros((count, 2), np.float64)
    rng = np.random.default_rng(20261102)
    chords = [(45, [57, 60, 64]), (48, [55, 60, 64]), (41, [53, 57, 60]), (43, [55, 59, 62])]
    def note(start, length, midi, amp, pan=0., pluck=False):
        offset = int(start * sr)
        n = min(int(length * sr), count - offset)
        if n <= 0:
            return
        t = np.arange(n) / sr
        f = 440 * 2 ** ((midi - 69) / 12)
        signal = np.sin(2 * np.pi * f * t) + .16 * np.sin(2 * np.pi * 2.002 * f * t)
        env = (1 - np.exp(-t * 100)) * np.exp(-t * 5) if pluck else np.minimum(t / .25, 1) * np.minimum((n / sr - t) / .6, 1)
        a = (pan + 1) * np.pi / 4
        mix[offset:offset + n] += signal[:, None] * np.maximum(env, 0)[:, None] * amp * np.array([np.cos(a), np.sin(a)])
    for bar, start in enumerate(np.arange(0, DURATION, 4 * beat)):
        bass, chord = chords[bar % 4]
        for k, midi in enumerate(chord):
            note(start, 4 * beat + .5, midi, .045, [-.45, 0., .45][k])
        note(start, beat * 1.7, bass, .055)
        if bar > 1:
            for j, k in enumerate([0, 2, 1, 2, 1, 0, 2, 1]):
                note(start + j * beat / 2, .6, chord[k] + 12, .035, [-.25, .25][j % 2], True)
        if 4 < bar < 21:
            for j in range(4):
                offset = int((start + j * beat) * sr)
                n = min(int(.22 * sr), count - offset)
                if n <= 0:
                    continue
                t = np.arange(n) / sr
                kick = np.sin(2 * np.pi * (46 * t + 35 * .035 * (1 - np.exp(-t / .035)))) * np.exp(-t * 25) * .065
                mix[offset:offset + n] += kick[:, None]
            for j in range(8):
                offset = int((start + j * beat / 2) * sr)
                n = min(int(.065 * sr), count - offset)
                if n > 0:
                    noise = np.r_[0., np.diff(rng.normal(size=n))] * np.exp(-np.arange(n) / sr * 80) * .004
                    mix[offset:offset + n] += noise[:, None] * np.array([.8, 1.])
    axis = np.arange(count) / sr
    mix *= (np.minimum(axis / 1.5, 1) * np.maximum(0, np.minimum((DURATION - axis) / 2.5, 1)))[:, None]
    mix *= .6 / np.abs(mix).max()
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(2); stream.setsampwidth(2); stream.setframerate(sr)
        stream.writeframes(np.rint(mix * 32767).astype('<i2').tobytes())
    return dict(rights='Original procedural composition for this project; no samples, songs or third-party music.',
                seed=20261102, sample_rate_hz=sr, bpm=96, duration_s=DURATION)


def sam3d_assets():
    """Bind genuine model/launch, captured native orbits and a raw source projection."""
    review_path = local('data/sam3d-native-review-20261101/earbud-a-v2/review-manifest.json')
    review = read(review_path)
    if (review.get('fixture') is not False or review['candidate']['selected'] is not False
            or review['binding']['program_sha256'] != PROGRAM_SHA
            or review['binding']['track_id'] != 'earbud_a'
            or review['renderer']['display_background'] != 'neutral'
            or review['renderer']['native_color_values_modified'] is not False):
        raise ValueError('Actual same-source neutral native comparison required')
    review_dir = review_path.parent
    records = [record(review_path, 'verified_native_review_manifest')]
    for row in review['files']:
        path = review_dir / row['path']
        if sha(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('Genuine native review artifact changed')
        records.append(record(path, 'unchanged_verified_native_review_artifact'))
    proof_path = local('output/sam3d-first-run-20261101/actual-v2/actual-execution-review.json')
    proof = read(proof_path)
    if (proof.get('status') != 'actual_cluster_execution_and_local_native_import_verified'
            or proof.get('offline_model_attempts') != 1 or proof.get('all_nine_returned_files_match_remote_sha256') is not True
            or proof['native_import']['sha256'] != review['import_receipt']['sha256']
            or next((r['sha256'] for r in proof['output_files'] if r['path']=='object.glb'), None) != review['candidate']['asset_sha256']):
        raise ValueError('Actual independently reviewed execution/import binding differs')
    for binding in proof['root_execution_snapshots']:
        checked(binding)
    records.append(record(proof_path, 'independent_root_actual_execution_review'))
    for name in ('first-object-v5-final.json','first-object-v5-transfer.json','first-object-v5-launch.json'):
        path = proof_path.parent / name
        if not path.is_file():
            raise ValueError('Actual execution/transfer/launch receipts missing')
        records.append(record(path, 'actual_root_execution_or_transfer_receipt'))
    capture_dir = OUT / 'captures/sam3d-native-v2'
    receipt_path = capture_dir / 'capture-for-film.json'
    receipt = read(receipt_path)
    if (receipt['url'] != 'http://127.0.0.1:8765/files/data/sam3d-native-review-20261101/earbud-a-v2/index.html'
            or not 40 <= len(receipt['candidate_frames']) <= 52):
        raise ValueError('Actual genuine SAM3D CUA orbit required')
    records.append(record(receipt_path, 'actual_CUA_native_inspection_capture_receipt'))
    checked(receipt['capture_receipt']); records.append(receipt['capture_receipt'])
    for captured in receipt['full_screenshots']:
        checked(captured); records.append(captured)
    frames = []
    for row in receipt['candidate_frames']:
        path = capture_dir / row['file']
        path.relative_to(capture_dir)
        checked(row['image'])
        records.append(record(path, 'actual_unchanged_SAM3D_native_canvas_capture'))
        frames.append(Image.open(path).convert('RGB'))
    baseline_path = capture_dir / receipt['baseline']
    records.append(record(baseline_path, 'actual_preserved_TripoSR_canvas_capture'))
    baseline = Image.open(baseline_path).convert('RGB')
    diagnostic_dir = local('data/sam3d-source-projection-20261102/raw-layout-v1/diagnostic.json').parent
    diagnostic_path = diagnostic_dir / 'diagnostic.json'
    diagnostic = read(diagnostic_path)
    if (diagnostic['format'] != 'sam3d-fixed-raw-source-projection-diagnostic'
            or diagnostic['render']['native_appearance']['source_glb_sha256'] != review['candidate']['asset_sha256']
            or diagnostic['optimized'] is not False or diagnostic['native_glb_changed'] is not False
            or diagnostic['source_video_pose_accepted'] is not False
            or diagnostic['camera']['independent_camera_calibration'] is not False
            or diagnostic['render']['SAM_mask_used_to_clip_render'] is not False):
        raise ValueError('Genuine raw single-frame projection scope changed')
    records.append(record(diagnostic_path, 'raw_same_frame_projection_diagnostic_not_pose_acceptance'))
    for row in diagnostic['files']:
        path = diagnostic_dir / row['path']
        if sha(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('Raw projection artifact changed')
        records.append(record(path, 'unchanged_raw_projection_artifact'))
    source = Image.open(diagnostic_dir / 'source-rgb.png').convert('RGB')
    anchor = Image.open(review_dir / 'source-rgb.png').convert('RGB')
    if source.size != anchor.size or source.tobytes() != anchor.tobytes():
        raise ValueError('Projected source pixels differ from exact model input')
    crop = diagnostic['display_crop_xyxy']
    return {'frames': frames, 'baseline': baseline, 'source': source.crop(crop),
            'overlay': Image.open(diagnostic_dir / 'source-overlay.png').convert('RGB').crop(crop),
            'silhouette': Image.open(diagnostic_dir / 'silhouette-comparison.png').convert('RGB').crop(crop),
            'provenance': {'records': records, 'actual_execution_review': proof,
                'review_manifest': review, 'diagnostic': diagnostic,
                'fixed_editorial_crop_xyxy': crop,
                'native_orbit_is_editorial_camera_only': True,
                'other_component_models_remain_TripoSR': True,
                'single_frame_projection_not_tracking': True}}


def assets(capture=None):
    prep = read(OUT / 'preparation.json')
    for row in [prep['program'], prep['source'], prep['inherited_render_manifest'], prep['mcp_trace'], *prep['robot']]:
        checked(row)
    a = {'prep': prep, 'program': read(checked(prep['program'])),
         'source': [], 'mask': [], 'context': {}, 'parts': {}}
    crop = prep['fixed_editorial_crop_xyxy']
    for row in prep['sequence']:
        a['source'].append(Image.open(checked(row['source'])).convert('RGB').crop(crop))
        a['mask'].append(Image.open(checked(row['overlay'])).convert('RGB').crop(crop))
        for mask in row['masks']:
            checked(mask)
    for row in prep['context']:
        a['context'][row['id']] = Image.open(checked(row['image'])).convert('RGB')
    inherited = prep['inherited_render_manifest_content']
    for row in inherited['inputs']:
        checked(row)
    mapping = {row['path']: row for row in inherited['rendered_asset_inputs']}
    native = read(OUT / 'native-renders-v1/manifest.json')
    if native['program']['sha256'] != PROGRAM_SHA:
        raise ValueError('Native display belongs to another program')
    for component in native['components']:
        checked(component['asset'])
        a['parts'][component['component']] = [Image.open(checked(row)).convert('RGB') for row in component['frames']]
    a['native_manifest'] = record(OUT / 'native-renders-v1/manifest.json', 'actual_current_program_native_render_provenance')
    a['camera_source'] = Image.open(local(inherited['camera_source_image'])).convert('RGB')
    for key in ('observed', 'inferred'):
        a[key] = [Image.open(checked(mapping[f'output/showcase-studio-20261031/camera-patch-orbits/{key}-{n:03d}.jpg'])).convert('RGB') for n in range(24)]
    robot_receipt_path = OUT / 'captures/robot-v3/capture-receipt.json'
    robot_receipt = read(robot_receipt_path)
    if robot_receipt['url'] != 'http://127.0.0.1:8765/workspace/robotics.html?case=flow-rigid-parts-jabra-20261031-v1--flow-procedure-1' or len(robot_receipt['frames']) != 48:
        raise ValueError('Corrected actual robot capture is not source-bound')
    a['robot'] = [Image.open(Path(row['path'])).convert('RGB') for row in robot_receipt['frames']]
    a['robot_display_manifest'] = dict(receipt=record(robot_receipt_path, 'actual_corrected_rehearsal_capture'),
        frames=[record(Path(row['path']), 'actual_UI_scene_panel_screenshot', authored_progress=row['authored_progress']) for row in robot_receipt['frames']])
    a['agent'] = Image.open(ROOT / 'output/submission-review-20261101/fresh-36-call-desktop-v2.png').convert('RGB')
    if capture:
        row = read(local(capture))
        a['library_capture'] = row
        a['library'] = Image.open(checked(row['image'])).convert('RGB')
    a['sam3d'] = sam3d_assets()
    return a


def source_index(a, u):
    return min(len(a['source']) - 1, int(u * len(a['source'])))


def frame(section, u, a, live=None):
    n = source_index(a, u)
    if section == 0:
        im = base(dark=True)
        text(im, (68, 210), 'From a video.', 91, BG, True)
        text(im, (68, 322), 'To a robot', 91, LIME, True)
        text(im, (68, 434), 'task draft.', 91, BG, True)
        text(im, (74, 619), 'Parts. Actions. Evidence.', 36, '#c8d6bf')
        text(im, (74, 678), 'Reusable assets for robotics authoring.', 31, '#c8d6bf')
        chip(im, (74, 849), 'ORDINARY VIDEO → INTERACTION ASSET')
        source = live if live is not None else a['source'][n]
        card(im, source, (1260, 85, 520, 890), background='#263e32', radius=30)
        footer(im, 'Working prototype · source video → parts + actions + evidence', True)
    elif section == 1:
        im = base('01 / Meta perception', 'One recording. Separate parts.', 'Visible component masks from the real SAM 3.1 video API.')
        card(im, a['source'][n], (68, 340, 650, 565), background=SOFT)
        card(im, a['mask'][n], (770, 340, 650, 565), background=SOFT)
        chip(im, (94, 363), 'SOURCE RGB', fill=BG)
        chip(im, (796, 363), 'SAM 3.1', fill=LIME)
        for j, (key, value) in enumerate([('case_base', 'case_base'), ('case_lid', 'case_lid'), ('earbud_a', 'earbud_a'), ('earbud_b', 'earbud_b')]):
            y = 415 + j * 98
            ImageDraw.Draw(im).ellipse((1480, y + 4, 1503, y + 27), fill=PART_COLORS[key])
            text(im, (1525, y), value, 29, INK, True)
        text(im, (1481, 848), f'{a["prep"]["sequence"][n]["timestamp_s"]:.3f} s', 30, MUTED, True)
        footer(im, 'Exact saved SAM masks · proposed component IDs · source samples at 3 fps')
    elif section == 2:
        im = base('02 / Muse Spark reasoning', 'A procedure with receipts.', 'Relationships and action order stay linked to the frames behind them.')
        card(im, a['context']['source_000021'], (70, 340, 345, 615), background=SOFT)
        x, y = 490, 341
        selected = min(3, int(u * 4))
        for j, step in enumerate(a['program']['steps']):
            active = j == selected
            ImageDraw.Draw(im).rounded_rectangle((x, y + 140 * j, 1829, y + 140 * j + 117), 18, fill=INK if active else '#ffffff')
            color = BG if active else INK
            text(im, (x + 27, y + 140 * j + 25), f'0{j + 1}', 33, LIME if active else MUTED, True)
            text(im, (x + 110, y + 140 * j + 22), step['title'], 34, color, True)
            text(im, (x + 113, y + 140 * j + 71), f'{step["start_s"]:.3f}–{step["end_s"]:.3f} s  ·  {len(step["evidence_ids"])} cited samples', 23, '#b9ceb2' if active else MUTED)
        footer(im, 'Model proposals linked to exact source frames · reviewable, not physical ground truth')
    elif section == 3:
        sam = a['sam3d']
        if u < .56:
            im = base('03 / First genuine SAM 3D part', 'A new shape. The same source.', 'One automatic RGB/mask anchor. An unchanged, unselected native candidate.')
            index = min(len(sam['frames'])-1, int(u / .56 * len(sam['frames'])))
            card(im, sam['source'], (68, 357, 420, 480), background=SOFT)
            card(im, sam['baseline'], (528, 357, 420, 480), background=SOFT)
            card(im, sam['frames'][index], (988, 332, 864, 572), background=SOFT)
            chip(im, (88, 377), 'SOURCE RGB', fill=BG, size=19)
            chip(im, (548, 377), 'TRIPOSR BASELINE', fill=BG, size=19)
            chip(im, (1012, 356), 'SAM 3D · FIRST PART', fill=LIME, size=21)
            text(im, (77, 866), '13.001 s · editorial crop', 23, MUTED)
            text(im, (1000, 931), 'Actual native inspection orbit · independent coordinates', 24, MUTED)
            footer(im, 'New earbud candidate only · other components retain TripoSR drafts · physical shape unverified')
        else:
            im = base('03 / Raw model layout → source', 'And back into the recorded frame.', 'The model’s own layout, with a camera estimated from its pointmap. No pose fitting.')
            card(im, sam['source'], (68, 340, 790, 555), background=SOFT)
            card(im, sam['overlay'], (918, 340, 790, 555), background=SOFT)
            chip(im, (94, 364), 'ORIGINAL SOURCE', fill=BG)
            chip(im, (944, 364), 'RAW LAYOUT PROJECTION', fill=LIME)
            text(im, (87, 927), 'Same exact source pixels · 13.001144 s', 24, MUTED)
            text(im, (938, 927), 'Single-frame initializer · estimated camera', 24, MUTED)
            footer(im, 'Editorial crop only · no manual or reused pose · motion, physical scale and contact remain unvalidated')
    elif section == 4:
        im = base('04 / Interaction program', 'More than geometry.', 'Components, actions, citations and unknowns travel together.', dark=True)
        ImageDraw.Draw(im).rounded_rectangle((68, 340, 1240, 941), 24, fill='#203c2e')
        text(im, (99, 373), 'READABLE VIEW OF THE SAVED PROGRAM', 19, '#b3cca7', True)
        code = [('component', 'earbud_a'), ('relationship', 'removable_from → case_base'), ('action', 'remove earbuds to table'), ('evidence', 'source_000021 · source_000033'), ('status', 'inferred · reviewable'), ('physical_validation', 'false')]
        visible = min(6, 2 + int(u * 6))
        for j, (key, value) in enumerate(code[:visible]):
            text(im, (100, 435 + j * 75), key + ':', 25, '#b6c9af', mono=True)
            text(im, (476, 435 + j * 75), value, 25, LIME if j < 4 else '#e7c183', mono=True)
        for j, evidence_id in enumerate(('source_000021', 'source_000033')):
            card(im, a['context'][evidence_id], (1310 + j * 255, 340, 230, 409), background='#263e32', radius=20)
            evidence = next(r for r in a['prep']['context'] if r['id'] == evidence_id)
            text(im, (1314 + j * 255, 775), f'{evidence["timestamp_s"]:.3f} s', 26, '#c8d8be', True)
        text(im, (1310, 848), 'Source evidence,\nnot silent assumptions.', 30, BG, True, spacing=12)
        footer(im, 'Descriptive interaction program · source evidence travels with the asset', True)
    elif section == 5:
        im = base('05 / Library + MCP', 'An agent can inspect the asset.', 'Discover components, retrieve models and reopen exact cited frames.')
        if u < .55 and 'library' in a:
            card(im, a['library'], (68, 327, 1786, 612), background=SOFT, cover=False)
            footer(im, 'Actual local library capture · saved assets grouped by source · quality and unknowns remain attached')
        else:
            card(im, a['agent'], (68, 327, 1170, 612), background=SOFT, cover=False)
            for j, name in enumerate(('search_twins', 'get_procedure', 'get_asset', 'get_evidence')):
                ImageDraw.Draw(im).rounded_rectangle((1298, 350 + j * 130, 1845, 452 + j * 130), 16, fill=INK)
                text(im, (1322, 379 + j * 130), name + '()', 27, LIME, mono=True)
            footer(im, 'Real local SDK run · 36 calls · 3 native GLBs · 6 cited JPEGs · scripted client, no robot execution')
    elif section == 6:
        im = base('06 / Robot task authoring', 'Start authoring the next task.', 'Stage the native assets in an illustrative robot rehearsal.')
        index = min(len(a['robot']) - 1, int(u * len(a['robot'])))
        # The capture is the exact existing scene panel including its disclaimer.
        screen = a['robot'][index].crop((200, 65, 950, 565))
        card(im, screen, (68, 325, 1300, 645), background=SOFT, cover=False)
        chip(im, (95, 353), 'ACTUAL APP · EDITORIAL CROP', fill=BG, size=19)
        chip(im, (1411, 354), 'ILLUSTRATIVE', fill='#f7dfaa')
        text(im, (1412, 435), 'Existing drafts.\nAuthored anchors.\nInspectable task.', 34, INK, True, spacing=20)
        text(im, (1412, 654), 'Next: validate scale,\ncollision, grasp\nand contact.', 29, MUTED, spacing=17)
        footer(im, 'Illustrative rehearsal with existing TripoSR assets · physics not validated · authored motion shown faster')
    else:
        im = base(dark=True)
        text(im, (68, 205), 'One recording.', 91, BG, True)
        text(im, (68, 317), 'A reusable', 91, LIME, True)
        text(im, (68, 429), 'robotics asset.', 91, BG, True)
        text(im, (76, 614), 'Build a library of parts, actions and evidence.', 34, '#c8d6bf')
        for j, label in enumerate(('INSPECT', 'REVIEW', 'EXPORT', 'MCP')):
            chip(im, (76 + j * 245, 760), label, fill=LIME if j == 3 else '#294735', color=INK if j == 3 else BG, size=23)
        card(im, a['mask'][min(10, len(a['mask']) - 1)], (1280, 145, 510, 665), background='#263e32')
        text(im, (1288, 845), 'Meta SAM 3.1 · SAM 3D\nMuse Spark', 25, LIME, True, spacing=12)
        text(im, (76, 903), 'INTERACTION TWIN', 35, BG, True)
        footer(im, 'Source-backed assets for task authoring · physical validation comes next', True)
    start = sum(c['seconds'] for c in CHAPTERS[:section])
    progress = (start + u * CHAPTERS[section]['seconds']) / DURATION
    ImageDraw.Draw(im).line((0, H - 4, round(W * progress), H - 4), fill=LIME if section in (0, 4, 7) else INK, width=4)
    return im


def storyboard(take, capture=None):
    directory = OUT / take
    directory.mkdir(exist_ok=True)
    a = assets(capture)
    montage = Image.new('RGB', (1600, 1800), '#d6dccf')
    rows = []
    for section in range(8):
        im = frame(section, .6, a)
        path = directory / f'storyboard-{section + 1:02d}.jpg'
        im.save(path, quality=96)
        montage.paste(im.resize((800, 450)), ((section % 2) * 800, (section // 2) * 450))
        rows.append(record(path, 'editorial_storyboard_with_genuine_inputs'))
    montage.save(directory / 'storyboard.jpg', quality=95)
    fresh(directory / 'storyboard.json', dict(chapters=CHAPTERS, frames=rows,
                                              source=record(OUT / 'preparation.json', 'frozen_preparation')))
    print(json.dumps({'storyboard': str(directory / 'storyboard.jpg')}), flush=True)


def mix_audio(directory):
    score = directory / 'original-score.wav'
    rights = original_score(score)
    speech = read(OUT / 'speech-v5.json')
    inputs = []
    filters = []
    for j, row in enumerate(speech['rows']):
        inputs += ['-i', str(checked(row['audio']))]
        duration = CHAPTERS[j]['seconds']
        filters.append(f'[{j}:a]adelay=350,apad,atrim=duration={duration},asetpts=PTS-STARTPTS[a{j}]')
    filters.append(''.join(f'[a{j}]' for j in range(8)) + 'concat=n=8:v=0:a=1,volume=1.0,pan=stereo|c0=c0|c1=c0[voice]')
    inputs += ['-i', str(score)]
    filters.append('[8:a]volume=0.10[music]')
    filters.append('[voice][music]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.85:level=false[mix]')
    dest = directory / 'narration-music.wav'
    subprocess.run(['ffmpeg', '-hide_banner', '-v', 'error', *inputs, '-filter_complex', ';'.join(filters),
                    '-map', '[mix]', '-ar', '48000', '-ac', '2', '-c:a', 'pcm_s16le', str(dest)], check=True)
    fresh(directory / 'audio-rights.json', dict(score=record(score, 'original_procedural_music'), composition=rights,
                                               speech=record(OUT / 'speech-v5.json', 'bundled_local_TTS'),
                                               mixed=record(dest, 'original_music_with_local_voice')))
    return dest


def encode(take, capture=None):
    if not take.replace('-', '').isalnum():
        raise ValueError('Simple versioned take required')
    directory = OUT / take
    directory.mkdir(exist_ok=True)
    dest = MEDIA / (PREFIX + '-' + take + '.mp4')
    if dest.exists():
        raise ValueError('Preserve existing takes')
    a = assets(capture)
    audio = mix_audio(directory)
    snapshot = directory / 'builder-at-encode.py'
    shutil.copyfile(Path(__file__), snapshot)
    sys.path.insert(0, str(ROOT / 'scripts'))
    from build_showcase_media import VideoClip
    clip = VideoClip(checked(a['prep']['source']), 12, CHAPTERS[0]['seconds'], size=(540, 960))
    argv = ['ffmpeg', '-hide_banner', '-v', 'warning', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
            '-r', str(FPS), '-i', 'pipe:0', '-i', str(audio), '-map', '0:v', '-map', '1:a', '-c:v', 'libx264',
            '-preset', 'fast', '-crf', '18', '-threads', '4', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
            '-movflags', '+faststart', '-t', str(DURATION), str(dest)]
    log = (directory / 'encode.log').open('x')
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stderr=log)
    start, count = time.monotonic(), 0
    chapters = []
    try:
        for section, chapter in enumerate(CHAPTERS):
            frames = chapter['seconds'] * FPS
            for n in range(frames):
                live = clip.frame() if section == 0 else None
                im = frame(section, n / max(1, frames - 1), a, live)
                # Smooth editorial chapter entry; no source/object interpolation.
                lead = min(1., n / 8)
                if section and lead < 1:
                    previous = frame(section - 1, 1., a)
                    im = Image.blend(previous, im, lead)
                proc.stdin.write(im.tobytes()); count += 1
            if section == 0:
                clip.close()
            chapters.append(dict(chapter=section + 1, start_s=sum(c['seconds'] for c in CHAPTERS[:section]), **chapter))
            print(json.dumps({'chapter': section + 1, 'frames': count}), flush=True)
        proc.stdin.close()
        if proc.wait(timeout=180):
            raise ValueError('Encoder failed; retain log')
    finally:
        if clip.process.poll() is None:
            clip.close()
        if proc.poll() is None:
            proc.terminate(); proc.wait(timeout=20)
        log.close()
    fresh(directory / 'source-manifest.json', dict(format='source-backed-interaction-library-film', version=1,
        output=record(dest, 'actual_H264_AAC_film'), width=W, height=H, fps=FPS, duration_s=DURATION, frames=count,
        render_seconds=round(time.monotonic() - start, 3), chapters=chapters,
        preparation=record(OUT / 'preparation.json', 'frozen_genuine_inputs'), preparation_content=a['prep'],
        preparation_scope='Inherited V1 baseline input receipt is historical; current first genuine SAM3D part is bound separately in sam3d_provenance.',
        builder=record(snapshot, 'exact_encoder_source'), audio=record(directory / 'audio-rights.json', 'local_voice_original_music'),
        library_capture=a.get('library_capture'),
        agent_capture=record(ROOT / 'output/submission-review-20261101/fresh-36-call-desktop-v2.png', 'actual_prior_CUA_capture'),
        current_native_render_manifest=a['native_manifest'],
        current_robot_display_manifest=a['robot_display_manifest'],
        source_playback='First5seconds actual12..17s video decoded24fps; remaining source overlays use exact saved3fps samples. No invented frames or poses.',
        robot_editorial_crop_xyxy=[200,65,950,565],
        robot_playback='48 actual UI states of an18second authored rehearsal, held and shown in7seconds; not recovered video motion or physics.',
        camera_orbits='Prior frozen authored inspection cameras; native model geometry unchanged.',
        new_api_calls=0, model_inference_calls=0, SAM3D_output_included=True, first_part_only=True, sam3d_provenance=a['sam3d']['provenance'], published=False))
    print(json.dumps({'status': 'encoded', 'output': str(dest), 'seconds': DURATION}), flush=True)


def verify(take):
    directory = OUT / take
    manifest = read(directory / 'source-manifest.json')
    path = checked(manifest['output'])
    probe = read_probe(path)
    video = next(r for r in probe['streams'] if r['codec_type'] == 'video')
    audio = next(r for r in probe['streams'] if r['codec_type'] == 'audio')
    if (video['width'], video['height'], video['avg_frame_rate'], video['codec_name']) != (W, H, '24/1', 'h264'):
        raise ValueError('Video output differs')
    if not 57.99 <= float(probe['format']['duration']) <= 58.1 or audio['codec_name'] != 'aac':
        raise ValueError('Duration/audio differs')
    subprocess.run(['ffmpeg', '-hide_banner', '-v', 'error', '-i', str(path), '-f', 'null', '-'], check=True)
    raw = subprocess.check_output(['ffmpeg', '-hide_banner', '-v', 'error', '-i', str(path), '-vn', '-f', 'f32le', '-acodec', 'pcm_f32le', '-'])
    samples = np.frombuffer(raw, dtype='<f4').reshape(-1, 2)
    peak = float(np.abs(samples).max())
    if not .01 < peak < 1 or not np.isfinite(samples).all():
        raise ValueError('Invalid/nonfinite/clipped audio')
    capture = cv2.VideoCapture(str(path))
    montage = Image.new('RGB', (1600, 1800), '#d6dccf')
    decoded, counts, blank = [], {}, []
    frame_no = 0
    previous = None
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        if bgr.shape != (H, W, 3):
            raise ValueError('Decoded dimensions differ')
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if gray.std() < 3:
            blank.append(frame_no)
        thumb = cv2.resize(gray, (192, 108)).astype(float)
        delta = float(np.sqrt(np.mean((thumb - previous) ** 2))) if previous is not None else 0
        section = min(7, int(np.searchsorted(np.cumsum([c['seconds'] * FPS for c in CHAPTERS]), frame_no, side='right')))
        row = counts.setdefault(str(section + 1), {'frames': 0, 'changed_frames': 0})
        row['frames'] += 1; row['changed_frames'] += delta > .1
        previous = thumb
        frame_no += 1
    if frame_no != DURATION * FPS or blank:
        raise ValueError('Missing or blank output frames')
    capture.release()
    capture = cv2.VideoCapture(str(path))
    for i, chapter in enumerate(manifest['chapters']):
        at = chapter['start_s'] + chapter['seconds'] * .6
        capture.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
        ok, bgr = capture.read()
        if not ok:
            raise ValueError('Encoded chapter failed to decode')
        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        output = directory / f'decoded-chapter-{i + 1:02d}.jpg'
        im.save(output, quality=96)
        montage.paste(im.resize((800, 450)), ((i % 2) * 800, (i // 2) * 450))
        decoded.append(record(output, 'actual_decoded_MP4_frame', timestamp_s=at))
        if i == 0:
            shutil.copyfile(output, MEDIA / (PREFIX + '-' + take + '-poster.jpg'))
    capture.release()
    montage.save(directory / 'verified-storyboard.jpg', quality=95)
    result = dict(status='passed', video=probe, fully_decoded=True, decoded_frames=frame_no,
                  chapter_frame_changes=counts, blank_frames=blank, chapter_stills=decoded,
                  audio={'peak_dbfs': float(20 * np.log10(peak)), 'rms_dbfs': float(20 * np.log10(np.sqrt(np.mean(samples * samples)))),
                         'clipped_samples': int((np.abs(samples) >= 1).sum())},
                  full_visual_review_required=True, publication=False)
    fresh(directory / 'verification.json', result)
    print(json.dumps({'status': 'verified', 'frames': frame_no, 'audio': result['audio']}), flush=True)


def audit_sources(take):
    """Recheck every source/renderer/capture digest in the encoded film receipt."""
    manifest_path = OUT / take / 'source-manifest.json'
    manifest = read(manifest_path)
    checked_rows = {}
    historical_witnesses = {}
    def walk(value, location=()):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                if value['path']=='portable_twin.py' and value['sha256']=='16a4c3e596dc2a7cbe0d81e9f072c5657c34f4109417469e890e0d92ea828eaf':
                    witness={'path':'output/sam3d-large-native-20261102/historical/portable_twin-before-large-index-fix.py','sha256':value['sha256']}
                    checked(witness)
                    checked_rows[witness['path']]=witness['sha256']
                    historical_witnesses[value['path']]={'frozen_declared_sha256':value['sha256'],'verified_historical_bytes':witness,
                        'origin':'Reconstructed by reversing the recorded role-specific index-limit patch; exact frozen SHA equality required.',
                        'past_execution_independently_proved_by_this_recovery':False}
                else:
                    checked(value)
                    checked_rows[value['path']] = value['sha256']
            for key, child in value.items():
                # These exact embedded documents use their own relative file
                # namespaces. Their entire bytes and declared artifacts are
                # already frozen in provenance.records; resolve proof outputs
                # separately below instead of treating them as workspace paths.
                if location == ('sam3d_provenance',) and key in {'actual_execution_review','review_manifest','diagnostic'}:
                    continue
                walk(child, location + (key,))
        elif isinstance(value, list):
            for child in value: walk(child, location)
    walk(manifest)
    proof=manifest['sam3d_provenance']['actual_execution_review']
    for row in proof['output_files']:
        bound=dict(row,path='data/sam3d-gpu-20261101/earbud-a-sample40-output-v1/'+row['path'])
        walk(bound)
    walk(proof['native_import'])
    walk(proof['root_execution_snapshots'])
    walk(record(ROOT/'portable_twin.py','current_decoder_with_bounded_triangle_index_support'))
    walk(record(ROOT/'output/sam3d-large-native-20261102/historical/recovery.json','historical_decoder_exact_hash_recovery_record'))
    native = read(checked(manifest['current_native_render_manifest']))
    walk(native)
    audio = read(checked(manifest['audio']))
    walk(audio)
    speech_manifest = read(checked(audio['speech']))
    walk(speech_manifest)
    fresh(OUT / take / 'source-integrity.json', dict(status='all_declared_bytes_verified',
          source_manifest=record(manifest_path, 'actual_encoded_source_manifest'),
          verifier=record(Path(__file__), 'scoped_source_auditor_after_encode'),
          historical_implementation_witnesses=historical_witnesses,
          distinct_files_checked=len(checked_rows), files=checked_rows,
          physical_accuracy_verified=False, model_inference_calls=0, api_calls=0))
    print(json.dumps({'status': 'source_integrity_passed', 'files': len(checked_rows)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'native-render', 'speech', 'storyboard', 'encode', 'verify', 'audit-sources'))
    p.add_argument('--take', default='v1')
    p.add_argument('--library-capture', help='Project-local capture receipt with image path+SHA and actual URL')
    p.add_argument('--speech-take', default='speech-v5')
    a = p.parse_args()
    if a.action == 'prepare': prepare()
    elif a.action == 'native-render': native_render()
    elif a.action == 'speech': speech(a.speech_take)
    elif a.action == 'storyboard': storyboard(a.take, a.library_capture)
    elif a.action == 'encode': encode(a.take, a.library_capture)
    elif a.action == 'verify': verify(a.take)
    else: audit_sources(a.take)


if __name__ == '__main__':
    main()
