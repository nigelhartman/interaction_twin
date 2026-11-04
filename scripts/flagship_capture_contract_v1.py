#!/usr/bin/env python3
"""Local, read-only acceptance gate for exact current-app trailer captures.

This checks saved lineage, complete clocks, viewport bounds and decoded pixels.
It does not establish physical accuracy or prove a browser trace is truthful.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ROLES = {'multiple_models': {'phone', 'cable', 'charger'},
         'jabra_parts': {'case_base', 'case_lid', 'earbud_a', 'earbud_b'}}
IDS = {'phone': 'flow-capture-phone-20261103-v1--flow-procedure-1',
       'charger': 'flow-capture-charger-20261103-v1--flow-procedure-1',
       'jabra': 'flow-sam3d-default-e2e-20261102-v1--flow-procedure-1'}


def require(value, message):
    if not value:
        raise ValueError(message)


def file_ref(path, root=ROOT):
    root = root.resolve()
    path = Path(path).resolve()
    require(path.is_relative_to(root.resolve()), 'Artifact must be inside project.')
    raw = path.read_bytes()
    return {'path': path.relative_to(root).as_posix(),
            'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def read_ref(reference, root=ROOT):
    require(isinstance(reference, dict), 'Exact reference required.')
    name = reference.get('path', '')
    require(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9_./-]+', name)
            and not any(p.startswith('.') or not p for p in name.split('/')),
            'Unsafe artifact reference.')
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), 'Artifact escaped project.')
    actual = file_ref(path, root)
    require(all(reference.get(k) == v for k, v in actual.items()),
            'Artifact bytes changed: ' + name)
    return path


def current_context(root=ROOT):
    index = json.loads((root / 'showcase/public/library-index.json').read_text())
    catalog_ref = dict(index['catalog'], path='showcase/public/' + index['catalog']['path'])
    catalog = json.loads(read_ref(catalog_ref, root).read_text())
    aligned = json.loads((root / 'showcase/public/aligned-scenes.json').read_text())
    result = {'catalog': catalog_ref, 'records': {}}
    for role, ident in IDS.items():
        record = next(r for r in catalog['records'] if r['id'] == ident)
        entry = next(r for r in aligned['scenes'] if r['source_sha256'] == record['source']['sha256'])
        scene = json.loads(read_ref(entry['manifest'], root).read_text())
        observed = json.loads(read_ref(entry['observed_surfaces'], root).read_text()) if entry.get('observed_surfaces') else None
        result['records'][role] = {'record': record, 'entry': entry, 'scene': scene, 'observed': observed}
    return result


def clock(trace, key, duration, minimum):
    require(isinstance(trace, list) and len(trace) >= minimum, 'Capture clock is too sparse.')
    times = [r.get(key) for r in trace]
    require(all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) for t in times), 'Invalid capture clock.')
    require(0 <= times[0] <= .05 and duration - .05 <= times[-1] <= duration + .05,
            'Capture does not span complete source clock.')
    require(all(a < b for a, b in zip(times, times[1:])), 'Capture clock reversed or repeated.')
    require(max(b-a for a,b in zip(times,times[1:])) <= .25, 'Capture clock has a visible gap.')
    return times


def _exact_native_assets(context):
    return {c['id']: c['asset'] for c in context['scene']['components']}


def validate_receipt(kind, doc, video, context, root=ROOT):
    """Validate declarations against *current exact* curated source assets."""
    raw = Path(video).read_bytes()
    require(doc.get('version') == 1 and doc.get('bytes') == len(raw)
            and doc.get('video_sha256') == hashlib.sha256(raw).hexdigest(), 'Video does not match capture receipt.')
    require(doc.get('new_API_calls') == 0 and doc.get('physical_accuracy_verified', False) is False,
            'Unexpected API/physical claim.')
    require(doc.get('new_GPU_calls', doc.get('new_remote_GPU_calls')) == 0, 'Unexpected GPU inference.')
    if kind in ROLES:
        require(doc.get('format') == 'interaction-twin-flagship-multiple-model-capture', 'Wrong model capture type.')
        capture_catalog = doc.get('current_catalog')
        if capture_catalog != context['catalog']:
            # Additive motion metadata may be published while videos encode.
            # Never follow its mutable alias: prove the exact historical bytes,
            # then require *all* actual captured scene/model bindings below to
            # remain equal to the current curated selection.
            require(isinstance(capture_catalog,dict) and capture_catalog.get('path') ==
                    'showcase/public/library-catalog-'+str(capture_catalog.get('sha256'))+'.json'
                    and (root/capture_catalog['path']).is_file(), 'Model capture catalog is stale or unavailable.')
            historical = json.loads(read_ref(capture_catalog,root).read_text())
            for binding in context['records'].values():
                record = binding['record']
                prior = next((r for r in historical['records'] if r['id'] == record['id']), None)
                require(prior and prior['source'] == record['source'] and prior['interaction'] == record['interaction'],
                        'Historical capture source/program changed.')
        roles = ROLES[kind]
        require(set(doc.get('roles', [])) == roles and len(doc.get('roles', [])) == len(roles), 'Wrong or duplicate model roles.')
        require(doc.get('geometry_loaded_actual_WebGL') is True and doc.get('source_assets_modified') is False
                and doc.get('charging_or_insertion_executed') is False, 'Model scope changed.')
        require(doc.get('canvas') == [1920, 1080] and doc.get('duration_s') == 8, 'Wrong model capture dimensions/clock.')
        clock(doc.get('trace'), 'time_s', 8, 100)
        models = doc.get('models', [])
        require(len(models) == len(roles) and {m['role'] for m in models} == roles, 'Model inventory incomplete.')
        for model in models:
            role = model['role']
            binding = context['records']['jabra' if kind == 'jabra_parts' else 'phone' if role == 'phone' else 'charger']
            require(model.get('source_sha256') == binding['record']['source']['sha256']
                    and model.get('scene_ref') == binding['entry']['manifest']
                    and model.get('source_program') == binding['scene']['original_program'], 'Model source binding differs.')
            if role == 'phone':
                observation = model.get('observation') or {}
                sample = next((r for r in binding['observed']['samples'] if r['source_sample_index'] == observation.get('source_sample_index')), None)
                require(sample and sample.get('asset') == model['asset'] and sample.get('timestamp_s') == observation.get('timestamp_s'), 'Phone is not the exact current supported surface.')
                require(model.get('representation') == 'source_visible_2.5D_estimated_depth', 'Partial phone representation was mislabeled.')
            else:
                ident = {'cable': 'power_cable', 'charger': 'adapter_early'}.get(role, role)
                require(model['asset'] == _exact_native_assets(binding)[ident], 'Model is not current native/declared-derived component.')
            read_ref(model['asset'], root)
        for row in doc['trace']:
            require({v['role'] for v in row['views']} == roles and len(row['views']) == len(roles), 'Missing model view.')
            for i, view in enumerate(row['views']):
                x, y, w, h = ((i % 2 * 960, i // 2 * 540, 960, 540) if len(roles) == 4 else (i * 640, 0, 640, 1080))
                points = view['projected_bounds']
                require(len(points) == 8 and all(len(p) == 3 and x+11.9 <= p[0] <= x+w-11.9 and y+11.9 <= p[1] <= y+h-11.9 and -1 <= p[2] <= 1 for p in points), 'A model was clipped.')
    elif kind == 'robot':
        require(doc.get('format') == 'interaction-twin-flagship-current-robot-capture', 'Wrong robot capture type.')
        duration = doc.get('task_duration_s', 0)
        clock(doc.get('trace'), 'time_s', duration, 200)
        require(abs(doc.get('duration_s', 0) - duration) < .15 and doc.get('canvas') == [1280, 800], 'Robot capture speed/dimensions differ.')
        require(doc.get('captured_full_task') is True and doc.get('final_attached') is None
                and doc.get('final_closed_reference') is True and doc.get('both_earbuds_released') is True,
                'Robot did not complete both releases and closure.')
        goals = doc.get('final_goals', [])
        require({g['id'] for g in goals} == {'earbud_a','earbud_b'} and all(not g['attached'] and 0 <= g['error'] < 1e-5 for g in goals), 'Robot final goal mismatch.')
        require(doc.get('authored_robot_staging') is True and doc.get('physical_contact_verified') is False and doc.get('simulation_ready') is False, 'Robot scope changed.')
        expected = _exact_native_assets(context['records']['jabra'])
        require({r['sha256'] for r in doc['native_asset_refs']} == {r['sha256'] for r in expected.values()}, 'Robot used obsolete model geometry.')
        binding = doc['typed_plan']['source_binding']
        # The full saved plan has separate typed source binding; exact references
        # may be nested, so compare its unchanged source/program identifiers too.
        serialized = json.dumps(binding, sort_keys=True)
        require(context['records']['jabra']['record']['source']['sha256'] in serialized, 'Robot source differs.')
        require(context['records']['jabra']['scene']['original_program']['sha256'] in serialized, 'Robot program differs.')
        for reference in doc['native_asset_refs']:
            read_ref(reference, root)
        require(all(0 <= row['capture_wall_time_s'] <= duration + .3 for row in doc['trace']), 'Robot wall clock differs.')
    elif kind == 'connected':
        require(doc.get('format') == 'interaction-twin-connected-full-bounds-capture', 'Wrong connected capture type.')
        binding = context['records']['jabra']
        require(doc.get('source_sha256') == binding['record']['source']['sha256'] and doc.get('scene_ref') == binding['entry']['manifest'], 'Connected scene is stale.')
        require(doc.get('canvas') == [1920,1080] and doc.get('duration_s') == 12, 'Wrong connected dimensions/clock.')
        clock(doc.get('trace'), 'film_s', 12, 150)
        require(doc.get('native_geometry_unchanged') is True and doc.get('original_part_poses_unchanged') is True
                and doc.get('all_four_parts_required') is True, 'Connected scene scope changed.')
        for reference in doc['inputs']:
            read_ref(reference, root)
        required = set(_exact_native_assets(binding))
        for row in doc['trace']:
            require({b['id'] for b in row['projected_native_boxes']} == required, 'A connected part vanished.')
            require(all(len(p) == 3 and 59.9 <= p[0] <= 1860.1 and 59.9 <= p[1] <= 1020.1 and -1 <= p[2] <= 1 for b in row['projected_native_boxes'] for p in b['points']), 'A connected part was clipped.')
    else:
        raise ValueError('Unknown capture kind.')
    return {'kind': kind, 'trace_rows': len(doc['trace']), 'lineage_and_clock_passed': True,
            'capture_catalog_equals_current': doc.get('current_catalog') == context['catalog'],
            'physical_accuracy_verified': False}


def probe_and_pixels(video, expected_duration, dimensions, crop=None):
    """Decode real pixels: catches a plausible clock around an empty WebGL buffer."""
    from PIL import Image
    command = ['ffprobe','-v','error','-count_frames','-show_entries',
               'stream=width,height,nb_read_frames:format=duration','-of','json',str(video)]
    probe = json.loads(subprocess.check_output(command))
    stream = probe['streams'][0]
    require([stream['width'], stream['height']] == list(dimensions), 'Decoded video dimensions differ.')
    require(int(stream['nb_read_frames']) >= expected_duration * 24, 'Too few real decoded frames.')
    timestamps = [float(v.split(',')[0]) for v in subprocess.check_output(['ffprobe','-v','error','-select_streams','v',
        '-show_entries','frame=best_effort_timestamp_time','-of','csv=p=0',str(video)]).decode().splitlines() if v]
    require(len(timestamps) == int(stream['nb_read_frames']) and all(a < b for a,b in zip(timestamps,timestamps[1:])),
            'Encoded frame timestamps are invalid.')
    encoded_end = timestamps[-1]
    require(0 <= timestamps[0] <= .1 and 0 <= expected_duration-encoded_end <= max(.15,expected_duration*.05),
            'Encoded clock lost too much of the recorded interval.')
    require(max(b-a for a,b in zip(timestamps,timestamps[1:])) <= .15, 'Encoded capture has a visible frame gap.')
    samples = []
    previous = None
    moving_pairs = 0
    for phase in [0, .15, .33, .5, .7, .85, .99]:
        raw = subprocess.check_output(['ffmpeg','-nostdin','-v','error','-ss',str(phase*encoded_end),
            '-i',str(video),'-frames:v','1','-f','image2pipe','-vcodec','png','pipe:1'])
        import io
        image = Image.open(io.BytesIO(raw)).convert('RGB')
        if crop:
            image = image.crop(crop)
        small = image.resize((160,90))
        colors = small.getcolors(160*90)
        dominant = max(n for n,c in colors)/(160*90)
        require(len(colors) > 100 and dominant < .99, 'Capture is a blank/cleared render buffer.')
        if previous is not None:
            from PIL import ImageChops, ImageStat
            difference = sum(ImageStat.Stat(ImageChops.difference(previous,small)).mean)/3
            moving_pairs += difference > .05
        previous = small
        samples.append({'encoded_time_s': phase*encoded_end, 'colors_160x90': len(colors), 'dominant_fraction': dominant})
    require(moving_pairs >= 3, 'Captured scene lacks expected real movement.')
    return {'decoded_frames': int(stream['nb_read_frames']), 'moving_sample_pairs': moving_pairs,
            'sampled_pixels': samples, 'decoded_pixels_passed': True,
            'encoded_last_timestamp_s': encoded_end, 'capture_runtime_duration_s': expected_duration,
            'encoded_vs_runtime_end_difference_s': expected_duration-encoded_end,
            'editorial_clock_policy': 'linear_encoded_clock_normalization_no_new_pose_or_frame_interpolation'}


def make_manifest(pairs, folder, root=ROOT):
    """Accept four existing local pairs without modifying the captured originals."""
    require(set(pairs) == {'robot','multiple_models','jabra_parts','connected'}, 'All four exact capture pairs are required.')
    require(not folder.exists(), 'Use a new intake output directory.')
    context = current_context(root)
    accepted = {}
    for kind, paths in pairs.items():
        video, receipt = map(Path, paths)
        doc = json.loads(receipt.read_text())
        checks = validate_receipt(kind, doc, video, context, root)
        duration = doc['duration_s']
        crop = [0,88,1280,718] if kind == 'robot' else None
        checks['decoded_video'] = probe_and_pixels(video, duration, doc['canvas'], crop)
        accepted[kind] = {'video': file_ref(video, root), 'receipt': file_ref(receipt, root),
                          'duration_s': duration,
                          'encoded_last_timestamp_s': checks['decoded_video']['encoded_last_timestamp_s'],
                          'validation': checks}
    # A source pointer could have moved while decoding all four videos.
    require(current_context(root)['catalog'] == context['catalog'], 'Catalog moved during intake.')
    folder.mkdir(parents=True)
    result = {'format': 'interaction-twin-flagship-current-app-captures', 'version': 1,
              'current_catalog': context['catalog'], 'authored_robot_staging': True,
              **accepted, 'new_API_calls': 0, 'new_GPU_inference_calls': 0,
              'physical_accuracy_verified': False}
    (folder/'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, required=True,
                        help='JSON map of four kinds to [video_path, receipt_path].')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    made = make_manifest(json.loads(args.pairs.read_text()), args.output.resolve())
    print(json.dumps({'manifest': file_ref(args.output.resolve()/'manifest.json'),
                      'accepted': list(ROLES)+['robot','connected']}))
