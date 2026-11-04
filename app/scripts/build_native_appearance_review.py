#!/usr/bin/env python3
"""Build an isolated CPU rendering verification page, never a model result.

The known textured shapes and poses below are SOFTWARE TEST FIXTURES. They are
not offered to the production catalog, import service, benchmark or tracker.
The separate Jabra panel reuses an unchanged learned asset and a saved pose.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignment_render import render_mesh
from native_alignment import native_representation_receipt
from native_appearance import load_native_glb, render_native_mesh

FORMAT = 'interaction-twin-native-appearance-software-review'
BASELINE = 'data/recording-144929-1fps-v3/alignment/generated-earbud-v1/experiment.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def font(size):
    for name in ('/System/Library/Fonts/Supplemental/Arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def calibration_texture():
    image = Image.new('RGB', (384, 384), '#142d3c')
    draw = ImageDraw.Draw(image)
    for y in range(8):
        for x in range(8):
            draw.rectangle((x*48, y*48, x*48+47, y*48+47), fill=('#74d7bc' if (x+y) % 2 else '#285168'))
    for x, y, color, label in [(8, 8, '#e8a278', '01'), (286, 8, '#b6a4e9', '02'),
                                (8, 286, '#d4e7d7', '03'), (286, 286, '#79bbdf', '04')]:
        draw.rounded_rectangle((x, y, x+89, y+89), radius=15, fill=color)
        draw.text((x+16, y+20), label, fill='#122330', font=font(40))
    draw.rounded_rectangle((99, 151, 286, 232), radius=14, fill='#132b3b')
    draw.text((115, 174), 'UV / TEST', font=font(27), fill='#e2f2ef')
    return image


def fixture_glb(parts, texture, materials=None):
    """Encode known opaque geometry and an embedded test image as a tiny GLB."""
    materials = materials or [{'doubleSided': True, 'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}}}]
    doc = {'asset': {'version': '2.0', 'generator': 'SOFTWARE TEST FIXTURE; no inference'},
           'scene': 0, 'scenes': [{'nodes': [0]}], 'nodes': [{'mesh': 0}],
           'meshes': [{'primitives': []}], 'accessors': [], 'bufferViews': [],
           'materials': materials, 'images': [], 'textures': [{'source': 0, 'sampler': 0}],
           'samplers': [{'magFilter': 9729, 'minFilter': 9729, 'wrapS': 33071, 'wrapT': 33071}],
           'extras': {'fixture': True, 'learned_inference_executed': False}}
    binary = bytearray()
    def view(raw):
        binary.extend(b'\0' * (-len(binary) % 4))
        index = len(doc['bufferViews'])
        doc['bufferViews'].append({'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(raw)})
        binary.extend(raw)
        return index
    def accessor(value, kind, component=5126):
        array = np.asarray(value, dtype='<f4' if component == 5126 else '<u4')
        index = len(doc['accessors'])
        doc['accessors'].append({'bufferView': view(array.tobytes()), 'componentType': component,
                                 'count': len(array), 'type': kind})
        return index
    for part in parts:
        attrs = {'POSITION': accessor(part['vertices'], 'VEC3')}
        if part.get('uv') is not None:
            attrs['TEXCOORD_0'] = accessor(part['uv'], 'VEC2')
        faces = np.asarray(part.get('faces', [[0, 1, 2], [0, 2, 3]])).reshape(-1, 1)
        doc['meshes'][0]['primitives'].append({'attributes': attrs, 'indices': accessor(faces, 'SCALAR', 5125),
                                              'material': part.get('material', 0)})
    stream = io.BytesIO(); texture.save(stream, format='PNG')
    doc['images'].append({'bufferView': view(stream.getvalue()), 'mimeType': 'image/png'})
    doc['buffers'] = [{'byteLength': len(binary)}]
    text = json.dumps(doc, separators=(',', ':'), allow_nan=False).encode()
    text += b' ' * (-len(text) % 4); binary.extend(b'\0' * (-len(binary) % 4))
    chunks = struct.pack('<II', len(text), 0x4E4F534A) + text + struct.pack('<II', len(binary), 0x004E4942) + binary
    return struct.pack('<III', 0x46546C67, 2, 12+len(chunks)) + chunks


def quad(x0, x1, y0=-.8, y1=.8, z=0, material=0, uv=None):
    return {'vertices': [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]],
            'uv': uv if uv is not None else [[0, 0], [1, 0], [1, 1], [0, 1]], 'material': material}


def cube_parts():
    # Twenty-four vertices deliberately preserve six independent UV face seams.
    faces = [
        [[-.8, -.8, -.8], [.8, -.8, -.8], [.8, .8, -.8], [-.8, .8, -.8]],
        [[.8, -.8, .8], [-.8, -.8, .8], [-.8, .8, .8], [.8, .8, .8]],
        [[-.8, -.8, .8], [-.8, -.8, -.8], [-.8, .8, -.8], [-.8, .8, .8]],
        [[.8, -.8, -.8], [.8, -.8, .8], [.8, .8, .8], [.8, .8, -.8]],
        [[-.8, -.8, .8], [.8, -.8, .8], [.8, -.8, -.8], [-.8, -.8, -.8]],
        [[-.8, .8, -.8], [.8, .8, -.8], [.8, .8, .8], [-.8, .8, .8]],
    ]
    return [{'vertices': vertices, 'uv': [[0, 0], [1, 0], [1, 1], [0, 1]], 'material': index}
            for index, vertices in enumerate(faces)]


def cube_materials():
    return [{'doubleSided': True, 'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}, 'baseColorFactor': color}}
            for color in ([1, 1, 1, 1], [.85, .65, 1, 1], [.65, 1, .9, 1], [1, .7, .9, 1], [1, .85, .6, 1], [.8, .9, 1, 1])]


def contained_image(image, size):
    """Enlarge a display crop without changing its aspect ratio or source data."""
    factor = min(size[0]/image.width, size[1]/image.height)
    return image.resize((max(1, round(image.width*factor)), max(1, round(image.height*factor))), Image.Resampling.NEAREST)


def rotation(x, y):
    x, y = np.deg2rad([x, y]); cx, sx, cy, sy = np.cos(x), np.sin(x), np.cos(y), np.sin(y)
    return np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])


def pinhole(width, height, focal):
    return np.array([[focal, 0, (width-1)/2], [0, focal, (height-1)/2], [0, 0, 1.]])


def display(rendered, mode='rgb'):
    h, w = rendered['mask'].shape
    yy, xx = np.mgrid[:h, :w]
    pixels = np.empty((h, w, 3), np.uint8)
    pixels[:] = [14, 28, 39]
    pixels[((xx % 32 == 0) | (yy % 32 == 0))] = [22, 40, 51]
    mask = rendered['mask']
    if mode == 'rgb':
        pixels[mask] = rendered['rgb'][mask]
    elif mode == 'depth':
        values = rendered['depth'][mask]
        low, high = float(values.min()), float(values.max())
        t = (values-low) / max(high-low, 1e-12)
        pixels[mask] = np.rint(np.array([129, 231, 193])*(1-t[:, None]) + np.array([99, 100, 180])*t[:, None]).astype(np.uint8)
    else:
        palette = np.array([[113, 218, 187], [180, 160, 235], [235, 166, 117], [111, 182, 223], [226, 140, 176], [196, 214, 149]], np.uint8)
        pixels[mask] = palette[(rendered['face_index'][mask]//2) % len(palette)]
    return Image.fromarray(pixels)


def emit_render(directory, stem, result, relative, *, raw=True):
    paths = {}
    for mode in ('rgb', 'depth', 'faces'):
        name = f'{stem}-{mode}.png'; display(result, mode).save(directory/name)
        paths[mode] = relative(name)
    if raw:
        Image.fromarray(result['rgb']).save(directory/f'{stem}-raw-rgb.png')
        np.savez_compressed(directory/f'{stem}-depth-faces.npz', depth=result['depth'], face_index=result['face_index'])
        paths.update(raw_rgb=relative(f'{stem}-raw-rgb.png'), raw_depth=relative(f'{stem}-depth-faces.npz'))
    return paths


def comparison_metrics(left, right):
    if left['rgb'].shape != right['rgb'].shape:
        raise ValueError('Baseline render dimensions differ')
    difference = np.abs(left['rgb'].astype(np.int16)-right['rgb'].astype(np.int16))
    depth_equal = np.array_equal(left['depth'], right['depth'])
    return {'rgb_max_absolute_difference_uint8': int(difference.max()),
            'rgb_mean_absolute_difference_uint8': float(difference.mean()),
            'changed_rgb_pixels': int(np.any(difference != 0, axis=2).sum()),
            'mask_mismatched_pixels': int(np.count_nonzero(left['mask'] != right['mask'])),
            'depth_arrays_exactly_equal_including_background': bool(depth_equal),
            'face_index_arrays_exactly_equal': bool(np.array_equal(left['face_index'], right['face_index'])),
            'pixel_equality': bool(np.array_equal(left['rgb'], right['rgb']) and depth_equal and np.array_equal(left['mask'], right['mask']))}


def build_baseline(out, relative):
    import trimesh
    reference_path = ROOT/BASELINE
    reference = json.loads(reference_path.read_text())
    metadata_path = Path(reference['inputs']['manifest'])
    if sha(metadata_path) != reference['inputs']['manifest_sha256']:
        raise ValueError('Preserved Jabra model metadata hash differs')
    metadata = json.loads(metadata_path.read_text())
    asset = metadata_path.parent/metadata['outputs']['glb_path']
    if sha(asset) != reference['inputs']['mesh_sha256']:
        raise ValueError('Preserved Jabra GLB hash differs')
    source = metadata_path.parent/metadata['source']['image_path']
    if sha(source) != reference['inputs']['rgb_sha256']:
        raise ValueError('Preserved source RGB hash differs')
    old = trimesh.load(asset, force='mesh', process=False)
    native = load_native_glb(asset)
    K = np.asarray(reference['intrinsics']['matrix'])
    R, t = np.asarray(reference['candidate']['rotation']), np.asarray(reference['candidate']['translation'])
    width, height = reference['intrinsics']['width'], reference['intrinsics']['height']
    begin = time.perf_counter()
    left = render_mesh(old.vertices, old.faces, K, R, t, width, height, vertex_colors=np.asarray(old.visual.vertex_colors)[:, :3])
    old_seconds = time.perf_counter()-begin
    begin = time.perf_counter(); right = render_native_mesh(native, K, R, t, width, height)
    new_seconds = time.perf_counter()-begin
    metrics = comparison_metrics(left, right)
    ys, xs = np.where(left['mask'] | right['mask'])
    crop = (max(0, int(xs.min())-24), max(0, int(ys.min())-24), min(width, int(xs.max())+25), min(height, int(ys.max())+25))
    observed = np.asarray(Image.open(source).convert('RGB'))
    paths = {}
    for label, render in [('legacy', left), ('native', right)]:
        Image.fromarray(render['rgb']).save(out/f'jabra-{label}-raw.png')
    # Display crops are derived from projected support, never a new fit or pose.
    for label, render in [('legacy', left), ('native', right), ('source', None)]:
        pixels = observed.copy()
        if render is not None:
            mask = render['mask']
            pixels[mask] = np.rint(.35*observed[mask]+.65*render['rgb'][mask]).astype(np.uint8)
        panel = Image.new('RGB', (480, 400), '#0e1c27')
        part = contained_image(Image.fromarray(np.asarray(pixels, np.uint8)).crop(crop), (440, 360))
        panel.paste(part, ((480-part.width)//2, (400-part.height)//2))
        name = f'jabra-{label}.png'; panel.save(out/name); paths[label] = relative(name)
    difference = np.abs(left['rgb'].astype(np.int16)-right['rgb'].astype(np.int16)).max(2)
    diff_rgb = np.full((height, width, 3), [14, 28, 39], np.uint8)
    diff_rgb[left['mask'] | right['mask']] = [24, 59, 53]
    diff_rgb[difference > 0] = [240, 138, 115]
    diff = contained_image(Image.fromarray(diff_rgb).crop(crop), (440, 360))
    panel = Image.new('RGB', (480, 400), '#0e1c27'); panel.paste(diff, ((480-diff.width)//2, (400-diff.height)//2))
    panel.save(out/'jabra-difference.png'); paths['difference'] = relative('jabra-difference.png')
    return {'fixture': False, 'title': 'Preserved Jabra earbud', 'model_backend': metadata['backend'],
            'appearance': metadata['appearance'], 'source_timestamp_s': reference['source_timestamp_s'],
            'reference_status': reference['status'], 'manual_geometry_or_pose_added': False,
            'source_reference': {'path': BASELINE, 'sha256': sha(reference_path)},
            'asset': {'path': str(asset.relative_to(ROOT)), 'sha256': sha(asset), 'bytes': asset.stat().st_size},
            'source_rgb': {'path': str(source.relative_to(ROOT)), 'sha256': sha(source)},
            'native_appearance': native_representation_receipt(native), 'metrics': metrics,
            'render_seconds': {'legacy': old_seconds, 'native': new_seconds}, 'images': paths,
            'automatic_display_crop_xyxy': list(crop), 'full_source_dimensions': [width, height],
            'pose': {'rotation': R.tolist(), 'translation': t.tolist(), 'K': K.tolist(), 'new_pose_fitting_executed': False},
            'physical_accuracy_verified': False, 'hidden_insertion_recovered': False}


def optional_real_results():
    """Read completed real receipts; never launch or finish an experiment."""
    result = {}
    run_path = ROOT/'data/native-appearance-real-fit-v1/run.json'
    fit_path = run_path.parent/'fit/experiment.json'
    if run_path.is_file() and fit_path.is_file():
        run, fit = json.loads(run_path.read_text()), json.loads(fit_path.read_text())
        if run.get('status') == 'complete' and run.get('synthetic_fixture') is False and run.get('gpu_used') is False:
            value = {'fixture': False, 'receipt_path': str(run_path.relative_to(ROOT)), 'receipt_sha256': sha(run_path),
                     'experiment_path': str(fit_path.relative_to(ROOT)), 'experiment_sha256': sha(fit_path),
                     'preview_path': str((fit_path.parent/'detail-preview.jpg').relative_to(ROOT)),
                     'wall_seconds': run['wall_seconds'], 'reference_iou': fit['reference_full_resolution_iou'],
                     'gate_reasons': fit['usable_pose_gate']['reasons'],
                     'physical_pose_verified': False, 'quality_improvement_claimed': False}
            observer = run_path.parent/'tracking-cpu-v1/observer.html'
            tracking = observer.parent/'experiment.json'
            if observer.is_file() and tracking.is_file():
                record = json.loads(tracking.read_text())
                if record.get('status') == 'complete_conditional_tracking_experiment' and record.get('runtime', {}).get('device') == 'cpu':
                    value['tracking_observer_path'] = str(observer.relative_to(ROOT))
                    value['tracking_experiment_sha256'] = sha(tracking)
            result['real_fit'] = value
    completed = []
    for path in (ROOT/'data/model-upgrades/runs').glob('upgrade-*/run.json'):
        try:
            run = json.loads(path.read_text())
            if run.get('request', {}).get('mode') != 'refit_native_cpu' or run.get('status') not in ('complete', 'complete_with_gaps'):
                continue
            motion = next(p for p in reversed(run['phases']) if p['name'] == 'motion' and p.get('result', {}).get('experiment'))
            experiment_path = (ROOT/motion['result']['experiment']).resolve()
            if not experiment_path.is_relative_to(ROOT/'data'):
                continue
            tracking = json.loads(experiment_path.read_text())
            if tracking.get('status') != 'complete_conditional_tracking_experiment' or tracking.get('runtime', {}).get('device') != 'cpu':
                continue
            observer = experiment_path.parent/'observer.html'
            completed.append((run.get('completed_at', ''), {'fixture': False, 'run_id': run['id'],
                              'receipt_path': str(path.relative_to(ROOT)), 'receipt_sha256': sha(path),
                              'tracking_experiment': str(experiment_path.relative_to(ROOT)), 'tracking_sha256': sha(experiment_path),
                              'observer_path': str(observer.relative_to(ROOT)) if observer.is_file() else None,
                              'comparison': run.get('result', {}).get('comparison'), 'physical_pose_verified': False}))
        except (ValueError, OSError, KeyError, TypeError, StopIteration):
            continue
    if completed:
        result['native_cpu_upgrade'] = sorted(completed, key=lambda item: item[0])[-1][1]
    return result


def build(version):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,39}', version):
        raise ValueError('Use a short lowercase version name')
    out = ROOT/'data'/f'native-appearance-review-{version}'
    out.mkdir(exist_ok=False)
    relative = lambda name: str((out/name).relative_to(ROOT))
    texture = calibration_texture(); texture.save(out/'software-test-texture.png')
    raw = fixture_glb(cube_parts(), texture, cube_materials()); (out/'software-fixture-cube.glb').write_bytes(raw)
    mesh = load_native_glb(raw)
    width, height = 640, 400; K = pinhole(width, height, 465)
    frames = []
    for index in range(10):
        yaw = -42+index*84/9; pitch = -19+6*np.sin(index*np.pi/9)
        R = rotation(pitch, yaw); t = np.array([0., 0., 4.1])
        result = render_native_mesh(mesh, K, R, t, width, height)
        images = emit_render(out, f'fixture-{index:02d}', result, relative)
        frames.append({'index': index, 'fixture': True, 'known_pose': True, 'yaw_deg': float(yaw), 'pitch_deg': float(pitch),
                       'rotation': R.tolist(), 'translation': t.tolist(), 'images': images,
                       'visible_pixels': int(result['mask'].sum()), 'known_depth_range': [float(result['depth'][result['mask']].min()), float(result['depth'][result['mask']].max())]})
    examples = []
    material = lambda color: {'doubleSided': True, 'pbrMetallicRoughness': {'baseColorFactor': color, 'baseColorTexture': {'index': 0}}}
    definitions = [
        ('perspective', 'Perspective follows the surface.', 'A tilted test plane keeps its texture in native UV coordinates. The displayed pattern is sampled with perspective-correct weights.',
         [quad(-1.3, 1.3)], None, rotation(-8, -43), [0, 0, 3.3]),
        ('occlusion', 'The nearer triangle wins.', 'Two known test surfaces overlap. One shared depth buffer determines which material reaches each pixel.',
         [quad(-1.25, 1.1, -.85, .8, .45), quad(-.45, 1.05, -.45, .6, -.2, 1)], [material([.65, 1, .9, 1]), material([1, .65, .9, 1])], rotation(-9, -16), [0, 0, 3.3]),
        ('seams', 'A seam is allowed to stay a seam.', 'The two test panels share geometric edge positions and keep separate UV vertices. We do not merge or bake them into vertex colours.',
         [quad(-1.3, 0, uv=[[0, 0], [.48, 0], [.48, 1], [0, 1]]), quad(0, 1.3, uv=[[.53, 0], [1, 0], [1, 1], [.53, 1]])], None, rotation(-4, -20), [0, 0, 3.3]),
    ]
    for key, title, summary, parts, materials, R, t in definitions:
        raw = fixture_glb(parts, texture, materials); (out/f'software-fixture-{key}.glb').write_bytes(raw)
        native = load_native_glb(raw); k = pinhole(480, 320, 360)
        result = render_native_mesh(native, k, R, np.asarray(t), 480, 320)
        images = emit_render(out, key, result, relative)
        examples.append({'id': key, 'title': title, 'summary': summary, 'fixture': True, 'images': images,
                         'asset_path': relative(f'software-fixture-{key}.glb'), 'asset_sha256': hashlib.sha256(raw).hexdigest(),
                         'vertices': len(native.vertices), 'unique_positions': len(np.unique(native.vertices, axis=0)),
                         'visible_triangles': len(np.unique(result['face_index'][result['mask']])),
                         'known_pose': {'rotation': R.tolist(), 'translation': t, 'K': k.tolist()}})
    baseline = build_baseline(out, relative)
    manifest = {'format': FORMAT, 'version': '1', 'fixture': True,
                'title': 'Native appearance · software verification',
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'Isolated rendering verification with explicitly synthetic fixtures and a preserved real learned baseline; no reconstruction or tracking inference.',
                'learned_inference_executed_for_this_review': False, 'network_or_credentials_used': False,
                'physical_accuracy_verified': False, 'production_catalog_modified': False,
                'fixture_asset': {'path': relative('software-fixture-cube.glb'), 'sha256': sha(out/'software-fixture-cube.glb')},
                'texture_path': relative('software-test-texture.png'), 'fixture_renderer_receipt': native_representation_receipt(mesh),
                'fixture_camera': {'K': K.tolist(), 'width': width, 'height': height, 'units': 'arbitrary test coordinates'},
                'frames': frames, 'examples': examples, 'baseline': baseline,
                **optional_real_results(),
                'builder_source': {'path': str(Path(__file__).relative_to(ROOT)), 'sha256': sha(Path(__file__))},
                'limitations': ['Known fixture geometry and poses test software; they are not recovered from a video.',
                                'Jabra pixels test backward compatibility, not geometry or pose accuracy.',
                                'No genuine SAM 3D or SAM 3.1 model inference ran for this review.',
                                'Generated appearance does not establish observed hidden surfaces or successful insertion.']}
    manifest['artifacts'] = [{'path': relative(path.name), 'sha256': sha(path), 'bytes': path.stat().st_size}
                             for path in sorted(out.iterdir()) if path.is_file()]
    write_json(out/'review.json', manifest)
    pointer = {'format': FORMAT+'-pointer', 'manifest_path': relative('review.json'), 'manifest_sha256': sha(out/'review.json'),
               'fixture': True, 'physical_accuracy_verified': False}
    write_json(ROOT/'workspace/appearance-review.json', pointer)
    # A local script payload supports file:// inspection without a fetch bypass.
    # The same manifest is exposed as JSON for independent inspection and hashes.
    payload = json.dumps({'pointer': pointer, 'review': manifest}, allow_nan=False, separators=(',', ':'))
    for char, replacement in [('<', '\\u003c'), ('>', '\\u003e'), ('&', '\\u0026'), ('\u2028', '\\u2028'), ('\u2029', '\\u2029')]:
        payload = payload.replace(char, replacement)
    (ROOT/'workspace/appearance-review-data.js').write_text('window.NATIVE_APPEARANCE_REVIEW='+payload+';\n')
    return {'directory': str(out), 'page': str(ROOT/'workspace/appearance.html'), 'manifest': relative('review.json'),
            'fixture_frames': len(frames), 'baseline_metrics': baseline['metrics'], 'artifact_count': len(manifest['artifacts'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default='v1', help='Fresh output suffix; an existing directory is never replaced')
    args = parser.parse_args()
    print(json.dumps(build(args.version), indent=2))


if __name__ == '__main__':
    main()
