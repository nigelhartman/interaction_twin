#!/usr/bin/env python3
"""Build a new, source-linked Studio trailer without inference or old outputs.

Native geometry is never rewritten. Orbits and graphic layouts are editorial
inspection cameras. Source masks and screenshots are actual saved outputs.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from native_appearance import load_native_glb, render_native_mesh
from portable_twin import decode_glb_bytes
from build_showcase_media import font, centered_mesh, VideoClip
font = lru_cache(maxsize=128)(font)

OUT = ROOT / 'output/showcase-studio-20261031'
MEDIA = ROOT / 'showcase/media'
W, H, FPS = 1600, 900, 24
BG, INK, MUTED, TEAL = '#f5f2eb', '#17323a', '#60767d', '#087f83'
BLUE, AMBER, PURPLE, LINE = '#4d80ca', '#d99b29', '#8870bd', '#d9e2de'
INPUT = 'data/observed-cotracker-experiment-20261101/jabra-rigid-input-v2/geometry-input.json'
PARTS = 'data/moving-parts-reconstruction-20261031/jabra-parts-triposr-v1/display-manifest.json'
CAMERA = 'data/observed-app-runs-20261031/observed-112663a646eba4fb38ec3893'
DURATIONS = [4, 6, 6, 6, 6, 8, 6]
GRAPHIC_NAME = 'studio-pipeline-v2'


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def relative(path): return Path(path).resolve().relative_to(ROOT).as_posix()
def local(value):
    path = (ROOT / value.removeprefix('/files/')).resolve()
    if not path.is_relative_to(ROOT): raise ValueError('Media input escaped the project')
    return path
def record(path, kind, **extra): return dict(path=relative(path), sha256=sha(path), kind=kind, **extra)
def write(path, value):
    with Path(path).open('x') as stream: json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
def label(canvas, xy, value, size=24, color=INK, bold=False, spacing=9):
    ImageDraw.Draw(canvas).multiline_text(xy, value, font=font(size, bold), fill=color, spacing=spacing)
def panel(canvas, image, box, fill='white', contain=True):
    x, y, width, height = box
    piece = Image.new('RGB', (width, height), fill)
    content = ImageOps.contain(image.convert('RGB'), (width, height), Image.Resampling.LANCZOS) if contain else ImageOps.fit(image.convert('RGB'), (width, height), Image.Resampling.LANCZOS)
    piece.paste(content, ((width-content.width)//2, (height-content.height)//2))
    mask = Image.new('L', (width, height)); ImageDraw.Draw(mask).rounded_rectangle((0, 0, width-1, height-1), 24, fill=255)
    canvas.paste(piece, (x, y), mask)
    ImageDraw.Draw(canvas).rounded_rectangle((x, y, x+width-1, y+height-1), 24, outline=LINE, width=2)
def chip(canvas, xy, text, color=TEAL):
    x, y = xy; width = int(font(18, True).getlength(text)) + 28
    ImageDraw.Draw(canvas).rounded_rectangle((x, y, x+width, y+32), 12, fill=color)
    label(canvas, (x+14, y+5), text, 18, 'white', True)
def base(chapter, title, subtitle, footer, number):
    canvas = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(canvas)
    d.rounded_rectangle((62, 43, 74, 62), 5, fill=TEAL)
    label(canvas, (90, 39), 'INTERACTION TWIN', 20, INK, True)
    label(canvas, (64, 92), chapter.upper(), 17, TEAL, True)
    label(canvas, (60, 126), title, 54, INK, True)
    label(canvas, (64, 197), subtitle, 24, MUTED)
    d.line((64, 840, 1536, 840), fill=LINE, width=2)
    label(canvas, (64, 855), footer, 17, MUTED)
    label(canvas, (1446, 855), number, 17, MUTED, True)
    return canvas


def graphic():
    """Code-native research schematic; never presented as a model output."""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="1680" height="850" viewBox="0 0 1680 850">
<defs><marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="none" stroke="#60767d" stroke-width="1.6"/></marker>
<style>text{font-family:Arial,sans-serif;fill:#17323a} .h{font-size:25px;font-weight:700}.s{font-size:20px;fill:#60767d}.b{fill:white;stroke:#d5dfda;stroke-width:2}.a{fill:none;stroke:#60767d;stroke-width:3;marker-end:url(#a)}.tag{font-size:16px;font-weight:700;letter-spacing:1px}</style></defs>
<rect width="1680" height="850" rx="24" fill="#f5f2eb"/>
<text x="64" y="72" font-size="40" font-weight="700">Show it once. Reuse the interaction.</text>
<text x="66" y="110" class="s">Video → inspectable parts, source-linked geometry and a descriptive program for agents</text>
<rect x="64" y="280" width="220" height="240" rx="22" class="b"/>
<rect x="131" y="304" width="86" height="118" rx="12" fill="#e8f1ed" stroke="#087f83" stroke-width="3"/>
<path d="M168 341l25 20-25 20z" fill="#087f83"/>
<text x="91" y="463" class="h">Captured video</text><text x="92" y="496" class="s">Source time + RGB</text>
<rect x="340" y="222" width="280" height="150" rx="22" fill="#eef6f5" stroke="#087f83" stroke-width="2"/>
<text x="368" y="262" class="h">SAM 3.1</text><text x="368" y="298" class="s">Separate visible parts</text><text x="368" y="330" class="s">Masks + time + visibility</text>
<rect x="340" y="410" width="280" height="150" rx="22" fill="#eef6f5" stroke="#087f83" stroke-width="2"/>
<text x="368" y="451" class="h">Muse Spark</text><text x="368" y="486" class="s">Propose parts + relations</text><text x="368" y="519" class="s">Steps + source evidence</text>
<path d="M284 364H310V297H338M310 364V486H338" class="a"/>
<rect x="684" y="174" width="360" height="198" rx="22" fill="#eef3fb" stroke="#4d80ca" stroke-width="2"/>
<text x="710" y="210" class="tag" style="fill:#4d80ca">OBSERVED BRANCH</text>
<text x="710" y="249" class="h">Track → multiview math</text><text x="710" y="286" class="s">SIFT / CoTracker · triangulation</text><text x="710" y="319" class="s">Stereo patches · held-out checks</text><text x="710" y="351" class="s">Original source RGB texture</text>
<rect x="684" y="414" width="360" height="198" rx="22" fill="#fcf4e6" stroke="#d99b29" stroke-width="2"/>
<text x="710" y="451" class="tag" style="fill:#b67c13">INFERRED BRANCH · OPTIONAL</text>
<text x="710" y="490" class="h">Model / completion priors</text><text x="710" y="527" class="s">TripoSR part previews</text><text x="710" y="560" class="s">MoGe depth / small-gap filling</text><text x="710" y="591" class="s">Separate layers · accuracy unknown</text>
<path d="M620 297H651V273H680M651 297V513H680" class="a"/>
<rect x="1108" y="258" width="260" height="274" rx="22" class="b"/>
<text x="1136" y="300" class="h">Interaction asset</text><text x="1136" y="343" class="s">GLB + component IDs</text><text x="1136" y="381" class="s">Procedure + relations</text><text x="1136" y="419" class="s">Evidence + quality</text><text x="1136" y="457" class="s">Versions + limitations</text><text x="1136" y="496" class="s">Observed ≠ inferred</text>
<path d="M1044 273H1076V349H1105M1044 513H1076V447H1105M620 486H649V659H1238V535" class="a"/>
<rect x="1430" y="298" width="190" height="194" rx="22" fill="#f2eef8" stroke="#8870bd" stroke-width="2"/>
<text x="1457" y="343" class="h">MCP</text><text x="1457" y="382" class="s">Agent discovery</text><text x="1457" y="418" class="s">Inspect + export</text><text x="1457" y="454" class="s">Read-only tools</text>
<path d="M1368 395H1426" class="a"/>
<rect x="64" y="719" width="1556" height="76" rx="18" fill="#e7eeea"/>
<text x="88" y="751" class="s">PIPELINE SCHEMATIC · Working local prototype. Separate part gauges; physical scale, common assembly and contact remain unverified.</text>
<text x="88" y="779" class="s">Geometry generation is not physical simulation. SAM 3D / cluster reconstruction has not run in this demonstration.</text></svg>'''
    path = MEDIA / (GRAPHIC_NAME + '.svg')
    with path.open('x') as stream: stream.write(svg)
    # This SVG uses only rectangles, text and orthogonal paths. Render those
    # same explicit elements with existing Pillow; no renderer download.
    raster = Image.new('RGB', (1680, 850), BG); draw = ImageDraw.Draw(raster)
    for element in ET.fromstring(svg):
        tag = element.tag.rsplit('}', 1)[-1]; a = element.attrib
        if tag == 'rect':
            x, y = float(a.get('x', 0)), float(a.get('y', 0))
            box = (x, y, x+float(a['width']), y+float(a['height']))
            draw.rounded_rectangle(box, float(a.get('rx', 0)), fill=a.get('fill', 'white'),
                outline=a.get('stroke', '#d5dfda' if a.get('class') == 'b' else None), width=int(a.get('stroke-width', 2)))
        elif tag == 'text':
            cls = a.get('class'); size = int(a.get('font-size', {'h':25, 's':20, 'tag':16}.get(cls, 20)))
            color = '#60767d' if cls == 's' else INK
            if 'fill:' in a.get('style', ''): color = a['style'].split('fill:', 1)[1].split(';')[0]
            draw.text((float(a['x']), float(a['y'])), element.text or '',
                font=font(size, cls in {'h', 'tag'} or a.get('font-weight') == '700'), fill=color, anchor='ls')
        elif tag == 'path' and a.get('class') == 'a':
            tokens = re.findall(r'[MHV]|-?\d+(?:\.\d+)?', a['d']); i = 0; points = []
            def finish():
                if len(points) < 2: return
                draw.line(points, fill=MUTED, width=3, joint='curve')
                tip, before = np.asarray(points[-1]), np.asarray(points[-2]); direction = tip-before
                direction = direction / np.linalg.norm(direction); side = np.array([-direction[1], direction[0]])
                wings = [tuple(tip-direction*11+side*6), tuple(tip), tuple(tip-direction*11-side*6)]
                draw.line(wings, fill=MUTED, width=3)
            while i < len(tokens):
                command = tokens[i]; i += 1
                if command == 'M':
                    finish(); points = [(float(tokens[i]), float(tokens[i+1]))]; i += 2
                elif command == 'H': points.append((float(tokens[i]), points[-1][1])); i += 1
                elif command == 'V': points.append((points[-1][0], float(tokens[i]))); i += 1
            finish()
    draw.polygon([(168,341),(193,361),(168,381)], fill=TEAL)
    raster.save(MEDIA / (GRAPHIC_NAME + '.png'))
    return [record(path, 'illustrative_pipeline_schematic'), record(MEDIA / (GRAPHIC_NAME + '.png'), 'same_explicit_schematic_elements_Pillow_raster')]


def prepare():
    OUT.mkdir(parents=True, exist_ok=True); MEDIA.mkdir(exist_ok=True)
    if (OUT / 'preparation.json').exists(): raise ValueError('Preserve the existing prepared media')
    source = read(ROOT / INPUT)
    frames_dir = OUT / 'mask-sequence'; frames_dir.mkdir()
    sequence = []
    for row in source['evidence']:
        if not 11.99 <= row['timestamp_s'] <= 18.01: continue
        rgb_path = local(row['frame_path'])
        if sha(rgb_path) != row['frame_sha256']: raise ValueError('Source frame changed')
        inputs, masks = [], []
        for observation in row['observations']:
            path = local(observation['mask_path'])
            if sha(path) != observation['mask_sha256']: raise ValueError('SAM mask changed')
            masks.append(np.asarray(Image.open(path)) > 127)
            inputs.append(record(path, 'genuine_SAM31_video_mask', component_id=observation['track_id'],
                                 native_instance=observation.get('native_instance')))
        # Extend the legacy helper's 3-colour list explicitly for the fourth
        # component; source mask geometry stays unchanged.
        rgb = np.array(Image.open(rgb_path).convert('RGB'))
        colors = {'case_base':[20, 186, 177], 'case_lid':[236, 140, 68],
                  'earbud_a':[119, 139, 240], 'earbud_b':[197, 111, 208]}
        import cv2
        for mask, source_mask in zip(masks, inputs):
            color = np.asarray(colors[source_mask['component_id']])
            rgb[mask] = np.rint(.62 * rgb[mask] + .38 * color).astype(np.uint8)
            edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
            rgb[edge] = color
        output = frames_dir / (row['id'] + '.jpg'); Image.fromarray(rgb).save(output, quality=95)
        sequence.append(dict(file=relative(output), sha256=sha(output), source_pts_s=row['timestamp_s'],
            source_frame=record(rgb_path, 'exact_saved_source_RGB'), masks=inputs,
            rendering='tinted_mask_overlay_edges_only_no_geometry_or_interpolation'))
    if not sequence: raise ValueError('No real source mask frames available')
    parts = read(ROOT / PARTS); orbits = OUT / 'native-part-orbits'; orbits.mkdir()
    assets = []
    for component in parts['components']:
        path = local(component['glb_url'])
        if sha(path) != component['glb_sha256']: raise ValueError('Native learned GLB changed')
        decoded = decode_glb_bytes(path.read_bytes())
        subdir = orbits / component['proposal_id']; subdir.mkdir()
        for n in range(48):
            image = centered_mesh(decoded['primitives'], (620, 510), -35 + n * 360 / 48)
            pane = Image.new('RGB', image.size, '#e9efeb'); pane.paste(image, (0, 0), image)
            pane.save(subdir / f'{n:03d}.jpg', quality=94)
        assets.append(record(path, 'untouched_single_image_TripoSR_prior', component_id=component['proposal_id'],
            source_image=record(local(component['source_image_url']), 'original_model_source_image'),
            source_mask=record(local(component['mask_url']), 'original_model_source_mask'),
            geometry_modified=False, physical_assembly_established=False, output_directory=relative(subdir)))
    dense = read(local(CAMERA + '-dense/viewer-display.json'))
    completion = read(local(CAMERA + '-completion/viewer.json'))
    export = read(local(CAMERA + '-workflow/result.json'))['exports']
    observed_glb = local(export['observed']['glb_url']); inferred_glb = local(export['inferred']['inferred_only_url'])
    if sha(observed_glb) != export['observed']['glb_sha256'] or sha(inferred_glb) != export['inferred']['inferred_only_sha256']:
        raise ValueError('Recorded observed or inferred GLB changed')
    observed = load_native_glb(observed_glb); inferred = load_native_glb(inferred_glb)
    source_frame = next(r for r in dense['frames'] if r['id'] == dense['surface']['source_frame_id'])
    camera = next(c for c in dense['cameras'] if c['frame_id'] == source_frame['id'])
    rotations = OUT / 'camera-patch-orbits'; rotations.mkdir()
    center = (observed.vertices.min(0) + observed.vertices.max(0)) / 2
    span = float(np.max(observed.vertices.max(0) - observed.vertices.min(0)))
    size = (700, 420); K = np.array([[650., 0, size[0]/2], [0, 650., size[1]/2], [0, 0, 1.]])
    for n in range(24):
        angle = -18 + 36*n/23
        R = Rotation.from_euler('y', angle, degrees=True).as_matrix() @ np.asarray(camera['R'])
        t = np.array([0., 0., span*2.1]) - R @ center
        obs = render_native_mesh(observed, K, R, t, *size)
        inf = render_native_mesh(inferred, K, R, t, *size)
        backdrop = np.full((size[1], size[0], 3), [233, 239, 235], np.uint8)
        image = backdrop.copy(); image[obs['mask']] = obs['rgb'][obs['mask']]
        combined = image.copy(); use = inf['mask'] & (~obs['mask'] | (inf['depth'] < obs['depth']))
        amber_rgb = np.array([226, 166, 63])
        combined[use] = np.rint(.25*inf['rgb'][use] + .75*amber_rgb).astype(np.uint8)
        Image.fromarray(image).save(rotations / f'observed-{n:03d}.jpg', quality=95)
        Image.fromarray(combined).save(rotations / f'inferred-{n:03d}.jpg', quality=95)
    source_image = local(source_frame['image_url'])
    manifest = {'format': 'source-linked-studio-showcase-media', 'version': 1,
        'build_implementation': record(Path(__file__), 'media_builder_source'),
        'inputs': [record(ROOT / INPUT, 'frozen_actual_SAM31_video_input'), record(ROOT / PARTS, 'native_model_display_manifest'),
                   record(source_image, 'camera_original_source_RGB', source_pts_s=source_frame['timestamp_s']),
                   record(observed_glb, 'actual_observed_open_patch'), record(inferred_glb, 'actual_separate_inferred_depth_prior'),
                   record(local(CAMERA + '-workflow/result.json'), 'actual_workflow_receipts')],
        'source_video': record(local(source['source']['path']), 'upright_SDR_recorded_video',
            original_sha256=source['source']['original_sha256']),
        'mask_sequence': sequence, 'native_assets': assets, 'camera_source_image': relative(source_image),
        'camera_orbit_directory': relative(rotations),
        'geometry_counts': {'observed': len(dense['surface']['triangles']), 'inferred': len(completion['inferred_surface']['triangles'])},
        'display_policy': {'source_mask_sequence': 'Exact saved 3fps source timestamps held; no invented frames or masks.',
            'native_turntable': 'Authored inspection camera rotations and centering only; independent part gauges, no recovered assembly.',
            'camera_surface_orbit': 'Authored inspection camera orbit over unchanged observed and separately inferred arrays; not recovered object motion.',
            'inferred_display': 'Amber tint is a display overlay only; no source GLB/material bytes are rewritten.',
            'native_appearance': 'Original source RGB for observed patch; original generated vertex colour plus display lighting for TripoSR.',
            'editorial': 'Typography, panels, arrows and scene transitions are authored; no fabricated UI or success status.',
            'physical_accuracy_verified': False, 'simulation_ready': False, 'sam3d_run': False,
            'audio': 'Silent; no external soundtrack', 'private_capture': 'Local only; no publication performed'},
        'new_api_calls': 0, 'new_model_inference_calls': 0, 'cluster_contacted': False,
        'outputs': graphic()}
    write(OUT / 'preparation.json', manifest)
    print(json.dumps({'status': 'prepared', 'mask_frames': len(sequence), 'source': str(OUT)}), flush=True)


def frame(section, u, assets, live=None):
    if section == 0:
        c = Image.new('RGB', (W, H), BG)
        label(c, (68, 52), 'INTERACTION TWIN', 22, TEAL, True)
        label(c, (62, 208), 'Show it once.\nReuse the\ninteraction.', 81, INK, True, 8)
        label(c, (69, 556), 'Recorded video → parts, geometry\nand a source-linked program for agents.', 28, MUTED, spacing=10)
        chip(c, (70, 760), 'WORKING LOCAL PROTOTYPE')
        panel(c, live, (1062, 62, 430, 776))
    elif section == 1:
        c = base('01 · Meta perception', 'Find the parts.', 'Actual SAM 3.1 video masks. Every sample keeps its source timestamp.',
            'Model-proposed masks; physical identity and occluded states remain uncertain.', '02 / 07')
        n = min(len(assets['mask_sequence'])-1, int(u*len(assets['mask_sequence'])))
        record_ = assets['mask_sequence'][n]
        panel(c, assets['source_frames'][n], (66, 270, 313, 555))
        panel(c, assets['mask_frames'][n], (435, 270, 313, 555))
        label(c, (829, 306), 'ONE RECORDING.\nSEPARATE COMPONENTS.', 33, INK, True, 12)
        names = [('Case base', TEAL), ('Case lid', '#d78f3f'), ('Earbud A', '#778bf0'), ('Earbud B', '#c56fd0')]
        for j, (name, color) in enumerate(names):
            ImageDraw.Draw(c).ellipse((833, 440+j*60, 849, 456+j*60), fill=color)
            label(c, (873, 432+j*60), name, 27)
        chip(c, (833, 739), f'EXACT SOURCE · {record_["source_pts_s"]:.3f} s')
    elif section == 2:
        c = base('02 · Observed geometry', 'Recover what the camera saw.', 'Source correspondences → multiview math → a genuine source-textured patch.',
            'Visible open patch only. Intrinsics, scale and shape accuracy are unverified.', '03 / 07')
        panel(c, assets['camera_source'], (67, 295, 700, 410))
        n = min(23, int(u*24)); panel(c, assets['observed'][n], (829, 295, 700, 410), '#e9efeb')
        chip(c, (83, 730), 'ORIGINAL SOURCE RGB')
        chip(c, (846, 730), '5,377 OBSERVED TRIANGLES', BLUE)
        label(c, (850, 782), 'Illustrative inspection-camera orbit', 20, MUTED)
    elif section == 3:
        c = base('03 · Explicit uncertainty', 'Keep inferred geometry separate.', 'Optional depth-prior completion adds a distinct, inspectable layer.',
            'Every amber face is inferred. Source colour does not validate its depth.', '04 / 07')
        n = min(23, int(u*24)); panel(c, assets['observed'][n], (67, 290, 700, 420), '#e9efeb')
        panel(c, assets['inferred'][n], (829, 290, 700, 420), '#e9efeb')
        chip(c, (84, 735), 'OBSERVED PATCH', BLUE)
        chip(c, (845, 735), '+ 27,895 INFERRED TRIANGLES', '#b77d19')
        label(c, (851, 786), 'MoGe 2 Small · unknown physical accuracy', 20, MUTED)
    elif section == 4:
        c = base('04 · Independent part previews', 'Inspect each component.', 'Separate native single-image priors. Original GLBs stay unchanged.',
            'TripoSR comparison models; independent gauges. Common scale, hinge and assembly are unfinished.', '05 / 07')
        n = min(47, int(u*48))
        panel(c, assets['base_orbit'][n], (67, 278, 700, 512), '#e9efeb')
        panel(c, assets['lid_orbit'][n], (829, 278, 700, 512), '#e9efeb')
        chip(c, (84, 299), 'CASE BASE · SINGLE-IMAGE PRIOR')
        chip(c, (846, 299), 'CASE LID · SINGLE-IMAGE PRIOR')
        label(c, (1007, 800), 'Camera orbit + display lighting', 19, MUTED)
    elif section == 5:
        c = Image.new('RGB', (W, H), BG)
        label(c, (64, 35), 'RECONSTRUCTION STUDIO', 22, TEAL, True)
        label(c, (651, 31), 'One workspace. Inspect the whole flow.', 32, INK, True)
        index = min(len(assets['studio_screens'])-1, int(u*len(assets['studio_screens'])))
        panel(c, assets['studio_screens'][index], (32, 98, 1536, 718))
        label(c, (64, 855), 'Actual app screenshots · preserved successes and unsupported stages · source checks and exports', 17, MUTED)
        label(c, (1446, 855), '06 / 07', 17, MUTED, True)
    else:
        c = Image.new('RGB', (W, H), INK)
        label(c, (67, 52), 'INTERACTION TWIN', 22, '#8dd5cc', True)
        label(c, (62, 190), 'Assets agents\ncan inspect.', 76, '#f5f2eb', True, 10)
        label(c, (68, 401), 'Geometry. Components. Evidence.\nProcedures. Versions. Limits.', 30, '#c7d8d7', spacing=12)
        for j, name in enumerate(['OBSERVED', 'INFERRED', 'SOURCE-LINKED']):
            chip(c, (68, 570+j*48), name, [BLUE, '#b77d19', TEAL][j])
        screen = assets['screens'][-1]
        panel(c, screen, (640, 147, 890, 541), '#f5f2eb')
        label(c, (644, 727), 'Read-only MCP resources + downloadable artifacts', 24, '#c7d8d7')
        label(c, (68, 827), 'Working local prototype · metric scale, physical assembly and contact remain unverified.', 20, '#a9c1c3')
    return c


def encode(screenshots):
    if len(screenshots) < 2: raise ValueError('Two or more actual app screenshots with source URLs are required')
    for item in screenshots:
        source = local(item['path'])
        if not source.is_relative_to(ROOT / 'output') or source.suffix.lower() not in {'.png','.jpg','.jpeg'}:
            raise ValueError('Use only actual captured image files under project output')
        if not source.is_file() or not item.get('url') or not item.get('description'):
            raise ValueError('Actual screenshot file, URL and visible-state description are required')
        with Image.open(source) as image: image.verify()
    manifest = read(OUT / 'preparation.json')
    # Preserve the original preparation and its exact code while freezing the
    # final encoder source separately after visual layout review.
    snapshot = OUT / 'media-builder-at-encode.py'
    snapshot.write_bytes(Path(__file__).read_bytes())
    original_code = OUT / 'media-builder-at-prepare.py'
    if original_code.exists():
        if sha(original_code) != manifest['build_implementation']['sha256']:
            raise ValueError('Frozen preparation code changed')
        manifest['build_implementation'] = record(original_code, 'exact_media_prepare_builder_snapshot')
    manifest['encoder_implementation'] = record(snapshot, 'exact_final_media_encoder_snapshot')
    manifest['original_preparation'] = record(OUT / 'preparation.json', 'preserved_preparation_receipt')
    manifest['outputs'] += [record(MEDIA / (GRAPHIC_NAME + extension), 'final_pipeline_schematic') for extension in ['.svg', '.png']]
    dependencies = OUT / 'frozen-renderers'; dependencies.mkdir()
    manifest['renderer_dependencies'] = []
    for name in ['scripts/build_showcase_media.py', 'native_appearance.py', 'alignment_render.py', 'portable_twin.py']:
        source = ROOT / name; frozen = dependencies / Path(name).name
        shutil.copyfile(source, frozen)
        manifest['renderer_dependencies'].append(record(frozen, 'existing_renderer_code_snapshot',
            original_path=name, no_old_prepare_or_encode_called=True))
    manifest['rendered_asset_inputs'] = [record(path, 'source_linked_editorial_native_camera_render') for path in
        sorted((OUT / 'native-part-orbits').glob('*/*.jpg')) + sorted((OUT / 'camera-patch-orbits').glob('*.jpg'))]
    manifest['render_recipe'] = {'native_part_orbit_angles_degrees':[-35+n*360/48 for n in range(48)],
        'native_part_initial_inspection_euler_xyz_degrees':[-18,'variable',12],
        'camera_patch_y_orbit_degrees':[-18+36*n/23 for n in range(24)],
        'camera_patch_K':[[650,0,350],[0,650,210],[0,0,1]],
        'camera_patch_translation':'[0,0,2.1*observed_bbox_max_span] - display_R @ observed_bbox_center',
        'geometry_modified':False,'inspection_camera_is_authored':True}
    screen_folder = OUT / 'screenshots'; screen_folder.mkdir()
    copied, screen_records = [], []
    for n, item in enumerate(screenshots):
        source = local(item['path'])
        if not item.get('url') or not item.get('description'): raise ValueError('Screenshot URL and visible-state description are required')
        destination = screen_folder / f'{n:02d}-{source.name}'
        shutil.copyfile(source, destination); copied.append(destination)
        captured = record(destination, 'actual_app_screenshot', original=record(source, 'unaltered_capture'),
            url=item['url'], description=item['description'], captured_at=item.get('captured_at'),
            role=item.get('role','studio'), authored_UI=False)
        if item.get('provenance_path'):
            proof = local(item['provenance_path'])
            if not proof.is_relative_to(ROOT / 'output') or proof.suffix != '.json':
                raise ValueError('Screenshot capture proof must be project output JSON')
            proof_data = read(proof)
            if proof_data.get('sha256') and proof_data['sha256'] != sha(source):
                raise ValueError('Actual screenshot differs from capture proof')
            frozen = screen_folder / f'{n:02d}-capture-provenance.json'; shutil.copyfile(proof, frozen)
            captured['capture_proof'] = record(frozen, 'unaltered_capture_provenance')
            captured['capture_file_mtime_utc'] = proof_data.get('captured_at_utc_from_filesystem_mtime')
        screen_records.append(captured)
    sequence = manifest['mask_sequence']; rotations = local(manifest['camera_orbit_directory'])
    assets = dict(mask_sequence=sequence, source_frames=[Image.open(local(r['source_frame']['path'])).convert('RGB') for r in sequence],
        mask_frames=[Image.open(local(r['file'])).convert('RGB') for r in sequence],
        camera_source=Image.open(local(manifest['camera_source_image'])).convert('RGB'),
        observed=[Image.open(rotations / f'observed-{n:03d}.jpg').convert('RGB') for n in range(24)],
        inferred=[Image.open(rotations / f'inferred-{n:03d}.jpg').convert('RGB') for n in range(24)],
        base_orbit=[Image.open(OUT / 'native-part-orbits/case_base' / f'{n:03d}.jpg').convert('RGB') for n in range(48)],
        lid_orbit=[Image.open(OUT / 'native-part-orbits/case_lid' / f'{n:03d}.jpg').convert('RGB') for n in range(48)],
        screens=[Image.open(p).convert('RGB') for p in copied],
        studio_screens=[Image.open(p).convert('RGB') for p,item in zip(copied,screenshots) if item.get('role') != 'agents'])
    if not assets['studio_screens']: raise ValueError('At least one real Studio screenshot is required')
    output = MEDIA / 'studio-trailer-v1.mp4'
    if output.exists(): raise ValueError('Preserve the previous trailer; use a new version')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W}x{H}', '-r', str(FPS), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast',
        '-crf', '19', '-threads', '4', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    log = (OUT / 'encode.log').open('x')
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
    live = VideoClip(local(manifest['source_video']['path']), 12, DURATIONS[0], size=(432, 768))
    story, frame_count, started = [], 0, time.monotonic()
    try:
        for section, seconds in enumerate(DURATIONS):
            for n in range(seconds*FPS):
                u = n / max(1, seconds*FPS-1)
                image = frame(section, u, assets, live.frame() if section == 0 else None)
                if section not in {0, 6}:
                    ImageDraw.Draw(image).line((64, 830, 64+int(1472*frame_count/(sum(DURATIONS)*FPS)), 830), fill=TEAL, width=3)
                if n == seconds*FPS//2:
                    path = OUT / f'storyboard-{section+1:02d}.jpg'; image.save(path, quality=96); story.append(path)
                process.stdin.write(image.tobytes()); frame_count += 1
            if section == 0: live.close()
            print(json.dumps({'phase': 'editorial_chapter_encoded', 'chapter': section+1, 'frames': frame_count}), flush=True)
        process.stdin.close()
        if process.wait(timeout=180): raise RuntimeError('Trailer encode failed; preserved encode.log')
    finally:
        if live.process.poll() is None: live.close()
        if process.poll() is None: process.terminate(); process.wait(timeout=20)
        log.close()
    poster = MEDIA / 'studio-trailer-poster-v1.jpg'
    shutil.copyfile(story[0], poster)
    storyboard = Image.new('RGB', (1200, 1350), '#dce5df')
    for n, path in enumerate(story): storyboard.paste(Image.open(path).resize((600, 338)), ((n%2)*600, (n//2)*338))
    storyboard.save(OUT / 'storyboard.jpg', quality=94)
    webm = MEDIA / 'studio-trailer-v1.webm'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(output), '-an', '-c:v', 'libvpx-vp9',
        '-threads', '2', '-b:v', '0', '-crf', '34', str(webm)], check=True)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(output)], text=True))
    if any(row['codec_type'] == 'audio' for row in probe['streams']): raise ValueError('Trailer must be silent')
    if abs(float(probe['format']['duration'])-sum(DURATIONS)) > .1: raise ValueError('Unexpected trailer duration')
    manifest.update(screenshots=screen_records, trailer={'path': relative(output), 'duration_s': sum(DURATIONS),
        'width': W, 'height': H, 'fps': FPS, 'frames': frame_count, 'render_seconds': round(time.monotonic()-started, 3),
        'source_playback': 'First4seconds decode recorded source interval12..16s; frame-rate conversion drops/duplicates only.',
        'chapters': [{'chapter': n+1, 'start_s': sum(DURATIONS[:n]), 'duration_s': value} for n, value in enumerate(DURATIONS)]})
    manifest['outputs'] += [record(path, 'new_showcase_media') for path in [output, webm, poster, OUT / 'storyboard.jpg']]
    write(OUT / 'source-manifest.json', manifest); write(OUT / 'ffprobe.json', probe)
    print(json.dumps({'status': 'encoded', 'duration_s': sum(DURATIONS), 'frames': frame_count, 'output': relative(output)}), flush=True)


def verify():
    """Check real files/decoding and save frames from the encoded trailer."""
    manifest = read(OUT / 'source-manifest.json'); checked = []
    def visit(value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                path = local(value['path'])
                if sha(path) != value['sha256']: raise ValueError('Media provenance hash mismatch: ' + str(path))
                checked.append(relative(path))
            if isinstance(value.get('file'), str) and isinstance(value.get('sha256'), str):
                path = local(value['file'])
                if sha(path) != value['sha256']: raise ValueError('Mask-overlay frame hash mismatch')
                checked.append(relative(path))
            for item in value.values(): visit(item)
        elif isinstance(value, list):
            for item in value: visit(item)
    visit(manifest)
    probes = []
    for suffix in ['mp4', 'webm']:
        video = MEDIA / ('studio-trailer-v1.' + suffix)
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(video)], text=True))
        videos = [s for s in probe['streams'] if s['codec_type'] == 'video']
        if len(videos) != 1 or any(s['codec_type'] == 'audio' for s in probe['streams']): raise ValueError('Unexpected audio/video stream count')
        stream = videos[0]
        if (stream['width'], stream['height']) != (W,H) or stream['avg_frame_rate'] != '24/1': raise ValueError('Unexpected encoded video dimensions/rate')
        if abs(float(probe['format']['duration'])-sum(DURATIONS)) > .1: raise ValueError('Unexpected encoded video duration')
        subprocess.run(['ffmpeg', '-hide_banner', '-v', 'error', '-i', str(video), '-an', '-f', 'null', '-'], check=True)
        probes.append({'path': relative(video), 'sha256': sha(video), 'codec': stream['codec_name'],
            'duration_s': float(probe['format']['duration']), 'fully_decoded_without_errors': True, 'audio_streams': 0})
    import cv2
    capture = cv2.VideoCapture(str(MEDIA / 'studio-trailer-v1.mp4'))
    frames = []; montage = Image.new('RGB', (1200,1352), '#dce5df')
    try:
        for n, seconds in enumerate(DURATIONS):
            timestamp = sum(DURATIONS[:n]) + seconds/2
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp*1000)
            success, array = capture.read()
            if not success or array.std() < 10: raise ValueError('Encoded trailer frame is missing or blank')
            image = Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
            path = OUT / f'verified-frame-{n+1:02d}.png'; image.save(path)
            montage.paste(image.resize((600,338)), ((n%2)*600,(n//2)*338))
            frames.append(record(path, 'actual_decoded_trailer_frame', trailer_time_s=timestamp))
    finally: capture.release()
    montage.save(OUT / 'verified-storyboard.jpg', quality=95)
    report = {'format':'studio-showcase-media-verification','passed':True,
        'provenance_contracts_checked':len(checked),'unique_provenance_paths':len(set(checked)),
        'source_manifest_sha256':sha(OUT/'source-manifest.json'),'encoded_videos':probes,
        'actual_decoded_frames':frames,'visual_contact_sheet':relative(OUT/'verified-storyboard.jpg'),
        'native_model_inputs_byte_preserved':True,'new_api_calls':0,'new_model_inference_calls':0,
        'manual_object_geometry_or_recovered_poses_used':False,'authored_inspection_camera_orbits':True,
        'cluster_contacted':False,'public_publication':False,
        'scope':'File identity, full CPU video decoding and real-frame inspection; no physical geometry or simulator validation.'}
    write(OUT/'verification.json',report)
    print(json.dumps({'status':'verified','provenance_contracts':len(checked),'videos':len(probes),'decoded_frames':len(frames)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'encode', 'verify'])
    parser.add_argument('--screenshots', type=Path, help='JSON array of actual screenshot path/url/description records')
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'verify': verify()
    elif args.screenshots: encode(read(args.screenshots))
    else: raise ValueError('Actual screenshot records are required before encoding')
