#!/usr/bin/env python3
"""Package full-duration video, object IDs and camera/battery pose candidates."""
import base64
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stage4_observer import DOCUMENT as SIMPLE_DOCUMENT

CASE = ROOT / 'data/camera-battery-full-scene'


def read(path):
    return json.loads(path.read_text())


def uri(path, mime):
    return 'data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode()


def mesh_data(path):
    mesh = trimesh.load(path, force='mesh')
    # A readable wire projection needs structural edges, not every triangulation
    # diagonal on a coplanar camera panel.
    adjacent = mesh.face_adjacency_edges[mesh.face_adjacency_angles > np.deg2rad(12)]
    if len(mesh.faces) < 600:
        # Cylinder side edges reveal its long axis and round ends.
        adjacent = mesh.edges_unique
    return {'vertices': mesh.vertices.tolist(), 'faces': mesh.faces.tolist(),
            'colors': np.array(mesh.visual.vertex_colors)[:, :3].tolist(), 'edges': adjacent.tolist()}


DOCUMENT = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Full video / object alignment observer</title><style>BASE_STYLE
#scene-labels{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.chapters{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}.chapters button{font-size:13px;padding:6px 11px}.legend{display:flex;flex-wrap:wrap;gap:10px;margin:17px 0 6px}.legend span{border:1px solid var(--line);padding:6px 10px;border-radius:7px;font-size:12px}.legend b{font-weight:550}#object-list{color:#9eb4c6;font-size:12px;min-height:20px;margin-top:8px}.panel-tools{min-height:90px}.coverage{color:#9eb4c6;font-size:12px}.footer{padding-bottom:15px}.footnote{font-size:12px;color:#c1ac80;margin-top:12px}
</style></head><body><main>
<header><div><h1>One video. Separate objects. A shared 3D view.</h1><p>The full 74-second recording, with readable IDs and inspected pose candidates.</p></div><span class="stage">CAMERA + BATTERIES · LOCAL PROTOTYPE</span></header>
<div class="views"><section class="panel"><div class="panel-head"><h2>Full video + object IDs</h2><span>Native video · sampled tracking</span></div><div class="viewport"><div class="video-box"><video id="video" muted playsinline loop preload="metadata" poster="POSTER_URI" src="VIDEO_URI"></video><canvas id="overlay" width="960" height="540" aria-label="Object IDs and projected 3D geometry"></canvas></div></div><div class="panel-tools"><label><input id="ids-toggle" type="checkbox" checked>Object IDs</label><label><input id="wire-toggle" type="checkbox" checked>3D wire projections</label><small style="color:var(--muted)">Wireframes include hidden geometry; hands are not removed.</small></div></section>
<section class="panel"><div class="panel-head"><h2>Camera + battery scene</h2><span id="mode-label">Orbit · shared camera coordinates</span></div><div class="viewport"><canvas id="scene" aria-label="Orbit the camera and battery 3D scene"></canvas><canvas id="scene-labels"></canvas><div class="scene-hint">Drag to orbit · scroll to zoom · Shift + drag to pan</div><div id="error" role="alert"></div></div><div class="panel-tools"><button id="orbit-view" class="active">Orbit</button><button id="camera-view">Match video</button><button id="reset-view">Reset</button><label><input id="xray-toggle" type="checkbox" checked>Transparent camera</label><label><input id="plane-toggle" type="checkbox">Video plane</label><label><input id="path-toggle" type="checkbox" checked>Battery paths</label><label><input id="rejected-toggle" type="checkbox">Rejected poses</label></div></section></div>
<div class="chapters"><button data-time="0">Start</button><button data-time="34">Open compartment</button><button data-time="39.006">Two batteries / withdrawal</button><button data-time="47.81">First insertion</button><button data-time="51.8">Occluded approach</button><button data-time="54.05">Close / inspect camera</button><button data-time="69.07">Camera back</button></div>
<section class="timeline"><div class="row"><button id="play">▶ Play</button><label class="speed">Speed <select id="speed" aria-label="Playback speed"><option value="0.25">0.25×</option><option value="0.5" selected>0.5×</option><option value="1">1×</option><option value="2">2×</option></select></label><span id="frame-count"></span><span class="spacer"></span><span id="time"></span></div><input id="scrubber" type="range" min="0" max="74241" step="1" value="0" aria-label="Full video timeline"><div class="ticks"><span>0 s</span><span>Video and 3D scene share this timeline</span><span>74.24 s</span></div><div class="status-row"><span id="status" class="badge"></span><span id="note"></span></div><div id="object-list"></div></section>
<div class="legend"><span style="color:#79aaff"><b>camera_00001</b> · fitted shell</span><span style="color:#c5a3ff"><b>cover_00001</b> · 2D candidate only</span><span style="color:#42e0c2"><b>battery_00001</b> · first distinct battery</span><span style="color:#ffa94f"><b>battery_00002</b> · second distinct battery</span><span style="color:#efcf75"><b>battery_track_… ?</b> · later identity unresolved</span></div>
<div class="footer"><p><strong>The camera is a video-proportioned approximation.</strong> Its shell, lens and open battery wells use explicit shape priors. Camera poses are fitted from assistant-marked landmarks; poses between marked frames are approximate. The optional plane is a flat video reference.</p><p><strong>We show observed motion where it is supported.</strong> Both outward batteries have IDs. Later observations keep a “?” because hand occlusion obscures their identity. Hidden seating and the mostly occluded second approach are left unknown. A full-duration video does not mean full-duration battery pose coverage.</p></div><p class="coverage">COVERAGE_TEXT</p><p class="footnote">Camera–battery registration is a combined monocular candidate. Shared nominal units and image alignment do not verify contact, physical depth, exact fit or successful insertion.</p>
</main><script id="observer-data" type="application/json">DATA_JSON</script><script>RUNTIME_JS</script></body></html>'''


def main():
    registry = read(CASE / 'registry.json')
    observations = read(CASE / 'observations.json')
    manifest = read(CASE / 'manifest.json')
    pilot = read(ROOT / 'data/camera-battery-withdrawal-15fps/alignment/battery-pilot-v2/alignment.json')
    camera = read(CASE / 'camera-poses-v1.json')
    batteries = read(CASE / 'battery-poses-v1.json')
    assets = {'camera': mesh_data(CASE / 'geometry/camera-v1/camera.ply'),
              'battery': mesh_data(ROOT / batteries['asset_path'])}
    tracks = [{'id': 'camera_00001', 'object_id': 'camera_00001', 'label': 'camera_00001',
               'asset': 'camera', 'color': '#79aaff', 'scale': camera.get('asset_scale', pilot['asset_scale']),
               'source_track_id': 'camera', 'max_age_s': .56,
               'poses': sorted(camera['frames'], key=lambda row: row['timestamp_s'])}]
    colors = {'battery_00001': '#42e0c2', 'battery_00002': '#ffa94f', 'battery_track_00003': '#dc927b',
              'battery_track_00004': '#73bdd7', 'battery_track_00007': '#f2cd62', 'battery_track_00008': '#ef9cbf'}
    for track in batteries['tracklets']:
        identity = track.get('object_id') or track['observation_id']
        source_track = track.get('source_track_id', track.get('source_track', track.get('track')))
        if track['observation_id'] == 'battery_track_00007':
            source_track = 'in_first_dense'
        elif track['observation_id'] == 'battery_track_00008':
            source_track = 'in_second_dense'
        if source_track == 'visible_end':
            continue
        age = .045 if identity == 'battery_00001' else (.065 if source_track in ['in_first_dense', 'in_second_dense'] else .13)
        tracks.append({'id': identity, 'object_id': track.get('object_id'), 'candidate_object_ids': track.get('candidate_object_ids', []),
                       'label': identity + (' ?' if track.get('object_id') is None else ''),
                       'asset': 'battery', 'color': colors.get(identity, '#f2cd62'), 'scale': batteries['asset_scale'],
                       'source_track_id': source_track, 'max_age_s': age,
                       'poses': sorted(track['frames'], key=lambda row: row['timestamp_s'])})
    data = {'assets': assets, 'tracks': tracks, 'registry': registry, 'observations': observations['frames'],
            'K': pilot['camera']['intrinsics'], 'duration': manifest['source']['video']['duration_s']}
    for track in tracks:
        assert track['scale'] == pilot['asset_scale'], 'All components must use one explicit common nominal gauge'
        for row in track['poses']:
            if row.get('rotation') is not None:
                rotation = np.array(row['rotation'])
                assert np.isfinite(rotation).all() and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
    video = CASE / 'full-source.mp4'
    poster = ROOT / 'data/camera-battery/frames/000001.jpg'
    coverage = {'timeline_samples': len(observations['frames']), 'native_2d_observations': sum(len(row['objects']) for row in observations['frames']),
                'camera_pose_samples': len(camera['frames']),
                'battery_pose_candidates': {t['id']: sum(row.get('rotation') is not None and not row.get('status', '').startswith('unsupported') for row in t['poses']) for t in tracks if t['asset'] == 'battery'}}
    text = f"{coverage['timeline_samples']} evidence timestamps · {coverage['native_2d_observations']} native 2D observations · {coverage['camera_pose_samples']} camera pose samples. Full video plays at its native cadence; pose coverage is sampled and includes explicit approximations."
    style = re.search(r'<style>(.*?)</style>', SIMPLE_DOCUMENT, re.S).group(1)
    document = DOCUMENT.replace('BASE_STYLE', style).replace('COVERAGE_TEXT', text)
    document = document.replace('POSTER_URI', uri(poster, 'image/jpeg')).replace('VIDEO_URI', uri(video, 'video/mp4'))
    document = document.replace('DATA_JSON', json.dumps(data, separators=(',', ':'), allow_nan=False).replace('</', '<\\/'))
    document = document.replace('RUNTIME_JS', (ROOT / 'full_scene_observer.js').read_text())
    path = CASE / 'observer.html'
    path.write_text(document)
    result = {'status': 'generated_pending_review', 'html': str(path), 'html_bytes': path.stat().st_size,
              'full_video_duration_s': data['duration'], 'coverage': coverage,
              'source_sha256': manifest['source']['sha256'],
              'inputs_sha256': {name: hashlib.sha256((CASE / name).read_bytes()).hexdigest() for name in ['registry.json', 'observations.json', 'camera-poses-v1.json', 'battery-poses-v1.json', 'full-source.mp4']},
              'scale': pilot['asset_scale'], 'external_network_dependencies': False,
              'physical_registration_verified': False, 'second_insertion_recovered': False}
    (CASE / 'observer-metadata.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
