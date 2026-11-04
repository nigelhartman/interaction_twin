#!/usr/bin/env python3
"""Full-source video beside an orbitable scene of actual generated-mesh poses.

Only the explicitly supplied new experiment and its hashed GLB/video are read.
The earlier sampled image review remains intact. No pose interpolation is added.
"""
import argparse
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'supported_conditional_motion_candidate', 'reference_candidate_unverified_geometry_and_pose'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_case(pack, metadata):
    """Explicit motion-source case wins over the mesh's reconstruction case.

    A dense motion run may reuse a mesh generated from a different sampling of
    the exact same video. Never silently fall back when an explicit case fails.
    """
    wanted=pack['provenance']['source_video_sha256']
    if pack.get('case_path'):
        case=Path(pack['case_path']).resolve()
        if not (case/'manifest.json').is_file() or read(case/'manifest.json')['source']['sha256']!=wanted:
            raise ValueError('Explicit motion-source case does not match source-video hash')
        return case
    selection=Path(metadata['source']['selection_manifest_path'])
    case=next((p for p in selection.parents if (p/'manifest.json').is_file() and read(p/'manifest.json')['source']['sha256']==wanted),None)
    if case is None:
        raise ValueError('No matching source-video case')
    return case


def source_projection(k, width, height, near, far):
    """OpenGL projection after CV(+Ydown,+Zforward)→GL axis conversion.

    Integer source pixels correspond to centres (u+.5,v+.5) in the viewport.
    """
    k = np.asarray(k, float)
    if k.shape != (3, 3) or min(width, height, near) <= 0 or far <= near:
        raise ValueError('Invalid calibrated-view dimensions or clipping planes')
    return np.array([[2*k[0, 0]/width, 0, 1-2*(k[0, 2]+.5)/width, 0],
                     [0, 2*k[1, 1]/height, 2*(k[1, 2]+.5)/height-1, 0],
                     [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)],
                     [0, 0, -1, 0]], float)


def pose_matrix(record):
    r = np.asarray(record['candidate_rotation'], float)
    t = np.asarray(record['candidate_translation'], float)
    if r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all() or not np.isfinite(t).all() or not np.allclose(r.T@r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1., atol=1e-5):
        raise ValueError('Invalid rigid pose')
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = r, t
    return matrix


def scene_layout(vertices, frames):
    """Bounds and path segments from supported poses; never bridge a gap."""
    vertices = np.asarray(vertices, float)
    canonical_center = (vertices.min(0)+vertices.max(0))/2
    transformed, segments, current = [], [], []
    for record in frames:
        if record['status'] not in ALLOWED:
            if current:
                segments.append(current)
                current = []
            continue
        matrix = pose_matrix(record)
        xyz = vertices@matrix[:3, :3].T+matrix[:3, 3]
        transformed.append(xyz)
        centre = canonical_center@matrix[:3, :3].T+matrix[:3, 3]
        current.append({'sample_index': record['sample_index'], 'timestamp_s': record['timestamp_s'], 'position': centre.tolist()})
    if current:
        segments.append(current)
    if not transformed:
        raise ValueError('No supported/reference poses for a 3D observer')
    all_xyz = np.concatenate(transformed)
    lo, hi = all_xyz.min(0), all_xyz.max(0)
    span = float(np.linalg.norm(hi-lo))
    if not np.isfinite(span) or span <= 0 or lo[2] <= 0:
        raise ValueError('Scene must have finite extent in front of its conditional camera')
    return {'bounds': [lo.tolist(), hi.tolist()], 'target': ((lo+hi)/2).tolist(), 'span': span,
            'initial_distance': span*1.8, 'source_near': max(1e-5, float(lo[2])*.05),
            'source_far': float(hi[2]+span*8), 'path_segments': segments,
            'bounds_policy': 'all actual raw GLB vertices transformed by supported/reference saved poses; no manual scene target'}


DOCUMENT = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Generated earbud · video and interactive 3D</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0e1620;color:#e4edf5;font:15px/1.5 system-ui}main{max-width:1550px;margin:auto;padding:24px}h1{font-size:27px;margin:0}p{color:#aebdcb}header{margin-bottom:20px}.views{display:grid;grid-template-columns:1fr 1fr;gap:18px}.panel{border:1px solid #324353;border-radius:13px;background:#172331;overflow:hidden}h2{font-size:16px;margin:12px 16px}.viewport{height:min(65vh,620px);min-height:340px;background:#08101a;position:relative}video,canvas{display:block;width:100%;height:100%;object-fit:contain}canvas{touch-action:none;cursor:grab}.tools{padding:12px 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}button,select{font:inherit;color:inherit;border:1px solid #476076;background:#22394d;border-radius:6px;padding:7px 10px;cursor:pointer}button.active{border-color:#51bea3;background:#214f45}label{font-size:13px;display:flex;gap:6px}input{accent-color:#65d6bb}.timeline{border:1px solid #324353;border-radius:12px;padding:16px;margin-top:18px;background:#172331}.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}#scrubber{width:100%;margin:14px 0}#status{padding:8px 10px;background:#254939;border-radius:6px}#status.unsupported{background:#5c3037}#status.reference{background:#5c4b28}#source-time{font-variant-numeric:tabular-nums}#note,#age{color:#afc1d0;font-size:13px}#mode-label{font-size:12px;color:#b1c2d0}#error{display:none;position:absolute;inset:20px;background:#622d35;padding:15px;border-radius:9px}small{color:#aebdcb}a{color:#77e1cb}.limits{display:grid;grid-template-columns:1fr 1fr;gap:18px;font-size:13px}#hint{position:absolute;bottom:10px;left:12px;background:#101b29c9;color:#bdd3e5;padding:4px 8px;border-radius:6px;font-size:12px;pointer-events:none}@media(max-width:900px){.views,.limits{grid-template-columns:1fr}main{padding:14px}.viewport{height:60vh}h1{font-size:23px}}
</style><main><header><h1>Full video beside the moving generated earbud</h1><p>Drag the 3D scene to orbit. These are conditional camera-relative pose candidates: generated shape, physical scale and true pose remain unknown.</p></header>
<div class="views"><section class="panel"><h2>Full source-timing video</h2><div class="viewport"><video id="video" controls muted playsinline preload="metadata" src="VIDEO" poster="POSTER"></video></div><div class="tools"><small>Complete recording · native presentation times · no image-sequence playback</small></div></section>
<section class="panel"><h2>Interactive WebGL scene <span id="mode-label"></span></h2><div class="viewport"><canvas id="scene" aria-label="Orbit, pan and zoom the actual generated mesh"></canvas><div id="hint">Drag to orbit · Shift drag to pan · scroll to zoom</div><div id="error" role="alert"></div></div><div class="tools"><button id="orbit" class="active">Orbit</button><button id="match">Match video camera</button><button id="reset">Reset</button><label><input id="path-toggle" type="checkbox" checked>Past motion path</label><label><input id="wire-toggle" type="checkbox">Wireframe</label></div></section></div>
<section class="timeline"><div class="row"><button id="play">Play</button><button id="previous">Previous sample</button><button id="next">Next sample</button><span id="source-time"></span><label>Speed <select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option></select></label></div><input id="scrubber" aria-label="Full source video time" type="range" min="0" step="0.001"><div class="row"><span id="status"></span><span id="age"></span></div><p id="note"></p></section>
<div class="limits"><p><strong>What moves:</strong> the untouched TripoSR GLB and its original model-predicted colours, at fixed native scale1.0. The saved pose maps it into the recording camera coordinates. Orbiting changes your inspection viewpoint; it does not recover new geometry.</p><p><strong>What stays unknown:</strong> metric scale, physical pose accuracy, hidden surfaces, real contact and seating. Saved SAMPLING_FPS fps poses use the most recent preceding sample; no future sample, interpolation or reacquisition fills a gap. The mesh is hidden outside the pilot or at unsupported samples. The path never connects across withheld intervals.</p></div>
<p><a href="observer.html">Sampled source / raw projection diagnostics</a> · <a href="experiment.json">Motion evidence</a> · <a href="observer-3d-metadata.json">Viewer provenance</a></p>
<script id="generated-motion-data" type="application/json">DATA</script><script>JAVASCRIPT</script></main></html>'''


def build(experiment, output):
    experiment, output = Path(experiment).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Output exists; preserve the prior reviewer')
    pack = read(experiment)
    mesh_path = Path(pack['provenance']['mesh_file'])
    if sha(mesh_path) != pack['provenance']['mesh_sha256']:
        raise ValueError('Actual generated mesh hash mismatch')
    metadata_path = Path(pack['provenance']['model_metadata_file'])
    metadata = read(metadata_path)
    case = source_case(pack,metadata)
    manifest = read(case/'manifest.json')
    video = case/manifest['source']['file']
    if sha(video) != pack['provenance']['source_video_sha256']:
        raise ValueError('Full source-timing video hash mismatch')
    if pack.get('fixed_native_mesh_scale') != 1.:
        raise ValueError('This observer preserves the saved native mesh scale exactly')
    helper_spec = importlib.util.spec_from_file_location('generated_mesh_reader', ROOT/'scripts/stage3_reconstruction_review.py')
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    primitives, stats = helper.mesh_primitives(mesh_path)
    vertices = np.concatenate([np.asarray(p['vertices']) for p in primitives])
    scene = scene_layout(vertices, pack['frames'])
    rows = []
    last_time = -np.inf
    frame_rows=read(case/'frames/index.json')['frames']
    source_rows={r['index']:r for r in frame_rows}
    for rec in pack['frames']:
        timestamp = rec['timestamp_s']
        if not np.isfinite(timestamp) or timestamp <= last_time:
            raise ValueError('Saved pose presentation times must strictly increase')
        source_row=source_rows.get(rec['sample_index'])
        if source_row is None or abs(source_row['timestamp_s']-timestamp)>1e-6:
            raise ValueError('Motion sample index/PTS does not match its source case')
        last_time = timestamp
        allowed = rec['status'] in ALLOWED
        rows.append({'sample_index': rec['sample_index'], 'timestamp_s': timestamp, 'status': rec['status'],
                     'visible': allowed, 'reference': rec['status'].startswith('reference_'), 'reasons': rec['reasons'],
                     'camera_from_native_mesh': pose_matrix(rec).tolist() if allowed else None})
    duration = manifest['source']['video']['duration_s']
    if not 0 <= rows[0]['timestamp_s'] < rows[-1]['timestamp_s'] <= duration:
        raise ValueError('Pilot must lie within full source duration')
    data = {'object_id': pack['object_id'], 'source_duration_s': duration, 'image_size': [pack['intrinsics']['width'], pack['intrinsics']['height']],
            'intrinsics': pack['intrinsics']['matrix'], 'sampling_fps': manifest['sampling']['fps'], 'frames': rows,
            'primitives': primitives, 'mesh_stats': stats, 'scene': scene,
            'source_projection_rows': source_projection(pack['intrinsics']['matrix'], pack['intrinsics']['width'], pack['intrinsics']['height'], scene['source_near'], scene['source_far']).tolist(),
            'initial_source_time_s': next(r['timestamp_s'] for r in rows if r['reference']),
            'allowed_statuses': sorted(ALLOWED), 'coordinate_axes': '+Xright,+Ydown,+Zforward camera-relative conditional coordinates'}
    js = (ROOT/'generated_motion_observer.js').read_text()
    poster = case/next(r['file'] for r in frame_rows if r['index'] == pack['reference_sample_index'])
    document = DOCUMENT.replace('VIDEO', html.escape(quote(os.path.relpath(video, output.parent), safe='/'), quote=True)).replace('POSTER', html.escape(quote(os.path.relpath(poster, output.parent), safe='/'), quote=True))
    document = document.replace('SAMPLING_FPS',f"{manifest['sampling']['fps']:g}").replace('DATA', json.dumps(data, separators=(',', ':')).replace('</', '<\\/')).replace('JAVASCRIPT', js.replace('</script', '<\\/script'))
    output.write_text(document)
    record = {'status': 'built_orbitable_webgl_reviewer', 'source_experiment_sha256': sha(experiment), 'actual_glb_sha256': sha(mesh_path),
              'source_video_sha256': sha(video), 'original_source_sha256': manifest['source']['original_sha256'],
              'motion_source_case':str(case),'explicit_motion_source_case_used':bool(pack.get('case_path')),
              'source_video_file': os.path.relpath(video, output.parent), 'source_video_timing': 'complete original-PTS-preserving normalized video, not sampled preview',
              'source_image_size': data['image_size'], 'scene_bounds_and_target': scene,
              'mesh_appearance_preserved': True, 'mesh_scale': 1., 'pose_interpolation': False, 'point_reacquisition': False,
              'sample_selection': 'most recent preceding sourcePTS; hide beforepilot,afterpilot,unsupported',
              'motion_path': 'past supported/reference consecutive samples only; split at every unsupported interval',
              'camera_projection': 'K in native portrait pixels with integer-pixel-centre +0.5 viewport mapping',
              'reference_pose_and_physical_shape_verified': False, 'no_manual_geometry_or_old_pose_files_read': True,
              'external_network_dependencies': False, 'glb_geometry_embedded': True, 'source_video_embedded': False,
              'javascript_sha256': sha(ROOT/'generated_motion_observer.js'), 'html_sha256': sha(output), 'html_bytes': output.stat().st_size}
    (output.parent/'observer-3d-metadata.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'html': str(output), 'source_duration_s': duration, 'triangles': stats['faces'], 'supported_or_reference_poses': sum(r['visible'] for r in rows)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    build(args.experiment, args.out)


if __name__ == '__main__':
    main()
