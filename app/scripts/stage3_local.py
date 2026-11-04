#!/usr/bin/env python3
"""Prepare a small evidence package and run an explicit local depth backend."""
import argparse, hashlib, json, os, re, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry import intrinsics_from_hfov, lift_pixels, write_cloud

MODEL = 'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf'

def read(p): return json.loads(p.read_text())
def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def get_paths(args):
    for slug in [args.case, args.run]:
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', slug): raise ValueError('Invalid case/run slug')
    case = ROOT/'data'/args.case
    return case, case/'geometry'/args.run

def proxy(path):
    # Explicit arbitrary fixture: not reconstructed or fitted to this video.
    n = 32
    with path.open('w') as f:
        f.write('# UNFITTED PROCEDURAL TEST PROXY. Arbitrary units; not SAM 3D output.\n')
        for z in [-2, 2]:
            for i in range(n):
                t = i*2*np.pi/n; f.write(f'v {0.5*np.cos(t):.7g} {0.5*np.sin(t):.7g} {z}\n')
        for i in range(n):
            j = (i+1)%n; f.write(f'f {i+1} {j+1} {j+1+n} {i+1+n}\n')
        f.write('f '+' '.join(str(i) for i in range(n, 0, -1))+'\n')
        f.write('f '+' '.join(str(i) for i in range(n+1, 2*n+1))+'\n')

def prepare(args):
    case, out = get_paths(args)
    if out.exists(): raise ValueError('Run exists; preserve it and use a fresh run name')
    parent = case/'segmentation'/args.mask_run
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', args.mask_run): raise ValueError('Invalid mask run')
    rows = read(case/'frames/index.json')['frames']; composed = read(parent/'frames.json')['frames']
    selected = []
    for row, record in zip(rows, composed):
        if not args.start <= row['timestamp_s'] <= args.end: continue
        comp = next((c for c in record['components'] if c['track_id'] == args.track and c['area_pixels'] > 0), None)
        if comp and args.track not in record['withheld']: selected.append((row, comp))
    if len(selected) < 3: raise ValueError('Need at least three reviewed mask observations')
    (out/'inputs').mkdir(parents=True)
    samples = []
    for row, comp in selected:
        name = f"{row['index']:06d}"
        image = out/'inputs'/f'{name}.jpg'; mask = out/'inputs'/f'{name}-mask.png'
        # Copy so the experiment is immutable if upstream files are later corrected.
        image.write_bytes((case/row['file']).read_bytes()); mask.write_bytes((parent/comp['visible_mask_file']).read_bytes())
        samples.append({'sample_index':row['index'], 'timestamp_s':row['timestamp_s'], 'width':row['width'], 'height':row['height'], 'image':str(image.relative_to(out)), 'mask':str(mask.relative_to(out)), 'image_sha256':digest(image), 'mask_sha256':digest(mask), 'mask_area_pixels':comp['area_pixels']})
    k = intrinsics_from_hfov(samples[0]['width'], samples[0]['height'], args.hfov)
    proxy(out/'unit-proxy-battery.obj')
    spec = next(t for t in read(parent/'annotations.json')['tracks'] if t['id']==args.track)
    pack = {'schema_version':'0.3', 'stage':3, 'status':'prepared', 'source_case':args.case, 'source_sha256':read(case/'manifest.json')['source']['sha256'], 'parent_mask_run':args.mask_run, 'track_id':args.track, 'candidate_entity_ids':spec['candidate_entity_ids'], 'samples':samples,
      'camera':{'intrinsics':k.tolist(), 'origin':'assumed_horizontal_fov', 'horizontal_fov_degrees':args.hfov, 'calibrated':False, 'extrinsics':'not_estimated', 'coordinates':'+X right, +Y down, +Z forward; frame-local recording-camera coordinates'},
      'depth_backend':{'model':MODEL, 'role':'temporary_non_Meta_local_development_backend', 'output_semantics':'predicted Z-depth in model meter units; physical scale unverified'},
      'geometry_asset':{'file':'unit-proxy-battery.obj', 'kind':'unfitted_procedural_test_proxy', 'units':'arbitrary', 'reconstructed_from_video':False, 'sam3d_output':False, 'purpose':'asset-loader/export fixture only; excluded from depth and tracking evaluation'},
      'planned_gpu_backends':{'geometry':'Meta SAM 3D Objects', 'depth_and_camera':'MegaSaM/MoGe', 'point_tracking':'TAPIP3D'},
      'limitations':['Intrinsics are assumed, not recovered from the recording.', 'Independent monocular depth predictions are not temporally calibrated or measured depth.', 'No world-space trajectory, fitted component pose or inferred attachment is established.', 'Only one outward tracklet is used; physical identity across occlusion is not resolved.']}
    write(out/'experiment.json', pack); print(json.dumps({'experiment':str(out), 'samples':len(samples)}))

def depth(args):
    case, out = get_paths(args); pack = read(out/'experiment.json')
    import torch
    if args.device == 'mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable; no silent CPU fallback')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1': raise RuntimeError('Disable CPU operator fallback for this measured test')
    os.environ['HF_HOME'] = str(ROOT/'models/hf-cache')
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    model_dir = ROOT/'models/depth-anything-v2-metric-indoor-small'
    provenance_file = model_dir/'provenance.json'
    if not provenance_file.exists():
        revision = HfApi().model_info(MODEL, token=False).sha
        snapshot_download(MODEL, revision=revision, local_dir=model_dir, allow_patterns=['config.json','preprocessor_config.json','model.safetensors'], token=False)
        write(provenance_file, {'repo':MODEL, 'revision':revision, 'checkpoint_sha256':digest(model_dir/'model.safetensors')})
    provenance = read(provenance_file)
    if digest(model_dir/'model.safetensors') != provenance['checkpoint_sha256']: raise ValueError('Depth checkpoint changed')
    start = time.monotonic()
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    model = AutoModelForDepthEstimation.from_pretrained(model_dir, local_files_only=True).to(args.device).eval()
    directory = out/'depth'; directory.mkdir(exist_ok=True)
    prior = read(directory/'metadata.json') if (directory/'metadata.json').exists() else None
    if prior and (prior['model_provenance'] != provenance or prior['input_size'] != args.input_size or prior['device'] != args.device or prior['camera'] != pack['camera']): raise ValueError('Depth settings or camera changed; use a new experiment')
    outputs = {r['sample_index']:r for r in prior['frames']} if prior else {}
    selected = pack['samples'][:args.limit] if args.limit else pack['samples']
    for row in selected:
        if digest(out/row['image']) != row['image_sha256'] or digest(out/row['mask']) != row['mask_sha256']: raise ValueError('Experiment inputs changed')
        i = row['sample_index']; dest = directory/f'{i:06d}.npy'
        if i in outputs:
            if not dest.exists() or digest(dest) != outputs[i]['depth_sha256']: raise ValueError('Existing depth missing or changed')
            continue
        image = Image.open(out/row['image']).convert('RGB'); t = time.monotonic()
        inputs = processor(images=image, return_tensors='pt', size={'height':args.input_size,'width':args.input_size})
        inputs = {k:v.to(args.device) for k,v in inputs.items()}
        with torch.inference_mode():
            predicted = model(**inputs).predicted_depth
            predicted = torch.nn.functional.interpolate(predicted[:,None], size=(row['height'],row['width']), mode='bicubic', align_corners=False)[0,0]
            if args.device == 'mps': torch.mps.synchronize()
            array = predicted.cpu().numpy().astype(np.float32)
        if not np.isfinite(array).all() or np.any(array <= 0): raise ValueError('Invalid model depth')
        np.save(dest, array); mask = np.asarray(Image.open(out/row['mask'])) > 0
        ys,xs = np.where(mask); step = max(1, len(xs)//5000); uv = np.c_[xs[::step],ys[::step]]
        xyz = lift_pixels(uv, array, pack['camera']['intrinsics']); colors = np.asarray(image)[ys[::step],xs[::step]]
        cloud = directory/f'{i:06d}-visible-surface.ply'; write_cloud(cloud, xyz, colors)
        rec = {'sample_index':i,'timestamp_s':row['timestamp_s'],'file':str(dest.relative_to(out)), 'depth_sha256':digest(dest),'point_cloud':str(cloud.relative_to(out)), 'inference_and_export_s':round(time.monotonic()-t,3),'object_depth_p10_p50_p90':np.percentile(array[mask],[10,50,90]).tolist()}
        outputs[i] = rec
        write(directory/'metadata.json', {'status':'complete' if len(outputs)==len(pack['samples']) else 'partial', 'model_provenance':provenance, 'input_size':args.input_size,'device':args.device,'torch':torch.__version__, 'cpu_operator_fallback_enabled':False, 'camera':pack['camera'], 'units':'model_predicted_meters_UNCALIBRATED', 'frames':sorted(outputs.values(),key=lambda r:r['sample_index']), 'last_invocation_wall_time_s':round(time.monotonic()-start,3), 'accuracy_measured':False})
        print(f'Depth sample {i}: {rec["inference_and_export_s"]:.3f}s', flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=['prepare','depth']); p.add_argument('--case',default='camera-battery-action-5fps');p.add_argument('--run',default='local-battery-v1');p.add_argument('--track',default='out_left');p.add_argument('--mask-run',default='components-v2');p.add_argument('--start',type=float,default=39.0);p.add_argument('--end',type=float,default=42.45);p.add_argument('--hfov',type=float,default=60);p.add_argument('--device',choices=['mps','cpu'],default='mps');p.add_argument('--input-size',type=int,default=392);p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.input_size < 112 or a.input_size % 14: p.error('Input size must be a multiple of 14, at least 112')
    if a.limit is not None and a.limit <= 0: p.error('Limit must be positive')
    {'prepare':prepare,'depth':depth}[a.command](a)

if __name__=='__main__': main()
