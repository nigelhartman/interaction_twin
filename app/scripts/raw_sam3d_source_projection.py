#!/usr/bin/env python3
"""One bounded CPU render of a raw SAM3D layout with estimated pointmap K.

Diagnostic only: no optimization, model inference, tracking or pose acceptance.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
import sam3d_milestone as M
import upgrade_bundles as U
from native_appearance import load_native_glb, render_native_mesh
from sam3d_source_projection import recover_estimated_camera, raw_layout_projection


def worker(plan_path):
    path=U._local(plan_path,ROOT);plan=U.read(path)
    if plan.get('format')!='sam3d-fixed-raw-source-projection-plan' or plan.get('optimized') is not False:
        raise ValueError('Unsupported fixed projection plan')
    glb=U._local(plan['native_glb']['path'],ROOT)
    if U.sha(glb)!=plan['native_glb']['sha256']:raise ValueError('Native GLB changed')
    mesh=load_native_glb(glb)
    pose=plan['raw_projection'];w,h=plan['image_size']
    rendered=render_native_mesh(mesh,plan['camera']['K_estimated'],pose['R'],pose['t'],w,h,scale=1.)
    Image.fromarray(rendered['rgb']).save(path.parent/'render-rgb.png')
    Image.fromarray(rendered['mask'].astype(np.uint8)*255).save(path.parent/'render-mask.png')
    np.savez_compressed(path.parent/'render-depth-and-faces.npz',depth=rendered['depth'],face_index=rendered['face_index'])
    points=mesh.vertices@np.asarray(pose['R']).T+pose['t'];valid=np.isfinite(points).all(1)&(points[:,2]>1e-8)
    uv=points[valid]@np.asarray(plan['camera']['K_estimated']).T;uv=uv[:,:2]/uv[:,2:]
    report={'native_appearance':mesh.receipt,'positive_depth_native_vertices':int(valid.sum()),
            'projected_vertex_bbox_xyxy':np.r_[uv.min(0),uv.max(0)].tolist() if len(uv) else None,
            'complete_geometry_rendered':True,'SAM_mask_used_to_clip_render':False,
            'frame_bounds_clip_only':True,'optimized':False,'physical_accuracy_verified':False}
    U.write(path.parent/'render-receipt.json',report)


def diagnostic_records(directory):
    """Diagnostic console logs may be empty; native-result inventories may not."""
    rows=[];total=0
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file() or path.name.startswith('.'):
            raise ValueError('Unexpected diagnostic artifact')
        size=path.stat().st_size;total+=size
        if size>U.MAX_FILE or (size==0 and path.name!='worker.log') or total>U.MAX_TOTAL or len(rows)>=U.MAX_FILES:
            raise ValueError('Diagnostic artifact budget exceeded')
        rows.append({'path':path.name,'sha256':U.sha(path),'bytes':size})
    return rows


def build(input_bundle,adapter_output,output,*,finalize_existing=False):
    output=M._directory(output,ROOT,exists=False)
    if output.exists()!=finalize_existing:raise ValueError('Choose a fresh directory, or explicitly finalize an existing completed render')
    checked,source,files,metadata=M.inspect_output(input_bundle,adapter_output)
    rgb,mask=source['rgb'],source['mask'];h,w=rgb.shape[:2]
    with U._safe_layout(files['layout-and-pointmap.npz']) as layout:
        if 'pointmap' not in layout.files or layout['pointmap'].shape!=(h,w,3):
            raise ValueError('Exact native pointmap/RGB mapping required; no guessed resizing')
        camera=recover_estimated_camera(layout['pointmap'])
        pose=raw_layout_projection(layout['rotation'],layout['translation'],layout['scale'])
    glb=M._directory(adapter_output,ROOT)/'object.glb'
    if any(output.is_relative_to(p) for p in (source['directory'],glb.parent)):
        raise ValueError('Diagnostic must be separate from frozen source/output')
    plan={'format':'sam3d-fixed-raw-source-projection-plan','version':1,'image_size':[w,h],
          'source_binding':checked['source_binding'],
          'native_glb':{'path':glb.relative_to(ROOT).as_posix(),'sha256':U.digest(files['object.glb'])},
          'layout_sha256':U.digest(files['layout-and-pointmap.npz']),
          'source_rgb_sha256':U.digest(source['files']['image.png']),'source_mask_sha256':U.digest(source['files']['mask.png']),
          'camera':camera,'raw_projection':pose,'render_timeout_seconds':120,'maximum_renders':1,
          'optimized':False,'old_pose_reused':False,'new_models_or_api_calls':0,
          'physical_accuracy_verified':False,'source_video_pose_accepted':False,'simulation_ready':False}
    started=time.monotonic()
    if finalize_existing:
        if U.read(output/'plan.json')!=plan or (output/'diagnostic.json').exists():raise ValueError('Existing render plan differs or is already finalized')
        rendered_receipt=U.read(output/'render-receipt.json')
        if rendered_receipt.get('native_appearance',{}).get('source_glb_sha256')!=plan['native_glb']['sha256']:
            raise ValueError('Existing render belongs to a different native asset')
        for name in ['render-rgb.png','render-mask.png','render-depth-and-faces.npz']:
            if not (output/name).is_file() or not (output/name).stat().st_size:raise ValueError('Existing render is incomplete; no retry')
    else:
        output.mkdir(parents=True,exist_ok=False);U.write(output/'plan.json',plan)
        try:
            result=subprocess.run([sys.executable,'-I',str(Path(__file__).resolve()),'--worker-plan',str(output/'plan.json')],
                                  capture_output=True,text=True,timeout=120)
        except subprocess.TimeoutExpired as error:
            U.write(output/'timeout.json',{'status':'render_timeout_no_retry','seconds':120,'optimized':False})
            raise ValueError('Raw render exceeded120seconds; evidence retained, no retry') from error
        (output/'worker.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise ValueError('Raw render failed; worker log retained')
    render=np.asarray(Image.open(output/'render-rgb.png').convert('RGB'))
    projected=np.asarray(Image.open(output/'render-mask.png'))>0
    overlay=rgb.copy();overlay[projected]=np.rint(rgb[projected]*.35+render[projected]*.65).astype(np.uint8)
    comparison=np.zeros_like(rgb);comparison[mask]=[235,167,54];comparison[projected]=[66,194,184];comparison[mask&projected]=[232,237,220]
    Image.fromarray(rgb).save(output/'source-rgb.png');Image.fromarray(mask.astype(np.uint8)*255).save(output/'source-mask.png')
    Image.fromarray(overlay).save(output/'source-overlay.png');Image.fromarray(comparison).save(output/'silhouette-comparison.png')
    yy,xx=np.where(mask|projected);crop=None
    if len(xx):
        padding=20;crop=[max(0,int(xx.min())-padding),max(0,int(yy.min())-padding),min(w,int(xx.max())+padding+1),min(h,int(yy.max())+padding+1)]
        panel=Image.new('RGB',(1200,510),'#f4f5ef');draw=ImageDraw.Draw(panel)
        for i,(array,label) in enumerate([(rgb,'Original source'),(overlay,'Raw layout + estimated K'),(comparison,'SAM amber / model teal / overlap white')]):
            image=Image.fromarray(array).crop(crop);image.thumbnail((380,410),Image.Resampling.NEAREST if i==2 else Image.Resampling.BICUBIC)
            panel.paste(image,(i*400+(400-image.width)//2,55+(410-image.height)//2));draw.text((i*400+16,20),label,fill='#153d32')
        draw.text((16,485),'Display crop only. No silhouette fitting, tracking, camera calibration or physical pose validation.',fill='#153d32')
        panel.save(output/'detail-crop.png')
    intersection=int((mask&projected).sum());union=int((mask|projected).sum())
    receipt={'format':'sam3d-fixed-raw-source-projection-diagnostic','version':1,'status':'one_raw_projection_rendered',
          'plan_sha256':U.sha(output/'plan.json'),'camera':camera,'raw_projection':pose,
          'render':U.read(output/'render-receipt.json'),
          'render_elapsed_seconds':None if finalize_existing else time.monotonic()-started,
          'existing_render_finalized_without_retry':finalize_existing,
          'same_frame_silhouette_diagnostic':{'SAM_visible_pixels':int(mask.sum()),'raw_projected_pixels':int(projected.sum()),
              'intersection_pixels':intersection,'union_pixels':union,'IoU':intersection/union if union else None,
              'used_for_camera_or_layout_selection':False,'physical_accuracy_metric':False},
          'display_crop_xyxy':crop,'crop_affects_render_or_measurements':False,
          'native_glb_changed':U.sha(glb)!=plan['native_glb']['sha256'],'optimized':False,
          'new_models_or_api_calls':0,'source_video_pose_accepted':False,'physical_accuracy_verified':False,'simulation_ready':False,
          'implementation':{'script_sha256':U.sha(Path(__file__)), 'math_sha256':U.sha(ROOT/'sam3d_source_projection.py'),
                            'native_renderer_sha256':U.sha(ROOT/'native_appearance.py')},
          'limitations':['Camera is an estimated prior fit to the SAME model-inferred pointmap; held-out grid is not independent physical calibration.',
                         'MoGe force_projection=False permits ray inconsistency; recovered K is not the exact omitted pipeline camera.',
                         'Raw model layout is not refined or accepted as a source-video pose. Same-frame mask overlap is descriptive only.',
                         'No geometry, program, catalog, selected model, assembly, contact or simulation state is changed.'],
          'files':diagnostic_records(output)}
    if receipt['native_glb_changed']:raise ValueError('Native GLB changed during diagnostic')
    U.write(output/'diagnostic.json',receipt)
    return {'status':receipt['status'],'output':output.relative_to(ROOT).as_posix(),
            'camera_heldout':camera['heldout_grid'],'silhouette_diagnostic':receipt['same_frame_silhouette_diagnostic']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker-plan',type=Path)
    parser.add_argument('--finalize-existing',action='store_true')
    parser.add_argument('--input-bundle',type=Path);parser.add_argument('--adapter-output',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.worker_plan:worker(args.worker_plan);return 0
    if not all([args.input_bundle,args.adapter_output,args.output]):parser.error('Input bundle, adapter output and fresh output required')
    try:report=build(args.input_bundle,args.adapter_output,args.output,finalize_existing=args.finalize_existing)
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(json.dumps({'status':'raw_projection_unavailable','error':str(error)}));return 2
    print(json.dumps(report,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
