#!/usr/bin/env python3
"""Actual local image-to-3D attempt; hidden geometry is inferred, never measured."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['mps','cpu'],default='mps');p.add_argument('--resolution',type=int,default=96);a=p.parse_args()
    import torch,trimesh
    from skimage.measure import marching_cubes
    if a.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no silent fallback')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise RuntimeError('CPU operator fallback must be disabled')
    os.environ['HF_HOME']=str(ROOT/'models/hf-cache');os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
    sys.path.insert(0,str(ROOT/'models/triposr-code'))
    from tsr.system import TSR
    base=ROOT/'data/camera-battery-withdrawal-15fps/geometry';parent=base/'local-battery-v3';out=base/'battery-reconstruction-v1'
    out.mkdir(exist_ok=False);start=time.monotonic()
    metadata={'status':'running','backend':'TripoSR','device':a.device,'mesh_extraction':'explicit CPU scikit-image marching cubes on actual learned density','cpu_operator_fallback_enabled':False,'source_run':'local-battery-v3','source_sample':16,'model_provenance':read(ROOT/'models/triposr-checkpoint/provenance.json'),'local_code_adaptation':'Optional torchmcubes import; native extractor is not used. No learned modules or weights altered.','metric_scale_verified':False,'hidden_surfaces':'generative inference, not observed','resolution':a.resolution}
    write(out/'metadata.json',metadata)
    try:
        pack=read(parent/'experiment.json');row=next(r for r in pack['samples'] if r['sample_index']==16)
        image=np.array(Image.open(parent/row['image']).convert('RGB'));mask=np.array(Image.open(parent/row['mask']))>0;ys,xs=np.where(mask);x0,x1=xs.min(),xs.max()+1;y0,y1=ys.min(),ys.max()+1
        rgba=np.dstack([image,mask.astype(np.uint8)*255]);Image.fromarray(rgba).save(out/'source-masked.png')
        crop=rgba[y0:y1,x0:x1];side=int(np.ceil(max(crop.shape[:2])/.85));square=np.full((side,side,3),128,np.uint8);cy=(side-crop.shape[0])//2;cx=(side-crop.shape[1])//2;alpha=crop[:,:,3:]/255;square[cy:cy+crop.shape[0],cx:cx+crop.shape[1]]=(crop[:,:,:3]*alpha+128*(1-alpha)).astype(np.uint8);input_image=Image.fromarray(square);input_image.save(out/'model-input.png')
        metadata['input_sha256']=hashlib.sha256((out/'model-input.png').read_bytes()).hexdigest();metadata['input_policy']='SAM visible pixels only, gray background; hand occlusions remain missing, not inpainted'
        model=TSR.from_pretrained(str(ROOT/'models/triposr-checkpoint'),'config.yaml','model.ckpt').to(a.device).eval();model.renderer.set_chunk_size(8192)
        if a.device=='mps':metadata['recommended_mps_memory_gib']=torch.mps.recommended_max_memory()/2**30
        inference_start=time.monotonic()
        with torch.inference_mode():
            scene=model([input_image],device=a.device)
            if a.device=='mps':torch.mps.synchronize()
        metadata['image_to_triplane_s']=round(time.monotonic()-inference_start,3);print('TripoSR learned triplane generated',flush=True)
        radius=float(model.renderer.cfg.radius);axis=torch.linspace(-radius,radius,a.resolution);grid=torch.stack(torch.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
        density=[]
        with torch.inference_mode():
            for block in grid.split(8192):
                prediction=model.renderer.query_triplane(model.decoder,block.to(a.device),scene[0]);density.append(prediction['density_act'][:,0].cpu())
        volume=torch.cat(density).reshape(a.resolution,a.resolution,a.resolution).numpy();np.save(out/'learned-density.npy',volume)
        vertices,faces,_,_=marching_cubes(volume,level=25.0,spacing=(2*radius/(a.resolution-1),)*3);vertices-=radius
        color_parts=[]
        with torch.inference_mode():
            for block in torch.from_numpy(vertices.copy()).float().split(8192):color_parts.append(model.renderer.query_triplane(model.decoder,block.to(a.device),scene[0])['color'].cpu().numpy())
        colors=(np.clip(np.concatenate(color_parts),0,1)*255).astype(np.uint8);mesh=trimesh.Trimesh(vertices=vertices,faces=faces,vertex_colors=colors,process=True)
        mesh.export(out/'battery-raw.glb');mesh.export(out/'battery-raw.obj')
        parts=mesh.split(only_watertight=False);main_mesh=max(parts,key=lambda part:part.area);main_mesh.export(out/'battery-main.glb');main_mesh.export(out/'battery-main.obj')
        metadata.update(status='candidate_generated_pending_visual_review',image_to_triplane_s=metadata['image_to_triplane_s'],wall_time_s=round(time.monotonic()-start,3),raw_vertices=len(mesh.vertices),raw_faces=len(mesh.faces),connected_components=len(parts),main_vertices=len(main_mesh.vertices),main_faces=len(main_mesh.faces),main_watertight=bool(main_mesh.is_watertight),main_bounds=main_mesh.bounds.tolist(),units='TripoSR normalized asset units; uncalibrated',main_component_policy='Largest connected surface; raw mesh preserved',geometry_reconstructed_by_model=True,sam3d_output=False)
        write(out/'metadata.json',metadata);print(json.dumps(metadata,indent=2))
    except Exception as exc:
        metadata.update(status='failed',error=str(exc),wall_time_s=round(time.monotonic()-start,3));write(out/'metadata.json',metadata);raise
if __name__=='__main__':main()
