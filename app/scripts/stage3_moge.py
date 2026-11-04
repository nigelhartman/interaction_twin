#!/usr/bin/env python3
"""Versioned local MoGe-2 geometry estimates; never overwrite the DA baseline."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
MODEL='Ruicheng/moge-2-vits-normal'
def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);p.add_argument('--device',choices=['mps','cpu'],default='mps');a=p.parse_args()
    import torch
    if a.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no silent fallback')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise RuntimeError('CPU operator fallback must be disabled')
    os.environ['HF_HOME']=str(ROOT/'models/hf-cache');os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
    from huggingface_hub import HfApi,hf_hub_download
    from moge.model.v2 import MoGeModel
    from importlib.metadata import distribution
    parent=ROOT/'data/camera-battery-withdrawal-15fps/geometry/local-battery-v2';pack=read(parent/'experiment.json');out=parent/'moge-v1';out.mkdir(exist_ok=True)
    model_dir=ROOT/'models/moge-2-vits-normal';model_dir.mkdir(exist_ok=True);prov_file=model_dir/'provenance.json'
    if not prov_file.exists():
        revision=HfApi().model_info(MODEL,token=False).sha
        ckpt=Path(hf_hub_download(MODEL,'model.pt',revision=revision,local_dir=model_dir,token=False))
        package=json.loads(distribution('moge').read_text('direct_url.json'))
        write(prov_file,{'repo':MODEL,'revision':revision,'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'code_commit':package['vcs_info']['commit_id'],'implementation':'moge.model.v2 only; CUDA-only v3 is not imported'})
    provenance=read(prov_file);checkpoint=model_dir/'model.pt'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=provenance['checkpoint_sha256']:raise ValueError('MoGe checkpoint changed')
    model=MoGeModel.from_pretrained(str(checkpoint)).to(a.device).eval();start=time.monotonic()
    prior=read(out/'metadata.json') if (out/'metadata.json').exists() else None
    if prior and (prior['model_provenance']!=provenance or prior['device']!=a.device):raise ValueError('Model/device changed; preserve this run')
    records={r['sample_index']:r for r in prior['frames']} if prior else {}
    for row in pack['samples'][:a.limit] if a.limit else pack['samples']:
        if hashlib.sha256((parent/row['image']).read_bytes()).hexdigest()!=row['image_sha256']:raise ValueError('Source image changed')
        i=row['sample_index'];file=out/f'{i:06d}.npz'
        if i in records:
            if hashlib.sha256(file.read_bytes()).hexdigest()!=records[i]['sha256']:raise ValueError('Existing MoGe result changed')
            continue
        original=Image.open(parent/row['image']).convert('RGB');image=original.resize((640,360),Image.Resampling.LANCZOS)
        tensor=torch.from_numpy(np.array(image).copy()).permute(2,0,1).float().to(a.device)/255
        t=time.monotonic()
        with torch.inference_mode():
            output=model.infer(tensor,num_tokens=1200,use_fp16=False,apply_mask=True)
            if a.device=='mps':torch.mps.synchronize()
            depth=output['depth'].cpu().numpy().astype(np.float32);points=output['points'].cpu().numpy().astype(np.float32);intrinsics=output['intrinsics'].cpu().numpy().astype(np.float32);valid=output['mask'].cpu().numpy().astype(bool)
        np.savez_compressed(file,depth=depth,points=points,intrinsics_normalized=intrinsics,valid=valid)
        fov=float(np.rad2deg(2*np.arctan(.5/intrinsics[0,0])))
        records[i]={'sample_index':i,'timestamp_s':row['timestamp_s'],'file':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'inference_and_export_s':round(time.monotonic()-t,3),'estimated_hfov_degrees':fov,'intrinsics_normalized':intrinsics.tolist(),'valid_fraction':float(valid.mean())}
        write(out/'metadata.json',{'status':'complete' if len(records)==len(pack['samples']) else 'partial','model_provenance':provenance,'device':a.device,'cpu_operator_fallback_enabled':False,'input_hw':[360,640],'num_tokens':1200,'inference_precision':'float32','intrinsics':'model estimated, not calibrated','units':'nominal model meters; physical scale unverified','source_sha256':pack['source_sha256'],'frames':sorted(records.values(),key=lambda r:r['sample_index']),'last_invocation_s':round(time.monotonic()-start,3)})
        print(f'MoGe sample {i}: {records[i]["inference_and_export_s"]:.2f}s, estimated HFOV {fov:.1f}',flush=True)
if __name__=='__main__':main()
