#!/usr/bin/env python3
"""Pool estimated optics and diagnose/correct relative depth-scale drift using held-out points."""
import json,os,shutil,sys,time,hashlib
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from geometry import lift_pixels,write_cloud,rigid_fit
from consistency import pixel_intrinsics,estimate_depth_scale,heldout_shape_error
def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
    start=time.monotonic();base=ROOT/'data/camera-battery-withdrawal-15fps/geometry';source=base/'local-battery-v2';out=base/'local-battery-v3'
    if out.exists():raise ValueError('Preserve existing run; do not overwrite it')
    pack=read(source/'experiment.json');moge=read(source/'moge-v1/metadata.json')
    if moge['status']!='complete':raise ValueError('MoGe input incomplete')
    rows=pack['samples'];old=np.load(source/'motion-cotracker/tracks.npz');meta=read(source/'motion-cotracker/metadata.json');anchor=next(i for i,r in enumerate(rows) if r['sample_index']==meta['anchor_sample'])
    k_norm=np.median([r['intrinsics_normalized'] for r in moge['frames']],axis=0);k=pixel_intrinsics(k_norm,rows[0]['width'],rows[0]['height']);fov=float(np.rad2deg(2*np.arctan(.5/k_norm[0,0])))
    raw_depth=[];raw_points=[]
    for j,row in enumerate(rows):
        prediction=np.load(source/'moge-v1'/f"{row['sample_index']:06d}.npz");d=prediction['depth'].copy();d[~prediction['valid']]=np.nan
        d=cv2.resize(d,(row['width'],row['height']),interpolation=cv2.INTER_LINEAR)
        raw_depth.append(d);raw_points.append(lift_pixels(old['uv'][j],d,k))
    raw_points=np.asarray(raw_points);reference=raw_points[anchor];training=np.flatnonzero(np.arange(reference.shape[0])%3!=0);heldout=np.flatnonzero(np.arange(reference.shape[0])%3==0)
    out.mkdir();shutil.copytree(source/'inputs',out/'inputs',copy_function=os.link);shutil.copy2(source/'unit-proxy-battery.obj',out/'unit-proxy-battery.obj');(out/'depth').mkdir();(out/'motion-cotracker').mkdir()
    pack['camera'].update(intrinsics=k.tolist(),origin='MoGe-2 median estimated intrinsics under fixed-optics assumption',horizontal_fov_degrees=fov,calibrated=False)
    pack['depth_backend']={'model':'Ruicheng/moge-2-vits-normal','role':'temporary_non_Meta_local_geometry_backend','consistency':'shared estimated intrinsics and per-frame relative depth-scale correction from training point pairs','output_semantics':'model-predicted depth; physical distance remains unverified'}
    pack['limitations']+=['A fixed recording-camera lens/zoom is assumed for this short continuous clip.','Depth-scale correction assumes one rigid battery; it is not a metric camera calibration.','Training and held-out point IDs are disjoint; evaluation measures internal consistency, not physical accuracy.']
    write(out/'experiment.json',pack)
    records=[];diagnostics=[];xyz=[];evaluation=[]
    span=float(2*np.percentile(np.linalg.norm(reference-np.nanmedian(reference,axis=0),axis=1),90))
    for j,(row,depth,points) in enumerate(zip(rows,raw_depth,raw_points)):
        valid=old['valid'][j]&np.isfinite(points).all(axis=1)&np.isfinite(reference).all(axis=1)
        proposed=estimate_depth_scale(reference,points,valid,training)
        accepted=.5<=proposed<=2
        factor=proposed if accepted else 1.0
        corrected=depth*factor;lifted=points*factor;xyz.append(lifted)
        eval_row={'sample_index':row['sample_index'],'timestamp_s':row['timestamp_s'],'scale_candidate':proposed,'scale_applied':factor,'correction_accepted':accepted,'DA_heldout_pair_error':heldout_shape_error(old['xyz_camera'][anchor],old['xyz_camera'][j],valid,heldout),'MoGe_sharedK_heldout_pair_error':heldout_shape_error(reference,points,valid,heldout),'corrected_heldout_pair_error':heldout_shape_error(reference,lifted,valid,heldout)};evaluation.append(eval_row)
        file=out/'depth'/f"{row['sample_index']:06d}.npy";np.save(file,corrected.astype(np.float32))
        mask=(np.asarray(Image.open(out/row['mask']))>0)&np.isfinite(corrected)&(corrected>0);ys,xs=np.where(mask);step=max(1,int(np.ceil(len(xs)/5000)));uv=np.c_[xs[::step],ys[::step]];cloud=out/'depth'/f"{row['sample_index']:06d}-visible-surface.ply";image=np.array(Image.open(out/row['image']).convert('RGB'));write_cloud(cloud,lift_pixels(uv,corrected,k),image[ys[::step],xs[::step]])
        records.append({'sample_index':row['sample_index'],'timestamp_s':row['timestamp_s'],'file':str(file.relative_to(out)),'depth_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'point_cloud':str(cloud.relative_to(out)),'scale_correction':factor,'usable_for_3d':accepted,'status':'relative_depth_corrected' if accepted else 'raw_diagnostic_only_scale_correction_withheld'})
        rec={'sample_index':row['sample_index'],'timestamp_s':row['timestamp_s'],'tracked_points':int(old['valid'][j].sum()),'pose_exported':False,'depth_usable':accepted,'status':'depth_correction_withheld' if not accepted else 'insufficient_correspondences'}
        if accepted and valid.sum()>=6:
            transform,errors=rigid_fit(reference[valid],lifted[valid]);ratio=float(np.median(errors)/span);rec.update(status='diagnostic_only_low_residual' if ratio<=.1 else 'nonrigid_lifted_evidence',median_rigid_residual_model_units=float(np.median(errors)),residual_fraction_of_reference_span=ratio,diagnostic_transform=transform.tolist())
        diagnostics.append(rec)
    comparable=[r for j,r in enumerate(evaluation) if j!=anchor and r['correction_accepted']]
    summary={key:float(np.median([r[key] for r in comparable if r[key] is not None])) for key in ['DA_heldout_pair_error','MoGe_sharedK_heldout_pair_error','corrected_heldout_pair_error']}
    report={'status':'complete_diagnostic_not_calibrated','source_run':'local-battery-v2','shared_hfov_degrees':fov,'per_frame_hfov_range':[min(r['estimated_hfov_degrees'] for r in moge['frames']),max(r['estimated_hfov_degrees'] for r in moge['frames'])],'training_point_ids':training.tolist(),'heldout_point_ids':heldout.tolist(),'anchor_sample':meta['anchor_sample'],'allowed_depth_correction_range':[.5,2],'comparable_nonanchor_frames':len(comparable),'median_heldout_relative_pair_errors':summary,'frames':evaluation,'limitations':['No measured geometry, physical scale or recording-camera extrinsics.','The anchor geometry is model-predicted; improved consistency does not prove correctness.','Correction uses a rigid-component prior and is withheld outside the predeclared bounds.','Scale correction changes depth estimates, not the size of an animated component asset.']}
    write(out/'consistency.json',report)
    write(out/'depth/metadata.json',{'status':'complete','model_provenance':moge['model_provenance'],'source_run':'local-battery-v2/moge-v1','device':'mps_for_model; cpu_for_consistency','camera':pack['camera'],'units':'model_units_relative_depth_corrected_NOT_verified_meters','cpu_operator_fallback_enabled':False,'frames':records,'accuracy_measured':False})
    arrays={name:old[name] for name in old.files};arrays['xyz_camera']=np.asarray(xyz,dtype=np.float32);arrays['xyz_usable']=np.array([r['depth_usable'] for r in diagnostics])[:,None]&old['valid'];np.savez_compressed(out/'motion-cotracker/tracks.npz',**arrays)
    meta.update(source_2d_run='local-battery-v2/motion-cotracker; unchanged 2D tracks reused',frames=diagnostics,coordinate_frame='per-frame recording-camera coordinates; shared MODEL-ESTIMATED optics, no world/extrinsic calibration',low_residual_nonanchor_frames=sum(r['status']=='diagnostic_only_low_residual' for j,r in enumerate(diagnostics) if j!=anchor),consistency_processing_s=round(time.monotonic()-start,3))
    write(out/'motion-cotracker/metadata.json',meta);print(json.dumps({**summary,'shared_hfov_degrees':fov,'comparable_nonanchor_frames':len(comparable),'withheld_samples':[r['sample_index'] for r in evaluation if not r['correction_accepted']]},indent=2))
if __name__=='__main__':main()
