#!/usr/bin/env python3
"""Fit battery proportions from visible evidence; complete unseen shape with an explicit cylinder prior."""
import json,sys
from pathlib import Path
import numpy as np
import cv2,trimesh
from PIL import Image
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from geometry import lift_pixels
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def estimate_view(parent,row,k):
    mask=np.array(Image.open(parent/row['mask']))>0;mask=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0
    ys,xs=np.where(mask);uv=np.c_[xs,ys];depth=np.load(parent/'depth'/f"{row['sample_index']:06d}.npy");xyz=lift_pixels(uv,depth,k);good=np.isfinite(xyz).all(1);uv=uv[good];xyz=xyz[good]
    center=np.median(xyz,axis=0);_,_,basis=np.linalg.svd(xyz-center,full_matrices=False);axis=basis[0]
    if axis[1]<0:axis=-axis
    image_center=uv.mean(0);_,_,image_basis=np.linalg.svd(uv-image_center,full_matrices=False);local=(uv-image_center)@image_basis.T
    pixels_length=float(np.percentile(local[:,0],99.5)-np.percentile(local[:,0],.5));pixels_width=float(np.percentile(local[:,1],97)-np.percentile(local[:,1],3));z=float(np.median(xyz[:,2]));radius=pixels_width*z/(2*k[0,0]);sin=np.linalg.norm(axis[:2])
    if sin<.3:raise ValueError('Axis too foreshortened for this simple evidence fit')
    length=(pixels_length*z/k[0,0]-2*radius*abs(axis[2]))/sin
    if not 1.5<length/(2*radius)<8:raise ValueError('Evidence does not support a reasonable cylindrical hypothesis')
    return {'sample_index':row['sample_index'],'timestamp_s':row['timestamp_s'],'radius_model_units':radius,'length_model_units':float(length),'length_over_diameter':float(length/(2*radius)),'axis_camera':axis.tolist(),'visible_center_camera':center.tolist(),'length_pixels':pixels_length,'width_pixels':pixels_width},uv,xyz

def closed_battery_mesh(radius,length,segments=64):
    # Small inferred edge bevel and flat end closures are explicit shape priors.
    rings=[(-length/2,.94*radius),(-length/2+.06*radius,radius),(length/2-.06*radius,radius),(length/2,.94*radius)]
    vertices=[]
    for z,r in rings:
        vertices += [[r*np.cos(t),r*np.sin(t),z] for t in np.arange(segments)*2*np.pi/segments]
    vertices += [[0,0,-length/2],[0,0,length/2]];faces=[]
    for j in range(len(rings)-1):
        for i in range(segments):
            n=(i+1)%segments;a=j*segments+i;b=j*segments+n;c=(j+1)*segments+i;d=(j+1)*segments+n
            faces += [[a,b,d],[a,d,c]]
    for i in range(segments):
        n=(i+1)%segments;faces += [[4*segments,n,i],[4*segments+1,3*segments+i,3*segments+n]]
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=True);mesh.fix_normals()
    if mesh.volume<0:mesh.invert()
    return mesh

def main():
    base=ROOT/'data/camera-battery-withdrawal-15fps/geometry';parent=base/'local-battery-v3';out=base/'battery-fitted-v1';out.mkdir(exist_ok=False)
    pack=read(parent/'experiment.json');k=np.array(pack['camera']['intrinsics']);chosen=[1,8,16,24,32,40,46];estimates=[];reference=None
    for i in chosen:
        row=next(r for r in pack['samples'] if r['sample_index']==i);estimate,uv,xyz=estimate_view(parent,row,k);estimates.append(estimate)
        if i==16:reference=(estimate,row,uv,xyz)
    ratio=float(np.median([v['length_over_diameter'] for v in estimates]));radius_model=float(np.median([v['radius_model_units'] for v in estimates]));mesh=closed_battery_mesh(.5,ratio)
    estimate,row,uv,xyz=reference;axis=np.array(estimate['axis_camera']);surface_center=np.array(estimate['visible_center_camera']);front=-surface_center;front-=axis*np.dot(front,axis);front/=np.linalg.norm(front);side=np.cross(axis,front);rotation=np.stack([front,side,axis],axis=1)
    axial=xyz@axis;center=surface_center+axis*((np.percentile(axial,.5)+np.percentile(axial,99.5))/2-np.dot(surface_center,axis))-.7*radius_model*front
    vertices_camera=np.asarray(mesh.vertices)*(2*radius_model)@rotation.T+center
    distance,nearest=cKDTree(xyz).query(vertices_camera);normals_camera=np.asarray(mesh.vertex_normals)@rotation.T;to_camera=-vertices_camera;to_camera/=np.linalg.norm(to_camera,axis=1)[:,None]
    evidence_appearance=(distance<.65*radius_model)&((normals_camera*to_camera).sum(1)>.15)
    source=np.array(Image.open(parent/row['image']).convert('RGB'));pixels=np.rint(uv[nearest]).astype(int);colors=np.full((len(mesh.vertices),4),[148,159,166,255],np.uint8);colors[evidence_appearance,:3]=source[pixels[evidence_appearance,1],pixels[evidence_appearance,0]];mesh.visual.vertex_colors=colors
    mesh.export(out/'battery.glb');mesh.export(out/'battery.obj');mesh.export(out/'battery.ply')
    leave_one_out=[{'sample_index':v['sample_index'],'relative_aspect_difference':float(abs(np.median([u['length_over_diameter'] for u in estimates if u['sample_index']!=v['sample_index']])/v['length_over_diameter']-1))} for v in estimates]
    metadata={'status':'candidate_generated_pending_visual_review','method':'multi-view evidence-fitted cylindrical completion','source_sha256':pack['source_sha256'],'candidate_entity_ids':pack['candidate_entity_ids'],'source_run':'local-battery-v3','proportion_evidence':estimates,'length_over_diameter':ratio,'asset_units':'diameter = 1; physical dimensions unverified','estimated_model_units_per_asset_unit':2*radius_model,'appearance_source_sample':16,'appearance_evidence_vertex_ids':np.flatnonzero(evidence_appearance).tolist(),'appearance_evidence_fraction':float(evidence_appearance.mean()),'inferred_appearance':'neutral grey; not sampled from occluded/background pixels','unseen_geometry':'circular cross-sections, hidden back, flat end caps and a small edge bevel supplied by explicit shape prior','geometry_measured':False,'generative_model_output':False,'sam3d_output':False,'watertight':bool(mesh.is_watertight),'winding_consistent':bool(mesh.is_winding_consistent),'euler_number':int(mesh.euler_number),'vertices':len(mesh.vertices),'faces':len(mesh.faces),'leave_one_view_out_aspect_check':leave_one_out,'reference_camera_transform_for_appearance_only':{'rotation':rotation.tolist(),'translation':center.tolist(),'accepted_video_alignment':False},'limitations':['This is an evidence-fitted geometric approximation, not a measured battery scan.','Length and radius inherit monocular depth and estimated-intrinsics uncertainty.','Circular cross-sections and hidden end geometry are priors; terminals and exact manufacturing details are not recovered.','Axial roll, unseen markings and exact metric size remain unknown.','Source appearance is partial; nearest-surface transfer may contain registration/segmentation error.']}
    write(out/'metadata.json',metadata);print(json.dumps({k:metadata[k] for k in ['method','length_over_diameter','watertight','vertices','faces','appearance_evidence_fraction']},indent=2))
if __name__=='__main__':main()
