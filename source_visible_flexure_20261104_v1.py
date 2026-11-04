"""CPU-only, evidence-bound source-visible nonrigid surfaces.

This is an additive learned 2.5D depth-prior representation, not a SAM-native
mesh or a recovered rigid/material mechanism. It never paints hidden surfaces.
"""
from pathlib import Path
import argparse,hashlib,json,os,socket,time,copy
import cv2
import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2
import trimesh

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/source-visible-flexure/wallet-20261104-v1'
OUT=ROOT/'output/experience-refinement-20261104-v1/wallet/source-visible-flexure-v1'
SCENE=ROOT/'data/rigidity-feasibility/overnight-camera-wallet-20261103-v1/wallet/scene.json'
REVIEW=ROOT/'output/experience-refinement-20261104-v1/wallet/muse-review/attempt/result.json'
POLICY={'device':'cpu','checkpoint':'models/moge-2-vits-normal/model.pt','input_dimensions':[252,448],
 'num_tokens':600,'use_fp16':False,'downloads_allowed':False,'network_disabled':True,
 'source_grid_stride_px':3,'mask_core_erosion_px':2,'maximum_relative_triangle_depth_edge':.06,
 'source_RGB_sampling':'nearest_exact_original_pixel','hidden_surfaces':'absent_not_inpainted',
 'source_camera':'unchanged_estimated_native_scene_K','depth_gauge':'per_frame_native_holder_centroid_depth_over_prior_mask_median',
 'timing':'nearest_source_keyframe_within_0.18s_only_no_geometric_interpolation',
 'maximum_source_time_error_s':.18,'maximum_contiguous_source_gap_s':.5,
 'frame_selection':'all_21_to_30_plus_exact_Muse_open_rows_excluding_occluded_or_uncertain',
 'numerical_fold_parameters':'fallible_learned_surface_plane_estimates_only_not_measured',
 'physical_accuracy_verified':False,'simulation_ready':False,'native_SAM_contract_compatible':False}
def ref(path):
    path=Path(path);raw=path.read_bytes();return {'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
def save(path,value):
    with Path(path).open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def checked(r):
    p=ROOT/r['path'];assert ref(p)['sha256']==r['sha256'];return p
def resize_depth(depth,valid,width,height):
    valid=valid&np.isfinite(depth)&(depth>0);weights=cv2.resize(valid.astype(np.float32),(width,height),interpolation=cv2.INTER_LINEAR)
    inv=np.zeros(depth.shape,np.float32);inv[valid]=1/depth[valid];up=cv2.resize(inv,(width,height),interpolation=cv2.INTER_LINEAR)
    out=np.full((height,width),np.nan,np.float32);good=(weights>.999)&(up>0);out[good]=weights[good]/up[good];return out
def mesh_from_depth(rgb,depth,mask,K,stride=3):
    h,w=mask.shape;safe=cv2.erode(mask.astype(np.uint8),np.ones((POLICY['mask_core_erosion_px'],)*2,np.uint8))>0
    yy,xx=np.mgrid[0:h:stride,0:w:stride];z=depth[yy,xx];good=safe[yy,xx]&np.isfinite(z)&(z>0);idx=np.full(good.shape,-1,np.int32);idx[good]=np.arange(good.sum())
    sx,sy=xx[good],yy[good];zz=z[good];vertices=np.c_[(sx-K[0,2])/K[0,0]*zz,(sy-K[1,2])/K[1,1]*zz,zz]
    faces=[]
    for a,b,c in [(idx[:-1,:-1],idx[1:,:-1],idx[:-1,1:]),(idx[1:,:-1],idx[1:,1:],idx[:-1,1:])]:
        tri=np.c_[a.ravel(),b.ravel(),c.ravel()];tri=tri[(tri>=0).all(1)]
        if len(tri):
            depths=vertices[tri,2];tri=tri[np.ptp(depths,axis=1)/depths.mean(1)<POLICY['maximum_relative_triangle_depth_edge']];faces.append(tri)
    faces=np.concatenate(faces)if faces else np.zeros((0,3),np.int32)
    # Positive image X/Y increases point downwards. These faces point towards
    # the camera (-Z); observed RGB is never projected onto their hidden backs.
    if len(faces):
        n=np.cross(vertices[faces[0,1]]-vertices[faces[0,0]],vertices[faces[0,2]]-vertices[faces[0,0]])
        if n[2]>0:faces=faces[:,::-1]
    return vertices,faces,rgb[sy,sx],np.c_[sx,sy]
def plane(points):
    origin=np.median(points,axis=0);_,_,U=np.linalg.svd(points-origin,full_matrices=False);n=U[-1]
    if n[2]>0:n=-n
    dist=(points-origin)@n;return n,float(-origin@n),float(np.median(abs(dist))),origin
def virtual_fold(vertices,faces,source_xy,holder_mask,flap_mask,K):
    """Report unstable/ambiguous fits, never manufacture a physical hinge."""
    result={'format':'interaction-twin-virtual-flexure-parameters','type':'virtual_flexure_band','status':'unknown',
      'angle_rad':None,'origin_camera':None,'axis_camera':None,'angle_is_measured':False,'metric_scale_verified':False,
      'source_of_3D':'learned_monocular_depth_prior','physical_accuracy_verified':False,'reason':'insufficient_distinct_plane_evidence'}
    if len(faces)<100:return result
    tri=vertices[faces];norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm/=np.maximum(np.linalg.norm(norm,axis=1)[:,None],1e-12)
    centers=tri.mean(1);xy=np.rint(source_xy[faces].mean(1)).astype(int);host=(holder_mask[xy[:,1],xy[:,0]]>0)&(flap_mask[xy[:,1],xy[:,0]]==0)
    chosen=np.flatnonzero(host)[::max(1,int(host.sum()/3000))]
    if len(chosen)<100:return result
    # Deterministic normal clustering; the plane/angle is explicitly a model
    # hypothesis. The corresponding actual inferred points are also fitted.
    normals=norm[chosen];initial=normals[[np.argmin(normals[:,0]),np.argmax(normals[:,0])]]
    try:means,labels=kmeans2(normals,initial,iter=30,minit='matrix',missing='raise')
    except Exception as e:result['reason']='normal_cluster_failure_'+type(e).__name__;return result
    groups=[centers[chosen[labels==i]]for i in range(2)];fractions=[len(a)/len(chosen)for a in groups];result['surface_cluster_fractions']=fractions
    if min(fractions)<.16:result['reason']='one_cluster_too_small_or_occluded';return result
    fits=[plane(a)for a in groups];n0,d0,e0,c0=fits[0];n1,d1,e1,c1=fits[1];cos=float(np.clip(n0@n1,-1,1));normal_angle=float(np.arccos(cos));opening=float(np.pi-normal_angle)
    result.update({'surface_plane_normals':[n0.tolist(),n1.tolist()],'surface_plane_residuals':[e0,e1],
      'opening_angle_prior_rad':opening,'opening_angle_prior_deg':float(np.degrees(opening)),
      'angle_uncertainty':'unquantified_monocular_prior_and_plane_partition_ambiguity_not_calibrated',
      'numerical_estimate_may_not_correspond_to_the_actual_spine':True})
    axis=np.cross(n0,n1);s=float(np.linalg.norm(axis))
    if s<.12:result['reason']='near_parallel_leaf_planes_axis_unidentifiable';return result
    axis/=s;center=(c0+c1)/2;origin=np.linalg.solve(np.stack([n0,n1,axis]),np.array([-d0,-d1,axis@center]));extent=float(np.linalg.norm(np.ptp(vertices,axis=0)));dist=min(np.linalg.norm(origin-c0),np.linalg.norm(origin-c1))
    if dist>extent*.7:result['reason']='plane_intersection_outside_visible_surface_support';return result
    result.update({'status':'prior_estimate','origin_camera':origin.tolist(),'axis_camera':axis.tolist(),'angle_rad':opening,
      'reason':'two_source_visible_prior_surface_planes_virtual_intersection_not_rigid_joint',
      'band_width_status':'unknown_not_authored','band_width':None,'attachment_verified':False})
    return result
def render_board(rgb,mask,vertices,faces,colors,K,path,title):
    from alignment_render import render_mesh
    from PIL import ImageDraw,ImageFont
    h,w=mask.shape;renders=[]
    for angle in [0,24]:
        if angle:
            from scipy.spatial.transform import Rotation
            center=vertices.mean(0);R=Rotation.from_euler('y',angle,degrees=True).as_matrix();v=(vertices-center)@R.T+center
        else:v=vertices
        rr=render_mesh(v,faces,K,np.eye(3),np.zeros(3),w,h,vertex_colors=colors)
        image=np.full_like(rgb,[238,241,247]);image[rr['mask']]=rr['rgb'][rr['mask']];renders.append(image)
    board=Image.new('RGB',(w*3,h+85),'#f6f8fc');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',20)
    for n,(im,label)in enumerate(zip([rgb,*renders],['Actual recording','Source-visible depth-prior mesh','Same observed surface · orbit24°'])):
        board.paste(Image.fromarray(im),(n*w,45));draw.text((n*w+12,12),label,fill='#172033',font=font)
    draw.text((12,h+53),title,fill='#172033',font=font);board.save(path)
def run():
    DATA.mkdir(parents=True,exist_ok=False);OUT.mkdir(parents=True,exist_ok=False);scene=json.loads(SCENE.read_text());program=json.loads(checked(scene['original_program']).read_text());review=json.loads(REVIEW.read_text())
    native=next(c for c in scene['components']if c['id']=='wallet_holder');checkpoint=ROOT/POLICY['checkpoint'];provenance=json.loads(checkpoint.with_name('provenance.json').read_text());assert ref(checkpoint)['sha256']==provenance['checkpoint_sha256']
    observations={r['source_index']:r for r in review['observations']};wanted=sorted(set(range(21,31))|{i for i,r in observations.items()if r['wallet_state']=='open'})
    protocol={'format':'interaction-twin-source-visible-flexure-protocol','version':1,'policy':POLICY,'source_scene':ref(SCENE),'source_program':scene['original_program'],'Muse_qualitative_review':ref(REVIEW),'checkpoint':ref(checkpoint),'checkpoint_provenance':provenance,'implementation':ref(Path(__file__)),'source_indices_before_inference':wanted,'frozen_before_inference':True}
    save(OUT/'protocol.json',protocol)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    def no_network(*a,**kw):raise RuntimeError('Network disabled for CPU source-visible depth prior')
    socket.socket.connect=no_network;socket.create_connection=no_network
    import torch
    from moge.model.v2 import MoGeModel
    torch.set_num_threads(4);model=MoGeModel.from_pretrained(str(checkpoint)).to('cpu').eval();K=np.asarray(scene['camera']['K']);rows=[];t0=time.monotonic()
    for e in program['evidence']:
        index=e['source_index'];base={'source_sample_index':index,'timestamp_s':e['timestamp_s'],'status':'unknown','native_closed_asset_preserved':native['asset'],'qualitative_state':observations.get(index,{'wallet_state':'unreviewed','pouch_flap_state':'unreviewed'})}
        if index not in wanted:
            base['reason']='no_source_visible_flexure_mesh_requested_or_qualitative_occlusion_barrier';rows.append(base);continue
        pose=next(r for r in native['poses']if r['source_sample_index']==index)
        if pose['status']=='unknown':base['reason']='original_parent_unknown_preserved';rows.append(base);continue
        source=ROOT/e['frame_path'];assert ref(source)['sha256']==e['frame_sha256'];rgb=np.asarray(Image.open(source).convert('RGB'));h,w=rgb.shape[:2];labels={};refs=[]
        for cid in ['wallet_holder','pouch_flap','gold_coin']:
            item=next((o for o in e['observations']if o['track_id']==cid),None)
            if item:mp=checked({'path':item['mask_path'],'sha256':item['mask_sha256']});labels[cid]=np.asarray(Image.open(mp))>0;refs.append({'track_id':cid,**ref(mp)})
            else:labels[cid]=np.zeros((h,w),bool)
        combined=(labels['wallet_holder']|labels['pouch_flap'])&~labels['gold_coin'];image=cv2.resize(rgb,tuple(POLICY['input_dimensions']),interpolation=cv2.INTER_AREA);tensor=torch.from_numpy(image).permute(2,0,1).float()/255;started=time.monotonic()
        with torch.inference_mode():prediction=model.infer(tensor,num_tokens=POLICY['num_tokens'],use_fp16=False,apply_mask=True)
        raw={k:v.cpu().numpy()for k,v in prediction.items()if hasattr(v,'cpu')};frame_dir=DATA/f'source-{index:03d}';frame_dir.mkdir();np.savez_compressed(frame_dir/'depth-prior.npz',**raw)
        depth=resize_depth(raw['depth'],raw['mask'].astype(bool),w,h);valid=combined&np.isfinite(depth)&(depth>0)
        if valid.sum()<300:base['reason']='insufficient_source_mask_valid_depth';rows.append(base);continue
        M=np.asarray(pose['camera_from_native_mesh']);gauge=float(M[2,3]);factor=gauge/float(np.median(depth[valid]));depth*=factor
        vertices,faces,colors,xy=mesh_from_depth(rgb,depth,combined,K,POLICY['source_grid_stride_px'])
        if len(faces)<200:base['reason']='insufficient_source_surface_faces';rows.append(base);continue
        mesh=trimesh.Trimesh(vertices=vertices,faces=faces,vertex_colors=colors,process=False)
        mesh.metadata.update({'source_visible_only':True,'learned_depth_prior':True,'SAM_native_mesh':False,'physical_accuracy_verified':False});asset=frame_dir/'observed-surface.glb';asset.write_bytes(mesh.export(file_type='glb'))
        np.savez_compressed(frame_dir/'source-pixel-map.npz',source_xy=xy,depth=vertices[:,2],faces=faces)
        flex=virtual_fold(vertices,faces,xy,labels['wallet_holder'],labels['pouch_flap'],K)
        base.update({'status':'source_observed_prior','reason':'exact_source_RGB_SAM_region_with_CPU_learned_monocular_depth_prior',
          'asset':ref(asset),'source_RGB':ref(source),'source_masks':refs,'depth_prior':ref(frame_dir/'depth-prior.npz'),'source_pixel_map':ref(frame_dir/'source-pixel-map.npz'),
          'camera_from_asset':np.eye(4).tolist(),'depth_scale_gauge':factor,'native_prior_translation_z':gauge,'vertices':len(vertices),'triangles':len(faces),'source_visible_only':True,
          'virtual_flexure':flex,'wall_seconds':time.monotonic()-started,'new_API_calls':0,'new_GPU_calls':0,'physical_accuracy_verified':False})
        save(frame_dir/'provenance.json',base);base['provenance']=ref(frame_dir/'provenance.json');rows.append(base)
        if index in [14,21,27,28,30,94]:render_board(rgb,combined,vertices,faces,colors,K,OUT/f'comparison-{index:03d}.png',f'Source{index:03d} · learned2.5Dsurface, not SAMnative or measured physics')
        print(index,len(vertices),len(faces),flex['status'],round(time.monotonic()-started,2),flush=True)
    descriptor={'format':'interaction-twin-source-visible-flexure-series','version':1,'kind':'source_observed_monocular_depth_prior','source_scene':ref(SCENE),'source_program':scene['original_program'],'source':scene['source'],'camera':scene['camera'],'policy':POLICY,'protocol':ref(OUT/'protocol.json'),
      'replacement_component_ids':['wallet_holder','pouch_flap'],'unaffected_component_ids':['gold_coin'],'closed_native_assets':[c['asset']for c in scene['components']if c['id']in['wallet_holder','pouch_flap']],
      'semantic_relationships':[{'type':'virtual_flexure_band','scope':'two_wallet_leaves','state_source':'fallible_Muse_RGB_observations','measured_joint':False,'joint_position':'unknown_unless_per_frame_prior_intersection_supported','joint_angle':'unknown_unless_per_frame_prior_plane_hypothesis_supported'},
       {'type':'attached_flexible_flap','parent':'wallet_holder','child':'pouch_flap','state_source':'fallible_Muse_snap_and_mouth_RGB_observations','geometry_under_hand':'unknown_absent','rigid_revolute_joint':False}],
      'samples':rows,'physical_accuracy_verified':False,'simulation_ready':False,'manual_geometry_or_poses_used':False,'native_SAM_contract_compatible':False,'new_API_calls':0,'new_GPU_calls':0,
      'limitations':['Source-visible surface depth is a learned monocular prior, not measured geometry.','No hidden backs, pocket interiors under occlusion, watertight volumes, material model, grasp or contact certification.','Independent source frames may change inferred depth and topology; no geometric interpolation is permitted.','The qualitative flexure/flap states are fallible Muse observations, distinct from numerical joint recovery.','Native SAM models and descriptive program remain intact and downloadable as originals.']}
    save(DATA/'descriptor.json',descriptor);save(OUT/'delivery.json',{'format':'interaction-twin-source-visible-flexure-delivery','descriptor':ref(DATA/'descriptor.json'),'supported_sources':[r['source_sample_index']for r in rows if r['status']=='source_observed_prior'],'unknown_sources':[r['source_sample_index']for r in rows if r['status']=='unknown'],'implementation':ref(Path(__file__)),'wall_seconds':time.monotonic()-t0,'production_state_changed':False,'public_eligible':False,'new_API_calls':0,'new_GPU_calls':0});print(ref(DATA/'descriptor.json'),flush=True)
if __name__=='__main__':run()
