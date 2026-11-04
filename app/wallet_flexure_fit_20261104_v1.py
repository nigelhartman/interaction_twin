"""Bounded automatic soft-fold proxy research; never changes the approved twin.

Native PCA, split and source SAM masks determine all geometry/pose parameters.
The two-surface fold is a display approximation, not recovered leather physics.
"""
from pathlib import Path
import json,hashlib,argparse,time
import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from native_appearance import load_native_glb

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'output/experience-refinement-20261104-v1/wallet'
SCENE=ROOT/'data/rigidity-feasibility/overnight-camera-wallet-20261103-v1/wallet/scene.json'

def ref(path):
    path=Path(path);raw=path.read_bytes();return {'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
def save(path,obj):
    with Path(path).open('x')as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def context():
    s=json.loads(SCENE.read_text());p=json.loads((ROOT/s['original_program']['path']).read_text());c=next(c for c in s['components']if c['id']=='wallet_holder');m=load_native_glb((ROOT/c['asset']['path']).read_bytes())
    v=m.vertices;center=v.mean(0);eig,U=np.linalg.eigh(np.cov((v-center).T));U=U[:,np.argsort(eig)[::-1]]
    if np.linalg.det(U)<0:U[:,2]*=-1
    q=(v-center)@U;split=float(np.median(q[:,2]));return s,p,c,m,center,U,q,split
def fold(points,angle,hinge,split,side=1):
    q=np.array(points,copy=True);moving=side*(q[:,2]-split)>=0
    # All points in one measured shell receive one continuous virtual-fold
    # transform. Cross-split faces are explicitly excluded from the derivative.
    R=Rotation.from_rotvec([0,float(angle),0]).as_matrix();q[moving]=(q[moving]-hinge)@R.T+hinge
    return q
def leaf_hulls(q,split):
    hulls=[]
    for positive in [False,True]:
        chosen=q[(q[:,2]>=split)==positive]
        # Convex native-PCA silhouette is only an optimization proxy. Full native
        # split surfaces, not these hulls, are the concrete display derivative.
        ids=cv2.convexHull(chosen[:,:2].astype(np.float32),returnPoints=False)[:,0]
        hulls.append(chosen[ids])
    return hulls
def affine_from_params(par,prior,center,U,scale,K,target_box):
    R=Rotation.from_rotvec(par[:3]).as_matrix()@prior
    z=np.exp(par[3]);cx,cy,tw,th=target_box
    t=np.array([(cx+par[4]*tw-K[0,2])/K[0,0]*z,(cy+par[5]*th-K[1,2])/K[1,1]*z,z])
    return R*scale,t
def project(q,A,t,K):
    p=q@A.T+t
    if p[:,2].min()<=.01:return None
    h=p@K.T;return h[:,:2]/h[:,2:]
def raster(hulls,angle,hinge,split,side,A,t,K,shape):
    image=np.zeros(shape,np.uint8)
    for hull in hulls:
        pts=fold(hull,angle,hinge,split,side) if np.mean(hull[:,2])>=split else hull
        # If negative shell moves, choose that shell instead.
        if side<0:pts=fold(hull,angle,hinge,split,side)if np.mean(hull[:,2])<split else hull
        uv=project(pts,A,t,K)
        if uv is None:return image
        cv2.fillConvexPoly(image,cv2.convexHull(np.rint(uv).astype(np.int32)),1)
    return image
def score(mask,target):
    inter=np.count_nonzero((mask>0)&target);union=np.count_nonzero((mask>0)|target);return inter/max(union,1)
def fit_frame(s,p,c,center,U,q,split,hulls,index,rng):
    e=p['evidence'][index-1];obs=next(o for o in e['observations']if o['track_id']=='wallet_holder');mp=ROOT/obs['mask_path'];assert ref(mp)['sha256']==obs['mask_sha256']
    full=cv2.imread(str(mp),0)>0;factor=3;target=cv2.resize(full.astype(np.uint8),(180,320),interpolation=cv2.INTER_NEAREST)>0
    yy,xx=np.nonzero(target);box=[float((xx.min()+xx.max())/2),float((yy.min()+yy.max())/2),float(np.ptp(xx)),float(np.ptp(yy))]
    K=np.asarray(s['camera']['K'],float)/factor;K[2,2]=1
    prior_pose=next(r for r in c['poses']if r['source_sample_index']==index)
    if prior_pose['status']=='unknown':return {'source_sample_index':index,'timestamp_s':e['timestamp_s'],'status':'unknown','reason':'original_holder_pose_unknown_preserved','source_mask':ref(mp)}
    M=np.asarray(prior_pose['camera_from_native_mesh']);scale=np.cbrt(np.linalg.det(M[:3,:3]));prior=M[:3,:3]@U/scale
    native_depth=float((M[:3,:3]@center+M[:3,3])[2]);bounds=[(-np.pi,np.pi)]*3+[(np.log(native_depth*.5),np.log(native_depth*3)),(-.7,.7),(-.7,.7),(0,np.pi)]
    choices=[]
    for edge in [-1,1]:
        hinge=np.array([np.quantile(q[:,0],.01 if edge<0 else .99),np.median(q[:,1]),split])
        def objective(par):
            A,t=affine_from_params(par,prior,center,U,scale,K,box);im=raster(hulls,par[6],hinge,split,1,A,t,K,target.shape)
            return 1-score(im,target)
        # Predeclared deterministic global starts; no source-authored angles.
        starts=[np.array([0.,0.,0.,np.log(native_depth),0.,0.,a])for a in np.linspace(0,np.pi,5)]
        for _ in range(130):
            starts.append(np.array([*rng.uniform(-np.pi,np.pi,3),np.log(native_depth*rng.uniform(.6,2.4)),*rng.uniform(-.12,.12,2),rng.uniform(0,np.pi)]))
        ranked=sorted((objective(x),i,x)for i,x in enumerate(starts))[:8]
        for _,_,start in ranked:
            result=minimize(objective,start,method='Powell',bounds=bounds,options={'maxiter':40,'maxfev':2200,'xtol':.002,'ftol':.0002})
            val=objective(result.x);choices.append({'objective':float(val),'edge':edge,'hinge':hinge,'parameters':result.x,'prior_rotation':prior,'scale':scale,'target_box':box,'evaluations':int(result.nfev)})
    choices.sort(key=lambda r:r['objective']);best=choices[0];A,t=affine_from_params(best['parameters'],prior,center,U,scale,K,box)
    fits=[{'iou':1-r['objective'],'angle_rad':float(r['parameters'][6]),'edge':r['edge']}for r in choices]
    near=[r for r in fits if r['iou']>=1-best['objective']-.025]
    return {'source_sample_index':index,'timestamp_s':e['timestamp_s'],'status':'estimated_proxy'if 1-best['objective']>=.6 else'unknown','reason':'automatic_native_PCA_two_surface_silhouette_fit_not_independent_shape_validation',
      'source_RGB':{'path':e['frame_path'],'sha256':e['frame_sha256']},'source_mask':ref(mp),'proxy_iou':1-best['objective'],'fold_angle_rad':float(best['parameters'][6]),'fold_angle_deg':float(np.degrees(best['parameters'][6])),
      'angle_near_optimal_range_deg':[float(np.degrees(min(r['angle_rad']for r in near))),float(np.degrees(max(r['angle_rad']for r in near)))],
      'fold_edge_sign':best['edge'],'hinge_PCA':best['hinge'].tolist(),'camera_from_PCA':np.c_[A,t].tolist(),'parameter_vector':best['parameters'].tolist(),
      'initialization_source':'unchanged_existing_native_pose_plus_bounded_global_search','top_fit_candidates':fits,'whole_pose_refit':True,'camera_calibrated':False,'metric_scale_verified':False,'physical_accuracy_verified':False}
def run(indices):
    out=BASE/'fit-v1';out.mkdir(exist_ok=False);s,p,c,m,center,U,q,split=context();hulls=leaf_hulls(q,split);rng=np.random.default_rng(407);rows=[];start=time.monotonic()
    save(out/'native-frame.json',{'format':'interaction-twin-automatic-native-flexure-frame','asset':c['asset'],'PCA_origin':center.tolist(),'PCA_axes':U.tolist(),'native_PCA_extent':np.ptp(q,axis=0).tolist(),'split_thin_coordinate':split,'split_rule':'native_median_thin_PCA_coordinate','fold_axis_PCA':[0,1,0],'fold_edge_rule':'native_1_or_99_percentile_width_selected_by_source_mask_fit','manual_points':False,'implementation':ref(Path(__file__))})
    for index in indices:
        row=fit_frame(s,p,c,center,U,q,split,hulls,index,rng);rows.append(row);save(out/f'frame-{index:03d}.json',row);print(index,row['status'],round(row.get('proxy_iou',0),3),round(row.get('fold_angle_deg',0),1),flush=True)
    save(out/'report.json',{'format':'interaction-twin-source-driven-soft-fold-proxy-research','version':1,'baseline_scene':ref(SCENE),'source_program':s['original_program'],'native_frame':ref(out/'native-frame.json'),'frames':rows,'wall_seconds':time.monotonic()-start,'new_API_calls':0,'new_GPU_calls':0,'physical_accuracy_verified':False,'public_eligible':False,'limitations':['2-surface virtual fold is a bounded display approximation for deformable leather.','Source SAM silhouettes used for fitting; reported overlap is not heldout validation.','Uncalibrated projection and occlusion can make fold angle nonidentifiable; near-optimal range is recorded.','Current model does not yet solve separately moving pouch flap or material deformation.','Native geometry and original program/scene stay unchanged.']})
    print(ref(out/'report.json'),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--indices',default='1,10,14,18,21,30,94,99');args=a.parse_args();run([int(i)for i in args.indices.split(',')])
