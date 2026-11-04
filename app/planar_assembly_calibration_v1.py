"""Source-reviewed, global panel calibration and nonintersecting rear hinge.

No points, angles or dimensions are authored. The width/depth footprints come
from original mesh support planes, linked by source-reviewed flush-panel rules.
"""
from __future__ import annotations
import math
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

import sam3d_mechanisms as M
import sam3d_common_scene as S
import occluded_hinge_recovery_v1 as R

POLICY={'footprint_quantiles':[.001,.999],'maximum_affine_axis_change':.20,'numeric_clearance_epsilon_fraction':.0001,
        'maximum_parent_rotation_rad':math.radians(12),'maximum_parent_translation_extent_fraction':.12,'maximum_parent_depth_fraction':.15,
        'angle_grid_count':181,'maximum_fit_evaluations':650,'sparse_vertices':1400,'minimum_union_iou':.65,
        'minimum_visible_part_iou':.55,'maximum_heldout_union_regression':.05,'physical_accuracy_verified':False}


def transform(vertices,matrix):return np.asarray(vertices)@np.asarray(matrix)[:3,:3].T+np.asarray(matrix)[:3,3]


def normal_from_supported_interior(vertices,matrix):
    normal=np.linalg.svd(vertices-vertices.mean(0),full_matrices=False)[2][-1];pose=np.asarray(matrix);normal*=np.sign((pose[:3,:3]@normal)@(-pose[:3,3]));return normal


def calibrate(parent,child,joint,parent_interior_normal,child_interior_normal,*,width_quantiles=None):
    parent=np.asarray(parent);child=np.asarray(child);sp,sc=joint['parent_native_scale'],joint['child_native_scale']
    pn=np.asarray(parent_interior_normal);cn=np.asarray(child_interior_normal)
    u=np.linalg.svd(parent-parent.mean(0),full_matrices=False)[2][0];u-=pn*(u@pn);u/=np.linalg.norm(u)
    if u@np.asarray(joint['axis'])>0:u*=-1
    front=np.cross(u,pn);front/=np.linalg.norm(front)
    old_anchor=np.asarray(joint['parent_anchor_native'])
    if front@(old_anchor-parent.mean(0))>0:front*=-1;u*=-1
    # Positive opening moves the front edge away from the keyboard plane.
    axis=np.cross(front,pn);axis/=np.linalg.norm(axis)
    basis=np.stack([u,front,pn],axis=1);pq=parent@basis;plo,phi=np.quantile(pq,POLICY['footprint_quantiles'],axis=0);pe=phi-plo
    if width_quantiles is not None:
        R.require(width_quantiles in ([.001,.999],[.005,.995],[.01,.99]),'Predeclared native robust-width hypothesis required')
        pe[0]=float(np.diff(np.quantile(pq[:,0],width_quantiles))[0])
    cb=np.linalg.svd(child-child.mean(0),full_matrices=False)[2];cq=(child-child.mean(0))@cb.T;clo,chi=np.quantile(cq,POLICY['footprint_quantiles'],axis=0);ce=chi-clo
    scales=np.array([sp*pe[0]/(sc*ce[0]),sp*pe[1]/(sc*ce[1]),1.])
    R.require(max(abs(scales-1))<=POLICY['maximum_affine_axis_change'],'Source-reviewed panel calibration exceeds bounded independent scale correction')
    linear=cb.T@np.diag(scales)@cb;center=child.mean(0);affine=np.eye(4);affine[:3,:3]=linear;affine[:3,3]=center-linear@center;derived=transform(child,affine)
    cw=cb[0];cw-=cn*(cw@cn);cw/=np.linalg.norm(cw)
    source=np.stack([cw,cn,np.cross(cw,cn)],axis=1)
    candidates=[];old=M.rotation_pose(M.forward(np.eye(4),joint['parent_from_joint'],joint['axis'],0.,joint['joint_from_child_zero']))[0]
    for sign in [-1.,1.]:
        right=u*sign;target=np.stack([right,-pn,np.cross(right,-pn)],axis=1);rotation=target@source.T
        candidates.append((Rotation.from_matrix(rotation@old.T).magnitude(),rotation))
    rotation=min(candidates,key=lambda r:r[0])[1];relative=np.eye(4);relative[:3,:3]=rotation*(sc/sp)
    dq=transform(derived,relative)@basis;dlo,dhi=np.quantile(dq,POLICY['footprint_quantiles'],axis=0)
    offset=np.zeros(3);offset[:2]=(plo[:2]+phi[:2]-dlo[:2]-dhi[:2])/2
    epsilon=pe[0]*POLICY['numeric_clearance_epsilon_fraction'];offset[2]=pq[:,2].max()+epsilon-dq[:,2].min();relative[:3,3]=basis@offset
    closedq=transform(derived,relative)@basis
    # The observed attachment is the inner rear edge. A mid-thickness pivot
    # lifts this edge in open states and creates a visibly detached seam.
    # Once the lid swings behind the rear edge it may cross the INFINITE deck
    # plane safely: the finite keyboard ends there. Test the finite footprint.
    pivot_q=np.array([(closedq[:,0].min()+closedq[:,0].max())/2,closedq[:,1].min()-epsilon,closedq[:,2].min()]);pivot=basis@pivot_q
    child_pivot=transform(pivot[None],np.linalg.inv(relative))[0]
    a=np.eye(4);a[:3,:3]/=sp;a[:3,3]=pivot
    b=np.eye(4);b[:3,:3]=rotation*sc;b[:3,3]=-rotation@child_pivot*sc
    exact=M.forward(np.eye(4),a,axis,0.,b);R.require(np.allclose(exact,relative,atol=1e-10),'Closed rear-hinge transformation mismatch')
    checks=[];infinite=[]
    for q in np.linspace(0,math.pi,181):
        moved=transform(derived,M.forward(np.eye(4),a,axis,float(q),b))@basis
        above_footprint=(moved[:,0]>=pq[:,0].min())&(moved[:,0]<=pq[:,0].max())&(moved[:,1]>=pivot_q[1])&(moved[:,1]<=pq[:,1].max())
        infinite.append(float(moved[:,2].min()-pq[:,2].max()))
        checks.append(float(moved[above_footprint,2].min()-pq[:,2].max())if above_footprint.any()else float(epsilon))
    R.require(min(checks)>=-1e-8,'Calibrated lid crosses the keyboard support plane')
    report={'method':'source_reviewed_flush_footprints_opposed_panel_normals_nonpenetrating_rear_edge_hinge','parent_frame':basis.tolist(),
            'parent_interior_normal':pn.tolist(),'child_interior_normal':cn.tolist(),'parent_footprint_extents':pe.tolist(),'child_original_intrinsic_extents':ce.tolist(),
            'intrinsic_lid_axis_scale':scales.tolist(),'affine_matrix':affine.tolist(),'closed_child_from_parent_native':relative.tolist(),
            'pivot_parent_native':pivot.tolist(),'pivot_child_derived_native':child_pivot.tolist(),'axis_parent_native':axis.tolist(),
            'minimum_signed_deck_clearance_native':min(checks),'minimum_infinite_plane_clearance_native':min(infinite),'clearance_scope':'finite_parent_footprint_only; crossing_behind_rear_edge_is_not_deck_penetration','clearance_angle_sample_count':181,'numeric_clearance_epsilon_native':epsilon,
            'closed_signed_deck_clearance_native':checks[0],'closed_lid_footprint_extents':np.ptp(closedq,axis=0).tolist(),
            'width_support_quantiles':width_quantiles or POLICY['footprint_quantiles'],
            'physical_contact_verified':False,'geometry_watertightness_verified':False,'physical_accuracy_verified':False}
    return derived,a,axis,b,report


def quad_pose(vertices,quad,camera,scale,prior,interior_normal,visible_face):
    """Use the shared PnP fitter with a source-derived contour or homography."""
    center=np.asarray(vertices).mean(0);basis=np.linalg.svd(vertices-center,full_matrices=False)[2];coords=(vertices-center)@basis.T;lo,hi=np.percentile(coords,[.3,99.7],axis=0)
    corners=np.array([[lo[0],lo[1],0],[hi[0],lo[1],0],[hi[0],hi[1],0],[lo[0],hi[1],0]])@basis+center
    previous=M.rotation_pose(prior)[0];rows=[]
    for reverse in [False,True]:
        for shift in range(4):
            pixels=np.ascontiguousarray(np.roll(quad[::-1]if reverse else quad,shift,axis=0),np.float64)
            try:_,rs,ts,es=cv2.solvePnPGeneric(np.ascontiguousarray(corners*scale,np.float64),pixels,np.asarray(camera['K']),None,flags=cv2.SOLVEPNP_IPPE)
            except cv2.error:continue
            for rr,tt,error in zip(rs,ts,es):
                rotation=cv2.Rodrigues(rr)[0];t=tt.ravel();facing=float((rotation@interior_normal)@(-t)/np.linalg.norm(t))
                if (visible_face=='interior'and facing<.05)or(visible_face=='exterior'and facing>-.05):continue
                matrix=np.eye(4);matrix[:3,:3]=rotation*scale;matrix[:3,3]=t
                if transform(vertices,matrix)[:,2].min()<=1e-5:continue
                distance=float(Rotation.from_matrix(rotation@previous.T).magnitude());rows.append({'matrix':matrix,'quad_reprojection_px':float(error[0]),'facing':facing,'objective':float(error[0])+distance*2})
    R.require(rows,'No supported positive-depth source-visible panel PnP');return min(rows,key=lambda r:r['objective'])


def fit_sample(parent,child,pmask,cmask,camera,a,axis,b,prior,normal_parent,normal_child,*,closed=False,visible_panel='laptop_base',quad=None):
    k=np.asarray(camera['K']);sp=M.rotation_pose(prior)[2];sc=M.rotation_pose(b)[2]
    observed=pmask if visible_panel=='laptop_base'else cmask;vertices=parent if visible_panel=='laptop_base'else child
    if quad is None:quad=R.mask_quad(observed)
    if visible_panel=='laptop_base':
        initial=quad_pose(vertices,quad,camera,sp,prior,normal_parent,'exterior'if closed else'interior')['matrix']
    else:
        cp=M.forward(prior,a,axis,0.,b);initial=quad_pose(vertices,quad,camera,sc,cp,normal_child,'exterior')['matrix']@np.linalg.inv(M.forward(np.eye(4),a,axis,0.,b))
    pv=parent[np.linspace(0,len(parent)-1,min(len(parent),POLICY['sparse_vertices']),dtype=int)];cv=child[np.linspace(0,len(child)-1,min(len(child),POLICY['sparse_vertices']),dtype=int)]
    union=pmask|cmask;tu=S.mask_shape(union);tp=S.mask_shape(pmask)if pmask.any()else None;tc=S.mask_shape(cmask)if cmask.any()else None
    ratio=initial[2,3]/k[0,0];bounds=[(-POLICY['maximum_parent_rotation_rad'],POLICY['maximum_parent_rotation_rad'])]*3+[(-tu['extent'][0]*ratio*.12,tu['extent'][0]*ratio*.12),(-tu['extent'][1]*ratio*.12,tu['extent'][1]*ratio*.12),(-initial[2,3]*.15,initial[2,3]*.15)]
    if not closed:bounds.append((0,math.pi))
    def matrices(x):
        p=initial.copy();p[:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix()@initial[:3,:3];p[:3,3]+=x[3:6];q=0. if closed else float(x[6]);return p,M.forward(p,a,axis,q,b),q
    def loss(x):
        p,c,q=matrices(x);up,vp=M.project(pv,p,k);uc,vc=M.project(cv,c,k)
        if not vp.all()or not vc.all():return 1e5
        lu=S.silhouette_score(np.r_[up,uc],tu)[0]
        if closed:
            visible=S.silhouette_score(up if visible_panel=='laptop_base'else uc,tp if visible_panel=='laptop_base'else tc)[0]
            return .7*lu+.3*visible+.001*np.sum((x[:3]/.2)**2)
        fp=float((M.rotation_pose(p)[0]@normal_parent)@(-p[:3,3])/np.linalg.norm(p[:3,3]));fc=float((M.rotation_pose(c)[0]@normal_child)@(-c[:3,3])/np.linalg.norm(c[:3,3]))
        if min(fp,fc)<.025:return 10+max(0.,.025-fp)+max(0.,.025-fc)
        return .4*lu+.3*S.silhouette_score(up,tp)[0]+.3*S.silhouette_score(uc,tc)[0]+.001*np.sum((x[:3]/.2)**2)
    if closed:x=np.zeros(6)
    else:x=min([np.r_[np.zeros(6),q]for q in np.linspace(0,math.pi,POLICY['angle_grid_count'])],key=loss)
    optimized=minimize(loss,x,method='Powell',bounds=bounds,options={'maxfev':POLICY['maximum_fit_evaluations'],'maxiter':5,'xtol':.001,'ftol':.0001});x=optimized.x if optimized.fun<loss(x)else x;p,c,q=matrices(x)
    up,_=M.project(parent,p,k);uc,_=M.project(child,c,k);scores={'union':S.silhouette_score(np.r_[up,uc],tu)[1]}
    if tp is not None:scores['parent']=S.silhouette_score(up,tp)[1]
    if tc is not None:scores['child']=S.silhouette_score(uc,tc)[1]
    return {'parent':p,'child':c,'q':q,'scores':scores,'objective':float(loss(x)),'fit_evaluations':int(optimized.nfev),'source_quad':quad.tolist()}
