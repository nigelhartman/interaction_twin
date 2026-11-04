"""Bounded visible two-panel source fit through an existing fixed native hinge."""
from __future__ import annotations
import math
import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

import occluded_hinge_recovery_v1 as R
import sam3d_common_scene as S
import sam3d_mechanisms as M

POLICY={
    'angle_grid_count':181,
    'maximum_opening_search_rad':math.pi,
    'maximum_parent_rotation_component_rad':math.radians(15),
    'maximum_parent_translation_extent_fraction':.15,
    'maximum_parent_depth_fraction':.20,
    'maximum_fit_evaluations':1600,
    'sparse_vertices':1400,
    'minimum_full_triangle_component_iou':.60,
    'minimum_full_triangle_union_iou':.72,
    'minimum_source_foreground_coverage':.75,
    'minimum_interior_facing_cosine':.05,
    'maximum_projected_hinge_seam_distance_px':32.,
    'new_api_calls':0,'new_gpu_calls':0,'physical_accuracy_verified':False,
}


def fit(parent_vertices,child_vertices,parent_mask,child_mask,camera,joint,prior_parent,parent_interior_normal,child_interior_normal,seam_pixels):
    pv=np.asarray(parent_vertices,float);cv=np.asarray(child_vertices,float);k=np.asarray(camera['K'],float)
    initial=R.panel_pose(pv,parent_mask,camera,joint['parent_native_scale'],prior_parent,parent_interior_normal,visible_face='interior')
    original=np.asarray(initial['camera_from_native_mesh']);rp,tp,sp=M.rotation_pose(original)
    p=pv[np.linspace(0,len(pv)-1,min(len(pv),POLICY['sparse_vertices']),dtype=int)]
    c=cv[np.linspace(0,len(cv)-1,min(len(cv),POLICY['sparse_vertices']),dtype=int)]
    targetp,targetc,targetu=S.mask_shape(parent_mask),S.mask_shape(child_mask),S.mask_shape(parent_mask|child_mask)
    a,b=np.asarray(joint['parent_from_joint']),np.asarray(joint['joint_from_child_zero']);axis=joint['axis']
    def matrices(x):
        parent=original.copy();parent[:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix()@original[:3,:3];parent[:3,3]+=x[3:6]
        child=M.forward(parent,a,axis,float(x[6]),b);return parent,child
    def facing(matrix,normal):
        rotation,translation,_=M.rotation_pose(matrix)
        return float((rotation@normal)@(-translation)/np.linalg.norm(translation))
    def seam_distance(parent):
        pixel,valid=M.project(np.asarray([joint['parent_anchor_native']]),parent,k)
        return float(M.line_distance(pixel,np.asarray(seam_pixels))[0][0])if valid[0]else float('inf')
    def loss(x):
        parent,child=matrices(x);pu,vp=M.project(p,parent,k);cu,vc=M.project(c,child,k)
        if not vp.all()or not vc.all():return 1e5
        lp=S.silhouette_score(pu,targetp)[0];lc=S.silhouette_score(cu,targetc)[0];lu=S.silhouette_score(np.r_[pu,cu],targetu)[0]
        fp,fc=facing(parent,parent_interior_normal),facing(child,child_interior_normal)
        if min(fp,fc)<POLICY['minimum_interior_facing_cosine']:return 10.+max(0.,.05-fp)+max(0.,.05-fc)
        seam=seam_distance(parent)/POLICY['maximum_projected_hinge_seam_distance_px']
        regularizer=.002*np.sum(np.square(x[:3]/POLICY['maximum_parent_rotation_component_rad']))
        return float(.25*lp+.4*lc+.35*lu+.06*min(seam,3.)+regularizer)
    grid=np.linspace(0.,POLICY['maximum_opening_search_rad'],POLICY['angle_grid_count'])
    starts=[np.r_[np.zeros(6),angle]for angle in grid];x=min(starts,key=loss)
    native_per_pixel=original[2,3]/k[0,0];dx=targetu['extent'][0]*POLICY['maximum_parent_translation_extent_fraction']*native_per_pixel;dy=targetu['extent'][1]*POLICY['maximum_parent_translation_extent_fraction']*native_per_pixel;dz=original[2,3]*POLICY['maximum_parent_depth_fraction']
    bounds=[(-POLICY['maximum_parent_rotation_component_rad'],POLICY['maximum_parent_rotation_component_rad'])]*3+[(-dx,dx),(-dy,dy),(-dz,dz),(0.,POLICY['maximum_opening_search_rad'])]
    optimized=minimize(loss,x,method='Powell',bounds=bounds,options={'maxfev':POLICY['maximum_fit_evaluations'],'maxiter':8,'xtol':.001,'ftol':.0001})
    chosen=optimized.x if optimized.fun<loss(x)else x;parent,child=matrices(chosen)
    return {'parent_matrix':parent.tolist(),'child_matrix':child.tolist(),'q_rad':float(chosen[6]),'objective':float(loss(chosen)),
            'initial_parent_PnP':initial,'parent_delta':chosen[:6].tolist(),'bounds':[list(b)for b in bounds],
            'evaluations':int(optimized.nfev),'interior_facing_cosine':{'parent':facing(parent,parent_interior_normal),'child':facing(child,child_interior_normal)},
            'projected_hinge_seam_distance_px':seam_distance(parent),'manual_geometry_or_poses_used':False,'physical_accuracy_verified':False}


def gate(metrics,fit):
    return bool(all(metrics[c]['mask_iou']>=POLICY['minimum_full_triangle_component_iou']and metrics[c]['source_foreground_coverage']>=POLICY['minimum_source_foreground_coverage']for c in ['parent','child'])
                and metrics['union']['mask_iou']>=POLICY['minimum_full_triangle_union_iou']
                and min(fit['interior_facing_cosine'].values())>=POLICY['minimum_interior_facing_cosine']
                and fit['projected_hinge_seam_distance_px']<=POLICY['maximum_projected_hinge_seam_distance_px'])
