"""Fast full-triangle silhouette PROXY for candidate search only.

OpenCV boundary coverage differs from the native pixel-centre rasterizer. The
original transformed triangles, winding/doubleSided semantics and near clipping
are retained. Native exact rendering must independently rerank/check finalists.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from alignment_render import _clip_near, _inputs

POLICY={'algorithm':'original_triangle_opencv_search_proxy_v2','fixed_point_shift':8,
        'near_plane':1e-8,'geometry_decimation_or_hull_used':False,'native_geometry_changed':False,
        'final_scoring_uses_proxy':False,'pixel_coverage_equivalent_to_native':False,
        'opaque_silhouette_only':True,'backface_policy':'native primitive doubleSided/winding; historical legacy two-sided',
        'near_clip_policy':'clip original triangle to Z>=near, then viewport; retain resulting convex polygon'}


def viewport_clip(polygon,width,height):
    polygon=np.asarray(polygon,float)
    for axis,bound,positive in [(0,-1.,True),(0,float(width),False),(1,-1.,True),(1,float(height),False)]:
        if not len(polygon):break
        result=[]
        for i,b in enumerate(polygon):
            a=polygon[i-1];inside_a=a[axis]>=bound if positive else a[axis]<=bound
            inside_b=b[axis]>=bound if positive else b[axis]<=bound
            if inside_a!=inside_b:
                result.append(a+(bound-a[axis])/(b[axis]-a[axis])*(b-a))
            if inside_b:result.append(b)
        polygon=np.asarray(result,float).reshape(-1,2)
    return polygon


def silhouette(mesh,rotation,translation,k,size,origin_offset=None):
    width,height=size
    if not isinstance(width,int) or not isinstance(height,int) or width<=0 or height<=0 or width*height>16_777_216:raise ValueError('Bounded positive mask dimensions required')
    translation=np.asarray(translation,float)
    if origin_offset is not None:translation=translation-np.asarray(rotation)@np.asarray(origin_offset)
    vertices,faces,k,rotation,translation,_=_inputs(mesh.vertices,mesh.faces,k,rotation,translation,1.)
    camera=vertices@rotation.T+translation;triangles=camera[faces]
    eligible=np.ones(len(faces),bool)
    if not mesh.appearance.legacy_vertex_compatibility:
        eligible[:]=False
        for part in mesh.appearance.primitives:
            indices=np.arange(part.face_offset,part.face_offset+len(part.faces))
            tri=triangles[indices]
            if not part.double_sided:
                normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
                indices=indices[part.winding_sign*np.einsum('ij,ij->i',normals,tri[:,0])<0]
            eligible[indices]=True
    near=POLICY['near_plane'];eligible&=triangles[:,:,2].max(1)>=near
    target=np.zeros((height,width),np.uint8);shift=POLICY['fixed_point_shift'];factor=1<<shift
    selected=triangles[eligible]
    wholly_in_front=(selected[:,:,2]>=near).all(1)
    positive=selected[wholly_in_front]
    q=positive@k.T;uv=q[:,:,:2]/q[:,:,2:]
    finite=np.isfinite(uv).all((1,2))
    wholly_inside=finite&(uv[:,:,0]>=-1).all(1)&(uv[:,:,0]<=width).all(1)&(uv[:,:,1]>=-1).all(1)&(uv[:,:,1]<=height).all(1)
    # Same unchanged per-triangle convex union; batch projection/bounds avoid
    # allocating small NumPy arrays thousands of times per candidate pose.
    polygons=np.rint(uv[wholly_inside]*factor).astype(np.int32)
    for polygon in polygons:cv2.fillConvexPoly(target,polygon,1,lineType=cv2.LINE_8,shift=shift)
    slow=np.concatenate([selected[~wholly_in_front],positive[~wholly_inside]])
    for tri in slow:
        if (tri[:,2]<near).any():tri,_=_clip_near(tri,np.zeros((3,3)),near)
        if len(tri)<3:continue
        q=tri@k.T;uv=q[:,:2]/q[:,2:]
        if not np.isfinite(uv).all():continue
        if uv[:,0].max()<-1 or uv[:,0].min()>width or uv[:,1].max()<-1 or uv[:,1].min()>height:continue
        if (uv[:,0]<-1).any() or (uv[:,0]>width).any() or (uv[:,1]<-1).any() or (uv[:,1]>height).any():uv=viewport_clip(uv,width,height)
        if len(uv)<3:continue
        cv2.fillConvexPoly(target,np.rint(uv*factor).astype(np.int32),1,lineType=cv2.LINE_8,shift=shift)
    return target.astype(bool)


def iou(a,b):return float(np.count_nonzero(a&b)/max(1,np.count_nonzero(a|b)))


def refine(mesh,origin_offset,k,target,rotation,translation,max_evaluations=160):
    """Same bounded Powell parameterization; only search raster is replaced."""
    height,width=target.shape
    observed_boundary=cv2.morphologyEx(target.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
    observed_distance=cv2.distanceTransform(1-observed_boundary,cv2.DIST_L2,3)
    q0=np.r_[np.zeros(3),translation[:2]/translation[2],np.log(translation[2])]
    limits=np.array([.6,.6,.6,.12,.12,.35]);best={'value':np.inf,'q':q0.copy()};calls=0
    def unpack(q):
        r=Rotation.from_rotvec(q[:3]).as_matrix()@rotation;z=np.exp(q[5]);return r,np.r_[q[3:5]*z,z]
    def loss(q):
        nonlocal calls
        calls+=1;r,t=unpack(q);rendered=silhouette(mesh,r,t,k,(width,height),origin_offset)
        if not rendered.any():return 5.
        boundary=cv2.morphologyEx(rendered.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
        distance=cv2.distanceTransform(1-boundary.astype(np.uint8),cv2.DIST_L2,3)
        chamfer=.5*(observed_distance[boundary].mean()+distance[observed_boundary>0].mean())
        value=1-iou(rendered,target)+.25*chamfer/np.hypot(width,height)
        if value<best['value']:best.update(value=float(value),q=q.copy())
        return float(value)
    result=minimize(loss,q0,method='Powell',bounds=np.c_[q0-limits,q0+limits],
        options={'maxfev':max_evaluations,'maxiter':12,'xtol':.001,'ftol':.0001})
    r,t=unpack(best['q'])
    return {'rotation':r,'translation':t,'loss':best['value'],'evaluations':calls,
        'optimizer_success':bool(result.success),'budget_exhausted':bool(result.nfev>=max_evaluations),
        'search_raster':POLICY['algorithm'],'search_proxy_iou':iou(silhouette(mesh,r,t,k,(width,height),origin_offset),target)}
