"""Contour-adaptive observed-surface tessellation from cached depth priors.

No new source pixels, hidden surfaces, dimensions or poses are authored. Depth
on newly selected visible pixels is an explicitly estimated cached-prior sample.
"""
from __future__ import annotations
import math
import cv2
import numpy as np
from scipy.spatial import Delaunay,cKDTree

MAX_VERTICES=1800
def conservative_triangle_inside(xy,safe):
 """Every pixel square intersecting the triangle must remain observed.

 The triangle/square separating-axis test includes contact at an edge or
 corner. Unlike a fill-center raster, this covers partial boundary pixels.
 """
 p=np.asarray(xy,np.int32);lo=p.min(0);hi=p.max(0)
 if(lo<0).any()or hi[0]>=safe.shape[1]or hi[1]>=safe.shape[0]:return False
 region=safe[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
 if region.all():return True
 y,x=np.nonzero(~region);centres=np.c_[x+lo[0],y+lo[1]].astype(float)
 possible=np.ones(len(centres),bool)
 for a,b in zip(p,np.roll(p,-1,axis=0)):
  normal=np.array([-(b-a)[1],(b-a)[0]],float);proj=p@normal;c=centres@normal;r=.5*np.abs(normal).sum();possible&=(c+r>=proj.min())&(c-r<=proj.max())
 return not bool(possible.any())
def source_points(safe,maximum=MAX_VERTICES):
 contours,_=cv2.findContours(safe.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
 contours=[c for c in contours if len(c)>=3 and abs(cv2.contourArea(c))>=2]
 if not contours:raise ValueError('No source contour with triangle support')
 epsilon=0.
 for candidate in [0.,.15,.3,.45,.6,.8,1.,1.3,1.7,2.]:
  pieces=[c if candidate==0 else cv2.approxPolyDP(c,candidate,True)for c in contours];boundary=np.unique(np.concatenate([c[:,0]for c in pieces if len(c)>=3]),axis=0)
  if len(boundary)<maximum*.72:epsilon=candidate;break
 if len(boundary)>=maximum:raise ValueError('Source contour exceeds fixed vertex budget')
 count=int(safe.sum());distance=cv2.distanceTransform(safe.astype(np.uint8),cv2.DIST_L2,3);budget=maximum-len(boundary);stride=max(2,math.ceil(math.sqrt(count/max(1,budget))))
 while True:
  yy,xx=np.mgrid[0:safe.shape[0]:stride,0:safe.shape[1]:stride];good=(safe[yy,xx]>0)&(distance[yy,xx]>=max(2,stride*.35));interior=np.c_[xx[good],yy[good]];points=np.unique(np.concatenate([boundary,interior]),axis=0)
  if len(points)<=maximum:break
  stride+=1
 return points.astype(np.int32),{'contour_approximation_epsilon_px':epsilon,'contour_vertices':len(boundary),'interior_grid_stride_px':stride,'source_points_before_triangle_pruning':len(points)}
def cached_depth(xy,parent_xy,parent_depth,parent_stride,safe=None):
 parent_xy=np.asarray(parent_xy,float);parent_depth=np.asarray(parent_depth,float);query=np.asarray(xy,float);tree=cKDTree(parent_xy);dist,nearest=tree.query(query);ids=np.repeat(nearest[:,None],3,axis=1);weights=np.zeros((len(query),3),float);weights[:,0]=1.;modes=np.ones(len(query),np.uint8)
 limit=float(math.ceil(math.sqrt(2)*(parent_stride+2)));valid=dist<=limit
 if safe is not None:
  for n in np.flatnonzero(valid):
   valid[n]=conservative_triangle_inside(np.asarray([query[n],parent_xy[nearest[n]],parent_xy[nearest[n]]]),safe)
 triangulation=Delaunay(parent_xy);simplex=triangulation.find_simplex(query);inside=simplex>=0
 for n in np.flatnonzero(inside):
  s=simplex[n];chosen=triangulation.simplices[s];p=parent_xy[chosen];edge=max(np.linalg.norm(p[a]-p[b])for a,b in [(0,1),(1,2),(2,0)])
  if edge>parent_stride*2.5 or(safe is not None and not conservative_triangle_inside(p,safe)):continue
  transform=triangulation.transform[s];w=transform[:2]@(query[n]-transform[2]);w=np.r_[w,1-w.sum()]
  if w.min()<-1e-8:continue
  ids[n]=chosen;weights[n]=w;modes[n]=0;valid[n]=True
 z=np.einsum('ij,ij->i',parent_depth[ids],weights).astype(np.float32);valid&=np.isfinite(z)&(z>0)
 return z,ids.astype(np.uint16),weights,modes,valid,{'maximum_nearest_sampling_distance_px':limit,'maximum_interpolation_parent_edge_px':float(parent_stride*2.5),'linear_samples':int((modes==0).sum()),'nearest_samples':int((modes==1).sum())}
def build(rgb,safe,K,parent_xy,parent_depth,parent_stride):
 # One extra pixel for vertex placement lets the conservative pixel-square
 # proof retain smooth diagonals without sampling outside the original core.
 points_mask=cv2.erode(safe.astype(np.uint8),np.ones((3,3),np.uint8))>0
 xy,selection=source_points(points_mask);z,ids,weights,modes,valid,depth_policy=cached_depth(xy,parent_xy,parent_depth,parent_stride,safe);xy,z,ids,weights,modes=xy[valid],z[valid],ids[valid],weights[valid],modes[valid];faces=Delaunay(xy).simplices
 keep=[]
 for face in faces:
  p=xy[face];a=p[1]-p[0];b=p[2]-p[0];area=abs(int(a[0])*int(b[1])-int(a[1])*int(b[0]));depths=z[face]
  keep.append(area>=1 and np.ptp(depths)/depths.mean()<.06 and conservative_triangle_inside(p,safe))
 faces=faces[np.asarray(keep)];used=np.unique(faces);remap=np.full(len(xy),-1,np.int32);remap[used]=np.arange(len(used));faces=remap[faces];xy,z,ids,weights,modes=xy[used],z[used],ids[used],weights[used],modes[used];K=np.asarray(K,float);vertices=np.c_[(xy[:,0]-K[0,2])/K[0,0]*z,(xy[:,1]-K[1,2])/K[1,1]*z,z].astype(np.float32)
 tri=vertices[faces].astype(float);normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);reverse=np.einsum('ij,ij->i',normal,tri.mean(1))>=0;faces[reverse]=faces[reverse,::-1]
 return {'vertices':vertices,'faces':faces,'colors':rgb[xy[:,1],xy[:,0]],'source_xy':xy.astype(np.uint16),'depth':z,'parent_depth_vertex_ids':ids,'parent_depth_weights':weights,'depth_sampling_modes':modes,'policy':{**selection,**depth_policy,'fixed_maximum_vertices':MAX_VERTICES,'triangles_rejected':int((~np.asarray(keep)).sum()),'source_footprint_proof_version':2,'source_footprint_rule':'all_unit_pixel_squares_intersecting_closed_triangle_inside_original_2px_SAM_core','vertex_placement_extra_inset_px':1,'depth_derivation':'cached_monocular_prior_barycentric_or_bounded_nearest_not_measured_depth'}}
