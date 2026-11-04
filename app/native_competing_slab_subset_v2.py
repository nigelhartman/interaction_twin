"""Numerically conservative competing-slab classifier, private candidates only.

V1 remains immutable with its already-rendered private hypotheses. Near-orthogonal
normal dot products must exceed a fixed .05 confidence margin before a face is
assigned to the competing slab; roundoff cannot arbitrarily delete side walls.
"""
import numpy as np
from native_planar_subset_v1 import require
from native_planar_surface_subset_v2 import hypotheses as surface_hypotheses
from native_competing_slab_subset_v1 import _plane_pair_interval


def hypotheses(vertices,faces):
    vertices=np.asarray(vertices,float);faces=np.asarray(faces,int)
    require(vertices.ndim==2 and vertices.shape[1]==3 and np.isfinite(vertices).all(),
            'Finite original native vertices required')
    triangles=vertices[faces]
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    lengths=np.linalg.norm(cross,axis=1);area=lengths/2
    normals=cross/np.maximum(lengths[:,None],1e-20);centers=triangles.mean(1)
    seeds=surface_hypotheses(vertices,faces);result=[]
    for desired in seeds:
        target=np.asarray(desired['selection']['axis'])
        rivals=[r for r in seeds if abs(np.dot(target,r['selection']['axis']))<.2]
        if not rivals:continue
        try:lo,hi,pair=_plane_pair_interval(centers,normals,area,target)
        except ValueError:continue
        q=vertices@target;band=np.all(((q>=lo)&(q<=hi))[faces],axis=1)
        records=[];wrong=np.zeros(len(faces),bool)
        for rival in rivals:
            other=np.asarray(rival['selection']['axis'])
            try:other_lo,other_hi,other_pair=_plane_pair_interval(centers,normals,area,other)
            except ValueError:continue
            cq=centers@other
            classified=(cq>=other_lo)&(cq<=other_hi)&(
                np.abs(normals@other)>np.abs(normals@target)+.05)
            wrong|=classified
            records.append({'axis':other.tolist(),'plane_pair':other_pair,
                            'rejected_face_count':int((band&classified).sum())})
        keep=band&~wrong
        if not records or not .15*len(faces)<=keep.sum()<=.9*len(faces):continue
        result.append({'id':desired['id'].replace('planar_surface','competing_slab'),
            'operation':'indexed_triangle_subset_compact',
            'selection':{'method':'source_planar_premise_competing_native_surface_pairs_v2',
                'seed':desired['selection'],'target_axis':target.tolist(),
                'target_plane_pair':pair,'rivals':records,'normal_assignment_margin':.05,
                'source_single_planar_rigid_object_premise_requires_visual_review':True,
                'manual_geometry_or_poses_used':False},
            'face_indices':np.flatnonzero(keep),'retained_faces':int(keep.sum()),
            'original_faces':len(faces),'geometry_displayshape_nonwatertight_possible':True,
            'physical_accuracy_verified':False})
    return result
