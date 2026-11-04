"""Native-face hypotheses for a source-observed single planar object.

This is a private candidate generator, never a shape correctness gate. It removes
only original indexed triangles and needs separate source-image validation. The
competing planes are estimated from area-weighted native face normals; no object
coordinates, hand-authored vertices, faces or poses are accepted as inputs.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from native_planar_subset_v1 import require
from native_planar_surface_subset_v2 import hypotheses as surface_hypotheses


def _plane_pair_interval(centers, normals, area, axis):
    aligned = np.abs(normals @ axis) > .93
    require(aligned.sum() >= 100, 'Insufficient broad native surface evidence')
    q = centers[aligned] @ axis
    histogram, edges = np.histogram(q, bins=151, weights=area[aligned])
    density = gaussian_filter1d(histogram, 1.2)
    peaks, _ = find_peaks(density, height=.08*density.max(), distance=5)
    require(len(peaks) >= 2, 'Two distinct native broad surfaces required')
    selected = sorted(sorted(peaks, key=lambda i: -density[i])[:2])
    lower, upper = selected
    while lower > 0 and density[lower] >= .04*density[selected[0]]:
        lower -= 1
    while upper < len(density)-1 and density[upper] >= .04*density[selected[1]]:
        upper += 1
    lo, hi = float(edges[lower]), float(edges[upper+1])
    require(0 < hi-lo < .3*np.ptp(centers@axis), 'Native plane pair is not thin')
    return lo, hi, {'method':'area_weighted_parallel_face_pair_density',
        'normal_cosine_minimum':.93, 'histogram_bins':151,
        'gaussian_sigma_bins':1.2, 'selected_peak_bins':list(map(int, selected)),
        'interval':[lo, hi], 'manual_plane_coordinates_used':False}


def hypotheses(vertices, faces):
    vertices=np.asarray(vertices, float);faces=np.asarray(faces, int)
    require(vertices.ndim==2 and vertices.shape[1]==3 and np.isfinite(vertices).all(),
            'Finite original native vertices required')
    triangles=vertices[faces]
    cross=np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
    lengths=np.linalg.norm(cross, axis=1);area=lengths/2
    normals=cross/np.maximum(lengths[:,None], 1e-20);centers=triangles.mean(1)
    seeds=surface_hypotheses(vertices, faces);result=[]
    for desired in seeds:
        target=np.asarray(desired['selection']['axis'])
        rivals=[r for r in seeds if abs(np.dot(target,r['selection']['axis'])) < .2]
        if not rivals:
            continue
        try:
            lo,hi,pair=_plane_pair_interval(centers,normals,area,target)
        except ValueError:
            continue
        q=vertices@target
        band=np.all(((q>=lo)&(q<=hi))[faces],axis=1)
        rival_records=[];wrong=np.zeros(len(faces),bool)
        for rival in rivals:
            other=np.asarray(rival['selection']['axis'])
            try:
                other_lo,other_hi,other_pair=_plane_pair_interval(centers,normals,area,other)
            except ValueError:
                continue
            cq=centers@other
            # Real end walls lie at the target object's outer long-axis ends.
            # The conflicting slab's broad surfaces lie inside its own thin
            # native pair and face that pair, rather than the desired pair.
            classified=(cq>=other_lo)&(cq<=other_hi)&(
                np.abs(normals@other)>np.abs(normals@target))
            wrong|=classified
            rival_records.append({'axis':other.tolist(),'plane_pair':other_pair,
                                  'rejected_face_count':int((band&classified).sum())})
        keep=band&~wrong
        if not rival_records or not .15*len(faces)<=keep.sum()<=.9*len(faces):
            continue
        result.append({'id':desired['id'].replace('planar_surface','competing_slab'),
            'operation':'indexed_triangle_subset_compact',
            'selection':{'method':'source_planar_premise_competing_native_surface_pairs',
                'seed':desired['selection'],'target_axis':target.tolist(),
                'target_plane_pair':pair,'rivals':rival_records,
                'source_single_planar_rigid_object_premise_requires_visual_review':True,
                'manual_geometry_or_poses_used':False},
            'face_indices':np.flatnonzero(keep),'retained_faces':int(keep.sum()),
            'original_faces':len(faces),'geometry_displayshape_nonwatertight_possible':True,
            'physical_accuracy_verified':False})
    return result
