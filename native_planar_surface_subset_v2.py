"""Data-derived dominant surface-normal refinement of planar subset hypotheses."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
import native_planar_subset_v1 as P


def hypotheses(vertices,faces):
    vertices=np.asarray(vertices,float);faces=np.asarray(faces,int);triangles=vertices[faces]
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]);area=np.linalg.norm(cross,axis=1);normals=cross/np.maximum(area[:,None],1e-20)
    output=[]
    for original in P.hypotheses(vertices,faces):
        normal=np.asarray(original['selection']['axis']);eligible=np.abs(normals@normal)>.9
        if eligible.sum()<100:continue
        covariance=np.einsum('ni,nj,n->ij',normals[eligible],normals[eligible],area[eligible]);eigenvalues,axes=np.linalg.eigh(covariance);normal=axes[:,-1]
        if normal[np.argmax(abs(normal))]<0:normal=-normal
        q=vertices@normal;hist,edges=np.histogram(q,bins=151);density=gaussian_filter1d(hist.astype(float),1.8);peak=int(density.argmax());left=right=peak
        while left>0 and density[left]>=.3*density[peak]:left-=1
        while right<150 and density[right]>=.3*density[peak]:right+=1
        lower,upper=float(edges[left]),float(edges[right+1]);span=float(np.ptp(q));keep=np.all(((q>=lower)&(q<=upper))[faces],axis=1)
        if left==0 or right==150 or not .04<=(upper-lower)/span<=.25 or not .15*len(faces)<=keep.sum()<=.9*len(faces):continue
        index=original['selection']['axis_index_ascending_variance'];output.append({'id':f'planar_surface_{index}','operation':'indexed_triangle_subset_compact',
            'selection':{'method':'PCA_seed_area_weighted_dominant_surface_normal_projection_density','pca_seed':original['selection'],
                'normal_cosine_selection_minimum':.9,'area_weighted_normal_eigenvalues':eigenvalues.tolist(),'axis':normal.tolist(),
                'histogram_bins':151,'gaussian_sigma_bins':1.8,'density_cutoff_fraction_of_peak':.3,'slab_lower':lower,'slab_upper':upper,
                'manual_plane_coordinates_used':False},'face_indices':np.flatnonzero(keep),'retained_faces':int(keep.sum()),'original_faces':len(faces),
            'geometry_displayshape_nonwatertight_possible':True,'physical_accuracy_verified':False})
    return output
