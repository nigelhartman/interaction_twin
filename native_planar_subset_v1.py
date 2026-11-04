"""Automatic planar consensus hypotheses and exact native triangle subsets.

Candidate generation is geometric only. A candidate is never accepted merely
because it removes a protrusion: source views, appearance and a separate review
must support it. Original vertices/colours/UVs/materials and bytes are preserved.
Removing faces can open boundaries; these are declared display derivatives.
"""
from __future__ import annotations
import copy
import hashlib
import json
import struct

import numpy as np
from scipy.ndimage import gaussian_filter1d
from native_appearance import load_native_glb


def require(value, message):
    if not value: raise ValueError(message)


def hypotheses(vertices, faces):
    vertices=np.asarray(vertices,float);faces=np.asarray(faces,int)
    require(vertices.ndim==2 and vertices.shape[1]==3 and len(vertices)>=100 and np.isfinite(vertices).all(), 'Finite native geometry required')
    require(faces.ndim==2 and faces.shape[1]==3 and len(faces)>=100 and faces.min()>=0 and faces.max()<len(vertices), 'Original indexed triangles required')
    eigenvalues, axes=np.linalg.eigh(np.cov(vertices.T)); result=[]
    for index in range(3):
        normal=axes[:,index]
        if normal[np.argmax(abs(normal))]<0: normal=-normal
        q=vertices@normal;hist,edges=np.histogram(q,bins=151);density=gaussian_filter1d(hist.astype(float),1.8)
        peak=int(np.argmax(density));threshold=.30*density[peak]
        left=peak
        while left>0 and density[left]>=threshold:left-=1
        right=peak
        while right<len(density)-1 and density[right]>=threshold:right+=1
        lower,upper=float(edges[left]),float(edges[right+1]);span=float(np.ptp(q))
        # Exclude an edge/surface peak and broad solids; this operation is only
        # a central thin-slab hypothesis for a visibly planar rigid object.
        if left==0 or right==len(density)-1 or not .06<= (upper-lower)/span <=.35:continue
        keep_vertices=(q>=lower)&(q<=upper);keep_faces=np.all(keep_vertices[faces],axis=1)
        retained=int(keep_faces.sum())
        if not .15*len(faces)<=retained<=.9*len(faces):continue
        face_indices=np.flatnonzero(keep_faces)
        result.append({'id':f'planar_consensus_{index}', 'operation':'indexed_triangle_subset',
            'selection':{'method':'native_PCA_axis_projection_density_connected_peak', 'axis_index_ascending_variance':index,
                'axis':normal.tolist(),'eigenvalue':float(eigenvalues[index]),'histogram_bins':151,'gaussian_sigma_bins':1.8,
                'density_cutoff_fraction_of_peak':.30,'slab_lower':lower,'slab_upper':upper,'slab_fraction_of_native_extent':(upper-lower)/span,
                'manual_plane_coordinates_used':False},'face_indices':face_indices,'retained_faces':retained,'original_faces':len(faces),
            'geometry_displayshape_nonwatertight_possible':True, 'physical_accuracy_verified':False})
    return result


def rewrite_triangle_subset(raw, face_indices):
    """Replace only index prefix while retaining every other binary byte.

    Currently supports one identity-node/triangle primitive. Every selected face
    remains in native order, with its original vertex/colour/UV/material mapping.
    """
    original=load_native_glb(raw);chosen=np.asarray(face_indices)
    require(chosen.ndim==1 and np.issubdtype(chosen.dtype,np.integer) and 1<=len(chosen)<len(original.faces), 'Nonempty proper original face subset required')
    require(chosen.min()>=0 and chosen.max()<len(original.faces) and np.all(np.diff(chosen)>0), 'Sorted unique original face IDs required')
    require(raw[:4]==b'glTF' and struct.unpack_from('<I',raw,4)[0]==2, 'GLB2 required')
    jlen,jkind=struct.unpack_from('<II',raw,12);require(jkind==0x4e4f534a,'GLB JSON first required');document=json.loads(raw[20:20+jlen])
    off=20+jlen;blen,bkind=struct.unpack_from('<II',raw,off);require(bkind==0x004e4942 and off+8+blen==len(raw),'Exactly one embedded binary chunk required')
    require(len(document.get('nodes',[]))==1 and document['nodes'][0].get('mesh')==0 and not set(document['nodes'][0])-{'name','mesh','extras'},'One identity native mesh required')
    require(len(document.get('meshes',[]))==1 and len(document['meshes'][0]['primitives'])==1,'One native primitive required')
    primitive=document['meshes'][0]['primitives'][0];require(primitive.get('mode',4)==4 and 'indices'in primitive,'Indexed triangles required')
    # Native decoded indices are verified after writing; scene transformations
    # or primitive splitting may not silently renumber the selected vertices.
    accessor=document['accessors'][primitive['indices']];view=document['bufferViews'][accessor['bufferView']]
    require(accessor['type']=='SCALAR' and accessor['componentType']==5125 and not accessor.get('sparse') and not view.get('byteStride'), 'Packed original uint32 indices required')
    indices=np.asarray(original.faces[chosen],dtype='<u4').reshape(-1);data=bytearray(raw[off+8:]);prefix=bytes(data)
    start=view.get('byteOffset',0)+accessor.get('byteOffset',0);old_size=accessor['count']*4;encoded=indices.tobytes()
    require(len(encoded)<=old_size and start+old_size<=len(data),'Original index allocation required')
    for other in document['bufferViews']:
        if other is view:continue
        a,b=other.get('byteOffset',0),other.get('byteOffset',0)+other['byteLength']
        require(b<=start or a>=start+old_size,'Index buffer overlaps another native attribute/image view')
    data[start:start+len(encoded)]=encoded
    accessor.update(count=len(indices),min=[int(indices.min())],max=[int(indices.max())])
    document.setdefault('asset',{}).setdefault('extras',{})['interaction_twin_derivative']='indexed_triangle_subset_display_v1'
    encoded_json=json.dumps(document,separators=(',',':'),allow_nan=False).encode();encoded_json+=b' '*((-len(encoded_json))%4)
    result=struct.pack('<4sII',b'glTF',2,12+8+len(encoded_json)+8+len(data))+struct.pack('<II',len(encoded_json),0x4e4f534a)+encoded_json+struct.pack('<II',len(data),0x004e4942)+data
    decoded=load_native_glb(result)
    require(np.array_equal(decoded.vertices,original.vertices) and np.array_equal(decoded.faces,original.faces[chosen]),'Derived native vertex/face identity changed')
    if original.vertex_colors is not None:require(np.array_equal(decoded.vertex_colors,original.vertex_colors),'Native vertex colours changed')
    nonindex_before=prefix[:start]+prefix[start+old_size:];nonindex_after=bytes(data[:start])+bytes(data[start+old_size:])
    require(nonindex_before==nonindex_after,'Native non-index binary bytes changed')
    return result,{'operation':'indexed_triangle_subset','original_face_count':len(original.faces),'derived_face_count':len(chosen),
        'original_nonindex_binary_sha256':hashlib.sha256(nonindex_before).hexdigest(),'derived_nonindex_binary_sha256':hashlib.sha256(nonindex_after).hexdigest(),
        'nonindex_binary_bytes_preserved':True,'all_native_vertices_preserved':True,'original_vertex_colours_preserved':True,
        'original_uv_material_image_bytes_preserved':True,'added_faces':0,'added_vertices':0,
        'geometry_displayshape_nonwatertight_possible':True,'physical_accuracy_verified':False}
