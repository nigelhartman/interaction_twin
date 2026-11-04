"""Exact compact native triangle subsets; no re-mesh or interpolated vertices."""
from __future__ import annotations
import hashlib
import json
import struct
import numpy as np
from native_appearance import load_native_glb
from native_planar_subset_v1 import require


DTYPES={5120:np.dtype('i1'),5121:np.dtype('u1'),5122:np.dtype('<i2'),5123:np.dtype('<u2'),5125:np.dtype('<u4'),5126:np.dtype('<f4')}
DIMS={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}


def rewrite_compact_subset(raw, face_indices):
    original=load_native_glb(raw);chosen=np.asarray(face_indices)
    require(chosen.ndim==1 and np.issubdtype(chosen.dtype,np.integer) and 1<=len(chosen)<len(original.faces) and chosen.min()>=0 and chosen.max()<len(original.faces) and np.all(np.diff(chosen)>0),'Sorted unique proper source face subset required')
    jlen,jkind=struct.unpack_from('<II',raw,12);require(raw[:4]==b'glTF' and jkind==0x4e4f534a,'GLB2 JSON first required')
    document=json.loads(raw[20:20+jlen]);off=20+jlen;blen,bkind=struct.unpack_from('<II',raw,off)
    require(bkind==0x004e4942 and off+8+blen==len(raw),'One embedded GLB binary chunk required')
    require(len(document.get('nodes',[]))==1 and document['nodes'][0].get('mesh')==0 and not set(document['nodes'][0])-{'name','mesh','extras'},'One identity native mesh required')
    require(len(document.get('meshes',[]))==1 and len(document['meshes'][0]['primitives'])==1,'One native primitive required')
    primitive=document['meshes'][0]['primitives'][0];require(primitive.get('mode',4)==4 and 'indices'in primitive and not primitive.get('targets'),'Static indexed triangles required')
    require(set(primitive['attributes'])<={'POSITION','NORMAL','COLOR_0','TEXCOORD_0'},'Only exact static position/normal/colour/UV attributes supported')
    ids=np.unique(original.faces[chosen]);inverse=np.full(len(original.vertices),-1,np.int64);inverse[ids]=np.arange(len(ids));new_faces=inverse[original.faces[chosen]]
    data=bytearray(raw[off+8:]);ranges=[];proofs=[]
    def rewrite(accessor_id, values, label):
        accessor=document['accessors'][accessor_id];view=document['bufferViews'][accessor['bufferView']]
        require(not accessor.get('sparse') and accessor['componentType']in DTYPES and accessor['type']in DIMS,'Packed nonsparse attribute required')
        dt=DTYPES[accessor['componentType']];dim=DIMS[accessor['type']];stride=dt.itemsize*dim
        require(view.get('byteStride',stride)==stride,'Interleaved native attributes are not supported')
        start=view.get('byteOffset',0)+accessor.get('byteOffset',0);old_size=accessor['count']*stride;encoded=np.asarray(values,dtype=dt).tobytes()
        require(len(encoded)<=old_size and start+old_size<=len(data),'Native attribute allocation differs')
        for a,b in ranges:require(start+old_size<=a or start>=b,'Native attribute allocations overlap')
        ranges.append((start,start+old_size));data[start:start+len(encoded)]=encoded;accessor['count']=len(values)
        a=np.asarray(values).reshape(-1,dim)
        if 'min'in accessor:accessor['min']=a.min(0).tolist()
        if 'max'in accessor:accessor['max']=a.max(0).tolist()
        proofs.append({'semantic':label,'accessor':accessor_id,'component_type':accessor['componentType'],'type':accessor['type'],
            'selected_values_sha256':hashlib.sha256(encoded).hexdigest(),'values_interpolated':False})
    for semantic,accessor_id in primitive['attributes'].items():
        a=document['accessors'][accessor_id];v=document['bufferViews'][a['bufferView']];dt=DTYPES[a['componentType']];dim=DIMS[a['type']]
        require(a['count']==len(original.vertices),'Native accessor vertex count differs from decoded original')
        start=v.get('byteOffset',0)+a.get('byteOffset',0);values=np.frombuffer(raw[off+8:],dt,count=a['count']*dim,offset=start).reshape(a['count'],dim)
        rewrite(accessor_id,values[ids],semantic)
    index_accessor=document['accessors'][primitive['indices']];require(index_accessor['componentType']==5125 and index_accessor['type']=='SCALAR','Original packed uint32 triangle indices required')
    rewrite(primitive['indices'],new_faces.reshape(-1),'INDICES')
    unchanged=np.ones(len(data),bool)
    for a,b in ranges:unchanged[a:b]=False
    original_binary=raw[off+8:];require(bytes(np.asarray(data,dtype=np.uint8)[unchanged])==bytes(np.frombuffer(original_binary,np.uint8)[unchanged]),'Non-attribute/index binary image/material payload changed')
    document.setdefault('asset',{}).setdefault('extras',{})['interaction_twin_derivative']='indexed_triangle_subset_compact_display_v1'
    encoded_json=json.dumps(document,separators=(',',':'),allow_nan=False).encode();encoded_json+=b' '*((-len(encoded_json))%4)
    result=struct.pack('<4sII',b'glTF',2,12+8+len(encoded_json)+8+len(data))+struct.pack('<II',len(encoded_json),0x4e4f534a)+encoded_json+struct.pack('<II',len(data),0x004e4942)+data
    derived=load_native_glb(result)
    require(np.array_equal(derived.vertices,original.vertices[ids]) and np.array_equal(derived.faces,new_faces),'Compact native vertex/face mapping differs')
    if original.vertex_colors is not None:require(np.array_equal(derived.vertex_colors,original.vertex_colors[ids]),'Compact native colours differ')
    return result,ids,{'operation':'indexed_triangle_subset_compact','original_vertices':len(original.vertices),'derived_vertices':len(ids),
        'original_faces':len(original.faces),'derived_faces':len(chosen),'attribute_proofs':proofs,'all_selected_attribute_values_exact':True,
        'native_images_and_nonattribute_binary_preserved':True,'face_order_and_winding_preserved':True,'added_faces':0,'added_vertices':0,
        'geometry_displayshape_nonwatertight_possible':True,'manual_geometry_or_poses_used':False,'physical_accuracy_verified':False}
