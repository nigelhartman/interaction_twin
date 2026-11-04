"""Exact source-JPEG UV appearance, preserving source-RGB sample provenance."""
import copy,json,struct
import numpy as np
from source_visible_flexure_contract_v1 import require
def exact_source_texture(raw,source_xy,source_RGB_bytes,width,height):
    size,kind=struct.unpack_from('<II',raw,12);require(kind==0x4e4f534a,'GLB JSON required');doc=json.loads(raw[20:20+size]);off=20+size;length,kind=struct.unpack_from('<II',raw,off);require(kind==0x004e4942 and off+8+length==len(raw),'One embedded BIN required');binary=raw[off+8:]
    require(source_RGB_bytes[:2]==b'\xff\xd8','Exact original JPEG required');require(len(doc['meshes'])==1 and len(doc['meshes'][0]['primitives'])==1,'One observed primitive required');p=doc['meshes'][0]['primitives'][0];attrs=p['attributes'];require('COLOR_0'in attrs and 'TEXCOORD_0'not in attrs,'Original source-RGB attribute required')
    # Keep exact sampled RGB as a provenance attribute. GLTF shader must not
    # multiply those colors into the source texture a second time.
    attrs['_SOURCE_RGB']=attrs.pop('COLOR_0');uv=(np.asarray(source_xy,np.float32)+.5)/np.array([width,height],np.float32)
    def append(data):
        nonlocal binary
        binary+=b'\0'*((-len(binary))%4);start=len(binary);binary+=data;index=len(doc['bufferViews']);doc['bufferViews'].append({'buffer':0,'byteOffset':start,'byteLength':len(data)});return index
    v=append(uv.astype('<f4').tobytes());index=len(doc['accessors']);doc['accessors'].append({'bufferView':v,'componentType':5126,'count':len(uv),'type':'VEC2','min':uv.min(0).tolist(),'max':uv.max(0).tolist()});attrs['TEXCOORD_0']=index
    image_view=append(source_RGB_bytes);doc['images']=[{'bufferView':image_view,'mimeType':'image/jpeg','name':'Exact original source JPEG; only SAM-supported surface UVs'}];doc['samplers']=[{'magFilter':9729,'minFilter':9729,'wrapS':33071,'wrapT':33071}];doc['textures']=[{'sampler':0,'source':0}]
    doc['materials']=[{'name':'Source-observed exact JPEG, unlit, unknown backs culled','pbrMetallicRoughness':{'baseColorFactor':[1,1,1,1],'baseColorTexture':{'index':0,'texCoord':0},'metallicFactor':0,'roughnessFactor':1},'doubleSided':False,'extensions':{'KHR_materials_unlit':{}}}];p['material']=0
    doc.setdefault('asset',{}).setdefault('extras',{}).update({'appearance_mode':'exact_source_JPEG_projected_UV','sampled_source_RGB_attribute':'_SOURCE_RGB','vertex_RGB_multiplied_into_texture':False})
    binary+=b'\0'*((-len(binary))%4);doc['buffers'][0]['byteLength']=len(binary);encoded=json.dumps(doc,separators=(',',':'),allow_nan=False).encode();encoded+=b' '*((-len(encoded))%4);return struct.pack('<4sII',b'glTF',2,12+8+len(encoded)+8+len(binary))+struct.pack('<II',len(encoded),0x4e4f534a)+encoded+struct.pack('<II',len(binary),0x004e4942)+binary
def visible_components(source_xy,faces,masks,component_ids):
    """An ID is visible only if an entire retained triangle has that label."""
    xy=np.asarray(source_xy,int);faces=np.asarray(faces,int);out=[]
    for cid in component_ids:
        if cid not in masks:continue
        support=np.asarray(masks[cid],bool)[xy[:,1],xy[:,0]]
        if support[faces].all(1).any():out.append(cid)
    return out
