"""Evidence-limited UV appearance from real source RGB, with native fallback.

Only a native texel's unambiguous, source-visible mesh point can receive the
exact RGB of a foreground source pixel. Geometry/UVs/other native texels stay
unchanged. No inpainting, generated imagery, background or hidden-face painting.
"""
from __future__ import annotations
import copy,hashlib,io,json,struct
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from native_appearance import load_native_glb,render_native_mask
from alignment_render import render_mesh
import sam3d_mechanisms as M
import native_assembly_refinement_v1 as N

ROOT=Path(__file__).resolve().parent
FORMAT='interaction-twin-observed-surface-texture-projection'
POLICY={'maximum_texture_pixels':1048576,'source_mask_erosion_px':3,'maximum_relative_source_depth_error':.002,
        'minimum_painted_texels':1000,'maximum_changed_material_images':1,'source_color_sampling':'nearest_original_RGB_pixel_no_colour_synthesis',
        'UV_overlap_policy':'first_and_last_original_face_raster_ownership_must_agree','hidden_surface_policy':'native_texture_fallback',
        'physical_accuracy_verified':False}


def require(value,message):
    if not value:raise ValueError(message)


def unpack(raw):
    require(raw[:4]==b'glTF'and struct.unpack_from('<I',raw,4)[0]==2,'Embedded GLB2 required')
    n,kind=struct.unpack_from('<II',raw,12);require(kind==0x4e4f534a,'GLB JSON first required');doc=json.loads(raw[20:20+n]);off=20+n;length,kind=struct.unpack_from('<II',raw,off);require(kind==0x004e4942 and off+8+length==len(raw),'One embedded BIN required');return doc,raw[off+8:]


def texture(raw):
    doc,binary=unpack(raw);require(len(doc['meshes'])==len(doc['meshes'][0]['primitives'])==1,'Single native primitive projection supported')
    primitive=doc['meshes'][0]['primitives'][0];material=doc['materials'][primitive.get('material',0)];index=material['pbrMetallicRoughness']['baseColorTexture']['index'];image_index=doc['textures'][index]['source'];image=doc['images'][image_index];view=doc['bufferViews'][image['bufferView']];start=view.get('byteOffset',0);pixels=np.asarray(Image.open(io.BytesIO(binary[start:start+view['byteLength']])).convert('RGB')).copy();require(pixels.shape[0]*pixels.shape[1]<=POLICY['maximum_texture_pixels'],'Native texture exceeds bounded projection raster');return pixels,image_index


def rewrite_texture(raw,pixels,image_index):
    doc,binary=unpack(raw);buffer=io.BytesIO();Image.fromarray(pixels).save(buffer,format='PNG');png=buffer.getvalue();body=binary+b'\0'*((-len(binary))%4);offset=len(body);body+=png;body+=b'\0'*((-len(body))%4)
    view=len(doc['bufferViews']);doc['bufferViews'].append({'buffer':0,'byteOffset':offset,'byteLength':len(png)});doc['images'][image_index]={'bufferView':view,'mimeType':'image/png'};doc['buffers'][0]['byteLength']=len(body)
    doc.setdefault('asset',{}).setdefault('extras',{})['interaction_twin_observed_texture']='source_visible_UV_texels_with_native_fallback_v1'
    encoded=json.dumps(doc,separators=(',',':'),allow_nan=False).encode();encoded+=b' '*((-len(encoded))%4);total=12+8+len(encoded)+8+len(body)
    return struct.pack('<4sII',b'glTF',2,total)+struct.pack('<II',len(encoded),0x4e4f534a)+encoded+struct.pack('<II',len(body),0x004e4942)+body


def atlas_owners(mesh,width,height):
    require(len(mesh.appearance.primitives)==1,'One UV primitive required');uv=mesh.appearance.primitives[0].uv;require(uv is not None and len(uv)==len(mesh.vertices),'Original per-vertex UVs required')
    points=np.c_[uv*np.array([width,height])-.5,np.ones(len(uv))];colors=np.zeros((len(uv),3),np.uint8)
    first=render_mesh(points,mesh.faces,np.eye(3),np.eye(3),np.zeros(3),width,height,vertex_colors=colors)['face_index']
    last=render_mesh(points,mesh.faces[::-1],np.eye(3),np.eye(3),np.zeros(3),width,height,vertex_colors=colors)['face_index'];last=np.where(last>=0,len(mesh.faces)-1-last,-1)
    return first,(first>=0)&(first==last)


def surface_points(mesh,width,height,x,y,face_ids):
    uv=mesh.appearance.primitives[0].uv*np.array([width,height])-.5;tri=uv[mesh.faces[face_ids]];a,b,c=tri[:,0],tri[:,1],tri[:,2];den=(b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1]);require(np.all(abs(den)>1e-12),'Degenerate recorded UV triangles')
    w0=((b[:,1]-c[:,1])*(x-c[:,0])+(c[:,0]-b[:,0])*(y-c[:,1]))/den;w1=((c[:,1]-a[:,1])*(x-c[:,0])+(a[:,0]-c[:,0])*(y-c[:,1]))/den;bary=np.c_[w0,w1,1-w0-w1]
    require(bary.min()>=-1e-8,'Projection proof texel lies outside its original UV face');points=np.einsum('ni,nij->nj',bary,mesh.vertices[mesh.faces[face_ids]]);return points


def project_candidates(mesh,texels,owners,valid,source_mask,camera,matrix):
    th,tw=texels.shape[:2];h,w=source_mask.shape;rotation,translation,scale=M.rotation_pose(matrix);source_depth=render_native_mask(mesh,np.asarray(camera['K']),rotation,translation,w,h,scale=scale)['depth'];foreground=cv2.erode(source_mask.astype(np.uint8),np.ones((POLICY['source_mask_erosion_px'],)*2,np.uint8))>0
    yy,xx=np.where(valid);collected=[]
    for start in range(0,len(xx),65536):
        x,y=xx[start:start+65536],yy[start:start+65536];ids=owners[y,x];points=surface_points(mesh,tw,th,x,y,ids);cam=points@np.asarray(matrix)[:3,:3].T+np.asarray(matrix)[:3,3];uvw=cam@np.asarray(camera['K']).T;uv=uvw[:,:2]/uvw[:,2:];xy=np.rint(uv).astype(int)
        inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)&(cam[:,2]>1e-5)
        good=np.flatnonzero(inside);good=good[foreground[xy[good,1],xy[good,0]]]
        z=source_depth[xy[good,1],xy[good,0]];good=good[np.isfinite(z)&(abs(cam[good,2]-z)<=POLICY['maximum_relative_source_depth_error']*cam[good,2])]
        if len(good):collected.append(np.c_[x[good],y[good],ids[good],xy[good]])
    require(collected,'No source-visible, unambiguous foreground UV texels');return np.concatenate(collected).astype(np.int32)


def rewrite_vertex_colors(raw,colors):
    doc,binary=unpack(raw);primitive=doc['meshes'][0]['primitives'][0];accessor=doc['accessors'][primitive['attributes']['COLOR_0']];view=doc['bufferViews'][accessor['bufferView']]
    require(accessor['componentType']==5121 and accessor['type']in {'VEC3','VEC4'}and accessor.get('normalized')is True and not accessor.get('sparse'),'Packed byte native vertex colors required')
    channels=3 if accessor['type']=='VEC3'else 4;require(view.get('byteStride',channels)==channels and accessor['count']==len(colors),'Exact packed native color count required')
    start=view.get('byteOffset',0)+accessor.get('byteOffset',0);length=accessor['count']*channels;body=bytearray(binary);array=np.frombuffer(bytes(body[start:start+length]),np.uint8).reshape(-1,channels).copy();array[:,:3]=colors;body[start:start+length]=array.tobytes()
    encoded=json.dumps(doc,separators=(',',':'),allow_nan=False).encode();encoded+=b' '*((-len(encoded))%4);total=12+8+len(encoded)+8+len(body)
    return struct.pack('<4sII',b'glTF',2,total)+struct.pack('<II',len(encoded),0x4e4f534a)+encoded+struct.pack('<II',len(body),0x004e4942)+body


def vertex_mapping(mesh,mask,camera,matrix):
    h,w=mask.shape;r,t,scale=M.rotation_pose(matrix);depth=render_native_mask(mesh,np.asarray(camera['K']),r,t,w,h,scale=scale)['depth'];eroded=cv2.erode(mask.astype(np.uint8),np.ones((POLICY['source_mask_erosion_px'],)*2,np.uint8))>0
    cam=mesh.vertices@np.asarray(matrix)[:3,:3].T+np.asarray(matrix)[:3,3];pixel=cam@np.asarray(camera['K']).T;pixel=np.rint(pixel[:,:2]/pixel[:,2:]).astype(int);inside=(pixel[:,0]>=0)&(pixel[:,0]<w)&(pixel[:,1]>=0)&(pixel[:,1]<h)&(cam[:,2]>1e-5);ids=np.flatnonzero(inside);xy=pixel[ids];ids=ids[eroded[xy[:,1],xy[:,0]]];xy=pixel[ids];z=depth[xy[:,1],xy[:,0]];ids=ids[np.isfinite(z)&(abs(cam[ids,2]-z)<=POLICY['maximum_relative_source_depth_error']*cam[ids,2])]
    return np.c_[ids,pixel[ids]].astype(np.int32)


def _bake_vertices(asset_ref,source_rgb_ref,source_mask_ref,camera,matrix,output,*,root):
    out=root/output;raw=N.checked(asset_ref,root);mesh=load_native_glb(raw);require(mesh.vertex_colors is not None and mesh.receipt['legacy_vertex_compatibility'],'Exact native byte RGB needed')
    rgb=np.asarray(Image.open(io.BytesIO(N.checked(source_rgb_ref,root))).convert('RGB'));mask=np.asarray(Image.open(io.BytesIO(N.checked(source_mask_ref,root))))>0;require(rgb.shape[:2]==mask.shape==(camera['height'],camera['width']),'Original source-camera pixels required')
    mapping=vertex_mapping(mesh,mask,camera,matrix);require(len(mapping)>=POLICY['minimum_painted_texels'],'Insufficient source-visible native vertex colors');colors=mesh.vertex_colors.copy();ids,x,y=mapping.T;colors[ids]=rgb[y,x];changed=rewrite_vertex_colors(raw,colors)
    out.mkdir(parents=True);asset=out/'asset.glb';asset.write_bytes(changed);np.savez_compressed(out/'projection-map.npz',mapping=mapping)
    report={'format':FORMAT,'version':1,'appearance_mode':'vertex_rgb','input_asset_ref':asset_ref,'output_asset_ref':N.ref(asset,root),'source_rgb_ref':source_rgb_ref,'source_mask_ref':source_mask_ref,
        'source_camera':camera,'camera_from_native_mesh':np.asarray(matrix).tolist(),'projection_map_ref':N.ref(out/'projection-map.npz',root),'policy':POLICY,'painted_vertex_count':len(mapping),'native_fallback_vertex_count':len(colors)-len(mapping),
        'native_geometry_and_UV_unchanged':True,'source_RGB_pixels_copied_without_synthesis':True,'manual_pixel_editing_used':False,'new_api_calls':0,'new_gpu_calls':0,'physical_accuracy_verified':False}
    N.write(out/'projection-report.json',report);return {'asset_ref':N.ref(asset,root),'operation':{'kind':'observed_surface_texture_projection','projection_report_ref':N.ref(out/'projection-report.json',root)},'report':report}


def bake(asset_ref,source_rgb_ref,source_mask_ref,camera,matrix,output,*,root=ROOT):
    root=Path(root).resolve();out=root/output;require(not out.exists(),'Observed projection outputs are immutable');raw=N.checked(asset_ref,root);mesh=load_native_glb(raw)
    if mesh.vertex_colors is not None and mesh.receipt['legacy_vertex_compatibility']:return _bake_vertices(asset_ref,source_rgb_ref,source_mask_ref,camera,matrix,output,root=root)
    original,image_index=texture(raw);rgb=np.asarray(Image.open(io.BytesIO(N.checked(source_rgb_ref,root))).convert('RGB'));mask=np.asarray(Image.open(io.BytesIO(N.checked(source_mask_ref,root))))>0
    require(rgb.shape[:2]==mask.shape==(camera['height'],camera['width']),'Original source-camera pixels required')
    owners,valid=atlas_owners(mesh,original.shape[1],original.shape[0]);mapping=project_candidates(mesh,original,owners,valid,mask,camera,matrix);require(len(mapping)>=POLICY['minimum_painted_texels'],'Insufficient observed surface texture coverage')
    changed=original.copy();x,y,face,sx,sy=mapping.T;changed[y,x]=rgb[sy,sx];derived=rewrite_texture(raw,changed,image_index);require(len(derived)<=16*1024*1024,'Projected native asset exceeds portable GLB budget')
    out.mkdir(parents=True);asset=out/'asset.glb';asset.write_bytes(derived);np.savez_compressed(out/'projection-map.npz',mapping=mapping);Image.fromarray(original).save(out/'native-texture.png');Image.fromarray(changed).save(out/'observed-texture.png')
    report={'format':FORMAT,'version':1,'input_asset_ref':asset_ref,'output_asset_ref':N.ref(asset,root),'source_rgb_ref':source_rgb_ref,'source_mask_ref':source_mask_ref,
            'source_camera':camera,'camera_from_native_mesh':np.asarray(matrix).tolist(),'projection_map_ref':N.ref(out/'projection-map.npz',root),'policy':POLICY,
            'painted_texel_count':len(mapping),'native_fallback_texel_count':original.shape[0]*original.shape[1]-len(mapping),'texture_dimensions':[original.shape[1],original.shape[0]],
            'native_geometry_and_UV_unchanged':True,'source_RGB_pixels_copied_without_synthesis':True,'manual_pixel_editing_used':False,'new_api_calls':0,'new_gpu_calls':0,'physical_accuracy_verified':False}
    N.write(out/'projection-report.json',report);return {'asset_ref':N.ref(asset,root),'operation':{'kind':'observed_surface_texture_projection','projection_report_ref':N.ref(out/'projection-report.json',root)},'report':report}


def validate_operation(original,derived,operation,read):
    report=json.loads(read(operation['projection_report_ref']));require(report.get('format')==FORMAT and report.get('version')==1 and report.get('policy')==POLICY and report.get('manual_pixel_editing_used')is False and report.get('physical_accuracy_verified')is False,'Unknown or altered observed projection proof')
    before_raw=read(report['input_asset_ref']);after_raw=read(report['output_asset_ref']);before=load_native_glb(before_raw);after=load_native_glb(after_raw)
    require(np.array_equal(before.vertices,after.vertices)and np.array_equal(before.faces,after.faces)and np.array_equal(after.vertices,derived.vertices)and np.array_equal(after.faces,derived.faces),'Appearance projection modified geometry')
    require(len(before.appearance.primitives)==len(after.appearance.primitives)==1,'One projected native primitive required');a,b=before.appearance.primitives[0],after.appearance.primitives[0]
    require(np.array_equal(a.uv,b.uv)and np.array_equal(a.factor,b.factor)and a.sampler==b.sampler and a.double_sided==b.double_sided,'Appearance projection modified UVs or native material factors')
    if report.get('appearance_mode')=='vertex_rgb':
        mapping=np.load(io.BytesIO(read(report['projection_map_ref'])),allow_pickle=False)['mapping'];require(mapping.ndim==2 and mapping.shape[1]==3 and len(mapping)==report['painted_vertex_count'],'Exact native vertex projection map required')
        mask=np.asarray(Image.open(io.BytesIO(read(report['source_mask_ref']))))>0;actual=vertex_mapping(before,mask,report['source_camera'],np.asarray(report['camera_from_native_mesh']));require(np.array_equal(mapping,actual),'Recorded visible native vertex/source-pixel map cannot be reproduced')
        rgb=np.asarray(Image.open(io.BytesIO(read(report['source_rgb_ref']))).convert('RGB'));expected=before.vertex_colors.copy();ids,x,y=mapping.T;expected[ids]=rgb[y,x];require(np.array_equal(expected,after.vertex_colors)and np.array_equal(after.vertex_colors,derived.vertex_colors),'Vertex colors include invented, hidden or unreported source pixel edits');return True
    require(np.array_equal(a.tint,b.tint),'UV appearance projection modified native tint')
    prior,_=texture(before_raw);actual,_=texture(after_raw);mapping=np.load(io.BytesIO(read(report['projection_map_ref'])),allow_pickle=False)['mapping'];require(mapping.ndim==2 and mapping.shape[1]==5 and len(mapping)==report['painted_texel_count']and np.issubdtype(mapping.dtype,np.integer),'Bounded explicit observed projection map required')
    x,y,faces,sx,sy=mapping.T;require(len(set(map(tuple,mapping[:,:2])))==len(mapping),'Duplicate painted native texels')
    owners,valid=atlas_owners(before,prior.shape[1],prior.shape[0]);require(valid[y,x].all()and np.array_equal(owners[y,x],faces),'Projection painted an ambiguous or different UV face')
    rgb=np.asarray(Image.open(io.BytesIO(read(report['source_rgb_ref']))).convert('RGB'));mask=np.asarray(Image.open(io.BytesIO(read(report['source_mask_ref']))))>0
    source=report['source_camera'];matrix=np.asarray(report['camera_from_native_mesh']);points=surface_points(before,prior.shape[1],prior.shape[0],x,y,faces);cam=points@matrix[:3,:3].T+matrix[:3,3];pixel=cam@np.asarray(source['K']).T;pixel=np.rint(pixel[:,:2]/pixel[:,2:]).astype(int);require(np.array_equal(pixel,np.c_[sx,sy]),'Projection source pixel indices do not follow the recorded source camera')
    r,t,scale=M.rotation_pose(matrix);depth=render_native_mask(before,np.asarray(source['K']),r,t,source['width'],source['height'],scale=scale)['depth'];eroded=cv2.erode(mask.astype(np.uint8),np.ones((POLICY['source_mask_erosion_px'],)*2,np.uint8))>0
    require(eroded[sy,sx].all()and np.isfinite(depth[sy,sx]).all()and (abs(cam[:,2]-depth[sy,sx])<=POLICY['maximum_relative_source_depth_error']*cam[:,2]).all(),'Projection sampled hidden/background source pixels')
    expected=prior.copy();expected[y,x]=rgb[sy,sx];require(np.array_equal(expected,actual),'Projected texture contains unreported edits or invented pixels');return True
