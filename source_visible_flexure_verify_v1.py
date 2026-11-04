"""Expensive publication-time source-pixel proof; no model inference."""
from pathlib import Path
import json,hashlib,struct
import numpy as np
import cv2
from PIL import Image
from source_visible_flexure_contract_v1 import ROOT,validate_descriptor,checked,require,reference
def covered_pixel_squares_inside(points,safe):
    """Independent conservative closed triangle/unit-square overlap proof.

    A source texel is supported only if every square touched by the source
    triangle is inside the original SAM core. Unused bounding-box corners
    need not be supported; actual triangle edge/corner contacts must be.
    """
    p=np.asarray(points,dtype=np.int64);low=p.min(axis=0);high=p.max(axis=0)
    if(low<0).any()or high[0]>=safe.shape[1]or high[1]>=safe.shape[0]:return False
    sy,sx=np.nonzero(~safe[low[1]:high[1]+1,low[0]:high[0]+1])
    if not len(sx):return True
    q=np.column_stack((sx+low[0],sy+low[1]));touches=np.ones(len(q),bool)
    for i in range(3):
        edge=p[(i+1)%3]-p[i];axis=np.array([-edge[1],edge[0]],np.int64);extent=np.abs(axis).sum();t=2*(p@axis);c=2*(q@axis)
        touches&=(c+extent>=t.min())&(c-extent<=t.max())
    return not touches.any()
def verify_cached_depth(mapping,parent_mapping,row,safe):
    """Check actual interpolation weights and no crossing of unknown pixels."""
    xy=mapping['source_xy'].astype(float);z=mapping['depth'].astype(np.float32);parent_xy=parent_mapping['source_xy'].astype(float);parent_z=parent_mapping['depth'].astype(float)
    ids=mapping['parent_depth_vertex_ids'];weights=mapping['parent_depth_weights'];modes=mapping['depth_sampling_modes'];n=len(xy);policy=row['depth_sampling']
    require(ids.shape==(n,3)and np.issubdtype(ids.dtype,np.integer)and(weights.shape==(n,3))and modes.shape==(n,)and np.issubdtype(modes.dtype,np.integer),'Cached-depth weight map shape differs')
    require(np.isfinite(weights).all()and(weights>=-1e-8).all()and np.max(abs(weights.sum(1)-1))<1e-8 and(ids>=0).all()and(ids<len(parent_xy)).all()and np.isin(modes,[0,1]).all(),'Cached-depth indices/weights are invalid')
    require(np.isfinite(parent_z).all()and(parent_z>0).all(),'Original cached prior has invalid depth')
    expected=np.einsum('ij,ij->i',parent_z[ids],weights).astype(np.float32);require(np.array_equal(expected,z),'Cached depth does not equal the exact declared original weighted samples')
    nearest=0;linear=0
    for index,(target,chosen,w,mode)in enumerate(zip(xy,ids,weights,modes)):
        p=parent_xy[chosen]
        if mode==0:
            linear+=1;require(np.max(abs(w@p-target))<1e-6,'Barycentric cached-depth coordinates differ from source pixel')
            require(max(np.linalg.norm(p[a]-p[b])for a,b in [(0,1),(1,2),(2,0)])<=policy['maximum_interpolation_parent_edge_px']+1e-8,'Cached interpolation crosses an overlong original depth edge')
            require(covered_pixel_squares_inside(p,safe),'Cached depth interpolation crosses an unknown source pixel')
        else:
            nearest+=1;require(np.array_equal(chosen,np.repeat(chosen[0],3))and np.array_equal(w,[1.,0.,0.]),'Nearest cached depth carries unapproved weights')
            distance=np.linalg.norm(target-p[0]);minimum=np.linalg.norm(parent_xy-target,axis=1).min();require(distance<=policy['maximum_nearest_sampling_distance_px']+1e-8 and abs(distance-minimum)<1e-8,'Cached nearest sample is distant or not nearest')
            require(covered_pixel_squares_inside(np.asarray([target,p[0],p[0]]),safe),'Nearest depth sampling crosses an unknown source pixel')
    return {'exact_cached_depth_weight_derivation':True,'barycentric_samples':linear,'bounded_nearest_samples':nearest,'no_depth_sampling_across_unknown_source_pixels':True}
def decode_surface(raw):
    require(raw[:4]==b'glTF'and struct.unpack_from('<I',raw,4)[0]==2,'Only GLB2 source surfaces supported');size,kind=struct.unpack_from('<II',raw,12);require(kind==0x4e4f534a,'Source GLB JSON required');doc=json.loads(raw[20:20+size]);off=20+size;length,kind=struct.unpack_from('<II',raw,off);require(kind==0x004e4942 and off+8+length==len(raw),'Single exact embedded BIN required');binary=raw[off+8:]
    require(len(doc['meshes'])==1 and len(doc['meshes'][0]['primitives'])==1 and not doc.get('skins')and not doc.get('animations'),'One static source-camera surface required');p=doc['meshes'][0]['primitives'][0];require(p.get('mode',4)==4,'Triangle mesh required')
    for n in doc.get('nodes',[]):require(not any(k in n for k in ['matrix','translation','rotation','scale']),'Source camera geometry cannot carry hidden node transforms')
    def accessor(i):
        a=doc['accessors'][i];v=doc['bufferViews'][a['bufferView']];require(not a.get('sparse')and v.get('buffer',0)==0,'Packed source accessor required');dtype={5121:np.uint8,5123:np.uint16,5125:np.uint32,5126:np.float32}[a['componentType']];channels={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']];item=np.dtype(dtype).itemsize*channels;require(v.get('byteStride',item)==item,'Interleaved source arrays unsupported');offset=v.get('byteOffset',0)+a.get('byteOffset',0);out=np.frombuffer(binary,dtype=dtype,count=a['count']*channels,offset=offset).reshape(-1,channels).copy();return out
    attrs={k:accessor(v)for k,v in p['attributes'].items()};faces=accessor(p['indices']).reshape(-1,3);return {'document':doc,'binary':binary,'vertices':attrs['POSITION'],'faces':faces,'colors':attrs.get('_SOURCE_RGB',attrs.get('COLOR_0'))[:,:3],'UV':attrs.get('TEXCOORD_0'),'attributes':attrs}
def validate_surface_files(reference_value,root=ROOT):
    root=Path(root).resolve();result=validate_descriptor(reference_value,root);d=result['document'];K=np.asarray(d['camera']['K'],float);reports=[];bytes_checked=0
    def read(r):
        p=root/r['path'];require(not p.is_symlink()and p.resolve().is_relative_to(root),'Surface proof path escaped root');return p.read_bytes()
    for r in result['artifacts']:bytes_checked+=len(checked(r,read))
    parent_proof=None
    if d.get('source_footprint_proof_version',1)==2:
        parent_proof=validate_surface_files(d['cached_depth_parent'],root)
    scene=result['source_scene'];originals=[]
    for cid in d['replacement_component_ids']:
        asset=next(c['asset']for c in scene['components']if c['id']==cid);checked(asset,read);originals.append(asset)
    for row in d['samples']:
        if row['status']=='unknown':continue
        raw=checked(row['asset'],read);mesh=decode_surface(raw);mapping=np.load(root/row['source_pixel_map']['path'],allow_pickle=False)
        if d.get('source_footprint_proof_version',1)==2:
            require(mapping['source_xy'].ndim==2 and mapping['source_xy'].shape[1]==2 and np.issubdtype(mapping['source_xy'].dtype,np.integer)and np.issubdtype(mapping['faces'].dtype,np.integer),'Adaptive source-pixel coordinates/topology must be exact integers')
            require(mapping['depth'].dtype==np.float32 and np.isfinite(mapping['depth']).all()and(mapping['depth']>0).all(),'Adaptive cached depths must be finite positive float32 samples')
        xy=mapping['source_xy'].astype(int);z=mapping['depth'].astype(float);faces=mapping['faces'].astype(np.int64)
        require(len(xy)==row['vertices']==len(mesh['vertices'])and len(faces)==row['triangles']==len(mesh['faces']),'Source map/mesh counts differ');require(np.array_equal(mesh['faces'],faces),'Surface face order or topology differs from exact source map')
        expect=np.c_[(xy[:,0]-K[0,2])/K[0,0]*z,(xy[:,1]-K[1,2])/K[1,1]*z,z];error=float(np.max(abs(expect-mesh['vertices'])));require(error<5e-7,'Surface vertices do not unproject exact recorded source pixels/depth')
        rgb=np.asarray(Image.open(root/row['source_RGB']['path']).convert('RGB'));require((xy[:,0]>=0).all()and(xy[:,0]<rgb.shape[1]).all()and(xy[:,1]>=0).all()and(xy[:,1]<rgb.shape[0]).all(),'Source RGB coordinate escaped image');require(np.array_equal(mesh['colors'],rgb[xy[:,1],xy[:,0]]),'Surface colors are not exact original source RGB samples')
        if row.get('appearance_mode')=='exact_source_JPEG_projected_UV':
            uv=(xy.astype(np.float32)+.5)/np.array([rgb.shape[1],rgb.shape[0]],np.float32);require(np.array_equal(mesh['UV'],uv),'Observed texture UVs do not match exact source pixels')
            require('COLOR_0'not in mesh['attributes']and '_SOURCE_RGB'in mesh['attributes']and row.get('vertex_RGB_multiplied_into_texture')is False,'Source RGB is being multiplied into the texture')
            gd=mesh['document'];require(len(gd.get('images',[]))==1 and gd['images'][0].get('mimeType')=='image/jpeg'and 'uri'not in gd['images'][0],'Exactly one embedded source JPEG required');iv=gd['bufferViews'][gd['images'][0]['bufferView']];start=iv.get('byteOffset',0);image_bytes=mesh['binary'][start:start+iv['byteLength']];require(image_bytes==checked(row['source_RGB'],read),'Embedded JPEG differs from original source bytes')
        masks={r['track_id']:np.asarray(Image.open(root/r['path']))>0 for r in row['source_masks']};shape=rgb.shape[:2];ids=d['replacement_component_ids'];union=np.zeros(shape,bool)
        for cid in ids:union|=masks.get(cid,np.zeros(shape,bool))
        for cid in d.get('unaffected_component_ids',[]):union&=~masks.get(cid,np.zeros(shape,bool))
        safe=cv2.erode(union.astype(np.uint8),np.ones((2,2),np.uint8))>0
        require(safe[xy[:,1],xy[:,0]].all(),'Surface vertex outside source SAM foreground core')
        if 'visible_component_ids'in row:
            visible=[cid for cid in ids if cid in masks and masks[cid][xy[:,1],xy[:,0]][faces].all(1).any()];require(row['visible_component_ids']==visible,'Visible component labels lack actual retained-triangle support')
        proof_version=d.get('source_footprint_proof_version',1);depth_proof={}
        if proof_version==1:
            # Historical semantics are intentionally unchanged.
            for face in faces:
                p=xy[face];lo=p.min(0);hi=p.max(0);require(bool(safe[lo[1]:hi[1]+1,lo[0]:hi[0]+1].all()),'Surface triangle crosses an unknown/occluded source pixel')
        else:
            for face in faces:require(covered_pixel_squares_inside(xy[face],safe),'Conservative source triangle footprint crosses an unknown/occluded source pixel')
            parent_map=np.load(root/row['cached_depth_parent_map']['path'],allow_pickle=False);depth_proof=verify_cached_depth(mapping,parent_map,row,safe)
            depths=z[faces];require((np.ptp(depths,axis=1)/depths.mean(1)<.06).all(),'Observed adaptive surface crosses a depth discontinuity')
        n=np.cross(mesh['vertices'][faces[:,1]]-mesh['vertices'][faces[:,0]],mesh['vertices'][faces[:,2]]-mesh['vertices'][faces[:,0]]);require(((n*mesh['vertices'][faces].mean(1)).sum(1)<0).all(),'A source patch has reversed/degenerate camera-facing winding')
        size,kind=struct.unpack_from('<II',raw,12);gltf=json.loads(raw[20:20+size]);require('KHR_materials_unlit'in gltf.get('extensionsUsed',[])and all(m.get('doubleSided')is False and 'KHR_materials_unlit'in m.get('extensions',{})for m in gltf.get('materials',[])),'Observed backfaces or appearance policy changed')
        prov=json.loads(checked(row['provenance'],read));require(prov=={k:v for k,v in row.items()if k!='provenance'},'Per-frame provenance differs from descriptor')
        reports.append({'source_sample_index':row['source_sample_index'],'vertices':len(xy),'triangles':len(faces),'maximum_unprojection_error':error,'exact_source_RGB_vertices':True,('all_triangle_source_pixel_rectangles_within_saved_SAM_core'if proof_version==1 else'all_conservatively_covered_pixel_squares_within_saved_SAM_core'):True,'hidden_backs_culled':True,'numeric_joint_parameters_unknown':True,**depth_proof})
    report={'format':'interaction-twin-source-visible-surface-publication-proof','version':1,'descriptor':reference(reference_value),'all_passed':True,'supported_sources':result['supported_sources'],'unknown_sources':result['unknown_sources'],'original_native_assets_checked_unchanged':originals,'samples':reports,'artifact_count':len(result['artifacts']),'artifact_bytes_checked':bytes_checked,'new_inference_calls':0,'physical_accuracy_verified':False,'independent_geometry_correctness_established':False}
    if parent_proof is not None:report.update({'source_footprint_proof_version':2,'cached_depth_parent_proof':{'descriptor':parent_proof['descriptor'],'all_passed':parent_proof['all_passed'],'supported_sources':parent_proof['supported_sources'],'unknown_sources':parent_proof['unknown_sources']}})
    return report
