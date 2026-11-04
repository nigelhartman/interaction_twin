"""Offline generation/verification of compact, source-observed RGB states."""
from __future__ import annotations
import copy,hashlib,io,json,struct
from pathlib import Path
import numpy as np
from PIL import Image
import native_assembly_refinement_v1 as N
import observed_surface_projection_v1 as O
from native_appearance import load_native_glb
import source_appearance_states_v1 as S

ROOT=Path(__file__).resolve().parent
METHOD={'source_depth_policy':O.POLICY,'probe_selection':'jointly_source_visible_31_and_44; source31_maxRGB<100; source44_meanRGB>160; meanRGB_rise>90',
 'minimum_visible_probe_count':300,'dark_maximum_switched_probe_fraction':.20,
 'logo_minimum_switched_probe_fraction':.60,'per_probe_switch_fraction_of_reference_rise':.35,
 'observations_are_source_appearance_not_electronic_state':True}

def pack(mapping,rgb,total):
    return struct.pack('<4sIII',b'ITCP',1,total,len(mapping))+b''.join(struct.pack('<IHHBBB',int(i),int(x),int(y),*map(int,rgb[y,x]))for i,x,y in mapping)

def native_vertex_scope(raw,vertex_count):
    """Same narrow display scope as NativeByteRgbDisplayV1; no float/texture transfer assumptions."""
    mesh=load_native_glb(raw);native,binary=O.unpack(raw);primitives=[p for m in native['meshes']for p in m['primitives']]
    S.require(len(primitives)==1 and mesh.receipt['legacy_vertex_compatibility']and mesh.vertex_colors is not None and len(mesh.vertices)==vertex_count,'Only one legacy native byteRGB primitive supports compact observed patches')
    primitive=primitives[0];color=native['accessors'][primitive['attributes']['COLOR_0']];view=native['bufferViews'][color['bufferView']];channels=4 if color['type']=='VEC4'else 3 if color['type']=='VEC3'else 0;materials=native.get('materials',[]);material=materials[primitive['material']]if 'material'in primitive else {};pbr=material.get('pbrMetallicRoughness',{})
    S.require(color['componentType']==5121 and color.get('normalized')is True and channels>0 and not color.get('sparse')and view.get('byteStride',channels)==channels and color['count']==vertex_count and material.get('alphaMode','OPAQUE')=='OPAQUE'and not material.get('extensions')and not pbr.get('baseColorTexture')and pbr.get('baseColorFactor',[1,1,1,1])==[1,1,1,1],'Unsupported texture/float/nontrivial native color material')
    start=view.get('byteOffset',0)+color.get('byteOffset',0);values=np.frombuffer(binary,dtype=np.uint8,count=color['count']*channels,offset=start).reshape(-1,channels);S.require(channels==3 or bool((values[:,3]==255).all()),'Opaque native vertex alpha required')
    return mesh

def visible(mesh,pose,scene,read):
    mask=np.asarray(Image.open(io.BytesIO(read(pose['source_mask']))))>0
    return O.vertex_mapping(mesh,mask,scene['camera'],pose['camera_from_native_mesh'])

def photometry(mapping,rgb,probe_ids,dark_reference,light_reference):
    lookup={int(i):(int(x),int(y))for i,x,y in mapping};valid=[(j,lookup[int(i)])for j,i in enumerate(probe_ids)if int(i)in lookup]
    if len(valid)<METHOD['minimum_visible_probe_count']:return {'status':'unknown','visible_probe_count':len(valid),'switched_probe_fraction':None,'mean_source_RGB':None}
    indices=np.array([j for j,_ in valid]);xy=np.array([xy for _,xy in valid]);color=rgb[xy[:,1],xy[:,0]].astype(float);lo=dark_reference[indices].mean(1);hi=light_reference[indices].mean(1)
    fraction=float(np.mean(color.mean(1)>lo+METHOD['per_probe_switch_fraction_of_reference_rise']*(hi-lo)))
    label='source_observed_dark'if fraction<=METHOD['dark_maximum_switched_probe_fraction']else'source_observed_logo'if fraction>=METHOD['logo_minimum_switched_probe_fraction']else'unknown'
    return {'status':label,'visible_probe_count':len(valid),'switched_probe_fraction':fraction,'mean_source_RGB':color.mean(0).tolist()}

def build(*,root=ROOT,output='data/laptop-display-appearance-20261104-v1'):
    root=Path(root).resolve();out=root/output;S.require(not out.exists(),'Source appearance artifacts are immutable');out.mkdir(parents=True)
    prior_path=root/'data/laptop-mating-20261104-v1/final-candidate-v2/declared-geometry.json';prior=json.loads(prior_path.read_bytes());scene=json.loads(N.checked(prior['scene_ref'],root));program=json.loads(N.checked(prior['evidence_program'],root));c=next(c for c in scene['components']if c['id']=='laptop_lid');mesh=load_native_glb(N.checked(c['asset'],root));poses={p['source_sample_index']:p for p in c['poses']};ev={e['source_index']:e for e in program['evidence']}
    def read(ref,**kwargs):return N.checked(ref,root)
    mappings={};colors={};keyframes=[]
    for n,label in [(31,'source_observed_dark'),(44,'source_observed_logo')]:
        pose=poses[n];frame=N.ref(root/ev[n]['frame_path'],root);rgb=np.asarray(Image.open(io.BytesIO(read(frame))).convert('RGB'));mapping=visible(mesh,pose,scene,read);mappings[n]=mapping
        table=np.full((len(mesh.vertices),3),-1,dtype=np.int16);table[mapping[:,0]]=rgb[mapping[:,2],mapping[:,1]];colors[n]=table
        patch=out/f'source-{n:06}-rgb.itcp';patch.write_bytes(pack(mapping,rgb,len(mesh.vertices)))
        keyframes.append({'id':f'appearance_source_{n:06}','source_sample_index':n,'timestamp_s':pose['timestamp_s'],'appearance_label':label,'source_rgb_ref':frame,'source_mask_ref':pose['source_mask'],
            'source_camera':scene['camera'],'camera_from_native_mesh':pose['camera_from_native_mesh'],'patch_ref':N.ref(patch,root),'painted_vertex_count':len(mapping),'native_fallback_vertex_count':len(mesh.vertices)-len(mapping)})
        print('keyframe',n,len(mapping),flush=True)
    a,b=colors[31],colors[44];probe=np.flatnonzero((a.min(1)>=0)&(b.min(1)>=0)&(a.max(1)<100)&(b.mean(1)>160)&(b.mean(1)-a.mean(1)>90)).astype('<u4');S.require(len(probe)>=METHOD['minimum_visible_probe_count'],'Source states do not provide sufficient distinct visible probes')
    probe_path=out/'probe-native-vertex-ids.u32';probe_path.write_bytes(probe.tobytes());rows=[]
    for pose in c['poses']:
        n=pose['source_sample_index'];row={'source_sample_index':n,'timestamp_s':pose['timestamp_s'],'status':'unknown','appearance_label':'unknown','keyframe_id':None,'reason':'no_supported_open_display_source_observation'}
        # Native faces are not painted based merely on an inferred closed state.
        if pose['status']=='estimated'and 27<=n<=54:
            frame=N.ref(root/ev[n]['frame_path'],root);rgb=np.asarray(Image.open(io.BytesIO(read(frame))).convert('RGB'));mapping=mappings[n]if n in mappings else visible(mesh,pose,scene,read);stats=photometry(mapping,rgb,probe,a[probe],b[probe]);row.update(source_rgb_ref=frame,source_mask_ref=pose['source_mask'],photometric_observation=stats)
            if stats['status']!='unknown':row.update(status='source_observed_appearance_estimate',appearance_label=stats['status'],keyframe_id='appearance_source_000031'if stats['status']=='source_observed_dark'else'appearance_source_000044',reason='actual_saved_RGB_photometry_at_source_visible_native_probes')
            print('observation',n,stats,flush=True)
        rows.append(row)
    doc={'format':S.FORMAT,'version':1,'previous_derived_assembly_ref':N.ref(prior_path,root),'scene_ref':prior['scene_ref'],'evidence_program':prior['evidence_program'],'source':prior['source'],'component_id':'laptop_lid','baseline_asset_ref':c['asset'],'vertex_count':len(mesh.vertices),'keyframes':keyframes,'observations':rows,'display_policy':S.POLICY,
        'photometric_method':METHOD,'probe_native_vertex_ids_ref':N.ref(probe_path,root),'source_state_assertion_ids':['a_screen_dark_31','a_screen_lit_44'],
        'physical_accuracy_verified':False,'simulation_ready':False,'electronic_state_verified':False,'new_api_calls':0,'new_gpu_calls':0,'manual_geometry_or_poses_used':False,
        'limitations':['Source-projected RGB is observed appearance on estimated native surface/camera, not a reconstructed display or verified power state.','Only source-visible vertices are repainted; all hidden/unmapped vertices retain static approved native appearance.','Between saved observations nearest compatible appearance is an estimate; incompatible state changes and unknown poses are barriers.','Reflections/exposure are copied from actual source and may appear fixed to the surface in orbit view.','Screen appearance does not fix remaining source27 width or native bevel seam.']}
    N.write(out/'source-appearance-states.json',doc);validate_files(doc,root=root);new=copy.deepcopy(prior);new['source_appearance_states_ref']=N.ref(out/'source-appearance-states.json',root);N.write(out/'declared-geometry.json',new)
    return {'sidecar_ref':N.ref(out/'source-appearance-states.json',root),'derived_assembly_ref':N.ref(out/'declared-geometry.json',root),'keyframe_count':len(keyframes),'patch_bytes':sum(k['patch_ref']['bytes']for k in keyframes),'supported_appearance_samples':sum(r['status']!='unknown'for r in rows)}

def validate_files(document,*,root=ROOT):
    root=Path(root).resolve()
    def read(ref,**kwargs):return N.checked(ref,root)
    validated=S.validate(document,read);scene=validated['scene'];mesh=native_vertex_scope(read(document['baseline_asset_ref']),document['vertex_count']);S.require(document['photometric_method']==METHOD,'Unknown appearance projection method')
    program=json.loads(read(document['evidence_program']));assertions={a['id']:a for a in program['assertions']}
    S.require(document['source_state_assertion_ids']==['a_screen_dark_31','a_screen_lit_44']and all(assertions.get(i,{}).get('variable_id')=='screen_display'and assertions[i].get('value')==value and assertions[i].get('evidence_id')==evidence for i,value,evidence in [('a_screen_dark_31','dark','source_000031'),('a_screen_lit_44','illuminated','source_000044')]),'Direct source-state model assertions missing or changed')
    mappings={};tables={}
    for key in document['keyframes']:
        raw=read(key['patch_ref']);parsed=S.decode_patch(raw);mapping=np.array([r[:3]for r in parsed['records']],np.int32);mask=np.asarray(Image.open(io.BytesIO(read(key['source_mask_ref']))))>0;actual=O.vertex_mapping(mesh,mask,key['source_camera'],key['camera_from_native_mesh']);S.require(np.array_equal(mapping,actual),'Compact RGB patch paints hidden/unsupported source surface');rgb=np.asarray(Image.open(io.BytesIO(read(key['source_rgb_ref']))).convert('RGB'));S.require(raw==pack(actual,rgb,len(mesh.vertices)),'Compact patch includes invented or changed source RGB');mappings[key['source_sample_index']]=mapping
        table=np.full((len(mesh.vertices),3),-1,dtype=np.int16);table[mapping[:,0]]=rgb[mapping[:,2],mapping[:,1]];tables[key['source_sample_index']]=table
    a,b=tables[31],tables[44];expected=np.flatnonzero((a.min(1)>=0)&(b.min(1)>=0)&(a.max(1)<100)&(b.mean(1)>160)&(b.mean(1)-a.mean(1)>90)).astype('<u4');S.require(expected.tobytes()==read(document['probe_native_vertex_ids_ref']),'Photometric probes differ from actual source-state keyframes')
    poses={p['source_sample_index']:p for p in next(c for c in scene['components']if c['id']==document['component_id'])['poses']}
    for row in document['observations']:
        if 'photometric_observation'not in row:continue
        n=row['source_sample_index'];mapping=mappings[n]if n in mappings else visible(mesh,poses[n],scene,read);rgb=np.asarray(Image.open(io.BytesIO(read(row['source_rgb_ref']))).convert('RGB'));stats=photometry(mapping,rgb,expected,a[expected],b[expected]);S.require(stats==row['photometric_observation'],'Source appearance observation cannot be recomputed');S.require(stats['status']==row['appearance_label'],'Unsupported source appearance label')
    return {'validated':True,'keyframes':len(document['keyframes']),'patch_bytes':sum(k['patch_ref']['bytes']for k in document['keyframes']),'probe_count':len(expected),'geometry_camera_and_pose_unchanged':True,'exact_source_RGB_verified':True}

if __name__=='__main__':print(json.dumps(build()),flush=True)
