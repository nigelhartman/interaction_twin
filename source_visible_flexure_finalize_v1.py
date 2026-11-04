"""Additive source-mask footprint fence and sanitized delivery, no inference."""
from pathlib import Path
import copy,json,struct,time
import cv2,numpy as np,trimesh
from PIL import Image
from native_appearance import load_native_glb
from source_visible_flexure_20261104_v2 import ROOT,SCENE,REVIEW,ref,save,render_board

PARENT=ROOT/'data/source-visible-flexure/wallet-20261104-v2/descriptor.json'
DATA=ROOT/'data/source-visible-flexure/wallet-20261104-v3'
OUT=ROOT/'output/experience-refinement-20261104-v1/wallet/source-visible-flexure-v3'
def unlit(raw):
    size,kind=struct.unpack_from('<II',raw,12);assert kind==0x4e4f534a;doc=json.loads(raw[20:20+size]);off=20+size;length,kind=struct.unpack_from('<II',raw,off);assert kind==0x004e4942;binary=raw[off+8:off+8+length]
    doc['materials']=[{'name':'Exact observed source RGB, unknown backs culled','pbrMetallicRoughness':{'baseColorFactor':[1,1,1,1],'metallicFactor':0,'roughnessFactor':1},'doubleSided':False,'extensions':{'KHR_materials_unlit':{}}}]
    doc.setdefault('extensionsUsed',[]).append('KHR_materials_unlit')
    for m in doc['meshes']:
        for p in m['primitives']:p['material']=0
    doc['asset']['extras']={'interaction_twin_kind':'source_observed_monocular_depth_prior','native_SAM_mesh':False,'hidden_surfaces':'absent_not_inpainted','physical_accuracy_verified':False}
    enc=json.dumps(doc,separators=(',',':')).encode();enc+=b' '*((-len(enc))%4);return struct.pack('<4sII',b'glTF',2,12+8+len(enc)+8+len(binary))+struct.pack('<II',len(enc),0x4e4f534a)+enc+struct.pack('<II',len(binary),0x004e4942)+binary
def footprint_keep(xy,faces,safe):
    """Conservative full pixel bounding-rectangle support for each triangle."""
    p=xy[faces].astype(int);lo=p.min(1);hi=p.max(1);integral=cv2.integral(safe.astype(np.uint8));x0,y0=lo.T;x1,y1=(hi+1).T
    counts=integral[y1,x1]-integral[y0,x1]-integral[y1,x0]+integral[y0,x0];return counts==(x1-x0)*(y1-y0)
def run():
    DATA.mkdir(parents=True,exist_ok=False);OUT.mkdir(parents=True,exist_ok=False);parent=json.loads(PARENT.read_text());doc=copy.deepcopy(parent);K=np.asarray(doc['camera']['K']);rows=[];counts=[];t0=time.monotonic()
    protocol={'format':'interaction-twin-source-visible-flexure-protocol','version':3,'policy':copy.deepcopy(doc['policy']),
      'depth_model':{'name':'Ruicheng/moge-2-vits-normal','checkpoint_sha256':next(r['depth_prior_checkpoint']['sha256']for r in doc['samples']if r['status']=='source_observed_prior'),'device':'cpu','num_tokens':600,'input_dimensions':[252,448]},
      'source_scene':doc['source_scene'],'source_program':doc['source_program'],'source_video_sha256':doc['source']['sha256'],
      'semantic_review':{'provider':'Meta Model API','model':'muse-spark-1.3','result_sha256':ref(REVIEW)['sha256'],'fallible_qualitative_observations_only':True},
      'native_geometry_unchanged':True,'new_API_calls':0,'new_GPU_calls':0,'network_access_during_inference':False,'physical_accuracy_verified':False,
      'triangle_support':'every pixel of each triangle source-coordinate bounding rectangle lies inside2px_eroded_saved_SAM_union_excluding_coin',
      'sourceRGB':'exact original RGB at each vertex; triangle interpolation only, no synthesis or inpainting',
      'unknown_surfaces':'absent; original unknown samples and Muse occluded58/uncertain67 remain barriers',
      'implementation_sha256':ref(Path(__file__))['sha256'],'inference_implementation_sha256':ref(ROOT/'source_visible_flexure_20261104_v2.py')['sha256'],
      'parent_research_descriptor_sha256':ref(PARENT)['sha256']}
    protocol['policy']['triangle_support']=protocol['triangle_support'];save(DATA/'protocol.json',protocol)
    for row in doc['samples']:
        if row['status']!='source_observed_prior':rows.append(row);continue
        index=row['source_sample_index'];rgb=np.asarray(Image.open(ROOT/row['source_RGB']['path']).convert('RGB'));masks={r['track_id']:np.asarray(Image.open(ROOT/r['path']))>0 for r in row['source_masks']};shape=rgb.shape[:2];combined=(masks.get('wallet_holder',np.zeros(shape,bool))|masks.get('pouch_flap',np.zeros(shape,bool)))&~masks.get('gold_coin',np.zeros(shape,bool));safe=cv2.erode(combined.astype(np.uint8),np.ones((2,2),np.uint8))>0
        raw=(ROOT/row['asset']['path']).read_bytes();mesh=load_native_glb(raw);mapping=np.load(ROOT/row['source_pixel_map']['path'],allow_pickle=False);xy=mapping['source_xy'];keep=footprint_keep(xy,mesh.faces,safe);faces=mesh.faces[keep];used=np.unique(faces);remap=np.full(len(mesh.vertices),-1,np.int32);remap[used]=np.arange(len(used));faces=remap[faces];vertices=mesh.vertices[used];colors=mesh.vertex_colors[used];xy=xy[used]
        out=DATA/f'source-{index:03d}';out.mkdir();new=trimesh.Trimesh(vertices=vertices,faces=faces,vertex_colors=colors,process=False);asset=out/'observed-surface.glb';asset.write_bytes(unlit(new.export(file_type='glb')));np.savez_compressed(out/'source-pixel-map.npz',source_xy=xy.astype(np.uint16),depth=vertices[:,2].astype(np.float32),faces=faces.astype(np.uint16))
        row['parent_research_asset_sha256']=row['asset']['sha256'];row['asset']=ref(asset);row['source_pixel_map']=ref(out/'source-pixel-map.npz');row['depth_model']={'name':'Ruicheng/moge-2-vits-normal','checkpoint_sha256':row.pop('depth_prior_checkpoint')['sha256'],'device':'cpu'};row.pop('provenance',None);row['triangles_pruned_by_unknown_pixel_footprint']=int((~keep).sum());row['vertices']=len(vertices);row['triangles']=len(faces);row['triangle_source_footprint_verified']=True;row['hidden_backs_culled']=True;row['unlit_exact_vertex_RGB']=True;save(out/'provenance.json',row);row['provenance']=ref(out/'provenance.json');rows.append(row);counts.append({'source':index,'vertices':len(vertices),'triangles':len(faces),'removed':int((~keep).sum())})
        if index in [14,27,28,99]:render_board(rgb,combined,vertices,faces,colors,K,OUT/f'comparison-{index:03d}.png',f'Source{index:03d} · sourceRGBreprojection + CPUlearneddepth; uncertaingeometryabsent')
    doc['samples']=rows;doc['protocol']=ref(DATA/'protocol.json');doc['policy']=protocol['policy'];doc['version']=1;doc['delivery_revision']=3;doc['depth_model']=protocol['depth_model'];doc['presentation_policy']={'supported_time':'source_observed_prior_mesh','unknown_time':'withhold_holder_and_flap_instead_of_wrong_closed_mesh','original_native':'individual_downloads_or_exact_qualitatively_closed6_10_99_only','geometric_interpolation':False,'show_estimated_required':True}
    save(DATA/'descriptor.json',doc);summary={'format':'interaction-twin-source-visible-flexure-report','version':3,'descriptor':ref(DATA/'descriptor.json'),'source_scene':doc['source_scene'],'source_program':doc['source_program'],'source_sha256':doc['source']['sha256'],'supported_sources':[r['source_sample_index']for r in rows if r['status']=='source_observed_prior'],'unknown_sources':[r['source_sample_index']for r in rows if r['status']=='unknown'],'samples':counts,'total_GLΒ_bytes':sum(r['asset']['bytes']for r in rows if r['status']=='source_observed_prior'),'numeric_joint_parameters':'unknown','new_API_calls':0,'new_GPU_calls':0,'wall_seconds':time.monotonic()-t0,'physical_accuracy_verified':False,'source_projection_overlap_is_not_independent_geometry_validation':True}
    save(DATA/'report.json',summary);save(OUT/'delivery.json',{'descriptor':ref(DATA/'descriptor.json'),'report':ref(DATA/'report.json'),'protocol':ref(DATA/'protocol.json'),'comparisons':[ref(OUT/f'comparison-{i:03d}.png')for i in [14,27,28,99]],'new_API_calls':0,'new_GPU_calls':0,'production_state_changed':False});print(json.dumps({'descriptor':ref(DATA/'descriptor.json'),'summary':{k:v for k,v in summary.items()if k!='samples'}},indent=2))
if __name__=='__main__':run()
