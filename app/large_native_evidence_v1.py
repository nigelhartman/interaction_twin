"""Private large-native evidence and explicitly derived inspection previews.

No model, network, publication or canonical import. Existing portable limits are
unchanged. The native archive contract accepts only one embedded, identity-node,
vertex-colour triangle mesh; unsupported exporters fail closed.
"""
from __future__ import annotations
import io,json,hashlib,struct,traceback
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import initial_sam3d_import as N
import initial_sam3d_handoff as H
import sam3d_milestone as M
import sam3d_cluster_bridge as B
import upgrade_bundles as U
import portable_twin as P
ROOT=Path(__file__).resolve().parent
FORMAT='interaction-twin-large-native-evidence'
MAX_ARCHIVE_GLB=64*2**20
MAX_ARCHIVE_VERTICES=2_000_000
MAX_ARCHIVE_INDICES=6_000_000

def require(v,message):
 if not v:raise ValueError(message)

def reference(path,root=ROOT):
 root=Path(root).resolve();path=Path(path)
 if not path.is_absolute():path=root/path
 require(path.resolve()==path and path.is_relative_to(root) and path.relative_to(root).parts[0]in{'data','output'} and not any(p.is_symlink()for p in [path,*path.parents]),'Ordinary private evidence file required')
 raw=path.read_bytes();require(0<len(raw)<=U.MAX_FILE,'Bounded private evidence required')
 return {'path':path.relative_to(root).as_posix(),'sha256':U.digest(raw),'bytes':len(raw)}

def checked(ref,root=ROOT):
 require(isinstance(ref,dict)and set(ref)=={'path','sha256','bytes'},'Exact file reference required')
 path=Path(root)/ref['path'];require(reference(path,root)==ref,'Evidence bytes changed');return path.read_bytes()

def colour_mesh(raw):
 """Narrow archive reader, separate from the smaller portable runtime contract."""
 require(isinstance(raw,bytes)and 20<=len(raw)<=MAX_ARCHIVE_GLB,'Archive GLB byte bound exceeded')
 magic,version,length=struct.unpack_from('<III',raw);require((magic,version,length)==(0x46546c67,2,len(raw)),'Complete GLB2 required')
 chunks={};at=12
 while at<len(raw):
  require(at+8<=len(raw),'Truncated archive chunk');size,kind=struct.unpack_from('<II',raw,at);at+=8
  require(kind not in chunks and size%4==0 and size<=len(raw)-at,'Archive chunk bounds differ');chunks[kind]=raw[at:at+size];at+=size
 require(set(chunks)=={0x4e4f534a,0x004e4942} and len(chunks[0x4e4f534a])<=4*2**20,'One JSON and embedded BIN required')
 doc=U._json_bytes(chunks[0x4e4f534a].rstrip(b' \0'));binary=chunks[0x004e4942]
 require(doc.get('asset',{}).get('version')=='2.0' and not any(doc.get(k)for k in ['extensionsUsed','extensionsRequired','skins','animations','textures','images','materials']),'Only simple static native vertex colours supported')
 require(doc.get('scene',0)==0 and doc.get('scenes')==[{'nodes':[0]}] and len(doc.get('nodes',[]))==1,'One native scene node required')
 node=doc['nodes'][0];require(node.get('mesh')==0 and set(node)<= {'name','mesh','extras'},'Archive reader refuses transformed or nested nodes')
 require(len(doc.get('meshes',[]))==1 and len(doc['meshes'][0].get('primitives',[]))==1,'One native primitive required')
 primitive=doc['meshes'][0]['primitives'][0]
 require(primitive.get('mode',4)==4 and set(primitive)<= {'mode','attributes','indices'} and set(primitive.get('attributes',{}))=={'POSITION','COLOR_0'},'Only static coloured triangles supported')
 require(doc.get('buffers')==[{'byteLength':len(binary)}],'One exact embedded buffer required')
 views=doc.get('bufferViews',[]);accessors=doc.get('accessors',[])
 require(len(views)==len(accessors)==3,'Exact native position/colour/index accessors required')
 def array(index,ctype,kind,dtype,columns,max_count,normalized=False):
  require(type(index)is int and 0<=index<len(accessors),'Invalid accessor index');a=accessors[index]
  require(a.get('componentType')==ctype and a.get('type')==kind and a.get('normalized',False)is normalized and set(a)<= {'componentType','type','bufferView','byteOffset','count','min','max','normalized'},'Unsupported archive accessor')
  count=a.get('count');require(type(count)is int and 0<count<=max_count,'Archive accessor bound exceeded')
  vi=a.get('bufferView');require(type(vi)is int and 0<=vi<len(views),'Invalid archive view');v=views[vi]
  require(v.get('buffer',0)==0 and set(v)<= {'buffer','byteOffset','byteLength'},'Only packed native views supported')
  start=v.get('byteOffset',0);size=v.get('byteLength');offset=a.get('byteOffset',0);item=np.dtype(dtype).itemsize
  require(all(type(x)is int for x in [start,size,offset])and start>=0 and size>0 and offset>=0 and start+size<=len(binary)and offset+count*columns*item<=size,'Archive array escapes embedded bytes')
  out=np.frombuffer(binary,dtype=dtype,count=count*columns,offset=start+offset).reshape(count,columns).copy();require(np.isfinite(out).all(),'Nonfinite archive arrays');return out
 vertices=array(primitive['attributes']['POSITION'],5126,'VEC3','<f4',3,MAX_ARCHIVE_VERTICES).astype(float)
 colours=array(primitive['attributes']['COLOR_0'],5121,'VEC4','u1',4,MAX_ARCHIVE_VERTICES,True)
 faces=array(primitive.get('indices'),5125,'SCALAR','<u4',1,MAX_ARCHIVE_INDICES).reshape(-1)
 require(len(faces)%3==0 and len(colours)==len(vertices)and faces.max()<len(vertices),'Invalid native triangle indices/colours')
 require(np.all(colours[:,3]==255),'Only opaque archive colours supported')
 require(float(np.linalg.norm(np.ptp(vertices,axis=0)))>1e-10,'Degenerate archive mesh')
 return vertices,faces.reshape(-1,3),colours

def encode_colour_mesh(vertices,faces,colours):
 arrays=[np.asarray(faces,dtype='<u4').reshape(-1),np.asarray(vertices,dtype='<f4'),np.asarray(colours,dtype='u1')]
 blocks=[a.tobytes()for a in arrays];views=[];at=0
 for b in blocks:views.append({'buffer':0,'byteOffset':at,'byteLength':len(b)});at+=len(b)
 binary=b''.join(blocks);binary+=b'\0'*((-len(binary))%4)
 doc={'asset':{'version':'2.0','generator':'Interaction Twin derived voxel preview v1'},'scene':0,'scenes':[{'nodes':[0]}],'nodes':[{'mesh':0}],'meshes':[{'name':'derived_preview','primitives':[{'attributes':{'POSITION':1,'COLOR_0':2},'indices':0,'mode':4}]}],'buffers':[{'byteLength':len(binary)}],'bufferViews':views,'accessors':[{'componentType':5125,'type':'SCALAR','bufferView':0,'count':int(np.asarray(faces).size)},{'componentType':5126,'type':'VEC3','bufferView':1,'count':len(vertices)},{'componentType':5121,'normalized':True,'type':'VEC4','bufferView':2,'count':len(colours)}]}
 js=json.dumps(doc,separators=(',',':'),allow_nan=False).encode();js+=b' '*((-len(js))%4)
 return struct.pack('<III',0x46546c67,2,28+len(js)+len(binary))+struct.pack('<II',len(js),0x4e4f534a)+js+struct.pack('<II',len(binary),0x004e4942)+binary

def derive_preview(vertices,faces,colours,*,target_faces=350_000):
 require(type(target_faces)is int and 100<=target_faces<=500_000,'Bounded preview target required')
 lo=vertices.min(0);span=float(np.ptp(vertices,axis=0).max());history=[];original_tree=cKDTree(vertices)
 for resolution in [384,320,256,224,192,160,128,96,64,48,32]:
  cell=span/resolution;keys=np.floor((vertices-lo)/cell).astype(np.int64);_,inverse=np.unique(keys,axis=0,return_inverse=True);count=int(inverse.max())+1
  counts=np.bincount(inverse);centres=np.column_stack([np.bincount(inverse,weights=vertices[:,j])/counts for j in range(3)])
  mapped=inverse[faces];valid=(mapped[:,0]!=mapped[:,1])&(mapped[:,0]!=mapped[:,2])&(mapped[:,1]!=mapped[:,2]);mapped=mapped[valid]
  if len(mapped):_,first=np.unique(np.sort(mapped,axis=1),axis=0,return_index=True);mapped=mapped[np.sort(first)]
  used,compact=np.unique(mapped,return_inverse=True);small_vertices=centres[used];small_faces=compact.reshape(-1,3)
  history.append({'resolution':resolution,'voxel_cell_native_units':cell,'vertices':len(small_vertices),'faces':len(small_faces)})
  if not 0<len(small_faces)<=target_faces:continue
  nearest_distance,nearest=original_tree.query(small_vertices,workers=1);small_colours=colours[nearest];raw=encode_colour_mesh(small_vertices,small_faces,small_colours)
  if len(raw)>P.MAX_ASSET:continue
  decoded=P.decode_glb_bytes(raw);dist,_=cKDTree(small_vertices).query(vertices,workers=1);diag=float(np.linalg.norm(np.ptp(vertices,axis=0)))
  metrics={'original_vertex_to_preview_vertex_distance_native_units':{'mean':float(dist.mean()),'p95':float(np.quantile(dist,.95)),'max':float(dist.max())},'preview_vertex_to_original_vertex_distance_native_units':{'mean':float(nearest_distance.mean()),'max':float(nearest_distance.max())},'p95_distance_fraction_original_diagonal':float(np.quantile(dist,.95)/diag),'max_distance_fraction_original_diagonal':float(dist.max()/diag),'bounds_before':[vertices.min(0).tolist(),vertices.max(0).tolist()],'bounds_after':[small_vertices.min(0).tolist(),small_vertices.max(0).tolist()],'surface_hausdorff_measured':False,'watertightness_or_self_intersection_verified':False}
  require(metrics['max_distance_fraction_original_diagonal']<.04,'Preview collapse exceeds bounded vertex deviation')
  return raw,decoded,{'algorithm':'deterministic_uniform_voxel_centroid_clustering_v1','resolution':resolution,'target_max_faces':target_faces,'history':history,'colour_transfer':'Exact uint8 RGBA of nearest original vertex (KDTree exact lookup); positions/topology changed, appearance is a derived approximation.','collapsed_and_duplicate_triangles_removed':True,'metrics':metrics}
 raise ValueError('No bounded derived preview met geometry/byte/deviation gates')

def inspect(job_directory,*,root=ROOT):
 root=Path(root).resolve();directory=M._directory(job_directory,root);require(directory.parent==root/'data/sam3d-upload-jobs'and len(directory.name)==64,'One exact existing upload job required')
 bridge=U.read(directory/'bridge.json');handoff=bridge['handoff'];checked(handoff,root);source=N.inspect_input(handoff['path'],root=root)
 require(directory.name==handoff['sha256'] and bridge.get('job_id')=='upload-'+handoff['sha256'][:40] and source['handoff']==handoff,'Existing native attempt/handoff identity differs')
 require(bridge.get('format')=='sam3d-upload-cluster-bridge'and bridge.get('version')==1 and bridge.get('maximum_model_commands')==1 and bridge.get('automatic_retry')is False and bridge.get('parent_allocation_modified')is False,'Existing one-attempt policy differs')
 execution=B.verify_execution(directory,bridge['job_id'],handoff['sha256']);inventory=U.read(directory/'return-inventory.json')
 require(inventory.get('all_sha256_verified')is True and {r['path']for r in inventory['files']}=={'native/'+n for n in B.NATIVE}|{'evidence/'+n for n in B.EVIDENCE},'Exact completed return inventory required')
 for r in inventory['files']:require(reference(directory/'returned'/r['path'],root)=={'path':(directory/'returned'/r['path']).relative_to(root).as_posix(),'sha256':r['sha256'],'bytes':r['bytes']},'Returned inventory hash/length differs')
 failure=None
 try:N.inspect_output(handoff['path'],directory/'returned/native',root=root)
 except ValueError as error:
  frames=[];tb=error.__traceback__
  while tb:frames.append(tb.tb_frame.f_code);tb=tb.tb_next
  require(len(frames)>=5 and frames[-1]is P._bytes.__code__ and frames[-2]is P.decode_glb_bytes.__code__ and M.inspect_native_output.__code__ in frames and str(error)=='Resource is empty, not bytes, or exceeds its package budget','Output failed for a reason other than legacy portable capacity')
  failure={'type':'ValueError','message':str(error),'path':'initial_sam3d_import.inspect_output → sam3d_milestone.inspect_native_output → portable_twin.decode_glb_bytes/_bytes','meaning':'Existing source/runtime/output attestations passed up to the legacy GLB byte guard; full canonical import did not complete.'}
 require(failure is not None,'Use the ordinary importer for portable native assets')
 files=M._tree(directory/'returned/native');raw=files['object.glb'];require(len(raw)>P.MAX_ASSET,'Oversize native GLB required')
 vertices,faces,colours=colour_mesh(raw);md=U._json_bytes(files['metadata.json']);require(md['outputs']['appearance_mode']=='vertex-color','This archive contract supports only generated vertex colours')
 splat=M._splat_header(files['object-splat.ply'])
 with U._safe_layout(files['layout-and-pointmap.npz'])as layout:
  fields={};optional={'present':False}
  for name in layout.files:
   value=layout[name];require(value.dtype.kind in'fiu','Non-numeric native layout')
   if name in {'pointmap','pointmap_colors'}:
    require(value.ndim==3 and value.shape[-1]==3 and min(value.shape[:2])>0,'Invalid optional native pointmap')
    if name=='pointmap':require(not np.isinf(value).any(),'Infinite pointmap');optional={'present':True,'finite_points':int(np.isfinite(value).all(-1).sum()),'missing_pixels_preserved':True}
    else:require(np.isfinite(value).all(),'Nonfinite pointmap colours')
   else:require(np.isfinite(value).all(),'Nonfinite native layout')
   fields[name]={'shape':list(value.shape),'dtype':str(value.dtype)}
  require(fields==md['outputs']['layout_fields'],'Native layout declarations differ')
  if 'pointmap'in fields and 'pointmap_colors'in fields:require(fields['pointmap']['shape']==fields['pointmap_colors']['shape'],'Pointmap shape differs')
  pose=M.native_layout_to_cv(rotation_wxyz=layout['rotation'],translation=layout['translation'],scale_xyz=layout['scale'],source_commit=M.SAM3D_COMMIT,scale_semantics=M.DECODED_SCALE_SEMANTICS)
 report={'format':FORMAT,'version':1,'status':'gpu_generated_legacy_package_capacity_blocked','component_id':source['binding']['component_id'],'source':source['binding']['source'],'source_binding':reference(source['directory']/'source-binding.json',root),'initial_handoff':handoff,'original_bridge':reference(directory/'bridge.json',root),'return_inventory':reference(directory/'return-inventory.json',root),'returned_files':[reference(directory/'returned'/r['path'],root)for r in inventory['files']],'execution':execution,'native_glb':reference(directory/'returned/native/object.glb',root),'native_layout':pose.metadata(),'optional_pointmap':optional,'splat_contract':splat,'legacy_import_failure':failure,'capacity':{'native_bytes':len(raw),'portable_max_asset_bytes':P.MAX_ASSET,'native_vertices':len(vertices),'native_faces':len(faces),'native_indices':int(faces.size),'portable_max_triangle_indices':P.MAX_TRIANGLE_INDICES},'native_archive_validation':'narrow static opaque identity-node vertex-colour GLB structural/finite/index contract; exact source/runtime/output inventory, layout and Gaussian PLY verified; distinct from canonical portable import','legacy_canonical_import_completed':False,'native_original_modified':False,'catalog_changed':False,'selected':False,'physical_accuracy_verified':False,'metric_scale_verified':False,'assembly_verified':False,'simulation_ready':False,'new_model_commands':0,'paid_api_calls':0,'limitations':['This may preserve a semantically disputed decomposition; structural validity does not approve component identity.','Hidden surfaces and opaque generated colours are model-inferred; transparent physical glass is not reproduced physically.','A reduced preview is a new derivative; original GLB, source, receipts and failed bridge remain unchanged.']}
 return report,source,vertices,faces,colours

def build(job_directory,destination,*,root=ROOT,target_faces=350_000):
 root=Path(root).resolve();destination=Path(destination)
 if not destination.is_absolute():destination=root/destination
 require(destination.resolve()==destination and destination.is_relative_to(root/'output') and not destination.exists(),'Fresh task-owned output directory required')
 report,source,v,f,c=inspect(job_directory,root=root);raw,decoded,reduction=derive_preview(v,f,c,target_faces=target_faces)
 destination.mkdir(parents=True,exist_ok=False);(destination/'preview-derived.glb').write_bytes(raw);(destination/'implementation.py').write_bytes(Path(__file__).read_bytes())
 report.update(preview_glb=reference(destination/'preview-derived.glb',root),preview_original_geometry_modified=True,preview_only=True,preview_mesh_stats=decoded['mesh_stats'],reduction=reduction,implementation=reference(destination/'implementation.py',root))
 report['dependencies']=[{'path':Path(module.__file__).resolve().relative_to(root).as_posix(),'sha256':U.sha(Path(module.__file__).resolve()),'bytes':Path(module.__file__).stat().st_size}for module in [N,H,M,B,U,P]]
 require(checked(report['native_glb'],root)==(Path(root)/report['native_glb']['path']).read_bytes() and report['native_glb']['sha256']==next(r['sha256']for r in report['returned_files']if r['path'].endswith('/native/object.glb')),'Original identity changed during derivative build')
 (destination/'evidence.json').write_bytes(U.encoded(report));return report,source,decoded

def verify(report_ref,*,root=ROOT):
 """Recompute original proofs and exact derivative; no model or workflow mutation."""
 root=Path(root).resolve();saved=U._json_bytes(checked(report_ref,root))
 require(saved.get('format')==FORMAT and saved.get('version')==1,'Unknown large-native evidence format')
 require(checked(saved['implementation'],root)==Path(__file__).read_bytes(),'Archived inspector implementation differs')
 expected_dependencies=[{'path':Path(module.__file__).resolve().relative_to(root).as_posix(),'sha256':U.sha(Path(module.__file__).resolve()),'bytes':Path(module.__file__).stat().st_size}for module in [N,H,M,B,U,P]]
 require(saved.get('dependencies')==expected_dependencies,'Inspection dependency changed')
 directory=(root/saved['original_bridge']['path']).parent;current,_,v,f,c=inspect(directory,root=root)
 require(all(saved.get(k)==value for k,value in current.items()),'Original source/execution evidence projection differs')
 raw,decoded,reduction=derive_preview(v,f,c,target_faces=saved['reduction']['target_max_faces'])
 require(raw==checked(saved['preview_glb'],root)and reduction==saved['reduction']and decoded['mesh_stats']==saved['preview_mesh_stats'],'Derived preview or metrics differ from original')
 require(saved['preview_only']is True and saved['preview_original_geometry_modified']is True and saved['native_original_modified']is False,'Derivative scope changed')
 return {'status':'large_native_evidence_and_derived_preview_verified','original_sha256':saved['native_glb']['sha256'],'preview_sha256':saved['preview_glb']['sha256'],'legacy_canonical_import_completed':False,'owner_approved':False,'model_calls':0}
