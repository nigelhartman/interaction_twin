"""Additive contour mesh upgrades using only exact existing depth samples.

This creates candidates, never changes a catalog or an original descriptor.
"""
from pathlib import Path
import argparse,copy,hashlib,json,time
import cv2,numpy as np,trimesh
from PIL import Image
from source_visible_boundary_mesh_v1 import build
from source_visible_flexure_finalize_v1 import unlit
from source_visible_texture_v1 import exact_source_texture,visible_components
from source_visible_flexure_verify_v1 import decode_surface,validate_surface_files
from source_visible_flexure_contract_v1 import validate_descriptor
ROOT=Path(__file__).resolve().parent
RULE='all_unit_pixel_squares_intersecting_closed_triangle_inside_original_2px_SAM_core'
def ref(path):
 p=Path(path);raw=p.read_bytes();return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
def checked(r):
 p=ROOT/r['path'];actual=ref(p);assert all(actual[k]==r[k]for k in actual);return p
def save(path,value):
 with Path(path).open('x')as f:json.dump(value,f,indent=2);f.write('\n')
def silhouette(xy,faces,shape):
 mask=np.zeros(shape,np.uint8)
 for face in faces:cv2.fillConvexPoly(mask,xy[face].astype(np.int32),1)
 return mask>0
def metrics(actual,target):
 border=actual&~cv2.erode(actual.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool);tb=target&~cv2.erode(target.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool);dist=cv2.distanceTransform((~tb).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
 return {'constructive_source_silhouette_IoU':float((actual&target).sum()/max(1,(actual|target).sum())),'source_boundary_distance_median_px':float(np.median(dist[border])),'source_boundary_distance_p95_px':float(np.percentile(dist[border],95)),'outside_original_SAM_pixels':int((actual&~target).sum())}
def run(name,parent_version,new_version):
 start=time.monotonic();parent_path=ROOT/f'data/source-visible-flexure/{name}-20261104-v{parent_version}/descriptor.json';parent_ref=ref(parent_path);old=validate_descriptor(parent_ref,ROOT)['document'];out=ROOT/f'data/source-visible-flexure/{name}-20261104-v{new_version}';out.mkdir(exist_ok=False);d=copy.deepcopy(old);d['delivery_revision']=new_version;d['source_footprint_proof_version']=2;d['cached_depth_parent']=parent_ref
 policy=d['policy'];policy.update({'source_grid_stride_px':'contour_adaptive_plus_interior_grid','triangle_support':RULE,'stored_depth':'exact_original_cached_sparse_prior_samples_with_barycentric_or_bounded_nearest_weights','contour_smoothing':'automatic_source_contour_only_no_hidden_completion','depth_sampling_crosses_unknown_source_pixels':False})
 protocol=copy.deepcopy(json.loads(checked(old['protocol']).read_text()));protocol.update({'version':5,'cached_depth_parent':parent_ref,'source_footprint_proof_version':2,'policy':policy,'new_CPU_depth_inference_calls':0,'component_support':'visible IDs require a retained triangle; exact source-mask union core bounds every conservatively covered pixel square','implementation_sha256':ref(Path(__file__))['sha256'],'mesh_implementation_sha256':ref(ROOT/'source_visible_boundary_mesh_v1.py')['sha256']});save(out/'protocol.json',protocol);d['protocol']=ref(out/'protocol.json');reports=[];rows=[]
 for previous in old['samples']:
  if previous['status']=='unknown':rows.append(copy.deepcopy(previous));continue
  tick=time.monotonic();row=copy.deepcopy(previous);index=row['source_sample_index'];rgb=np.asarray(Image.open(checked(row['source_RGB'])).convert('RGB'));shape=rgb.shape[:2];masks={r['track_id']:np.asarray(Image.open(checked(r)))>0 for r in row['source_masks']};zero=np.zeros(shape,bool);union=np.logical_or.reduce([masks.get(c,zero)for c in d['replacement_component_ids']])
  for c in d.get('unaffected_component_ids',[]):union&=~masks.get(c,zero)
  safe=cv2.erode(union.astype(np.uint8),np.ones((2,2),np.uint8))>0;parent=np.load(checked(previous['source_pixel_map']),allow_pickle=False)
  new=build(rgb,safe,np.asarray(d['camera']['K']),parent['source_xy'],parent['depth'],previous['source_grid_stride_px']);frame=out/f'source-{index:03d}';frame.mkdir();asset=frame/'observed-surface.glb';asset.write_bytes(exact_source_texture(unlit(trimesh.Trimesh(vertices=new['vertices'],faces=new['faces'],vertex_colors=new['colors'],process=False).export(file_type='glb')),new['source_xy'],checked(row['source_RGB']).read_bytes(),shape[1],shape[0]));mp=frame/'source-pixel-map.npz';np.savez_compressed(mp,**{k:new[k]for k in ['source_xy','depth','faces','parent_depth_vertex_ids','parent_depth_weights','depth_sampling_modes']})
  sampling={k:new['policy'][k]for k in ['maximum_nearest_sampling_distance_px','maximum_interpolation_parent_edge_px']};sampling.update({'method':'cached_monocular_prior_barycentric_or_bounded_nearest_not_measured_depth','parent_grid_stride_px':previous['source_grid_stride_px']})
  for key in ['geometry_unchanged_from_footprint_verified_parent','parent_research_asset_sha256','triangles_pruned_by_unknown_pixel_footprint']:row.pop(key,None)
  for key in ['depth_prior_model_intrinsics','valid_mask_depth_fraction','depth_scale_gauge','native_prior_translation_z']:
   if key in row:row['cached_parent_'+key]=row.pop(key)
  row.update({'reason':'exact_source_contour_adaptive_mesh_with_cached_estimated_depth_not_new_inference','topology_refined_from_cached_depth':True,'source_footprint_proof_version':2,'cached_depth_parent_map':previous['source_pixel_map'],'depth_sampling':sampling,'adaptive_mesh_policy':new['policy'],'source_grid_stride_px':new['policy']['interior_grid_stride_px'],'vertices':len(new['vertices']),'triangles':len(new['faces']),'triangles_pruned_by_geometric_and_source_footprint_gates':new['policy']['triangles_rejected'],'asset':ref(asset),'source_pixel_map':ref(mp),'visible_component_ids':visible_components(new['source_xy'],new['faces'],masks,d['replacement_component_ids']),'wall_seconds':time.monotonic()-tick,'new_CPU_depth_inference_calls':0})
  row['component_surface_support']=[{'component_id':c,'source_supported_triangle_count':int(masks[c][new['source_xy'][:,1],new['source_xy'][:,0]][new['faces']].all(1).sum())if c in masks else 0}for c in d['replacement_component_ids']]
  row.pop('provenance',None);save(frame/'provenance.json',row);row['provenance']=ref(frame/'provenance.json');rows.append(row);before=metrics(silhouette(parent['source_xy'],parent['faces'],shape),union);after=metrics(silhouette(new['source_xy'],new['faces'],shape),union);reports.append({'source_sample_index':index,'before':before,'after':after,'vertices':row['vertices'],'triangles':row['triangles'],'visible_component_ids_before':previous['visible_component_ids'],'visible_component_ids_after':row['visible_component_ids'],'seconds':row['wall_seconds']});print(name,index,'IoU',round(before['constructive_source_silhouette_IoU'],3),'->',round(after['constructive_source_silhouette_IoU'],3),'vertices',row['vertices'],flush=True)
 d['samples']=rows;d['limitations'].append('Contour adaptation improves source-facing boundary sampling only; depth is a reproducible interpolation of an existing monocular prior, not independently measured 3D or repaired native geometry.');save(out/'descriptor.json',d);exact=ref(out/'descriptor.json');proof=validate_surface_files(exact,ROOT);save(out/'publication-proof.json',proof);save(out/'boundary-quality-report.json',{'format':'interaction-twin-source-visible-boundary-upgrade-report','version':1,'descriptor':exact,'parent':parent_ref,'samples':reports,'new_API_calls':0,'new_GPU_calls':0,'new_CPU_depth_inference_calls':0,'public_adopted':False,'independent_3D_accuracy_established':False,'wall_seconds':time.monotonic()-start});return exact
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name',choices=['wallet','phone','camera']);p.add_argument('parent_version',type=int);p.add_argument('new_version',type=int);a=p.parse_args();print(json.dumps(run(a.name,a.parent_version,a.new_version)),flush=True)
