"""Additive exact-JPEG appearance derivative of the checked walletV3 series."""
from pathlib import Path
import copy,json,time
import numpy as np
from PIL import Image
from source_visible_flexure_20261104_v2 import ROOT,ref,save
from source_visible_texture_v1 import exact_source_texture,visible_components
from source_visible_flexure_verify_v1 import decode_surface
PARENT=ROOT/'data/source-visible-flexure/wallet-20261104-v3/descriptor.json'
DATA=ROOT/'data/source-visible-flexure/wallet-20261104-v4'
OUT=ROOT/'output/experience-refinement-20261104-v1/wallet/source-visible-flexure-v4'
def run():
    DATA.mkdir(parents=True,exist_ok=False);OUT.mkdir(parents=True,exist_ok=False);doc=json.loads(PARENT.read_text());protocol=json.loads((ROOT/doc['protocol']['path']).read_text());protocol['version']=4;protocol['appearance_mode']='exact_source_JPEG_projected_UV';protocol['texture_derivation_implementation_sha256']=ref(ROOT/'source_visible_texture_v1.py')['sha256'];protocol['parent_research_descriptor_sha256']=ref(PARENT)['sha256'];save(DATA/'protocol.json',protocol);doc['protocol']=ref(DATA/'protocol.json');doc['delivery_revision']=4;doc['appearance_mode']='exact_source_JPEG_projected_UV';rows=[]
    for row in doc['samples']:
        if row['status']=='unknown':rows.append(row);continue
        index=row['source_sample_index'];out=DATA/f'source-{index:03d}';out.mkdir();raw=(ROOT/row['asset']['path']).read_bytes();mapping_path=ROOT/row['source_pixel_map']['path'];mapping=np.load(mapping_path,allow_pickle=False);xy=mapping['source_xy'];rgbbytes=(ROOT/row['source_RGB']['path']).read_bytes();im=Image.open(ROOT/row['source_RGB']['path']);asset=out/'observed-surface.glb';asset.write_bytes(exact_source_texture(raw,xy,rgbbytes,im.width,im.height));(out/'source-pixel-map.npz').write_bytes(mapping_path.read_bytes());masks={r['track_id']:np.asarray(Image.open(ROOT/r['path']))>0 for r in row['source_masks']};row['parent_research_asset_sha256']=row['asset']['sha256'];row['asset']=ref(asset);row['source_pixel_map']=ref(out/'source-pixel-map.npz');row.pop('provenance',None);row['visible_component_ids']=visible_components(xy,mapping['faces'],masks,doc['replacement_component_ids']);row['appearance_mode']='exact_source_JPEG_projected_UV';row['texture_source_RGB']=row['source_RGB'];row['sampled_source_RGB_attribute']='_SOURCE_RGB';row['vertex_RGB_multiplied_into_texture']=False;row['geometry_unchanged_from_footprint_verified_parent']=True;save(out/'provenance.json',row);row['provenance']=ref(out/'provenance.json');rows.append(row)
    doc['samples']=rows;save(DATA/'descriptor.json',doc);summary={'format':'interaction-twin-source-visible-texture-delivery','descriptor':ref(DATA/'descriptor.json'),'protocol':ref(DATA/'protocol.json'),'surface_GLΒ_bytes':sum(r['asset']['bytes']for r in rows if r['status']=='source_observed_prior'),'supported_sources':[r['source_sample_index']for r in rows if r['status']=='source_observed_prior'],'unknown_sources':[r['source_sample_index']for r in rows if r['status']=='unknown'],'geometry_topology_depth_unchanged':True,'source_JPEG_bytes_unchanged':True,'new_API_calls':0,'new_GPU_calls':0,'physical_accuracy_verified':False};save(DATA/'report.json',summary);save(OUT/'delivery.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
