"""Review renders for the explicitly typed observed surface, not native SAM."""
from pathlib import Path
import io,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
from native_appearance import NativeMesh,NativeAppearance,NativePrimitive,render_native_mesh,_srgb_to_linear
from source_visible_flexure_verify_v1 import decode_surface
from source_visible_flexure_20261104_v2 import ROOT,ref,save
def preview_mesh(raw):
    d=decode_surface(raw);g=d['document'];iv=g['bufferViews'][g['images'][0]['bufferView']];start=iv.get('byteOffset',0);rgb=np.asarray(Image.open(io.BytesIO(d['binary'][start:start+iv['byteLength']])).convert('RGB'))/255
    part=NativePrimitive(vertices=d['vertices'],faces=d['faces'],uv=d['UV'],tint=np.ones((len(d['vertices']),3)),factor=np.ones(3),texture_linear=_srgb_to_linear(rgb),sampler={'filter':'linear','wrap_s':33071,'wrap_t':33071},face_offset=0,double_sided=False,winding_sign=1)
    return NativeMesh(vertices=d['vertices'],faces=d['faces'],vertex_colors=None,appearance=NativeAppearance((part,),False),receipt={'explicit_observed_surface_adapter':True,'native_SAM':False})
def board(row,doc,path):
    rgb=np.asarray(Image.open(ROOT/row['source_RGB']['path']).convert('RGB'));h,w=rgb.shape[:2];mesh=preview_mesh((ROOT/row['asset']['path']).read_bytes());K=np.asarray(doc['camera']['K']);imgs=[rgb]
    for angle in [0,24]:
        R=Rotation.from_euler('y',angle,degrees=True).as_matrix();center=mesh.vertices.mean(0);t=center-R@center;render=render_native_mesh(mesh,K,R,t,w,h);im=np.full_like(rgb,[238,241,247]);im[render['mask']]=render['rgb'][render['mask']];imgs.append(im)
    b=Image.new('RGB',(w*3,h+85),'#f6f8fc');draw=ImageDraw.Draw(b);font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',20)
    for i,(im,text)in enumerate(zip(imgs,['Actual source video','Exact sourceJPEG · learned2.5Ddepth','Same partial surface · orbit24°'])):b.paste(Image.fromarray(im),(i*w,45));draw.text((i*w+10,12),text,fill='#172033',font=font)
    draw.text((10,h+53),f'Source{row["source_sample_index"]:03d} · observedRGBreprojection; learneddepthuncertain, hiddenregionsabsent',fill='#172033',font=font);b.save(path)
def run():
    p=ROOT/'data/source-visible-flexure/wallet-20261104-v4/descriptor.json';doc=json.loads(p.read_text());out=ROOT/'output/experience-refinement-20261104-v1/wallet/source-visible-flexure-v4';images=[]
    for index in [14,27,28,99]:
        row=next(r for r in doc['samples']if r['source_sample_index']==index);path=out/f'comparison-{index:03d}.png';board(row,doc,path);images.append({'source_sample_index':index,'image':ref(path),'source_RGB':row['source_RGB'],'surface_asset':row['asset'],'visible_component_ids':row['visible_component_ids']})
    packet={'format':'interaction-twin-observed-surface-visual-review-packet','descriptor':ref(p),'images':images,'appearance':'exactsourceJPEG_UVnotindependentgeometryevidence','depth':'CPUlearnedMoGeprior_notmeasured','hidden_geometry':'absent_notinpainted','joint_parameters':'unknown','new_API_calls':0,'new_GPU_calls':0,'physical_accuracy_verified':False};save(out/'visual-review-packet.json',packet);print(json.dumps(packet,indent=2))
if __name__=='__main__':run()
