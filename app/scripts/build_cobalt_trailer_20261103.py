#!/usr/bin/env python3
"""New local Cobalt film. Saved native assets only; no inference or publication.

The earlier films and public references are left unchanged. Every image/frame,
mask, mesh and UI capture is bound in a new private production receipt. Native
triangles/UVs are rendered unchanged with the existing CPU diagnostic renderer.
Editorial crops and layouts are identified. The connected replay shows exact
saved source samples, without manufacturing missing observations or poses.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from native_appearance import load_native_glb, render_native_mesh
from sam3d_mechanisms import rotation_pose

OUT = ROOT / 'output/cobalt-trailer-20261103/v1'
MEDIA = ROOT / 'showcase/media'
NAME = 'interaction-twin-cobalt-20261103-v1'
W, H, FPS, DURATION = 1920, 1080, 24, 55
BG, INK, MUTED, BLUE, LINE, SOFT = '#f7f8fb', '#151923', '#626b7b', '#3766ed', '#dfe3eb', '#eef2ff'
PARTS = ('case_base', 'case_lid', 'earbud_a', 'earbud_b')
COLORS = {'case_base': '#3766ed', 'case_lid': '#d48d47', 'earbud_a': '#9179cb', 'earbud_b': '#409baf'}
LABELS = {'case_base': 'Case base', 'case_lid': 'Connected lid', 'earbud_a': 'Earbud A', 'earbud_b': 'Earbud B'}
CHAPTERS = [
 (0, 5, 'Show it once. Keep how it works.', 'Recorded demonstration and an editorial layout of the same four native SAM 3D components.'),
 (5, 12, 'Find the moving parts.', 'Actual saved SAM video masks and four native SAM 3D meshes. Colours identify candidate masks.'),
 (12, 24, 'Keep the connection.', 'Source-linked estimated scene replay at the saved three-frame-per-second observation rate. Unsupported parts stay absent.'),
 (24, 32, 'Geometry with a description.', 'Actual object-program fields: component IDs, estimated revolute joint, removable relationships, and source evidence.'),
 (32, 38, 'A library, built from demonstrations.', 'Actual Cobalt library screenshot and recorded examples. Human review controls what enters the curated library.'),
 (38, 44, 'For people. For their agents.', 'Actual agent setup page. Read-only JSON discovery and local stdio MCP; no hosted MCP service is claimed.'),
 (44, 51, 'Start a robot task.', 'Actual Cobalt robotics-lab screenshot using native SAM 3D meshes. Robot motion and anchors are authored; contact and physics are not simulated.'),
 (51, 55, 'Interaction Twin', 'A reusable starting point for robotics development. Shape, physical scale, contact and simulation remain unverified.'),
]


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()
def local(value):
    p = Path(value)
    if not p.is_absolute(): p = ROOT / p
    p = p.absolute(); rel = p.relative_to(ROOT)
    if '..' in rel.parts or any(x.startswith('.') for x in rel.parts) or p.resolve() != p or not p.is_file():
        raise ValueError('Only regular visible project artifacts are allowed: ' + str(rel))
    return p
def checked(ref):
    p = local(ref['path'])
    if sha(p) != ref['sha256']: raise ValueError('Input changed: ' + ref['path'])
    return p
def ref(path, kind=None, **extra):
    path = local(path)
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), bytes=path.stat().st_size,
                **({'kind': kind} if kind else {}), **extra)
def write_new(path, obj):
    with Path(path).open('x') as f: json.dump(obj, f, indent=2, allow_nan=False); f.write('\n')


@lru_cache(maxsize=100)
def font(size, bold=False, mono=False):
    path = '/System/Library/Fonts/Menlo.ttc' if mono else '/System/Library/Fonts/Supplemental/Arial' + (' Bold' if bold else '') + '.ttf'
    return ImageFont.truetype(path, size)
def txt(im, xy, value, size=30, color=INK, bold=False, mono=False, spacing=10):
    ImageDraw.Draw(im).multiline_text(xy, value, font=font(size,bold,mono), fill=color, spacing=spacing)
def rounded(im, box, fill='white', radius=25, outline=LINE, width=1):
    ImageDraw.Draw(im).rounded_rectangle(tuple(map(round,box)), radius, fill=fill, outline=outline, width=width)
def line(im, xy, fill=LINE, width=2): ImageDraw.Draw(im).line(xy, fill=fill, width=width)
def chip(im, xy, value, fill=SOFT, color=BLUE, size=22):
    x,y=xy; width=round(font(size,True).getlength(value))+30
    rounded(im,(x,y,x+width,y+42),fill,12,None);txt(im,(x+15,y+8),value,size,color,True)
    return width

def logo(im,x,y,size=42):
    # Native project vector mark, reproduced as vector-like editorial drawing.
    p=Image.new('RGBA',(192,192));d=ImageDraw.Draw(p);s=4
    d.rounded_rectangle((0,0,191,191),44,fill=BLUE)
    def points(v):return[(round(a*s),round(b*s))for a,b in v]
    d.polygon(points([(10,27),(24,20),(38,27),(24,35)]),fill='#c1d1ff')
    d.polygon(points([(10,27),(11,36),(24,43),(24,35)]),fill='#eff4ff')
    d.polygon(points([(38,27),(37,36),(24,43),(24,35)]),fill='#eff4ff')
    d.line(points([(10,13),(24,6),(38,13),(24,21),(10,13)]),fill='white',width=6,joint='curve')
    d.line(points([(11,14),(11,19),(24,26),(37,19),(37,14)]),fill='white',width=6,joint='curve')
    d.line(points([(10,24),(38,24)]),fill='#a6bdff',width=4);d.ellipse((30,86,50,106),fill='white')
    im.paste(p.resize((size,size),Image.Resampling.LANCZOS),(x,y),p.resize((size,size),Image.Resampling.LANCZOS))

def base(kicker, title, subtitle=None, *, dark=False):
    bg=INK if dark else BG;fg='white'if dark else INK
    im=Image.new('RGB',(W,H),bg);logo(im,68,46);txt(im,(123,51),'Interaction Twin',27,fg,True)
    txt(im,(68,130),kicker.upper(),21,'#a9bbf7'if dark else BLUE,True)
    txt(im,(64,169),title,68,fg,True,spacing=6)
    if subtitle:txt(im,(68,264),subtitle,27,'#bcc5d8'if dark else MUTED)
    return im

def footer(im,text,dark=False):
    line(im,[(68,1008),(1852,1008)],'#30394c'if dark else LINE)
    txt(im,(68,1028),text,21,'#b7c1d6'if dark else MUTED)

def paste_card(im,source,box,*,fill='white',cover=False,radius=24,crop=None):
    x,y,w,h=map(round,box)
    source=source.convert('RGB');source=source.crop(crop)if crop else source
    p=Image.new('RGB',(w,h),fill)
    img=(ImageOps.fit if cover else ImageOps.contain)(source,(w,h),Image.Resampling.LANCZOS)
    p.paste(img,((w-img.width)//2,(h-img.height)//2))
    mask=Image.new('L',(w,h));ImageDraw.Draw(mask).rounded_rectangle((0,0,w-1,h-1),radius,fill=255)
    im.paste(p,(x,y),mask);rounded(im,(x,y,x+w-1,y+h-1),None,radius,LINE)

def cross_image(im,box,image,fraction):
    # Editorial screenshot pan/zoom only, not model/object motion.
    x,y,w,h=box;image=image.convert('RGB');cw=image.width;ch=image.height
    scale=1+.035*fraction
    sw=int(cw/scale);sh=int(ch/scale)
    crop=(int((cw-sw)*fraction),int((ch-sh)*fraction),int((cw-sw)*fraction)+sw,int((ch-sh)*fraction)+sh)
    paste_card(im,image,box,crop=crop)


def composite(meshes, components, k, width, height, sample_index):
    depth=np.full((height,width),np.inf,np.float32);rgb=np.full((height,width,3),(239,242,249),np.uint8)
    supported=[]
    for c in components:
        row=next(r for r in c['poses']if r['source_sample_index']==sample_index)
        if row['status']!='estimated' or row['camera_from_native_mesh']is None:continue
        r,t,scale=rotation_pose(row['camera_from_native_mesh'])
        rendered=render_native_mesh(meshes[c['id']],k,r,t,width,height,scale=scale)
        mask=rendered['mask']&(rendered['depth']<depth)
        rgb[mask]=rendered['rgb'][mask];depth[mask]=rendered['depth'][mask];supported.append(c['id'])
    return Image.fromarray(rgb),supported


def prepare():
    if (OUT/'preparation.json').exists():
        p=read(OUT/'preparation.json')
        for r in p['inputs']:checked(r)
        for r in p['outputs']:checked(r)
        print(json.dumps({'reused_verified_preparation':True,'output':str(OUT)}));return
    OUT.mkdir(parents=True,exist_ok=True);assets=OUT/'assets';assets.mkdir()
    registry=local('showcase/public/aligned-scenes.json');reg=read(registry)
    assert len(reg['scenes'])==1,'Explicit current connected-scene selection required'
    entry=reg['scenes'][0];scene_path=checked(entry['manifest']);scene=read(scene_path)
    program_path=checked(scene['original_program']);program=read(program_path)
    object_path=checked(entry['object_program']);obj=read(object_path)
    assert program['source']['sha256']==scene['source']['sha256']==entry['source_sha256']==obj['source']['sha256']
    assert obj['constrained_scene']['sha256']==entry['manifest']['sha256']
    assert not scene['physical_accuracy_verified'] and not scene['simulation_ready']
    assert {c['id']for c in scene['components']}==set(PARTS)
    catalog=read(local('showcase/public/cases-index.json'))
    assert any(c['interaction']['source']['sha256']==program['source']['sha256']for c in catalog['cases'])
    source=checked(scene['source']);inputs=[ref(registry,'current_curated_connected_scene_registry'),ref(scene_path,'current_connected_native_scene'),ref(program_path,'source_linked_interaction_program'),ref(object_path,'descriptive_object_program'),ref(source,'source_recording')]
    snapshots={key:local(value)for key,value in {
      'library':'output/cobalt-studio-20261102/library-preview.png',
      'agents':'output/overnight-pipeline-20261102/agents-one-line.png',
      'robotics':'output/cobalt-studio-20261102/robotics-lab.png'}.items()}
    for key,path in snapshots.items():inputs.append(ref(path,'actual_Cobalt_UI_capture',screen=key))
    music=local('output/trailer-20261102/v4/original-instrumental-only.wav');inputs.append(ref(music,'original_procedural_project_instrumental'))
    meshes={}
    for c in scene['components']:
        asset=checked(c['asset']);inputs.append(ref(asset,'unchanged_native_SAM3D_glb',component=c['id']))
        meshes[c['id']]=load_native_glb(asset)
    outputs=[];sequence=[]
    # No temporal fit, interpolation, correction or invented hidden state: the
    # camera replay uses the exact saved 3fps observation matrices.
    indices=list(range(37,62))+list(range(94,102))
    k=np.asarray(scene['camera']['K'],float)*(.75);k[2,2]=1
    for n,index in enumerate(indices):
        evidence=next(e for e in program['evidence']if e['source_index']==index)
        frame=checked({'path':evidence['frame_path'],'sha256':evidence['frame_sha256']})
        inputs.append(ref(frame,'actual_source_frame',source_index=index,timestamp_s=evidence['timestamp_s']))
        rgb=np.asarray(Image.open(frame).convert('RGB'));overlay=rgb.copy();masks=[]
        for obs in evidence['observations']:
            path=checked({'path':obs['mask_path'],'sha256':obs['mask_sha256']});inputs.append(ref(path,'actual_saved_SAM_mask',component=obs['track_id'],source_index=index))
            if obs['audit_status']=='reject_wrong_or_lost_mask':continue
            mask=np.asarray(Image.open(path).convert('L'))>127;color=np.array(tuple(int(COLORS[obs['track_id']][j:j+2],16)for j in(1,3,5)))
            overlay[mask]=np.rint(overlay[mask]*.64+color*.36).astype(np.uint8)
            edge=cv2.morphologyEx(mask.astype('uint8'),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0;overlay[edge]=color
            masks.append(obs['track_id'])
        mask_path=assets/f'masks-{index:03}.png';Image.fromarray(overlay).save(mask_path)
        native,supported=composite(meshes,scene['components'],k,405,720,index)
        native_path=assets/f'connected-{index:03}.png';native.save(native_path)
        outputs.extend([ref(mask_path,'actual_mask_tint_editorial_palette'),ref(native_path,'full_native_triangle_scene_render')])
        sequence.append({'source_index':index,'timestamp_s':evidence['timestamp_s'],'source_frame':ref(frame),'mask_overlay':ref(mask_path),'native_render':ref(native_path),'supported_components':supported,'mask_components':masks})
        if n%5==0:print(json.dumps({'prepared_scene_samples':n+1,'of':len(indices)}),flush=True)
    # Isolated native parts use only an editorial camera/layout. Their original
    # mesh coordinates and textures are not rewritten or mesh-decimated.
    gallery={}
    for c in scene['components']:
        mesh=meshes[c['id']];row=next(r for r in c['poses']if r['source_sample_index']==41)
        rot,_,_=rotation_pose(row['camera_from_native_mesh']);pts=mesh.vertices@rot.T
        middle=(pts.min(0)+pts.max(0))/2;extent=pts.max(0)-pts.min(0)
        focal=520.;z=max(extent[0]/(370/focal),extent[1]/(260/focal))+.55*extent[2]
        trans=-middle+np.array([0,0,z]);kg=np.array([[focal,0,240],[0,focal,170],[0,0,1]])
        render=render_native_mesh(mesh,kg,rot,trans,480,340)
        pixels=np.full((340,480,3),(239,242,249),np.uint8);pixels[render['mask']]=render['rgb'][render['mask']]
        path=assets/f'native-{c["id"]}.png';Image.fromarray(pixels).save(path);gallery[c['id']]=ref(path,'native_part_editorial_camera')
        outputs.append(gallery[c['id']])
    # Readable projection of real fields, explicitly not a new file-format claim.
    joint=obj['joints'][0]
    excerpt={'format':obj['format'],'components':[c['component_id']for c in obj['component_assets']],
        'joint':{k:joint[k]for k in('type','parent_id','child_id','status','axis','limits_rad')},
        'removable_relationships':obj['removable_relationships'],
        'source_sha256':obj['source']['sha256'],'physical_accuracy_verified':obj['physical_accuracy_verified'],'simulation_ready':obj['simulation_ready']}
    excerpt_path=OUT/'actual-object-program-excerpt.json';write_new(excerpt_path,excerpt);outputs.append(ref(excerpt_path,'readable_projection_of_actual_object_program_fields'))
    builder=OUT/'builder-at-prepare.py';shutil.copyfile(__file__,builder)
    # Deduplicate before source manifest; source hashes are rechecked before encode.
    inputs=list({r['path']:r for r in inputs}.values())
    write_new(OUT/'preparation.json',{'format':'interaction-twin-cobalt-film-preparation','version':1,'inputs':inputs,'outputs':outputs,
        'builder':ref(builder),'source':ref(source),'program':ref(program_path),'object_program':ref(object_path),'scene':ref(scene_path),
        'gallery':gallery,'sequence':sequence,'screenshots':{key:ref(path)for key,path in snapshots.items()},'music':ref(music),
        'chapters':[dict(start_s=a,end_s=b,title=c,scope=d)for a,b,c,d in CHAPTERS],
        'render':{'width':W,'height':H,'fps':FPS,'duration_s':DURATION,'scene_render_width':405,'scene_render_height':720,'scene_observation_rate_hz':3,'source_crop_xyxy':[0,260,540,800],'native_crop_xyxy':[0,195,405,600]},
        'scope':{'new_model_calls':0,'new_API_calls':0,'new_GPU_jobs':0,'source_GLBS_and_poses_modified':False,'native_triangles_decimated':False,'camera_orbits_invented':False,'layout_and_UI_pans_editorial':True,'robot_anchors_and_motion_authored_not_physics':True,'physical_accuracy_verified':False,'simulation_ready':False,'published':False}})
    print(json.dumps({'prepared':True,'samples':len(sequence),'inputs':len(inputs),'native_parts':len(gallery)}),flush=True)


@lru_cache(maxsize=160)
def img(path):return Image.open(checked(path)if isinstance(path,dict)else local(path)).convert('RGB')
# dict refs cannot be lru keys; callers use declared exact paths after prepare verification.
def pic(ref):return img(ref['path'])

def sequence_row(prep,clock):
    return min(prep['sequence'],key=lambda r:abs(r['timestamp_s']-clock))

def timestamp(value):return f'{value:05.2f} s'

def decoded_source_frame(prep,clock,fallback):
    manifest=prep.get('_decoded_clips')
    if not manifest:return fallback
    name='handling'if clock<25 else'closing'
    clip=next(c for c in manifest['clips']if c['name']==name)
    i=round((clock-clip['source_start_s'])*clip['fps'])
    return pic(clip['frames'][max(0,min(len(clip['frames'])-1,i))])

def scene_pair(im,prep,clock,y=350,height=590):
    row=sequence_row(prep,clock)
    source=decoded_source_frame(prep,clock,pic(row['source_frame']))
    for x,title,picture,crop in[(68,'CAPTURED VIDEO',source,(0,260,540,800)),(992,'CONNECTED SCENE · ESTIMATED',pic(row['native_render']),(0,195,405,600))]:
        rounded(im,(x,y,x+860,y+height),'white',24);txt(im,(x+26,y+21),title,20,BLUE,True)
        paste_card(im,picture,(x+18,y+60,824,height-122),crop=crop,fill='#eff2f9',radius=16)
        label=f'Video {timestamp(clock)}'if x==68 else f'Saved sample {timestamp(row["timestamp_s"])}'
        txt(im,(x+26,y+height-44),label,21,MUTED)
    missing=set(PARTS)-set(row['supported_components'])
    if missing:txt(im,(1018,y+height-44),f'{len(row["supported_components"])} supported parts · unknowns withheld',21,MUTED)
    else:txt(im,(1310,y+height-44),'4 native parts',21,MUTED)


def film_frame(prep,t):
    if t<5:
        im=base('Demonstrations, made reusable.','Show it once.\nKeep how it works.')
        txt(im,(70,410),'Turn captured interactions into\n3D parts, connections and evidence.',36,MUTED,spacing=14)
        for j,label in enumerate(['SAM 3.1','Muse','SAM 3D Objects']):chip(im,(70+j*218,565),label)
        rounded(im,(70,678,690,898),'white',24)
        txt(im,(98,710),'A starting point for robotics.',32,INK,True)
        txt(im,(98,766),'A descriptive asset — with the\nrecording behind every estimate.',27,MUTED,spacing=10)
        row=sequence_row(prep,16.67)
        paste_card(im,pic(row['native_render']),(810,130,1040,790),crop=(0,195,405,600),fill='#eff2f9')
        chip(im,(839,159),'Native SAM 3D components')
        txt(im,(840,859),'Saved connected scene · estimated',23,MUTED)
        footer(im,'Research prototype  /  Recorded evidence  /  Native meshes  /  Descriptive programs')
    elif t<12:
        q=t-5;im=base('01 / Perceive + reconstruct','Find the moving parts.','SAM 3.1 tracks the recording. SAM 3D Objects generates each component.')
        row=sequence_row(prep,12+q)
        paste_card(im,pic(row['mask_overlay']),(68,335,755,610),crop=(0,260,540,800))
        chip(im,(90,880),'Actual masks · candidate boundaries',size=21)
        for j,identifier in enumerate(PARTS):
            x=882+(j%2)*490;y=335+(j//2)*305
            paste_card(im,pic(prep['gallery'][identifier]),(x,y,470,280),fill='#eff2f9')
            chip(im,(x+16,y+227),LABELS[identifier],size=19)
        footer(im,'Four unchanged native meshes  /  Editorial component layout  /  Generated geometry remains unverified')
    elif t<24:
        q=t-12;im=base('02 / Connect + replay','Keep the connection.','A persistent hinge keeps the lid attached. Removable components remain separate.')
        clock=12+q if q<8 else 30.67+(q-8)*(.667)
        scene_pair(im,prep,clock)
        footer(im,'Video + exact saved 3fps estimates  /  Final closing shot slowed  /  Unknown parts withheld; depth and angles estimated')
    elif t<32:
        q=(t-24)/8;im=base('03 / Describe','Geometry with a description.','Components, joints, actions and the frames that explain them.')
        row=sequence_row(prep,19.0)
        paste_card(im,pic(row['source_frame']),(68,340,630,585),crop=(0,260,540,800))
        chip(im,(92,363),'source_000058 · 19.00 s',size=21)
        rounded(im,(750,340,1852,935),'#151e32',24,None)
        code=[('{', '#e9efff'),('  "components": [', '#a9bfff'),('    "case_base", "case_lid",', '#e9efff'),('    "earbud_a", "earbud_b"', '#e9efff'),('  ],', '#a9bfff'),('  "joint": {', '#a9bfff'),('    "type": "revolute",', '#e9efff'),('    "parent_id": "case_base",', '#e9efff'),('    "child_id": "case_lid",', '#e9efff'),('    "status": "estimated"', '#a9c8ff'),('  }', '#a9bfff'),('}', '#e9efff')]
        for j,(value,color)in enumerate(code):txt(im,(790,370+j*42),value,27,color,mono=True)
        footer(im,'Readable excerpt of the actual object program  /  Native frames, joint axis, angle samples and source hashes retained')
    elif t<38:
        q=(t-32)/6;im=base('04 / Curate + discover','A library, built from demonstrations.','Browse by object, action or mechanism. Inspect the recording and download the twin.')
        cross_image(im,(68,335,1240,610),pic(prep['screenshots']['library']),q)
        for j,(number,title,note)in enumerate([('01','Inspect','Video + connected scene'),('02','Understand','Parts + descriptive program'),('03','Reuse','Native meshes + evidence')]):
            y=370+j*177;txt(im,(1370,y),number,26,BLUE,True);txt(im,(1370,y+46),title,36,INK,True);txt(im,(1370,y+101),note,23,MUTED)
        footer(im,'Actual Cobalt library view  /  Human review controls the curated collection')
    elif t<44:
        q=(t-38)/6;im=base('05 / Agent access','For people. For their agents.','One shared library. Source-linked programs. Read-only discovery.')
        cross_image(im,(68,340,1280,590),pic(prep['screenshots']['agents']),q)
        rounded(im,(1392,340,1852,930),'#151e32',24,None)
        txt(im,(1424,382),'LOCAL MCP TOOLS',21,'#a9bfff',True)
        for j,(tool,detail)in enumerate([('search_twins','Find an interaction'),('get_kinematics','Read the object program'),('get_evidence','Retrieve its source')]):
            txt(im,(1424,460+j*130),tool,25,'white',mono=True);txt(im,(1424,504+j*130),detail,22,'#b7c1d6')
        footer(im,'Actual setup page  /  Read-only JSON access  /  MCP runs in a local checkout; no hosted MCP is claimed')
    elif t<51:
        q=(t-44)/7;im=base('06 / Robotics authoring','Start a robot task.','Choose components. Stage a removal and reinsertion. Export a task draft.')
        cross_image(im,(68,330,1450,620),pic(prep['screenshots']['robotics']),q)
        txt(im,(1560,410),'Parts',35,INK,True);txt(im,(1560,459),'Native SAM 3D',23,MUTED)
        txt(im,(1560,565),'Task',35,INK,True);txt(im,(1560,614),'Take out → put back',23,MUTED)
        chip(im,(1556,760),'Kinematic preview',size=20)
        footer(im,'Actual robotics-lab UI  /  Robot, anchors and motion are authored  /  Contact and physics are not simulated')
    else:
        q=t-51;im=base('From demonstrations to reusable interactions.','Interaction Twin',dark=True)
        txt(im,(70,328),'For the next builder.\nAnd their robot.',70,'white',True,spacing=8)
        row=sequence_row(prep,19.0)
        paste_card(im,pic(row['native_render']),(1120,228,730,670),crop=(0,195,405,600),fill='#eff2f9')
        for j,value in enumerate(['3D parts','Connected motion','Source-linked meaning']):chip(im,(70,592+j*66),value,fill='#28395e',color='#d5dfff',size=25)
        txt(im,(70,875),'Designed & built by Nigel Hartman',27,'#b7c1d6')
        footer(im,'Research prototype  /  Physical scale, contact and simulation readiness require validation',True)
    # Smooth editorial chapter transitions; no fabricated object motion.
    start=max(a for a,_,_,_ in CHAPTERS if a<=t)
    fade=min(1.,(t-start)/.22)if start else min(1.,t/.35)
    if fade<1:im=Image.blend(Image.new('RGB',(W,H),BG if t<51 else INK),im,fade)
    return im


def encode():
    prep=read(OUT/'preparation.json')
    for r in prep['inputs']+prep['outputs']:checked(r)
    clips_path=local(OUT/'decoded-source-clips.json');clips=read(clips_path)
    assert clips['source']['sha256']==prep['source']['sha256']
    checked(clips['source'])
    for clip in clips['clips']:
        assert clip['fps']==FPS and clip['frames']
        for row in clip['frames']:checked(row)
    prep['_decoded_clips']=clips
    # Explicit provenance for the visible model labels. No models are run here.
    mask_metadata=local('data/flow-sam3d-default-e2e-20261102-v1/segmentation/flow-components-1/metadata.json')
    assert read(mask_metadata)['model']=='sam-3.1'
    program=read(checked(prep['program']))
    assert program['reasoning']['provider']=='meta_api'and program['reasoning']['model'].startswith('muse-')
    encode_builder=OUT/'builder-at-encode.py';shutil.copyfile(__file__,encode_builder)
    MEDIA.mkdir(exist_ok=True)
    silent=OUT/'picture.mp4';target=MEDIA/(NAME+'.mp4')
    if silent.exists()or target.exists():raise ValueError('New version name required; existing films are never overwritten')
    command=['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(silent)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    proof_times=[2.5,8.,16.,26.,35.,41.,47.,53.]
    proofs=[];started=time.monotonic()
    try:
        for n in range(DURATION*FPS):
            t=n/FPS;im=film_frame(prep,t)
            if any(abs(t-x)<.5/FPS for x in proof_times):
                path=OUT/f'proof-{int(t*10):03}.jpg';im.save(path,quality=95);proofs.append(ref(path,'film_proof_frame',film_time_s=t))
            process.stdin.write(np.asarray(im,dtype=np.uint8).tobytes())
            if n%(FPS*5)==0:print(json.dumps({'encoded_seconds':int(t),'wall_seconds':round(time.monotonic()-started,1)}),flush=True)
        process.stdin.close();code=process.wait()
        if code:raise RuntimeError(f'ffmpeg failed: {code}')
    except BaseException:
        process.kill();process.wait();raise
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(silent),'-i',str(checked(prep['music'])),'-map','0:v','-map','1:a','-t',str(DURATION),'-c:v','copy','-af',f'afade=t=out:st={DURATION-2.5}:d=2.5,loudnorm=I=-20:TP=-2:LRA=7','-c:a','aac','-b:a','192k','-movflags','+faststart',str(target)],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(target)],text=True))
    assert abs(float(probe['format']['duration'])-DURATION)<.1
    assert len([s for s in probe['streams']if s['codec_type']=='video'])==1
    assert len([s for s in probe['streams']if s['codec_type']=='audio'])==1
    video=next(s for s in probe['streams']if s['codec_type']=='video');assert(video['width'],video['height'])==(W,H)
    poster=MEDIA/(NAME+'-poster.jpg');film_frame(prep,2.5).save(poster,quality=96)
    sheet=Image.new('RGB',(1280,4*386),BG)
    for j,r in enumerate(proofs):
        x=(j%2)*640;y=(j//2)*386;thumb=Image.open(checked(r)).resize((640,360),Image.Resampling.LANCZOS);sheet.paste(thumb,(x,y));txt(sheet,(x+10,y+363),f'{r["film_time_s"]:.1f}s · {CHAPTERS[j][2]}',16,MUTED)
    sheetpath=OUT/'storyboard-proof.jpg';sheet.save(sheetpath,quality=96)
    captions=['WEBVTT','']
    for a,b,title,scope in CHAPTERS:
        def stamp(x):return f'00:{int(x)//60:02}:{int(x)%60:02}.000'
        captions.extend([f'{stamp(a)} --> {stamp(b)}',title,scope,''])
    vtt=MEDIA/(NAME+'.vtt');vtt.write_text('\n'.join(captions))
    write_new(OUT/'verification.json',{'format':'interaction-twin-cobalt-film-verification','version':1,'film':ref(target),'poster':ref(poster),'captions':ref(vtt),'storyboard':ref(sheetpath),'proof_frames':proofs,'ffprobe':probe,'decoded_source_clips':ref(clips_path),'mask_model_metadata':ref(mask_metadata),'builder_at_encode':ref(encode_builder),'new_model_or_API_calls':0,'published':False,'landing_references_changed':False,'original_films_changed':False,'music_rights':'Original procedural project composition; no third-party samples. Instrumental only; no speech.','render_wall_seconds':time.monotonic()-started})
    support=OUT/'index.html'
    support.write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cobalt trailer review · Interaction Twin</title><style>body{margin:0;background:#f7f8fb;color:#151923;font:17px -apple-system,BlinkMacSystemFont,Arial,sans-serif}main{max-width:1120px;margin:60px auto;padding:24px}small{color:#3766ed;font-weight:700;letter-spacing:.12em}h1{font-size:46px;letter-spacing:-.04em;margin:18px 0}p{line-height:1.7;color:#626b7b;max-width:900px}video,img{width:100%;border-radius:20px;border:1px solid #dfe3eb;background:white}a{color:#3766ed}section{margin:40px 0}.note{background:white;border:1px solid #dfe3eb;border-radius:20px;padding:24px}li{margin:12px 0;line-height:1.6}</style><main><small>LOCAL REVIEW · 55 SECONDS · CURRENT NATIVE ASSETS</small><h1>Show it once. Keep how it works.</h1><p>A new Cobalt film made from the actual source recording, saved SAM masks, four native SAM 3D meshes, a connected estimated scene and current app captures. Existing public films and landing references remain unchanged.</p><video controls playsinline preload="metadata" poster="../../../showcase/media/'''+NAME+'''-poster.jpg"><source src="../../../showcase/media/'''+NAME+'''.mp4" type="video/mp4"><track kind="subtitles" srclang="en" label="English" src="../../../showcase/media/'''+NAME+'''.vtt"></video><section><a href="../../../showcase/media/'''+NAME+'''.mp4" download>Download MP4</a> · <a href="verification.json">Verification &amp; provenance</a> · <a href="preparation.json">Exact input manifest</a></section><section class="note"><strong>What the film shows</strong><ul><li>Source-linked masks and unchanged native textured geometry.</li><li>Exact saved three-fps estimated scene samples. Unsupported poses remain absent; no synthetic observations are added.</li><li>Editorial crops, component layouts and screenshot pans.</li><li>An authored kinematic robotics-lab rehearsal. Contact, physical scale and physics are unverified.</li><li>Read-only agent discovery and local stdio MCP, without implying a hosted MCP service.</li></ul></section><section><h2>Storyboard proof</h2><img src="storyboard-proof.jpg" alt="Eight proof frames from the 55-second Cobalt trailer"></section><p>Designed and built by Nigel Hartman. Research prototype. Original procedural instrumental; no third-party music or API inference used for this film.</p></main>''')
    print(json.dumps({'complete':True,'film':str(target),'review':str(support),'duration_s':DURATION,'proof':str(sheetpath)}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','encode','all']);args=parser.parse_args()
    if args.command in ('prepare','all'):prepare()
    if args.command in ('encode','all'):encode()
if __name__=='__main__':main()
