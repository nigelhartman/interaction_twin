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

OUT = ROOT / 'output/cobalt-trailer-20261103/v2'
MEDIA = ROOT / 'showcase/media'
NAME = 'interaction-twin-cobalt-20261103-v2'
W, H, FPS, DURATION = 1920, 1080, 24, 55
BG, INK, MUTED, BLUE, LINE, SOFT = '#f7f8fb', '#151923', '#626b7b', '#3766ed', '#dfe3eb', '#eef2ff'
PARTS = ('case_base', 'case_lid', 'earbud_a', 'earbud_b')
COLORS = {'case_base': '#3766ed', 'case_lid': '#d48d47', 'earbud_a': '#9179cb', 'earbud_b': '#409baf'}
LABELS = {'case_base': 'Case base', 'case_lid': 'Connected lid', 'earbud_a': 'Earbud A', 'earbud_b': 'Earbud B'}
CHAPTERS = [
 (0, 5, 'Show it once. Keep how it works.', 'A source-linked connected scene of four unchanged native SAM 3D components.'),
 (5, 13, 'Find the moving parts.', 'Actual saved SAM video masks and four native SAM 3D meshes. Colours identify candidate masks.'),
 (13, 21, 'Keep the connection.', 'Source 15–20 seconds played at 0.625× with exact nearest saved 3fps scene estimates; no poses are fabricated.'),
 (21, 29, 'Geometry with a description.', 'Actual object-program fields: component IDs, estimated revolute joint, removable relationships, and source evidence.'),
 (29, 36, 'A library, built from demonstrations.', 'Actual Cobalt library screenshot. Human review controls what enters the curated library.'),
 (36, 43, 'For people. For their agents.', 'Actual setup page. Read-only JSON discovery and local stdio MCP; no hosted MCP service is claimed.'),
 (43, 51, 'Start a robot task.', 'Actual timed screenshot sequence of the current robotics-lab playback. Robot motion and anchors are authored; contact and physics are not simulated.'),
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
        for row in p['inputs']+p['outputs']:checked(row)
        print(json.dumps({'reused_verified_preparation':True}));return
    original=local('output/cobalt-trailer-20261103/v1/preparation.json');p=read(original)
    for row in p['inputs']+p['outputs']:checked(row)
    OUT.mkdir(parents=True)
    receipt=local('output/overnight-pipeline-20261102/robot-ui-capture/capture.json');robot=read(receipt)
    assert robot['kind']=='actual_cua_browser_screenshot_sequence'and robot['authored_robot_not_physics']is True
    assert 'flow-sam3d-default-e2e-20261102-v1--flow-procedure-1'in robot['url']
    assert len(robot['frames'])==151 and robot['frames'][0]['captured_ms']==0
    assert all(b['captured_ms']>a['captured_ms']for a,b in zip(robot['frames'],robot['frames'][1:]))
    rows=[]
    for frame in robot['frames']:
        path=local(receipt.parent/frame['file'])
        assert Image.open(path).size==(1280,720)
        rows.append(dict(**ref(path,'actual_CUA_UI_screenshot'),captured_ms=frame['captured_ms']))
    p['inputs'].extend([ref(original,'reused_exact_native_asset_preparation'),ref(receipt,'actual_timed_CUA_capture_receipt'),*rows])
    p['robot_capture']={'receipt':ref(receipt),'frames':rows,'recorded_elapsed_ms':robot['elapsed_ms'],'capture_last_frame_ms':rows[-1]['captured_ms'],'editorial_playback_duration_s':8,'authored_robot_not_physics':True}
    p['chapters']=[dict(start_s=a,end_s=b,title=c,scope=d)for a,b,c,d in CHAPTERS]
    p['render']['source_crop_xyxy']='Shared frame-dependent editorial vertical crop, derived only from the saved native render; identical source-image coordinates.'
    p['render']['native_crop_xyxy']='Full native image width, editorial vertical crop includes all rasterized native parts; no pose or mesh changes.'
    p['render']['source_clock_excerpt']={'start_s':15,'end_s':20,'playback_rate':.625,'selection':'continuous saved interval with all four parts supported'}
    p['scope']['robot_segment']='Actual timed CUA playback sequence, sampled by captured_ms and compressed from 8.452s into 8s.'
    builder=OUT/'builder-at-prepare.py';shutil.copyfile(__file__,builder);p['builder']=ref(builder)
    write_new(OUT/'preparation.json',p)
    clips=local('output/cobalt-trailer-20261103/v1/decoded-source-clips.json')
    shutil.copyfile(clips,OUT/'decoded-source-clips.json')
    print(json.dumps({'prepared':True,'native_renders_reused_verified':len(p['sequence']),'robot_frames':len(rows),'duration_s':DURATION}),flush=True)


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

def native_crop(row):
    pixels=np.asarray(pic(row['native_render']));mask=np.any(pixels!=np.array([239,242,249]),axis=-1)
    ys,xs=np.nonzero(mask)
    if not len(xs):return(0,0,405,720)
    height=min(720,max(405,int(ys.max()-ys.min()+81)))
    top=max(0,min(720-height,int((ys.min()+ys.max()-height)/2)))
    return(0,top,405,top+height)

def robot_frame(prep,elapsed):
    capture=prep['robot_capture'];clock=elapsed/8*capture['capture_last_frame_ms']
    row=max((r for r in capture['frames']if r['captured_ms']<=clock),key=lambda r:r['captured_ms'])
    return pic(row)

def scene_pair(im,prep,clock,y=350,height=590):
    row=sequence_row(prep,clock);crop=native_crop(row);sourcecrop=tuple(x*4/3 for x in crop)
    source=decoded_source_frame(prep,clock,pic(row['source_frame']))
    for x,title,picture,cropping in[(68,'CAPTURED VIDEO',source,sourcecrop),(992,'CONNECTED SCENE · ESTIMATED',pic(row['native_render']),crop)]:
        rounded(im,(x,y,x+860,y+height),'white',24);txt(im,(x+26,y+21),title,20,BLUE,True)
        paste_card(im,picture,(x+18,y+60,824,height-122),crop=cropping,fill='#eff2f9',radius=16)
        label=f'Video {timestamp(clock)}'if x==68 else f'Saved sample {timestamp(row["timestamp_s"])}'
        txt(im,(x+26,y+height-44),label,21,MUTED)
    txt(im,(1510,y+height-44),'4 native parts',21,MUTED)


def film_frame(prep,t):
    if t<5:
        im=base('Demonstrations, made reusable.','Show it once.\nKeep how it works.')
        txt(im,(70,410),'Turn captured interactions into\n3D parts, connections and evidence.',36,MUTED,spacing=14)
        for j,label in enumerate(['SAM 3.1','Muse','SAM 3D Objects']):chip(im,(70+j*218,565),label)
        rounded(im,(70,678,690,898),'white',24)
        txt(im,(98,710),'A starting point for robotics.',32,INK,True)
        txt(im,(98,766),'A descriptive asset — with the\nrecording behind every estimate.',27,MUTED,spacing=10)
        row=sequence_row(prep,13.333)
        paste_card(im,pic(row['native_render']),(810,130,1040,790),crop=native_crop(row),fill='#eff2f9')
        chip(im,(839,159),'Native SAM 3D components')
        txt(im,(840,859),'Saved connected scene · estimated',23,MUTED)
        footer(im,'Research prototype  /  Recorded evidence  /  Native meshes  /  Descriptive programs')
    elif t<13:
        q=t-5;im=base('01 / Perceive + reconstruct','Find the moving parts.','SAM 3.1 tracks the recording. SAM 3D Objects generates each component.')
        row=sequence_row(prep,12+q*7/8)
        paste_card(im,pic(row['mask_overlay']),(68,335,755,610),crop=(0,260,540,800))
        chip(im,(90,880),'Actual masks · candidate boundaries',size=21)
        for j,identifier in enumerate(PARTS):
            x=882+(j%2)*490;y=335+(j//2)*305
            paste_card(im,pic(prep['gallery'][identifier]),(x,y,470,280),fill='#eff2f9')
            chip(im,(x+16,y+227),LABELS[identifier],size=19)
        footer(im,'Four unchanged native meshes  /  Editorial component layout  /  Generated geometry remains unverified')
    elif t<21:
        q=t-13;im=base('02 / Connect + replay','Keep the connection.','A persistent hinge keeps the lid attached. Removable components remain separate.')
        clock=15+q*.625
        scene_pair(im,prep,clock)
        footer(im,'Source 15–20s at 0.625×  /  Exact saved 3fps estimates  /  Shared editorial crop; depth and opening angles estimated')
    elif t<29:
        q=(t-21)/8;im=base('03 / Describe','Geometry with a description.','Components, joints, actions and the frames that explain them.')
        row=sequence_row(prep,19.0)
        paste_card(im,pic(row['source_frame']),(68,340,630,585),crop=(0,260,540,800))
        chip(im,(92,363),'source_000058 · 19.00 s',size=21)
        rounded(im,(750,340,1852,935),'#151e32',24,None)
        code=[('{', '#e9efff'),('  "components": [', '#a9bfff'),('    "case_base", "case_lid",', '#e9efff'),('    "earbud_a", "earbud_b"', '#e9efff'),('  ],', '#a9bfff'),('  "joint": {', '#a9bfff'),('    "type": "revolute",', '#e9efff'),('    "parent_id": "case_base",', '#e9efff'),('    "child_id": "case_lid",', '#e9efff'),('    "status": "estimated"', '#a9c8ff'),('  }', '#a9bfff'),('}', '#e9efff')]
        for j,(value,color)in enumerate(code):txt(im,(790,370+j*42),value,27,color,mono=True)
        footer(im,'Readable excerpt of the actual object program  /  Native frames, joint axis, angle samples and source hashes retained')
    elif t<36:
        q=(t-29)/7;im=base('04 / Curate + discover','A library, built from demonstrations.','Browse by object, action or mechanism. Inspect the recording and download the twin.')
        cross_image(im,(68,335,1240,610),pic(prep['screenshots']['library']),q)
        for j,(number,title,note)in enumerate([('01','Inspect','Video + connected scene'),('02','Understand','Parts + descriptive program'),('03','Reuse','Native meshes + evidence')]):
            y=370+j*177;txt(im,(1370,y),number,26,BLUE,True);txt(im,(1370,y+46),title,36,INK,True);txt(im,(1370,y+101),note,23,MUTED)
        footer(im,'Actual Cobalt library view  /  Human review controls the curated collection')
    elif t<43:
        q=(t-36)/7;im=base('05 / Agent access','For people. For their agents.','One shared library. Source-linked programs. Read-only discovery.')
        cross_image(im,(68,340,1280,590),pic(prep['screenshots']['agents']),q)
        rounded(im,(1392,340,1852,930),'#151e32',24,None)
        txt(im,(1424,382),'LOCAL MCP TOOLS',21,'#a9bfff',True)
        for j,(tool,detail)in enumerate([('search_twins','Find an interaction'),('get_kinematics','Read the object program'),('get_evidence','Retrieve its source')]):
            txt(im,(1424,460+j*130),tool,25,'white',mono=True);txt(im,(1424,504+j*130),detail,22,'#b7c1d6')
        footer(im,'Actual setup page  /  Read-only JSON access  /  MCP runs in a local checkout; no hosted MCP is claimed')
    elif t<51:
        q=(t-43)/8;im=base('06 / Robotics authoring','Start a robot task.','Choose components. Stage a removal and reinsertion. Export a task draft.')
        paste_card(im,robot_frame(prep,t-43),(68,330,1450,620))
        txt(im,(1560,410),'Parts',35,INK,True);txt(im,(1560,459),'Native SAM 3D',23,MUTED)
        txt(im,(1560,565),'Task',35,INK,True);txt(im,(1560,614),'Take out → put back',23,MUTED)
        chip(im,(1556,760),'Kinematic preview',size=20)
        footer(im,'Actual UI playback · timed capture  /  Robot, anchors and motion are authored  /  Contact and physics are not simulated')
    else:
        q=t-51;im=base('From demonstrations to reusable interactions.','Interaction Twin',dark=True)
        txt(im,(70,328),'For the next builder.\nAnd their robot.',70,'white',True,spacing=8)
        row=sequence_row(prep,13.333)
        paste_card(im,pic(row['native_render']),(1120,228,730,670),crop=native_crop(row),fill='#eff2f9')
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
    proof_times=[2.5,8.,17.,25.,32.,39.,47.,53.]
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
    support.write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cobalt trailer review · Interaction Twin</title><style>body{margin:0;background:#f7f8fb;color:#151923;font:17px -apple-system,BlinkMacSystemFont,Arial,sans-serif}main{max-width:1120px;margin:60px auto;padding:24px}small{color:#3766ed;font-weight:700;letter-spacing:.12em}h1{font-size:46px;letter-spacing:-.04em;margin:18px 0}p{line-height:1.7;color:#626b7b;max-width:900px}video,img{width:100%;border-radius:20px;border:1px solid #dfe3eb;background:white}a{color:#3766ed}section{margin:40px 0}.note{background:white;border:1px solid #dfe3eb;border-radius:20px;padding:24px}li{margin:12px 0;line-height:1.6}</style><main><small>LOCAL REVIEW · 55 SECONDS · CURRENT NATIVE ASSETS</small><h1>Show it once. Keep how it works.</h1><p>A new Cobalt film made from the actual source recording, saved SAM masks, four native SAM 3D meshes, a connected estimated scene and current app captures. Existing public films and landing references remain unchanged.</p><video controls playsinline preload="metadata" poster="../../../showcase/media/'''+NAME+'''-poster.jpg"><source src="../../../showcase/media/'''+NAME+'''.mp4" type="video/mp4"><track kind="subtitles" srclang="en" label="English" src="../../../showcase/media/'''+NAME+'''.vtt"></video><section><a href="../../../showcase/media/'''+NAME+'''.mp4" download>Download MP4</a> · <a href="verification.json">Verification &amp; provenance</a> · <a href="preparation.json">Exact input manifest</a></section><section class="note"><strong>What the film shows</strong><ul><li>Source-linked masks and unchanged native textured geometry.</li><li>A continuous supported 15–20s source excerpt, slowed to 0.625×, with exact saved three-fps estimates. Shared editorial crops do not change the scene.</li><li>Editorial crops, component layouts and screenshot pans.</li><li>Actual timed UI playback of the authored kinematic robotics-lab rehearsal. Contact, physical scale and physics are unverified.</li><li>Read-only agent discovery and local stdio MCP, without implying a hosted MCP service.</li></ul></section><section><h2>Storyboard proof</h2><img src="storyboard-proof.jpg" alt="Eight proof frames from the 55-second Cobalt trailer"></section><p>Designed and built by Nigel Hartman. Research prototype. Original procedural instrumental; no third-party music or API inference used for this film.</p></main>''')
    print(json.dumps({'complete':True,'film':str(target),'review':str(support),'duration_s':DURATION,'proof':str(sheetpath)}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','encode','all']);args=parser.parse_args()
    if args.command in ('prepare','all'):prepare()
    if args.command in ('encode','all'):encode()
if __name__=='__main__':main()
