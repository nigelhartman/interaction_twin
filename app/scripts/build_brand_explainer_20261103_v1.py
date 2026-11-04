#!/usr/bin/env python3
"""A new55-second source-grounded explainer; no inference or publication.

Native WebGL footage is recorded separately by the visible bounded capture
page. This compositor never changes geometry/poses or hides its provenance.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from marketing_sources import brand,validated_brand,file_ref,checked
from marketing_render import load_brand_artwork

OUT=ROOT/'output/brand-trailer-20261103';W,H,FPS,DURATION=1920,1080,24,55
BG,INK,MUTED,BLUE,LINE='#f7f8fb','#151923','#626b7b','#3766ed','#dfe3eb'
PARTS=['case_base','case_lid','earbud_a','earbud_b']
LABELS={'case_base':'Case base','case_lid':'Hinged lid','earbud_a':'Earbud A','earbud_b':'Earbud B'}
COLORS={'case_base':'#3766ed','case_lid':'#d48d47','earbud_a':'#9179cb','earbud_b':'#409baf'}
CHAPTERS=[
 (0,3,'Show it once. Keep how it works.','A real filmed interaction; the newTwinframe brand.'),
 (3,8,'Start with a demonstration.','Original owner-supplied footage. One case, a lid and two earbuds.'),
 (8,14,'SAM follows the moving parts.','Actual saved SAM3.1 masks. Mask observations are3fps; display repeats samples.'),
 (14,21,'SAM 3D builds each mesh.','Four unchanged nativeGLBs. Independent asset views; camera orbit is editorial.'),
 (21,27,'Work out how parts connect.','Muse proposes relationships; geometry fits estimated joints and motion. Review is not physical certification.'),
 (27,35,'Replay the connected interaction.','Estimated joint motion, linked to the recording. Saved constrained FK and bounded interpolation; generated mating surfaces remain unverified.'),
 (35,41,'More than a mesh.','Readable projection of actual component, joint, relationship and evidence fields.'),
 (41,46,'Find it. Inspect it. Reuse it.','Current curated library metadata; read-only discovery and local MCP. NativeGLB plus descriptiveJSON.'),
 (46,52,'Start a reusable robot task.','Actual browser capture. Robot staging and position cues are authored; no physics or trained policy.'),
 (52,55,'Capture once. Reuse the interaction.','Designed and built byNigelHartman. Physical geometry, scale and contact remain unverified.'),
]

def read(path):return json.loads(Path(path).read_text())
def ref(path,**extra):return file_ref(path,ROOT)|extra
def verify(row):return checked({k:row[k]for k in ['path','sha256','bytes']},ROOT)
def write(path,value):
    with path.open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def run(args):subprocess.run(args,check=True,capture_output=True,timeout=180)
@lru_cache(maxsize=80)
def font(size,bold=False,mono=False):
    name='/System/Library/Fonts/Menlo.ttc'if mono else'/System/Library/Fonts/Supplemental/Arial'+(' Bold'if bold else'')+'.ttf'
    return ImageFont.truetype(name,size)
def txt(im,xy,value,size=30,color=INK,bold=False,mono=False,spacing=10):ImageDraw.Draw(im).multiline_text(xy,value,font=font(size,bold,mono),fill=color,spacing=spacing)
def box(im,rect,fill='white',radius=22,outline=LINE):ImageDraw.Draw(im).rounded_rectangle(tuple(map(round,rect)),radius,fill=fill,outline=outline,width=2)
def line(im,pts,color=LINE,width=3):ImageDraw.Draw(im).line(pts,fill=color,width=width,joint='curve')
def ease(p):p=max(0,min(1,p));return p*p*(3-2*p)
@lru_cache(maxsize=12)
def image(relative):return Image.open(ROOT/relative).convert('RGB')
def contain(im,picture,rect,fill=BG):
    x,y,w,h=map(round,rect);layer=Image.new('RGB',(w,h),fill);pic=ImageOps.contain(picture.convert('RGB'),(w,h),Image.Resampling.LANCZOS);layer.paste(pic,((w-pic.width)//2,(h-pic.height)//2));im.paste(layer,(x,y))
def mark(im,prep,x,y,size,alpha=1.):
    art=load_brand_artwork(prep['brand'],ROOT).resize((size,size),Image.Resampling.LANCZOS)
    if alpha<1:art.putalpha(art.getchannel('A').point(lambda a:round(a*alpha)))
    im.paste(art,(round(x),round(y)),art)
def base(prep,kicker,title,sub=None):
    im=Image.new('RGB',(W,H),BG);mark(im,prep,72,40,46);txt(im,(132,48),'Interaction Twin',27,INK,True)
    txt(im,(72,133),kicker.upper(),21,BLUE,True);txt(im,(68,174),title,66,INK,True,spacing=6)
    if sub:txt(im,(72,263),sub,29,MUTED)
    return im
def foot(im,value):line(im,[(72,1006),(1848,1006)]);txt(im,(72,1030),value,20,MUTED)
def chip(im,x,y,value,*,color=BLUE,fill='#eef2ff',size=22):
    width=round(font(size,True).getlength(value))+28;box(im,(x,y,x+width,y+42),fill,12,None);txt(im,(x+14,y+8),value,size,color,True)

def decode(path,folder,start,duration):
    folder.mkdir();run(['ffmpeg','-nostdin','-v','error','-ss',str(start),'-i',str(path),'-t',str(duration),'-vf','fps=24','-q:v','2',str(folder/'%04d.jpg')])
    rows=[ref(p)for p in sorted(folder.glob('*.jpg'))]
    if len(rows)<max(1,int(duration*FPS)-2):raise ValueError('Decoded source sequence is shorter than declared')
    return {'start_s':start,'duration_s':duration,'fps':24,'frames':rows,'selection':'FFmpeg nearest-frame24fps display; no generated frames'}

def prepare():
    target=OUT/'preparation.json'
    if target.exists():
        p=read(target)
        for row in p['inputs']:verify(row)
        for clip in p['clips'].values():
            for row in clip['frames']:verify(row)
        print(json.dumps({'preparation_reused':True}));return
    OUT.mkdir(parents=True,exist_ok=True)
    old_path=ROOT/'output/cobalt-trailer-20261103/v2/preparation.json';old=read(old_path)
    for key in ['source','program','object_program','scene','music']:verify(old[key])
    scene=read(verify(old['scene']));program=read(verify(old['program']));obj=read(verify(old['object_program']))
    if obj['constrained_scene']!=old['scene']or obj['evidence_program']!=old['program']or scene['source']['sha256']!=old['source']['sha256']:raise ValueError('Source/program/scene binding changed')
    if set(PARTS)!={c['id']for c in scene['components']}:raise ValueError('Expected exact four native components')
    inputs=[ref(old_path),*[old[k]for k in ['source','program','object_program','scene','music']]]
    for c in scene['components']:verify(c['asset']);inputs.append(c['asset'])
    metadata=ROOT/'data/flow-sam3d-default-e2e-20261102-v1/segmentation/flow-components-1/metadata.json'
    if read(metadata)['model']!='sam-3.1'or program['reasoning']['provider']!='meta_api'or not program['reasoning']['model'].startswith('muse-'):raise ValueError('Model attribution differs from saved runs')
    inputs.append(ref(metadata));clips={}
    for name,start,duration in [('intro',1.,6.),('interaction',12.,8.25),('closing',31.,1.25)]:clips[name]=decode(verify(old['source']),OUT/('source-'+name),start,duration)
    robot=ROOT/'showcase/media/robot-workbench-jabra-20261103-v1.mp4';robotref=ref(robot)
    if robotref['sha256']!='6b1d1b1b100c2247ff5a80b933340c34a8214ef697e3a1984bd1639beea41417':raise ValueError('Actual robot capture changed')
    clips['robot']=decode(robot,OUT/'robot',5.,6.);inputs.append(robotref)
    masks=[]
    for row in old['sequence']:
        if 12<=row['timestamp_s']<=20.1:
            verify(row['mask_overlay']);masks.append(row);inputs.extend([row['mask_overlay'],row['source_frame']])
    from curated_twin_catalog import CuratedTwinCatalog
    catalog=CuratedTwinCatalog(ROOT);library=catalog.library_search(limit=10)['records']
    snapshot=brand(ROOT)
    if snapshot['document']['version']!=2 or not snapshot.get('logo_raster'):raise ValueError('Final approved raster brandV2 required')
    validated_brand(snapshot,ROOT);verify(snapshot['logo_raster'])
    joint=obj['joints'][0]
    excerpt={'components':[a['component_id']for a in obj['component_assets']],
        'joint':{key:joint[key]for key in ['id','type','parent_id','child_id','status']},
        'evidence_ids':joint['evidence_ids'],'removable_relationships':obj['removable_relationships'],
        'source_sha256':obj['source']['sha256'],'physical_accuracy_verified':False}
    write(OUT/'object-program-excerpt.json',excerpt)
    shutil.copyfile(__file__,OUT/'builder-at-prepare.py')
    p={'format':'interaction-twin-brand-explainer-preparation','version':1,'brand':snapshot,'inputs':list({r['path']:r for r in inputs}.values()),
        'source':old['source'],'scene':old['scene'],'program':old['program'],'object_program':old['object_program'],'music':old['music'],'clips':clips,'masks':masks,
        'gallery':old['gallery'],'library':library,'excerpt':excerpt,'chapters':[dict(start_s=a,end_s=b,title=c,scope=d)for a,b,c,d in CHAPTERS],
        'render':{'width':W,'height':H,'fps':FPS,'duration_s':DURATION},'scope':{'model_or_API_calls':0,'new_geometry':False,'physics_performed':False,'published':False}}
    for row in old['gallery'].values():verify(row)
    write(target,p);write(OUT/'storyboard.json',{'chapters':p['chapters'],'why':'Keep the shape. Keep how it works. A reusable object program gives builders and agents parts, connections and source evidence.'})
    print(json.dumps({'prepared':True,'source_frames':sum(len(c['frames'])for c in clips.values()),'mask_samples':len(masks)}),flush=True)

def accept_capture(version=1):
    suffix=''if version==1 else'-v2'
    target=OUT/f'native-capture-preparation{suffix}.json'
    if target.exists():raise ValueError('Native capture already frozen; use another version for a replacement')
    clip=OUT/f'native-capture{suffix}.webm';receipt=OUT/f'native-capture{suffix}.json';document=read(receipt);p=read(OUT/'preparation.json')
    if document.get('format')!='interaction-twin-native-browser-capture'or document['config']['scene']!=p['scene']or document['config']['source_sha256']!=p['source']['sha256']:raise ValueError('Native capture source binding changed')
    if document['config']['width']!=1920 or document['config']['height']!=1080 or document['geometry_modified']or document['independent_child_poses_used']:raise ValueError('Native capture contract differs')
    scene=read(verify(p['scene']))
    if document['assets']!=[{'id':c['id'],'asset':c['asset']}for c in scene['components']]:raise ValueError('Captured native assets differ')
    if version==2:
        if document.get('version')!=2 or document.get('joint_annotation',{}).get('joint_id')!='j1_case_base_case_lid':raise ValueError('Exact diagnostic joint annotation required')
        validate_annotation_trace(document,scene)
    capture=decode(clip,OUT/f'native-frames{suffix}',0.,17.)
    write(target,{'clip':ref(clip),'receipt':ref(receipt),'frames':capture,'scope':'Actual WebGL canvas. Independent asset camera orbits0–7s; source-constrained FK7–17s. No independent lid placement. V2 blue marker/axis are explicit source-joint annotations, not mating geometry.'})
    print(json.dumps({'native_capture_accepted':True,'frames':len(capture['frames'])}),flush=True)

def clip_frame(clip,elapsed):
    n=max(0,min(len(clip['frames'])-1,round(elapsed*clip['fps'])));return image(clip['frames'][n]['path'])
def source_frame(p,clock):
    name='intro'if clock<12 else'closing'if clock>=31 else'interaction';clip=p['clips'][name];return clip_frame(clip,clock-clip['start_s'])
def native_frame(p,t):return clip_frame(p['_native']['frames'],t)
def fixed_union_bbox(paths,region,*,padding=36):
    """Editorial crop of blank render pixels; one fixed box for a whole shot.

    The threshold detects the dark native parts against this capture's fixed
    light background. A union over every decoded frame prevents tracking zoom
    and retains the complete visible silhouette throughout the shot.
    """
    left,top,right,bottom=region;bounds=[]
    if padding<0 or right<=left or bottom<=top:raise ValueError('Invalid editorial crop region')
    for path in paths:
        a=np.asarray(Image.open(path).convert('RGB'))
        if a.shape[:2]!=(H,W):raise ValueError('Expected native1080p capture frame')
        ys,xs=np.where(a[top:bottom,left:right].min(axis=2)<200)
        if len(xs):bounds.append((int(xs.min()+left),int(ys.min()+top),int(xs.max()+left+1),int(ys.max()+top+1)))
    if not bounds:raise ValueError('No native geometry visible in crop region')
    return [max(left,min(b[0]for b in bounds)-padding),max(top,min(b[1]for b in bounds)-padding),min(right,max(b[2]for b in bounds)+padding),min(bottom,max(b[3]for b in bounds)+padding)]
def validate_annotation_trace(document,scene):
    """Verify saved annotation math using the browser's real FK contract.

    No browser, inference or network. The only child process is a fixed Node
    verifier reading the exact declared scene and bounded JSON from stdin.
    """
    trace=document.get('projection_trace')
    if not isinstance(trace,list)or not 200<=len(trace)<=1024:raise ValueError('Bounded complete joint annotation trace required')
    if trace[0].get('capture_time_s',100)>7.15 or trace[-1].get('capture_time_s',0)<16.8:raise ValueError('Joint annotation trace does not span the connected shot')
    code=r'''const fs=require('node:fs'),C=require('./showcase/aligned-scene-contract.js'),d=JSON.parse(fs.readFileSync(0,'utf8')),s=C.validateScene(d.scene,d.document.config.source_sha256),j=s.mechanics.joints.find(j=>j.id===d.document.joint_annotation.joint_id),cam=d.document.capture_camera;
const mul=(m,v)=>m.map(r=>r.reduce((a,x,i)=>a+x*v[i],0)),near=(a,b)=>Array.isArray(a)&&a.length===b.length&&a.every((v,i)=>Number.isFinite(v)&&Math.abs(v-b[i])<1e-5),flip=[[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]],rigidInverse=m=>{const r=[0,1,2].map(i=>[m[0][i],m[1][i],m[2][i],0]);for(let i=0;i<3;i++)r[i][3]=-r[i].slice(0,3).reduce((a,x,k)=>a+x*m[k][3],0);return[...r,[0,0,0,1]];},view=rigidInverse(cam.matrix_world),project=p=>{const v=mul(cam.projection_matrix,mul(view,[...p,1]));return[(v[0]/v[3]+1)*960,(1-v[1]/v[3])*540,v[2]/v[3]];};let previous=-1;
for(const r of d.document.projection_trace){if(!(r.capture_time_s>previous&&r.capture_time_s>=7&&r.capture_time_s<=17.1&&r.joint_id===j.id&&r.annotation_only===true))throw Error('Invalid typed annotation timeline');previous=r.capture_time_s;const q=r.capture_time_s-7,t=q<=6?15+q*5/6:31+(q-6)/4;if(Math.abs(t-r.source_time_s)>1e-6)throw Error('Source clock changed');const state=C.sceneAt(s,t,true),parent=state.parts.find(p=>p.id===j.parent_id),angle=state.joints.find(p=>p.id===j.id);if(!parent.camera_from_native_mesh||angle.q_rad===null||Math.abs(angle.q_rad-r.q_rad)>1e-5)throw Error('Unsupported source joint');const world=C.multiply(C.multiply(flip,parent.camera_from_native_mesh),j.parent_from_joint),pivot=mul(world,[0,0,0,1]).slice(0,3),direction=mul(world,[...j.axis,0]).slice(0,3),len=Math.hypot(...direction),axis=direction.map(x=>x/len),L=d.document.joint_annotation.axis_length_world;if(!Number.isFinite(L)||L<=0)throw Error('Invalid diagnostic axis length');const a=pivot.map((v,i)=>v-axis[i]*L/2),b=pivot.map((v,i)=>v+axis[i]*L/2);if(!near(r.pivot_world,pivot)||!near(r.axis_world,axis)||!near(r.pivot_canvas,project(pivot))||!Array.isArray(r.axis_canvas)||r.axis_canvas.length!==2||!near(r.axis_canvas[0],project(a))||!near(r.axis_canvas[1],project(b)))throw Error('Joint pivot/axis projection differs');}console.log(JSON.stringify({verified:true,rows:d.document.projection_trace.length}));'''
    result=subprocess.run(['node','-e',code],cwd=ROOT,input=json.dumps({'document':document,'scene':scene}),text=True,capture_output=True,timeout=30)
    if result.returncode:raise ValueError('Joint annotation evidence failed: '+result.stderr[-1000:])
    return json.loads(result.stdout)
def prepare_framing(capture_version=1):
    suffix=''if capture_version==1 else'-v2';prep_path=OUT/f'native-capture-preparation{suffix}.json'
    target=OUT/('editorial-framing-v3.json'if capture_version==1 else'editorial-framing-v4.json');native=read(prep_path);prep=read(OUT/'preparation.json')
    if target.exists():
        saved=read(target)
        if saved['native_capture_preparation']!=ref(prep_path)or saved['robot_source']!=next(r for r in prep['inputs']if r['path'].endswith('robot-workbench-jabra-20261103-v1.mp4')):raise ValueError('Editorial framing source changed')
        return saved
    paths=[verify(row)for row in native['frames']['frames']]
    quads=[(0,0,960,540),(960,0,1920,540),(0,540,960,1080),(960,540,1920,1080)]
    saved={'format':'interaction-twin-fixed-editorial-framing','version':4 if capture_version==2 else 3,'native_capture_preparation':ref(prep_path),
      'method':'Fixed union of dark native silhouettes over every decoded shot frame, padded; no per-frame crop, generated pixels, changed meshes or poses.',
      'independent_native':{key:fixed_union_bbox(paths[:7*FPS],region,padding=18)for key,region in zip(PARTS,quads)},
      'connected_native':fixed_union_bbox(paths[7*FPS:],(0,0,W,H),padding=55),
      'robot_source':next(r for r in prep['inputs']if r['path'].endswith('robot-workbench-jabra-20261103-v1.mp4')),
      'robot_canvas_crop':[90,88,1190,718],'robot_crop_scope':'Editorial crop of actual browser-canvas recording, removing the old UI header/footer; no robot/object pixels are generated.'}
    write(target,saved);return saved
def connected_picture(p,t):return native_frame(p,t).crop(p['_framing']['connected_native'])
def descriptive_lines(p):
    obj=p['_object_program'];joint=obj['joints'][0]
    sample=next(row for row in joint['samples']if row['source_sample_index']==21)
    if 'source_000021'not in joint['evidence_ids']:raise ValueError('Displayed evidence citation is absent')
    return ['{',f'  "parent_id": "{joint["parent_id"]}",',f'  "child_id": "{joint["child_id"]}",',
      f'  "type": "{joint["type"]}",',f'  "status": "{joint["status"]}",',
      '  "axis": ['+', '.join(f'{n:.2f}'for n in joint['axis'])+'],',
      f'  "q_rad": {sample["q_rad"]:.2f},',f'  "timestamp_s": {sample["timestamp_s"]:.2f},',
      '  "evidence_ids": ["source_000021"],','  "physical_accuracy_verified": false','}']
def node(im,x,y,title,active=True):
    box(im,(x,y,x+410,y+114),'#eef2ff'if active else'white',18,BLUE if active else LINE);txt(im,(x+24,y+23),title,30,INK,True)
def film_frame(p,t):
    if t<3:
        im=Image.new('RGB',(W,H),BG);progress=ease(t/1.8);size=round(150*(.94+.06*progress));mark(im,p,96,205,size,min(1,t/.45))
        txt(im,(92,420),'Interaction Twin',88,INK,True);txt(im,(96,548),'Keep the shape.\nKeep how it works.',51,MUTED,spacing=12)
        contain(im,source_frame(p,1+t),(1050,80,770,910));txt(im,(1050,952),'A real demonstration → a reusable interaction',25,BLUE,True)
    elif t<8:
        im=base(p,'01 / Capture','Start with a demonstration.')
        txt(im,(84,415),'One case.\nA lid.\nTwo removable earbuds.',48,INK,True,spacing=22)
        txt(im,(86,680),'The recording is the evidence.',29,MUTED);contain(im,source_frame(p,1+t-3),(950,215,850,750))
        foot(im,'Original captured video · no special capture hardware or calibration board')
    elif t<14:
        im=base(p,'02 / Segment','SAM follows the moving parts.','SAM 3.1 separates the base, lid and both earbud tracklets.')
        clock=12+(t-8)*.8;row=min(p['masks'],key=lambda r:abs(r['timestamp_s']-clock));contain(im,image(row['mask_overlay']['path']),(80,340,900,620))
        for i,key in enumerate(PARTS):
            y=380+i*130;box(im,(1080,y,1120,y+40),COLORS[key],11,None);txt(im,(1152,y-3),LABELS[key],38,INK,True)
        foot(im,'Actual saved masks · 3fps observations repeated for display · source-linked component IDs')
    elif t<21:
        im=base(p,'03 / Reconstruct','SAM 3D builds each mesh.','Separate geometry is useful. It still needs a description of how it works.')
        captured=native_frame(p,t-14)
        for i,key in enumerate(PARTS):
            x=90+(i%2)*900;y=330+(i//2)*330
            contain(im,captured.crop(p['_framing']['independent_native'][key]),(x,y,840,265))
            chip(im,x+270,y+275,LABELS[key],size=25)
        foot(im,'Four unchanged native GLBs · independent asset views · camera motion is editorial')
    elif t<27:
        q=t-21;im=base(p,'04 / Reason + review','Work out how parts connect.','Muse proposes relationships; geometry fits estimated joints and motion.')
        center=(940,575);nodes={'case_base':(720,515),'case_lid':(130,365),'earbud_a':(1320,365),'earbud_b':(1320,675)}
        for i,key in enumerate(['case_lid','earbud_a','earbud_b']):
            if q>i*.65:
                x,y=nodes[key];line(im,[(925,570),(x+205,y+57)],BLUE,5)
        for key,(x,y)in nodes.items():node(im,x,y,LABELS[key],q>.3)
        chip(im,300,540,'revolute · estimated',size=23);chip(im,1260,575,'removable',size=23)
        for i,label in enumerate(['Propose relationships','Check against evidence','Owner reviews the result']):
            if q>1.5+i*.6:txt(im,(110+i*600,882),str(i+1).zfill(2)+'  '+label,26,BLUE,True)
        foot(im,'Symbolic explanation of saved relationships · model proposals and checks are not physical certification')
    elif t<35:
        q=t-27;clock=15+q*5/6 if q<=6 else 31+(q-6)/4
        im=base(p,'05 / Connect','Replay the connected interaction.','Estimated joint motion, linked to the recording.')
        contain(im,source_frame(p,clock),(55,335,570,610));contain(im,connected_picture(p,7+q),(690,315,1190,625))
        chip(im,110,945,'Captured video',size=22);chip(im,1050,945,'Connected scene · estimated',size=22)
        foot(im,'Source-constrained FK + bounded interpolation · editorial time jump at33s · unknown poses withheld')
    elif t<41:
        im=base(p,'06 / Describe','More than a mesh.','Keep parts, connections, actions and source evidence together.')
        contain(im,connected_picture(p,10),(65,330,820,560));box(im,(920,338,1848,915),'#151923',22,None)
        txt(im,(950,358),'SAVED EVIDENCE EXCERPT · 6.67s',21,'#a6bdff',True)
        for i,value in enumerate(descriptive_lines(p)):txt(im,(950,405+i*44),value,26,'#c1d1ff'if i in [0,10]else'white',mono=True)
        chip(im,92,946,'3D parts · GLB',size=24);chip(im,590,946,'Object program · JSON',size=24);chip(im,1280,946,'Source evidence',size=24)
        foot(im,'Readable excerpt of saved fields · axis, angle and scale are estimates · all parts retain source evidence')
    elif t<46:
        im=base(p,'07 / Reuse','Find it. Inspect it. Reuse it.','Use it in robotics authoring. Let agents inspect it.')
        cards=[('Earbuds + case','Open · remove · return'),('Wallet + coin','Containment · remove'),('Camera + batteries','Open · exchange')]
        for i,(title,sub)in enumerate(cards):
            y=350+i*168;box(im,(76,y,820,y+140));txt(im,(106,y+24),title,34,INK,True);txt(im,(108,y+82),sub,25,MUTED)
        box(im,(900,350,1848,904),'#151923',22,None);txt(im,(945,395),'READ-ONLY AGENT TOOLS',22,'#a6bdff',True)
        for i,(name,note)in enumerate([('search_twins','Find a useful object'),('get_kinematics','Read its connections'),('get_evidence','Inspect the recording')]):txt(im,(945,470+i*124),name,35,'white',mono=True);txt(im,(945,523+i*124),note,24,'#b9c3d7')
        foot(im,'Current curated examples · GLB + JSON downloads · local MCP, not a hosted service')
    elif t<52:
        im=base(p,'08 / Build on it','Start a reusable robot task.','Native objects become an editable starting point for robotics development.')
        picture=clip_frame(p['clips']['robot'],t-46).crop(p['_framing']['robot_canvas_crop'])
        contain(im,picture,(40,326,1840,650));foot(im,'Actual browser canvas · editorial crop · authored position cues · no physical simulation or trained policy')
    else:
        im=Image.new('RGB',(W,H),BG);mark(im,p,100,145,132);txt(im,(96,362),'Capture once.\nReuse the interaction.',86,INK,True,spacing=12)
        txt(im,(102,624),'3D parts. Connections. Evidence.',38,MUTED);txt(im,(104,758),'Explore the object library.',34,BLUE,True)
        txt(im,(104,916),'Designed and built by Nigel Hartman',26,MUTED);foot(im,'Research prototype · generated geometry, physical scale, grasp and contact remain unverified')
    start=max(a for a,_,_,_ in CHAPTERS if a<=t);fade=min(1.,(t-start)/.22)if start else min(1.,t/.25)
    return Image.blend(Image.new('RGB',(W,H),BG),im,fade)if fade<1 else im

def encode(name,capture_version=1):
    if not re.fullmatch('[a-z0-9-]{1,40}',name):raise ValueError('Bounded fresh render name required')
    suffix=''if capture_version==1 else'-v2';native_preparation=OUT/f'native-capture-preparation{suffix}.json'
    framing_path=OUT/('editorial-framing-v3.json'if capture_version==1 else'editorial-framing-v4.json')
    p=read(OUT/'preparation.json');p['_native']=read(native_preparation)
    p['_framing']=prepare_framing(capture_version);p['_object_program']=read(verify(p['object_program']))
    for row in p['inputs']+[p['_native']['clip'],p['_native']['receipt']]:verify(row)
    validated_brand(p['brand'],ROOT)
    for clip in [*p['clips'].values(),p['_native']['frames']]:
        for row in clip['frames']:verify(row)
    target=OUT/'renders'/name;target.mkdir(parents=True,exist_ok=False);shutil.copyfile(__file__,target/'builder-at-encode.py')
    write(target/'storyboard.json',{'format':'interaction-twin-brand-explainer-storyboard','version':2,
      'chapters':[dict(start_s=a,end_s=b,title=c,scope=d)for a,b,c,d in CHAPTERS],
      'why':'Keep the shape. Keep how it works. A reusable object program retains parts, connections and source evidence.',
      'formats_shown':['GLB','JSON','source evidence'],'claims':{'physical_accuracy_verified':False,'robot_motion':'authored position-only rehearsal','MCP':'local read-only tools, not hosted service'}})
    picture=target/'picture.mp4';movie=target/'interaction-twin-explainer.mp4';started=time.monotonic()
    proc=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','24','-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','19','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(picture)],stdin=subprocess.PIPE)
    proofs=[];proof_times=[1.5,5.,10.,17.,24.,30.,38.,43.,49.,53.]
    try:
        for n in range(DURATION*FPS):
            t=n/FPS;im=film_frame(p,t)
            if any(abs(t-x)<.5/FPS for x in proof_times):
                path=target/f'proof-{n:04}.jpg';im.save(path,quality=96);proofs.append(ref(path,film_time_s=t))
            proc.stdin.write(np.asarray(im,np.uint8).tobytes())
            if n%(FPS*5)==0:print(json.dumps({'film_s':int(t),'wall_s':round(time.monotonic()-started,1)}),flush=True)
        proc.stdin.close();code=proc.wait(timeout=90)
        if code:raise ValueError('FFmpeg failed')
    except BaseException:proc.kill();proc.wait();raise
    run(['ffmpeg','-nostdin','-v','error','-i',str(picture),'-i',str(verify(p['music'])),'-map','0:v:0','-map','1:a:0','-t','55','-c:v','copy','-af','loudnorm=I=-22:TP=-2:LRA=7,afade=t=in:st=0:d=0.6,afade=t=out:st=53:d=2','-ar','48000','-c:a','aac','-b:a','192k','-movflags','+faststart',str(movie)])
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(movie)]))
    video=next(s for s in probe['streams']if s['codec_type']=='video')
    if(video['width'],video['height'])!=(1920,1080)or abs(float(probe['format']['duration'])-55)>.1 or not any(s['codec_type']=='audio'for s in probe['streams']):raise ValueError('Final media contract failed')
    sheet=Image.new('RGB',(1600,5*480),BG)
    for i,row in enumerate(proofs):
        x=(i%2)*800;y=(i//2)*480;sheet.paste(image(row['path']).resize((800,450),Image.Resampling.LANCZOS),(x,y));txt(sheet,(x+12,y+454),f'{row["film_time_s"]:.1f}s · {CHAPTERS[i][2]}',17,MUTED)
    sheet.save(target/'proof-sheet.jpg',quality=96);film_frame(p,30).save(target/'poster.jpg',quality=96)
    vtt=['WEBVTT',''];srt=[]
    def stamp(t,sep):return f'00:{int(t)//60:02}:{int(t)%60:02}{sep}000'
    for i,(a,b,title,scope)in enumerate(CHAPTERS,1):vtt.extend([stamp(a,'.')+' --> '+stamp(b,'.'),title,scope,'']);srt.extend([str(i),stamp(a,',')+' --> '+stamp(b,','),title,''])
    (target/'captions.vtt').write_text('\n'.join(vtt));(target/'captions.srt').write_text('\n'.join(srt))
    write(target/'verification.json',{'format':'interaction-twin-brand-explainer-verification','version':2,'movie':ref(movie),'poster':ref(target/'poster.jpg'),'proof_sheet':ref(target/'proof-sheet.jpg'),'proof_frames':proofs,'storyboard':ref(target/'storyboard.json'),'captions':ref(target/'captions.vtt'),'srt':ref(target/'captions.srt'),'ffprobe':probe,'preparation':ref(OUT/'preparation.json'),'native_capture_preparation':ref(native_preparation),'editorial_framing':ref(framing_path),'builder':ref(target/'builder-at-encode.py'),'elapsed_wall_s':time.monotonic()-started,'new_API_or_model_calls':0,'geometry_modified':False,'physics_performed':False,'published':False,'landing_changed':False,'music_rights':'Original procedural project instrumental; no external samples or tracks.'})
    (target/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>New explainer · local review</title><style>body{margin:40px auto;max-width:1200px;background:#f7f8fb;color:#151923;font:18px Arial}video,img{width:100%;border-radius:18px}p{line-height:1.6}a{color:#3766ed}</style><h1>Keep the shape. Keep how it works.</h1><p>New55-second explainer: actual footage, saved SAM masks, unchanged native meshes, constrained connections and an authored robot rehearsal. No new inference or publication.</p><video controls poster="poster.jpg" src="interaction-twin-explainer.mp4"><track kind="subtitles" src="captions.vtt" srclang="en" label="English"></video><p><a href="interaction-twin-explainer.mp4" download>DownloadMP4</a> · <a href="verification.json">Exactprovenance</a></p><img src="proof-sheet.jpg" alt="Tenproof frames"><p>Geometry, physical scale, grasp and contact remain unverified. Robot task is authored. Designed and built by Nigel Hartman.</p>')
    print(json.dumps({'complete':True,'movie':str(movie),'proof':str(target/'proof-sheet.jpg'),'duration_s':55}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','accept-capture','encode']);parser.add_argument('--name',default='v1');parser.add_argument('--capture-version',type=int,choices=[1,2],default=1);args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='accept-capture':accept_capture(args.capture_version)
    else:encode(args.name,args.capture_version)
if __name__=='__main__':main()
