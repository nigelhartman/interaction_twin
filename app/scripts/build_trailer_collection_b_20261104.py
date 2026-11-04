#!/usr/bin/env python3
"""Three distinct editorial stories using hash-pinned, authentic saved assets.
No hosted inference, part edits, new poses, publication or simulation occurs.
"""
from __future__ import annotations
import argparse,hashlib,json,math,subprocess,sys,time,wave
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'output/trailer-collection-20261104-v1/set-b'
DEST=ROOT/'showcase/media/trailer-collection-20261104-v1'
MASTER=ROOT/'output/dynamic-trailer-20261104-v3/render-candidate-v5/master-frames'
ROBOT=ROOT/'output/dynamic-trailer-20261104-v3/render-candidate-v5/robot-frames'
MOTION=ROOT/'output/motion-trailer-20261104-v5/interaction-twin-motion-20261104-v5.webm'
MOTION_RECEIPT=MOTION.with_suffix('.json')
ROBOT_RECEIPT=ROOT/'output/trailer-refinement-20261103/robot-complete-capture-v3.json'
LOGO=ROOT/'showcase/brand/v2/mark-1024.png'
BG='#f7f8fb';INK='#151923';BLUE='#3766ed';MUTED='#68758a';LINE='#dfe4ee';DARK='#121927';FPS=30
SPECS={'04-proof':{'size':(1920,1080),'duration':30,'title':'The proof','story':'Follow a real observation into a connected twin, inspect its source evidence and keep the geometry, actions and connections for an agent.','style':'Continuous landscape split-screen documentary; restrained ambient pulse.'},'05-pocket-worlds':{'size':(1080,1920),'duration':26,'title':'Pocket worlds','story':'Everyday objects become a small museum of useful parts, then a library for robotics agents.','style':'Vertical kinetic museum; real native WebGL camera orbits, original footage inserts and syncopated percussion.'},'06-agent':{'size':(1080,1080),'duration':38,'title':"An agent’s first day",'story':'An illustrated agent asks for an interaction, retrieves the real twin and program, then completes both placements in the actual local robot rehearsal.','style':'Square conversational product story; typed instruction, inspected source, task timeline and warm arpeggio.'}}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):
 p=Path(p);return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def run(args,timeout=600):return subprocess.run(args,check=True,capture_output=True,text=True,timeout=timeout)
@lru_cache(maxsize=32)
def font(n,bold=False):return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if bold else'')+'.ttf',n)
@lru_cache(maxsize=20)
def pic(p):return Image.open(p).convert('RGB')
def mframe(t):return pic(str(MASTER/f'{max(1,min(1110,round(t*30)+1)):05}.jpg'))
def rframe(t):return pic(str(ROBOT/f'{max(1,min(806,round(t*30)+1)):05}.jpg'))
def ease(t):t=max(0,min(1,t));return t*t*(3-2*t)
def text(im,xy,s,size=32,color=INK,bold=False):ImageDraw.Draw(im).text(xy,s,font=font(size,bold),fill=color)
def multiline(im,xy,s,size=32,color=INK,bold=False,spacing=5):ImageDraw.Draw(im).multiline_text(xy,s,font=font(size,bold),fill=color,spacing=spacing)
def rr(im,box,fill='white',radius=24,outline=None):ImageDraw.Draw(im).rounded_rectangle(box,radius,fill=fill,outline=outline,width=2)
def fit(im,picture,box,bg=None):
 x,y,x2,y2=map(round,box);w,h=x2-x,y2-y
 if bg:rr(im,box,bg,22)
 layer=ImageOps.contain(picture,(w,h),Image.Resampling.LANCZOS);im.paste(layer,(x+(w-layer.width)//2,y+(h-layer.height)//2))
def brand(im,small=False):
 size=42 if small else 48;logo=Image.open(LOGO).convert('RGBA').resize((size,size),Image.Resampling.LANCZOS);im.paste(logo,(48,42),logo);text(im,(48+size+16,46),'Interaction Twin',25 if small else 29,bold=True)
def label(im,xy,s,size=19,color=BLUE):text(im,xy,s.upper(),size,color,True)
def center(im,y,s,size=60,color=INK,bold=True):
 b=ImageDraw.Draw(im).textbbox((0,0),s,font=font(size,bold));text(im,((im.width-(b[2]-b[0]))//2,y),s,size,color,bold)
def endcard(size,t,subtitle='Show it once. Keep how it works.'):
 im=Image.new('RGB',size,BG);w,h=size;logo=Image.open(LOGO).convert('RGBA').resize((96,96));im.paste(logo,((w-96)//2,h//2-180),logo);center(im,h//2-40,'Interaction Twin',65 if w>1200 else 56);center(im,h//2+60,subtitle,32 if w>1200 else 27,MUTED,False);return im

def preparation():
 OUT.mkdir(parents=True,exist_ok=True);DEST.mkdir(parents=True,exist_ok=True)
 receipt=json.loads(MOTION_RECEIPT.read_text())
 assert receipt['version']==5 and receipt['canvas']==[1920,1080] and receipt['duration_s']==37
 assert receipt['bytes']==MOTION.stat().st_size and receipt['physical_accuracy_verified']is False
 for r in receipt['inputs']+receipt['capture_runtime_source_refs']:
  p=ROOT/r['path'];assert sha(p)==r['sha256'] and p.stat().st_size==r['bytes'],r['path']
 robot=json.loads(ROBOT_RECEIPT.read_text());assert robot['captured_full_task'] and all(g['released']and g['reached'] for g in robot['final_goals'])
 assert all(abs(v)<1e-8 for v in robot['final_joint_q_rad'].values()) and robot['final_gripper_attached']is None
 assert len(list(MASTER.glob('*.jpg')))>=1100 and len(list(ROBOT.glob('*.jpg')))>=800
 write(OUT/'input-validation.json',{'motion':ref(MOTION),'motion_receipt':ref(MOTION_RECEIPT),'all_capture_inputs_verified':True,'robot_receipt':ref(ROBOT_RECEIPT),'complete_goals_verified':True,'derived_cache_origin':'Existing final V3 render-candidate-v5 decoded frames, final master and robot originals independently retained.','logo':ref(LOGO),'new_api_calls':0,'new_gpu_calls':0})
 return receipt,robot

def native_crop_audit(receipt):
 """Full original vertices and exact saved draw matrices bound every native part."""
 from native_appearance import load_native_glb
 scene=json.loads((ROOT/receipt['scene_ref']['path']).read_text());w,h=scene['source']['width'],scene['source']['height'];k=np.asarray(scene['camera']['K']);near,far=.001,10000
 projection=np.asarray([[2*k[0,0]/w,-2*k[0,1]/w,1-2*(k[0,2]+.5)/w,0],[0,2*k[1,1]/h,2*(k[1,2]+.5)/h-1,0],[0,0,-(far+near)/(far-near),-2*far*near/(far-near)],[0,0,-1,0]])
 vertices={}
 for component in scene['components']:
  m=load_native_glb(ROOT/component['asset']['path']);vertices[component['id']]=np.column_stack([m.vertices,np.ones(len(m.vertices))]).T
 viewports={0:[950,315,907,660],2:[55,292,1810,715],3:[90,305,1260,690]};checks=[];unions={str(n):[float('inf'),float('inf'),float('-inf'),float('-inf')]for n in viewports}
 for i,row in enumerate(receipt['projection_trace']):
  chapter=row['chapter_index']
  if chapter not in viewports:continue
  angle=row['editorial_camera_orbit_rad'];c,ss=math.cos(angle),math.sin(angle);view=np.asarray([[c,0,-ss,0],[0,1,0,0],[ss,0,c,0],[0,0,0,1]])
  l,tt,r,b=row['source_crop'];x,y,vw,vh=viewports[chapter];scale=min(vw/(r-l),vh/(b-tt));offset=np.asarray([x+(vw-(r-l)*scale)/2,y+(vh-(b-tt)*scale)/2])
  for part in row['parts']:
   q=projection@view@np.diag([1,-1,-1,1])@np.asarray(part['camera_from_native_mesh'])@vertices[part['id']];assert np.isfinite(q).all()and(q[3]>0).all();ndc=q[:2]/q[3];uv=np.column_stack([(ndc[0]+1)*w/2,(1-ndc[1])*h/2]);canvas=(uv-[l,tt])*scale+offset;lo,hi=canvas.min(0),canvas.max(0);bounds=[float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1])];u=unions[str(chapter)];u[:]=[min(u[0],bounds[0]),min(u[1],bounds[1]),max(u[2],bounds[2]),max(u[3],bounds[3])]
   checks.append({'capture_trace_index':i,'chapter':chapter,'master_s':row['film_s'],'source_s':row['source_s'],'part':part['id'],'native_vertex_bounds':bounds})
 crops={key:[max(0,math.floor(u[0]-12)),max(0,math.floor(u[1]-12)),min(1920,math.ceil(u[2]+12)),min(1080,math.ceil(u[3]+12))]for key,u in unions.items()}
 for row in checks:
  box=crops[str(row['chapter'])];b=row['native_vertex_bounds'];assert b[0]>=box[0]+10 and b[1]>=box[1]+10 and b[2]<=box[2]-10 and b[3]<=box[3]-10
 value={'format':'interaction-twin-trailer-full-native-framing-audit','version':1,'motion_receipt':ref(MOTION_RECEIPT),'scene_ref':receipt['scene_ref'],'native_assets':[c['asset']for c in scene['components']],'fixed_native_crops':crops,'padding_px':12,'full_native_vertices':True,'component_bound_checks':len(checks),'part_crops_or_clips':0,'method':'Project every original native vertex through each exact saved part matrix, editorial camera rotation, scene K and chapter viewport. Fit the conservative per-chapter union with12px padding; no mesh or pose change.','checks':checks}
 write(OUT/'native-framing-audit.json',value);return crops

def native_frame(chapter,clock):return mframe(clock).crop(NATIVE_CROPS[str(chapter)])

def orbit_assets():
 """Use authentic, existing native views; pan/cut is editorial, not new part motion."""
 folder=OUT/'native-orbits';folder.mkdir(exist_ok=True);rows=[]
 inspection=ROOT/'output/overnight-pipeline-20261102/camera-wallet-native-inspection-v1/inspection.json';doc=json.loads(inspection.read_text());coin=next(r for r in doc['records']if r['component_id']=='gold_coin');assert sha(ROOT/coin['native_asset_ref']['path'])==coin['native_asset_ref']['sha256']
 sources={
 'earbuds':[(MASTER/'00550.jpg',(90,305,1350,995))],
 'laptop':[(ROOT/'output/precision-polish-20261104-v1/browser-proof/laptop-boot-source44.png',(755,240,1082,500))],
 'charger':[(ROOT/'output/assembly-quality-20261104/repaired-source-orbit-v1/charger_after-assembly-render.png',(340,55,660,385))],
 'coin':[(ROOT/f'output/overnight-pipeline-20261102/camera-wallet-native-inspection-v1/wallet-gold_coin-view-{n:03}.png',(0,0,256,256))for n in [0,60,120]]}
 for name,items in sources.items():
  views=[]
  for i in range(18):
   source,crop=items[min(len(items)-1,i*len(items)//18)];picture=Image.open(source).convert('RGB').crop(crop);side=480;picture=ImageOps.contain(picture,(side,side));im=Image.new('RGB',(side,side),BG);im.paste(picture,((side-picture.width)//2,(side-picture.height)//2));target=folder/f'{name}-editorial-{i:02}.jpg';im.save(target,quality=93);views.append({'image':ref(target),'source':ref(source),'crop_xyxy':list(crop)})
  rows.append({'id':name,'views':views,'geometry_modified':False,'camera_scope':'Preserved preflight-only editorial view candidates; FINAL film uses the separate verified continuous GPU-orbit capture.'})
 gpu=ROOT/'output/trailer-collection-20261104-v1/native-orbits/native-orbits-20261104-v1.webm';gpu_receipt=gpu.with_suffix('.json');g=json.loads(gpu_receipt.read_text());assert g['format']=='interaction-twin-native-editorial-orbits' and g['duration_s']==12 and g['canvas']==[1080,1080]and g['geometry_modified']is False and g['editorial_camera_motion_only']is True
 assert sha(gpu)==g['video_sha256'] and gpu.stat().st_size==g['bytes']
 for row in g['assets']:assert sha(ROOT/row['asset_ref']['path'])==row['asset_ref']['sha256']
 frames=OUT/'native-orbits-gpu';frames.mkdir(exist_ok=True)
 if not (frames/'00360.jpg').exists():run(['ffmpeg','-nostdin','-v','error','-i',str(gpu),'-vf','fps=30,scale=720:720','-q:v','3','-y',str(frames/'%05d.jpg')])
 assert len(list(frames.glob('*.jpg')))>=359
 gpu_context={'capture':ref(gpu),'receipt':ref(gpu_receipt),'frame_cache_scale':[720,720],'segments_s':[[0,3],[3,6],[6,9],[9,12]],'all_native_GLBS_verified':True,'camera_orbit_only_no_new_part_poses':True}
 # Tiny actual source inserts: complete portrait composition, no object-masking edits.
 source_clips=[('laptop','data/flow-capture-laptop-20261103-v1/source/video.mp4',28.0),('charger','data/flow-capture-charger-20261103-v1/source/video.mp4',35.0),('coin','data/flow-rigid-parts-wallet-20261031-v1/source/video.mp4',23.0)]
 inserts=[]
 for name,path,start in source_clips:
  target=folder/name;target.mkdir(exist_ok=True)
  if not (target/'00045.jpg').exists():run(['ffmpeg','-nostdin','-v','error','-ss',str(start),'-i',str(ROOT/path),'-t','3','-vf','fps=15,scale=270:480','-q:v','4','-y',str(target/'%05d.jpg')])
  inserts.append({'id':name,'source':ref(ROOT/path),'source_interval_s':[start,start+3],'decoded_fps':15,'files':[ref(v)for v in sorted(target.glob('*.jpg'))]})
 write(OUT/'native-orbits.json',{'assets':rows,'actual_gpu_orbits':gpu_context,'source_inserts':inserts,'coin_original_binding_receipt':ref(inspection),'geometry_modified':False,'camera_only_editorial_motion':True,'new_render_cpu_attempt':'Three preserved full-triangle case-body renders were generated but are not used in final film; slow attempt stopped, no automatic retry.'});return rows

def connected_preparation():
 capture=ROOT/'output/trailer-collection-20261104-v1/connected-full-bounds/connected-full-bounds-20261104-v1.webm';receipt=capture.with_suffix('.json');validation=capture.parent/'validation.json';d=json.loads(receipt.read_text());v=json.loads(validation.read_text());assert d['format']=='interaction-twin-connected-full-bounds-capture' and d['duration_s']==12 and d['canvas']==[1920,1080] and d['camera_is_original_video_camera']is False and d['original_part_poses_unchanged']is True and d['native_geometry_unchanged']is True
 assert d['bytes']==capture.stat().st_size and d['video_sha256']==sha(capture) and v['no_clipped_components']is True
 for row in d['inputs']:
  target=ROOT/row['path'];assert target.stat().st_size==row['bytes']and sha(target)==row['sha256']
 boxes={0:[float('inf'),float('inf'),float('-inf'),float('-inf')],1:[float('inf'),float('inf'),float('-inf'),float('-inf')]};checks=0
 for row in d['trace']:
  assert {p['id']for p in row['parts']}=={'case_base','case_lid','earbud_a','earbud_b'}and all(p['camera_from_native_mesh']for p in row['parts']);u=boxes[row['chapter_index']]
  for part in row['projected_native_boxes']:
   points=np.asarray(part['points']);assert points.shape==(8,3)and np.isfinite(points).all();assert(points[:,0]>=60-1e-6).all()and(points[:,0]<=1860+1e-6).all()and(points[:,1]>=60-1e-6).all()and(points[:,1]<=1020+1e-6).all();lo,hi=points[:,:2].min(0),points[:,:2].max(0);u[:]=[min(u[0],lo[0]),min(u[1],lo[1]),max(u[2],hi[0]),max(u[3],hi[1])];checks+=8
 crops={str(c):[math.floor(u[0]-14),math.floor(u[1]-14),math.ceil(u[2]+14),math.ceil(u[3]+14)]for c,u in boxes.items()};folder=OUT/'connected-native-frames';folder.mkdir(exist_ok=True)
 if not(folder/'00359.jpg').exists():run(['ffmpeg','-nostdin','-v','error','-i',str(capture),'-vf','fps=30','-q:v','2','-y',str(folder/'%05d.jpg')])
 paths=sorted(folder.glob('*.jpg'));assert len(paths)>=359
 context={'capture':ref(capture),'receipt':ref(receipt),'capture_validation':ref(validation),'segment_native_union_crops':crops,'padding_px':14,'full_native_box_corner_checks':checks,'all_components_inside_original_and_editorial_bounds':True,'camera_policy':'Expanded editorial camera keeps whole original parts; not the original source-video camera. All original part poses/joints and native geometry unchanged.','source_intervals':d['intervals'],'legacy_V5_intro_not_used_as_native_footage':True}
 write(OUT/'connected-full-bounds-use.json',context);return d,crops,len(paths)

def connected_frame(clock):
 n=max(1,min(CONNECTED_COUNT,round(clock*30)+1));chapter=0 if clock<6 else 1
 return pic(str(OUT/'connected-native-frames'/f'{n:05}.jpg')).crop(CONNECTED_CROPS[str(chapter)])

def proof(t):
 im=Image.new('RGB',(1920,1080),BG)
 if t<1.8:return endcard(im.size,t,'Every interaction leaves a story.')
 if t>=28:return endcard(im.size,t-28)
 brand(im);p=(t-1.8)/26.2
 title='Watch what happens.' if t<11 else 'Keep how it connects.' if t<19 else 'Reuse what it knows.'
 label(im,(62,135),'A source-linked interaction');text(im,(58,181),title,65,bold=True)
 if t<11:
  mt=.10+(t-1.8)/9.2*3.73;f=mframe(mt);left=f.crop((77,369,900,958));right=connected_frame(mt*1.5);source=17.35+mt/4*2.95
 elif t<19:
  mt=9.10+(t-11)/8*8.72;f=mframe(mt);left=f.crop((1458,370,1772,777));right=connected_frame(6+(mt-9)*2/3);source=24.68+(mt-9)/9*.97
 else:
  mt=18.10+(t-19)/9*5.70;f=mframe(mt);left=mframe(9.15+(t-19)/9*8.55).crop((1458,370,1772,777));right=connected_frame(6+mt-18);source=24.68+(mt-18)/6*.97
 rr(im,(62,300,926,904),DARK,24);rr(im,(954,300,1858,904),'#eef1f7',24);fit(im,left,(77,348,912,883));fit(im,right,(970,345,1842,883))
 label(im,(85,317),'Original video',18,'#dfe5f2');label(im,(980,317),'Connected 3D replay',18)
 d=ImageDraw.Draw(im);d.line((940,413,940,795),fill=LINE,width=2);d.ellipse((929,580,951,602),fill=BLUE)
 text(im,(64,943),f'Source evidence · {source:05.2f} s',23,MUTED)
 if t<11:
  text(im,(960,943),'4 separate parts. One connected lid.',29,BLUE,True)
 elif t<19:
  text(im,(960,943),'Hinge + removable components',29,BLUE,True)
 else:
  text(im,(960,943),'Geometry + actions + source evidence',29,BLUE,True)
 # Quiet evolving connection/program strip, never obscures the observed parts.
 steps=['Observe','Connect','Reuse'];active=0 if t<11 else 1 if t<19 else 2
 for n,s in enumerate(steps):
  x=64+n*164;d.line((x,1022,x+125,1022),fill=BLUE if n<=active else LINE,width=4);text(im,(x,984),s,20,BLUE if n==active else MUTED,n==active)
 if t>=23:
  rr(im,(1266,994,1858,1046),'#e8eeff',14);text(im,(1286,1008),'Ready for an agent to inspect →',24,BLUE,True)
 return im

def pocket(t):
 im=Image.new('RGB',(1080,1920),BG)
 if t<1.6:return endcard(im.size,t,'Everyday objects. New possibilities.')
 if t>=24:return endcard(im.size,t-24,'A library of how things work.')
 brand(im,True)
 if t<5.0:
  label(im,(64,205),'The world is full of interactions',21);multiline(im,(58,267),'Small things.\nUseful worlds.',91,bold=True,spacing=-2)
  f=mframe(.12+(t-1.6)/3.4*3.70);fit(im,f.crop((65,365,916,975)),(60,530,1020,1425),DARK)
  center(im,1520,'Film what happens.',45);center(im,1594,'Keep more than its shape.',29,MUTED,False)
 elif t<17:
  n=min(3,int((t-5)/3));a=(t-5)%3/3;names=['Case body','Laptop lid','USB-C adapter','Coin'];verbs=['Contains. Connects.','Open. Move. Close.','Connect. Insert.','Take. Place. Reuse.'];ids=['earbuds','laptop','charger','coin'];label(im,(64,205),f'{n+1:02} / 04    captured from ordinary footage',20)
  text(im,(58,269),names[n],96,bold=True);text(im,(64,390),verbs[n],37,BLUE,True)
  rr(im,(52,525,1028,1360),'white',40,LINE)
  orbit_time=(3.05+a*1.35)if n==1 else(n*3+a*2.96+.02);idx=min(359,round(orbit_time*30));asset=pic(str(OUT/'native-orbits-gpu'/f'{idx+1:05}.jpg'));fit(im,asset,(70,548,1010,1336))
  # These are native asset orbits, never claimed as inferred joint motion.
  label(im,(415,1430),'A useful interaction',20);multiline(im,(414,1484),'Parts. Connections.\nActions. Evidence.',34,bold=True,spacing=12)
  if n==0:source=mframe(.12+a*3.70).crop((65,365,916,975))
  else:source=pic(str(OUT/'native-orbits'/ids[n]/f'{min(45,round(a*44)+1):05}.jpg'))
  fit(im,source,(64,1394,360,1780),DARK)
  ImageDraw.Draw(im).rounded_rectangle((414,1692,1016,1698),3,fill=LINE);ImageDraw.Draw(im).rounded_rectangle((414,1692,414+602*a,1698),3,fill=BLUE)
 elif t<21:
  label(im,(64,205),'From pixels to a useful library',21);multiline(im,(58,267),'Shape.\nMeaning.\nMotion.',94,bold=True,spacing=-5)
  f=mframe(18.10+(t-17)/4*5.72);fit(im,native_frame(3,18.10+(t-17)/4*5.72),(48,680,1032,1450))
  for n,s in enumerate(['3D parts','Connected actions','Source evidence']):
   y=1518+n*85;rr(im,(62,y,1018,y+64),'white',17,LINE);text(im,(86,y+15),s,28,BLUE,True)
 else:
  label(im,(64,205),'For agents and robotics',21);multiline(im,(58,267),'Build with\nwhat you see.',87,bold=True,spacing=0)
  f=rframe(1+(t-21)/3*23.90);fit(im,f.crop((570,110,1300,1040)),(45,525,1035,1660))
  center(im,1730,'Prototype the interaction.',35,BLUE)
 return im

def typed(im,xy,s,t,duration,size=30,color=INK):
 count=min(len(s),round(len(s)*ease(t/duration)));text(im,xy,s[:count]+('│' if count<len(s)else''),size,color,False)

def agent(t):
 im=Image.new('RGB',(1080,1080),BG)
 if t<1.8:return endcard(im.size,t,'An agent’s first day.')
 if t>=36:return endcard(im.size,t-36)
 brand(im,True);label(im,(48,124),'Agent workflow · illustrated conversation',16)
 if t<8:
  text(im,(45,180),'“What can I do with this?”',48,bold=True)
  rr(im,(48,280,1032,450),'white',24,'#c8d6ff');label(im,(75,306),'You',17);typed(im,(74,348),'Find a case with removable earbuds.',t-2,2.5,31)
  if t>4:
   rr(im,(48,486,1032,652),'#eaf0ff',24);label(im,(75,512),'Library via local MCP',17);typed(im,(74,558),'Found: 4 parts. Hinged lid. Source video.',t-4.2,2.0,29)
  f=mframe(.12+(t-1.8)/6.2*3.70);fit(im,connected_frame(.10+(t-1.8)/6.2*5.70),(240,674,840,1045))
 elif t<15:
  text(im,(45,180),'First, understand the object.',47,bold=True)
  f=mframe(18.1+(t-8)/7*5.72);fit(im,connected_frame(6.1+(t-8)/7*5.70),(40,274,1040,785))
  rr(im,(48,802,1032,1028),DARK,22)
  rows=[('parts','case_base · case_lid · earbud_a · earbud_b'),('connection','revolute hinge'),('evidence','video + masks + observed motion')]
  for n,(key,value)in enumerate(rows):
   y=828+n*61;label(im,(76,y),key,15,'#90aaff');text(im,(260,y-2),value,24,'#f0f3fa')
 elif t<34.5:
  # Every original task phase, both placements, closed lid and withdrawal fit.
  elapsed=t-15;source=elapsed/19.5*26.9;f=rframe(source);text(im,(45,180),'Then, prototype the task.',48,bold=True)
  fit(im,f.crop((570,110,1300,1040)),(116,250,964,980))
  r=json.loads(ROBOT_RECEIPT.read_text()) if False else ROBOT_DATA
  row=min(r['trace'],key=lambda row:abs(row['time_s']-source));goals=row['goals'];completed=sum(g['released']and g['reached']for g in goals)
  actions=['Open the case','Move the earbuds','Release on the targets','Close and withdraw']
  action=row['action_id'];number=int(action.removeprefix('action_'))if action.startswith('action_')else 0
  phase=0 if number<=13 and action not in ['film_withdraw_arm','film_final_hold']else 1 if number in [14,15,16,17,18,19,25,26,27,28,29,30]else 2 if 20<=number<=35 else 3
  rr(im,(48,918,1032,1036),'white',20,LINE);text(im,(74,939),actions[phase],29,BLUE,True);text(im,(74,983),f'{completed} / 2 earbuds released on target',22,MUTED)
  label(im,(691,996),'Local rehearsal',14)
 else:
  text(im,(45,180),'The result is reusable.',49,bold=True);f=rframe(26.85);fit(im,f.crop((570,110,1300,1040)),(140,265,940,884));rr(im,(48,903,1032,1034),'#eaf0ff',22);text(im,(77,931),'2 placements. Case closed.',34,BLUE,True);text(im,(77,985),'Models + program + evidence, together.',24,MUTED)
 return im

CUES={
'04-proof':[(0,1.8,'Every interaction leaves a story.'),(1.8,11,'Watch a real interaction beside its connected 3D replay.'),(11,19,'Separate parts. A connected hinge. Source-linked movement.'),(19,28,'Keep the geometry, actions and evidence for your agent.'),(28,30,'Interaction Twin. Show it once. Keep how it works.')],
'05-pocket-worlds':[(0,1.6,'Everyday objects. New possibilities.'),(1.6,5,'Small things. Useful worlds. Film what happens.'),(5,8,'Earbuds. Open. Remove. Return.'),(8,11,'Laptop lid. A moving part with a useful story.'),(11,14,'Charger. Connect. Insert.'),(14,17,'Coin. Take. Place. Reuse.'),(17,21,'Shape. Meaning. Motion. A library of how things work.'),(21,24,'For agents and robotics: prototype the interaction.'),(24,26,'Interaction Twin.')],
'06-agent':[(0,1.8,'An agent’s first day.'),(1.8,8,'Find a case with removable earbuds. Found: four parts, a hinged lid and source video.'),(8,15,'First, understand the object: its parts, connection and evidence.'),(15,25,'Then prototype the task: open the case and move the earbuds.'),(25,34.5,'Release both earbuds on their targets. Close the case and withdraw.'),(34.5,36,'Models, program and evidence, together.'),(36,38,'Interaction Twin. Show it once. Keep how it works.')]}

def soundtrack(path,duration,style):
 """Original deterministic, sample-free stereo procedural music; three moods."""
 rate=48000;n=round(duration*rate);t=np.arange(n)/rate;y=np.zeros(n);bpm=[72,112,86][style];beat=60/bpm;rng=np.random.default_rng(8204+style)
 # Shared harmonic family, deliberately different rhythm/arrangement.
 notes=[146.8324,174.6141,220.,261.6256]
 if style==0:
  for i,hz in enumerate(notes):y+=.030*np.sin(2*np.pi*hz*t+np.sin(t*.09+i)*.5)*(1+.22*np.sin(t*.31+i))
  for at in np.arange(0,duration,beat):
   start=int(at*rate);m=min(int(.14*rate),n-start);q=np.arange(m)/rate;y[start:start+m]+=.045*np.sin(2*np.pi*960*q)*np.exp(-q*52)
 elif style==1:
  for j,at in enumerate(np.arange(0,duration,beat/2)):
   start=int(at*rate);m=min(int(.22*rate),n-start);q=np.arange(m)/rate
   if j%2==0:y[start:start+m]+=.20*np.sin(2*np.pi*(52*q+2.8*(1-np.exp(-q*22))))*np.exp(-q*20)
   if j%4==2:y[start:start+m]+=.075*rng.standard_normal(m)*np.exp(-q*32)
   y[start:start+m]+=.025*rng.standard_normal(m)*np.exp(-q*100)
   hz=notes[(j//2)%4]*2;y[start:start+m]+=.058*np.sin(2*np.pi*hz*q)*np.exp(-q*11)
 else:
  for j,at in enumerate(np.arange(0,duration,beat/2)):
   start=int(at*rate);m=min(int(1.2*rate),n-start);q=np.arange(m)/rate;hz=notes[(j%8)//2]*(2 if j%2 else 1);env=np.exp(-q*4)*(1-np.exp(-q*120));y[start:start+m]+=.075*(np.sin(2*np.pi*hz*q)+.2*np.sin(2*np.pi*hz*2*q))*env
   if j%2==0:y[start:start+m]+=.06*np.sin(2*np.pi*55*q)*np.exp(-q*21)
 fade=np.minimum(1,t/.7)*np.minimum(1,(duration-t)/1.1);y=np.tanh(y)*fade
 stereo=np.stack([y,np.roll(y,round(.007*rate))*.98],1);stereo=np.clip(stereo,-.9,.9);pcm=(stereo*32767).astype('<i2')
 with wave.open(str(path),'wb')as w:w.setnchannels(2);w.setsampwidth(2);w.setframerate(rate);w.writeframes(pcm.tobytes())
 return {'composition':'Original deterministic procedural composition; no imported samples','seed':8204+style,'tempo_bpm':bpm,'style':['Ambient proof pulse','Syncopated museum percussion','Warm agent arpeggio'][style],'stereo_48khz':True,'speech':False,'caption_kind':'on-screen narrative description, not spoken dialogue','raw_peak':float(abs(stereo).max())}

def stamp(t):
 ms=round(t*1000);s,ms=divmod(ms,1000);m,s=divmod(s,60);return f'00:{m:02}:{s:02}.{ms:03}'
def artifacts(name):
 cues=CUES[name];(DEST/f'{name}.vtt').write_text('WEBVTT\n\n'+''.join(f'{i+1}\n{stamp(a)} --> {stamp(b)}\n{s}\n\n'for i,(a,b,s)in enumerate(cues)));(DEST/f'{name}-transcript.txt').write_text('\n\n'.join(s for a,b,s in cues)+'\n')

def render(name):
 spec=SPECS[name];w,h=spec['size'];duration=spec['duration'];fn={'04-proof':proof,'05-pocket-worlds':pocket,'06-agent':agent}[name];folder=OUT/name;folder.mkdir(exist_ok=True);snapshot=folder/'builder-source.py';snapshot.write_bytes(Path(__file__).read_bytes());silent=folder/'silent.mp4';target=DEST/f'{name}.mp4';frames=round(duration*FPS)
 encoder=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-threads','3','-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(silent)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
 samples=[];sheet=Image.new('RGB',(1280,3*384),BG);d=ImageDraw.Draw(sheet);sample_times=np.linspace(2,duration-2.3,6).tolist();sample_indices={round(v*FPS):(i,v)for i,v in enumerate(sample_times)}
 try:
  for n in range(frames):
   t=n/FPS;im=fn(t);assert im.size==(w,h);encoder.stdin.write(im.tobytes())
   if n in sample_indices:
    i,ts=sample_indices[n];thumb=ImageOps.contain(im,(630,350));sheet.paste(thumb,(i%2*640+(640-thumb.width)//2,i//2*384));d.text((i%2*640+14,i//2*384+356),f'{ts:.2f} s',font=font(18),fill=INK);path=folder/f'proof-{n:04}.jpg';im.save(path,quality=90);samples.append(ref(path))
   if n%300==0:print(name,n,'/',frames,flush=True)
 finally:encoder.stdin.close()
 errors=encoder.stderr.read().decode();code=encoder.wait();assert code==0,errors
 sheet.save(folder/'contact-sheet.jpg',quality=91);music=folder/'original-score.wav';audio=soundtrack(music,duration,list(SPECS).index(name));run(['ffmpeg','-nostdin','-v','error','-i',str(silent),'-i',str(music),'-map','0:v','-map','1:a','-af','loudnorm=I=-18:TP=-2:LRA=8,afade=t=out:st='+str(duration-.8)+':d=0.8','-c:v','copy','-c:a','aac','-b:a','160k','-t',str(duration),'-movflags','+faststart','-y',str(target)])
 poster=DEST/f'{name}-poster.jpg';run(['ffmpeg','-nostdin','-v','error','-ss',str(7 if name!='06-agent'else 10),'-i',str(target),'-frames:v','1','-q:v','2','-y',str(poster)]);artifacts(name)
 probe=json.loads(run(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(target)]).stdout);assert abs(float(probe['format']['duration'])-duration)<.08;streams=probe['streams'];assert [(s['codec_type'])for s in streams]==['video','audio'];assert streams[0]['width']==w and streams[0]['height']==h and streams[0]['pix_fmt']=='yuv420p'
 # Decode every final frame for actual content/black checks; no image cache.
 scan=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-i',str(target),'-an','-vf','scale=96:96','-pix_fmt','rgb24','-f','rawvideo','pipe:1'],stdout=subprocess.PIPE);black=[];flat=[];count=0;prev=None;diff=[]
 while True:
  raw=scan.stdout.read(96*96*3)
  if not raw:break
  assert len(raw)==96*96*3;arr=np.frombuffer(raw,np.uint8).astype(float)
  if arr.mean()<6:black.append(count)
  if arr.std()<2:flat.append(count)
  if prev is not None:diff.append(float(np.mean(abs(arr-prev))))
  prev=arr;count+=1
 assert scan.wait()==0 and count==frames and not black and not flat
 rec={'format':'interaction-twin-trailer-choice','version':1,'id':name,**spec,'movie':ref(target),'poster':ref(poster),'captions':ref(DEST/f'{name}.vtt'),'transcript':ref(DEST/f'{name}-transcript.txt'),'builder':ref(snapshot),'working_builder_path':str(Path(__file__).relative_to(ROOT)),'inputs':json.loads((OUT/'input-validation.json').read_text()),'native_orbits':ref(OUT/'native-orbits.json')if name=='05-pocket-worlds'else None,'native_framing_audit':ref(OUT/'native-framing-audit.json'),'paired_source_clock_audit':ref(OUT/'04-source-clock-audit.json')if name=='04-proof'else None,'proof_frames':samples,'contact_sheet':ref(folder/'contact-sheet.jpg'),'audio':{**audio,'file':ref(music)},'ffprobe':probe,'qa':{'decoded_frames':count,'black_frames':black,'blank_frames':flat,'mean_consecutive_small_frame_difference':float(np.mean(diff)),'all_final_frames_checked':True},'editorial_scope':{'source':'Exact saved owner footage and saved supported FK scene capture; cuts/re-timing/orbit are editorial.','agent_conversation':'Illustrated natural-language conversation; actual saved metadata/library query and complete robot trace underlie the result, not a newly executed AI conversation.','robot':'Complete authored local kinematic rehearsal, both earbud goal releases, closed lid and withdrawal. No physical grasp/contact/force validation.','native_assets':'Original triangle positions/indices/colors preserved; optional standalone camera orbits are editorial.','physics_verified':False,'trained_robot_policy':False},'connected_full_bounds_capture':ref(OUT/'connected-full-bounds-use.json'),'original_cache_reproduction':ref(OUT/'source-cache-verification.json'),'new_api_calls':0,'new_gpu_calls':0,'external_media':False,'published_to_social':False}
 write(DEST/f'{name}-provenance.json',rec);write(folder/'receipt.json',rec)
 # Intermediate silent bitstream is ours, final retains all video bytes.
 silent.unlink();print('Complete',name,target.stat().st_size,flush=True);return rec

def main():
 p=argparse.ArgumentParser();p.add_argument('--only',choices=list(SPECS));p.add_argument('--proofs-only',action='store_true');args=p.parse_args();global ROBOT_DATA,NATIVE_CROPS,CONNECTED_DATA,CONNECTED_CROPS,CONNECTED_COUNT
 motion_data,ROBOT_DATA=preparation();NATIVE_CROPS=native_crop_audit(motion_data);orbit_assets();CONNECTED_DATA,CONNECTED_CROPS,CONNECTED_COUNT=connected_preparation()
 if args.proofs_only:
  for name,spec in SPECS.items():
   folder=OUT/name;folder.mkdir(exist_ok=True);fn={'04-proof':proof,'05-pocket-worlds':pocket,'06-agent':agent}[name];sheet=Image.new('RGB',(1280,1152),BG)
   for i,t in enumerate(np.linspace(2,spec['duration']-2.3,6)):
    im=fn(float(t));thumb=ImageOps.contain(im,(630,350));sheet.paste(thumb,(i%2*640+(640-thumb.width)//2,i//2*384));text(sheet,(i%2*640+14,i//2*384+356),f'{t:.2f} s',18)
   sheet.save(folder/'preflight-contact-sheet.jpg',quality=90)
  return
 receipts=[render(name)for name in ([args.only]if args.only else SPECS)];write(OUT/'delivery.json',{'format':'interaction-twin-trailer-collection-set-b','trailers':receipts,'new_api_calls':0,'new_gpu_calls':0})
if __name__=='__main__':main()
