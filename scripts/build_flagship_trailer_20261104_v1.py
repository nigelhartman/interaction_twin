#!/usr/bin/env python3
"""Additive, visual-first 45 s film. No inference or source geometry edits.

Storyboards may use explicitly labelled historic capture placeholders. A final
render requires a new current-app robot/multi-model capture manifest; it cannot
silently ship the old staging footage. All previous films stay untouched.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, subprocess, sys, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
import flagship_capture_contract_v1 as CAPTURE
spec=importlib.util.spec_from_file_location('flagship_existing_media',ROOT/'scripts/build_trailer_collection_a_20261104.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
W,H,FPS,D=1920,1080,30,45
# Editorial crop of the recorded scene only. It excludes recorder furniture and
# generous table margins; no part/robot pose or geometric pixel is retouched.
ROBOT_CROP=(200,100,1140,620)
BG,INK,BLUE,MUTED=A.PAPER,A.INK,A.BLUE,A.MUTED
BASE=ROOT/'output/flagship-trailer-20261104-v1'
BEATS=[(0,3,'Video → interactive twins'),(3,6,'Find the parts'),(6,9,'Generate their shape'),(9,12,'Keep how it works'),(12,27,'Prototype the next move'),(27,31,'Find your next interaction'),(31,35,'Match the interfaces'),(35,39,'Prepare a new task'),(39,43,'Build a reusable library'),(43,45,'Interaction Twin')]
CUES=[(.2,3,'Film how an object works.'),(3.2,8.8,'SAM finds its parts. SAM 3D gives them shape.'),(9,12,'Keep their connections and source evidence.'),(12.2,13.8,'Place the case.'),(14.2,16.3,'Open the case.'),(16.7,21,'Move both earbuds to the tray.'),(23.5,26.6,'Then close it.'),(27.2,31,'Find the parts your agent needs.'),(31.2,35,'Match their interfaces across recordings.'),(35.2,39,'Prepare a source-linked task draft.'),(39.2,42.7,'Real demonstrations. Reusable interactions.'),(43,45,'Interaction Twin.')]
OWNER_IDS={'Earbuds':'flow-sam3d-default-e2e-20261102-v1--flow-procedure-1','Phone':'flow-capture-phone-20261103-v1--flow-procedure-1','Charger':'flow-capture-charger-20261103-v1--flow-procedure-1','Laptop':'flow-capture-laptop-20261103-v1--flow-procedure-1'}

def ref(p):
 p=Path(p);raw=p.read_bytes();return {'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def checked(r):
 p=ROOT/r['path'];assert p.resolve()==p and p.is_relative_to(ROOT) and not any(x.startswith('.')for x in p.relative_to(ROOT).parts)
 assert ref(p)==r,'Capture/evidence bytes changed';return p
def text(im,xy,s,size=30,color=INK,bold=False,mono=False):A.text(ImageDraw.Draw(im),xy,s,size,color,bold,mono)
def card(im,box,fill='white',line='#e1e6ef',radius=24):ImageDraw.Draw(im).rounded_rectangle(box,radius=radius,fill=fill,outline=line,width=2)
def fit(im,pic,box):A.contain(im,pic,box)
def label(im,xy,s,size=21,color=BLUE):text(im,xy,s,size,color,True)
def headline(im,s):text(im,(66,116),s,55,INK,True)
def header(im,a,t):
 logo=a.logo.resize((36,36),Image.Resampling.LANCZOS);im.paste(logo,(66,45),logo);text(im,(115,48),'Interaction Twin',24,INK,True)
 ImageDraw.Draw(im).line((66,1033,1854,1033),fill='#e1e6ef',width=3);ImageDraw.Draw(im).line((66,1033,66+round(1788*t/D),1033),fill=BLUE,width=3)

def all_bounds_crop(points, margin=36):
 return(max(0,math.floor(min(p[0]for p in points))-margin),max(0,math.floor(min(p[1]for p in points))-margin),min(W,math.ceil(max(p[0]for p in points))+margin),min(H,math.ceil(max(p[1]for p in points))+margin))

def robot_editorial_knots(plan):
 """Retime saved actions, preserving their full order and both goal releases."""
 actions=plan['actions'];ends=[];elapsed=0
 for row in actions:
  elapsed+=row['duration_s'];ends.append(elapsed)
 opening=next(i for i,row in enumerate(actions)if row['operation']=='set_joint'and row['value']>0)
 approach=max(0,opening-1);open_start=ends[approach-1]if approach else 0
 goals=plan['goal']['component_ids'];assert len(goals)==2
 releases=[]
 for goal in goals:
  index=next(i for i,row in enumerate(actions)if row['operation']=='detach'and row['object_id']==goal)
  # Keep each placement and the immediately following retreat in its beat.
  releases.append(ends[index+1]if index+1<len(actions)and actions[index+1]['operation']=='move_to'else ends[index])
 sources=[0,open_start,ends[opening],*releases,elapsed]
 assert all(a<b for a,b in zip(sources,sources[1:])), 'Robot beats must preserve full action order.'
 return list(zip([0,2,4.5,8,11.5,15],sources))

def interpolate_clock(knots,film_s):
 t=max(knots[0][0],min(knots[-1][0],film_s))
 for (a,x),(b,y)in zip(knots,knots[1:]):
  if t<=b:return x+(y-x)*(t-a)/(b-a)
 return knots[-1][1]

class Assets:
 def __init__(self,fresh=None):
  self.old=A.Assets();self.logo=self.old.logo;self.cache={};self.readers={};self.fresh=json.loads(fresh.read_text())if fresh else None
  p=ROOT/'showcase/public/library-index.json';self.index_ref=ref(p);index=json.loads(p.read_text());self.catalog_ref=ref(p.parent/index['catalog']['path']);catalog=json.loads((ROOT/self.catalog_ref['path']).read_text());self.records={name:next(r for r in catalog['records']if r['id']==ident)for name,ident in OWNER_IDS.items()}
  self.task_ref=ref(ROOT/'showcase/demos/charge-phone-v1/task.json');self.task=json.loads(checked(self.task_ref).read_text());self.summary=json.loads((ROOT/'showcase/demos/charge-phone-v1/summary.json').read_text())
  assert self.task['claims']['charging_success_verified']is False and self.summary['live_llm_agent']is False
  if self.fresh:
   assert self.fresh['format']=='interaction-twin-flagship-current-app-captures'and self.fresh['version']==1
   assert self.fresh['current_catalog']==self.catalog_ref and self.fresh['authored_robot_staging']is True
   # Recompute eligibility rather than trusting a historical intake PASS flag.
   # A valid timeline around a cleared WebGL buffer is still an invalid film.
   context=CAPTURE.current_context(ROOT);self.capture_validation={}
   for key in ['robot','multiple_models','jabra_parts','connected']:
    row=self.fresh[key];receipt=json.loads(checked(row['receipt']).read_text());movie=checked(row['video'])
    checks=CAPTURE.validate_receipt(key,receipt,movie,context,ROOT)
    checks['decoded_video']=CAPTURE.probe_and_pixels(movie,receipt['duration_s'],receipt['canvas'],(0,88,1280,718)if key=='robot'else None)
    self.capture_validation[key]=checks
   for key in ['robot','multiple_models','jabra_parts','connected']:checked(self.fresh[key]['video']);checked(self.fresh[key]['receipt'])
   r=json.loads(checked(self.fresh['robot']['receipt']).read_text());assert r['captured_full_task']and r['final_attached']is None and r['final_closed_reference']and r['both_earbuds_released']
   self.robot_receipt=r;self.robot_knots=robot_editorial_knots(r['typed_plan'])
   m=json.loads(checked(self.fresh['multiple_models']['receipt']).read_text());assert set(m['roles'])=={'phone','cable','charger'}and m['geometry_loaded_actual_WebGL']and m['charging_or_insertion_executed']is False
   j=json.loads(checked(self.fresh['jabra_parts']['receipt']).read_text());assert set(j['roles'])=={'case_base','case_lid','earbud_a','earbud_b'}and j['geometry_loaded_actual_WebGL']
   self.models_crop=all_bounds_crop([p for row in m['trace']for view in row['views']for p in view['projected_bounds']])
   self.parts_crop=all_bounds_crop([p for row in j['trace']for view in row['views']for p in view['projected_bounds']])
   c=json.loads(checked(self.fresh['connected']['receipt']).read_text());assert c['native_geometry_unchanged']and c['original_part_poses_unchanged']
   points=[p for row in c['trace']if row['chapter_index']==1 for b in row['projected_native_boxes']for p in b['points']];self.connected_crop=(max(0,math.floor(min(p[0]for p in points))-16),max(0,math.floor(min(p[1]for p in points))-16),min(W,math.ceil(max(p[0]for p in points))+16),min(H,math.ceil(max(p[1]for p in points))+16))
 def picture(self,p):
  if str(p)not in self.cache:self.cache[str(p)]=Image.open(p).convert('RGB')
  return self.cache[str(p)].copy()
 def decode(self,key,p,t,size):
  old=self.readers.get(key)
  if old is None or t<old.time-.04:
   if old:old.close()
   old=self.readers[key]=A.RawReader(p,max(0,t),size)
  return old.get(t).copy()
 def source(self,p):return self.old.raw_source(24.68+min(1,p)*.97,crop='hero')
 def mask(self,p):return self.old.sam(min(1,p))[0]
 def connected(self,p):
  if self.fresh:
   row=self.fresh['connected'];end=row['encoded_last_timestamp_s'];return self.decode('fresh-connected',checked(row['video']),(6+min(.999,p)*5.9)/12*end,(1920,1080)).crop(self.connected_crop)
  return self.old.hero(min(1,p))[0]
 def parts(self,p):
  if self.fresh:
   row=self.fresh['jabra_parts'];end=row['encoded_last_timestamp_s'];return self.decode('fresh-parts',checked(row['video']),min(end-.04,p*end),(1920,1080)).crop(self.parts_crop)
  return self.old.orbit(min(.999,p)*2.85)
 def robot(self,p):
  if self.fresh:
   row=self.fresh['robot'];end=row['encoded_last_timestamp_s'];self.robot_runtime_s=interpolate_clock(self.robot_knots,p*15);t=self.robot_runtime_s/self.robot_receipt['task_duration_s']*end;return self.decode('fresh-robot',checked(row['video']),min(end-.04,t),(1280,800)).crop(ROBOT_CROP)
  return self.old.robot_image(min(26.89,p*26.9))
 def models(self,p):
  if self.fresh:
   row=self.fresh['multiple_models'];end=row['encoded_last_timestamp_s'];return self.decode('fresh-models',checked(row['video']),min(end-.04,p*end),(1920,1080)).crop(self.models_crop)
  return None
 def evidence(self,role):
  source=next(r for r in self.task['sources']if r['role']==('phone'if role=='Phone'else'charger'))
  ident={'Phone':'phone_port','Cable':'cable_free_end','Charger':'adapter_port'}[role]
  row=next(r for r in source['interfaces']if r['id']==ident)['evidence'][0]
  p=ROOT/row['frame_url'].removeprefix('/files/');return self.picture(p)
 def breadth(self,name,p):
  row=self.records[name];start={'Earbuds':18,'Phone':23,'Charger':22,'Laptop':26}[name];return self.decode('breadth-'+name,ROOT/row['source']['path'],start+p*3.2,(360,640))
 def close(self):
  self.old.close()
  for r in self.readers.values():r.close()
  self.readers.clear()

def frame(t,a,*,draft=False):
 im=Image.new('RGB',(W,H),BG);header(im,a.old,t);trace={'film_s':t,'editorial':True}
 if t<3:
  headline(im,'Video → interactive twins');p=t/3
  for x,title,pic in [(66,'Video',a.source(p)),(680,'SAM',a.mask(p)),(1294,'Connected twin',a.connected(p))]:
   card(im,(x,228,x+560,954));fit(im,pic,(x+16,244,528,628));label(im,(x+26,895),title,27)
  for x in [638,1252]:text(im,(x,516),'→',42,BLUE,True)
  trace.update(kind='source_mask_connected_montage',source_scope='saved_supported_Jabra_intervals_editorially_mapped')
 elif t<6:
  headline(im,'Find the parts');p=(t-3)/3
  card(im,(66,220,1110,970));fit(im,a.mask(p),(88,238,1000,710))
  for i,s in enumerate(['Case','Lid','Earbud A','Earbud B']):
   y=310+i*125;ImageDraw.Draw(im).ellipse((1234,y+10,1250,y+26),fill=['#3766ed','#dc8a3d','#8d72d8','#24a4c5'][i]);text(im,(1290,y),s,47,INK,True)
  label(im,(1218,834),'SAM 3.1',25);trace.update(kind='actual_saved_segmentation')
 elif t<9:
  headline(im,'Generate their shape');fit(im,a.parts((t-6)/3),(100,195,1720,790));label(im,(66,939),'SAM 3D · separate parts',24)
  trace.update(kind='actual_native_WebGL_orbits',geometry_modified=False)
 elif t<12:
  headline(im,'Keep how it works');fit(im,a.connected((t-9)/3),(66,205,1140,780));card(im,(1250,315,1852,858));p=(t-9)/3
  for i,s in enumerate(['Parts','Connections','Actions','Evidence']):
   if p>i*.17:label(im,(1296+round((1-A.ease(min(1,(p-i*.17)/.22)))*35),369+i*113),s,35)
  label(im,(1277,921),'Muse + source constraints',22);trace.update(kind='supported_connected_scene_and_saved_program_fields')
 elif t<27:
  p=(t-12)/15;headline(im,'Prototype the next move');fit(im,a.robot(p),(40,190,1840,780))
  label(im,(70,970),'Robot task prototype',22,MUTED)
  phase='Place the case'if p*15<2 else'Open the case'if p*15<4.5 else'Move both earbuds'if p*15<11.5 else'Close the case'
  text(im,(1400,151),phase,25,BLUE,True);trace.update(kind='current_robot'if a.fresh else'historic_robot_placeholder',robot_progress=p,full_source_order_preserved=True)
  if a.fresh:trace.update(captured_task_runtime_s=a.robot_runtime_s,editorial_retiming_knots=a.robot_knots,caption=phase)
 elif t<31:
  headline(im,'Find your next interaction');p=(t-27)/4;models=a.models(p*.5)
  if models:fit(im,models,(40,213,1840,780))
  else:
   for i,name in enumerate(['Phone','Cable','Charger']):
    x=66+i*614;card(im,(x,245,x+560,958));fit(im,a.evidence(name),(x+20,270,520,590));label(im,(x+28,903),name,31)
  for i,s in enumerate(['Phone · partial surface','Cable','Charger']):label(im,(115+i*614,938),s,28)
  trace.update(kind='actual_multi_model_load'if models else'evidence_only_model_capture_pending',all_three_roles=['phone','cable','charger'])
 elif t<35:
  headline(im,'Match the interfaces');p=(t-31)/4;models=a.models(.5+p*.5)
  if models:fit(im,models,(40,213,1840,780))
  else:
   for i,name in enumerate(['Phone','Cable','Charger']):
    x=66+i*614;card(im,(x,245,x+560,868));fit(im,a.evidence(name),(x+20,270,520,545));label(im,(x+28,805),name,29)
  for x,s in [(260,'USB-C'),(878,'USB-C'),(1492,'USB-C')]:label(im,(x,951),s,31)
  d=ImageDraw.Draw(im);phase=A.ease(p);d.line([(395,918),(395+round(518*phase),918)],fill=BLUE,width=5);d.line([(1010,918),(1010+round(518*phase),918)],fill=BLUE,width=5)
  label(im,(72,194),'Owner-described interfaces',20,MUTED)
  trace.update(kind='cross_recording_interface_candidates',headset_USB_C_compatibility_claimed=False,physical_fit_verified=False)
 elif t<39:
  headline(im,'Prepare a new task');p=(t-35)/4;models=a.models(p)
  if models:fit(im,models,(66,390,1130,580))
  else:
   for i,s in enumerate(['Phone','Cable','Charger']):fit(im,a.evidence(s),(66+i*375,440,350,490))
  card(im,(66,223,1854,365),fill='white');words=a.task['instruction'];count=min(len(words),int(A.ease(min(1,p*2.4))*len(words)));text(im,(102,265),words[:count]+('│'if count<len(words)else''),38,INK,True)
  if p>.25:
   card(im,(1250,405,1854,966),fill='#eaf0ff',line='#cfdbff');label(im,(1290,445),'Source-linked task draft',28)
   # Condensed editorial labels over actual saved task templates. These are
   # source-linked proposals, never executed insertion or charging actions.
   rows=[('Retrieve parts','2 source recordings'),('Reuse charger step','Charger · 23.06 s'),('Propose phone link','Phone · 23.76 s')]
   for i,(s,note)in enumerate(rows):
    if p>.28+i*.16:label(im,(1292,538+i*123),s,32);text(im,(1294,585+i*123),note,24,MUTED)
  label(im,(73,982),'Saved MCP workflow',20,MUTED)
  trace.update(kind='actual_saved_cross_recording_MCP_task_explained',saved_MCP_calls=a.summary['mcp_calls'],live_LLM_conversation=False,charging_or_robot_insertion_executed=False)
 elif t<43:
  headline(im,'Build a reusable library');p=(t-39)/4
  for i,name in enumerate(['Earbuds','Phone','Charger','Laptop']):
   x=66+i*453;card(im,(x,224,x+430,965));fit(im,a.breadth(name,p),(x+10,242,410,636));label(im,(x+24,908),name,29)
  trace.update(kind='four_owner_recordings',recording_ids=list(OWNER_IDS.values()),camera_recording_excluded=True)
 else:
  im=A.ending(t-43,a.old);trace.update(kind='brand')
 if draft and not a.fresh:text(im,(69,1008),'STORYBOARD · fresh current-app robot and model capture pending',17,'#8e5217')
 return im,trace

def storyboard(folder,a):
 folder.mkdir(parents=True,exist_ok=False);proofs=[]
 for i,t in enumerate([.9,3.8,6.8,10,14,20,25.8,28.5,32.8,36.5,40.4,44]):
  im,trace=frame(t,a,draft=True);p=folder/f'frame-{i:02}.jpg';im.save(p,quality=94);proofs.append({'time_s':t,'file':ref(p),'trace':trace})
 A.contact_sheet(proofs,folder/'storyboard.jpg')
 story={'format':'interaction-twin-flagship-storyboard','version':1,'duration_s':D,'title':'Video to reusable interactions','beats':[{'start_s':s,'end_s':e,'title':v}for s,e,v in BEATS],'narration':[{'start_s':s,'end_s':e,'text':v}for s,e,v in CUES],'frames':proofs,'final_capture_required':['current_robot','simultaneously_loaded_phone_cable_charger'],'scope':'Current phone/cable/charger interface descriptions support candidates, not physical insertion or charging. Headset USB-C is not evidenced. Native original assets and source-visible 2.5D remain distinct.','music':'Original project procedural score; no external samples.','new_API_calls':0,'new_GPU_calls':0,'website_modified':False}
 write(folder/'storyboard.json',story);print(json.dumps({'storyboard':ref(folder/'storyboard.json'),'contact_sheet':ref(folder/'storyboard.jpg')}))

def audio(folder):
 folder.mkdir(parents=True,exist_ok=False)
 name='flagship-20261104-v1';A.STORIES[name]={'duration_s':D,'cues':CUES,'voice':'Samantha','music':{'bpm':96,'kind':'mission','seed':8404}}
 target,cues=A.make_audio(name,folder)
 write(folder/'narration.json',{'cues':cues,'local_voice':'Samantha','new_API_calls':0,'new_GPU_calls':0})
 print(json.dumps({'audio':ref(target),'audio_provenance':ref(folder/'audio.json')}));return target

def render(folder,media,a,audio_path):
 if not a.fresh:raise ValueError('Final video requires fresh current-app captures; historic placeholders cannot ship.')
 folder.mkdir(parents=True,exist_ok=False);media.mkdir(parents=True,exist_ok=False);(folder/'builder-source.py').write_bytes(Path(__file__).read_bytes());movie=media/'interaction-twin-flagship-45s.mp4'
 command=['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-i',str(audio_path),'-map','0:v','-map','1:a','-t',str(D),'-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(movie)]
 encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE);traces=[];proofs=[];proof_at={round(t*FPS)for t in [.9,3.8,6.8,10,14,20,25.8,28.5,32.8,36.5,40.4,44]};start=time.monotonic()
 try:
  for n in range(D*FPS):
   im,trace=frame(n/FPS,a);trace['frame']=n;traces.append(trace)
   if n in proof_at:
    p=folder/f'proof-{n:04}.jpg';im.save(p,quality=95);proofs.append({'time_s':n/FPS,'file':ref(p)})
   encoder.stdin.write(im.tobytes())
  encoder.stdin.close();err=encoder.stderr.read();code=encoder.wait()
  if code:raise RuntimeError(err.decode()[-2000:])
 except BaseException:encoder.kill();encoder.wait();raise
 probe=A.verify_video(movie,{'duration_s':D});A.contact_sheet(proofs,folder/'contact-sheet.jpg')
 A.V.run(['ffmpeg','-nostdin','-v','error','-ss','16','-i',str(movie),'-frames:v','1','-q:v','2',str(media/'poster.jpg')])
 cues=[{'start_s':s,'end_s':e,'text':v}for s,e,v in CUES];vtt='WEBVTT\n\n'+''.join(f'{i}\n{A.V.stamp(r["start_s"])} --> {A.V.stamp(r["end_s"])}\n{r["text"]}\n\n'for i,r in enumerate(cues,1));(media/'captions.vtt').write_text(vtt);(media/'transcript.txt').write_text('\n\n'.join(r['text']for r in cues)+'\n')
 write(folder/'editorial-clock.json',{'frames':traces,'full_robot_source_order_preserved':True,'part_geometry_modified':False,'source_observation_and_native_representations_separate':True})
 result={'format':'interaction-twin-flagship-film','version':1,'duration_s':D,'canvas':[W,H],'fps':FPS,'movie':ref(movie),'poster':ref(media/'poster.jpg'),'captions':ref(media/'captions.vtt'),'transcript':ref(media/'transcript.txt'),'builder_snapshot':ref(folder/'builder-source.py'),'audio':ref(audio_path),'current_catalog':a.catalog_ref,'fresh_current_app_captures':a.fresh,'capture_revalidation':a.capture_validation,'saved_cross_recording_task':a.task_ref,'story':BEATS,'clock':ref(folder/'editorial-clock.json'),'contact_sheet':ref(folder/'contact-sheet.jpg'),'probe':probe,'new_API_calls':0,'new_GPU_calls':0,'original_music':True,'external_media_used':False,'all_four_breadth_recordings_owner_authored':True,'camera_recording_included':False,'physical_accuracy_verified':False,'robotics_scope':'authored_kinematic_prototype','agent_scope':'illustrated_replay_of_actual_saved_MCP_task_draft_not_live_LLM_or_insertion','website_modified':False,'publication':False,'elapsed_s':time.monotonic()-start}
 result.update(capture_validation_helper=ref(ROOT/'scripts/flagship_capture_contract_v1.py'),editorial_robot_pacing={'knots_film_elapsed_to_captured_task_s':a.robot_knots,'all_actions_retained_in_original_order':True,'new_pose_interpolation':False},editorial_crops_xyxy={'robot':ROBOT_CROP,'multiple_models':a.models_crop,'four_parts':a.parts_crop,'connected':a.connected_crop},audio_provenance=ref(audio_path.parent/'audio.json'))
 write(media/'provenance.json',result);write(folder/'verification.json',result);print(json.dumps({'movie':result['movie'],'provenance':ref(media/'provenance.json')}))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--storyboard',type=Path);p.add_argument('--audio-folder',type=Path);p.add_argument('--fresh-captures',type=Path);p.add_argument('--render-folder',type=Path);p.add_argument('--media-folder',type=Path);p.add_argument('--audio',type=Path);args=p.parse_args()
 if args.audio_folder:audio(args.audio_folder.resolve())
 if args.storyboard or args.render_folder:
  a=Assets(args.fresh_captures)
  try:
   if args.storyboard:storyboard(args.storyboard.resolve(),a)
   if args.render_folder:
    if not(args.media_folder and args.audio):p.error('--render-folder needs --media-folder and --audio')
    render(args.render_folder.resolve(),args.media_folder.resolve(),a,args.audio.resolve())
  finally:a.close()
 if not(args.audio_folder or args.storyboard or args.render_folder):p.error('Choose an output action.')
if __name__=='__main__':main()
