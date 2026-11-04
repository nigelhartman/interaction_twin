#!/usr/bin/env python3
"""Exactly180-second local narrated demo draft from actual saved app evidence.

Additive only: no API, remote voice, media upload, source/model edit or trailer
replacement. Screenshot pixels are preserved in declared crops/scales/editorial
pans. Original video frames are sampled; no invented source or robot frames.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,math,subprocess,sys,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/submission-finalization-20261103/walkthrough'
MEDIA=ROOT/'showcase/media'
PREFIX='submission-walkthrough-20261103-v1'
W,H,FPS,TARGET=1920,1080,24,180
BG,INK,MUTED,BLUE,LINE='#f7f8fb','#151923','#626b7b','#3766ed','#dfe3eb'
FILM=ROOT/'showcase/media/interaction-twin-explainer-20261103-v6-refined45.mp4'
ROBOT=ROOT/'output/trailer-refinement-20261103/robot-complete-capture-v3.webm'
VOICE='Samantha';RATE=162
CHAPTERS=[
 {'id':'opening','start':0,'end':10,'title':'A video in. An object program out.',
  'narration':'A video shows parts, connections, and a task. Interaction Twin turns that demonstration into an inspectable, reusable object program.'},
 {'id':'source','start':10,'end':24,'title':'Start with an ordinary recording.',
  'narration':"Start with ordinary footage. This is Nigel's own earbud recording: a case opens, the earbuds move, and the lid closes. We keep the original source and its timestamps."},
 {'id':'models','start':24,'end':42,'title':'Separate the parts. Recover their shape.',
  'narration':'SAM three point one separates visible components. SAM three D Objects generates a separate draft mesh for each part. Muse Spark proposes what each part is, how parts relate, and which actions the video supports. These are saved model outputs.'},
 {'id':'connections','start':42,'end':58,'title':'Keep the parts connected.',
  'narration':'Static meshes are only the beginning. Local geometry fitting estimates connected joints and motion from the saved evidence. Here the lid follows a hinge, while the earbuds remain removable. This presentation uses a documented derived assembly.'},
 {'id':'program','start':58,'end':76,'title':'More than a mesh.',
  'narration':'The result is more descriptive than a mesh. Each object program links component assets, estimated joints and angles, relationships, demonstrated steps, and source evidence. Download the geometry and JSON. Unknown physical properties remain explicit, rather than becoming hidden assumptions.'},
 {'id':'library','start':76,'end':94,'title':'A library built from demonstrations.',
  'narration':'The curated library currently contains six recordings and fifteen SAM three D models. Browse parts and interactions, expand an example, compare its source and estimated replay, or inspect its downloads. New uploads stay private until the owner reviews and approves them.'},
 {'id':'agents','start':94,'end':108,'title':'The same library, for your agent.',
  'narration':'An agent can search the same approved library through read only M C P tools. It retrieves exact identifiers, source linked programs, geometry, and evidence. The one line connection guide makes this discoverable.'},
 {'id':'composition','start':108,'end':133,'title':'One task. Two recordings.',
  'narration':'For example: prepare a task to charge this phone. A real scripted M C P client made forty two calls across the phone and charger recordings, verifying six native models and nine source images. Typed owner context provides USB C interface candidates. A separate Muse assessment critiques the draft. Fit, cable identity, power compatibility, and charging success remain unverified.'},
 {'id':'robot','start':133,'end':161,'title':'Rehearse a reusable robot task.',
  'narration':'The robotics lab makes the handoff tangible. This complete browser rehearsal uses an authored five degree of freedom arm and the generated earbud parts. It opens the connected case, removes both earbuds, releases each onto a goal, closes the lid, and withdraws. The motion is a typed kinematic demonstration, not a trained policy, contact simulation, or physical robot execution.'},
 {'id':'limits','start':161,'end':173,'title':'Inspectable today. Physical validation next.',
  'narration':'Our next work is physical validation: accurate scale and contact, better generated geometry, flexible cables, and simulator testing. Ordinary video remains the input; no checkerboard is required.'},
 {'id':'end','start':173,'end':180,'title':'From demonstrations to reusable object programs.',
  'narration':'Interaction Twin. Built by Nigel Hartman. From demonstrations to reusable object programs.'},
]


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def ref(path,kind,**extra):
 path=Path(path);return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size,kind=kind,**extra)
def read(path):return json.loads(Path(path).read_text())
def write_new(path,data):
 with Path(path).open('x')as f:json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def run(args,log):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE);Path(log).write_bytes(r.stdout+r.stderr)
 if r.returncode:raise ValueError('Local media command failed; inspect '+str(log))
 return r.stdout
def probe(path):return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))
def font(size,bold=False):return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold'if bold else'')+'.ttf',size)
def text(im,xy,value,size=32,fill=INK,bold=False):ImageDraw.Draw(im).multiline_text(xy,value,font=font(size,bold),fill=fill,spacing=11)
def ease(v):v=max(0,min(1,v));return v*v*(3-2*v)
def stamp(value):
 ms=round(value*1000);return f'{ms//3600000:02d}:{ms//60000%60:02d}:{ms//1000%60:02d}.{ms%1000:03d}'


def prepare():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'narration').mkdir(exist_ok=True)
 if (OUT/'preparation.json').exists():raise ValueError('Prepared source contract is immutable')
 if sha(FILM)!='3545dcb03d2c5101fa3c986e32122617fda205635c9b026aba2ae7bab3526627':raise ValueError('Official V6 film changed')
 rp=read(ROOT/'output/trailer-refinement-20261103/robot-complete-capture-v3.json')
 if not rp.get('captured_full_task')or rp.get('duration_s')!=26.900000000000002 or not all(g['released']and g['reached']for g in rp['final_goals'])or rp['final_gripper_attached']is not None:
  raise ValueError('Complete robot proof missing')
 paths=[('film',FILM,'exact_unchanged_official_V6_actual_Jabra_chapters'),('robot',ROBOT,'actual_complete_authored_browser_robot_recording'),
  ('robot_proof',ROOT/'output/trailer-refinement-20261103/robot-complete-capture-v3.json','recorded_646_state_complete_task_proof'),
  ('film_proof',ROOT/'output/brand-trailer-20261103/trailer-v6-delivery.json','exact_source_and_derived_geometry_provenance'),
  ('brand',ROOT/'showcase/brand/v2/mark-1024.png','original_brand_v2_mark'),
  ('task',ROOT/'showcase/demos/charge-phone-v1/task.json','actual_MCP_returned_deterministic_task'),
  ('mcp_trace',ROOT/'showcase/demos/charge-phone-v1/trace.json','actual_42_call_stdio_trace'),
  ('muse_review',ROOT/'showcase/demos/charge-phone-v1/muse-review.json','separate_real_metadata_only_Muse_critic')]
 inputs={name:ref(path,kind)for name,path,kind in paths}
 captures={}
 for name in ['library-sam3d','jabra-description','jabra-downloads','agents-page','composition-top','composition-task','composition-trace']:
  captures[name]=ref(OUT/'captures'/f'{name}.png','actual_CUA_app_screenshot_or_pixel_preserving_declared_crop')
 for name in ['jabra-description','jabra-downloads']:
  captures[name]['original_capture']=ref(OUT/'captures'/f'{name}-full.png','actual_CUA_full_page_capture')
  captures[name]['crop_bounds']=read(OUT/'captures'/f'{name}-bounds.json')
 prepare={'format':'interaction-twin-submission-walkthrough-preparation','version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
  'target_duration_s':TARGET,'frame_rate':FPS,'resolution':[W,H],'chapters':CHAPTERS,'inputs':inputs,'captures':captures,
  'robot_source_timing':{'declared_capture_duration_s':rp['duration_s'],'event_trace_rows':len(rp['trace']),'actual_decoded_frames':645,'fps':24,'encoded_span_s':26.875,'source_frame_range':[0,644],'final_state_hold_s':1.125},
  'voice':VOICE,'voice_source':'bundled_local_macOS_say_synthetic_voice','rate':RATE,
  'policy':{'new_api_calls':0,'remote_media_upload':False,'external_music':False,'official45sfilm_unchanged':True,
   'youtube_camera_footage_included':False,'fresh_model_run_demonstrated':False,'robot_motion':'authored_kinematic_rehearsal',
   'physical_simulation_executed':False,'physical_accuracy_verified':False,'user_review_required_before_submission':True},
  'editorial_rules':['Actual screenshots retain pixels; only declared crop, scale and vertical pan are applied.',
   'V6 actual saved-model/replay chapters reused and slowed with frame holds; no novel frames generated.',
   'Robot segment includes all645decoded frames at24fps (26.875s) at1x plus final-state hold; source event metadata reports26.9s.',
   'Narration is local synthetic voice; no implication that this is the creator speaking.']}
 write_new(OUT/'preparation.json',prepare)
 script='Interaction Twin — 180-second narrated submission DEMO DRAFT\nLocal synthetic narration: macOS Samantha. Review before submission.\n\n'
 for c in CHAPTERS:script+=f'{stamp(c["start"])} – {stamp(c["end"])} | {c["title"]}\n{c["narration"]}\n\n'
 (OUT/'transcript.txt').write_text(script)
 print(json.dumps({'prepared':True,'chapters':len(CHAPTERS),'words':sum(len(c['narration'].split())for c in CHAPTERS)}))


def speech():
 (OUT/'narration').mkdir(exist_ok=True)
 plan=OUT/'narration-plan.json'
 if not plan.exists():write_new(plan,{'format':'local_walkthrough_narration_plan','voice':VOICE,'rate':RATE,'chapters':CHAPTERS})
 rows=[]
 for j,c in enumerate(CHAPTERS):
  stem=OUT/'narration'/f'{j:02d}-{c["id"]}';txt=stem.with_suffix('.txt');aiff=stem.with_suffix('.aiff');wav=stem.with_suffix('.wav')
  if aiff.exists()or wav.exists():raise ValueError('Retain existing speech take; do not overwrite')
  txt.write_text(c['narration']+'\n')
  run(['say','-v',VOICE,'-r',str(RATE),'-f',str(txt),'-o',str(aiff)],stem.with_suffix('.say.log'))
  run(['ffmpeg','-hide_banner','-v','error','-i',str(aiff),'-ar','48000','-ac','2','-c:a','pcm_s16le',str(wav)],stem.with_suffix('.convert.log'))
  dur=float(probe(wav)['format']['duration']);available=c['end']-c['start']-.8
  if not dur>.1:raise ValueError('Local system speech returned no audio')
  tempo=max(1,dur/available)
  if tempo>1.2:raise ValueError('Narration too long; preserve take and revise planned script')
  final=wav
  if tempo>1.00001:
   final=stem.with_name(stem.name+'-tempo').with_suffix('.wav')
   run(['ffmpeg','-hide_banner','-v','error','-i',str(wav),'-af',f'atempo={tempo:.9f}','-ar','48000','-ac','2','-c:a','pcm_s16le',str(final)],stem.with_suffix('.tempo.log'))
  fd=float(probe(final)['format']['duration'])
  row={'chapter':c['id'],'start_s':c['start']+.35,'duration_s':fd,'source_duration_s':dur,'tempo':tempo,'audio':ref(final,'local_synthetic_narration_PCM'),'original':ref(aiff,'unaltered_macOS_say_output')}
  rows.append(row);print(json.dumps({'chapter':c['id'],'audio_s':fd,'tempo':tempo}),flush=True)
 write_new(OUT/'speech.json',{'voice':VOICE,'rate':RATE,'local_synthetic_voice':True,'rows':rows})
 args=['ffmpeg','-hide_banner','-v','error']
 for r in rows:args+=['-i',str(ROOT/r['audio']['path'])]
 filters=[]
 for j,r in enumerate(rows):
  delay=round(r['start_s']*1000);filters.append(f'[{j}:a]adelay={delay}|{delay},apad,atrim=duration=180[a{j}]')
 filters.append(''.join(f'[a{j}]'for j in range(len(rows)))+f'amix=inputs={len(rows)}:normalize=0,alimiter=limit=0.95,atrim=duration=180[a]')
 args+=['-filter_complex',';'.join(filters),'-map','[a]','-ar','48000','-ac','2','-c:a','pcm_s16le',str(OUT/'narration-timeline.wav')]
 run(args,OUT/'audio-mix.log')
 vtt='WEBVTT\n\n'
 for j,(c,r)in enumerate(zip(CHAPTERS,rows)):
  words=c['narration'].split();groups=[];current=[]
  for word in words:
   current.append(word)
   if len(current)>=12:groups.append(' '.join(current));current=[]
  if current:groups.append(' '.join(current))
  total=sum(len(g.split())for g in groups);cursor=r['start_s']
  for k,g in enumerate(groups):
   end=cursor+r['duration_s']*len(g.split())/total
   vtt+=f'{j+1}-{k+1}\n{stamp(cursor)} --> {stamp(end)}\n{g}\n\n';cursor=end
 (OUT/'captions.vtt').write_text(vtt)


class Clip:
 def __init__(self,path):
  self.path=Path(path);self.cap=cv2.VideoCapture(str(path));self.last=-1;self.frame=None
  if not self.cap.isOpened():raise ValueError('Actual source video cannot be decoded')
  self.fps=self.cap.get(cv2.CAP_PROP_FPS) or 24
 def at(self,t):
  target=max(0,int(t*self.fps+.0001))
  if target==self.last:return self.frame
  # These browser WebMs have no duration/index. Random CAP_PROP_POS_FRAMES
  # seeking silently returns the first frame, so decode every source frame in
  # sequence instead. On rewind, reopen and replay; never substitute a frame.
  if target<self.last:
   self.cap.release();self.cap=cv2.VideoCapture(str(self.path));self.last=-1;self.frame=None
  while self.last<target:
   ok,bgr=self.cap.read()
   if not ok:raise ValueError('Source video frame unavailable at'+str(t))
   self.last+=1;self.frame=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
  return self.frame


def panel(im,source,box):
 x,y,w,h=box;fit=ImageOps.contain(source,(w,h),Image.Resampling.LANCZOS);im.paste(fit,(x+(w-fit.width)//2,y+(h-fit.height)//2))
def screenshot_panel(im,source,u,box=(70,220,1780,790),start=0,end=1):
 x,y,w,h=box;scaled=source.resize((w,round(source.height*w/source.width)),Image.Resampling.LANCZOS)
 if scaled.height<=h:im.paste(scaled,(x,y+(h-scaled.height)//2));return
 top=round((scaled.height-h)*(start+(end-start)*ease(u)));im.paste(scaled.crop((0,top,w,top+h)),(x,y))
def base(index):
 im=Image.new('RGB',(W,H),BG);logo=Image.open(ROOT/'showcase/brand/v2/mark-1024.png').convert('RGBA');logo.thumbnail((43,43));im.paste(logo,(64,36),logo)
 text(im,(122,42),'INTERACTION TWIN',24,INK,True);text(im,(1630,45),'DEMO DRAFT',17,MUTED)
 text(im,(68,118),CHAPTERS[index]['title'],49,INK,True)
 return im

def frame(index,u,assets,film,robot):
 c=CHAPTERS[index];im=base(index)
 if index==0:
  text(im,(82,365),'Capture once.\nKeep how it works.',78,INK,True)
  text(im,(87,640),'Parts  /  Connections  /  Actions  /  Evidence',28,MUTED)
  logo=assets['brand'].copy();logo.thumbnail((500,500));im.paste(logo,(1280,320),logo if logo.mode=='RGBA'else None)
 elif index==1:
  # Real owner footage is already embedded unaltered in exact V6 chapter2.
  im=film.at(2+u*3).copy();text(im,(1630,45),'DEMO DRAFT',17,MUTED)
 elif index==2:
  source=5+u*9 # V6 actual mask and native-shape chapters, frame-held slowdown.
  im=film.at(source).copy();text(im,(1630,45),'DEMO DRAFT',17,MUTED)
 elif index==3:
  im=film.at(14+u*10).copy();text(im,(1630,45),'DEMO DRAFT',17,MUTED)
 elif index==4:
  if u<.58:screenshot_panel(im,assets['jabra-description'],u/.58,(70,225,1780,780),0,.75)
  else:screenshot_panel(im,assets['jabra-downloads'],(u-.58)/.42,(70,225,1780,780),0,1)
 elif index==5:screenshot_panel(im,assets['library-sam3d'],u,(70,225,1780,780),0,.78)
 elif index==6:screenshot_panel(im,assets['agents-page'],u,(70,225,1780,780),0,.55)
 elif index==7:
  if u<.43:screenshot_panel(im,assets['composition-top'],u/.43,(70,225,1780,780),0,1)
  elif u<.80:screenshot_panel(im,assets['composition-task'],(u-.43)/.37,(70,225,1780,780),0,1)
  else:screenshot_panel(im,assets['composition-trace'],(u-.8)/.2,(70,225,1780,780),0,1)
 elif index==8:
  text(im,(72,187),'Authored browser rehearsal · complete task at 1×',20,MUTED)
  panel(im,robot.at(min(u*28,644/24)),(35,230,1850,800))
 elif index==9:
  pairs=[('01','Metric geometry + contact'),('02','Flexible cable behavior'),('03','Independent simulator validation')]
  for j,(n,t)in enumerate(pairs):text(im,(92,344+j*160),n,30,BLUE,True);text(im,(205,330+j*160),t,48,INK,True)
  text(im,(95,884),'Ordinary video stays the input.',31,MUTED)
 else:
  text(im,(85,350),'Interaction Twin',90,INK,True);text(im,(90,495),'From demonstrations to\nreusable object programs.',56,BLUE,True)
  text(im,(94,758),'Built by Nigel Hartman',28,MUTED);text(im,(94,918),'Narrated draft · local synthetic voice · review before submission',20,MUTED)
 return im


def render():
 prep=read(OUT/'preparation.json');speech=read(OUT/'speech.json')
 for item in [*prep['inputs'].values(),*prep['captures'].values()]:
  if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Prepared input bytes changed')
 output=OUT/(PREFIX+'.mp4')
 if output.exists():raise ValueError('Existing movie must remain unchanged')
 assets={n:Image.open(ROOT/r['path']).convert('RGB')for n,r in prep['captures'].items()};assets['brand']=Image.open(ROOT/prep['inputs']['brand']['path']).convert('RGBA')
 film=Clip(FILM);robot=Clip(ROBOT)
 args=['ffmpeg','-hide_banner','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{W}x{H}','-framerate',str(FPS),'-i','pipe:0','-i',str(OUT/'narration-timeline.wav'),'-map','0:v:0','-map','1:a:0','-c:v','libx264','-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-ar','48000','-ac','2','-frames:v',str(TARGET*FPS),'-t',str(TARGET),'-movflags','+faststart',str(output)]
 log=(OUT/'render.log').open('wb');process=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=log,stderr=log);started=time.monotonic();chapter=0
 try:
  for k in range(TARGET*FPS):
   t=k/FPS
   while chapter<len(CHAPTERS)-1 and t>=CHAPTERS[chapter]['end']:chapter+=1
   c=CHAPTERS[chapter];im=frame(chapter,(t-c['start'])/(c['end']-c['start']),assets,film,robot)
   process.stdin.write(im.tobytes())
   if k%240==0:print(json.dumps({'render_seconds':t,'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
  process.stdin.close();code=process.wait()
  if code:raise ValueError('Render failed; log preserved')
 finally:
  if process.poll()is None:process.terminate()
  log.close()
 print(json.dumps({'render_complete':True,'path':str(output.relative_to(ROOT)),'elapsed_s':round(time.monotonic()-started,1)}))


def verify():
 movie=OUT/(PREFIX+'.mp4');d=probe(movie);write_new(OUT/'ffprobe.json',d)
 video=next(s for s in d['streams']if s['codec_type']=='video');audio=next(s for s in d['streams']if s['codec_type']=='audio')
 if float(d['format']['duration'])!=180 or int(video.get('nb_frames',0))!=4320 or video['r_frame_rate']!='24/1' or [video['width'],video['height']]!=[W,H]:raise ValueError('Exact180second4320frame contract failed')
 if int(audio['sample_rate'])!=48000 or audio['channels']!=2:raise ValueError('Audio contract failed')
 run(['ffmpeg','-hide_banner','-v','error','-xerror','-i',str(movie),'-f','null','-'],OUT/'full-decode.log')
 run(['ffmpeg','-hide_banner','-i',str(movie),'-af','volumedetect','-vn','-sn','-f','null','-'],OUT/'audio-levels.log')
 cap=cv2.VideoCapture(str(movie));sheet=Image.new('RGB',(1600,1350),BG);frames=[]
 for j,c in enumerate(CHAPTERS):
  at=c['start']+(c['end']-c['start'])*.48;cap.set(cv2.CAP_PROP_POS_MSEC,at*1000);ok,bgr=cap.read()
  if not ok:raise ValueError('Contact-sheet frame unavailable')
  im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));path=OUT/f'qa-{j:02d}-{c["id"]}.jpg';im.save(path,quality=95)
  thumb=im.resize((533,300),Image.Resampling.LANCZOS);x=(j%3)*533;y=(j//3)*337;sheet.paste(thumb,(x,y));text(sheet,(x+9,y+307),f'{stamp(at)} · {c["id"]}',17,INK)
  frames.append(ref(path,'decoded_final_movie_frame',time_s=at))
 cap.release();sheet.save(OUT/'contact-sheet.jpg',quality=95)
 # Explicit end-state frames: both releases, closed lid and withdrawal.
 finalproof=[]
 for at in [158,159.8,160.8]:
  path=OUT/f'qa-robot-goals-{at}.png';run(['ffmpeg','-hide_banner','-v','error','-ss',str(at),'-i',str(movie),'-frames:v','1',str(path)],OUT/f'qa-robot-goals-{at}.log');finalproof.append(ref(path,'decoded_complete_robot_end_state',time_s=at))
 manifest={'format':'interaction-twin-submission-walkthrough-delivery','version':1,'status':'reviewable_narrated_demo_draft','movie':ref(movie,'exact180second_narrated_demo_draft'),
  'preparation':ref(OUT/'preparation.json','source_contract'),'speech':ref(OUT/'speech.json','local_synthetic_voice_timing'),
  'transcript':ref(OUT/'transcript.txt','exact_spoken_transcript'),'captions':ref(OUT/'captions.vtt','English_WebVTT'),
  'contact_sheet':ref(OUT/'contact-sheet.jpg','decoded_final_movie_QA'),'frames':frames,'robot_end_state_frames':finalproof,
  'full_decode':{'passed':True,'duration_s':180,'frames':4320,'audio_rate':48000,'channels':2,'log':ref(OUT/'full-decode.log','full_video_audio_decode_log')},
  'new_api_calls':0,'published':False,'submitted':False,'official45s_trailer_unchanged':sha(FILM)=='3545dcb03d2c5101fa3c986e32122617fda205635c9b026aba2ae7bab3526627',
  'local_synthetic_voice':True,'physical_accuracy_verified':False,'robot_execution_available':False,'fresh_model_run_demonstrated':False,
  'review_needed':'Creator should review narration and may replace local synthetic voice with a personal recording before submission.'}
 for source,suffix in [(movie,'.mp4'),(OUT/'captions.vtt','.vtt'),(OUT/'transcript.txt','.txt')]:
  dest=MEDIA/(PREFIX+suffix)
  if dest.exists():raise ValueError('Do not replace any existing media version')
  import shutil;shutil.copyfile(source,dest);manifest.setdefault('public_media',[]).append(ref(dest,'same_bytes_reviewable_media_copy'))
 write_new(OUT/'delivery.json',manifest);print(json.dumps({'status':manifest['status'],'movie':manifest['movie'],'full_decode':True,'official45s_unchanged':manifest['official45s_trailer_unchanged']}))


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','speech','render','verify']);a=p.parse_args();globals()[a.action]()
