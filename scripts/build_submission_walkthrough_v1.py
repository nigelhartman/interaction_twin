#!/usr/bin/env python3
"""Separate, local narrated walkthrough draft of genuine saved app results.

macOS bundled speech + ffmpeg; no model calls, downloads or publication.
Never writes the 54-second trailer or its inputs. Editorial layouts are labelled.
"""
from __future__ import annotations
import argparse, hashlib, json, math, shutil, subprocess, sys, time
from pathlib import Path
from functools import lru_cache
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import build_overnight_trailer_v2 as style
from build_showcase_media import VideoClip
OUT=ROOT/'output/submission-walkthrough-20261101-v1'
MEDIA=ROOT/'showcase/media'; PREFIX='submission-walkthrough-20261101-v3';RENDER=OUT/'render-v3'
W,H,FPS=1920,1080,24; TARGET_S=172; VOICE='Samantha'; RATE=160
BG=style.BG; INK=style.INK; MUTED=style.MUTED; TEAL=style.TEAL; BLUE=style.BLUE; AMBER=style.AMBER
CHAPTERS=[
 dict(title='A video contains physical know-how.',subtitle='A narrated walkthrough of saved local results',weight=18,
  narration='A product demonstration contains more than a shape. It contains parts, changing states, and an order of actions. Interaction Twin makes that context inspectable and reusable for technical guides and simulation authoring. This walkthrough shows saved local results, with their evidence and their limits.'),
 dict(title='Start with an ordinary recording.',subtitle='Actual earbuds recording · source footage, no synthetic frames',weight=19,
  narration='Here is the actual earbuds recording. The case opens, the earbuds leave the holder, and they return. In the workspace, the source sits beside a separate generated component. The app preserves the recording, sample timestamps, masks, and failures so each later claim can be traced back to pixels.'),
 dict(title='Separate the parts. Describe the task.',subtitle='Official Meta SAM 3.1 + Muse Spark · fallible model proposals',weight=29,
  narration='Meta SAM three point one supplies the visible component masks. Muse Spark proposes component descriptions, relationships, states, and source linked steps. The saved procedure opens the case, removes the earbuds, returns them, and closes the lid. Tripo S R supplies three independent rough component drafts. These are model generated shapes with unknown scale. The lid relationship and left right identity remain proposals that a person can inspect and correct.'),
 dict(title='Keep observed and inferred shape separate.',subtitle='Camera example · original source RGB and an optional inferred layer',weight=27,
  narration='The camera example tests a second reconstruction route. SAM masks isolate the target; feature correspondences and multi view geometry recover a partial visible surface. Original source pixels provide its appearance. The amber optional layer uses an inferred depth prior. It stays separate from the observed patch. Missing backs, cavities, calibration, and complete shape remain unresolved. The inspection camera orbit shown here is editorial.'),
 dict(title='Follow short motion. Keep the gaps.',subtitle='Exact source samples · image consistency, not measured physical accuracy',weight=34,
  narration='Motion Studio checks short intervals against exact source frames and unchanged native models. Green samples pass conditional image checks. Unverified references are amber; unsupported poses stay absent. Each interval has its own gauge. The base and lid gain some conditional support, while the preserved baseline is stronger for the earbud. A separate twelve frame per second lid experiment adds visible samples, but ties its three frame per second control at six of six common timestamps. More frames have not demonstrated more physical accuracy.'),
 dict(title='Package the context for an agent.',subtitle='Actual read-only MCP SDK client · 36 saved calls · no robot execution',weight=29,
  narration='The same fresh Jabra project is available through read only MCP tools. A real scripted SDK client made thirty six calls, retrieved three native models and six cited images, and packaged the four step procedure with a separate motion report. The exact source and mask can be reopened in the app. This handoff uses the original three frame per second base experiment; the denser lid study remains separate. The client was scripted, and no physical simulation ran.'),
 dict(title='A useful draft. A clear next milestone.',subtitle='Calibrated capture → stronger geometry → verified assembly → simulation',weight=16,
  narration='Today the result is an inspectable authoring draft. The next milestone is calibrated, overlapping capture with less occlusion, then verified shape, common scale, joints, and contact. That is the route toward reusable robot simulation assets: preserve what the video supports, and make every missing assumption visible.')]

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def local(v):return style.local(v)
def record(p,kind,**k):return dict(path=rel(p),sha256=sha(p),kind=kind,**k)
def fresh(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def run(args,log=None):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if log:Path(log).write_bytes(p.stdout+p.stderr)
 if p.returncode:raise RuntimeError('Local command failed; preserved log: '+str(log))
 return p.stdout
def probe(p):return json.loads(run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(p)]))
def check_records(v):
 n=0
 if isinstance(v,dict):
  if isinstance(v.get('path'),str) and isinstance(v.get('sha256'),str):
   if sha(local(v['path']))!=v['sha256']:raise ValueError('Input identity changed: '+v['path'])
   n+=1
  for x in v.values():n+=check_records(x)
 elif isinstance(v,list):
  for x in v:n+=check_records(x)
 return n
def prepare():
 OUT.mkdir(exist_ok=True,parents=True);MEDIA.mkdir(exist_ok=True)
 if (OUT/'preparation.json').exists():raise ValueError('Prepared version is immutable')
 prior=ROOT/'output/overnight-session-20261031/overnight-trailer-20261031-v2-source-manifest.json'
 inherited=read(prior);checked=check_records(inherited)
 preview=ROOT/'data/mcp-fresh-jabra-motion-handoff-20261101-v1-preview.json'
 if sha(preview)!='d0d05271576d1af649e7390502f13f4513de7ccbb03ace078f98ca4d6bd027bd':raise ValueError('Fresh package changed')
 captures=[]
 urls={'project-quality.png':'quality','project-procedure.png':'procedure','project-replay.png':'replay'}
 for name,view in urls.items():
  p=OUT/'captures'/name
  captures.append(record(p,'actual_CUA_app_screenshot',url='http://127.0.0.1:8765/workspace/index.html?case=flow-rigid-parts-jabra-20261031-v1--flow-procedure-1&view='+view,view=view,capture_mtime_utc_s=p.stat().st_mtime))
 for p in ['output/submission-review-20261101/fresh-36-call-desktop-v2.png','output/submission-review-20261101/density-lid-overview-desktop.png','output/submission-review-20261101/density-lid-source-native-desktop.png','output/motion-lab-20261031/visual-review/final-source-native-desktop.png']:
  captures.append(record(local(p),'existing_actual_CUA_capture',proof=record(local(p).with_suffix('.json'),'capture_proof') if local(p).with_suffix('.json').exists() else None))
 frozen=OUT/'builder-at-prepare.py';shutil.copyfile(__file__,frozen)
 prep={'format':'narrated-submission-walkthrough-preparation','version':1,'status':'prepared','chapters':CHAPTERS,'target_duration_s':TARGET_S,'voice':VOICE,'speech_rate_words_per_minute':RATE,'tts':'Bundled macOS say; no remote voice service','inherited_v2_manifest':record(prior,'unchanged54secondfilm_full_source_provenance'),'inherited_source_contracts_checked':checked,'captures':captures,'program':inherited['preparation_content']['program'],'source_video':inherited['preparation_content']['source_video'],'fresh_handoff':record(preview,'actual_scripted36call_same_program_handoff'),'style_helper':record(ROOT/'scripts/build_overnight_trailer_v2.py','existing_style_helpers_only'),'video_decode_helper':record(ROOT/'scripts/build_showcase_media.py','existing_recorded_video_decoder'),'builder':record(frozen,'exact_builder_at_prepare'),'policy':{'new_api_calls':0,'new_model_inference':0,'tts_local':True,'published':False,'private_user_recordings_local_only':True,'live_upload_demonstrated':False,'robot_or_simulator_executed':False,'LLM_planner_in_trace':False,'physical_accuracy_verified':False,'native_geometry_modified':False,'pose_interpolation':False}}
 fresh(OUT/'preparation.json',prep)
 script='Interaction Twin — narrated submission walkthrough draft\n\n'
 for j,c in enumerate(CHAPTERS):script+=f'{j+1}. {c["title"]}\n{c["narration"]}\n\n'
 (OUT/'narration-script.txt').write_text(script)
 print(json.dumps({'status':'prepared','words':sum(len(c['narration'].split()) for c in CHAPTERS),'input_contracts':checked}),flush=True)
def speech(take='speech-local-v2',rate=RATE):
 if not take.replace('-','').isalnum():raise ValueError('Simple versioned speech-take name required')
 prep=read(OUT/'preparation.json');check_records(prep);audio=OUT/take;audio.mkdir()
 rows=[]
 for j,c in enumerate(prep['chapters']):
  text=audio/f'{j+1:02d}.txt';aiff=audio/f'{j+1:02d}.aiff';wav=audio/f'{j+1:02d}.wav';text.write_text(c['narration']+'\n')
  run(['say','-v',VOICE,'-r',str(rate),'-f',str(text),'-o',str(aiff)],audio/f'{j+1:02d}-say.log')
  run(['ffmpeg','-hide_banner','-v','error','-i',str(aiff),'-ar','48000','-ac','2','-c:a','pcm_s16le',str(wav)],audio/f'{j+1:02d}-convert.log')
  seconds=float(probe(wav)['format'].get('duration',0))
  if seconds<=.1:raise ValueError('System voice returned empty audio; preserve this attempt and use authorized local-system execution')
  rows.append(dict(chapter=j+1,audio=record(wav,'actual_local_TTS_narration'),original=record(aiff,'original_macOS_say_output'),text=record(text,'exact_spoken_text'),duration_s=seconds))
  print(json.dumps({'speech_chapter':j+1,'seconds':seconds}),flush=True)
 minimum=[math.ceil((r['duration_s']+1.5)*FPS) for r in rows];remaining=TARGET_S*FPS-sum(minimum)
 if remaining<0:raise ValueError('Narration exceeds fixed172s budget; preserve this draft and review speech rate')
 weights=np.array([c['weight'] for c in CHAPTERS],float);extra=np.floor(remaining*weights/weights.sum()).astype(int);extra[0]+=remaining-int(extra.sum())
 for j,r in enumerate(rows):r['frames']=int(minimum[j]+extra[j]);r['chapter_duration_s']=r['frames']/FPS;r['narration_lead_s']=.75
 fresh(OUT/'speech.json',{'voice':VOICE,'rate':rate,'planned_rate':prep['speech_rate_words_per_minute'],'take':take,'rows':rows,'total_duration_s':sum(r['chapter_duration_s'] for r in rows),'speech_seconds':sum(r['duration_s'] for r in rows)})
def retime_speech():
 """Preserve actual local voice PCM, derive an explicitly tempo-adjusted narration."""
 prep=read(OUT/'preparation.json');check_records(prep);source=OUT/'speech-local-v2';dest=OUT/'speech-tempo-v4';dest.mkdir();rows=[];tempo=1.08
 for j,c in enumerate(prep['chapters']):
  original=source/f'{j+1:02d}.wav';text_path=source/f'{j+1:02d}.txt';wav=dest/f'{j+1:02d}.wav'
  if text_path.read_text().strip()!=c['narration']:raise ValueError('Voice source differs from planned transcript')
  run(['ffmpeg','-hide_banner','-v','error','-i',str(original),'-af',f'atempo={tempo}','-ar','48000','-ac','2','-c:a','pcm_s16le',str(wav)],dest/f'{j+1:02d}-retime.log')
  seconds=float(probe(wav)['format']['duration']);rows.append(dict(chapter=j+1,audio=record(wav,'local_voice_pitch_preserving_tempo_derivative'),original=record(original,'unchanged_actual_macOS_voice_PCM'),original_aiff=record(source/f'{j+1:02d}.aiff','unaltered_macOS_say_output'),text=record(text_path,'exact_spoken_text'),duration_s=seconds,source_duration_s=float(probe(original)['format']['duration']),tempo=tempo))
 minimum=[math.ceil((r['duration_s']+1.5)*FPS) for r in rows];remaining=TARGET_S*FPS-sum(minimum)
 if remaining<0:raise ValueError('Voice derivative still exceeds172seconds')
 weights=np.array([c['weight'] for c in CHAPTERS],float);extra=np.floor(remaining*weights/weights.sum()).astype(int);extra[0]+=remaining-int(extra.sum())
 for j,r in enumerate(rows):r['frames']=int(minimum[j]+extra[j]);r['chapter_duration_s']=r['frames']/FPS;r['narration_lead_s']=.5;r['narration_tail_s']=r['chapter_duration_s']-.5-r['duration_s'];assert r['narration_tail_s']>=1
 fresh(OUT/'speech.json',{'voice':VOICE,'source_requested_rate':160,'pitch_preserving_tempo':tempo,'requested_rate_adjustment_note':'The160and174requested-rate takes produced byte-identical PCM; final narration uses the explicitly declared1.08tempo derivative of the160take.','rows':rows,'total_duration_s':sum(r['chapter_duration_s'] for r in rows),'speech_seconds':sum(r['duration_s'] for r in rows),'source_take':'speech-local-v2','derivative_take':'speech-tempo-v4'})
def assets():
 inherited=read(local(read(OUT/'preparation.json')['inherited_v2_manifest']['path']));p=inherited['preparation_content']
 a={'prior':inherited,'prep':p,'source':Image.open(local(p['source_sequence'][0]['source']['path'])).convert('RGB'),'masks':[Image.open(local(r['overlay']['path'])).convert('RGB') for r in p['source_sequence']],'camera':Image.open(local(p['camera_source']['path'])).convert('RGB'),'orbits':{g['kind']:[Image.open(local(r['path'])).convert('RGB') for r in g['frames']] for g in p['camera_orbits']},'motion':inherited['motion_sequence']}
 a['motion_source']=[Image.open(local(r['source_crop']['path'])).convert('RGB') for r in a['motion']['frames']];a['motion_native']=[Image.open(local(r['native_crop']['path'])).convert('RGB') for r in a['motion']['frames']]
 for name in ['quality','procedure','replay']:a[name]=Image.open(OUT/'captures'/f'project-{name}.png').convert('RGB')
 for name,pth in [('agents','fresh-36-call-desktop-v2.png'),('density','density-lid-overview-desktop.png'),('density_pose','density-lid-source-native-desktop.png')]:a[name]=Image.open(ROOT/'output/submission-review-20261101'/pth).convert('RGB')
 a['gap']=Image.open(OUT/'captures/withheld-earbud606.png').convert('RGB')
 a['agents']=a['agents'].crop((40,12,1475,522)) # Declared crop only, preserving every screenshot pixel.
 return a
def text(im,xy,s,size=30,color=INK,bold=False):style.txt(im,xy,s,size,color,bold)
def panel(im,p,box):style.panel(im,p,box,bg=BG)
def footer(im,textline):text(im,(72,1019),textline,24,MUTED)
def base(j):
 im=Image.new('RGB',(W,H),BG);text(im,(72,42),'INTERACTION TWIN',25,TEAL,True);text(im,(1615,45),'NARRATED DRAFT',19,MUTED,True)
 text(im,(72,111),f'{j+1:02d} / {CHAPTERS[j]["title"]}',55,INK,True);text(im,(74,194),CHAPTERS[j]['subtitle'],27,MUTED)
 ImageDraw.Draw(im).line((72,995,1848,995),fill='#354855',width=2);return im
def motion_index(u,seconds,rows):
 elapsed=max(0.,(u-.36)*seconds*.5)
 if elapsed>=rows[-1]['timestamp_s']-rows[0]['timestamp_s']:return len(rows)-1
 t=rows[0]['timestamp_s']+elapsed
 return max(z for z,row in enumerate(rows) if row['timestamp_s']<=t)
def frame(j,u,a,live=None):
 im=base(j)
 if j==0:
  text(im,(74,325),'A shape is only\none part of\nthe task.',84,INK,True);text(im,(80,664),'Parts · states · action order · evidence',34,MUTED)
  style.tag(im,(80,825),'TECHNICAL GUIDES + SIMULATION AUTHORING');panel(im,a['source'],(1275,275,470,650))
  footer(im,'Saved local results · local synthetic voice · no live upload or robot execution')
 elif j==1:
  if u<.75:
   panel(im,live if live is not None else a['source'],(90,270,470,670));text(im,(690,320),'One ordinary\nobject interaction.',60,INK,True)
   text(im,(698,535),'Open → remove → return → close',34,TEAL,True);text(im,(698,647),'Keep the original timing, source images,\ncomponent masks and saved failures.',31,MUTED)
  else:panel(im,a['replay'],(70,267,1780,700))
  footer(im,'Actual recorded source at1× → saved app capture · independent asset inspection has no assembly pose')
 elif j==2:
  if u<.48:
   n=min(len(a['masks'])-1,int(u/.48*len(a['masks'])));panel(im,a['masks'][n],(90,275,390,660))
   for z,(label,detail) in enumerate([('SAM 3.1','Visible component masks'),('Muse Spark','Components, states and cited steps'),('TripoSR','Independent rough 3D drafts')]):
    text(im,(630,320+z*180),label,46,TEAL,True);text(im,(635,385+z*180),detail,31,MUTED)
  else:panel(im,a['procedure' if u<.83 else 'quality'],(70,267,1780,700))
  footer(im,'Actual saved Meta proposals and local generated assets · physical identity and common scale unverified')
 elif j==3:
  n=min(23,int(u*24));panel(im,a['camera'],(72,315,850,510));inferred=u>.5;panel(im,a['orbits']['inferred' if inferred else 'observed'][n],(996,315,850,510))
  style.tag(im,(90,862),'ORIGINAL SOURCE RGB',BLUE);style.tag(im,(1014,862),'+ INFERRED PRIOR LAYER' if inferred else 'PARTIAL OBSERVED SURFACE',AMBER if inferred else BLUE)
  footer(im,'Separate camera experiment · authored inspection-camera orbit · missing backs/cavities remain unknown')
 elif j==4:
  if u<.18:panel(im,a['density'],(70,267,1780,700))
  elif u<.36:
   panel(im,a['gap'],(70,267,1780,700));style.tag(im,(90,932),'REAL WITHHELD EARBUD SAMPLE · POSE HIDDEN',AMBER)
  elif u<.75:
   # Two seconds of actual saved source-time replayed at0.5×, then its final sample held. No interpolation.
   rows=a['motion']['frames'];n=motion_index(u,a['chapter_seconds'],rows);r=rows[n]
   panel(im,a['motion_source'][n],(74,310,850,535));panel(im,a['motion_native'][n],(995,310,850,535))
   style.tag(im,(90,875),f'EXACT SOURCE {r["timestamp_s"]:.3f} s');style.tag(im,(1010,875),'UNVERIFIED REFERENCE' if r['status']=='reference_unverified' else 'CONDITIONAL IMAGE SUPPORT',AMBER if r['status']=='reference_unverified' else BLUE)
  else:
   panel(im,a['density_pose'],(70,267,1780,650));text(im,(88,935),'Common timestamps: 6/6 vs6/6 · extra samples do not prove greater accuracy',28,TEAL,True)
  footer(im,'12 fps earbud example · exact source20.168433 s · saved scope gate · no pose displayed' if .18<=u<.36 else '12 fps lid study · half-speed sample hold · physical pose unmeasured · no interpolation or cross-gap bridge')
 elif j==5:
  panel(im,a['agents'],(72,300,1776,632));footer(im,'Actual 36-call scripted client · original 3 fps BASE report · separate 12 fps LID study · simulation not executed')
 else:
  labels=[('1','Calibrated, overlapping capture',TEAL),('2','Verified shape + common scale',BLUE),('3','Joints, contact + simulator validation',AMBER)]
  for z,(n,title,color) in enumerate(labels):
   style.tag(im,(84,325+z*175),n,color);text(im,(174,326+z*175),title,45,INK,True)
  text(im,(84,886),'Preserve the evidence. Make missing assumptions visible.',37,TEAL,True)
  footer(im,'Authoring draft · complete twins and physical robot simulation remain future work')
 return im
def storyboard(name='draft-01'):
 a=assets();a['chapter_seconds']=34;montage=Image.new('RGB',(1440,1620),BG);rows=[]
 for j in range(7):
  im=frame(j,.5,a);p=OUT/f'{name}-chapter-{j+1:02d}.jpg';im.save(p,quality=96);montage.paste(im.resize((720,405)),((j%2)*720,(j//2)*405));rows.append(record(p,'editorial_walkthrough_layout_actual_inputs'))
 path=OUT/f'{name}-storyboard.jpg';montage.save(path,quality=95);fresh(OUT/f'{name}-storyboard.json',{'chapters':rows,'montage':record(path,'storyboard')});print(rel(path),flush=True)
def audio_timeline(speech):
 rows=speech['rows'];args=['ffmpeg','-hide_banner','-v','error']
 for r in rows:args+=['-i',str(local(r['audio']['path']))]
 args+=['-stream_loop','-1','-i',str(local(read(OUT/'preparation.json')['inherited_v2_manifest']['path']).parent/'overnight-trailer-20261031-v2-original-score.wav')]
 filters=[]
 for j,r in enumerate(rows):
  delay=round(r.get('narration_lead_s',.75)*1000);filters.append(f'[{j}:a]adelay={delay}|{delay},apad,atrim=duration={r["chapter_duration_s"]:.9f},asetpts=PTS-STARTPTS[a{j}]')
 filters.append(''.join(f'[a{j}]' for j in range(len(rows)))+f'concat=n={len(rows)}:v=0:a=1[voice]')
 filters.append(f'[{len(rows)}:a]atrim=duration={TARGET_S},volume=0.045,afade=t=out:st={TARGET_S-3}:d=3[music]')
 filters.append('[voice][music]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.85:level=false[mixed]')
 destination=RENDER/'narration-mix.wav';run(args+['-filter_complex',';'.join(filters),'-map','[mixed]','-ar','48000','-ac','2','-c:a','pcm_s16le',str(destination)],RENDER/'audio-mix.log');return destination
def encode():
 RENDER.mkdir(exist_ok=False,parents=True)
 prep=read(OUT/'preparation.json');check_records(prep);speech=read(OUT/'speech.json');check_records(speech);a=assets();check_records(a['prior']);check_records(read(OUT/'capture-supplement.json'));audio=audio_timeline(speech)
 snapshot=RENDER/'builder-at-encode.py';shutil.copyfile(__file__,snapshot);encoder_sha=sha(snapshot)
 dest=MEDIA/(PREFIX+'.mp4')
 if dest.exists():raise ValueError('Encoded walkthrough version already exists')
 command=['ffmpeg','-hide_banner','-v','warning','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-i',str(audio),'-map','0:v:0','-map','1:a:0','-c:v','libx264','-preset','fast','-crf','19','-threads','4','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart','-t',str(TARGET_S),str(dest)]
 log=(RENDER/'encode.log').open('x');proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log);chapters=[];count=0;started=time.monotonic();live=None
 try:
  for j,r in enumerate(speech['rows']):
   a['chapter_seconds']=r['chapter_duration_s'];cache={};clip_frames=14*FPS
   if j==1:live=VideoClip(local(prep['source_video']['path']),12,14,size=(540,960))
   for n in range(r['frames']):
    u=n/max(1,r['frames']-1);image=None
    if j==1 and u<.75:image=frame(j,u,a,live.frame() if n<clip_frames else live.last)
    else:
     if j==2:key=('mask',min(len(a['masks'])-1,int(u/.48*len(a['masks'])))) if u<.48 else ('procedure' if u<.83 else 'quality')
     elif j==3:key=(min(23,int(u*24)),u>.5)
     elif j==4:
      if u<.18:key='overview'
      elif u<.36:key='actual_withheld_gap_UI'
      elif u>=.75:key='source_native_UI'
      else:
       key=motion_index(u,a['chapter_seconds'],a['motion']['frames'])
     else:key=int(u>=.75) if j==1 else 0
     if key not in cache:cache[key]=frame(j,u,a).tobytes()
     image=cache[key]
    proc.stdin.write(image.tobytes() if isinstance(image,Image.Image) else image);count+=1
   if live:live.close();live=None
   chapters.append(dict(chapter=j+1,title=CHAPTERS[j]['title'],start_s=sum(x['duration_s'] for x in chapters),duration_s=r['chapter_duration_s'],frames=r['frames'],narration=CHAPTERS[j]['narration']))
   print(json.dumps({'encoded_chapter':j+1,'frames':count}),flush=True)
  proc.stdin.close()
  if proc.wait(timeout=180):raise RuntimeError('Walkthrough encoder failed')
 finally:
  if live:live.close()
  if proc.poll() is None:proc.terminate();proc.wait(timeout=20)
  log.close()
 if sha(__file__)!=encoder_sha:raise ValueError('Encoder source changed during execution; preserve output as unaccepted attempt')
 fresh(RENDER/'source-manifest.json',{'format':'narrated-submission-walkthrough','version':2,'status':'encoded_local_draft','output':record(dest,'separate_narrated_H264_AAC_walkthrough'),'width':W,'height':H,'fps':FPS,'duration_s':TARGET_S,'frames':count,'render_seconds':time.monotonic()-started,'chapters':chapters,'preparation':record(OUT/'preparation.json','frozen_input_preparation'),'preparation_content':prep,'inherited_actual_sources':a['prior'],'capture_supplement':record(OUT/'capture-supplement.json','actual_later_withheld_sample_capture_proof'),'capture_supplement_content':read(OUT/'capture-supplement.json'),'speech':record(OUT/'speech.json','local_TTS_receipts_and_timing'),'speech_content':speech,'narration_mix':record(audio,'actual_local_narration_plus_low_background_original_score'),'original_music':record(ROOT/'output/overnight-session-20261031/overnight-trailer-20261031-v2-original-score.wav','unchanged_original_procedural_music_reused_at0.045gain'),'builder':record(snapshot,'exact_encoder_source'),'source_playback':'Chapter2 decodes source12..26s at1x/24fps, brief final hold if needed, then switches to actual saved app capture at75percent ofchapter; frame conversion drops/duplicates only. SavedSAM overlays use exact sampled source frames.','motion_playback':'Existing25 native12fpslid source/mask and saved pose pairs at0.5x then hold; no invented/interpolated poses. Actual withheld606earbud capture is a separate explicit example. Original3fpsBASE MCP resource shown separately.','editorial_camera_orbit':True,'actual_screenshots_not_live_screencast':True,'new_api_calls':0,'model_inference_calls':0,'published_or_submitted':False,'physical_accuracy_verified':False})
def verify():
 manifest=read(RENDER/'source-manifest.json');contracts=check_records(manifest);video=local(manifest['output']['path']);p=probe(video)
 vs=next(s for s in p['streams'] if s['codec_type']=='video');aus=next(s for s in p['streams'] if s['codec_type']=='audio')
 assert(vs['width'],vs['height'],vs['avg_frame_rate'],vs['codec_name'])==(W,H,'24/1','h264');assert aus['codec_name']=='aac';assert 160<=float(p['format']['duration'])<=175
 run(['ffmpeg','-hide_banner','-v','error','-i',str(video),'-f','null','-'],RENDER/'full-decode.log')
 raw=run(['ffmpeg','-hide_banner','-v','error','-i',str(video),'-vn','-f','f32le','-acodec','pcm_f32le','-']);sound=np.frombuffer(raw,dtype='<f4').reshape(-1,2);peak=float(np.abs(sound).max());rms=float(np.sqrt(np.mean(sound*sound)));assert 0.01<rms<.4 and 0<peak<1
 cap=cv2.VideoCapture(str(video));count=0;bad=[];chapterframes=[]
 while True:
  ok,rgb=cap.read()
  if not ok:break
  if rgb.shape!=(H,W,3) or float(rgb.std())<3:bad.append(count)
  count+=1
 cap.release();assert count==TARGET_S*FPS and not bad
 montage=Image.new('RGB',(1440,1620),BG);cap=cv2.VideoCapture(str(video))
 for j,c in enumerate(manifest['chapters']):
  t=c['start_s']+c['duration_s']*.5;cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,bgr=cap.read();assert ok;im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));path=RENDER/f'decoded-chapter-{j+1:02d}.jpg';im.save(path,quality=96);montage.paste(im.resize((720,405)),((j%2)*720,(j//2)*405));chapterframes.append(record(path,'actual_encoded_decoded_chapter',timestamp_s=t))
  if j==0:shutil.copyfile(path,MEDIA/(PREFIX+'-poster.jpg'))
 cap.release();story=RENDER/'verified-storyboard.jpg';montage.save(story,quality=95)
 rows=manifest['inherited_actual_sources']['motion_sequence']['frames'];motion_chapter=manifest['chapters'][4];mapping=[]
 for n in range(motion_chapter['frames']):
  u=n/max(1,motion_chapter['frames']-1)
  if .36<=u<.75:
   k=motion_index(u,motion_chapter['duration_s'],rows);mapping.append({'output_frame':round(motion_chapter['start_s']*FPS)+n,'source_sample_index':rows[k]['sample_index'],'source_pts_s':rows[k]['timestamp_s']})
 assert len({r['source_sample_index'] for r in mapping})==25
 fresh(RENDER/'verification.json',{'format':'submission-walkthrough-verification','status':'passed','contracts_checked':contracts,'duration_s':float(p['format']['duration']),'decoded_frames':count,'expected_frames':TARGET_S*FPS,'uniform_or_invalid_frames':bad,'full_ffmpeg_video_audio_decode':True,'audio':{'peak_dbfs':20*math.log10(peak),'rms_dbfs':20*math.log10(rms),'clipped_samples':int((np.abs(sound)>=1).sum()),'channels':2,'sample_rate_hz':48000},'decoded_chapters':chapterframes,'storyboard':record(story,'actual_encoded_chapter_montage'),'motion_source_map':mapping,'unique_native_motion_samples_shown':25,'new_api_calls':0,'new_model_inference':0,'published':False})
 print(json.dumps({'status':'passed','frames':count,'duration':TARGET_S,'contracts':contracts,'peak_dbfs':20*math.log10(peak)}),flush=True)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','speech','retime_speech','storyboard','encode','verify']);p.add_argument('--draft-name',default='draft-01');p.add_argument('--speech-take',default='speech-local-v2');p.add_argument('--speech-rate',type=int,default=RATE);a=p.parse_args()
 if a.action=='storyboard':storyboard(a.draft_name)
 elif a.action=='speech':speech(a.speech_take,a.speech_rate)
 else:globals()[a.action]()
if __name__=='__main__':main()
