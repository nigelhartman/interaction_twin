#!/usr/bin/env python3
"""45s motion-first edit of actual browser captures, never a replacement renderer.

Source poses/geometry are not modified here. Fixed editorial framing and
documented time remapping preserve the full robot task and released goals.
No API calls, publishing, arbitrary commands or external music.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT=Path(__file__).resolve().parents[1]
W,H,FPS,DURATION=1920,1080,30,45
BASE=ROOT/'output/dynamic-trailer-20261104-v1'
BG,INK,BLUE,MUTED='#f7f8fb','#151923','#3766ed','#68758a'
MASTER_MAP=[(0,4,0,4),(4,8,4,9),(8,16,9,18),(16,20,18,24),(20,25,24,31),(25,29,31,37)]
VOICE=[
    (0,4,'Film a real interaction.'),
    (4,8,'SAM follows the moving parts.'),
    (8,16,'SAM 3D reconstructs them. Muse reasons about their connections, and our pipeline fits the motion.'),
    (16,20,'Together: a 3D object and a program describing how it works.'),
    (20,25,'Every recording adds a searchable twin to your library.'),
    (25,29,'Your agent can find its parts, actions, and evidence.'),
    (29,39,'Build prototype robot tasks, from opening a case to removing and placing its earbuds.'),
    (39,43,'Test the parts with NVIDIA Newton physics.'),
    (43,45,'Interaction Twin.')]
ROBOT_REF={'path':'output/trailer-refinement-20261103/robot-complete-capture-v3.webm',
    'sha256':'32f8db744c0e2fe46004529c70bc518e57bfb589cf78375dc7555de028ebdd97','bytes':3212314}
ROBOT_JSON={'path':'output/trailer-refinement-20261103/robot-complete-capture-v3.json',
    'sha256':'69743e4a6edf48ddb91593e0483b37044ab2fd575be5fb967064017e387371b1','bytes':1001851}
LOGO_REF={'path':'showcase/brand/v2/mark-1024.png',
    'sha256':'881e92e291fe72ee49d58a25f70caf65ab1912e39262647bd6a999ceff8afb68','bytes':42331}
MUSIC=ROOT/'output/trailer-20261102/v4/original-instrumental-only.wav'
NEWTON=ROOT/'output/robotics-physics-20261104-v1/newton-contact-native-1080p-v1.mp4'
NEWTON_REPORT=ROOT/'output/robotics-physics-20261104-v1/replay-native-v3.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):
    p=Path(p).resolve();return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
def checked(r):
    relative=r.get('path')
    if not isinstance(relative,str)or not relative.startswith(('data/','output/','showcase/','workspace/'))or any(part.startswith('.')or not part for part in relative.split('/'))or '\\'in relative or ':'in relative:
        raise ValueError('Only explicit non-hidden workspace presentation artifacts are allowed')
    p=ROOT/relative
    if p.resolve()!=p or not p.is_file() or sha(p)!=r['sha256'] or p.stat().st_size!=r['bytes']:
        raise ValueError('Pinned source changed: '+r['path'])
    return p
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def run(args,*,timeout=300):return subprocess.run(args,check=True,capture_output=True,text=True,timeout=timeout)
def duration(p):
    return float(run(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1',str(p)]).stdout)
def master_time(t):
    if not math.isfinite(t)or not 0<=t<29:raise ValueError('Master editorial time outside29s')
    for a,b,sa,sb in MASTER_MAP:
        if a<=t<b:return sa+(t-a)*(sb-sa)/(b-a)
    raise AssertionError()
def robot_time(t):
    if not math.isfinite(t)or not 0<=t<=10:raise ValueError('Robot editorial time outside10s')
    if t<=.6:return t/.6
    if t<=8.6:return 1+(t-.6)*23.9/8
    return 24.9+(t-8.6)*2/1.4
def stamp(t,sep='.'):
    ms=round(t*1000);h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);s,ms=divmod(ms,1000)
    return f'{h:02}:{m:02}:{s:02}{sep}{ms:03}'
@lru_cache(maxsize=24)
def font(size,bold=False):
    return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold'if bold else'')+'.ttf',size)
@lru_cache(maxsize=16)
def frame(p):return Image.open(p).convert('RGB')
def decode(p,folder):
    folder.mkdir(exist_ok=False)
    run(['ffmpeg','-nostdin','-v','error','-i',str(p),'-vf',f'fps={FPS}','-q:v','2',str(folder/'%05d.jpg')])
    paths=sorted(folder.glob('*.jpg'))
    if not paths:raise ValueError('Capture decoded no frames')
    with Image.open(paths[0])as im:
        if im.size!=(W,H):raise ValueError('Capture is not native1920x1080')
    return paths
def source_frame(paths,t):
    return frame(str(paths[max(0,min(len(paths)-1,round(t*FPS)))]))
def voice_one(row,folder):
    i,start,end,text=row
    aiff=folder/f'{i:02}-samantha.aiff';wav=folder/f'{i:02}-voice.wav'
    if not aiff.exists():run(['/usr/bin/say','-v','Samantha','-r','175','-o',str(aiff),text],timeout=90)
    raw=duration(aiff);available=end-start-.18;tempo=max(1.,raw/available)
    if tempo>1.35:raise ValueError('Narration is too rushed for its scene: '+text)
    run(['ffmpeg','-nostdin','-v','error','-i',str(aiff),'-af',f'atempo={tempo:.9f},loudnorm=I=-17:TP=-2:LRA=7,afade=t=in:st=0:d=0.015','-ar','48000','-ac','2',str(wav)])
    actual=duration(wav)
    if actual>available+.03:raise ValueError('Voice extends into the next shot')
    return {'start_s':start+.04,'end_s':start+.04+actual,'window_end_s':end,'text':text,
        'voice':'Apple Samantha / local synthesis','rate_words_minute':175,'atempo':tempo,
        'source':ref(aiff),'processed':ref(wav)}
def prepare_voice(out,source_folder=None):
    folder=out/'voice';folder.mkdir(exist_ok=True)
    if source_folder:
        for i in range(len(VOICE)):
            source=Path(source_folder)/f'{i:02}-samantha.aiff';target=folder/source.name
            if not source.is_file()or duration(source)<=.1:raise ValueError('Complete local voice source required')
            if source.resolve()!=target.resolve():shutil.copyfile(source,target)
    rows=[(i,a,b,t)for i,(a,b,t)in enumerate(VOICE)]
    with ThreadPoolExecutor(max_workers=3)as pool:
        segments=list(pool.map(lambda r:voice_one(r,folder),rows))
    args=['ffmpeg','-nostdin','-v','error']
    for s in segments:args+=['-i',str(ROOT/s['processed']['path'])]
    filters=[];labels=[]
    for i,s in enumerate(segments):
        delay=round(s['start_s']*1000);filters.append(f'[{i}:a]adelay={delay}|{delay},apad=whole_dur=45[a{i}]');labels.append(f'[a{i}]')
    filters.append(''.join(labels)+f'amix=inputs={len(segments)}:normalize=0,atrim=0:45,alimiter=limit=0.89[voice]')
    target=out/'voice-45.wav'
    run(args+['-filter_complex',';'.join(filters),'-map','[voice]','-ar','48000','-ac','2',str(target)])
    write(out/'narration.json',{'segments':segments,'duration_s':45,'hosted_speech':False})
    return segments,target
def captions(out,segments):
    cues=[]
    for s in segments:
        words=s['text'].split();count=max(1,math.ceil(len(s['text'])/70));bounds=[0]
        bad={'a','an','the','and','or','to','from','for','with','about','their','our','its','of'}
        for i in range(1,count):
            index=(i*len(words))//count
            while index<len(words)-4 and words[index-1].lower().strip(',.:;')in bad:index+=1
            bounds.append(index)
        bounds.append(len(words));groups=[' '.join(words[a:b])for a,b in zip(bounds,bounds[1:])]
        counts=sum(len(g.split())for g in groups);clock=s['start_s']
        for group in groups:
            end=clock+(s['end_s']-s['start_s'])*len(group.split())/counts
            cues.append((clock,end,group));clock=end
    vtt='WEBVTT\n\n';srt=''
    for i,(a,b,text)in enumerate(cues,1):
        vtt+=f'{i}\n{stamp(a)} --> {stamp(b)}\n{text}\n\n'
        srt+=f'{i}\n{stamp(a,",")} --> {stamp(b,",")}\n{text}\n\n'
    (out/'captions.vtt').write_text(vtt);(out/'captions.srt').write_text(srt)
    (out/'transcript.txt').write_text('\n\n'.join(s['text']for s in segments)+'\n')
    return cues
def robot_crop(receipt):
    c=receipt['camera_fit'];mn,mx=c['ndc_min'],c['ndc_max'];pad=38
    box=[max(0,math.floor((1+mn[0])*W/2)-pad),max(0,math.floor((1-mx[1])*H/2)-pad),
         min(W,math.ceil((1+mx[0])*W/2)+pad),min(H,math.ceil((1-mn[1])*H/2)+pad)]
    if not c['all_points_inside']or c['edge_margin_fraction']<.02:raise ValueError('Whole robot view not verified')
    for goal in receipt['final_goal_canvas']:
        x,y=goal['canvas'][:2]
        if not box[0]+8<x<box[2]-8 or not box[1]+8<y<box[3]-8:raise ValueError('Editorial crop hides a robot goal')
    return tuple(box)
def robot_stage(receipt,elapsed):
    row=min(receipt['trace'],key=lambda r:abs(r['time_s']-robot_time(elapsed)))
    action=row['action_id']
    if action=='film_initial_hold':return 0
    if action in {'film_withdraw_arm','film_final_hold'}:return 4
    number=int(action.removeprefix('action_'))
    if number<=11:return 0
    if number<=13:return 1
    if 14<=number<=19 or 25<=number<=30:return 2
    if 20<=number<=24 or 31<=number<=35:return 3
    return 4
def robot_frame(picture,elapsed,crop,logo,active=0):
    im=Image.new('RGB',(W,H),picture.getpixel((0,0)));layer=ImageOps.contain(picture.crop(crop),(1070,950),Image.Resampling.LANCZOS)
    im.paste(layer,(90+(1070-layer.width)//2,90+(950-layer.height)//2))
    draw=ImageDraw.Draw(im);draw.text((1190,210),'ROBOTICS LAB',font=font(20,True),fill=BLUE)
    draw.text((1185,255),'Build a next move.',font=font(49,True),fill=INK)
    labels=['Place the case.','Open the lid.','Remove each earbud.','Place on the tray.','Close the case.']
    for i,text in enumerate(labels):
        draw.text((1190,365+i*65),text,font=font(32,i==active),fill=BLUE if i==active else '#8792a4')
        if i==active:draw.rounded_rectangle((1158,374+i*65,1166,401+i*65),radius=4,fill=BLUE)
    im.paste(logo.resize((43,43),Image.Resampling.LANCZOS),(1190,800),logo.resize((43,43),Image.Resampling.LANCZOS))
    draw.text((1247,808),'Interaction Twin',font=font(25,True),fill=INK)
    return im
def newton_frame(picture,elapsed,logo):
    im=picture.copy();d=ImageDraw.Draw(im)
    d.text((92,80),'GENERATED PARTS → PHYSICS',font=font(21,True),fill=BLUE)
    d.text((86,130),'Real gravity. Real contact.',font=font(59,True),fill=INK)
    return im
def end_frame(t,logo):
    im=Image.new('RGB',(W,H),BG);draw=ImageDraw.Draw(im)
    progress=min(1.,max(0.,t/.45));ease=1-(1-progress)**3
    size=round(144+8*ease);art=logo.resize((size,size),Image.Resampling.LANCZOS)
    if progress<1:art.putalpha(art.getchannel('A').point(lambda a:round(a*progress)))
    x=round(310-20*(1-ease));im.paste(art,(x,405),art)
    draw.text((510,405),'Interaction Twin',font=font(88,True),fill=INK)
    draw.text((515,525),'Show it once. Keep how it works.',font=font(36),fill=MUTED)
    draw.rounded_rectangle((515,608,round(515+700*min(1,t/1.6)),613),radius=2,fill=BLUE)
    return im
def verify_motion(clip,receipt):
    d=read(receipt)
    if d.get('format')!='interaction-twin-dynamic-film-capture' or d.get('version')!=2:
        raise ValueError('Only the repaired version2 motion capture is accepted; no draft fallback')
    if d.get('canvas')!=[W,H]or d.get('fps')!=30 or d.get('duration_s')!=37 or d.get('bytes')!=clip.stat().st_size:
        raise ValueError('Motion capture clock/dimensions/bytes changed')
    if d.get('physical_accuracy_verified')is not False or d.get('new_api_calls')!=0 or d.get('camera_microphone_capture')is not False:
        raise ValueError('Unsupported capture claims')
    interval=d.get('fully_supported_source_interval_s');trace=d.get('projection_trace')
    if interval!=[17.35,20.3]or not isinstance(trace,list)or len(trace)<100:
        raise ValueError('The repaired capture needs its fully supported source interval and actual projection trace')
    last=-1
    for row in trace:
        if not isinstance(row,dict)or not isinstance(row.get('film_s'),(int,float))or not last<row['film_s']<=37 or not 17.35-1e-6<=row.get('source_s',-1)<=20.3+1e-6:
            raise ValueError('Projection trace clock escapes the verified source interval')
        last=row['film_s'];parts=row.get('parts',[])
        if {p.get('id')for p in parts}!={'case_base','case_lid','earbud_a','earbud_b'}or len(parts)!=4:
            raise ValueError('The filmed connected twin is missing a native component')
        for part in parts:
            matrix=part.get('camera_from_native_mesh')
            if not isinstance(matrix,list)or len(matrix)!=4 or not all(isinstance(r,list)and len(r)==4 and all(isinstance(v,(int,float))and math.isfinite(v)for v in r)for r in matrix):
                raise ValueError('A native component is withheld in the film; incomplete poses cannot be promoted')
    for r in d.get('inputs',[]):checked(r)
    return d
def build(motion,receipt,out):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    motion=motion.resolve();receipt=receipt.resolve();md=verify_motion(motion,receipt)
    robot=checked(ROBOT_REF);rj=read(checked(ROBOT_JSON));crop=robot_crop(rj)
    if rj.get('captured_full_task')is not True or rj.get('final_gripper_attached')is not None or not all(g['released']and g['reached']for g in rj['final_goals']):
        raise ValueError('Completed robot release/closed-lid task required')
    nr=read(NEWTON_REPORT)
    if nr.get('claims',{}).get('real_physics_engine_execution')is not True:raise ValueError('Actual Newton execution required')
    logo_path=ROOT/LOGO_REF['path']
    if sha(logo_path)!=LOGO_REF['sha256']:raise ValueError('Approved V2 logo changed')
    logo=Image.open(logo_path).convert('RGBA')
    segments,voice=prepare_voice(out,BASE/'voice-prepare-v3/voice');cues=captions(out,segments)
    master_frames=decode(motion,out/'master-frames');robot_frames=decode(robot,out/'robot-frames');physics_frames=decode(NEWTON,out/'physics-frames')
    if len(master_frames)<1106 or len(robot_frames)<805 or len(physics_frames)<280:raise ValueError('A source capture is incomplete')
    picture=out/'silent-45.mp4'
    encoder=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(picture)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    proofs=[];proof_times=[0,4.4,8.8,12,15.3,17.8,21.5,26,30,34.5,38.4,40.3,42,44.1]
    proof_indices={round(t*FPS)for t in proof_times}
    try:
        for n in range(FPS*DURATION):
            t=n/FPS
            if t<29:im=source_frame(master_frames,master_time(t)).copy()
            elif t<39:im=robot_frame(source_frame(robot_frames,robot_time(t-29)),t-29,crop,logo,robot_stage(rj,t-29))
            elif t<43:im=newton_frame(source_frame(physics_frames,.4+(t-39)*1.8/4),t-39,logo)
            else:im=end_frame(t-43,logo)
            if n in proof_indices:
                p=out/f'proof-{n:04}.jpg';im.save(p,quality=94);proofs.append({'time_s':t,'file':ref(p)})
            encoder.stdin.write(im.tobytes())
        encoder.stdin.close();stderr=encoder.stderr.read();code=encoder.wait()
        if code:raise ValueError('Video encoder failed: '+stderr.decode()[-2000:])
    except BaseException:
        encoder.kill();raise
    music_filter='loudnorm=I=-29:TP=-5:LRA=7,afade=t=in:st=0:d=0.8,afade=t=out:st=43:d=2'
    movie=out/'interaction-twin-dynamic-20261104-v1.mp4'
    filt=f'[1:a]{music_filter},asplit=2[music][music_meter];[2:a]asplit=2[voice][sidechain];[music][sidechain]sidechaincompress=threshold=.015:ratio=5:attack=15:release=250[duck];[duck][voice]amix=inputs=2:normalize=0,alimiter=limit=.89,atrim=0:45[a];[music_meter]anullsink'
    run(['ffmpeg','-nostdin','-v','error','-i',str(picture),'-i',str(MUSIC),'-i',str(voice),'-filter_complex',filt,'-map','0:v:0','-map','[a]','-t','45','-c:v','copy','-c:a','aac','-b:a','192k','-ar','48000','-ac','2','-movflags','+faststart',str(movie)])
    probe=json.loads(run(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(movie)]).stdout)
    video=next(s for s in probe['streams']if s['codec_type']=='video');audio=next(s for s in probe['streams']if s['codec_type']=='audio')
    if video['width']!=W or video['height']!=H or int(video['nb_read_frames'])!=1350 or abs(float(probe['format']['duration'])-45)>.001 or int(audio['sample_rate'])!=48000 or audio['channels']!=2:
        raise ValueError('Final video duration/frame/audio check failed')
    run(['ffmpeg','-nostdin','-v','error','-ss','12','-i',str(movie),'-frames:v','1','-q:v','2',str(out/'poster.jpg')])
    sheet=Image.new('RGB',(1440,math.ceil(len(proofs)/4)*226),BG);d=ImageDraw.Draw(sheet)
    for i,p in enumerate(proofs):
        im=Image.open(ROOT/p['file']['path']);im.thumbnail((360,202))
        x=(i%4)*360;y=(i//4)*226;sheet.paste(im,(x,y));d.text((x+8,y+205),f"{p['time_s']:.1f} s",font=font(14),fill=MUTED)
    sheet.save(out/'contact-sheet.jpg',quality=93)
    write(out/'verification.json',{'format':'interaction-twin-motion-first-trailer','version':1,'movie':ref(movie),'poster':ref(out/'poster.jpg'),'captions':ref(out/'captions.vtt'),'srt':ref(out/'captions.srt'),'transcript':ref(out/'transcript.txt'),'contact_sheet':ref(out/'contact-sheet.jpg'),'proof_frames':proofs,'render':{'width':W,'height':H,'fps':FPS,'duration_s':45,'frames':1350,'stereo_sample_rate':48000},'source_captures':{'motion':ref(motion),'motion_receipt':ref(receipt),'robot':ROBOT_REF,'robot_receipt':ROBOT_JSON,'newton':ref(NEWTON),'newton_execution':ref(NEWTON_REPORT)},'time_mapping':{'master':MASTER_MAP,'robot':{'display_start_s':29,'display_duration_s':10,'source_duration_s':26.9,'initial_hold_display_s':.6,'whole_motion_display_s':8,'released_goals_closed_lid_hold_display_s':1.4},'newton':{'display_start_s':39,'display_end_s':43,'source_start_s':.4,'source_end_s':2.2},'brand':[43,45]},'robot_fixed_crop':crop,'robot_scope':'Complete authored kinematic rehearsal, no force/contact/grasp simulation. Both earbuds released onto goals, lid closed and tool withdrawn.','physics_scope':'Actual Newton CPU release/gravity/contact with assumed dimensions and materials; not robot grasp validation.','music':ref(MUSIC),'music_rights':'Original procedural project composition; no third-party samples.','speech':{'provider':'local Apple Samantha synthetic voice','rate_words_per_minute':175,'segments':segments},'captions_count':len(cues),'logo':ref(logo_path),'builder':ref(Path(__file__)),'elapsed_s':time.monotonic()-started,'new_api_calls':0,'new_gpu_calls':0,'public_media_replaced':False,'landing_changed':False,'published':False,'ffprobe':probe})
    print(json.dumps({'movie':ref(movie),'verification':ref(out/'verification.json'),'wall_s':time.monotonic()-started}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--motion',type=Path);p.add_argument('--receipt',type=Path);p.add_argument('--out',type=Path,default=BASE/'render-v1');p.add_argument('--prepare-voice',action='store_true')
    a=p.parse_args()
    if a.prepare_voice:
        a.out.mkdir(parents=True,exist_ok=True);segments,path=prepare_voice(a.out);captions(a.out,segments);print(json.dumps({'voice':ref(path),'segments':len(segments)}))
    elif a.motion and a.receipt:build(a.motion,a.receipt,a.out)
    else:p.error('--motion and --receipt are required')
