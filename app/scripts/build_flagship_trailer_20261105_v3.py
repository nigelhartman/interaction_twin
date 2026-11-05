#!/usr/bin/env python3
"""Output-only 45-second motion draft. No inference, publication or asset edits.

The first270 pictures are decoded from the approved V2 movie without editorial
changes. Later robot pictures fill the canvas. A charging placeholder is allowed
only in a storyboard; motion exports require the exact new capture contract.
"""
from __future__ import annotations
import argparse, hashlib, json, math, subprocess, time
from pathlib import Path
from types import SimpleNamespace
from PIL import Image, ImageDraw, ImageOps
import build_flagship_trailer_20261104_v1 as F

ROOT=F.ROOT; A=F.A; W,H,FPS,D=1920,1080,30,45
OUT=ROOT/'output/trailer-rework-20261105-v1'
PREVIOUS=ROOT/'showcase/media/flagship-20261104-v2/provenance.json'
REVALIDATION=ROOT/'output/flagship-trailer-20261104-v2/post-semantic-current-context-check-v1.json'
INTAKE=ROOT/'output/flagship-trailer-20261104-v1/intake-v1/manifest.json'
OLD_AUDIO=ROOT/'output/flagship-trailer-20261104-v1/audio-v3/audio.json'
CASE_CROP=(160,134,1120,674)  # Actual WebGL canvas, no recorder header/footer.
PROMPT='Connect the charger to this phone.'
BEATS=[(0,9,'Video → masks → parts → connected program'),(9,23,'Complete earbud robot rehearsal'),
       (23,26,'Different recorded objects'),(26,29,'One task request'),
       (29,42,'Complete charging rehearsal'),(42,45,'Interaction Twin')]
PROOF_TIMES=(.7,2.7,4.8,7.4,8.9,9.4,11.4,16.2,21.8,22.8,24.4,27.7,29.5,34.5,39.8,41.6,43.5)

def output_path(path):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT) or path==OUT:raise ValueError('Draft outputs must stay in the new output folder.')
    return path

def read(pointer):return json.loads(F.checked(pointer).read_text())

class Assets:
    def __init__(self,charging=None,case_capture=None):
        self.previous_ref=F.ref(PREVIOUS);self.previous=read(self.previous_ref)
        self.opening=F.checked(self.previous['movie']);self.intake=read(F.ref(INTAKE))
        self.revalidation_ref=F.ref(REVALIDATION);revalidated=read(self.revalidation_ref)
        context=F.CAPTURE.current_context(ROOT);self.catalog=context['catalog']
        if revalidated['current_catalog']!=self.catalog:raise ValueError('Capture context needs a fresh separate revalidation.')
        self.capture_validation={}
        for kind in ('robot','multiple_models','jabra_parts','connected'):
            row=self.intake[kind]
            self.capture_validation[kind]=F.CAPTURE.validate_receipt(kind,read(row['receipt']),F.checked(row['video']),context,ROOT)
        self.case=self.intake['robot'];self.case_doc=read(self.case['receipt'])
        self.knots=F.robot_editorial_knots(self.case_doc['typed_plan'])
        self.models=self.intake['multiple_models'];self.readers={};self.canonical_capture_frames={}
        self.capture_implementation_refs=[F.ref(ROOT/name)for name in ['showcase/case-motion-capture-20261105-v1.html','showcase/case-motion-capture-20261105-v1.js','scripts/validate_case_motion_capture_20261105_v1.cjs','scripts/bind_case_motion_capture_20261105_v1.py','showcase/charging-rehearsal-20261105-v1.html','showcase/charging-rehearsal-20261105-v1.js','showcase/charging-rehearsal-20261105-v1.bindings.json','showcase/charging-rehearsal-plan-20261105-v1.js','showcase/charging-rehearsal-view-20261105-v1.js','scripts/validate_charging_capture_20261105_v1.cjs','scripts/bind_charging_capture_20261105_v1.py']]

        self.logo=Image.open(ROOT/'showcase/brand/v2/mark-1024.png').convert('RGBA')
        self.case_capture_ref=F.ref(case_capture.resolve())if case_capture else None
        self.case_capture=read(self.case_capture_ref)if case_capture else None
        if self.case_capture:
            c=self.case_capture
            subprocess.run(['node',str(ROOT/'scripts/validate_case_motion_capture_20261105_v1.cjs'),str(F.checked(self.case_capture_ref))],check=True,capture_output=True,text=True)
            if c.get('video_ref_requires_local_download_binding')is not False or c.get('local_validation',{}).get('passed')is not True:raise ValueError('Case clip requires local downloaded-byte binding.')
            if c['prior_capture_receipt']!=self.case['receipt']:raise ValueError('Case plan differs from the exact historically reviewed task.')
            F.checked(c['video_ref']);F.checked(c['exact_browser_receipt_ref'])
        self.charging_ref=F.ref(charging.resolve())if charging else None
        self.charging=read(self.charging_ref)if charging else None
        if self.charging:
            c=self.charging
            if not(c.get('format')=='interaction-twin-charging-rehearsal-capture'and c.get('version')==1
                   and c.get('canvas')==[W,H]and abs(c.get('duration_s',0)-13)<.15
                   and c.get('authored_kinematic_rehearsal')is True
                   and c.get('physical_insertion_verified')is False and c.get('charging_success_verified')is False):
                raise ValueError('Exact authored charging capture required.')
            subprocess.run(['node',str(ROOT/'scripts/validate_charging_capture_20261105_v1.cjs'),str(F.checked(self.charging_ref))],check=True,capture_output=True,text=True)
            if c.get('video_ref_requires_local_download_binding')is not False or c.get('local_validation',{}).get('passed')is not True:raise ValueError('Charging clip requires local downloaded-byte binding.')
            F.checked(c['video_ref']);F.checked(c['exact_browser_receipt_ref'])
    def capture_end(self,key,capture):
        if key not in self.canonical_capture_frames:
            # MediaRecorder variable PTS can yield one fewer30fps output picture.
            # Count the same fps conversion used by RawReader, then hold the last
            # actual decoded picture only during the independently proven settle.
            result=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(F.checked(capture['video_ref'])),'-vf','fps=30','-progress','pipe:1','-f','null','-'],check=True,capture_output=True,text=True)
            count=[int(line.split('=',1)[1])for line in result.stdout.splitlines()if line.startswith('frame=')][-1]
            if count<300:raise ValueError('Complete normalized capture frame clock required.')
            self.canonical_capture_frames[key]=count
        return (self.canonical_capture_frames[key]-1)/FPS
    def decode(self,key,path,t,size):
        old=self.readers.get(key)
        if old is None or t<old.time-.04:
            if old:old.close()
            old=self.readers[key]=A.RawReader(path,max(0,t),size)
        return old.get(t).copy()
    def case_picture(self,t):
        film_elapsed=max(0,min(14,t-9))
        if self.case_capture:
            c=self.case_capture;at=min(self.capture_end('case_fresh',c),film_elapsed)
            runtime=F.interpolate_clock(c['editorial_clock_knots'],film_elapsed*15/14)
            return self.decode('case_fresh',F.checked(c['video_ref']),at,(W,H)),runtime
        runtime=F.interpolate_clock(self.knots,film_elapsed*15/14)
        encoded=runtime/self.case_doc['task_duration_s']*self.case['encoded_last_timestamp_s']
        picture=self.decode('case',F.checked(self.case['video']),min(self.case['encoded_last_timestamp_s']-.04,encoded),(1280,800))
        return picture.crop(CASE_CROP).resize((W,H),Image.Resampling.LANCZOS),runtime
    def model_picture(self,t):
        p=max(0,min(.999,(t-23)/6));end=self.models['encoded_last_timestamp_s']
        im=self.decode('models',F.checked(self.models['video']),min(end-.04,p*end),(W,H))
        # Gentle editorial push on the actual moving native capture; no asset/pose edit.
        scale=1+.06*p;size=(round(W*scale),round(H*scale));large=im.resize(size,Image.Resampling.LANCZOS);left=(size[0]-W)//2;top=(size[1]-H)//2
        return large.crop((left,top,left+W,top+H))
    def close(self):
        for reader in self.readers.values():reader.close()
        self.readers.clear()

def scope(im):
    # One restrained scope label, outside the centered robot/endpoint action.
    F.text(im,(64,1018),'Authored robot preview',22,F.MUTED)

def frame(t,a):
    trace={'film_s':t,'editorial':True,'physical_success_verified':False}
    if t<9:
        im=a.decode('opening',a.opening,t,(W,H))
        return im,{**trace,'kind':'unchanged_V2_opening_picture','source_movie_s':t,'editorial_pixel_changes':False}
    if t<23:
        im,runtime=a.case_picture(t);scope(im)
        return im,{**trace,'kind':'fullscreen_case_robot','captured_task_runtime_s':runtime,'source_order_preserved':True,'fresh_full_canvas':bool(a.case_capture),'crop':None if a.case_capture else CASE_CROP,'case_capture':a.case_capture_ref}
    if t<29:
        im=a.model_picture(t)
        if t>=26:
            draw=ImageDraw.Draw(im);box=(180,72,1740,214)
            draw.rounded_rectangle(box,radius=22,fill='white',outline='#d9e1f3',width=2)
            F.text(im,(222,111),'›',45,F.BLUE,True)
            count=min(len(PROMPT),math.floor(max(0,t-26.12)/1.65*len(PROMPT)))
            F.text(im,(276,114),PROMPT[:count]+('│'if count<len(PROMPT)else''),42,F.INK,True)
        return im,{**trace,'kind':'moving_native_breadth'if t<26 else'typed_task_request','prompt':PROMPT if t>=26 else None,'live_inference':False,'editorial_slow_push_scale':1+.06*max(0,min(.999,(t-23)/6))}
    if t<42:
        if a.charging:
            c=a.charging;at=min(a.capture_end('charging',c),max(0,t-29))
            im=a.decode('charging',F.checked(c['video_ref']),at,(W,H))  # Captured top-left scope label stays; no duplicate overlay.
            return im,{**trace,'kind':'fullscreen_charging_robot','capture_s':at,'receipt':a.charging_ref,'authored_endpoint_proxies':True}
        im=a.model_picture(28.9);draw=ImageDraw.Draw(im);draw.rectangle((120,420,1800,680),fill='white')
        F.text(im,(180,485),'DRAFT · charging rehearsal capture pending',44,F.MUTED,True)
        return im,{**trace,'kind':'explicit_pending_capture_placeholder','placeholder':True}
    im=A.ending(t-42,SimpleNamespace(logo=a.logo))
    return im,{**trace,'kind':'brand_end'}

def storyboard(folder,a):
    folder=output_path(folder);folder.mkdir(parents=True,exist_ok=False);rows=[]
    for n,t in enumerate(PROOF_TIMES):
        im,trace=frame(t,a);p=folder/f'frame-{n:02}.jpg';im.save(p,quality=95)
        rows.append({'time_s':t,'file':F.ref(p),'trace':trace})
    A.contact_sheet(rows,folder/'storyboard.jpg')
    F.write(folder/'storyboard.json',{'format':'interaction-twin-fullscreen-motion-storyboard','version':3,'duration_s':D,'beats':BEATS,'frames':rows,'capture_revalidation':a.revalidation_ref,'charging_capture':a.charging_ref,'case_capture':a.case_capture_ref,'new_API_calls':0,'publication':False})
    print(json.dumps({'storyboard':F.ref(folder/'storyboard.json'),'board':F.ref(folder/'storyboard.jpg')},indent=2))

def audio(folder,music=None,music_provenance=None,voice_cache=None):
    folder=output_path(folder);folder.mkdir(parents=True,exist_ok=False);previous=read(F.ref(OLD_AUDIO))
    # Original local normal voice only. Existing takes are reused at1x.
    reused=[(0,.2),(1,3.2),(3,9.2),(4,11.05),(5,13.39),(6,19.73),(7,23.12),(10,38.6),(11,42.2)]
    voices=[]
    for index,start in reused:
        v=previous['voice'][index];F.checked(v['processed']);assert v['atempo']==1
        voices.append({**v,'start_s':start,'end_s':start+v['end_s']-v['start_s'],'reused_exact_take':True})
    for index,(start,latest,words)in enumerate([(26.1,28.9,PROMPT),(29.15,32.5,'Rehearse the connection.')]):
        raw=folder/f'normal-voice-new-{index}.aiff';wav=raw.with_suffix('.wav')
        if voice_cache:
            import shutil
            cached=output_path(voice_cache)
            shutil.copyfile(cached/raw.name,raw)
        else:
            subprocess.run(['/usr/bin/say','-v','Samantha','-r','173','-o',str(raw),words],check=True)
        subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(raw),'-af','loudnorm=I=-18:TP=-3:LRA=7','-ar','48000','-ac','2',str(wav)],check=True)
        duration=A.V.duration(wav)
        if start+duration>latest:raise ValueError('Normal narration needs a longer slot, not speedup: '+words)
        voices.append({'start_s':start,'end_s':start+duration,'text':words,'raw':F.ref(raw),'processed':F.ref(wav),'atempo':1,'local_voice':'Samantha','reused_exact_take':False})
    voices.sort(key=lambda v:v['start_s'])
    for left,right in zip(voices,voices[1:]):assert left['end_s']+.05<=right['start_s'],'Narration overlaps'
    music_source_ref=F.ref(music.resolve())if music else previous['music'];music_path=F.checked(music_source_ref)
    music_note_ref=F.ref(music_provenance.resolve())if music_provenance else None
    if music and not music_note_ref:raise ValueError('Owner-supplied music needs its source/license provenance note.')
    selected=folder/'music-bed-selected.wav'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(music_path),'-t','45','-af','loudnorm=I=-24:TP=-5:LRA=8,afade=t=in:st=0:d=0.25,afade=t=out:st=43.6:d=1.4','-ar','48000','-ac','2',str(selected)],check=True)
    music_ref=F.ref(selected);music_path=selected
    arguments=['ffmpeg','-nostdin','-v','error','-i',str(music_path)];filters=[];labels=[]
    for n,v in enumerate(voices,1):
        arguments+=['-i',str(F.checked(v['processed']))];delay=round(v['start_s']*1000)
        filters.append(f'[{n}:a]adelay={delay}|{delay},apad=whole_dur=45[v{n}]');labels.append(f'[v{n}]')
    filters+=[''.join(labels)+f'amix=inputs={len(labels)}:normalize=0,atrim=0:45,asplit=2[voice][side]',
             '[0:a]volume=.70[music];[music][side]sidechaincompress=threshold=.012:ratio=4:attack=8:release=240[duck];[duck][voice]amix=inputs=2:normalize=0,loudnorm=I=-17:TP=-3:LRA=8[a]']
    target=folder/'normal-voice-draft-mix.wav'
    subprocess.run(arguments+['-filter_complex',';'.join(filters),'-map','[a]','-t','45','-ar','48000','-ac','2',str(target)],check=True)
    F.write(folder/'audio.json',{'format':'interaction-twin-local-normal-voice-draft','version':1,'mix':F.ref(target),'music':music_ref,'music_source':music_source_ref,'music_provenance':music_note_ref,'music_selected_interval_s':[0,45],'standalone_music_distribution':False,'voice':voices,'hosted_speech':False,'ElevenLabs_used':False,'all_speech_atempo':1,'new_API_calls':0})
    def stamp(value):
        ms=round(value*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02}.{ms%1000:03}'
    (folder/'narration.vtt').write_text('WEBVTT\n\n'+'\n\n'.join(f"{n}\n{stamp(v['start_s'])} --> {stamp(v['end_s'])}\n{v['text']}"for n,v in enumerate(voices,1))+'\n')
    (folder/'narration.txt').write_text('\n'.join(v['text']for v in voices)+'\n')
    print(json.dumps({'mix':F.ref(target),'audio':F.ref(folder/'audio.json')},indent=2))

def render_motion_draft(folder,a,audio_path,full_quality=False,review_receipts=()):
    if not a.charging or not a.case_capture:raise ValueError('Motion draft requires both fresh, source-bound full-canvas recordings; no placeholder or clipped-case export.')
    folder=output_path(folder)
    if full_quality and len(review_receipts)<2:raise ValueError('The1080p draft requires both completed independent review rounds.')
    checked_reviews=[F.ref(Path(p).resolve())for p in review_receipts]
    folder.mkdir(parents=True,exist_ok=False);size=(W,H)if full_quality else(960,540);movie=folder/('full-quality-draft-45s.mp4'if full_quality else'motion-draft-45s.mp4')
    command=['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}','-r','30','-i','pipe:0','-i',str(audio_path),'-map','0:v','-map','1:a','-t','45','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(movie)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE);traces=[];started=time.monotonic()
    try:
        for n in range(D*FPS):
            im,trace=frame(n/FPS,a);traces.append(dict(trace,frame=n));process.stdin.write((im if full_quality else im.resize(size,Image.Resampling.LANCZOS)).tobytes())
        process.stdin.close();error=process.stderr.read();assert process.wait()==0,error.decode()[-2000:]
    except BaseException:process.kill();process.wait();raise
    F.write(folder/'draft-receipt.json',{'format':'interaction-twin-fullscreen-motion-draft','version':3,'movie':F.ref(movie),'duration_s':D,'canvas':list(size),'fps':FPS,'full_quality_draft':full_quality,'completed_review_rounds':checked_reviews,'frames':traces,'previous_film':a.previous['movie'],'capture_revalidation':a.revalidation_ref,'charging_capture':a.charging_ref,'case_capture':a.case_capture_ref,'audio':F.ref(audio_path),'builder':F.ref(Path(__file__)),'normalized_capture_frame_counts':a.canonical_capture_frames,'end_frame_hold_note':'Only the proven settled endpoint may repeat its last actual decoded frame when normalizing variable-PTS captures to30fps. No in-action interpolation or synthetic picture.','capture_implementation_refs':a.capture_implementation_refs,'charging_implementation_freeze':F.ref(ROOT/'output/charging-rehearsal-20261105-v1/implementation-freeze.json'),'source_face_derivation_check':F.ref(ROOT/'output/charging-rehearsal-20261105-v1/source-face-fixture-derivation-check.json'),'capture_source_measurements':'Browser-receipt staging measurements plus separate CPU source-face derivation check; not a physical geometry validation.','phone_display_fixture':a.charging['asset_staging']['phone']['display_derivative'],'first270_source_pictures_unedited_before_encoding':True,'encoded_byte_identity_claimed':False,'publication':False,'new_API_calls':0,'new_GPU_calls':0,'elapsed_s':time.monotonic()-started})
    print(json.dumps({'movie':F.ref(movie),'receipt':F.ref(folder/'draft-receipt.json')},indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--storyboard',type=Path);p.add_argument('--audio-folder',type=Path);p.add_argument('--voice-cache',type=Path);p.add_argument('--music',type=Path);p.add_argument('--music-provenance',type=Path);p.add_argument('--charging-receipt',type=Path);p.add_argument('--case-receipt',type=Path);p.add_argument('--motion-draft',type=Path);p.add_argument('--audio',type=Path);p.add_argument('--full-quality',action='store_true');p.add_argument('--review-receipt',type=Path,action='append',default=[]);args=p.parse_args()
    if args.audio_folder:audio(args.audio_folder,args.music,args.music_provenance,args.voice_cache)
    if args.storyboard or args.motion_draft:
        assets=Assets(args.charging_receipt,args.case_receipt)
        try:
            if args.storyboard:storyboard(args.storyboard,assets)
            if args.motion_draft:
                if not args.audio:p.error('--motion-draft requires --audio')
                render_motion_draft(args.motion_draft,assets,args.audio.resolve(),args.full_quality,args.review_receipt)
        finally:assets.close()
    if not(args.storyboard or args.audio_folder or args.motion_draft):p.error('Choose a draft action.')

if __name__=='__main__':main()
