#!/usr/bin/env python3
"""Output-only180s draft: new45s intro, saved walkthrough, full-screen robots.

No capture, API, hosted speech, geometry mutation or publication. The original
walkthrough video is decoded unchanged outside its27s robot chapter. A new
normal local narration replaces only that chapter; saved speech elsewhere stays
at1x. The entire new45s intro video stream is copied without re-encoding.
"""
from __future__ import annotations
import argparse,hashlib,json,subprocess,time,wave
from pathlib import Path
import numpy as np
import build_flagship_trailer_20261105_v3 as M

ROOT=M.ROOT;F=M.F;A=M.A;OUT=M.OUT
OLD_PROVENANCE=ROOT/'showcase/media/submission-final-20261104-v2/provenance.json'
OLD_TAIL=ROOT/'output/submission-final-20261104-v2/walkthrough-tail.mp4'
ROBOT_CUES=[(.35,13.8,'The saved object program becomes a task you can inspect and edit. Here, the robot opens the case, moves both earbuds to their goals, and closes it.'),
            (14.1,26.7,'A second preview rehearses connecting a cable between the charger and phone. These are authored kinematics. Physical contact, insertion, charging, and robot execution still need validation.')]

def command(args):return subprocess.run(args,check=True,capture_output=True,text=True)
def write(p,d):F.write(p,d)
def stamp(s):
    ms=round(s*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02}.{ms%1000:03}'
def read_wav(p):
    with wave.open(str(p),'rb')as w:
        if(w.getnchannels(),w.getsampwidth(),w.getframerate())!=(2,2,48000):raise ValueError('48k stereo16bit input required.')
        return np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').reshape(-1,2).copy()
def write_wav(p,data):
    with wave.open(str(p),'wb')as w:w.setparams((2,2,48000,len(data),'NONE','not compressed'));w.writeframes(np.clip(np.rint(data),-32768,32767).astype('<i2').tobytes())

def audio(folder,voice_cache=None):
    folder=M.output_path(folder);folder.mkdir(parents=True,exist_ok=False)
    old=json.loads(OLD_PROVENANCE.read_text());saved=json.loads(F.checked(old['audio']).read_text());intro=json.loads((OUT/'audio-v4/audio.json').read_text())
    if any(v['atempo']!=1 for v in saved['voice']+intro['voice']):raise ValueError('Normal speech only.')
    voices=[]
    for i,(start,latest,text)in enumerate(ROBOT_CUES):
        raw=folder/f'normal-robot-{i}.aiff';wav=raw.with_suffix('.wav')
        if voice_cache:
            import shutil
            shutil.copyfile(M.output_path(voice_cache)/raw.name,raw)
        else:subprocess.run(['/usr/bin/say','-v','Samantha','-r','173','-o',str(raw),text],check=True)
        command(['ffmpeg','-nostdin','-v','error','-i',str(raw),'-af','loudnorm=I=-18:TP=-3:LRA=7','-ar','48000','-ac','2',str(wav)])
        duration=A.V.duration(wav)
        if duration+start>latest:raise ValueError('Narration must fit at1x; do not speed it up.')
        voices.append({'start_s':start,'end_s':start+duration,'text':text,'raw':F.ref(raw),'processed':F.ref(wav),'atempo':1,'local_voice':'Samantha'})
    bed=folder/'robot-score-section.wav';command(['ffmpeg','-nostdin','-v','error','-ss','78','-i',str(F.checked(saved['music'])),'-t','27','-ar','48000','-ac','2',str(bed)])
    args=['ffmpeg','-nostdin','-v','error','-i',str(bed)];filters=[];labels=[]
    for i,v in enumerate(voices,1):
        args+=['-i',str(F.checked(v['processed']))];delay=round(v['start_s']*1000);filters.append(f'[{i}:a]adelay={delay}|{delay},apad=whole_dur=27[v{i}]');labels.append(f'[v{i}]')
    filters+=[''.join(labels)+'amix=inputs=2:normalize=0,atrim=0:27,asplit=2[voice][side]','[0:a]volume=0.52[music];[music][side]sidechaincompress=threshold=0.018:ratio=5:attack=12:release=270[duck];[duck][voice]amix=inputs=2:normalize=0,loudnorm=I=-17:TP=-3:LRA=8[a]']
    robot=folder/'robot-chapter-mix.wav';command(args+['-filter_complex',';'.join(filters),'-map','[a]','-t','27','-ar','48000','-ac','2',str(robot)])
    previous=read_wav(F.checked(saved['mix']));replacement=read_wav(robot);opening=read_wav(F.checked(intro['mix']));assert len(previous)==135*48000 and len(replacement)==27*48000 and len(opening)==45*48000
    # The two50ms joins occur in music-only regions. Saved outside-chapter PCM
    # stays byte-identical; blend only inside the newly replaced chapter.
    a,b=78*48000,105*48000;replacement=replacement.astype(np.float64);join=2400;weight=np.linspace(0,1,join)[:,None]
    replacement[:join]=previous[a:a+join]*(1-weight)+replacement[:join]*weight
    replacement[-join:]=replacement[-join:]*(1-weight)+previous[b-join:b]*weight
    tail=previous.copy();tail[a:b]=np.clip(np.rint(replacement),-32768,32767).astype(np.int16)
    assert np.array_equal(tail[:a],previous[:a])and np.array_equal(tail[b:],previous[b:])
    target=folder/'normal-voice-180s-mix.wav';write_wav(target,np.concatenate([opening,tail]))
    cues=[dict(v)for v in intro['voice']]
    for i,v in enumerate(saved['voice']):
        if i!=5:cues.append({**v,'start_s':v['start_s']+45,'end_s':v['end_s']+45})
    cues.extend({**v,'start_s':v['start_s']+123,'end_s':v['end_s']+123}for v in voices);cues.sort(key=lambda v:v['start_s'])
    assert all(a['end_s']<b['start_s']for a,b in zip(cues,cues[1:]));assert cues[-1]['end_s']<=180
    (folder/'captions.vtt').write_text('WEBVTT\n\n'+'\n\n'.join(f"{i}\n{stamp(v['start_s'])} --> {stamp(v['end_s'])}\n{v['text']}"for i,v in enumerate(cues,1))+'\n')
    (folder/'transcript.txt').write_text('\n\n'.join(v['text']for v in cues)+'\n')
    write(folder/'audio.json',{'format':'interaction-twin-local-extended-audio-draft','version':4,'mix':F.ref(target),'voice':cues,'new_robot_voice':voices,'previous_tail_audio':old['audio'],'opening_audio':F.ref(OUT/'audio-v4/audio.json'),'old_tail_PCM_unchanged_outside_s':[78,105],'music_join_blend_ms':50,'duration_s':180,'all_speech_atempo':1,'hosted_speech':False,'ElevenLabs_used':False,'new_API_calls':0})
    print(json.dumps({'audio':F.ref(folder/'audio.json'),'mix':F.ref(target)},indent=2),flush=True)

def video_hashes(p):
    lines=command(['ffmpeg','-nostdin','-v','error','-i',str(p),'-map','0:v:0','-an','-f','framemd5','pipe:1']).stdout
    return[line.rsplit(',',1)[1].strip()for line in lines.splitlines()if line and not line.startswith('#')]

def build(folder,opening,audio_folder):
    folder=M.output_path(folder);opening=M.output_path(opening);audio_folder=M.output_path(audio_folder);assert opening.is_file()
    full=json.loads((opening.parent/'draft-receipt.json').read_text());assert full['movie']==F.ref(opening)and full['full_quality_draft']is True and full['canvas']==[1920,1080]and len(full['completed_review_rounds'])>=2
    old=json.loads(OLD_PROVENANCE.read_text());F.checked(old['movie']);assert A.verify_video(OLD_TAIL,{'duration_s':135})and A.verify_video(opening,{'duration_s':45})
    aud=json.loads((audio_folder/'audio.json').read_text());audio_path=F.checked(aud['mix']);assert aud['all_speech_atempo']==1
    folder.mkdir(parents=True,exist_ok=False);(folder/'builder-source.py').write_bytes(Path(__file__).read_bytes());tail=folder/'patched-walkthrough-tail-video.mp4';proofs=[];clock=[]
    assets=M.Assets(F.checked(full['charging_capture']),F.checked(full['case_capture']));reader=None
    encoder=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','30','-i','pipe:0','-an','-t','135','-c:v','libx264','-threads','4','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(tail)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    start=time.monotonic()
    try:
        for n in range(4050):
            t=n/30
            if t<78 or t>=105:
                if reader is None:reader=A.RawReader(OLD_TAIL,t,(1920,1080))
                image=reader.get(t).copy();trace={'kind':'saved_current_walkthrough_picture','source_tail_s':t,'editorial_pixel_changes':False}
            else:
                if reader:reader.close();reader=None
                source_t=9+t-78 if t<92 else 29+t-92
                image,trace=M.frame(source_t,assets);trace={**trace,'source_rehearsal_editorial_s':source_t}
            clock.append({**trace,'film_s':t+45})
            if n in{0,540,1080,1320,1500,1890,2340,2370,2490,2640,2730,2760,2880,3030,3090,3145,3150,3450,3840,4000}:
                p=folder/f'frame-{n:04}.jpg';image.save(p,quality=94);proofs.append({'time_s':t+45,'file':F.ref(p)})
            encoder.stdin.write(image.tobytes())
        encoder.stdin.close();err=encoder.stderr.read();assert encoder.wait()==0,err.decode()[-2000:]
    except BaseException:encoder.kill();encoder.wait();raise
    finally:
        if reader:reader.close()
        assets.close()
    opening_silent=folder/'opening-video-stream.mp4';command(['ffmpeg','-nostdin','-v','error','-i',str(opening),'-map','0:v:0','-an','-c:v','copy',str(opening_silent)])
    listing=folder/'concat.txt';listing.write_text(''.join("file '"+str(p).replace("'","'\\''")+"'\n"for p in(opening_silent,tail)))
    silent=folder/'joined-video-stream.mp4';command(['ffmpeg','-nostdin','-v','error','-f','concat','-safe','0','-i',str(listing),'-map','0:v:0','-an','-c:v','copy','-t','180',str(silent)])
    movie=folder/'interaction-twin-180s-v4-draft.mp4';command(['ffmpeg','-nostdin','-v','error','-i',str(silent),'-i',str(audio_path),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k','-t','180','-movflags','+faststart',str(movie)])
    probe=A.verify_video(movie,{'duration_s':180});frames=video_hashes(movie);first=video_hashes(opening);assert len(frames)==5400 and len(first)==1350 and frames[:1350]==first,'New45s opening must remain decoded-pixel identical.'
    write(folder/'opening-copy-proof.json',{'format':'interaction-twin-opening-stream-copy-proof','version':1,'opening':F.ref(opening),'movie':F.ref(movie),'opening_decoded_frames_equal':1350,'opening_video_reencoded':False,'new_model_calls':0})
    write(folder/'clock.json',clock);A.contact_sheet(proofs,folder/'storyboard.jpg')
    for name in('captions.vtt','transcript.txt'):(folder/name).write_bytes((audio_folder/name).read_bytes())
    write(folder/'provenance.json',{'format':'interaction-twin-final-submission-draft','version':4,'movie':F.ref(movie),'duration_s':180,'fps':30,'canvas':[1920,1080],'probe':probe,'new_opening':F.ref(opening),'opening_draft_receipt':F.ref(opening.parent/'draft-receipt.json'),'opening_copy_proof':F.ref(folder/'opening-copy-proof.json'),'previous_walkthrough_tail':F.ref(OLD_TAIL),'previous_walkthrough_provenance':F.ref(OLD_PROVENANCE),'saved_tail_pictures_retained_outside_tail_s':[78,105],'replacement_fullscreen_chapter_film_s':[123,150],'replacement_actions':[{'case':full['case_capture'],'duration_s':14},{'charging':full['charging_capture'],'duration_s':13}],'audio':F.ref(audio_folder/'audio.json'),'captions':F.ref(folder/'captions.vtt'),'clock':F.ref(folder/'clock.json'),'builder':F.ref(Path(__file__)),'builder_snapshot':F.ref(folder/'builder-source.py'),'all_speech_atempo':1,'authored_kinematics_not_physical_execution':True,'phone_display_fixture_disclosed_in_opening_receipt':True,'new_API_calls':0,'new_remote_GPU_calls':0,'published_defaults_changed':False,'publication':False,'elapsed_s':time.monotonic()-start})
    print(json.dumps({'movie':F.ref(movie),'provenance':F.ref(folder/'provenance.json'),'opening_copy_proof':F.ref(folder/'opening-copy-proof.json')},indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--audio-folder',type=Path);ap.add_argument('--voice-cache',type=Path);ap.add_argument('--build-folder',type=Path);ap.add_argument('--opening',type=Path);ap.add_argument('--audio',type=Path);a=ap.parse_args()
    if a.audio_folder:audio(a.audio_folder,a.voice_cache)
    if a.build_folder:
        if not a.opening or not a.audio:ap.error('--build-folder requires --opening and --audio folder')
        build(a.build_folder,a.opening,a.audio)
    if not(a.audio_folder or a.build_folder):ap.error('Choose an output-only draft action.')
if __name__=='__main__':main()
