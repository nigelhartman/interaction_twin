#!/usr/bin/env python3
"""Local,1x narrative paragraphs for output-only45s/180s trailer drafts.

No image/model/API requests. Native macOS Samantha voice only. Never modifies a
previous take, script, liked movie or published default. Every paragraph keeps
its natural duration; a slot overflow fails rather than accelerating speech.
"""
from __future__ import annotations
import argparse,hashlib,json,re,shutil,subprocess,wave
from pathlib import Path
import numpy as np
import build_flagship_trailer_20261105_v3 as M
ROOT=M.ROOT;F=M.F;A=M.A;OUT=M.OUT

def run(args):return subprocess.run(args,check=True,capture_output=True,text=True)
def stamp(s):
    ms=round(s*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02}.{ms%1000:03}'
def pcm(p):
    with wave.open(str(p),'rb')as w:
        if(w.getnchannels(),w.getsampwidth(),w.getframerate())!=(2,2,48000):raise ValueError('48k stereo16bit PCM required.')
        return np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').reshape(-1,2).copy()
def wav(p,frames):
    with wave.open(str(p),'wb')as w:w.setparams((2,2,48000,len(frames),'NONE','not compressed'));w.writeframes(frames.astype('<i2').tobytes())

def build(script,folder,duration,intro_folder=None,reuse_folder=None,voice_rate=150):
    script=M.output_path(script);folder=M.output_path(folder);d=json.loads(script.read_text());assert d['format']=='interaction-twin-narrative-script-draft';assert duration in(45,180)
    rows=d[f'{duration}s']['paragraphs'];assert len(rows)==(3 if duration==45 else 11)
    words=sum(len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b",r['text']))for r in rows)
    if not(60<=words<=90 if duration==45 else 330<=words<=380):raise ValueError('Narrative word budget differs from reviewed scope.')
    folder.mkdir(parents=True,exist_ok=False);reuse=json.loads((M.output_path(reuse_folder)/'audio.json').read_text())if reuse_folder else None
    if duration==180:
        if not intro_folder:raise ValueError('Exact new45s audio is required for the180s draft.')
        intro_folder=M.output_path(intro_folder);intro=json.loads((intro_folder/'audio.json').read_text());assert intro['duration_s']==45 and intro['all_speech_atempo']==1
        if [v['text']for v in intro['voice']]!=[r['text']for r in rows[:3]]:raise ValueError('Opening speech differs from reviewed shared story.')
        selected=rows[3:];offset=45;length=135
    else:selected=rows;offset=0;length=45
    voices=[]
    for i,row in enumerate(selected):
        text=row['text'];start=row['start_s']-offset;latest=row['latest_end_s']-offset;raw=folder/f'paragraph-{i:02}.aiff';processed=raw.with_suffix('.wav')
        saved=next((v for v in(reuse or{}).get('voice',[])if v['text']==text and v.get('local_voice_rate_setting')==voice_rate),None)
        if saved:
            shutil.copyfile(F.checked(saved['raw']),raw)
        else:subprocess.run(['/usr/bin/say','-v','Samantha','-r',str(voice_rate),'-o',str(raw),text],check=True)
        run(['ffmpeg','-nostdin','-v','error','-i',str(raw),'-af','loudnorm=I=-18:TP=-3:LRA=7','-ar','48000','-ac','2',str(processed)])
        seconds=A.V.duration(processed)
        if start<0 or start+seconds>latest:raise ValueError(f'Paragraph{i+1} needs{seconds:.3f}s at1x, longer than slot{latest-start:.3f}s; revise words or timing, never speed up.')
        voices.append({'start_s':start,'end_s':start+seconds,'latest_end_s':latest,'text':text,'raw':F.ref(raw),'processed':F.ref(processed),'atempo':1,'local_voice':'Samantha','local_voice_rate_setting':voice_rate,'reused_exact_raw_take':bool(saved),'paragraph_number':row['paragraph']})
    assert all(a['end_s']+.05<=b['start_s']for a,b in zip(voices,voices[1:])), 'Natural paragraphs overlap.'
    if duration==45:
        bed=json.loads((OUT/'audio-v4/audio.json').read_text())['music'];volume='.70';threshold='.012';ratio='4';attack='8';release='240'
    else:
        old=json.loads((ROOT/'showcase/media/submission-final-20261104-v2/provenance.json').read_text());bed=json.loads(F.checked(old['audio']).read_text())['music'];volume='.52';threshold='.018';ratio='5';attack='12';release='270'
    args=['ffmpeg','-nostdin','-v','error','-i',str(F.checked(bed))];filters=[];labels=[]
    for i,v in enumerate(voices,1):
        args+=['-i',str(F.checked(v['processed']))];delay=round(v['start_s']*1000);filters.append(f'[{i}:a]adelay={delay}|{delay},apad=whole_dur={length}[v{i}]');labels.append(f'[v{i}]')
    filters+=[''.join(labels)+f'amix=inputs={len(voices)}:normalize=0,atrim=0:{length},asplit=2[voice][side]',f'[0:a]volume={volume}[music];[music][side]sidechaincompress=threshold={threshold}:ratio={ratio}:attack={attack}:release={release}[duck];[duck][voice]amix=inputs=2:normalize=0,loudnorm=I=-17:TP=-3:LRA=8[a]']
    mixed=folder/('mix.wav'if duration==45 else'tail-mix.wav');run(args+['-filter_complex',';'.join(filters),'-map','[a]','-t',str(length),'-ar','48000','-ac','2',str(mixed)])
    if duration==180:
        first=pcm(F.checked(intro['mix']));tail=pcm(mixed);assert len(first)==45*48000 and len(tail)==135*48000
        mixed=folder/'mix.wav';wav(mixed,np.concatenate([first,tail]));voices=[dict(v)for v in intro['voice']]+[{**v,'start_s':v['start_s']+45,'end_s':v['end_s']+45,'latest_end_s':v['latest_end_s']+45}for v in voices]
    assert len(pcm(mixed))==duration*48000
    cues='WEBVTT\n\n'+'\n\n'.join(f"{i}\n{stamp(v['start_s'])} --> {stamp(v['end_s'])}\n{v['text']}"for i,v in enumerate(voices,1))+'\n';(folder/'captions.vtt').write_text(cues);(folder/'transcript.txt').write_text('\n\n'.join(v['text']for v in voices)+'\n')
    F.write(folder/'audio.json',{'format':'interaction-twin-continuous-story-narration','version':3,'duration_s':duration,'words':words,'script':F.ref(script),'mix':F.ref(mixed),'voice':voices,'music':bed,'intro_audio':F.ref(intro_folder/'audio.json')if duration==180 else None,'music_provenance':F.ref(OUT/'music-intake-v1/provenance.json'),'all_speech_atempo':1,'broad_explanatory_paragraphs_not_action_microtakes':True,'hosted_speech':False,'ElevenLabs_used':False,'new_API_calls':0,'publication':False,'builder':F.ref(Path(__file__))})
    print(json.dumps({'audio':F.ref(folder/'audio.json'),'mix':F.ref(mixed),'captions':F.ref(folder/'captions.vtt')},indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--script',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--duration',type=int,choices=[45,180],required=True);p.add_argument('--intro-audio',type=Path);p.add_argument('--reuse-audio',type=Path);p.add_argument('--rate',type=int,choices=range(125,181),default=150);a=p.parse_args();build(a.script,a.out,a.duration,a.intro_audio,a.reuse_audio,a.rate)
if __name__=='__main__':main()
