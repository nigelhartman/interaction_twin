#!/usr/bin/env python3
"""Exact chapter-level captions for the saved local narration, without an aligner.

Whole-paragraph cues use the measured lead and PCM duration of each chapter.
These are explicitly chapter captions; no word-level alignment is asserted.
"""
from pathlib import Path
import hashlib,json,shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/submission-walkthrough-20261101-v1'
RENDER=OUT/'render-v3'
MEDIA=ROOT/'showcase/media';PREFIX='submission-walkthrough-20261101-v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def record(p,kind):return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'kind':kind}
def stamp(seconds):
 ms=round(seconds*1000);hours,ms=divmod(ms,3600000);minutes,ms=divmod(ms,60000);seconds,ms=divmod(ms,1000)
 return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}'
def main():
 manifest_path=RENDER/'source-manifest.json';manifest=json.loads(manifest_path.read_text())
 speech=manifest['speech_content'];rows=speech['rows'];assert len(rows)==len(manifest['chapters'])==7
 vtt=['WEBVTT','','NOTE Chapter captions. Exact measured narration chapter boundaries; no word-level alignment.','']
 text=['INTERACTION TWIN — NARRATED SUBMISSION WALKTHROUGH DRAFT','172 seconds · local synthetic Samantha narration · saved results, not a live upload','',
       'Voice: installed macOS Samantha; local speech with a declared, pitch-preserving tempo factor of 1.08.','The 54-second trailer and the model-directed MCP assessment remain separate artifacts.','']
 cues=[]
 for chapter,voice in zip(manifest['chapters'],rows):
  assert chapter['chapter']==voice['chapter']
  transcript=(ROOT/voice['text']['path']).read_text().strip();assert transcript==chapter['narration']
  assert sha(ROOT/voice['text']['path'])==voice['text']['sha256']
  start=chapter['start_s']+voice['narration_lead_s'];end=start+voice['duration_s']
  assert chapter['start_s']<=start<end<=chapter['start_s']+chapter['duration_s']
  vtt.extend([str(chapter['chapter']),f'{stamp(start)} --> {stamp(end)}',transcript,''])
  text.extend([f'{stamp(chapter["start_s"])} — {chapter["title"]}',transcript,''])
  cues.append({'chapter':chapter['chapter'],'start_s':start,'end_s':end,'text_sha256':hashlib.sha256(transcript.encode()).hexdigest(),'alignment':'whole_chapter_paragraph_exact_audio_boundary'})
 vtt_path=MEDIA/(PREFIX+'.vtt');txt_path=MEDIA/(PREFIX+'-transcript.txt')
 with vtt_path.open('x') as f:f.write('\n'.join(vtt))
 with txt_path.open('x') as f:f.write('\n'.join(text))
 source=RENDER/'captions-builder.py';shutil.copyfile(__file__,source)
 proof={'format':'narration-chapter-captions','version':1,'source_manifest':record(manifest_path,'exact_encoded_walkthrough_manifest'),'builder':record(source,'caption_derivation_source'),'webvtt':record(vtt_path,'optional_English_chapter_captions'),'transcript':record(txt_path,'plain_complete_exact_narration'),'cues':cues,'word_level_alignment':False,'narration_text_changed':False,'inference_calls':0,'api_calls':0}
 with (RENDER/'captions.json').open('x') as f:json.dump(proof,f,indent=2);f.write('\n')
 print(json.dumps({'status':'prepared','cues':len(cues),'vtt':str(vtt_path.relative_to(ROOT))}))
if __name__=='__main__':main()
