#!/usr/bin/env python3
"""Make a fresh instrumental-only share cut; copy verified V3 video packets.

Reads the original procedural score only. No system voice, model/API request,
source geometry changes or publication. Narrated earlier cuts remain private.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,subprocess,wave
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PREFIX='interaction-library-trailer-20261102'
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def record(path,kind):
 p=Path(path);return {'path':p.relative_to(ROOT).as_posix(),'sha256':digest(p),'bytes':p.stat().st_size,'kind':kind}
def fresh(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def packets(path):
 return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_packets','-show_entries','packet=pts,dts,duration,size,data_hash','-show_data_hash','sha256','-of','json',str(path)],text=True))['packets']
def main():
 source_dir=ROOT/'output/trailer-20261102/v3';out=ROOT/'output/trailer-20261102/v4';out.mkdir(exist_ok=False)
 source_manifest=source_dir/'source-manifest.json';source=json.loads(source_manifest.read_text())
 verified=json.loads((source_dir/'verification.json').read_text())
 if verified.get('status')!='passed' or verified.get('decoded_frames')!=1392:raise ValueError('Verified V3 visual source required')
 source_video=ROOT/source['output']['path']
 if digest(source_video)!=source['output']['sha256']:raise ValueError('Private source video changed')
 rights=json.loads((source_dir/'audio-rights.json').read_text());score=ROOT/rights['score']['path']
 if digest(score)!=rights['score']['sha256'] or rights['composition']['seed']!=20261102:raise ValueError('Original procedural score binding differs')
 with wave.open(str(score),'rb') as w:
  if (w.getframerate(),w.getnchannels(),w.getsampwidth())!=(48000,2,2):raise ValueError('Expected original PCM score')
  samples=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').reshape(-1,2).astype(float)/32768.
 if len(samples)!=58*48000:raise ValueError('Original score duration differs')
 rms=float(np.sqrt(np.mean(samples*samples)));peak=float(np.abs(samples).max());gain=min(.10/rms,.75/peak)
 music=out/'original-instrumental-only.wav'
 with wave.open(str(music),'wb') as w:
  w.setnchannels(2);w.setsampwidth(2);w.setframerate(48000);w.writeframes(np.rint(samples*gain*32767).astype('<i2').tobytes())
 audio={'format':'interaction-twin-original-instrumental-share-audio','version':1,'original_score':record(score,'original_procedural_music_without_voice'),
        'composition':rights['composition'],'gain':gain,'target_rms_dbfs':-20,'system_voice_audio_included':False,'narration_included':False,
        'output':record(music,'music_only_PCM'),'third_party_music_samples':False,'rights_scope':'Original procedural score only; source footage publication permissions require separate review.'}
 fresh(out/'audio-rights.json',audio)
 dest=ROOT/'showcase/media'/(PREFIX+'-v4.mp4')
 if dest.exists():raise ValueError('Preserve previous share cut')
 command=['ffmpeg','-hide_banner','-v','error','-i',str(source_video),'-i',str(music),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart','-t','58',str(dest)]
 subprocess.run(command,check=True)
 before,after=packets(source_video),packets(dest)
 if before!=after or len(after)!=1392:raise ValueError('Video packet payload/timestamps changed')
 fresh(out/'video-packet-verification.json',{'format':'share-cut-video-packet-comparison','version':1,'status':'identical_video_packet_payloads_and_timestamps','packets':1392,
       'private_source_container_sha256':digest(source_video),'share_output_sha256':digest(dest),'packet_sequence_sha256':hashlib.sha256(json.dumps(after,sort_keys=True).encode()).hexdigest(),
       'source_audio_copied':False,'system_voice_audio_present':False})
 manifest={'format':'source-backed-interaction-library-instrumental-share-film','version':1,'output':record(dest,'instrumental_only_H264_AAC_share_cut'),
      'width':1920,'height':1080,'fps':24,'duration_s':58,'frames':1392,'chapters':source['chapters'],
      'visual_lineage':{'source_manifest':record(source_manifest,'historical_V3_visual_source_recipe'),'source_integrity':record(source_dir/'source-integrity.json','historical_V3_source_artifact_hash_audit'),
                        'video_packets':record(out/'video-packet-verification.json','exact_video_packet_equality_proof'),'private_source_audio_not_used':True},
      'audio':record(out/'audio-rights.json','original_instrumental_only_audio_provenance'),'builder':record(Path(__file__),'share_cut_builder'),
      'SAM3D_output_included':True,'first_earbud_only_in_film':True,'other_featured_models_and_robot_rehearsal_use_TripoSR':True,
      'system_voice_audio_present':False,'narration_present':False,'new_api_calls':0,'model_inference_calls':0,'published':False,'submitted':False,
      'limitations':['New native model remains unselected in an independent nonmetric gauge.','Raw same-frame projection uses an estimated camera; no silhouette/pose fitting or validated temporal tracking.',
                     'Robot rehearsal remains illustrative, using existing TripoSR drafts; physics/contact are unvalidated.','Source-footage permissions and final official submission requirements need review before publication.'],
      'created_at_utc':datetime.now(timezone.utc).isoformat()}
 fresh(out/'source-manifest.json',manifest)
 # Independently recheck all frozen ancestral evidence. A mutable historical
 # decoder path is resolved only by an exact, already recorded hash witness.
 ancestry=json.loads((source_dir/'source-integrity.json').read_text());rows={};witnesses={}
 known={'bb9a6ff7e03b1fc68b113cf5d53432d1686b458cfba5f1ad2fe4cf8e586bdcd2':'output/sam3d-large-native-20261102/historical/portable_twin-after-index-before-position-cap.py'}
 for name,expected in ancestry['files'].items():
  path=ROOT/name
  if digest(path)!=expected:
   if name!='portable_twin.py' or expected not in known:raise ValueError('Ancestral source artifact changed: '+name)
   witness=ROOT/known[expected]
   if digest(witness)!=expected:raise ValueError('Historical decoder witness differs')
   witnesses[name]={'declared_sha256':expected,'verified_bytes':record(witness,'exact_historical_code_hash_witness')};path=witness
  rows[path.relative_to(ROOT).as_posix()]=expected
 fresh(out/'source-integrity.json',{'status':'all_ancestral_artifacts_reverified_and_video_payloads_identical','ancestral_artifacts':len(ancestry['files']),
       'historical_code_witnesses':witnesses,'files':rows,'private_ancestral_voice_files_not_in_share_package':True,
       'output':record(dest,'music_only_share_film'),'audio_rights':record(out/'audio-rights.json','voice_free_audio_provenance'),
       'system_voice_audio_present':False,'published':False,'new_api_calls':0,'model_inference_calls':0})
 print(json.dumps({'status':'instrumental_share_cut_ready','path':str(dest),'duration_s':58,'video_packets':1392,'ancestral_artifacts_reverified':len(rows),'system_voice_audio_present':False}))
if __name__=='__main__':main()
