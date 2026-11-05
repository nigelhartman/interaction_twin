"""Prepare11 unique timestamp-TTS request bodies. Offline; never reads credentials."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def ref(p):
 b=p.read_bytes();return{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'output/elevenlabs-final-20261105-v1');a=ap.parse_args();out=a.output.resolve()
 if out.exists():raise FileExistsError('Fresh additive output required.')
 old=ROOT/'output/trailer-rework-20261105-v1/narration-story-v2';p45=old/'audio45-natural150-v1/audio.json';p180=old/'audio180-natural150-v2/audio.json';short=json.loads(p45.read_text());long=json.loads(p180.read_text());assert short['voice']==long['voice'][:3]
 out.mkdir(parents=True);(out/'requests').mkdir();(out/'responses').mkdir();settings={'stability':.5,'similarity_boost':.75,'style':0,'use_speaker_boost':True,'speed':1.0};cues=[]
 for i,r in enumerate(long['voice']):
  ident=f'cue-{i:02}';end=min(r['latest_end_s'],long['voice'][i+1]['start_s']-.15)if i+1<len(long['voice'])else r['latest_end_s'];body={'text':r['text'],'model_id':'eleven_multilingual_v2','voice_settings':settings,'apply_text_normalization':'auto'}
  if i:body['previous_text']=long['voice'][i-1]['text']
  if i+1<len(long['voice']):body['next_text']=long['voice'][i+1]['text']
  request=out/'requests'/f'{ident}.json';request.write_text(json.dumps(body,indent=2)+'\n');cues.append({'id':ident,'start_s':r['start_s'],'latest_end_s':end,'original_latest_end_s':r['latest_end_s'],'available_s':round(end-r['start_s'],3),'text':r['text'],'text_sha256':hashlib.sha256(r['text'].encode()).hexdigest(),'request':ref(request),'response_json':(out/'responses'/f'{ident}.json').relative_to(ROOT).as_posix(),'film_durations_s':[45,180]if i<3 else[180]})
 plan={'format':'interaction-twin-elevenlabs-timed-cue-plan','version':1,'voice_id':'cBTYF7NiJbR8Bvmdeuse','endpoint':'https://api.elevenlabs.io/v1/text-to-speech/cBTYF7NiJbR8Bvmdeuse/with-timestamps?output_format=mp3_44100_128','method':'POST','model_id':'eleven_multilingual_v2','voice_settings':settings,'unique_requests':len(cues),'total_text_characters':sum(len(c['text'])for c in cues),'first_three_cues_reused_byte_identically_between_films':True,'audio_45_source':ref(p45),'audio_180_source':ref(p180),'script_source':ref(old/'script-v4.json'),'films':[{'duration_s':45,'cue_ids':[c['id']for c in cues[:3]],'source_video':ref(ROOT/'showcase/media/flagship-20261105-v3/interaction-twin-flagship-45s.mp4')},{'duration_s':180,'cue_ids':[c['id']for c in cues],'source_video':ref(ROOT/'showcase/media/submission-20261105-v3/interaction-twin-3min.mp4')}],'cues':cues,'no_network_or_credentials_accessed':True,'new_API_calls':0,'publication':False,'official_docs':['https://elevenlabs.io/docs/api-reference/text-to-speech/convert-with-timestamps','https://elevenlabs.io/docs/eleven-creative/playground/text-to-speech','https://elevenlabs.io/docs/overview/models']};p=out/'cue-plan.json';p.write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({'plan':ref(p),'unique_requests':len(cues),'characters':plan['total_text_characters'],'first_cue_window_s':cues[0]['available_s']},indent=2))
if __name__=='__main__':main()
