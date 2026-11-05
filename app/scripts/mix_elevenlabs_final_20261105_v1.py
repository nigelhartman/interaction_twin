"""Offline timed ElevenLabs response intake, captions, ducked music, video-copy remux.
No credentials, network, TTS calls or source mutation. Fresh output only.
"""
from pathlib import Path
import argparse,base64,hashlib,json,math,re,subprocess
ROOT=Path(__file__).resolve().parents[1]
def run(args):return subprocess.run([str(x)for x in args],check=True,capture_output=True,text=True)
def ref(p):
 p=Path(p).resolve();b=p.read_bytes()
 try:name=p.relative_to(ROOT).as_posix()
 except ValueError:name=str(p)
 return{'path':name,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def checked(r):
 p=ROOT/r['path'];assert ref(p)==r,('Exact source changed',r['path']);return p
def probe(p):return json.loads(run(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type,width,height,r_frame_rate,nb_frames','-of','json',p]).stdout)
def seconds(p):return float(probe(p)['format']['duration'])
def loudness(p,I=-16,TP=-2.5,LRA=9):
 r=run(['ffmpeg','-hide_banner','-nostdin','-i',p,'-af',f'loudnorm=I={I}:TP={TP}:LRA={LRA}:print_format=json','-f','null','-']);matches=re.findall(r'\{\s*"input_i".*?\}',r.stderr,re.S)
 if not matches:raise ValueError('Loudness measurement missing.')
 d=json.loads(matches[-1]);assert all(math.isfinite(float(d[k]))for k in['input_i','input_tp','input_lra','input_thresh','target_offset']),d;return d

def normalize(source,target,I=-16,TP=-2.5,LRA=9):
 d=loudness(source,I,TP,LRA);f=f'loudnorm=I={I}:TP={TP}:LRA={LRA}:measured_I={d["input_i"]}:measured_TP={d["input_tp"]}:measured_LRA={d["input_lra"]}:measured_thresh={d["input_thresh"]}:offset={d["target_offset"]}:linear=true:print_format=summary';run(['ffmpeg','-hide_banner','-nostdin','-y','-i',source,'-af',f,'-ar','48000','-ac','2','-c:a','pcm_s16le',target]);return d

def alignment(response,text):
 for key in ['alignment','normalized_alignment']:
  d=response.get(key)
  if not isinstance(d,dict):continue
  chars=d.get('characters');starts=d.get('character_start_times_seconds');ends=d.get('character_end_times_seconds')
  if not isinstance(chars,list)or not isinstance(starts,list)or not isinstance(ends,list)or not 1<=len(chars)==len(starts)==len(ends)<=10000:continue
  if not all(isinstance(x,str)and 0<len(x)<=8 for x in chars):continue
  joined=''.join(chars)
  if re.sub(r'\s+',' ',joined).strip()!=re.sub(r'\s+',' ',text).strip():continue
  if not all(isinstance(a,(int,float))and isinstance(b,(int,float))and math.isfinite(a)and math.isfinite(b)and 0<=a<=b<=120 for a,b in zip(starts,ends)):continue
  if any(b<a-.02 for a,b in zip(starts,starts[1:])):continue
  if any(len(c)!=1 for c in chars):raise ValueError('Caption intake expects documented character-level alignment.')
  return{'provider_field':key,'text':joined,'starts':starts,'ends':ends}
 raise ValueError('Exact original-text character alignment required; no guessed captions.')

def caption_groups(a,offset,trim,tempo):
 words=[]
 for m in re.finditer(r'\S+',a['text']):
  ix=range(m.start(),m.end());words.append({'text':m.group(),'start_s':offset+(min(a['starts'][i]for i in ix)-trim)/tempo,'end_s':offset+(max(a['ends'][i]for i in ix)-trim)/tempo})
 groups=[];group=[]
 for w in words:
  if group and(len(' '.join(x['text']for x in group+[w]))>76 or len(group)>=12 or w['end_s']-group[0]['start_s']>4.2):groups.append(group);group=[]
  group.append(w)
  if re.search(r'[.!?]$',w['text'])and len(group)>=4:groups.append(group);group=[]
 if group:groups.append(group)
 return[{'start_s':g[0]['start_s'],'end_s':g[-1]['end_s'],'text':' '.join(w['text']for w in g)}for g in groups]

def slice_alignment(a,lo,hi):
 while lo<hi and a['text'][lo].isspace():lo+=1
 while hi>lo and a['text'][hi-1].isspace():hi-=1
 return{'provider_field':a['provider_field'],'text':a['text'][lo:hi],'starts':a['starts'][lo:hi],'ends':a['ends'][lo:hi]}

def prepare_segment(c,a,mp3,response_path,out,max_tempo,ident=None,source_start=None,source_end=None):
 ident=ident or c['id'];duration=seconds(mp3);spoken=[i for i,ch in enumerate(a['text'])if not ch.isspace()];start=max(0,min(a['starts'][i]for i in spoken)-.025)if source_start is None else source_start;end=min(duration,max(a['ends'][i]for i in spoken)+.065)if source_end is None else source_end
 if max(a['ends'])>duration+.12 or not 0<=start<end<=duration:raise ValueError('Audio/alignment duration differs: '+ident)
 available=c['latest_end_s']-c['start_s'];tempo=1 if end-start<=available+1e-6 else(end-start+.01)/available
 if tempo>max_tempo+1e-9:raise ValueError(f'{ident} requires {end-start:.3f}s, window {available:.3f}s; exceeds authorized local tempo cap{max_tempo}. No words truncated.')
 trimmed=out/(ident+'-trim.wav');processed=out/(ident+'.wav');f=f'atrim=start={start:.9f}:end={end:.9f},asetpts=PTS-STARTPTS,highpass=f=70'
 if tempo>1:f+=f',atempo={tempo:.10f}'
 f+=',afade=t=in:d=0.008,afade=t=out:st='+str(max(0,(end-start)/tempo-.008))+':d=0.008';run(['ffmpeg','-hide_banner','-nostdin','-y','-i',mp3,'-af',f,'-ar','48000','-ac','2','-c:a','pcm_s16le',trimmed]);measurement=normalize(trimmed,processed,I=-16,TP=-3,LRA=7);actual=seconds(processed)
 if c['start_s']+actual>c['latest_end_s']+.025:raise ValueError('Processed speech exceeds cue window: '+ident)
 captions=caption_groups(a,c['start_s'],start,tempo)
 return{'id':ident,'source_cue_id':c['id'],'text':a['text'],'start_s':c['start_s'],'end_s':c['start_s']+actual,'latest_end_s':c['latest_end_s'],'response_ref':ref(response_path),'raw_audio':ref(mp3),'processed':ref(processed),'provider_alignment_field':a['provider_field'],'speech_trim_s':[start,end],'atempo':tempo,'captions':captions,'normalization_input':measurement}

def prepare_cue(c,response_path,out,max_tempo):
 if not response_path.is_file()or response_path.stat().st_size>40*1024*1024:raise ValueError('Bounded completed timestamp JSON required: '+str(response_path))
 d=json.loads(response_path.read_text());raw=base64.b64decode(d['audio_base64'],validate=True)
 if not 100<len(raw)<=20*1024*1024:raise ValueError('Audio payload outside bounds.')
 a=alignment(d,c['text']);mp3=out/(c['id']+'.mp3');mp3.write_bytes(raw)
 if c['id']!='cue-02':return[prepare_segment(c,a,mp3,response_path,out,max_tempo)]
 phrase='Show an interaction once.';split=a['text'].find(phrase)
 if split<1:raise ValueError('Exact closing tagline missing from shared cue02.')
 first=slice_alignment(a,0,split);last=slice_alignment(a,split,len(a['text']));prior_end=max(first['ends']);next_start=min(last['starts']);boundary=(prior_end+next_start)/2
 if next_start<prior_end-.015:raise ValueError('Sentence timing overlaps; cannot safely split cue02 without words crossing.')
 tail_start=max(boundary,next_start-.025);tail_end=min(seconds(mp3),max(last['ends'])+.065);tail_duration=tail_end-tail_start;closing_end=44.3;closing_start=closing_end-tail_duration;head_end=min(boundary,prior_end+.065);head=dict(c,latest_end_s=closing_start-.3);tail=dict(c,start_s=closing_start,latest_end_s=closing_end)
 if head['latest_end_s']<=c['start_s']or closing_start<26:raise ValueError('Closing tagline leaves no explanation window.')
 rows=[prepare_segment(head,first,mp3,response_path,out,max_tempo,ident='cue-02-explanation',source_end=head_end),prepare_segment(tail,last,mp3,response_path,out,max_tempo,ident='cue-02-closing',source_start=tail_start,source_end=tail_end)]
 for r in rows:r['timing_override']='Shared cue02 split only at aligned sentence boundary; final tagline ends44.3s in both films; no new TTS call.'
 return rows

def timestamp(t,separator='.'):
 ms=round(max(0,t)*1000);h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);s,ms=divmod(ms,1000);return f'{h:02}:{m:02}:{s:02}{separator}{ms:03}'
def write_captions(rows,out,duration):
 groups=[g for r in rows for g in r['captions']];last=-1
 for g in groups:
  if not 0<=g['start_s']<g['end_s']<=duration or g['start_s']<last-.02:raise ValueError('Caption overlap/outside film.')
  last=g['end_s']
 (out/'captions.vtt').write_text('WEBVTT\n\n'+'\n'.join(f'{i+1}\n{timestamp(g["start_s"])} --> {timestamp(g["end_s"])}\n{g["text"]}\n'for i,g in enumerate(groups)))
 (out/'captions.srt').write_text('\n'.join(f'{i+1}\n{timestamp(g["start_s"],",")} --> {timestamp(g["end_s"],",")}\n{g["text"]}\n'for i,g in enumerate(groups)))
 (out/'transcript.txt').write_text('\n\n'.join(r['text']for r in rows)+'\n');return groups

def packets(p):return json.loads(run(['ffprobe','-v','error','-select_streams','v:0','-show_packets','-show_data_hash','sha256','-show_entries','packet=pts_time,dts_time,duration_time,size,flags,data_hash','-of','json',p]).stdout)['packets']
def build_film(film,prepared,music,out):
 duration=film['duration_s'];rows=[r for i in film['cue_ids']for r in prepared[i]];folder=out/f'film-{duration}';folder.mkdir();args=['ffmpeg','-hide_banner','-nostdin','-y','-f','lavfi','-t',duration,'-i','anullsrc=r=48000:cl=stereo'];parts=[];labels=['[0:a]']
 for i,r in enumerate(rows,1):args+=['-i',ROOT/r['processed']['path']];parts.append(f'[{i}:a]adelay={round(r["start_s"]*1000)}:all=1[c{i}]');labels.append(f'[c{i}]')
 parts.append(''.join(labels)+f'amix=inputs={len(labels)}:normalize=0:duration=longest:dropout_transition=0,atrim=duration={duration}[voice]');graph=folder/'voice-filter.txt';graph.write_text(';\n'.join(parts));voice=folder/'voice.wav';run(args+['-filter_complex_script',graph,'-map','[voice]','-ar','48000','-ac','2','-c:a','pcm_s16le',voice])
 music_duration=seconds(music);music_tempo=min(1,music_duration/duration)
 if music_tempo<.95:raise ValueError('Music is too short for a <=5%tempo fit; select a longer track or prepare an explicit loop.')
 bed_raw=folder/'music-raw.wav';f=f'atempo={music_tempo:.10f},apad,atrim=duration={duration},asetpts=PTS-STARTPTS,afade=t=in:d=0.65,afade=t=out:st={duration-1.5}:d=1.5';run(['ffmpeg','-hide_banner','-nostdin','-y','-i',music,'-af',f,'-ar','48000','-ac','2','-c:a','pcm_s16le',bed_raw]);bed=folder/'music-normalized.wav';normalize(bed_raw,bed,I=-22,TP=-5,LRA=9)
 raw_mix=folder/'mix-pre-normalization.wav';duck='[0:a]asplit[main][key];[1:a]volume=0.42[bed];[bed][key]sidechaincompress=threshold=0.02:ratio=6:attack=12:release=350[duck];[main][duck]amix=inputs=2:normalize=0:duration=longest:dropout_transition=0,atrim=duration='+str(duration)+'[mix]';run(['ffmpeg','-hide_banner','-nostdin','-y','-i',voice,'-i',bed,'-filter_complex',duck,'-map','[mix]','-ar','48000','-ac','2','-c:a','pcm_s16le',raw_mix]);mix=folder/'mix.wav';normalization=normalize(raw_mix,mix,I=-16,TP=-2.5,LRA=9)
 video=checked(film['source_video']);movie=folder/('interaction-twin-45s.mp4'if duration==45 else'interaction-twin-3min.mp4');run(['ffmpeg','-hide_banner','-nostdin','-y','-i',video,'-i',mix,'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k','-ar','48000','-ac','2','-t',duration,'-movflags','+faststart',movie]);old_packets,new_packets=packets(video),packets(movie)
 if old_packets!=new_packets:raise ValueError('Encoded video packet/timestamp identity failed; do not publish.')
 p=probe(movie);assert abs(float(p['format']['duration'])-duration)<.03;qa=loudness(movie)
 if abs(float(qa['input_i'])+16)>1 or float(qa['input_tp'])> -1.5:raise ValueError('Final encoded loudness/true-peak gate failed: '+str(qa))
 captions=write_captions(rows,folder,duration);receipt={'format':'interaction-twin-elevenlabs-final-audio-remux','version':1,'duration_s':duration,'source_video':film['source_video'],'movie':ref(movie),'video_packets_unchanged':True,'video_packets':len(new_packets),'video_reencoded':False,'cue_ids':film['cue_ids'],'cues':rows,'music_input':ref(music),'music_only_fit_tempo':music_tempo,'music_base_volume':.42,'music_ducking':{'threshold':.02,'ratio':6,'attack_ms':12,'release_ms':350},'mix':ref(mix),'normalization_input':normalization,'encoded_audio_measurement':qa,'captions':ref(folder/'captions.vtt'),'caption_count':len(captions),'transcript':ref(folder/'transcript.txt'),'new_API_calls_by_this_offline_helper':0,'publication':False};(folder/'audio-remux.json').write_text(json.dumps(receipt,indent=2)+'\n');return{'movie':ref(movie),'receipt':ref(folder/'audio-remux.json'),'captions':ref(folder/'captions.vtt')}

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--music',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--film',choices=['45','180','both'],default='both');ap.add_argument('--responses',type=Path);ap.add_argument('--max-tempo',type=float,default=1.15);a=ap.parse_args()
 if not 1<=a.max_tempo<=1.15:raise ValueError('Authorized local post-tempo cap is1.00–1.15; used only for overflow.')
 if a.output.exists():raise FileExistsError('Fresh additive output required.')
 plan=json.loads(a.plan.read_text());assert plan['format']=='interaction-twin-elevenlabs-timed-cue-plan'and plan['version']==1 and len(plan['cues'])==11;films=[f for f in plan['films']if a.film=='both'or str(f['duration_s'])==a.film];wanted={x for f in films for x in f['cue_ids']};a.output.mkdir(parents=True);cueout=a.output/'cues';cueout.mkdir();prepared={}
 for c in plan['cues']:
  if c['id']not in wanted:continue
  checked(c['request']);response=a.responses/(c['id']+'.json')if a.responses else ROOT/c['response_json'];prepared[c['id']]=prepare_cue(c,response,cueout,a.max_tempo)
 deliverables=[build_film(f,prepared,a.music.resolve(),a.output.resolve())for f in films];report={'format':'interaction-twin-elevenlabs-offline-final-delivery','version':1,'plan':ref(a.plan),'voice_id':plan['voice_id'],'model_id':plan['model_id'],'voice_settings':plan['voice_settings'],'unique_raw_responses':len(prepared),'first_three_exact_raw_and_processed_takes_shared_between_films':a.film=='both','post_tempo_authorized_cap':a.max_tempo,'all_narration_atempo_1':all(r['atempo']==1 for rows in prepared.values()for r in rows),'processed_segments':sum(len(rows)for rows in prepared.values()),'shared_tagline_alignment_retimed_to_end44_3s':True,'deliverables':deliverables,'builder':ref(Path(__file__)),'network_or_credentials_accessed':False,'new_API_calls':0,'publication':False};(a.output/'delivery.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
