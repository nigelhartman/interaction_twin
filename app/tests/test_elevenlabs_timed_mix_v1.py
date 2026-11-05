"""Offline fixtures; no speech provider, credentials or actual generated responses."""
import importlib.util,sys,unittest,json,base64,tempfile
from unittest.mock import patch
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'scripts/mix_elevenlabs_final_20261105_v1.py';s=importlib.util.spec_from_file_location('timedmix',P);M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
def response(text):return{'alignment':{'characters':list(text),'character_start_times_seconds':[i*.05 for i in range(len(text))],'character_end_times_seconds':[(i+1)*.05 for i in range(len(text))]}}
class TimedMix(unittest.TestCase):
 def test_exact_text_alignment_required(self):
  self.assertEqual(M.alignment(response('Source-linked parts.'),'Source-linked parts.')['provider_field'],'alignment')
  with self.assertRaises(ValueError):M.alignment(response('Unsupported claim.'),'Source-linked parts.')
 def test_invalid_and_reversed_clock_rejected(self):
  for change in [lambda a:a['character_end_times_seconds'].pop(),lambda a:a['character_start_times_seconds'].__setitem__(3,-1),lambda a:a['character_end_times_seconds'].__setitem__(2,float('nan')),lambda a:a['character_start_times_seconds'].__setitem__(4,.01)]:
   r=response('Exact words.');change(r['alignment'])
   with self.assertRaises(ValueError):M.alignment(r,'Exact words.')
 def test_caption_clock_follows_trim_and_tempo(self):
  a=M.alignment(response('Keep the source evidence close.'),'Keep the source evidence close.');c=M.caption_groups(a,26,.025,1)
  self.assertEqual(c[0]['text'],'Keep the source evidence close.');self.assertAlmostEqual(c[0]['start_s'],25.975);self.assertAlmostEqual(c[-1]['end_s'],26+len(a['text'])*.05-.025)
 def test_caption_chunks_keep_all_words(self):
  text=' '.join(['source-linked']*30)+'.';a=M.alignment(response(text),text);c=M.caption_groups(a,1,0,1)
  self.assertEqual(' '.join(x['text']for x in c),text);self.assertTrue(all(len(x['text'])<=76 for x in c))
 def test_shared_tagline_uses_same_response_and_ends_on_endcard(self):
  text='Describe a new task before physical validation. Show an interaction once. Keep how it works.';r=response(text);r['audio_base64']=base64.b64encode(b'x'*200).decode()
  cue={'id':'cue-02','text':text,'start_s':26,'latest_end_s':44.5}
  def stub(c,a,mp3,response_path,out,max_tempo,ident=None,**kwargs):return{'id':ident,'start_s':c['start_s'],'latest_end_s':c['latest_end_s'],'text':a['text'],'source_response':str(response_path)}
  with tempfile.TemporaryDirectory()as d:
   folder=Path(d);p=folder/'response.json';p.write_text(json.dumps(r))
   with patch.object(M,'seconds',return_value=10),patch.object(M,'prepare_segment',side_effect=stub):rows=M.prepare_cue(cue,p,folder,1.15)
  self.assertEqual(rows[0]['start_s'],26);self.assertEqual(rows[1]['latest_end_s'],44.3);self.assertGreater(rows[1]['start_s'],40);self.assertEqual(rows[0]['source_response'],rows[1]['source_response']);self.assertEqual(' '.join(x['text']for x in rows),text)
 def test_caption_timestamp_rounding(self):self.assertEqual(M.timestamp(179.9996),'00:03:00.000');self.assertEqual(M.timestamp(1.234,','),'00:00:01,234')
if __name__=='__main__':unittest.main()
