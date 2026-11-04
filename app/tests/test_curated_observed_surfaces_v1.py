"""Read-only observed-surface transport fences; synthetic bytes do not certify3D."""
import asyncio
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from curated_observed_surfaces_v1 import CuratedObservedSurfaces
from curated_twin_catalog import CuratedTwinCatalog
from twin_catalog import TwinCatalog
from observed_surface_presentation_v1 import derive
import source_visible_flexure_contract_v1 as F

ROOT=Path(__file__).resolve().parents[1]

class ObservedReaderTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve();self.assets={};self.id='synthetic-source--procedure';self.source={'path':'data/fixture/video.mp4','sha256':'a'*64,'duration_s':2.2};self.camera={'K':[[2,0,1],[0,2,1],[0,0,1]],'width':2,'height':2,'calibrated':False};self.native=self.write('data/fixture/native.glb',b'glTF-synthetic-native');self.other=self.write('data/other/unrelated.glb',b'glTF-other-public-twin');evidence=[]
  for i in range(1,4):
   rgb=self.write(f'data/fixture/frame{i}.jpg',b'\xff\xd8synthetic'+bytes([i]));mask=self.write(f'data/fixture/mask{i}.png',b'\x89PNGsynthetic'+bytes([i]));evidence.append({'source_index':i,'timestamp_s':float(i-1),'frame_path':rgb['path'],'frame_sha256':rgb['sha256'],'observations':[{'track_id':'body','mask_path':mask['path'],'mask_sha256':mask['sha256']}]})
  self.program=self.write('data/fixture/program.json',{'source':self.source,'evidence':evidence});self.scene=self.write('data/fixture/scene.json',{'original_program':self.program,'source':self.source,'camera':self.camera,'components':[{'id':'body','asset':self.native,'poses':[{'source_sample_index':i,'status':'unknown'if i==2 else'estimated'}for i in range(1,4)]}]});protocol={'source_scene':self.scene,'source_program':self.program,'source_video_sha256':self.source['sha256'],'physical_accuracy_verified':False,'network_access_during_inference':False,'new_GPU_calls':0,'depth_model':{'device':'cpu','checkpoint_sha256':'f'*64}};self.protocol=self.write('data/fixture/observed/protocol.json',protocol);rows=[]
  for e in evidence:
   i=e['source_index'];r={'source_sample_index':i,'timestamp_s':e['timestamp_s'],'status':'unknown'}
   if i!=2:
    folder=f'data/fixture/observed/source-{i:03d}';r.update({'status':'source_observed_prior','asset':self.write(folder+'/surface.glb',b'glTF-source-surface'+bytes([i])),'source_pixel_map':self.write(folder+'/map.npz',b'NPZsynthetic'+bytes([i])),'source_RGB':self.assets[e['frame_path']],'source_masks':[{'track_id':'body',**self.assets[e['observations'][0]['mask_path']]}],'physical_accuracy_verified':False,'source_visible_only':True,'triangle_source_footprint_verified':True,'hidden_backs_culled':True,'unlit_exact_vertex_RGB':True,'vertices':3,'triangles':1,'camera_from_asset':[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]],'virtual_flexure':{'status':'unknown','angle_rad':None,'axis_camera':None,'origin_camera':None},'depth_model':{'checkpoint_sha256':'f'*64},'visible_component_ids':['body']});r['provenance']=self.write(folder+'/provenance.json',r)
   rows.append(r)
  self.document={'format':F.FORMAT,'version':1,'kind':'source_observed_monocular_depth_prior','source_program':self.program,'source_scene':self.scene,'source':self.source,'camera':self.camera,'protocol':self.protocol,'replacement_component_ids':['body'],'unaffected_component_ids':[],'closed_native_assets':[self.native],'policy':{'maximum_source_time_error_s':.18,'timing':'nearest_source_keyframe_within_0.18s_only_no_geometric_interpolation','hidden_surfaces':'absent_not_inpainted','numerical_fold_parameters':'unknown_not_identifiably_recovered'},'samples':rows,'physical_accuracy_verified':False,'simulation_ready':False,'manual_geometry_or_poses_used':False,'native_SAM_contract_compatible':False};self.record={'id':self.id,'interaction':self.program,'source':self.source,'curation':{'status':'owner_approved'}};self.records=[self.record];self.pin();self.publish();self.catalog=CuratedTwinCatalog(self.root);self.reader=CuratedObservedSurfaces(self.catalog)
 def write(self,path,value,admit=True):
  raw=value if isinstance(value,bytes)else json.dumps(value,sort_keys=True).encode();p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);r={'path':path,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
  if admit:self.assets[path]=r
  return r
 def pin(self):self.record['observed_surfaces']=self.write('data/fixture/observed/descriptor.json',self.document)
 def presentation(self,value=None):
  value=derive(self.document,self.record['observed_surfaces'])if value is None else value;self.record['observed_surface_presentation']=self.write('data/fixture/presentation.json',value);return value
 def publish(self):
  value={'format':'interaction-twin-object-program-library','version':1,'records':self.records};raw=json.dumps(value).encode();sha=hashlib.sha256(raw).hexdigest();name='library-catalog-'+sha+'.json';self.write('showcase/public/'+name,raw,False);self.write('showcase/public/library-index.json',{'format':'interaction-twin-library-index','version':1,'catalog':{'path':name,'sha256':sha,'bytes':len(raw)}},False);self.write('showcase/public/cases-index.json',{'curated':True,'cases':[{'id':r['id'],'interaction_path':r['interaction']['path'],'original_sha256':r['interaction']['sha256']}for r in self.records]},False);self.write('showcase/public/public-assets.json',{'format':'interaction-twin-public-assets','version':1,'assets':list(self.assets.values())},False);entry={'source_sha256':self.source['sha256'],'manifest':self.scene}
  if' observed_surfaces'.strip()in self.record:entry['observed_surfaces']=self.record['observed_surfaces']
  if self.record.get('observed_surface_presentation'):entry['observed_surface_presentation']=self.record['observed_surface_presentation']
  self.write('showcase/public/aligned-scenes.json',{'format':'sam3d-aligned-scene-index','version':1,'scenes':[entry]},False)
 def listing(self):return self.reader.inspect(self.id)
 def test_exact_resources_roles_and_no_geometry_or_inference_claim(self):
  listing=self.listing();self.assertEqual(listing,self.listing());self.assertEqual(listing['status'],'source_visible_depth_observations');self.assertFalse(listing['new_inference']);self.assertFalse(listing['native_kinematics_replaced']);self.assertFalse(listing['physical_accuracy_verified']);self.assertFalse(listing['simulation_ready'])
  for item in listing['artifacts']:
   self.assertEqual(self.reader.artifact(self.id,item['artifact_id']),(self.root/item['path']).read_bytes());self.assertEqual(item['uri'],f'twin://{self.id}/observed-surfaces/{item["artifact_id"]}')
  self.assertNotIn(self.other['path'],{r['path']for r in listing['artifacts']})
 def test_private_owner_or_arbitrary_twin_and_artifact_ids_reject(self):
  with self.assertRaises(ValueError):CuratedObservedSurfaces(TwinCatalog(self.root))
  for name in ['private-working--procedure','../../.secrets/key','unknown']:
   with self.subTest(name=name),self.assertRaises(ValueError):self.reader.inspect(name)
  for name in [self.native['path'],'A'*64,'a'*63,'../../private']:
   with self.subTest(name=name),self.assertRaises(ValueError):self.reader.artifact(self.id,name)
 def test_no_untyped_private_runtime_or_unrelated_inventory_expansion(self):
  private=self.write('data/private/runtime.bin',b'PRIVATE-NOT-ADMITTED',False);self.document['provider_reply']={'model_runtime_ref':private,'other_twin_ref':self.other};self.document['samples'][1]['source_RGB']=private;self.pin();self.publish();paths={r['path']for r in self.listing()['artifacts']};self.assertNotIn(private['path'],paths);self.assertNotIn(self.other['path'],paths);self.assertFalse(any(p.startswith('models/')or'.private/'in p or'.secrets/'in p for p in paths))
 def test_unrelated_even_public_native_asset_cannot_become_this_source_original(self):
  self.document['closed_native_assets']=[self.other];self.pin();self.publish()
  with self.assertRaisesRegex(ValueError,'Original surface assets|Original native|replacement'):self.listing()
 def test_publication_transport_itself_rejects_wrong_native_before_any_pointer_update(self):
  forged=copy.deepcopy(self.document);forged['closed_native_assets']=[self.other]
  with self.assertRaisesRegex(ValueError,'Original surface assets'):
   F.validate(forged,lambda r:self.catalog._bytes(r,maximum=16*1024*1024))
 def test_publication_config_requires_explicit_safe_descriptor_selection(self):
  from curated_publication import observed_surface_displays
  self.presentation();pins={self.source['sha256']:{'descriptor':self.record['observed_surfaces'],'presentation':self.record['observed_surface_presentation']}};self.write('config/observed-surface-displays.json',{'version':1,'recordings':pins},False);self.assertEqual(observed_surface_displays(self.root),pins)
  for mutation in ['source','path','extra']:
   bad=copy.deepcopy(pins)
   if mutation=='source':bad={'UPPER-INVALID-SOURCE':next(iter(bad.values()))}
   elif mutation=='path':bad[self.source['sha256']]['descriptor']['path']='data/../.private/key'
   else:bad[self.source['sha256']]['scan_candidates']=True
   self.write('config/observed-surface-displays.json',{'version':1,'recordings':bad},False)
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):observed_surface_displays(self.root)
 def test_cross_twin_token_is_not_reusable_even_for_identical_assets(self):
  first=self.listing()['artifacts'][0]['artifact_id'];second=copy.deepcopy(self.record);second['id']='second-curated--procedure';self.records.append(second);self.publish();other=self.reader.inspect(second['id']);self.assertNotEqual(first,other['artifacts'][0]['artifact_id'])
  with self.assertRaisesRegex(ValueError,'stale'):self.reader.artifact(second['id'],first)
 def test_stale_descriptor_ids_fail_after_approval_changes(self):
  token=self.listing()['artifacts'][0]['artifact_id'];self.document['owner_note']='new approved descriptive revision';self.pin();self.publish()
  with self.assertRaisesRegex(ValueError,'stale'):self.reader.artifact(self.id,token)
 def test_source_camera_original_unknown_or_asset_on_unknown_forgeries_reject(self):
  baseline=copy.deepcopy(self.document)
  for mutation in ['source','camera','unknown_asset','unknown_promoted']:
   self.document=copy.deepcopy(baseline)
   if mutation=='source':self.document['source']['sha256']='b'*64
   elif mutation=='camera':self.document['camera']['K'][0][0]+=1
   elif mutation=='unknown_asset':self.document['samples'][1]['asset']=self.document['samples'][0]['asset']
   else:self.document['samples'][1]=copy.deepcopy(self.document['samples'][0]);self.document['samples'][1].update(source_sample_index=2,timestamp_s=1.)
   self.pin();self.publish()
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.listing()
 def test_changed_bytes_or_removed_admission_reject_after_listing(self):
  item=next(x for x in self.listing()['artifacts']if'source_RGB'in x['roles']);p=self.root/item['path'];raw=p.read_bytes();p.write_bytes(raw[:-1]+b'!')
  with self.assertRaises(ValueError):self.reader.artifact(self.id,item['artifact_id'])
  p.write_bytes(raw);self.assets.pop(item['path']);self.publish()
  with self.assertRaises(ValueError):self.reader.artifact(self.id,item['artifact_id'])
 def test_optional_editorial_presentation_is_separate_exact_resource(self):
  policy=self.presentation();self.publish();listing=self.listing();self.assertEqual(listing['presentation']['document'],policy);self.assertEqual(listing['document']['policy']['maximum_source_time_error_s'],.18);item=next(i for i in listing['artifacts']if'editorial_source_sample_presentation'in i['roles']);self.assertEqual(json.loads(self.reader.artifact(self.id,item['artifact_id'])),policy)
 def test_forged_or_cross_descriptor_presentation_fails(self):
  policy=self.presentation();self.publish()
  for mutation in ['cap','source','unknown']:
   bad=copy.deepcopy(policy)
   if mutation=='cap':bad['display_repeat_cap_s']=.6
   elif mutation=='source':bad['source_observations']['sha256']='0'*64
   else:bad['samples'][1]['status']='source_observed_prior'
   self.presentation(bad);self.publish()
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.listing()
 def test_current_source_or_presentation_switch_during_read_fails_closed(self):
  self.presentation();self.publish();item=next(i for i in self.listing()['artifacts']if i['path']==self.native['path']);original=self.catalog._bytes
  for mutation in ['source','presentation']:
   fired=[False];original_record=copy.deepcopy(self.record)
   def race(r,*,maximum):
    raw=original(r,maximum=maximum)
    if r==self.native and not fired[0]:
     fired[0]=True
     if mutation=='source':self.record['interaction']=self.write('data/fixture/other-program.json',{'different':'program'})
     else:self.record.pop('observed_surface_presentation')
     self.publish()
    return raw
   with self.subTest(mutation=mutation),patch.object(self.catalog,'_bytes',side_effect=race),self.assertRaises(ValueError):self.reader.artifact(self.id,item['artifact_id'])
   self.record.clear();self.record.update(original_record);self.publish()
 def test_unavailable_is_explicit_without_native_or_private_fallback(self):
  self.record.pop('observed_surfaces');self.publish();listing=self.listing();self.assertEqual(listing['status'],'unavailable');self.assertEqual(listing['artifacts'],[])
 def test_module_remains_lightweight_without_cv_or_model_imports(self):
  code='import sys;sys.path.insert(0,'+repr(str(ROOT))+');import curated_observed_surfaces_v1;assert not any(n.split(".")[0] in {"numpy","cv2","PIL","scipy","torch","moge"} for n in sys.modules)';r=subprocess.run([sys.executable,'-I','-S','-c',code],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
 def test_actual_MCP_tool_and_resource_are_readonly_curated_even_private_mode(self):
  try:from mcp import Client
  except ImportError:self.skipTest('Modern SDK is available in .venv-mcp')
  spec=importlib.util.spec_from_file_location('observed_reader_test_server',ROOT/'scripts/serve_twin_mcp.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  async def exercise():
   with patch.object(module,'CuratedTwinCatalog',return_value=self.catalog):server=module.create_server(catalog=TwinCatalog(self.root),private_owner=True)
   async with Client(server,read_timeout_seconds=30)as client:
    tool=next(t for t in(await client.list_tools()).tools if t.name=='get_observed_surfaces');self.assertTrue(tool.annotations.read_only_hint);self.assertFalse(tool.annotations.destructive_hint);result=await client.call_tool('get_observed_surfaces',{'twin_id':self.id});self.assertFalse(result.is_error);listing=result.structured_content or json.loads(result.content[0].text);self.assertEqual(listing['access'],'owner_curated_public_snapshot');item=next(i for i in listing['artifacts']if i['path']==self.native['path']);resource=(await client.read_resource(item['uri'])).contents[0];self.assertEqual(base64.b64decode(resource.blob,validate=True),(self.root/self.native['path']).read_bytes());bad=await client.call_tool('get_observed_surfaces',{'twin_id':'private-unapproved'});self.assertTrue(bad.is_error)
  asyncio.run(exercise())

class SavedObservedReaderTests(unittest.TestCase):
 def test_real_phone_camera_candidates_in_explicit_fenced_approval(self):
  for name in ['phone','camera']:
   path=ROOT/f'data/source-visible-flexure/{name}-20261104-v2/descriptor.json'
   if not path.exists():self.skipTest('Exact source candidates absent in portable source-only checkout')
   raw=path.read_bytes();pointer={'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)};validated=F.validate_descriptor(pointer);document=validated['document'];allowed={r['path']:r for r in validated['artifacts']}
   for r in document['closed_native_assets']:allowed[r['path']]=r
   presentation=ROOT/f'data/source-visible-presentations/{name}-20261104-v1/presentation.json';b=presentation.read_bytes();presentation_ref={'path':presentation.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)};allowed[presentation_ref['path']]=presentation_ref;record={'id':name+'-fenced--procedure','interaction':document['source_program'],'source':document['source'],'observed_surfaces':pointer,'observed_surface_presentation':presentation_ref};case={'id':record['id'],'interaction_path':record['interaction']['path'],'original_sha256':record['interaction']['sha256']};registry={'scenes':[{'source_sha256':document['source']['sha256'],'manifest':document['source_scene'],'observed_surfaces':pointer,'observed_surface_presentation':presentation_ref}]}
   class FencedCatalog(CuratedTwinCatalog):
    def _snapshot(self):return [record],[case]
    def _assets(self):return allowed
    def _json(self,relative,maximum=8*1024*1024):
     if relative=='showcase/public/aligned-scenes.json':return registry,json.dumps(registry).encode()
     return super()._json(relative,maximum)
   reader=CuratedObservedSurfaces(FencedCatalog(ROOT));listing=reader.inspect(record['id']);self.assertEqual(listing['document'],document);self.assertEqual({a['path']for a in listing['artifacts']},set(allowed));self.assertFalse(any(p.startswith('models/')or'.private/'in p or'.secrets/'in p for p in allowed));wanted={'descriptive_surface_program','source_RGB','asset','source_pixel_map','protocol','editorial_source_sample_presentation'}
   for role in wanted:
    item=next(a for a in listing['artifacts']if role in a['roles']);self.assertEqual(reader.artifact(record['id'],item['artifact_id']),(ROOT/item['path']).read_bytes())

if __name__=='__main__':unittest.main()
