"""Synthetic byte authority fixtures; source tracking truth is not simulated."""
import asyncio,base64,copy,hashlib,importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from curated_twin_catalog import CuratedTwinCatalog
from curated_motion_hypotheses_v1 import CuratedMotionHypotheses
from twin_catalog import TwinCatalog
import motion_hypothesis_contract_v1 as T
from tests import test_motion_hypothesis_contract_v1 as motion_fixtures
ROOT=Path(__file__).resolve().parents[1]

class MotionReaderTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve();self.document,self.files,_=motion_fixtures.fixture();self.assets={}
  for relative,raw in self.files.items():self.write(relative,raw)
  self.id=self.document['twin_id'];self.source=self.document['source_ref'];self.scene=self.document['baseline_scene_ref'];self.program=self.document['evidence_program_ref'];self.native=json.loads(self.files[self.scene['path']])['components'][0]['asset'];self.record={'id':self.id,'source':self.source,'interaction':self.program,'curation':{'status':'owner_approved'}};self.records=[self.record];self.pin();self.publish();self.catalog=CuratedTwinCatalog(self.root);self.reader=CuratedMotionHypotheses(self.catalog)
 def write(self,relative,value,admit=True):
  raw=value if isinstance(value,bytes)else json.dumps(value,sort_keys=True).encode();p=self.root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);r={'path':relative,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
  if admit:self.assets[relative]=r
  return r
 def pin(self):self.record['motion_hypothesis']=self.write('data/temporal-fixture/hypotheses.json',self.document)
 def publish(self):
  raw=json.dumps({'format':'interaction-twin-object-program-library','version':1,'records':self.records},separators=(',',':')).encode()+b'\n';sha=hashlib.sha256(raw).hexdigest();name='library-catalog-'+sha+'.json';self.write('showcase/public/'+name,raw,False);self.write('showcase/public/library-index.json',{'format':'interaction-twin-library-index','version':1,'catalog':{'path':name,'sha256':sha,'bytes':len(raw)}},False);self.write('showcase/public/library-catalog.json',raw,False);self.write('showcase/public/cases-index.json',{'curated':True,'cases':[{'id':r['id'],'interaction_path':r['interaction']['path'],'original_sha256':r['interaction']['sha256']}for r in self.records]},False);self.write('showcase/public/public-assets.json',{'format':'interaction-twin-public-assets','version':1,'assets':list(self.assets.values())},False);row={'source_sha256':self.source['sha256'],'manifest':self.scene}
  if self.record.get('motion_hypothesis'):row['motion_hypothesis']=self.record['motion_hypothesis']
  self.write('showcase/public/aligned-scenes.json',{'format':'sam3d-aligned-scene-index','version':1,'scenes':[row]},False)
 def listing(self):return self.reader.inspect(self.id)
 def test_exact_current_document_and_only_typed_artifacts_are_returned(self):
  result=self.listing();self.assertEqual(result['document'],self.document);self.assertEqual(result['status'],'source_conditioned_motion_hypotheses');self.assertFalse(result['new_inference']);self.assertTrue(result['source_unknowns_preserved']);self.assertFalse(result['native_geometry_modified']);self.assertFalse(result['physical_accuracy_verified'])
  for item in result['artifacts']:
   self.assertEqual(self.reader.artifact(self.id,item['artifact_id']),(self.root/item['path']).read_bytes());self.assertEqual(item['uri'],f'twin://{self.id}/motion-hypotheses/{item["artifact_id"]}')
  paths={i['path']for i in result['artifacts']};self.assertNotIn(self.document['catalog_ref']['path'],paths);self.assertNotIn(self.document['implementation_ref']['path'],paths)
 def test_untyped_runtime_or_provider_references_are_not_discovered(self):
  private=self.write('data/private/runtime.bin',b'PRIVATE-NOT-ROUTABLE',False);self.document['provider_reply']={'runtime':private};self.pin();self.publish();self.assertNotIn(private['path'],{r['path']for r in self.listing()['artifacts']})
 def test_private_owner_catalog_paths_unknown_twins_and_unlisted_ids_reject(self):
  with self.assertRaises(ValueError):CuratedMotionHypotheses(TwinCatalog(self.root))
  for twin in ['private-working','../../.secrets/','unknown']:
   with self.subTest(twin=twin),self.assertRaises(ValueError):self.reader.inspect(twin)
  for value in ['../../key','A'*64,'a'*63,self.native['path']]:
   with self.subTest(value=value),self.assertRaises(ValueError):self.reader.artifact(self.id,value)
 def test_stale_and_cross_twin_ids_fail(self):
  token=self.listing()['artifacts'][0]['artifact_id'];self.document['review_note']='Explicit changed descriptor';self.pin();self.publish()
  with self.assertRaisesRegex(ValueError,'stale'):self.reader.artifact(self.id,token)
  other=copy.deepcopy(self.record);other['id']='another-approved-source';other.pop('motion_hypothesis');self.records.append(other);self.publish()
  with self.assertRaises(ValueError):self.reader.artifact(other['id'],token)
 def test_changed_source_camera_original_anchor_unknown_id_and_child_matrix_reject(self):
  original=copy.deepcopy(self.document)
  for mutation in ['source','camera','anchor','unknown','child','claims']:
   self.document=copy.deepcopy(original);episode=self.document['groups'][0]['episodes'][0]
   if mutation=='source':self.document['source_ref']['sha256']='0'*64
   elif mutation=='camera':self.document['camera']['K'][0][0]+=1
   elif mutation=='anchor':episode['anchor_before']['camera_from_native_mesh'][0][3]+=.1
   elif mutation=='unknown':episode['original_unknown_indices']=[]
   elif mutation=='child':episode['predictions'][0]['component_poses']['lid'][0][3]+=.1
   else:self.document['physical_accuracy_verified']=True
   self.pin();self.publish()
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.listing()
 def test_declared_scene_and_current_program_must_be_identical(self):
  self.record['interaction']=self.write('data/temporal-fixture/another-program.json',{'unrelated':'program'});self.publish()
  with self.assertRaises(ValueError):self.listing()
 def test_changed_artifact_or_missing_admission_reject(self):
  item=next(r for r in self.listing()['artifacts']if r['path']==self.native['path']);p=self.root/item['path'];raw=p.read_bytes();p.write_bytes(raw+b'!')
  with self.assertRaises(ValueError):self.reader.artifact(self.id,item['artifact_id'])
  p.write_bytes(raw);self.assets.pop(item['path']);self.publish()
  with self.assertRaises(ValueError):self.reader.artifact(self.id,item['artifact_id'])
 def test_full_current_inventory_is_revalidated_after_resource_read(self):
  item=next(r for r in self.listing()['artifacts']if r['path']==self.native['path']);original=self.catalog._bytes;calls=[0]
  def race(r,*,maximum):
   raw=original(r,maximum=maximum)
   if r==self.native:
    calls[0]+=1
    if calls[0]==2:self.record.pop('motion_hypothesis');self.publish()
   return raw
  with patch.object(self.catalog,'_bytes',side_effect=race),self.assertRaises(ValueError):self.reader.artifact(self.id,item['artifact_id'])
 def test_config_requires_exact_explicit_data_selection(self):
  from curated_publication import source_motion_displays
  pins={self.source['sha256']:{'descriptor':self.record['motion_hypothesis']}};self.write('config/source-motion-displays.json',{'version':1,'recordings':pins},False);self.assertEqual(source_motion_displays(self.root),pins)
  for mutation in ['source','path','extra']:
   bad=copy.deepcopy(pins)
   if mutation=='source':bad={'UPPER-INVALID':next(iter(bad.values()))}
   elif mutation=='path':bad[self.source['sha256']]['descriptor']['path']='data/../.private/key'
   else:bad[self.source['sha256']]['discover_candidates']=True
   self.write('config/source-motion-displays.json',{'version':1,'recordings':bad},False)
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):source_motion_displays(self.root)
 def test_absent_selection_is_unavailable_without_private_fallback(self):
  self.record.pop('motion_hypothesis');self.publish();self.assertEqual(self.listing()['status'],'unavailable');self.assertEqual(self.listing()['artifacts'],[])
 def test_describe_exposes_only_exact_pinned_motion_and_its_unverified_scope(self):
  with patch.object(TwinCatalog,'describe',return_value={'id':self.id}):description=self.catalog.describe(self.id)
  pointer=description['motion_hypothesis'];self.assertEqual(pointer['artifact'],self.record['motion_hypothesis']);self.assertEqual(pointer['inspection_tool'],'get_motion_hypotheses');self.assertTrue(pointer['original_observations_preserved']);self.assertFalse(pointer['physical_accuracy_verified']);self.assertFalse(pointer['native_geometry_modified'])
 def test_reader_import_remains_stdlib_only(self):
  code='import sys;sys.path.insert(0,'+repr(str(ROOT))+');import curated_motion_hypotheses_v1;assert not any(x.split(".")[0] in {"numpy","cv2","torch","PIL","scipy","moge"} for x in sys.modules)';result=subprocess.run([sys.executable,'-I','-S','-c',code],capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr)
 def test_actual_MCP_readonly_tool_and_resource_keep_curated_boundary_in_private_mode(self):
  try:from mcp import Client
  except ImportError:self.skipTest('Modern SDK is installed in .venv-mcp')
  spec=importlib.util.spec_from_file_location('motion_reader_actual_sdk',ROOT/'scripts/serve_twin_mcp.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  async def run():
   with patch.object(module,'CuratedTwinCatalog',return_value=self.catalog):server=module.create_server(catalog=TwinCatalog(self.root),private_owner=True)
   async with Client(server,read_timeout_seconds=30)as client:
    tool=next(t for t in(await client.list_tools()).tools if t.name=='get_motion_hypotheses');self.assertTrue(tool.annotations.read_only_hint);self.assertFalse(tool.annotations.destructive_hint);result=await client.call_tool('get_motion_hypotheses',{'twin_id':self.id});self.assertFalse(result.is_error);listing=result.structured_content or json.loads(result.content[0].text);self.assertEqual(listing['document'],self.document);item=listing['artifacts'][0];resource=(await client.read_resource(item['uri'])).contents[0];self.assertEqual(base64.b64decode(resource.blob,validate=True),(self.root/item['path']).read_bytes());bad=await client.call_tool('get_motion_hypotheses',{'twin_id':'private-unapproved'});self.assertTrue(bad.is_error)
  asyncio.run(run())

class SavedMotionReaderTests(unittest.TestCase):
 def test_actual_canonical_motion_with_real_MCP_resources_in_exact_prospective_snapshot(self):
  try:from mcp import Client
  except ImportError:self.skipTest('Modern SDK is installed in .venv-mcp')
  path=ROOT/'data/temporal-motion-hypotheses/laptop-continuous-20261104-v1/hypotheses.json'
  if not path.is_file():self.skipTest('Saved source motion is excluded from source-only checkout')
  raw=path.read_bytes();pointer={'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)};document=json.loads(raw);original=CuratedTwinCatalog(ROOT);records,cases=copy.deepcopy(original._snapshot());allowed=dict(original._assets());registry,_=original._json('showcase/public/aligned-scenes.json');registry=copy.deepcopy(registry);record=next(r for r in records if r['id']==document['twin_id']);scene=next(r for r in registry['scenes']if r['source_sha256']==record['source']['sha256']);self.assertEqual(record['interaction'],document['evidence_program_ref']);self.assertEqual(scene['manifest'],document['baseline_scene_ref']);record['motion_hypothesis']=pointer;scene['motion_hypothesis']=pointer;validated=T.validate(document,lambda r,maximum=None:(ROOT/r['path']).read_bytes())
  for r in [pointer,*validated['artifacts']]:
   self.assertTrue(r['path']not in allowed or allowed[r['path']]==r);allowed[r['path']]=r
  class FencedCatalog(CuratedTwinCatalog):
   def _snapshot(self):return records,cases
   def _assets(self):return allowed
   def _json(self,relative,maximum=8*1024*1024):
    if relative=='showcase/public/aligned-scenes.json':return registry,json.dumps(registry).encode()
    return super()._json(relative,maximum=maximum)
  catalog=FencedCatalog(ROOT);reader=CuratedMotionHypotheses(catalog);listing=reader.inspect(record['id']);self.assertEqual(listing['document'],document);self.assertEqual(len(listing['artifacts']),len({r['path']for r in [pointer,*validated['artifacts']]}));self.assertFalse(any(r['path'].startswith(('models/','output/','.secrets/','showcase/'))for r in listing['artifacts']));spec=importlib.util.spec_from_file_location('motion_real_prospective_sdk',ROOT/'scripts/serve_twin_mcp.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  async def run():
   with patch.object(module,'CuratedTwinCatalog',return_value=catalog):server=module.create_server(catalog=catalog)
   async with Client(server,read_timeout_seconds=60)as client:
    result=await client.call_tool('get_motion_hypotheses',{'twin_id':record['id']});self.assertFalse(result.is_error);actual=result.structured_content or json.loads(result.content[0].text);self.assertEqual(actual['document'],document);self.assertEqual(actual['artifact'],pointer);self.assertEqual(actual['artifacts'],listing['artifacts'])
    for ref in [pointer,document['display_continuity_ref'],document['flow_proofs_ref']]:
     item=next(r for r in actual['artifacts']if r['path']==ref['path']);resource=(await client.read_resource(item['uri'])).contents[0];self.assertEqual(base64.b64decode(resource.blob,validate=True),(ROOT/ref['path']).read_bytes())
  asyncio.run(run())

if __name__=='__main__':unittest.main()
