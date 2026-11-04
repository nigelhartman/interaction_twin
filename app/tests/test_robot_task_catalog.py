"""Robot transport/packaging tests; canonical task semantics have separate tests."""
import copy
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import hashlib
from email.message import Message
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import public_robotics_api as API
import robot_task_catalog as K
import robot_task_program as R
from robot_starter import preview_server as S
from tests import test_curated_twin_catalog as fixtures

class RobotTransportTests(unittest.TestCase):
    def setUp(self):
        f=fixtures.CuratedTwinCatalogTests(methodName='runTest');f.setUp();self.addCleanup(f.doCleanups);self.f=f;self.root=f.root
        self.transport=K.RobotTaskCatalog(self.root)
        for name,_ in K.FIXED_FILES.values():
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes((K.ROOT/name).read_bytes())
        self.program={'status':'ready','kit_id':R.KIT_ID,'twin_id':f.id,'claims':{'authored_rehearsal':True}}
        self.refs=[f.record['interaction'],f.ref(f.doc['assets'][0]['path'])]
        # Test adapter exact byte gating independently of canonical planner rules.
        self.patches=[patch.object(R,'validate_robot_task',return_value={'valid':True}),patch.object(R,'task_bindings',side_effect=lambda p:copy.deepcopy(self.refs))]
        for p in self.patches:p.start();self.addCleanup(p.stop)

    def test_deterministic_bundle_bytes_hash_inventory_and_original_asset(self):
        a=self.transport.export_bundle(self.program);b=self.transport.export_bundle(self.program)
        self.assertEqual(a,b);raw=self.transport.bundle_artifact(a['bundle_id']);self.assertEqual(hashlib.sha256(raw).hexdigest(),a['bundle_id'])
        with zipfile.ZipFile(BytesIO(raw))as z:
            names=z.namelist();self.assertFalse(any('.private'in p or '.secrets'in p for p in names))
            prefix='Interaction-Twin-Robot-Starter/'
            self.assertEqual(z.read(prefix+self.refs[1]['path']),b'glTFcontract-test')
            self.assertEqual(json.loads(z.read(prefix+'task-program.json')),self.program)
            self.assertEqual(json.loads(z.read(prefix+'robot-task.schema.json')),R.task_schema())
            envelope=json.loads(z.read(prefix+'kit-response.json'))
            self.assertEqual(envelope['resource_url_scope'],'packaged_manifest_files');self.assertFalse(envelope['mcp_resources_available'])
            for descriptor in envelope['resources'].values():
                self.assertTrue(descriptor['uri'].startswith('twin://robot-kits/'))
                self.assertFalse(descriptor['url'].startswith('/api/'))
                body=z.read(prefix+descriptor['url'].removeprefix('/'))
                self.assertEqual(len(body),descriptor['bytes']);self.assertEqual(hashlib.sha256(body).hexdigest(),descriptor['sha256'])
            manifest=json.loads(z.read(prefix+'bundle-manifest.json'))
            for row in manifest['files']:
                body=z.read(prefix+row['path']);self.assertEqual(len(body),row['bytes']);self.assertEqual(hashlib.sha256(body).hexdigest(),row['sha256'])
            self.assertFalse(manifest['hardware_execution_authorized']);self.assertFalse(manifest['private_data_included'])
        live=self.transport.get_kit(self.program['kit_id'])
        self.assertTrue(all(d['url'].startswith('/api/robotics/')for d in live['resources'].values()))
        self.assertNotIn('mcp_resources_available',live)

    def test_rechecks_current_curated_bytes_and_private_refs(self):
        artifact=self.transport.export_bundle(self.program)
        path=self.root/self.refs[1]['path'];path.write_bytes(b'glTFchanged-byte')
        with self.assertRaises(ValueError):self.transport.bundle_artifact(artifact['bundle_id'])
        self.refs=[self.f.file('data/private/never-approved.glb',b'glTFprivate')]
        with self.assertRaisesRegex(ValueError,'not published'):self.transport.export_bundle(self.program)

    def test_changed_runtime_does_not_return_under_old_descriptor(self):
        a=self.transport.export_bundle(self.program)
        (self.root/K.FIXED_FILES['controller'][0]).write_bytes(b'// updated original source\n')
        with self.assertRaisesRegex(ValueError,'dependencies changed'):self.transport.bundle_artifact(a['bundle_id'])
        b=self.transport.export_bundle(self.program);self.assertNotEqual(a['bundle_id'],b['bundle_id'])

    def test_unknownkit_path_symlink_invalid_and_blocked_task_fail(self):
        for kit,asset in [('unknown','kit'),(R.KIT_ID,'../../.secrets/meta.env')]:
            with self.subTest(kit=kit,asset=asset),self.assertRaises(ValueError):self.transport.kit_artifact(kit,asset)
        path=self.root/K.FIXED_FILES['controller'][0];path.unlink();path.symlink_to(self.root/self.refs[1]['path'])
        with self.assertRaises(ValueError):self.transport.export_bundle(self.program)
        self.assertIn('controller',self.transport.get_kit()['missing_starter_files'])
        with self.assertRaises(ValueError):self.transport.export_bundle({**self.program,'status':'blocked'})
        with patch.object(R,'validate_robot_task',return_value={'valid':False}),self.assertRaises(ValueError):self.transport.export_bundle(self.program)
        with self.assertRaises(ValueError):K.encoded({'x':float('nan')})
        recursive={};recursive['self']=recursive
        with self.assertRaises(ValueError):K.encoded(recursive)

    def test_parallel_exports_serialize_bounded_package_builds_without_deadlock(self):
        original=self.transport._build_bundle;state={'active':0,'maximum':0};guard=threading.Lock()
        def measured(program):
            with guard:state['active']+=1;state['maximum']=max(state['maximum'],state['active'])
            try:time.sleep(.01);return original(program)
            finally:
                with guard:state['active']-=1
        with patch.object(self.transport,'_build_bundle',side_effect=measured),ThreadPoolExecutor(max_workers=3)as executor:
            values=list(executor.map(lambda _:self.transport.export_bundle(self.program),range(3)))
        self.assertEqual(state['maximum'],1);self.assertEqual(len({v['bundle_id']for v in values}),1)

    def test_cache_is_bounded_and_unknown_bundle_ids_do_not_read_files(self):
        for i in range(10):self.transport.export_bundle({**self.program,'test_instance':i})
        self.assertEqual(len(self.transport._bundles),8)
        for value in ['../secret','f'*64]:
            with self.subTest(value=value),self.assertRaises(ValueError):self.transport.bundle_artifact(value)

    def test_standalone_inventory_exact_bytes_and_no_unlisted_private_file(self):
        artifact=self.transport.export_bundle(self.program);raw=self.transport.bundle_artifact(artifact['bundle_id'])
        with tempfile.TemporaryDirectory()as temp:
            with zipfile.ZipFile(BytesIO(raw))as z:z.extractall(temp)
            base=Path(temp)/'Interaction-Twin-Robot-Starter';doc,files,_=S.inventory(base)
            self.assertEqual(doc['task_program_sha256'],hashlib.sha256(K.encoded(self.program)).hexdigest())
            (base/'extra-private.txt').write_text('NEVER_SERVED')
            self.assertNotIn('extra-private.txt',files)
            (base/'task-program.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'differ'):S.inventory(base)

    def test_new_brand_mark_is_manifest_bound_without_replacing_legacy_mark(self):
        artifact=self.transport.export_bundle(self.program);raw=self.transport.bundle_artifact(artifact['bundle_id'])
        with zipfile.ZipFile(BytesIO(raw))as archive:
            prefix='Interaction-Twin-Robot-Starter/'
            for key in ['mark','brand-mark-v2']:
                relative=K.FIXED_FILES[key][0]
                self.assertEqual(archive.read(prefix+relative),(self.root/relative).read_bytes())
            manifest=json.loads(archive.read(prefix+'bundle-manifest.json'))
            declared={row['path']:row for row in manifest['files']}
            self.assertIn('showcase/mark.svg',declared);self.assertIn('showcase/brand/v2/mark.svg',declared)
            envelope=json.loads(archive.read(prefix+'kit-response.json'))
            self.assertEqual(envelope['resources']['brand-mark-v2']['url'],'/showcase/brand/v2/mark.svg')

    def test_workbench_navigation_dependency_is_included_exactly_in_starter_and_resource_map(self):
        artifact=self.transport.export_bundle(self.program);raw=self.transport.bundle_artifact(artifact['bundle_id'])
        with zipfile.ZipFile(BytesIO(raw))as archive:
            prefix='Interaction-Twin-Robot-Starter/'
            relative='showcase/robot-workbench-ui-v2.js'
            self.assertEqual(archive.read(prefix+relative),(self.root/relative).read_bytes())
            self.assertIn('src="robot-workbench-ui-v2.js"',archive.read(prefix+'showcase/robot-workbench.html').decode())
            envelope=json.loads(archive.read(prefix+'kit-response.json'))
            self.assertEqual(envelope['resources']['workbench-navigation']['url'],'/'+relative)
            self.assertEqual(S.ARTIFACT_PATHS['workbench-navigation'],relative)

    def test_starter_handler_serves_only_manifest_and_loopback_with_hash_rechecks(self):
        artifact=self.transport.export_bundle(self.program);raw=self.transport.bundle_artifact(artifact['bundle_id'])
        with tempfile.TemporaryDirectory()as temp:
            with zipfile.ZipFile(BytesIO(raw))as z:z.extractall(temp)
            base=Path(temp)/'Interaction-Twin-Robot-Starter';H=S.handler(base)
            (base/'unlisted-private.txt').write_text('NEVER_SERVED')
            def request(path,host='127.0.0.1:1234',head=False):
                h=H.__new__(H);h.path=path;h.headers=Message();h.headers['Host']=host;h.wfile=BytesIO();sent=[]
                h.send_response=lambda code:sent.append(code);h.send_header=lambda *args:None;h.end_headers=lambda:None;h.send_error=lambda code:sent.append(code)
                h._serve(head);return sent,h.wfile.getvalue()
            self.assertEqual(request('/files/'+self.refs[1]['path']),([200],b'glTFcontract-test'))
            self.assertEqual(request('/files/'+self.refs[1]['path'],head=True),([200],b''))
            self.assertEqual(request('/api/robotics/kit?kit_id='+R.KIT_ID)[0],[200])
            self.assertEqual(request('/api/robotics/kits/'+R.KIT_ID+'/artifacts/controller')[0],[200])
            self.assertEqual(request('/api/robotics/kits/'+R.KIT_ID+'/artifacts/kit')[0],[200])
            self.assertEqual(request('/api/robotics/kits/'+R.KIT_ID+'/artifacts/schema')[0],[200])
            self.assertEqual(request('/api/robotics/kits/not-declared/artifacts/controller')[0],[404])
            for path in ['/unlisted-private.txt','/../unlisted-private.txt','/%2e%2e/secret','/%252e%252e/secret','/api/robotics/plan?instruction=anything','/.private/credential','/showcase/robot-controller.js?surprise=1']:
                with self.subTest(path=path):self.assertIn(request(path)[0][0],[400,404])
            self.assertEqual(request('/robot-kit.json',host='attacker.example')[0],[403])
            (base/self.refs[1]['path']).write_bytes(b'glTFtampered-xx')
            self.assertEqual(request('/files/'+self.refs[1]['path'])[0],[409])

    def test_same_process_public_download_matches_exact_edited_task_bundle(self):
        edited={**self.program,'edited_action_title':'Edited task test'}
        original=self.transport.export_bundle(self.program);updated=self.transport.export_bundle(edited)
        self.assertNotEqual(original['artifact']['sha256'],updated['artifact']['sha256'])
        with patch.object(API,'_catalog',return_value=self.transport):
            status,raw,mime,_=API.response('/api/robotics/bundles/'+updated['bundle_id'],{})
        self.assertEqual((status,mime),(200,'application/zip'));self.assertEqual(hashlib.sha256(raw).hexdigest(),updated['artifact']['sha256'])
        with zipfile.ZipFile(BytesIO(raw))as archive:
            self.assertEqual(json.loads(archive.read('Interaction-Twin-Robot-Starter/task-program.json')),edited)

    def test_public_api_strict_query_kit_and_no_unknown_route_claim(self):
        with patch.object(API,'_catalog',return_value=self.transport):
            status,body,mime,_=API.response('/api/robotics/kit','kit_id='+R.KIT_ID)
            self.assertEqual(status,200);self.assertEqual(json.loads(body)['kit']['id'],R.KIT_ID)
            for route,q in [('/api/robotics/kit','kit_id=x&kit_id=y'),('/api/robotics/kits','anything=x'),('/api/robotics/plan','twin_id=x&instruction='),('/api/robotics/plan','twin_id=x&instruction='+'a'*2001),('/api/robotics/kits','x='+'a'*8193)]:
                with self.subTest(q=q[:90]):self.assertEqual(API.response(route,q)[0],400)
            self.assertIsNone(API.response('/api/robotics/private',''));self.assertIsNone(API.response('/api/uploads',''))
            status,body,_,_=API.response('/api/robotics/kits/'+R.KIT_ID+'/artifacts/kit',{})
            self.assertEqual(status,200);self.assertEqual(json.loads(body),R.get_robot_kit())
            status,body,_,_=API.response('/api/robotics/kits/'+R.KIT_ID+'/artifacts/no-such-file',{})
            self.assertEqual(status,400);self.assertNotIn(b'.private',body)

if __name__=='__main__':unittest.main()
