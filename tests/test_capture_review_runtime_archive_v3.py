"""Exact archived HTML extends old JS selections; no mutable-runtime fallback."""
import ast
import copy
import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import ProxyHandler,Request,build_opener
import zipfile

from scripts import launch_capture_review as V1,launch_capture_review_v2 as V2
from scripts import launch_capture_review_v3 as V3,launch_capture_review_v4 as V4,launch_capture_review_v5 as V5
from scripts import restore_capture_review_runtime_20261104_v2 as B
from scripts import restore_capture_review_runtime_20261104_v3 as A

PRIVATE=(A.ROOT/A.MANIFEST).is_file()
def manifest():return json.loads(V1.checked({'path':A.MANIFEST,'sha256':A.MANIFEST_SHA,'bytes':A.MANIFEST_BYTES}))

@unittest.skipUnless(PRIVATE,'Private exact historical HTML archive is not included')
class RuntimeArchiveV3Tests(unittest.TestCase):
    def test_prior_launchers_configs_snapshots_and_runtime_manifests_remain_byte_identical(self):
        data=manifest();pins=json.loads(V1.checked(data['prior_file_pins_ref']))
        for name,pointer in pins.items():
            with self.subTest(name=name):self.assertEqual(hashlib.sha256(V1.checked(pointer)).hexdigest(),pointer['sha256'])
        prior_command=V1.checked(data['prior_command_ref']).decode()
        self.assertIn('launch_capture_review_v4.py',prior_command)
        self.assertIn(hashlib.sha256(Path(V4.__file__).read_bytes()).hexdigest(),prior_command)

    def test_only_html_path_changes_after_v2_all_original_source_model_url_data_remain_exact(self):
        data=manifest();self.assertEqual(len(data['snapshots']),2)
        self.assertEqual(sum(len(r['original_snapshot_refs'])for r in data['snapshots']),4)
        for record in data['snapshots']:
            old=json.loads(V1.checked(record['source_snapshot_ref']));new=json.loads(V1.checked(record['restored_snapshot_ref']))
            changes={url for url in old['routes']if old['routes'][url]!=new['routes'][url]}
            self.assertEqual(changes,A.ALLOWED_URLS)
            expected=copy.deepcopy(old)
            for url in changes:
                original,archived=old['routes'][url],new['routes'][url]
                self.assertEqual((original['sha256'],original['bytes']),(archived['sha256'],archived['bytes']))
                self.assertTrue(archived['path'].startswith(A.ARCHIVE_PREFIX));V1.checked(archived)
                expected['routes'][url]=archived
                with self.assertRaisesRegex(ValueError,'Frozen review file'):V1.checked(original)
            self.assertEqual(new,expected)

    def test_html_is_exact_member_of_previously_verified_archive(self):
        data=manifest();V1.checked(data['source_zip']);V1.checked(data['source_zip_build_receipt'])
        self.assertEqual(data['source_zip']['sha256'],'96e0e6fb814b8063d650e724d6e1f9759440ff0f6529d6b9799383f719b07f5a')
        with zipfile.ZipFile(A.ROOT/data['source_zip']['path'])as archive:
            for row in data['archive_members']:
                raw=archive.read(row['member']);self.assertEqual(raw,V1.checked(row['extracted_ref']))
                self.assertEqual(hashlib.sha256(raw).hexdigest(),'4640bb63fb821c29f4c166ace777c13538511bb6dca8f89ea053a6e75cd2766c')

    def test_original_and_already_archived_identities_are_bound_and_unmapped_selection_never_falls_back(self):
        for launcher in [V1,V2,V3,V4]:
            raw=launcher.load_configuration()['viewers'][2];selected=A.restore_spec(raw)
            self.assertEqual(A.restore_spec(selected),selected)
            for candidate in [raw,selected]:
                for key,value in [('id','foreign'),('port',59001),('kind','batch'),('prefix','/abcdefghijklmnopqrstuvwxyz/')]:
                    changed=copy.deepcopy(candidate);changed[key]=value
                    with self.subTest(launcher=launcher.__name__,key=key),self.assertRaisesRegex(ValueError,'viewer identity changed'):A.restore_spec(changed)
            prepared=launcher.prepare_spec(selected)
            self.assertEqual(prepared['port'],raw['port'])
        batch=V4.load_configuration()['viewers'][0]
        self.assertEqual(A.restore_spec(batch),batch);self.assertIsNot(A.restore_spec(batch),batch)

    def test_changed_html_or_selection_manifest_is_rejected_before_any_implicit_fallback(self):
        data=manifest();record=data['snapshots'][1];raw=V4.load_configuration()['viewers'][2]
        for kind in ['html','manifest']:
            with tempfile.TemporaryDirectory()as folder:
                root=Path(folder)
                pointers=[{'path':B.MANIFEST,'sha256':B.MANIFEST_SHA,'bytes':B.MANIFEST_BYTES},
                          {'path':A.MANIFEST,'sha256':A.MANIFEST_SHA,'bytes':A.MANIFEST_BYTES},
                          record['source_snapshot_ref'],record['restored_snapshot_ref'],record['runtime_substitutions']['/showcase/aligned-scene.html']['archived_ref']]
                for pointer in pointers:
                    path=root/pointer['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(V1.checked(pointer))
                target=root/(A.MANIFEST if kind=='manifest'else pointers[-1]['path']);target.write_bytes(target.read_bytes()+b'changed')
                with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'selection manifest changed|Frozen review file'):A.restore_spec(raw,root=root)

    def test_v5_prepares_all_original_routes_preserving_source_http_and_process_ownership_guards(self):
        config=V5.load_configuration();specs=[V5.prepare_spec(s)for s in config['viewers']]
        self.assertEqual([s['port']for s in specs],[52175,65139,52154]);self.assertEqual([len(s['routes'])for s in specs],[147,15,23])
        def functions(path):return{n.name:ast.dump(n,include_attributes=False)for n in ast.parse(Path(path).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        old,new=functions(V4.__file__),functions(V5.__file__);self.assertEqual(set(old),set(new))
        for name in set(old)-{'load_configuration','owned_listener','reopen'}:
            with self.subTest(name=name):self.assertEqual(old[name],new[name])
        own={'pid':777,'uid':44,'argv':['python',str(V5.ROOT/'scripts/launch_capture_review_v5.py'),'--serve',specs[2]['id']]}
        self.assertTrue(V5.owned_listener(specs[2],own,uid=44));own['argv'][1]=str(V5.ROOT/'scripts/launch_capture_review_v4.py');self.assertFalse(V5.owned_listener(specs[2],own,uid=44))
        command=(V5.ROOT/'Open Capture Review.command').read_text();self.assertIn('launch_capture_review_v5.py',command);self.assertIn(hashlib.sha256(Path(V5.__file__).read_bytes()).hexdigest(),command)

    def test_isolated_http_delivers_exact_archived_html_and_rejects_private_cross_origin_and_unknown_routes(self):
        selected=V5.load_configuration()['viewers'][2];snapshot=json.loads(V5.checked(selected['snapshot_ref']));pointer=snapshot['routes']['/showcase/aligned-scene.html'];spec={'id':'archived-html-test','routes':{'/showcase/aligned-scene.html':pointer}}
        server=ThreadingHTTPServer(('127.0.0.1',0),V5.handler(spec));worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        try:
            opener=build_opener(ProxyHandler({}));base=f'http://127.0.0.1:{server.server_port}'
            with opener.open(base+'/showcase/aligned-scene.html')as response:
                self.assertEqual(response.status,200);self.assertEqual(response.read(),V1.checked(pointer));self.assertTrue(response.headers['Content-Type'].startswith('text/html'))
            for route,headers,status in [('/.secrets',{},404),('/admin/',{},404),('/showcase/aligned-scene.html?x=1',{},404),('/showcase/aligned-scene.html',{'Origin':'null'},403)]:
                with self.subTest(route=route),self.assertRaises(HTTPError)as error:opener.open(Request(base+route,headers=headers))
                self.assertEqual(error.exception.code,status);error.exception.close()
        finally:server.shutdown();server.server_close();worker.join(timeout=3)

if __name__=='__main__':unittest.main()
