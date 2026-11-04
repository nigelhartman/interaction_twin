"""Explicit archive selection preserves old bytes and strict private boundaries."""
import ast
import copy
import hashlib
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import ProxyHandler,Request,build_opener
from urllib.error import HTTPError

from scripts import launch_capture_review as V1,launch_capture_review_v2 as V2,launch_capture_review_v3 as V3
from scripts import restore_capture_review_runtime_20261103 as A
from scripts import restore_capture_review_runtime_20261104_v2 as B
from scripts import restore_capture_review_runtime_20261104_v3 as C

PRIVATE=(A.ROOT/A.MANIFEST).is_file()


class RuntimeArchiveTests(unittest.TestCase):
    @unittest.skipUnless(PRIVATE,'Private frozen capture runtime archive is not included')
    def test_old_selections_stay_pinned_and_stale_runtime_stays_rejected(self):
        for launcher in [V1,V2]:
            config=launcher.load_configuration();spec=config['viewers'][2]
            snapshot=json.loads(launcher.checked(spec['snapshot_ref']))
            # Legitimate public runtime updates must never pass the old pin.
            with self.assertRaisesRegex(ValueError,'Frozen review file'):
                launcher.checked(snapshot['routes']['/showcase/aligned-scene-contract.js'])
            restored=C.restore_spec(B.restore_spec(A.restore_spec(spec)))
            self.assertNotEqual(restored['snapshot_ref'],spec['snapshot_ref'])
            actual=launcher.prepare_spec(restored)
            self.assertEqual(actual['port'],spec['port'])
            self.assertEqual(actual['routes']['/manifest.json'],snapshot['routes']['/manifest.json'])
            expected='e4eb3534368e7cfe97793d987a53aef18e493a959aefa38719efda2b753e081d' if launcher is V1 else '3cf390429a6f271902d4a506ca91b2b29c1a95b8a9faa675545fc4ccbfd3e435'
            self.assertEqual(hashlib.sha256(launcher.checked(spec['snapshot_ref'])).hexdigest(),expected)

    @unittest.skipUnless(PRIVATE,'Private frozen capture runtime archive is not included')
    def test_only_two_filesystem_refs_change_all_url_source_model_fields_identical(self):
        manifest=json.loads((A.ROOT/A.MANIFEST).read_text())
        self.assertEqual(hashlib.sha256((A.ROOT/A.MANIFEST).read_bytes()).hexdigest(),A.MANIFEST_SHA)
        for record in manifest['snapshots']:
            source=json.loads(V1.checked(record['source_snapshot_ref']));restored=json.loads(V1.checked(record['restored_snapshot_ref']))
            changes={name for name in source['routes']if source['routes'][name]!=restored['routes'][name]}
            self.assertEqual(changes,A.ALLOWED_URLS)
            a=copy.deepcopy(source);b=copy.deepcopy(restored);a.pop('routes');b.pop('routes');self.assertEqual(a,b)
            for url in changes:
                old,new=source['routes'][url],restored['routes'][url]
                self.assertEqual(new['sha256'],old['sha256']);self.assertEqual(new['bytes'],old['bytes'])
                self.assertTrue(new['path'].startswith(A.ARCHIVE_PREFIX))
                self.assertEqual(hashlib.sha256(V1.checked(new)).hexdigest(),old['sha256'])

    @unittest.skipUnless(PRIVATE,'Private frozen capture runtime archive is not included')
    def test_viewer_identity_cannot_be_changed_by_restoration(self):
        original=V2.load_configuration()['viewers'][2]
        for field,value in [('id','foreign-view'),('port',59001),('prefix','/abcdefghijklmnopqrstuvwxyz/'),('kind','batch')]:
            value_spec=copy.deepcopy(original);value_spec[field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'viewer identity changed'):A.restore_spec(value_spec)
        batch=V2.load_configuration()['viewers'][0]
        self.assertEqual(A.restore_spec(batch),batch)
        self.assertIsNot(A.restore_spec(batch),batch)

    def test_archive_manifest_changed_is_rejected_before_any_fallback(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);p=root/A.MANIFEST;p.parent.mkdir(parents=True);p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'selection manifest changed'):A.restore_spec({'id':'anything'},root=root)

    @unittest.skipUnless(PRIVATE,'Private frozen capture runtime archive is not included')
    def test_new_default_selection_prepares_all_original_routes_and_http_guards_unchanged(self):
        c=V3.load_configuration();specs=[V3.prepare_spec(C.restore_spec(B.restore_spec(s)))for s in c['viewers']]
        self.assertEqual([s['port']for s in specs],[52175,65139,52154]);self.assertEqual([len(s['routes'])for s in specs],[147,15,23])
        def functions(path):return{n.name:ast.dump(n,include_attributes=False)for n in ast.parse(Path(path).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        before,after=functions(V2.__file__),functions(V3.__file__)
        self.assertEqual(set(before),set(after))
        for name in set(before)-{'load_configuration','owned_listener','reopen'}:
            with self.subTest(name=name):self.assertEqual(before[name],after[name])
        own={'pid':919,'uid':44,'argv':['python',str(V3.ROOT/'scripts/launch_capture_review_v3.py'),'--serve',specs[2]['id']]}
        self.assertTrue(V3.owned_listener(specs[2],own,uid=44));own['argv'][1]=str(V3.ROOT/'scripts/launch_capture_review_v2.py');self.assertFalse(V3.owned_listener(specs[2],own,uid=44))

    @unittest.skipUnless(PRIVATE,'Private frozen capture runtime archive is not included')
    def test_actual_isolated_http_delivers_exact_original_runtime_and_rejects_unknown_paths(self):
        record=json.loads((A.ROOT/A.MANIFEST).read_text())['snapshots'][1]
        snapshot=json.loads(V1.checked(record['restored_snapshot_ref']))
        spec={'id':'runtime-archive-fixture','routes':{url:snapshot['routes'][url]for url in A.ALLOWED_URLS}}
        server=ThreadingHTTPServer(('127.0.0.1',0),V3.handler(spec));worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        try:
            opener=build_opener(ProxyHandler({}));base=f'http://127.0.0.1:{server.server_port}'
            for url,item in spec['routes'].items():
                with opener.open(base+url)as response:
                    self.assertEqual(response.status,200);self.assertEqual(response.read(),V1.checked(item))
                    self.assertEqual(response.headers['X-Interaction-Twin-Review'],spec['id']+':'+V3.CONFIG_SHA)
            for route,headers,status in [('/.secrets',{},404),('/showcase/not-listed.js',{},404),('/showcase/aligned-scene-contract.js',{'Origin':'null'},403),('/showcase/aligned-scene-contract.js?x=1',{},404)]:
                with self.subTest(route=route),self.assertRaises(HTTPError)as error:opener.open(Request(base+route,headers=headers))
                self.assertEqual(error.exception.code,status);error.exception.close()
        finally:server.shutdown();server.server_close();worker.join(timeout=3)


if __name__=='__main__':unittest.main()
