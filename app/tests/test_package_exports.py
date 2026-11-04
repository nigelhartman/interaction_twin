import json
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import Mock, patch

import package_exports as E
import stage5
from test_revisions import CASE, published_fixture

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('package_export_server',ROOT/'scripts/serve_workspace.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)


def fake_export(twin_id, directory, **options):
    directory.mkdir()
    (directory/'index.html').write_text('<title>Offline package</title>')
    (directory/'manifest.json').write_text('{}')
    archive=Path(str(directory)+'.zip');archive.write_bytes(b'test archive')
    return {'directory':str(directory),'index':str(directory/'index.html'),
            'manifest':str(directory/'manifest.json'),'zip':str(archive),
            'manifest_sha256':stage5.sha(directory/'manifest.json')}


class PackageExportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source,self.doc=published_fixture(self.root)
        self.body={'base_sha256':stage5.sha(self.source)}

    def tearDown(self):
        self.temp.cleanup()

    def test_success_publishes_only_confined_local_links_and_omits_video_by_default(self):
        exporter=Mock(side_effect=fake_export)
        result=E.create_export(CASE,self.body,root=self.root,exporter=exporter)
        self.assertEqual(result['status'],'complete')
        self.assertFalse(result['source_video_included'])
        self.assertFalse(exporter.call_args.kwargs['include_video'])
        self.assertFalse(result['physical_simulation_ready'])
        for key in ['viewer_url','manifest_url','zip_url']:
            self.assertTrue(result[key].startswith('/files/data/portable-exports/export-'))
            self.assertTrue((self.root/result[key][7:]).is_file())

    def test_stale_source_and_unpublished_ids_cannot_start_export(self):
        for twin_id,body in [(CASE,{'base_sha256':'f'*64}),('unpublished',self.body)]:
            exporter=Mock()
            with self.subTest(twin_id=twin_id),self.assertRaises(ValueError):
                E.create_export(twin_id,body,root=self.root,exporter=exporter)
            exporter.assert_not_called()
        self.assertFalse((self.root/'data/portable-exports').exists())

    def test_request_does_not_accept_paths_or_coerced_privacy_options(self):
        for extra in [{'output':'arbitrary-directory'},{'include_video':'false'}, {'include_video':1}]:
            exporter=Mock()
            with self.subTest(extra=extra),self.assertRaises(ValueError):
                E.create_export(CASE,self.body|extra,root=self.root,exporter=exporter)
            exporter.assert_not_called()

    def test_changed_evidence_prevents_packaging_even_with_unchanged_program_hash(self):
        evidence=self.root/self.doc['evidence'][0]['frame_path'];evidence.write_bytes(b'changed')
        exporter=Mock()
        with self.assertRaises(ValueError):
            E.create_export(CASE,self.body,root=self.root,exporter=exporter)
        exporter.assert_not_called()

    def test_failed_export_retains_partial_files_and_truthful_failure_receipt(self):
        def fail(twin,directory,**kwargs):
            directory.mkdir();(directory/'partial.bin').write_bytes(b'partial')
            raise ValueError('A component failed verification')
        with self.assertRaisesRegex(ValueError,'verification'):
            E.create_export(CASE,self.body,root=self.root,exporter=fail)
        receipts=list((self.root/'data/portable-exports').glob('*-failure.json'))
        self.assertEqual(len(receipts),1)
        self.assertEqual(json.loads(receipts[0].read_text())['status'],'failed')
        self.assertEqual(len(list((self.root/'data/portable-exports').glob('*/partial.bin'))),1)

    def test_symlink_storage_or_unexpected_returned_file_cannot_be_published(self):
        (self.root/'data/portable-exports').symlink_to(self.root/'workspace',target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'symlinks'):
            E.create_export(CASE,self.body,root=self.root,exporter=fake_export)
        (self.root/'data/portable-exports').unlink()
        def wrong(twin,directory,**kwargs):
            result=fake_export(twin,directory,**kwargs);result['index']=str(self.source)
            return result
        with self.assertRaisesRegex(ValueError,'declared index'):
            E.create_export(CASE,self.body,root=self.root,exporter=wrong)


class PackageHTTPContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source,_=published_fixture(self.root)
        self.patcher=patch.object(S,'ROOT',self.root);self.patcher.start()
        self.server=S.WorkspaceServer(('127.0.0.1',0),S.Handler)
        from tests.owner_http_fixture import OwnerHTTPSession
        self.owner_http=OwnerHTTPSession(self.server);self.addCleanup(self.owner_http.close)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.base=f'http://127.0.0.1:{self.server.server_port}'
        self.url=self.base+f'/api/cases/{CASE}/portable-package'
        self.body={'base_sha256':stage5.sha(self.source)}

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)
        self.patcher.stop();self.temp.cleanup()

    def request(self,body,headers=None):
        request=urllib.request.Request(self.url,data=json.dumps(body).encode(),headers=headers or {'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=10) as response:
            return response.status,json.load(response)

    def test_real_route_creates_local_files_and_stale_requests_return_conflict(self):
        with patch('portable_twin.export_case',side_effect=fake_export) as exporter:
            status,result=self.request(self.body)
            self.assertEqual(status,201)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(exporter.call_count,1)
            for key in ['viewer_url','manifest_url','zip_url']:
                self.assertTrue((self.root/result[key][7:]).is_file())
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.request({'base_sha256':'f'*64})
            self.assertEqual(error.exception.code,409)
            self.assertEqual(exporter.call_count,1)

    def test_cross_origin_or_extra_fields_never_call_exporter(self):
        with patch('portable_twin.export_case') as exporter:
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.request(self.body,{'Content-Type':'application/json','Origin':'https://example.com'})
            self.assertEqual(error.exception.code,403)
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.request(self.body|{'output_path':'data/arbitrary'})
            self.assertEqual(error.exception.code,400)
            exporter.assert_not_called()


if __name__=='__main__':
    unittest.main()
