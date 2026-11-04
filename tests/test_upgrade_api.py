import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

import stage5
import upgrade_api as A
from test_revisions import CASE,published_fixture

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('upgrade_http_server',ROOT/'scripts/serve_workspace.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)


class UpgradeFixture:
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source,self.doc=published_fixture(self.root)
        p=self.root/'workspace/cases.json';value=json.loads(p.read_text());value['cases'][0]['title']='Test source';p.write_text(json.dumps(value))
        self.body={'twin_id':CASE,'track_id':'part','base_sha256':stage5.sha(self.source),'mode':'refit_existing'}

    def tearDown(self):self.temp.cleanup()


class UpgradeBoundaryContracts(UpgradeFixture,unittest.TestCase):
    def test_catalog_uses_only_published_hash_valid_evidence_and_keeps_unavailable_backends_explicit(self):
        result=A.catalog(root=self.root)
        self.assertEqual(result['errors'],[]);self.assertEqual(len(result['projects']),1)
        item=result['projects'][0];self.assertEqual(item['base_sha256'],self.body['base_sha256'])
        self.assertFalse(item['components'][0]['has_asset'])
        self.assertEqual(item['components'][0]['baseline_metrics']['total_samples'],3)
        self.assertTrue(item['poster_url'].startswith('/files/data/case/'))
        self.assertFalse(next(x for x in result['backends'] if x['id']=='sam3d_objects')['available'])
        self.assertFalse(result['policy']['hosted_reasoning'])
        (self.root/self.doc['evidence'][0]['frame_path']).write_bytes(b'changed')
        changed=A.catalog(root=self.root)
        self.assertEqual(changed['projects'],[]);self.assertEqual(len(changed['errors']),1)

    def test_browser_cannot_supply_commands_paths_or_coerce_imported_modes(self):
        self.assertEqual(A.canonical_request(self.body),self.body)
        for change in [{'command':'run something'},{'output_path':'data/chosen'},{'mode':'remote_shell'},
                       {'twin_id':'../../unpublished'},{'base_sha256':'bad'},
                       {'imported_result_id':'import-'+'1'*32},{'mode':'imported_sam3d'}]:
            with self.subTest(change=change),self.assertRaises(ValueError):A.canonical_request(self.body|change)

    def test_run_receipts_do_not_follow_symlinks_or_accept_mismatched_ids(self):
        folder=self.root/'data/model-upgrades/runs/upgrade-example';folder.mkdir(parents=True)
        path=folder/'run.json';path.write_text(json.dumps({'id':'upgrade-example','status':'complete'}))
        self.assertEqual(A.run_record('upgrade-example',root=self.root)['status'],'complete')
        path.write_text(json.dumps({'id':'upgrade-other','status':'complete'}))
        with self.assertRaisesRegex(ValueError,'differs'):A.run_record('upgrade-example',root=self.root)
        path.unlink();path.symlink_to(self.source)
        with self.assertRaisesRegex(ValueError,'unavailable'):A.run_record('upgrade-example',root=self.root)
        self.assertEqual(A.runs(root=self.root)['runs'],[])

    def test_correcting_an_older_receipt_does_not_make_it_the_latest_experiment(self):
        for name,date in [('upgrade-new','2026-10-29T19:15:00+00:00'),('upgrade-old','2026-10-29T19:11:00+00:00')]:
            path=self.root/'data/model-upgrades/runs'/name/'run.json';path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'id':name,'status':'complete','created_at':date}))
        self.assertEqual([r['id'] for r in A.runs(root=self.root)['runs']],['upgrade-new','upgrade-old'])


class UpgradeHTTPContracts(UpgradeFixture,unittest.TestCase):
    def setUp(self):
        super().setUp();self.patcher=patch.object(S,'ROOT',self.root);self.patcher.start()
        self.server=S.WorkspaceServer(('127.0.0.1',0),S.Handler)
        from tests.owner_http_fixture import OwnerHTTPSession
        self.owner_http=OwnerHTTPSession(self.server);self.addCleanup(self.owner_http.close)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.base=f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2);self.patcher.stop();super().tearDown()

    def request(self,path,body=None,headers=None):
        data=json.dumps(body).encode() if body is not None else None
        req=urllib.request.Request(self.base+path,data=data,headers=headers or {'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=10) as response:return response.status,json.load(response)

    def test_real_routes_for_catalog_and_plan_validate_before_execution(self):
        code,data=self.request('/api/upgrades/catalog');self.assertEqual(code,200);self.assertEqual(len(data['projects']),1)
        with patch.object(A,'plan',return_value={'can_run':False,'blockers':['Missing asset']}) as planner:
            code,data=self.request('/api/upgrades/plan',self.body)
            self.assertEqual(code,200);self.assertFalse(data['can_run']);planner.assert_called_once()

    def test_same_origin_and_single_worker_rules_cover_upgrade_requests(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request('/api/upgrades/plan',self.body,{'Content-Type':'application/json','Origin':'https://example.com'})
        self.assertEqual(error.exception.code,403)
        class Busy:
            id='run-busy';state={'status':'running'}
            def cancel(self):pass
        self.server.active=Busy()
        try:
            with patch.object(A,'plan',return_value={'can_run':True}),self.assertRaises(urllib.error.HTTPError) as error:
                self.request('/api/upgrades/runs',self.body)
            self.assertEqual(error.exception.code,409)
        finally:self.server.active=None

    def test_result_upload_is_bounded_and_requires_a_prepared_job(self):
        req=urllib.request.Request(self.base+'/api/upgrades/jobs/job-'+'1'*32+'/result',data=b'not a zip',headers={'Content-Type':'text/plain'})
        with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(req,timeout=10)
        self.assertEqual(error.exception.code,400)
        self.assertFalse((self.root/'data/upgrade-uploads').exists())


if __name__=='__main__':unittest.main()
