"""Isolated admin fixture boundaries; no owner session, socket or models."""
from email.message import Message
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from scripts.admin_review_fixture_20261104_v1 import ROOT, CANDIDATE, handler_for, prepare

class AdminReviewFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required=[ROOT/'showcase/public/public-assets.json',ROOT/'showcase/public/library-catalog.json',ROOT/'showcase/public/aligned-scenes.json']
        if any(not path.is_file() for path in required):
            raise unittest.SkipTest('Saved public catalog and approved artifacts are separate from the source-only checkout')
        cls.temp=tempfile.TemporaryDirectory(prefix='qa-fixture-test-',dir=ROOT/'output')
        cls.fixture=prepare(Path(cls.temp.name));cls.Handler=handler_for(cls.fixture)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def request(self,route,method='GET',range_header=None):
        h=self.Handler.__new__(self.Handler);h.path=route;h.command=method;h.headers=Message();h.wfile=BytesIO();sent=[]
        if range_header:h.headers['Range']=range_header
        h.send_response=lambda status:sent.append(('status',status));h.send_header=lambda key,value:sent.append((key,value));h.end_headers=lambda:None
        getattr(h,'do_'+method)();return sent,h.wfile.getvalue()
    def test_fixture_has_distinct_saved_scenes_and_only_public_allowlisted_data(self):
        f=self.fixture;meta=f['metadata'];self.assertNotEqual(meta['before_scene_ref']['sha256'],meta['after_scene_ref']['sha256'])
        self.assertIn('synthetic',meta['history'].lower());self.assertIn('not the owner service',(f['destination']/'admin/index.html').read_text())
        self.assertTrue(all(p.startswith('data/') and target.is_relative_to(ROOT/'data') for p,target in f['data'].items()))
        self.assertTrue(all(not any(part.startswith('.') for part in Path(p).parts) for p in f['data']))
    def test_every_mutation_is_rejected_without_consuming_a_body_or_writing_files(self):
        before={p:p.stat().st_mtime_ns for p in self.fixture['destination'].rglob('*') if p.is_file()}
        for route in ['/api/admin/candidates/'+CANDIDATE+'/reject','/api/admin/candidates/'+CANDIDATE+'/approve','/api/admin/candidates/'+CANDIDATE+'/repair','/api/admin/setup','/api/admin/login','/api/run','/api/admin/operations/gpu/refresh']:
            with self.subTest(route=route):
                sent,raw=self.request(route,'POST');self.assertEqual(sent[0],('status',405));self.assertIn(b'disabled',raw)
        after={p:p.stat().st_mtime_ns for p in self.fixture['destination'].rglob('*') if p.is_file()};self.assertEqual(after,before)
    def test_unknown_private_files_and_real_api_routes_never_escape_the_fixture(self):
        for route in ['/files/.private/admin-workflow.sqlite3','/files/.secrets/meta.env','/admin/../private.json','/files/data/not-admitted.json','/api/meta/budget','/api/admin/candidates/candidate_'+'0'*32]:
            with self.subTest(route=route):self.assertEqual(self.request(route)[0][0],('status',404))
    def test_source_video_ranges_and_historical_adapter_use_the_exact_fixture_files(self):
        path=self.fixture['metadata']['source_ref']['path'];sent,raw=self.request('/files/'+path,range_header='bytes=1-4')
        self.assertEqual(sent[0],('status',206));self.assertEqual(raw,(ROOT/path).read_bytes()[1:5])
        for route in ['/admin/revision-scene.html','/admin/revision-scene.js','/admin/review-refinement.js']:
            sent,raw=self.request(route);self.assertEqual(sent[0],('status',200));self.assertEqual(raw,self.fixture['files'][route].read_bytes())

if __name__=='__main__':unittest.main()
