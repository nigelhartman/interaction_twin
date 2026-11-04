"""The new robot-task reads cannot bypass owner-only APIs or become execution."""
from email.message import Message
import importlib.util
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('robotics_boundary_server',ROOT/'scripts/serve_workspace.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)


class RoboticsHTTPBoundaryTests(unittest.TestCase):
    def request(self,path,method='GET',host='127.0.0.1:8766',response=None):
        h=S.Handler.__new__(S.Handler)
        h.path=path;h.headers=Message();h.headers['Host']=host
        h.headers['Content-Length']='2';h.headers['Content-Type']='application/json'
        h.wfile=BytesIO();h.rfile=BytesIO(b'{}');h.connection=SimpleNamespace(settimeout=lambda _:None)
        h.server=SimpleNamespace(admin_auth=None,public_asset_cache={})
        sent=[];errors=[]
        h.send_response=lambda status:sent.append(('status',status))
        h.send_header=lambda key,value:sent.append((key,value))
        h.end_headers=lambda:None
        h.json=lambda status,value:errors.append((status,value))
        with patch('public_robotics_api.response',return_value=response) as adapter:
            getattr(h,'do_'+method)()
        return h,sent,errors,adapter

    def test_exact_read_returns_bounded_adapter_bytes_without_owner_session(self):
        body=b'{"kit":{"id":"generic5dof_tabletop_v1"}}'
        h,sent,errors,adapter=self.request('/api/robotics/kit?kit_id=generic5dof_tabletop_v1',response=(200,body,'application/json; charset=utf-8',{}))
        self.assertEqual(errors,[]);self.assertEqual(h.wfile.getvalue(),body)
        self.assertIn(('Content-Length',str(len(body))),sent)
        self.assertIn(('X-Content-Type-Options','nosniff'),sent)
        self.assertIn(('Cache-Control','no-store'),sent)
        self.assertEqual(adapter.call_args.args,('/api/robotics/kit',{'kit_id':['generic5dof_tabletop_v1']}))

    def test_binary_download_keeps_declared_mime_and_attachment(self):
        body=b'PK\x03\x04fixture'
        h,sent,errors,_=self.request('/api/robotics/bundles/'+'a'*64,response=(200,body,'application/zip',{'Content-Disposition':'attachment; filename="robot-task.zip"'}))
        self.assertFalse(errors);self.assertEqual(h.wfile.getvalue(),body)
        self.assertIn(('Content-Type','application/zip'),sent)
        self.assertIn(('Content-Disposition','attachment; filename="robot-task.zip"'),sent)

    def test_head_has_identical_length_and_no_body(self):
        h,sent,errors,_=self.request('/api/robotics/kits','HEAD',response=(200,b'12345','application/json',{}))
        self.assertFalse(errors);self.assertEqual(h.wfile.getvalue(),b'')
        self.assertIn(('Content-Length','5'),sent)

    def test_unknown_robotics_routes_still_require_owner_authentication(self):
        _,_,errors,adapter=self.request('/api/robotics/execute')
        self.assertEqual(errors[0][0],401);adapter.assert_called_once()

    def test_owner_apis_never_reach_public_adapter(self):
        for path in ['/api/config','/api/runs','/api/admin/candidates','/api/meta/budget']:
            with self.subTest(path=path):
                _,_,errors,adapter=self.request(path)
                self.assertEqual(errors[0][0],401);adapter.assert_not_called()

    def test_posting_to_robotics_never_executes_or_mutates(self):
        _,_,errors,adapter=self.request('/api/robotics/plan','POST')
        self.assertEqual(errors[0][0],403);adapter.assert_not_called()

    def test_nonlocal_host_is_rejected_before_robotics_reads(self):
        _,_,errors,adapter=self.request('/api/robotics/kits',host='attacker.example')
        self.assertEqual(errors[0][0],403);adapter.assert_not_called()


if __name__=='__main__':unittest.main()
