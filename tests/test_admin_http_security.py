"""Exercise actual HTTP handler authorization before uploads, jobs and reads."""
from email.message import Message
import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import admin_auth as A

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('secured_workspace',ROOT/'scripts/serve_workspace.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)


class AdminHTTPSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.auth=A.AdminAuth(self.root)
        A.configure_password(self.root,'Test fixture owner password')
        anon,token=self.auth.status();self.anon=f'{A.COOKIE}={token}';self.anon_csrf=anon['csrf_token']
        owned,token=self.auth.login('Test fixture owner password',self.anon,self.anon_csrf)
        self.cookie=f'{A.COOKIE}={token}';self.csrf=owned['csrf_token']
        self.server=SimpleNamespace(admin_auth=self.auth,public_asset_cache={},lock=__import__('threading').Lock(),active=None)
        for name in ['workspace','showcase/public','admin']:(self.root/name).mkdir(parents=True,exist_ok=True)
        self.original_static=S.static_path

    def request(self,route,method='GET',body=None,*,headers=None):
        h=S.Handler.__new__(S.Handler);h.server=self.server;h.path=route;h.headers=Message()
        h.headers['Host']='127.0.0.1:8765'
        for key,value in (headers or {}).items():h.headers[key]=value
        raw=body if isinstance(body,bytes) else json.dumps(body).encode() if body is not None else b''
        if method=='POST':
            h.headers['Content-Length']=str(len(raw))
            if not h.headers.get('Content-Type'):h.headers['Content-Type']='application/json'
        h.rfile=BytesIO(raw);h.wfile=BytesIO();h.connection=SimpleNamespace(settimeout=lambda _:None)
        calls=[];sent=[];h.json=lambda status,value:calls.append((status,value));h.send_response=lambda status:sent.append(('status',status));h.send_header=lambda k,v:sent.append((k,v));h.end_headers=lambda:None
        with patch.object(S,'ROOT',self.root),patch.object(S,'static_path',side_effect=lambda route:self.original_static(route,root=self.root)),patch.object(S,'public_url',side_effect=lambda path:'/files/'+path.relative_to(self.root).as_posix()):
            if method=='POST':h.do_POST()
            else:h.do_GET()
        return h,calls,sent

    def owner(self,**extra):return {'Cookie':self.cookie,'X-CSRF-Token':self.csrf,'Origin':'http://127.0.0.1:8765',**extra}

    def test_every_existing_mutation_is_guarded_before_body_or_side_effects(self):
        routes=['/api/upload','/api/run','/api/runs/run-example/cancel','/api/cases/example/revisions',
                '/api/cases/example/portable-package','/api/upgrades/jobs','/api/upgrades/runs',
                '/api/observed/jobs','/api/motion/jobs','/api/calibration/jobs','/api/intelligence/run',
                '/api/admin/candidates']
        for route in routes:
            with self.subTest(route=route):
                _,calls,_=self.request(route,'POST',{},headers={'Origin':'http://127.0.0.1:8765'})
                self.assertEqual(calls[0][0],401)
        self.assertFalse((self.root/'data/uploads').exists())

    def test_owner_cookie_alone_never_bypasses_origin_and_csrf(self):
        for headers in [{'Cookie':self.cookie,'X-CSRF-Token':self.csrf},
                        self.owner(Origin='https://attacker.example'),
                        {'Cookie':self.cookie,'Origin':'http://127.0.0.1:8765'},
                        self.owner(**{'X-CSRF-Token':'wrong'})]:
            with self.subTest(headers=sorted(headers)):
                _,calls,_=self.request('/api/upload','POST',b'fixture',headers=headers)
                self.assertEqual(calls[0][0],403)
        self.assertFalse((self.root/'data/uploads').exists())

    def test_authenticated_binary_upload_is_private_from_anonymous_direct_url(self):
        h,calls,_=self.request('/api/upload','POST',b'private-video-fixture',headers=self.owner(**{'X-Filename':'capture.mp4','Content-Type':'application/octet-stream'}))
        self.assertEqual(calls[0][0],201);saved=calls[0][1]
        self.assertEqual((self.root/saved['input_path']).read_bytes(),b'private-video-fixture')
        _,anonymous,_=self.request(saved['url']);self.assertEqual(anonymous[0][0],404)
        h,owner,sent=self.request(saved['url'],headers={'Cookie':self.cookie})
        self.assertEqual(sent[0],('status',200));self.assertEqual(h.wfile.getvalue(),b'private-video-fixture')

    def test_publication_requires_exact_current_asset_bytes_and_supports_ranges(self):
        path=self.root/'data/approved/video.mp4';path.parent.mkdir(parents=True);path.write_bytes(b'approved-video-fixture')
        ref={'path':'data/approved/video.mp4','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
        manifest=self.root/'showcase/public/public-assets.json';manifest.write_text(json.dumps({'format':'interaction-twin-public-assets','version':1,'assets':[ref]}))
        h,calls,sent=self.request('/files/'+ref['path'],headers={'Range':'bytes=1-4'})
        self.assertEqual(sent[0],('status',206));self.assertEqual(h.wfile.getvalue(),b'ppro')
        path.write_bytes(b'changed-video-fixture!')
        _,calls,_=self.request('/files/'+ref['path']);self.assertEqual(calls[0][0],404)
        ref['path']='.secrets/admin-auth.json';manifest.write_text(json.dumps({'format':'interaction-twin-public-assets','version':1,'assets':[ref]}))
        with patch.object(S,'public_static',wraps=S.public_static) as policy:
            _,calls,_=self.request('/files/data/approved/video.mp4');self.assertEqual(calls[0][0],404)

    def test_unfiltered_catalogs_and_private_apis_do_not_leak_without_login(self):
        for relative in ['workspace/cases.js','workspace/cases.json','workspace/cases-index.js','showcase/library-index.js',
                         'showcase/library-catalog.json','showcase/aligned-scenes.json','showcase/library-catalog.json.tmp',
                         'showcase/library-index.json.tmp','showcase/library-catalog-private.js','showcase/private-review.zip',
                         'showcase/public/library-index.json.tmp']:
            path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('private fixture')
            _,calls,_=self.request('/'+relative);self.assertEqual(calls[0][0],404,relative)
        for route in ['/api/config','/api/cases','/api/runs','/api/runs/summary','/api/meta/budget','/api/admin/snapshot','/api/observed/catalog']:
            _,calls,_=self.request(route);self.assertEqual(calls[0][0],401,route)
        _,calls,_=self.request('/files/.private/admin-workflow.sqlite3');self.assertEqual(calls[0][0],404)

    def test_preauth_csrf_login_contract_and_owner_dispatch_identity(self):
        h,calls,_=self.request('/api/admin/auth');self.assertEqual(calls[0][0],200)
        token=A.cookie_token(h.extra_response_headers[0][1]);status=calls[0][1]
        login={'Cookie':f'{A.COOKIE}={token}','X-CSRF-Token':status['csrf_token'],'Origin':'http://127.0.0.1:8765'}
        h,calls,_=self.request('/api/admin/login','POST',{'password':'Test fixture owner password'},headers=login)
        self.assertTrue(calls[0][1]['authenticated']);self.assertIn('HttpOnly',h.extra_response_headers[0][1])
        dispatched=[]
        fake=SimpleNamespace(dispatch=lambda h,r,m,b:dispatched.append((h.admin_actor,r,m,b)) or h.json(200,{'ok':True}) or True)
        with patch.dict(__import__('sys').modules,{'admin_http':fake}):
            _,calls,_=self.request('/api/admin/candidates','POST',{'input_path':'data/private.mp4','processing_consent':True},headers=self.owner())
        self.assertEqual(dispatched[0][0],'nigel');self.assertEqual(calls[0][0],200)

    def test_public_browser_scene_requires_exact_curated_case_reference_and_bytes(self):
        raw=b'{"label":"approved browser scene fixture"}';digest=hashlib.sha256(raw).hexdigest()
        relative='scenes/'+digest+'.json';path=self.root/'showcase/public'/relative
        path.parent.mkdir();path.write_bytes(raw)
        route='/showcase/public/'+relative
        _,denied,_=self.request(route);self.assertEqual(denied[0][0],404)
        index=self.root/'showcase/public/cases-index.json'
        index.write_text(json.dumps({'curated':True,'cases':[{'scene_ref':{'path':relative,'sha256':digest,'bytes':len(raw)}}]}))
        h,calls,sent=self.request(route);self.assertEqual(sent[0],('status',200));self.assertEqual(h.wfile.getvalue(),raw)
        path.write_bytes(raw.replace(b'approved',b'private!'))
        _,denied,_=self.request(route);self.assertEqual(denied[0][0],404)


if __name__=='__main__':unittest.main()
