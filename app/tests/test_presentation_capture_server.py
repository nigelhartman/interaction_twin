"""The local production server exposes only frozen film inputs, never a folder."""
import hashlib,json,tempfile,threading,unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import build_opener,ProxyHandler,Request
from scripts.serve_presentation_capture import load_configuration,handler

class PresentationServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.folder=self.root/'output/capture';self.folder.mkdir(parents=True)
        self.model=self.folder/'lid.glb';self.model.write_bytes(b'LOCAL SYNTHETIC MODEL')
        self.manifest=self.folder/'inputs.json';self.manifest.write_text('{"synthetic":true}')
        self.config=self.folder/'server.json'
        self.value={'format':'interaction-twin-local-presentation-server','version':1,'routes':{'/robot-capture-manifest.json':self.ref(self.manifest)},'files':[self.ref(self.manifest),self.ref(self.model)]}
        self.save()
    def ref(self,path):
        raw=path.read_bytes();return{'path':path.relative_to(self.root).as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    def save(self):self.config.write_text(json.dumps(self.value))
    def test_actual_loopback_only_frozen_routes_head_and_tamper(self):
        routes=load_configuration(self.config,self.root);server=ThreadingHTTPServer(('127.0.0.1',0),handler(routes,self.root));worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        self.addCleanup(lambda:(server.shutdown(),server.server_close(),worker.join(2)))
        opener=build_opener(ProxyHandler({}));origin='http://127.0.0.1:'+str(server.server_port)
        for path in ['/robot-capture-manifest.json','/files/output/capture/inputs.json','/files/output/capture/lid.glb']:
            with opener.open(origin+path,timeout=3)as response:self.assertEqual(response.status,200);self.assertTrue(response.read());self.assertEqual(response.headers['Cache-Control'],'no-store')
        with opener.open(Request(origin+'/robot-capture-manifest.json',method='HEAD'),timeout=3)as response:self.assertEqual(response.read(),b'');self.assertGreater(int(response.headers['Content-Length']),0)
        for path in ['/.private/admin.sqlite3','/files/output/unlisted.json','/%2e%2e/.secrets/meta.env','/robot-capture-manifest.json?extra=1']:
            with self.subTest(path=path),self.assertRaises(HTTPError)as caught:opener.open(origin+path,timeout=3)
            self.assertIn(caught.exception.code,[400,404])
        self.model.write_bytes(b'TAMPERED')
        with self.assertRaises(HTTPError)as caught:opener.open(origin+'/files/output/capture/lid.glb',timeout=3)
        self.assertEqual(caught.exception.code,409)
    def test_configuration_rejects_private_code_duplicate_refs_and_symlinks(self):
        cases=[{**self.value,'routes':{'/api/admin':self.ref(self.manifest)}},{**self.value,'files':[self.ref(self.model),self.ref(self.model)]}]
        private=self.root/'.private/credential.json';private.parent.mkdir();private.write_text('{}');cases.append({**self.value,'files':[self.ref(private)]})
        for value in cases:
            self.config.write_text(json.dumps(value))
            with self.subTest(value=value),self.assertRaises(ValueError):load_configuration(self.config,self.root)
        self.save();raw=self.model.read_bytes();self.model.unlink();other=self.folder/'other.glb';other.write_bytes(raw);self.model.symlink_to(other)
        with self.assertRaises(ValueError):load_configuration(self.config,self.root)

if __name__=='__main__':unittest.main()
