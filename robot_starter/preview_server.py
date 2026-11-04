#!/usr/bin/env python3
"""Offline, read-only preview for this exact manifest-bound robot starter."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path, PurePosixPath
import re
from urllib.parse import parse_qs, unquote, urlsplit

MAX_TOTAL=96*1024*1024
ARTIFACT_PATHS={
 'kit':'robot-kit.json','schema':'robot-task.schema.json','controller':'showcase/robot-controller.js',
 'task-controller':'showcase/robot-task-plan.js','recorder':'showcase/robot-recorder.js','workbench':'showcase/robot-workbench.html',
 'workbench-runtime':'showcase/robot-workbench.js','workbench-style':'showcase/robot-workbench.css',
 'workbench-navigation':'showcase/robot-workbench-ui-v2.js',
 'scene-contract':'showcase/aligned-scene-contract.js','mark':'showcase/mark.svg',
 'three':'workspace/robot-vendor/three.module.js','three-core':'workspace/robot-vendor/three.core.js',
 'orbit-controls':'workspace/robot-vendor/OrbitControls.js','gltf-loader':'workspace/robot-vendor/GLTFLoader.js',
 'geometry-utils':'workspace/robot-vendor/BufferGeometryUtils.js','three-license':'workspace/robot-vendor/LICENSE',
 'starter-server':'preview_server.py'}

def inventory(root):
    root=Path(root).resolve();path=root/'bundle-manifest.json'
    if path.is_symlink()or not path.is_file()or path.stat().st_size>256*1024:raise ValueError('Exact bounded bundle manifest required')
    raw=path.read_bytes()
    if len(raw)>256*1024:raise ValueError('Manifest changed beyond its bound')
    doc=json.loads(raw)
    if doc.get('format')!='interaction-twin-robot-starter-bundle'or doc.get('version')!=1:raise ValueError('Unsupported starter manifest')
    rows=doc.get('files');files={}
    if not isinstance(rows,list)or not 1<=len(rows)<=96:raise ValueError('Bounded manifest file inventory required')
    total=0
    for row in rows:
        name=row.get('path');size=row.get('bytes');sha=row.get('sha256')
        if not isinstance(name,str)or not name or len(name)>600 or '\\'in name or '%'in name or any(p in {'','.','..'}or p.startswith('.')for p in name.split('/'))or PurePosixPath(name).is_absolute():raise ValueError('Unsafe manifest path')
        if name in files or type(size)is not int or not 0<size<=32*1024*1024 or not isinstance(sha,str)or not re.fullmatch('[a-f0-9]{64}',sha):raise ValueError('Invalid manifest reference')
        target=root/name
        if target.resolve()!=target or any(p.is_symlink()for p in [target,*target.parents])or not target.is_file():raise ValueError('Manifest file unavailable')
        if target.stat().st_size!=size or total+size>MAX_TOTAL:raise ValueError('Manifest size differs or exceeds its bound')
        body=target.read_bytes();total+=len(body)
        if total>MAX_TOTAL or len(body)!=size or hashlib.sha256(body).hexdigest()!=sha:raise ValueError('Manifest file bytes differ')
        files[name]=(sha,size)
    if hashlib.sha256((root/'task-program.json').read_bytes()).hexdigest()!=doc.get('task_program_sha256'):raise ValueError('Task program differs from the exact bundle')
    return doc,files,raw

def handler(root):
    root=Path(root).resolve();manifest,files,manifest_raw=inventory(root)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):self.send_error(405)
        def do_GET(self):self._serve(False)
        def do_HEAD(self):self._serve(True)
        def _serve(self,head):
            host=self.headers.get('Host','').split(':')[0]
            if host not in {'127.0.0.1','localhost'}:self.send_error(403);return
            if len(self.path)>8192:self.send_error(414);return
            parsed=urlsplit(self.path)
            if parsed.scheme or parsed.netloc:self.send_error(400);return
            if parsed.path=='/':
                self.send_response(302);self.send_header('Location','/showcase/robot-workbench.html?bundle=1');self.end_headers();return
            artifact=re.fullmatch(r'/api/robotics/kits/([A-Za-z0-9_-]+)/artifacts/([A-Za-z0-9_-]+)',parsed.path)
            if artifact:
                kit_id,artifact_id=artifact.groups()
                if parsed.query or kit_id!=manifest['kit_id']or artifact_id not in ARTIFACT_PATHS:self.send_error(404);return
                name=ARTIFACT_PATHS[artifact_id]
            elif parsed.path=='/api/robotics/kits/'+manifest['kit_id']:
                if parsed.query:self.send_error(400);return
                name='kit-response.json'
            elif parsed.path=='/api/robotics/kit':
                try:q=parse_qs(parsed.query,keep_blank_values=True,strict_parsing=True,max_num_fields=2)
                except ValueError:self.send_error(400);return
                if set(q)-{'kit_id'}or('kit_id'in q and q['kit_id']!=[manifest['kit_id']]):self.send_error(400);return
                name='kit-response.json'
            else:
                if parsed.query and not(parsed.path=='/showcase/robot-workbench.html'and parsed.query=='bundle=1'):self.send_error(400);return
                path=unquote(parsed.path)
                if '%'in path or '\\'in path or any(p in {'.','..'}or p.startswith('.')for p in path.split('/')if p):self.send_error(404);return
                name=path.removeprefix('/')
                if name.startswith('files/data/'):name=name.removeprefix('files/')
            if name=='bundle-manifest.json':body=manifest_raw
            elif name in files:
                path=root/name;expected_sha,size=files[name]
                if path.resolve()!=path or any(p.is_symlink()for p in [path,*path.parents]):self.send_error(404);return
                try:
                    if path.stat().st_size!=size:self.send_error(409);return
                    body=path.read_bytes()
                except OSError:self.send_error(404);return
                if len(body)!=size or hashlib.sha256(body).hexdigest()!=expected_sha:self.send_error(409);return
            else:self.send_error(404);return
            mime=mimetypes.guess_type(name)[0]or'application/octet-stream'
            if name.endswith('.js'):mime='text/javascript'
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
            if not head:self.wfile.write(body)
    return Handler

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=0);args=parser.parse_args()
    if not 0<=args.port<=65535:parser.error('Port must be between0 and65535')
    root=Path(__file__).resolve().parent
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler(root))
    print('http://127.0.0.1:'+str(server.server_address[1])+'/showcase/robot-workbench.html?bundle=1',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
