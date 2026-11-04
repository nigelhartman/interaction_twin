#!/usr/bin/env python3
"""Serve one frozen local film-capture allowlist, never the project directory."""
import argparse
import hashlib
import json
import mimetypes
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
RUNTIME={
 'showcase/robot-task-capture-v2.html','showcase/robot-task-capture-v2.js','showcase/robot-presentation-v2.js',
 'showcase/robot-task-capture-v3.html','showcase/robot-task-capture-v3.js','showcase/robot-camera-fit-v1.js',
 'showcase/brand-refined-capture-v1.html','showcase/brand-refined-capture-v1.js',
 'showcase/robot-controller.js','showcase/robot-task-plan.js','showcase/aligned-scene-contract.js',
 'workspace/robot-vendor/three.module.js','workspace/robot-vendor/three.core.js',
 'workspace/robot-vendor/OrbitControls.js','workspace/robot-vendor/GLTFLoader.js','workspace/robot-vendor/BufferGeometryUtils.js',
}
MAX_FILE=32*1024*1024

def checked(reference,root=ROOT):
    root=Path(root).resolve()
    if (not isinstance(reference,dict)or set(reference)!={'path','sha256','bytes'}or
        not isinstance(reference['path'],str)or'\\'in reference['path']or
        type(reference['bytes'])is not int or not 0<reference['bytes']<=MAX_FILE or
        not isinstance(reference['sha256'],str)or len(reference['sha256'])!=64 or
        any(c not in'0123456789abcdef'for c in reference['sha256'])):raise ValueError('Exact bounded capture reference required')
    relative=Path(reference['path'])
    if relative.is_absolute()or any(p in {'','..','.'}or p.startswith('.')for p in relative.parts):raise ValueError('Ordinary capture path required')
    path=root/relative
    if (not path.is_relative_to(root)or path.resolve()!=path or any(p.is_symlink()for p in[path,*path.parents])or
        not path.is_file()or path.stat().st_size!=reference['bytes']):raise ValueError('Capture reference is unavailable or changed')
    raw=path.read_bytes()
    if len(raw)!=reference['bytes']or hashlib.sha256(raw).hexdigest()!=reference['sha256']:raise ValueError('Capture input bytes changed')
    return raw

def load_configuration(path,root=ROOT):
    root=Path(root).resolve();path=Path(path)
    if not path.is_absolute():path=root/path
    if not path.is_relative_to(root/'output')or path.resolve()!=path or any(p.is_symlink()for p in[path,*path.parents])or not 0<path.stat().st_size<=128000:raise ValueError('Bounded ordinary output capture configuration required')
    value=json.loads(path.read_bytes())
    if (set(value)!={'format','version','routes','files'}or value['format']!='interaction-twin-local-presentation-server'or type(value['version'])is not int or value['version']!=1 or
        not isinstance(value['files'],list)or not 1<=len(value['files'])<=48 or not isinstance(value['routes'],dict)or not 1<=len(value['routes'])<=2):raise ValueError('Unknown local capture configuration')
    routes={};total=0
    for ref in value['files']:
        name=ref.get('path')if isinstance(ref,dict)else None
        if name in RUNTIME:route='/'+name
        elif isinstance(name,str)and name.startswith(('data/','output/'))and Path(name).suffix.lower()in{'.json','.glb'}:route='/files/'+name
        else:raise ValueError('Only fixed capture runtime and exact JSON/GLB inputs are allowed')
        checked(ref,root);total+=ref['bytes']
        if total>128*1024*1024 or route in routes:raise ValueError('Bounded unique capture inventory required')
        routes[route]=ref
    for route,ref in value['routes'].items():
        if route not in{'/robot-capture-manifest.json','/native-capture-manifest.json'}or ref not in value['files']:raise ValueError('Only pinned robot/native manifests may have named routes')
        routes[route]=ref
    return routes

def handler(routes,root=ROOT):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_args):pass
        def do_GET(self):self.respond(False)
        def do_HEAD(self):self.respond(True)
        def respond(self,head):
            parts=urlsplit(self.path)
            if parts.query or parts.fragment or'%'in parts.path or len(self.path)>2048:self.send_error(400);return
            ref=routes.get(parts.path)
            if not ref:self.send_error(404);return
            try:raw=checked(ref,root)
            except(ValueError,OSError):self.send_error(409,'Frozen capture input changed');return
            mime=mimetypes.guess_type(ref['path'])[0]or'application/octet-stream'
            if ref['path'].endswith('.js'):mime='text/javascript'
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(raw)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Referrer-Policy','no-referrer');self.end_headers()
            if not head:self.wfile.write(raw)
    return Handler

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('configuration',type=Path);parser.add_argument('--port',type=int,default=0);args=parser.parse_args()
    if not 0<=args.port<=65535:parser.error('Use a valid loopback port')
    routes=load_configuration(args.configuration);server=ThreadingHTTPServer(('127.0.0.1',args.port),handler(routes));
    print(f'http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
