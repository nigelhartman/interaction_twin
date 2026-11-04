#!/usr/bin/env python3
"""Short-lived exact allowlist for local screenshots of the saved MCP example."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import hashlib,json,mimetypes,os
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/submission-finalization-20261103/walkthrough'
names=['index.html','demo.js','demo.css','task.json','trace.json','summary.json','muse-review.json']
paths=['showcase/demos/charge-phone-v1/'+n for n in names]+['showcase/style.css','showcase/cobalt.css','showcase/brand/v2/mark.svg']
task=json.loads((ROOT/'showcase/demos/charge-phone-v1/task.json').read_text())
for source in task['sources']:
 interface=next(i for i in source['interfaces']if i['id']==('phone_port'if source['role']=='phone'else'cable_free_end'))
 paths.append(interface['evidence'][0]['frame_url'].removeprefix('/files/'))
routes={}
for name in paths:
 p=ROOT/name
 if p.resolve()!=p or not p.is_file()or any(x.startswith('.')for x in p.relative_to(ROOT).parts):raise ValueError('Only explicit ordinary public inputs allowed')
 route='/files/'+name if name.startswith('data/')else'/'+name
 routes[route]={'path':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  route=urlsplit(self.path).path;ref=routes.get(route)
  if ref is None:self.send_error(404);return
  p=ROOT/ref['path'];data=p.read_bytes()
  if hashlib.sha256(data).hexdigest()!=ref['sha256']:self.send_error(409);return
  self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(p)[0]or'application/octet-stream');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
receipt={'format':'exact_allowlist_submission_capture_server','pid':os.getpid(),'host':'127.0.0.1','port':server.server_port,'routes':routes,'no_api_or_private_routes':True}
with (OUT/'capture-server-v2.json').open('x')as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({'pid':os.getpid(),'url':f'http://127.0.0.1:{server.server_port}/showcase/demos/charge-phone-v1/index.html','routes':len(routes)}),flush=True)
server.serve_forever()
