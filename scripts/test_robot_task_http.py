#!/usr/bin/env python3
"""Actual edited-task download smoke for the public read-only robot adapter.

Starts only an owned ephemeral loopback transport; never the owner/admin server,
models, network services, inference, scheduler or hardware control.
"""
import argparse
import copy
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import sys
import threading
import urllib.request
from urllib.parse import urlsplit
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import public_robotics_api as API


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    catalog=API._catalog(ROOT)
    twin='flow-sam3d-default-e2e-20261102-v1--flow-procedure-1'
    task=catalog.plan(twin,'Place the case on the table, open the lid, take out both earbuds and place them on the tray, then close it.')
    original=catalog.export_bundle(task)
    edited=copy.deepcopy(task);edited['actions'][0]['duration_s']+=.25;edited['actions'][0]['title']='Agent edited approach'
    assert catalog.validate(edited)['valid'] is True
    expected=catalog.export_bundle(edited);assert expected['artifact']['sha256']!=original['artifact']['sha256']
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            parsed=urlsplit(self.path);value=API.response(parsed.path,parsed.query,root=ROOT)
            if value is None:self.send_error(404);return
            status,body,mime,headers=value
            self.send_response(status);self.send_header('Content-Length',str(len(body)));self.send_header('Content-Type',mime)
            for key,value in headers.items():self.send_header(key,value)
            self.end_headers();self.wfile.write(body)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        url='http://127.0.0.1:'+str(server.server_address[1])+expected['artifact']['url']
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url,timeout=30)as response:
            assert response.headers['Content-Type']=='application/zip';body=response.read()
        assert len(body)==expected['artifact']['bytes']and hashlib.sha256(body).hexdigest()==expected['artifact']['sha256']
        with zipfile.ZipFile(BytesIO(body))as archive:
            exact=json.loads(archive.read('Interaction-Twin-Robot-Starter/task-program.json'));assert exact==edited and exact['source_binding']==task['source_binding']
        report={'format':'interaction-twin-edited-task-http-smoke','version':1,'status':'passed',
            'original_zip_sha256':original['artifact']['sha256'],'edited_zip_sha256':expected['artifact']['sha256'],
            'edited_zip_bytes':len(body),'exact_edited_json_downloaded':True,'source_binding_unchanged':True,
            'transport':'actual owned ephemeral loopback HTTP using public_robotics_api.response',
            'network_scope':'localhost only','new_api_calls':0,'gpu_calls':0,'physical_accuracy_verified':False,'hardware_execution_available':False}
    finally:server.shutdown();server.server_close();thread.join(timeout=5)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x')as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report))
if __name__=='__main__':main()
