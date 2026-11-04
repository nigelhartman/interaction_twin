#!/usr/bin/env python3
"""Verify one trusted exported starter with isolated Python and owned loopback HTTP."""
import argparse
import hashlib
import json
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('zip',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    archive_path=args.zip.resolve();raw=archive_path.read_bytes()
    if len(raw)>96*1024*1024:raise ValueError('Bounded starter ZIP required')
    report={'format':'interaction-twin-robot-starter-http-smoke','version':1,'zip_sha256':hashlib.sha256(raw).hexdigest(),'network_scope':'owned ephemeral loopback only','new_api_calls':0,'gpu_calls':0}
    with tempfile.TemporaryDirectory(prefix='robot-starter-check-')as temporary:
        with zipfile.ZipFile(archive_path)as archive:
            if len(archive.infolist())>97 or sum(row.file_size for row in archive.infolist())>97*1024*1024:raise ValueError('Bounded declared archive required')
            for row in archive.infolist():
                path=Path(row.filename)
                if path.is_absolute()or'..'in path.parts or not row.filename.startswith('Interaction-Twin-Robot-Starter/'):raise ValueError('Unsafe archive entry')
            archive.extractall(temporary)
        base=Path(temporary)/'Interaction-Twin-Robot-Starter'
        node=shutil.which('node')
        if node:
            javascript=r"""
const fs=require('fs'),p=require('path'),b=process.argv[1];
const P=require(p.join(b,'showcase/robot-task-plan.js')),C=require(p.join(b,'showcase/robot-controller.js'));require(p.join(b,'showcase/aligned-scene-contract.js'));
const initial=JSON.parse(fs.readFileSync(p.join(b,'task-program.json'),'utf8')),kit=JSON.parse(fs.readFileSync(p.join(b,'robot-kit.json'),'utf8'));
const edited=JSON.parse(JSON.stringify(initial));edited.actions[0].duration_s+=.25;edited.actions[0].title='Edited JSON preview';
const checked=P.validate(edited,initial,kit),options=typeof P.jointToolPosition==='function'?{joint_tool_position:(step,q,state)=>P.jointToolPosition(checked,step,q,state)}:{},compiled=C.compile(P.translate(checked,kit),options);P.checkOutcome(checked,compiled);
if(checked.actions[0].title!=='Edited JSON preview'||JSON.stringify(checked.source_binding)!==JSON.stringify(initial.source_binding))throw Error('Editor did not preserve the exact binding/edit');
const invalid=JSON.parse(JSON.stringify(edited));invalid.objects[0].asset_ref.path='data/private/never-published.glb';let rejected=false;
try{P.validate(invalid,initial,kit);}catch{rejected=true;}if(!rejected)throw Error('Editor accepted substituted geometry');
console.log(JSON.stringify({edited_task_compiled:true,edited_action_title_preserved:true,source_binding_unchanged:true,foreign_geometry_rejected:true,actions:checked.actions.length,duration_s:compiled.duration_s}));
"""
            result=subprocess.run([node,'-e',javascript,str(base)],capture_output=True,text=True,timeout=20)
            if result.returncode:raise ValueError('Packaged editor/controller rejected edited task: '+result.stderr[-2000:])
            report['packaged_editor_logic']=json.loads(result.stdout)
        else:report['packaged_editor_logic']={'status':'not_checked_node_unavailable','node_not_required_for_starter':True}
        process=subprocess.Popen([sys.executable,'-I','-S',str(base/'preview_server.py'),'--port','0'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,cwd=base)
        try:
            if not select.select([process.stdout],[],[],10)[0]:raise ValueError('Preview server startup timed out')
            url=process.stdout.readline().strip()
            if not url.startswith('http://127.0.0.1:'):raise ValueError('Unexpected local starter startup: '+process.stderr.read()[-2000:])
            origin=url.split('/showcase/')[0];opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));verified=[]
            for row in json.loads((base/'bundle-manifest.json').read_text())['files']:
                target=origin+('/files/'if row['path'].startswith('data/')else'/')+row['path']
                with opener.open(target,timeout=10)as response:body=response.read()
                assert len(body)==row['bytes']and hashlib.sha256(body).hexdigest()==row['sha256'];verified.append(row['path'])
            envelope=json.loads((base/'kit-response.json').read_text());resources=0
            assert envelope['resource_url_scope']=='packaged_manifest_files'and envelope['mcp_resources_available']is False
            for ref in envelope['resources'].values():
                assert ref['url'].startswith('/')and not ref['url'].startswith('/api/')and ref['uri'].startswith('twin://robot-kits/')
                with opener.open(origin+ref['url'],timeout=10)as response:body=response.read()
                assert len(body)==ref['bytes']and hashlib.sha256(body).hexdigest()==ref['sha256'];resources+=1
            report['advertised_packaged_resource_urls_verified']=resources
            for target in ['/api/robotics/plan?instruction=anything','/.private/admin-workflow.sqlite3','/unlisted.txt','/%2e%2e/secret']:
                try:opener.open(origin+target,timeout=10)
                except urllib.error.HTTPError as error:assert error.code in [400,404]
                else:raise ValueError('Undeclared path was exposed')
            with opener.open(urllib.request.Request(url,method='HEAD'),timeout=10)as response:assert response.read()==b''and int(response.headers['Content-Length'])>0
            report.update(status='passed',verified_files=len(verified),unlisted_private_paths_rejected=True,head_body_empty=True,isolated_python_no_site=True)
        finally:
            process.terminate();process.wait(timeout=5)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x')as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report))
if __name__=='__main__':main()
