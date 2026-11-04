#!/usr/bin/env python3
"""Isolated, read-only owner-desk QA. Public saved assets; synthetic history/session.

No owner store, credentials, models, subprocesses or real service are used.
All mutation routes return 405. This server always binds to loopback.
"""
from __future__ import annotations
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'output/submission-final-polish-20261104-v1/admin-review-fixture-v1'
CANDIDATE='candidate_'+'a'*32
REV_BEFORE='revision_'+'b'*32
REV_AFTER='revision_'+'c'*32
LABEL='QA fixture — not the owner service'


def read(path):
    return json.loads(path.read_text())


def ref(path):
    raw=path.read_bytes()
    return {'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def prepare(destination):
    required=[ROOT/'showcase/public/public-assets.json',ROOT/'showcase/public/library-catalog.json',ROOT/'showcase/public/aligned-scenes.json']
    if any(not path.is_file() for path in required):
        raise ValueError('The read-only admin QA fixture requires the separate saved public catalog and approved data artifacts; the source-only checkout does not include them.')
    destination=destination.resolve()
    if not destination.is_relative_to(ROOT/'output'):
        raise ValueError('Fixture output must stay under the project output directory')
    destination.mkdir(parents=True,exist_ok=True)
    public={r['path']:r for r in read(ROOT/'showcase/public/public-assets.json')['assets']}
    catalog=read(ROOT/'showcase/public/library-catalog.json')
    record=next(r for r in catalog['records'] if any(p.get('id')=='case_base' for p in r.get('components',[])))
    registry=read(ROOT/'showcase/public/aligned-scenes.json')
    row=next(r for r in registry['scenes'] if r['source_sha256']==record['source']['sha256'])
    program=read(ROOT/record['interaction']['path'])
    declaration=read(ROOT/row['derived_assembly']['path'])
    before_ref=declaration.get('baseline_scene_ref')
    if not before_ref:
        before_ref=read(ROOT/row['object_program']['path'])['constrained_scene']
    after_ref=row['manifest']
    before_scene=read(ROOT/before_ref['path'])
    after_scene=read(ROOT/after_ref['path'])
    if before_scene['source']['sha256']!=program['source']['sha256'] or after_scene['source']['sha256']!=program['source']['sha256']:
        raise ValueError('Fixture scenes must share the exact approved public source clock')
    if before_scene['original_program']!=record['interaction'] or after_scene['original_program']!=record['interaction']:
        raise ValueError('Fixture scenes must bind the same exact source program')
    source_ref=public[program['source']['path']]
    native=[public[a['path']] for a in program.get('assets',[]) if a.get('backend')=='Meta SAM 3D Objects']
    allowed={}
    def collect(value):
        if isinstance(value,dict):
            path=value.get('path');expected=value.get('sha256')
            if isinstance(path,str) and path in public and isinstance(expected,str):
                admitted=public[path]
                if expected!=admitted['sha256'] or value.get('bytes',admitted['bytes'])!=admitted['bytes']:
                    raise ValueError('A referenced public artifact changed')
                if path not in allowed:
                    target=(ROOT/path).resolve()
                    if not target.is_relative_to(ROOT/'data') or target.is_symlink() or ref(target)!=admitted:
                        raise ValueError('Only exact admitted data files may be served')
                    allowed[path]=admitted
                    if target.suffix=='.json' and admitted['bytes']<=8*1024*1024:
                        collect(read(target))
            for child in value.values():collect(child)
        elif isinstance(value,list):
            for child in value:collect(child)
    for value in [record['interaction'],row,before_ref,source_ref,*native]:collect(value)
    artifacts=list(allowed.values())
    def revision(number,id,scene_ref):
        manifest={'format':'interaction-twin-private-candidate-revision','version':1,'candidate_id':CANDIDATE,'revision_id':id,'number':number,'run_id':'QA-public-source-fixture','source_ref':source_ref,'interaction_ref':record['interaction'],'object_program_ref':row['object_program'],'scene_ref':scene_ref,'artifacts':artifacts,'base_revision_id':REV_BEFORE if number==2 else None,'job_id':'job_fixture','memory_context_sha256':'0'*64,'human_guidance_used':False,'training_performed':False,'memory_only_not_weight_training':True,'repair_execution':[],'pipeline_status':'QA_fixture','created_utc':'2026-11-04T20:00:00Z','physical_accuracy_verified':False,'simulation_ready':False,'review':{'roles':[{'role':name,'status':'pass','checks':['QA fixture contract; no actual new review was run.'],'issues':[]} for name in ['evidence','geometry','mechanics','semantics']],'all_required_roles_completed':True,'role_count':4,'blocking_failures':0,'hosted_roles':[],'hosted_review_enabled':False},'fixture_only':True}
        fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return {'id':id,'number':number,'fingerprint':fingerprint,'manifest':manifest,'created_utc':manifest['created_utc']}
    before=revision(1,REV_BEFORE,before_ref);after=revision(2,REV_AFTER,after_ref)
    candidate={'id':CANDIDATE,'current_revision_id':REV_AFTER,'source_sha256':source_ref['sha256'],'status':'awaiting_review','run_id':'QA-public-source-fixture','title':'QA fixture · Earbuds (public saved assets)','created_utc':'2026-11-04T20:00:00Z','updated_utc':'2026-11-04T20:00:00Z'}
    detail={'candidate':candidate,'revisions':[after,before],'jobs':[],'repair_plans':[],'decisions':[],'feedback':[],'improvement_rounds':[],'improvement_stops':[],'best_revision_id':REV_AFTER,'fixture_only':True}
    files={}
    for folder in ['admin','workspace/robot-vendor']:
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and not path.is_symlink() and path.suffix in {'.html','.js','.css','.svg','.png','.ico'}:
                relative=path.relative_to(ROOT);target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);files['/'+relative.as_posix()]=target
    for path in (ROOT/'showcase').glob('*'):
        if path.is_file() and path.suffix in {'.html','.js','.css','.svg','.png','.ico'}:
            relative=path.relative_to(ROOT);target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);files['/'+relative.as_posix()]=target
    for path in (ROOT/'showcase/brand/v2').rglob('*'):
        if path.is_file() and path.suffix in {'.svg','.png','.css'}:
            relative=path.relative_to(ROOT);target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);files['/'+relative.as_posix()]=target
    for relative,value in [('showcase/public/aligned-scenes.json',{**registry,'scenes':[row]}),('showcase/public/library-catalog.json',{**catalog,'records':[record]})]:
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(value));files['/'+relative]=target
    index=destination/'admin/index.html'
    index.write_text(index.read_text().replace('<body>','<body><div style="background:#fff3cf;color:#5b470e;text-align:center;padding:10px 18px;font:12px system-ui;border-bottom:1px solid #edda9e"><strong>'+LABEL+'</strong> · Public saved assets, synthetic revision history/session. All writes and model calls are disabled.</div>',1))
    metadata={'fixture_only':True,'label':LABEL,'history':'Synthetic revision IDs over two exact previously saved public scene snapshots; not real owner decisions.','source_ref':source_ref,'before_scene_ref':before_ref,'after_scene_ref':after_ref,'data_refs':list(allowed.values()),'post_policy':'All POST routes return 405; no request writes a file or runs a model.','copied_frontend_files':sorted(files)}
    (destination/'fixture-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (destination/'fixture-detail.json').write_text(json.dumps(detail,indent=2)+'\n')
    return {'destination':destination,'files':files,'data':{path:ROOT/path for path in allowed},'detail':detail,'record':record,'metadata':metadata}


def handler_for(fixture):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def json_response(self,value,code=200):
            raw=json.dumps(value).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.end_headers()
            if self.command!='HEAD':self.wfile.write(raw)
        def do_POST(self):
            self.json_response({'error':LABEL+': writes, model calls and real owner actions are disabled.','fixture_only':True},405)
        def do_HEAD(self):self.do_GET()
        def do_GET(self):
            route=unquote(urlsplit(self.path).path)
            if route in {'/','/admin','/admin/'}:route='/admin/index.html'
            candidate=fixture['detail']['candidate']
            if route=='/api/admin/auth':return self.json_response({'authenticated':True,'configured':True,'setup_available':False,'csrf_token':'Q'*43,'fixture_only':True})
            if route=='/api/admin/snapshot':return self.json_response({'candidates':[candidate],'jobs':[],'budget':{'available':False},'runtime':{'configured':False},'fixture_only':True})
            if route=='/api/admin/candidates/'+CANDIDATE:return self.json_response(fixture['detail'])
            if route=='/api/admin/memory':return self.json_response({'entries':[],'revoked_ids':[],'fixture_only':True})
            if route=='/api/admin/operations':return self.json_response({'budget':{'status':'unavailable'},'gpu':{'status':'fixture_no_gpu'},'processing':{'status':'available','counts_by_state':{}},'marketing':{'status':'available','counts_by_state':{}},'fixture_only':True})
            if route=='/api/library':return self.json_response({'records':[fixture['record']],'fixture_only':True})
            if route.startswith('/api/'):return self.json_response({'error':'Not a QA fixture route.','fixture_only':True},404)
            target=fixture['files'].get(route)
            if route.startswith('/files/') and route[7:] in fixture['data']:target=fixture['data'][route[7:]]
            if not target or not target.is_file():return self.json_response({'error':'Not an allowlisted QA fixture file.'},404)
            size=target.stat().st_size;start=0;end=size-1;status=200;range_header=self.headers.get('Range')
            if range_header:
                match=re.fullmatch(r'bytes=(\d+)-(\d*)',range_header)
                if not match:return self.json_response({'error':'Unsupported range'},416)
                start=int(match[1]);end=min(size-1,int(match[2]) if match[2] else size-1)
                if start>end or start>=size:return self.json_response({'error':'Range outside fixture'},416)
                status=206
            self.send_response(status);self.send_header('Content-Type',mimetypes.guess_type(target.name)[0] or 'application/octet-stream');self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes');self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            if self.command=='HEAD':return
            with target.open('rb') as stream:
                stream.seek(start);remaining=end-start+1
                while remaining:
                    block=stream.read(min(65536,remaining))
                    if not block:break
                    self.wfile.write(block);remaining-=len(block)
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--port',type=int,default=0);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    if args.port and not 1024<=args.port<=65535:parser.error('Use an unprivileged loopback port or zero for a random port')
    fixture=prepare(args.output)
    print(json.dumps({'fixture_only':True,'label':LABEL,'output':str(fixture['destination']),'public_data_files':len(fixture['data']),'before_scene':fixture['metadata']['before_scene_ref'],'after_scene':fixture['metadata']['after_scene_ref']}),flush=True)
    if args.prepare_only:return
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(fixture));print(f'QA fixture URL: http://127.0.0.1:{server.server_address[1]}/admin/',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
