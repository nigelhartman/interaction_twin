#!/usr/bin/env python3
"""Extract and audit an owned preview ZIP, using its stdlib-only server.

No production server, model, provider API or credential is accessed. With
--keep-server the isolated read-only process remains for browser verification.
"""
from __future__ import annotations
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import re
import posixpath
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlencode
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not value: raise ValueError(message)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def request(port, route, method='GET', headers=None):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=30)
    try:
        connection.request(method, route, body=b'' if method == 'POST' else None, headers=headers or {})
        response = connection.getresponse(); raw = response.read()
        return response.status, raw, dict(response.getheaders())
    finally: connection.close()


def smoke(build_receipt, receipt, keep_server=False):
    build_receipt, receipt = Path(build_receipt).resolve(), Path(receipt).resolve()
    require(build_receipt.is_relative_to(ROOT / 'output') and receipt.is_relative_to(ROOT / 'output') and not receipt.exists(),
            'Read a local owned build receipt and choose a fresh output receipt')
    build = json.loads(build_receipt.read_text())
    archive_ref = build['archive_ref']; archive = ROOT / archive_ref['path']
    require(archive.resolve() == archive and archive.is_relative_to(ROOT / 'output') and archive.suffix == '.zip'
            and archive.is_file() and archive.stat().st_size <= 512 * 1024 * 1024, 'Choose the owned bounded output ZIP')
    raw = archive.read_bytes()
    require(len(raw) == archive_ref['bytes'] and sha(raw) == archive_ref['sha256'], 'Owned preview ZIP bytes changed')
    isolated = Path(tempfile.mkdtemp(prefix='interaction-twin-judge-isolated-', dir='/private/tmp')).resolve()
    server = None
    successful = False
    started = time.time()
    try:
        with ZipFile(archive) as zipped:
            names = zipped.namelist()
            require(1 <= len(names) <= 4001 and len(names) == len(set(names)), 'Bounded unique ZIP required')
            # Match the measured builder's 768 MiB payload + 2 MiB manifest.
            require(sum(x.file_size for x in zipped.infolist()) <= 770 * 1024 * 1024, 'Bounded ZIP expansion required')
            for item in zipped.infolist():
                require(item.filename.startswith('Interaction-Twin-judge-preview/') and '\\' not in item.filename and
                        not any(p.startswith('.') for p in item.filename.split('/')) and
                        ((item.external_attr >> 16) & 0o170000) != 0o120000,
                        'Unsafe archive member')
            require(zipped.testzip() is None, 'Preview ZIP CRC failed')
            zipped.extractall(isolated)
        folder = isolated / 'Interaction-Twin-judge-preview'
        manifest_path = folder / 'bundle-manifest.json'
        manifest = json.loads(manifest_path.read_text()); files = manifest['files']
        by_path = {item['path']: item for item in files}
        for item in files:
            actual = (folder / item['path']).read_bytes()
            require(len(actual) == item['bytes'] and sha(actual) == item['sha256'], 'Isolated extracted file mismatch')
        log_path = isolated / 'preview-server.log'
        with log_path.open('wb') as log:
            server = subprocess.Popen([sys.executable, '-I', '-S', str(folder / 'preview_server.py'), '--port', '0'],
                cwd=folder, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        port = None
        for _ in range(100):
            require(server.poll() is None, 'Isolated stdlib-only preview did not start')
            found = re.search(r'http://127\.0\.0\.1:(\d+)/', log_path.read_text())
            if found: port = int(found[1]); break
            time.sleep(.2)
        require(port is not None, 'Isolated preview readiness timed out')
        code, body, _ = request(port, '/health')
        health = json.loads(body)
        require(code == 200 and health['read_only'] and health['inference_available'] is False, 'Read-only health contract failed')
        checks = []
        admitted_count = 0
        for item in files:
            if not item['http']: continue
            route = '/files/' + item['path'] if item['path'].startswith('data/') else '/' + item['path']
            code, body, headers = request(port, route)
            require(code == 200 and len(body) == item['bytes'] and sha(body) == item['sha256'],
                    'Frozen HTTP response differs: ' + route)
            require(headers.get('X-Content-Type-Options') == 'nosniff', 'Missing content-type protection')
            if item['path'].startswith('data/') and item['path'].endswith('.html'):
                require(headers.get('Content-Disposition', '').startswith('attachment;') and
                        headers.get('Content-Security-Policy', '').startswith('sandbox;'), 'Historical HTML must stay a sandboxed attachment')
            admitted_count += 1
            if admitted_count % 300 == 0: print(json.dumps({'http_files_verified': admitted_count}), flush=True)
        checks.append('every_declared_http_file_exact_hash_and_length')
        film_html=(folder/'showcase/film.html').read_text();film_match=re.search(r'<source\s+[^>]*src="([^"]+\.mp4)"',film_html)
        require(film_match is not None,'Current film page has no exact MP4 source')
        film_path=posixpath.normpath(posixpath.join('showcase',film_match.group(1)));film=by_path.get(film_path)
        require(film is not None,'HTML-selected current film is missing from frozen inventory')
        route = '/' + film['path']; original = (folder / film['path']).read_bytes()
        code, body, headers = request(port, route, headers={'Range': 'bytes=13-111'})
        require(code == 206 and body == original[13:112] and headers.get('Content-Range') == f'bytes 13-111/{len(original)}', 'Valid video byte range failed')
        require(request(port, route, headers={'Range': 'bytes=999999999-'})[0] == 416, 'Out-of-range video request accepted')
        code, body, headers = request(port, route, 'HEAD')
        require(code == 200 and not body and headers.get('Content-Length') == str(len(original)), 'Exact HEAD contract failed')
        checks += ['video_range_and_head_exact', 'invalid_range_refused']
        physics=manifest.get('frozen_physics_snapshot');require(physics is not None,'Current preview omitted the Newton contact experiment')
        replay=json.loads((folder/physics['replay_ref']['path']).read_bytes());require(len(replay['objects'])==3 and len(replay['frames'])==241,'Frozen Newton sequence incomplete')
        require(request(port,'/showcase/physics-bench.html')[0]==200,'Newton bench page unavailable')
        for ref in [physics['replay_ref'],physics['starter_ref'],*physics['asset_refs']]:
            route='/files/'+ref['path']if ref['path'].startswith('data/')else'/'+ref['path'];code,blob,_=request(port,route);require(code==200 and len(blob)==ref['bytes']and sha(blob)==ref['sha256'],'Frozen Newton dependency differs')
        checks+=['newton_bench_exact_replay_three_meshes_and_starter_zip']
        code, body, _ = request(port, '/api/library')
        library = json.loads(body)
        require(code == 200 and library['total'] == 6 and {r['id'] for r in library['records']} == set(manifest['curated_ids']),
                'Frozen six-record catalog differs')
        code, body, _ = request(port, '/api/library?' + urlencode({'query': 'USB-C'}))
        require(code == 200 and json.loads(body)['total'] == 2, 'Displayed USB-C title search failed')
        require(request(port, '/api/library/facets')[0] == 200, 'Read-only facets unavailable')
        checks += ['six_exact_curated_records', 'shared_library_search_and_facets']
        negatives = ['/admin', '/api/admin/auth', '/api/admin/snapshot', '/api/upload', '/api/config',
            '/api/meta/budget', '/preview_server.py', '/library_query.py', '/.secrets/fixture-private.env',
            '/files/.private/fixture-owner.json', '/files/data/../private.json',
            '/files/data/%252e%252e/private.json', '/showcase/demos/charge-phone-v1/private.json','/showcase/media/private.zip',
            '/showcase/demos/charge-phone-v1/model-prompt.json', '/api/robotics/execute']
        for route in negatives: require(request(port, route)[0] == 404, 'Private/unsupported route became available: ' + route)
        require(request(port, '/api/upload', 'POST')[0] == 405, 'Frozen preview accepted upload mutation')
        checks += ['private_owner_and_runtime_routes_absent', 'mutations_unavailable']
        index = json.loads((folder / 'showcase/demos/robotics/index.json').read_text())
        code, body, _ = request(port, '/api/robotics/kit?' + urlencode({'kit_id': index['kit_id']}))
        require(code == 200 and json.loads(body)['snapshot_mode'] is True, 'Frozen robot kit unavailable')
        for task in index['tasks']:
            params = urlencode({'kit_id': index['kit_id'], 'twin_id': task['twin_id'], 'instruction': task['instruction']})
            code, body, _ = request(port, '/api/robotics/plan?' + params)
            require(code == 200 and json.loads(body) == json.loads((folder / task['task']['path']).read_text()),
                    'Saved robot baseline API fields differ')
            code, body, _ = request(port, '/api/robotics/bundle?' + params)
            require(code == 200 and sha(body) == task['starter']['sha256'], 'Saved robot starter download differs')
        bad = urlencode({'kit_id': index['kit_id'], 'twin_id': index['tasks'][0]['twin_id'], 'instruction': 'Do an invented physical task.'})
        require(request(port, '/api/robotics/plan?' + bad)[0] == 404, 'Frozen server invented a new task')
        checks += ['saved_robot_kit_and_exact_task_and_starter_endpoints', 'unsupported_server_planning_refused']
        node = shutil.which('node'); require(node, 'Existing Node runtime needed for packaged local compiler check')
        source = r"""
const fs=require('node:fs'),path=require('node:path'),root=process.argv[1],read=p=>JSON.parse(fs.readFileSync(path.join(root,p))),C=require(path.join(root,'showcase/robot-controller.js')),A=require(path.join(root,'showcase/aligned-scene-contract.js')),P=require(path.join(root,'showcase/robot-task-plan.js')),index=read('showcase/demos/robotics/index.json'),kit=read(index.kit.path),results=[];
for(const row of index.tasks){const baseline=read(row.task.path),program=read(baseline.source_binding.program_ref.path),descriptive=baseline.source_binding.object_program_ref?read(baseline.source_binding.object_program_ref.path):null;if(P.sourceBinding(baseline,program,descriptive)!==true)throw Error('Actual three-argument browser source binding failed');const texts=row.task.path.includes('/jabra/')?[row.instruction,'Open the lid and remove earbud_a to the tray.','Return earbud_a from the tray to the open case, then close it.']:['Move gold_coin to the tray.','Move gold_coin to the table.'];for(const text of texts){const task=P.compileInstruction(text,baseline,kit);P.validate(task,baseline,kit);const timeline=C.compile(P.translate(task,kit),{joint_tool_position:(step,q,state)=>P.jointToolPosition(task,step,q,state)});P.checkOutcome(task,timeline);for(let i=0;i<=100;i++){const state=C.at(timeline,timeline.duration_s*i/100);if(!state.ik.reachable||!state.ik.within_limits)throw Error('Packaged task lost reachability');}results.push({case_id:row.twin_id,instruction:text,actions:task.actions.length,authored_rehearsal:task.claims.authored_rehearsal,simulation_ready:task.claims.simulation_ready});}const wrong=JSON.parse(JSON.stringify(baseline));wrong.claims.simulation_ready=true;let rejected=false;try{P.validate(wrong,baseline,kit)}catch(_){rejected=true}if(!rejected)throw Error('Unsafe physical claim accepted');}console.log(JSON.stringify(results));
"""
        local_compiler = subprocess.run([node, '-e', source, str(folder)], cwd=isolated, capture_output=True, text=True, timeout=60, check=True)
        compiler_results = json.loads(local_compiler.stdout)
        checks += ['actual_three_argument_browser_source_binding_for_each_snapshot','packaged_local_instruction_compiler_and_101_state_joint_checks', 'false_physical_readiness_claim_refused']
        result = {'format': 'interaction-twin-isolated-judge-preview-smoke', 'version': 1, 'status': 'passed',
            'archive_ref': archive_ref, 'isolated_root': str(folder), 'stdlib_only_launch': True,
            'launch_flags': ['-I', '-S'], 'url': f'http://127.0.0.1:{port}/',
            'server_pid': server.pid if keep_server else None, 'server_retained_for_browser_qa': keep_server,
            'exact_extracted_files': len(files), 'exact_http_files_verified': admitted_count,
            'total_file_bytes': manifest['total_file_bytes'], 'checks': checks,'current_film_path':film_path,
            'newton_snapshot':{'replay_ref':physics['replay_ref'],'starter_ref':physics['starter_ref'],'native_meshes':len(physics['asset_refs']),'saved_frames':len(replay['frames'])},
            'privacy_negative_routes_checked': len(negatives), 'local_compiler_results': compiler_results,
            'new_api_calls': 0, 'new_gpu_calls': 0, 'production_server_changed': False,
            'physical_accuracy_verified': False, 'simulation_ready': False,
            'browser_pixels_verified_by_this_script': False, 'elapsed_s': round(time.time() - started, 3)}
        receipt.parent.mkdir(parents=True, exist_ok=True)
        with receipt.open('x') as stream: json.dump(result, stream, indent=2); stream.write('\n')
        successful = True
        print(json.dumps({k: result[k] for k in ['status', 'url', 'server_pid', 'exact_http_files_verified', 'elapsed_s']}))
        return result
    finally:
        if server and not (keep_server and successful):
            server.terminate()
            try: server.wait(timeout=10)
            except subprocess.TimeoutExpired: server.kill(); server.wait(timeout=5)
        if not (keep_server and successful): shutil.rmtree(isolated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-receipt', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--keep-server', action='store_true')
    args = parser.parse_args()
    smoke(args.build_receipt, args.receipt, args.keep_server)
