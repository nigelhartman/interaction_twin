#!/usr/bin/env python3
"""Export a backend-free, project-subpath-safe demo. No publishing or inference.

Only current curated, owner-recorded examples are exported. Artifact bytes stay
unchanged; deployment URL changes apply to copied viewer code only. A closed
reference walk is bounded by the existing public asset allowlist.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_judge_preview as J

CAMERA = 'flow-meta-video-camera-20261030-v1--flow-procedure-1'
FROZEN = ROOT / 'output/final-quality-20261104-v1/packages-v1/judge-preview-final-v1/Interaction-Twin-judge-preview'

def encoded(v): return (json.dumps(v, separators=(',', ':'), ensure_ascii=False, allow_nan=False)+'\n').encode()
def identity(path, raw): return {'path': path, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
def strings(v):
    if isinstance(v, str): yield v
    elif isinstance(v, dict):
        for x in v.values(): yield from strings(x)
    elif isinstance(v, list):
        for x in v: yield from strings(x)

def closure(seeds, approved, root=ROOT):
    pending = list(strings(seeds)); found = {}; omitted = []
    while pending:
        path = pending.pop().removeprefix('/files/')
        if path not in approved or path in found: continue
        pointer = approved[path]
        if path.endswith('.html') or J.omitted_provenance(pointer):
            if path not in omitted: omitted.append(path)
            continue
        J.check(root / path, pointer, root=root)
        found[path] = pointer
        if path.endswith('.json'):
            pending.extend(strings(json.loads((root / path).read_bytes())))
    return found, sorted(omitted)

def rewrite_viewer(text, base, name):
    # These are URL literals in trusted copied viewer code, not saved data.
    for prefix in ('/files/', '/workspace/', '/showcase/'):
        for quote in ('"', "'", '`'):
            text = text.replace(quote+prefix, quote+base+prefix.lstrip('/'))
        text = text.replace('${origin}'+prefix, '${origin}'+base+prefix.lstrip('/'))
    if name.endswith('agent-charge-example.js'):
        # Validate the original evidence URL before mapping it to Pages.
        text = text.replace('return value;', 'return '+json.dumps(base)+"+value.slice(1);")
        text = text.replace("const expected = '"+base+"showcase/library.html?case='", "const expected = '/showcase/library.html?case='")
        text = text.replace('return expected;', 'return '+json.dumps(base)+'+expected.slice(1);')
    if name.endswith('library.js'):
        text = text.replace("'/api/library?'+params", "'public/library-catalog.json'")
    if name=='showcase/app.js':
        text=text.replace('async function processor(){',"async function processor(){if(window.StaticTwin){$('processor-status').textContent='Read-only demo · saved results';return;}")
    if name.endswith('robot-workbench.js'):
        # Read saved exact baselines, then use the existing bounded JS compiler.
        text = text.replace('async function jsonURL(url){', "async function jsonURL(url){if(url.startsWith('/api/'))return StaticTwin.robotJSON(url);")
        text = text.replace("'/api/robotics/bundle?'+params", 'StaticTwin.bundleURL(params)')
    if name.endswith('agent-setup.html'):
        start = text.index('<h2>1. Discover')
        end = text.index('<h2>2. Inspect', start)
        text = text[:start]+'<h2>1. Explore the saved public catalog</h2><p>This is a static demonstration. Browse the <a href="library.html">library</a> or read its <a href="public/library-catalog.json">saved catalog JSON</a>. A live HTTP API and hosted MCP endpoint are not running here. To use MCP tools, run the local project as described below.</p>'+text[end:]
    return text

def adapter(base):
    return '''/* Static saved-baseline transport. No server, inference or model calls. */
(() => { 'use strict';
const base=BASE; let frozen,snapshot;
async function read(path){const r=await fetch(base+path);if(!r.ok)throw Error('Saved demo file unavailable');return r.json();}
const index=()=>frozen||(frozen=read('showcase/demos/robotics/index.json').then(i=>{snapshot=i;return i;}));
function selectTask(i,params){const matches=i.tasks.filter(t=>t.twin_id===params.get('twin_id')&&t.instruction===params.get('instruction'));if(matches.length!==1||params.get('kit_id')!==i.kit_id)throw Error('No saved baseline for this task');return matches[0];}
async function taskFor(params){return selectTask(await index(),params);}
window.StaticTwin={
 async robotCapabilities(){const i=await index();return{twin_ids:[...new Set(i.tasks.map(t=>t.twin_id))],scope:'saved_baselines'};},
 async robotJSON(url){const u=new URL(url,location.origin);if(u.pathname==='/api/library')return read('showcase/public/library-catalog.json');
 if(u.pathname==='/api/robotics/kit'){const i=await index();if(u.searchParams.get('kit_id')!==i.kit_id)throw Error('Unknown kit');return{kit:await read(i.kit.path),snapshot_mode:true};}
 if(u.pathname==='/api/robotics/plan'){const t=await taskFor(u.searchParams);return read(t.task.path);}throw Error('This static demo has no API');},
 bundleURL(params){if(!snapshot)throw Error('Load a saved baseline first');const t=selectTask(snapshot,params);if(!/^showcase\\/demos\\/robotics\\/[a-z0-9_-]+\\/starter\\.zip$/.test(t.starter.path))throw Error('Invalid saved starter path');return base+t.starter.path;}
};
document.addEventListener('DOMContentLoaded',()=>{if(window.self!==window.top)return;const b=document.createElement('div');b.className='static-demo-note';b.textContent='Read-only demo · saved results';document.body.prepend(b);});
})();
'''.replace('BASE', json.dumps(base), 1)

def build(output, base):
    if not re.fullmatch(r'/(?:[A-Za-z0-9_-]+/)*', base): raise ValueError('Ordinary absolute Pages base path required')
    if output.exists(): raise ValueError('Use a fresh output directory')
    public = ROOT/'showcase/public'
    catalog = json.loads((public/'library-catalog.json').read_bytes())
    original_catalog = identity('showcase/public/library-catalog.json', (public/'library-catalog.json').read_bytes())
    excluded = next(r for r in catalog['records'] if r['id']==CAMERA)
    catalog['records'] = [r for r in catalog['records'] if r['id']!=CAMERA]
    ids = {r['id'] for r in catalog['records']}; sources={r['source']['sha256'] for r in catalog['records']}
    cases=json.loads((public/'cases-index.json').read_bytes()); cases['cases']=[r for r in cases['cases'] if r['id'] in ids]
    scenes=json.loads((public/'aligned-scenes.json').read_bytes()); scenes['scenes']=[r for r in scenes['scenes'] if r['source_sha256'] in sources]
    catalog['summary']={'recordings':len(ids), 'native_asset_drafts':len({c['asset']['sha256'] for r in catalog['records'] for c in r['components'] if c.get('asset')}), 'components':sum(r['counts']['components'] for r in catalog['records']), 'steps':sum(r['counts']['steps'] for r in catalog['records']), 'eligible_saved_versions':len(ids)}
    catalog['static_export']={'backend_available':False,'source_catalog':original_catalog,'selection':'five_owner_recorded_examples; third_party_camera_excluded_pending_rights','original_artifact_bytes_modified':False}
    approved={r['path']:r for r in json.loads((public/'public-assets.json').read_bytes())['assets']}
    demo={n:json.loads((ROOT/'showcase/demos/charge-phone-v1'/n).read_bytes())for n in ['task.json','trace.json','summary.json','muse-review.json']}
    robot=json.loads((FROZEN/'showcase/demos/robotics/index.json').read_bytes())
    physics=json.loads((FROZEN/'bundle-manifest.json').read_bytes())['frozen_physics_snapshot']
    assets,omitted=closure([catalog['records'],cases,scenes,demo,robot,physics],approved)
    # The excluded recording must never be a reachable source or frame subtree.
    assert not any(p.startswith('data/flow-meta-video-camera-20261030-v1/') for p in assets)
    assert excluded['source']['path'] not in assets and len(ids)==5
    output.mkdir(parents=True); inventory=[]
    def put(name, raw, role, original=None):
        if any(r['path']==name for r in inventory):
            assert role=='filtered_static_catalog' and name.startswith('showcase/public/')
            inventory[:]=[r for r in inventory if r['path']!=name]
        p=output/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        inventory.append({**identity(name,raw),'role':role,**({'original_ref':original}if original else {})})
    for path,pointer in sorted(assets.items()): put('files/'+path,(ROOT/path).read_bytes(),'unchanged_curated_artifact',pointer)
    static=J.static_dependencies(ROOT)
    for name in sorted(set(static+J.RUNTIME+['workspace/robot-vendor/'+n for n in ['three.module.js','three.core.js','GLTFLoader.js','OrbitControls.js','BufferGeometryUtils.js','LICENSE','manifest.json']])):
        raw=(ROOT/name).read_bytes(); original=identity(name,raw)
        if Path(name).suffix in {'.html','.js','.css'}:
            text=rewrite_viewer(raw.decode(),base,name)
            if name.endswith('.html'):
                text=text.replace('</head>', '<script src="'+base+'static-demo.js"></script><style>.static-demo-note{padding:7px 24px;text-align:center;background:#eef2ff;color:#3659b8;font:12px system-ui;letter-spacing:.03em}</style></head>')
            raw=text.encode()
        put(name,raw,'static_viewer',original)
    for row in robot['tasks']:
        assert row['twin_id'] in ids
    for name in ['index.json','robot-kit.json','jabra/task.json','jabra/starter.zip','wallet/task.json','wallet/starter.zip']:
        path='showcase/demos/robotics/'+name
        put(path,(FROZEN/path).read_bytes(),'saved_robot_baseline')
    for pointer in [physics['replay_ref'],physics['starter_ref']]:
        if not (output/pointer['path']).exists():
            J.check(ROOT/pointer['path'],pointer,root=ROOT);put(pointer['path'],(ROOT/pointer['path']).read_bytes(),'saved_physics_experiment',pointer)
    # Lazy case scenes are separate from /files and keep their original bytes.
    for case in cases['cases']:
        pointer=case.get('scene_ref') or (case.get('scene') or {}).get('scene_ref')
        if pointer:
            J.check(public/pointer['path'],pointer,root=ROOT)
            put('showcase/public/'+pointer['path'],(public/pointer['path']).read_bytes(),'unchanged_lazy_scene',pointer)
    # Record-level lazy scene references are the authoritative selection.
    for row in catalog['records']:
        pointer=row['scene_case'].get('scene_ref')
        if not pointer: continue
        name='showcase/public/'+pointer['path']
        if not (output/name).exists():
            J.check(public/pointer['path'],pointer,root=ROOT);put(name,(public/pointer['path']).read_bytes(),'unchanged_lazy_scene',pointer)
    raw=encoded(catalog);digest=hashlib.sha256(raw).hexdigest();catname='library-catalog-'+digest+'.json'
    index={'format':'interaction-twin-library-index','version':1,'catalog':identity(catname,raw),'summary':catalog['summary']}
    for name,value in [('library-catalog.json',catalog),(catname,catalog),('library-index.json',index),('cases-index.json',cases),('aligned-scenes.json',scenes),('public-assets.json',{'format':'interaction-twin-public-assets','version':1,'assets':list(assets.values())})]:put('showcase/public/'+name,encoded(value),'filtered_static_catalog')
    for name,var,value in [('library-index.js','INTERACTION_LIBRARY_INDEX',index),('cases-index.js','TWIN_CASES',cases)]:put('showcase/public/'+name,('window.'+var+'='+encoded(value).decode()+';\n').encode(),'filtered_static_catalog')
    put('static-demo.js',adapter(base).encode(),'static_baseline_transport')
    put('index.html',('<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=showcase/index.html"><title>Interaction Twin</title><a href="showcase/index.html">Open Interaction Twin →</a></html>').encode(),'entry')
    put('.nojekyll',b'','hosting')
    manifest={'format':'interaction-twin-static-pages-export','version':1,'base_path':base,'source_catalog':original_catalog,'summary':catalog['summary'],'curated_ids':sorted(ids),'backend_available':False,'admin_included':False,'mcp_runtime_included':False,'new_api_calls':0,'new_gpu_calls':0,'published':False,'original_artifact_bytes_modified':False,'omitted_inert_provenance':omitted,'excluded_third_party_recording':CAMERA,'files':inventory,'total_bytes':sum(r['bytes']for r in inventory)}
    assert manifest['total_bytes']<900*1024*1024 and len(inventory)<5000
    put('export-manifest.json',encoded(manifest),'manifest')
    return {'output':str(output),'files':len(inventory),'bytes':manifest['total_bytes'],'summary':catalog['summary'],'published':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--base',default='/interaction-twin/');a=p.parse_args();print(json.dumps(build(a.output.resolve(),a.base),indent=2))
