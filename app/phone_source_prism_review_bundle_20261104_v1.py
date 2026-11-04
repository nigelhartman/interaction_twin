"""Copy an exact, private, self-contained review bundle; never serve/publish it."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import zipfile

import phone_source_regeneration_20261104_v1 as P
import phone_source_prism_20261104_v1 as G
import phone_source_prism_evaluation_20261104_v1 as E

ROOT=P.ROOT
OUTPUT=P.OUTPUT/'private-full-review-v1'

HTML='''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phone reconstruction · private comparison</title><link rel="stylesheet" href="review.css">
<script type="importmap">{"imports":{"three":"./vendor/three.module.js"}}</script></head>
<body><main><header><a class="brand" href="./">INTERACTION TWIN <span>PRIVATE REVIEW</span></a>
<h1>One phone. A cleaner shape.</h1><p>A source-derived 2.5D phone, built automatically from existing video and SAM masks.</p>
<div class="tags"><span>Actual front + back RGB</span><span>One watertight body</span><span id="coverage">Verifying saved evidence…</span></div></header>
<section class="comparison"><article><div class="pane-header"><strong>Source recording</strong><span>32.36 seconds</span></div>
<video id="video" controls playsinline muted preload="auto" aria-label="Original phone recording"></video></article>
<article><div class="pane-header"><strong id="model-label">Source-derived repair · 2.5D</strong>
<div class="segmented"><button id="source-view" aria-pressed="true">Video camera</button><button id="orbit-view" aria-pressed="false">Orbit</button></div></div>
<div class="stage"><canvas id="scene" aria-label="Interactive phone reconstruction"></canvas><div id="unavailable" hidden>Phone pose unknown at this source time.</div>
<div class="scene-caption"><b id="scene-clock">0.000 s</b><span id="pose-state">Loading exact saved geometry…</span></div></div></article></section>
<div class="transport"><button id="play">Play</button><input id="timeline" type="range" min="0" max="32.358022" step="0.001" value="17.493789" aria-label="Source video time">
<output id="clock">17.494 s</output><button id="previous">← Sample</button><button id="next">Sample →</button></div>
<nav class="links"><a id="compare-original" href="?model=original">Compare original SAM 3D ↗</a><a href="data/descriptive-sidecar.json" download>Download description ↓</a>
<a href="assets/source-phone.glb" download>Download source-derived GLB ↓</a><a href="data/new-test-report.json">New frame tests ↗</a></nav>
<aside><b>A research comparison, with uncertainty kept visible.</b>
<p>This phone uses a flat-face silhouette extrusion and an estimated edge thickness. It is <strong>not a new SAM 3D generation</strong>. Camera bumps, port depth and physical fit are not reconstructed.</p>
<p id="validation">The three prior mask-withheld RGB-tracking checks passed. On four new disjoint frames, one tracking prediction passed and three were withheld for insufficient confidence. This version is not the public default.</p>
<p id="unknown-note">Saved unknown states stay unknown. Large rotation jumps have explicit playback barriers. The cable retains its original generated asset and estimates; connector contact, grasping and insertion remain unverified.</p></aside>
<details><summary>Inspect the exact saved files</summary><div class="links"><a href="data/candidate-scene.json">Full scene JSON</a><a href="data/full-scene-report.json">Coverage report</a><a href="data/mask-withheld-report.json">Prior comparison checks</a><a href="manifest.json">Hash manifest</a></div></details>
<div id="error" role="alert" hidden></div><footer>Local private review · no upload, API call or GPU inference is launched by this page.</footer>
</main><script src="vendor/aligned-scene-contract.js"></script><script type="module" src="review.js"></script></body></html>'''

CSS='''*{box-sizing:border-box}body{margin:0;background:#f5f7fc;color:#172033;font:15px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1340px;margin:auto;padding:38px 28px}a{color:#315eea;text-decoration:none}a:hover{text-decoration:underline}button{font:inherit;cursor:pointer;border:1px solid #ccd5e8;border-radius:8px;background:white;color:#24314c;padding:9px 13px}button:hover,button[aria-pressed=true]{background:#315eea;color:white;border-color:#315eea}.brand{font-size:12px;letter-spacing:.13em;font-weight:800}.brand span{margin-left:18px;font-size:10px;color:#74819a}h1{font-size:clamp(30px,4vw,52px);letter-spacing:-.055em;line-height:1.1;margin:22px 0 12px}header>p{font-size:18px;color:#61708a;margin:0 0 20px}.tags{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:28px}.tags span{background:#e7edff;color:#3655ab;border-radius:5px;padding:5px 9px;font-size:12px}.comparison{display:grid;grid-template-columns:1fr 1fr;border:1px solid #d4dced;border-radius:16px;overflow:hidden;background:#101827}.comparison article{min-width:0}.comparison article+article{border-left:1px solid #354056}.pane-header{min-height:62px;display:flex;gap:12px;justify-content:space-between;align-items:center;color:#dce4f5;padding:13px 18px;font-size:13px}.pane-header>span{color:#99a7c0;font-size:12px}.segmented{display:flex;gap:5px}.segmented button{font-size:11px;padding:6px 10px;background:#253046;border-color:#354259;color:#c4cee2}.segmented button[aria-pressed=true]{background:#315eea;color:white}video,.stage{display:block;width:100%;height:min(65vw,570px);max-height:65vh;background:#111a29}video{object-fit:contain}.stage{position:relative}canvas{width:100%;height:100%;display:block;touch-action:none}.scene-caption{position:absolute;bottom:12px;left:14px;right:14px;border-radius:7px;background:#152239dc;color:#c6d3f0;padding:9px 12px;font-size:11px;pointer-events:none}.scene-caption b{float:right;color:#fff;font-variant-numeric:tabular-nums}.scene-caption span{display:block;padding-right:65px}#unavailable{position:absolute;left:12%;right:12%;top:42%;text-align:center;background:#edf2fbea;border:1px solid #c8d2e6;color:#41526f;border-radius:10px;padding:14px;font-size:12px}#unavailable[hidden]{display:none}.transport{display:flex;align-items:center;gap:12px;margin:17px 0}#timeline{flex:1;min-width:50px;accent-color:#315eea}output{font-size:12px;min-width:65px;font-variant-numeric:tabular-nums}.links{display:flex;gap:12px 24px;flex-wrap:wrap;font-size:13px}.links a{padding:5px 0}aside{margin:26px 0 16px;padding:21px 25px;border:1px solid #d6deef;border-radius:12px;background:white;color:#62718a;font-size:13px}aside>b{color:#243451}aside p{margin:9px 0}details{border-top:1px solid #d6deef;padding-top:16px;color:#62718a;font-size:12px}summary{cursor:pointer}footer{margin:22px 0 0;font-size:11px;color:#7d8ba3}#error{padding:20px;background:#fee9e9;color:#922f2f;border-radius:10px;margin:20px 0}@media(max-width:750px){main{padding:24px 15px}.comparison{grid-template-columns:1fr}.comparison article+article{border-left:0;border-top:1px solid #354056}video,.stage{height:480px;max-height:65vh}.transport{flex-wrap:wrap;gap:8px}.transport button{font-size:12px}.transport #timeline{flex-basis:50%}.brand span{display:block;margin:5px 0}.pane-header{font-size:12px}}'''

JS='''import * as THREE from 'three';
import {GLTFLoader} from './vendor/GLTFLoader.js';
import {OrbitControls} from './vendor/OrbitControls.js';
const $=id=>document.getElementById(id),C=globalThis.AlignedSceneContract;
const originalMode=new URLSearchParams(location.search).get('model')==='original';
const manifest=await fetch('manifest.json',{cache:'no-store'}).then(r=>r.json());
async function bytes(local){const spec=manifest.files.find(r=>r.local===local);if(!spec)throw Error('Undeclared review asset.');const r=await fetch(local,{cache:'no-store'});if(!r.ok)throw Error('Saved review file unavailable.');const raw=await r.arrayBuffer();if(raw.byteLength!==spec.bytes)throw Error('Review file length changed.');const digest=[...new Uint8Array(await crypto.subtle.digest('SHA-256',raw))].map(v=>v.toString(16).padStart(2,'0')).join('');if(digest!==spec.sha256)throw Error('Review file hash changed.');return raw;}
async function json(local){return JSON.parse(new TextDecoder().decode(await bytes(local)));}
try{
const [candidate,baseline]=await Promise.all([json('data/candidate-scene.json'),json('data/baseline-scene.json')]);
const manifestScene=originalMode?baseline:candidate;C.validateScene(manifestScene,manifestScene.source.sha256);
if(originalMode){$('model-label').textContent='Original SAM 3D · unchanged';$('compare-original').textContent='View source-derived repair ↗';$('compare-original').href='./';}
$('coverage').textContent=originalMode?'Original phone:38/47 saved estimates':'Repair phone:37/47 saved estimates';
const manager=new THREE.LoadingManager();manager.setURLModifier(url=>{if(url.startsWith('data:')||url.startsWith('blob:'))return url;throw Error('External model resources are disabled.');});
const scene=new THREE.Scene(),renderer=new THREE.WebGLRenderer({canvas:$('scene'),antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio||1,2));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.autoClear=false;
const sourceCamera=new THREE.PerspectiveCamera();sourceCamera.projectionMatrix.set(...C.projection(manifestScene.camera.K,manifestScene.source.width,manifestScene.source.height).flat());sourceCamera.projectionMatrixInverse.copy(sourceCamera.projectionMatrix).invert();sourceCamera.matrixAutoUpdate=false;sourceCamera.matrix.identity();sourceCamera.matrixWorld.identity();sourceCamera.matrixWorldInverse.identity();
const camera=new THREE.PerspectiveCamera(42,1,.001,100),controls=new OrbitControls(camera,$('scene'));controls.enableDamping=true;controls.enabled=false;let orbit=false,fitted=false;
const flip=new THREE.Matrix4().makeScale(1,-1,-1),groups=[];
for(const component of manifestScene.components){const spec=manifest.asset_mapping.find(r=>r.source_path===component.asset.path);if(!spec)throw Error('Scene asset mapping unavailable.');const raw=await bytes(spec.local);const model=await new Promise((resolve,reject)=>new GLTFLoader(manager).parse(raw,'',resolve,reject));
model.scene.traverse(o=>{if(!o.isMesh)return;const materials=Array.isArray(o.material)?o.material:[o.material];const basic=materials.map(old=>{const m=new THREE.MeshBasicMaterial({color:old.color||0xffffff,map:old.map||null,vertexColors:!!o.geometry.attributes.color,side:THREE.DoubleSide});if(o.geometry.attributes.color?.array instanceof Uint8Array&&o.geometry.attributes.color.normalized&&!m.map){m.toneMapped=false;m.onBeforeCompile=s=>{s.fragmentShader=s.fragmentShader.replace('#include <colorspace_fragment>','// Saved source byte RGB already encodes display colour.');};m.customProgramCacheKey=()=> 'private-review-source-byte-rgb-v1';}return m;});o.material=Array.isArray(o.material)?basic:basic[0];materials.forEach(m=>m.dispose());});
const group=new THREE.Group();group.matrixAutoUpdate=false;group.visible=false;group.add(model.scene);scene.add(group);groups.push({component,group});}
const videoRaw=await bytes('assets/source-video.mp4'),videoURL=URL.createObjectURL(new Blob([videoRaw],{type:'video/mp4'}));$('video').src=videoURL;
let time=17.493789;const requested=Number(new URLSearchParams(location.search).get('t'));if(Number.isFinite(requested)&&new URLSearchParams(location.search).has('t')&&requested>=0&&requested<=manifestScene.source.duration_s)time=requested;
function apply(){const current=C.sceneAt(manifestScene,time,true);let phone=null;for(const item of groups){const pose=current.parts.find(r=>r.id===item.component.id);item.group.visible=Boolean(pose?.camera_from_native_mesh);if(item.group.visible){item.group.matrix.copy(flip).multiply(new THREE.Matrix4().set(...pose.camera_from_native_mesh.flat()));item.group.matrixWorldNeedsUpdate=true;}if(item.component.id==='phone_blue')phone=pose;}
$('unavailable').hidden=Boolean(phone?.camera_from_native_mesh);$('pose-state').textContent=phone?.camera_from_native_mesh?(phone.interpolated?'Between adjacent supported estimates':phone.method?.includes('RGB')?'Saved automatic RGB-tracking estimate':'Saved automatic mask-fitting estimate'):'Unknown phone pose · no replacement placement';$('scene-clock').textContent=time.toFixed(3)+' s';$('clock').textContent=time.toFixed(3)+' s';$('timeline').value=String(time);
if(!fitted&&groups.some(r=>r.group.visible)){scene.updateMatrixWorld(true);const box=new THREE.Box3();groups.filter(r=>r.group.visible).forEach(r=>box.union(new THREE.Box3().setFromObject(r.group)));const center=box.getCenter(new THREE.Vector3()),span=Math.max(box.getSize(new THREE.Vector3()).length(),.1);controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(span*.7,span*.2,span*1.7));camera.near=span*.0001;camera.far=100;camera.updateProjectionMatrix();controls.update();fitted=true;}}
function seek(t){time=Math.max(0,Math.min(manifestScene.source.duration_s,t));$('video').currentTime=time;apply();}
$('video').addEventListener('loadedmetadata',()=>seek(time),{once:true});$('video').addEventListener('timeupdate',()=>{time=$('video').currentTime;apply();});$('video').addEventListener('play',()=>$('play').textContent='Pause');$('video').addEventListener('pause',()=>$('play').textContent='Play');$('play').onclick=()=>$('video').paused?$('video').play():$('video').pause();$('timeline').oninput=()=>seek(Number($('timeline').value));
const samples=candidate.components.find(c=>c.id==='phone_blue').poses.filter(r=>Number.isSafeInteger(r.source_sample_index));$('previous').onclick=()=>{const before=samples.filter(r=>r.timestamp_s<time-.02);seek(before.length?before.at(-1).timestamp_s:0);};$('next').onclick=()=>seek(samples.find(r=>r.timestamp_s>time+.02)?.timestamp_s||manifestScene.source.duration_s);
function setOrbit(value){orbit=value;controls.enabled=orbit;$('source-view').setAttribute('aria-pressed',String(!orbit));$('orbit-view').setAttribute('aria-pressed',String(orbit));} $('source-view').onclick=()=>setOrbit(false);$('orbit-view').onclick=()=>setOrbit(true);
new ResizeObserver(()=>{const box=$('scene').getBoundingClientRect();renderer.setSize(box.width,box.height,false);camera.aspect=box.width/box.height;camera.updateProjectionMatrix();}).observe($('scene'));
function draw(){if(!$('video').paused){time=$('video').currentTime;apply();}controls.update();const{width,height}=$('scene').getBoundingClientRect();renderer.setViewport(0,0,width,height);renderer.setScissorTest(false);renderer.setClearColor(0x111a29);renderer.clear();if(orbit){renderer.setClearColor(0xeef1f7);renderer.clear();renderer.render(scene,camera);}else{const w=Math.min(width,height*manifestScene.source.width/manifestScene.source.height),h=w*manifestScene.source.height/manifestScene.source.width;renderer.setViewport((width-w)/2,(height-h)/2,w,h);renderer.setScissor((width-w)/2,(height-h)/2,w,h);renderer.setScissorTest(true);renderer.setClearColor(0xeef1f7);renderer.clear();renderer.render(scene,sourceCamera);renderer.setScissorTest(false);}requestAnimationFrame(draw);}apply();draw();globalThis.phoneReview={scene:manifestScene,mode:originalMode?'original':'source_prism',seek,setOrbit,status:()=>({time,phoneVisible:groups.find(r=>r.component.id==='phone_blue').group.visible})};
addEventListener('pagehide',()=>{URL.revokeObjectURL(videoURL);controls.dispose();renderer.dispose();});
}catch(error){$('error').hidden=false;$('error').textContent=error.message+' No substitute data is shown.';throw error;}
'''

SERVER='''#!/usr/bin/env python3
"""Read-only loopback server for the exact copied private review inventory."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,unquote
import argparse,json,mimetypes,re
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'manifest.json').read_text())
allowed={r['local']for r in manifest['files']}|{'manifest.json'}
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_HEAD(self):self.send_file(False)
 def do_GET(self):self.send_file(True)
 def send_file(self,body):
  path=unquote(urlsplit(self.path).path)
  name='index.html'if path=='/'else path.lstrip('/')
  if name not in allowed:self.send_error(404);return
  target=ROOT/name
  if not target.is_file()or target.is_symlink()or not target.resolve().is_relative_to(ROOT):self.send_error(404);return
  size=target.stat().st_size;start,end=0,size-1;partial=False
  if self.headers.get('Range'):
   match=re.fullmatch(r'bytes=(\\d+)-(\\d*)',self.headers['Range'])
   if not match:self.send_error(416);return
   start=int(match[1]);end=min(size-1,int(match[2]))if match[2]else size-1
   if not 0<=start<=end<size:self.send_error(416);return
   partial=True
  self.send_response(206 if partial else 200)
  self.send_header('Content-Type',mimetypes.guess_type(name)[0]or'application/octet-stream')
  self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes')
  self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
  if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
  self.end_headers()
  if body:
   try:
    with target.open('rb')as stream:
     stream.seek(start);remaining=end-start+1
     while remaining:
      block=stream.read(min(1024*1024,remaining))
      if not block:break
      self.wfile.write(block);remaining-=len(block)
   except(BrokenPipeError,ConnectionResetError):pass
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8871);args=parser.parse_args()
 if not 1024<=args.port<=65535:raise ValueError('Unprivileged loopback port required')
 print(f'Private phone review: http://127.0.0.1:{args.port}/',flush=True)
 ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
'''


def build():
    OUTPUT.mkdir(exist_ok=False)
    originals={}
    def copy(source,local):
        source=Path(source);target=OUTPUT/local;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target);originals[local]=P.ref(source)
    full=G.OUTPUT/'full-scene-v1'
    for source,local in [(E.BASE,'data/baseline-scene.json'),(full/'scene.json','data/candidate-scene.json'),
        (full/'descriptive-sidecar.json','data/descriptive-sidecar.json'),(full/'report.json','data/full-scene-report.json'),
        (G.OUTPUT/'report.json','data/geometry-report.json'),
        (G.OUTPUT/'RGB-tracking-evaluation-v1/report.json','data/mask-withheld-report.json'),
        (G.OUTPUT/'new-disjoint-tests-v1/report.json','data/new-test-report.json'),
        (G.OUTPUT/'new-disjoint-tests-v1/attempt/comparison.png','data/new-test-comparison.png'),
        (G.OUTPUT/'render/native-contact-sheet.png','data/model-contact-sheet.png'),
        (P.CASE/'source/video.mp4','assets/source-video.mp4')]:copy(source,local)
    baseline=json.loads(E.BASE.read_text());candidate=json.loads((full/'scene.json').read_text())
    mapping=[];seen=set()
    for scene in [candidate,baseline]:
        for component in scene['components']:
            source=component['asset']['path']
            if source in seen:continue
            seen.add(source)
            local='assets/source-phone.glb'if source==P.ref(G.OUTPUT/'asset.glb')['path']else f'assets/original-{component["id"]}.glb'
            copy(ROOT/source,local);mapping.append({'source_path':source,'local':local,'sha256':component['asset']['sha256']})
    for name in ['three.module.js','three.core.js','GLTFLoader.js','OrbitControls.js','BufferGeometryUtils.js','LICENSE']:
        copy(ROOT/'workspace/robot-vendor'/name,'vendor/'+name)
    copy(ROOT/'showcase/aligned-scene-contract.js','vendor/aligned-scene-contract.js')
    for name,value in [('index.html',HTML),('review.css',CSS),('review.js',JS),('serve-review.py',SERVER)]:
        (OUTPUT/name).write_text(value)
    (OUTPUT/'README.txt').write_text('PRIVATE PHONE RECONSTRUCTION REVIEW\n\nRun: python3 serve-review.py --port 8871\nOpen: http://127.0.0.1:8871/\nOriginal native comparison: http://127.0.0.1:8871/?model=original\n\nThe server binds only127.0.0.1 and serves only the exact copied manifest inventory. It never serves the project root or credentials, runs models, calls an API, or publishes anything. No process was started by the builder.\n\nThe source-derived phone is explicitly a2.5D prism hypothesis, not a new SAM3D model. The original remains untouched. Source camera and orbit controls share the source video clock. Unknown poses stay hidden. Description/GLB downloads are local files.\n\nThis bundle contains the user recording for private inspection. Public distribution permission is not inferred.\n')
    files=[]
    for file in sorted(OUTPUT.rglob('*')):
        if file.is_file():
            row=P.ref(file);local=file.relative_to(OUTPUT).as_posix()
            files.append({'local':local,'sha256':row['sha256'],'bytes':row['bytes'],
                **({'original':originals[local]}if local in originals else{})})
    manifest={'format':'interaction-twin-private-phone-review-bundle','version':1,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'implementation':P.ref(Path(__file__)),
        'files':files,'asset_mapping':mapping,'candidate_scene':P.ref(full/'scene.json'),
        'original_scene':P.ref(E.BASE),'source_video':P.ref(P.CASE/'source/video.mp4'),
        'private_only':True,'inference_or_API_commands':False,'server_started':False}
    P.save(OUTPUT/'manifest.json',manifest)
    # Syntax checks are bounded and execute no web/model/server code.
    compile(SERVER,str(OUTPUT/'serve-review.py'),'exec')
    archive=P.OUTPUT/'Interaction-Twin-private-phone-review-v1.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3)as z:
        for file in sorted(OUTPUT.rglob('*')):
            if file.is_file():z.write(file,file.relative_to(OUTPUT))
    receipt={'format':'interaction-twin-private-phone-review-bundle-build','version':1,
        'bundle_manifest':P.ref(OUTPUT/'manifest.json'),'archive':P.ref(archive),
        'file_count':len(files)+1,'total_file_bytes':sum(r['bytes']for r in files)+(OUTPUT/'manifest.json').stat().st_size,
        'server_bind':'127.0.0.1','exact_file_allowlist':True,'server_started':False,
        'runner_command':['python3',str(OUTPUT/'serve-review.py'),'--port','8871'],
        'source_clock_default_s':17.493789,'public_files_or_catalog_changed':False}
    P.save(P.OUTPUT/'private-full-review-v1-build.json',receipt)
    return receipt


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
