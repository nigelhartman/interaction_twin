/* Genuine initial-native review V2. No inference, source poses or assembly. */
(async()=>{'use strict';
const $=id=>document.getElementById(id),SHA=/^[a-f0-9]{64}$/;
const FILES=['candidate-scene.json','candidate-geometry.bin','candidate.glb','mask-overlay.png','model-metadata.json','source-mask.png','source-rgb.png'];
let renderer=null,urls=[];
function assert(v,m){if(!v)throw Error(m)}
const sha=raw=>InitialNativeGeometry.hash(raw);
async function bounded(response,limit){
 assert(response.ok,'Saved resource unavailable');const declared=Number(response.headers.get('content-length'));
 assert(!Number.isFinite(declared)||declared<=limit,'Resource length exceeds bound');
 const reader=response.body?.getReader();
 if(!reader){const raw=await response.arrayBuffer();assert(raw.byteLength<=limit,'Resource exceeds byte bound');return raw}
 let size=0,parts=[];
 try{for(;;){const p=await reader.read();if(p.done)break;size+=p.value.byteLength;assert(size<=limit,'Resource exceeds byte bound');parts.push(p.value)}}
 catch(e){await reader.cancel();throw e}finally{reader.releaseLock()}
 const out=new Uint8Array(size);let at=0;for(const p of parts){out.set(p,at);at+=p.byteLength}return out.buffer;
}
function blob(raw,type){const u=URL.createObjectURL(new Blob([raw],{type}));urls.push(u);return u}
try{
 const pinned=document.body.dataset.manifestSha;assert(SHA.test(pinned),'Missing manifest pin');
 const raw=await bounded(await fetch('review-manifest.json',{cache:'no-store',credentials:'omit'}),128*1024);
 assert(await sha(raw)===pinned,'Review manifest changed');const d=JSON.parse(new TextDecoder().decode(raw));
 assert(d.format==='sam3d-initial-native-review'&&d.version===2&&d.fixture===false&&d.native_only===true&&d.native_up==='Y'&&SHA.test(d.native_glb_sha256),'Unsupported binary initial native review');
 for(const k of ['baseline_required','original_geometry_modified','old_pose_reused','source_video_placement_established','physical_accuracy_verified','metric_scale_verified','assembly_verified','contact_verified','simulation_ready','selected','catalog_changed'])assert(d.scope?.[k]===false,'Unexpected physical or selection claim');
 const t=d.geometry_transport;
 assert(t?.format==='lossless-native-geometry'&&t.version===1&&t.byte_order==='little'&&t.float_encoding==='ieee754-float64'&&t.index_encoding==='uint32'&&t.combined_scene_budget_bytes===64*1024*1024,'Unsupported geometry transport');
 assert(Array.isArray(d.files)&&d.files.length===FILES.length,'Review inventory differs');
 const names=new Set(),bytes=new Map(),records=new Map();
 for(const f of d.files){
  assert(FILES.includes(f.path)&&!names.has(f.path)&&SHA.test(f.sha256)&&Number.isSafeInteger(f.bytes)&&f.bytes>0,'Invalid review resource');
  names.add(f.path);records.set(f.path,f);
  const limit=f.path.endsWith('.glb')?16*1024*1024:['candidate-scene.json','candidate-geometry.bin'].includes(f.path)?64*1024*1024:8*1024*1024;
  assert(f.bytes<=limit,'Review resource exceeds unchanged native budget');
 }
 const descriptor=records.get('candidate-scene.json'),geometry=records.get('candidate-geometry.bin');
 assert(descriptor.bytes+geometry.bytes<=64*1024*1024&&descriptor.bytes===t.descriptor_bytes&&geometry.bytes===t.geometry_bytes&&geometry.sha256===t.geometry_sha256,'Combined binary scene budget or hash differs');
 assert(records.get('candidate.glb').sha256===d.native_glb_sha256,'Native GLB identity differs');
 for(const f of d.files){
  const value=await bounded(await fetch(f.path,{cache:'force-cache',credentials:'omit'}),f.bytes);
  assert(value.byteLength===f.bytes&&await sha(value)===f.sha256,'Source or native bytes changed');bytes.set(f.path,value);
 }
 $('status').textContent='Source and model verified · decoding exact binary geometry…';
 const scene=await InitialNativeGeometry.decodeScene(bytes.get('candidate-scene.json'),bytes.get('candidate-geometry.bin'),{sourceHash:d.native_glb_sha256,meshStats:d.mesh_stats,componentId:d.component_id,imageSize:d.image_size});
 assert(scene.geometry_sha256===t.geometry_sha256,'Scene/manifest binary identity differs');
 $('title').textContent=d.title;
 $('source-time').textContent='Source sample '+d.source.sample_index+' · '+d.source.timestamp_s.toFixed(6)+' s';
 $('source-rgb').src=blob(bytes.get('source-rgb.png'),'image/png');$('source-mask').src=blob(bytes.get('mask-overlay.png'),'image/png');
 $('mask-toggle').onchange=()=>{$('source-mask').hidden=!$('mask-toggle').checked};
 $('appearance').textContent=d.appearance;
 $('counts').textContent=d.mesh_stats.rendered_vertices.toLocaleString()+' vertices · '+d.mesh_stats.faces.toLocaleString()+' faces · unchanged generated model';
 $('binding').textContent=JSON.stringify({component_id:d.component_id,source:d.source,native_glb_sha256:d.native_glb_sha256,initial_handoff:d.initial_handoff,import_receipt:d.import_receipt,geometry_transport:d.geometry_transport,scope:d.scope},null,2);
 renderer=new TwinScene.Scene($('native-canvas'),scene.scene,message=>{throw Error(message)});
 renderer.inspect('native',d.component_id);$('reset').onclick=()=>renderer.reset();
 $('native-canvas').addEventListener('keydown',e=>{
  if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-'].includes(e.key))return;
  e.preventDefault();if(e.key==='ArrowLeft')renderer.yaw-=.12;if(e.key==='ArrowRight')renderer.yaw+=.12;
  if(e.key==='ArrowUp')renderer.pitch=Math.min(1.4,renderer.pitch+.1);if(e.key==='ArrowDown')renderer.pitch=Math.max(-1.4,renderer.pitch-.1);
  if(e.key==='+'||e.key==='=')renderer.distance*=.9;if(e.key==='-')renderer.distance*=1.1;renderer.draw();
 });
 $('download').onclick=()=>{const a=document.createElement('a');a.href=blob(bytes.get('candidate.glb'),'model/gltf-binary');a.download=d.component_id+'-native.glb';a.click()};
 $('metadata').href=blob(bytes.get('model-metadata.json'),'application/json');
 $('content').hidden=false;renderer.draw();
 $('status').textContent='Verified initial source + native model · lossless binary display · candidate unselected · no source-video placement.';
 window.addEventListener('pagehide',()=>{renderer?.dispose();urls.forEach(URL.revokeObjectURL)});
}catch(e){renderer?.dispose();urls.forEach(URL.revokeObjectURL);$('content').hidden=true;$('status').textContent='Initial native review unavailable: '+e.message}
})();
