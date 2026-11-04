/* Local immutable native review. No inference, original-pose reuse or catalog. */
(function(root){'use strict';
const FILES=['baseline-scene.json','baseline.glb','candidate-scene.json','candidate.glb','source-mask.png','source-rgb.png'];
const HASH=/^[0-9a-f]{64}$/;
const expectedSize={'.glb':16*1024*1024,'.png':8*1024*1024,'.json':64*1024*1024};
function assert(value,message){if(!value)throw new Error(message)}
async function digest(bytes){return Array.from(new Uint8Array(await root.crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('')}
function validateManifest(value,binding){
 assert(value&&value.format==='sam3d-native-review'&&value.version===1&&value.fixture===false&&value.native_comparison_only===true,'Unsupported or fixture review.');
 const b=value.binding;assert(b&&b.case_id===binding.caseId&&b.track_id===binding.trackId&&HASH.test(b.program_sha256)&&HASH.test(b.video_sha256)&&b.time_domain==='source_video_presentation_seconds'&&Number.isFinite(b.timestamp_s)&&b.timestamp_s>=0&&Number.isInteger(b.sample_index)&&b.sample_index>0&&b.width>0&&b.height>0,'The review source binding differs.');
 assert(value.scope?.independent_gauges===true,'Independent native gauges are required.');
 for(const field of ['geometry_bytes_modified','pose_reused','source_video_alignment_established','metric_scale_verified','physical_accuracy_verified','assembly_verified','contact_verified','simulation_executed','preferred_model_changed'])assert(value.scope[field]===false,'Unsupported geometry, pose or physical claim.');
 assert(value.candidate?.selected===false&&value.candidate.native_up==='Y'&&value.baseline?.native_up==='Z'&&HASH.test(value.candidate.asset_sha256)&&HASH.test(value.baseline.asset_sha256),'The native candidate must remain unselected.');
 assert(value.renderer?.pbr_rendering===false&&value.renderer?.splat_rendering===false,'This review shows native unlit meshes only.');
 assert(Array.isArray(value.files)&&value.files.length===FILES.length,'Review file inventory differs.');
 const names=new Set();for(const file of value.files){assert(file&&FILES.includes(file.path)&&!names.has(file.path)&&HASH.test(file.sha256)&&Number.isInteger(file.bytes)&&file.bytes>0,'Unsafe or duplicated review file.');names.add(file.path);const suffix=file.path.slice(file.path.lastIndexOf('.'));assert(file.bytes<=expectedSize[suffix],'Review artifact exceeds the native budget.')}
 assert(value.files.find(x=>x.path==='candidate.glb').sha256===value.candidate.asset_sha256&&value.files.find(x=>x.path==='baseline.glb').sha256===value.baseline.asset_sha256,'Model bytes and review identity differ.');
 return value;
}
function validateScene(value,asset,track){
 const scene=value?.scene,objects=scene?.objects;
 assert(value?.source_glb_sha256===asset.asset_sha256&&scene.native_only===true&&scene.physical_assembly_verified===false&&scene.source_projection_rows===null&&Array.isArray(scene.timeline)&&scene.timeline.length===0&&Array.isArray(objects)&&objects.length===1,'Native scene does not match its verified GLB.');
 const object=objects[0];assert(object.object_id===track&&object.native_up===asset.native_up&&Array.isArray(object.frames)&&object.frames.length===0&&object.max_hold_s===0&&object.source_projection_rows===null&&Array.isArray(object.primitives)&&object.primitives.length>0,'Pose or source projection cannot enter a native-only scene.');
 const layout=object.native_scene;assert(layout&&Array.isArray(layout.target)&&layout.target.length===3&&layout.target.every(Number.isFinite)&&Number.isFinite(layout.span)&&layout.span>0&&Number.isFinite(layout.initial_distance)&&layout.initial_distance>0,'Native display framing is invalid.');
 return scene;
}
async function readResponse(response,maxBytes){
 assert(response.ok,'A review artifact is unavailable.');const declared=Number(response.headers?.get?.('content-length'));assert(!Number.isFinite(declared)||declared<=maxBytes,'Review response exceeds its declared byte budget.');
 if(!response.body?.getReader){const raw=await response.arrayBuffer();assert(raw.byteLength<=maxBytes,'Review response exceeds its byte budget.');return raw}
 const reader=response.body.getReader(),parts=[];let size=0;try{for(;;){const part=await reader.read();if(part.done)break;size+=part.value.byteLength;if(size>maxBytes){await reader.cancel();throw new Error('Review response exceeds its byte budget.')}parts.push(part.value)}}finally{reader.releaseLock?.()}
 const raw=new Uint8Array(size);let offset=0;for(const part of parts){raw.set(part,offset);offset+=part.byteLength}return raw.buffer;
}
async function readVerified(url,expected,fetchImpl=root.fetch.bind(root)){
 const response=await fetchImpl(url,{cache:'no-store',credentials:'same-origin'}),raw=await readResponse(response,expected.bytes);assert(raw.byteLength===expected.bytes&&await digest(raw)===expected.sha256,'Review artifact bytes changed.');return raw;
}
const API={validateManifest,validateScene,readVerified,readResponse,digest,FILES};if(typeof module!=='undefined'&&module.exports)module.exports=API;root.SAM3DNativeReview=API;
if(!root.document)return;
let epoch=1,renderers=[],urls=[];const $=id=>document.getElementById(id),status=$('status');
function fail(error){status.textContent='Native review unavailable: '+(error?.message||'source or model verification failed');status.classList.add('error');$('panels').hidden=true;$('details').hidden=true;for(const r of renderers)r.dispose();renderers=[]}
function keyboard(canvas,renderer){canvas.addEventListener('keydown',event=>{let handled=true;switch(event.key){case'ArrowLeft':renderer.yaw+=.12;break;case'ArrowRight':renderer.yaw-=.12;break;case'ArrowUp':renderer.pitch=Math.min(1.4,renderer.pitch+.1);break;case'ArrowDown':renderer.pitch=Math.max(-1.4,renderer.pitch-.1);break;case'+':case'=':renderer.distance*=.9;break;case'-':renderer.distance*=1.1;break;default:handled=false}if(handled){event.preventDefault();renderer.draw()}})}
async function start(){const token=epoch,b=document.body.dataset;assert(HASH.test(b.manifestSha),'Missing pinned review manifest.');
 const response=await fetch('review-manifest.json',{cache:'no-store',credentials:'same-origin'}),raw=await readResponse(response,128*1024);assert(await digest(raw)===b.manifestSha,'The pinned review manifest changed.');
 const manifest=validateManifest(JSON.parse(new TextDecoder().decode(raw)),{caseId:b.caseId,trackId:b.trackId});
 const verified=await Promise.all(manifest.files.map(async file=>[file.path,await readVerified(file.path,file)]));if(token!==epoch)return;
 const bytes=new Map(verified),decode=name=>JSON.parse(new TextDecoder().decode(bytes.get(name)));
 const baseline=validateScene(decode('baseline-scene.json'),manifest.baseline,manifest.binding.track_id),candidate=validateScene(decode('candidate-scene.json'),manifest.candidate,manifest.binding.track_id);
 for(const [name,id]of[['source-rgb.png','source-rgb'],['source-mask.png','source-mask']]){const url=URL.createObjectURL(new Blob([bytes.get(name)],{type:'image/png'}));urls.push(url);$(id).src=url}
 $('source-time').textContent='Sample '+manifest.binding.sample_index+' · '+manifest.binding.timestamp_s.toFixed(6)+' s';$('baseline-title').textContent=manifest.baseline.backend+' baseline';
 for(const [prefix,scene,asset]of[['baseline',baseline,manifest.baseline],['candidate',candidate,manifest.candidate]]){const renderer=new root.TwinScene.Scene($(prefix+'-canvas'),scene,error=>fail(new Error(error)));renderer.inspect('native',manifest.binding.track_id);renderers.push(renderer);$(prefix+'-reset').onclick=()=>renderer.reset();keyboard($(prefix+'-canvas'),renderer);$(prefix+'-counts').textContent=asset.mesh_stats.rendered_vertices.toLocaleString()+' vertices · '+asset.mesh_stats.faces.toLocaleString()+' faces'}
 $('mask-toggle').onclick=()=>{const show=$('source-mask').hidden;$('source-mask').hidden=!show;$('mask-toggle').textContent=show?'Hide mask':'Show mask';$('mask-toggle').setAttribute('aria-pressed',String(show))};
 $('panels').hidden=false;$('details').hidden=false;for(const r of renderers)r.draw();status.textContent='Verified native comparison · same source RGB/mask · candidate unselected · no source-video alignment.';
}
root.addEventListener('pagehide',()=>{epoch++;for(const r of renderers)r.dispose();renderers=[];for(const url of urls)URL.revokeObjectURL(url);urls=[]});start().catch(fail);
})(globalThis);
