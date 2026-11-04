/* Exact private laptop source replay. Renderer and unknown policy are shared. */
(function(root){'use strict';
const C=root.AlignedSceneContract,SHA=/^[a-f0-9]{64}$/;
function assert(value,message){if(!value)throw Error(message)}
function same(a,b){return a?.path===b?.path&&a?.sha256===b?.sha256&&a?.bytes===b?.bytes}
function validate(manifest,scene,program){
 assert(manifest?.format==='interaction-twin-private-laptop-hinge-review'&&manifest.version===1&&manifest.physical_accuracy_verified===false,'Unsupported private hinge review');
 C.ref(manifest.scene_ref,8*1024*1024);C.ref(manifest.program_ref,8*1024*1024);C.ref(manifest.source_video_ref,64*1024*1024);
 C.validateScene(scene,manifest.source_video_ref.sha256);
 assert(same(scene.original_program,manifest.program_ref)&&program?.source?.sha256===scene.source.sha256&&program.source.path===manifest.source_video_ref.path,'Source, program and hinge scene differ');
 assert(Array.isArray(manifest.assets)&&manifest.assets.length===scene.components.length&&new Set(manifest.assets.map(a=>a.id)).size===manifest.assets.length,'Exact native asset inventory required');
 for(const component of scene.components){const item=manifest.assets.find(a=>a.id===component.id);assert(item&&same(item.asset_ref,component.asset),'Native asset binding differs');C.ref(item.asset_ref);assert(program.assets?.some(a=>a.track_id===component.id&&a.path===item.asset_ref.path&&a.sha256===item.asset_ref.sha256),'Native component not bound to the source program');}
 assert(scene.mechanics?.joints.length===1&&!(scene.mechanics.unresolved_children||[]).length,'One resolved estimated joint is required');
 return scene;
}
function stateAt(scene,time){
 assert(Number.isFinite(time)&&time>=0&&time<=scene.source.duration_s,'Source time is outside the recording');const value=C.sceneAt(scene,time,true),joint=value.joints[0];return{...value,joint,supported:Number.isFinite(joint?.q_rad)&&value.parts.some(p=>p.id===joint.parent_id&&p.camera_from_native_mesh)&&value.parts.some(p=>p.id===joint.child_id&&p.camera_from_native_mesh)};
}
function validMessage(event,frame,origin,binding){const d=event?.data;return event.source===frame&&event.origin===origin&&d?.source_sha256===binding.source&&d?.manifest_sha256===binding.scene&&['interaction-twin:aligned-ready','interaction-twin:aligned-state','interaction-twin:aligned-error'].includes(d?.type)?d:null}
const api={validate,stateAt,validMessage};root.LaptopHingeReview=api;if(typeof module!=='undefined'&&module.exports)module.exports=api;if(typeof document==='undefined')return;
const $=id=>document.getElementById(id);let manifest,scene,disposed=false,frameReady=false,view='source',initial=0,raf=0,lastSent=-1;const urls=[];
function error(message){$('status').textContent=message;$('status').className='error';for(const id of['play','timeline','source-view','orbit-view','supported','reset'])$(id).disabled=true;}
async function bytes(reference){C.ref(reference,64*1024*1024);const response=await fetch('/files/'+reference.path,{cache:'no-store',credentials:'omit'});assert(response.ok,'A source-bound artifact is unavailable');const raw=await response.arrayBuffer();assert(raw.byteLength===reference.bytes,'An artifact length changed');const digest=[...new Uint8Array(await crypto.subtle.digest('SHA-256',raw))].map(v=>v.toString(16).padStart(2,'0')).join('');assert(digest===reference.sha256,'An artifact hash changed');return raw;}
function update(reset=false){if(!scene||disposed)return;const time=Math.min(scene.source.duration_s,Math.max(0,$('video').currentTime||0)),state=stateAt(scene,time);$('clock').textContent=time.toFixed(3)+' s';$('timeline').value=String(time);$('motion-state').textContent=state.supported?'Connected motion · estimated':'Joint unavailable · connected assembly withheld';$('joint-value').textContent=state.supported?(state.joint.q_rad*180/Math.PI).toFixed(1)+'° relative':'Unknown';if(frameReady&&(reset||Math.abs(time-lastSent)>.012)){lastSent=time;$('scene').contentWindow.postMessage({type:'interaction-twin:aligned-time',source_sha256:scene.source.sha256,manifest_sha256:manifest.scene_ref.sha256,time_s:time,show_estimated:true,view,reset},location.origin);}}
function animate(){if(disposed)return;update();if(!$('video').paused)raf=requestAnimationFrame(animate)}
function setTime(time){if(!scene||disposed)return;$('video').currentTime=Math.min(scene.source.duration_s,Math.max(0,time));update();}
function setView(value){view=value;$('source-view').setAttribute('aria-pressed',String(value==='source'));$('orbit-view').setAttribute('aria-pressed',String(value==='orbit'));lastSent=-1;update();}
async function start(){
 const response=await fetch('/manifest.json',{cache:'no-store',credentials:'omit'});assert(response.ok,'Private review manifest unavailable');const text=await response.text();assert(text.length<=128*1024,'Review manifest exceeds bound');manifest=JSON.parse(text);
 const [sceneRaw,programRaw]=await Promise.all([bytes(manifest.scene_ref),bytes(manifest.program_ref)]);if(disposed)return;scene=validate(manifest,JSON.parse(new TextDecoder().decode(sceneRaw)),JSON.parse(new TextDecoder().decode(programRaw)));
 const saved=scene.mechanics.joints[0].samples,supported=saved.filter(s=>s.status!=='unknown'&&Number.isFinite(s.q_rad)).length;initial=C.previewTime(scene).time_s;$('coverage').textContent=supported+' supported saved samples · '+(saved.length-supported)+' unknown · no missing motion invented';
 const sourceRaw=await bytes(manifest.source_video_ref);if(disposed)return;const blob=URL.createObjectURL(new Blob([sourceRaw],{type:'video/mp4'}));urls.push(blob);
 $('video').onloadedmetadata=()=>{if(disposed)return;if(!Number.isFinite($('video').duration)||Math.abs($('video').duration-scene.source.duration_s)>.3){error('Source-video clock differs from the scene. No substitute replay is shown.');return;}$('timeline').max=String(scene.source.duration_s);for(const id of['play','timeline','source-view','orbit-view','supported','reset'])$(id).disabled=false;setTime(initial);};$('video').src=blob;
 const params=new URLSearchParams({...manifest.scene_ref,source:scene.source.sha256,t:String(initial)});$('scene').src='/showcase/aligned-scene.html?'+params;$('scene').hidden=false;
 $('scene-download').href='/files/'+manifest.scene_ref.path;$('program-download').href='/files/'+manifest.program_ref.path;$('scene-download').hidden=false;$('program-download').hidden=false;
 $('status').textContent='Verified source + native assets · one estimated persistent hinge · no physical acceptance.';
}
$('play').onclick=()=>{$('video').paused?$('video').play().catch(e=>error(e.message)):$('video').pause()};$('video').addEventListener('play',()=>{$('play').textContent='Pause';cancelAnimationFrame(raf);animate()});$('video').addEventListener('pause',()=>{$('play').textContent='Play';cancelAnimationFrame(raf);update()});for(const event of['seeked','timeupdate'])$('video').addEventListener(event,()=>update());$('timeline').addEventListener('input',()=>setTime(Number($('timeline').value)));$('source-view').onclick=()=>setView('source');$('orbit-view').onclick=()=>setView('orbit');$('supported').onclick=()=>{$('video').pause();setTime(initial)};$('reset').onclick=()=>update(true);
addEventListener('message',event=>{if(!manifest||!scene||disposed)return;const d=validMessage(event,$('scene').contentWindow,location.origin,{source:scene.source.sha256,scene:manifest.scene_ref.sha256});if(!d)return;if(d.type==='interaction-twin:aligned-ready'){frameReady=true;lastSent=-1;update()}else if(d.type==='interaction-twin:aligned-error')error(String(d.message||'Connected replay unavailable').slice(0,1000));});
addEventListener('pagehide',()=>{disposed=true;cancelAnimationFrame(raf);$('video').pause();$('scene').removeAttribute('src');urls.forEach(URL.revokeObjectURL)});start().catch(e=>{if(!disposed)error(e.message)});
})(globalThis);
