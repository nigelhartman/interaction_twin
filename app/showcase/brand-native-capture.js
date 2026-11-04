/* New local production surface: original meshes + source-bound constrained FK. */
import * as THREE from 'three';
import {GLTFLoader} from '/workspace/robot-vendor/GLTFLoader.js';
const $=id=>document.getElementById(id),C=globalThis.AlignedSceneContract,canvas=$('native');
const CONFIG={scene:{path:'data/mechanism-scenes/5211f84dfd15d53d5bf9a0b8fffb87f69ec6cd7e1033049bca4ef9ad6ba40030/constrained/scene.json',sha256:'3e11c91756ee390fc76265f92df89cf7446fee9bf320c18132d7f7f737dc413a',bytes:440102},source_sha256:'7474ebd9d06c55b63fc073f632e71e5125353ace2538fa7bcf632cfdc1c3d0f2',duration_s:17,turntable_s:7,fps:24,width:1920,height:1080,maximum_bytes:60*1024*1024};
let renderer,manifest,parts=[],isolated=[],connected,camera,ready=false,playStart=null,recording=null,downloadURL=null,manifestURL=null,raf=null,disposed=false,lastT=0,annotation,annotationSpan=0,projectionTrace=[],lastTraceIndex=-1,lastJointEvidence=null;
const flip=new THREE.Matrix4().makeScale(1,-1,-1),matrix=rows=>new THREE.Matrix4().set(...rows.flat());
const rows=m=>Array.from({length:4},(_,r)=>Array.from({length:4},(_,c)=>m.elements[c*4+r]));
async function checked(ref){C.ref(ref,64*1024*1024);const response=await fetch('/files/'+ref.path,{cache:'force-cache'});if(!response.ok)throw Error('A curated production asset is unavailable.');const raw=await response.arrayBuffer();if(raw.byteLength!==ref.bytes)throw Error('Production asset length changed.');const sha=[...new Uint8Array(await crypto.subtle.digest('SHA-256',raw))].map(x=>x.toString(16).padStart(2,'0')).join('');if(sha!==ref.sha256)throw Error('Production asset hash changed.');return raw;}
async function model(ref){const raw=await checked(ref),manager=new THREE.LoadingManager();manager.setURLModifier(url=>{if(url.startsWith('blob:')||url.startsWith('data:'))return url;throw Error('External GLB resources disabled.');});return new Promise((resolve,reject)=>new GLTFLoader(manager).parse(raw,'',resolve,reject));}
function sourceClock(q){return q<=6?15+q*5/6:31+(q-6)/4;}
function apply(t){const state=C.sceneAt(manifest,sourceClock(t),true),byId=new Map(state.parts.map(x=>[x.id,x]));for(const p of parts){const pose=byId.get(p.id);p.group.visible=!!pose?.camera_from_native_mesh;if(p.group.visible){p.group.matrix.copy(flip).multiply(matrix(pose.camera_from_native_mesh));p.group.matrixWorldNeedsUpdate=true;}}connected.updateMatrixWorld(true);updateAnnotation(state,byId,t);return state;}
function makeAnnotation(span){
 annotationSpan=span;const group=new THREE.Group(),geometry=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3()]);
 const axis=new THREE.Line(geometry,new THREE.LineBasicMaterial({color:0x3766ed,depthTest:false,transparent:true,opacity:.95}));axis.renderOrder=20;group.add(axis);
 const dot=new THREE.Mesh(new THREE.SphereGeometry(span*.006,12,8),new THREE.MeshBasicMaterial({color:0x3766ed,depthTest:false}));dot.renderOrder=21;group.add(dot);
 const labelCanvas=document.createElement('canvas');labelCanvas.width=640;labelCanvas.height=96;const ctx=labelCanvas.getContext('2d');ctx.fillStyle='rgba(247,248,251,.96)';ctx.fillRect(0,0,640,96);ctx.font='bold 38px Arial';ctx.fillStyle='#3766ed';ctx.fillText('Estimated joint · annotation',14,59);
 const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(labelCanvas),depthTest:false,transparent:true}));sprite.scale.set(span*.46,span*.069,1);sprite.renderOrder=22;group.add(sprite);connected.add(group);annotation={group,axis,dot,sprite};
}
function updateAnnotation(state,byId,t){
 if(!annotation)return;const joint=manifest.mechanics.joints.find(j=>j.id==='j1_case_base_case_lid'),parent=byId.get(joint.parent_id),angle=state.joints.find(j=>j.id===joint.id);
 annotation.group.visible=!!parent?.camera_from_native_mesh&&typeof angle?.q_rad==='number'&&Number.isFinite(angle.q_rad);lastJointEvidence=null;if(!annotation.group.visible)return;
 // The joint origin and its axis use the SAME parent-native -> joint frame
 // chain as constrained FK. They are diagnostic annotations, not new parts.
 const world=flip.clone().multiply(matrix(parent.camera_from_native_mesh)).multiply(matrix(joint.parent_from_joint));
 const pivot=new THREE.Vector3().setFromMatrixPosition(world),direction=new THREE.Vector3(...joint.axis).transformDirection(world),half=direction.clone().multiplyScalar(annotationSpan*.07),a=pivot.clone().sub(half),b=pivot.clone().add(half);
 annotation.axis.geometry.setFromPoints([a,b]);annotation.dot.position.copy(pivot);annotation.sprite.position.copy(pivot).add(new THREE.Vector3(annotationSpan*.19,annotationSpan*.13,0));
 const project=p=>{const v=p.clone().project(camera);return [(v.x+1)*960,(1-v.y)*540,v.z];};
 lastJointEvidence={capture_time_s:t+7,source_time_s:sourceClock(t),joint_id:joint.id,q_rad:angle.q_rad,pivot_world:pivot.toArray(),axis_world:direction.toArray(),pivot_canvas:project(pivot),axis_canvas:[project(a),project(b)],annotation_only:true};
}
function draw(t){
 if(!ready||disposed)return;lastT=Math.min(CONFIG.duration_s,Math.max(0,t));renderer.setScissorTest(false);renderer.setViewport(0,0,1920,1080);renderer.setClearColor(0xf7f8fb);renderer.clear();
 if(lastT<7){for(let i=0;i<isolated.length;i++){const item=isolated[i],x=(i%2)*960,y=i<2?540:0,angle=Math.sin(lastT/7*Math.PI*2)*.22;item.camera.position.copy(item.center).add(new THREE.Vector3(Math.sin(angle)*item.distance,item.distance*.12,Math.cos(angle)*item.distance));item.camera.lookAt(item.center);renderer.setViewport(x,y,960,540);renderer.setScissor(x,y,960,540);renderer.setScissorTest(true);renderer.render(item.scene,item.camera);}}
 else{apply(lastT-7);renderer.setScissorTest(false);renderer.setViewport(0,0,1920,1080);renderer.render(connected,camera);if(recording&&lastJointEvidence){const index=Math.floor(lastT*24);if(index!==lastTraceIndex&&projectionTrace.length<1024){projectionTrace.push(lastJointEvidence);lastTraceIndex=index;}}}
 renderer.setScissorTest(false);
}
function loop(now){if(disposed)return;if(playStart!==null){const t=Math.min(17,(now-playStart)/1000);draw(t);if(t>=17){playStart=null;if(recording)stopRecord();}}raf=requestAnimationFrame(loop);}
function play(){if(!ready)throw Error('Wait for verified native assets.');playStart=performance.now();draw(0);}
function stopRecord(){if(!recording)return;const item=recording;recording=null;clearTimeout(item.timer);playStart=null;if(item.recorder.state!=='inactive')item.recorder.stop();}
function record(){
 if(recording)throw Error('A capture is already running.');if(!ready)throw Error('Wait for verified assets.');
 const types=['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm','video/mp4'],mime=types.find(t=>MediaRecorder.isTypeSupported(t));if(!mime||!canvas.captureStream)throw Error('Canvas recording is unavailable.');
 if(downloadURL)URL.revokeObjectURL(downloadURL);if(manifestURL)URL.revokeObjectURL(manifestURL);$('download').hidden=true;$('manifest-download').hidden=true;
 const stream=canvas.captureStream(24),chunks=[],started=performance.now();let recorder;try{recorder=new MediaRecorder(stream,{mimeType:mime,videoBitsPerSecond:14000000});}catch(error){stream.getTracks().forEach(track=>track.stop());throw error;}let bytes=0,invalid=false;
 const item={stream,recorder,timer:null};recording=item;projectionTrace=[];lastTraceIndex=-1;$('record').disabled=true;$('play').disabled=true;
 recorder.ondataavailable=event=>{if(invalid||!event.data.size)return;bytes+=event.data.size;if(bytes>CONFIG.maximum_bytes){invalid=true;chunks.length=0;stopRecord();$('status').textContent='Stopped at the60MB bound; no clip was retained.';}else chunks.push(event.data);};
 recorder.onerror=()=>{invalid=true;chunks.length=0;$('status').textContent='Canvas recorder failed; no retry or asset change.';stopRecord();};
 recorder.onstop=()=>{clearTimeout(item.timer);stream.getTracks().forEach(track=>track.stop());recording=null;if(disposed)return;$('record').disabled=false;$('play').disabled=false;if(invalid||!chunks.length)return;
  const blob=new Blob(chunks,{type:recorder.mimeType||mime});downloadURL=URL.createObjectURL(blob);const extension=blob.type.startsWith('video/mp4')?'mp4':'webm';$('download').href=downloadURL;$('download').download='native-capture-v2.'+extension;$('download').textContent='Download native clip';$('download').hidden=false;
  const receipt={format:'interaction-twin-native-browser-capture',version:2,config:CONFIG,assets:manifest.components.map(p=>({id:p.id,asset:p.asset})),chapters:[{start_s:0,end_s:7,scope:'Independent unchanged native meshes; editorial camera orbit only.'},{start_s:7,end_s:17,scope:'Saved constrained scene with bounded FK interpolation; source15–20s then31–32s; editorial jump atcapture13s. Unknown earbud poses withheld. Blue joint marker/axis/label are diagnostic annotations from actual FK, not model geometry.'}],joint_annotation:{joint_id:'j1_case_base_case_lid',chain:'flip * parent.camera_from_native_mesh * parent_from_joint',axis_frame:'joint_frame_after_parent_from_joint',axis_length_world:annotationSpan*.14,dot_radius_world:annotationSpan*.006,depth_test:false,physical_contact_verified:false},capture_camera:{matrix_world:rows(camera.matrixWorld),projection_matrix:rows(camera.projectionMatrix)},projection_trace:projectionTrace,captured_elapsed_s:(performance.now()-started)/1000,bytes:blob.size,mime:blob.type,canvas_dimensions:[1920,1080],camera_microphone_access:false,geometry_modified:false,independent_child_poses_used:false,physical_accuracy_verified:false,simulation_performed:false,new_model_calls:0};
  manifestURL=URL.createObjectURL(new Blob([JSON.stringify(receipt,null,2)+'\n'],{type:'application/json'}));$('manifest-download').href=manifestURL;$('manifest-download').download='native-capture-v2.json';$('manifest-download').hidden=false;$('status').textContent='Capture ready · '+(blob.size/1024/1024).toFixed(1)+'MB · download links are now available.';
 };
 try{recorder.start(250);play();item.timer=setTimeout(stopRecord,19000);$('status').textContent='Recording17seconds of the actual1920×1080 canvas…';}catch(error){invalid=true;chunks.length=0;stopRecord();stream.getTracks().forEach(track=>track.stop());$('record').disabled=false;$('play').disabled=false;throw error;}
}
async function start(){
 manifest=C.validateScene(JSON.parse(new TextDecoder().decode(await checked(CONFIG.scene))),CONFIG.source_sha256);const expected=['case_base','case_lid','earbud_a','earbud_b'];if(manifest.components.length!==4||expected.some(id=>!manifest.components.some(p=>p.id===id)))throw Error('Exact four-component curated scene required.');
 renderer=new THREE.WebGLRenderer({canvas,antialias:true,preserveDrawingBuffer:true});renderer.setSize(1920,1080,false);renderer.setPixelRatio(1);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.autoClear=false;connected=new THREE.Scene();
 for(const component of manifest.components){const native=await model(component.asset);native.scene.traverse(o=>{if(!o.isMesh)return;const originals=Array.isArray(o.material)?o.material:[o.material],basic=originals.map(m=>new THREE.MeshBasicMaterial({color:m.color||0xffffff,map:m.map||null,vertexColors:!!o.geometry.attributes.color,side:THREE.DoubleSide,transparent:m.transparent,opacity:m.opacity}));o.material=Array.isArray(o.material)?basic:basic[0];originals.forEach(m=>m.dispose());});
  const group=new THREE.Group();group.matrixAutoUpdate=false;group.add(native.scene);connected.add(group);parts.push({id:component.id,group});
  const anchor=component.poses.find(p=>p.source_sample_index===41);if(!anchor?.camera_from_native_mesh)throw Error('A saved editorial view anchor is unavailable.');const isoScene=new THREE.Scene(),iso=new THREE.Group();iso.matrixAutoUpdate=false;iso.matrix.copy(flip).multiply(matrix(anchor.camera_from_native_mesh));iso.add(native.scene.clone(true));isoScene.add(iso);isoScene.updateMatrixWorld(true);const box=new THREE.Box3().setFromObject(iso),center=box.getCenter(new THREE.Vector3()),span=box.getSize(new THREE.Vector3()).length(),cam=new THREE.PerspectiveCamera(37,960/540,span*.0001,span*100);isolated.push({scene:isoScene,camera:cam,center,distance:span*1.65});
 }
 const union=new THREE.Box3();for(const t of [0,3,5,6,8,10]){apply(t);for(const p of parts)if(p.group.visible)union.union(new THREE.Box3().setFromObject(p.group));}if(union.isEmpty())throw Error('No supported constrained scene.');const center=union.getCenter(new THREE.Vector3()),span=union.getSize(new THREE.Vector3()).length();camera=new THREE.PerspectiveCamera(38,1920/1080,span*.0001,span*100);camera.position.copy(center).add(new THREE.Vector3(span*.12,span*.25,span*1.65));camera.lookAt(center);
 camera.updateMatrixWorld(true);makeAnnotation(span);ready=true;draw(0);$('status').textContent='Ready · four exact native GLBs, constrained FK and diagnostic joint annotation verified.';$('play').disabled=false;$('record').disabled=false;raf=requestAnimationFrame(loop);
}
$('play').onclick=()=>{try{play();}catch(e){$('status').textContent=e.message;}};$('record').onclick=()=>{try{record();}catch(e){$('status').textContent=e.message;}};
addEventListener('pagehide',()=>{disposed=true;cancelAnimationFrame(raf);stopRecord();if(downloadURL)URL.revokeObjectURL(downloadURL);if(manifestURL)URL.revokeObjectURL(manifestURL);renderer?.dispose();});
globalThis.BrandNativeCapture=Object.freeze({snapshot:()=>({ready,time_s:lastT,recording:!!recording,config:CONFIG}),seek:t=>{playStart=null;draw(t);}});
start().catch(error=>{$('status').textContent=error.message+' No substitute asset is shown.';});
