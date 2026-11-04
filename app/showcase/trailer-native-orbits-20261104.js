import * as THREE from 'three';
import {GLTFLoader} from '/workspace/robot-vendor/GLTFLoader.js';
import './native-byte-rgb-display-v1.js';
const $=id=>document.getElementById(id),canvas=$('orbit-canvas'),W=1080,D=12,FPS=30;
const specs=[
 {case_id:'flow-sam3d-default-e2e-20261102-v1--flow-procedure-1',component:'case_base',label:'Case base'},
 {case_id:'flow-capture-laptop-20261103-v1--flow-procedure-1',component:'laptop_lid',label:'Laptop lid'},
 {case_id:'flow-capture-charger-20261103-v1--flow-procedure-1',component:'adapter_early',label:'USB-C adapter'},
 {case_id:'flow-rigid-parts-wallet-20261031-v1--flow-procedure-1',component:'gold_coin',label:'Coin'}
];
let renderer,scene,camera,parts=[],raf,disposed=false,recording=false,urls=[];
const digest=async b=>[...new Uint8Array(await crypto.subtle.digest('SHA-256',b))].map(n=>n.toString(16).padStart(2,'0')).join('');
async function checked(ref){InteractionLibrary.ref(ref);if(ref.bytes>32*1024*1024)throw Error('Native asset exceeds capture bound.');const r=await fetch('/files/'+ref.path,{credentials:'omit'});if(!r.ok)throw Error('Exact original asset unavailable.');const b=await r.arrayBuffer();if(b.byteLength!==ref.bytes||await digest(b)!==ref.sha256)throw Error('Original asset bytes differ.');return b;}
function draw(time){
 const index=Math.min(3,Math.floor(Math.max(0,time)/3)),fraction=Math.min(1,(time-index*3)/3);
 parts.forEach((p,i)=>{p.group.visible=i===index;});
 const yaw=(38+fraction*54)*Math.PI/180,radius=2.4;camera.position.set(Math.sin(yaw)*radius,1.03,Math.cos(yaw)*radius);camera.lookAt(0,0,0);camera.updateMatrixWorld(true);renderer.render(scene,camera);
 const box=parts[index].bounds,projections=[];for(const x of[box.min.x,box.max.x])for(const y of[box.min.y,box.max.y])for(const z of[box.min.z,box.max.z]){const v=new THREE.Vector3(x,y,z).project(camera);projections.push([(v.x+1)*W/2,(1-v.y)*W/2,v.z]);}
 if(projections.some(p=>p[0]<36||p[0]>W-36||p[1]<36||p[1]>W-36||p[2]<-1||p[2]>1))throw Error('Full native bounds do not fit the editorial camera.');
 return {time_s:time,index,component:specs[index].component,camera_position:camera.position.toArray(),projected_native_box:projections};
}
async function start(){
 const catalog=await InteractionLibrary.loadCatalog();scene=new THREE.Scene();scene.background=new THREE.Color('#f7f8fb');camera=new THREE.PerspectiveCamera(36,1,.01,50);
 renderer=new THREE.WebGLRenderer({canvas,antialias:true,preserveDrawingBuffer:true});renderer.setPixelRatio(1);renderer.setSize(W,W,false);renderer.outputColorSpace=THREE.SRGBColorSpace;
 for(const spec of specs){const record=catalog.records.find(r=>r.id===spec.case_id),review=Sam3DModelReviews.forComponent(record,spec.component);if(!review?.native_glb)throw Error('Curated native component unavailable: '+spec.component);const raw=await checked(review.native_glb),manager=new THREE.LoadingManager();manager.setURLModifier(url=>{if(url.startsWith('blob:')||url.startsWith('data:'))return url;throw Error('External asset resources forbidden.');});const asset=await new Promise((resolve,reject)=>new GLTFLoader(manager).parse(raw,'',resolve,reject));
  asset.scene.traverse(o=>{if(!o.isMesh)return;const materials=Array.isArray(o.material)?o.material:[o.material],basic=materials.map(m=>new THREE.MeshBasicMaterial({color:m.color||0xffffff,map:m.map||null,vertexColors:!!o.geometry.attributes.color,side:THREE.DoubleSide,transparent:m.transparent,opacity:m.opacity}));o.material=Array.isArray(o.material)?basic:basic[0];materials.forEach(m=>m.dispose());});NativeByteRgbDisplayV1.apply(asset.scene,raw,{kind:'verified_sam3d_review_asset',asset_sha256:review.native_glb.sha256});
  const box=new THREE.Box3().setFromObject(asset.scene),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3()),span=Math.max(size.x,size.y,size.z);if(!Number.isFinite(span)||span<=0)throw Error('Finite native bounds required.');const group=new THREE.Group();group.add(asset.scene);group.scale.setScalar(1/span);group.position.copy(center).multiplyScalar(-1/span);scene.add(group);group.updateMatrixWorld(true);parts.push({group,bounds:new THREE.Box3().setFromObject(group)});spec.asset_ref=review.native_glb;spec.display_uniform_scale=1/span;spec.display_translation=group.position.toArray();
 }
 for(let t=0;t<=D;t+=.05)draw(t);draw(0);$('orbit-status').textContent='Ready · four exact original assets · full native bounds verified for all camera samples.';$('orbit-record').disabled=false;
 $('orbit-record').onclick=record;
}
function record(){
 if(recording||disposed)return;const mime=['video/webm;codecs=vp9','video/webm;codecs=vp8'].find(m=>MediaRecorder.isTypeSupported(m));if(!mime)throw Error('Canvas recording unavailable.');recording=true;$('orbit-record').disabled=true;for(const url of urls)URL.revokeObjectURL(url);urls=[];
 const stream=canvas.captureStream(FPS),recorder=new MediaRecorder(stream,{mimeType:mime,videoBitsPerSecond:12000000}),chunks=[],trace=[];let bytes=0,started=null,failed=false,last=-1;
 recorder.ondataavailable=e=>{bytes+=e.data.size;if(bytes>48*1024*1024){failed=true;recorder.stop();}else if(e.data.size)chunks.push(e.data);};
 recorder.onerror=()=>{failed=true;};recorder.onstop=async()=>{cancelAnimationFrame(raf);stream.getTracks().forEach(t=>t.stop());recording=false;if(disposed)return;$('orbit-record').disabled=false;if(failed||!chunks.length||trace.at(-1)?.time_s<D-.02){$('orbit-status').textContent='Incomplete capture withheld.';return;}const blob=new Blob(chunks,{type:mime}),url=URL.createObjectURL(blob);urls.push(url);$('orbit-download').href=url;$('orbit-download').download='native-orbits-20261104-v1.webm';$('orbit-download').hidden=false;const proof={format:'interaction-twin-native-editorial-orbits',version:1,duration_s:D,canvas:[W,W],fps:FPS,assets:specs,trace,geometry_modified:false,poses_recovered:false,editorial_camera_motion_only:true,display_normalization:'uniform centering per independent native asset; original bytes unchanged',native_rgb_policy:NativeByteRgbDisplayV1.POLICY,bytes:blob.size,mime,video_sha256:await digest(await blob.arrayBuffer()),new_API_calls:0,new_remote_GPU_calls:0,camera_or_microphone_capture:false,physical_accuracy_verified:false};const pr=URL.createObjectURL(new Blob([JSON.stringify(proof,null,2)],{type:'application/json'}));urls.push(pr);$('orbit-receipt').href=pr;$('orbit-receipt').download='native-orbits-20261104-v1.json';$('orbit-receipt').hidden=false;$('orbit-status').textContent='Capture complete · all four originals · full native geometry visible.';};
 draw(0);recorder.start(200);$('orbit-status').textContent='Recording four native-asset camera orbits…';
 function frame(now){if(disposed||recorder.state!=='recording')return;if(started===null)started=now;const t=Math.min(D,(now-started)/1000);try{const row=draw(t);if(t-last>=.09||t===D){trace.push(row);last=t;}}catch(e){failed=true;$('orbit-status').textContent=e.message;recorder.stop();return;}if(t===D){recorder.stop();return;}raf=requestAnimationFrame(frame);}raf=requestAnimationFrame(frame);
}
start().catch(e=>{$('orbit-status').textContent=e.message;$('orbit-record').disabled=true;});
addEventListener('pagehide',()=>{disposed=true;cancelAnimationFrame(raf);for(const p of parts)p.group.traverse(o=>{o.geometry?.dispose();for(const m of Array.isArray(o.material)?o.material:[o.material]){m?.map?.dispose();m?.dispose();}});renderer?.dispose();for(const url of urls)URL.revokeObjectURL(url);});
