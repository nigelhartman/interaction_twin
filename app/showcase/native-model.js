import * as THREE from 'three';
import {GLTFLoader} from '/workspace/robot-vendor/GLTFLoader.js';
import {OrbitControls} from '/workspace/robot-vendor/OrbitControls.js';
import './native-byte-rgb-display-v1.js';
const $=id=>document.getElementById(id),canvas=$('model');
let renderer,controls,group,raf,observer,disposed=false;
async function checked(ref){
  InteractionLibrary.ref(ref);
  const response=await fetch('/files/'+ref.path);if(!response.ok)throw Error('Saved model unavailable.');
  const raw=await response.arrayBuffer();if(raw.byteLength!==ref.bytes)throw Error('Saved model length changed.');
  const sha=[...new Uint8Array(await crypto.subtle.digest('SHA-256',raw))].map(v=>v.toString(16).padStart(2,'0')).join('');
  if(sha!==ref.sha256)throw Error('Saved model hash changed.');return raw;
}
async function start(){
  const query=new URLSearchParams(location.search),catalog=await InteractionLibrary.loadCatalog();
  const record=catalog.records.find(r=>r.id===query.get('case'))||catalog.records.find(r=>r.source.sha256===query.get('source'));
  const review=Sam3DModelReviews.forComponent(record,query.get('component'));
  if(!review?.native_glb)throw Error('No verified SAM 3D model is available for this part yet.');
  const raw=await checked(review.native_glb);
  const manager=new THREE.LoadingManager();manager.setURLModifier(url=>{if(url.startsWith('blob:')||url.startsWith('data:'))return url;throw Error('External model resources are disabled.');});
  const asset=await new Promise((resolve,reject)=>new GLTFLoader(manager).parse(raw,'',resolve,reject));
  if(disposed)return;
  renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor(0xedf0f5);renderer.outputColorSpace=THREE.SRGBColorSpace;
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(38,1,.01,100);
  group=new THREE.Group();group.add(asset.scene);scene.add(group);
  asset.scene.traverse(o=>{if(o.isMesh){const original=Array.isArray(o.material)?o.material:[o.material];const basic=original.map(m=>new THREE.MeshBasicMaterial({color:m.color||0xffffff,map:m.map||null,vertexColors:!!o.geometry.attributes.color,side:THREE.DoubleSide,transparent:m.transparent,opacity:m.opacity}));o.material=Array.isArray(o.material)?basic:basic[0];original.forEach(m=>m.dispose());}});
  NativeByteRgbDisplayV1.apply(asset.scene,raw,{kind:'verified_sam3d_review_asset',asset_sha256:review.native_glb.sha256});
  const bounds=new THREE.Box3().setFromObject(asset.scene),center=bounds.getCenter(new THREE.Vector3()),size=bounds.getSize(new THREE.Vector3()),radius=Math.max(size.x,size.y,size.z);
  if(!Number.isFinite(radius)||radius<=0)throw Error('Invalid model bounds.');
  // Display-only centering and uniform normalization; the original asset is unchanged.
  group.position.copy(center).multiplyScalar(-1/radius);group.scale.setScalar(1/radius);
  function reset(){camera.position.set(1.35,.95,1.7);controls?.target.set(0,0,0);controls?.update();}
  controls=new OrbitControls(camera,canvas);controls.enableDamping=true;controls.minDistance=.65;controls.maxDistance=6;controls.autoRotate=query.get('spin')==='1';controls.autoRotateSpeed=1.2;reset();
  $('reset').onclick=reset;canvas.onkeydown=e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','-','='].includes(e.key))return;e.preventDefault();if(e.key.startsWith('Arrow'))group.rotation.y+=(e.key==='ArrowLeft'||e.key==='ArrowUp'?-1:1)*.1;else camera.position.multiplyScalar(e.key==='-'?1.1:.9);};
  observer=new ResizeObserver(()=>{const {width,height}=canvas.getBoundingClientRect();renderer.setSize(width,height,false);camera.aspect=width/Math.max(1,height);camera.updateProjectionMatrix();});observer.observe(canvas);
  $('label').textContent=review.label+' · SAM 3D Objects';$('detail').textContent='Original generated mesh · independent coordinates';$('status').hidden=true;
  function draw(){if(disposed)return;controls.update();renderer.render(scene,camera);raf=requestAnimationFrame(draw);}draw();
}
start().catch(e=>{$('status').hidden=false;$('status').textContent=e.message+' No older model is substituted.';});
addEventListener('pagehide',()=>{disposed=true;cancelAnimationFrame(raf);observer?.disconnect();controls?.dispose();group?.traverse(o=>{o.geometry?.dispose();for(const m of Array.isArray(o.material)?o.material:[o.material]){m?.map?.dispose();m?.dispose();}});renderer?.dispose();});
