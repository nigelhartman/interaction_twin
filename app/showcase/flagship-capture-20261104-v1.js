/* Additive production capture. Current curated bytes/current workbench only. */
import * as THREE from 'three';
import {GLTFLoader} from '/workspace/robot-vendor/GLTFLoader.js';
const $=id=>document.getElementById(id),C=globalThis.AlignedSceneContract,F=globalThis.SourceVisibleFlexure;
const ids={phone:'flow-capture-phone-20261103-v1--flow-procedure-1',charger:'flow-capture-charger-20261103-v1--flow-procedure-1',jabra:'flow-sam3d-default-e2e-20261102-v1--flow-procedure-1'};
const W=1920,H=1080,D=8,FPS=30,MAX=64*1024*1024,flip=new THREE.Matrix4().makeScale(1,-1,-1),matrix=r=>new THREE.Matrix4().set(...r.flat());
const digest=async raw=>[...new Uint8Array(await crypto.subtle.digest('SHA-256',raw))].map(v=>v.toString(16).padStart(2,'0')).join('');
const encoded=v=>new TextEncoder().encode(JSON.stringify(v));
let catalogRef,catalog,modelSet='devices',models=[],renderer,disposed=false,recording=false,raf=null,urls=[],robotSession=null,robotPlan=null,robotTimeline=null,robotKit=null,robotTrace=[];
async function checked(ref){
 C.ref(ref,32*1024*1024);const r=await fetch('/files/'+ref.path,{cache:'no-store',credentials:'omit'});if(!r.ok)throw Error('Exact curated input unavailable.');
 const raw=await r.arrayBuffer();if(raw.byteLength!==ref.bytes||await digest(raw)!==ref.sha256)throw Error('Curated input bytes changed.');return raw;
}
function link(id,blob,name){const u=URL.createObjectURL(blob);urls.push(u);$(id).href=u;$(id).download=name;$(id).hidden=false;}
function jsonLink(id,doc,name){link(id,new Blob([JSON.stringify(doc,null,2)+'\n'],{type:'application/json'}),name);}
async function currentCatalog(){
 const r=await fetch('public/library-index.json',{cache:'no-store'});if(!r.ok)throw Error('Current library pointer unavailable.');const i=await r.json();catalogRef={...i.catalog,path:'showcase/public/'+i.catalog.path};
 const raw=await(await fetch('public/'+i.catalog.path,{cache:'no-store'})).arrayBuffer();if(raw.byteLength!==i.catalog.bytes||await digest(raw)!==i.catalog.sha256)throw Error('Current catalog changed.');return InteractionLibrary.validateCatalog(JSON.parse(new TextDecoder().decode(raw)));
}
async function prepare(role,record,componentId){
 const binding=await AlignedScenes.find(record.source.sha256);if(!binding)throw Error('A source-bound scene is required.');let asset,pose,observation=null,representation='saved_native_or_declared_derived_mesh';
 if(role==='phone'&&binding.observed_surfaces){
  const doc=F.validate(binding.observed_surfaces.document,{source_program:binding.manifest.original_program,source_scene:binding.reference,source:binding.manifest.source});
  // This film's inspection shot uses a broadside saved observation. The task
  // evidence remains the distinct connector-interaction observations.
  const requested=Number(new URLSearchParams(location.search).get('phone_sample')||5),selected=doc.samples.find(r=>r.source_sample_index===requested);observation=selected?F.at(doc,selected.timestamp_s,true):null;if(!observation)throw Error('Requested supported source-visible phone observation unavailable.');asset=observation.asset;pose=observation.camera_from_asset;representation='source_visible_2.5D_estimated_depth';
 }else{
  const component=binding.manifest.components.find(c=>c.id===componentId);if(!component)throw Error('Required component unavailable.');asset=component.asset;
  const candidate=component.poses.filter(p=>p.status!=='unknown'&&p.camera_from_native_mesh).sort((a,b)=>(b.convex_silhouette_fit_iou||0)-(a.convex_silhouette_fit_iou||0))[0];if(!candidate)throw Error('A saved supported camera orientation is required.');pose=candidate.camera_from_native_mesh;
 }
 const raw=await checked(asset),manager=new THREE.LoadingManager();manager.setURLModifier(u=>{if(u.startsWith('blob:')||u.startsWith('data:'))return u;throw Error('External model resources are disabled.');});
 const gltf=await new Promise((resolve,reject)=>new GLTFLoader(manager).parse(raw,'',resolve,reject));
 gltf.scene.traverse(o=>{if(!o.isMesh)return;const old=Array.isArray(o.material)?o.material:[o.material],materials=old.map(m=>new THREE.MeshBasicMaterial({color:m.color||0xffffff,map:m.map||null,vertexColors:representation==='source_visible_2.5D_estimated_depth'?false:!!o.geometry.attributes.color,side:representation==='source_visible_2.5D_estimated_depth'?THREE.FrontSide:THREE.DoubleSide,toneMapped:false}));o.material=Array.isArray(o.material)?materials:materials[0];old.forEach(m=>m.dispose());});
 if(representation!=='source_visible_2.5D_estimated_depth')NativeByteRgbDisplayV1.apply(gltf.scene,raw,{kind:'verified_sam3d_scene_asset',asset_sha256:asset.sha256});
 const scene=new THREE.Scene();scene.background=new THREE.Color('#edf0f5');const poseGroup=new THREE.Group();poseGroup.matrixAutoUpdate=false;poseGroup.matrix.copy(flip).multiply(matrix(pose));poseGroup.add(gltf.scene);scene.add(poseGroup);scene.updateMatrixWorld(true);
 const bounds=new THREE.Box3().setFromObject(poseGroup,true),center=bounds.getCenter(new THREE.Vector3()),size=bounds.getSize(new THREE.Vector3()),span=Math.max(size.x,size.y,size.z);if(!Number.isFinite(span)||span<=0)throw Error('Model bounds are empty.');
 const camera=new THREE.PerspectiveCamera(37,W/3/H,.001,span*100);const radius=size.length()/2,distance=radius/Math.sin(37*Math.PI/360)*1.24;
 return{role,asset,scene,poseGroup,camera,center,span,distance,source_sha256:record.source.sha256,scene_ref:binding.reference,source_program:binding.manifest.original_program,representation,observation:observation?{source_sample_index:observation.source_sample_index,timestamp_s:observation.timestamp_s,visible_component_ids:observation.visible_component_ids}:null,display_only_independent_coordinate_system:true};
}
function drawModels(t){
 renderer.setScissorTest(false);renderer.setViewport(0,0,W,H);renderer.setClearColor('#edf0f5');renderer.clear();const trace=[];
 for(const[i,m]of models.entries()){
  const rect=models.length===4?{x:i%2*W/2,y:i<2?H/2:0,w:W/2,h:H/2}:{x:i*W/3,y:0,w:W/3,h:H},aspect=rect.w/rect.h;
  const angle=Math.sin(t/D*Math.PI*2)*(m.representation==='source_visible_2.5D_estimated_depth'?.095:.35),distance=m.distance/Math.min(1,aspect);m.camera.position.copy(m.center).add(new THREE.Vector3(Math.sin(angle)*distance,0,Math.cos(angle)*distance));m.camera.lookAt(m.center);m.camera.updateMatrixWorld(true);m.camera.aspect=aspect;m.camera.updateProjectionMatrix();
  renderer.setViewport(rect.x,rect.y,rect.w,rect.h);renderer.setScissor(rect.x,rect.y,rect.w,rect.h);renderer.setScissorTest(true);renderer.render(m.scene,m.camera);
  const b=new THREE.Box3().setFromObject(m.poseGroup,true),points=[];for(const x of[b.min.x,b.max.x])for(const y of[b.min.y,b.max.y])for(const z of[b.min.z,b.max.z]){const p=new THREE.Vector3(x,y,z).project(m.camera);points.push([(p.x+1)/2*rect.w+rect.x,(1-p.y)/2*rect.h+(H-rect.y-rect.h),p.z]);}
  const top=H-rect.y-rect.h;if(points.some(p=>p[0]<rect.x+12||p[0]>rect.x+rect.w-12||p[1]<top+12||p[1]>top+rect.h-12||p[2]<-1||p[2]>1))throw Error('A full model left its independent viewport.');trace.push({role:m.role,camera_position:m.camera.position.toArray(),projected_bounds:points});
 }
 renderer.setScissorTest(false);return{time_s:t,views:trace};
}
function recordModels(){
 if(recording||disposed)return;const mime=['video/webm;codecs=vp9','video/webm;codecs=vp8'].find(m=>MediaRecorder.isTypeSupported(m));if(!mime)throw Error('Canvas recording is unavailable.');
 recording=true;$('models-record').disabled=true;$('load-parts').disabled=true;$('load-devices').disabled=true;$('status').textContent='Recording actual model viewers…';const kind=modelSet;const stream=$('models').captureStream(FPS),recorder=new MediaRecorder(stream,{mimeType:mime,videoBitsPerSecond:14000000}),chunks=[],trace=[];let total=0,start=null,failed=false;
 recorder.ondataavailable=e=>{total+=e.data.size;if(total>MAX){failed=true;recorder.stop();}else if(e.data.size)chunks.push(e.data);};recorder.onerror=()=>{failed=true;};
 recorder.onstop=async()=>{stream.getTracks().forEach(t=>t.stop());recording=false;$('models-record').disabled=false;$('load-parts').disabled=false;$('load-devices').disabled=false;if(failed||trace.at(-1)?.time_s<D-.02){$('status').textContent='Incomplete model capture withheld.';return;}const blob=new Blob(chunks,{type:mime}),stem=kind==='jabra'?'flagship-four-parts-20261104-v1':'flagship-three-models-20261104-v1';link('models-download',blob,stem+'.webm');const doc={format:'interaction-twin-flagship-multiple-model-capture',version:1,current_catalog:catalogRef,roles:models.map(m=>m.role),models:models.map(({role,asset,source_sha256,scene_ref,source_program,representation,observation,display_only_independent_coordinate_system})=>({role,asset,source_sha256,scene_ref,source_program,representation,observation,display_only_independent_coordinate_system})),duration_s:D,canvas:[W,H],fps:FPS,bytes:blob.size,video_sha256:await digest(await blob.arrayBuffer()),trace,geometry_loaded_actual_WebGL:true,source_assets_modified:false,camera_motion_only:true,charging_or_insertion_executed:false,physical_accuracy_verified:false,new_API_calls:0,new_GPU_calls:0};jsonLink('models-receipt',doc,stem+'.json');$('status').textContent='Complete · actual models, exact curated references and full bounds.';};
 drawModels(0);recorder.start(200);function step(now){if(disposed||recorder.state!=='recording')return;start??=now;const t=Math.min(D,(now-start)/1000);try{trace.push(drawModels(t));}catch(e){failed=true;$('status').textContent=e.message;recorder.stop();return;}if(t===D)recorder.stop();else raf=requestAnimationFrame(step);}raf=requestAnimationFrame(step);
}
async function prepareRobot(){
 const frame=$('robot'),work=frame.contentWindow?.RobotWorkbench;if(!work?.snapshot().ready)return false;work.play(false);robotPlan=work.getPlan();const snapshot=work.snapshot();
 const envelope=await(await fetch('/api/robotics/kit?kit_id='+encodeURIComponent(snapshot.kit_id),{cache:'no-store'})).json();robotKit=envelope.kit;
 RobotTaskPlan.validate(robotPlan,robotPlan,robotKit);robotTimeline=RobotWorkbenchController.compile(RobotTaskPlan.translate(robotPlan,robotKit),{joint_tool_position:(step,q,state)=>RobotTaskPlan.jointToolPosition(robotPlan,step,q,state)});RobotTaskPlan.checkOutcome(robotPlan,robotTimeline);
 if(!robotPlan.goal.component_ids.includes('earbud_a')||!robotPlan.goal.component_ids.includes('earbud_b')||!robotPlan.goal.close_container)throw Error('The complete both-earbud/closed-case task is required.');$('robot-record').disabled=false;$('robot-status').textContent='Current workbench ready · choose a camera in the live workbench, then record.';return true;
}
function recordRobot(){
 const frame=$('robot'),work=frame.contentWindow.RobotWorkbench,canvas=frame.contentDocument.getElementById('workbench-canvas');work.play(false);work.seek(0);robotTrace=[];let started=performance.now();
 const speed=frame.contentDocument.getElementById('task-speed');if(speed&&Number(speed.value)!==1)throw Error('Use 1× playback for a source-clock capture.');
 const sample=()=>{const s=work.snapshot();if(s.ready&&(!robotTrace.length||s.time_s>robotTrace.at(-1).time_s+.015))robotTrace.push({time_s:s.time_s,capture_wall_time_s:(performance.now()-started)/1000,state:s.state});return s;};
 robotSession?.dispose();robotSession=RobotPreviewRecording.create({canvas,getState:sample,startPlayback:()=>{work.play(false);work.seek(0);work.play(true);},stopPlayback:()=>work.play(false),getCaption:()=>frame.contentDocument.getElementById('current-step')?.textContent||'Robot task prototype',onStatus:s=>$('robot-status').textContent=s,onReady:async clip=>{
  const final=work.snapshot(),state=final.state,goals=robotPlan.goal.component_ids.map(id=>({id,error:Math.hypot(...state.objects[id].position.map((v,i)=>v-robotTimeline.final.objects[id].position[i])),attached:state.attached===id})),closed=Object.values(state.object_joints).every(v=>Math.abs(v)<1e-7),fullTrace=robotTrace.length>=200&&robotTrace[0].time_s<=.05&&robotTrace.at(-1).time_s>=robotTimeline.duration_s-.05,complete=fullTrace&&final.time_s>=robotTimeline.duration_s-.05&&state.attached===null&&goals.every(g=>g.error<1e-5&&!g.attached)&&closed;
  if(!complete){$('robot-status').textContent='Incomplete final goal/closed state; clip withheld.';return;}
  $('robot-download').href=clip.url;$('robot-download').download='flagship-current-robot-20261104-v1.'+(clip.mime.startsWith('video/mp4')?'mp4':'webm');$('robot-download').hidden=false;
  const bytes=await(await fetch(clip.url)).arrayBuffer(),doc={format:'interaction-twin-flagship-current-robot-capture',version:1,current_catalog:catalogRef,typed_plan:robotPlan,typed_plan_snapshot_JSON_sha256:await digest(encoded(robotPlan)),robot_kit:robotKit,native_asset_refs:final.native_asset_refs,duration_s:clip.duration_s,task_duration_s:robotTimeline.duration_s,bytes:clip.bytes,video_sha256:await digest(bytes),canvas:[1280,800],geometry_safe_viewport_xyxy:[0,88,1280,718],trace:robotTrace,captured_full_task:true,final_attached:state.attached,final_closed_reference:closed,both_earbuds_released:goals.every(g=>g.error<1e-5&&!g.attached),final_goals:goals,authored_robot_staging:true,physical_contact_verified:false,simulation_ready:false,new_API_calls:0,new_GPU_calls:0};jsonLink('robot-receipt',doc,'flagship-current-robot-20261104-v1.json');$('robot-status').textContent='Complete · both earbuds released on typed goals, closed reference and full task trace.';
 }});robotSession.start();
}
async function start(){
 catalog=await currentCatalog();const phone=catalog.records.find(r=>r.id===ids.phone),charger=catalog.records.find(r=>r.id===ids.charger);if(!phone||!charger)throw Error('Both source recordings are required.');
 renderer=new THREE.WebGLRenderer({canvas:$('models'),antialias:true,preserveDrawingBuffer:true});renderer.setSize(W,H,false);renderer.setPixelRatio(1);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.autoClear=false;
 models=await Promise.all([prepare('phone',phone,'phone_blue'),prepare('cable',charger,'power_cable'),prepare('charger',charger,'adapter_early')]);drawModels(0);$('models-record').disabled=false;$('models-record').onclick=recordModels;$('status').textContent='Ready · actual phone, cable and charger geometry loaded independently.';
 async function switchModels(kind){if(recording)return;$('models-record').disabled=true;$('load-parts').disabled=true;$('load-devices').disabled=true;const row=catalog.records.find(r=>r.id===ids.jabra),newModels=kind==='jabra'?await Promise.all(['case_base','case_lid','earbud_a','earbud_b'].map(id=>prepare(id,row,id))):await Promise.all([prepare('phone',phone,'phone_blue'),prepare('cable',charger,'power_cable'),prepare('charger',charger,'adapter_early')]);for(const old of models)old.scene.traverse(o=>{o.geometry?.dispose();for(const m of Array.isArray(o.material)?o.material:[o.material]){m?.map?.dispose();m?.dispose();}});models=newModels;modelSet=kind;drawModels(0);$('models-record').disabled=false;$('load-parts').disabled=false;$('load-devices').disabled=false;$('status').textContent=kind==='jabra'?'Ready · four independently loaded original case parts.':'Ready · phone, cable and charger loaded together.';}
 $('load-parts').disabled=false;$('load-devices').disabled=false;$('load-parts').onclick=()=>switchModels('jabra').catch(e=>$('status').textContent=e.message);$('load-devices').onclick=()=>switchModels('devices').catch(e=>$('status').textContent=e.message);
 $('robot-record').onclick=recordRobot;const poll=setInterval(async()=>{if(disposed){clearInterval(poll);return;}try{if(await prepareRobot())clearInterval(poll);}catch(e){clearInterval(poll);$('robot-status').textContent=e.message;}},700);
}
start().catch(e=>{$('status').textContent=e.message;});
addEventListener('pagehide',()=>{disposed=true;cancelAnimationFrame(raf);robotSession?.dispose();for(const m of models)m.scene.traverse(o=>{o.geometry?.dispose();for(const p of Array.isArray(o.material)?o.material:[o.material]){p?.map?.dispose();p?.dispose();}});renderer?.dispose();for(const u of urls)URL.revokeObjectURL(u);});
