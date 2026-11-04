import * as THREE from 'three';
import {GLTFLoader} from '/workspace/robot-vendor/GLTFLoader.js';
import {OrbitControls} from '/workspace/robot-vendor/OrbitControls.js';
import './nonrigid-observations-contract.js';
import './native-byte-rgb-display-v1.js';
import './source-appearance-states-v1.js';
import './auto-scene-camera-v1.js';
import './source-visible-flexure-v1.js';
import './source-surface-presentation-v1.js';
import './estimated-motion-v1.js';
const C=AlignedSceneContract,$=id=>document.getElementById(id),query=new URLSearchParams(location.search),source=query.get('source'),binding={path:query.get('path'),sha256:query.get('sha256'),bytes:Number(query.get('bytes'))};
let scene,renderer,controls,camera,sourceCamera,manifest,observer,raf,disposed=false,ready=false,view='source',time=0,showEstimated=true,parts=[],fitted=false,showAxes=false,parentStateReceived=false,resetRequested=false;
const pendingStates=[];
const landing=query.get('presentation')==='landing',inspection=landing?globalThis.AutoSceneCameraV1.create():null;
let cameraReturn=null;
if(landing)document.documentElement.dataset.presentation='landing';
const jointAxes=[];
const flip=new THREE.Matrix4().makeScale(1,-1,-1);
let motionHypothesis=null;
let surfaces=null,surfacePresentation=null,surfaceRoot=null,surfaceRow=null,surfaceWanted=null;
const surfaceCache=new Map();let surfaceInFlight=0,surfaceDeferred=null;
async function loadSurfaceSeries(){
 const found=await AlignedScenes.find(source);if(!found)return;
 if(found.reference.path!==binding.path||found.reference.sha256!==binding.sha256||found.reference.bytes!==binding.bytes)throw Error('Observed surfaces belong to another exact scene.');
 if(found.motion_hypothesis){const M=globalThis.EstimatedMotionV1;if(!M)throw Error('Motion hypothesis reader unavailable.');motionHypothesis=M.validate(found.motion_hypothesis.document,manifest,binding);}
 if(!found.observed_surfaces)return;
 const d=found.observed_surfaces.document,F=globalThis.SourceVisibleFlexure;
 if(!F)throw Error('The observed-surface validator is unavailable.');
 F.validate(d,{...manifest,source_scene:binding});
 if(JSON.stringify(d.camera)!==JSON.stringify(manifest.camera)||d.replacement_component_ids.some(id=>!manifest.components.some(c=>c.id===id)))throw Error('Observed surface camera or component identity differs.');
 surfaces=d;if(found.observed_surfaces.presentation){const P=globalThis.SourceSurfacePresentationV1;if(!P)throw Error('Observed presentation validator unavailable.');surfacePresentation=P.validate(found.observed_surfaces.presentation.document,d,found.observed_surfaces.artifact);}
}
function releaseSurface(entry){entry.group?.removeFromParent();entry.group?.traverse(o=>{o.geometry?.dispose();for(const m of Array.isArray(o.material)?o.material:[o.material]){m?.map?.dispose();m?.dispose();}});}
function trimSurfaces(){for(const[key,entry]of surfaceCache){if(surfaceCache.size<=10)break;if(key!==surfaceWanted&&entry.group){releaseSurface(entry);surfaceCache.delete(key);}}}
function updateSurface(){
 if(!surfaces)return;
 surfaceRow=surfacePresentation?globalThis.SourceSurfacePresentationV1.at(surfacePresentation,surfaces,time,showEstimated):globalThis.SourceVisibleFlexure.at(surfaces,time,showEstimated);surfaceWanted=surfaceRow?.asset.sha256??null;
 for(const[key,entry]of surfaceCache)if(entry.group)entry.group.visible=key===surfaceWanted;
 if(!surfaceRow){surfaceDeferred=null;return;}
 const row=surfaceRow,key=row.asset.sha256;if(surfaceCache.has(key))return;
 if(surfaceInFlight>=2){surfaceDeferred=row;return;}surfaceInFlight++;
 const entry={row,group:null};surfaceCache.set(key,entry);
 asset(row.asset).then(model=>{
  delete model.nativeRgbSource;
  if(disposed){entry.group=model.scene;releaseSurface(entry);return;}
  model.scene.traverse(o=>{if(!o.isMesh)return;const old=Array.isArray(o.material)?o.material:[o.material];
   if(!old.some(m=>m.map)&&o.geometry.attributes.color){const color=o.geometry.attributes.color,values=[];for(let i=0;i<color.count;i++){const c=new THREE.Color(color.getX(i),color.getY(i),color.getZ(i)).convertSRGBToLinear();values.push(c.r,c.g,c.b);}o.geometry.setAttribute('color',new THREE.Float32BufferAttribute(values,3));}
   const basic=old.map(m=>new THREE.MeshBasicMaterial({color:m.color||0xffffff,map:m.map||null,vertexColors:!m.map&&!!o.geometry.attributes.color,side:THREE.FrontSide,transparent:m.transparent,opacity:m.opacity,toneMapped:false}));o.material=Array.isArray(o.material)?basic:basic[0];old.forEach(m=>m.dispose());
  });entry.group=model.scene;entry.group.visible=key===surfaceWanted;surfaceRoot.add(entry.group);trimSurfaces();if(key===surfaceWanted)apply();
 }).catch(error=>{if(disposed)return;entry.error=String(error.message||error);if(key===surfaceWanted)apply();}).finally(()=>{surfaceInFlight=Math.max(0,surfaceInFlight-1);if(disposed)return;if(surfaceDeferred?.asset.sha256===surfaceWanted){surfaceDeferred=null;apply();}});
}
let nonrigid=null,cableAnnotation=null,cableCanvas=null,cableContext=null,cableDrawKey=null;
async function loadNonrigid(){
 const found=await AlignedScenes.find(source),ref=found?.derived_assembly?.document?.nonrigid_observations_ref;
 if(!ref)return;
 const N=globalThis.NonrigidObservationsV1;if(!N)throw Error('The declared cable observation validator is unavailable.');
 if(!N.sameRef(found.reference,binding))throw Error('Cable observations belong to another exact scene binding.');
 N.reference(ref);const document=await AlignedScenes.checked(ref);
 const [trace,program]=await Promise.all([AlignedScenes.checked(document.trace_ref),AlignedScenes.checked(document.evidence_program)]);
 nonrigid=N.validate(document,trace,program,manifest);
}
let appearance=null,appearanceTarget=null,appearanceBaseline=null,appearanceAppliedKey=undefined;
async function boundedBytes(ref){C.ref(ref,8*1024*1024);const r=await fetch('/files/'+ref.path,{cache:'force-cache'});if(!r.ok)throw Error('Saved source appearance patch unavailable.');const b=await r.arrayBuffer();if(b.byteLength!==ref.bytes)throw Error('Source appearance patch length changed.');const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',b))].map(v=>v.toString(16).padStart(2,'0')).join('');if(hash!==ref.sha256)throw Error('Source appearance patch hash changed.');return b;}
async function loadAppearance(){const found=await AlignedScenes.find(source),ref=found?.derived_assembly?.document?.source_appearance_states_ref;if(!ref)return;const A=globalThis.SourceAppearanceStatesV1;if(!A)throw Error('The source appearance validator is unavailable.');if(found.reference.path!==binding.path||found.reference.sha256!==binding.sha256||found.reference.bytes!==binding.bytes)throw Error('Source appearance belongs to another exact scene binding.');const d=await AlignedScenes.checked(ref),[previous,program]=await Promise.all([AlignedScenes.checked(d.previous_derived_assembly_ref),AlignedScenes.checked(d.evidence_program)]);A.validate(d,previous,manifest,program);if(d.scene_ref.path!==binding.path||d.scene_ref.sha256!==binding.sha256||d.scene_ref.bytes!==binding.bytes)throw Error('Appearance source scene differs.');const patches=new Map();for(const k of d.keyframes)patches.set(k.id,A.decodePatch(await boundedBytes(k.patch_ref),d.vertex_count));appearance={document:d,patches};}
function bindAppearance(){if(!appearance)return;const part=parts.find(p=>p.component.id===appearance.document.component_id),targets=[];part?.group.traverse(o=>{if(o.isMesh&&o.geometry?.attributes?.color)targets.push(o);});if(targets.length!==1||targets[0].geometry.attributes.color.count!==appearance.document.vertex_count)throw Error('The source RGB patch does not match one exact native vertex inventory.');const mesh=targets[0];appearanceTarget=mesh.geometry.attributes.color.clone();mesh.geometry.setAttribute('color',appearanceTarget);appearanceBaseline=appearanceTarget.array.slice();}
function applyAppearance(){if(!appearance)return null;const key=globalThis.SourceAppearanceStatesV1.at(appearance.document,time),id=key?.id||null;if(appearanceAppliedKey!==id){if(key)globalThis.SourceAppearanceStatesV1.applyPatch(appearanceTarget,appearanceBaseline,appearance.patches.get(key.id));else{appearanceTarget.array.set(appearanceBaseline);appearanceTarget.needsUpdate=true;}appearanceAppliedKey=id;}return key;}
function drawCable(width,height,sourceView=true){
 if(!cableCanvas)return;
 const ratio=Math.min(devicePixelRatio||1,2),key=[view,sourceView,cableAnnotation?.source_index??'',width,height,ratio].join(':');
 if(key===cableDrawKey)return;cableDrawKey=key;
 cableCanvas.width=Math.round(width*ratio);cableCanvas.height=Math.round(height*ratio);
 cableContext.setTransform(ratio,0,0,ratio,0,0);cableContext.clearRect(0,0,width,height);
 if(view!=='source'||!sourceView||!cableAnnotation)return;
 const a=cableAnnotation,aspect=a.width/a.height,w=Math.min(width,height*aspect),h=w/aspect,x=(width-w)/2,y=(height-h)/2;
 cableContext.save();cableContext.beginPath();cableContext.rect(x,y,w,h);cableContext.clip();
 cableContext.beginPath();a.points_px.forEach((p,i)=>{const px=x+p[0]*w/a.width,py=y+p[1]*h/a.height;if(i)cableContext.lineTo(px,py);else cableContext.moveTo(px,py);});
 cableContext.lineJoin='round';cableContext.lineCap='round';cableContext.strokeStyle='#152c55';cableContext.lineWidth=4;cableContext.stroke();cableContext.strokeStyle='#74a4ff';cableContext.lineWidth=2;cableContext.stroke();cableContext.restore();
}
function announce(type,fields={}){if(parent!==window)parent.postMessage({type,source_sha256:source,manifest_sha256:binding.sha256,...fields},location.origin);}
async function asset(ref){C.ref(ref);const response=await fetch('/files/'+ref.path,{cache:'force-cache'});if(!response.ok)throw Error('A native component is unavailable.');const raw=await response.arrayBuffer();if(raw.byteLength!==ref.bytes)throw Error('A component length changed.');const sha=[...new Uint8Array(await crypto.subtle.digest('SHA-256',raw))].map(v=>v.toString(16).padStart(2,'0')).join('');if(sha!==ref.sha256)throw Error('A component hash changed.');const manager=new THREE.LoadingManager();manager.setURLModifier(url=>{if(url.startsWith('blob:')||url.startsWith('data:'))return url;throw Error('External native resources are disabled.');});return new Promise((resolve,reject)=>new GLTFLoader(manager).parse(raw,'',resolve,reject)).then(model=>{model.nativeRgbSource=raw;return model;});}
function matrix(rows){return new THREE.Matrix4().set(...rows.flat());}
function apply(){if(!ready||disposed)return;let visible=0,interpolated=0,saved=0;const statuses=[];
 updateSurface();const surfaceReady=surfaceWanted&&surfaceCache.get(surfaceWanted)?.group,replaced=new Set(surfaces?.replacement_component_ids||[]),surfaceIds=new Set(surfaceReady?surfaceRow?.visible_component_ids||[]:[]);
 const appearanceKey=applyAppearance();
 const connected=motionHypothesis?globalThis.EstimatedMotionV1.sceneAt(manifest,motionHypothesis,time,showEstimated):C.sceneAt(manifest,time,showEstimated),byId=new Map(connected.parts.map(p=>[p.id,p]));
 cableAnnotation=nonrigid?globalThis.NonrigidObservationsV1.at(nonrigid,time,view,showEstimated):null;
 for(const p of parts){const pose=byId.get(p.component.id),flexible=nonrigid&&p.component.id===nonrigid.component_id,observed=replaced.has(p.component.id);p.group.visible=!observed&&!flexible&&Boolean(pose.camera_from_native_mesh);if(p.group.visible){p.group.matrix.copy(flip).multiply(matrix(pose.camera_from_native_mesh));p.group.matrixWorldNeedsUpdate=true;visible++;if(pose.interpolated)interpolated++;else saved++;}if(observed&&surfaceIds.has(p.component.id))visible++;statuses.push({id:p.component.id,label:p.component.label,status:observed?surfaceIds.has(p.component.id)?'estimated':'unknown':flexible?'nonrigid_rigid_replay_withheld':pose.status,visible:observed?surfaceIds.has(p.component.id):p.group.visible,interpolated:!observed&&!flexible&&!!pose.interpolated});}
 if(surfaces){const surfaceError=surfaceWanted&&surfaceCache.get(surfaceWanted)?.error;$('surface-caption').hidden=false;$('surface-caption').textContent=surfaceError?'Surface unavailable':surfaceReady?'Observed surface · '+surfaceRow.timestamp_s.toFixed(3)+'s':surfaceRow?'Loading surface…':'Surface unknown here';$('surface-caption').title=surfaceError||'Source-visible surfaces with estimated monocular depth. Hidden surfaces and numerical physical joints are unknown. Original SAM models remain available separately.';}
 if(!surfaces&&motionHypothesis){$('surface-caption').hidden=!connected.motion_hypothesis?.used;$('surface-caption').textContent='Estimated continuity';$('surface-caption').title='A separate source-conditioned estimate fills an original tracking gap. The original unknown observation is preserved; depth and joint state are estimates.';}
 for(const item of jointAxes){const parentPose=byId.get(item.joint.parent_id),childPose=byId.get(item.joint.child_id),flexible=nonrigid&&[item.joint.parent_id,item.joint.child_id].includes(nonrigid.component_id),observed=[item.joint.parent_id,item.joint.child_id].some(id=>replaced.has(id));item.group.visible=!observed&&!flexible&&showAxes&&Boolean(parentPose?.camera_from_native_mesh)&&Boolean(childPose?.camera_from_native_mesh);if(item.group.visible){const jointWorld=flip.clone().multiply(matrix(parentPose.camera_from_native_mesh)).multiply(matrix(item.joint.parent_from_joint));item.group.matrix.copy(jointWorld);item.group.matrixWorldNeedsUpdate=true;}}
 $('clock').textContent=time.toFixed(3)+' s';$('summary').textContent=visible?visible+' parts · '+saved+' saved estimates'+(interpolated?' · '+interpolated+' between samples':''):cableAnnotation?'':showEstimated?'No supported estimate at this time.':'Estimated motion hidden. No accepted 6D poses are available.';
 if(cableAnnotation)$('summary').textContent+=($('summary').textContent?' · ':'')+'Observed cable · 2D · source '+cableAnnotation.source_index;
 if(appearance)$('summary').textContent+=' · '+(appearanceKey?'Source-observed screen appearance':'Screen appearance unknown · native fallback');
 if(!fitted&&visible){scene.updateMatrixWorld(true);const box=new THREE.Box3();for(const p of parts)if(p.group.visible)box.union(new THREE.Box3().setFromObject(p.group));const activeSurface=surfaceCache.get(surfaceWanted)?.group;if(activeSurface?.visible)box.union(new THREE.Box3().setFromObject(activeSurface,true));if(!box.isEmpty()){const target=box.getCenter(new THREE.Vector3()),span=Math.max(box.getSize(new THREE.Vector3()).length(),.01);controls.target.copy(target);camera.position.copy(target).add(new THREE.Vector3(span*.8,span*.35,span*1.5));controls.minDistance=span*.05;controls.maxDistance=span*20;camera.near=Math.max(span*.0001,.00001);camera.far=Math.max(span*100,1000);camera.updateProjectionMatrix();controls.update();fitted=true;}}
 const reportedJoints=connected.joints.map(j=>replaced.has(j.parent_id)||replaced.has(j.child_id)?{...j,status:'unknown',q_rad:null,reason:'Source-visible depth observation; numerical material joint unknown'}:j);
 const jointText=reportedJoints.map(j=>{const spec=[...manifest.mechanics.joints,...(manifest.mechanics.unresolved_children||[])].find(s=>s.id===j.id),name=spec.label||parts.find(p=>p.component.id===spec.child_id)?.component.label||j.id,unresolved=spec.angle_quality?.angle_visual_consistency===false;return name+': '+(j.q_rad===null?'joint unavailable':(unresolved?'configuration ':'')+Math.round(j.q_rad*180/Math.PI)+(unresolved?'° · opening angle unresolved':'° estimate')+(j.interpolated?' · between source samples':j.source_sample_index!==null&&j.source_sample_index!==undefined?' · source '+j.source_sample_index:''));});
 $('joint-summary').hidden=!jointText.length;$('joint-summary').textContent=jointText.join(' · ');
 announce('interaction-twin:aligned-state',{time_s:time,visible,parts:statuses,joints:reportedJoints,view,show_estimated:showEstimated,...(connected.motion_hypothesis?{motion_hypothesis:connected.motion_hypothesis}:{}),...(surfaces?{observed_surfaces:{representation:'source_visible_2.5D_depth_prior',visible:!!surfaceReady,source_sample_index:surfaceRow?.source_sample_index??null,source_time_error_s:surfaceRow?.source_time_error_s??null,interpolated:false,numerical_joint_parameters:'unknown',hidden_surfaces:'absent',observation_timestamp_s:surfaceRow?.timestamp_s??null,presentation_mode:surfaceRow?.presentation_mode??'strict_source_observation',current_geometry:false}}:{}),...(appearance?{source_appearance:{component_id:appearance.document.component_id,keyframe_id:appearanceKey?.id??null,appearance_label:appearanceKey?.appearance_label??null,status:appearanceKey?'source_observed_appearance_estimate':'unknown_native_fallback',electronics_simulated:false}}:{}),...(nonrigid?{nonrigid:{component_id:nonrigid.component_id,visible:!!cableAnnotation,source_index:cableAnnotation?.source_index??null,depth:'undefined',interpolated:false}}:{})});
}
function inspectionTarget(){scene.updateMatrixWorld(true);const box=new THREE.Box3();for(const p of parts)if(p.group.visible)box.union(new THREE.Box3().setFromObject(p.group));const activeSurface=surfaceCache.get(surfaceWanted)?.group;if(activeSurface?.visible)box.union(new THREE.Box3().setFromObject(activeSurface,true));return box.isEmpty()?new THREE.Vector3(0,0,-1):box.getCenter(new THREE.Vector3());}
function inspectionDepth(){return Math.max(.01,Math.abs(inspectionTarget().z));}
function beginInspection(event){
 if(!landing||!ready)return;if(event.type==='pointermove'&&!event.buttons)return;
 const now=performance.now(),state=inspection.state(now);
 if(state.phase==='source'){
  const target=inspectionTarget(),box=new THREE.Box3();for(const p of parts)if(p.group.visible)box.union(new THREE.Box3().setFromObject(p.group));const activeSurface=surfaceCache.get(surfaceWanted)?.group;if(activeSurface?.visible)box.union(new THREE.Box3().setFromObject(activeSurface,true));
  const span=box.isEmpty()?1:Math.max(box.getSize(new THREE.Vector3()).length(),.01),{width,height}=$('scene').getBoundingClientRect();
  camera.aspect=width/Math.max(1,height);camera.updateProjectionMatrix();controls.target.copy(target);camera.position.copy(target).add(new THREE.Vector3(0,span*.18,span*1.7));camera.lookAt(target);controls.update();camera.updateMatrixWorld(true);
 }
 cameraReturn=null;inspection.activity(now);
}
function draw(){
 if(disposed)return;const inspecting=inspection?.state(performance.now());
 if(!landing||inspecting.phase==='manual')controls?.update();
 if(landing&&inspecting.phase==='return'){
  if(!cameraReturn)cameraReturn={position:camera.position.clone(),quaternion:camera.quaternion.clone(),target:controls.target.clone(),projection:camera.projectionMatrix.clone(),sourceTarget:new THREE.Vector3(0,0,-inspectionDepth())};
  camera.position.copy(cameraReturn.position).lerp(sourceCamera.position,inspecting.progress);camera.quaternion.copy(cameraReturn.quaternion).slerp(sourceCamera.quaternion,inspecting.progress);controls.target.copy(cameraReturn.target).lerp(cameraReturn.sourceTarget,inspecting.progress);
  for(let i=0;i<16;i++)camera.projectionMatrix.elements[i]=cameraReturn.projection.elements[i]+(sourceCamera.projectionMatrix.elements[i]-cameraReturn.projection.elements[i])*inspecting.progress;
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();camera.updateMatrixWorld(true);
 }
 if(landing&&inspecting.phase==='source')cameraReturn=null;
 if(renderer){
  const{width,height}=renderer.domElement.getBoundingClientRect();renderer.setViewport(0,0,width,height);renderer.setScissorTest(false);renderer.setClearColor(0x141a25);renderer.clear();
  const sourceRect=()=>{const aspect=manifest.source.width/manifest.source.height,w=Math.min(width,height*aspect),h=w/aspect;return{x:(width-w)/2,y:(height-h)/2,w,h};};
  if(landing){
   const rect=sourceRect(),p=inspecting.phase==='source'?1:inspecting.phase==='return'?inspecting.progress:0;
   renderer.setViewport(rect.x*p,rect.y*p,width+(rect.w-width)*p,height+(rect.h-height)*p);renderer.setScissor(rect.x*p,rect.y*p,width+(rect.w-width)*p,height+(rect.h-height)*p);renderer.setScissorTest(true);renderer.setClearColor(0xedf0f5);renderer.clear();renderer.render(scene,inspecting.phase==='source'?sourceCamera:camera);renderer.setScissorTest(false);
  }else if(view==='source'){const r=sourceRect();renderer.setViewport(r.x,r.y,r.w,r.h);renderer.setScissor(r.x,r.y,r.w,r.h);renderer.setScissorTest(true);renderer.setClearColor(0xedf0f5);renderer.clear();renderer.render(scene,sourceCamera);renderer.setScissorTest(false);}
  else{renderer.setClearColor(0xedf0f5);renderer.clear();renderer.render(scene,camera);}
  drawCable(width,height,!landing||inspecting.phase==='source');
 }
 raf=requestAnimationFrame(draw);
}
function applyDesiredState(){if(!ready||disposed)return;controls.enabled=landing||view==='orbit';if(resetRequested){fitted=false;inspection?.reset();cameraReturn=null;resetRequested=false;}apply();}
function receiveState(d){
 if(disposed)return;
 // The manifest and native models load independently. Keep bounded, authenticated
 // requests before duration is known; no renderer/controls may be touched yet.
 if(!manifest){pendingStates.push({time_s:d.time_s,view:d.view,show_estimated:d.show_estimated,reset:d.reset===true});if(pendingStates.length>64)pendingStates.shift();return;}
 if(d.time_s>manifest.source.duration_s)return;
 time=d.time_s;showEstimated=d.show_estimated;view=d.view;parentStateReceived=true;resetRequested=resetRequested||d.reset===true;
 applyDesiredState();
}
async function start(){
 C.ref(binding,8*1024*1024);const verified=C.validateScene(await AlignedScenes.checked(binding),source);if(disposed)return;manifest=verified;
 for(const state of pendingStates.splice(0))receiveState(state);
 await loadNonrigid();await loadAppearance();await loadSurfaceSeries();if(disposed)return;
 if(manifest.mechanics?.joints.length||manifest.mechanics?.unresolved_children?.length)$('scene-kind').textContent='Connected scene · estimated';
 // Source-surface mode never reveals these native replacements, even at an
 // unknown observation or in orbit. Keep every validated component/pose in the
 // scene, but fetch/decode only meshes this viewer can actually use. Separate
 // native inspection/downloads retain their original independent loaders.
 const replacedIds=new Set(surfaces?.replacement_component_ids||[]),omitNative=component=>replacedIds.has(component.id)&&appearance?.document.component_id!==component.id;
 const nativeLoading={policy:'omit_permanently_surface_replaced_meshes_only',requested_bytes:0,omitted_bytes:0,omitted_component_ids:[]};
 let nativeAborted=false,loaded;const initialModels=[];try{loaded=await Promise.all(manifest.components.map(async component=>{if(omitNative(component)){nativeLoading.omitted_bytes+=component.asset.bytes;nativeLoading.omitted_component_ids.push(component.id);return{component,model:null};}nativeLoading.requested_bytes+=component.asset.bytes;const model=await asset(component.asset);if(disposed||nativeAborted){releaseSurface({group:model.scene});return{component,model:null};}initialModels.push(model);return{component,model};}));}catch(error){nativeAborted=true;for(const model of initialModels)releaseSurface({group:model.scene});throw error;}if(disposed){nativeAborted=true;for(const model of initialModels)releaseSurface({group:model.scene});return;}
 scene=new THREE.Scene();if(surfaces){surfaceRoot=new THREE.Group();surfaceRoot.matrixAutoUpdate=false;surfaceRoot.matrix.copy(flip);scene.add(surfaceRoot);}renderer=new THREE.WebGLRenderer({canvas:$('scene'),antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.autoClear=false;renderer.outputColorSpace=THREE.SRGBColorSpace;
 if(nonrigid){cableCanvas=document.createElement('canvas');cableCanvas.setAttribute('aria-hidden','true');Object.assign(cableCanvas.style,{position:'absolute',inset:'0',width:'100%',height:'100%',pointerEvents:'none'});$('scene').insertAdjacentElement('afterend',cableCanvas);cableContext=cableCanvas.getContext('2d');if(!cableContext)throw Error('The source-pixel cable overlay is unavailable.');}
 camera=new THREE.PerspectiveCamera(42,1,.001,10000);sourceCamera=new THREE.PerspectiveCamera();sourceCamera.projectionMatrix.copy(matrix(C.projection(manifest.camera.K,manifest.source.width,manifest.source.height)));sourceCamera.projectionMatrixInverse.copy(sourceCamera.projectionMatrix).invert();sourceCamera.matrixAutoUpdate=false;sourceCamera.matrix.identity();sourceCamera.matrixWorld.identity();sourceCamera.matrixWorldInverse.identity();
 for(const{component,model}of loaded){const group=new THREE.Group();group.matrixAutoUpdate=false;group.visible=false;scene.add(group);if(model){group.add(model.scene);model.scene.traverse(o=>{if(o.isMesh){const old=Array.isArray(o.material)?o.material:[o.material],basic=old.map(m=>new THREE.MeshBasicMaterial({color:m.color||0xffffff,map:m.map||null,vertexColors:!!o.geometry.attributes.color,side:THREE.DoubleSide,transparent:m.transparent,opacity:m.opacity}));o.material=Array.isArray(o.material)?basic:basic[0];old.forEach(m=>m.dispose());}});NativeByteRgbDisplayV1.apply(model.scene,model.nativeRgbSource,{kind:'verified_sam3d_scene_asset',asset_sha256:component.asset.sha256});delete model.nativeRgbSource;}parts.push({component,group});}
 bindAppearance();
 for(const joint of manifest.mechanics?.joints||[]){if(replacedIds.has(joint.parent_id)||replacedIds.has(joint.child_id))continue;const part=parts.find(p=>p.component.id===joint.parent_id),box=new THREE.Box3().setFromObject(part.group),length=Math.max(box.getSize(new THREE.Vector3()).length()*part.component.native_uniform_scale*.28,.001),direction=new THREE.Vector3(...joint.axis),group=new THREE.Group();group.matrixAutoUpdate=false;group.visible=false;const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints([direction.clone().multiplyScalar(-length),direction.clone().multiplyScalar(length)]),new THREE.LineBasicMaterial({color:0xefaa58,depthTest:false,transparent:true,opacity:.95}));line.renderOrder=10;const anchor=new THREE.Mesh(new THREE.SphereGeometry(length*.035,10,8),new THREE.MeshBasicMaterial({color:0xffca85,depthTest:false}));anchor.renderOrder=11;group.add(line,anchor);scene.add(group);jointAxes.push({joint,group});}
 if(jointAxes.length){$('joint-axis').hidden=false;$('joint-axis').onclick=()=>{showAxes=!showAxes;$('joint-axis').setAttribute('aria-pressed',String(showAxes));$('joint-axis').textContent=showAxes?'Hide joint axis':'Show joint axis';apply();};}
 controls=new OrbitControls(camera,$('scene'));if(surfaces){controls.minAzimuthAngle=-Math.PI/3;controls.maxAzimuthAngle=Math.PI/3;controls.minPolarAngle=Math.PI*.22;controls.maxPolarAngle=Math.PI*.78;}controls.enableDamping=!landing;controls.enabled=landing||view==='orbit';
 if(landing)for(const name of['pointerdown','pointermove','wheel'])$('scene').addEventListener(name,beginInspection,{capture:true,passive:true});
 observer=new ResizeObserver(()=>{const{width,height}=$('scene').getBoundingClientRect();renderer.setSize(width,height,false);camera.aspect=width/Math.max(1,height);camera.updateProjectionMatrix();});observer.observe($('scene'));
 // Direct inspection opens on a supported saved estimate. Embedded playback
 // receives its source time from the owning video after the ready handshake.
 if(parent===window&&!parentStateReceived){const requested=Number(query.get('t'));time=query.has('t')&&Number.isFinite(requested)&&requested>=0&&requested<=manifest.source.duration_s?requested:C.previewTime(manifest,surfaces).time_s;}
 ready=true;$('status').hidden=true;applyDesiredState();draw();announce('interaction-twin:aligned-ready',{native_loading:nativeLoading});
}
addEventListener('message',event=>{const d=C.message(event,parent,location.origin,source,binding.sha256);if(d)receiveState(d);});
start().catch(error=>{if(disposed)return;$('status').hidden=false;$('status').textContent=error.message+' No substitute scene is shown.';announce('interaction-twin:aligned-error',{message:error.message});});
addEventListener('pagehide',()=>{disposed=true;ready=false;pendingStates.length=0;cancelAnimationFrame(raf);observer?.disconnect();controls?.dispose();for(const p of [...parts,...jointAxes])p.group.traverse(o=>{o.geometry?.dispose();for(const m of Array.isArray(o.material)?o.material:[o.material]){m?.map?.dispose();m?.dispose();}});for(const entry of surfaceCache.values())releaseSurface(entry);surfaceCache.clear();surfaceDeferred=null;renderer?.dispose();cableCanvas?.remove();});
