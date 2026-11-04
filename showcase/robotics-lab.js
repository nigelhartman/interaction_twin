/* Local evidence-linked authoring sandbox. Robot motion is deliberately illustrative. */
import * as THREE from '../workspace/robot-vendor/three.module.js';
import { OrbitControls } from '../workspace/robot-vendor/OrbitControls.js';
import { GLTFLoader } from '../workspace/robot-vendor/GLTFLoader.js';

const $ = id => document.getElementById(id);
const copy = value => JSON.parse(JSON.stringify(value));
const defaults = () => ({source:[0,.36,0],target:[.95,.18,.45],lift:.48,rotation:[0,0,0],assetSize:.18,contextSize:.72,mode:'out-and-back'});
const V = a => new THREE.Vector3(...a);
const smooth = t => t*t*(3-2*t);
const mix = (a,b,t) => a.map((v,i)=>v+(b[i]-v)*t);
const elevated = (a,h) => [a[0],a[1]+h,a[2]];
let author=defaults(), cases=[], libraryCatalog=null, availableAssets=[], active=null, selectedStep=null, movingAsset=null, contextAsset=null, playing=false, progress=0, previousTick=null, loadVersion=0, assetVersion=0;
let scene,camera,renderer,controls,movingGroup,contextGroup,armGroup,pathGroup,anchorGroup,links=[],joints=[],palm,fingers=[],floor;
const loadingManager=new THREE.LoadingManager();loadingManager.setURLModifier(url=>{if(url.startsWith('blob:')||url.startsWith('data:'))return url;throw new Error('External resource references are not allowed in the native GLB rehearsal.');});
let assetReady=false, programVerified=false, gltfLoader=new GLTFLoader(loadingManager), compiledPhases=[], duration=18, currentObjectPosition=[0,0,0],renderDirty=true;
let publicReplay=null,publicReplayReady=false,replayView='source',publicMode='replay',robotSceneAttempted=false;
const shoulder=[-1.45,.76,-.72],armLengths=[1.55,1.45];
const robotGaps=['verified_metric_object_scale','verified_shape_and_common_scene_registration','actual_robot_kinematic_description_and_joint_limits','gripper_and_grasp_frames','collision_geometries_and_collision_free_motion','mass_inertia_friction_and_contact_parameters','insertion_clearance_and_contact_model','simulated_grasp_stability_and_success_tests'];

function artifactURL(path){
 if(typeof path!=='string'||!path||path.startsWith('/')||path.includes('\\')||!/^[-a-zA-Z0-9_./]+$/.test(path)||path.split('/').some(s=>s==='..'||s===''))throw new Error('The saved artifact path is not a relative local path.');
 return location.protocol==='file:'?new URL('../'+path,import.meta.url).href:new URL('/files/'+path.split('/').map(encodeURIComponent).join('/'),location.origin).href;
}
function error(message){$('error').hidden=!message;$('error').textContent=message||'';}
function text(tag,value,className){const e=document.createElement(tag);e.textContent=value;if(className)e.className=className;return e;}
async function digest(bytes){if(!crypto?.subtle)throw new Error('Artifact verification requires localhost or HTTPS.');return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),n=>n.toString(16).padStart(2,'0')).join('');}
async function verifiedBytes(path,expected,expectedBytes=null){
 if(!/^[a-f0-9]{64}$/.test(expected||''))throw new Error('This artifact has no valid original SHA256.');
 const response=await fetch(artifactURL(path),{cache:'no-store'});if(!response.ok)throw new Error('Saved artifact could not be read ('+response.status+').');
 const bytes=await response.arrayBuffer();if(bytes.byteLength>80*1024*1024)throw new Error('Saved asset exceeds the rehearsal viewer limit.');
 if(expectedBytes!==null&&bytes.byteLength!==expectedBytes)throw new Error('Saved SAM 3D asset length changed.');
 if(await digest(bytes)!==expected)throw new Error('Saved artifact hash mismatch. Original assets have not been loaded.');return bytes;
}

function makePhases(config){
 const a=config.mode==='in'?config.target:config.source,b=config.mode==='in'?config.source:config.target;
 const ah=elevated(a,config.lift),bh=elevated(b,config.lift),home=[-.65,1.05,-.42];
 const p=[];
 const add=(id,title,from,to,objectFrom,objectTo,gripFrom,gripTo,note)=>p.push({id,title,from:[...from],to:[...to],object_from:[...objectFrom],object_to:[...objectTo],grip_from:gripFrom,grip_to:gripTo,note});
 add('approach','Approach the authored anchor',home,ah,a,a,0,0,'Approach clearance is illustrative; obstacles are not checked.');
 add('descend','Descend to the authored grasp',ah,a,a,a,0,0,'This grasp point is authored, not detected from the video.');
 add('grasp','Close the illustrative gripper',a,a,a,a,0,1,'Attachment is animated. No force or grasp stability is computed.');
 add('lift','Lift the generated component',a,ah,a,ah,1,1,'The original learned mesh follows the authored gripper path.');
 add('transfer',config.mode==='in'?'Transfer toward the container':'Transfer to the outside anchor',ah,bh,ah,bh,1,1,'No collision avoidance or actual robot joint limits.');
 add('lower','Lower to the destination',bh,b,bh,b,1,1,'This destination is a staging point, not verified seating.');
 add('release','Release the component',b,b,b,b,1,0,'The component is placed by animation, without dynamics.');
 add('retreat','Retreat from the placement',b,bh,b,b,0,0,'A visual task rehearsal, not a simulated success.');
 if(config.mode==='out-and-back'){
  add('return_descend','Return to the component',bh,b,b,b,0,0,'Reverse transfer is an authored task, not recovered motion.');
  add('return_grasp','Re-grasp the component',b,b,b,b,0,1,'The gripper closure is illustrative.');
  add('return_lift','Lift for the return',b,bh,b,bh,1,1,'Relative component size is normalized for display.');
  add('return_transfer','Move back to the container',bh,ah,bh,ah,1,1,'A desired path, with no hidden contact or assembly inferred.');
  add('return_lower','Lower to the container-side anchor',ah,a,ah,a,1,1,'This does not prove the part fits or is fully inserted.');
  add('return_release','Release at the authored anchor',a,a,a,a,1,0,'Insertion and seating remain unverified.');
  add('return_retreat','Retreat and inspect the draft',a,ah,a,a,0,0,'Export retains every missing simulation requirement.');
 }
 return p.map((phase,i)=>({...phase,start:i/p.length,end:(i+1)/p.length}));
}
function frameAt(phases,t){const u=Math.max(0,Math.min(1,t)),phase=phases[Math.min(phases.length-1,Math.floor(u*phases.length))],local=Math.min(1,Math.max(0,(u-phase.start)/(phase.end-phase.start))),f=smooth(local);return{phase,tip:mix(phase.from,phase.to,f),object:mix(phase.object_from,phase.object_to,f),grip:phase.grip_from+(phase.grip_to-phase.grip_from)*f};}

function mesh(geometry,color,extra={}){return new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color,roughness:.48,metalness:.12,...extra}));}
function sphere(radius,color){return mesh(new THREE.SphereGeometry(radius,24,16),color);}
function rod(radius,length,color){return mesh(new THREE.CylinderGeometry(radius,radius,length,24),color);}
function segment(object,a,b){const x=V(a),y=V(b),d=y.clone().sub(x);object.position.copy(x.add(y).multiplyScalar(.5));object.scale.set(1,d.length(),1);object.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),d.normalize());}
function makeLabel(label,color){const canvas=document.createElement('canvas');canvas.width=128;canvas.height=128;const c=canvas.getContext('2d');c.fillStyle=color;c.beginPath();c.arc(64,64,40,0,Math.PI*2);c.fill();c.font='600 46px sans-serif';c.textAlign='center';c.textBaseline='middle';c.fillStyle='#ffffff';c.fillText(label,64,66);const texture=new THREE.CanvasTexture(canvas);texture.colorSpace=THREE.SRGBColorSpace;const s=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,depthTest:false}));s.scale.set(.2,.2,.2);return s;}
function initializeScene(){
 scene=new THREE.Scene();scene.background=new THREE.Color('#edf0f5');scene.fog=new THREE.Fog('#edf0f5',7,16);
 camera=new THREE.PerspectiveCamera(38,1,.01,40);camera.position.set(3.5,2.8,4.5);
 renderer=new THREE.WebGLRenderer({canvas:$('robot-canvas'),antialias:true,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio||1,2));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.15;renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
 controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,.55,0);controls.enableDamping=true;controls.minDistance=1.5;controls.maxDistance=10;controls.maxPolarAngle=Math.PI*.48;controls.addEventListener('change',()=>{renderDirty=true;});
 scene.add(new THREE.HemisphereLight('#ffffff','#8994a8',2.6));const key=new THREE.DirectionalLight('#ffffff',4);key.position.set(1,6,3);key.castShadow=true;key.shadow.mapSize.set(1024,1024);key.shadow.camera.left=-4;key.shadow.camera.right=4;key.shadow.camera.top=4;key.shadow.camera.bottom=-4;key.shadow.bias=-.001;scene.add(key);
 const fill=new THREE.DirectionalLight('#dce9ff',1.7);fill.position.set(-3,2,-3);scene.add(fill);
 floor=mesh(new THREE.PlaneGeometry(200,200),'#e0e5ed',{roughness:1});floor.rotation.x=-Math.PI/2;floor.position.y=-.015;floor.receiveShadow=true;scene.add(floor);
 const grid=new THREE.GridHelper(8,32,'#b8c3d5','#d7dfe9');grid.position.y=-.009;grid.material.opacity=.48;grid.material.transparent=true;scene.add(grid);
 const tabletop=mesh(new THREE.BoxGeometry(3.5,.045,2.5),'#f0f2f7',{roughness:.9});tabletop.position.set(.15,-.035,.1);tabletop.receiveShadow=true;scene.add(tabletop);
 armGroup=new THREE.Group();scene.add(armGroup);const base=rod(.24,.16,'#8e9aac');base.position.set(shoulder[0],.08,shoulder[2]);base.castShadow=true;armGroup.add(base);const baseRing=rod(.245,.04,'#3766ed');baseRing.position.set(shoulder[0],.18,shoulder[2]);armGroup.add(baseRing);const tower=rod(.15,.55,'#e6eaf2');tower.position.set(shoulder[0],.45,shoulder[2]);tower.castShadow=true;armGroup.add(tower);
 for(let i=0;i<2;i++){const l=rod(i?.077:.095,1,'#f3f5fa');l.castShadow=true;armGroup.add(l);links.push(l);const j=sphere(i?.12:.14,'#53617a');j.castShadow=true;armGroup.add(j);joints.push(j);}
 const wrist=sphere(.095,'#566783');armGroup.add(wrist);joints.push(wrist);
 palm=mesh(new THREE.BoxGeometry(.25,.11,.15),'#465d90');palm.castShadow=true;armGroup.add(palm);
 for(const sign of[-1,1]){const finger=new THREE.Group(),stem=mesh(new THREE.BoxGeometry(.027,.18,.05),'#adbad1'),pad=mesh(new THREE.BoxGeometry(.047,.055,.06),'#2f3d57');stem.position.y=-.095;pad.position.set(-sign*.01,-.17,0);finger.add(stem,pad);finger.userData.sign=sign;armGroup.add(finger);fingers.push(finger);}
 movingGroup=new THREE.Group();contextGroup=new THREE.Group();scene.add(movingGroup,contextGroup);pathGroup=new THREE.Group();anchorGroup=new THREE.Group();scene.add(pathGroup,anchorGroup);
 new ResizeObserver(resize).observe($('viewport'));resize();updateStaging();
}
function resize(){if(!renderer)return;const w=$('viewport').clientWidth,h=$('viewport').clientHeight;if(!w||!h)return;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();renderDirty=true;}
function disposeTree(group){group.traverse(o=>{o.geometry?.dispose();const mats=Array.isArray(o.material)?o.material:[o.material];for(const m of mats){if(!m)continue;for(const k of ['map','normalMap','roughnessMap','metalnessMap','emissiveMap','aoMap'])m[k]?.dispose();m.dispose();}});group.clear();}
function renderArm(grasp,grip){
 const end=[grasp[0],grasp[1]+.25,grasp[2]],delta=V(end).sub(V(shoulder)),r=Math.hypot(delta.x,delta.z),d=delta.length(),reachable=d<=armLengths[0]+armLengths[1]&&d>=Math.abs(armLengths[0]-armLengths[1]);
 const safeD=Math.max(.10001,Math.min(d,armLengths[0]+armLengths[1]-.0001)),dir=new THREE.Vector3(delta.x,0,delta.z).normalize(),theta=Math.atan2(delta.y,r)+Math.acos(THREE.MathUtils.clamp((armLengths[0]**2+safeD**2-armLengths[1]**2)/(2*armLengths[0]*safeD),-1,1));
 const elbow=V(shoulder).add(dir.multiplyScalar(armLengths[0]*Math.cos(theta))).add(new THREE.Vector3(0,armLengths[0]*Math.sin(theta),0)).toArray();
 const displayEnd=reachable?end:V(shoulder).add(delta.normalize().multiplyScalar(safeD)).toArray();segment(links[0],shoulder,elbow);segment(links[1],elbow,displayEnd);joints[0].position.copy(V(shoulder));joints[1].position.copy(V(elbow));joints[2].position.copy(V(displayEnd));palm.position.set(displayEnd[0],displayEnd[1]-.045,displayEnd[2]);
 for(const f of fingers){f.position.set(displayEnd[0]+f.userData.sign*(.13-grip*.06),displayEnd[1]-.075,displayEnd[2]);}
 $('robot-status').classList.toggle('unreachable',!reachable);$('robot-status').textContent=reachable?'Display-only reach calculation · collisions and contact unchecked':'Authored anchor is outside this illustrative arm’s reach · export remains blocked for simulation';return reachable;
}
function updateStaging(){
 compiledPhases=makePhases(author);duration=author.mode==='out-and-back'?18:10;
 if(!scene)return;disposeTree(pathGroup);disposeTree(anchorGroup);
 const points=[];for(const p of compiledPhases){points.push(V(elevated(p.from,.25)),V(elevated(p.to,.25)));}
 const line=new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineDashedMaterial({color:'#8fa7e7',dashSize:.045,gapSize:.035,transparent:true,opacity:.65}));line.computeLineDistances();pathGroup.add(line);
 for(const [i,a]of[author.source,author.target].entries()){
  const color=i?'#8b79c9':'#3766ed',ring=new THREE.Mesh(new THREE.TorusGeometry(.12,.006,8,48),new THREE.MeshBasicMaterial({color}));ring.rotation.x=Math.PI/2;ring.position.copy(V(a));anchorGroup.add(ring);
  const stem=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(a[0],.01,a[2]),V(a)]),new THREE.LineDashedMaterial({color,dashSize:.035,gapSize:.025}));stem.computeLineDistances();anchorGroup.add(stem);
  const label=makeLabel(i?'B':'A',color);label.position.set(a[0],a[1]+.17,a[2]);anchorGroup.add(label);
 }
 pathGroup.visible=$('show-path').checked;anchorGroup.visible=$('show-anchors').checked;
 updateModelTransforms();renderProgress();
}
function updateModelTransforms(){
 if(!movingGroup)return;
 for(const [group,size,rotation]of[[movingGroup,author.assetSize,author.rotation],[contextGroup,author.contextSize,[0,0,0]]]){
  const root=group.children[0];if(!root)continue;root.scale.setScalar(size/root.userData.nativeSpan);root.rotation.set(...rotation.map(n=>n*Math.PI/180));
 }
 contextGroup.position.copy(V(contextOrigin()));
}
function contextOrigin(){const root=contextGroup?.children[0],height=root?.userData.nativeSize?.[1],span=root?.userData.nativeSpan;const aboveTable=Number.isFinite(height)&&span>0?author.contextSize*height/span*.5+.02:.02;return[author.source[0],Math.max(aboveTable,author.source[1]-author.contextSize*.35),author.source[2]];}
function renderProgress(){
 if(!compiledPhases.length)return;renderDirty=true;const f=frameAt(compiledPhases,progress);currentObjectPosition=f.object;
 if(scene){movingGroup.position.copy(V(f.object));renderArm(f.tip,f.grip);}
 $('phase-label').textContent=f.phase.title;$('phase-note').textContent=f.phase.note;$('scrub').value=progress;$('clock').textContent=(progress*duration).toFixed(1)+' / '+duration.toFixed(1)+' s';
}
function playback(on){playing=on&&assetReady;$('play').textContent=playing?'Ⅱ':'▶';$('play').setAttribute('aria-label',playing?'Pause authored robot rehearsal':'Play authored robot rehearsal');}
function tick(now){const dt=previousTick===null?0:Math.min(.1,(now-previousTick)/1000);previousTick=now;if(playing&&!document.hidden){progress+=dt/duration*Number($('speed').value);if(progress>=1){if($('loop').checked)progress%=1;else{progress=1;playback(false);}}renderProgress();}if(renderer){controls.update();if(!document.hidden&&renderDirty){renderer.render(scene,camera);renderDirty=false;}}requestAnimationFrame(tick);}

async function loadNative(asset,group,version){
 if(!asset||!group)return false;
 const bytes=await verifiedBytes(asset.path,asset.sha256,asset.bytes);if(version!==assetVersion)return false;
 const gltf=await gltfLoader.parseAsync(bytes,'');if(version!==assetVersion){disposeTree(gltf.scene);return false;}
 const box=new THREE.Box3().setFromObject(gltf.scene),span=Math.max(...box.getSize(new THREE.Vector3()).toArray());if(!Number.isFinite(span)||span<=0){disposeTree(gltf.scene);throw new Error('Generated mesh has no finite display bounds.');}
 const holder=new THREE.Group();holder.userData.nativeSpan=span;holder.userData.nativeSize=box.getSize(new THREE.Vector3()).toArray();holder.userData.nativeCenter=box.getCenter(new THREE.Vector3()).toArray();const centering=new THREE.Group();centering.position.copy(box.getCenter(new THREE.Vector3()).multiplyScalar(-1));centering.add(gltf.scene);holder.add(centering);
 gltf.scene.traverse(o=>{if(o.isMesh){o.castShadow=true;o.receiveShadow=true;}});disposeTree(group);group.add(holder);updateModelTransforms();return true;
}
function movingTrack(){return active?.interaction.tracklets.find(t=>t.id===movingAsset?.track_id);}
function selectDefaultAssets(c){return RobotRehearsalAssets.defaultAssets(availableAssets,c.interaction.tracklets||[]);}
async function selectAssets(){
 const version=++assetVersion;playback(false);assetReady=false;$('play').disabled=true;$('export-task').disabled=true;error('');
 movingAsset=availableAssets.find(a=>a.track_id===$('moving-asset').value);contextAsset=availableAssets.find(a=>a.track_id===$('container-asset').value)||null;
 if(contextAsset?.track_id===movingAsset?.track_id){contextAsset=null;$('container-asset').value='';}
 $('asset-provenance').textContent=movingAsset?`SAM 3D Objects · unchanged native GLB · ${movingAsset.selected?'selected in this program':'unselected candidate'} · video alignment unverified · ${movingAsset.track_id}`:'SAM 3D geometry is pending for this recording. Source evidence remains available.';
 if(contextGroup)disposeTree(contextGroup);if(movingGroup)disposeTree(movingGroup);
 if(!movingAsset){$('phase-label').textContent='SAM 3D geometry pending';$('phase-note').textContent='Source-linked steps are available. No older model is substituted.';$('scrub').disabled=true;renderStep();return;}
 if(!renderer){$('export-task').disabled=!programVerified||!movingAsset;return;}
 try{const results=await Promise.all([loadNative(movingAsset,movingGroup,version),contextAsset?loadNative(contextAsset,contextGroup,version):Promise.resolve(true)]);if(version!==assetVersion)return;assetReady=results.every(Boolean);$('play').disabled=!assetReady;$('scrub').disabled=!assetReady;$('export-task').disabled=!programVerified||!assetReady;renderProgress();}
 catch(e){if(version===assetVersion){error('Asset loading stopped: '+e.message);assetReady=false;}}
 renderStep();
}
function renderStep(){
 if(!active)return;selectedStep=active.interaction.steps.find(s=>s.id===$('step-select').value)||active.interaction.steps[0]||null;
 $('step-description').textContent=selectedStep?.description||'No procedure step is available.';
 const limits=[...(selectedStep?.limitations||[])];if(selectedStep&&movingAsset&&!selectedStep.participant_track_ids.includes(movingAsset.track_id))limits.push('This step does not identify the selected generated tracklet. The draft makes no physical identity claim between them.');
 $('step-limits').hidden=!limits.length;$('step-limits').textContent=limits.join(' · ');
 $('evidence-thumbnails').replaceChildren();for(const id of selectedStep?.evidence_ids||[]){const e=active.interaction.evidence.find(x=>x.id===id);if(!e)continue;const b=text('button','');b.type='button';b.title='Seek original source to '+e.timestamp_s.toFixed(3)+' seconds';const im=document.createElement('img');im.alt='Cited source sample '+e.source_index;im.loading='lazy';im.src=artifactURL(e.frame_path);b.append(im,text('span',e.timestamp_s.toFixed(3)+' s'));b.onclick=()=>seekSource(e.timestamp_s);$('evidence-thumbnails').append(b);}
}
function seekSource(t){const v=$('source-video');if(v.readyState>=1)v.currentTime=Math.max(0,Math.min(v.duration||active.duration_s,t));else v.addEventListener('loadedmetadata',()=>{v.currentTime=t;},{once:true});}
async function loadCase(id){
 const version=++loadVersion;++assetVersion;playback(false);progress=0;programVerified=false;assetReady=false;availableAssets=[];movingAsset=null;contextAsset=null;selectedStep=null;$('play').disabled=true;$('scrub').disabled=true;$('export-task').disabled=true;error('');
 if(contextGroup)disposeTree(contextGroup);if(movingGroup)disposeTree(movingGroup);
 for(const key of ['moving-asset','container-asset']){$(key).replaceChildren();$(key).disabled=true;}
 const selected=cases.find(c=>c.id===id);active=null;clearPublicReplay();if(publicMode==='workbench')$('workbench-frame').removeAttribute('src');if(!selected){error('That twin is not in the curated public library.');return;}
 $('program-link').removeAttribute('href');$('program-link').textContent='Original program verification pending';
 for(const id of ['open-evidence','workspace-link'])$(id).removeAttribute('href');$('source-video').removeAttribute('poster');$('step-select').replaceChildren();$('step-description').textContent='';$('step-limits').textContent='';$('evidence-thumbnails').replaceChildren();$('asset-provenance').textContent='Verifying generated parts…';
 $('case-select').value=selected.id;$('case-summary').textContent='Verifying the original program and SAM 3D source bindings…';$('source-video').pause();$('source-video').removeAttribute('src');$('phase-label').textContent='Verifying saved source';$('phase-note').textContent='SAM 3D candidates stay separate from the original evidence program.';
 try{
  const bytes=await verifiedBytes(selected.interaction_path,selected.original_sha256);if(version!==loadVersion)return;
  const original=JSON.parse(new TextDecoder().decode(bytes));if(original.case_id!==selected.interaction.case_id)throw new Error('Source program identity differs from the saved catalog.');
  // Read the hash-verified original for every citation. Rehearsal candidates are
  // a separate view and never replace its historical asset list or observations.
  active={...selected,interaction:original};availableAssets=RobotRehearsalAssets.forProgram(libraryCatalog,original);programVerified=true;
  $('case-summary').textContent=availableAssets.length+' generated parts';$('workspace-link').href=$('open-evidence').href='library.html?case='+encodeURIComponent(active.id);if(publicMode==='replay')updatePublicReplay(active,version);if(publicMode==='workbench')loadWorkbenchCase(active.id);$('program-link').href=artifactURL(active.interaction_path);$('program-link').textContent='Original procedure JSON ↗';$('source-video').src=artifactURL(original.source.path);$('source-video').poster=artifactURL(active.poster_path);$('source-caption').textContent=`Original video · ${active.duration_s.toFixed(1)} s · ${original.reasoning.model||'saved model'} procedure`;
  const params=new URLSearchParams(location.search);params.set('case',active.id);history.replaceState(null,'',location.pathname+'?'+params.toString());
  for(const key of ['moving-asset','container-asset']){if(key==='container-asset'){const opt=text('option','No context asset');opt.value='';$(key).append(opt);}for(const a of availableAssets){const track=original.tracklets.find(t=>t.id===a.track_id);const o=text('option',(track?.label||a.track_id)+' · SAM 3D · '+a.track_id);o.value=a.track_id;$(key).append(o);}$(key).disabled=!availableAssets.length;}
  const chosen=selectDefaultAssets(active);$('moving-asset').value=chosen.moving?.track_id||'';$('container-asset').value=chosen.context?.track_id||'';$('step-select').replaceChildren();for(const s of original.steps){const o=text('option',s.title);o.value=s.id;$('step-select').append(o);}
  const suggested=original.steps.find(s=>s.participant_track_ids.includes(chosen.moving?.track_id)&&/remove|extract|out/i.test(s.title))||original.steps.find(s=>s.participant_track_ids.includes(chosen.moving?.track_id));if(suggested)$('step-select').value=suggested.id;
  author=defaults();updateInputs();updateStaging();renderStep();if(!['workbench','physics'].includes(publicMode))await selectAssets();if(version!==loadVersion)return;seekSource(selectedStep?.start_s||0);
 }
 catch(e){if(version===loadVersion){$('case-summary').textContent='This twin is unavailable. Choose another recording.';$('aligned-message').textContent='Source verification stopped. No previous scene is retained.';error('Source verification stopped: '+e.message);}}
}
function slider(parent,label,index,key,min,max,step){const row=text('label','','range-row'),s=text('span',label),input=document.createElement('input'),out=document.createElement('output');input.type='range';input.min=min;input.max=max;input.step=step;input.setAttribute('aria-label',(key==='source'?'Container anchor ':key==='target'?'Outside anchor ':'Asset rotation ')+label);input.dataset.authorKey=key;input.dataset.authorIndex=index;input.oninput=()=>{author[key][index]=Number(input.value);out.textContent=Number(input.value).toFixed(key==='rotation'?0:2);updateStaging();};row.append(s,input,out);$(parent).append(row);}
function updateInputs(){document.querySelectorAll('[data-author-key]').forEach(input=>{const v=author[input.dataset.authorKey][Number(input.dataset.authorIndex)];input.value=v;input.nextElementSibling.textContent=Number(v).toFixed(input.dataset.authorKey==='rotation'?0:2);});for(const [id,key]of[['lift','lift'],['asset-size','assetSize'],['context-size','contextSize']]){$(id).value=author[key];$(id+'-value').textContent=author[key].toFixed(2);}$('mode').value=author.mode;}
function exportDraft(){
 if(!active||!programVerified||!movingAsset)throw new Error('Verify an original source program and choose an asset first.');
 const step=selectedStep,evidence=(step?.evidence_ids||[]).map(id=>active.interaction.evidence.find(e=>e.id===id)).filter(Boolean).map(e=>({id:e.id,timestamp_s:e.timestamp_s,frame_path:e.frame_path,frame_sha256:e.frame_sha256}));
 const refs=[{role:'manipulated_component',asset:movingAsset,group:movingGroup},{role:'context_component',asset:contextAsset,group:contextGroup}].filter(r=>r.asset).map(r=>({...copy(r.asset),role:r.role,original_bytes_verified_in_viewer:assetReady,authored_placement:true,original_evidence_program_asset_list_changed:false,display_size:r.role==='manipulated_component'?author.assetSize:author.contextSize,display_normalization:'independent_maximum_native_bounding_box_span',native_bounds_center:r.group?.children[0]?.userData.nativeCenter||null,native_bounds_size:r.group?.children[0]?.userData.nativeSize||null,display_rotation_degrees:r.role==='manipulated_component'?[...author.rotation]:[0,0,0]}));
 return{format:'interaction-twin-robot-task-draft',version:'0.1',created_at:new Date().toISOString(),status:'illustrative_authoring_only',simulation_ready:false,physical_execution_allowed:false,
  provenance:{case_id:active.id,original_program_path:active.interaction_path,original_program_sha256:active.original_sha256,original_program_hash_verified:programVerified,original_evidence_program_unchanged:true,source_video:copy(active.interaction.source),reasoning:copy(active.interaction.reasoning),geometry_selection:'source_and_track_bound_SAM3D_candidates_separate_from_original_program',robot_motion_origin:'authored_in_browser_not_recovered_video_poses',native_asset_bytes_modified:false},
  evidence_citation:step?{step_id:step.id,title:step.title,start_s:step.start_s,end_s:step.end_s,participant_track_ids:[...step.participant_track_ids],selected_asset_is_step_participant:step.participant_track_ids.includes(movingAsset.track_id),limitations:[...step.limitations],samples:evidence}:null,
  assets:refs,authoring:{coordinate_frame:'separate_authored_staging_Y_up',units:'dimensionless_display_units_not_metres',source_anchor:[...author.source],outside_anchor:[...author.target],anchor_semantics:'desired_component_center_and_gripper_contact_illustration_not_verified_grasp_or_cavity',lift_height:author.lift,mode:author.mode,duration_s:duration,component_rotation_degrees:[...author.rotation],context_origin:contextOrigin(),context_placement_rule:'authored_anchor_offset_with_display_floor_clearance_not_recovered_pose',shape_scale_or_assembly_verified:false},
  robot:{kind:'illustrative_threejs_articulated_arm',actual_robot_model:null,display_reach_solver:'two_link_planar_geometric_illustration_with_base_yaw',shoulder:[...shoulder],link_lengths:[...armLengths],units:'dimensionless_display_units',joint_limits_verified:false,collision_checking:false,dynamics:false,grasp_constraint:'animation_attachment_only',control_commands_generated:false},
  authored_sequence:copy(compiledPhases),readiness:{simulation_ready:false,blocking_requirements:[...robotGaps],simulator_executed:false,real_robot_executed:false,successful_insertion_verified:false},
  limitations:['Robot, gripper, floor and task path are authored illustrative geometry.','Object meshes are genuine learned assets, with unverified shape and physical dimensions.','Per-component display normalization does not establish common metric scale or assembly.','Video evidence and the robot rehearsal use independent timelines.','A cited procedure step does not validate the authored trajectory or a physical grasp.','No contact, seating, forces, collisions, real robot IK, controller or simulator has been validated.']};
}
function downloadTask(){try{const data=exportDraft(),blob=new Blob([JSON.stringify(data,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=(active.interaction.case_id||'interaction')+'-robot-task-draft.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){error(e.message);}}
for(const [parent,key]of[['source-sliders','source'],['target-sliders','target']])for(const [i,axis]of['X','Y','Z'].entries())slider(parent,axis,i,key,i===1?.08:-1.2,i===1?.9:1.5,.01);
for(const [i,axis]of['X','Y','Z'].entries())slider('rotation-sliders',axis,i,'rotation',-180,180,1);
for(const [id,key]of[['lift','lift'],['asset-size','assetSize'],['context-size','contextSize']])$(id).oninput=()=>{author[key]=Number($(id).value);$(id+'-value').textContent=author[key].toFixed(2);updateStaging();};
$('mode').onchange=()=>{author.mode=$('mode').value;progress=0;updateStaging();};$('play').onclick=()=>playback(!playing);$('scrub').oninput=()=>{playback(false);progress=Number($('scrub').value);renderProgress();};$('restart').onclick=()=>{progress=0;renderProgress();};$('show-path').onchange=()=>{if(pathGroup)pathGroup.visible=$('show-path').checked;renderDirty=true;};$('show-anchors').onchange=()=>{if(anchorGroup)anchorGroup.visible=$('show-anchors').checked;renderDirty=true;};$('reset-camera').onclick=()=>{if(!camera)return;camera.position.set(3.5,2.8,4.5);controls.target.set(0,.55,0);controls.update();};$('reset-authoring').onclick=()=>{author=defaults();progress=0;playback(false);updateInputs();updateStaging();};$('moving-asset').onchange=selectAssets;$('container-asset').onchange=selectAssets;$('case-select').onchange=()=>loadCase($('case-select').value);$('step-select').onchange=()=>{renderStep();seekSource(selectedStep?.start_s||0);};$('export-task').onclick=downloadTask;
if(new URLSearchParams(location.search).get('embed')==='1')document.body.classList.add('embedded');
$('robot-canvas').tabIndex=0;$('robot-canvas').setAttribute('aria-description','Illustrative robot task authoring. Space plays or pauses; arrow keys move the authored timeline. Drag to orbit. Source video and robot use independent timelines.');
document.addEventListener('keydown',event=>{if(['INPUT','SELECT','TEXTAREA','VIDEO','BUTTON','A'].includes(event.target.tagName))return;if(event.code==='Space'){event.preventDefault();playback(!playing);}else if(['ArrowLeft','ArrowRight','Home','End'].includes(event.code)){event.preventDefault();playback(false);progress=event.code==='Home'?0:event.code==='End'?1:Math.max(0,Math.min(1,progress+(event.code==='ArrowRight'?.025:-.025)));renderProgress();}});
window.RobotRehearsal={makePhases,frameAt,exportDraft,snapshot:()=>({case_id:active?.id,moving_track:movingAsset?.track_id,context_track:contextAsset?.track_id,geometry_backend:movingAsset?.backend||null,available_sam3d_components:availableAssets.map(a=>a.track_id),original_program_asset_count:active?.interaction.assets?.length||0,program_verified:programVerified,asset_ready:assetReady,simulation_ready:false,progress,playing,phase_count:compiledPhases.length,authoring:copy(author),object_position:[...currentObjectPosition]})};
function ensureRobotScene(){if(robotSceneAttempted)return;robotSceneAttempted=true;try{initializeScene();}catch(e){renderer=null;$('webgl-error').hidden=false;$('webgl-error').textContent='The 3D rehearsal is unavailable: '+e.message+'. Source evidence and the task draft can still be inspected.';}}
updateInputs();requestAnimationFrame(tick);



function clearPublicReplay(){
 publicReplay=null;publicReplayReady=false;
 const frame=$('aligned-frame');frame.removeAttribute('src');frame.hidden=true;
 $('aligned-message').hidden=false;$('aligned-message').textContent='Checking for a source-linked scene…';
 $('replay-orbit').disabled=true;$('replay-source').disabled=true;
 $('scene-program').hidden=true;$('replay-note').textContent='Source pixels and reconstructed motion share one timeline.';
}
async function updatePublicReplay(entry,version){
 try{
  const row=await AlignedScenes.find(entry.interaction.source.sha256);if(version!==loadVersion)return;
  if(!row){$('aligned-message').textContent='Connected replay is not available for this recording yet. Try Quick rehearsal to inspect its generated parts.';$('replay-note').textContent='Source-linked video is available. Connected motion is pending.';return;}
  const reviews=Sam3DModelReviews.forRecording(libraryCatalog.records.find(r=>r.id===entry.id));
  if(!AlignedSceneContract.matchesProgram(row.manifest,entry,reviews,row.derived_assembly))throw Error('Scene components do not match this exact curated program.');
  publicReplay=row;const ref=row.reference,params=new URLSearchParams({source:entry.interaction.source.sha256,path:ref.path,sha256:ref.sha256,bytes:String(ref.bytes),presentation:'landing'});
  $('aligned-frame').src='aligned-scene.html?'+params;$('aligned-frame').hidden=false;$('aligned-message').hidden=true;
  $('replay-orbit').disabled=false;$('replay-source').disabled=false;
  if(row.object_program){$('scene-program').href=artifactURL(row.object_program.path);$('scene-program').textContent=row.derived_assembly?'Download derived assembly program ↓':'Download object program ↓';$('scene-program').hidden=false;}
  const time=AlignedSceneContract.previewTime(row.manifest,row.observed_surfaces?.document).time_s;seekSource(time);
  $('replay-note').textContent='Estimated 3D motion. Physical fit is unverified.';
 }catch(e){if(version===loadVersion){$('aligned-message').hidden=false;$('aligned-message').textContent=e.message+' No substitute scene is shown.';}}
}
function syncPublicReplay(reset=false){
 if(!publicReplay||!publicReplayReady||!active||publicMode!=='replay')return;
 $('aligned-frame').contentWindow.postMessage({type:'interaction-twin:aligned-time',source_sha256:active.interaction.source.sha256,manifest_sha256:publicReplay.reference.sha256,time_s:Math.max(0,Math.min(active.duration_s,$('source-video').currentTime||0)),show_estimated:true,view:replayView,reset},location.origin);
}
function loadWorkbenchCase(id){if(!cases.some(c=>c.id===id))throw Error('Choose an exact curated twin for the workbench.');const src='robot-workbench.html?embed=1&case='+encodeURIComponent(id),frame=$('workbench-frame');if(frame.getAttribute?frame.getAttribute('src')!==src:frame.src!==src)frame.src=src;}
function setPublicMode(mode){
 if(!['replay','robot','workbench','physics'].includes(mode))throw Error('Choose a supported robotics lab mode.');
 publicMode=mode;
 const notes={replay:'Watch the video and connected 3D scene together.',workbench:'Press play to try a robot task with this object.',robot:'Build a simple path. Positions and size are display settings, not recovered robot motion.',physics:'A saved Newton CPU contact experiment with assumed dimensions and materials.'};$('lab-mode-note').textContent=notes[mode];
 const params=new URLSearchParams(location.search);params.set('mode',mode);history.replaceState(null,'',location.pathname+'?'+params.toString());
 for(const name of ['replay','workbench']){$('mode-'+name).setAttribute('aria-selected',String(mode===name));$('mode-'+name).tabIndex=(mode==='workbench'?'workbench':'replay')===name?0:-1;}
 for(const name of ['robot','physics'])$('mode-'+name).setAttribute('aria-pressed',String(mode===name));
 for(const name of ['replay','workbench','robot','physics'])$(name+'-panel').hidden=mode!==name;
 const advanced=$('lab-advanced');advanced.hidden=mode==='workbench';if(mode==='robot'||mode==='physics')advanced.open=true;
 $('lab-case-picker').hidden=mode==='physics';
 const physics=$('physics-frame');if(mode==='physics'&&!physics.src)physics.src='physics-bench.html?embed=1';physics.contentWindow?.postMessage({type:'interaction-twin:physics-visibility',active:mode==='physics'},location.origin);
 const frame=$('workbench-frame');if(mode==='workbench'&&active)loadWorkbenchCase(active.id);frame.contentWindow?.postMessage({type:'interaction-twin:workbench-visibility',active:mode==='workbench'},location.origin);
 if(mode==='replay'){playback(false);if(active&&!publicReplay)updatePublicReplay(active,loadVersion);syncPublicReplay();}
 else{$('source-video').pause();if(mode==='robot'){ensureRobotScene();if(programVerified&&!assetReady)selectAssets().catch(e=>error(e.message));resize();}else playback(false);}
}
$('mode-replay').onclick=()=>setPublicMode('replay');$('mode-robot').onclick=()=>setPublicMode('robot');$('mode-workbench').onclick=()=>setPublicMode('workbench');$('mode-physics').onclick=()=>setPublicMode('physics');
for(const mode of ['source','orbit'])$('replay-'+mode).onclick=()=>{replayView=mode;for(const item of ['source','orbit'])$('replay-'+item).setAttribute('aria-pressed',String(item===mode));syncPublicReplay(mode==='orbit');};
for(const type of ['timeupdate','seeked','loadedmetadata'])$('source-video').addEventListener(type,()=>syncPublicReplay());
addEventListener('message',event=>{
 if(event.origin===location.origin&&event.source===$('physics-frame').contentWindow&&event.data?.type==='interaction-twin:physics-height'){if(Number.isFinite(event.data.height)&&event.data.height>=400&&event.data.height<=2000)$('physics-frame').style.height=Math.ceil(event.data.height)+'px';return;}
 if(event.origin===location.origin&&event.source===$('workbench-frame').contentWindow&&event.data?.type==='interaction-twin:workbench-height'){if(Number.isFinite(event.data.height)&&event.data.height>=400&&event.data.height<=2500)$('workbench-frame').style.height=Math.ceil(event.data.height)+'px';return;}
 const d=event.data;if(!publicReplay||event.origin!==location.origin||event.source!==$('aligned-frame').contentWindow||d?.source_sha256!==active?.interaction.source.sha256||d.manifest_sha256!==publicReplay.reference.sha256)return;
 if(d.type==='interaction-twin:aligned-ready'){publicReplayReady=true;syncPublicReplay();}
 if(d.type==='interaction-twin:aligned-error'){$('aligned-message').hidden=false;$('aligned-message').textContent=d.message;}
});
function sourceTick(){if(publicMode==='replay'&&!$('source-video').paused)syncPublicReplay();requestAnimationFrame(sourceTick);}requestAnimationFrame(sourceTick);
const modes=['replay','workbench'],labTabs=modes.map(mode=>$('mode-'+mode));for(const tab of labTabs)tab.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();const next=event.key==='Home'?0:event.key==='End'?modes.length-1:(labTabs.indexOf(tab)+(event.key==='ArrowLeft'?-1:1)+modes.length)%modes.length;setPublicMode(modes[next]);labTabs[next].focus();});
window.RoboticsLab={setMode:setPublicMode,snapshot:()=>({...window.RobotRehearsal.snapshot(),public_mode:publicMode,curated_only:true,connected_replay:!!publicReplay,source_sha256:active?.interaction.source.sha256})};

try{const response=await fetch('public/cases-index.json',{cache:'no-store'});if(!response.ok)throw new Error('Saved project catalog returned '+response.status);const payload=await response.json();if(payload.curated!==true)throw Error('Only the curated public catalog is supported.');libraryCatalog=await InteractionLibrary.loadCatalog();cases=(payload.cases||[]).filter(c=>{const r=libraryCatalog.records.find(r=>r.id===c.id);return r&&r.interaction.path===c.interaction_path&&r.interaction.sha256===c.original_sha256&&Array.isArray(c.interaction?.tracklets);});const canonical=new Map(libraryCatalog.records.map((r,i)=>[r.id,i]));cases.sort((a,b)=>(canonical.get(a.id)??1000)-(canonical.get(b.id)??1000)||a.id.localeCompare(b.id));$('case-select').replaceChildren();for(const c of cases){const count=RobotRehearsalAssets.forProgram(libraryCatalog,c.interaction).length;const record=libraryCatalog.records.find(r=>r.id===c.id);const opt=text('option',`${record?.title||c.title} · ${count?count+' parts':'3D pending'}`);opt.value=c.id;$('case-select').append(opt);}$('case-select').disabled=false;if(!cases.length)throw new Error('No curated object programs are available yet.');const requested=new URLSearchParams(location.search).get('case');if(requested&&!cases.some(c=>c.id===requested))throw Error('That requested twin is not in the curated public library.');const initial=cases.some(c=>c.id===requested)?requested:[...cases].sort((a,b)=>RobotRehearsalAssets.forProgram(libraryCatalog,b.interaction).length-RobotRehearsalAssets.forProgram(libraryCatalog,a.interaction).length)[0].id;await loadCase(initial);}catch(e){error('The rehearsal could not open: '+e.message);}

if(['replay','robot','workbench','physics'].includes(new URLSearchParams(location.search).get('mode')))setPublicMode(new URLSearchParams(location.search).get('mode'));
