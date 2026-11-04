/* Read-only inspection of real saved geometry experiments. No inference or authored points. */
import * as THREE from './robot-vendor/three.module.js';
import { OrbitControls } from './robot-vendor/OrbitControls.js';
import { sourceCameraMatrices,letterbox } from './geometry-camera.mjs';
import { createAutomaticCameraPanel, resolveAutomaticCameraHypothesis } from './automatic-camera.js';
import { readCameraPermalink, writeCameraPermalink } from './camera-permalink.mjs';

const $=id=>document.getElementById(id);
const SUMMARY='/files/data/observed-geometry-experiment-20261031/summary.json';
const finite=n=>typeof n==='number'&&Number.isFinite(n);
const vector=(v,n)=>Array.isArray(v)&&v.length===n&&v.every(finite);
const node=(tag,value,cls)=>{const e=document.createElement(tag);if(value!==undefined&&value!==null)e.textContent=String(value);if(cls)e.className=cls;return e;};
const human=s=>String(s).replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase());
let summary,runs=[],run=null,frameIndex=0,loadEpoch=0,frameEpoch=0,loadedPhotoEpoch=0,maskImage=null,selectedPoint=null;
let renderer,scene,camera,controls,pointGroup,cameraGroup,selectedGroup,surfaceGroup,dirty=true,viewCenter=new THREE.Vector3(),viewRadius=1;
let surfaceTexture=null;
let sourceMode=false,sourceViewCamera,inferredGroup,inferredTexture=null,completionData=null,completionEpoch=0;
let exportsData=[],completionExports=[],validationData=[],completionOptions=[],movingData=[],metaReview=null,samParts=null,partAssets=[],trackingComparison=null;
let jobCatalog=[],jobRuntimeAvailable=false,activeJob=null,lastJob=null,jobTimer=null,workflowPolicies=[],savedJobRows=[];
let cameraObjects=[];
let centralCenter=new THREE.Vector3(),centralRadius=1;
let activeCameraBinding=null;
const sameCameraBinding=(a,b)=>!!a&&!!b&&a.caseId===b.caseId&&a.trackId===b.trackId&&a.sourceSha256===b.sourceSha256;

function renderRunChoices(){
 $('run-select').replaceChildren(...runs.map((r,i)=>{const o=node('option',r.id+' · '+human(r.status||'saved')+' · '+(r.label||r.id));o.value=i;return o;}));
}
async function inspectCameraHypothesis(candidate,{scroll=true}={}){
 if(candidate.diagnosticOnly!==true||!candidate.viewer?.data||!sameCameraBinding(activeCameraBinding,candidate.sourceBinding))throw new Error('Only a verified diagnostic camera candidate can be inspected.');
 const id='automatic-camera-'+candidate.id;
 let item=runs.find(r=>r.id===id);
 if(!item){item={id,label:candidate.label+' · diagnostic',status:candidate.viewer.data.status,viewer_url:candidate.viewer.url,original_viewer_url:candidate.viewer.url,mask_provider:'Saved SAM component masks · frozen camera comparison'};runs.push(item);}
 item.verifiedViewerData=candidate.viewer.data;item.cameraBinding=candidate.sourceBinding;item.cameraHypothesisId=candidate.id;
 renderRunChoices();$('run-select').value=String(runs.indexOf(item));await selectRun(runs.indexOf(item));
 if(scroll)$('workbench').scrollIntoView({behavior:'smooth',block:'start'});
}
const automaticCameraPanel=createAutomaticCameraPanel($('automatic-camera-panel'),{
 onInspectFrame:async(frame)=>{if(!sameCameraBinding(activeCameraBinding,frame))throw new Error('The selected reconstruction belongs to another source.');const id=frame.frameId??frame.id??frame.frame_id;const i=run?.frames?.findIndex(f=>String(f.id)===String(id)&&Math.abs(f.timestamp_s-frame.ptsS)<1e-6);if(!(i>=0))throw new Error('This exact frame is absent from the selected reconstruction. Inspect a comparison hypothesis to load its complete frozen frame set.');await selectFrame(i);$('workbench').scrollIntoView({behavior:'smooth',block:'start'});},
 onInspectHypothesis:inspectCameraHypothesis
});
async function verifiedCameraJSON(url,digest){
 if(!/^[a-f0-9]{64}$/.test(digest||''))throw new Error('Missing camera source identity');
 const response=await fetch(safeURL(url),{cache:'no-store'});if(!response.ok)throw new Error('Camera source is unavailable');
 const bytes=await response.arrayBuffer();if(bytes.byteLength>2*1024*1024)throw new Error('Camera binding exceeds its display bound');
 const actual=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
 if(actual!==digest)throw new Error('Camera source identity changed');return JSON.parse(new TextDecoder().decode(bytes));
}
async function bindCameraPanel(item,data,epoch){
 try{
  let binding=item.cameraBinding;
  if(!binding){
   const path=new URL(item.viewer_url,location.href);path.pathname=path.pathname.replace(/[^/]+$/,'protocol.json');
   const protocol=await verifiedCameraJSON(path.href,data.input_protocol_sha256);
   if(!protocol.program_path||!protocol.program_sha256||!protocol.source?.sha256||!protocol.track_id)return;
   const program=await verifiedCameraJSON('/files/'+protocol.program_path,protocol.program_sha256);
   if(program.source?.sha256!==protocol.source.sha256)return;
   binding={caseId:program.case_id,trackId:protocol.track_id,sourceSha256:protocol.source.sha256};
  }
  if(epoch===loadEpoch){activeCameraBinding=binding;await automaticCameraPanel.setBinding(binding);}
 }catch{if(epoch===loadEpoch){activeCameraBinding=null;automaticCameraPanel.setBinding(null);}}
}

function safeURL(value){
 if(typeof value!=='string')throw new Error('A saved local artifact URL is missing.');
 const u=new URL(value,location.href);
 if(u.origin!==location.origin||!u.pathname.startsWith('/files/')||u.search||u.hash||decodeURIComponent(u.pathname).split('/').some(x=>x==='..')||u.pathname.includes('\\'))throw new Error('Only same-origin /files/ artifacts can be inspected.');
 return u.href;
}
async function json(url){const response=await fetch(safeURL(url),{cache:'no-store'});if(!response.ok)throw new Error('Saved experiment could not be read ('+response.status+').');return response.json();}
function error(message){$('error').hidden=!message;$('error').textContent=message||'';}
function disposeGroup(group){if(!group)return;group.traverse(o=>{o.geometry?.dispose();if(Array.isArray(o.material))o.material.forEach(m=>m.dispose());else o.material?.dispose();});group.clear();}
function clearSurface(){disposeGroup(surfaceGroup);surfaceTexture?.dispose();surfaceTexture=null;}
function init3D(){
 try{
  renderer=new THREE.WebGLRenderer({canvas:$('geometry-canvas'),antialias:true,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor('#e7ebe3');renderer.outputColorSpace=THREE.SRGBColorSpace;
  scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(45,1,.001,10000);camera.position.set(2,1.3,2);controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.addEventListener('change',()=>dirty=true);
  sourceViewCamera=new THREE.Camera();inferredGroup=new THREE.Group();pointGroup=new THREE.Group();cameraGroup=new THREE.Group();selectedGroup=new THREE.Group();surfaceGroup=new THREE.Group();scene.add(pointGroup,cameraGroup,selectedGroup,surfaceGroup,inferredGroup);new ResizeObserver(resize).observe($('viewport'));resize();
  renderer.domElement.addEventListener('pointerdown',()=>{if(sourceMode)setViewMode(false);});
  const tick=()=>{if(!document.hidden){if(!sourceMode)controls.update();if(dirty){drawScene();dirty=false;}}requestAnimationFrame(tick);};requestAnimationFrame(tick);
 }catch(e){$('geometry-empty').hidden=false;$('geometry-empty').textContent='3D preview is unavailable in this browser. The actual image evidence and saved measurements remain available.';$('fit-view').disabled=true;$('focus-center').disabled=true;}
}
function resize(){if(!renderer)return;const {clientWidth:w,clientHeight:h}=$('viewport');if(!w||!h)return;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();dirty=true;}
function fit(mode='all'){if(!camera||!controls)return;setViewMode(false);const center=mode==='center'?centralCenter:viewCenter,radius=mode==='center'?centralRadius:viewRadius;controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(1.15,-.75,-1.7).normalize().multiplyScalar(radius*3.1));camera.near=Math.max(radius/10000,1e-6);camera.far=Math.max(viewRadius*1000,10);camera.updateProjectionMatrix();controls.minDistance=radius*.02;controls.maxDistance=viewRadius*100;controls.update();$('view-scope').textContent=mode==='center'?'Central 80% view · all points retained':'All-point view';dirty=true;}
function frameCamera(){const frame=run?.frames?.[frameIndex];return(run?.cameras||[]).find(c=>String(c.frame_id)===String(frame?.id));}
function setViewMode(source){
 sourceMode=!!source&&!!frameCamera();if(controls)controls.enabled=!sourceMode;
 $('view-source').setAttribute('aria-pressed',String(sourceMode));$('view-orbit').setAttribute('aria-pressed',String(!sourceMode));$('view-source').classList.toggle('active',sourceMode);$('view-orbit').classList.toggle('active',!sourceMode);$('photo-opacity').disabled=!sourceMode;
 $('alignment-image').hidden=!sourceMode||loadedPhotoEpoch!==frameEpoch||Number($('photo-opacity').value)===0;
 $('alignment-note').textContent=sourceMode?'Saved camera · letterboxed projection. Source-frame agreement is a construction check.':'Orbit mode · photo comparison hidden.';
 if(sourceMode)$('view-scope').textContent='Saved video camera · comparison view';dirty=true;
}
function drawScene(){
 const w=$('viewport').clientWidth,h=$('viewport').clientHeight,c=frameCamera(),f=run?.frames?.[frameIndex];
 renderer.setScissorTest(false);renderer.setViewport(0,0,w,h);renderer.clear();
 if(!sourceMode||!c||!f){renderer.render(scene,camera);return;}
 const a=sourceCameraMatrices(c.R,c.t,c.K,f.width,f.height,Math.max(viewRadius/10000,1e-6),Math.max(viewRadius*1000,10));sourceViewCamera.matrixAutoUpdate=false;sourceViewCamera.matrix.fromArray(a.world);sourceViewCamera.updateMatrixWorld(true);sourceViewCamera.projectionMatrix.fromArray(a.projection);sourceViewCamera.projectionMatrixInverse.copy(sourceViewCamera.projectionMatrix).invert();
 const rect=letterbox(w,h,f.width,f.height),img=$('alignment-image');Object.assign(img.style,{left:rect.x+'px',top:rect.y+'px',width:rect.width+'px',height:rect.height+'px',opacity:String(Number($('photo-opacity').value)/100)});
 renderer.setViewport(rect.x,rect.y,rect.width,rect.height);renderer.setScissor(rect.x,rect.y,rect.width,rect.height);renderer.setScissorTest(true);const camerasVisible=cameraGroup.visible;cameraGroup.visible=false;renderer.render(scene,sourceViewCamera);cameraGroup.visible=camerasVisible;renderer.setScissorTest(false);renderer.setViewport(0,0,w,h);
}
function pointSupported(point){return point.heldout_scored_projection_supported_in_at_least_two_views===true;}
function pointCloud(points,supported){
 const positions=[],colors=[];
 for(const p of points){if(!vector(p.xyz,3)||pointSupported(p)!==supported)continue;positions.push(...p.xyz);const rgb=vector(p.rgb,3)?p.rgb:[128,150,120];const c=new THREE.Color().setRGB(...rgb.map(n=>Math.max(0,Math.min(255,n))/255),THREE.SRGBColorSpace);colors.push(c.r,c.g,c.b);}
 const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));const material=new THREE.PointsMaterial({vertexColors:true,size:supported?4:3.5,sizeAttenuation:false,transparent:!supported,opacity:supported?1:.7});const cloud=new THREE.Points(geometry,material);cloud.userData.supported=supported;return cloud;
}
function cameraWorld(R,t,v){return[0,1,2].map(i=>R[0][i]*(v[0]-t[0])+R[1][i]*(v[1]-t[1])+R[2][i]*(v[2]-t[2]));}
function validMatrix(v){return Array.isArray(v)&&v.length===3&&v.every(row=>vector(row,3));}
function renderSurface(){
 const s=run?.surface,epoch=loadEpoch;
 $('show-surface').disabled=true;$('show-texture').disabled=true;$('surface-label').textContent='No observed surface patch';
 if(!s||s.kind!=='observed_partial_patch'||!Array.isArray(s.vertices)||!s.vertices.length||s.vertices.length>60000||!s.vertices.every(p=>vector(p,3))||!Array.isArray(s.triangles)||s.triangles.length>100000||!s.triangles.every(t=>Array.isArray(t)&&t.length===3&&t.every(i=>Number.isInteger(i)&&i>=0&&i<s.vertices.length)))return;
 const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(s.vertices.flat(),3));geometry.setIndex(s.triangles.flat());
 const hasColors=Array.isArray(s.colors)&&s.colors.length===s.vertices.length&&s.colors.every(c=>vector(c,3));
 if(hasColors){const colors=s.colors.flatMap(rgb=>{const c=new THREE.Color().setRGB(...rgb.map(n=>Math.max(0,Math.min(255,n))/255),THREE.SRGBColorSpace);return[c.r,c.g,c.b];});geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));}
 const hasUV=Array.isArray(s.uv)&&s.uv.length===s.vertices.length&&s.uv.every(uv=>vector(uv,2));
 if(hasUV)geometry.setAttribute('uv',new THREE.Float32BufferAttribute(s.uv.flat(),2));
 const material=new THREE.MeshBasicMaterial({vertexColors:hasColors,side:THREE.FrontSide,polygonOffset:true,polygonOffsetFactor:1,polygonOffsetUnits:1});
 const mesh=new THREE.Mesh(geometry,material);mesh.userData.hasColors=hasColors;surfaceGroup.add(mesh);$('show-surface').disabled=false;$('surface-label').textContent='Observed patch ('+s.triangles.length.toLocaleString()+' triangles)';
 if(hasUV&&s.texture_url){try{const texture=new THREE.TextureLoader().load(safeURL(s.texture_url),loaded=>{if(epoch!==loadEpoch){loaded.dispose();return;}loaded.colorSpace=THREE.SRGBColorSpace;surfaceTexture=loaded;$('show-texture').disabled=false;applyGeometryToggles();},undefined,()=>{if(epoch===loadEpoch){$('show-texture').disabled=true;$('geometry-note').textContent+=' Source texture could not load; measured point colours remain available.';}});texture.colorSpace=THREE.SRGBColorSpace;}catch{/* Actual vertex colours remain available if source URL is invalid. */}}
}
function frustum(c,scale){
 if(!validMatrix(c.R)||!validMatrix(c.K)||!vector(c.t,3)||c.K[0][0]<=0||c.K[1][1]<=0)return null;
 const frame=(run.frames||[]).find(f=>String(f.id)===String(c.frame_id));if(!frame||!finite(frame.width)||!finite(frame.height))return null;
 const center=cameraWorld(c.R,c.t,[0,0,0]);
 const corners=[[0,0],[frame.width,0],[frame.width,frame.height],[0,frame.height]].map(([u,v])=>cameraWorld(c.R,c.t,[(u-c.K[0][2])/c.K[0][0]*scale,(v-c.K[1][2])/c.K[1][1]*scale,scale]));
 const vertices=[];for(let i=0;i<4;i++){vertices.push(...center,...corners[i],...corners[i],...corners[(i+1)%4]);}
 const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));const line=new THREE.LineSegments(g,new THREE.LineBasicMaterial({color:'#9aaba0',transparent:true,opacity:.5}));line.userData.frameID=String(c.frame_id);return line;
}
function renderGeometry(){
 if(!renderer)return;disposeGroup(pointGroup);disposeGroup(cameraGroup);disposeGroup(selectedGroup);clearSurface();cameraObjects=[];
 const points=Array.isArray(run.points)?run.points.filter(p=>vector(p.xyz,3)):[];
 const bbox=new THREE.Box3();for(const p of points)bbox.expandByPoint(new THREE.Vector3(...p.xyz));
 const centers=(run.cameras||[]).filter(c=>validMatrix(c.R)&&vector(c.t,3)).map(c=>cameraWorld(c.R,c.t,[0,0,0]));
 if(bbox.isEmpty())for(const c of centers)bbox.expandByPoint(new THREE.Vector3(...c));
 if(bbox.isEmpty()){viewCenter.set(0,0,0);viewRadius=1;}else{bbox.getCenter(viewCenter);viewRadius=Math.max(bbox.getSize(new THREE.Vector3()).length()/2,1e-4);}
 centralCenter.copy(viewCenter);centralRadius=viewRadius;
 if(points.length>=10){const bounds=[0,1,2].map(i=>points.map(p=>p.xyz[i]).sort((a,b)=>a-b));const lo=bounds.map(a=>a[Math.floor((a.length-1)*.1)]),hi=bounds.map(a=>a[Math.ceil((a.length-1)*.9)]);const box=new THREE.Box3(new THREE.Vector3(...lo),new THREE.Vector3(...hi));box.getCenter(centralCenter);centralRadius=Math.max(box.getSize(new THREE.Vector3()).length()/2,1e-4);}
 pointGroup.add(pointCloud(points,true),pointCloud(points,false));
 renderSurface();
 const supportedCount=points.filter(pointSupported).length;$('supported-label').textContent='Test-supported points ('+supportedCount+')';$('candidate-label').textContent='Other candidates ('+(points.length-supportedCount)+')';
 for(const c of run.cameras||[]){const line=frustum(c,viewRadius*.13);if(line){cameraObjects.push(line);cameraGroup.add(line);}}
 $('fit-view').disabled=points.length===0&&cameraObjects.length===0;$('focus-center').disabled=points.length<10;
 $('geometry-status').textContent=points.length.toLocaleString()+' saved points · '+cameraObjects.length+' viewpoints';$('geometry-empty').hidden=points.length>0;
 if(!points.length)$('geometry-empty').textContent='No accepted sparse reconstruction. Inspect the image features and failure reason; no substitute object is shown.';
 applyGeometryToggles();fit();
}
function applyGeometryToggles(){if(!pointGroup)return;for(const c of pointGroup.children)c.visible=$(c.userData.supported?'show-supported':'show-candidates').checked;cameraGroup.visible=$('show-cameras').checked;surfaceGroup.visible=$('show-surface').checked;for(const mesh of surfaceGroup.children){const useTexture=!!surfaceTexture&&$('show-texture').checked;mesh.material.map=useTexture?surfaceTexture:null;mesh.material.vertexColors=!useTexture&&mesh.userData.hasColors;mesh.material.wireframe=$('show-wireframe').checked;mesh.material.needsUpdate=true;}if(inferredGroup){inferredGroup.visible=$('show-inferred').checked;for(const mesh of inferredGroup.children){const textured=!!inferredTexture&&$('inferred-photo').checked;mesh.material.map=textured?inferredTexture:null;mesh.material.color.set(textured?'#ffffff':'#dca136');mesh.material.wireframe=$('show-wireframe').checked;mesh.material.needsUpdate=true;}}$('inferred-live').hidden=!$('show-inferred').checked;dirty=true;}
function highlightFrame(){const frame=run?.frames?.[frameIndex];for(const o of cameraObjects){const selected=o.userData.frameID===String(frame?.id);o.material.color.set(selected?'#2b794e':'#9aaba0');o.material.opacity=selected?1:.4;}dirty=true;}
function showMetrics(metrics){
 const box=$('metrics');box.replaceChildren();
 const m=metrics||{},format=value=>value===null||value===undefined?'—':typeof value==='number'?(Number.isInteger(value)?value.toLocaleString():Math.abs(value)<.01&&value!==0?value.toExponential(2):Number(value.toFixed(3)).toLocaleString()):typeof value==='boolean'?(value?'Yes':'No'):String(value);
 const photo=validationData.find(d=>d.source_run_id===run?.id)?.metrics;
 const rows=run?.kind==='observed_surface'?[['Training views',format(run.frames.length)],['Stereo candidates',format(m.bounded_candidates)],['Pair photo passes',format(m.pair_photometric_candidates)],['Surface points',format(m.accepted_partial_surface_points)],['Visible triangles',format(m.observed_local_triangles)],['Extra train support',m.accepted_partial_surface_points>0?'≥ 2 views':'—'],['Test views with pose',photo?photo.heldout_views_with_saved_anchor_pose+' / '+photo.heldout_views_total:'—'],['Camera calibrated','No']]:[['Source views',format(m.selected_views)],['Feature tracks',format(m.feature_tracks)],['Sparse points',format(m.recovered_points)],['Registered fit views',format(m.registered_train_views)+' / '+format(m.training_views)],['Passing test views',format(m.heldout_views_passing)+' / '+format(m.heldout_views)],['Scored test observations',format(m.heldout_scored_observations)],['Held-out median · px',format(m.heldout_median_px)],['Held-out 90th · px',format(m.heldout_p90_px)]];
 for(const [label,value]of rows){const cell=node('div','', 'metric');cell.append(node('strong',value),node('span',label));box.append(cell);}
 const details=$('extra-metrics');details.replaceChildren();for(const [key,value]of Object.entries(m)){if(value!==null&&!['number','string','boolean'].includes(typeof value))continue;details.append(node('dt',human(key)),node('dd',format(value)));}
}
async function selectRun(index){
 const epoch=++loadEpoch;activeCameraBinding=null;automaticCameraPanel.setBinding(null);frameEpoch++;error('');$('run-select').disabled=true;$('run-reason').textContent='Reading saved result…';
 loadedPhotoEpoch=0;$('alignment-image').removeAttribute('src');$('asset-downloads').replaceChildren();$('export-status').textContent='Reading this record’s own export status…';$('photo-validation').replaceChildren(node('p','Reading this record’s own checks…'));$('completion-info').replaceChildren(node('p','Reading matching completion candidates…'));$('completion-select').replaceChildren();$('completion-select').disabled=true;$('preview-completion').disabled=true;
 $('mask-provenance').textContent='';
 $('camera-warning').hidden=true;$('camera-warning').textContent='';
 run=null;maskImage=null;selectedPoint=null;cameraObjects=[];
 completionEpoch++;completionData=null;disposeGroup(inferredGroup);inferredTexture?.dispose();inferredTexture=null;$('show-inferred').checked=false;$('show-inferred').disabled=true;$('inferred-label').textContent='Inferred gap fill';$('inferred-live').hidden=true;setViewMode(false);
 for(const group of[pointGroup,cameraGroup,selectedGroup])disposeGroup(group);clearSurface();dirty=true;
 $('frame-image').hidden=true;$('image-empty').hidden=false;$('image-empty').textContent='Reading this experiment’s original frames…';
 const overlay=$('feature-overlay');overlay.getContext('2d').clearRect(0,0,overlay.width,overlay.height);
 $('geometry-empty').hidden=false;$('geometry-empty').textContent='Reading this experiment’s recovered geometry…';$('geometry-status').textContent='Loading saved geometry';
 $('metrics').replaceChildren();$('extra-metrics').replaceChildren();$('limitations').replaceChildren();$('result-explanation').textContent='Reading the experiment’s own result.';$('raw-link').hidden=true;
 $('frame-status').textContent='No frame selected';$('frame-time').textContent='—';
 $('frame-note').textContent='Loading this experiment’s own image observations…';
 for(const id of['previous','next','frame-slider'])$(id).disabled=true;
 try{
  const item=runs[index],data=item.verifiedViewerData?structuredClone(item.verifiedViewerData):await json(item.viewer_url);if(epoch!==loadEpoch)return;
  run=data;bindCameraPanel(item,data,epoch);frameIndex=0;selectedPoint=null;maskImage=null;run.frames=Array.isArray(run.frames)?run.frames:[];run.points=Array.isArray(run.points)?run.points:[];
  $('run-reason').textContent=run.reason||human(run.status||item.status||'Saved result');$('mask-provenance').textContent='Masks: '+(item.mask_provider||run.mask_provider||'Provider not specified in this saved experiment.');$('raw-link').href=safeURL(item.viewer_url);$('raw-link').hidden=false;
  const dense=run.kind==='observed_surface';
  $('geometry-title').textContent=dense?'02 / Observed visible surface':'02 / Recovered sparse geometry';
  $('result-explanation').textContent=(run.reason||'Read the saved metrics and image observations before assessing this result.')+'\n\n'+(dense?'The observed layer contains stereo samples that pass mask, texture and additional training-view checks. Adjacent accepted samples form a source-textured patch. Missing regions remain absent in that layer. Optional completion adds separate, explicitly inferred faces. Image agreement does not establish accurate physical shape.':'A sparse reconstruction is a collection of matched surface points. It is not a complete, textured object. Held-out image checks test consistency conditional on the saved masks.')+'\n\nCoordinate scale is arbitrary. The camera and physical 3D shape remain unverified.';
  $('geometry-note').textContent=dense?'Observed triangles join adjacent accepted samples. The texture comes from the source frame. Optional inferred fill is a separate layer with unknown physical confidence. Held-out image checks below cover limited regions; shape and scale remain unverified.':'The selected frame’s viewpoint is highlighted. “Test-supported” means the saved point was independently scored in at least two held-out views. It does not establish correct physical geometry. All other points remain candidates.';
  $('show-candidates').checked=!dense;$('show-supported').checked=true;$('show-surface').checked=true;$('show-texture').checked=true;
  $('camera-warning').hidden=false;
  $('camera-warning').textContent=item.cameraBinding?'Unselected camera hypothesis · focal '+Number(run.cameras?.[0]?.K?.[0]?.[0]).toFixed(1)+' px · unknown scale. Image agreement does not establish physical accuracy.':run.camera_assumption?'Learned camera assumption · fresh reconstruction · unknown scale. Consistent frame estimates do not establish accurate intrinsics.':run.calibration?'Camera estimate remains '+(run.calibration.observable?'provisional':'ambiguous')+' · fitted focal '+Number(run.calibration.focal_px??run.cameras?.[0]?.K?.[0]?.[0]).toFixed(1)+' px · no measured calibration. Training checks do not establish reliable lens parameters.':dense?'Diagnostic surface · '+(String(item.source_sparse_status).includes('self_calibration')?'fitted but ambiguous':'assumed')+' focal '+(finite(item.stereo_focal_px)?item.stereo_focal_px.toFixed(1)+' px':'unknown')+' · unknown scale. A plausible-looking patch is not a calibrated scan.':'Assumed camera intrinsics · unknown scale · physical shape not established.';
  const limits=[...(Array.isArray(summary.limitations)?summary.limitations:[]),...(Array.isArray(run.limitations)?run.limitations:[])];$('limitations').replaceChildren(...[...new Set(limits)].map(value=>node('li',value)));
  if(!limits.length)$('limitations').append(node('li','No complete surface, metric scale, hidden geometry or physically accurate assembly is established.'));
  showMetrics(run.metrics);renderGeometry();$('frame-slider').max=Math.max(0,run.frames.length-1);$('frame-slider').disabled=!run.frames.length;
  renderAppPanels(item,epoch).catch(()=>{if(epoch!==loadEpoch)return;$('asset-downloads').replaceChildren();$('export-status').textContent='Saved sidecar unavailable. No export is substituted.';$('completion-select').disabled=true;$('preview-completion').disabled=true;$('completion-info').replaceChildren(node('p','Optional completion record unavailable.'));$('photo-validation').replaceChildren(node('p','Optional test report unavailable.'));});renderMethodSwitch();
  // Prefer an actual registered view for initial inspection; every source sample remains selectable.
  const sourceID=run.surface?.source_frame_id;
  await selectFrame(Math.max(0,run.frames.findIndex(frame=>sourceID?String(frame.id)===String(sourceID):frame.registered)));
 }catch(e){if(epoch!==loadEpoch)return;error(e.message);$('run-reason').textContent='Saved experiment unavailable.';$('geometry-empty').textContent='No geometry loaded for this experiment.';$('image-empty').textContent='No source frame loaded for this experiment.';$('result-explanation').textContent='The saved result could not be read. No other experiment is being substituted.';}finally{if(epoch===loadEpoch)$('run-select').disabled=runs.length===0;}
}
async function selectFrame(index){
 if(!run)return;const epoch=++frameEpoch;frameIndex=Math.max(0,Math.min(index,run.frames.length-1));const frame=run.frames[frameIndex];maskImage=null;
 $('frame-slider').value=frameIndex;$('previous').disabled=!frame||frameIndex===0;$('next').disabled=!frame||frameIndex===run.frames.length-1;highlightFrame();
 const img=$('frame-image');img.hidden=true;loadedPhotoEpoch=0;$('alignment-image').removeAttribute('src');$('feature-overlay').getContext('2d').clearRect(0,0,$('feature-overlay').width,$('feature-overlay').height);$('image-empty').hidden=false;
 $('view-source').disabled=!frameCamera();if(sourceMode&&!frameCamera())setViewMode(false);$('alignment-image').hidden=true;
 if(!frame){$('image-empty').textContent='This experiment has no saved source frames.';$('frame-time').textContent='—';$('frame-status').textContent='No saved frame';return;}
 $('image-empty').textContent='Loading original frame…';$('frame-time').textContent=finite(frame.timestamp_s)?frame.timestamp_s.toFixed(3)+' s':'Time unavailable';$('frame-status').textContent=(frameIndex+1)+' / '+run.frames.length+' · '+(frame.split==='heldout'?'Test view · ':'')+(frame.registered?'Registered':'Not registered');
 try{
  const source=safeURL(frame.image_url);await new Promise((resolve,reject)=>{const loader=new Image();loader.onload=()=>resolve(loader);loader.onerror=()=>reject(new Error('Original source frame could not be loaded.'));loader.src=source;}).then(loaded=>{if(epoch!==frameEpoch)return;if(loaded.naturalWidth!==frame.width||loaded.naturalHeight!==frame.height)throw new Error('Saved image dimensions do not match the experiment.');img.src=source;$('alignment-image').src=source;loadedPhotoEpoch=epoch;img.hidden=false;$('image-empty').hidden=true;setViewMode(sourceMode);});
  if(epoch!==frameEpoch)return;
  const canvas=$('feature-overlay');canvas.width=frame.width;canvas.height=frame.height;drawFeatures();
  if(frame.mask_url){try{const mask=await new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=reject;image.src=safeURL(frame.mask_url);});if(epoch===frameEpoch&&mask.naturalWidth===frame.width&&mask.naturalHeight===frame.height){maskImage=mask;drawFeatures();}}catch{if(epoch===frameEpoch&&$('show-mask').checked)$('frame-note').textContent='The saved mask could not be loaded. Feature observations remain available.';}}
 }catch(e){if(epoch===frameEpoch){$('image-empty').textContent=e.message;$('image-empty').hidden=false;}}
}
function drawFeatures(){
 const frame=run?.frames?.[frameIndex];if(!frame)return;const canvas=$('feature-overlay'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
 const unit=Math.max(1,canvas.width/650),obs=Array.isArray(frame.observations)?frame.observations:[];
 if($('show-mask').checked&&maskImage){const scratch=document.createElement('canvas');scratch.width=canvas.width;scratch.height=canvas.height;const c=scratch.getContext('2d');c.drawImage(maskImage,0,0);const d=c.getImageData(0,0,scratch.width,scratch.height);for(let i=0;i<d.data.length;i+=4){const on=d.data[i]+d.data[i+1]+d.data[i+2]>60;d.data[i]=130;d.data[i+1]=205;d.data[i+2]=95;d.data[i+3]=on?65:0;}c.putImageData(d,0,0);ctx.drawImage(scratch,0,0);}
 let projected=0,heldout=0,anchors=0,stereo=0;
 for(const o of obs){if(!vector(o.xy,2))continue;const [x,y]=o.xy,color=o.role==='heldout'?'#ff619b':o.role==='pose_anchor'?'#ffd163':'#45a2ff';if(o.role==='heldout')heldout++;if(o.role==='pose_anchor')anchors++;if(o.role==='stereo_match')stereo++;const chosen=String(o.track_id)===String(selectedPoint);
  if($('show-projections').checked&&vector(o.predicted_xy,2)){projected++;const [px,py]=o.predicted_xy;ctx.strokeStyle=color;ctx.lineWidth=unit*.9;ctx.globalAlpha=.8;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(px,py);ctx.stroke();ctx.globalAlpha=1;ctx.lineWidth=unit*1.3;ctx.beginPath();ctx.moveTo(px-3*unit,py-3*unit);ctx.lineTo(px+3*unit,py+3*unit);ctx.moveTo(px-3*unit,py+3*unit);ctx.lineTo(px+3*unit,py-3*unit);ctx.stroke();}
  if($('show-features').checked){ctx.fillStyle=color;ctx.strokeStyle='#14222e';ctx.lineWidth=unit;ctx.beginPath();ctx.arc(x,y,(chosen?5:2.6)*unit,0,Math.PI*2);ctx.fill();ctx.stroke();if(chosen){ctx.strokeStyle='#fff';ctx.beginPath();ctx.arc(x,y,7*unit,0,Math.PI*2);ctx.stroke();}}
 }
 const selected=obs.find(o=>String(o.track_id)===String(selectedPoint));
 $('frame-note').textContent=selected?'Track '+selected.track_id+' · '+(selected.role==='heldout'?'held-out score':selected.role==='pose_anchor'?'test-view pose anchor':'geometry fit')+' observation · '+(finite(selected.error_px)?selected.error_px.toFixed(3)+' px saved projection residual':'no saved 3D projection')+'. This is an image-space consistency check, not physical accuracy.':obs.length+' saved observations · '+heldout+' held-out scores · '+anchors+' pose anchors · '+projected+' displayed projections. Only saved points are drawn; a small residual alone does not establish correct physical shape.';
 if(run.kind==='observed_surface')$('frame-note').textContent=stereo?stereo+' saved stereo matches · source-pair reprojections are construction checks, not held-out accuracy. Extra training views provide photometric support.':(frame.observations_reason||'No pairwise stereo observations are displayed in this view. Extra training-view support is stored per point.');
 if(frame.feature_filter&&finite(frame.feature_filter.core_margin_px))$('frame-note').textContent+=' Features restricted to mask core (at least '+Number(frame.feature_filter.core_margin_px.toFixed(2))+' px); raw mask unchanged.';
}
function selectPoint(event){
 const frame=run?.frames?.[frameIndex];if(!frame)return;const canvas=$('feature-overlay'),r=canvas.getBoundingClientRect();const scale=Math.min(r.width/frame.width,r.height/frame.height),left=r.left+(r.width-frame.width*scale)/2,top=r.top+(r.height-frame.height*scale)/2,x=(event.clientX-left)/scale,y=(event.clientY-top)/scale;
 let nearest=null,best=(14/scale)**2;for(const o of frame.observations||[]){if(!vector(o.xy,2))continue;const d=(x-o.xy[0])**2+(y-o.xy[1])**2;if(d<best){nearest=o;best=d;}}
 selectedPoint=nearest?.track_id??null;drawFeatures();if(selectedGroup){disposeGroup(selectedGroup);const p=run.points.find(p=>String(p.id)===String(selectedPoint));if(p&&vector(p.xyz,3)){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(p.xyz,3));selectedGroup.add(new THREE.Points(g,new THREE.PointsMaterial({color:'#f54890',size:10,sizeAttenuation:false,depthTest:false})));}dirty=true;}
}
function link(label,url,cls='evidence-link'){const a=node('a',label,cls);a.href=safeURL(url);a.target='_blank';a.rel='noopener';return a;}
function stat(label,value){const d=node('div');d.append(node('strong',value),node('span',label));return d;}
function selectRecord(id){const i=runs.findIndex(r=>r.id===id);if(i<0)return;$('run-select').value=i;selectRun(i);}
function renderMethodSwitch(){const host=$('method-switch');host.replaceChildren();const choices=[['camera-body-stereo-v1','Original camera surface'],['camera-body-stereo-refined-k-v2','Focal-fit candidate · ambiguous']];if(run?.kind==='observed_surface'&&!choices.some(([id])=>id===run.id))choices.push([run.id,'Selected workflow surface']);for(const[id,label]of choices){const item=runs.find(r=>r.id===id);if(!item)continue;const button=node('button',label);button.classList.toggle('active',item.id===run?.id);button.setAttribute('aria-pressed',String(item.id===run?.id));button.onclick=()=>selectRecord(item.id);host.append(button);}host.append(node('span','All attempts, point-track studies and failures stay in the selector.','method-caption'));}
function renderTrackingStudy(){
 const host=$('tracking-study-body');if(!host||!trackingComparison)return;host.replaceChildren();const table=node('table'),head=node('tr');for(const title of['Same 35 source views','Sparse points','Train views registered','Test views passing'])head.append(node('th',title));const thead=node('thead');thead.append(head);table.append(thead);const body=node('tbody');
 for(const a of trackingComparison.attempts.filter(a=>a.group==='camera_same_views')){const m=a.metrics,row=node('tr'),label=node('td'),button=node('button',a.method==='sift'?'Image features':'Tracked points');button.onclick=()=>{selectRecord(a.id);$('workbench').scrollIntoView({behavior:'smooth'});};label.append(button);row.append(label,node('td',m.recovered_points),node('td',m.registered_train_views+' / '+m.training_views),node('td',m.heldout_views_passing+' / '+m.heldout_views));body.append(row);}table.append(body);host.append(table,node('p','Tracked points increase conditional temporal coverage on this matched camera test. Different point sets and uncertain camera calibration prevent a physical accuracy claim. Its downstream surface is smaller and has no eligible independent photo checks; the original surface remains preferred.'));
 const base=trackingComparison.attempts.find(a=>a.id==='jabra-base-49-fresh-masks-v3');if(base){const p=node('p','Separate moving-base experiment: '+base.metrics.recovered_points+' sparse points. The lid remains unsupported, and no common assembly or hinge was recovered. This diagnostic is not bound as a replacement production asset.');const b=node('button','Inspect moving-base points →');b.onclick=()=>{selectRecord(base.id);$('workbench').scrollIntoView({behavior:'smooth'});};host.append(p,b);}host.append(link('All attempts & independent audit ↗',summary.extensions.tracking_comparison));
}
async function loadExtensions(){
 const entries=Object.entries(summary.extensions||{}),results=await Promise.allSettled(entries.map(async([key,url])=>[key,await json(url)]));
 for(const result of results){if(result.status!=='fulfilled'){console.warn('A saved studio sidecar could not be read.');continue;}const[k,d]=result.value;if(!d||typeof d!=='object')continue;if(k==='exports')exportsData=Array.isArray(d.exports)?d.exports:[];if(k==='completion_exports')completionExports=Array.isArray(d.exports)?d.exports:[];if(k==='validation')validationData=Array.isArray(d.runs)?d.runs:[];if(k==='moving_parts')movingData=Array.isArray(d.components)?d.components:[];if(k==='meta_review')metaReview=d;if(k==='sam_parts')samParts=d;if(k==='part_assets')partAssets=Array.isArray(d.components)?d.components:[];if(k==='tracking_comparison')trackingComparison=d;}
 exportsData.push(...(summary.workflow_exports||[]));validationData.push(...(summary.workflow_validation||[]));completionExports.push(...(summary.workflow_completion_exports||[]));completionOptions=Array.isArray(summary.completion_candidates)?summary.completion_candidates:[];renderTrackingStudy();try{renderMovingParts();}catch{$('moving-components').replaceChildren(node('p','Optional component records could not be read.'));}try{renderMetaReview();}catch{$('meta-review').replaceChildren(node('p','Optional Meta advice record could not be read.'));}
}
async function renderAppPanels(item,epoch){
 const downloads=$('asset-downloads');downloads.replaceChildren();const asset=exportsData.find(e=>e.source_run_id===item.id);
 $('export-status').textContent=asset?'Source-textured GLB, open surface, arbitrary scale. The evidence bundle includes original frames, masks and provenance.':'This record has no accepted surface export. Its source evidence and failure remain inspectable.';
 if(asset){downloads.append(link('Download observed GLB ↓',asset.glb_url),link('Evidence bundle ↓',asset.zip_url),link('Provenance ↗',asset.metadata_url));}
 const validation=validationData.find(d=>d.source_run_id===item.id),box=$('photo-validation');box.replaceChildren();
 if(validation){try{const d=await json(validation.report_url);if(epoch!==loadEpoch)return;const m=d.metrics,stats=node('div',undefined,'studio-stats');stats.append(stat('points checked / total',m.unique_points_checked+' / '+m.total_dense_points),stat('agree / eligible checks',m.photo_consistent_checks+' / '+m.eligible_photo_checks),stat('test views with pose',m.heldout_views_with_saved_anchor_pose+' / '+m.heldout_views_total));box.append(stats,node('p','Conditional image agreement on limited textured regions. Other test views lack a saved pose. These checks do not establish physical shape accuracy.'));const strip=node('div',undefined,'validation-strip');for(const f of d.frames.filter(f=>f.overlay_url)){const a=link('',f.overlay_url,'');const img=node('img');img.src=safeURL(f.overlay_url);img.alt='Held-out photometric check at '+f.timestamp_s.toFixed(3)+' seconds';a.append(img,node('span',f.timestamp_s.toFixed(3)+' s · '+(f.counts.photo_consistent||0)+' agree / '+(f.counts.photo_inconsistent||0)+' disagree'));strip.append(a);}box.append(strip,link('Full check record ↗',validation.report_url));}catch{if(epoch===loadEpoch)box.append(node('p','Saved test report unavailable. No validation is being substituted.'));}}
 else box.append(node('p','No dense held-out photometric report for this record. Sparse test results, where available, are shown above.'));
 if(epoch!==loadEpoch)return;
 const options=completionOptions.filter(c=>c.source_run_id===item.id);$('completion-select').replaceChildren(...options.map((o,i)=>{const option=node('option',o.label);option.value=i;return option;}));$('completion-select').disabled=!options.length;$('preview-completion').disabled=true;$('completion-info').replaceChildren(node('p',options.length?'Loading a separate inferred completion candidate…':'No matching completion candidate. Missing geometry stays absent.'));
 if(options.length)await loadCompletion(options[0],epoch);
 if(epoch!==loadEpoch)return;const u=writeCameraPermalink(new URL(location.href),item);history.replaceState(null,'',u.pathname+u.search+u.hash);
}
async function loadCompletion(option,epoch=loadEpoch){
 const thisCompletion=++completionEpoch;
 disposeGroup(inferredGroup);inferredTexture?.dispose();inferredTexture=null;completionData=null;$('show-inferred').checked=false;$('show-inferred').disabled=true;$('preview-completion').disabled=true;applyGeometryToggles();
 try{const d=await json(option.viewer_url);if(epoch!==loadEpoch||thisCompletion!==completionEpoch)return;const s=d.inferred_surface;if(!s){$('completion-info').replaceChildren(node('p',d.reason||'This completion candidate did not pass its checks.'));return;}
 const original=runs.find(r=>r.id===run.id);if(safeURL(d.observed_surface_url)!==safeURL(original.original_viewer_url)&&safeURL(d.observed_surface_url)!==safeURL(original.viewer_url))throw new Error('Completion is not bound to this observed surface');
 if(!Array.isArray(s.vertices)||s.vertices.length>60000||!s.vertices.every(p=>vector(p,3))||!Array.isArray(s.triangles)||s.triangles.length>100000||!s.triangles.every(t=>Array.isArray(t)&&t.length===3&&t.every(i=>Number.isInteger(i)&&i>=0&&i<s.vertices.length)))throw new Error('Invalid inferred patch');
 const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(s.vertices.flat(),3));g.setIndex(s.triangles.flat());if(Array.isArray(s.uv)&&s.uv.length===s.vertices.length&&s.uv.every(p=>vector(p,2)))g.setAttribute('uv',new THREE.Float32BufferAttribute(s.uv.flat(),2));
 inferredGroup.add(new THREE.Mesh(g,new THREE.MeshBasicMaterial({color:'#dca136',side:THREE.FrontSide,polygonOffset:true,polygonOffsetFactor:1,polygonOffsetUnits:1})));completionData=d;$('show-inferred').disabled=false;$('inferred-label').textContent='Inferred fill ('+s.triangles.length.toLocaleString()+' triangles)';$('preview-completion').disabled=false;
 const info=$('completion-info');info.replaceChildren(node('p',d.reason||d.title),node('p',s.triangles.length.toLocaleString()+' added triangles · every added face is inferred, with unknown physical confidence. The observed surface stays unchanged.'));
 if(d.depth_completion_report)info.append(node('p','Learned depth is a visual hypothesis. Source-view compatibility does not establish detailed shape, cavities or contact.'));
 if(d.depth_completion_report&&summary.completion_mask_check_url)info.append(link('Native SAM mask filter check ↗',summary.completion_mask_check_url));
 if(String(d.status).includes('component_review'))info.append(node('p','Component boundaries need review; an imperfect mask may include hands or neighbouring parts.'));
 const exported=completionExports.find(e=>e.completion_id===d.id);if(exported)info.append(link('Optional observed + inferred GLB ↓',exported.glb_url),link('Completion provenance ↗',exported.metadata_url));
 info.append(link('Inspect completion record ↗',option.viewer_url));
 if(s.texture_url&&g.attributes.uv)new THREE.TextureLoader().load(safeURL(s.texture_url),texture=>{if(epoch!==loadEpoch||completionData?.id!==d.id){texture.dispose();return;}texture.colorSpace=THREE.SRGBColorSpace;inferredTexture=texture;applyGeometryToggles();});
 applyGeometryToggles();
 }catch(e){if(epoch===loadEpoch&&thisCompletion===completionEpoch)$('completion-info').replaceChildren(node('p','Completion unavailable: '+e.message));}
}
function renderMovingParts(){
 const host=$('moving-components');host.replaceChildren();const camera=node('article',undefined,'component-card');camera.append(node('span','SOURCE-DERIVED SURFACE','part-state'),node('h3','Camera body'),node('p','A rigid component reconstructed in its own coordinate system. Door and removable batteries need separate component assets.'));const b=node('button','Inspect body surface →');b.onclick=()=>selectRecord('camera-body-stereo-v1');camera.append(b);host.append(camera);
 for(const part of movingData){if(!part?.metrics)continue;const card=node('article',undefined,'component-card');card.append(node('span','SEPARATE COMPONENT TEST','part-state'),node('h3',part.label),node('p',part.metrics.selected_views+' source views · '+part.metrics.recovered_points+' recovered points. Current views lack enough stable surface matches; no shape or pose is invented.'));const button=node('button','Inspect source & mask →');button.onclick=()=>selectRecord(part.id);card.append(button);host.append(card);}
 if(samParts){for(const part of(Array.isArray(samParts.proposals)?samParts.proposals:[])){const frames=Array.isArray(part.frames)?part.frames:[],count=frames.reduce((n,f)=>n+(Array.isArray(f.instances)?f.instances.length:0),0);const card=node('article',undefined,'component-card');card.append(node('span',count?'META SAM 3.1 · INDEPENDENT IMAGE MASKS':'META SAM 3.1 · NO NATIVE MATCH','part-state'),node('h3',part.label),node('p',count?'Separate masks from model-localized visual prompts. Motion and physical identity remain unverified.':'The saved text-only request returned no native matches. This is a preserved failed proposal.'));const f=frames.find(f=>f.overlay_url);if(f){const a=link('',f.overlay_url,'');const image=node('img');image.src=safeURL(f.overlay_url);image.alt=part.label+' SAM 3.1 masks';image.style.width='100%';image.style.borderRadius='7px';a.append(image);card.append(a);}const asset=partAssets.find(a=>a.proposal_id===part.id);if(asset){card.append(node('p','Local TripoSR preview · separate model-inferred asset; shape, scale and placement unvalidated.'),link('Inspect this 3D part ↗',asset.review_url),link('Download part GLB ↓',asset.glb_url));}host.append(card);}}
}
function renderMetaReview(){
 const host=$('meta-review');host.replaceChildren();if(!metaReview){host.append(node('p','No saved Meta reconstruction review is available yet. Opening this page does not start a paid request.'));return;}
 $('meta-review-status').textContent=metaReview.model+' · saved visual review';host.append(node('p',metaReview.summary));const grid=node('div',undefined,'insight-grid');
 for(const p of(Array.isArray(metaReview.proposed_components)?metaReview.proposed_components:[])){const card=node('article');card.append(node('h3',p.label),node('small',human(p.motion_type)+' · proposed component'),node('p',p.reason),node('small','SAM concept: '+p.sam_prompt));const f=(Array.isArray(metaReview.input_frames)?metaReview.input_frames:[]).find(f=>(Array.isArray(p.evidence_ids)?p.evidence_ids:[]).includes(f.id));if(f)card.append(link('Source evidence ↗',f.image_url));grid.append(card);}host.append(grid);
 const actions=node('ul',undefined,'capture-list');for(const a of(metaReview.capture_actions||[]).slice(0,6))actions.append(node('li',a.action));host.append(actions,node('p','Model advice is fallible. It does not establish joint axes, metric geometry or physical contact.','advisory-footnote'));
}
async function localAPI(path,body){
 if(!/^\/api\/observed\/(?:catalog|jobs(?:\/observed-[a-f0-9]{24}(?:\/cancel)?)?)$/.test(path))throw new Error('Invalid local job path');
 const response=await fetch(path,body===undefined?{cache:'no-store'}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});let data;try{data=await response.json();}catch{throw new Error('Local reconstruction worker is unavailable.');}if(!response.ok)throw new Error(typeof data.error==='string'?data.error.slice(0,180):'The local worker rejected this operation.');return data;
}
function jobOptions(preferCache=false){
 const item=jobCatalog.find(c=>c.id===$('job-case').value),caches=(item?.sparse_caches||[]).filter(c=>c.track_id===$('job-component').value),gated=$('job-workflow').value==='gated-local-v1';
 $('job-cache').replaceChildren(...caches.map(c=>{const o=node('option',c.label||c.id);o.value=c.id;return o;}));
 const cacheOption=[...$('job-provider').options].find(o=>o.value==='reuse_verified');if(cacheOption)cacheOption.disabled=!caches.length;
 const policy=workflowPolicies.find(w=>w.id==='gated-local-v1');for(const option of $('job-provider').options){if(policy?.provider_availability?.[option.value]===false)option.disabled=true;}for(const option of $('job-completion').options){option.disabled=policy?.completion_availability?.[option.value]===false;}
 if(preferCache)$('job-provider').value=caches.length?'reuse_verified':'sift';
 if($('job-provider').value==='reuse_verified'&&!caches.length)$('job-provider').value='sift';
 const learnedOption=[...$('job-camera').options].find(o=>o.value==='moge_prior');if(learnedOption)learnedOption.disabled=policy?.camera_availability?.moge_prior!==true||$('job-provider').value==='reuse_verified';
 if(!gated||$('job-provider').value==='reuse_verified')$('job-camera').value='assumed';$('job-camera').disabled=!gated||$('job-provider').value==='reuse_verified';
 $('job-provider').disabled=!gated;$('job-completion').disabled=!gated;$('job-cache-label').hidden=!gated||$('job-provider').value!=='reuse_verified';
 const component=item?.components?.find(c=>c.track_id===$('job-component').value);$('start-observed').disabled=!!activeJob||!component?.eligible||!jobRuntimeAvailable;
 $('job-policy').textContent=gated?'Saved RGB and SAM masks → sparse geometry → eligible dense surface → held-out checks → optional separate fill → exports. Unsupported stages keep their reasons. Select an experimental camera estimate only with fresh reconstruction. No paid requests or original asset replacement.':'A bounded CPU image-feature test on up to 35 saved views. Sparse candidates and failures remain separate. No paid requests or original asset replacement.';
}
function jobComponents(){const item=jobCatalog.find(c=>c.id===$('job-case').value),requested=new URLSearchParams(location.search).get('track');$('job-component').replaceChildren(...(item?.components||[]).map(c=>{const o=node('option',c.label+' · '+c.source_observations+' observations');o.value=c.track_id;o.disabled=!c.eligible;return o;}));$('job-component').disabled=!item;const eligible=item?.components?.find(c=>c.eligible&&c.track_id===requested)||item?.components?.find(c=>c.eligible);if(eligible)$('job-component').value=eligible.track_id;jobOptions(true);}
async function connectJobs(){try{const previous=$('job-case').value,d=await localAPI('/api/observed/catalog');jobRuntimeAvailable=d.runtime_available===true;jobCatalog=Array.isArray(d.cases)?d.cases:[];workflowPolicies=Array.isArray(d.workflows)?d.workflows:[];$('job-case').replaceChildren(...jobCatalog.map(c=>{const o=node('option',c.title);o.value=c.id;return o;}));const params=new URLSearchParams(location.search),requested=params.get('case')||previous;if(jobCatalog.some(c=>c.id===requested))$('job-case').value=requested;$('job-case').disabled=!jobCatalog.length||!jobRuntimeAvailable;jobComponents();$('observed-job-status').textContent=jobRuntimeAvailable?'Ready · existing source pixels and masks only.':'Prepared local model environment is unavailable.';const saved=await localAPI('/api/observed/jobs');savedJobRows=Array.isArray(saved.jobs)?saved.jobs:[];$('saved-observed-job').replaceChildren(...savedJobRows.map(j=>{const o=node('option',(j.label||j.track_id)+' · '+human(j.workflow_status||j.status)+' · '+j.id.slice(-8));o.value=j.id;return o;}));const running=savedJobRows.find(j=>['created','queued','running','cancellation_requested'].includes(j.status));if(running)watchJob(running.id);else{const chosen=savedJobRows.find(j=>j.id===params.get('job'))||savedJobRows.find(j=>j.case_id===requested)||savedJobRows[0];if(chosen)showJob(chosen);}}catch(e){$('start-observed').disabled=true;$('observed-job-status').textContent=e.message+' Existing saved reconstructions remain available.';}}
function showJob(job){
 lastJob=job;const running=['created','queued','running','cancellation_requested'].includes(job.status);activeJob=running?job.id:null;$('cancel-observed').hidden=!running;$('cancel-observed').disabled=job.cancellation_requested===true;const selected=jobCatalog.find(c=>c.id===$('job-case').value)?.components?.find(c=>c.track_id===$('job-component').value);$('start-observed').disabled=running||!selected?.eligible||!jobRuntimeAvailable;
 const savedIndex=savedJobRows.findIndex(row=>row.id===job.id);if(savedIndex<0)savedJobRows.unshift(job);else savedJobRows[savedIndex]=job;
 let savedOption=[...$('saved-observed-job').options].find(o=>o.value===job.id);if(!savedOption){savedOption=node('option');savedOption.value=job.id;$('saved-observed-job').prepend(savedOption);}savedOption.textContent=(job.label||job.track_id)+' · '+human(job.workflow_status||job.status)+' · '+job.id.slice(-8);$('saved-observed-job').value=job.id;
 $('observed-job-status').textContent=(job.label||job.track_id||'Component')+' · '+human(job.workflow_status||job.status)+(job.result_status?' · '+human(job.result_status):'')+(job.phase?' · '+human(job.phase):'');$('open-observed-result').hidden=!job.viewer_url;
 $('job-stages').replaceChildren(...(job.stages||[]).map(s=>{const row=node('li',undefined,'stage-'+s.status),top=node('div');top.append(node('strong',human(s.stage)),node('span',human(s.status)+(finite(s.elapsed_s)?' · '+s.elapsed_s.toFixed(1)+' s':'')));row.append(top);if(s.reason)row.append(node('p',s.reason));if(s.result?.gate?.reasons?.length)row.append(node('p',s.result.gate.reasons.join(' · ')));if(s.result?.camera_report_url)row.append(link('Camera estimate record ↗',s.result.camera_report_url));return row;}));
 const actions=$('job-capture');actions.replaceChildren();if(job.capture_diagnostics?.length){actions.append(node('strong','What would help this component'));const ul=node('ul');for(const a of job.capture_diagnostics)ul.append(node('li',a.action||a.code));actions.append(ul);}const downloads=$('job-downloads');downloads.replaceChildren();if(job.exports?.observed)downloads.append(link('Observed GLB ↓',job.exports.observed.glb_url),link('Source bundle ↓',job.exports.observed.zip_url));if(job.exports?.inferred)downloads.append(link('Observed + inferred GLB ↓',job.exports.inferred.glb_url));if(job.workflow_manifest_url)downloads.append(link('Stage report ↗',job.workflow_manifest_url));
}
async function watchJob(id){clearTimeout(jobTimer);try{const j=await localAPI('/api/observed/jobs/'+id);showJob(j);if(activeJob)jobTimer=setTimeout(()=>watchJob(id),1500);}catch(e){activeJob=null;$('cancel-observed').hidden=true;$('observed-job-status').textContent='Job status unavailable: '+e.message+' Refresh sources to reconnect.';}}
$('job-case').onchange=jobComponents;$('job-component').onchange=()=>jobOptions(true);$('job-provider').onchange=()=>jobOptions();$('job-workflow').onchange=()=>jobOptions();$('job-camera').onchange=()=>jobOptions();$('refresh-job-catalog').onclick=connectJobs;$('start-observed').onclick=async()=>{if(activeJob)return;$('start-observed').disabled=true;try{const body={case_id:$('job-case').value,track_id:$('job-component').value};if($('job-workflow').value==='gated-local-v1'){Object.assign(body,{workflow:'gated-local-v1',feature_provider:$('job-provider').value,completion:$('job-completion').value,camera_mode:$('job-camera').value});if(body.feature_provider==='reuse_verified')body.reuse_sparse_id=$('job-cache').value;}const j=await localAPI('/api/observed/jobs',body);activeJob=j.id;watchJob(j.id);}catch(e){$('observed-job-status').textContent=e.message;jobOptions();}};
$('cancel-observed').onclick=async()=>{if(!activeJob)return;try{await localAPI('/api/observed/jobs/'+activeJob+'/cancel',{});$('observed-job-status').textContent='Cancellation requested; existing artifacts will be preserved.';}catch(e){$('observed-job-status').textContent=e.message;}};
$('saved-observed-job').onchange=()=>{const j=savedJobRows.find(j=>j.id===$('saved-observed-job').value);if(j&&!activeJob)showJob(j);};
$('open-observed-result').onclick=async()=>{if(!lastJob?.viewer_url)return;const dense=lastJob.result_kind==='observed_surface',id=lastJob.id+(dense?'-dense':'-sparse');if(!runs.some(r=>r.id===id)){runs.push({id,label:lastJob.label,status:lastJob.result_status,kind:dense?'observed_surface':'sparse_geometry',viewer_url:lastJob.viewer_url,original_viewer_url:lastJob.viewer_url,mask_provider:lastJob.mask_provider||'Saved source component masks'});$('run-select').replaceChildren(...runs.map((r,i)=>{const o=node('option',r.id+' · '+human(r.status||'saved'));o.value=i;return o;}));}if(lastJob.exports?.observed&&!exportsData.some(e=>e.source_run_id===id))exportsData.push(lastJob.exports.observed);const validation=lastJob.stages?.find(s=>s.stage==='validation')?.result?.report_url;if(validation&&!validationData.some(v=>v.source_run_id===id))validationData.push({source_run_id:id,report_url:validation});const completion=lastJob.stages?.find(s=>s.stage==='completion')?.result;if(completion?.viewer_url&&!completionOptions.some(c=>c.viewer_url===completion.viewer_url))completionOptions.push({id:lastJob.id+'-completion',source_run_id:id,label:'Optional workflow completion',viewer_url:completion.viewer_url});if(lastJob.exports?.inferred&&!completionExports.some(e=>e.source_run_id===id))completionExports.push(lastJob.exports.inferred);selectRecord(id);$('workbench').scrollIntoView({behavior:'smooth',block:'start'});};
for(const id of ['show-features','show-projections','show-mask'])$(id).addEventListener('change',drawFeatures);
for(const id of ['show-supported','show-candidates','show-cameras','show-surface','show-texture','show-inferred','show-wireframe','inferred-photo'])$(id).addEventListener('change',applyGeometryToggles);
$('view-source').onclick=()=>setViewMode(true);$('view-orbit').onclick=()=>setViewMode(false);$('photo-opacity').oninput=()=>setViewMode(sourceMode);$('preview-completion').onclick=()=>{$('show-inferred').checked=true;applyGeometryToggles();$('viewport').scrollIntoView({behavior:'smooth',block:'center'});};$('completion-select').onchange=()=>loadCompletion(completionOptions.filter(c=>c.source_run_id===run?.id)[Number($('completion-select').value)]);
$('frame-slider').addEventListener('input',e=>selectFrame(Number(e.target.value)));$('previous').addEventListener('click',()=>selectFrame(frameIndex-1));$('next').addEventListener('click',()=>selectFrame(frameIndex+1));$('run-select').addEventListener('change',e=>selectRun(Number(e.target.value)));$('fit-view').addEventListener('click',()=>fit());$('focus-center').addEventListener('click',()=>fit('center'));$('image-wrapper').addEventListener('click',selectPoint);
init3D();
try{
 summary=await json(SUMMARY);runs=Array.isArray(summary.runs)?summary.runs:[];$('api-status').textContent='Saved results · browsing uses no credits';await loadExtensions();
 renderRunChoices();
 if(!runs.length){$('run-select').append(node('option','No experiments saved yet'));throw new Error('No geometry results are available yet. The experiment may still be running. Reload once it finishes.');}
 const cameraLink=readCameraPermalink(new URL(location.href));
 if(cameraLink){
  const epoch=loadEpoch;let candidate;
  try{candidate=await resolveAutomaticCameraHypothesis(cameraLink);}catch(e){if(epoch===loadEpoch)throw e;}
  if(epoch===loadEpoch&&candidate){activeCameraBinding=candidate.sourceBinding;await inspectCameraHypothesis(candidate,{scroll:false});}
 }else{
  const requested=new URLSearchParams(location.search).get('run')||summary.default_run_id,selected=Math.max(0,runs.findIndex(r=>r.id===requested));$('run-select').value=selected;await selectRun(selected);
 }
}catch(e){error(e.message);$('run-reason').textContent='No saved result loaded.';if(!run)$('run-select').selectedIndex=-1;}
connectJobs();
