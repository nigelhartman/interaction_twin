/* Execute the actual viewer with real Three math; defer manifest/GLTF callbacks.
 * DOM, WebGL renderer and transport are test boundaries, not browser pixels. */
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),{createHash,webcrypto}=require('node:crypto');
const C=require('../showcase/aligned-scene-contract.js'),ROOT=path.resolve(__dirname,'..');
let T;test.before(async()=>{T=await import('../workspace/robot-vendor/three.module.js');});
const source='a'.repeat(64),manifestSHA='b'.repeat(64),native=Buffer.from('synthetic asynchronous GLTF callback fixture'),nativeSHA=createHash('sha256').update(native).digest('hex');
const transform=x=>[[1,0,0,x],[0,1,0,0],[0,0,1,3],[0,0,0,1]];
function fixture(){return{format:'sam3d-common-video-scene',version:1,original_program:{path:'data/source/procedure/interaction.json',sha256:'c'.repeat(64),bytes:100},source:{sha256:source,width:540,height:960,duration_s:3,time_domain:'source_video_presentation_seconds'},camera:{K:[[800,0,269.5],[0,810,479.5],[0,0,1]],width:540,height:960,world_from_camera:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},common_frame:{id:'current_source_camera_cv',coordinate_convention:'+Xright+Ydown+Zforward',metric_scale_verified:false,assembly_verified:false,camera_motion_removed:false},physical_accuracy_verified:false,accepted_motion_poses:0,timing:{max_hold_s:0,max_interpolation_gap_s:1,max_estimated_gap_s:1},components:['base','lid'].map((id,i)=>({id,label:id,asset:{path:`data/native/${id}.glb`,sha256:nativeSHA,bytes:native.length},poses:[{timestamp_s:0,status:'estimated',camera_from_native_mesh:transform(i)},{timestamp_s:1,status:'estimated',camera_from_native_mesh:transform(i+1)},{timestamp_s:2,status:'estimated',camera_from_native_mesh:transform(i+2)}]}))};}
function deferred(){let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});return{promise,resolve,reject};}
async function until(predicate,message){for(let i=0;i<200;i++){if(predicate())return;await new Promise(resolve=>setTimeout(resolve,1));}assert.fail(message);}
function viewer({legacyHandler=false,direct=false,t,nonrigidFind=async()=>null,nonrigidArtifacts={},binaryArtifacts={},modelColors=null,landing=false}={}){
 const manifest=deferred(),listeners=new Map(),messages=[],loaders=[],renders=[],renderers=[],orbit=[],fetches=[],overlays=[],models=[];let frame,now=0;
 const elements=new Map(['scene','status','clock','summary','joint-summary','joint-axis','scene-kind','surface-caption'].map(id=>[id,{id,hidden:id==='joint-axis',textContent:'',attrs:{},events:new Map(),addEventListener(name,callback){this.events.set(name,callback)},setAttribute(name,value){this.attrs[name]=value},getBoundingClientRect(){return{width:800,height:600}},insertAdjacentElement(where,element){assert.equal(where,'afterend');overlays.push(element);}}]));
 const localWindow={},parent=direct?localWindow:{postMessage(value,origin){assert.equal(origin,'http://127.0.0.1:8766');messages.push(structuredClone(value));}};
 class Renderer{constructor({canvas}){this.domElement=canvas;this.disposed=false;renderers.push(this)}setPixelRatio(){}setSize(){}setViewport(){}setScissorTest(){}setClearColor(){}clear(){}setScissor(){}render(scene,camera){renders.push({scene,camera})}dispose(){this.disposed=true}}
 class Controls{constructor(camera,canvas){this.camera=camera;this.canvas=canvas;this.target=new T.Vector3();this.enableHistory=[];orbit.push(this)}set enabled(value){this.enableHistory.push(value);this._enabled=value}get enabled(){return this._enabled}update(){}dispose(){this.disposed=true}}
 class Loader{parse(raw,base,resolve,reject){loaders.push({resolve,reject,raw:Buffer.from(raw)});}}
 const params=new URLSearchParams({source,path:'data/scene/scene.json',sha256:manifestSHA,bytes:'100'});if(t!==undefined)params.set('t',String(t));if(landing)params.set('presentation','landing');
 const sandbox={performance:{now:()=>now},AutoSceneCameraV1:require('../showcase/auto-scene-camera-v1.js'),SourceVisibleFlexure:require('../showcase/source-visible-flexure-v1.js'),SourceSurfacePresentationV1:require('../showcase/source-surface-presentation-v1.js'),console,URLSearchParams,Map,Set,Float32Array,Uint8Array,ArrayBuffer,TextDecoder,devicePixelRatio:1,crypto:webcrypto,THREE:{...T,WebGLRenderer:Renderer},GLTFLoader:Loader,OrbitControls:Controls,AlignedSceneContract:C,NativeByteRgbDisplayV1:require('../showcase/native-byte-rgb-display-v1.js'),SourceAppearanceStatesV1:require('../showcase/source-appearance-states-v1.js'),NonrigidObservationsV1:require('../showcase/nonrigid-observations-contract.js'),AlignedScenes:{checked:r=>r.path in nonrigidArtifacts?Promise.resolve(nonrigidArtifacts[r.path]):manifest.promise,find:nonrigidFind},document:{documentElement:{dataset:{}},getElementById:id=>{assert(elements.has(id),id);return elements.get(id)},createElement:kind=>{assert.equal(kind,'canvas');const calls=[],ctx={calls};for(const name of['setTransform','clearRect','save','beginPath','rect','clip','moveTo','lineTo','stroke','restore'])ctx[name]=(...args)=>calls.push([name,...args]);return{style:{},setAttribute(){},getContext:()=>ctx,context:ctx,remove(){this.removed=true}}}},location:{origin:'http://127.0.0.1:8766',search:'?'+params},parent,window:localWindow,requestAnimationFrame:callback=>{frame=callback;return 1},cancelAnimationFrame(){},ResizeObserver:class{observe(){}disconnect(){this.disconnected=true}},addEventListener(name,callback){listeners.set(name,callback)},fetch:async url=>{fetches.push(url);const key=url.replace(/^\/files\//,'');if(key in binaryArtifacts)return{ok:true,arrayBuffer:async()=>Uint8Array.from(await binaryArtifacts[key]).buffer};assert.match(url,/^\/files\/data\/native\/(base|lid|power_cable)\.glb$/);return{ok:true,arrayBuffer:async()=>Uint8Array.from(native).buffer}}};
 let code=fs.readFileSync(path.join(ROOT,'showcase/aligned-scene.js'),'utf8');assert.match(code,/import '\.\/native-byte-rgb-display-v1\.js'/);
 code=code.replace(/^import .*\n/gm,'');
 if(legacyHandler){const current="addEventListener('message',event=>{const d=C.message(event,parent,location.origin,source,binding.sha256);if(d)receiveState(d);});";assert(code.includes(current));code=code.replace(current,"addEventListener('message',event=>{const d=C.message(event,parent,location.origin,source,binding.sha256);if(!d||!manifest||d.time_s>manifest.source.duration_s)return;time=d.time_s;showEstimated=d.show_estimated;view=d.view;controls.enabled=view==='orbit';if(d.reset)fitted=false;apply();});");}
 const context=vm.createContext(sandbox);vm.runInContext(code,context,{filename:'showcase/aligned-scene.js'});
 const send=(fields={},event={})=>listeners.get('message')({origin:sandbox.location.origin,source:parent,data:{type:'interaction-twin:aligned-time',source_sha256:source,manifest_sha256:manifestSHA,time_s:1,view:'source',show_estimated:true,...fields},...event});
 const resolveModel=(index,{texture=false,colors=modelColors}={})=>{const scene=new T.Group();let geometry=new T.BoxGeometry(1,1,1);if(colors){geometry=new T.BufferGeometry();geometry.setAttribute('position',new T.Float32BufferAttribute([0,0,0,1,0,0,0,1,0],3));geometry.setAttribute('color',new T.Uint8BufferAttribute(colors,4,true));}if(loaders[index].raw.toString().startsWith('observed source surface '))geometry.translate(0,0,3);const map=texture?new T.Texture():null,material=new T.MeshBasicMaterial({color:0xffffff,map});geometry.disposalCount=0;geometry.addEventListener('dispose',()=>geometry.disposalCount++);material.disposalCount=0;material.addEventListener('dispose',()=>material.disposalCount++);if(map){map.disposalCount=0;map.addEventListener('dispose',()=>map.disposalCount++);}const mesh=new T.Mesh(geometry,material);scene.add(mesh);models[index]={scene,geometry,mesh,map,originalMaterial:material};loaders[index].resolve({scene});return models[index];};
 return{manifest,messages,loaders,renders,renderers,orbit,fetches,elements,overlays,models,send,resolveModel,frame:()=>frame?.(),advance:n=>{now=n;frame?.()},interact:(name,fields={})=>elements.get('scene').events.get(name)?.({type:name,...fields}),hide:()=>listeners.get('pagehide')(),state:()=>messages.filter(m=>m.type==='interaction-twin:aligned-state').at(-1),ready:()=>messages.some(m=>m.type==='interaction-twin:aligned-ready'),errors:()=>messages.filter(m=>m.type==='interaction-twin:aligned-error')};
}

test('pre-manifest and in-flight model messages preserve the latest valid state without touching absent controls',async()=>{
 const v=viewer();assert.doesNotThrow(()=>v.send({time_s:.2,view:'orbit',show_estimated:true,reset:true}));v.manifest.resolve(fixture());await until(()=>v.loaders.length===2,'Both native parsers must be pending');
 assert.doesNotThrow(()=>v.send({time_s:1,view:'source',show_estimated:true}));assert.doesNotThrow(()=>v.send({time_s:2,view:'orbit',show_estimated:false}));
 assert.equal(v.orbit.length,0);assert.equal(v.renders.length,0);assert.equal(v.messages.length,0,'No state or ready handshake before assets exist');v.resolveModel(0);await new Promise(resolve=>setImmediate(resolve));assert.equal(v.renderers.length,0,'One ready model cannot expose a partial scene');
 v.resolveModel(1);await until(v.ready,'Ready handshake was not emitted');assert.equal(v.state().time_s,2);assert.equal(v.state().view,'orbit');assert.equal(v.state().show_estimated,false);assert.equal(v.state().visible,0);assert.equal(v.elements.get('clock').textContent,'2.000 s');assert.equal(v.orbit[0].enabled,true);assert(v.orbit[0].enableHistory.every(Boolean),'Startup must not overwrite queued orbit with enabled=false');assert.equal(v.errors().length,0);
 v.send({time_s:1,view:'source',show_estimated:true});assert.equal(v.state().visible,2);assert.equal(v.orbit[0].enabled,false);v.orbit[0].target.set(999,999,999);v.send({time_s:1,view:'orbit',show_estimated:true});assert.equal(v.orbit[0].target.x,999,'Ordinary updates preserve inspection orbit');v.send({time_s:1,view:'orbit',show_estimated:true,reset:true});assert.notEqual(v.orbit[0].target.x,999,'Explicit reset must refit after readiness');v.hide();
});

test('deferred duration validation rejects out-of-range state and authenticated-message boundaries remain intact',async()=>{
 const v=viewer();v.send({time_s:1,view:'orbit',reset:true});v.send({time_s:999,view:'source'});v.send({time_s:2,view:'source'},{origin:'https://untrusted.invalid'});v.manifest.resolve(fixture());await until(()=>v.loaders.length===2,'Native parsers missing');
 v.send({time_s:NaN,view:'source'});v.send({time_s:2,source_sha256:'f'.repeat(64)});v.send({time_s:2,manifest_sha256:'f'.repeat(64)});v.send({time_s:2},{source:{}});v.send({time_s:9});v.resolveModel(0);v.resolveModel(1);await until(v.ready,'No ready');assert.equal(v.state().time_s,1);assert.equal(v.state().view,'orbit');assert.equal(v.errors().length,0);v.hide();
});

test('the previous unguarded handler reproduces the observed undefined-controls failure',async()=>{
 const v=viewer({legacyHandler:true});v.manifest.resolve(fixture());await until(()=>v.loaders.length===2,'Native parsers missing');assert.throws(()=>v.send({time_s:1,view:'orbit'}),/Cannot set properties of undefined.*enabled/);v.hide();v.resolveModel(0);v.resolveModel(1);
});

test('direct inspection keeps its explicit initial source time when there is no parent state',async()=>{
 const v=viewer({direct:true,t:1});v.manifest.resolve(fixture());await until(()=>v.loaders.length===2,'Native parsers missing');v.resolveModel(0);v.resolveModel(1);await until(()=>v.renders.length===1,'Direct scene did not render');assert.equal(v.elements.get('clock').textContent,'1.000 s');assert.equal(v.orbit[0].enabled,false);assert.equal(v.messages.length,0);v.hide();
});

test('pagehide before manifest or model completion prevents late viewer construction and announcements',async()=>{
 const a=viewer();a.send({view:'orbit',reset:true});a.hide();a.manifest.resolve(fixture());await new Promise(resolve=>setImmediate(resolve));assert.equal(a.fetches.length,0);assert.equal(a.renderers.length,0);assert.equal(a.messages.length,0);
 const b=viewer();b.manifest.resolve(fixture());await until(()=>b.loaders.length===2,'Native parsers missing');b.hide();assert.doesNotThrow(()=>b.send({view:'orbit',reset:true}));b.resolveModel(0);b.resolveModel(1);await new Promise(resolve=>setImmediate(resolve));assert.equal(b.renderers.length,0);assert.equal(b.messages.length,0);
});

test('pagehide during initial native parsing also releases every late geometry and texture',async()=>{
 const v=viewer();v.manifest.resolve(fixture());await until(()=>v.loaders.length===2,'Native parsers missing');const first=v.resolveModel(0,{texture:true});await new Promise(r=>setImmediate(r));v.hide();const second=v.resolveModel(1,{texture:true});await new Promise(r=>setImmediate(r));assert.equal(v.renderers.length,0);assert.equal(v.messages.length,0);for(const m of [first,second]){assert.equal(m.geometry.disposalCount,1,'Unrendered native geometry must not leak after pagehide');assert.equal(m.map.disposalCount,1,'Unrendered native texture must not leak after pagehide');}
});

test('failed native parsing preserves the error and never lets pending messages fabricate readiness',async()=>{
 const v=viewer();v.manifest.resolve(fixture());await until(()=>v.loaders.length===2,'Native parsers missing');v.send({time_s:2,view:'orbit'});v.loaders[0].reject(Error('Synthetic parse failure'));await until(()=>v.errors().length===1,'Native error missing');assert.equal(v.ready(),false);assert.equal(v.orbit.length,0);assert.equal(v.renders.length,0);assert.match(v.elements.get('status').textContent,/No substitute scene/);v.send({time_s:1,view:'source'});assert.equal(v.errors().length,1);v.hide();v.resolveModel(1);
});

test('a failed native parser aborts startup and releases another parser that succeeds late',async()=>{
 const v=viewer();v.manifest.resolve(fixture());await until(()=>v.loaders.length===2,'Native parsers missing');v.loaders[0].reject(Error('First native parser failed'));await until(()=>v.errors().length===1,'Failed parser error missing');const late=v.resolveModel(1,{texture:true});await new Promise(r=>setImmediate(r));assert.equal(v.ready(),false);assert.equal(v.renderers.length,0);assert.equal(late.geometry.disposalCount,1,'Another late successful native parser must not leak after startup abort');assert.equal(late.map.disposalCount,1);assert.equal(v.errors().length,1);v.hide();assert.equal(late.geometry.disposalCount,1,'A subsequent pagehide must not double-dispose released parser resources');
});

test('nonrigid observations from another exact scene binding fail before native parsing',async()=>{
 const v=viewer({nonrigidFind:async()=>({reference:{path:'data/foreign/scene.json',sha256:'d'.repeat(64),bytes:100},derived_assembly:{document:{nonrigid_observations_ref:{path:'data/nonrigid/observations.json',sha256:'e'.repeat(64),bytes:100}}}})});
 v.manifest.resolve(fixture());await until(()=>v.errors().length===1,'Nonrigid scene-binding error missing');assert.match(v.errors()[0].message,/another exact scene binding/);assert.equal(v.fetches.length,0);assert.equal(v.ready(),false);v.hide();
});

test('the actual viewer draws saved source-pixel cable paths and suppresses only their rigid replay in both cameras',async()=>{
 const m=fixture();m.source.path='data/video.mp4';m.components[1].id='power_cable';m.components[1].asset.path='data/native/power_cable.glb';
 const ref=(path,sha='d'.repeat(64))=>({path,sha256:sha,bytes:100});
 const evidence=[0,1,2].map(i=>({source_index:i,timestamp_s:i,id:'frame-'+i,frame_path:`data/frames/${i}.jpg`,frame_sha256:'e'.repeat(64),observations:[{track_id:'power_cable',mask_path:`data/masks/${i}.png`,mask_sha256:'f'.repeat(64)}]}));
 const{width,height,...programSource}=m.source,p={source:programSource,tracklets:[{id:'power_cable',limitations:['Deformable cable; no rigid pose.']}],evidence};
 const rows=evidence.map(e=>({source_index:e.source_index,timestamp_s:e.timestamp_s,evidence_id:e.id,frame_ref:ref(e.frame_path,e.frame_sha256),mask_ref:ref(e.observations[0].mask_path,e.observations[0].mask_sha256),frame_size_px:[540,960],between_samples:'absent_no_temporal_interpolation_or_endpoint_identity_assignment',status:e.source_index===1?'unknown':'observed_2d_path_candidate',trace:e.source_index===1?null:{status:'observed_2d_path_candidate',coordinate_system:'original_binary_mask_pixels_top_left_xy',endpoint_identity:'unassigned_lexicographic_order_not_connector_identity',mask_size_px:[540,960],points_px:[[10,20],[30,40]]}}));
 const traceRef=ref('data/cable/trace.json'),docRef=ref('data/cable/observations.json'),trace={format:'interaction-twin-cable-trace-2d',version:1,track_id:'power_cable',program_ref:m.original_program,source_sha256:source,source_ref:{path:p.source.path,sha256:source},selection:{program_evidence_count:3},samples:rows};
 const d={format:'interaction-twin-nonrigid-observations',version:1,depth:'undefined',endpoint_physical_identity:'unknown',invisible_spans_completed:false,physical_accuracy_verified:false,simulation_ready:false,source_time_domain:'source_video_presentation_seconds',evidence_program:m.original_program,trace_ref:traceRef,source:p.source,component_id:'power_cable',original_trace_sha256:traceRef.sha256,fresh_skeleton_recomputation_verified:true,sample_count:3,display_policy:{rigid_native_pose_replay:'withheld_for_known_nonrigid_component',native_static_asset_inspection_retained:true,viewer_representation:'source_pixel_polyline_overlay_video_camera_only',temporal_interpolation:'none',maximum_nearest_sample_offset_s:.36,unknown_rows_are_barriers:true,display_between_samples:'nearest_saved_annotation_estimate_not_motion_measurement'}};
 const v=viewer({nonrigidFind:async()=>({reference:{path:'data/scene/scene.json',sha256:manifestSHA,bytes:100},derived_assembly:{document:{nonrigid_observations_ref:docRef}}}),nonrigidArtifacts:{[docRef.path]:d,[traceRef.path]:trace,[m.original_program.path]:p}});
 v.send({time_s:.1});v.manifest.resolve(m);await until(()=>v.loaders.length===2,'Native static assets must remain loaded');v.resolveModel(0);v.resolveModel(1);await until(v.ready,'Cable viewer not ready');
 assert.equal(v.state().visible,1);assert.equal(v.state().parts.find(p=>p.id==='power_cable').status,'nonrigid_rigid_replay_withheld');assert.equal(v.state().nonrigid.visible,true);assert.equal(v.state().nonrigid.source_index,0);assert.equal(v.state().nonrigid.interpolated,false);assert.match(v.elements.get('summary').textContent,/Observed cable · 2D/);assert.equal(v.overlays.length,1);
 const overlay=v.overlays[0],points=overlay.context.calls.filter(c=>['moveTo','lineTo'].includes(c[0]));assert.deepEqual(points,[['moveTo',237.5,12.5],['lineTo',250,25]],'Only exact saved source pixels pass through source letterboxing');
 v.send({time_s:.5});v.frame();assert.equal(v.state().nonrigid.visible,false,'Nearest unknown wins midpoint tie');assert.equal(overlay.context.calls.at(-1)[0],'clearRect');
 v.send({time_s:2,view:'orbit'});v.frame();assert.equal(v.state().nonrigid.visible,false);assert.equal(v.state().parts.find(p=>p.id==='power_cable').visible,false,'Orbit must never substitute a rigid cable coil');
 v.send({time_s:2,view:'source'});v.frame();assert.equal(v.state().nonrigid.visible,true);v.hide();assert.equal(overlay.removed,true);
});

function appearanceFixture(){
 const m=fixture(),A=require('../showcase/source-appearance-states-v1.js');
 const ref=(name,bytes=100,sha='d'.repeat(64))=>({path:'data/appearance/'+name,sha256:sha,bytes});
 const rgb=[0,1,2].map(i=>ref(`rgb-${i}.jpg`,100,'e'.repeat(64))),masks=[0,1,2].map(i=>ref(`mask-${i}.png`,100,'f'.repeat(64)));
 m.components.forEach(c=>c.poses.forEach((p,i)=>Object.assign(p,{source_sample_index:i,source_mask:masks[i]})));
 const program={source:m.source,evidence:[0,1,2].map(i=>({source_index:i,timestamp_s:i,frame_path:rgb[i].path,frame_sha256:rgb[i].sha256}))};
 const sceneRef={path:'data/scene/scene.json',sha256:manifestSHA,bytes:100},previousRef=ref('previous.json'),programRef=m.original_program,sidecarRef=ref('states.json');
 const previous={source:m.source,scene_ref:sceneRef,evidence_program:programRef,component_assets:m.components.map(c=>({component_id:c.id,native_glb:c.asset}))};
 function patch(color){const b=Buffer.alloc(27);b.write('ITCP');b.writeUInt32LE(1,4);b.writeUInt32LE(3,8);b.writeUInt32LE(1,12);b.writeUInt32LE(1,16);b.writeUInt16LE(2,20);b.writeUInt16LE(3,22);b.set(color,24);return b;}
 const rawDark=patch([4,5,6]),rawBoot=patch([210,220,230]);
 const dark=ref('dark.itcp',rawDark.length,createHash('sha256').update(rawDark).digest('hex')),boot=ref('boot.itcp',rawBoot.length,createHash('sha256').update(rawBoot).digest('hex'));
 const keys=[['dark',0,'source_observed_dark',dark],['boot',2,'source_observed_logo',boot]].map(([id,i,label,patchRef])=>({id,source_sample_index:i,timestamp_s:i,appearance_label:label,source_rgb_ref:rgb[i],source_mask_ref:masks[i],source_camera:m.camera,camera_from_native_mesh:m.components[1].poses[i].camera_from_native_mesh,patch_ref:patchRef,painted_vertex_count:1,native_fallback_vertex_count:2}));
 const d={format:A.FORMAT,version:1,display_policy:A.POLICY,physical_accuracy_verified:false,simulation_ready:false,electronic_state_verified:false,previous_derived_assembly_ref:previousRef,evidence_program:programRef,scene_ref:sceneRef,source:m.source,baseline_asset_ref:m.components[1].asset,component_id:'lid',vertex_count:3,keyframes:keys,
  observations:[0,1,2].map(i=>({source_sample_index:i,timestamp_s:i,status:i===1?'unknown':'source_observed_appearance_estimate',keyframe_id:i===1?null:i===0?'dark':'boot',appearance_label:i===1?'unknown':i===0?'source_observed_dark':'source_observed_logo',source_rgb_ref:rgb[i],source_mask_ref:masks[i]}))};
 const found={reference:sceneRef,derived_assembly:{document:{source_appearance_states_ref:sidecarRef}}};
 return {scene:m,document:d,found,sidecarRef,dark,boot,rawDark,rawBoot,
  artifacts:{[sidecarRef.path]:d,[previousRef.path]:previous,[programRef.path]:program},
  binary:{[dark.path]:rawDark,[boot.path]:rawBoot},baseline:[10,20,30,255,40,50,60,128,70,80,90,64]};
}

test('actual viewer applies exact dark→unknown baseline→boot patches with alpha and provenance preserved',async()=>{
 const f=appearanceFixture(),v=viewer({nonrigidFind:async()=>f.found,nonrigidArtifacts:f.artifacts,binaryArtifacts:f.binary,modelColors:f.baseline});
 v.send({time_s:0});v.manifest.resolve(f.scene);await until(()=>v.loaders.length===2,'Appearance patches must validate before native parsers');
 assert.deepEqual(v.fetches.slice(0,2),['/files/'+f.dark.path,'/files/'+f.boot.path]);assert.equal(v.ready(),false);
 v.resolveModel(0);v.resolveModel(1);await until(v.ready,'Source appearance viewer never became ready');
 // Crypto/parse completions may finish out of order under full-suite load.
 // Published scene groups retain the manifest's base/lid order via Promise.all.
 const groups=v.renders[0].scene.children;
 const base=groups[0].children[0].children[0].geometry.attributes.color.array,lid=groups[1].children[0].children[0].geometry.attributes.color.array;
 assert.deepEqual([...base],f.baseline,'A lid patch must never repaint its sibling');
 assert.deepEqual([...lid],[10,20,30,255,4,5,6,128,70,80,90,64]);
 assert.deepEqual(v.state().source_appearance,{component_id:'lid',keyframe_id:'dark',appearance_label:'source_observed_dark',status:'source_observed_appearance_estimate',electronics_simulated:false});
 assert.equal(v.state().source_sha256,source);assert.equal(v.state().manifest_sha256,manifestSHA);
 v.send({time_s:1});assert.deepEqual([...lid],f.baseline,'Unknown observation must restore the exact native baseline');
 assert.equal(v.state().source_appearance.keyframe_id,null);assert.equal(v.state().source_appearance.status,'unknown_native_fallback');assert.match(v.elements.get('summary').textContent,/Screen appearance unknown · native fallback/);
 v.send({time_s:2,view:'orbit'});assert.deepEqual([...lid],[10,20,30,255,210,220,230,128,70,80,90,64]);
 assert.equal(v.state().source_appearance.keyframe_id,'boot');assert.equal(v.state().source_appearance.electronics_simulated,false);assert.equal(v.state().view,'orbit');
 v.send({time_s:.2,view:'source'});assert.deepEqual([...lid],f.baseline,'A gap touching an unknown row cannot smear dark or boot colours');
 v.send({time_s:0});assert.deepEqual([...lid],[10,20,30,255,4,5,6,128,70,80,90,64],'Returning to a saved keyframe restores its exact source RGB');
 v.hide();assert.equal(v.renderers[0].disposed,true);assert.equal(v.errors().length,0);
});

test('foreign appearance bindings and corrupted or mismatched patches fail before any native parsing',async()=>{
 const cases=[
  {name:'foreign source binding',mutate:f=>{f.found.reference={...f.found.reference,sha256:'9'.repeat(64)};},error:/another exact scene binding/},
  {name:'corrupt patch hash',mutate:f=>{f.binary[f.dark.path]=Buffer.from(f.rawDark);f.binary[f.dark.path][26]^=1;},error:/patch hash changed/},
  {name:'corrupt patch length',mutate:f=>{f.binary[f.dark.path]=f.rawDark.subarray(1);},error:/patch length changed/},
  {name:'wrong native vertex inventory',mutate:f=>{const b=Buffer.from(f.rawDark);b.writeUInt32LE(4,8);f.binary[f.dark.path]=b;f.document.keyframes[0].patch_ref={...f.dark,sha256:createHash('sha256').update(b).digest('hex')};},error:/native vertex count differs/},
  {name:'foreign keyframe RGB',mutate:f=>{f.document.keyframes[0].source_rgb_ref={...f.document.keyframes[0].source_rgb_ref,sha256:'9'.repeat(64)};},error:/projection differs/}
 ];
 for(const row of cases){const f=appearanceFixture();row.mutate(f);const v=viewer({nonrigidFind:async()=>f.found,nonrigidArtifacts:f.artifacts,binaryArtifacts:f.binary,modelColors:f.baseline});v.manifest.resolve(f.scene);await until(()=>v.errors().length===1,row.name+' error missing');assert.match(v.errors()[0].message,row.error,row.name);assert.equal(v.loaders.length,0,row.name+' reached native parser');assert.equal(v.renderers.length,0);assert.equal(v.ready(),false);v.hide();}
});

test('delayed appearance bytes preserve latest queued time and pagehide prevents late readiness',async()=>{
 const f=appearanceFixture(),pending=deferred();f.binary[f.dark.path]=pending.promise;
 const v=viewer({nonrigidFind:async()=>f.found,nonrigidArtifacts:f.artifacts,binaryArtifacts:f.binary,modelColors:f.baseline});
 v.send({time_s:0});v.manifest.resolve(f.scene);await until(()=>v.fetches.includes('/files/'+f.dark.path),'Deferred patch was not requested');
 v.send({time_s:2,view:'orbit'});assert.equal(v.loaders.length,0);assert.equal(v.ready(),false);assert.equal(v.messages.length,0);
 pending.resolve(f.rawDark);await until(()=>v.loaders.length===2,'Native parsers did not start after verified patch bytes');v.resolveModel(0);v.resolveModel(1);await until(v.ready,'No delayed appearance readiness');assert.equal(v.state().time_s,2);assert.equal(v.state().source_appearance.keyframe_id,'boot');v.hide();
 const hidden=appearanceFixture(),delayed=deferred();hidden.binary[hidden.dark.path]=delayed.promise;
 const h=viewer({nonrigidFind:async()=>hidden.found,nonrigidArtifacts:hidden.artifacts,binaryArtifacts:hidden.binary,modelColors:hidden.baseline});h.manifest.resolve(hidden.scene);await until(()=>h.fetches.includes('/files/'+hidden.dark.path),'Second deferred patch missing');h.hide();delayed.resolve(hidden.rawDark);await until(()=>h.fetches.includes('/files/'+hidden.boot.path),'Pending patch loader did not settle');await new Promise(resolve=>setImmediate(resolve));assert.equal(h.loaders.length,0);assert.equal(h.renderers.length,0);assert.equal(h.messages.length,0);assert.equal(h.errors().length,0);
});

test('landing drag uses a centered inspection camera, then returns exactly without changing source time or poses',async()=>{
 const v=viewer({landing:true});v.send({time_s:1});v.manifest.resolve(fixture());await until(()=>v.loaders.length===2,'Native parsers missing');v.resolveModel(0);v.resolveModel(1);await until(v.ready,'Landing viewer missing');
 const sourceCamera=v.renders.at(-1).camera,projection=sourceCamera.projectionMatrix.clone(),state=JSON.stringify(v.state());
 v.interact('pointermove',{buttons:0});v.advance(100);assert.equal(v.renders.at(-1).camera,sourceCamera,'Hover must not leave video camera');
 v.interact('pointerdown');const orbit=v.orbit[0];v.advance(101);assert.equal(v.renders.at(-1).camera,orbit.camera);assert.equal(orbit.enabled,true);
 assert.notEqual(orbit.target.x,0,'Orbit targets the actual part bounds, not the empty optical axis');assert(orbit.target.z<0);
 assert(orbit.camera.projectionMatrix.elements.every(Number.isFinite),'Inspection projection must be finite');const projected=orbit.target.clone().project(orbit.camera);assert(Math.abs(projected.x)<1e-10&&Math.abs(projected.y)<1e-10,'Inspection frames the actual parts at center');
 const start=new T.Vector3(.2,.1,.2);orbit.camera.position.copy(start);orbit.camera.lookAt(orbit.target);const q=orbit.camera.quaternion.clone();
 v.advance(3099);assert.equal(v.renders.at(-1).camera,orbit.camera);assert.equal(orbit.camera.position.x,.2);
 v.advance(3525);assert(orbit.camera.position.x>0&&orbit.camera.position.x<.2);assert(orbit.camera.quaternion.angleTo(q)>0);
 assert.equal(JSON.stringify(v.state()),state,'Inspection does not mutate poses, support, time or estimate flags');
 v.advance(3950);assert.equal(v.renders.at(-1).camera,sourceCamera);assert.deepEqual(sourceCamera.position.toArray(),[0,0,0]);assert.deepEqual(sourceCamera.quaternion.toArray(),[0,0,0,1]);
 v.hide();
});

function surfaceFixture({count=3,unknown=[1]}={}){
 const m=fixture();m.source.duration_s=Math.max(3,count);const ref=(path,sha='d'.repeat(64),bytes=100)=>({path,sha256:sha,bytes}),binary={};
 const samples=Array.from({length:count},(_,i)=>{const row={source_sample_index:i,timestamp_s:i,status:unknown.includes(i)?'unknown':'source_observed_prior'};if(row.status==='unknown')return row;const raw=Buffer.from('observed source surface '+i),asset=ref('data/surfaces/surface-'+i+'.glb',createHash('sha256').update(raw).digest('hex'),raw.length);binary[asset.path]=raw;return{...row,asset,source_RGB:ref('data/frames/'+i+'.jpg'),source_pixel_map:ref('data/surfaces/map-'+i+'.npz'),provenance:ref('data/surfaces/proof-'+i+'.json'),source_masks:[ref('data/masks/'+i+'.png')],physical_accuracy_verified:false,source_visible_only:true,triangle_source_footprint_verified:true,vertices:3,hidden_backs_culled:true,unlit_exact_vertex_RGB:true,virtual_flexure:{status:'unknown',angle_rad:null,origin_camera:null,axis_camera:null},camera_from_asset:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],visible_component_ids:i===0?['base']:['base','lid']};});
 const descriptor={format:'interaction-twin-source-visible-flexure-series',version:1,kind:'source_observed_monocular_depth_prior',physical_accuracy_verified:false,simulation_ready:false,native_SAM_contract_compatible:false,manual_geometry_or_poses_used:false,source_program:m.original_program,source_scene:ref('data/scene/scene.json',manifestSHA),protocol:ref('data/surfaces/protocol.json'),source:m.source,camera:m.camera,replacement_component_ids:['base','lid'],samples,policy:{maximum_source_time_error_s:.18,timing:'nearest_source_keyframe_within_0.18s_only_no_geometric_interpolation',hidden_surfaces:'absent_not_inpainted',numerical_fold_parameters:'unknown_not_identifiably_recovered'}};
 const found={reference:descriptor.source_scene,observed_surfaces:{artifact:ref('data/surfaces/descriptor.json'),document:descriptor}};return {scene:m,descriptor,found,binary};
}
async function startSurface(f,options={}){const v=viewer({nonrigidFind:async()=>f.found,binaryArtifacts:f.binary,...options});v.send({time_s:0});v.manifest.resolve(f.scene);await until(()=>v.loaders.length===2,'Native inspection assets did not load');v.resolveModel(0);v.resolveModel(1);await until(v.ready,'Surface viewer not ready');await until(()=>v.loaders.length===3,'Initial observed surface parser missing');return v;}

test('deferred observed surfaces obey latest seek and unknown barriers without restoring native replacements',async()=>{
 const f=surfaceFixture(),original=JSON.stringify(f.scene),v=await startSurface(f);assert.equal(v.state().visible,0);assert(v.models.slice(0,2).every(m=>m.scene.parent.visible===false),'Native holder/lid must stay hidden while surface loads');
 v.send({time_s:2});await until(()=>v.loaders.length===4,'Second source surface not requested');const stale=v.resolveModel(2,{texture:true});await new Promise(r=>setImmediate(r));assert.equal(stale.scene.visible,false,'Late previous frame may be cached but not shown');assert.equal(v.state().time_s,2);assert.equal(v.state().visible,0);
 const current=v.resolveModel(3,{texture:true});await until(()=>v.state().visible===2,'Latest exact source surface never became visible');assert.equal(current.scene.visible,true);assert.equal(v.state().observed_surfaces.source_sample_index,2);assert.equal(v.state().observed_surfaces.interpolated,false);assert.equal(v.elements.get('clock').textContent,'2.000 s');
 v.send({time_s:1});assert.equal(current.scene.visible,false);assert.equal(v.state().visible,0);assert.equal(v.state().observed_surfaces.visible,false);assert(v.state().parts.every(p=>p.status==='unknown'&&!p.visible));assert.match(v.elements.get('surface-caption').textContent,/unknown/);assert(v.models.slice(0,2).every(m=>m.scene.parent.visible===false),'Unknown surface must not fall back to the wrong closed native model');
 v.send({time_s:2,show_estimated:false});assert.equal(v.state().visible,0);assert.equal(v.state().observed_surfaces.visible,false);assert.equal(JSON.stringify(f.scene),original,'Surface loading/selection must not rewrite source poses or timing');v.hide();
});

test('exact-JPEG surfaces use unlit front faces without multiplying vertex colors and dispose retained texture resources',async()=>{
 const f=surfaceFixture(),v=await startSurface(f),model=v.resolveModel(2,{texture:true,colors:[20,40,60,255,80,100,120,255,140,160,180,255]});await until(()=>v.state().visible===1,'Surface not displayed');
 assert.equal(model.mesh.material.map,model.map);assert.equal(model.mesh.material.vertexColors,false,'Exact source texture must not multiply sampled vertex RGB');assert.equal(model.mesh.material.side,T.FrontSide);assert.equal(model.mesh.material.toneMapped,false);assert.equal(model.originalMaterial.disposalCount,1);assert.equal(model.map.disposalCount,0);assert.equal(model.geometry.attributes.color.itemSize,4,'Provenance/unused color attribute must not be repainted when a texture exists');
 assert.equal(v.state().parts.find(p=>p.id==='lid').visible,false,'Source-mask reference alone must not claim a missing component visible');v.hide();assert.equal(model.geometry.disposalCount,1);assert.equal(model.map.disposalCount,1);assert.equal(v.renderers[0].disposed,true);
});

test('pagehide releases late observed GLTFs and ignores late load errors without DOM/state mutation',async()=>{
 const f=surfaceFixture(),v=await startSurface(f),before={caption:v.elements.get('surface-caption').textContent,messages:v.messages.length};v.hide();const late=v.resolveModel(2,{texture:true});await new Promise(r=>setImmediate(r));assert.equal(late.geometry.disposalCount,1);assert.equal(late.map.disposalCount,1);assert.equal(late.scene.parent,null);assert.equal(v.messages.length,before.messages);assert.equal(v.elements.get('surface-caption').textContent,before.caption);
 const g=surfaceFixture(),w=await startSurface(g);const caption=w.elements.get('surface-caption').textContent,count=w.messages.length;w.hide();w.loaders[2].reject(Error('Late source surface failure'));await new Promise(r=>setImmediate(r));assert.equal(w.elements.get('surface-caption').textContent,caption);assert.equal(w.messages.length,count);assert.equal(w.errors().length,0);
});

test('failed observed surface remains explicitly unavailable after a seek and never becomes endless loading or native fallback',async()=>{
 const f=surfaceFixture(),v=await startSurface(f);v.loaders[2].reject(Error('Synthetic observed texture failure'));await until(()=>v.elements.get('surface-caption').textContent==='Surface unavailable','Source surface failure missing');v.send({time_s:1});v.send({time_s:0});assert.equal(v.elements.get('surface-caption').textContent,'Surface unavailable');assert.match(v.elements.get('surface-caption').title,/Synthetic observed texture failure/);assert.equal(v.state().visible,0);assert(v.models.slice(0,2).every(m=>m.scene.parent.visible===false));assert.equal(v.loaders.length,3,'A failed display must not silently redispatch work');v.hide();
});

test('fast surface seeks bound inflight parsing to two and coalesce deferred requests to the latest source only',async()=>{
 const f=surfaceFixture({count:18,unknown:[]}),v=await startSurface(f);for(let i=1;i<18;i++){v.send({time_s:i});await new Promise(r=>setImmediate(r));}await new Promise(r=>setTimeout(r,5));assert.equal(v.loaders.length,4,'Only two observed GLTF parses may be in flight');assert.equal(v.state().time_s,17);assert.equal(v.state().visible,0);
 v.resolveModel(2,{texture:true});await until(()=>v.loaders.length===5,'Latest coalesced source was not dispatched');assert.equal(v.loaders[4].raw.toString(),'observed source surface 17');assert.equal(v.fetches.filter(p=>p.startsWith('/files/data/surfaces/')).length,3,'Intermediate source seeks must not fan out into fetch/parse work');v.resolveModel(3,{texture:true});v.resolveModel(4,{texture:true});await until(()=>v.state().observed_surfaces.source_sample_index===17&&v.state().visible===2,'Final source surface missing');v.hide();
});

test('unknown seek clears a coalesced surface request before earlier parsers finish',async()=>{
 const f=surfaceFixture({count:6,unknown:[4]}),v=await startSurface(f);v.send({time_s:1});await until(()=>v.loaders.length===4,'Second inflight source missing');v.send({time_s:3});v.send({time_s:4});v.resolveModel(2,{texture:true});v.resolveModel(3,{texture:true});await new Promise(r=>setTimeout(r,5));assert.equal(v.loaders.length,4,'Unknown barrier must cancel the deferred desired frame');assert.equal(v.state().time_s,4);assert.equal(v.state().visible,0);assert.equal(v.state().observed_surfaces.visible,false);v.hide();
});

test('observed landing inspection stays in the front cone while source clock, projection and poses remain untouched',async()=>{
 const f=surfaceFixture(),original=JSON.stringify(f.scene),v=await startSurface(f,{landing:true});v.resolveModel(2,{texture:true});await until(()=>v.state().visible===1,'Observed landing surface missing');v.frame();const sourceCamera=v.renders.at(-1).camera,projection=sourceCamera.projectionMatrix.toArray(),state=JSON.stringify(v.state()),orbit=v.orbit[0];
 assert.equal(orbit.minAzimuthAngle,-Math.PI/3);assert.equal(orbit.maxAzimuthAngle,Math.PI/3);assert.equal(orbit.minPolarAngle,Math.PI*.22);assert.equal(orbit.maxPolarAngle,Math.PI*.78);v.interact('pointerdown');v.advance(1);assert.equal(v.renders.at(-1).camera,orbit.camera);assert.equal(orbit.target.z<0,true);const direction=orbit.camera.position.clone().sub(orbit.target);assert(Math.abs(Math.atan2(direction.x,direction.z))<=Math.PI/3);assert.equal(JSON.stringify(v.state()),state);assert.equal(JSON.stringify(f.scene),original);assert.deepEqual(sourceCamera.projectionMatrix.toArray(),projection);v.advance(4000);assert.equal(v.renders.at(-1).camera,sourceCamera);assert.equal(v.elements.get('clock').textContent,'0.000 s');v.hide();
});

test('editorial source repeats report the actual observation clock without changing the requested video time or strict query',async()=>{
 const f=surfaceFixture(),P=require('../showcase/source-surface-presentation-v1.js'),F=require('../showcase/source-visible-flexure-v1.js');f.found.observed_surfaces.presentation={artifact:{path:'data/surfaces/presentation.json',sha256:'e'.repeat(64),bytes:100},document:P.derive(f.descriptor,f.found.observed_surfaces.artifact)};const v=await startSurface(f);v.resolveModel(2,{texture:true});await until(()=>v.state().visible===1,'Repeated source surface missing');assert.equal(F.at(f.descriptor,.35,true),null,'Strict observation query must remain .18s');v.send({time_s:.35});assert.equal(v.state().time_s,.35);assert.equal(v.state().observed_surfaces.source_sample_index,0);assert.equal(v.state().observed_surfaces.observation_timestamp_s,0);assert.equal(v.state().observed_surfaces.source_time_error_s,.35);assert.equal(v.state().observed_surfaces.presentation_mode,'editorial_source_sample_repeat');assert.equal(v.state().observed_surfaces.current_geometry,false);assert.equal(v.state().observed_surfaces.interpolated,false);assert.equal(v.elements.get('clock').textContent,'0.350 s');assert.match(v.elements.get('surface-caption').textContent,/0\.000s/);v.send({time_s:1});assert.equal(v.state().observed_surfaces.visible,false);assert.equal(v.state().visible,0);v.hide();
});

test('foreign observed-surface program, scene, camera and hidden pose claims fail before native or surface parsing',async()=>{
 const cases=[
  {name:'foreign exact scene',change:f=>f.found.reference.sha256='9'.repeat(64)},
  {name:'foreign original program',change:f=>f.descriptor.source_program={...f.descriptor.source_program,sha256:'9'.repeat(64)}},
  {name:'different source camera',change:f=>f.descriptor.camera={...f.descriptor.camera,K:[[900,0,269.5],[0,810,479.5],[0,0,1]]}},
  {name:'unknown numerical joint',change:f=>f.descriptor.samples[0].virtual_flexure.angle_rad=.5},
  {name:'hidden source pose',change:f=>f.descriptor.samples[0].camera_from_asset[0][3]=1}
 ];
 for(const row of cases){const f=surfaceFixture();row.change(f);const v=viewer({nonrigidFind:async()=>f.found,binaryArtifacts:f.binary});v.manifest.resolve(f.scene);await until(()=>v.errors().length===1,row.name+' was not rejected');assert.equal(v.loaders.length,0,row.name+' reached a GLTF parser');assert.equal(v.fetches.length,0,row.name+' reached binary resources');assert.equal(v.ready(),false);assert.match(v.elements.get('status').textContent,/No substitute scene/);v.hide();}
});
