const test=require('node:test'),assert=require('node:assert/strict');
const C=require('../showcase/aligned-scene-contract.js');
const source='a'.repeat(64),sha='b'.repeat(64),asset={path:'data/native/part.glb',sha256:sha,bytes:100};
function matrix(s=1,x=0){return[[s,0,0,x],[0,s,0,0],[0,0,s,3],[0,0,0,1]];}
function fixture(){return{format:'sam3d-common-video-scene',version:1,original_program:{path:'data/source/procedure/interaction.json',sha256:sha,bytes:100},source:{sha256:source,width:540,height:960,duration_s:3,time_domain:'source_video_presentation_seconds'},camera:{K:[[800,0,269.5],[0,810,479.5],[0,0,1]],width:540,height:960,world_from_camera:matrix(1,0).map((r,i)=>r.map((v,j)=>i===j?1:0))},common_frame:{id:'current_source_camera_cv',coordinate_convention:'+Xright+Ydown+Zforward',metric_scale_verified:false,assembly_verified:false,camera_motion_removed:false},physical_accuracy_verified:false,accepted_motion_poses:0,timing:{max_hold_s:0,max_interpolation_gap_s:.8,max_estimated_gap_s:1},components:[{id:'part',label:'Part',asset:{...asset},poses:[{timestamp_s:0,status:'estimated',camera_from_native_mesh:matrix(2,0)},{timestamp_s:.5,status:'estimated',camera_from_native_mesh:matrix(4,1)}]}]};}
test('source, common frame and unverified scope are mandatory',()=>{
 assert.equal(C.validateScene(fixture(),source).components.length,1);
 for(const change of [s=>s.source.sha256='c'.repeat(64),s=>s.common_frame.assembly_verified=true,s=>s.physical_accuracy_verified=true,s=>s.accepted_motion_poses=1,s=>s.camera.width=960,s=>s.camera.world_from_camera[0][3]=1,s=>s.timing.max_hold_s=1]){const s=fixture();change(s);assert.throws(()=>C.validateScene(s,source));}
});
test('native scale is applied once, reflection and nonuniform geometry transformations reject',()=>{
 assert.equal(C.similarity(matrix(3)).s,3);
 for(const mutate of [m=>m[0][0]=-1,m=>m[0][0]=2,m=>m[3][0]=.1,m=>m[0][0]=NaN]){const m=matrix();mutate(m);assert.throws(()=>C.similarity(m));}
 const midpoint=C.interpolate(matrix(2,0),matrix(4,1),.5);assert.equal(C.similarity(midpoint).s,3);assert.equal(midpoint[0][3],.5);
});
test('source camera projection preserves integer pixel centers and skew after CV-to-Three basis conversion',()=>{
 const K=[[710,12,253.25],[0,820,470.75],[0,0,1]],W=540,H=960,P=C.projection(K,W,H);
 for(let i=0;i<100;i++){const x=(i-50)/70,y=((i*7)%100-50)/100,z=1+i/50,p=[x,-y,-z,1];const clip=P.map(r=>r.reduce((v,q,j)=>v+q*p[j],0));const uv=[(clip[0]/clip[3]+1)*W/2-.5,(1-clip[1]/clip[3])*H/2-.5],expected=[(K[0][0]*x+K[0][1]*y)/z+K[0][2],K[1][1]*y/z+K[1][2]];assert.ok(Math.abs(uv[0]-expected[0])<1e-10);assert.ok(Math.abs(uv[1]-expected[1])<1e-10);}
});
test('all parts share source time and bounded interpolation retains unknown gaps',()=>{
 const s=fixture(),c=s.components[0];assert.equal(C.at(c,0,s.timing).interpolated,undefined);
 const mid=C.at(c,.25,s.timing);assert.equal(mid.status,'estimated');assert.equal(mid.interpolated,true);assert.equal(mid.camera_from_native_mesh[0][3],.5);
 assert.equal(C.at(c,.25,s.timing,false).status,'unknown');assert.equal(C.at(c,0,s.timing,false).status,'unknown');
 assert.equal(C.at(c,-.1,s.timing).camera_from_native_mesh,null);assert.equal(C.at(c,1,s.timing).camera_from_native_mesh,null);
 c.poses.splice(1,0,{timestamp_s:.25,status:'unknown',camera_from_native_mesh:null});assert.equal(C.at(c,.2,s.timing).status,'unknown');assert.equal(C.at(c,.3,s.timing).status,'unknown');
 const distant=fixture();distant.components[0].poses[1].timestamp_s=2;assert.equal(C.at(distant.components[0],1,distant.timing).status,'unknown');
});
test('unknown poses cannot expose a render matrix and source samples remain strictly ordered',()=>{
 const s=fixture();s.components[0].poses[0].status='unknown';assert.throws(()=>C.validateScene(s,source));
 const duplicate=fixture();duplicate.components[0].poses[1].timestamp_s=0;assert.throws(()=>C.validateScene(duplicate,source));
});
test('iframe messages require parent window, same origin, exact source and manifest',()=>{
 const parent={},origin='http://127.0.0.1:8765',data={type:'interaction-twin:aligned-time',source_sha256:source,manifest_sha256:sha,time_s:.3,show_estimated:true,view:'source'};
 const event={origin,source:parent,data};assert.equal(C.message(event,parent,origin,source,sha),data);
 for(const bad of [{...event,origin:'https://example.com'},{...event,source:{}},{...event,data:{...data,source_sha256:'c'.repeat(64)}},{...event,data:{...data,manifest_sha256:'d'.repeat(64)}},{...event,data:{...data,time_s:NaN}}])assert.equal(C.message(bad,parent,origin,source,sha),null);
});
test('registry cannot alias sources or external artifacts',()=>{
 const index={format:'sam3d-aligned-scene-index',version:1,scenes:[{source_sha256:source,manifest:{path:'data/scene/scene.json',sha256:sha,bytes:100}}]};assert.equal(C.validateIndex(index),index);index.scenes.push(index.scenes[0]);assert.throws(()=>C.validateIndex(index));
 assert.throws(()=>C.ref({...asset,path:'data/../private/file'}));assert.throws(()=>C.ref({...asset,path:'https://example.com/model.glb'}));
});
test('initial preview chooses a saved time with the most parts, then best minimum silhouette fit',()=>{
 const scene=fixture();scene.components[0].poses=[{timestamp_s:0,status:'unknown',camera_from_native_mesh:null},...scene.components[0].poses.map((p,i)=>({...p,timestamp_s:1+i,convex_silhouette_fit_iou:i?.8:.6}))];
 scene.components.push({...scene.components[0],id:'part_b',poses:scene.components[0].poses.map(p=>({...p}))});
 assert.deepEqual(C.previewTime(scene),{time_s:2,count:2,min_fit_iou:.8});
 scene.components[1].poses[2]={timestamp_s:2,status:'unknown',camera_from_native_mesh:null};assert.equal(C.previewTime(scene).time_s,1);
 assert.equal(C.previewTime({components:[{poses:[]}]}).time_s,0);
});
function connectedFixture(){const s=fixture(),base=s.components[0];base.id='base';base.poses=[{timestamp_s:0,status:'estimated',camera_from_native_mesh:matrix(2,0)},{timestamp_s:.5,status:'estimated',camera_from_native_mesh:C.multiply(C.axisRotation([0,1,0],Math.PI/4),matrix(2,.4))}];s.components.push({id:'lid',label:'Lid',asset:{...asset},poses:base.poses.map((p,i)=>({...p,camera_from_native_mesh:matrix(.8,i?-100:100)}))},{id:'removable',label:'Removable object',asset:{...asset},poses:base.poses.map(p=>({...p}))});s.mechanics={format:'sam3d-constrained-mechanisms',version:1,physical_accuracy_verified:false,joints:[{id:'hinge',type:'revolute',status:'estimated',parent_id:'base',child_id:'lid',parent_from_joint:[[.5,0,0,.2],[0,.5,0,.05],[0,0,.5,0],[0,0,0,1]],joint_from_child_zero:[[.8,0,0,-.2],[0,.8,0,0],[0,0,.8,0],[0,0,0,1]],axis:[0,0,1],limits_rad:[0,Math.PI],samples:[{timestamp_s:0,status:'estimated',q_rad:0,source_sample_index:1},{timestamp_s:.5,status:'estimated',q_rad:Math.PI/2,source_sample_index:2}]}]};return s;}
const point=(matrix,xyz)=>matrix.map(row=>row.reduce((v,q,j)=>v+q*[...xyz,1][j],0)).slice(0,3);
test('hinge anchors stay identical at saved and interpolated times despite deliberately incompatible raw child poses',()=>{
 const s=C.validateScene(connectedFixture(),source);
 for(let i=0;i<=100;i++){const time=i/200,state=C.sceneAt(s,time,true),parent=state.parts.find(p=>p.id==='base'),child=state.parts.find(p=>p.id==='lid'),a=point(parent.camera_from_native_mesh,[.2,.05,0]),b=point(child.camera_from_native_mesh,[.25,0,0]);assert.equal(child.constrained,true);assert.ok(Math.hypot(...a.map((v,j)=>v-b[j]))<1e-10,'The lid may not detach between saved source frames.');assert.ok(Math.abs(child.camera_from_native_mesh[0][3])<10,'Must not reuse independently fitted child world translations.');assert.ok(Math.abs(C.similarity(child.camera_from_native_mesh).s-.8)<1e-10,'Child native scale must be applied once.');}
});
test('attached child world-pose samples are never read for replay or interpolation',()=>{
 const s=connectedFixture();Object.defineProperty(s.components[1],'poses',{get(){throw Error('Independent child world poses must not be consulted');}});
 const frame=C.sceneAt(s,.25,true);assert.ok(frame.parts.find(p=>p.id==='lid').camera_from_native_mesh);assert.equal(frame.parts.find(p=>p.id==='lid').constrained,true);
});
test('missing joint or parent input hides the connected assembly, while removable objects stay independent',()=>{
 for(const missing of ['joint','parent']){const s=connectedFixture();if(missing==='joint')s.mechanics.joints[0].samples.splice(1,0,{timestamp_s:.25,status:'unknown',q_rad:null});else s.components[0].poses.splice(1,0,{timestamp_s:.25,status:'unknown',camera_from_native_mesh:null});const frame=C.sceneAt(s,.125,true);assert.equal(frame.parts.find(p=>p.id==='base').camera_from_native_mesh,null);assert.equal(frame.parts.find(p=>p.id==='lid').camera_from_native_mesh,null);assert.ok(frame.parts.find(p=>p.id==='removable').camera_from_native_mesh);}
 const hidden=C.sceneAt(connectedFixture(),.25,false);assert.ok(hidden.parts.every(p=>p.camera_from_native_mesh===null));
});
test('mechanics graph source, topology, frames, axis and angle boundaries fail closed',()=>{
 for(const mutate of [s=>s.mechanics.source_sha256='c'.repeat(64),s=>s.mechanics.joints[0].axis=[0,0,2],s=>s.mechanics.joints[0].samples[0].q_rad=-10,s=>s.mechanics.joints[0].parent_from_joint[0][0]=2,s=>s.mechanics.joints.push({...s.mechanics.joints[0],id:'cycle',parent_id:'lid',child_id:'base'})]){const s=connectedFixture();mutate(s);assert.throws(()=>C.validateScene(s,source));}
});
test('unresolved persistent attachments hide the entire connected assembly without reading child world poses',()=>{
 const scene=connectedFixture();scene.mechanics.unresolved_children=[{id:'unresolved',type:'prismatic',status:'unknown',parent_id:'lid',child_id:'removable',reason:'No supported sliding fit'}];C.validateScene(scene,source);
 for(const component of scene.components.slice(1))Object.defineProperty(component,'poses',{get(){throw Error('A persistent child cannot fall back to independent world poses');}});
 const frame=C.sceneAt(scene,.25,true);assert.ok(frame.parts.every(p=>p.camera_from_native_mesh===null));assert.ok(frame.joints.every(j=>j.status==='unknown'&&j.q_rad===null));
 const cycle=connectedFixture();cycle.mechanics.unresolved_children=[{id:'cycle',type:'fixed',status:'unknown',parent_id:'lid',child_id:'base'}];assert.throws(()=>C.validateScene(cycle,source));
});
function guardedNonrigidFixture(){
 const scene=connectedFixture(),same=(path,hash='c')=>({path:'data/guard/'+path,sha256:hash.repeat(64),bytes:100}),reason='rigid_motion_withheld_due_to_explicit_source_nonrigidity';
 const proposal=same('proposal.json'),oldScene=same('original-scene.json','d'),oldMechanics=same('original-mechanics.json','e');
 scene.inputs=[scene.original_program,proposal,oldScene,oldMechanics];
 scene.components[1].poses=scene.components[1].poses.map(p=>({...p,status:'unknown',camera_from_native_mesh:null,reason,method:'source_nonrigidity_feasibility_withholding',joint_id:'hinge',accepted_motion_evidence:false}));
 scene.mechanics.joints=[];scene.mechanics.proposal=proposal;scene.mechanics.unresolved_children=[{id:'hinge',type:'revolute',status:'unknown',parent_id:'base',child_id:'lid',reason}];
 scene.mechanics.rigidity_feasibility={source_sha256:source,interaction_ref:scene.original_program,proposal_ref:proposal,source_scene_ref:oldScene,source_mechanics_ref:oldMechanics,withheld_joint_ids:['hinge'],withheld_component_ids:['lid'],rigidity_established:false,flexible_motion_recovered:false,native_geometry_modified:false,unaffected_parent_and_removable_poses_preserved:true,physical_accuracy_verified:false,new_api_calls:0,new_gpu_calls:0,automatic_approval:false,findings:[{joint_id:'hinge',parent_id:'base',child_id:'lid',proposal_joint_type:'revolute',reason:'explicit_nonrigidity_in_persistent_joint_proposal',measured_nonrigidity_verified:false,declarations:[{limitation_index:0,source_limitation:'deformable leather, not rigid',assertion_clause:'not rigid',normalized_assertion:'not rigid'}]}]};
 return scene;
}
test('exact source-bound nonrigid withholding preserves the unaffected parent and removable saved poses without a rigid child or invented angle',()=>{
 const scene=C.validateScene(guardedNonrigidFixture(),source);
 for(const time of [0,.125,.25,.5]){const state=C.sceneAt(scene,time,true),part=id=>state.parts.find(p=>p.id===id);assert.deepEqual(part('base').camera_from_native_mesh,C.at(scene.components[0],time,scene.timing).camera_from_native_mesh);assert.deepEqual(part('removable').camera_from_native_mesh,C.at(scene.components[2],time,scene.timing).camera_from_native_mesh);assert.equal(part('lid').camera_from_native_mesh,null);assert.equal(part('lid').status,'unknown');assert.deepEqual(state.joints,[{id:'hinge',parent_id:'base',child_id:'lid',status:'unknown',q_rad:null,reason:'rigid_motion_withheld_due_to_explicit_source_nonrigidity'}]);}
 assert.ok(C.sceneAt(scene,.25,false).parts.every(p=>p.camera_from_native_mesh===null));
});
test('a nonrigid reason without the exact withholding guard never exempts an unresolved assembly',()=>{
 const scene=guardedNonrigidFixture();delete scene.mechanics.rigidity_feasibility;C.validateScene(scene,source);const state=C.sceneAt(scene,.25,true);assert.equal(state.parts.find(p=>p.id==='base').camera_from_native_mesh,null);assert.equal(state.parts.find(p=>p.id==='lid').camera_from_native_mesh,null);assert.ok(state.parts.find(p=>p.id==='removable').camera_from_native_mesh);
});
test('nonrigid guard rejects mismatched source/program/proposal, missing saved inputs, altered withheld IDs and fabricated child motion',()=>{
 const changes=[s=>s.mechanics.rigidity_feasibility.source_sha256='f'.repeat(64),s=>s.mechanics.rigidity_feasibility.interaction_ref={...s.original_program,sha256:'f'.repeat(64)},s=>s.mechanics.rigidity_feasibility.proposal_ref={...s.mechanics.proposal,sha256:'f'.repeat(64)},s=>s.inputs.pop(),s=>s.mechanics.rigidity_feasibility.withheld_joint_ids=['foreign'],s=>s.mechanics.rigidity_feasibility.withheld_component_ids=['removable'],s=>s.mechanics.rigidity_feasibility.unaffected_parent_and_removable_poses_preserved=false,s=>s.mechanics.rigidity_feasibility.flexible_motion_recovered=true,s=>s.mechanics.rigidity_feasibility.findings[0].declarations[0].normalized_assertion='may be deformable',s=>s.mechanics.rigidity_feasibility.findings[0].declarations[0].assertion_clause='deformable',s=>s.components[1].poses[0].camera_from_native_mesh=matrix(),s=>s.components[1].poses[0].status='estimated',s=>s.components[1].poses[0].accepted_motion_evidence=true,s=>s.components[1].poses[0].q_rad=.5,s=>s.mechanics.unresolved_children[0].q_rad=.5];
 for(const change of changes){const scene=guardedNonrigidFixture();change(scene);assert.throws(()=>C.validateScene(scene,source));assert.throws(()=>C.sceneAt(scene,.25,true));}
});
test('preserving a guarded nonrigid parent never fills its unknown rows or a removable identity gap',()=>{
 const scene=guardedNonrigidFixture();scene.components[0].poses.splice(1,0,{timestamp_s:.25,status:'unknown',camera_from_native_mesh:null});scene.components[2].poses.splice(1,0,{timestamp_s:.25,status:'unknown',camera_from_native_mesh:null});C.validateScene(scene,source);
 for(const time of [.125,.25,.375])assert.ok(C.sceneAt(scene,time,true).parts.every(p=>p.camera_from_native_mesh===null));
 assert.ok(C.sceneAt(scene,0,true).parts.find(p=>p.id==='base').camera_from_native_mesh);assert.ok(C.sceneAt(scene,.5,true).parts.find(p=>p.id==='removable').camera_from_native_mesh);
});
test('a guarded nonrigid descendant needs an explicit withheld ancestor chain, while unrelated rigid failures still hide their assembly',()=>{
 const scene=guardedNonrigidFixture(),edge={id:'dependent',parent_id:'lid',child_id:'removable',type:'fixed',status:'unknown',reason:'rigid_motion_withheld_due_to_unsupported_ancestor'};
 scene.mechanics.unresolved_children.push(edge);scene.mechanics.rigidity_feasibility.withheld_joint_ids.push('dependent');scene.mechanics.rigidity_feasibility.withheld_component_ids.push('removable');
 scene.components[2].poses=scene.components[2].poses.map(p=>({...p,status:'unknown',camera_from_native_mesh:null,reason:edge.reason,method:'source_nonrigidity_feasibility_withholding',joint_id:'dependent',accepted_motion_evidence:false}));
 C.validateScene(scene,source);let state=C.sceneAt(scene,.25,true);assert.ok(state.parts.find(p=>p.id==='base').camera_from_native_mesh);assert.ok(state.parts.filter(p=>p.id!=='base').every(p=>p.camera_from_native_mesh===null));assert.ok(state.joints.every(j=>j.status==='unknown'&&j.q_rad===null));
 const foreign=structuredClone(scene);foreign.mechanics.unresolved_children[1].parent_id='base';assert.throws(()=>C.validateScene(foreign,source));
 scene.components.push({id:'unknown_attachment',label:'Unknown attachment',asset:{...asset},poses:[]});scene.mechanics.unresolved_children.push({id:'unsupported_rigid',parent_id:'base',child_id:'unknown_attachment',type:'prismatic',status:'unknown'});C.validateScene(scene,source);state=C.sceneAt(scene,.25,true);assert.ok(state.parts.every(p=>p.camera_from_native_mesh===null));
});
test('aligned publication binds the exact program and every current source-bound original native asset',()=>{
 const scene=fixture(),program={interaction:{source:{sha256:source}},original_sha256:sha,interaction_path:scene.original_program.path},reviews=[{component_id:'part',source_sha256:source,native_glb:{...asset,path:'data/verified-copy/candidate.glb'}}];
 assert.equal(C.matchesProgram(scene,program,reviews),true,'Byte-identical native copies may have different review paths');
 for(const mutate of [x=>x.p.original_sha256='c'.repeat(64),x=>x.p.interaction_path='data/old/interaction.json',x=>x.p.interaction.source.sha256='c'.repeat(64),x=>x.r[0].native_glb.sha256='c'.repeat(64),x=>x.r[0].native_glb.bytes++,x=>x.r[0].source_sha256='c'.repeat(64),x=>x.r[0].component_id='another_part',x=>x.r=[]]){const x={p:structuredClone(program),r:structuredClone(reviews)};mutate(x);assert.equal(C.matchesProgram(scene,x.p,x.r),false);}
});
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),savedMechanics=path.join(__dirname,'../data/sam3d-mechanisms-jabra-20261102-v1/scene.json');
test('actual immutable Jabra hinge remains connected through the reported 16.707s frame and dense source timeline',{skip:!fs.existsSync(savedMechanics)},()=>{
 const raw=fs.readFileSync(savedMechanics);assert.equal(crypto.createHash('sha256').update(raw).digest('hex'),'6738c73e030deab8c1b67f838afc4012b1fc5fa097b8a4754f8a867ebe1ecdcb');const scene=JSON.parse(raw);C.validateScene(scene,scene.source.sha256);
 let supported=0,hidden=0;for(const t of [16.707,...Array.from({length:1001},(_,i)=>scene.source.duration_s*i/1000)]){const frame=C.sceneAt(scene,t,true);for(const joint of scene.mechanics.joints){const parent=frame.parts.find(p=>p.id===joint.parent_id),child=frame.parts.find(p=>p.id===joint.child_id);assert.equal(Boolean(parent.camera_from_native_mesh),Boolean(child.camera_from_native_mesh),'An unsupported assembly must disappear together');if(!child.camera_from_native_mesh){hidden++;continue;}supported++;const a=point(parent.camera_from_native_mesh,joint.parent_anchor_native),b=point(child.camera_from_native_mesh,joint.child_anchor_native);assert.ok(Math.hypot(...a.map((v,i)=>v-b[i]))<1e-10,'Shared actual native vertex anchors must remain coincident between source frames');if(t===16.707){const q=frame.joints.find(j=>j.id===joint.id);assert.equal(q.interpolated,true);assert.ok(Math.abs(q.q_rad-.4937309753752873)<1e-10);}}}assert.ok(supported>500);assert.ok(hidden>100);
});
const savedGuardedWallet=path.join(__dirname,'../data/rigidity-feasibility/overnight-camera-wallet-20261103-v1/wallet/scene.json');
test('actual immutable wallet guard exposes preserved holder and coin observations while its flexible flap remains unknown',{skip:!fs.existsSync(savedGuardedWallet)},()=>{
 const raw=fs.readFileSync(savedGuardedWallet);assert.equal(crypto.createHash('sha256').update(raw).digest('hex'),'9ae828d5e8806ffaea9b057e8d6cb4812cb1fd390636c0fce88717fdc47bfc10');const scene=JSON.parse(raw);C.validateScene(scene,scene.source.sha256);let parentVisible=0,coinVisible=0;
 for(const time of scene.components[0].poses.map(p=>p.timestamp_s)){const state=C.sceneAt(scene,time,true);for(const id of['wallet_holder','gold_coin']){const displayed=state.parts.find(p=>p.id===id),component=scene.components.find(c=>c.id===id);assert.deepEqual(displayed.camera_from_native_mesh,C.at(component,time,scene.timing).camera_from_native_mesh);if(displayed.camera_from_native_mesh){if(id==='wallet_holder')parentVisible++;else coinVisible++;}}assert.equal(state.parts.find(p=>p.id==='pouch_flap').camera_from_native_mesh,null);assert.equal(state.joints[0].status,'unknown');assert.equal(state.joints[0].q_rad,null);assert.match(state.joints[0].reason,/explicit_source_nonrigidity/);}
 assert.equal(parentVisible,93);assert.equal(coinVisible,14);const frame=C.sceneAt(scene,24.168789,true);assert.equal(frame.parts.filter(p=>p.camera_from_native_mesh).length,2);assert.ok(C.previewTime(scene).count>=2);
});

test('surface-first opening selects actual supported observations and preserves unknown intervals',()=>{
 const scene=fixture(),row=(i,t,extra={})=>({source_sample_index:i,timestamp_s:t,status:'source_observed_prior',asset:{...asset},visible_component_ids:['part'],...extra}),observations={format:'interaction-twin-source-visible-flexure-series',version:1,source:{sha256:source},source_program:scene.original_program,physical_accuracy_verified:false,simulation_ready:false,samples:[row(1,0,{status:'unknown'}),row(2,1),row(3,2,{qualitative_state:{confidence:.8,wallet_state:'open'}})]};
 const before=JSON.stringify(observations),chosen=C.previewTime(scene,observations);assert.equal(chosen.time_s,2);assert.equal(chosen.source_sample_index,3);assert.equal(chosen.current_geometry,false);assert.equal(chosen.representation,'source_visible_observation');assert.equal(JSON.stringify(observations),before,'Picking an opening observation cannot fill or rewrite unknown rows');
 assert.throws(()=>C.previewTime(scene,{...observations,source:{sha256:'c'.repeat(64)}}));assert.throws(()=>C.previewTime(scene,{...observations,source_program:{...scene.original_program,sha256:'d'.repeat(64)}}));assert.throws(()=>C.previewTime(scene,{...observations,samples:[row(2,9)]}));assert.throws(()=>C.previewTime(scene,{...observations,samples:[row(2,1,{visible_component_ids:['foreign']})]}));
 assert.equal(C.previewTime(scene,{...observations,samples:[row(1,0,{status:'unknown'})]}).count,0,'No supported surface must not fall back to a misleading native pose');
 const ambiguous={...observations,samples:[row(2,1),row(3,2,{qualitative_state:{confidence:.4,wallet_state:'open'}})]};assert.equal(C.previewTime(scene,ambiguous).time_s,1,'Low-confidence state descriptions cannot decide the opening view');
});

test('every current surface-enabled twin opens at a supported saved surface, never its obsolete native preview time',t=>{
 const fs=require('node:fs'),path=require('node:path'),root=path.resolve(__dirname,'..'),indexPath=path.join(root,'showcase/public/aligned-scenes.json');if(!fs.existsSync(indexPath))return t.skip('Current curated saved artifacts are not in the source-only package.');
 const index=JSON.parse(fs.readFileSync(indexPath));let checked=0;for(const entry of index.scenes.filter(r=>r.observed_surfaces)){const scene=JSON.parse(fs.readFileSync(path.join(root,entry.manifest.path))),observations=JSON.parse(fs.readFileSync(path.join(root,entry.observed_surfaces.path))),selected=C.previewTime(scene,observations),row=observations.samples.find(r=>r.source_sample_index===selected.source_sample_index);assert.ok(row&&row.status==='source_observed_prior'&&row.asset);assert.equal(selected.time_s,row.timestamp_s);assert.ok(selected.count>0);checked++;}assert.equal(checked,3);
});
