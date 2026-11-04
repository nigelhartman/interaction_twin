/* Display-only math checks; no inferred sample is saved or sent to agents. */
const test=require('node:test'),assert=require('node:assert/strict');
const {Scene,displayRecordAt,recordAt,rigidParts,GAP_POLICY}=require('../workspace/scene.js');
const SHA='a'.repeat(64);
function matrix({x=0,y=0,z=2,degrees=0,scale=1}={}){const a=degrees*Math.PI/180,c=Math.cos(a)*scale,s=Math.sin(a)*scale;return[[c,-s,0,x],[s,c,0,y],[0,0,scale,z],[0,0,0,1]];}
function row(t,options={}){return {sample_index:Math.round(t*10),timestamp_s:t,status:'supported_conditional_motion_candidate',visible:true,reference:false,reasons:[],camera_from_native_mesh:matrix(options)};}
function object(frames=[row(0),row(1,{x:1,degrees:60})]){return{object_id:'part',frames,max_hold_s:.1,native_scene:{span:1},fixed_depth_gauge_scale:1,provenance:{mesh_sha256:SHA}};}
function near(a,b){assert.ok(Math.abs(a-b)<1e-7,`${a} != ${b}`);}
test('bounded gap uses translation LERP and shortest-arc SLERP without mutating source',()=>{
 const o=object(),before=JSON.stringify(o),raw=recordAt(o,.5),r=displayRecordAt(o,.5);
 assert.equal(r.estimated,true);assert.equal(r.visible,true);assert.equal(r.row.sample_index,null);assert.equal(r.row.physical_pose_verified,false);
 near(r.row.camera_from_native_mesh[0][3],.5);near(r.row.camera_from_native_mesh[0][0],Math.cos(Math.PI/6));near(r.row.camera_from_native_mesh[1][0],.5);
 assert.deepEqual(r.row.source_endpoint_times_s,[0,1]);assert.equal(JSON.stringify(o),before);assert.deepEqual(recordAt(o,.5),raw);assert.equal(raw.visible,false);
});
test('exact endpoints remain original rows and raw toggle returns existing sampler behavior',()=>{
 const o=object();for(const t of [0,1]){const r=displayRecordAt(o,t);assert.equal(r.row,recordAt(o,t).row);assert.equal(r.estimated,false);}
 assert.deepEqual(displayRecordAt(o,.05,false),{...recordAt(o,.05),estimated:false});assert.equal(displayRecordAt(o,.5,false).visible,false);
});
test('no extrapolation, long gap, excessive translation or rotation',()=>{
 for(const [o,t]of[[object(),-.01],[object(),1.01],[object([row(0),row(2.01)]),1],[object([row(0),row(1,{x:2.001})]),.5],[object([row(0),row(1,{degrees:90.001})]),.5]])assert.equal(displayRecordAt(o,t).estimated,false);
 assert.equal(displayRecordAt(object([row(0),row(2,{x:2,degrees:90})]),1).estimated,true);
});
test('shortest rotation crosses 180 degrees coherently instead of unwinding',()=>{
 const r=displayRecordAt(object([row(0,{degrees:170}),row(1,{degrees:-170})]),.5);assert.equal(r.estimated,true);near(r.row.camera_from_native_mesh[0][0],-1);near(r.row.camera_from_native_mesh[1][0],0);
});
test('missing correspondence rows may be ghosted; rejection rows stay unknown',()=>{
 const missing={sample_index:5,timestamp_s:.5,status:'unsupported',visible:false,reference:false,reasons:['fewer_than_6_valid_training_correspondences'],camera_from_native_mesh:null};
 assert.equal(displayRecordAt(object([row(0),missing,row(1)]),.5).estimated,true);
 for(const reason of ['source_mask_not_plausible_in_model_audit','component_scope_model_rejected_mask','raw_full_mesh_silhouette_gate_failed','heldout_error_above_3px_median_or_8px_p90','insufficient_training_consensus','unknown_future_gate']){
  assert.equal(displayRecordAt(object([row(0),{...missing,reasons:[reason]},row(1)]),.5).estimated,false,reason);
 }
 assert.equal(displayRecordAt(object([row(0),{...missing,reasons:[]},row(1)]),.5).estimated,false);
 assert.equal(displayRecordAt(object([row(0),{...missing,status:'unsupported_candidate_projection_retained'},row(1)]),.5).estimated,false);
 const endpoint=object();endpoint.frames[1].reasons=['source_mask_not_plausible_in_model_audit'];assert.equal(displayRecordAt(endpoint,.5).estimated,false);
});
test('mesh, track, identity branch, gauge changes and orientation reinitialization block estimates',()=>{
 const change=(key,value)=>{const o=object();o.frames.splice(1,0,{timestamp_s:.5,status:'unsupported',visible:false,reasons:['fewer_than_6_valid_training_correspondences'],[key]:value});return o;};
 for(const [key,value]of[['mesh_sha256','b'.repeat(64)],['asset_sha256','b'.repeat(64)],['track_id','other'],['fixed_depth_gauge_scale',2],['identity_branch_id','other'],['reasons',['independent_reference_orientation_unverified']]])assert.equal(displayRecordAt(change(key,value),.5).estimated,false,key);
 const o=object();o.frames[0].identity_branch_id='one';o.frames[1].identity_branch_id='two';assert.equal(displayRecordAt(o,.5).estimated,false);
 const same=object();for(const f of same.frames)f.identity_branch_id='same';assert.equal(displayRecordAt(same,.5).estimated,true);
});
test('only proper finite rigid transforms with a fixed positive uniform scale are eligible',()=>{
 for(const mutate of [o=>o.frames[1].camera_from_native_mesh[0][0]=NaN,o=>o.frames[1].camera_from_native_mesh[3][0]=1,o=>o.frames[1].camera_from_native_mesh[0][0]=-1,o=>o.frames[1].camera_from_native_mesh[0][1]=.5,o=>o.frames[1].camera_from_native_mesh=matrix({scale:2}),o=>delete o.provenance.mesh_sha256,o=>o.fixed_depth_gauge_scale=0,o=>o.native_scene.span=NaN]){const o=object([row(0),row(1)]);mutate(o);assert.equal(displayRecordAt(o,.5).estimated,false);}
 assert.equal(rigidParts(matrix({scale:2})).scale,2);assert.equal(GAP_POLICY.opacity,.42);
});
test('empty native assets and unordered or duplicate source times do not gain video placement',()=>{
 assert.equal(displayRecordAt(object([]),.5).estimated,false);for(const frames of [[row(1),row(0)],[row(0),row(0),row(1)]])assert.equal(displayRecordAt(object(frames),.5).estimated,false);
});
test('native inspection hides inline controls and clears estimated source-pose status',()=>{
 const o=object(),host={style:{display:'flex'}},button={setAttribute(){}},badge={};
 const view={inspection:'native',estimateGaps:true,gapUI:{host,button,badge},enabled:new Set(['part']),objectsInView:()=>[o],displayRecord:()=>({estimated:true,gap_s:1}),draw(){Scene.prototype.gapLabel.call(this);}};
 const statuses=Scene.prototype.setTime.call(view,.5);assert.equal(host.hidden,true);assert.equal(host.style.display,'none');assert.equal(badge.hidden,true);assert.equal(badge.textContent,'');assert.equal(statuses[0].native_asset,true);assert.equal(statuses[0].estimated,false);assert.equal(statuses[0].row,null);
 view.inspection='component';Scene.prototype.gapLabel.call(view);assert.equal(host.hidden,false);assert.equal(host.style.display,'flex');assert.equal(badge.hidden,false);assert.match(badge.textContent,/display only/);
});
