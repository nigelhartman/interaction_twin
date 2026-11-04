/* Exact offline derived task → browser FK; no model or renderer claims. */
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process');
const ROOT=path.resolve(__dirname,'..'),C=require('../showcase/robot-controller.js'),P=require('../showcase/robot-task-plan.js');require('../showcase/aligned-scene-contract.js');
const raw=JSON.parse(fs.readFileSync(path.join(ROOT,'output/submission-finalization-20261103/jabra/robot-plans-private-probe-v3.json'))),plan=raw.plans[0].plan;
const kit=JSON.parse(cp.execFileSync(path.join(ROOT,'.venv-sam/bin/python'),['-c','import json;from robot_task_program import get_robot_kit;print(json.dumps(get_robot_kit()))'],{cwd:ROOT,encoding:'utf8'}));
const program=JSON.parse(fs.readFileSync(path.join(ROOT,plan.source_binding.program_ref.path))),derived=JSON.parse(fs.readFileSync(path.join(ROOT,plan.source_binding.object_program_ref.path))),copy=v=>JSON.parse(JSON.stringify(v));
function compiled(p=plan){P.validate(p,plan,kit);const t=C.compile(P.translate(p,kit),{joint_tool_position:(s,q,v)=>P.jointToolPosition(p,s,q,v)});P.checkOutcome(p,t);return t;}
function bound(ref){const raw=fs.readFileSync(path.join(ROOT,ref.path)),n=raw.readUInt32LE(12),doc=JSON.parse(raw.subarray(20,20+n)),bin=28+n,p=doc.meshes[0].primitives[0],a=doc.accessors[p.attributes.POSITION],bv=doc.bufferViews[a.bufferView],start=bin+(bv.byteOffset||0)+(a.byteOffset||0),stride=bv.byteStride||12,low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];assert.equal(a.componentType,5126);for(let i=0;i<a.count;i++)for(let k=0;k<3;k++){const v=raw.readFloatLE(start+i*stride+k*4);low[k]=Math.min(low[k],v);high[k]=Math.max(high[k],v);}return{low,high};}

test('declared derivative retains exact original source and cannot masquerade as native generation',()=>{
 assert.equal(P.sourceBinding(plan,program,derived),true);assert.equal(plan.objects.filter(o=>o.asset_scope==='curated_geometric_derivative').length,1);assert.equal(plan.source_binding.object_program_ref.sha256,raw.source_bound_derivative_ref.sha256);
 for(const change of[d=>d.claims.declared_derivative_geometry_used=false,d=>d.source_audit_summary.default_eligible=false,d=>d.evidence_program.sha256='0'.repeat(64),d=>d.component_assets.find(a=>a.component_id==='case_lid').native_glb.sha256='0'.repeat(64)]){const d=copy(derived);change(d);assert.throws(()=>P.sourceBinding(plan,program,d));}
 const native=JSON.parse(fs.readFileSync(path.join(ROOT,derived.base_object_program_ref.path)));assert.throws(()=>P.sourceBinding(plan,program,native));
});

test('forty-action rehearsal closes both endpoints, releases both goals and withdraws with bounded IK',()=>{
 const t=compiled();assert.equal(plan.actions.length,40);assert.equal(t.initial.object_joints['case_base:'+plan.joints[0].id],0);assert.equal(t.final.object_joints['case_base:'+plan.joints[0].id],0);assert.equal(t.final.attached,null);assert.equal(t.final.gripper,0);
 assert.equal(plan.actions.at(-2).title,'Withdraw from the completed task');assert.ok(C.distance(t.final.tool,t.final.objects.case_base.position)>.3);
 const base=plan.objects.find(o=>o.id==='case_base'),baseBounds=bound(base.asset_ref),baseHalf=(baseBounds.high[1]-baseBounds.low[1])/2*base.display_max_dimension/Math.max(...baseBounds.high.map((v,k)=>v-baseBounds.low[k]));assert.ok(Math.abs(t.final.objects.case_base.position[1]-baseHalf)<1e-8);assert.ok(Math.abs(t.initial.objects.case_base.position[1]-baseHalf)<1e-8);
 for(const id of['earbud_a','earbud_b']){const object=plan.objects.find(o=>o.id===id),b=bound(object.asset_ref),span=Math.max(...b.high.map((v,k)=>v-b.low[k])),half=(b.high[1]-b.low[1])/2*object.display_max_dimension/span;assert.ok(Math.abs(t.final.objects[id].position[1]-half-.009)<1e-8);assert.equal(t.final.objects[id].parent_id,undefined);}
 assert.ok(C.distance(t.final.objects.earbud_a.position,t.final.objects.earbud_b.position)>.1);
 for(let i=0;i<=600;i++){const state=C.at(t,t.duration_s*i/600);assert.equal(state.ik.reachable,true);assert.equal(state.ik.within_limits,true);assert.ok(state.ik.residual<.002);assert.equal(state.objects.case_lid.parent_id,'case_base');}
});

test('single-earbud, separate return and open-only keep exact derivative provenance',()=>{
 for(const row of raw.plans){const p=row.plan;assert.equal(row.validation.valid,true);P.validate(p,p,kit);assert.equal(P.sourceBinding(p,program,derived),true);const t=C.compile(P.translate(p,kit),{joint_tool_position:(s,q,v)=>P.jointToolPosition(p,s,q,v)});P.checkOutcome(p,t);assert.equal(t.final.attached,null);assert.equal(p.claims.simulation_ready,false);}
});

test('standalone local prompts preserve measured display support and completed withdrawal',()=>{
 const local=P.compileInstruction(raw.plans[0].text,plan,kit),a=compiled(local),b=compiled();assert.equal(local.actions.length,40);
 for(const id of['case_base','earbud_a','earbud_b'])assert.ok(C.distance(a.final.objects[id].position,b.final.objects[id].position)<1e-8);
 assert.ok(C.distance(a.final.tool,b.final.tool)<1e-8);
 const returned=P.compileInstruction(raw.plans[2].text,plan,kit),t=compiled(returned),source=raw.plans[2].plan,expected=C.compile(P.translate(source,kit),{joint_tool_position:(s,q,v)=>P.jointToolPosition(source,s,q,v)});
 assert.ok(C.distance(t.final.objects.earbud_a.position,expected.final.objects.earbud_a.position)<1e-8);assert.equal(t.final.attached,null);
});
