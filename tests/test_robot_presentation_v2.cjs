const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const ROOT=path.resolve(__dirname,'..'),C=require('../showcase/robot-controller.js');require('../showcase/robot-task-plan.js');require('../showcase/aligned-scene-contract.js');const R=require('../showcase/robot-presentation-v2.js');
const baseline=JSON.parse(fs.readFileSync(path.join(ROOT,'output/brand-trailer-20261103/robot-capture-v2/baseline-task.json'))),kit=JSON.parse(fs.readFileSync(path.join(ROOT,'output/brand-trailer-20261103/robot-capture-v2/robot-kit.json')));
test('real complete task contains both pickups, releases, closed endpoints and a withdrawn final hold',()=>{
 const original=JSON.stringify(baseline),r=R.build(baseline,kit),timeline=r.timeline;
 assert.equal(JSON.stringify(baseline),original);assert.equal(r.proof.original_actions,38);assert.equal(r.plan.actions.length,40);
 assert.deepEqual(r.chronology.initial_hold_s,[0,1]);assert.equal(r.chronology.final_hold_s[1]-r.chronology.final_hold_s[0],2);
 const initial=C.at(timeline,0),final=C.at(timeline,timeline.duration_s),joint=r.plan.joints[0],key=joint.parent_id+':'+joint.id;
 assert.equal(initial.object_joints[key],0);assert.equal(final.object_joints[key],0);assert.equal(final.attached,null);assert.equal(final.gripper,0);
 for(const goal of r.goals){assert.deepEqual(final.objects[goal.id].position,goal.position);assert.equal(final.objects[goal.id].parent_id,undefined);assert.equal(timeline.phases.filter(p=>p.step.op==='attach'&&p.step.object_id===goal.id).length,1);assert.equal(timeline.phases.filter(p=>p.step.op==='detach'&&p.step.object_id===goal.id).length,1);}
 assert.ok(R.goalState(r,final).every(g=>g.reached));assert.ok(C.distance(r.goals[0].position,r.goals[1].position)>.08);assert.ok(C.distance(final.tool,final.objects.case_base.position)>.3);
});
test('all 24fps complete task frames have valid IK and carried objects follow the actual tool',()=>{
 const r=R.build(baseline,kit);let carried=new Set(),maxResidual=0;
 for(let frame=0;frame<=Math.ceil(r.timeline.duration_s*24);frame++){const s=C.at(r.timeline,Math.min(frame/24,r.timeline.duration_s));assert.equal(s.ik.reachable,true);maxResidual=Math.max(maxResidual,s.ik.residual);if(s.attached){carried.add(s.attached);assert.ok(s.objects[s.attached].position.every((v,i)=>Math.abs(v-s.tool[i]-s.attachment_offset[i])<1e-10));assert.equal(s.gripper,1);}}
 assert.ok(maxResidual<.002);assert.deepEqual([...carried].sort(),['case_base','earbud_a','earbud_b']);
});
test('truncating a transfer or ending attached cannot manufacture successful goal placement',()=>{
 for(const operation of['detach','move_to']){const p=structuredClone(baseline);p.actions=p.actions.filter(a=>!(a.operation===operation&&(operation==='detach'?a.object_id==='earbud_b':['Transfer earbud_b','Lower earbud_b'].includes(a.title))));assert.throws(()=>R.build(p,kit));}
 const p=structuredClone(baseline);p.goal.close_container=false;assert.throws(()=>R.build(p,kit),/complete remove/);
});
test('a geometric derivative cannot rebind another source inventory or claim verified physics',()=>{
 const r={format:'interaction-twin-robot-presentation-refinement',version:1,source_asset_refs:[],joint_id:baseline.joints[0].id,parent_id:'case_base',child_id:'case_lid',closed_q_rad:0,physical_accuracy_verified:false};
 assert.throws(()=>R.build(baseline,kit,r),/different source meshes/);
 r.source_asset_refs=baseline.objects.map(o=>({id:o.id,asset_ref:o.asset_ref}));r.physical_accuracy_verified=true;
 assert.throws(()=>R.build(baseline,kit,r),/not physical validation/);
});
test('actual mesh-derived closure and contained staging keep the full task reachable and both release goals exact',()=>{
 const refinement=JSON.parse(fs.readFileSync(path.join(ROOT,'output/brand-trailer-20261103/robot-capture-v2/refinement-inputs.json'))),r=R.build(baseline,kit,refinement);
 assert.equal(r.plan.format,'interaction-twin-robot-presentation-task');assert.equal(r.plan.version,2);assert.equal(r.plan.objects.find(o=>o.id==='case_lid').asset_scope,'mesh_derived_geometric_correction');
 for(const staged of refinement.contained_staging){const o=r.plan.objects.find(o=>o.id===staged.id),base=r.plan.objects.find(o=>o.id===staged.parent_id);assert.ok(Math.abs(o.initial_transform.position[1]-base.initial_transform.position[1]-staged.offset[1])<1e-12);assert.ok(staged.offset[1]+staged.display_half_height<=staged.opening_rim_offset_y-staged.clearance_display_units+1e-12);}
 for(let frame=0;frame<=Math.ceil(r.timeline.duration_s*24);frame++){const state=C.at(r.timeline,Math.min(frame/24,r.timeline.duration_s));assert.equal(state.ik.reachable,true);if(state.attached)assert.ok(state.objects[state.attached].position.every((v,k)=>Math.abs(v-state.tool[k]-state.attachment_offset[k])<1e-10));}
 assert.ok(R.goalState(r,r.timeline.final).every(g=>g.released&&g.reached&&g.goal_error_display_units<1e-10));
 const wrong=structuredClone(refinement);wrong.contained_staging[0].offset[1]+=.2;assert.throws(()=>R.build(baseline,kit,wrong),/fit below/);
});
