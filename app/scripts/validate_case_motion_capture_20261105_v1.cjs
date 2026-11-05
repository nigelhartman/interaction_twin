/* Exact local full-canvas rehearsal intake. No browser, API or rendering. */
'use strict';
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const ROOT=path.resolve(__dirname,'..');
const C=require('../showcase/robot-controller.js');require('../showcase/aligned-scene-contract.js');
const P=require('../showcase/robot-task-plan.js');
function ref(r){assert.ok(r&&typeof r.path==='string'&&!path.isAbsolute(r.path)&&!r.path.split('/').includes('..'));const p=path.join(ROOT,r.path),b=fs.readFileSync(p);assert.equal(b.length,r.bytes);assert.equal(crypto.createHash('sha256').update(b).digest('hex'),r.sha256);return p;}
function near(a,b,e=1e-8){if(Array.isArray(a)){assert.ok(Array.isArray(b)&&a.length===b.length);a.forEach((v,i)=>near(v,b[i],e));}else if(typeof a==='number')assert.ok(typeof b==='number'&&Number.isFinite(b)&&Math.abs(a-b)<e,`${a} != ${b}`);else if(a&&typeof a==='object'){assert.ok(b&&typeof b==='object');for(const k of Object.keys(a))near(a[k],b[k],e);}else assert.equal(a,b);}
function bounds(rows){assert.ok(Array.isArray(rows)&&rows.length===4);assert.deepEqual(rows.map(r=>r.id),['tool_and_fingers','case_assembly','earbud_a','earbud_b']);for(const r of rows){assert.equal(r.points.length,8);for(const [x,y,z]of r.points){assert.ok([x,y,z].every(Number.isFinite));assert.ok(x>=96&&x<=1824&&y>=65&&y<=1015&&z>=-1&&z<=1,'Recorded focus bounds leave the frame');}}}
function validate(d){
 assert.equal(d.format,'interaction-twin-fullscreen-case-rehearsal-capture');assert.equal(d.version,1);assert.equal(d.duration_s,14);assert.deepEqual(d.canvas,[1920,1080]);assert.equal(d.fps,30);
 assert.equal(d.authored_kinematic_rehearsal,true);for(const k of['source_geometry_modified','physical_contact_verified','simulation_ready'])assert.equal(d[k],false);assert.equal(d.new_API_calls,0);assert.equal(d.new_GPU_calls,0);
 const old=JSON.parse(fs.readFileSync(ref(d.prior_capture_receipt))),plan=d.typed_plan;assert.deepEqual(plan,old.typed_plan);assert.equal(d.typed_plan_snapshot_JSON_sha256,old.typed_plan_snapshot_JSON_sha256);assert.equal(crypto.createHash('sha256').update(JSON.stringify(plan)).digest('hex'),d.typed_plan_snapshot_JSON_sha256);ref(d.original_workbench_renderer);
 for(const r of d.native_asset_refs)ref(r);assert.deepEqual(d.native_asset_refs,plan.objects.map(o=>o.asset_ref));for(const key of['program_ref','object_program_ref'])if(plan.source_binding[key])ref(plan.source_binding[key]);
 const timeline=C.compile(P.translate(plan,old.robot_kit),{joint_tool_position:(step,q,state)=>P.jointToolPosition(plan,step,q,state)});P.checkOutcome(plan,timeline);
 const knots=d.editorial_clock_knots;assert.deepEqual(knots.map(r=>r[0]),[0,2,4.5,8,11.5,15]);assert.equal(knots[0][1],0);near(knots.at(-1)[1],timeline.duration_s);for(let i=1;i<knots.length;i++)assert.ok(knots[i][1]>knots[i-1][1]);
 function clock(t){const v=Math.min(15,t/14*15);for(let i=1;i<knots.length;i++){const[a,x]=knots[i-1],[b,y]=knots[i];if(v<=b)return x+(y-x)*(v-a)/(b-a);}return timeline.duration_s;}
 const f=d.framing;assert.equal(f.full_trajectory_samples,281);assert.equal(f.minimum_horizontal_margin_px,96);assert.equal(f.minimum_vertical_margin_px,65);assert.equal(f.projected_full_trajectory.length,281);for(let i=0;i<281;i++){near(f.projected_full_trajectory[i].time_s,i*.05);bounds(f.projected_full_trajectory[i].projected_bounds);}
 assert.ok(Array.isArray(d.trace)&&d.trace.length>=250&&d.trace.length<1500);assert.ok(d.trace[0].film_s<=.1&&d.trace.at(-1).film_s>=13.98);let last=-1;
 for(const r of d.trace){assert.ok(r.film_s>last&&r.film_s<=14);if(last>=0)assert.ok(r.film_s-last<=.15,'Capture action clock has a gap');last=r.film_s;const at=clock(r.film_s);near(r.captured_task_runtime_s,at);near(C.at(timeline,at),r.state);bounds(r.projected_bounds);const u=Math.max(0,Math.min(1,(r.film_s-1)/1.3)),mix=u*u*(3-2*u);near(f.wide.position.map((v,i)=>v+(f.medium.position[i]-v)*mix),r.camera_position);near(f.wide.target.map((v,i)=>v+(f.medium.target[i]-v)*mix),r.camera_target);}
 const final=C.at(timeline,timeline.duration_s);assert.equal(final.attached,null);assert.equal(d.final_attached,null);assert.equal(d.final_closed_reference,true);assert.equal(d.both_earbuds_released,true);assert.equal(d.full_source_action_order_preserved,true);assert.equal(d.final_goals.length,2);
 for(const g of d.final_goals){assert.ok(plan.goal.component_ids.includes(g.id));assert.equal(g.attached,false);assert.ok(g.error<1e-5);near(final.objects[g.id].position,timeline.final.objects[g.id].position);}
 assert.ok(d.video_ref&&/^[a-f0-9]{64}$/.test(d.video_ref.sha256)&&d.video_ref.bytes>0&&d.video_ref.bytes<=64*1024*1024);
 return{passed:true,trace_samples:d.trace.length,duration_s:14,typed_plan_unchanged:true,both_earbuds_released:true,case_closed:true,source_geometry_modified:false,physical_contact_verified:false,editorial_medium_shot:true};
}
module.exports={validate,near,bounds};
if(require.main===module){if(process.argv.length!==3)throw Error('Usage: node scripts/validate_case_motion_capture_20261105_v1.cjs EXACT_RECEIPT.json');console.log(JSON.stringify(validate(JSON.parse(fs.readFileSync(process.argv[2])))));}
