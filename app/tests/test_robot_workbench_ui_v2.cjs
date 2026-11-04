'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const U=require('../showcase/robot-workbench-ui-v2.js');
const plan={objects:[{id:'base',label:'Case',component_id:'case_base'},{id:'lid',label:'Lid',component_id:'case_lid'},{id:'a',label:'Wireless earbud',component_id:'earbud_a'},{id:'b',label:'Wireless earbud',component_id:'earbud_b'}],joints:[{id:'hinge',parent_id:'base',child_id:'lid'}],actions:[{operation:'wait'},{operation:'set_joint',joint_id:'hinge',value:1},{operation:'detach',object_id:'a'},{operation:'detach',object_id:'b'},{operation:'set_joint',joint_id:'hinge',value:0},{operation:'wait'}]};
const starts=[0,.3,1.8,2.4,2.8,4.6],ends=[.3,1.8,2.4,2.8,4.6,5.2],timeline={duration_s:5.2,phases:starts.map((start_s,index)=>({index,start_s,end_s:ends[index],before:{object_joints:{'base:hinge':index>1?1:0}}})),final:{objects:{a:{position:[0,0,1]},b:{position:[0,0,2]}}}};
const state=(time_s=0)=>({time_s,attached:null,objects:{a:{position:[0,0,0]},b:{position:[0,0,0]}},ik:{reachable:true}});
test('checkpoints use actual action end times and distinguish identical part labels without changing the plan',()=>{
 const before=JSON.stringify({plan,timeline}),rows=U.checkpoints(plan,timeline);
 assert.deepEqual(rows.map(r=>r.time_s),[1.8,2.4,2.8,4.6,5.2]);assert.deepEqual(rows.map(r=>r.title),['Open Lid','earbud a placed','earbud b placed','Close Lid','Final state']);
 assert.equal(JSON.stringify({plan,timeline}),before);assert.equal(rows[1].object_id,'a');assert.equal(rows[2].object_id,'b');
});
test('action navigation handles exact boundaries, inside an action, final hold and reverse seeking',()=>{
 for(const [t,d,expected]of[[0,-1,0],[0,1,.3],[.3,1,1.8],[.3,-1,0],[1.9,-1,1.8],[1.9,1,2.4],[5.2,1,5.2],[5.2,-1,4.6],[900,-1,4.6],[-9,1,.3]])assert.equal(U.adjacent(timeline,t,d),expected);
 for(const bad of[NaN,Infinity,'2'])assert.throws(()=>U.adjacent(timeline,bad,1));assert.throws(()=>U.adjacent(timeline,1,0));assert.throws(()=>U.adjacent({duration_s:0,phases:[]},0,1));
});
test('outcome status requires time, release, reachability and all staged positions, and never claims physical success',()=>{
 const initial=state();assert.deepEqual([U.status(plan,timeline,initial,false).label,U.status(plan,timeline,initial,false).placed],['Ready',0]);assert.equal(U.status(plan,timeline,{...initial,time_s:1},true).label,'Running');
 const done=state(5.2);done.objects=structuredClone(timeline.final.objects);assert.equal(U.status(plan,timeline,done,false).complete,true);assert.equal(U.status(plan,timeline,done,false).label,'Preview complete');assert.equal(U.status(plan,timeline,done,false).physical_success_verified,false);assert.ok(U.placements(plan,timeline,done).every(g=>g.physical_placement_verified===false));
 for(const changed of[{...done,time_s:4.9},{...done,attached:'a'},{...done,ik:{reachable:false}},{...done,objects:{...done.objects,a:{position:[.1,0,1]}}}])assert.equal(U.status(plan,timeline,changed,false).complete,false);
 assert.equal(U.status(plan,timeline,{...done,time_s:0},false).complete,false,'Matching initial positions cannot imply completed execution.');
});
test('camera presets are finite independent views and contain no execution or network path',()=>{
 assert.deepEqual(Object.keys(U.CAMERA_PRESETS),['overview','parts','top']);for(const preset of Object.values(U.CAMERA_PRESETS)){assert.equal(preset.position.length,3);assert.equal(preset.target.length,3);assert.ok([...preset.position,...preset.target].every(Number.isFinite));assert.notDeepEqual(preset.position,preset.target);}assert.notDeepEqual(U.CAMERA_PRESETS.parts,U.CAMERA_PRESETS.overview);
});

test('plain task copy keeps generated part identity and never upgrades the rehearsal claims',()=>{
 assert.equal(U.shortAction('12 / 40 · Attach authored earbud_a'),'Attach earbud a');assert.equal(U.shortAction('Move to authored waypoint'),'Move to the next point');assert.equal(U.shortAction('Open estimated lid'),'Open lid');assert.equal(U.taskSummary({...plan,goal:{operation:'open_close',open_container:false,close_container:true}}),'Close Lid.');assert.equal(U.taskSummary({...plan,goal:{operation:'open_close',open_container:true,close_container:true}}),'Open Lid, then close it.');const p={...plan,goal:{operation:'remove_components',component_ids:['a','b'],destination:'tray',close_container:true}};assert.equal(U.taskSummary(p),'Move both earbuds to the tray, then close the case.');assert.equal(U.taskSummary({...p,goal:{...p.goal,operation:'return_components'}}),'Return both earbuds to the case, then close the case.');assert.equal(U.status(plan,timeline,state(),false).physical_success_verified,false);
});
test('default tasks follow the selected source component inventory without inventing charging or unknown objects',()=>{
 const record=(...ids)=>({components:ids.map(id=>({id}))});
 const laptop=U.defaultTask(record('laptop_base','laptop_lid'));
 assert.equal(laptop.instruction,'Open the laptop_lid, then close it.');assert.doesNotMatch(laptop.instruction,/earbud/);
 assert.equal(U.defaultTask(record('camera_body','battery_door','batteries')).instruction,'Move batteries to the tray.');
 const phone=U.defaultTask(record('phone_blue','cable_connector'));assert.equal(phone.instruction,'Move cable_connector to the tray.');assert.match(phone.hint,/Charging.*not simulated/);
 assert.equal(U.defaultTask(record('wallet_holder','gold_coin')).instruction,'Move gold_coin to the tray.');
 assert.match(U.defaultTask(record('case_base','case_lid','earbud_a','earbud_b')).instruction,/both earbuds/);
 assert.equal(U.defaultTask(record('charger','socket','usb_c_cable')),null);
 assert.equal(U.defaultTask(record('unknown')),null);
});
test('primary workbench shows the scene and play controls while every technical tool stays in Advanced',()=>{
 const fs=require('node:fs'),path=require('node:path'),html=fs.readFileSync(path.join(__dirname,'../showcase/robot-workbench.html'),'utf8'),advanced=html.indexOf('<details id="workbench-advanced"');assert.ok(advanced>html.indexOf('id="play-task"'));assert.ok(html.indexOf('id="play-task"')<html.indexOf('id="workbench-viewport"'),'The primary action precedes the canvas, including on a short display');assert.ok(advanced>html.indexOf('id="task-time"'));assert.ok(advanced>html.indexOf('id="current-step"'));for(const id of['task-form','task-prompt','task-speed','previous-action','next-action','task-checkpoints','reach-state','joint-inspector','task-steps','task-json','record-preview','download-plan','download-kit'])assert.ok(html.indexOf('id="'+id+'"')>advanced,id+' must remain in the drawer');assert.doesNotMatch(html,/<details id="workbench-advanced"[^>]*\bopen[ >]/);const ids=[...html.matchAll(/id="([^"]+)"/g)].map(m=>m[1]);assert.equal(new Set(ids).size,ids.length,'Controls retain one unambiguous target after reflow');
});
