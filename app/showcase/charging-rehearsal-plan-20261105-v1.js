/* Authored cross-recording task. Position IK only; no physical insertion or charging. */
(function(root){'use strict';
const C=root.RobotWorkbenchController||(typeof require==='function'?require('./robot-controller.js'):null);
const DURATION=13,TIP_OFFSET=[0,0,-.059],INITIAL={end_a:[-.32,.036,.40],end_b:[.22,.036,.46]},HOME=[-.45,.60,.12];
const copy=x=>JSON.parse(JSON.stringify(x)),add=(a,b)=>a.map((v,i)=>v+b[i]);
function build(goals){
 for(const id of ['end_a','end_b'])C.vector(goals[id],id+' authored body goal');
 const robot=copy(C.DEFAULT_ROBOT);robot.base=[-.65,0,-.42];robot.gripper_closed=.060;
 const steps=[],push=(op,duration_s,extra={})=>steps.push({op,duration_s,...extra}),move=(p,d,title)=>push('move',d,{position:p,title}),grip=(v,d)=>push('gripper',d,{value:v,title:v?'Close authored gripper':'Open authored gripper'});
 function end(id,pick,goal,first){
  move(add(pick,[0,.24,0]),first?.65:.45,'Approach '+id);move(pick, .55,'Lower to '+id);grip(1,.20);push('attach',.05,{object_id:id,title:'Take '+id});move(add(pick,[0,.24,0]),first?.65:.50,'Lift '+id);
  move(add(goal,[0,.24,.105]),.70,'Align '+id+' with authored USB-C port');move(add(goal,[0,0,.105]),.55,'Lower to authored port axis');move(goal,first?.85:1.00,'Seat '+id+' at authored proxy goal');push('detach',.05,{object_id:id,title:'Release '+id+' at authored proxy goal'});grip(0,.25);move(add(goal,[0,.28,.12]),first?.35:.75,'Retreat from '+id);if(first)push('wait',.45,{title:'Hold the released charger connection clear of the gripper'});
 }
 end('end_a',INITIAL.end_a,goals.end_a,true);end('end_b',INITIAL.end_b,goals.end_b,false);move(HOME,.95,'Retreat to hold');
 const used=steps.reduce((n,s)=>n+s.duration_s,0);push('wait',DURATION-used,{title:'Hold both authored endpoint goals'});
 const timeline=C.compile({robot,objects:{end_a:{position:INITIAL.end_a},end_b:{position:INITIAL.end_b}},steps,initial_tool:HOME,initial_gripper:0},{attachment_tolerance:.001});
 if(Math.abs(timeline.duration_s-DURATION)>1e-9||steps.at(-1).duration_s<1.5)throw Error('Complete13s ordered rehearsal required.');
 const releases=timeline.phases.filter(p=>p.step.op==='detach').map(p=>({id:p.step.object_id,time_s:p.end_s}));
 return {timeline,goals:copy(goals),releases,steps,authored_kinematic_rehearsal:true,physical_insertion_verified:false,charging_success_verified:false};
}
function at(plan,time){const state=C.at(plan.timeline,time);state.ik.joints[4]=-state.ik.joints[0];const limits=plan.timeline.robot.joint_limits;if(state.ik.joints.some((q,i)=>q<limits[i][0]||q>limits[i][1]))throw Error('Authored wrist roll exceeds rig limits.');const endpoints={};for(const id of ['end_a','end_b']){const position=state.objects[id].position;endpoints[id]={position:position.slice(),tip:add(position,TIP_OFFSET),goal:plan.goals[id].slice(),position_error:C.distance(position,plan.goals[id]),released:state.time_s>=plan.releases.find(r=>r.id===id).time_s,attached:state.attached===id,quaternion_xyzw:[0,0,0,1]};}return {...state,endpoints,final_hold:state.time_s>=11.3,authored_kinematic_rehearsal:true,physical_insertion_verified:false,charging_success_verified:false};}
const api=Object.freeze({build,at,DURATION,TIP_OFFSET,INITIAL,HOME});root.ChargingRehearsalPlanV1=api;if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(globalThis);
