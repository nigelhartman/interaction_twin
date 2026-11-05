/* Pure bounds policy for this authored shot; no model or source edits. */
(function(root){'use strict';
const C=root.RobotWorkbenchController||(typeof require==='function'?require('./robot-controller.js'):null),Q=root.ChargingRehearsalPlanV1||(typeof require==='function'?require('./charging-rehearsal-plan-20261105-v1.js'):null),F=root.RobotCameraFitV1||(typeof require==='function'?require('./robot-camera-fit-v1.js'):null);
function cablePoints(state){const a=state.objects.end_a.position,b=state.objects.end_b.position;return [[a[0],a[1],a[2]+.052],[a[0],a[1],a[2]+.14],[-.23,.028,.64],[.14,.028,.66],[b[0],b[1],b[2]+.14],[b[0],b[1],b[2]+.052]];}
function fit(plan,modelBounds,portCenters){
 const points=[],corner=(low,high)=>{if(![low,high].every(p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite))||low.some((v,i)=>v>high[i]))throw Error('Finite ordered model bounds required.');for(let mask=0;mask<8;mask++)points.push([0,1,2].map(i=>mask&(1<<i)?high[i]:low[i]));},pad=(p,r)=>corner(p.map(v=>v-r),p.map(v=>v+r));
 if(!Array.isArray(modelBounds)||modelBounds.length!==2)throw Error('Both exact displayed model bounds required.');for(const b of modelBounds)corner(...b);
 const r=plan.timeline.robot;corner([r.base[0]-.18,-.005,r.base[2]-.18],[r.base[0]+.18,r.shoulder_height+.095,r.base[2]+.18]);
 let worst=0;for(let frame=0;frame<=390;frame++){const s=Q.at(plan,frame/30);if(!s.ik.reachable||!C.forward(s.ik.joints,r).within_limits)throw Error('Full-trajectory IK or roll limit failed.');worst=Math.max(worst,s.ik.residual);for(const k of ['origin','elbow','wrist','tip'])pad(s.ik[k],k==='tip'?.13:.10);for(const id of ['end_a','end_b'])pad(s.objects[id].position,.072);for(const p of cablePoints(s))pad(p,.035);}
 for(const p of Object.values(portCenters))pad(p,.04);
 const fitted=F.fit(points,{vertical_fov_deg:36,aspect:16/9,view_direction:[.42,1.55,1.65],edge_margin_fraction:.07});
 const close=(indices,start,end,ids,includeCable=false)=>{const ps=[],box=(lo,hi)=>{for(let mask=0;mask<8;mask++)ps.push([0,1,2].map(i=>mask&(1<<i)?hi[i]:lo[i]));},padding=(p,r)=>box(p.map(v=>v-r),p.map(v=>v+r));for(const i of indices)box(...modelBounds[i]);for(let f=Math.floor(start*30);f<=Math.ceil(end*30);f++){const s=Q.at(plan,f/30);if(ids.length)padding(s.tool,.125);for(const id of ids)padding(s.objects[id].position,.075);if(includeCable)for(const p of cablePoints(s))padding(p,.035);}for(const i of indices)padding(portCenters[i===0?'end_b':'end_a'],.04);return F.fit(ps,{vertical_fov_deg:36,aspect:16/9,view_direction:[.42,1.55,1.65],edge_margin_fraction:.09});};
 const a=close([1],1.2,5.3,['end_a']),b=close([0],5.3,10.35,['end_b']),final=close([0,1],11.3,13,[],true);
 const value=v=>({position:v.position,target:v.target,vertical_fov_deg:v.vertical_fov_deg,aspect:v.aspect,near:v.near,far:v.far});
 const keys=[[0,fitted],[.9,fitted],[2.05,a],[5.3,a],[6.2,b],[10.35,b],[11.15,final],[13,final]].map(([time_s,v])=>({time_s,...value(v)}));
 return {...fitted,method:'full_13s_FK_establishing_then_editorial_action_closeups_and_two_endpoint_hold',editorial_camera:true,full_rig_required_during_establishing_only:true,keys,action_view_proofs:{charger:a.proof,phone:b.proof,final:final.proof},frequency_hz:30,full_task_samples_checked:391,maximum_ik_residual:worst};
}
function cameraAt(fitted,t){const keys=fitted.keys,hi=keys.findIndex(k=>k.time_s>=t);if(hi<=0)return keys[hi===0?0:keys.length-1];const a=keys[hi-1],b=keys[hi],u=C.clamp((t-a.time_s)/(b.time_s-a.time_s),0,1),s=u*u*(3-2*u);return {...a,position:C.mix(a.position,b.position,s),target:C.mix(a.target,b.target,s)};}
const api=Object.freeze({fit,cablePoints,cameraAt});root.ChargingRehearsalViewV1=api;if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(globalThis);
