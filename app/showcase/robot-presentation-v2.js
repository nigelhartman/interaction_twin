/* Versioned local film rehearsal. Typed actions, no dynamics or robot commands. */
(function(root){'use strict';
const clone=v=>JSON.parse(JSON.stringify(v)),assert=(v,m)=>{if(!v)throw Error(m);},finite=n=>typeof n==='number'&&Number.isFinite(n);
function build(baseline,kit,refinement=null){
 const C=root.RobotWorkbenchController,P=root.RobotTaskPlan;
 assert(C&&P,'Load the typed task controller first.');
 P.validate(baseline,baseline,kit);
 assert(baseline.goal.operation==='remove_components'&&baseline.goal.destination==='tray'&&baseline.goal.open_container&&baseline.goal.close_container,'The capture needs a complete remove-to-tray-and-close task.');
 const plan=clone(baseline),originalActions=clone(baseline.actions);
 const initial={id:'film_initial_hold',operation:'wait',title:'Closed case · ready to begin',duration_s:1.};
 const home=C.forward((kit.robot||kit.kinematics).home,kit.robot||kit.kinematics).tip;
 plan.actions=[initial,...plan.actions.filter((a,i)=>!(i===plan.actions.length-1&&a.operation==='wait')),
  {id:'film_withdraw_arm',operation:'move_to',title:'Withdraw from the completed task',duration_s:1.2,position:home},
  {id:'film_final_hold',operation:'wait',title:'Both earbuds released at their goals · lid closed',duration_s:2.}];
 const checked=P.validate(plan,baseline,kit);
 if(refinement){
  assert(refinement.format==='interaction-twin-robot-presentation-refinement'&&refinement.version===1,'A typed versioned geometric refinement is required.');
  assert(P.canonical(refinement.source_asset_refs)===P.canonical(baseline.objects.map(o=>({id:o.id,asset_ref:o.asset_ref}))),'Refinement belongs to different source meshes.');
  const joint=checked.joints.find(j=>j.id===refinement.joint_id);
  assert(joint&&refinement.parent_id===joint.parent_id&&refinement.child_id===joint.child_id,'Refinement joint identity differs.');
  assert(refinement.closed_q_rad===0&&refinement.physical_accuracy_verified===false,'Refined closure is a zero-reference estimate, not physical validation.');
  assert(Array.isArray(refinement.render_asset_overrides)&&refinement.render_asset_overrides.length<=1,'Only the declared derived articulated child may replace its render geometry.');
  for(const item of refinement.render_asset_overrides){const ref=item.asset_ref;assert(item.id===joint.child_id&&item.scope==='mesh_derived_geometric_correction'&&ref&&/^(data|output)\/[A-Za-z0-9_./-]+\.glb$/.test(ref.path)&&!ref.path.split('/').some(p=>!p||p.startsWith('.'))&&/^[a-f0-9]{64}$/.test(ref.sha256)&&Number.isSafeInteger(ref.bytes)&&ref.bytes>0&&ref.bytes<=32*1024*1024,'Invalid explicitly bound geometric derivative.');const object=checked.objects.find(o=>o.id===item.id);object.asset_ref=clone(ref);object.asset_scope='mesh_derived_geometric_correction';}
  const source=refinement.source_joint,S=root.AlignedSceneContract;
  assert(source&&Array.isArray(source.axis)&&source.axis.length===3&&source.axis.every(finite)&&Math.abs(Math.hypot(...source.axis)-1)<1e-6&&Array.isArray(source.limits_rad)&&source.limits_rad.length===2&&source.limits_rad.every(finite)&&source.limits_rad[0]<=0&&source.limits_rad[1]>0&&source.physical_accuracy_verified===false,'Finite estimated joint geometry is required.');
  S.similarity(source.parent_from_joint);S.similarity(source.joint_from_child_zero);joint.source_joint=clone(source);
  assert(refinement.actuation&&refinement.actuation.asset_sha256===checked.objects.find(o=>o.id===joint.child_id).asset_ref.sha256,'The actuation cue must bind the exact displayed child geometry.');
  for(const action of checked.actions)if(action.operation==='set_joint'&&action.joint_id===joint.id){action.actuation=clone(refinement.actuation);if(action.value!==0){assert(finite(refinement.open_q_rad)&&refinement.open_q_rad>Math.PI/12&&refinement.open_q_rad<=source.limits_rad[1]&&refinement.open_q_rad>=source.limits_rad[0],'A bounded source-derived opening angle is required.');action.value=refinement.open_q_rad;}}
  const staging=refinement.contained_staging;
  assert(Array.isArray(staging)&&staging.length===checked.goal.component_ids.length&&new Set(staging.map(s=>s.id)).size===staging.length,'Every carried component needs one exact containment anchor.');
  for(const staged of staging){
   const object=checked.objects.find(o=>o.id===staged.id),attachment=checked.attachments.find(a=>a.child_id===staged.id);
   assert(object?.role==='removable'&&checked.goal.component_ids.includes(staged.id)&&attachment?.parent_id===joint.parent_id&&staged.parent_id===joint.parent_id&&staged.physical_seating_verified===false,'Staging cannot invent a component relationship.');
   assert(Array.isArray(staged.offset)&&staged.offset.length===3&&staged.offset.every(finite)&&Math.abs(staged.offset[0]-attachment.offset[0])<1e-10&&Math.abs(staged.offset[2]-attachment.offset[2])<1e-10,'Containment preserves original lateral placement.');
   assert([staged.display_half_height,staged.opening_rim_offset_y,staged.body_bottom_offset_y,staged.clearance_display_units].every(finite)&&staged.display_half_height>0&&staged.clearance_display_units>=.001&&staged.clearance_display_units<=.01&&staged.offset[1]+staged.display_half_height<=staged.opening_rim_offset_y-staged.clearance_display_units+1e-10&&staged.offset[1]-staged.display_half_height>=staged.body_bottom_offset_y+staged.clearance_display_units-1e-10,'Contained geometry must fit below the measured rim and above the body bottom.');
   const delta=staged.offset[1]-attachment.offset[1],parent=checked.objects.find(o=>o.id===attachment.parent_id);
   object.initial_transform.position=parent.initial_transform.position.map((v,i)=>v+staged.offset[i]);attachment.offset=staged.offset.slice();
   const attachIndex=checked.actions.findIndex(a=>a.operation==='attach'&&a.object_id===staged.id);assert(attachIndex>=0,'The staged part needs its real typed pickup.');
   let lower=attachIndex-1;while(lower>=0&&checked.actions[lower].operation!=='move_to')lower--;
   assert(lower>=0,'The typed pickup needs a reachable lower waypoint.');checked.actions[lower].position[1]+=delta;
  }
  // This is deliberately a new presentation derivative, never a silently
  // modified curated task or a claim that the native SAM mesh is unchanged.
  checked.format='interaction-twin-robot-presentation-task';checked.version=2;
 }
 const timeline=C.compile(P.translate(checked,kit),{joint_tool_position:(step,q,state)=>P.jointToolPosition(checked,step,q,state)});
 P.checkOutcome(checked,timeline);
 const goals=checked.goal.component_ids.map(id=>({id,position:timeline.final.objects[id].position.slice()}));
 assert(goals.length===2&&C.distance(goals[0].position,goals[1].position)>.08,'Two distinct released goal anchors are required.');
 assert(timeline.final.attached===null&&timeline.final.gripper===0,'Release the objects and open the gripper before the final hold.');
 for(const joint of checked.joints){const key=joint.parent_id+':'+joint.id;assert(Math.abs(timeline.initial.object_joints[key])<1e-10&&Math.abs(timeline.final.object_joints[key])<1e-10,'Both endpoints must be the closed reference.');}
 const finalHold=timeline.phases.at(-1);
 assert(finalHold.step.id==='film_final_hold'&&C.distance(timeline.final.tool,timeline.final.objects[checked.objects.find(o=>o.role==='assembly_base').id].position)>.3,'The final view must show a withdrawn tool.');
 return{format:'interaction-twin-robot-presentation',version:2,plan:checked,timeline,goals,
  chronology:{initial_hold_s:[0,initial.duration_s],complete_motion_s:[initial.duration_s,finalHold.start_s],final_hold_s:[finalHold.start_s,timeline.duration_s]},
  proof:{original_actions:originalActions.length,presented_actions:checked.actions.length,all_original_nonfinal_wait_actions_included:true,complete_goal:true,
   refined_geometry:!!refinement,authored_staging:true,physical_accuracy_verified:false,contact_or_grasp_verified:false,new_model_calls:0}};
}
function goalState(presentation,state){return presentation.goals.map(goal=>({id:goal.id,position:goal.position.slice(),observed_position:state.objects[goal.id].position.slice(),goal_error_display_units:C().distance(state.objects[goal.id].position,goal.position),released:state.attached!==goal.id&&!state.objects[goal.id].parent_id,
 reached:C().distance(state.objects[goal.id].position,goal.position)<1e-6&&state.attached!==goal.id&&!state.objects[goal.id].parent_id}));}
function C(){return root.RobotWorkbenchController;}
const api=Object.freeze({build,goalState});root.RobotPresentationV2=api;if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(globalThis);
