/* Source-conditioned pose hypotheses are separate from immutable observations. */
(function(root){'use strict';
const C=root.AlignedSceneContract||(typeof require==='function'?require('./aligned-scene-contract.js'):null),finite=Number.isFinite,fail=m=>{throw Error(m);},same=(a,b)=>a&&b&&a.path===b.path&&a.sha256===b.sha256&&a.bytes===b.bytes;
function validate(d,s,sceneRef){
 if(!C||d?.format!=='interaction-twin-source-conditioned-motion-hypotheses'||d.version!==1||d.physical_accuracy_verified!==false||d.simulation_ready!==false||d.source_unknowns_preserved!==true||d.native_geometry_modified!==false)fail('Unknown or overstated motion hypothesis.');
 C.ref(d.source_ref,256*1024*1024);C.ref(d.evidence_program_ref,8*1024*1024);C.ref(d.baseline_scene_ref,8*1024*1024);C.ref(d.flow_proofs_ref,16*1024*1024);
 if(d.source_ref.sha256!==s.source.sha256||!same(d.evidence_program_ref,s.original_program)||sceneRef&&!same(d.baseline_scene_ref,sceneRef))fail('Motion hypotheses belong to another source program or scene.');
 const p=d.policy;if(!p||!finite(p.maximum_anchor_gap_s)||p.maximum_anchor_gap_s<=0||p.maximum_anchor_gap_s>4||!finite(p.maximum_viewer_interpolation_gap_s)||p.maximum_viewer_interpolation_gap_s<=0||p.maximum_viewer_interpolation_gap_s>.22||p.original_unknown_poses_or_barriers_changed!==false||p.physical_accuracy_verified!==false)fail('Unbounded or overstated motion policy.');
 if(!Array.isArray(d.groups)||d.groups.length>32)fail('Bounded motion groups required.');const roots=new Set();let total=0;
 for(const g of d.groups){const component=s.components.find(c=>c.id===g.root_component_id),joints=(s.mechanics?.joints||[]).filter(j=>j.parent_id===g.root_component_id),ids=[g.root_component_id,...joints.map(j=>j.child_id)];if(!component||roots.has(component.id)||JSON.stringify(ids)!==JSON.stringify(g.component_ids)||!Array.isArray(g.episodes)||g.episodes.length>256)fail('Motion assembly identity differs.');roots.add(component.id);
  for(const e of g.episodes){const a=e.anchor_before,b=e.anchor_after;if(!a||!b||a.component_id!==component.id||b.component_id!==component.id||!finite(a.timestamp_s)||!finite(b.timestamp_s)||a.timestamp_s>=b.timestamp_s||b.timestamp_s-a.timestamp_s>p.maximum_anchor_gap_s+1e-9&&e.predictions.length)fail('Invalid bounded motion anchors.');
   for(const anchor of[a,b]){const row=component.poses.find(r=>r.source_sample_index===anchor.source_sample_index);if(!row||row.status!=='estimated'||row.timestamp_s!==anchor.timestamp_s||JSON.stringify(row.camera_from_native_mesh)!==JSON.stringify(anchor.camera_from_native_mesh))fail('Motion anchor differs from its original observation.');}
   if(!Array.isArray(e.predictions))fail('Motion episode lacks its prediction ledger.');
   const old=component.poses.filter(r=>r.timestamp_s>a.timestamp_s&&r.timestamp_s<b.timestamp_s&&r.status==='unknown').map(r=>r.source_sample_index);if(JSON.stringify(old)!==JSON.stringify(e.original_unknown_indices))fail('Motion changed the original unknown ledger.');if(!e.predictions.length)continue;
   if(typeof e.view_change_break!=='boolean'||!Array.isArray(e.constant_joint_states)||e.constant_joint_states.length!==joints.length)fail('Motion episode lacks its visibility or joint rules.');
   for(const key of['anchor_before_view_segment_id','anchor_after_view_segment_id'])if(e[key]!==undefined&&(typeof e[key]!=='string'||!e[key].startsWith(e.id+':')))fail('Invalid motion anchor view segment.');
   for(const j of joints){const state=e.constant_joint_states.find(v=>v.id===j.id),ja=j.samples.find(v=>v.source_sample_index===a.source_sample_index),jb=j.samples.find(v=>v.source_sample_index===b.source_sample_index);if(!state||!ja||!jb||ja.status!=='estimated'||jb.status!=='estimated'||!finite(state.q_rad)||Math.abs(ja.q_rad-jb.q_rad)>(p.maximum_constant_joint_state_difference_rad??0)+1e-9||Math.abs(state.q_rad-(ja.q_rad+jb.q_rad)/2)>1e-9||state.hidden_part_observed!==false)fail('Motion cannot invent a hidden joint angle.');}
   let previous=a.timestamp_s;for(const row of e.predictions){total++;if(total>12000||!finite(row.timestamp_s)||row.timestamp_s<=previous||row.timestamp_s>=b.timestamp_s||row.original_unknown_row_preserved!==true||!['estimated','withheld'].includes(row.status))fail('Invalid motion prediction ledger.');previous=row.timestamp_s;
    if(row.source_sample_index!==null&&row.source_sample_index!==undefined){const raw=component.poses.find(v=>v.source_sample_index===row.source_sample_index);if(!old.includes(row.source_sample_index)||raw?.timestamp_s!==row.timestamp_s)fail('Motion prediction aliases a different original row.');}
    if(row.status==='withheld'){if(Object.keys(row.component_poses||{}).length||row.joint_states?.length)fail('Withheld motion exposes a pose.');continue;}
    if(JSON.stringify(Object.keys(row.component_poses).sort())!==JSON.stringify([...ids].sort())||JSON.stringify(row.joint_states)!==JSON.stringify(e.constant_joint_states)||typeof row.view_segment_id!=='string'||!row.view_segment_id.startsWith(e.id+':'))fail('Estimated motion lost component, joint or view identity.');
    for(const matrix of Object.values(row.component_poses))C.similarity(matrix);
    for(const j of joints){const q=row.joint_states.find(v=>v.id===j.id).q_rad,expected=C.multiply(C.multiply(C.multiply(row.component_poses[component.id],j.parent_from_joint),C.axisRotation(j.axis,q)),j.joint_from_child_zero),actual=row.component_poses[j.child_id];if(expected.some((r,i)=>r.some((v,k)=>Math.abs(v-actual[i][k])>2e-6)))fail('Predicted child detached from its source joint.');}
   }
  }
 }
 return d;
}
function anchorNode(scene,group,e,anchor,segment){const state=C.sceneAt(scene,anchor.timestamp_s,true),poses={};for(const id of group.component_ids){const p=state.parts.find(v=>v.id===id);if(!p?.camera_from_native_mesh)return null;poses[id]=p.camera_from_native_mesh;}return{timestamp_s:anchor.timestamp_s,status:'estimated',component_poses:poses,joint_states:e.constant_joint_states,view_segment_id:segment,support:'original_anchor',source_sample_index:anchor.source_sample_index};}
function atEpisode(scene,doc,group,e,time){
 if(time<e.anchor_before.timestamp_s||time>e.anchor_after.timestamp_s||!e.predictions.length)return null;
 const first=e.predictions.find(r=>r.status==='estimated'),last=[...e.predictions].reverse().find(r=>r.status==='estimated');if(!first||!last)return null;
 const nodes=[anchorNode(scene,group,e,e.anchor_before,e.anchor_before_view_segment_id??first.view_segment_id),...e.predictions,anchorNode(scene,group,e,e.anchor_after,e.anchor_after_view_segment_id??last.view_segment_id)].filter(Boolean);let lo=0,hi=nodes.length;while(lo<hi){const m=(lo+hi)>>1;if(nodes[m].timestamp_s<time-1e-6)lo=m+1;else hi=m;}
 const b=nodes[lo],a=nodes[lo-1];if(b&&Math.abs(b.timestamp_s-time)<=1e-6)return b.status==='estimated'?{...b,interpolated:false}:null;
 if(!a||!b||a.status!=='estimated'||b.status!=='estimated'||a.view_segment_id!==b.view_segment_id||b.timestamp_s-a.timestamp_s>doc.policy.maximum_viewer_interpolation_gap_s+1e-9)return null;
 const ratio=(time-a.timestamp_s)/(b.timestamp_s-a.timestamp_s),rootId=group.root_component_id,poses={[rootId]:C.interpolate(a.component_poses[rootId],b.component_poses[rootId],ratio)};
 // Interpolate only the root. Re-evaluate FK so even intermediate frames keep
 // the estimated hinge attached instead of blending child translations apart.
 for(const j of(scene.mechanics?.joints||[]).filter(j=>j.parent_id===rootId)){const q=e.constant_joint_states.find(s=>s.id===j.id).q_rad;poses[j.child_id]=C.multiply(C.multiply(C.multiply(poses[rootId],j.parent_from_joint),C.axisRotation(j.axis,q)),j.joint_from_child_zero);}
 return{timestamp_s:time,status:'estimated',component_poses:poses,joint_states:e.constant_joint_states,view_segment_id:a.view_segment_id,support:'bounded_between_source_conditioned_estimates',source_sample_index:null,interpolated:true};
}
function sceneAt(scene,doc,time,showEstimated=true,baseline=C.sceneAt(scene,time,showEstimated)){
 if(!doc||!showEstimated||!finite(time)||time<0||time>scene.source.duration_s)return baseline;
 const parts=new Map(baseline.parts.map(p=>[p.id,p])),joints=new Map(baseline.joints.map(j=>[j.id,j])),used=[];
 for(const g of doc.groups){if(g.component_ids.every(id=>parts.get(id)?.camera_from_native_mesh))continue;const candidates=g.episodes.map(e=>({e,row:atEpisode(scene,doc,g,e,time)})).filter(x=>x.row);if(candidates.length!==1)continue;const{e,row}=candidates[0];for(const id of g.component_ids)parts.set(id,{...parts.get(id),status:'estimated',camera_from_native_mesh:row.component_poses[id],constrained:g.component_ids.length>1,interpolated:row.interpolated===true,motion_hypothesis:true,source_sample_index:row.source_sample_index,reason:'Separate source-conditioned estimate; original unknown observation preserved'});for(const j of(scene.mechanics?.joints||[]).filter(j=>j.parent_id===g.root_component_id)){const q=e.constant_joint_states.find(s=>s.id===j.id).q_rad;joints.set(j.id,{id:j.id,parent_id:j.parent_id,child_id:j.child_id,status:'estimated',q_rad:q,interpolated:row.interpolated===true,motion_hypothesis:true,source_sample_index:row.source_sample_index,reason:'Matching neighbouring saved joint states; hidden joint is not directly observed'});}used.push({root_component_id:g.root_component_id,component_ids:g.component_ids,episode_id:e.id,view_segment_id:row.view_segment_id,support:row.support,interpolated:row.interpolated===true});}
 return{parts:[...parts.values()],joints:[...joints.values()],...(used.length?{motion_hypothesis:{used:true,groups:used,original_unknowns_preserved:true,physical_accuracy_verified:false}}:{})};
}
const api={validate,sceneAt};root.EstimatedMotionV1=api;if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(globalThis);
