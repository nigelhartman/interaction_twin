/* Explicit source-frame repetition; never changes strict observation queries. */
(function(root){'use strict';
const fail=m=>{throw Error(m);},finite=x=>typeof x==='number'&&Number.isFinite(x),same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
function derive(doc,reference){
 if(doc?.format!=='interaction-twin-source-visible-flexure-series'||doc.kind!=='source_observed_monocular_depth_prior'||doc.policy?.maximum_source_time_error_s!==.18||!Array.isArray(doc.samples)||doc.samples.length<2)fail('Exact strict source observations required.');
 const rows=doc.samples,duration=doc.source.duration_s;
 const samples=rows.map((r,i)=>{const t=r.timestamp_s,left=i?t-rows[i-1].timestamp_s:rows[1].timestamp_s-t,right=i+1<rows.length?rows[i+1].timestamp_s-t:t-rows[i-1].timestamp_s;if(!finite(t)||!(left>0&&right>0))fail('Strict source time order required.');return {source_sample_index:r.source_sample_index,observation_timestamp_s:t,status:r.status,left_actual_source_gap_s:left,right_actual_source_gap_s:right,display_window_start_s:Math.max(0,t-Math.min(.55,left/2)),display_window_end_s:Math.min(duration,t+Math.min(.55,right/2)),visible_component_ids:r.status==='source_observed_prior'?(r.visible_component_ids||[]):[]};});
 return {format:'interaction-twin-observed-surface-presentation',version:1,source_observations:reference,source_sha256:doc.source.sha256,source_duration_s:duration,mode:'editorial_source_sample_repeat',strict_observation_query_maximum_source_time_error_s:.18,display_repeat_cap_s:.55,bound_derivation:'each_side_minimum_of0.55s_and_half_actual_adjacent_source_gap',nearest_row_includes_unknowns:true,search_past_unknown_rows:false,geometry_interpolated:false,current_geometry:false,new_tracking_or_geometry_recovered:false,physical_accuracy_verified:false,show_estimated_required:true,required_visible_label:'Observed surface · {observation_timestamp_s}s',required_return_fields:['observation_timestamp_s','source_time_error_s','presentation_mode','interpolated','current_geometry'],samples};
}
function validate(policy,descriptor,reference){if(!same(policy,derive(descriptor,reference)))fail('Editorial source cadence/uncertainty policy differs.');return policy;}
function at(policy,descriptor,time,showEstimated=true){
 if(!showEstimated||!finite(time)||time<0||time>policy.source_duration_s)return null;
 let chosen=null,distance=Infinity;for(const row of policy.samples){const error=Math.abs(row.observation_timestamp_s-time);if(error<distance){distance=error;chosen=row;}}
 if(!chosen||chosen.status!=='source_observed_prior'||time<chosen.display_window_start_s-1e-12||time>chosen.display_window_end_s+1e-12)return null;
 const row=descriptor.samples.find(r=>r.source_sample_index===chosen.source_sample_index);if(row?.status!=='source_observed_prior')fail('Editorial sample points to unknown observation.');
 return {...row,observation_timestamp_s:chosen.observation_timestamp_s,source_time_error_s:distance,requested_video_time_s:time,presentation_mode:'editorial_source_sample_repeat',representation:'source_visible_2.5D_depth_prior',interpolated:false,current_geometry:false,new_tracking_or_geometry_recovered:false};
}
const api={derive,validate,at};if(typeof module!=='undefined'&&module.exports)module.exports=api;root.SourceSurfacePresentationV1=api;
})(typeof window!=='undefined'?window:globalThis);
