/* Resolve rehearsal geometry without modifying the original evidence program. */
(function(root){'use strict';
const clone=value=>JSON.parse(JSON.stringify(value));
function forProgram(catalog,program,reviews=root.Sam3DModelReviews){
 if(!catalog||!Array.isArray(catalog.records)||!program?.source||!Array.isArray(program.tracklets)||!reviews?.forRecording)return[];
 const ids=new Set(program.tracklets.map(t=>t.id)),matches=catalog.records.filter(r=>r.source?.sha256===program.source.sha256);
 if(matches.length>1)throw Error('Several library records claim this exact source. No rehearsal geometry is substituted.');
 const record=matches[0];if(!record)return[];
 return reviews.forRecording(record).filter(row=>ids.has(row.component_id)&&row.backend==='Meta SAM 3D Objects'&&row.source_video_alignment_verified===false&&row.simulation_ready===false&&reviews.localRef(row.review_manifest)&&Number.isSafeInteger(row.source_sample_index)&&row.source_sample_index>=0&&Number.isFinite(row.source_timestamp_s)&&row.source_timestamp_s>=0).map(row=>{
  const selected=(program.assets||[]).some(asset=>asset.track_id===row.component_id&&asset.sha256===row.native_glb.sha256&&asset.backend==='Meta SAM 3D Objects');
  return{
  id:'sam3d_'+row.component_id,track_id:row.component_id,path:row.native_glb.path,sha256:row.native_glb.sha256,bytes:row.native_glb.bytes,
  backend:'Meta SAM 3D Objects',geometry_status:selected?'selected_native_geometry_pending_alignment':'unselected_native_candidate_pending_alignment',selected,
  physical_scale_known:false,physical_accuracy_verified:false,source_video_alignment_verified:false,simulation_ready:false,
  source_sha256:row.source_sha256,source_anchor:{sample_index:row.source_sample_index,timestamp_s:row.source_timestamp_s,time_domain:'source_video_presentation_seconds'},
  source_image:clone(row.source_image),source_mask:clone(row.source_mask),review_page:clone(row.review_page),
  review_manifest:clone(row.review_manifest),catalog_record_id:record.id,
  selection_scope:selected?'selected_in_current_evidence_program':'rehearsal_geometry_only_original_program_unchanged',
  selection_origin:'exact_current_program_component_and_asset_sha256_binding',pose_origin:'authored_staging_not_recovered_video_pose'
 };});
}
function defaultAssets(assets,tracklets){
 const ordered=[...assets.filter(a=>a.selected===true),...assets.filter(a=>a.selected!==true)];
 const label=a=>(tracklets.find(t=>t.id===a.track_id)?.label||'')+' '+a.track_id;
 const moving=ordered.find(a=>/earbud|coin|batter/i.test(label(a))&&!/case|door|flap|body|pouch/i.test(label(a)))||ordered[0]||null;
 const context=ordered.find(a=>a!==moving&&/case|wallet|camera.*body|container|base/i.test(label(a))&&!/earbud_a|earbud_b|batter|flap|lid|door/i.test(a.track_id))||ordered.find(a=>a!==moving)||null;
 return{moving,context};
}
const api=Object.freeze({forProgram,defaultAssets});root.RobotRehearsalAssets=api;if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(globalThis);
