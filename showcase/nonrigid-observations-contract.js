/* A saved source-pixel annotation. Never a rigid pose, tube or interpolator. */
(function(root){'use strict';
const fail=m=>{throw Error(m);},isObject=v=>v&&typeof v==='object'&&!Array.isArray(v),finite=n=>typeof n==='number'&&Number.isFinite(n);
const POLICY={rigid_native_pose_replay:'withheld_for_known_nonrigid_component',native_static_asset_inspection_retained:true,viewer_representation:'source_pixel_polyline_overlay_video_camera_only',temporal_interpolation:'none',maximum_nearest_sample_offset_s:.36,unknown_rows_are_barriers:true,display_between_samples:'nearest_saved_annotation_estimate_not_motion_measurement'};
function same(a,b){if(a===b)return true;if(Array.isArray(a)||Array.isArray(b))return Array.isArray(a)&&Array.isArray(b)&&a.length===b.length&&a.every((v,i)=>same(v,b[i]));if(isObject(a)&&isObject(b)){const x=Object.keys(a).sort(),y=Object.keys(b).sort();return same(x,y)&&x.every(k=>same(a[k],b[k]));}return false;}
function reference(r,max=8*1024*1024){if(!isObject(r)||!same(Object.keys(r).sort(),['bytes','path','sha256'])||typeof r.path!=='string'||!/^data\/[A-Za-z0-9_./-]+$/.test(r.path)||r.path.split('/').some(p=>!p||p.startsWith('.'))||!/^[a-f0-9]{64}$/.test(r.sha256)||!Number.isInteger(r.bytes)||r.bytes<=0||r.bytes>max)fail('Invalid exact nonrigid data reference.');return r;}
function sameRef(a,b){return !!a&&!!b&&a.path===b.path&&a.sha256===b.sha256&&a.bytes===b.bytes;}
function validate(document,trace,program,manifest){
 if(!isObject(document)||document.format!=='interaction-twin-nonrigid-observations'||document.version!==1)fail('Unknown nonrigid observation contract.');
 if(document.depth!=='undefined'||document.endpoint_physical_identity!=='unknown'||document.invisible_spans_completed!==false||document.physical_accuracy_verified!==false||document.simulation_ready!==false||document.source_time_domain!=='source_video_presentation_seconds'||!same(document.display_policy,POLICY))fail('The cable annotation cannot claim depth, completion or interpolated motion.');
 reference(document.evidence_program);reference(document.trace_ref);
 if(document.component_id!=='power_cable'||!sameRef(document.evidence_program,manifest.original_program)||!same(document.source,program.source)||!same(document.source,(({width,height,...source})=>source)(manifest.source))||trace.format!=='interaction-twin-cable-trace-2d'||trace.version!==1||trace.track_id!==document.component_id||!sameRef(trace.program_ref,document.evidence_program)||trace.source_sha256!==program.source.sha256||trace.source_ref.path!==program.source.path||trace.source_ref.sha256!==program.source.sha256||document.original_trace_sha256!==document.trace_ref.sha256||document.fresh_skeleton_recomputation_verified!==true)fail('Nonrigid source/program/trace identity differs.');
 const width=manifest.source.width,height=manifest.source.height;
 if(!Number.isInteger(width)||!Number.isInteger(height)||width<=0||height<=0||manifest.camera.width!==width||manifest.camera.height!==height||!Array.isArray(manifest.camera.K)||manifest.camera.K.length!==3||!manifest.camera.K.every(r=>Array.isArray(r)&&r.length===3&&r.every(finite)))fail('Source-pixel overlay camera dimensions differ.');
 const tracks=program.tracklets?.filter(t=>t.id===document.component_id)||[];
 if(tracks.length!==1||!tracks[0].limitations?.some(t=>String(t).toLowerCase().includes('deform')))fail('An explicit model-described flexible cable is required.');
 const rows=trace.samples,evidence=program.evidence;
 if(!Array.isArray(rows)||rows.length<1||rows.length>90||!Array.isArray(evidence)||rows.length!==evidence.length||document.sample_count!==rows.length||trace.selection?.program_evidence_count!==evidence.length)fail('The whole saved cable/source clock is required.');
 const byIndex=new Map(evidence.map(e=>[e.source_index,e]));
 if(byIndex.size!==evidence.length)fail('Duplicate source sample identity.');
 let previous=-1;
 for(const row of rows){
  const e=byIndex.get(row.source_index);
  if(!e||!finite(row.timestamp_s)||row.timestamp_s!==e.timestamp_s||!(row.timestamp_s>previous)||row.evidence_id!==e.id||row.timestamp_s<0||row.timestamp_s>manifest.source.duration_s)fail('Cable source sample/PTS changed.');
  previous=row.timestamp_s;reference(row.frame_ref,16*1024*1024);
  if(row.frame_ref.path!==e.frame_path||row.frame_ref.sha256!==e.frame_sha256||row.between_samples!=='absent_no_temporal_interpolation_or_endpoint_identity_assignment')fail('Cable RGB identity or temporal scope differs.');
  const observations=e.observations.filter(o=>o.track_id===document.component_id);
  if(observations.length>1)fail('Duplicate cable mask identity.');
  if(row.mask_ref!==null){reference(row.mask_ref,16*1024*1024);const o=observations[0];if(!o||row.mask_ref.path!==o.mask_path||row.mask_ref.sha256!==o.mask_sha256)fail('Cable mask source identity differs.');}
  if(row.trace===null)continue;
  const t=row.trace;
  if(t.coordinate_system!=='original_binary_mask_pixels_top_left_xy'||t.endpoint_identity!=='unassigned_lexicographic_order_not_connector_identity'||!same(t.mask_size_px,[width,height])||!same(row.frame_size_px,[width,height])||!Array.isArray(t.points_px)||t.points_px.length>12000||!t.points_px.every(p=>Array.isArray(p)&&p.length===2&&p.every(finite)&&p[0]>=0&&p[0]<width&&p[1]>=0&&p[1]<height))fail('Cable coordinates are not bounded original source pixels.');
 }
 return {document,trace,width,height,component_id:document.component_id,rows};
}
function at(checked,time,view='source',show=true){
 if(!checked||!finite(time)||view!=='source'||show!==true||time<0||time>checked.document.source.duration_s)return null;
 let nearest=null,distance=Infinity;
 const known=r=>r?.status==='observed_2d_path_candidate'&&r.trace?.status==='observed_2d_path_candidate'&&r.trace.points_px.length>=2;
 for(const row of checked.rows){const delta=Math.abs(row.timestamp_s-time);if(delta<distance-1e-12||(Math.abs(delta-distance)<=1e-12&&!known(row))){nearest=row;distance=delta;}}
 if(!nearest||distance>.36+1e-12||!known(nearest))return null;
 return {component_id:checked.component_id,source_index:nearest.source_index,evidence_id:nearest.evidence_id,timestamp_s:nearest.timestamp_s,offset_s:distance,points_px:nearest.trace.points_px,width:checked.width,height:checked.height,depth:'undefined',interpolated:false,scope:'nearest_saved_annotation_estimate_not_motion_measurement'};
}
const api={validate,at,sameRef,reference};root.NonrigidObservationsV1=api;if(typeof module!=='undefined')module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
