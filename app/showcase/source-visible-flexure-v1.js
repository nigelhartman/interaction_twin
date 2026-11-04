/* Exact source-frame 2.5D depth priors. Never rigid joints or interpolated mesh. */
(function(root){'use strict';
const fail=m=>{throw Error(m);},finite=x=>typeof x==='number'&&Number.isFinite(x),same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
function ref(r){if(!r||typeof r.path!=='string'||!/^data\/[A-Za-z0-9_./-]+$/.test(r.path)||r.path.split('/').some(p=>!p||p.startsWith('.'))||!/^[a-f0-9]{64}$/.test(r.sha256)||!Number.isSafeInteger(r.bytes)||r.bytes<=0||r.bytes>64*1024*1024)fail('Invalid exact source-surface data reference.');return r;}
function sameRef(a,b){return a&&b&&a.path===b.path&&a.sha256===b.sha256&&a.bytes===b.bytes;}
function validate(doc,manifest){
 if(doc?.format!=='interaction-twin-source-visible-flexure-series'||doc.version!==1||doc.kind!=='source_observed_monocular_depth_prior'||doc.physical_accuracy_verified!==false||doc.simulation_ready!==false||doc.native_SAM_contract_compatible!==false||doc.manual_geometry_or_poses_used!==false)fail('Unknown or overstated source-surface contract.');
 ref(doc.source_program);ref(doc.source_scene);ref(doc.protocol);
 const proof=doc.source_footprint_proof_version??1;if(![1,2].includes(proof))fail('Unknown source-footprint proof mode.');if(proof===2)ref(doc.cached_depth_parent);
 if(manifest){const program=manifest.source_program||manifest.original_program,scene=manifest.source_scene||manifest.manifest;if(program&&!sameRef(doc.source_program,program))fail('Source-surface original program differs.');if(scene&&!sameRef(doc.source_scene,scene))fail('Source-surface original scene differs.');if(manifest.source?.sha256&&manifest.source.sha256!==doc.source.sha256)fail('Source-surface video differs.');}
 if(doc.policy?.maximum_source_time_error_s!==.18||doc.policy?.timing!=='nearest_source_keyframe_within_0.18s_only_no_geometric_interpolation'||doc.policy?.hidden_surfaces!=='absent_not_inpainted'||doc.policy?.numerical_fold_parameters!=='unknown_not_identifiably_recovered')fail('Source-surface timing or unknown policy differs.');
 if(!Array.isArray(doc.samples)||!doc.samples.length||doc.samples.length>20000||!Array.isArray(doc.replacement_component_ids)||!doc.replacement_component_ids.length)fail('Missing source-surface ledger.');
 let previous=-1;const seen=new Set();for(const row of doc.samples){if(!finite(row.timestamp_s)||row.timestamp_s<=previous||!Number.isSafeInteger(row.source_sample_index)||seen.has(row.source_sample_index)||!['unknown','source_observed_prior'].includes(row.status))fail('Invalid source-surface time ledger.');previous=row.timestamp_s;seen.add(row.source_sample_index);
  if(row.status==='unknown'){if(row.asset)fail('Unknown surface exposes geometry.');continue;}
  ref(row.asset);ref(row.source_RGB);ref(row.source_pixel_map);ref(row.provenance);row.source_masks.forEach(ref);
  if((row.source_footprint_proof_version??1)!==proof)fail('Source surface proof mode differs from its descriptor.');if(proof===2)ref(row.cached_depth_parent_map);
  if(row.physical_accuracy_verified!==false||row.source_visible_only!==true||row.triangle_source_footprint_verified!==true||row.vertices>1800||row.hidden_backs_culled!==true||row.unlit_exact_vertex_RGB!==true)fail('Source surface lost its evidence fence.');
  const f=row.virtual_flexure;if(f?.status!=='unknown'||f.angle_rad!==null||f.origin_camera!==null||f.axis_camera!==null)fail('Source surface cannot assert a recovered material joint.');
  if(!same(row.camera_from_asset,[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))fail('Source-camera surface placement differs.');
 }
 return doc;
}
function at(doc,time,showEstimated=true){
 if(!showEstimated||!finite(time)||time<0||time>doc.source.duration_s)return null;
 const rows=doc.samples;let lo=0,hi=rows.length;while(lo<hi){const m=(lo+hi)>>1;if(rows[m].timestamp_s<time)lo=m+1;else hi=m;}
 const options=[rows[lo-1],rows[lo]].filter(Boolean);options.sort((a,b)=>Math.abs(a.timestamp_s-time)-Math.abs(b.timestamp_s-time)||a.timestamp_s-b.timestamp_s);const row=options[0];
 // Nearest means nearest ledger row, including unknowns. Never search past an
 // unknown to a farther known mesh, and never interpolate across that barrier.
 if(!row||row.status!=='source_observed_prior'||Math.abs(row.timestamp_s-time)>.18+1e-12)return null;
 return {...row,source_time_error_s:Math.abs(row.timestamp_s-time),interpolated:false,representation:'source_visible_2.5D_depth_prior'};
}
const api={validate,at};if(typeof module!=='undefined'&&module.exports)module.exports=api;root.SourceVisibleFlexure=api;
})(typeof window!=='undefined'?window:globalThis);
