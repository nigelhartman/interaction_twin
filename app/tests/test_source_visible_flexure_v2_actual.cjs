/* Actual proof2 transport/timing checks; not WebGL pixels or geometry proof. */
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const C=require('../showcase/source-visible-flexure-v1.js'),P=require('../showcase/source-surface-presentation-v1.js'),ROOT=path.resolve(__dirname,'..');
const cases=[['wallet',5],['phone',3],['camera',3]];
const exact=p=>{const raw=fs.readFileSync(path.join(ROOT,p));return{path:p,sha256:crypto.createHash('sha256').update(raw).digest('hex'),bytes:raw.length};};
function read(r){assert.deepEqual(exact(r.path),{path:r.path,sha256:r.sha256,bytes:r.bytes});return JSON.parse(fs.readFileSync(path.join(ROOT,r.path)));}
function nearest(rows,time){let chosen=null,best=Infinity;for(const row of rows){const delta=Math.abs(row.timestamp_s-time);if(delta<best){chosen=row;best=delta;}}return chosen;}
for(const [name,version]of cases){
 const relative=`data/source-visible-flexure/${name}-20261104-v${version}-1/descriptor.json`,filename=path.join(ROOT,relative);
 test(`actual ${name} proof2 preserves exact source identity, unknown barriers and strict timing`,{skip:!fs.existsSync(filename)},t=>{
  const reference=exact(relative),doc=read(reference),before=JSON.stringify(doc),parent=read(doc.cached_depth_parent),scene=read(doc.source_scene),program=read(doc.source_program);assert.equal(doc.source_footprint_proof_version,2);assert.equal(C.validate(doc,{source_program:scene.original_program,source_scene:doc.source_scene,source:scene.source}),doc);assert.deepEqual(doc.samples.map(r=>[r.source_sample_index,r.timestamp_s,r.status]),parent.samples.map(r=>[r.source_sample_index,r.timestamp_s,r.status]));assert.deepEqual(doc.replacement_component_ids,parent.replacement_component_ids);assert.deepEqual(doc.closed_native_assets,parent.closed_native_assets);assert.deepEqual(doc.camera,parent.camera);assert.equal(doc.samples.length,program.evidence.length);
  let checks=0;const times=new Set([0,doc.source.duration_s]);
  for(const [i,row]of doc.samples.entries()){
   const e=program.evidence[i];assert.equal(row.source_sample_index,e.source_index);assert.equal(row.timestamp_s,e.timestamp_s);
   if(row.status==='source_observed_prior'){assert.equal(row.topology_refined_from_cached_depth,true);assert.notEqual(row.geometry_unchanged_from_footprint_verified_parent,true);assert.deepEqual(row.cached_depth_parent_map,parent.samples[i].source_pixel_map);assert.equal(row.virtual_flexure.status,'unknown');assert.equal(row.virtual_flexure.angle_rad,null);assert.equal(row.physical_accuracy_verified,false);}
   for(const offset of [-.180001,-.18,-.179999,-.000001,0,.000001,.179999,.18,.180001])times.add(row.timestamp_s+offset);
   if(i){const midpoint=(row.timestamp_s+doc.samples[i-1].timestamp_s)/2;for(const offset of [-.000001,0,.000001])times.add(midpoint+offset);}
  }
  for(let time=0;time<=doc.source.duration_s;time+=.01)times.add(time);
  for(const time of times){if(time<0||time>doc.source.duration_s)continue;const expected=nearest(doc.samples,time),visible=expected.status==='source_observed_prior'&&Math.abs(expected.timestamp_s-time)<=.18+1e-12,actual=C.at(doc,time,true);assert.equal(Boolean(actual),visible,`${name} at ${time}`);if(actual){assert.equal(actual.source_sample_index,expected.source_sample_index);assert.equal(actual.interpolated,false);assert.equal(actual.representation,'source_visible_2.5D_depth_prior');assert.ok(actual.source_time_error_s<=.18+1e-12);}assert.equal(C.at(doc,time,false),null);checks++;}
  for(const mutation of ['parent','parentmap','mode','clock','joint','source']){const bad=structuredClone(doc),row=bad.samples.find(r=>r.status==='source_observed_prior');if(mutation==='parent')delete bad.cached_depth_parent;else if(mutation==='parentmap')delete row.cached_depth_parent_map;else if(mutation==='mode')bad.source_footprint_proof_version=3;else if(mutation==='clock')bad.policy.maximum_source_time_error_s=.55;else if(mutation==='joint')row.virtual_flexure.angle_rad=.1;else bad.source.sha256='0'.repeat(64);assert.throws(()=>C.validate(bad,{source_program:scene.original_program,source_scene:doc.source_scene,source:scene.source}));checks++;}
  assert.equal(JSON.stringify(doc),before);t.diagnostic(`${checks} exact identity/timing/tamper checks; no native geometry or physical claim`);
 });
 if(name==='wallet')continue;
 const policyPath=path.join(ROOT,`data/source-visible-presentations/${name}-20261104-v2/presentation.json`);
 test(`actual ${name} revised editorial policy changes only descriptor binding`,{skip:!fs.existsSync(filename)||!fs.existsSync(policyPath)},t=>{
  const reference=exact(relative),doc=read(reference),policy=JSON.parse(fs.readFileSync(policyPath)),old=JSON.parse(fs.readFileSync(path.join(ROOT,`data/source-visible-presentations/${name}-20261104-v1/presentation.json`)));P.validate(policy,doc,reference);const equivalent=structuredClone(policy);equivalent.source_observations=old.source_observations;assert.deepEqual(equivalent,old);assert.equal(policy.display_repeat_cap_s,.55);assert.equal(policy.strict_observation_query_maximum_source_time_error_s,.18);let checks=0;
  const times=new Set();for(const row of policy.samples)for(const v of [row.display_window_start_s,row.observation_timestamp_s,row.display_window_end_s])for(const dt of [-.000001,0,.000001])times.add(v+dt);for(let s=0;s<=doc.source.duration_s;s+=.01)times.add(s);
  for(const time of times){if(time<0||time>doc.source.duration_s)continue;const row=nearest(doc.samples,time),window=policy.samples.find(r=>r.source_sample_index===row.source_sample_index),visible=row.status==='source_observed_prior'&&time>=window.display_window_start_s-1e-12&&time<=window.display_window_end_s+1e-12,result=P.at(policy,doc,time,true);assert.equal(Boolean(result),visible);if(result){assert.equal(result.source_sample_index,row.source_sample_index);assert.equal(result.observation_timestamp_s,row.timestamp_s);assert.equal(result.current_geometry,false);assert.equal(result.interpolated,false);assert.equal(result.new_tracking_or_geometry_recovered,false);assert.ok(result.source_time_error_s<=.55+1e-12);}assert.equal(P.at(policy,doc,time,false),null);checks++;}
  assert.throws(()=>P.validate(old,doc,reference));t.diagnostic(`${checks} editorial boundary/unknown checks; original .18s query unchanged`);
 });
}
