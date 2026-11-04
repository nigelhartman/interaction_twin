const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const L=require('../showcase/library.js');
const ROOT=path.resolve(__dirname,'..'),catalog=JSON.parse(fs.readFileSync(path.join(ROOT,'showcase/library-catalog.json')));
catalog.records.sort((a,b)=>Number(b.components.some(c=>c.id==='earbud_a'))-Number(a.components.some(c=>c.id==='earbud_a')));
const copy=v=>JSON.parse(JSON.stringify(v));
test('initial SAM3D preparation cannot become an asset, submission or another source',()=>{
 const base=copy(catalog),r=base.records[0];r.initial_reconstruction_inputs=[{component_id:r.components[0].id,handoff:r.interaction,source:{video_sha256:r.source.sha256,timestamp_s:1,time_domain:'source_video_presentation_seconds'},status:'initial_reconstruction_prepared',baseline_required:false,no_inference:true,remote_job_submitted:false,simulation_ready:false}];
 assert.equal(L.validateCatalog(base),base);
 for(const mutate of [i=>i.source.video_sha256='b'.repeat(64),i=>i.component_id='other',i=>i.remote_job_submitted=true,i=>i.no_inference=false,i=>i.simulation_ready=true,i=>i.baseline_required=true,i=>i.source.timestamp_s=Infinity]){const d=copy(base);mutate(d.records[0].initial_reconstruction_inputs[0]);assert.throws(()=>L.validateCatalog(d));}
});
test('actual catalog has source families rather than experimental-variant counts',()=>{
 assert.equal(L.validateCatalog(catalog),catalog);
 assert.equal(catalog.summary.recordings,new Set(catalog.records.map(r=>r.source.original_sha256)).size);
 assert.ok(catalog.summary.eligible_saved_versions>catalog.summary.recordings);
});
for(const property of ['physical_accuracy_verified','simulation_ready'])test('reject promoted '+property,()=>{
 const d=copy(catalog);d.records[0][property]=true;assert.throws(()=>L.validateCatalog(d));
});
test('reject duplicate original recordings, forged counts and mixed program/scene',()=>{
 for(const mutate of [d=>d.records[1].source.original_sha256=d.records[0].source.original_sha256,d=>d.summary.native_asset_drafts++,d=>d.records[0].scene_case.original_sha256='0'.repeat(64)]){
  const d=copy(catalog);mutate(d);assert.throws(()=>L.validateCatalog(d));
 }
});
test('reject fixture-like manual model provenance or copied old pose declarations',()=>{
 for(const key of ['manual_geometry_used','manual_pose_used','source_video_pose_established_by_this_library']){
  const d=copy(catalog);d.records[0].components.find(c=>c.asset).asset[key]=true;assert.throws(()=>L.validateCatalog(d));
 }
});
test('all artifact paths are local data records without traversal/query/fragment',()=>{
 const r=catalog.records[0].thumbnail;
 for(const p of ['https://elsewhere/a.png','data/../secret.png','data/a.png?x=1','data/a.png#fragment','data/%2e%2e/x','data/.hidden/x','data/a\\b'])assert.throws(()=>L.ref({...r,path:p}));
 assert.equal(L.ref(r),r);
});
test('reject semantic references to absent components and invalid source times',()=>{
 for(const mutate of [d=>d.records[0].relationships[0].subject_track_id='missing',d=>d.records[0].steps[0].citations[0].timestamp_s=100000]){
  const d=copy(catalog);mutate(d);assert.throws(()=>L.validateCatalog(d));
 }
});
test('search works across components and actions, with real empty results',()=>{
 assert.ok(L.matches(catalog.records[0],'earbud remove'));
 assert.equal(catalog.records.filter(r=>L.matches(r,'totally-absent-part')).length,0);
 assert.equal(L.matches(catalog.records[0],'earbud','Missing category'),false);
});
// Preserve the real legacy scene as a frozen regression fixture. The current
// SAM 3D library intentionally has no aggregate scene or inherited old poses.
const legacySHA='50842ba93eda2b3265966f080db09408980d3e53eb95bc9aba302bea31cd9e50';
const legacyBytes=fs.readFileSync(path.join(ROOT,'showcase',`library-catalog-${legacySHA}.json`));
assert.equal(crypto.createHash('sha256').update(legacyBytes).digest('hex'),legacySHA);
const legacy=JSON.parse(legacyBytes),r=legacy.records.find(r=>r.components.some(c=>c.id==='earbud_a')&&r.scene_case.scene_ref);
assert.ok(r,'Frozen legacy Jabra scene is required for motion-stripping regressions');
const sceneBytes=fs.readFileSync(path.join(ROOT,'workspace',r.scene_case.scene_ref.path));
assert.equal(crypto.createHash('sha256').update(sceneBytes).digest('hex'),r.scene_case.scene_ref.sha256);
const pack=JSON.parse(sceneBytes);
test('current unaligned SAM 3D library preserves real assets without fabricating an aggregate scene',()=>{
 const current=catalog.records.find(r=>r.components.some(c=>c.asset?.backend==='Meta SAM 3D Objects'));
 assert.ok(current,'Actual SAM 3D upload must remain present in the current catalog');
 assert.equal(L.validateCatalog(catalog),catalog);
 assert.equal(current.scene_case.scene_ref,null);
 const assets=current.components.filter(c=>c.asset);assert.ok(assets.length>0);
 assert.ok(assets.every(c=>c.asset.backend==='Meta SAM 3D Objects'));
 assert.deepEqual(current.search_index.readiness,['geometry_draft']);
 assert.equal(current.search_index.native_supported_motion_samples,0);
 assert.equal(current.source_video_alignment_added,false);
 // The older scene is not a fallback for a fresh native mesh, even though the
 // recording and component IDs can be the same.
 assert.throws(()=>L.nativeScene(current,pack.scene,assets[0].id));
});
test('native inspection strips all motion and source projection without changing source scene',()=>{
 const before=JSON.stringify(pack.scene),c=r.components.find(c=>c.asset),native=L.nativeScene(r,pack.scene,c.id);
 assert.equal(native.native_only,true);assert.equal(native.objects.length,1);assert.equal(native.objects[0].object_id,c.id);
 assert.deepEqual(native.objects[0].frames,[]);assert.equal(native.objects[0].source_projection_rows,null);
 assert.equal(native.source_projection_rows,null);assert.equal(native.physical_assembly_verified,false);
 assert.equal(JSON.stringify(pack.scene),before);
});
test('assetless part returns null rather than another component model',()=>{
 const c=r.components.find(c=>!c.asset);assert.ok(c);assert.equal(L.nativeScene(r,pack.scene,c.id),null);
});
test('native geometry with another mesh identity or missing part is rejected',()=>{
 const s=copy(pack.scene);s.objects[0].provenance.mesh_sha256='0'.repeat(64);assert.throws(()=>L.nativeScene(r,s,r.components[0].id));
 const missing=copy(pack.scene);missing.objects.pop();assert.throws(()=>L.nativeScene(r,missing,r.components[0].id));
});
test('prepared queue cannot claim submission/inference and saved analysis cannot become commands',()=>{
 // Optional preparation sidecars may correctly disappear when their source
 // contract becomes stale. Keep this controller boundary independent of which
 // real recording currently has a prepared queue or a cached analysis.
 const base=copy(catalog),position=Math.max(0,base.records.findIndex(r=>r.offline_queue)),record=base.records[position];
 const fixtureRef=name=>({path:`data/controller-unit-fixtures/${name}`,sha256:'d'.repeat(64),bytes:128});
 record.offline_queue=record.offline_queue||{
  queue:fixtureRef('queue.json'),plan:fixtureRef('plan.json'),prepared_bundles:[{
   track_id:record.components[0].id,manifest:fixtureRef('job.json'),zip:fixtureRef('job.zip')
  }],unavailable_components:[],remote_submitted:false,inference_executed:false,existing_poses_reused:false
 };
 const savedAnalysis=record.cached_insights[0],requestKey='a'.repeat(64);
 record.cached_insights=[savedAnalysis?copy(savedAnalysis):{analysis:{path:`data/task-intelligence/${requestKey}/analysis.json`,sha256:'b'.repeat(64),bytes:128},request_key:requestKey,question:'Controller unit fixture',model:'muse-spark-1.3',human_reviewed:false,robot_commands:false}];
 assert.equal(L.validateCatalog(base),base);
 for(const key of ['remote_submitted','inference_executed','existing_poses_reused']){const d=copy(base);d.records[position].offline_queue[key]=true;assert.throws(()=>L.validateCatalog(d));}
 const wrongPart=copy(base);wrongPart.records[position].offline_queue.prepared_bundles.push({track_id:'absent_component',manifest:fixtureRef('job.json'),zip:fixtureRef('job.zip')});assert.throws(()=>L.validateCatalog(wrongPart));
 for(const key of ['robot_commands','human_reviewed']){const e=copy(base);e.records[position].cached_insights[0][key]=true;assert.throws(()=>L.validateCatalog(e));}
});
test('pending or failed selection removes the prior asset download and source href',()=>{
 const links=['/files/data/old/asset.glb','../workspace/index.html?case=old'].map(href=>({href,hidden:false,removeAttribute(key){delete this[key];}}));
 L.clearAssetLinks(links);for(const link of links){assert.equal(link.hidden,true);assert.equal(link.href,undefined);}
});
