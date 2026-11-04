/* Actual saved derivative/source lineage through the read-only JS client. */
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),{createHash,webcrypto}=require('node:crypto');
const ROOT=path.resolve(__dirname,'..'),C=require('../showcase/aligned-scene-contract.js'),clientSource=fs.readFileSync(path.join(ROOT,'showcase/aligned-scene-client.js'),'utf8');
const sidecarPath='data/assembly-refinements/jabra-source-refit-20261103-v2/derived-assembly.json';
const raw=name=>fs.readFileSync(path.join(ROOT,name)),json=name=>JSON.parse(raw(name)),copy=x=>JSON.parse(JSON.stringify(x));
const reference=(name,bytes=raw(name))=>({path:name,sha256:createHash('sha256').update(bytes).digest('hex'),bytes:bytes.length});
const d=json(sidecarPath),scene=json(d.scene_ref.path),base=json(d.base_object_program_ref.path),baseScene=json(base.constrained_scene.path),program=json(d.evidence_program.path);
const row={source_sha256:d.source.sha256,manifest:d.scene_ref,object_program:d.base_object_program_ref,derived_assembly:reference(sidecarPath)};
const caseRow={interaction:program,interaction_path:d.evidence_program.path,original_sha256:d.evidence_program.sha256};
const reviews=d.component_assets.map(c=>({component_id:c.component_id,source_sha256:d.source.sha256,native_glb:copy(c.original_native_glb)}));
function context(overrides=new Map(),selection=row){
 const requested=[],index={format:'sam3d-aligned-scene-index',version:1,scenes:[selection]};
 const sandbox={AlignedSceneContract:C,URL,TextDecoder,Uint8Array,crypto:webcrypto,location:{origin:'http://127.0.0.1:8766'},fetch:async url=>{
  const name=new URL(url,'http://127.0.0.1:8766').pathname;requested.push(name);
  if(name==='/showcase/public/aligned-scenes.json')return{ok:true,status:200,text:async()=>JSON.stringify(index)};
  assert.ok(name.startsWith('/files/data/'),'Only exact declared ordinary artifacts may be requested');
  const relative=name.slice('/files/'.length),bytes=overrides.get(relative)||raw(relative);
  return{ok:true,status:200,arrayBuffer:async()=>bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)};
 }};
 vm.runInNewContext(clientSource,sandbox,{filename:'showcase/aligned-scene-client.js'});return{client:sandbox.AlignedScenes,requested};
}
test('verified derived replay retains original SAM parts and separately exposes the exact native program',async()=>{
 const t=context(),saved=await t.client.find(d.source.sha256);
 assert.deepEqual(copy(saved.object_program),row.derived_assembly);assert.deepEqual(copy(saved.base_object_program),row.object_program);
 assert.equal(saved.derived_assembly.document.format,'interaction-twin-declared-derived-assembly');
 assert.equal(C.matchesProgram(saved.manifest,caseRow,reviews),false,'A changed lid cannot pretend to be the unchanged SAM result');
 assert.equal(C.matchesProgram(saved.manifest,caseRow,reviews,saved.derived_assembly),true);
 for(const name of [sidecarPath,d.base_object_program_ref.path,base.constrained_scene.path,d.refinement_report_ref.path,d.source_alignment_audit_ref.path])assert.ok(t.requested.includes('/files/'+name));
 assert.equal(t.requested.some(name=>name.endsWith('.glb')),false,'Scene lookup does not prefetch geometry or generate models');
});
test('unmodified native descriptive lookup remains supported without derivative context',async()=>{
 const nativeRow={source_sha256:d.source.sha256,manifest:base.constrained_scene,object_program:d.base_object_program_ref};
 const saved=await context(new Map(),nativeRow).client.find(d.source.sha256);
 assert.equal(saved.derived_assembly,undefined);assert.equal(saved.base_object_program,undefined);
 assert.deepEqual(copy(saved.object_program),d.base_object_program_ref);assert.equal(C.matchesProgram(saved.manifest,caseRow,reviews),true);
});
test('sidecar byte tampering is rejected even when the original native program is valid',async()=>{
 const bytes=Buffer.from(raw(sidecarPath));bytes[bytes.length-2]^=1;
 await assert.rejects(context(new Map([[sidecarPath,bytes]])).client.find(d.source.sha256),/hash changed/);
});
test('rehashing a foreign-source or ineligible declaration cannot authorize the changed lid',async()=>{
 for(const mutate of [x=>x.source.sha256='f'.repeat(64),x=>x.source_audit_summary.default_eligible=false,x=>x.base_object_program_ref.sha256='f'.repeat(64),x=>x.component_assets.find(c=>c.component_id==='case_lid').geometry_origin='unchanged_SAM_3D_inference',x=>x.joints[0].axis=[0,1,0]]){
  const changed=copy(d);mutate(changed);const bytes=Buffer.from(JSON.stringify(changed)),selection={...row,derived_assembly:reference(sidecarPath,bytes)};
  await assert.rejects(context(new Map([[sidecarPath,bytes]]),selection).client.find(d.source.sha256));
 }
});
test('derived mappings still require every original source-bound review and original model identity',()=>{
 C.validateDerivedAssembly(d,scene,{sceneReference:row.manifest,baseProgramReference:row.object_program,baseProgram:base,baseSceneReference:base.constrained_scene,baseScene,refinementReport:json(d.refinement_report_ref.path),sourceAudit:json(d.source_alignment_audit_ref.path)});const declared={artifact:row.derived_assembly,document:d};assert.equal(C.matchesProgram(scene,caseRow,reviews,declared),true);const alteredScene=copy(scene),pose=alteredScene.components.find(c=>c.poses.some(p=>p.camera_from_native_mesh)).poses.find(p=>p.camera_from_native_mesh);pose.camera_from_native_mesh[0][3]+=.01;assert.equal(C.matchesProgram(alteredScene,caseRow,reviews,declared),false,'A changed manifest cannot reuse an earlier derivative verification');
 for(const mutate of [x=>x.reviews.pop(),x=>x.reviews[0].source_sha256='f'.repeat(64),x=>x.reviews[1].native_glb=copy(d.component_assets[1].native_glb),x=>x.program.interaction.assets[0].sha256='f'.repeat(64)]){
  const x={reviews:copy(reviews),program:copy(caseRow)};mutate(x);assert.equal(C.matchesProgram(scene,x.program,x.reviews,declared),false);
 }
});

test('historical silhouette-only V1 remains a valid history document but cannot authorize a default replay',async()=>{const oldPath='data/assembly-refinements/jabra-source-refit-20261103-v1/derived-assembly.json',old=json(oldPath),oldRow={source_sha256:old.source.sha256,manifest:old.scene_ref,object_program:old.base_object_program_ref,derived_assembly:reference(oldPath)};await assert.rejects(context(new Map(),oldRow).client.find(old.source.sha256),/facing.*ambiguity gate/);assert.equal(C.matchesProgram(json(old.scene_ref.path),caseRow,reviews,{artifact:oldRow.derived_assembly,document:old}),false);});
