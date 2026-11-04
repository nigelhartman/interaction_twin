const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const S=require('../showcase/object-program-summary-v1.js'),ROOT=path.resolve(__dirname,'..');
const read=p=>JSON.parse(fs.readFileSync(path.join(ROOT,typeof p==='string'?p:p.path),'utf8'));
const catalog=read('showcase/public/library-catalog.json'),index=read('showcase/public/aligned-scenes.json'),clone=x=>JSON.parse(JSON.stringify(x));
function fixture(part){const record=catalog.records.find(r=>r.components.some(c=>c.id===part));assert(record);const row=index.scenes.find(s=>s.source_sha256===record.source.sha256);return{record,aligned:{reference:row.manifest,manifest:read(row.manifest),...(row.derived_assembly?{derived_assembly:{artifact:row.derived_assembly,document:read(row.derived_assembly)}}:{}),...(row.observed_surfaces?{observed_surfaces:{artifact:row.observed_surfaces,document:read(row.observed_surfaces)}}:{})}};}

test('current Jabra description distinguishes refined estimates, hinge coordinates and removable proposals',()=>{
 const f=fixture('case_base'),before=JSON.stringify(f),s=S.summarize(f.record,f.aligned);assert.equal(s.scope,'Refined estimated assembly');assert.equal(s.parts.length,4);assert.equal(new Set(s.parts).size,4,'Same-label earbuds retain distinct component identities');const hinge=s.connections.find(c=>c.kind==='Estimated hinge');assert(hinge);assert.match(hinge.detail,/Source configuration range 0–164°/);assert.match(hinge.detail,/Axis \[-1\.00, 0\.00, 0\.00\] in the joint frame/);assert.match(hinge.detail,/Physical stops are unverified/);assert.equal(s.connections.filter(c=>c.kind==='Removable relationship proposal').length,2);assert.equal(s.step_count,7);assert.deepEqual(s.steps.map(s=>s.index),[1,4,7]);assert.equal(s.evidence.length,3);assert.equal(JSON.stringify(f),before);
});

test('laptop description displays the admitted range without granting physical hinge or button execution',()=>{
 const f=fixture('laptop_base'),s=S.summarize(f.record,f.aligned),hinge=s.connections.find(c=>c.kind==='Estimated hinge');assert(hinge);assert.match(hinge.detail,/0–150°/);assert.match(s.scope_note,/Angles are estimates/);assert.equal(s.step_count,f.record.steps.length);assert(s.steps.every(step=>f.record.steps.some(raw=>raw.title===step.title&&raw.start_s===step.start_s&&raw.end_s===step.end_s)));assert.doesNotMatch(JSON.stringify(s),/calibrated|verified hinge|robot success|simulation ready/i);
});

test('wallet source-visible preview never turns its withheld flexible flap into a numerical hinge',()=>{
 const f=fixture('wallet_holder'),s=S.summarize(f.record,f.aligned);assert.equal(s.scope,'Source-visible surface preview');assert.match(s.scope_note,/Original generated 3D drafts remain separate/);const flap=s.connections.find(c=>c.kind==='Flexible connection');assert(flap);assert.equal(flap.detail,'Fold axis, joint position and angle remain unknown.');assert.doesNotMatch(JSON.stringify(s),/Estimated hinge|range [0-9]|Axis \[/);assert(s.connections.some(c=>c.kind==='Removable relationship proposal'));assert.equal(s.step_count,7);
});

test('phone source surfaces expose the removable proposal without promoting connector seating or native geometry',()=>{
 const f=fixture('phone_blue'),s=S.summarize(f.record,f.aligned);assert.equal(s.scope,'Source-visible surface preview');assert.equal(s.connections.length,1);assert.equal(s.connections[0].kind,'Removable relationship proposal');assert.match(s.connections[0].detail,/contact and seating are unverified/);assert.match(s.scope_note,/hidden geometry and numerical joint parameters are unknown/);assert.doesNotMatch(JSON.stringify(s),/repaired|charges the phone|verified insertion/i);
});

test('unresolved camera opening remains unknown and adapter tracklets are not silently merged',()=>{
 const camera=fixture('camera_body'),s=S.summarize(camera.record,camera.aligned);assert(s.connections.some(c=>c.kind==='Numerical joint unknown'));assert.doesNotMatch(JSON.stringify(s),/range -?[0-9]|Axis \[/);const charger=fixture('adapter_late'),c=S.summarize(charger.record,charger.aligned);assert.equal(c.parts.length,4);assert.doesNotMatch(JSON.stringify(c.connections),/possibly same entity|same adapter|verified insertion|electrical success/i);assert(c.connections.every(row=>/proposal/i.test(row.kind)));
});

test('summary links select only exact admitted citations, preserve zero seconds and do not fabricate source paths',()=>{
 for(const record of catalog.records){const before=JSON.stringify(record),s=S.summarize(record),allowed=[...record.relationships,...record.steps].flatMap(row=>row.citations||[]);assert(s.evidence.length<=3);for(const citation of s.evidence)assert(allowed.some(c=>c.id===citation.id&&c.timestamp_s===citation.timestamp_s&&JSON.stringify(c.frame)===JSON.stringify(citation.frame)));assert.equal(JSON.stringify(record),before);}
 const f=fixture('wallet_holder'),record=clone(f.record);for(const row of [...record.steps,...record.relationships])for(const c of row.citations)c.frame.path='data/../private/key';const s=S.summarize(record);assert.equal(s.evidence.length,0);assert.equal(S.summarize(f.record).evidence[0].timestamp_s,0);
});

test('foreign scene, procedure, refined assembly and observed-surface contexts reject rather than decorating the current twin',()=>{
 const cases=[
  f=>f.aligned.manifest.source.sha256='0'.repeat(64),
  f=>f.aligned.manifest.original_program.bytes++,
  f=>f.aligned.derived_assembly.document.source.sha256='0'.repeat(64),
  f=>f.aligned.derived_assembly.document.scene_ref.bytes++
 ];for(const change of cases){const f=clone(fixture('case_base'));change(f);assert.throws(()=>S.summarize(f.record,f.aligned),/another source|another saved scene/);}
 for(const change of [f=>f.aligned.observed_surfaces.document.source_program.bytes++,f=>f.aligned.observed_surfaces.document.source_scene.sha256='0'.repeat(64),f=>f.aligned.observed_surfaces.document.native_SAM_contract_compatible=true]){const f=clone(fixture('phone_blue'));change(f);assert.throws(()=>S.summarize(f.record,f.aligned),/another saved scene/);}
});

test('inconsistent native fitted angles are configuration parameters, not an opening-angle claim',()=>{
 const f=fixture('camera_body');delete f.aligned.observed_surfaces;const s=S.summarize(f.record,f.aligned);assert(s.connections.some(row=>row.kind==='Estimated configuration'&&/Opening angle is unresolved/.test(row.detail)));assert.doesNotMatch(JSON.stringify(s),/Estimated hinge/);
});
