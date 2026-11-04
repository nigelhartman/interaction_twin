const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),{createHash}=require('node:crypto');
const R=require('../showcase/sam3d-models.js'),ROOT=path.resolve(__dirname,'..');
const catalog=JSON.parse(fs.readFileSync(path.join(ROOT,'showcase/library-catalog.json')));
const copy=value=>JSON.parse(JSON.stringify(value));

test('each recording exposes only its genuinely generated SAM3D components',()=>{
 const matches=catalog.records.map(r=>R.forRecording(r).map(v=>v.component_id));
 assert.ok(matches.flat().length>=5);for(let i=0;i<matches.length;i++){const ids=new Set(catalog.records[i].components.map(c=>c.id));assert.ok(matches[i].every(id=>ids.has(id)));}
});
test('recording-source hash and exact component identity are both necessary',()=>{
 const r=catalog.records.find(r=>r.components.some(c=>c.id==='earbud_a'));
 assert.notEqual(r.source.original_sha256,r.source.sha256);
 assert.deepEqual(R.forRecording({...r,source:{...r.source,sha256:r.source.original_sha256}}),[]);
 assert.deepEqual(R.forRecording({...r,source:{...r.source,sha256:'0'.repeat(64)}}),[]);
 assert.deepEqual(R.forRecording({...r,components:[]}),[]);
 assert.equal(R.forComponent(r,'not_a_component'),null);
 assert.equal(R.forComponent(catalog.records.find(r=>r.components.some(c=>c.id==='pouch_flap')),'pouch_flap'),null);
});
test('lookup does not change original scene selection, counts or catalog bytes',()=>{
 const d=copy(catalog),before=JSON.stringify(d);
 for(const r of d.records)for(const c of r.components)R.forComponent(r,c.id);
 assert.equal(JSON.stringify(d),before);
 for(const r of catalog.records.flatMap(r=>R.forRecording(r))){assert.equal(r.selected,false);assert.equal(r.physical_accuracy_verified,false);assert.ok(r.native_glb&&r.source_mask);}
});
test('all generated links and source images match real frozen review bytes and source IDs',()=>{
 for(const r of catalog.records.flatMap(r=>R.forRecording(r))){
  for(const ref of[r.review_page,r.source_image]){
   const p=path.resolve(ROOT,ref.path);assert.ok(p.startsWith(ROOT+path.sep));
   const raw=fs.readFileSync(p);assert.equal(raw.length,ref.bytes);assert.equal(createHash('sha256').update(raw).digest('hex'),ref.sha256);
  }
  const m=JSON.parse(fs.readFileSync(path.join(ROOT,path.dirname(r.review_page.path),'review-manifest.json')));
  assert.equal(m.component_id||m.binding.track_id,r.component_id);
  assert.equal((m.source||m.binding).video_sha256,r.source_sha256);
  assert.equal(m.fixture,false);assert.equal(m.scope.physical_accuracy_verified,false);
 }
});
