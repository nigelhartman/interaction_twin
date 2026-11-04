const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const L=require('../showcase/library.js'),ROOT=path.resolve(__dirname,'..');
const saved=JSON.parse(fs.readFileSync(path.join(ROOT,'showcase/library-catalog.json')));
const clone=x=>JSON.parse(JSON.stringify(x));

test('task collections consume explicit typed metadata and never infer capabilities from names or prose',()=>{
 const record=clone(saved.records[0]);delete record.task_classes;
 record.title='hinged assemblies connector insertion flexible sliding mechanism';
 record.tags=L.TASK_CLASSES.map(c=>c.label);
 for(const c of L.TASK_CLASSES)assert.deepEqual(L.queryRecords([record],{task_classes:[c.id]}),[]);
 record.task_classes=['removable_components'];
 assert.deepEqual(L.queryRecords([record],{task_classes:['removable_components']}),[record]);
 assert.deepEqual(L.queryRecords([record],{task_classes:['hinged_assemblies']}),[]);
 assert.deepEqual(L.queryRecords([record],{task_classes:['hinged_assemblies','removable_components']}),[record]);
 assert.deepEqual(L.queryRecords([record],{task_classes:['removable_components'],parts:['not_a_part']}),[]);
});

test('unknown, duplicate or unbounded task classes cannot enter a validated catalog',()=>{
 for(const classes of [['invented_capability'],['removable_components','removable_components'],Array(10).fill('hinged_assemblies')]){
  const catalog=clone(saved);catalog.records[0].task_classes=classes;
  assert.throws(()=>L.validateCatalog(catalog),/source-linked task classes/);
 }
 const catalog=clone(saved);catalog.records[0].task_classes=['removable_components'];
 assert.equal(L.validateCatalog(catalog),catalog);
});

test('public project and library navigation contain no creation or admin entry points',()=>{
 for(const file of ['index.html','library.html']){
  const html=fs.readFileSync(path.join(ROOT,'showcase',file),'utf8');
  assert.doesNotMatch(html,/href=["'][^"']*(?:[?&]new=1|\/admin|workspace\/(?:index|upgrades|geometry-lab|submission)\.html)/i);
  assert.doesNotMatch(html,/>\s*(?:Upload(?:\s+your\s+own)?\s+(?:a\s+)?video|Build from your footage)/i);
  assert.match(html,/src="public\/library-index\.js"/);
  assert.match(html,/https:\/\/github\.com\/nigelhartman/);
  assert.match(html,/https:\/\/www\.linkedin\.com\/in\/nigelhartman\//);
 }
});
