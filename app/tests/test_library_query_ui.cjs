const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),{spawnSync}=require('node:child_process');
const ROOT=path.resolve(__dirname,'..'),L=require('../showcase/library.js');
const catalog=JSON.parse(fs.readFileSync(path.join(ROOT,'showcase/library-catalog.json'))),records=catalog.records;
const cases=[{},{query:'earbud'},{query:'insert'},{parts:['case_base']},{backends:['Meta SAM 3D Objects']},{backends:['sam3d'],relations:['removable_from']},{parts:['case_lid','gold_coin']},{relations:['not_a_relation']},{readiness:['semantic_only']},...['title','id','duration','components','assets','steps','version'].flatMap(sort=>['asc','desc'].map(direction=>({sort,direction})))];
test('human filters and sorting match the shared agent contract on actual saved programs',()=>{
 const code="import json,sys;from library_query import query_records;d=json.load(sys.stdin);print(json.dumps([[r['id'] for r in query_records(d['records'],**q)['records']] for q in d['queries']]))";
 const python=spawnSync(path.join(ROOT,'.venv-sam/bin/python'),['-c',code],{cwd:ROOT,input:JSON.stringify({records,queries:cases}),encoding:'utf8'});assert.equal(python.status,0,python.stderr);const expected=JSON.parse(python.stdout);
 cases.forEach((q,i)=>assert.deepEqual(L.queryRecords(records,q).map(r=>r.id),expected[i],JSON.stringify(q)));
});
test('a descriptive program stays valid while geometry is pending',()=>{
 const d=JSON.parse(JSON.stringify(catalog));for(const r of d.records){r.components.forEach(c=>c.asset=null);r.counts.assets=0;r.scene_case.scene_ref=null;r.sam3d_reviews=[];}d.summary.native_asset_drafts=0;assert.equal(L.validateCatalog(d),d);
});
test('curated display names and original model titles stay searchable for humans and agents',()=>{
 const record=JSON.parse(JSON.stringify(records[0]));record.title='Original model title';record.public_title='USB-C charger + cable';record.public_note='Review-only editorial note';record.task_classes=[];
 const queries=[{query:'USB-C charger'},{query:'original model title'},{query:'review-only editorial'},{task_classes:['connectors_and_insertion']}];
 const script="import json,sys;from library_query import derive_index,query_records;d=json.load(sys.stdin);r=d['record'];r['search_index']=derive_index(r);print(json.dumps({'record':r,'matches':[[x['id'] for x in query_records([r],**q)['records']] for q in d['queries']]}))";
 const result=spawnSync(path.join(ROOT,'.venv-sam/bin/python'),['-c',script],{cwd:ROOT,input:JSON.stringify({record,queries}),encoding:'utf8'});assert.equal(result.status,0,result.stderr);const expected=JSON.parse(result.stdout);
 queries.forEach((q,i)=>assert.deepEqual(L.queryRecords([expected.record],q).map(r=>r.id),expected.matches[i]));
 assert.deepEqual(expected.matches,[[record.id],[record.id],[],[]]);assert.equal(expected.record.title,'Original model title');
});
test('name sorting uses the displayed title while preserving original model titles',()=>{
 const rows=['Phone + USB-C connector','USB-C charger + cable','Laptop + charging cable'].map((title,i)=>{
  const r=JSON.parse(JSON.stringify(records[0]));r.id='display-sort-'+i;r.source.sha256=r.source.original_sha256=String(i+1).repeat(64);r.title=['AAA original','BBB original','ZZZ original'][i];r.public_title=title;return r;
 });
 const queries=[{sort:'title',direction:'asc'},{sort:'title',direction:'desc'}],script="import json,sys;from library_query import query_records;d=json.load(sys.stdin);print(json.dumps([[r['id'] for r in query_records(d['records'],**q)['records']] for q in d['queries']]))";
 const result=spawnSync(path.join(ROOT,'.venv-sam/bin/python'),['-c',script],{cwd:ROOT,input:JSON.stringify({records:rows,queries}),encoding:'utf8'});assert.equal(result.status,0,result.stderr);const expected=JSON.parse(result.stdout);
 queries.forEach((q,i)=>assert.deepEqual(L.queryRecords(rows,q).map(r=>r.id),expected[i]));
 assert.deepEqual(expected,[['display-sort-2','display-sort-0','display-sort-1'],['display-sort-1','display-sort-0','display-sort-2']]);assert.equal(rows[0].title,'AAA original');
});
