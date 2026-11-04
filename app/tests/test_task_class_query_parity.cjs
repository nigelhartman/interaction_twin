const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),{spawnSync}=require('node:child_process');
const ROOT=path.resolve(__dirname,'..'),L=require('../showcase/library.js');
test('typed collections have identical Python agent and browser OR/AND/sort behavior',()=>{
 const source=JSON.parse(fs.readFileSync(path.join(ROOT,'showcase/library-catalog.json'))).records[0];
 const records=[['hinged_assemblies','removable_components'],['sliding_mechanisms'],[]].map((classes,i)=>{
  const r=JSON.parse(JSON.stringify(source));r.id='class-fixture-'+i;r.source.sha256=r.source.original_sha256=String(i+1).repeat(64);r.task_classes=classes;r.title=['Container','Slider','Hinged connector words alone'][i];return r;
 });
 const queries=[{task_classes:['hinged_assemblies']},{task_classes:['sliding_mechanisms']},{task_classes:['HINGED_ASSEMBLIES','sliding_mechanisms']},{task_classes:['removable_components'],parts:[records[0].components[0].id]},{task_classes:['removable_components'],parts:['absent_part']},{task_classes:['connectors_and_insertion']},{task_classes:['hinged_assemblies','sliding_mechanisms'],sort:'title',direction:'desc'}];
 const script="import json,sys;from library_query import derive_index,query_records;d=json.load(sys.stdin);records=d['records'];[r.update(search_index=derive_index(r)) for r in records];print(json.dumps({'records':records,'matches':[[r['id'] for r in query_records(records,**q)['records']] for q in d['queries']]}))";
 const result=spawnSync(path.join(ROOT,'.venv-sam/bin/python'),['-c',script],{cwd:ROOT,input:JSON.stringify({records,queries}),encoding:'utf8'});
 assert.equal(result.status,0,result.stderr);const expected=JSON.parse(result.stdout);
 queries.forEach((q,i)=>assert.deepEqual(L.queryRecords(expected.records,q).map(r=>r.id),expected.matches[i],JSON.stringify(q)));
 assert.equal(expected.matches[0].length,1);assert.equal(expected.matches[2].length,2);assert.equal(expected.matches[5].length,0);
});
