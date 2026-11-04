'use strict';
const assert=require('node:assert/strict');const fs=require('node:fs');const path=require('node:path');const vm=require('node:vm');const test=require('node:test');
const root=path.resolve(__dirname,'..');const directory=path.join(root,'showcase/demos/charge-phone-v1');
const files=Object.fromEntries(['task.json','trace.json','summary.json','muse-review.json'].map(name=>[name,JSON.parse(fs.readFileSync(path.join(directory,name),'utf8'))]));
function setup(overrides={}){
  const calls=[],nodes=new Map();function element(tag){return{tag,children:[],dataset:{},textContent:'',append(...items){this.children.push(...items);}};}
  function get(id){if(!nodes.has(id))nodes.set(id,element('section'));return nodes.get(id);}
  const context={document:{getElementById:get,createElement:element,createTextNode:text=>({textContent:text})},fetch:async(url,options)=>{calls.push([url,options]);assert.ok(Object.hasOwn(files,url));return{ok:true,json:async()=>overrides[url]??files[url]};}};
  vm.runInNewContext(fs.readFileSync(path.join(directory,'demo.js'),'utf8'),context);return{nodes,get,calls};
}
test('saved workflow renders source-bound candidates and traces without API/model requests',async()=>{
  const r=setup();await new Promise(resolve=>setImmediate(resolve));
  assert.equal(r.get('demo-status').textContent,'');assert.equal(r.get('source-recordings').children.length,2);
  assert.equal(r.get('connection-candidates').children.length,3);assert.equal(r.get('task-steps').children.length,5);
  assert.equal(r.get('trace-calls').children.length,files['trace.json'].calls.length);
  assert.equal(r.get('context-attribution').children.length,2);assert.equal(r.calls.length,4);
  assert.ok(r.calls.every(([url,options])=>!url.startsWith('/')&&options.credentials==='omit'));
  const candidates=r.get('connection-candidates').children;assert.equal(candidates.filter(x=>x.dataset.cross==='true').length,1);
  assert.match(r.get('trace-description').textContent,/hash-checked/);
});
test('unexpected saved contract cannot silently become live execution',async()=>{
  const bad=JSON.parse(JSON.stringify(files['task.json']));bad.claims.robot_execution_available=true;
  const r=setup({'task.json':bad});await new Promise(resolve=>setImmediate(resolve));
  assert.match(r.get('demo-status').textContent,/unavailable/);assert.equal(r.get('source-recordings').children.length,0);
});
test('unknown external evidence URL is never inserted as a page link',async()=>{
  const bad=JSON.parse(JSON.stringify(files['task.json']));bad.sources[1].interfaces.find(i=>i.id==='cable_free_end').evidence[0].video_url='javascript:bad';
  const r=setup({'task.json':bad});await new Promise(resolve=>setImmediate(resolve));
  assert.match(r.get('demo-status').textContent,/unavailable/);
});
test('actual trace contains both recordings and verified bytes with matching saved hashes',()=>{
  const crypto=require('node:crypto');const digest=name=>crypto.createHash('sha256').update(fs.readFileSync(path.join(directory,name))).digest('hex');
  assert.equal(digest('task.json'),files['summary.json'].task_sha256);assert.equal(digest('trace.json'),files['summary.json'].trace_sha256);
  const trace=files['trace.json'];assert.equal(trace.transport,'actual_official_sdk_subprocess_stdio');assert.equal(trace.workflow,'deterministic_scripted_client_not_live_llm');
  assert.equal(trace.native_asset_bytes_verified.length,6);assert.equal(trace.source_evidence_bytes_verified.length,9);
  assert.equal(new Set(trace.native_asset_bytes_verified.map(r=>r.twin_id)).size,2);
  assert.equal(files['muse-review.json'].review_scope,'text_metadata_only_no_images_or_meshes');
  assert.equal(files['muse-review.json'].task_file_sha256,digest('task.json'));assert.equal(files['muse-review.json'].trace_file_sha256,digest('trace.json'));
});
