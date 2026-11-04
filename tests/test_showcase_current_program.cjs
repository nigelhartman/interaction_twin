/* Regression: stable recording IDs may now point to newer approved procedures. */
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const H=require('./showcase_app_harness.cjs'),root=path.resolve(__dirname,'..');
const catalog=JSON.parse(fs.readFileSync(path.join(root,'showcase/public/library-catalog.json')));
test('all three landing samples use exact current programs and connected scenes',async()=>{
 const env=H.setup(false,{withAligned:true});await env.initialPromise;
 for(const [key,id]of [['jabra','flow-sam3d-default-e2e-20261102-v1--flow-procedure-1'],['wallet','flow-rigid-parts-wallet-20261031-v1--flow-procedure-1'],['fresh','flow-capture-laptop-20261103-v1--flow-procedure-1']]){
  await env.get('sample-'+key).onclick();await H.flush();
  const record=catalog.records.find(r=>r.id===id),program=JSON.parse(fs.readFileSync(path.join(root,record.interaction.path)));
  assert.equal(env.get('asset-mode').value,'aligned',key);
  assert.equal(env.get('sample-video').src,'/files/'+record.source.path);
  assert.equal(env.get('sample-chapters').children.length,program.steps.length);
  assert.equal(env.get('sample-workspace').href,'library.html?case='+encodeURIComponent(id));
  assert.ok(env.requests.includes('/files/'+record.interaction.path));
  assert.equal(env.legacyScenes.length,0);
 }
});
test('changed current procedure bytes cannot install a replacement sample',async()=>{
 const env=H.setup(false,{withAligned:true,corruptProcedure:true});await env.initialPromise;await H.flush();
 assert.equal(env.get('sample-video').src,undefined);
 assert.equal(env.get('sample-native-frame').src,undefined);
 assert.match(env.get('sample-description').textContent,/could not be verified/);
});
