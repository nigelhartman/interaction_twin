/* Actual Workspace controller, synthetic source/DOM only; no model/render claims. */
const assert=require('node:assert/strict');
const {setup,flush}=require('./lazy_selection_harness.cjs');
function make(search=''){
 return setup({search,configure({cases,scene}){
  cases[0].interaction.tracklets.push({id:'lid',label:'lid',limitations:[]},{id:'two_d',label:'2D-only component',limitations:[]});
  scene.objects.push({object_id:'lid',label:'lid',frames:[]});
 }});
}
async function ready(e){e.requests[0].resolve();await flush();return e}
async function run(){
 const e=await ready(make('?case=fixture-a&view=replay&t=1.250000&track=lid'));
 assert.equal(e.sandbox.TwinWorkspace.renderer.selected,'lid');
 assert.equal(e.get('asset-select').value,'lid');
 assert.equal(new URLSearchParams(e.sandbox.location.search).get('track'),'lid');
 assert.equal(e.video().currentTime,1.25);
 e.sandbox.TwinWorkspace.selectObject('two_d');
 assert.equal(e.canvas().style.visibility,'hidden');assert.equal(e.get('scene-error').hidden,false);
 assert.match(e.get('scene-error').textContent,/No generated 3D asset for two_d/);
 assert.equal(e.get('asset-select').value,'');assert.equal(e.get('camera').disabled,true);
 assert.equal(e.get('scene-label').textContent,'2D evidence only');
 const saved=e.sandbox.location.search;
 e.select(1);e.requests[1].reject(Error('Unavailable synthetic source'));await flush();
 assert.equal(e.sandbox.TwinWorkspace.active.id,'fixture-a');
 assert.equal(e.canvas().style.visibility,'hidden');
 assert.match(e.get('scene-error').textContent,/two_d/);
 const restored=await ready(make(saved));
 assert.equal(restored.canvas().style.visibility,'hidden');
 assert.match(restored.get('scene-error').textContent,/two_d/);
 restored.get('asset-select').value='lid';restored.get('asset-select').onchange();
 assert.equal(restored.canvas().style.visibility,'');assert.equal(restored.get('scene-error').hidden,true);
 assert.equal(restored.sandbox.TwinWorkspace.renderer.selected,'lid');
 assert.equal(new URLSearchParams(restored.sandbox.location.search).get('track'),'lid');
 assert.equal(restored.sandbox.TwinWorkspace.selectObject('../unknown'),false);
 assert.equal(restored.sandbox.TwinWorkspace.renderer.selected,'lid');
 restored.video().paused=false;restored.select(1);await flush();
 assert.equal(restored.panel().inert,true);
 assert.equal(restored.canvas().style.visibility,'hidden','Queued pause must not reveal the old source canvas');
 restored.video().emit('timeupdate');
 assert.equal(restored.canvas().style.visibility,'hidden','Pending source load must keep the old canvas hidden on timeupdate');
 restored.requests[1].resolve();await flush();
 assert.equal(restored.sandbox.TwinWorkspace.active.id,'fixture-b');
 assert.equal(restored.sandbox.TwinWorkspace.renderer.selected,'part');
 assert.equal(new URLSearchParams(restored.sandbox.location.search).get('track'),'part');
 const invalid=await ready(make('?case=fixture-a&track=../missing'));
 assert.equal(invalid.sandbox.TwinWorkspace.renderer.selected,'part');
 const cross=await ready(make('?case=fixture-b&track=lid'));
 assert.equal(cross.sandbox.TwinWorkspace.renderer.selected,'part');
 assert.ok([...e.network,...restored.network,...invalid.network,...cross.network].every(r=>!r.options.method));
 console.log(JSON.stringify({status:'passed',actual_controller:true,synthetic_DOM_only:true,model_calls:0,checks:'registered permalink reload;2D-only blank;asset recovery;case binding;invalid ID fail closed'}));
}
run().catch(error=>{console.error(error);process.exitCode=1});
