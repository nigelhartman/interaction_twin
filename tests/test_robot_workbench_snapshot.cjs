/* Frozen-preview local instruction branch; no browser renderer or inference claims. */
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../showcase/robot-workbench.js'),'utf8');
test('frozen snapshot compiles an instruction locally without a live server planner',async()=>{
 const helper=source.match(/^async function runInstruction\(\)\{[^\n]+\}/m)?.[0];assert.ok(helper);
 for(const [bundle,snapshotMode]of[[false,true],[true,false]]){
  const calls=[],nodes={'task-prompt':{value:'Open the lid.'},'download-kit':{}},baseline={fixture:'baseline'},kit={fixture:'kit'},context={bundle,snapshotMode,baseline,kit,$:id=>nodes[id],P:{compileInstruction:(text,b,k)=>{assert.equal(text,'Open the lid.');assert.equal(b,baseline);assert.equal(k,kit);calls.push('local');return{fixture:'compiled'};}},activate:async(value,options)=>{assert.equal(value.fixture,'compiled');assert.equal(options.reloadModels,false);calls.push('activate');},jsonURL:()=>{throw Error('No live planner may run in the frozen branch');}};
  vm.runInNewContext(helper+';globalThis.run=runInstruction;',context);await context.run();assert.deepEqual(calls,['local','activate']);if(snapshotMode)assert.equal(nodes['download-kit'].textContent,'Baseline starter ↓');
 }
 assert.match(source,/snapshotMode=envelope\.snapshot_mode===true/);
});
