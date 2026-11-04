'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),crypto=require('node:crypto');
const ROOT=path.resolve(__dirname,'..'),read=name=>JSON.parse(fs.readFileSync(path.join(ROOT,name),'utf8')),copy=x=>JSON.parse(JSON.stringify(x));
const catalog=read('showcase/public/library-catalog.json'),index=read('showcase/public/cases-index.json'),Reviews=require('../showcase/sam3d-models.js');
globalThis.Sam3DModelReviews=Reviews;const Assets=require('../showcase/robotics-lab-assets.js');
function node(tag='div'){return{tagName:tag.toUpperCase(),children:[],dataset:{},style:{},value:'',textContent:'',hidden:false,disabled:false,checked:true,readyState:1,duration:100,currentTime:0,paused:true,src:'',events:{},focused:false,focus(){this.focused=true;},contentWindow:{messages:[],postMessage(m,o){this.messages.push({message:m,origin:o});}},classList:{add(){},toggle(){}},append(...rows){this.children.push(...rows)},replaceChildren(...rows){this.children=rows},setAttribute(k,v){this[k]=v},removeAttribute(k){delete this[k]},pause(){this.paused=true;},addEventListener(type,fn){this.events[type]=fn;}};}
async function controller({tamperProgram=false,curated=true,mutateIndex=null,sceneRow=null,search='',unavailableSource=null}={}){
 const calls=[],events={},nodes=new Map(),get=id=>{if(!nodes.has(id))nodes.set(id,node(['case-select','moving-asset','container-asset','step-select'].includes(id)?'select':'div'));return nodes.get(id);};
 const publicIndex=copy(index),publicCatalog=copy(catalog);publicIndex.curated=curated;if(mutateIndex)mutateIndex(publicIndex);
 // Isolated transport-availability negative: real source/program bytes and all
 // other records remain unchanged. No production catalog entry is modified.
 if(unavailableSource){const record=publicCatalog.records.find(r=>r.source.sha256===unavailableSource);assert.ok(record);record.sam3d_reviews=[];}
 const location={protocol:'http:',origin:'http://127.0.0.1:8766',pathname:'/showcase/robotics-lab.html',search};
 const ctx={console,TextDecoder,TextEncoder,URL,URLSearchParams,Uint8Array,Map,Set,Number,Math,crypto:crypto.webcrypto,location,history:{replaceState(){}},devicePixelRatio:1,THREE:{LoadingManager:class{setURLModifier(){}}},GLTFLoader:class{},OrbitControls:class{},document:{hidden:false,body:{classList:{add(){}}},getElementById:get,createElement:node,querySelectorAll(){return[];},addEventListener(){}},requestAnimationFrame(){},setTimeout(){},addEventListener(type,fn){events[type]=fn;},InteractionLibrary:{loadCatalog:async()=>copy(publicCatalog)},AlignedScenes:{find:async()=>require('./derived_scene_fixture.cjs')(copy(sceneRow))},AlignedSceneContract:require('../showcase/aligned-scene-contract.js'),fetch:async url=>{
  calls.push(String(url));if(url==='public/cases-index.json')return{ok:true,json:async()=>publicIndex};
  if(String(url).startsWith(location.origin+'/files/data/')){const relative=decodeURIComponent(String(url).split('/files/')[1]);assert.ok(index.cases.some(c=>c.interaction_path===relative),'Only exact curated procedure files may be requested');const raw=fs.readFileSync(path.join(ROOT,relative));const bytes=(typeof tamperProgram==='function'?tamperProgram(relative):tamperProgram)?Buffer.from('{}'):raw;return{ok:true,arrayBuffer:async()=>bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)};}
  throw Error('Unexpected request '+url);
 }};ctx.window=ctx;ctx.globalThis=ctx;vm.createContext(ctx);vm.runInContext(fs.readFileSync(path.join(ROOT,'showcase/sam3d-models.js'),'utf8'),ctx);vm.runInContext(fs.readFileSync(path.join(ROOT,'showcase/robotics-lab-assets.js'),'utf8'),ctx);
 const source=fs.readFileSync(path.join(ROOT,'showcase/robotics-lab.js'),'utf8').replace(/^import .*;\s*$/gm,'').replaceAll('import.meta.url',JSON.stringify(location.origin+'/showcase/robotics-lab.js'));
 await vm.runInContext('(async()=>{'+source+'})()',ctx,{filename:'showcase/robotics-lab.js'});for(let i=0;i<5;i++)await new Promise(setImmediate);return{ctx,get,calls,events};
}
test('public robotics lab resolves only exact current-source SAM components, never historical geometry',()=>{
 const counts=index.cases.map(c=>Assets.forProgram(catalog,c.interaction).length);assert.deepEqual(counts.sort((a,b)=>a-b),[2,3,3,3,4,4]);assert.equal(counts.reduce((n,v)=>n+v,0),19);
 for(const c of index.cases)for(const a of Assets.forProgram(catalog,c.interaction)){assert.equal(a.backend,'Meta SAM 3D Objects');assert.equal(a.source_sha256,c.interaction.source.sha256);assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(ROOT,a.path))).digest('hex'),a.sha256);}
});
test('public controller uses curated catalogs and original verified program, with editable kinematic export',async()=>{
 const h=await controller(),state=h.ctx.RoboticsLab.snapshot();assert.equal(state.curated_only,true);assert.equal(state.public_mode,'replay');assert.equal(state.program_verified,true);assert.equal(state.geometry_backend,'Meta SAM 3D Objects');assert.equal(state.available_sam3d_components.length,4);
 assert.equal(h.calls[0],'public/cases-index.json');assert.ok(h.calls.every(c=>!c.includes('/api/admin')&&!c.includes('workspace/cases')&&!c.includes('/old/')));
 h.ctx.RoboticsLab.setMode('robot');assert.equal(h.get('replay-panel').hidden,true);assert.equal(h.get('robot-panel').hidden,false);assert.equal(h.get('mode-robot')['aria-pressed'],'true');assert.equal(h.get('lab-advanced').open,true);assert.equal(h.get('mode-replay').tabIndex,0,'An advanced experiment preserves a keyboard return to the main tabs');h.get('mode-workbench').events.keydown({key:'Home',preventDefault(){}});assert.equal(h.get('mode-replay').tabIndex,0);assert.equal(h.get('mode-replay').focused,true);
 const draft=h.ctx.RobotRehearsal.exportDraft();assert.equal(draft.simulation_ready,false);assert.equal(draft.robot.dynamics,false);assert.equal(draft.physical_execution_allowed,false);assert.equal(draft.provenance.robot_motion_origin,'authored_in_browser_not_recovered_video_poses');assert.ok(draft.assets.every(a=>a.authored_placement&&a.backend==='Meta SAM 3D Objects'));
 const camera=index.cases.find(c=>c.interaction.tracklets.some(t=>t.id==='camera_body'));assert.ok(camera);h.get('case-select').value=camera.id;await h.get('case-select').onchange();for(let i=0;i<5;i++)await new Promise(setImmediate);assert.equal(h.ctx.RoboticsLab.snapshot().program_verified,true);assert.equal(h.ctx.RoboticsLab.snapshot().geometry_backend,'Meta SAM 3D Objects');assert.equal(h.ctx.RoboticsLab.snapshot().available_sam3d_components.length,3);assert.equal(h.get('export-task').disabled,false,'Exact source/native metadata permits the authored draft; this fixture does not construct a WebGL renderer.');
});
test('an isolated unavailable native-review fixture never borrows another twin geometry or enables an authored export',async()=>{
 const camera=index.cases.find(c=>c.interaction.tracklets.some(t=>t.id==='camera_body'));assert.ok(camera);const h=await controller({unavailableSource:camera.interaction.source.sha256});assert.equal(h.ctx.RoboticsLab.snapshot().available_sam3d_components.length,4);
 h.get('case-select').value=camera.id;await h.get('case-select').onchange();for(let i=0;i<5;i++)await new Promise(setImmediate);const state=h.ctx.RoboticsLab.snapshot();assert.equal(state.program_verified,true);assert.equal(state.available_sam3d_components.length,0);assert.equal(state.asset_ready,false);assert.equal(h.get('export-task').disabled,true);assert.match(h.get('phase-label').textContent,/geometry pending/);assert.ok(catalog.records.find(r=>r.source.sha256===camera.interaction.source.sha256).sam3d_reviews.length===3,'Actual curated native records remain untouched.');
});
test('public controller refuses non-curated indexes and changed procedure bytes',async()=>{
 const raw=await controller({curated:false});assert.match(raw.get('error').textContent,/curated public catalog/);assert.equal(raw.calls.length,1);
 const bad=await controller({tamperProgram:true});assert.equal(bad.ctx.RoboticsLab.snapshot().program_verified,false);assert.equal(bad.ctx.RoboticsLab.snapshot().geometry_backend,null);assert.equal(bad.get('export-task').disabled,true);assert.match(bad.get('error').textContent,/hash mismatch/);
});
test('unbound case-index references cannot escape the verified curated catalog',async()=>{
 const h=await controller({mutateIndex:data=>{for(const row of data.cases)row.interaction_path='data/unapproved/procedure.json';}});assert.equal(h.calls.length,1);assert.match(h.get('error').textContent,/No curated object programs/);assert.equal(h.ctx.RoboticsLab.snapshot().program_verified,false);
});
test('connected replay shares only a validated exact source/program/native scene binding',async()=>{
 const source=catalog.records.find(r=>r.components.some(c=>c.id==='case_base')).source.sha256,row=read('showcase/public/aligned-scenes.json').scenes.find(r=>r.source_sha256===source),manifest=read(row.manifest.path),selected=row.derived_assembly||row.object_program,sceneRow={reference:row.manifest,manifest,object_program:selected,...(row.derived_assembly?{base_object_program:row.object_program,derived_assembly:{artifact:row.derived_assembly,document:read(row.derived_assembly.path)}}:{})},h=await controller({sceneRow});
 assert.equal(h.ctx.RoboticsLab.snapshot().connected_replay,true);assert.match(h.get('aligned-frame').src,/aligned-scene\.html\?source=/);assert.equal(h.get('scene-program').href,'http://127.0.0.1:8766/files/'+selected.path);
 const ev={origin:'http://127.0.0.1:8766',source:h.get('aligned-frame').contentWindow,data:{type:'interaction-twin:aligned-ready',source_sha256:manifest.source.sha256,manifest_sha256:row.manifest.sha256}};h.events.message(ev);assert.equal(h.get('aligned-frame').contentWindow.messages.length,1);assert.equal(h.get('aligned-frame').contentWindow.messages[0].message.manifest_sha256,row.manifest.sha256);
 ev.origin='https://untrusted.example';h.events.message(ev);assert.equal(h.get('aligned-frame').contentWindow.messages.length,1);
 manifest.original_program.sha256='0'.repeat(64);const other=await controller({sceneRow:{reference:row.manifest,manifest}});assert.equal(other.ctx.RoboticsLab.snapshot().connected_replay,false);assert.match(other.get('aligned-message').textContent,/exact curated program|Derived assembly source\/evidence/);
});

test('failed case transition clears the previous source preview and evidence links',async()=>{
 const camera=index.cases.find(c=>c.interaction.tracklets.some(t=>t.id==='camera_body'));assert.ok(camera);const h=await controller({tamperProgram:relative=>relative===camera.interaction_path});
 assert.equal(h.ctx.RoboticsLab.snapshot().program_verified,true);assert.ok(h.get('source-video').poster);assert.ok(h.get('open-evidence').href);
 h.get('case-select').value=camera.id;await h.get('case-select').onchange();for(let i=0;i<5;i++)await new Promise(setImmediate);
 assert.equal(h.ctx.RoboticsLab.snapshot().program_verified,false);assert.equal(h.get('source-video').poster,undefined);assert.equal(h.get('open-evidence').href,undefined);assert.equal(h.get('evidence-thumbnails').children.length,0);assert.equal(h.get('export-task').disabled,true);assert.match(h.get('case-summary').textContent,/Choose another recording/);
});


test('workbench selection stays bound to the chosen curated twin and unknown cases never fall back',async()=>{
 const h=await controller(),first=h.ctx.RoboticsLab.snapshot().case_id;h.ctx.RoboticsLab.setMode('workbench');assert.equal(h.get('workbench-panel').hidden,false);assert.equal(h.get('lab-case-picker').hidden,false);assert.equal(h.get('robot-panel').hidden,true);assert.match(h.get('workbench-frame').src,new RegExp('case='+encodeURIComponent(first)));
 const wallet=index.cases.find(c=>/wallet/i.test(c.title)),next=wallet||index.cases.find(c=>c.id!==first);h.get('case-select').value=next.id;await h.get('case-select').onchange();assert.match(h.get('workbench-frame').src,new RegExp('case='+encodeURIComponent(next.id)));
 h.get('mode-workbench').events.keydown({key:'Home',preventDefault(){}});assert.equal(h.get('mode-replay').tabIndex,0);assert.equal(h.get('replay-panel').hidden,false);
 h.ctx.RoboticsLab.setMode('workbench');h.get('case-select').value='unapproved_private';await h.get('case-select').onchange();assert.equal(h.ctx.RoboticsLab.snapshot().case_id,undefined);assert.equal(h.get('workbench-frame').src,undefined);assert.match(h.get('error').textContent,/not in the curated/);assert.throws(()=>h.ctx.RoboticsLab.setMode('unknown'),/supported/);
});


test('an unknown explicit deep-link twin stops before selecting another recording in every lab mode',async()=>{
 for(const mode of['replay','robot','workbench','physics']){const h=await controller({search:'?case=unapproved_private&mode='+mode});assert.equal(h.ctx.RoboticsLab.snapshot().case_id,undefined);assert.equal(h.ctx.RoboticsLab.snapshot().program_verified,false);assert.match(h.get('error').textContent,/not in the curated/);assert.equal(h.get('workbench-frame').src,'');assert.equal(h.calls.length,1,'No substitute source program was fetched.');}
});

test('the lab has only two primary tabs; experiments stay in a closed advanced drawer',async()=>{
 const html=fs.readFileSync(path.join(ROOT,'showcase/robotics-lab.html'),'utf8'),tabs=html.match(/role="tab"/g)||[];assert.equal(tabs.length,2);assert.match(html,/<details id="lab-advanced" class="lab-advanced">/);assert.ok(html.indexOf('id="mode-robot"')>html.indexOf('id="lab-advanced"'));assert.ok(html.indexOf('id="mode-physics"')>html.indexOf('id="lab-advanced"'));
 const h=await controller();h.get('mode-replay').events.keydown({key:'End',preventDefault(){}});assert.equal(h.ctx.RoboticsLab.snapshot().public_mode,'workbench');assert.equal(h.get('workbench-panel').hidden,false);assert.equal(h.get('lab-advanced').hidden,true);h.get('mode-workbench').events.keydown({key:'ArrowRight',preventDefault(){}});assert.equal(h.ctx.RoboticsLab.snapshot().public_mode,'replay');h.ctx.RoboticsLab.setMode('physics');assert.equal(h.get('lab-advanced').open,true);assert.equal(h.get('physics-panel').hidden,false);assert.match(h.get('physics-frame').src,/physics-bench/);
});
