'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),crypto=require('node:crypto');
const ROOT=path.resolve(__dirname,'..'),Reviews=require('../showcase/sam3d-models.js');
globalThis.Sam3DModelReviews=Reviews;
const Assets=require('../workspace/robotics-assets.js'),copy=x=>JSON.parse(JSON.stringify(x));
const catalog=JSON.parse(fs.readFileSync(path.join(ROOT,'showcase/library-catalog.json'))),cases=JSON.parse(fs.readFileSync(path.join(ROOT,'workspace/cases.json'))).cases;
test('real SAM 3D rehearsal assets match exact footage and part IDs, preserving the original program',()=>{
 const record=catalog.records.find(r=>r.components.some(c=>c.id==='earbud_a')),entry=cases.find(c=>c.id===record.id),before=JSON.stringify(entry.interaction);
 const assets=Assets.forProgram(catalog,entry.interaction);
 assert.deepEqual(assets.map(a=>a.track_id).sort(),record.sam3d_reviews.map(r=>r.component_id).sort());
 assert.ok(assets.every(a=>a.backend==='Meta SAM 3D Objects'&&!a.source_video_alignment_verified&&!a.simulation_ready));
 for(const a of assets)assert.equal(a.selected,entry.interaction.assets.some(original=>original.track_id===a.track_id&&original.sha256===a.sha256&&original.backend==='Meta SAM 3D Objects'));
 assert.equal(JSON.stringify(entry.interaction),before);
 assert.ok(assets.every(a=>!entry.interaction.assets.includes(a)));
 const chosen=Assets.defaultAssets(assets,entry.interaction.tracklets);assert.match(chosen.moving.track_id,/earbud/);assert.equal(chosen.context.track_id,'case_base');
 if(assets.some(a=>a.selected&&/earbud/.test(a.track_id)))assert.equal(chosen.moving.selected,true);
 assets[0].source_image.path='changed';assert.notEqual(record.sam3d_reviews[0].source_image.path,'changed');
});
test('an older recording never supplies its historical mesh when SAM 3D geometry is missing',()=>{
 const program=copy(cases.find(c=>c.interaction.assets.length).interaction);program.source.sha256='f'.repeat(64);
 assert.deepEqual(Assets.forProgram(catalog,program),[]);
 assert.deepEqual(Assets.defaultAssets([],program.tracklets),{moving:null,context:null});
});
test('same footage with different track names cannot claim another part model',()=>{
 const record=catalog.records.find(r=>r.components.some(c=>c.id==='earbud_a')),program=copy(cases.find(c=>c.id===record.id).interaction);
 program.tracklets=[{id:'a_different_earbud',label:'Earbud'}];assert.deepEqual(Assets.forProgram(catalog,program),[]);
});
test('invalid provenance or ambiguous source catalog refuses geometry',()=>{
 const c=copy(catalog),record=c.records.find(r=>r.sam3d_reviews.length),program=cases.find(p=>p.id===record.id).interaction;
 const original=copy(record.sam3d_reviews);
 for(const mutate of [r=>r.backend='TripoSR',r=>r.selected=true,r=>r.source_video_alignment_verified=true,r=>r.simulation_ready=true,r=>r.source_sha256='f'.repeat(64)]){
  record.sam3d_reviews=copy(original);record.sam3d_reviews.forEach(mutate);assert.deepEqual(Assets.forProgram(c,program),[]);
 }
 record.sam3d_reviews=original;c.records.push(copy(record));assert.throws(()=>Assets.forProgram(c,program),/Several library records/);
});
test('selection requires the current program component and exact native SHA, while physics stays unverified',()=>{
 const record=catalog.records.find(r=>r.sam3d_reviews.length),entry=cases.find(c=>c.id===record.id),program=copy(entry.interaction),row=record.sam3d_reviews[0];
 const selectedAsset={track_id:row.component_id,sha256:row.native_glb.sha256,backend:'Meta SAM 3D Objects'};
 program.assets=[selectedAsset];let resolved=Assets.forProgram(catalog,program),chosen=resolved.find(a=>a.track_id===row.component_id);
 assert.equal(chosen.selected,true);assert.equal(chosen.geometry_status,'selected_native_geometry_pending_alignment');assert.equal(chosen.selection_scope,'selected_in_current_evidence_program');
 assert.equal(chosen.source_video_alignment_verified,false);assert.equal(chosen.physical_accuracy_verified,false);assert.equal(chosen.simulation_ready,false);
 for(const changed of [{...selectedAsset,sha256:'f'.repeat(64)},{...selectedAsset,track_id:'a_different_part'},{...selectedAsset,backend:'TripoSR'}]){
  program.assets=[changed];chosen=Assets.forProgram(catalog,program).find(a=>a.track_id===row.component_id);assert.equal(chosen.selected,false);assert.equal(chosen.geometry_status,'unselected_native_candidate_pending_alignment');
 }
});

function node(tag='div'){
 return{tagName:tag.toUpperCase(),children:[],dataset:{},style:{},value:'',textContent:'',hidden:false,disabled:false,checked:false,readyState:1,duration:8,currentTime:0,
  classList:{add(){},toggle(){}},append(...rows){this.children.push(...rows)},replaceChildren(...rows){this.children=rows},setAttribute(k,v){this[k]=v},removeAttribute(k){delete this[k]},pause(){},addEventListener(){}};
}
function contract(selectNative=false){
 const ref=(name)=>({path:'data/review/'+name,sha256:'b'.repeat(64),bytes:10});
 const programs=['jabra','camera'].map((name,i)=>({case_id:name,summary:'Verified '+name+' story',source:{path:'data/'+name+'/video.mp4',sha256:(i?'c':'a').repeat(64),duration_s:8,time_domain:'source_video_presentation_seconds'},reasoning:{model:'muse-spark-1.3'},tracklets:[{id:'earbud_a',label:'Earbud'},{id:'case_base',label:'Case base'}],assets:[{track_id:'earbud_a',backend:'TripoSR',path:'data/old/asset.glb',sha256:'d'.repeat(64)}],steps:[{id:'remove',title:'Remove earbud',description:'Observed action',start_s:1,end_s:2,participant_track_ids:['earbud_a','case_base'],limitations:[],evidence_ids:[]}],evidence:[]}));
 if(selectNative)programs[0].assets=programs[0].tracklets.map(t=>({track_id:t.id,backend:'Meta SAM 3D Objects',path:'data/imported/'+t.id+'/asset.glb',sha256:'b'.repeat(64)}));
 const entries=programs.map(p=>({id:p.case_id,title:p.case_id,duration_s:8,interaction_path:'data/'+p.case_id+'/program.json',original_sha256:crypto.createHash('sha256').update(JSON.stringify(p)).digest('hex'),poster_path:'data/'+p.case_id+'/frame.png',interaction:{...copy(p),summary:'UNVERIFIED embedded story'}}));
 const records=programs.map(p=>({id:p.case_id,source:p.source,components:p.tracklets.map(t=>({id:t.id})),sam3d_reviews:p.case_id==='camera'?[]:p.tracklets.map(t=>({component_id:t.id,label:t.label,backend:'Meta SAM 3D Objects',source_sha256:p.source.sha256,selected:false,physical_accuracy_verified:false,source_video_alignment_verified:false,simulation_ready:false,native_glb:ref(t.id+'.glb'),source_image:ref('source.png'),source_mask:ref('mask.png'),review_page:ref('index.html'),review_manifest:ref('review-manifest.json'),source_timestamp_s:1,source_sample_index:1}))}));
 return{programs,entries,catalog:{records}};
}
async function controller({badHash=false,selectNative=false}={}){
 const data=contract(selectNative),nodes=new Map(),get=id=>{if(!nodes.has(id))nodes.set(id,node(['case-select','moving-asset','container-asset','step-select'].includes(id)?'select':'div'));return nodes.get(id);};
 const location={protocol:'http:',origin:'http://127.0.0.1:8765',pathname:'/workspace/robotics.html',search:''},calls=[];
 const ctx={console,TextDecoder,TextEncoder,URL,URLSearchParams,Uint8Array,Map,Set,Number,Math,crypto:crypto.webcrypto,location,history:{replaceState(){}},devicePixelRatio:1,
  THREE:{LoadingManager:class{setURLModifier(){}}},GLTFLoader:class{},OrbitControls:class{},
  document:{hidden:false,body:{classList:{add(){}}},getElementById:get,createElement:node,querySelectorAll(){return[]},addEventListener(){}},
  requestAnimationFrame(){},setTimeout(){},InteractionLibrary:{loadCatalog:async()=>copy(data.catalog)},
  fetch:async url=>{calls.push(String(url));if(url==='cases.json')return{ok:true,json:async()=>({cases:copy(data.entries)})};const p=data.programs.find(p=>String(url).endsWith('/data/'+p.case_id+'/program.json'));if(p){const raw=Buffer.from(JSON.stringify(badHash?{...p,summary:'changed original'}:p));return{ok:true,arrayBuffer:async()=>raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength)}}throw Error('Unexpected geometry/network request '+url);}};
 ctx.window=ctx;ctx.globalThis=ctx;vm.createContext(ctx);
 vm.runInContext(fs.readFileSync(path.join(ROOT,'showcase/sam3d-models.js'),'utf8'),ctx);
 vm.runInContext(fs.readFileSync(path.join(ROOT,'workspace/robotics-assets.js'),'utf8'),ctx);
 const source=fs.readFileSync(path.join(ROOT,'workspace/robotics.js'),'utf8').replace(/^import .*;\s*$/gm,'').replaceAll('import.meta.url',JSON.stringify('http://127.0.0.1:8765/workspace/robotics.js'));
 await vm.runInContext('(async()=>{'+source+'})()',ctx,{filename:'workspace/robotics.js'});
 return{ctx,get,data,calls};
}
test('actual controller exports SAM 3D candidates with authored poses while retaining verified original evidence',async()=>{
 const h=await controller(),s=h.ctx.RobotRehearsal.snapshot();assert.equal(s.geometry_backend,'Meta SAM 3D Objects');assert.equal(s.original_program_asset_count,1);
 const exported=h.ctx.RobotRehearsal.exportDraft();assert.equal(exported.provenance.original_evidence_program_unchanged,true);assert.equal(exported.provenance.robot_motion_origin,'authored_in_browser_not_recovered_video_poses');
 assert.ok(exported.assets.every(a=>a.backend==='Meta SAM 3D Objects'&&a.selected===false&&a.authored_placement===true&&!a.source_video_alignment_verified));
 assert.equal(exported.simulation_ready,false);assert.equal(exported.provenance.reasoning.model,'muse-spark-1.3');
 assert.equal(h.get('case-summary').textContent,'Verified jabra story');assert.ok(h.calls.every(url=>!url.includes('/old/')));
 h.get('case-select').value='camera';await h.get('case-select').onchange();for(let i=0;i<5;i++)await new Promise(setImmediate);
 const empty=h.ctx.RobotRehearsal.snapshot();assert.equal(empty.moving_track,undefined);assert.equal(empty.asset_ready,false);assert.equal(h.get('export-task').disabled,true);assert.equal(h.get('phase-label').textContent,'SAM 3D geometry pending');assert.throws(()=>h.ctx.RobotRehearsal.exportDraft(),/choose an asset/);
});
test('actual controller refuses modified original program before exposing geometry',async()=>{
 const h=await controller({badHash:true});assert.equal(h.ctx.RobotRehearsal.snapshot().program_verified,false);assert.equal(h.ctx.RobotRehearsal.snapshot().geometry_backend,null);assert.match(h.get('error').textContent,/hash mismatch/);assert.equal(h.get('export-task').disabled,true);
});
test('actual controller labels current-program SAM selections in both UI and authored export',async()=>{
 const h=await controller({selectNative:true}),draft=h.ctx.RobotRehearsal.exportDraft();
 assert.match(h.get('asset-provenance').textContent,/selected in this program/);
 assert.ok(draft.assets.every(a=>a.selected===true&&a.geometry_status==='selected_native_geometry_pending_alignment'&&a.source_video_alignment_verified===false&&a.physical_accuracy_verified===false&&a.authored_placement===true));
 assert.equal(draft.provenance.original_evidence_program_unchanged,true);assert.equal(draft.simulation_ready,false);
});
