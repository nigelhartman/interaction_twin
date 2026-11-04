/* Execute the actual app with a minimal browser contract; no network or models. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'../workspace/app.js'),'utf8');
const scenario=process.argv[2];
function element(tag='div'){
  const item={tagName:tag.toUpperCase(),children:[],dataset:{},style:{},value:'',textContent:'',hidden:false,disabled:false,checked:false,open:false,files:[],paused:true,ended:false,currentTime:0,lastChild:{textContent:''},
    classList:{add(){},remove(){},toggle(){}},append(...children){this.children.push(...children)},replaceChildren(...children){this.children=children},addEventListener(){},getAttribute(name){return this[name]},setAttribute(name,value){this[name]=value},pause(){this.paused=true},load(){},showModal(){this.open=true;this.opens=(this.opens||0)+1},close(){this.open=false},click(){},closest(){return null}};
  Object.defineProperty(item,'options',{get(){return this.children}});
  Object.defineProperty(item,'firstChild',{get(){return this.children[0]}});
  return item;
}
function setup({search='',protocol='http:',profiles=[JSON.parse(fs.readFileSync(path.join(__dirname,'../profiles/meta_sam3d_default.json'),'utf8'))],deferHealth=false,offline=false,cases=[],request=null,nativeDefault=false,caseLoader=null}={}){
  const nodes=new Map(),events=new Map(),documentEvents=new Map(),calls=[];
  const location={protocol,pathname:'/workspace/index.html',search,hash:'',href:(protocol==='file:'?'file:///project':'http://127.0.0.1:8765')+'/workspace/index.html'+search};
  const get=id=>{if(!nodes.has(id)){const n=element();n.id=id;nodes.set(id,n)}return nodes.get(id)};
  get('cloud-consent').checked=true; // A previous consent must never carry forward.
  let healthResolve;
  const healthWait=new Promise(resolve=>healthResolve=resolve);
  class Scene{
    constructor(_canvas,data){this.data=data;this.enabled=new Set(data.objects.map(o=>o.object_id));this.inspection=data.native_only?'native':'assembly';this.selected=data.objects[0]?.object_id;this.mode='orbit'}
    dispose(){}draw(){}fitFrame(){}setTime(){return[]}
    inspect(mode,id){this.inspection=mode;this.selected=id}
    objectsInView(){return this.inspection==='component'?[this.data.objects.find(o=>o.object_id===this.selected)]:this.data.objects}
  }
  const sandbox={console,URL,URLSearchParams,Map,Set,Number,Math,Blob,location,TWIN_CASES:{cases},TwinScene:{Scene},
    document:{getElementById:get,querySelectorAll(){return[]},createElement:element,addEventListener(name,callback){documentEvents.set(name,callback)}},
    history:{replaceState(_state,_title,target){const base=protocol==='file:'?'file:///project':'http://127.0.0.1:8765';const parsed=new URL(target,base);location.pathname=parsed.pathname;location.search=parsed.search;location.hash=parsed.hash;location.href=parsed.href}},
    requestAnimationFrame(){},setTimeout(){return 1},setInterval(){return 1},clearInterval(){},
    fetch:async(route,options={})=>{calls.push({route,options});if(offline)throw new Error('Local processor is offline');if(route==='/health'){if(deferHealth)await healthWait;return{ok:true,json:async()=>({status:'ready'})}}if(route==='/api/config')return{ok:true,json:async()=>({profiles,max_upload_bytes:512*1024*1024})};if(route==='/api/runs')return{ok:true,json:async()=>({runs:[]})};if(request)return request(route,options);throw new Error('Unexpected request '+route)},
    addEventListener(name,callback){events.set(name,callback)}};
  sandbox.window=sandbox;
  sandbox.TWIN_NATIVE_DEFAULT=nativeDefault;
  if(caseLoader)sandbox.TwinCaseData={load:caseLoader};
  vm.runInNewContext(source,sandbox,{filename:'workspace/app.js'});
  return{sandbox,get,calls,events,documentEvents,healthResolve};
}
const flush=()=>new Promise(resolve=>setImmediate(resolve));
function exportFixture(id='export-fixture'){
 const doc={tracklets:[],evidence:[],variables:[],assertions:[],relationships:[],steps:[],limitations:[],reasoning:{model:'fixture',inference_location:'fixture'},source:{path:'data/fixture.mp4'}};
 return{id,title:id,original_sha256:'a'.repeat(64),video_path:'data/fixture.mp4',poster_path:'data/frame.jpg',duration_s:8,interaction:doc,scene:null,mask_frames:[],exports:[],labels:{assets:0,steps:0,source_samples:0}};
}
async function run(){
  if(scenario==='native_profile_filter'){
    const profiles=['meta_sam3d_default','meta_sam31_parts','local_low'].map(id=>JSON.parse(fs.readFileSync(path.join(__dirname,'../profiles',id+'.json'),'utf8')));
    const env=setup({profiles});await flush();
    assert.deepEqual(env.get('run-profile').children.map(o=>o.value),['meta_sam3d_default']);
    assert.equal(env.get('run-profile').value,'meta_sam3d_default');
    const legacy=setup({profiles,search:'?legacy=1'});await flush();
    assert.equal(legacy.get('run-profile').children.length,3);
    assert.ok(env.calls.every(c=>!c.options.method));
  }else if(scenario==='native_default'){
    const fixture=exportFixture('native-default-fixture');
    fixture.interaction.tracklets=[{id:'part',label:'Part',candidate_entity_ids:['part_candidate'],identity_status:'unverified',limitations:[]}];
    fixture.scene_ref={path:'saved-old-scene.json'};
    fixture.scene={objects:[{object_id:'part',label:'Part',frames:[{visible:true,timestamp_s:2}]}]};
    let lazyCalls=0;
    const env=setup({cases:[fixture],nativeDefault:true,caseLoader:()=>{lazyCalls++;throw Error('Old scene must not be loaded');}});
    await flush();
    assert.equal(lazyCalls,0);assert.equal(env.sandbox.TwinWorkspace.renderer,null);
    assert.equal(env.sandbox.TwinWorkspace.selectedObject,'part');
    assert.equal(env.get('scene-label').textContent,'SAM 3D components');
    assert.equal(env.get('coverage').children.length,0,'Must not display old pose coverage as SAM 3D motion');
    assert.match(env.get('sample-age').textContent,/alignment remains separate/);
  }else if(scenario==='delayed_deep_link'){
    const env=setup({search:'?new=1&case=demo&t=1.25',deferHealth:true});
    assert.equal(env.get('import-dialog').open,false,'Must not open while health is unresolved');
    env.healthResolve();await flush();
    assert.equal(env.get('import-dialog').open,true);assert.equal(env.get('import-dialog').opens,1);
    assert.equal(env.get('cloud-consent').checked,false);assert.equal(env.get('start-run').disabled,false);
    assert.equal(new URLSearchParams(env.sandbox.location.search).get('new'),null);
    assert.equal(new URLSearchParams(env.sandbox.location.search).get('case'),'demo');
    assert.equal(new URLSearchParams(env.sandbox.location.search).get('t'),'1.25');
    assert.ok(env.calls.every(c=>!c.options.method),'Opening the dialog must not upload or submit a run');
  }else if(scenario==='offline_file'){
    const env=setup({search:'?new=1',protocol:'file:'});await flush();
    assert.equal(env.get('import-dialog').open,false);assert.equal(env.calls.length,0);
    assert.equal(env.get('start-run').disabled,true);assert.match(env.get('load-error').textContent,/Launch Workspace/);
    env.sandbox.TwinWorkspace.openImport();assert.equal(env.get('import-dialog').open,true);
    assert.equal(env.get('start-run').disabled,true);assert.equal(env.get('cloud-consent').checked,false);
  }else if(scenario==='missing_profiles'){
    const env=setup({search:'?new=1',profiles:[]});await flush();
    assert.equal(env.get('import-dialog').open,false);assert.equal(env.get('start-run').disabled,true);
    assert.match(env.get('load-error').textContent,/No runnable local processing profile/);
    env.sandbox.TwinWorkspace.openImport();assert.match(env.get('import-status').textContent,/No runnable local processing profile/);
    assert.equal(env.get('cloud-consent').checked,false);
  }else if(scenario==='navigation'){
    const env=setup();await flush();assert.equal(env.get('import-dialog').open,false);
    env.sandbox.location.search='?new=1&case=demo';env.events.get('popstate')();await flush();
    assert.equal(env.get('import-dialog').open,true);assert.equal(env.get('cloud-consent').checked,false);
    env.get('import-dialog').close();env.get('cloud-consent').checked=true;
    env.documentEvents.get('click')({target:{closest(){return{id:'inline-cta'}}},preventDefault(){}});await flush();
    assert.equal(env.get('import-dialog').open,true);assert.equal(env.get('cloud-consent').checked,false);
    assert.ok(env.calls.every(c=>!c.options.method));
  }else if(scenario==='invalid_profile'){
    const env=setup({search:'?new=1'});await flush();env.get('run-profile').value='nonexistent-profile';
    env.get('cloud-consent').checked=true;env.get('video-file').files=[{name:'sample.mp4',size:40}];
    await env.get('import-form').onsubmit({preventDefault(){}});
    assert.match(env.get('import-status').textContent,/selected processing profile is unavailable/);
    assert.ok(env.calls.every(c=>!c.options.method),'An invalid profile must not save or send the selected video');
  }else if(scenario==='meta_provider_consent'){
    const profiles=[{id:'local_low',mode:'local_inference',title:'Codex baseline'},
      {id:'meta_low',mode:'local_inference',title:'Meta reasoning',reasoning_provider:'meta_api',disclosure:'Meta hosted reasoning; local geometry.'},
      {id:'meta_sam31_low',mode:'local_inference',title:'Meta SAM',reasoning_provider:'meta_api',segmentation_provider:'meta_sam31_api'},
      {id:'meta_sam31_video_low',mode:'local_inference',title:'Meta video',reasoning_provider:'meta_api',segmentation_provider:'meta_sam31_api',segmentation_mode:'video_boxes',disclosure:'Meta video; local geometry.'}];
    const env=setup({search:'?new=1&legacy=1',profiles});await flush();
    assert.equal(env.get('run-profile').value,'meta_sam31_video_low');
    assert.match(env.get('cloud-consent-label').textContent,/sampled video copy/);
    assert.match(env.get('cloud-consent-label').textContent,/sent to Meta/);
    assert.doesNotMatch(env.get('cloud-consent-label').textContent,/OpenAI/);
    assert.match(env.get('selected-profile-details').textContent,/local geometry/);
    env.get('cloud-consent').checked=true;env.get('run-profile').value='meta_sam31_low';env.get('run-profile').onchange();
    assert.equal(env.get('cloud-consent').checked,false,'Changing a hosted provider must clear previous consent');
    assert.match(env.get('cloud-consent-label').textContent,/paid SAM 3.1 segmentation/);
    env.get('cloud-consent').checked=true;env.get('run-profile').value='local_low';env.get('run-profile').onchange();
    assert.equal(env.get('cloud-consent').checked,false);assert.match(env.get('cloud-consent-label').textContent,/OpenAI through Codex/);
    assert.ok(env.calls.every(c=>!c.options.method),'Choosing profiles must not trigger generation or upload');
  }else if(scenario==='offline_http'){
    const env=setup({search:'?new=1',offline:true});await flush();
    assert.equal(env.get('import-dialog').open,false);assert.equal(env.get('start-run').disabled,true);
    assert.match(env.get('load-error').textContent,/Local processor is offline/);
    assert.equal(await env.sandbox.TwinWorkspace.requestImport(),false);
    assert.ok(env.calls.every(c=>!c.options.method));
  }else if(scenario==='component_jump'){
    const tracklets=['chosen','sibling'].map(id=>({id,label:id,limitations:[],identity_status:'physical_identity_unverified'}));
    const doc={tracklets,evidence:[],variables:[],assertions:[],relationships:[],steps:[],limitations:[],reasoning:{model:'fixture',inference_location:'fixture'},source:{path:'data/fixture.mp4'}};
    const scene={native_only:true,objects:[{object_id:'chosen',label:'chosen',frames:[{timestamp_s:4,visible:true}]},{object_id:'sibling',label:'sibling',frames:[{timestamp_s:2,visible:true}]}]};
    const env=setup({protocol:'file:',cases:[{id:'jump-fixture',title:'Jump fixture',video_path:'data/fixture.mp4',poster_path:'data/frame.jpg',duration_s:8,interaction:doc,scene,mask_frames:[],exports:[],labels:{assets:2,steps:0,source_samples:0}}]});
    const renderer=env.sandbox.TwinWorkspace.renderer;renderer.inspect('component','chosen');
    env.get('jump-supported').onclick();assert.equal(env.get('video').currentTime,4,'Selected part must not jump to a sibling-only pose');
    env.get('video').currentTime=0;renderer.enabled.delete('chosen');env.get('jump-supported').onclick();assert.equal(env.get('video').currentTime,0,'A hidden selected component must not jump to another part');
    renderer.enabled.add('chosen');renderer.inspect('assembly','chosen');env.get('jump-supported').onclick();assert.equal(env.get('video').currentTime,2,'Assembly mode still considers all enabled parts');
  }else if(scenario==='portable_export'){
    let finish;const env=setup({cases:[exportFixture(),exportFixture('other')],request:async(route,options)=>new Promise(resolve=>finish=resolve)});await flush();
    const panel=env.get('export-list').children[0],checkbox=panel.children[2].children[0],button=panel.children[3],status=panel.children[4];
    assert.equal(checkbox.checked,false);const pending=button.onclick();assert.equal(button.disabled,true);assert.equal(checkbox.disabled,true);
    const submitted=env.calls.filter(c=>c.options.method==='POST');assert.equal(submitted.length,1);
    assert.equal(submitted[0].route,'/api/cases/export-fixture/portable-package');assert.deepEqual(JSON.parse(submitted[0].options.body),{base_sha256:'a'.repeat(64),include_video:false});
    env.sandbox.TwinWorkspace.loadCase('other');const newPanel=env.get('export-list').children[0];
    const prefix='/files/data/portable-exports/export-'+'1'.repeat(32);
    finish({ok:true,json:async()=>({status:'complete',viewer_url:prefix+'/index.html',manifest_url:prefix+'/manifest.json',zip_url:prefix+'.zip'})});await pending;
    assert.equal(newPanel.children.length,6,'An old project export must not add links to the newly selected project');
    assert.equal(panel.children[5].children.length,3);assert.equal(panel.children[5].children[0].download,'export-fixture.zip');assert.match(status.textContent,/Ready/);assert.equal(button.disabled,false);
  }else if(scenario==='portable_bad_response'){
    const prefix='/files/data/portable-exports/export-'+'1'.repeat(32);
    const env=setup({cases:[exportFixture()],request:async()=>({ok:true,json:async()=>({status:'complete',viewer_url:prefix+'/index.html',manifest_url:prefix+'/../bad.json',zip_url:prefix+'.zip'})})});await flush();
    const panel=env.get('export-list').children[0];panel.children[2].children[0].checked=true;await panel.children[3].onclick();
    assert.equal(JSON.parse(env.calls.find(c=>c.options.method==='POST').options.body).include_video,true);
    assert.match(panel.children[4].textContent,/Unexpected package output path/);assert.equal(panel.children[5].children.length,0,'Validate every link before exposing any success links');
  }else throw new Error('Unknown scenario '+scenario);
  console.log(scenario+' passed');
}
run().catch(error=>{console.error(error);process.exitCode=1});
