/* Actual app/showcase controllers with deferred loader + synthetic UI doubles.
 * No private catalog, browser pixel/render claim, models, or network requests.
 */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const root=path.resolve(__dirname,'..'),surface=process.argv[2],scenario=process.argv[3];
const flush=()=>new Promise(resolve=>setImmediate(resolve));
function element(tag='div'){
 const listeners=new Map(),classes=new Set();
 const item={tagName:tag.toUpperCase(),children:[],dataset:{},attributes:{},style:{},value:'',textContent:'',className:'',hidden:false,inert:false,disabled:false,checked:false,open:false,files:[],paused:true,ended:false,currentTime:0,lastChild:{textContent:''},
  classList:{add(x){classes.add(x)},remove(x){classes.delete(x)},toggle(x,on){if(on)classes.add(x);else classes.delete(x)},contains(x){return classes.has(x)}},
  append(...rows){this.children.push(...rows)},prepend(...rows){this.children.unshift(...rows)},replaceChildren(...rows){this.children=rows},
  addEventListener(name,fn){if(!listeners.has(name))listeners.set(name,[]);listeners.get(name).push(fn)},emit(name){for(const f of listeners.get(name)||[])f({target:this})},
  getAttribute(name){return this.attributes[name]},setAttribute(name,value){this.attributes[name]=value},focus(){},click(){},closest(){return null},
  pause(){if(!this.paused){this.paused=true;queueMicrotask(()=>this.emit('pause'))}},play(){this.paused=false;queueMicrotask(()=>this.emit('play'));return Promise.resolve()},
  load(){this.currentTime=0;queueMicrotask(()=>this.emit('loadedmetadata'))},showModal(){this.open=true},close(){this.open=false;this.emit('close')}};
 Object.defineProperty(item,'options',{get(){return this.children}});Object.defineProperty(item,'firstChild',{get(){return this.children[0]}});return item;
}
function fixture(id,title){
 const source='data/synthetic-'+title+'/source.mp4';
 const interaction={case_id:id,source:{path:source},reasoning:{model:'muse-spark-1.3',inference_location:'synthetic_test_only'},
  tracklets:[{id:'part',label:'part',identity_status:'unverified',limitations:[]}],
  evidence:[{id:'source_000001',timestamp_s:0,frame_path:'data/synthetic-'+title+'/source.jpg',observations:[{track_id:'part',audit_status:'unknown'}]}],
  variables:[],assertions:[],relationships:[],limitations:[],
  steps:[{id:'step',title:'Synthetic '+title+' step',description:'UI fixture only',start_s:0,end_s:1,evidence_ids:['source_000001'],preconditions:[],postconditions:[],limitations:['Unverified fixture']}]};
 return{id,case_id:id,title:'Fixture '+title,original_sha256:'a'.repeat(64),description:'Synthetic UI test only',duration_s:3,
  interaction,video_path:source,poster_path:'data/synthetic-'+title+'/source.jpg',scene:null,scene_ref:{path:'synthetic-loader-fixture-only'},mask_frames:[],exports:[],labels:{assets:1,steps:1,source_samples:1}};
}
function setup(options={}){
 const ids=surface==='showcase'?['flow-rigid-parts-jabra-20261031-v1--flow-procedure-1','flow-meta-video-wallet-20261030-v1--flow-procedure-1','flow-capture-laptop-20261103-v1--flow-procedure-1']:['fixture-a','fixture-b','fixture-c'];
 const cases=ids.map((id,i)=>fixture(id,['A','B','C'][i])),nodes=new Map(),requests=[],network=[],instances=[],cache=new Map();
 const get=id=>{if(!nodes.has(id)){const node=element(['asset-mode','asset-choice'].includes(id)?'select':'div');node.id=id;nodes.set(id,node)}return nodes.get(id)};
 const tabs=['jabra','wallet','fresh'].map(key=>{const e=get('sample-'+key);e.dataset.sample=key;return e});
 const scene={native_only:true,objects:[{object_id:'part',label:'part',frames:[]}]};
 options.configure?.({cases,scene});
 class Scene{
  constructor(canvas,data){this.canvas=canvas;this.data=data;this.enabled=new Set(data.objects.map(o=>o.object_id));this.inspection='native';this.selected=data.objects[0]?.object_id;this.mode='orbit';this.disposed=false;instances.push(this)}
  dispose(){this.disposed=true}draw(){}fitFrame(){}setTime(){return[]}inspect(mode,id){this.inspection=mode;this.selected=id}objectsInView(){return this.data.objects}get sourceProjection(){return null}
 }
 const location={protocol:'http:',origin:'http://127.0.0.1:8765',pathname:surface==='showcase'?'/showcase/index.html':'/workspace/index.html',search:options.search||'',hash:''};location.href=location.origin+location.pathname+location.search;
 const sandbox={console,URL,URLSearchParams,Map,Set,Number,Math,Blob,AbortController,DOMException,queueMicrotask,location,TWIN_CASES:{cases,lazy_scenes:true},TwinScene:{Scene},
  TwinCaseData:{load(c,{signal}){if(cache.has(c.id))return Promise.resolve(cache.get(c.id));let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});requests.push({c,signal,resolve:()=>resolve({...c,scene}),reject});return promise}},
  document:{getElementById:get,querySelectorAll(selector){return selector==='[data-sample]'?tabs:[]},createElement:element,createTextNode(value){return{textContent:value}},addEventListener(){}},
  history:{replaceState(_s,_t,target){const p=new URL(target,location.href);location.pathname=p.pathname;location.search=p.search;location.hash=p.hash;location.href=p.href}},
  requestAnimationFrame(){},setTimeout(){return 1},setInterval(){return 1},clearInterval(){},addEventListener(){},open(){},
  matchMedia(){return{matches:false,addEventListener(){}}},
  fetch:async(route,options={})=>{network.push({route,options});if(route==='/health')return{ok:true,json:async()=>({status:'ready'})};if(route==='/api/config')return{ok:true,json:async()=>({profiles:[],max_upload_bytes:512*1024*1024})};if(route==='/api/runs/summary')return{ok:true,json:async()=>({runs:[]})};if(route==='/api/budget')return{ok:false,json:async()=>({})};throw Error('Unexpected request '+route)}};
 sandbox.window=sandbox;const source=surface==='showcase'?'showcase/app.js':'workspace/app.js';let controllerCode=fs.readFileSync(path.join(root,source),'utf8');
 // Explicit negative-control test only: mutate an in-memory copy, never source.
 if(process.argv[4]==='drop_generation_guards')controllerCode=controllerCode.replaceAll(/if\(epoch!==\w+Epoch\)return;/g,'');
 vm.runInNewContext(controllerCode,sandbox,{filename:source});
 const select=i=>surface==='showcase'?tabs[i].onclick():sandbox.TwinWorkspace.loadCase(ids[i]);
 const active=()=>surface==='showcase'?tabs.findIndex(t=>t.attributes['aria-selected']==='true'):ids.indexOf(sandbox.TwinWorkspace.active?.id);
 const title=()=>get(surface==='showcase'?'sample-title':'project-title').textContent;
 const video=()=>get(surface==='showcase'?'sample-video':'video');const canvas=()=>get(surface==='showcase'?'sample-scene':'scene');const panel=()=>get(surface==='showcase'?'sample-panel':'workspace-main');
 const message=()=>get(surface==='showcase'?'sample-description':'load-error').textContent;
 return{get,requests,network,instances,cache,cases,scene,select,active,title,video,canvas,panel,message,sandbox};
}
async function runNativeShowcase(){
 const native=require('./showcase_app_harness.cjs');
 const e=native.setup(false,{deferInitial:true,initialMode:scenario==='initial_failure'?'hash':false,
                            dropGenerationGuards:process.argv[4]==='drop_generation_guards'});
 await native.until(()=>e.catalogRequests.length===1,'initial real native catalog fetch');
 assert.equal(e.get('sample-native-frame').src,undefined,'A pending initial verification cannot display an earlier mesh.');
 if(scenario==='late_initial'){
  await e.tabs[1].onclick();await native.flush();
  assert.equal(native.selected(e,'wallet'),true);
  const source=e.get('sample-video').src,frame=e.get('sample-native-frame').src,title=e.get('sample-title').textContent;
  e.initialRelease();await e.initialPromise;await native.flush();
  assert.equal(native.selected(e,'wallet'),true,'A late initial catalog must not replace the newer source.');
  assert.equal(e.get('sample-video').src,source);assert.equal(e.get('sample-native-frame').src,frame);assert.equal(e.get('sample-title').textContent,title);
 }else if(scenario==='failed_fetch'||scenario==='failed_hash'){
  e.initialRelease();await e.initialPromise;await native.flush();
  assert.ok(e.get('sample-native-frame').src,'The actual initial SAM 3D model must verify before the failure test.');
  const oldSource=e.get('sample-video').src;
  const release=e.deferNext(scenario==='failed_hash'?'hash':'reject');const next=e.tabs[1].onclick();
  await native.until(()=>e.catalogRequests.length===2,'deferred next native catalog');
  release();await next;await native.flush();
  assert.equal(native.selected(e,'wallet'),true);assert.notEqual(e.get('sample-video').src,oldSource);
  assert.equal(e.get('sample-native-frame').src,undefined,'A failed verification must remove the previous recording’s native model.');
  assert.equal(e.get('sample-native-frame').hidden,true);assert.equal(e.get('asset-choice').children.length,0);
  assert.match(e.get('scene-error').textContent,/SAM 3D geometry is pending/);
  await e.tabs[0].onclick();await native.flush();
  assert.equal(native.selected(e,'jabra'),true);assert.ok(e.get('sample-native-frame').src,'A later valid catalog fetch must recover the actual native component.');
 }else if(scenario==='initial_failure'){
  e.initialRelease();await e.initialPromise;await native.flush();
  assert.equal(native.selected(e,'jabra'),true);assert.equal(e.get('sample-native-frame').src,undefined);
  assert.equal(e.get('sample-native-frame').hidden,true);assert.equal(e.get('asset-choice').children.length,0);
  assert.match(e.get('scene-error').textContent,/SAM 3D geometry is pending/);
 }else if(scenario==='cached_late_completion'){
  e.initialRelease();await e.initialPromise;await native.flush();
  const release=e.deferNext();const slow=e.tabs[1].onclick();
  // The cached data is the real saved catalog and still passes its true digest.
  // Its resolved promise must be obsolete once a later source is selected.
  const cached=e.tabs[0].onclick(),newest=e.tabs[2].onclick();
  await newest;await cached;await native.flush();
  assert.equal(native.selected(e,'fresh'),true);const source=e.get('sample-video').src;
  release();await slow;await native.flush();
  assert.equal(native.selected(e,'fresh'),true);assert.equal(e.get('sample-video').src,source);
  const selectedModel=new URL(e.get('sample-native-frame').src,'http://127.0.0.1:8765/showcase/');
  assert.equal(selectedModel.searchParams.get('source'),e.cases.find(c=>c.id==='flow-capture-laptop-20261103-v1--flow-procedure-1').interaction.source.sha256,'The newer laptop selection retains its own current native model.');
 }else throw Error('Unknown native showcase scenario');
 assert.equal(e.legacyScenes.length,0,'Native verification errors never instantiate the older geometry renderer.');
 const programs=new Set(JSON.parse(fs.readFileSync(path.join(root,'showcase/public/library-catalog.json'),'utf8')).records.map(r=>'/files/'+r.interaction.path));
 assert.ok(e.requests.every(route=>route==='/health'||route==='/showcase/public/library-index.json'||/^\/showcase\/public\/library-catalog(?:-[a-f0-9]{64})?\.json$/.test(route)||programs.has(route)));
 console.log(JSON.stringify({status:'passed',surface,scenario,actual_catalog_and_SHA_validation:true,actual_controller_source:true,browser_pixels:false,model_calls:0}));
}
async function run(){
 if(surface==='showcase')return runNativeShowcase();
 const e=setup();assert.equal(e.requests.length,1);assert.equal(e.panel().inert,true);assert.equal(e.canvas().style.visibility,'hidden');
 if(scenario==='late_initial'){
  e.select(1);assert.equal(e.requests[0].signal.aborted,true);e.requests[1].resolve();await flush();
  assert.equal(e.active(),1);const source=e.video().src,title=e.title(),instance=e.instances[0];
  e.requests[0].resolve();await flush();assert.equal(e.active(),1);assert.equal(e.video().src,source);assert.equal(e.title(),title);assert.equal(e.instances.length,1);assert.equal(instance.disposed,false);
 }else if(scenario==='failed_fetch'||scenario==='failed_hash'){
  e.requests[0].resolve();await flush();const source=e.video().src,title=e.title(),old=e.instances[0];
  e.select(1);assert.equal(e.panel().inert,true);assert.equal(e.canvas().style.visibility,'hidden');
  e.requests[1].reject(Error(scenario==='failed_hash'?'Saved scene bytes changed.':'Saved scene unavailable; no replacement is shown.'));await flush();
  assert.equal(e.active(),0);assert.equal(e.video().src,source);assert.equal(e.title(),title);assert.equal(old.disposed,false);assert.equal(e.panel().inert,false);assert.equal(e.canvas().style.visibility,'');
  assert.match(e.message(),/Could not load/);assert.match(e.message(),/Still showing/);
  if(surface==='showcase')assert.match(e.message(),/Still showing Earbuds/);else assert.match(e.message(),/Still showing Fixture A/);
  e.select(2);e.requests[2].resolve();await flush();assert.equal(e.active(),2);assert.equal(old.disposed,true);assert.ok(e.video().src.includes('synthetic-C'));assert.equal(e.instances.length,2);
 }else if(scenario==='initial_failure'){
  e.requests[0].reject(Error('Saved scene bytes changed.'));await flush();assert.equal(e.active(),-1);assert.equal(e.instances.length,0);assert.equal(e.canvas().style.visibility,'hidden');assert.equal(e.panel().inert,false);assert.match(e.message(),/No (project|sample) has been loaded/);
 }else if(scenario==='cached_late_completion'){
  e.requests[0].resolve();await flush();e.cache.set(e.cases[0].id,{...e.cases[0],scene:e.scene});
  const old=e.instances[0];e.select(1);e.select(0);e.select(2);await flush();
  // CachedA's resolved promise is stale by the time its microtask executes.
  assert.equal(e.active(),0);assert.equal(e.instances.length,1);assert.equal(e.panel().inert,true);assert.equal(e.requests[1].signal.aborted,true);
  e.requests[2].resolve();await flush();assert.equal(e.active(),2);assert.equal(old.disposed,true);assert.equal(e.instances.length,2);
  e.requests[1].resolve();await flush();assert.equal(e.active(),2);assert.equal(e.instances.length,2);assert.ok(e.video().src.includes('synthetic-C'));
 }else throw Error('Unknown scenario');
 assert.ok(e.network.every(r=>!r.options.method),'Selection must never submit upload/processing/model requests');
 console.log(JSON.stringify({status:'passed',surface,scenario,synthetic_UI_doubles:true,actual_controller_source:true,browser_pixels:false,model_calls:0}));
}
if(require.main===module)run().catch(error=>{console.error(error);process.exitCode=1});
else module.exports={setup,flush};
