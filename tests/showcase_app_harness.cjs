/* Execute the actual landing controller, catalog verifier and SAM 3D bindings. */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const {webcrypto}=require('node:crypto');
const root=path.join(__dirname,'..'),payload=JSON.parse(fs.readFileSync(path.join(root,'showcase/public/cases-index.json'),'utf8'));
const preservedCases=JSON.parse(fs.readFileSync(path.join(root,'workspace/cases.json'),'utf8'));
const publicAssets=new Set(JSON.parse(fs.readFileSync(path.join(root,'showcase/public/public-assets.json'),'utf8')).assets.map(r=>r.path));
const JABRA_SOURCE='7474ebd9d06c55b63fc073f632e71e5125353ace2538fa7bcf632cfdc1c3d0f2';
const code=fs.readFileSync(path.join(root,'showcase/app.js'),'utf8'),scenario=process.argv[2];
function element(tag='div'){
  const events=new Map(),classes=new Set();
  const e={tagName:tag.toUpperCase(),children:[],attributes:{},dataset:{},style:{},value:'',textContent:'',disabled:false,hidden:false,paused:true,ended:false,currentTime:0,
    classList:{add(c){classes.add(c)},remove(c){classes.delete(c)},toggle(c,on){if(on)classes.add(c);else classes.delete(c)},contains(c){return classes.has(c)}},
    append(...rows){this.children.push(...rows);if(this.tagName==='SELECT'&&!this.value&&rows[0])this.value=rows[0].value},prepend(...rows){this.children.unshift(...rows)},replaceChildren(...rows){this.children=rows;if(this.tagName==='SELECT')this.value=''},
    setAttribute(key,value){this.attributes[key]=value},removeAttribute(key){delete this.attributes[key];delete this[key]},focus(){},addEventListener(name,callback){if(!events.has(name))events.set(name,[]);events.get(name).push(callback)},emit(name){for(const callback of events.get(name)||[])callback({target:this})},
    pause(){if(!this.paused){this.paused=true;queueMicrotask(()=>this.emit('pause'))}},play(){this.paused=false;queueMicrotask(()=>this.emit('play'));return Promise.resolve()},load(){this.currentTime=0;queueMicrotask(()=>this.emit('loadedmetadata'))},showModal(){this.open=true},close(){this.open=false;this.emit('close')}};
  return e;
}
function setup(omitWallet=false,options={}){
  let cases=payload.cases.filter(c=>!omitWallet||c.id!=='flow-rigid-parts-wallet-20261031-v1--flow-procedure-1');
  // A preserved real procedure supplies actual shared step boundaries. Only its
  // gallery ID is aliased inside this isolated controller test, never on disk.
  if(scenario==='shared_step_boundaries'){const legacy=preservedCases.cases.find(c=>c.id==='flow-jabra-frozen-20261028--flow-procedure-1');cases=cases.map(c=>c.id==='flow-capture-laptop-20261103-v1--flow-procedure-1'?{...legacy,id:c.id}:c);}
  const nodes=new Map(),requests=[],catalogRequests=[],legacyScenes=[],posted=[],windowEvents=new Map();let deferred=null,failNext=false,initialRelease=null;
  function gate(mode=false){let release;const promise=new Promise(resolve=>release=resolve);return{value:{promise,mode},release};}
  if(options.deferInitial){const initial=gate(options.initialMode);deferred=initial.value;initialRelease=initial.release;}
  const get=id=>{if(!nodes.has(id)){const e=element(['asset-mode','asset-choice'].includes(id)?'select':'div');e.id=id;if(id==='sample-native-frame')e.contentWindow={postMessage(data,origin){posted.push({data,origin});}};nodes.set(id,e);}return nodes.get(id);};
  const tabs=['jabra','wallet','fresh'].map(key=>{const e=get('sample-'+key);e.dataset.sample=key;e.append(element('small'));return e;});
  const sandbox={console,URL,URLSearchParams,Set,Map,Number,Math,TextDecoder,TextEncoder,crypto:webcrypto,queueMicrotask,
    location:{protocol:'http:',origin:'http://127.0.0.1:8765',search:options.search||'',href:'http://127.0.0.1:8765/showcase/index.html'+(options.search||'')},TWIN_CASES:{cases},
    TwinScene:{Scene:function(){legacyScenes.push(true);throw Error('The SAM 3D default must never construct a historical scene.');}},
    document:{currentScript:{src:'http://127.0.0.1:8765/showcase/library.js'},getElementById(id){return id.startsWith('library-')?null:get(id);},querySelector(selector){const match=selector.match(/^#(sample-[a-z]+) small$/);return match?get(match[1]).children.find(e=>e.tagName==='SMALL')||null:null;},querySelectorAll(selector){return selector==='[data-sample]'?tabs:[];},createElement:element,createTextNode(value){return{textContent:value};}},
    matchMedia(){return{matches:false,addEventListener(){}};},requestAnimationFrame(){},open(){},addEventListener(name,callback){if(!windowEvents.has(name))windowEvents.set(name,[]);windowEvents.get(name).push(callback);},
    fetch:async route=>{
      const url=new URL(String(route),'http://127.0.0.1:8765');requests.push(url.pathname);
      if(url.pathname==='/health')return{ok:true,json:async()=>({status:'ready'})};
      if(options.withAligned&&url.pathname==='/showcase/public/aligned-scenes.json'){const file=path.join(root,'showcase/public/aligned-scenes.json');return fs.existsSync(file)?{ok:true,status:200,text:async()=>fs.readFileSync(file,'utf8')}:{ok:false,status:404};}
      if(url.pathname.startsWith('/files/data/')){const file=path.resolve(root,url.pathname.slice('/files/'.length));assert.ok(file.startsWith(path.join(root,'data')+path.sep));assert.ok(publicAssets.has(url.pathname.slice('/files/'.length)),'Only exact curated artifact paths may be served.');const raw=Buffer.from(fs.readFileSync(file));if(options.corruptProcedure&&url.pathname.endsWith('/interaction.json'))raw[raw.length-2]^=1;return{ok:true,arrayBuffer:async()=>raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength)};}
      if(url.pathname==='/showcase/public/library-index.json')return{ok:true,status:200,text:async()=>fs.readFileSync(path.join(root,'showcase/public/library-index.json'),'utf8'),json:async()=>JSON.parse(fs.readFileSync(path.join(root,'showcase/public/library-index.json'),'utf8'))};
      assert.match(url.pathname,/^\/showcase\/public\/library-catalog(?:-[a-f0-9]{64})?\.json$/,'Only saved catalog data may be fetched.');
      const raw=fs.readFileSync(path.join(root,url.pathname.slice(1)));
      const request={route:url.pathname,delivered:false};catalogRequests.push(request);
      let corrupt=false;
      if(deferred){const gate=deferred;deferred=null;await gate.promise;if(gate.mode===true||gate.mode==='reject')throw Error('Delayed catalog verification failure');corrupt=gate.mode==='hash';}
      if(failNext){failNext=false;throw Error('Catalog unavailable');}
      const returned=Buffer.from(raw);if(corrupt)returned[returned.length-2]^=1;
      return{ok:true,arrayBuffer:async()=>{request.delivered=true;return returned.buffer.slice(returned.byteOffset,returned.byteOffset+returned.byteLength);}};
    }};
  sandbox.window=sandbox;
  vm.createContext(sandbox);
  for(const name of ['showcase/public/library-index.js','showcase/sam3d-models.js','showcase/library.js'])vm.runInContext(fs.readFileSync(path.join(root,name),'utf8'),sandbox,{filename:name});
  if(options.withAligned)for(const name of ['showcase/aligned-scene-contract.js','showcase/aligned-scene-client.js'])vm.runInContext(fs.readFileSync(path.join(root,name),'utf8'),sandbox,{filename:name});
  // Observe the actual initial selection promise without replacing its behavior.
  let controller=code.replace("loadSample('jabra');processor();","window.__testInitialLoad=loadSample('jabra');processor();").replace("loadSample('linked');}else","window.__testInitialLoad=loadSample('linked');}else");
  assert.notEqual(controller,code,'The explicit initial-promise test hook must remain bound to the real controller.');
  if(options.dropGenerationGuards)controller=controller.replaceAll(/if\(epoch!==\w+Epoch\)return;/g,'');
  vm.runInContext(controller,sandbox,{filename:'showcase/app.js'});
  return{get,legacyScenes,requests,catalogRequests,tabs,cases,sandbox,posted,initialRelease,initialPromise:sandbox.__testInitialLoad,
    emitWindow(name,event){for(const callback of windowEvents.get(name)||[])callback(event);},
    deferNext(mode=false){const next=gate(mode);deferred=next.value;return next.release;},
    failNext(){failNext=true;}};
}
const flush=()=>new Promise(resolve=>setImmediate(resolve));
async function until(condition,message){for(let i=0;i<150;i++){if(condition())return;await flush();}throw Error('Controller did not settle: '+message);}
function selected(env,key){return env.get('sample-panel').attributes['aria-labelledby']==='sample-'+key;}
function rowForFrame(env){const url=new URL(env.get('sample-native-frame').src,'http://127.0.0.1:8765/showcase/');
  return{source:url.searchParams.get('source'),component:url.searchParams.get('component'),url};}
async function run(){
  const env=setup(scenario==='missing_sample',{withAligned:scenario==='aligned_sync'});
  await until(()=>selected(env,'jabra')&&env.get('sample-video').src,'initial Jabra native sample');
  assert.equal(env.legacyScenes.length,0);
  const actualCatalog=await env.sandbox.InteractionLibrary.loadCatalog();
  const jabraSource=env.cases.find(c=>c.interaction.source.sha256===JABRA_SOURCE).interaction.source.sha256;
  assert.ok(!env.cases.some(c=>c.id==='flow-rigid-parts-jabra-20261031-v1--flow-procedure-1'),'The real public index must exercise source-SHA fallback, not the historical Jabra ID.');
  assert.ok(env.get('sample-workspace').href.startsWith('library.html?case='),'Public samples link to curated inspection, never the private workspace.');
  const jabra=actualCatalog.records.find(r=>r.source.sha256===jabraSource);
  const realJabra=env.sandbox.Sam3DModelReviews.forRecording(jabra);
  assert.ok(realJabra.length>0,'The actual generated catalog must contain genuine native Jabra previews.');
  if(scenario==='aligned_sync'){
    const index=JSON.parse(fs.readFileSync(path.join(root,'showcase/public/aligned-scenes.json'),'utf8')),row=index.scenes.find(r=>r.source_sha256===jabraSource),manifest=JSON.parse(fs.readFileSync(path.join(root,row.manifest.path),'utf8'));
    const frame=env.get('sample-native-frame'),url=new URL(frame.src,'http://127.0.0.1:8765/showcase/');
    assert.equal(env.get('asset-mode').value,'aligned');assert.equal(url.pathname,'/showcase/aligned-scene.html');assert.equal(url.searchParams.get('sha256'),row.manifest.sha256);assert.equal(url.searchParams.get('source'),jabraSource);
    const opening=env.sandbox.AlignedSceneContract.previewTime(manifest);assert.ok(opening.time_s>0);assert.equal(env.get('sample-video').currentTime,opening.time_s);assert.equal(opening.count,manifest.components.length,'The initial saved frame should show every available part.');
    assert.equal(frame.loading,'eager');assert.equal(frame.hidden,false);assert.match(env.get('sample-caveat').textContent,/3 seconds/);assert.equal(url.searchParams.get('presentation'),'landing');assert.equal(env.get('asset-choice').hidden,true);assert.equal(env.get('estimated-motion-control').hidden,true);
    const validEvent=data=>({origin:'http://127.0.0.1:8765',source:frame.contentWindow,data:{source_sha256:jabraSource,manifest_sha256:row.manifest.sha256,...data}});
    const oldCount=env.posted.length;
    env.emitWindow('message',{...validEvent({type:'interaction-twin:aligned-ready'}),origin:'https://wrong.example'});assert.equal(env.posted.length,oldCount);
    env.emitWindow('message',validEvent({type:'interaction-twin:aligned-ready'}));assert.ok(env.posted.length>oldCount);
    env.get('sample-video').currentTime=5.25;env.get('sample-video').emit('seeked');let latest=env.posted.at(-1);assert.equal(latest.data.time_s,5.25);assert.equal(latest.data.view,'source');assert.equal(latest.data.manifest_sha256,row.manifest.sha256);
    env.get('view-orbit').onclick();assert.equal(env.posted.at(-1).data.view,'orbit');env.get('view-camera').onclick();assert.equal(env.posted.at(-1).data.view,'source');
    env.get('show-estimated-motion').checked=false;env.get('show-estimated-motion').onchange();assert.equal(env.posted.at(-1).data.show_estimated,false);
    const beforeAck=env.posted.length;env.emitWindow('message',validEvent({type:'interaction-twin:aligned-state',time_s:5.25,parts:manifest.components.map(c=>({id:c.id,label:c.label,status:'unknown',visible:false,interpolated:false}))}));assert.equal(env.posted.length,beforeAck,'An acknowledgement must not create a postMessage feedback loop.');
    env.get('asset-mode').value='native';env.get('asset-mode').onchange();assert.match(frame.src,/^native-model\.html\?/);assert.equal(frame.loading,'eager');assert.equal(frame.hidden,false);assert.equal(env.get('estimated-motion-control').hidden,true);
    env.get('asset-mode').value='aligned';env.get('asset-mode').onchange();assert.equal(new URL(frame.src,'http://127.0.0.1:8765/showcase/').searchParams.get('sha256'),row.manifest.sha256);assert.equal(frame.loading,'eager');assert.equal(frame.hidden,false);assert.equal(env.legacyScenes.length,0);
  }else if(scenario==='switch_native'){
    const initial=rowForFrame(env);assert.equal(initial.source,jabra.source.sha256);assert.equal(initial.component,realJabra[0].component_id);
    await env.get('sample-video').play();env.get('sample-wallet').onclick();await until(()=>selected(env,'wallet'),'wallet selection');
    const walletSource=env.cases.find(c=>c.id==='flow-rigid-parts-wallet-20261031-v1--flow-procedure-1').interaction.source.sha256;
    const wallet=actualCatalog.records.find(r=>r.source.sha256===walletSource);
    const rows=env.sandbox.Sam3DModelReviews.forRecording(wallet);assert.ok(rows.length>0);
    const current=rowForFrame(env);assert.equal(current.source,wallet.source.sha256);assert.equal(current.component,rows[0].component_id);
    assert.equal(env.get('asset-mode').value,'native');assert.equal(env.get('asset-mode').disabled,true);
    assert.equal(env.get('view-camera').disabled,true);assert.equal(env.get('sample-native-frame').hidden,false);
    assert.match(env.get('pose-note').textContent,/video alignment pending/);
    assert.ok(env.get('sample-video').src.includes('flow-rigid-parts-wallet-20261031-v1'));
    if(rows.length>1){env.get('asset-choice').value=rows[1].component_id;env.get('asset-choice').onchange();assert.equal(rowForFrame(env).component,rows[1].component_id);}
    env.get('sample-fresh').onclick();await until(()=>selected(env,'fresh'),'laptop selection');
    assert.equal(env.legacyScenes.length,0);
  }else if(scenario==='pending_no_substitute'){
    const camera=actualCatalog.records.find(r=>r.id==='flow-capture-laptop-20261103-v1--flow-procedure-1');
    assert.ok(camera.components.some(c=>c.asset),'The actual recording retains historical assets.');
    assert.ok(env.sandbox.Sam3DModelReviews.forRecording(camera).length>0,'Current camera now has native models.');
    env.sandbox.Sam3DModelReviews={...env.sandbox.Sam3DModelReviews,forRecording:()=>[]}; // Explicit unavailable-review fixture; never alter the genuine catalog.
    env.get('sample-fresh').onclick();await until(()=>selected(env,'fresh'),'pending laptop sample');
    assert.equal(env.get('sample-native-frame').hidden,true);assert.equal(env.get('sample-native-frame').src,undefined);
    assert.equal(env.get('asset-choice').disabled,true);assert.equal(env.get('scene-error').hidden,false);
    assert.match(env.get('scene-error').textContent,/SAM 3D geometry is pending/);
    assert.equal(env.get('sample-coverage').children.length,0,'Historical pose coverage must not be presented as native motion.');
    assert.equal(env.legacyScenes.length,0);
  }else if(scenario==='shared_step_boundaries'){
    env.get('sample-fresh').onclick();await until(()=>selected(env,'fresh'),'shared-boundary source');
    const current={interaction:JSON.parse(fs.readFileSync(path.join(root,actualCatalog.records.find(r=>r.id==='flow-capture-laptop-20261103-v1--flow-procedure-1').interaction.path),'utf8'))};
    for(const i of [1,5]){const step=current.interaction.steps[i];
      const chapter=env.get('sample-chapters').children.find(b=>Number(b.dataset.step)===i);assert.ok(chapter);chapter.onclick();await flush();
      assert.equal(env.get('sample-video').currentTime,step.start_s);assert.equal(env.get('selected-step-title').textContent,step.title);assert.equal(chapter.classList.contains('active'),true);}
  }else if(scenario==='missing_sample'){
    const original=env.get('sample-video').src,native=env.get('sample-native-frame').src;env.get('sample-wallet').onclick();await flush();
    assert.equal(env.get('sample-video').src,original);assert.equal(env.get('sample-native-frame').src,native);
    assert.equal(env.get('sample-title').textContent,'Earbuds & charging case');assert.match(env.get('sample-description').textContent,/current saved sample remains available/);
    env.get('sample-chapters').children[1].onclick();await flush();assert.ok(env.get('selected-step-title').textContent);
  }else if(scenario==='catalog_error'){
    env.failNext();env.get('sample-wallet').onclick();await until(()=>selected(env,'wallet'),'catalog failure state');
    assert.equal(env.get('sample-native-frame').src,undefined);assert.equal(env.get('sample-native-frame').hidden,true);
    assert.equal(env.get('asset-choice').children.length,0);assert.match(env.get('scene-error').textContent,/pending/);
    assert.equal(env.legacyScenes.length,0);
  }else if(scenario==='stale_success'||scenario==='stale_error'){
    const release=env.deferNext(scenario==='stale_error');const earlier=env.get('sample-wallet').onclick();
    env.get('sample-fresh').onclick();await until(()=>selected(env,'fresh'),'newer laptop selection');
    const selectedVideo=env.get('sample-video').src;release();await earlier;await flush();
    assert.equal(env.get('sample-video').src,selectedVideo,'A delayed earlier catalog result must not replace the newer selection.');
    assert.equal(selected(env,'fresh'),true);assert.match(env.get('sample-native-frame').src,/native-model\.html/);
  }else throw Error('Unknown scenario '+scenario);
  assert.ok(env.requests.every(route=>route==='/health'||route==='/showcase/public/library-index.json'||/^\/showcase\/public\/library-catalog(?:-[a-f0-9]{64})?\.json$/.test(route)||route.startsWith('/files/data/')||scenario==='aligned_sync'&&route==='/showcase/public/aligned-scenes.json'),'Controls must not start inference, uploads or paid API calls.');
  console.log(scenario+' passed');
}
if(require.main===module)run().catch(error=>{console.error(error);process.exitCode=1;});
else module.exports={setup,flush,until,selected,JABRA_SOURCE,payload};
