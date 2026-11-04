/* Shipped calibration controller; synthetic DOM/HTTP only, no app/model server. */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const {webcrypto,createHash}=require('node:crypto');
const SOURCE=path.join(__dirname,'../workspace/calibration-lab.js');
const idA='calibration-'+'a'.repeat(24),idB='calibration-'+'b'.repeat(24);
const hash=body=>createHash('sha256').update(JSON.stringify(body)).digest('hex');
const byteHash=body=>createHash('sha256').update(body).digest('hex');
const reportURL=id=>`/files/data/calibration-runs-20261101/${id}/result/report.json`;
const cameraURL=id=>`/files/data/calibration-runs-20261101/${id}/result/camera-calibration.json`;
const detectionsURL=id=>`/files/data/calibration-runs-20261101/${id}/result/detections.json`;
const overlayURL=(id,name='fixture')=>`/files/data/calibration-runs-20261101/${id}/result/overlays/${name}.jpg`;
// WebCrypto completes on a worker pool, independently of setImmediate turns.
// Drain actual submitted digests before asserting the next explicit HTTP stage;
// the harness still resolves every fetch itself and uses the real SHA-256.
const pendingDigests=new Set();
const trackedCrypto={subtle:{digest(...args){const work=webcrypto.subtle.digest(...args);pendingDigests.add(work);work.then(()=>pendingDigests.delete(work),()=>pendingDigests.delete(work));return work}}};
const flush=async()=>{for(let i=0;i<3;i++){await new Promise(setImmediate);if(pendingDigests.size)await Promise.allSettled([...pendingDigests])}};
class Element{
 constructor(tag='div',id=''){this.tagName=tag.toUpperCase();this.id=id;this.children=[];this.options=[];this.value='';this.hidden=false;this.disabled=false;this.style={};this.attrs={};this.listeners={};this._text='';this.classList={toggle:()=>{}};}
 set textContent(v){this._text=String(v);this.children=[];}
 get textContent(){return this._text+this.children.map(c=>typeof c==='string'?c:c.textContent).join('');}
 append(...values){this.children.push(...values);}
 replaceChildren(...values){this._text='';this.children=[...values];if(this.tagName==='SELECT'){this.options=[...values];this.value=this.options[0]?.value||'';}}
 add(v){this.options.push(v);this.children.push(v);}
 setAttribute(k,v){this.attrs[k]=String(v);}
 removeAttribute(k){delete this.attrs[k];}
 addEventListener(k,v){this.listeners[k]=v;}
 showModal(){this.open=true;} close(){this.open=false;} click(){}
}
function response(body){const binary=Buffer.isBuffer(body),data=binary?body:Buffer.from(JSON.stringify(body));return {ok:true,status:200,json:async()=>{if(binary)throw Error('Software fixture binary is not JSON');return body},arrayBuffer:async()=>data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength)};}
function report(id,{passing=false,source='a'.repeat(64),marker=id}={}){
 const body={format:'checkerboard-camera-calibration',version:1,status:passing?'candidate_passed_checks':'insufficient_evidence',source:{path:`data/uploads/${id}.mp4`,sha256:source},physical_accuracy_verified:false,object_metric_scale_established:false,new_api_calls:0,metrics:{sampled_views:10,eligible_train_views:8,sampled_heldout_views:2,training_rms_px:.1},diagnostics:[{action:marker}],image_size:[640,480],splits:{train:['1','2'],heldout:['3']}};
 return body;
}
function camera(id){return {format:'explicit-checkerboard-camera-record',version:1,source:{sha256:'a'.repeat(64)},physical_accuracy_verified:false,object_metric_scale_established:false,metric_scale_verified:false,heldout_checks_passed:true,K:[[400,0,320],[0,400,240],[0,0,1]],distortion_coefficients:[0,0,0,0,0]};}
function job(id,body=null,{status='completed',error='No camera for '+id}={}){return {id,status,error,request:{source_sha256:'a'.repeat(64)},created_at:'2026-11-01T00:00:00Z',...(body?{result_url:reportURL(id),result_sha256:hash(body)}:{})};}
function setup(mutation){
 const elements=new Map(),requests=[],timers=[],objectURLs=[],revokedURLs=[];
 const get=id=>{if(!elements.has(id))elements.set(id,new Element(id==='saved-calibration'?'select':'div',id));return elements.get(id)};
 get('board-cols').value='9';get('board-rows').value='6';get('result-error').hidden=true;
 const location={href:'http://127.0.0.1:18867/workspace/calibration-lab.html',search:''};
 class FixtureURL extends URL{}
 FixtureURL.createObjectURL=blob=>{const url='blob:software-fixture-'+objectURLs.length;objectURLs.push({url,blob});return url};
 FixtureURL.revokeObjectURL=url=>revokedURLs.push(url);
 const sandbox={console,URL:FixtureURL,URLSearchParams,Blob,TextDecoder,TextEncoder,Uint8Array,crypto:trackedCrypto,location,
  history:{replaceState(_a,_b,u){location.href=String(u);location.search=new URL(location.href).search}},
  document:{getElementById:get,createElement:t=>new Element(t),createElementNS:(_ns,t)=>new Element(t)},
  Option:function(text,value){const e=new Element('option');e.textContent=text;e.value=value;return e},
  setTimeout:(fn,ms)=>{const timer={fn,ms,active:true};timers.push(timer);return timer},clearTimeout:t=>{if(t)t.active=false},
  fetch:(url,options={})=>new Promise((resolve,reject)=>requests.push({url:String(url),options,resolve:body=>resolve(response(body)),reject})),
  XMLHttpRequest:function(){throw Error('This read-only harness must not upload')},XMLSerializer:function(){},};
 let source=fs.readFileSync(SOURCE,'utf8');
 if(mutation==='drop_postrender_guard'){const before=source;source=source.replace('if(epoch!==displayEpoch)return;if(isActive)pollTimer=', 'if(isActive)pollTimer=');assert.notEqual(source,before,'Mutation target missing');}
 if(mutation==='drop_refresh_guard'){const before=source;source=source.replace('if(epoch!==null&&epoch!==displayEpoch)return value.jobs;','');assert.notEqual(source,before,'Mutation target missing');}
 if(mutation==='drop_immediate_clear'){const before=source;source=source.replace("$('result-content').replaceChildren(node('p','Loading the selected attempt…','muted'));",'');assert.notEqual(source,before,'Mutation target missing');}
 vm.createContext(sandbox);vm.runInContext(source+'\n;globalThis.testAPI={loadJob,refresh,verified,evidenceURL,state:()=>({selectedJob,displayEpoch,busy})};',sandbox,{filename:SOURCE});
 const take=url=>{const request=requests.find(r=>r.url===url&&!r.taken);assert.ok(request,'Missing expected request '+url);request.taken=true;return request};
 const start=async(id,body,options)=>{get('saved-calibration').value=id;const promise=sandbox.testAPI.loadJob(id,options);take('/api/calibration/jobs/'+id).resolve(body);await flush();return {promise}};
 return {get,requests,timers,objectURLs,revokedURLs,sandbox,take,start,initial:async()=>{take('/api/calibration/jobs').resolve({jobs:[]});await flush()}};
}
async function run(name,mutation){
 const e=setup(mutation);await e.initial();const text=()=>e.get('result-content').textContent;
 if(name==='stale_metadata'){
  const pA=e.sandbox.testAPI.loadJob(idA);const a=e.take('/api/calibration/jobs/'+idA);
  const {promise:pB}=await e.start(idB,job(idB));await pB;a.resolve(job(idA,report(idA)));await pA;
  assert.equal(e.sandbox.testAPI.state().selectedJob,idB);assert.ok(text().includes(idB));assert.ok(!e.requests.some(r=>r.url===reportURL(idA)));
 }else if(name==='stale_report'||name==='stale_poll'){
  const aReport=report(idA);const {promise:pA}=await e.start(idA,job(idA,aReport,{status:name==='stale_poll'?'running':'completed'}));const held=e.take(reportURL(idA));
  const {promise:pB}=await e.start(idB,job(idB));await pB;held.resolve(aReport);await pA;
  assert.ok(text().includes(idB));assert.equal(e.sandbox.testAPI.state().selectedJob,idB);
  assert.equal(e.timers.filter(t=>t.active).length,0,'Stale poll continuation scheduled an older job');
 }else if(name==='stale_refresh'){
  const {promise:pA}=await e.start(idA,job(idA),{poll:true});const held=e.take('/api/calibration/jobs');
  const {promise:pB}=await e.start(idB,job(idB));await pB;held.resolve({jobs:[job(idA),job(idB)]});await pA;
  assert.equal(e.get('saved-calibration').value,idB,'Stale refresh replaced the newer selector');
 }else if(name==='clear_pending'){
  const a=report(idA);const {promise:pA}=await e.start(idA,job(idA,a));e.take(reportURL(idA)).resolve(a);await pA;assert.ok(text().includes(idA));
  const pB=e.sandbox.testAPI.loadJob(idB);assert.ok(!text().includes(idA),'Old source result remains visible while a new job loads');assert.match(text(),/Loading/);
  e.take('/api/calibration/jobs/'+idB).resolve(job(idB));await pB;
 }else if(name==='wrong_job'){
  const b=report(idB);const p=e.sandbox.testAPI.loadJob(idA);e.take('/api/calibration/jobs/'+idA).resolve(job(idB,b));await flush();
  const requested=e.requests.find(r=>r.url===reportURL(idB)&&!r.taken);if(requested){requested.taken=true;requested.resolve(b)}await p;
  assert.equal(e.get('result-error').hidden,false,'Wrong response job ID was accepted');assert.ok(!text().includes(idB),'Wrong job result was displayed');
 }else if(name==='wrong_source'||name==='wrong_hash'){
  const a=report(idA,{source:name==='wrong_source'?'b'.repeat(64):'a'.repeat(64)}),j=job(idA,a);if(name==='wrong_hash')j.result_sha256='0'.repeat(64);
  const {promise:p}=await e.start(idA,j);e.take(reportURL(idA)).resolve(a);await p;
  assert.equal(e.get('result-error').hidden,false);assert.ok(!text().includes('INTERNAL CHECKS PASSED'));
 }else if(name==='partial_camera_failure'){
  const a=report(idA,{passing:true}),c=camera(idA);a.camera_record={path:cameraURL(idA).slice(7),sha256:hash(c)};const j=job(idA,a);j.camera_record_url=cameraURL(idA);
  const {promise:p}=await e.start(idA,j);e.take(reportURL(idA)).resolve(a);await flush();e.take(cameraURL(idA)).resolve({...c,unbound:true});await p;
  assert.equal(e.get('result-error').hidden,false);assert.ok(!text().includes('INTERNAL CHECKS PASSED'),'Failed camera verification left a passed result visible');assert.equal(e.get('corner-gallery').children.length,0);
 }else if(name==='cross_attempt_artifacts'){
  // Correct digests/source identities are deliberately insufficient: each artifact
  // must also belong to the selected attempt, including nested gallery evidence.
  for(const kind of ['camera','detections','overlay']){
   const a=report(idA,{passing:kind==='camera'}),c=camera(idB),d={views:[]};
   let foreignURL,foreignBody;
   if(kind==='camera'){
    foreignURL=cameraURL(idB);foreignBody=c;
    a.camera_record={path:foreignURL.slice(7),sha256:hash(c)};
   }else if(kind==='detections'){
    foreignURL=detectionsURL(idB);foreignBody=d;
    a.provenance={detections:{path:foreignURL.slice(7),sha256:hash(d)}};
   }else{
    foreignURL=overlayURL(idB);foreignBody=Buffer.from('Explicit software fixture image bytes');
    d.views=[{id:'fixture',split:'train',detected:true,overlay_path:foreignURL.slice(7),overlay_sha256:byteHash(foreignBody)}];
    a.provenance={detections:{path:detectionsURL(idA).slice(7),sha256:hash(d)}};
   }
   const j=job(idA,a);if(kind==='camera')j.camera_record_url=foreignURL;
   const requestStart=e.requests.length,{promise:p}=await e.start(idA,j);
   e.take(reportURL(idA)).resolve(a);await flush();
   if(kind==='overlay'){e.take(detectionsURL(idA)).resolve(d);await flush();}
   // Resolve a regressed foreign fetch so the assertion is diagnostic, not a timeout.
   const unexpected=e.requests.slice(requestStart).find(r=>r.url===foreignURL&&!r.taken);
   if(unexpected){unexpected.taken=true;unexpected.resolve(foreignBody);}
   await p;
   assert.ok(!unexpected,'Cross-attempt '+kind+' reached fetch');
   assert.equal(e.get('result-error').hidden,false,'Cross-attempt '+kind+' was accepted');
   assert.match(e.get('result-error').textContent,/another calibration attempt/);
   assert.ok(!text().includes('INTERNAL CHECKS PASSED'));
   assert.equal(e.get('corner-gallery').children.length,0);
  }
 }else if(name==='successful_camera_gallery_fixture'||name==='staged_camera_gallery_commit'){
  // A tiny embedded software-fixture JPEG, not a photograph or calibration result.
  // This checks verified bytes -> Blob -> DOM wiring; it does not test image decoding.
  const jpeg=Buffer.from('/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDn6KKK/cz8cP/Z','base64');
  const a=report(idA,{passing:true,marker:'Explicit software fixture only'}),c=camera(idA);
  const d={views:['train','heldout'].map((split,i)=>({id:'fixture-'+(i+1),split,detected:true,overlay_path:overlayURL(idA,'fixture-'+(i+1)).slice(7),overlay_sha256:byteHash(jpeg)}))};
  a.camera_record={path:cameraURL(idA).slice(7),sha256:hash(c)};
  a.provenance={detections:{path:detectionsURL(idA).slice(7),sha256:hash(d)}};
  const j=job(idA,a);j.camera_record_url=cameraURL(idA);
  const assertPending=stage=>{if(name==='staged_camera_gallery_commit'){
   assert.equal(text(),'Loading the selected attempt…','Unverified report appeared while '+stage+' was pending');
   assert.ok(!text().includes('INTERNAL CHECKS PASSED'),'Unverified passed badge appeared while '+stage+' was pending');
   assert.equal(e.get('corner-gallery').children.length,0,'Partial gallery appeared while '+stage+' was pending');
   assert.equal(e.get('result-error').hidden,true);
  }};
  const {promise:p}=await e.start(idA,j);assertPending('report');
  e.take(reportURL(idA)).resolve(a);await flush();assertPending('camera');
  e.take(cameraURL(idA)).resolve(c);await flush();assertPending('detections');
  e.take(detectionsURL(idA)).resolve(d);await flush();assertPending('first overlay');
  e.take(overlayURL(idA,'fixture-1')).resolve(jpeg);await flush();assertPending('last overlay');
  e.take(overlayURL(idA,'fixture-2')).resolve(jpeg);await flush();
  await p;
  assert.equal(e.get('result-error').hidden,true,'Valid source-bound fixture was rejected');
  assert.ok(text().includes('INTERNAL CHECKS PASSED'));assert.ok(text().includes('Explicit software fixture only'));
  assert.ok(text().includes('Native-pixel camera matrix'));assert.ok(text().includes('400.000   0.000   320.000'));
  assert.ok(text().includes('Brown lens distortion'));assert.ok(text().includes('640 × 480 pixels'));
  assert.equal(e.sandbox.testAPI.state().selectedJob,idA);
  const figures=e.get('corner-gallery').children;assert.equal(figures.length,2);
  assert.match(figures[0].textContent,/fixture-1 · Training view · corners found/);
  assert.match(figures[1].textContent,/fixture-2 · Reserved view · corners found/);
  assert.equal(e.objectURLs.length,2);
  for(let i=0;i<figures.length;i++){
   const img=figures[i].children[0],objectURL=e.objectURLs[i];
   assert.equal(img.tagName,'IMG');assert.equal(img.src,objectURL.url);
   assert.match(img.alt,new RegExp('fixture-'+(i+1)));
   assert.equal(objectURL.blob.type,'image/jpeg');
   assert.deepEqual(Buffer.from(await objectURL.blob.arrayBuffer()),jpeg);
   img.onload();assert.ok(e.revokedURLs.includes(objectURL.url));
  }
  assert.equal(e.requests.length,7,'Successful fixture must fetch only its selected report/camera/detections/two overlays and metadata');
 }else if(name==='path_confinement'){
  const before=e.requests.length;
  for(const url of ['https://example.com/report.json','/files/data/calibration-runs-20261101-extra/a/report.json',reportURL(idA)+'/../outside.json',reportURL(idA)+'?x=1',reportURL(idA).replace('report.json','%2e%2e/private.json')])await assert.rejects(e.sandbox.testAPI.verified(url,'a'.repeat(64)));
  await e.sandbox.testAPI.loadJob('../bad');assert.equal(e.requests.length,before,'Invalid path reached fetch');
 }else if(name==='safe_text'){
  const literal='<img src=x onerror=alert(1)><script>evil()</script>',a=report(idA,{marker:literal});const {promise:p}=await e.start(idA,job(idA,a));e.take(reportURL(idA)).resolve(a);await p;
  assert.ok(text().includes(literal));const walk=node=>[node,...node.children.flatMap(c=>typeof c==='string'?[]:walk(c))];assert.ok(!walk(e.get('result-content')).some(n=>n.tagName==='IMG'||n.tagName==='SCRIPT'));
 }else throw Error('Unknown scenario '+name);
 assert.ok(e.requests.every(r=>!r.options.method||r.options.method==='GET'),'Read-only tests unexpectedly invoked a write');
 console.log(JSON.stringify({status:'passed',scenario:name,actual_controller:true,synthetic_DOM_and_HTTP_only:true,real_API_or_model_calls:0}));
}
run(process.argv[2],process.argv[3]).catch(error=>{console.error(error);process.exitCode=1});
