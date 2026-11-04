/* Exercises the actual shipped module with synthetic contract fixtures only. */
const fs=require('node:fs'), path=require('node:path'), assert=require('node:assert/strict'), {webcrypto}=require('node:crypto');
global.crypto=webcrypto;
const root=path.resolve(__dirname,'..');
const source=fs.readFileSync(path.join(root,'workspace/automatic-camera.js'),'utf8');
const scenario=process.argv[2];
const enc=new TextEncoder(), B=value=>enc.encode(value);
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return{promise,resolve};};
const response=bytes=>({ok:true,arrayBuffer:async()=>bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)});
const tick=()=>new Promise(r=>setImmediate(r));
class DomElement {
 constructor(tag='div'){this.tagName=tag.toUpperCase();this.children=[];this.parentElement=null;this.attrs={};this.hidden=false;this.disabled=false;this._text='';this.className='';this.value='';this.connectedRoot=false;this.classList={add:(...classes)=>{this.className=[...new Set([...this.className.split(/\s+/).filter(Boolean),...classes])].join(' ');},contains:name=>this.className.split(/\s+/).includes(name)};}
 set textContent(value){for(const child of this.children)if(typeof child!=='string')child.parentElement=null;this.children=[];this._text=String(value);}
 get textContent(){return this._text+this.children.map(child=>typeof child==='string'?child:child.textContent).join('');}
 append(...children){for(const child of children){if(typeof child!=='string')child.parentElement=this;this.children.push(child);}}
 replaceChildren(...children){for(const child of this.children)if(typeof child!=='string')child.parentElement=null;this.children=[];this._text='';this.append(...children);}
 setAttribute(key,value){this.attrs[key]=String(value);}
 getAttribute(key){return this.attrs[key]??null;}
 contains(value){return !!value&&(value===this||this.children.some(c=>typeof c!=='string'&&c.contains(value)));}
 get isConnected(){return this.connectedRoot||!!this.parentElement?.isConnected;}
 querySelectorAll(selector){const match=element=>selector.startsWith('.')?element.classList.contains(selector.slice(1)):element.tagName===selector.toUpperCase();return this.children.flatMap(child=>typeof child==='string'?[]:[...(match(child)?[child]:[]),...child.querySelectorAll(selector)]);}
 querySelector(selector){return this.querySelectorAll(selector)[0]||null;}
 focus(){document.activeElement=this;}
}
function domFixture(){
 const root=new DomElement('section');root.connectedRoot=true;
 global.document={createElement:tag=>new DomElement(tag),activeElement:null};
 global.Option=function(text,value){const option=new DomElement('option');option.textContent=text;option.value=value;return option;};
 return root;
}
async function waitForDOM(predicate){for(let i=0;i<200;i++){const value=predicate();if(value)return value;await new Promise(resolve=>setTimeout(resolve,2));}throw Error('Expected shipped-renderer DOM state did not arrive');}
(async()=>{
 const mod=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
 const digest=async bytes=>mod.sha256(bytes);
 async function fixture(suffix='a'){
  const binding={caseId:'case-'+suffix,sourceSha256:(suffix==='a'?'a':'b').repeat(64),trackId:'part-'+suffix};
  const frameBytes=B('SOURCE PIXELS '+suffix),frame={id:'frame-'+suffix,pts_s:2.25,path:'data/source-'+suffix+'/frame.jpg',sha256:await digest(frameBytes),split:'reserved',role:'Reserved pose anchors and scored points are separate.'};
  const viewer={format:'observed-geometry-sparse-experiment',kind:'automatic_camera_hypothesis',status:'automatic_camera_diagnostic_only',cameras:[{K:[[900,0,270],[0,900,480],[0,0,1]]}],input_protocol_sha256:'d'.repeat(64),new_api_calls:0,prior_geometry_or_depth_used:false,calibration:{camera_calibrated:false,physical_shape_verified:false,physical_scale_known:false,production_camera_changed:false,selected_production_camera_id:null,reserved_used_for_selection:false},frames:[{id:frame.id,timestamp_s:frame.pts_s,image_url:'/files/'+frame.path,width:540,height:960}]};
  const viewerBytes=B(JSON.stringify(viewer)),viewerIdentity={path:'data/camera-'+suffix+'/viewer.json',sha256:await digest(viewerBytes)};
  const original={format:'automatic-camera-comparison',version:1,case_id:binding.caseId,track_id:binding.trackId,source:{path:'data/source-'+suffix+'/video.mp4',sha256:binding.sourceSha256},camera_calibrated:false,physical_shape_verified:false,physical_scale_known:false,production_camera_changed:false,manual_geometry_or_poses_used:false,new_api_calls:0,input_protocol_sha256:'d'.repeat(64),source_frames:[{id:frame.id,timestamp_s:frame.pts_s,image_url:'/files/'+frame.path,image_sha256:frame.sha256,split:'heldout'}],hypotheses:[{id:'assumed',viewer:viewerIdentity,K:[[900,0,270],[0,900,480],[0,0,1]]}]};
  const rawBytes=B(JSON.stringify(original));
  const report={format:'automatic-camera-comparison-ui',version:1,case_id:binding.caseId,track_id:binding.trackId,source:original.source,image_size:[540,960],outcome:{status:'ambiguous',title:'Camera remains ambiguous',summary:'Synthetic contract fixture, not a real result.'},physical_accuracy_verified:false,object_metric_scale_established:false,applied_to_geometry:false,new_api_calls:0,hypotheses:[{id:'assumed',label:'Assumed camera',method:'Declared initial assumption',focal_px:[900,900],training:{median_px:1,p90_px:2,views:3,observations:20},reserved:{median_px:2,p90_px:4,views:2,observations:10},status:'diagnostic_only',reason:'Image agreement is not physical calibration.',viewer:viewerIdentity}],frames:[frame],limitations:['Scale is unknown.'],provenance:{report:{path:'data/camera-'+suffix+'/report.json',sha256:await digest(rawBytes)}}};
  const bytes=B(JSON.stringify(report));const row={case_id:binding.caseId,track_id:binding.trackId,source_sha256:binding.sourceSha256,report:{path:'data/camera-'+suffix+'/ui-report.json',sha256:await digest(bytes)}};
  return {binding,report,bytes,row,rawBytes,frameBytes,viewer,viewerBytes,viewerIdentity,original};
 }
 const a=await fixture('a'),b=await fixture('b'),states=[],calls=[];
 const routes=new Map(), entered=new Map();
 function register(f){routes.set('/files/'+f.row.report.path,f.bytes);routes.set('/files/'+f.report.provenance.report.path,f.rawBytes);routes.set('/files/'+f.report.frames[0].path,f.frameBytes);routes.set('/files/'+f.viewerIdentity.path,f.viewerBytes);}
 register(a);register(b);
 const fetchImpl=async(url,opts)=>{calls.push({url,opts});if(entered.has(url))entered.get(url).resolve();assert.equal(opts.method,'GET');assert(!url.includes('/api/'));if(!routes.has(url))throw Error('Missing fixture route '+url);const bytes=await routes.get(url);return response(bytes);};
 const c=mod.createAutomaticCameraController({records:[a.row,b.row],fetchImpl,commit:s=>states.push(s)});
 const last=()=>states.at(-1);
 switch(scenario){
  case'actual_record':{const row=mod.AUTOMATIC_CAMERA_RECORDS[0];assert(row,'No frozen actual comparison');const commits=[];const actual=mod.createAutomaticCameraController({records:mod.AUTOMATIC_CAMERA_RECORDS,fetchImpl:async url=>{assert(url.startsWith('/files/data/'));return response(fs.readFileSync(path.join(root,url.slice(7))));},commit:s=>commits.push(s)});await actual.setBinding({caseId:row.case_id,trackId:row.track_id,sourceSha256:row.source_sha256});assert.equal(commits.at(-1).status,'ready',commits.at(-1).message);assert.equal(commits.at(-1).report.hypotheses.length,3);assert.equal(commits.at(-1).report.frames.length,49);const frame=commits.at(-1).report.frames.find(f=>f.split==='reserved');await actual.selectFrame(frame.id);assert.equal(commits.at(-1).frame.id,frame.id);const inspected=await actual.inspectHypothesis('assumed_image_size');assert.equal(inspected.viewer.data.frames.length,49);assert.equal(inspected.sourceBinding.sourceSha256,row.source_sha256);break;}
  case'study_scope':{const root=domFixture(),panel=mod.createAutomaticCameraPanel(root,{records:[a.row],fetchImpl});await panel.setBinding(a.binding);assert.match(root.textContent,/SEPARATE FROZEN STUDY · 1 source frame · 2.250–2.250 s/);assert.match(root.textContent,/selected reconstruction may use a different frame subset and camera/);panel.dispose();break;}
  case'frame_callback_error':{const root=domFixture(),panel=mod.createAutomaticCameraPanel(root,{records:[a.row],fetchImpl,onInspectFrame:async()=>{throw Error('Exact frame is absent from the selected 35-frame subset.');}});await panel.setBinding(a.binding);const button=await waitForDOM(()=>root.querySelectorAll('button').find(x=>x.textContent==='Inspect in reconstruction ↗'));await button.onclick();const notice=root.querySelector('.ac-frame-callback-error');assert(notice);assert.equal(notice.hidden,false);assert.equal(notice.getAttribute('aria-live'),'polite');assert.equal(notice.getAttribute('role'),'status');assert.match(notice.textContent,/35-frame subset/);assert.equal(button.disabled,false);panel.dispose();break;}
  case'frame_callback_stale':{const root=domFixture(),wait=deferred(),panel=mod.createAutomaticCameraPanel(root,{records:[a.row,b.row],fetchImpl,onInspectFrame:async()=>{await wait.promise;throw Error('OLD SOURCE ERROR MUST NOT LEAK');}});await panel.setBinding(a.binding);const button=await waitForDOM(()=>root.querySelectorAll('button').find(x=>x.textContent==='Inspect in reconstruction ↗'));const clicking=button.onclick();assert.equal(button.disabled,true);await panel.setBinding(b.binding);await waitForDOM(()=>root.querySelectorAll('button').find(x=>x.textContent==='Inspect in reconstruction ↗'));wait.resolve();await clicking;assert(!root.textContent.includes('OLD SOURCE ERROR'),'Old frame callback error crossed source boundaries');assert.equal(root.querySelector('.ac-frame-callback-error').hidden,true);panel.dispose();break;}
  case'resolve':{const result=await mod.resolveAutomaticCameraHypothesis({...a.binding,hypothesisId:'assumed'},{records:[a.row],fetchImpl});assert.equal(result.diagnosticOnly,true);assert.equal(result.id,'assumed');assert.equal(result.sourceBinding.sourceSha256,a.binding.sourceSha256);assert.deepEqual(result.viewer.data,a.viewer);break;}
  case'resolve_wrong_binding':await assert.rejects(()=>mod.resolveAutomaticCameraHypothesis({...a.binding,trackId:'another-part',hypothesisId:'assumed'},{records:[a.row],fetchImpl}),/No saved automatic-camera/);assert.equal(calls.length,0);break;
  case'resolve_unknown_hypothesis':await assert.rejects(()=>mod.resolveAutomaticCameraHypothesis({...a.binding,hypothesisId:'other'},{records:[a.row],fetchImpl}),/No saved geometry/);assert.equal(calls.length,2);break;
  case'resolve_bad_id':await assert.rejects(()=>mod.resolveAutomaticCameraHypothesis({...a.binding,hypothesisId:'../private'},{records:[a.row],fetchImpl}),/identifier/);assert.equal(calls.length,0);break;
  case'resolve_changed_pixels':routes.set('/files/'+a.report.frames[0].path,B('changed'));await assert.rejects(()=>mod.resolveAutomaticCameraHypothesis({...a.binding,hypothesisId:'assumed'},{records:[a.row],fetchImpl}),/changed/);break;
  case'inspect':await c.setBinding(a.binding);{const result=await c.inspectHypothesis('assumed');assert.equal(result.sourceBinding.caseId,a.binding.caseId);assert.equal(result.diagnosticOnly,true);assert.deepEqual(result.viewer.data,a.viewer);assert.deepEqual(new Uint8Array(result.viewer.bytes),a.viewerBytes);}break;
  case'inspect_wrong_hash':await c.setBinding(a.binding);routes.set('/files/'+a.viewerIdentity.path,B('changed'));await assert.rejects(()=>c.inspectHypothesis('assumed'),/changed/);break;
  case'inspect_changed_source':await c.setBinding(a.binding);routes.set('/files/'+a.report.frames[0].path,B('changed'));await assert.rejects(()=>c.inspectHypothesis('assumed'),/changed/);break;
  case'inspect_wrong_camera':{const bad=structuredClone(a.viewer);bad.cameras[0].K[0][0]+=1;assert.throws(()=>mod.validateHypothesisViewer(bad,a.report.hypotheses[0],a.original,a.report),/another camera matrix/);break;}
  case'inspect_wrong_source_metadata':{const bad=structuredClone(a.viewer);bad.frames[0].timestamp_s+=1;assert.throws(()=>mod.validateHypothesisViewer(bad,a.report.hypotheses[0],a.original,a.report),/timestamp/);break;}
  case'inspect_wrong_physical_scope':for(const key of ['camera_calibrated','physical_shape_verified','physical_scale_known','production_camera_changed']){const bad=structuredClone(a.viewer);bad.calibration[key]=true;assert.throws(()=>mod.validateHypothesisViewer(bad,a.report.hypotheses[0],a.original,a.report),/diagnostic scope/);}break;
  case'inspect_stale_viewer':{await c.setBinding(a.binding);const wait=deferred();routes.set('/files/'+a.viewerIdentity.path,wait.promise);const old=c.inspectHypothesis('assumed');await c.setBinding(b.binding);wait.resolve(a.viewerBytes);assert.equal(await old,null,'Old hypothesis escaped after source switch');break;}
  case'inspect_stale_frame':{await c.setBinding(a.binding);const wait=deferred(),seen=deferred();routes.set('/files/'+a.report.frames[0].path,wait.promise);entered.set('/files/'+a.report.frames[0].path,seen);const old=c.inspectHypothesis('assumed');await seen.promise;await c.setBinding(b.binding);wait.resolve(a.frameBytes);assert.equal(await old,null,'Old pixel verification escaped after source switch');break;}
  case'original_source_frames':{const bad=structuredClone(a.original);bad.source_frames[0].image_sha256='e'.repeat(64);assert.throws(()=>mod.validateOriginalComparison(bad,a.report,a.binding),/frame hash/);break;}
  case'valid':await c.setBinding(a.binding);assert.equal(last().status,'ready');assert.equal(last().report.case_id,a.binding.caseId);await c.selectFrame('frame-a');assert.deepEqual(new Uint8Array(last().frame.bytes),a.frameBytes);assert.equal(calls.length,3);break;
  case'unsupported_source':await c.setBinding({...a.binding,sourceSha256:'c'.repeat(64)});assert.equal(last().status,'unsupported');assert.equal(calls.length,0);break;
  case'unsupported_component':await c.setBinding({...a.binding,trackId:'other'});assert.equal(last().status,'unsupported');assert.equal(calls.length,0);break;
  case'invalid_binding':await c.setBinding({...a.binding,caseId:'../escape'});assert.equal(last().status,'unsupported');assert.equal(calls.length,0);break;
  case'wrong_report_source':a.report.source.sha256='c'.repeat(64);{const bytes=B(JSON.stringify(a.report));routes.set('/files/'+a.row.report.path,bytes);a.row.report.sha256=await digest(bytes);}await c.setBinding(a.binding);assert.equal(last().status,'error');assert.equal(last().report,null);break;
  case'wrong_report_hash':routes.set('/files/'+a.row.report.path,B('changed'));await c.setBinding(a.binding);assert.equal(last().status,'error');assert.match(last().message,/changed/);break;
  case'wrong_raw_hash':routes.set('/files/'+a.report.provenance.report.path,B('changed'));await c.setBinding(a.binding);assert.equal(last().status,'error');assert.equal(last().report,null);break;
  case'wrong_frame_hash':await c.setBinding(a.binding);routes.set('/files/'+a.report.frames[0].path,B('changed'));await c.selectFrame('frame-a');assert.equal(last().frame,null);assert.match(last().frameError,/changed/);break;
  case'clear_pending':await c.setBinding(a.binding);const wait=deferred();routes.set('/files/'+b.row.report.path,wait.promise);const work=c.setBinding(b.binding);assert.equal(last().status,'loading');assert.equal(last().report,null,'Old camera result remains visible while another source is pending');assert.equal(last().frame,null);wait.resolve(b.bytes);await work;break;
  case'stale_report':{const wait=deferred();routes.set('/files/'+a.row.report.path,wait.promise);const old=c.setBinding(a.binding);await c.setBinding(b.binding);wait.resolve(a.bytes);await old;assert.equal(last().report.case_id,b.binding.caseId,'Stale report replaced newer source');break;}
  case'stale_original':{const wait=deferred();routes.set('/files/'+a.report.provenance.report.path,wait.promise);const seen=deferred();entered.set('/files/'+a.report.provenance.report.path,seen);const old=c.setBinding(a.binding);await seen.promise;assert.equal(calls.length,2);await c.setBinding(b.binding);wait.resolve(a.rawBytes);await old;assert.equal(last().report.case_id,b.binding.caseId,'Stale original-report verification replaced newer source');break;}
  case'stale_frame':{await c.setBinding(a.binding);const wait=deferred();routes.set('/files/'+a.report.frames[0].path,wait.promise);const old=c.selectFrame('frame-a');await c.setBinding(b.binding);wait.resolve(a.frameBytes);await old;assert.equal(last().report.case_id,b.binding.caseId);assert.equal(last().frame,null,'Old source frame reappeared after source switch');break;}
  case'frame_cleared_pending':{await c.setBinding(a.binding);await c.selectFrame('frame-a');const wait=deferred();routes.set('/files/'+a.report.frames[0].path,wait.promise);const old=c.selectFrame('frame-a');assert.equal(last().frame,null);assert.equal(last().frameLoading,true);wait.resolve(a.frameBytes);await old;break;}
  case'dispose':{const wait=deferred();routes.set('/files/'+a.row.report.path,wait.promise);const work=c.setBinding(a.binding);const n=states.length;c.dispose();wait.resolve(a.bytes);await work;assert.equal(states.length,n);break;}
  case'claim_scope':for(const field of ['physical_accuracy_verified','object_metric_scale_established','applied_to_geometry']){const bad=structuredClone(a.report);bad[field]=true;assert.throws(()=>mod.validateCameraReport(bad,a.binding),/diagnostic scope/);}break;
  case'empty_metric':{const bad=structuredClone(a.report);bad.hypotheses[0].reserved.observations=0;assert.throws(()=>mod.validateCameraReport(bad,a.binding),/Empty camera check/);break;}
  case'path_confinement':for(const bad of ['https://example.com/report.json','data/../.secrets/meta.env','data/.secrets/meta.env','data/camera/r.json?q=1','data/camera/%2e%2e/file','/data/camera/report.json','data/camera/./r.json','data/camera//r.json'])assert.throws(()=>mod.evidenceURL(bad),/Unexpected/);assert.equal(calls.length,0);break;
  case'duplicate_registry':{const d=mod.createAutomaticCameraController({records:[a.row,a.row],fetchImpl,commit:s=>states.push(s)});await d.setBinding(a.binding);assert.equal(last().status,'unsupported');assert.equal(calls.length,0);break;}
  default:throw Error('Unknown scenario '+scenario);
 }
 console.log('PASS '+scenario);
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
