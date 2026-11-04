/* Model Lab: local orchestration only. No reasoning, model or cluster API calls. */
(function(root){'use strict';
const MODES=new Set(['refit_existing','local_triposr','imported_sam3d','refit_native_cpu']);
const TERMINAL=new Set(['complete','complete_with_gaps','failed','cancelled','interrupted']);
const LIMIT=4*1024*1024;
const text=value=>typeof value==='string'?value:String(value??'');
const finite=value=>typeof value==='number'&&Number.isFinite(value);
function localURL(value){
 if(typeof value!=='string'||!value.startsWith('/')||value.startsWith('//')||/[\\\x00-\x1f]/.test(value))return null;
 const path=value.split(/[?#]/)[0];
 if(!/^\/(files\/data\/|workspace\/|api\/upgrades\/)/.test(path)||path.split('/').some(s=>s==='.'||s==='..'||/%(?:2e|2f|5c)/i.test(s)))return null;
 return value;
}
function mediaURL(value){
 if(typeof value!=='string'||!value.startsWith('data/')||value.split('/').some(s=>!s||s==='.'||s==='..'||s.startsWith('.'))||/[\\\x00-\x1f?#]/.test(value))return null;
 return (root.location?.protocol==='file:'?'../':'/files/')+value.split('/').map(encodeURIComponent).join('/');
}
function savedMediaURL(value){return root.location?.protocol==='file:'&&typeof value==='string'&&value.startsWith('../data/')?mediaURL(value.slice(3)):null;}
function requestFor(project,component,mode,imported){
 if(!project||!component||!MODES.has(mode)||!/^[a-f0-9]{64}$/.test(project.base_sha256||''))throw new Error('Choose a valid published project and component.');
 const request={twin_id:project.id,track_id:component.track_id,base_sha256:project.base_sha256,mode};
 if(mode==='imported_sam3d'){
  if(!matchingImport(imported,project,component))throw new Error('Choose a validated SAM 3D result with compatible appearance for this exact component and original program.');
  request.imported_result_id=imported.import_id||imported.id;
 }
 return request;
}
function matchingNativeImport(item,project,component){
 return !!(item&&project&&component&&item.backend==='sam3d_objects'&&item.fixture!==true&&item.twin_id===project.id&&item.track_id===component.track_id&&item.base_sha256===project.base_sha256&&item.status==='validated_candidate_pending_alignment'&&/^[A-Za-z0-9_-]{1,140}$/.test(item.import_id||item.id||''));
}
function matchingImport(item,project,component){return matchingNativeImport(item,project,component)&&item.alignment_eligible===true;}
function formatMetric(value,unit){
 if(value===null||value===undefined)return 'Not available';
 if(!finite(value))return 'Not available';
 if(unit==='%'||unit==='percent')return value.toFixed(1)+'%';
 if(unit==='px')return value.toFixed(2)+' px';
 if(unit==='count'||unit==='samples'||Number.isInteger(value))return String(value);
 return value.toFixed(3);
}
function progressValue(value){return finite(value)?Math.max(0,Math.min(100,value<=1?value*100:value)):0;}
function fallbackProjects(cases){return (cases||[]).map(c=>({id:c.id,title:c.title,base_sha256:c.original_sha256,video_url:mediaURL(c.video_path),poster_url:mediaURL(c.poster_path),duration_s:c.duration_s,components:(c.interaction?.tracklets||[]).map(t=>{const object=c.scene?.objects.find(o=>o.object_id===t.id);const anchor=c.interaction.evidence?.find(e=>e.observations?.some(o=>o.track_id===t.id));return{track_id:t.id,label:t.label,has_asset:!!object,backend:object?.provenance?.model_backend||'No generated model',source_frame_url:mediaURL(anchor?.frame_path||c.poster_path),source_timestamp_s:anchor?.timestamp_s,baseline_metrics:{conditional_supported:object?.frames.filter(f=>f.visible&&!f.reference).length||0,reference_samples:object?.frames.filter(f=>f.visible&&f.reference).length||0,total_samples:c.interaction.evidence?.length||0,silhouette_iou_median:null,residual_px_median:null}}})}));}
async function readJSON(response,limit=LIMIT){
 const contentLength=Number(response.headers?.get?.('content-length'));
 if(contentLength>limit)throw new Error('The local response is too large to display safely.');
 let result='';
 if(response.body?.getReader){const reader=response.body.getReader();const decoder=new TextDecoder();let bytes=0;try{while(true){const part=await reader.read();if(part.done)break;bytes+=part.value.byteLength;if(bytes>limit){await reader.cancel();throw new Error('The local response is too large to display safely.')}result+=decoder.decode(part.value,{stream:true})}result+=decoder.decode();}finally{reader.releaseLock?.();}}
 else{result=await response.text();if(result.length>limit)throw new Error('The local response is too large to display safely.');}
 let value;try{value=JSON.parse(result)}catch(_){throw new Error('The local processor returned an unreadable response.');}
 if(!response.ok)throw new Error(text(value.error||value.message||'The local request failed ('+response.status+').').slice(0,1800));
 return value;
}
async function api(path,options={}){
 if(root.location?.protocol==='file:')throw new Error('Start the local workspace launcher to enable processing. Saved models remain available here.');
 if(!path.startsWith('/api/upgrades/'))throw new Error('Unexpected local endpoint.');
 const {responseLimit=LIMIT,...fetchOptions}=options;
 const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),45000);
 try{return await readJSON(await root.fetch(path,{credentials:'same-origin',cache:'no-store',...fetchOptions,signal:controller.signal}),responseLimit);}
 catch(error){if(error.name==='AbortError')throw new Error('The local processor did not respond in time. Refresh status before retrying; a job may already have been created.');throw error;}
 finally{clearTimeout(timer);}
}
const ModelLab={localURL,mediaURL,requestFor,matchingImport,formatMetric,progressValue,fallbackProjects,readJSON,MODES};
if(typeof module!=='undefined'&&module.exports)module.exports=ModelLab;
root.ModelLab=ModelLab;
if(!root.document)return;
const $=id=>document.getElementById(id),node=(tag,value,className)=>{const n=document.createElement(tag);if(value!==undefined)n.textContent=text(value);if(className)n.className=className;return n;};
const state={projects:[],backends:[],project:null,component:null,mode:'refit_existing',plan:null,runs:[],jobs:[],imports:[],online:false,busy:false,refreshing:false,version:0,renderers:{before:null,after:null},sceneKeys:{before:null,after:null},scenes:new Map(),sceneRequests:new Map(),selectedRun:null,poll:null};
const savedCases=root.TWIN_CASES?.cases||[];
function message(id,value,error=false){const el=$(id);el.textContent=value||'';el.hidden=!value;el.classList.toggle('error',!!error);}
function errorMessage(error){const value=text(error?.message||error);return value==='Missing or oversized result receipt'?'This ZIP has no returned model result. Run the GPU input bundle, then import the ZIP created by its result packer.':value.slice(0,1800);}
function projectCase(id){return state.scenes.get(id)||(root.TWIN_CASES?.cases||savedCases).find(c=>c.id===id);}
function selectedImport(){return state.imports.find(x=>(x.import_id||x.id)===$('import-select').value);}
function invalidatePlan(){state.version++;state.plan=null;$('plan-stages').replaceChildren(node('li','Inspect the plan to check cached inputs and required model stages.','plan-placeholder'));$('plan-caption').textContent='Nothing starts until you inspect and approve this plan by pressing Start.';$('plan-sampling-note').hidden=true;message('plan-blockers','');setReadiness();}
function setReadiness(){
 const chosen=state.project&&state.component;
 const importReady=state.mode!=='imported_sam3d'||matchingImport(selectedImport(),state.project,state.component);
 const nativeOnly=state.imports.find(item=>matchingNativeImport(item,state.project,state.component)&&item.alignment_eligible!==true);
 const activeRun=state.runs.find(r=>!TERMINAL.has(r.status));
 $('inspect-plan').disabled=!state.online||!chosen||state.busy||!importReady;
 $('prepare-job').disabled=!state.online||!chosen||state.busy;
 $('start-run').disabled=!state.online||!state.plan?.can_run||state.busy||!!activeRun;
 const reason=!state.online?'Start the local workspace launcher to run model stages.':state.busy?'Saving the current request…':activeRun?'A local upgrade is already running. Follow its preserved progress below.':!importReady?(nativeOnly?'Native model imported for inspection. '+text(nativeOnly.alignment_ineligible_reason||'Its appearance does not meet the current local alignment contract.'):'Prepare a SAM 3D input bundle and import its genuine result before planning alignment.'):state.plan?.can_run?'Ready. Reuse cached reasoning and run only the planned local stages.':state.plan?.blockers?.length?state.plan.blockers.map(text).join(' '):'Inspect a plan to check prerequisites.';
 $('run-readiness').textContent=reason;
 const matches=state.imports.filter(item=>matchingImport(item,state.project,state.component));
 $('import-mode-chip').textContent=matches.length?'VALIDATED RESULT AVAILABLE':nativeOnly?'NATIVE INSPECTION ONLY':'RESULT REQUIRED';
 $('import-mode-chip').classList.toggle('lavender',matches.length>0);
 $('import-choice-row').hidden=state.mode!=='imported_sam3d';
 $('import-select').disabled=!matches.length||state.busy;
}
function selectOptions(element,rows,labelKey,valueKey,current){element.replaceChildren();for(const row of rows){const option=node('option',row[labelKey]);option.value=row[valueKey];element.append(option);}if(rows.some(r=>r[valueKey]===current))element.value=current;}
function renderImportChoice(){const previous=$('import-select').value;const matching=state.imports.filter(item=>matchingImport(item,state.project,state.component));const nativeOnly=state.imports.some(item=>matchingNativeImport(item,state.project,state.component)&&item.alignment_eligible!==true);$('import-select').replaceChildren();if(!matching.length){const option=node('option',nativeOnly?'Native model needs a compatible alignment adapter':'No matching SAM 3D result imported');option.value='';$('import-select').append(option);}else for(const item of matching){const id=item.import_id||item.id;const option=node('option',id+' · original output validated');option.value=id;$('import-select').append(option);}if(matching.some(i=>(i.import_id||i.id)===previous))$('import-select').value=previous;setReadiness();}
function scenePanel(which,caseData,track){
 const key=caseData?.id&&track?caseData.id+'::'+track:null;
 if(key&&key===state.sceneKeys[which]&&state.renderers[which])return true;
 state.renderers[which]?.dispose();state.renderers[which]=null;
 state.sceneKeys[which]=null;
 const empty=$(which+'-empty');empty.hidden=false;
 $('reset-'+which).disabled=true;
 const object=caseData?.scene?.objects.find(o=>o.object_id===track);
 if(!object)return false;
 try{const renderer=new root.TwinScene.Scene($(which+'-canvas'),caseData.scene,value=>{empty.replaceChildren(node('strong','The mesh could not be displayed'),node('p',value));empty.hidden=false;});renderer.inspect('native',track);state.renderers[which]=renderer;state.sceneKeys[which]=key;empty.hidden=true;$('reset-'+which).disabled=false;return true;}
 catch(error){empty.replaceChildren(node('strong','3D preview unavailable'),node('p',errorMessage(error)));return false;}
}
function baselineStats(){
 const m=state.component?.baseline_metrics||{},host=$('baseline-stats');host.replaceChildren();
 const coverage=node('strong',finite(m.conditional_supported)?String(m.conditional_supported):'—');coverage.append(node('small',' / '+(finite(m.total_samples)?m.total_samples:'—')+' source samples'));
 for(const [value,label]of[[coverage,'Conditional non-reference poses'],[formatMetric(m.silhouette_iou_median,'IoU'),'Median silhouette IoU'],[formatMetric(m.residual_px_median,'px'),'Median reprojection residual'],[finite(m.reference_samples)?String(m.reference_samples):'—','Reference initializations · unverified']]){const box=node('div',undefined,'stat');box.append(typeof value==='string'?node('strong',value):value,node('span',label));host.append(box);}
}
function showProject(id,track){
 const project=state.projects.find(p=>p.id===id)||state.projects[0];if(!project)return;
 state.project=project;$('project-select').value=project.id;
 selectOptions($('component-select'),project.components||[],'label','track_id',track);$('component-select').disabled=!project.components?.length;
 state.component=(project.components||[]).find(c=>c.track_id===$('component-select').value)||null;
 renderSelection();
}
function renderSelection(){
 invalidatePlan();renderImportChoice();const p=state.project,c=state.component;
 $('baseline-backend').textContent=c?.backend||'No generated model';$('program-hash').textContent=p.base_sha256?.slice(0,12)+'…'+p.base_sha256?.slice(-8);$('program-hash').title=p.base_sha256||'';
 $('source-count').textContent=(c?.baseline_metrics?.total_samples??'—')+' preserved source samples';
 $('open-workspace').href='index.html?case='+encodeURIComponent(p.id);
 $('source-video').pause();const poster=localURL(c?.source_frame_url)||savedMediaURL(c?.source_frame_url)||localURL(p.poster_url)||(root.location.protocol==='file:'?mediaURL(projectCase(p.id)?.poster_path):null);const video=localURL(p.video_url)||savedMediaURL(p.video_url)||(root.location.protocol==='file:'?mediaURL(projectCase(p.id)?.video_path):null);
 $('source-frame').hidden=!poster;if(poster){$('source-frame').src=poster;$('source-frame').alt=(c?.label||'Component')+' source anchor at '+(finite(c?.source_timestamp_s)?c.source_timestamp_s.toFixed(6):'unknown')+' seconds';}else $('source-frame').removeAttribute('src');
 $('source-empty').hidden=!!poster;$('source-video').hidden=true;
 if(video){$('source-video').src=video;$('source-video').poster=poster||'';$('source-video').load();}else $('source-video').removeAttribute('src');
 $('source-toggle').disabled=!video;$('source-toggle').textContent='Play recording';$('source-time').hidden=!finite(c?.source_timestamp_s);$('source-time').textContent=finite(c?.source_timestamp_s)?c.source_timestamp_s.toFixed(6)+' s':'';
 $('source-caption').textContent=finite(c?.source_timestamp_s)?'Preserved source frame · '+c.source_timestamp_s.toFixed(3)+' s':'Original source evidence';
 scenePanel('before',projectCase(p.id),c?.track_id);$('before-caption').textContent=(c?.backend||'No asset')+' · native coordinates';
 baselineStats();state.selectedRun=null;renderActivities();renderComparison();updateLocation();
}
function updateLocation(){if(root.location.protocol==='file:')return;const params=new URLSearchParams(root.location.search);params.set('case',state.project.id);if(state.component)params.set('track',state.component.track_id);root.history.replaceState(null,'','?'+params);}
function renderPlan(plan){
 state.plan=plan;$('plan-caption').textContent=plan.can_run?'Inputs checked. Review the stages, then start when ready.':'Some prerequisites are missing. Your source artifacts remain available.';
 const host=$('plan-stages');host.replaceChildren();
 for(const stage of plan.stages||[]){const action=['reuse','run','blocked'].includes(stage.action)?stage.action:'blocked';const li=node('li');li.append(node('span',action==='reuse'?'REUSE':action==='run'?'RUN LOCALLY':'REQUIRED','action-marker '+action),node('strong',stage.label||stage.id),node('p',stage.reason||''));host.append(li);}
 if(!host.children.length)host.append(node('li','The processor returned no stage details. Refresh the plan before starting.','plan-placeholder'));
 $('plan-sampling-note').hidden=!plan.source_sampling_may_differ_from_baseline;$('plan-sampling-note').textContent='This model’s original frame sampling differs from the baseline. Motion uses that saved sampling; comparison uses matching source times only.';
 message('plan-blockers',(plan.blockers||[]).map(text).join('\n'));setReadiness();
}
async function inspectPlan(){const version=state.version;state.busy=true;setReadiness();message('action-message','');try{const request=requestFor(state.project,state.component,state.mode,selectedImport());const plan=await api('/api/upgrades/plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});if(version!==state.version)return;renderPlan(plan);}catch(error){if(version===state.version){message('plan-blockers',errorMessage(error),true);state.plan=null;}}finally{state.busy=false;setReadiness();}}
async function startRun(){if(!state.plan?.can_run||state.busy)return;const version=state.version;const request=requestFor(state.project,state.component,state.mode,selectedImport());state.busy=true;setReadiness();message('action-message','');try{const result=await api('/api/upgrades/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});message('action-message','Local upgrade started. The progress below records real stages; you can leave this page and return later.');if(version===state.version)state.selectedRun=result.id||result.run?.id||null;await refreshActivity();}catch(error){message('action-message',errorMessage(error),true);}finally{state.busy=false;setReadiness();}}
async function prepareJob(){if(state.busy||!state.project||!state.component)return;const p=state.project,c=state.component,backend=$('gpu-backend').value;const request={twin_id:p.id,track_id:c.track_id,base_sha256:p.base_sha256,backend};if(backend==='sam31_image'){const prompt=$('mask-prompt').value.trim();if(!prompt){message('action-message','Describe the visible component for the SAM 3.1 anchor-mask comparison.',true);$('mask-prompt').focus();return;}request.prompt=prompt;}
 if(backend==='sam3d_objects')request.appearance=$('gpu-appearance').value||'vertex-color';
 state.busy=true;setReadiness();message('action-message','Preparing the local bundle…');try{await api('/api/upgrades/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});message('action-message','Input bundle prepared. Download it below. The GPU job has not been submitted or run.');await refreshActivity();}catch(error){message('action-message',errorMessage(error),true);}finally{state.busy=false;setReadiness();}}
function prettyStatus(status){return text(status||'unknown').replace(/_/g,' ');}
function matchingRecord(record){const context=record.request||record;return context.twin_id===state.project?.id&&context.track_id===state.component?.track_id;}
function addLink(host,label,url){const safe=localURL(url);if(!safe)return;const link=node('a',label);link.href=safe;host.append(link);}
function summaryLog(record){const phases=Array.isArray(record.phases)?record.phases:Object.entries(record.phases||{}).map(([id,value])=>({id,...(typeof value==='object'?value:{status:value})}));return [record.phase?('Current stage: '+record.phase):'',...phases.map(p=>[p.label||p.id,p.status,p.message||p.error].filter(Boolean).join(' · ')),record.error?'Error: '+text(record.error):''].filter(Boolean).join('\n').slice(0,15000);}
async function cancelRun(id,button){button.disabled=true;try{await api('/api/upgrades/runs/'+encodeURIComponent(id)+'/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});await refreshActivity();}catch(error){message('action-message',errorMessage(error),true);button.disabled=false;}}
function resumeRequest(run){
 if(!['failed','cancelled','interrupted'].includes(run.status)||text(run.result?.publication_status).startsWith('superseded_'))return null;
 const request=run.request;
 if(!request||request.base_sha256!==state.project?.base_sha256||!matchingRecord(run))return null;
 try{return requestFor(state.project,state.component,request.mode,state.imports.find(i=>(i.import_id||i.id)===request.imported_result_id));}catch(_){return null;}
}
async function resumeRun(run,button){
 const request=resumeRequest(run);if(!request||state.busy)return;
 button.disabled=true;state.busy=true;setReadiness();
 try{await api('/api/upgrades/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...request,resume_run_id:run.id||run.run_id})});message('action-message','Resuming after rechecking the same request and saved stage hashes. Completed valid stages are reused; no new reasoning is requested.');await refreshActivity();}
 catch(error){message('action-message',errorMessage(error),true);button.disabled=false;}
 finally{state.busy=false;setReadiness();}
}
async function importResult(job,file,input){if(!file)return;if(!file.name.toLowerCase().endsWith('.zip')){message('action-message','Choose the result ZIP produced by this job’s GPU workflow.',true);input.value='';return;}if(file.size>512*1024*1024){message('action-message','This result exceeds the 512 MiB result-package limit. No upload was made.',true);input.value='';return;}input.disabled=true;message('action-message','Validating this result against its exact job and input hashes…');try{const response=await api('/api/upgrades/jobs/'+encodeURIComponent(job.job_id||job.id)+'/result',{method:'POST',headers:{'Content-Type':'application/zip'},body:file});const result=response.import||response;const note=result.backend==='sam31_image'?'SAM 3.1 result imported as an anchor-mask comparison only.':result.backend==='sam3d_objects'&&result.alignment_eligible===true?'SAM 3D result validated and compatible with local alignment. Select it above to inspect a reuse plan.':result.backend==='sam3d_objects'?'SAM 3D native asset imported for inspection. '+text(result.alignment_ineligible_reason||'Its appearance does not meet the current local alignment contract.'):'Result imported and validated. Check its exact status below before choosing a next stage.';message('action-message',note);await refreshActivity();renderImportChoice();invalidatePlan();}catch(error){message('action-message',errorMessage(error),true);}finally{input.value='';input.disabled=false;}}
function renderActivities(){
 const host=$('activity-list');host.replaceChildren();const runs=state.runs.filter(matchingRecord),jobs=state.jobs.filter(matchingRecord);$('activity-count').textContent=runs.length+' local run'+(runs.length===1?'':'s')+' · '+jobs.length+' prepared handoff'+(jobs.length===1?'':'s');
 for(const run of runs){const id=run.id||run.run_id,mode=run.request?.mode||run.mode,card=node('article',undefined,'activity-card');const top=node('div',undefined,'activity-top'),title=node('div');title.append(node('strong',mode==='refit_existing'?'Refit current mesh':mode==='local_triposr'?'Local TripoSR reconstruction':mode==='refit_native_cpu'?'Native-material CPU refit':'Imported SAM 3D alignment'),node('small',id+' · '+(state.component?.label||run.request?.track_id||run.track_id)));top.append(title,node('span',prettyStatus(run.status).toUpperCase(),'chip '+(run.status==='failed'?'amber':TERMINAL.has(run.status)?'':'lavender')));card.append(top);const progress=node('div',undefined,'progress-track'),bar=node('span');bar.style.width=progressValue(run.progress)+'%';progress.append(bar);card.append(progress);if(run.message)card.append(node('p',run.message,'activity-message'));const actions=node('div',undefined,'activity-actions');if(!TERMINAL.has(run.status)){const cancel=node('button','Cancel local run');cancel.onclick=()=>cancelRun(id,cancel);actions.append(cancel);}if(resumeRequest(run)){const resume=node('button','Resume checked stages');resume.disabled=state.busy||state.runs.some(r=>!TERMINAL.has(r.status));resume.onclick=()=>resumeRun(run,resume);actions.append(resume);}if(run.result?.comparison){const compare=node('button','Inspect this comparison →');compare.onclick=()=>{state.selectedRun=id;renderComparison();$('compare-section').scrollIntoView({behavior:'smooth',block:'start'});};actions.append(compare);}addLink(actions,'Open result in workspace ↗',run.result?.workspace_url);addLink(actions,'Comparison receipt ↗',run.result?.comparison_url);card.append(actions);const log=summaryLog(run);if(log){const details=node('details'),summary=node('summary','Stage details & preserved failures');details.append(summary,node('pre',log));card.append(details);}host.append(card);}
 for(const job of jobs){const id=job.job_id||job.id,card=node('article',undefined,'activity-card'),top=node('div',undefined,'activity-top'),title=node('div');title.append(node('strong',job.backend_label||(job.backend==='sam3d_objects'?'SAM 3D Objects input bundle':'SAM 3.1 anchor-mask input bundle')),node('small',id+(finite(job.source?.timestamp_s)?' · '+job.source.timestamp_s.toFixed(6)+' s':'')));const valid=job.status==='prepared_not_submitted';top.append(title,node('span',valid?'INPUTS PREPARED':'INVALID SAVED JOB','chip '+(valid?'lavender':'amber')));card.append(top);if(!valid){card.append(node('p','This saved job failed validation. Its inputs cannot be downloaded or used for result import.','activity-message'));host.append(card);continue;}const imports=state.imports.filter(item=>item.job_id===id&&['validated_candidate_pending_alignment','validated_anchor_comparison_only'].includes(item.status)&&item.fixture!==true);card.append(node('p',imports.length?imports.length+' validated result'+(imports.length===1?'':'s')+' returned. Original model bytes and input bindings were checked.':'Run this bundle on your GPU when available. This page does not submit or monitor a cluster job.','activity-message'));const actions=node('div',undefined,'activity-actions');addLink(actions,'Download input ZIP ↓',job.download_url||(job.zip_relative_path?mediaURL(job.zip_relative_path):null));if(job.instructions_url)addLink(actions,'Instructions ↗',job.instructions_url);const fileLabel=node('label','Import result ZIP','file-label'),input=node('input');input.type='file';input.accept='.zip,application/zip';input.setAttribute('aria-label','Import result ZIP for '+id);input.onchange=()=>importResult(job,input.files?.[0],input);fileLabel.append(input);actions.append(fileLabel);card.append(actions);for(const item of imports){const imported=node('div',undefined,'activity-message');const sam3d=item.backend==='sam3d_objects',eligible=sam3d&&item.alignment_eligible===true;imported.append(node('span',sam3d?(eligible?'SAM 3D output · compatible with alignment':'Native SAM 3D asset · inspection only'):'SAM 3.1 anchor comparison only','chip '+(eligible?'':'subdued')),node('span',' '+(item.import_id||item.id)));if(sam3d&&!eligible)imported.append(node('p',item.alignment_ineligible_reason||'The imported appearance does not meet the current local alignment contract.'));addLink(imported,' · Native GLB ↓',item.asset_url||item.native_asset_url||(item.native_asset_path?mediaURL(item.native_asset_path):null));addLink(imported,' · Inspect mask comparison ↗',item.comparison_url);card.append(imported);}host.append(card);}
 if(!runs.length&&!jobs.length)host.append(node('div','No work has started for this component. Inspect a plan or prepare a GPU handoff.','empty-activity'));
 if(!state.selectedRun||!runs.some(r=>(r.id||r.run_id)===state.selectedRun))state.selectedRun=(runs.find(r=>r.result?.comparison)?.id)||null;
}
function renderComparison(){
 const run=state.runs.find(r=>(r.id||r.run_id)===state.selectedRun&&matchingRecord(r)),comparison=run?.result?.comparison;
 $('comparison-empty').hidden=!!comparison;$('comparison-results').hidden=!comparison;$('comparison-chip').textContent=comparison?'INTERNAL SOURCE-FRAME DIAGNOSTICS':'AWAITING A COMPLETED RUN';
 const resultId=run?.result?.workspace_case_id||run?.result?.result_case_id;const resultCase=projectCase(resultId);const hasAfter=scenePanel('after',resultCase,state.component?.track_id);
 $('after-caption').textContent=hasAfter?(['refit_existing','refit_native_cpu'].includes(run?.request?.mode||run?.mode)?'Same mesh · refitted motion':'Untouched candidate · native coordinates'):comparison?'Result saved · reload this page for its 3D preview':'No result has been imported';$('after-chip').textContent=hasAfter?'CANDIDATE':comparison?'SAVED':'PENDING';
 if(!comparison)return;
 if(!resultCase&&resultId&&state.online)loadResultScene(resultId,run.id||run.run_id);
 const count=finite(comparison.matched_sample_count)?comparison.matched_sample_count:comparison.timestamps_s?.length;const summary=$('comparison-summary');summary.replaceChildren();const copy=node('div');copy.append(node('strong',(finite(count)?count:'—')+' matched source timestamps'),node('p','Reference initializations excluded. Comparison does not verify physical pose or contact.'));summary.append(copy,node('span',prettyStatus(run.status),'chip subdued'));
 const host=$('comparison-rows');host.replaceChildren();for(const metric of comparison.metrics||[]){let population=metric.denominator!==undefined?'Denominator: '+text(metric.denominator):(finite(count)?count+' matched samples':'See comparison receipt');if(['IoU','px'].includes(metric.unit)&&finite(comparison.baseline?.conditional_supported)&&finite(comparison.candidate?.conditional_supported))population='Supported subsets: '+comparison.baseline.conditional_supported+' baseline / '+comparison.candidate.conditional_supported+' candidate of '+count+' matched';const row=node('tr');row.append(node('td',metric.label||metric.name||'Diagnostic'),node('td',formatMetric(metric.before,metric.unit)),node('td',formatMetric(metric.after,metric.unit)),node('td',population));host.append(row);}if(!host.children.length){const row=node('tr'),cell=node('td','No comparable fit measurements are available. Inspect the saved failure and coverage records.');cell.colSpan=4;row.append(cell);host.append(row);}
 const links=$('comparison-links');links.replaceChildren();addLink(links,'Open candidate in workspace ↗',run.result.workspace_url);addLink(links,'Read exact comparison receipt ↗',run.result.comparison_url);if(!hasAfter){const reload=node('button','Reload saved 3D preview','secondary');reload.onclick=()=>root.location.reload();links.append(reload);}
 $('comparison-scope').textContent=[comparison.metric_population_note,comparison.scope||'Internal source-frame diagnostics use model-dependent geometry, cameras and fitting evidence. These are not physical ground truth. Unknown poses remain absent.'].filter(Boolean).map(text).join(' ');
}
async function loadResultScene(caseId,runId){
 if(state.sceneRequests.has(caseId))return;
 const binding={project:state.project?.id,track:state.component?.track_id,run:runId};
 const task=api('/api/upgrades/scene?case='+encodeURIComponent(caseId),{responseLimit:16*1024*1024});state.sceneRequests.set(caseId,task);
 try{const response=await task;if(response.case?.id!==caseId||!Array.isArray(response.case?.scene?.objects))throw new Error('The result scene does not match its published case.');state.scenes.set(caseId,response.case);if(state.project?.id===binding.project&&state.component?.track_id===binding.track&&state.selectedRun===binding.run)renderComparison();}
 catch(error){if(state.project?.id===binding.project&&state.component?.track_id===binding.track&&state.selectedRun===binding.run){$('after-caption').textContent='Result preserved · 3D preview unavailable';$('after-empty').replaceChildren(node('strong','Could not load candidate preview'),node('p',errorMessage(error)));$('after-empty').hidden=false;}}
}
async function refreshActivity(){
 if(!state.online||state.refreshing)return;state.refreshing=true;
 try{const responses=await Promise.allSettled([api('/api/upgrades/runs'),api('/api/upgrades/jobs')]);const [runs,jobs]=responses;if(runs.status==='fulfilled')state.runs=runs.value.runs||[];if(jobs.status==='fulfilled'){state.jobs=jobs.value.jobs||[];state.imports=jobs.value.imports||[];}if(runs.status==='rejected'&&jobs.status==='rejected')throw runs.reason;renderImportChoice();renderActivities();renderComparison();schedulePoll();}
 catch(error){message('action-message','Could not refresh local progress: '+errorMessage(error),true);}
 finally{state.refreshing=false;}
}
function schedulePoll(){if(state.poll)clearTimeout(state.poll);state.poll=null;if(state.runs.some(r=>!TERMINAL.has(r.status))&&!document.hidden)state.poll=setTimeout(refreshActivity,3000);}
async function connect(){
 $('refresh-button').disabled=true;
 try{const catalog=await api('/api/upgrades/catalog');if(!Array.isArray(catalog.projects)||!catalog.projects.length)throw new Error('The local processor has no eligible saved projects. Open an existing project in the workspace first.');state.online=true;state.projects=catalog.projects;state.backends=catalog.backends||[];$('connection').textContent='Local processor connected';$('connection').prepend(node('i'));$('connection').classList.add('online');message('global-message','');const previous=state.project?.id;const track=state.component?.track_id;selectOptions($('project-select'),state.projects,'title','id',previous);$('project-select').disabled=false;const params=new URLSearchParams(root.location.search);showProject(previous||params.get('case')||params.get('twin'),track||params.get('track'));await refreshActivity();}
 catch(error){state.online=false;$('connection').textContent='Saved review only';$('connection').prepend(node('i'));$('connection').classList.remove('online');message('global-message',root.location.protocol==='file:'?'Saved review mode. Open “Launch Workspace” to prepare GPU bundles, import results and run cached local upgrades. No network requests are made in this mode.':'The local upgrade processor is unavailable. Saved models are still inspectable. Start or restart the workspace launcher, then press Refresh local status. '+errorMessage(error));if(!state.projects.length){state.projects=fallbackProjects(savedCases);selectOptions($('project-select'),state.projects,'title','id');$('project-select').disabled=!state.projects.length;const params=new URLSearchParams(root.location.search);showProject(params.get('case'),params.get('track'));}}
 finally{$('refresh-button').disabled=false;setReadiness();}
}
$('project-select').onchange=()=>showProject($('project-select').value);
$('component-select').onchange=()=>{state.component=state.project.components.find(c=>c.track_id===$('component-select').value);renderSelection();};
for(const input of document.querySelectorAll('input[name="upgrade-mode"]'))input.onchange=()=>{if(!input.checked)return;state.mode=input.value;invalidatePlan();renderImportChoice();};
$('import-select').onchange=invalidatePlan;$('inspect-plan').onclick=inspectPlan;$('start-run').onclick=()=>startRun().catch(error=>message('action-message',errorMessage(error),true));$('prepare-job').onclick=prepareJob;$('refresh-button').onclick=connect;
$('gpu-backend').onchange=()=>{$('prompt-row').hidden=$('gpu-backend').value!=='sam31_image';$('appearance-row').hidden=$('gpu-backend').value!=='sam3d_objects';};
$('source-toggle').onclick=()=>{const video=$('source-video');video.hidden=!video.hidden;$('source-frame').hidden=!video.hidden;$('source-toggle').textContent=video.hidden?'Play recording':'Show source anchor';$('source-time').hidden=!video.hidden||!finite(state.component?.source_timestamp_s);if(!video.hidden){video.currentTime=state.component?.source_timestamp_s||0;video.play().catch(()=>{});}else video.pause();};
for(const which of ['before','after'])$('reset-'+which).onclick=()=>state.renderers[which]?.reset();
document.addEventListener('visibilitychange',()=>{if(document.hidden){clearTimeout(state.poll);state.poll=null;}else if(state.online)refreshActivity();});
root.addEventListener('pagehide',()=>{clearTimeout(state.poll);state.renderers.before?.dispose();state.renderers.after?.dispose();});
ModelLab.state=state;ModelLab.refresh=connect;connect();
})(globalThis);
