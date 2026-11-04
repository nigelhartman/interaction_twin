/* Read-only, hash-bound catalog of genuine saved object programs. No inference. */
(function(root){'use strict';
const LIBRARY_SCRIPT_BASE=typeof document==='undefined'?null:new URL('public/',document.currentScript?.src||new URL('/showcase/library.js',location.origin).href).href;
const SHA=/^[a-f0-9]{64}$/, ID=/^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,180}$/;
const assert=(value,message)=>{if(!value)throw Error(message);};
const object=v=>v!==null&&typeof v==='object'&&!Array.isArray(v);
const finite=v=>typeof v==='number'&&Number.isFinite(v);
const TASK_CLASSES=Object.freeze([
 {id:'hinged_assemblies',label:'Hinged assemblies',description:'Parts connected by a proposed revolute joint.',icon:'<path d="m5 19 10-3 12 4-11 5Z"/><path d="m6 13 9-6 12 4-10 6Z"/><path d="m5 17 23-1" stroke-dasharray="2 3"/><circle cx="6" cy="17" r="1.6"/>'},
 {id:'removable_components',label:'Removable components',description:'Separate, remove and return a component.',icon:'<path d="M5 17h10v11H5zM20 4h9v10h-9zM12 13l9-8m-4 0h4v4"/><path d="m18 20 8 8m-5 0h5v-5"/>'},
 {id:'connectors_and_insertion',label:'Connectors & insertion',description:'Source-linked connector and insertion tasks.',icon:'<path d="M4 12h8v9H4m24-9h-8v9h8M12 15h8m-8 3h8M16 4v5m-3-2 3 3 3-3M16 28v-5m-3 2 3-3 3 3"/>'},
 {id:'sliding_mechanisms',label:'Sliding mechanisms',description:'Parts with a proposed linear constraint.',icon:'<path d="m4 10 17-5 8 5-17 5Z"/><path d="M4 10v15l17 4V14M7 18l10 2v5L7 23zM21 14l8-4v15l-8 4M6 4h10m-3-3 3 3-3 3"/>'},
 {id:'flexible_or_uncertain',label:'Flexible / uncertain',description:'Explicitly flexible or unresolved structure.',icon:'<path d="M4 21c4-17 9 12 14-5S27 3 29 16"/><path d="M5 28h9m7 0h6" stroke-dasharray="2 3"/><circle cx="25" cy="24" r="3"/>'}
]);
function taskClasses(record){return Array.isArray(record.task_classes)?record.task_classes:[];}
function ref(r,prefix='data/'){
 assert(object(r)&&typeof r.path==='string'&&r.path.startsWith(prefix)&&/^[a-zA-Z0-9_./-]+$/.test(r.path)&&!r.path.split('/').some(v=>!v||v.startsWith('.'))&&SHA.test(r.sha256)&&Number.isSafeInteger(r.bytes)&&r.bytes>0&&r.bytes<64*1024*1024,'Invalid local artifact binding.');
 return r;
}
function citations(rows,duration){
 assert(Array.isArray(rows)&&rows.length<1000,'Invalid source citations.');
 for(const r of rows){assert(object(r)&&ID.test(r.id)&&finite(r.timestamp_s)&&r.timestamp_s>=0&&r.timestamp_s<=duration+.1,'Invalid source time.');ref(r.frame);}
}
function validateCatalog(d){
 assert(object(d)&&d.format==='interaction-twin-object-program-library'&&d.version===1&&d.read_only===true&&d.no_inference===true&&d.published===false&&d.physical_accuracy_verified===false&&d.simulation_ready===false,'Unsupported library policy.');
 assert(Array.isArray(d.records)&&d.records.length<=100&&object(d.summary),'Invalid catalog records.');
 const ids=new Set(),sources=new Set(),assets=new Set();let parts=0,steps=0,versions=0;
 for(const r of d.records){
  assert(object(r)&&ID.test(r.id)&&!ids.has(r.id)&&typeof r.title==='string'&&typeof r.summary==='string'&&finite(r.duration_s)&&r.duration_s>0,'Invalid object program.');ids.add(r.id);
  if(r.public_title!==undefined)assert(typeof r.public_title==='string'&&r.public_title.trim().length>0&&r.public_title.length<=120,'Invalid curated display title.');
  if(r.public_note!==undefined)assert(typeof r.public_note==='string'&&r.public_note.trim().length>0&&r.public_note.length<=400&&!/[\u0000-\u001f\u007f]/.test(r.public_note),'Invalid curated display note.');
  ref(r.interaction);assert(object(r.source)&&SHA.test(r.source.sha256)&&SHA.test(r.source.original_sha256)&&!sources.has(r.source.original_sha256)&&r.source.time_domain==='source_video_presentation_seconds','Repeated or unbound recording.');sources.add(r.source.original_sha256);
  assert(r.source.path.startsWith('data/')&&!r.source.path.split('/').some(s=>!s||s.startsWith('.'))&&!/[\\%]/.test(r.source.path),'Invalid source path.');ref(r.thumbnail);
  assert(r.physical_accuracy_verified===false&&r.simulation_ready===false&&r.source_video_alignment_added===false&&r.preferred_model_changed===false,'Library must not claim physical validation or promote assets.');
  assert(object(r.reasoning)&&r.reasoning.provider==='meta_api'&&r.reasoning.manual_semantic_story_used===false&&r.reasoning.human_reviewed===false,'Automatic Meta program required.');
  assert(object(r.scene_case)&&r.scene_case.id===r.id&&r.scene_case.original_sha256===r.interaction.sha256,'Scene belongs to a different program.');if(r.scene_case.scene_ref){ref(r.scene_case.scene_ref,'scenes/');assert(r.scene_case.scene_ref.path===`scenes/${r.scene_case.scene_ref.sha256}.json`,'Scene content address differs.');}
  assert(Array.isArray(r.components)&&r.components.length>0&&Array.isArray(r.relationships)&&Array.isArray(r.steps)&&Array.isArray(r.tags)&&r.tags.every(s=>typeof s==='string'),'Invalid program structure.');
  if(r.task_classes!==undefined)assert(Array.isArray(r.task_classes)&&r.task_classes.length<=TASK_CLASSES.length&&new Set(r.task_classes).size===r.task_classes.length&&r.task_classes.every(id=>TASK_CLASSES.some(c=>c.id===id)),'Invalid source-linked task classes.');
  const componentIds=new Set();for(const c of r.components){
   assert(object(c)&&ID.test(c.id)&&!componentIds.has(c.id)&&typeof c.label==='string','Invalid component.');componentIds.add(c.id);
   if(c.asset!==null){const a=ref(c.asset);assert(a.track_id===c.id&&a.manual_geometry_used===false&&a.manual_pose_used===false&&a.physical_accuracy_verified===false&&a.source_video_pose_established_by_this_library===false&&a.mask_model==='sam-3.1','Generated part lacks automatic source provenance.');ref(a.metadata);assert(object(a.anchor)&&finite(a.anchor.timestamp_s)&&a.anchor.timestamp_s>=0&&a.anchor.timestamp_s<=r.duration_s+.1&&a.anchor.time_domain==='source_video_presentation_seconds','Invalid asset source anchor.');ref(a.anchor.image);ref(a.anchor.mask);assets.add(a.sha256);}
  }
  const nativeReviews=r.sam3d_reviews||[];assert(Array.isArray(nativeReviews),'Invalid native review list.');if(root.Sam3DModelReviews)assert(root.Sam3DModelReviews.forRecording(r).length===nativeReviews.length,'Native candidate source or component differs.');
  for(const q of r.relationships){assert(object(q)&&ID.test(q.id)&&componentIds.has(q.subject_track_id)&&componentIds.has(q.object_track_id)&&typeof q.predicate==='string','Invalid component relation.');citations(q.citations,r.duration_s);}
  for(const s of r.steps){assert(object(s)&&ID.test(s.id)&&typeof s.title==='string'&&finite(s.start_s)&&finite(s.end_s)&&s.start_s>=0&&s.end_s>=s.start_s&&s.end_s<=r.duration_s+.1&&Array.isArray(s.participant_track_ids)&&s.participant_track_ids.every(id=>componentIds.has(id)),'Invalid demonstrated action.');citations(s.citations,r.duration_s);}
  assert(object(r.readiness)&&r.readiness.simulation_ready===false&&r.readiness.simulation_runnable===false&&Array.isArray(r.readiness.checks)&&r.readiness.checks.every(c=>object(c)&&typeof c.title==='string'&&typeof c.detail==='string'&&typeof c.next_action==='string'&&['available','unknown','missing'].includes(c.status)),'Invalid readiness inventory.');
  assert(Array.isArray(r.other_saved_versions),'Invalid version list.');for(const v of r.other_saved_versions){assert(ID.test(v.id),'Invalid saved version.');ref(v.interaction);}
  if(r.offline_queue!==null){const q=r.offline_queue;assert(object(q)&&q.remote_submitted===false&&q.inference_executed===false&&q.existing_poses_reused===false&&Array.isArray(q.prepared_bundles)&&Array.isArray(q.unavailable_components)&&q.unavailable_components.every(id=>componentIds.has(id)),'Queue is not an offline preparation.');ref(q.queue);ref(q.plan);for(const b of q.prepared_bundles){assert(componentIds.has(b.track_id),'Unbound queued part.');ref(b.manifest);ref(b.zip);}}
  const inputs=r.initial_reconstruction_inputs||[];assert(Array.isArray(inputs)&&inputs.length<=100,'Invalid prepared input list.');for(const i of inputs){assert(object(i)&&componentIds.has(i.component_id)&&i.status==='initial_reconstruction_prepared'&&i.baseline_required===false&&i.no_inference===true&&i.remote_job_submitted===false&&i.simulation_ready===false&&object(i.source)&&i.source.video_sha256===r.source.sha256&&i.source.time_domain==='source_video_presentation_seconds'&&finite(i.source.timestamp_s)&&i.source.timestamp_s>=0&&i.source.timestamp_s<=r.duration_s+.1,'Prepared source input is not bound to this recording.');ref(i.handoff);}
  assert(Array.isArray(r.cached_insights),'Invalid cached assessment list.');for(const i of r.cached_insights){assert(object(i)&&SHA.test(i.request_key)&&i.human_reviewed===false&&i.robot_commands===false&&typeof i.question==='string'&&i.model==='muse-spark-1.3','Invalid fallible model assessment.');ref(i.analysis);assert(i.analysis.path===`data/task-intelligence/${i.request_key}/analysis.json`,'Cached assessment binding differs.');}
  assert(object(r.counts)&&r.counts.assets>=0&&r.counts.components===r.components.length&&r.counts.assets===r.components.filter(c=>c.asset).length&&r.counts.steps===r.steps.length&&r.counts.relationships===r.relationships.length,'Catalog counts differ from actual records.');
  parts+=r.components.length;steps+=r.steps.length;versions+=1+r.other_saved_versions.length;
 }
 for(const[k,v]of Object.entries({recordings:sources.size,native_asset_drafts:assets.size,components:parts,steps,eligible_saved_versions:versions}))assert(d.summary[k]===v,'Library totals differ from saved artifacts.');
 return d;
}
function matches(record,query='',tag='All recordings'){
 const haystack=[record.title,record.summary,...record.components.map(c=>c.label),...record.steps.map(s=>s.title),...record.relationships.map(r=>r.predicate.replaceAll('_',' '))].join(' ').toLowerCase();
 return(tag==='All recordings'||record.tags.includes(tag))&&query.toLowerCase().trim().split(/\s+/).every(q=>haystack.includes(q));
}
function queryRecords(records,options={}){
 const norm=v=>String(v||'').toLowerCase().trim().replace(/\s+/g,' '),fields=['parts','actions','relations','backends','readiness','task_classes'];
 const words=norm(options.query).split(' ').filter(Boolean);
 let rows=records.filter(r=>{const i=r.search_index||{};return words.every(w=>(i.text||norm([r.title,r.summary,...r.components.map(c=>c.label),...r.steps.map(s=>s.title)].join(' '))).includes(w))&&fields.every(f=>!(options[f]||[]).length||(options[f]||[]).some(v=>{let value=norm(v);if(f==='backends'){const compact=value.replace(/[^a-z0-9]/g,'');value=compact.includes('sam3d')?'sam3d':compact.includes('triposr')?'triposr':value;}return(f==='task_classes'?taskClasses(r):(i[f]||[])).includes(value);}));});
 const sort=options.sort||'title',direction=options.direction==='desc'?-1:1;
 const value=r=>sort==='title'?norm(r.public_title||r.title):sort==='id'?r.id:sort==='duration'?r.duration_s:sort==='version'?((r.id.match(/20[0-9]{6}/g)||[]).sort().at(-1)||'00000000'):(r.counts?.[sort]||0);
 rows.sort((a,b)=>{const x=value(a),y=value(b);return(x<y?-1:x>y?1:0)*direction||(a.id<b.id?-1:a.id>b.id?1:0);});
 return rows;
}
function nativeScene(record,scene,trackId){
 assert(object(scene)&&scene.physical_assembly_verified===false&&Array.isArray(scene.objects),'Unverified scene missing.');
 const parts=record.components.filter(c=>c.asset),ids=new Set();
 for(const o of scene.objects){const c=parts.find(c=>c.id===o.object_id);assert(c&&!ids.has(o.object_id)&&o.provenance?.mesh_sha256===c.asset.sha256&&o.provenance?.mesh_file===c.asset.path,'Native geometry belongs to a different saved asset.');ids.add(o.object_id);}
 assert(ids.size===parts.length,'Saved geometry is incomplete.');
 const component=record.components.find(c=>c.id===trackId);assert(component,'Unknown part.');
 if(!component.asset)return null;
 const selected=scene.objects.find(o=>o.object_id===trackId);assert(selected&&object(selected.native_scene),'Native asset layout missing.');
 return {...scene,objects:[{...selected,frames:[],max_hold_s:0,source_projection_rows:null}],native_only:true,timeline:[],source_projection_rows:null,physical_assembly_verified:false};
}
async function bytes(r,{signal}={}){
 ref(r);const response=await fetch(new URL('/files/'+r.path,location.origin),{signal,cache:'force-cache'});assert(response.ok,'Saved artifact is unavailable.');const raw=await response.arrayBuffer();assert(raw.byteLength===r.bytes,'Saved artifact length changed.');const sha=[...new Uint8Array(await crypto.subtle.digest('SHA-256',raw))].map(v=>v.toString(16).padStart(2,'0')).join('');assert(sha===r.sha256,'Saved artifact hash changed.');return raw;
}
async function loadCatalog(){
 let i=root.INTERACTION_LIBRARY_INDEX;const latest=await fetch(new URL('library-index.json',LIBRARY_SCRIPT_BASE),{cache:'no-store'});if(latest.status!==404){assert(latest.ok,'Current library index unavailable.');const raw=await latest.text();assert(raw.length<8192,'Library index outside budget.');i=JSON.parse(raw);}
 assert(object(i)&&i.format==='interaction-twin-library-index'&&i.version===1,'Library index missing.');const r=i.catalog;assert(object(r)&&(r.path==='library-catalog.json'||r.path===`library-catalog-${r.sha256}.json`)&&SHA.test(r.sha256)&&Number.isSafeInteger(r.bytes)&&r.bytes<8*1024*1024,'Invalid catalog identity.');
 const response=await fetch(new URL(r.path,LIBRARY_SCRIPT_BASE),{cache:'no-cache'});assert(response.ok,'Library catalog unavailable.');const raw=await response.arrayBuffer();assert(raw.byteLength===r.bytes,'Catalog length changed.');const sha=[...new Uint8Array(await crypto.subtle.digest('SHA-256',raw))].map(v=>v.toString(16).padStart(2,'0')).join('');assert(sha===r.sha256,'Catalog bytes changed.');return validateCatalog(JSON.parse(new TextDecoder().decode(raw)));
}
function clearAssetLinks(links){for(const link of links){if(link){link.removeAttribute('href');link.hidden=true;}}}
async function loadSourceCase(record,{signal}={}){
 const raw=await bytes(record.interaction,{signal}),program=JSON.parse(new TextDecoder().decode(raw));
 assert(program.source?.sha256===record.source.sha256&&Array.isArray(program.steps)&&Array.isArray(program.evidence)&&Array.isArray(program.assertions),'Procedure belongs to a different recording.');
 assert(record.scene_case?.id===record.id&&record.scene_case.original_sha256===record.interaction.sha256,'Curated sample belongs to a different procedure.');
 return{id:record.id,title:record.title,original_sha256:record.interaction.sha256,interaction_path:record.interaction.path,interaction:program,video_path:record.source.path,duration_s:record.duration_s,labels:{steps:program.steps.length,source_samples:program.evidence.length}};
}
const exports={validateCatalog,matches,nativeScene,ref,clearAssetLinks,loadCatalog,loadSourceCase,queryRecords,taskClasses,TASK_CLASSES};if(typeof module!=='undefined'&&module.exports)module.exports=exports;root.InteractionLibrary=exports;
if(typeof document==='undefined')return;
const $=id=>document.getElementById(id),el=(name,text,className)=>{const e=document.createElement(name);if(text!==undefined)e.textContent=text;if(className)e.className=className;return e;};
const fmt=t=>`${t.toFixed(2)} s`,human=s=>String(s||'').replaceAll('_',' '),fileURL=r=>'/files/'+r.path;
const workspace=(r,extra={})=>'index.html?'+new URLSearchParams({source:r.source.sha256,...(extra.t!==undefined?{t:extra.t}:{})})+'#samples';
const shortTitle=r=>r.public_title||r.title;
// Browser rendering hook also exercises the actual detail path in controller
// tests; no scene or inference is created by rendering saved metadata.
exports.renderDetails=details;
const urls=new Set(),cardHosts=new Map();let scene=null,epoch=0,controller=null,current=null,currentComponent=null,currentScene=null,visibleRecords=[];
function clearInspection(){scene?.dispose();scene=null;for(const u of urls)URL.revokeObjectURL(u);urls.clear();const canvas=$('library-scene');if(canvas){const replacement=canvas.cloneNode(false);canvas.replaceWith(replacement);}for(const id of['anchor-image','anchor-mask'])$(id)?.removeAttribute('src');clearAssetLinks([$('download-asset'),$('anchor-workspace'),$('open-sam3d-component')]);const frame=$('sam3d-native-frame');if(frame){frame.removeAttribute('src');frame.hidden=true;}if($('anchor-mask'))$('anchor-mask').hidden=true;if($('mask-toggle'))$('mask-toggle').checked=false;}
function blobURL(raw,type){const u=URL.createObjectURL(new Blob([raw],{type}));urls.add(u);return u;}
function counts(d,container){for(const [n,label]of[[d.summary.recordings,'source recordings'],[d.records.reduce((n,r)=>n+(root.Sam3DModelReviews?.forRecording(r).length||0),0),'SAM 3D models'],[d.summary.components,'component tracklets'],[d.summary.steps,'proposed task steps']]){const e=el('div',undefined,'library-count');e.append(el('strong',String(n)),el('span',label));container.append(e);}}
function paintCollections(catalog,selected='',choose=null){
 const box=$(choose?'library-collections':'task-class-preview');if(!box)return;box.replaceChildren();
 for(const definition of TASK_CLASSES){
  const count=catalog.records.filter(record=>taskClasses(record).includes(definition.id)).length;
  const card=el(choose?'button':'a',undefined,'task-class');
  if(choose){card.type='button';card.setAttribute('aria-pressed',String(selected===definition.id));card.addEventListener('click',()=>choose(selected===definition.id?'':definition.id));}
  else card.href='library.html?'+new URLSearchParams({task_classes:definition.id});
  const icon=el('span',undefined,'task-class-icon');icon.setAttribute('aria-hidden','true');icon.innerHTML='<svg viewBox="0 0 33 33">'+definition.icon+'</svg>';
  card.append(icon,el('strong',definition.label),el('span',String(count),`collection-count${count?'':' empty'}`));box.append(card);
 }
}
function citationLinks(record,rows){const box=el('div',undefined,'citation-links');for(const c of rows){const a=el('a',fmt(c.timestamp_s));a.href=fileURL(c.frame);a.target='_blank';a.rel='noopener';a.title=`Open exact source frame ${c.id} at ${fmt(c.timestamp_s)}`;box.append(a);}return box;}
function card(record,onSelect,preview=false){const button=el(preview?'a':'button',undefined,'library-card');if(preview)button.href='library.html?'+new URLSearchParams({case:record.id});else{button.type='button';button.setAttribute('aria-expanded','false');button.setAttribute('aria-controls','library-detail');button.dataset.case=record.id;button.addEventListener('click',()=>onSelect(record,true));}
 const media=el('div',undefined,'library-card-image'),im=el('img');im.alt='Saved source input from '+shortTitle(record);im.loading='lazy';media.append(im);const body=el('div',undefined,'library-card-copy');body.append(el('span',`${record.duration_s.toFixed(1)} SECOND RECORDING`,'small-label'),el('h3',shortTitle(record)),el('p',`${record.counts.components} components · ${root.Sam3DModelReviews?.forRecording(record).length||0} SAM 3D models · ${record.counts.steps} actions`,'library-card-meta'));if(record.public_note)body.append(el('p',record.public_note,'library-note'));const tags=el('div',undefined,'library-card-tags');for(const t of record.tags.slice(0,2))tags.append(el('span',t));body.append(tags,el('span',preview?'Explore twin ↗':'Inspect twin →','text-link'));button.append(media,body);
 bytes(record.thumbnail).then(raw=>{if(im.isConnected){const u=URL.createObjectURL(new Blob([raw],{type:'image/png'}));im.src=u;im.onload=()=>URL.revokeObjectURL(u);}}).catch(()=>{media.replaceChildren(el('span','Source preview unavailable'));});return button;
}
function resetCardHost(host){
 if(!host)return;host.classList.remove('is-expanded');const trigger=host.querySelector('.library-card');if(trigger){trigger.hidden=false;trigger.setAttribute('aria-expanded','false');}
}
function parkInspector(){
 const panel=$('library-detail'),parking=$('library-detail-parking');if(!panel)return;
 const host=panel.parentElement;if(host?.classList.contains('library-item'))resetCardHost(host);
 panel.hidden=true;if(parking&&panel.parentElement!==parking)parking.append(panel);
}
function mountInspector(record){
 const host=cardHosts.get(record.id),panel=$('library-detail');if(!host||host.parentElement!==$('library-cards')||host.hidden||!panel)return false;
 parkInspector();host.classList.add('is-expanded');const trigger=host.querySelector('.library-card');trigger.hidden=true;trigger.setAttribute('aria-expanded','true');host.append(panel);panel.hidden=false;return true;
}
function collapse({restoreFocus=false,updateURL=true}={}){
 const previous=current;++epoch;controller?.abort();controller=null;clearPreview();clearInspection();current=null;currentComponent=null;currentScene=null;parkInspector();
 const panel=$('library-detail');panel?.classList.remove('is-loading');panel?.setAttribute('aria-busy','false');
 if(updateURL){const url=new URL(location.href);url.searchParams.delete('case');if(url.hash==='#sam3d-models')url.hash='';history.replaceState(null,'',url);}
 if($('library-status'))$('library-status').textContent=`${visibleRecords.length} twins · select one to inspect`;
 if(restoreFocus&&previous)cardHosts.get(previous.id)?.querySelector('.library-card')?.focus();
}
function renderCards(records){
 const full=$('library-cards');if(!full)return;
 if(current&&!records.some(record=>record.id===current.id))collapse();
 visibleRecords=records;
 const wanted=new Set(records.map(record=>record.id));for(const child of [...full.children])if(!wanted.has(child.dataset.twin))child.remove();
 const hosts=records.map(record=>{let host=cardHosts.get(record.id);if(!host){host=el('article',undefined,'library-item');host.dataset.twin=record.id;host.append(card(record,select));cardHosts.set(record.id,host);}return host;});
 // Keep the expanded host connected: moving an ancestor reloads its iframe.
 const active=current?cardHosts.get(current.id):null,index=hosts.indexOf(active);
 if(index>=0&&active.parentElement===full){for(const host of hosts.slice(0,index))full.insertBefore(host,active);for(const host of hosts.slice(index+1))full.append(host);}
 else for(const host of hosts)full.append(host);
}
function showTab(name){for(const tab of document.querySelectorAll('.program-tabs [role=tab]')){const on=tab.dataset.panel===name;tab.setAttribute('aria-selected',String(on));tab.tabIndex=on?0:-1;$('panel-'+tab.dataset.panel).hidden=!on;}}
function sam3dReviews(record){
 const box=$('sam3d-model-cards');if(!box)return;
 const rows=root.Sam3DModelReviews?.forRecording(record)||[];box.replaceChildren();
 $('sam3d-models-count').textContent=rows.length?`${rows.length} GENERATED PARTS`:'NOT GENERATED YET';
 $('sam3d-models-description').textContent=rows.length?'Open a generated part to inspect its shape alongside the source frame used for reconstruction.':'SAM 3D geometry is pending for this recording. Its source evidence and descriptive program remain inspectable.';
 $('sam3d-models-note').hidden=!rows.length;
 for(const row of rows){
  const card=el('article',undefined,'sam3d-model-card'),image=el('img');image.alt='Actual source frame for the '+row.label+' SAM 3D model';image.loading='lazy';
  const copy=el('div',undefined,'sam3d-model-copy'),link=el('a','Open SAM 3D model →','button primary');link.href='native-model.html?'+new URLSearchParams({case:record.id,component:row.component_id});link.setAttribute('aria-label','Open '+row.label+' SAM 3D model');
  copy.append(el('span','SAM 3D OBJECTS','small-label'),el('h4',row.label),el('p','Interactive model + source frame'),link);card.append(image,copy);box.append(card);
  bytes(row.source_image).then(raw=>{if(image.isConnected){const u=URL.createObjectURL(new Blob([raw],{type:'image/png'}));image.src=u;image.onload=()=>URL.revokeObjectURL(u);}}).catch(()=>{if(card.isConnected){link.removeAttribute('href');link.setAttribute('aria-disabled','true');link.textContent='Saved preview unavailable';}});
 }
}
function details(record){
 sam3dReviews(record);
 $('detail-title').textContent=shortTitle(record);$('detail-summary').textContent=record.summary;const qualityNote=$('detail-quality-note');if(qualityNote){qualityNote.textContent=record.public_note||'';qualityNote.hidden=!record.public_note;}$('open-workspace').href=workspace(record);$('open-robotics').hidden=true;$('open-insights').href=workspace(record,{view:'intelligence'});
 const names=new Map(record.components.map(c=>[c.id,c.label]));const relations=$('relation-list');relations.replaceChildren();for(const r of record.relationships){const e=el('article',undefined,'library-relation');const line=el('div',undefined,'relation-nodes');line.append(el('strong',names.get(r.subject_track_id)),el('span',human(r.predicate)),el('strong',names.get(r.object_track_id)));e.append(line,el('p',r.rationale||'Source-linked model proposal.'),el('small',human(r.status||'model proposal'),'relation-meta'),citationLinks(record,r.citations));if(r.limitations?.length)e.append(el('p',r.limitations.join(' · '),'library-note'));relations.append(e);}
 const actions=$('action-list');actions.replaceChildren();for(const[sidx,s]of record.steps.entries()){const e=el('article',undefined,'library-action');e.append(el('span',String(sidx+1).padStart(2,'0'),'action-number'));const body=el('div');body.append(el('span',`${fmt(s.start_s)} → ${fmt(s.end_s)}`,'small-label'),el('h4',s.title),el('p',s.description||''),citationLinks(record,s.citations));if(s.limitations?.length)body.append(el('p',s.limitations.join(' · '),'library-note'));e.append(body);actions.append(e);}
 const inventory=$('readiness-inventory');inventory.replaceChildren();for(const[n,label]of[[record.counts.assets,'native visual drafts'],[record.counts.relationships,'proposed relations'],[record.counts.steps,'visual task steps'],[record.counts.evidence_samples,'evidence samples']]){const e=el('div');e.append(el('strong',String(n)),el('span',label));inventory.append(e);}
 const gaps=$('readiness-gaps');gaps.replaceChildren();for(const c of record.readiness.checks.filter(c=>c.blocking)){const e=el('details',undefined,'readiness-gap'),summary=el('summary',c.title);e.append(summary,el('p',c.detail),el('p',c.next_action,'library-note'));gaps.append(e);}
 $('download-program').href=fileURL(record.interaction);$('mcp-request').textContent=JSON.stringify({name:'describe_twin',arguments:{twin_id:record.id}},null,2);$('copy-status').textContent='';
 const sceneProvenance=['Geometry inspection','Native SAM parts · source-bound estimated replay where available'];
 const provenance=$('provenance-list');provenance.replaceChildren();for(const[label,value,href]of[['Original recording SHA256',record.source.original_sha256],['Saved program SHA256',record.interaction.sha256,fileURL(record.interaction)],['Visual reasoning',record.reasoning.model],['Segmentation','SAM 3.1 · model-generated prompts'],['Review state','Model proposals · physical identity unverified'],sceneProvenance]){const row=el('div',undefined,'provenance-row'),term=el('span',label),v=el(href?'a':'code',value);if(href)v.href=href;row.append(term,v);provenance.append(row);}
 $('version-count').textContent=`${record.other_saved_versions.length} other saved experimental version${record.other_saved_versions.length===1?'':'s'}`;const versions=$('saved-versions');versions.replaceChildren();for(const v of record.other_saved_versions){const a=el('span',v.id);versions.append(a);}
 const picker=$('component-picker');picker.replaceChildren();for(const c of record.components){const button=el('button');button.type='button';button.dataset.track=c.id;button.setAttribute('aria-pressed','false');button.append(el('strong',c.label),el('small',c.id),el('small',root.Sam3DModelReviews?.forComponent(record,c.id)?'SAM 3D preview available':c.asset?'Saved historical draft':'Geometry pending',c.asset?'':'no-asset'));button.addEventListener('click',()=>selectComponent(c));picker.append(button);}
 $('offline-queue').replaceChildren();$('offline-queue').hidden=true;$('cached-insights').replaceChildren();$('cached-insights').hidden=true;
 showTab('relations');
}
function selectionError(message){clearInspection();$('native-message').textContent=message;$('native-message').hidden=false;$('anchor-message').textContent='No replacement input is shown.';$('anchor-message').hidden=false;}
async function selectComponent(component){
 const token=++epoch;controller?.abort();controller=new AbortController();clearInspection();currentComponent=component;
 for(const b of document.querySelectorAll('#component-picker button'))b.setAttribute('aria-pressed',String(b.dataset.track===component.id));
 const newer=root.Sam3DModelReviews?.forComponent(current,component.id),newerLink=$('open-sam3d-component');
 $('library-scene').hidden=true;$('native-reset').hidden=true;$('native-label').textContent=component.label;
 if(newer){
  newerLink.href='native-model.html?'+new URLSearchParams({case:current.id,component:component.id});newerLink.hidden=false;
  $('native-backend').textContent='Meta SAM 3D Objects · generated component';$('native-appearance').textContent='Original generated appearance. Independent coordinates; fresh video alignment and assembly remain unverified.';
  $('native-message').hidden=true;$('anchor-message').hidden=false;$('anchor-message').textContent='Verifying this model’s actual SAM input…';$('mask-toggle').disabled=false;
  const frame=$('sam3d-native-frame');frame.loading='eager';frame.hidden=false;frame.src='native-model.html?'+new URLSearchParams({case:current.id,component:component.id});
  try{const[image,mask]=await Promise.all([bytes(newer.source_image,{signal:controller.signal}),bytes(newer.source_mask,{signal:controller.signal})]);if(token!==epoch)return;$('anchor-image').src=blobURL(image,'image/png');$('anchor-mask').src=blobURL(mask,'image/png');$('anchor-message').hidden=true;$('anchor-time').textContent=fmt(newer.source_timestamp_s);$('anchor-workspace').href=workspace(current,{track:component.id,t:newer.source_timestamp_s,view:'replay'});$('anchor-workspace').hidden=false;$('download-asset').href=fileURL(newer.native_glb);$('download-asset').hidden=false;}catch(e){if(token===epoch&&e.name!=='AbortError')selectionError(e.message);}return;
 }
 $('native-backend').textContent='SAM 3D geometry pending';$('native-appearance').textContent='The source-linked part and procedure are available. Historical geometry is kept in saved experiments.';$('native-message').textContent='No SAM 3D result for this component yet.';$('native-message').hidden=false;$('anchor-message').textContent='Choose a generated component to inspect its reconstruction input.';$('anchor-message').hidden=false;$('mask-toggle').disabled=true;
}


let previewEpoch=0,previewBinding=null,previewView='source',previewFrameRequest=0,previewOwner=null,previewInitialSeek=null,previewUserInteracted=false;
function inspectorView(name){
 for(const tab of document.querySelectorAll('.twin-view-tabs [role=tab]')){const on=tab.dataset.view===name;tab.setAttribute('aria-selected',String(on));tab.tabIndex=on?0:-1;const panel=$('view-panel-'+tab.dataset.view);if(panel)panel.hidden=!on;}
 if(name!=='preview')$('twin-source-video')?.pause?.();
 if(name==='parts'&&current&&!currentComponent)selectComponent(current.components.find(c=>root.Sam3DModelReviews?.forComponent(current,c.id))||current.components[0]);
}
function postPreview(reset=false){
 if(!previewBinding||!previewOwner)return;
 const video=$('twin-source-video'),frame=$('twin-alignment-frame'),time=Number(video?.currentTime);
 if(!Number.isFinite(time)||time<0||time>previewOwner.duration_s||time>previewBinding.manifest.source.duration_s)return;
 frame?.contentWindow?.postMessage({type:'interaction-twin:aligned-time',source_sha256:previewOwner.source.sha256,manifest_sha256:previewBinding.reference.sha256,time_s:time,playing:!video?.paused,show_estimated:true,view:previewView,reset},location.origin);
}
function clearPreview(){
 ++previewEpoch;previewOwner=null;previewBinding=null;previewView='source';previewInitialSeek=null;previewUserInteracted=false;if(previewFrameRequest&&root.cancelAnimationFrame)root.cancelAnimationFrame(previewFrameRequest);previewFrameRequest=0;
 const video=$('twin-source-video');video?.pause?.();video?.removeAttribute('src');video?.removeAttribute('poster');video?.load?.();
 const frame=$('twin-alignment-frame');frame?.removeAttribute('src');if(frame)frame.hidden=true;
 for(const id of ['preview-camera','preview-orbit'])if($(id))$(id).disabled=true;
}
function updatePreviewTimeline(){
 const video=$('twin-source-video'),range=$('preview-timeline'),time=Number(video?.currentTime);if(!range||!Number.isFinite(time)||time<0)return;
 range.value=String(time);range.setAttribute('aria-valuetext',fmt(time));$('preview-time').textContent=fmt(time);const play=$('preview-play');if(play){play.textContent=video.paused?'▶':'Ⅱ';play.setAttribute('aria-label',video.paused?'Play source video':'Pause source video');}
}
function applyInitialPreview(){
 const pending=previewInitialSeek,video=$('twin-source-video');if(!pending||pending.epoch!==previewEpoch||pending.source!==previewOwner?.source.sha256||previewUserInteracted||video.readyState<1)return;
 try{video.currentTime=pending.time_s;previewInitialSeek=null;updatePreviewTimeline();}catch(_){/* Metadata from a previous source is never used to force a seek. */}
}
function scrollToInspector(record,requested){
 if(!requested)return;const scroll=()=>{const panel=$('library-detail');if(current!==record||panel.hidden)return;panel.scrollIntoView({behavior:root.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches?'auto':'smooth',block:'start'});};
 if(root.requestAnimationFrame)root.requestAnimationFrame(scroll);else scroll();
}
function downloadRow(title,detail,binding){const row=el('a',undefined,'twin-download');row.href=fileURL(binding);row.download='';const copy=el('span');copy.append(el('strong',title),el('small',detail));row.append(copy,el('span','↓','download-arrow'));return row;}
function renderDownloads(record,aligned=null){
 const box=$('twin-downloads');if(!box)return;box.replaceChildren();
 if(aligned?.object_program)box.append(downloadRow(aligned.derived_assembly?'Refined twin program v1':'Descriptive twin program',aligned.derived_assembly?'JSON · declared geometry refinements, source audit and estimated joints':'JSON · components, joint frames, angles and source evidence',aligned.object_program));
 if(aligned?.motion_hypothesis)box.append(downloadRow('Motion continuity hypotheses','JSON · source-conditioned poses, inherited joint rules and exact gap evidence',aligned.motion_hypothesis.artifact));
 if(aligned?.observed_surfaces)box.append(downloadRow('Observed surface program','JSON · source-visible depth estimates, flexure states and exact surface/evidence references',aligned.observed_surfaces.artifact));
 if(aligned?.base_object_program)box.append(downloadRow('Original native object program','JSON · unchanged SAM 3D components and original estimated connections',aligned.base_object_program));
 box.append(downloadRow('Demonstrated procedure','JSON · proposed actions, relationships and citations',record.interaction));
 box.append(downloadRow('Source video','MP4 · the saved demonstration used by this twin',record.source));
 if(aligned?.reference)box.append(downloadRow('Connected scene','JSON · estimated poses and connection rules',aligned.reference));
 for(const component of aligned?.derived_assembly?.document?.component_assets||[]){if(component.native_glb.sha256!==component.original_native_glb.sha256)box.append(downloadRow((record.components.find(v=>v.id===component.component_id)?.label||component.component_id)+' · refined','GLB · declared automatic derivative; original available below',component.native_glb));}
 for(const row of root.Sam3DModelReviews?.forRecording(record)||[])box.append(downloadRow(row.label,'GLB · original SAM 3D Objects geometry',row.native_glb));
 if(!(root.Sam3DModelReviews?.forRecording(record)||[]).length)box.append(el('p','SAM 3D component downloads are pending for this recording.','library-note'));
}
function formatDocument(raw,binding,label){
 const text=new TextDecoder().decode(raw),doc=JSON.parse(text);$('program-document').textContent=JSON.stringify(doc,null,2);$('program-format').textContent=label;
 const link=$('description-download');link.href=fileURL(binding);link.hidden=false;
}
async function startPreview(record,raw){
 clearPreview();const token=previewEpoch;previewOwner=record;
 const video=$('twin-source-video'),frame=$('twin-alignment-frame'),status=$('twin-alignment-status');if(!video||!frame||!status)return;
 $('preview-duration').textContent=fmt(record.duration_s);const timeline=$('preview-timeline');if(timeline){timeline.max=String(record.duration_s);timeline.value='0';timeline.disabled=true;$('preview-time').textContent=fmt(0);$('preview-end').textContent=fmt(record.duration_s);}$('preview-lab').href='robotics-lab.html?'+new URLSearchParams({case:record.id});
 status.hidden=false;status.textContent='Checking the source-bound scene…';$('preview-scope').textContent='Scrub the video to inspect the corresponding 3D estimate.';
 video.src=fileURL(record.source);video.poster=fileURL(record.thumbnail);video.load?.();renderDownloads(record);formatDocument(raw,record.interaction,'Procedure JSON');
 try{
  const aligned=root.AlignedScenes?await root.AlignedScenes.find(record.source.sha256):null;if(token!==previewEpoch)return;
  if(!aligned){status.textContent='A connected scene is not available for this recording yet.';$('preview-scope').textContent='Inspect its generated parts and descriptive procedure in the other tabs.';return;}
  if(root.AlignedSceneContract?.matchesProgram){const program=JSON.parse(new TextDecoder().decode(raw)),binding={interaction:program,interaction_path:record.interaction.path,original_sha256:record.interaction.sha256},reviews=root.Sam3DModelReviews?.forRecording(record)||[];if(!root.AlignedSceneContract.matchesProgram(aligned.manifest,binding,reviews,aligned.derived_assembly))throw Error('Scene does not match this curated program and its original native components.');}
  previewBinding=aligned;previewView='source';const supported=root.AlignedSceneContract?.previewTime(aligned.manifest,aligned.observed_surfaces?.document);if(supported&&Number.isFinite(supported.time_s)&&supported.time_s>=0&&supported.time_s<=record.duration_s&&!previewUserInteracted){previewInitialSeek={epoch:token,source:record.source.sha256,time_s:supported.time_s};applyInitialPreview();}for(const id of ['preview-camera','preview-orbit'])$(id).disabled=false;
  $('preview-camera').setAttribute('aria-pressed','true');$('preview-orbit').setAttribute('aria-pressed','false');
  frame.loading='eager';frame.hidden=false;frame.src='aligned-scene.html?'+new URLSearchParams({source:record.source.sha256,path:aligned.reference.path,sha256:aligned.reference.sha256,bytes:String(aligned.reference.bytes),presentation:'landing'});
  status.hidden=true;$('preview-scope').textContent=aligned.derived_assembly?'Reviewed geometry refinements · source-linked estimates; physical fit and contact remain unverified.':'Source-linked replay · depth, opening angles and physical fit remain estimates.';renderDownloads(record,aligned);
  if(aligned.observed_surfaces){$('preview-scope').textContent='Observed surfaces · estimated depth; hidden geometry and numerical joint parameters are unknown. Original SAM models remain in 3D parts.';$('program-document').textContent=JSON.stringify(aligned.observed_surfaces.document,null,2);$('program-format').textContent='Source-visible surface program';$('description-download').href=fileURL(aligned.observed_surfaces.artifact);}
  else if(aligned.object_program){const doc=await root.AlignedScenes.checked(aligned.object_program);if(token!==previewEpoch)return;$('program-document').textContent=JSON.stringify(doc,null,2);$('program-format').textContent=aligned.derived_assembly?'Refined twin program v1':'Object program v'+doc.version;$('description-download').href=fileURL(aligned.object_program);}
 }catch(error){if(token!==previewEpoch)return;previewBinding=null;frame.removeAttribute('src');frame.hidden=true;status.hidden=false;status.textContent='Saved scene unavailable: '+error.message;renderDownloads(record);formatDocument(raw,record.interaction,'Procedure JSON');for(const id of ['preview-camera','preview-orbit'])$(id).disabled=true;}
}
function connectPreview(){
 const video=$('twin-source-video');if(!video)return;
 for(const type of ['timeupdate','seeked','loadedmetadata','pause','play'])video.addEventListener(type,()=>{if(type==='loadedmetadata'){$('preview-timeline').disabled=false;applyInitialPreview();}if(type==='play')previewUserInteracted=true;updatePreviewTimeline();postPreview();if(type==='play')tickPreview();});
 for(const type of ['pointerdown','keydown'])video.addEventListener(type,()=>{previewUserInteracted=true;previewInitialSeek=null;});
 $('preview-timeline')?.addEventListener('input',event=>{const time=Number(event.target.value);if(!previewOwner||!Number.isFinite(time)||time<0||time>previewOwner.duration_s)return;previewUserInteracted=true;previewInitialSeek=null;video.currentTime=time;updatePreviewTimeline();postPreview();});
 $('preview-play')?.addEventListener('click',()=>{previewUserInteracted=true;previewInitialSeek=null;if(video.paused){const play=video.play();play?.catch?.(()=>{});}else video.pause();});
 function tickPreview(){if(previewFrameRequest&&root.cancelAnimationFrame)root.cancelAnimationFrame(previewFrameRequest);if(!video.paused){postPreview();previewFrameRequest=root.requestAnimationFrame(tickPreview);}else previewFrameRequest=0;}
 for(const [id,view]of [['preview-camera','source'],['preview-orbit','orbit']])$(id)?.addEventListener('click',()=>{previewView=view;$('preview-camera').setAttribute('aria-pressed',String(view==='source'));$('preview-orbit').setAttribute('aria-pressed',String(view==='orbit'));postPreview(view==='orbit');});
 root.addEventListener('message',event=>{const d=event.data;if(!previewOwner||!previewBinding||event.origin!==location.origin||event.source!==$('twin-alignment-frame')?.contentWindow||d?.source_sha256!==previewOwner.source.sha256||d?.manifest_sha256!==previewBinding.reference.sha256)return;if(d.type==='interaction-twin:aligned-ready')postPreview();if(d.type==='interaction-twin:aligned-error'){$('twin-alignment-status').hidden=false;$('twin-alignment-status').textContent=d.message||'Connected scene unavailable.';}});
 for(const tab of document.querySelectorAll('.twin-view-tabs [role=tab]')){tab.addEventListener('click',()=>inspectorView(tab.dataset.view));tab.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();const tabs=[...document.querySelectorAll('.twin-view-tabs [role=tab]')],i=tabs.indexOf(tab),n=event.key==='Home'?0:event.key==='End'?tabs.length-1:(i+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;inspectorView(tabs[n].dataset.view);tabs[n].focus();});}
}

async function select(record,scroll=false){
 if(!mountInspector(record))return;
 const token=++epoch;controller?.abort();controller=new AbortController();clearInspection();clearPreview();current=record;currentComponent=null;currentScene=null;
 const panel=$('library-detail');panel.classList.add('is-loading');panel.setAttribute('aria-busy','true');$('detail-title').textContent=shortTitle(record);$('detail-summary').textContent='Verifying the saved program and geometry…';const qualityNote=$('detail-quality-note');if(qualityNote){qualityNote.textContent='';qualityNote.hidden=true;}$('open-workspace').href=workspace(record);$('library-status').textContent='Verifying this twin…';
 const url=new URL(location.href);url.searchParams.set('case',record.id);history.replaceState(null,'',url);scrollToInspector(record,scroll);
 try{const raw=await bytes(record.interaction,{signal:controller.signal});if(token!==epoch)return;details(record);panel.classList.remove('is-loading');panel.setAttribute('aria-busy','false');$('library-status').textContent='Verified source-linked object program';inspectorView(location.hash==='#sam3d-models'?'parts':'preview');startPreview(record,raw);if(location.hash==='#sam3d-models')$('sam3d-models')?.scrollIntoView({block:'start'});
 }catch(e){if(token===epoch&&e.name!=='AbortError'){panel.setAttribute('aria-busy','false');$('detail-summary').textContent='This saved twin is unavailable: '+e.message;$('library-status').textContent='Could not verify this entry. Choose another twin or minimize it.';}}
}
async function start(){
 const preview=$('library-preview'),full=$('library-cards');if(!preview&&!full)return;
 try{const d=await loadCatalog();if(preview){preview.replaceChildren(...d.records.map(r=>card(r,null,true)));const c=$('library-preview-counts');if(c){c.replaceChildren();counts(d,c);}paintCollections(d);}
 if(!full)return;$('library-counts').replaceChildren();counts(d,$('library-counts'));
 let selectedClass='';const names={parts:'Part',actions:'Demonstrated action',relations:'Relationship',backends:'Geometry model',readiness:'Draft status'},queryControls={};
 const initial=new URL(location.href).searchParams;$('library-search').value=initial.get('query')||'';selectedClass=TASK_CLASSES.some(c=>c.id===initial.get('task_classes'))?initial.get('task_classes'):'';
 for(const field of Object.keys(names)){
  const label=el('label',undefined,'library-select'),select=el('select');select.id='filter-'+field;select.setAttribute('aria-label',names[field]);
  label.append(el('span',names[field]),select);const any=el('option',({parts:'All parts',actions:'All demonstrated actions',relations:'All relationships',backends:'Any geometry model',readiness:'All draft statuses'})[field]);any.value='';select.append(any);
  const values=new Map();for(const r of d.records){const i=r.search_index||{};for(const v of i[field]||[])values.set(v,i.labels?.[field]?.[v]||human(v));}
  if(field==='backends'){values.delete('triposr');if(values.has('sam3d'))values.set('sam3d','Meta SAM 3D Objects');}
  for(const[v,text]of [...values].sort((a,b)=>a[1].localeCompare(b[1]))){const o=el('option',text);o.value=v;select.append(o);}
  select.value=values.has(initial.get(field))?initial.get(field):'';select.addEventListener('change',render);queryControls[field]=select;$('library-filters').append(label);
 }
 function options(){const o={query:$('library-search').value.slice(0,200)};for(const[f,c]of Object.entries(queryControls))o[f]=c.value?[c.value]:[];const[sort,direction]=$('library-sort').value.split(':');return{...o,...(selectedClass?{task_classes:[selectedClass]}:{}),sort,direction};}
 if([...$('library-sort').options].some(o=>o.value===initial.get('sort')+':'+initial.get('direction')))$('library-sort').value=initial.get('sort')+':'+initial.get('direction');
 function render(){const opts=options(),visible=queryRecords(d.records,opts);paintCollections(d,selectedClass,id=>{selectedClass=id;render();});if($('collection-all'))$('collection-all').setAttribute('aria-pressed',String(!selectedClass));renderCards(visible);$('library-empty').hidden=visible.length>0;$('library-status').textContent=`${visible.length} of ${d.records.length} twins · select one to inspect`;
 const params=new URLSearchParams();if(opts.query)params.set('query',opts.query);for(const f of [...Object.keys(names),'task_classes'])for(const v of opts[f]||[])params.append(f,v);params.set('sort',opts.sort);params.set('direction',opts.direction);
 $('library-query-json').href='/api/library?'+params;$('library-query-code').textContent=JSON.stringify({name:'search_twins',arguments:{...opts,group_by:'recording',limit:25}},null,2);
 const url=new URL(location.href);for(const k of ['query',...Object.keys(names),'task_classes','sort','direction'])url.searchParams.delete(k);for(const[k,v]of params)url.searchParams.append(k,v);history.replaceState(null,'',url);
 }
 if($('collection-all'))$('collection-all').addEventListener('click',()=>{selectedClass='';render();});$('library-search').addEventListener('input',render);$('library-sort').addEventListener('change',render);$('library-reset').addEventListener('click',()=>{selectedClass='';$('library-search').value='';for(const c of Object.values(queryControls))c.value='';$('library-sort').value='title:asc';render();});
 $('library-copy-query').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('library-query-code').textContent);$('library-query-copy-status').textContent='MCP search copied.';}catch{$('library-query-copy-status').textContent='Select and copy the query below.';}});
 for(const tab of document.querySelectorAll('.program-tabs [role=tab]')){tab.addEventListener('click',()=>showTab(tab.dataset.panel));tab.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const tabs=[...document.querySelectorAll('.program-tabs [role=tab]')],i=tabs.indexOf(tab),n=e.key==='Home'?0:e.key==='End'?tabs.length-1:(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;showTab(tabs[n].dataset.panel);tabs[n].focus();});}
 $('native-reset').addEventListener('click',()=>scene?.reset());$('mask-toggle').addEventListener('change',()=>{$('anchor-mask').hidden=!$('mask-toggle').checked;});
 $('library-scene').parentElement.addEventListener('keydown',e=>{if(e.target.id!=='library-scene'||!scene)return;const keys=['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-'];if(!keys.includes(e.key))return;e.preventDefault();if(e.key==='ArrowLeft')scene.yaw-=.15;if(e.key==='ArrowRight')scene.yaw+=.15;if(e.key==='ArrowUp')scene.pitch=Math.min(1.5,scene.pitch+.1);if(e.key==='ArrowDown')scene.pitch=Math.max(-1.5,scene.pitch-.1);if(e.key==='+'||e.key==='=')scene.distance*=.9;if(e.key==='-')scene.distance*=1.1;scene.draw();});
 $('copy-mcp').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('mcp-request').textContent);$('copy-status').textContent='Tool request copied. No tool was executed.';}catch(_){$('copy-status').textContent='Select and copy the request above.';}});
 $('minimize-twin').addEventListener('click',()=>collapse({restoreFocus:true}));connectPreview();root.addEventListener('pagehide',()=>{controller?.abort();clearPreview();clearInspection();});render();const requested=new URL(location.href).searchParams.get('case');if(!d.records.length){$('library-status').textContent='No curated object programs are available yet.';return;}const initialVisible=queryRecords(d.records,options());const requestedRecord=requested?initialVisible.find(r=>r.id===requested||r.other_saved_versions.some(v=>v.id===requested)):null;if(requestedRecord)await select(requestedRecord);
 }catch(e){if(full){$('library-status').textContent='The library could not be verified: '+e.message;$('library-counts').replaceChildren(el('span','Counts unavailable'));}if(preview)preview.replaceChildren(el('p','The curated library is temporarily unavailable.'));}
}
start();
})(globalThis);
