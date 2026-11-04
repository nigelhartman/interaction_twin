(function(){'use strict';
const $=id=>document.getElementById(id),video=$('sample-video');
const all=window.TWIN_CASES?.cases||[];
const definitions={
 jabra:{id:'flow-rigid-parts-jabra-20261031-v1--flow-procedure-1',title:'Earbuds & charging case',kind:'HINGED ASSEMBLY',description:'Four generated parts. A lid that stays connected. Actions linked to the original video.',chapters:[0,1,2,3],caveat:'Geometry uses Meta SAM 3D Objects. Components have independent coordinates; video alignment, assembly, contact and seating remain unverified.'},
 wallet:{id:'flow-rigid-parts-wallet-20261031-v1--flow-procedure-1',title:'Wallet & coin',kind:'FLEXIBLE OBJECT',description:'Opening, removing and returning a coin, described from the demonstration.',chapters:[0,1,3,5,6],caveat:'Coin identity across occlusion, flexible wallet geometry and hidden insertion are uncertain. Native asset inspection does not establish video placement or physical success.'},
 fresh:{id:'flow-capture-laptop-20261103-v1--flow-procedure-1',title:'Laptop & moving lid',kind:'FROM AN ORDINARY VIDEO',description:'Separate laptop parts, a connected lid and source-linked actions.',chapters:[0,1,2,4,5],caveat:'Drag the scene to explore. Release it to return to the video view.'}
};
definitions.jabra.source_sha256='7474ebd9d06c55b63fc073f632e71e5125353ace2538fa7bcf632cfdc1c3d0f2';
const deepParams=new URLSearchParams(location.search||''),deepSource=deepParams.get('source');
let deepTime=/^\d+(?:\.\d+)?$/.test(deepParams.get('t')||'')?Number(deepParams.get('t')):null;
let nativeReviews=[],nativeFrame=null;let active=null,definition=null,scene=null,currentStep=-1,seekAfterLoad=null,lastDraw=-1;
let aligned=null,alignedState=null,viewerMode='native',alignedView='source',showEstimated=true;
const jointFields=new Map();
let loadEpoch=0,loadController=null;
const reduce=window.matchMedia('(prefers-reduced-motion: reduce)');
function text(tag,value,cls){const e=document.createElement(tag);e.textContent=value;if(cls)e.className=cls;return e}
function fileURL(path){if(!path||/^(?:https?:|\/\/)/i.test(path))throw new Error('Expected a local project artifact');const clean=path.replace(/^\/+/, '');return location.protocol==='file:'?new URL('../'+clean,location.href).href:'/files/'+clean.split('/').map(encodeURIComponent).join('/')}
function supportedTimes(data){return [...new Set((data?.objects||[]).flatMap(o=>o.frames.filter(f=>f.visible).map(f=>f.timestamp_s)))].sort((a,b)=>a-b)}
function stepAt(steps,t){for(let i=steps.length-1;i>=0;i--)if(t>=steps[i].start_s&&t<=steps[i].end_s)return i;return-1}
function seek(t){if(!active)return;video.pause();const value=Math.max(0,Math.min(active.duration_s,Number(t)||0));video.currentTime=value;render(value,true)}
async function loadSample(key){
 const choice=definitions[key];if(!choice)return;let next=all.find(c=>c.id===choice.id)||all.find(c=>choice.source_sha256&&c.interaction.source.sha256===choice.source_sha256);
 
 
 if(!next){$('sample-description').textContent='The requested sample is not present. The current saved sample remains available.';return}
 const epoch=++loadEpoch;loadController?.abort();loadController=null;video.pause();
 let reviews=[],common=null;
 const results=await Promise.allSettled([InteractionLibrary.loadCatalog(),window.AlignedScenes?AlignedScenes.find(next.interaction.source.sha256):Promise.resolve(null)]);
 if(epoch!==loadEpoch)return;
 if(results[0].status==='fulfilled'){const catalog=results[0].value,record=catalog.records.find(r=>r.id===choice.id)||catalog.records.find(r=>r.source.sha256===next.interaction.source.sha256);reviews=Sam3DModelReviews.forRecording(record);if(record){try{next=await InteractionLibrary.loadSourceCase(record);if(epoch!==loadEpoch)return;}catch(error){if(epoch!==loadEpoch)return;$('sample-description').textContent='The current sample could not be verified. Choose another recording or open the library.';return;}}}
 if(results[1].status==='fulfilled'&&results[1].value&&AlignedSceneContract.matchesProgram(results[1].value.manifest,next,reviews,results[1].value.derived_assembly))common=results[1].value;
 nativeReviews=reviews;aligned=common;alignedState=null;

 next={...next,scene:{objects:[],timeline:[],native_only:true}};
 $('sample-panel').inert=false;$('sample-scene').style.visibility='';return installSample(key,choice,next);
}
function installSample(key,choice,next){
 definition=choice;active=next;
 video.pause();scene?.dispose();scene=null;currentStep=-1;lastDraw=-1;
 for(const b of document.querySelectorAll('[data-sample]')){const selected=b.dataset.sample===key;b.setAttribute('aria-selected',String(selected));b.tabIndex=selected?0:-1}
 $('sample-panel').setAttribute('aria-labelledby',key==='linked'?'sample-title':'sample-'+key);const badge=document.querySelector('#sample-'+key+' small');if(badge)badge.textContent=active.labels.steps+' source-linked steps';
 $('sample-title').textContent=definition.title;$('sample-kind').textContent=definition.kind;$('sample-description').textContent=definition.description;$('sample-caveat').textContent=definition.caveat;$('source-count').textContent=active.labels.source_samples+' source samples';
 $('sample-numbers').replaceChildren();for(const [number,label]of[[nativeReviews.length,'SAM 3D models'],[active.labels.steps,'steps'],[active.interaction.assertions.length,'state assertions']]){const e=text('span',label);e.prepend(text('strong',number));$('sample-numbers').append(e)}
 $('sample-workspace').href='library.html?case='+encodeURIComponent(active.id);
 $('sample-scrubber').max=active.duration_s;$('sample-coverage').replaceChildren();for(const t of supportedTimes(active.scene)){const e=document.createElement('i');e.style.left=(100*t/active.duration_s)+'%';$('sample-coverage').append(e)}
 $('scene-error').hidden=true;$('asset-choice').replaceChildren();const objects=active.scene?.objects||[];
 for(const r of nativeReviews){const option=text('option',r.label);option.value=r.component_id;$('asset-choice').append(option);}
 $('sample-scene').hidden=true;const available=Boolean(aligned?.manifest.components.some(c=>c.poses.some(p=>p.camera_from_native_mesh)));
 $('asset-mode').disabled=!available;const alignedOption=Array.from($('asset-mode').options||[]).find(o=>o.value==='aligned');if(alignedOption){alignedOption.disabled=!available;alignedOption.textContent='Interactive scene';}
 chooseMode(available?'aligned':'native');

 buildChapters();
 const times=supportedTimes(active.scene);seekAfterLoad=viewerMode==='aligned'&&aligned?AlignedSceneContract.previewTime(aligned.manifest,aligned.observed_surfaces?.document).time_s:active.scene?.native_only?0:active.scene?.initial_source_time_s??times[0]??0;
 if(deepTime!==null){seekAfterLoad=Math.max(0,Math.min(active.duration_s,deepTime));deepTime=null;}
 const nearest=active.interaction.evidence.reduce((a,b)=>Math.abs(a.timestamp_s-seekAfterLoad)<Math.abs(b.timestamp_s-seekAfterLoad)?a:b);
 video.poster=fileURL(nearest.frame_path);video.src=fileURL(active.video_path);video.load();render(seekAfterLoad,true);scene?.fitFrame();
 const step=stepAt(active.interaction.steps,seekAfterLoad);showStep(step<0?0:step,false);
}
function chooseNative(id){
 viewerMode='native';const row=nativeReviews.find(r=>r.component_id===id),frame=$('sample-native-frame');frame.hidden=!row;nativeFrame=row;
 if(row){frame.loading='eager';frame.hidden=false;frame.src='native-model.html?'+new URLSearchParams({source:row.source_sha256,component:row.component_id});$('scene-error').hidden=true;$('scene-title').textContent='Individual 3D part · '+row.label;}
 else{frame.removeAttribute('src');$('scene-error').textContent='SAM 3D geometry is pending for this recording. Its source video and Muse procedure are available.';$('scene-error').hidden=false;}
}
function postAligned(reset=false){if(viewerMode!=='aligned'||!aligned)return;const frame=$('sample-native-frame');frame.contentWindow?.postMessage({type:'interaction-twin:aligned-time',source_sha256:active.interaction.source.sha256,manifest_sha256:aligned.reference.sha256,time_s:Number(video.currentTime)||0,playing:!video.paused,show_estimated:showEstimated,view:alignedView,reset},location.origin);}
function chooseMode(mode){
 viewerMode=mode==='aligned'&&aligned?'aligned':'native';$('asset-mode').value=viewerMode;$('sample-caveat').textContent=viewerMode==='aligned'?'Drag to explore · returns to the video view after 3 seconds.':'Inspect each original generated part.';buildJointInspect();
 const estimates=$('estimated-motion-control');if(estimates)estimates.hidden=true;
 $('view-camera').disabled=viewerMode!=='aligned';$('view-orbit').disabled=viewerMode!=='aligned';
 $('asset-choice').disabled=viewerMode==='aligned'||!nativeReviews.length;$('asset-choice').hidden=viewerMode==='aligned'||!nativeReviews.length;$('sample-native-frame').title=viewerMode==='aligned'?'Interactive 3D scene':'Individual generated 3D part';
 if(viewerMode==='aligned'){nativeFrame=null;alignedState=null;const ref=aligned.reference,frame=$('sample-native-frame');frame.removeAttribute('src');frame.loading='eager';frame.hidden=false;frame.src='aligned-scene.html?'+new URLSearchParams({source:active.interaction.source.sha256,path:ref.path,sha256:ref.sha256,bytes:String(ref.bytes),presentation:'landing'});frame.hidden=false;$('scene-error').hidden=true;$('scene-title').textContent='Interactive 3D scene';postAligned();}
 else chooseNative($('asset-choice').value||nativeReviews[0]?.component_id);
}
function buildJointInspect(){
 const host=$('joint-inspection');if(!host)return;host.replaceChildren();jointFields.clear();const joints=viewerMode==='aligned'?[...(aligned?.manifest.mechanics?.joints||[]),...(aligned?.manifest.mechanics?.unresolved_children||[])]:[];host.hidden=!joints.length;
 for(const joint of joints){const row=text('div','','joint-inspect-row'),name=text('strong',joint.label||aligned.manifest.components.find(c=>c.id===joint.child_id)?.label||joint.id),value=text('span','Reading the source-linked joint estimate…'),actions=text('div','','joint-source-actions');jointFields.set(joint.id,value);row.append(name,value);
 const known=(joint.samples||[]).filter(p=>p.status==='estimated'&&Number.isFinite(p.q_rad)),low=known.reduce((a,b)=>!a||b.q_rad<a.q_rad?b:a,null),high=known.reduce((a,b)=>!a||b.q_rad>a.q_rad?b:a,null);
 for(const[label,sample]of[['Lowest angle',low],['Highest angle',high]]){const b=text('button',label);b.type='button';b.disabled=!sample;b.onclick=()=>{if(sample)seek(sample.timestamp_s);};actions.append(b);}row.append(actions);host.append(row);
 if(joint.type==='revolute'&&aligned.manifest.mechanics.joints.includes(joint)){const details=text('details','','joint-contract'),summary=text('summary','Inspect connection rules'),list=text('dl',''),numbers=v=>v.map(n=>Number(n).toFixed(4)).join(', ');details.append(summary,list);for(const[label,value]of[['Connection',joint.parent_id+' → '+joint.child_id+' · revolute'],['Axis',numbers(joint.axis)+' · joint frame'],['Parent anchor',numbers(joint.parent_anchor_native)+' · native mesh coordinates'],['Child anchor',numbers(joint.child_anchor_native)+' · native mesh coordinates'],['Fitted configuration range',joint.limits_rad.map(n=>(n*180/Math.PI).toFixed(1)+'°').join(' to ')],['Physical limits','Unknown · this fit does not measure physical stops']])list.append(text('dt',label),text('dd',value));if(joint.angle_quality?.angle_visual_consistency===false)details.append(text('p','Opening angle unresolved: fitted configurations do not consistently distinguish the cited open and closed states.'));const link=text('a','Read saved joint data ↗');link.href=fileURL(aligned.reference.path);link.target='_blank';link.rel='noopener';details.append(link);host.append(details);}else if(joint.reason)host.append(text('p',joint.reason,'joint-unresolved'));}
 if(aligned?.object_program){const link=text('a',aligned.derived_assembly?'Download derived assembly program ↗':'Download descriptive object program ↗','text-link');link.href=fileURL(aligned.object_program.path);link.target='_blank';link.rel='noopener';host.append(link);}
}
function updateJointInspect(){for(const[id,node]of jointFields){const joint=alignedState?.joints?.find(j=>j.id===id),spec=aligned?.manifest.mechanics?.joints.find(j=>j.id===id),unresolved=spec?.angle_quality?.angle_visual_consistency===false;node.textContent=joint&&joint.q_rad!==null?(unresolved?'Configuration ':'')+Math.round(joint.q_rad*180/Math.PI)+(unresolved?'° · opening angle unresolved':'° estimate')+(joint.interpolated?' · between source samples':Number.isSafeInteger(joint.source_sample_index)?' · source frame '+joint.source_sample_index:' · saved source time'):'Joint unavailable here · connected parts stay together or hidden';}}
window.addEventListener?.('message',event=>{
 if(viewerMode!=='aligned'||!aligned||event.origin!==location.origin||event.source!==$('sample-native-frame').contentWindow)return;
 const d=event.data;if(!d||d.source_sha256!==active.interaction.source.sha256||d.manifest_sha256!==aligned.reference.sha256)return;
 if(d.type==='interaction-twin:aligned-ready'){postAligned();return;}
 if(d.type==='interaction-twin:aligned-state'&&Array.isArray(d.parts)&&d.parts.length<=32){const ids=new Set(aligned.manifest.components.map(c=>c.id)),jids=new Set([...(aligned.manifest.mechanics?.joints||[]),...(aligned.manifest.mechanics?.unresolved_children||[])].map(j=>j.id));if(d.parts.some(p=>!ids.has(p.id)||!['observed','estimated','unknown'].includes(p.status)||typeof p.visible!=='boolean')||d.joints&&(!Array.isArray(d.joints)||d.joints.length>32||d.joints.some(j=>!jids.has(j.id)||!['estimated','unknown'].includes(j.status)||j.q_rad!==null&&!Number.isFinite(j.q_rad))))return;alignedState=d;render(video.currentTime,true,false);}
 if(d.type==='interaction-twin:aligned-error'){$('scene-error').hidden=false;$('scene-error').textContent=String(d.message||'The estimated scene could not load.')+' No substitute is shown.';}
});
function buildChapters(){
 const host=$('sample-chapters');host.replaceChildren();const unique=active.interaction.steps.map((_,i)=>i);
 for(const i of unique){const step=active.interaction.steps[i],b=document.createElement('button');b.dataset.step=i;b.append(text('span',String(i+1).padStart(2,'0')),document.createTextNode(step.title));b.onclick=()=>showStep(i,true);host.append(b)}
}
function showStep(i,move){
 const s=active.interaction.steps[i];if(!s)return;currentStep=i;
 for(const b of $('sample-chapters').children)b.classList.toggle('active',Number(b.dataset.step)===i);
 $('selected-step-title').textContent=s.title;$('selected-step-copy').textContent=s.description;$('selected-step-limit').textContent=s.limitations.find(value=>!/[a-z]+_[a-z]+|range beginning|cited only|target domain|window w\d/.test(value))||'A visual model proposal; hidden contact and physical completion remain unverified.';$('step-time').textContent=s.start_s.toFixed(3)+'–'+s.end_s.toFixed(3)+' S · STEP '+String(i+1).padStart(2,'0');
 if(move)seek(s.start_s);
}
function render(t,force=false,emit=true){
 if(!active||(!force&&Math.abs(t-lastDraw)<.008))return;lastDraw=t;$('sample-time').textContent=t.toFixed(3)+' s';$('sample-scrubber').value=t;$('demo-play').textContent=video.paused?'▷':'Ⅱ';$('demo-play').setAttribute('aria-label',video.paused?'Play sample':'Pause sample');
 const rows=scene?.setTime(t)||[],native=scene?.inspection==='native';$('pose-pills').replaceChildren();let visible=0,age=0;
 for(const row of rows){const obj=active.scene.objects.find(o=>o.object_id===row.id);if(row.visible)visible++;age=Math.max(age,row.visible?row.age:0);$('pose-pills').append(text('span',obj.label+': '+(native?'native asset':row.estimated?'estimated gap · display only':row.visible?row.row.reference?'reference':'conditional pose':'unknown'),'pose-pill'+(row.visible&&!native?' visible':'')))}
 $('scene-unavailable').hidden=native||visible>0||!scene;$('scene-title').textContent=native?'Separate generated asset':scene?.inspection==='component'?'Independent component motion':'Generated components';$('pose-note').textContent=native?'Native asset inspection · no video placement.':rows.some(r=>r.estimated)?'Estimated between saved endpoints · raw evidence unchanged.':visible?'Conditional pose · sample age '+age.toFixed(3)+' s · unverified scale.':'Unsupported poses stay hidden.';
 $('view-orbit').classList.toggle('active',scene?.mode==='orbit');$('view-camera').classList.toggle('active',scene?.mode==='source');$('view-camera').disabled=!scene||native||!scene.sourceProjection;
 if(nativeFrame){$('scene-title').textContent='Individual 3D part · '+nativeFrame.label;$('pose-note').textContent='Independent component coordinates · video alignment pending';$('scene-unavailable').hidden=true;$('pose-pills').replaceChildren(text('span','SAM 3D native component','pose-pill'));}
 if(viewerMode==='aligned'&&aligned){$('scene-title').textContent='Interactive 3D scene';$('pose-note').textContent=(aligned.manifest.mechanics?.joints.length||aligned.manifest.mechanics?.unresolved_children?.length)?'Connected motion · depth and joint fit are estimates':'Follows visible masks · depth and assembly unverified';if(aligned.observed_surfaces)$('pose-note').textContent='Source-visible surfaces with estimated depth · hidden surfaces and numerical fold parameters unknown.';updateJointInspect();$('scene-unavailable').hidden=true;$('view-camera').disabled=false;$('view-orbit').disabled=false;$('view-camera').classList.toggle('active',alignedView==='source');$('view-orbit').classList.toggle('active',alignedView==='orbit');$('pose-pills').replaceChildren(...(alignedState?.parts||[]).map(p=>text('span',p.label+': '+(p.visible?p.interpolated?'between samples':'saved estimate':'unknown'),'pose-pill'+(p.visible?' visible':''))));if(emit)postAligned();}
 const i=stepAt(active.interaction.steps,t);if(i>=0&&i!==currentStep)showStep(i,false);
}
video.addEventListener('loadedmetadata',()=>{if(seekAfterLoad!==null){const t=seekAfterLoad;seekAfterLoad=null;seek(t);scene?.fitFrame()}});
for(const e of['timeupdate','play','pause','seeked'])video.addEventListener(e,()=>render(video.currentTime,true));
video.addEventListener('error',()=>{$('sample-description').textContent='The saved video could not load. The generated components and full project evidence remain available.'});
 $('sample-scrubber').oninput=e=>seek(e.target.value);
$('demo-play').onclick=()=>{if(video.paused)video.play().catch(()=>{});else video.pause()};
$('asset-mode').onchange=()=>{chooseMode($('asset-mode').value);render(video.currentTime,true)};
$('asset-choice').onchange=()=>{chooseNative($('asset-choice').value);render(video.currentTime,true)};
$('view-orbit').onclick=()=>{if(viewerMode==='aligned'){alignedView='orbit';render(video.currentTime,true)}};
$('view-camera').onclick=()=>{if(viewerMode==='aligned'){alignedView='source';render(video.currentTime,true)}};
$('reset-view').onclick=()=>{if(viewerMode==='aligned')postAligned(true);else if(nativeFrame)chooseNative(nativeFrame.component_id);};
if($('show-estimated-motion')){$('show-estimated-motion').checked=true;$('show-estimated-motion').onchange=()=>{showEstimated=$('show-estimated-motion').checked;render(video.currentTime,true)};}
$('jump-pose').onclick=()=>{const shown=scene?.objectsInView?.()||active.scene?.objects||[];const times=supportedTimes({objects:shown});if(times.length){seek(times.find(t=>t>=video.currentTime)??times[0]);scene?.fitFrame()}};
$('step-source').onclick=()=>{const s=active.interaction.steps[currentStep],e=active.interaction.evidence.find(e=>e.id===s.evidence_ids[0]);if(e)window.open(fileURL(e.frame_path),'_blank','noopener')};
const tabs=Array.from(document.querySelectorAll('[data-sample]'));tabs.forEach((b,i)=>{b.onclick=()=>loadSample(b.dataset.sample);b.onkeydown=e=>{let n;if(e.key==='ArrowRight')n=(i+1)%tabs.length;else if(e.key==='ArrowLeft')n=(i+tabs.length-1)%tabs.length;else if(e.key==='Home')n=0;else if(e.key==='End')n=tabs.length-1;else return;e.preventDefault();loadSample(tabs[n].dataset.sample);tabs[n].focus()}});
const film=$('film-dialog'),trailer=$('trailer');if($('watch-film'))$('watch-film').onclick=()=>{video.pause();film.showModal();trailer.play().catch(()=>{})};if($('close-film'))$('close-film').onclick=()=>film.close();film?.addEventListener('close',()=>trailer.pause());
async function processor(){if(location.protocol==='file:'){$('processor-status').textContent='Start Launch Workspace.command to process a new video.';return}try{const r=await fetch('/health');const d=await r.json();$('processor-status').textContent=r.ok&&d.status==='ready'?'Local processor ready · hosted reasoning requires consent.':'Start the local launcher to process a video.'}catch{$('processor-status').textContent='Processor offline · saved samples remain inspectable.'}}
$('sample-scene').addEventListener('twin-display-change',()=>render(Number(video.currentTime)||0,true));
function tick(){if(!video.paused&&!video.ended)render(video.currentTime);requestAnimationFrame(tick)}requestAnimationFrame(tick);
const linked=/^[a-f0-9]{64}$/.test(deepSource||'')?all.find(c=>c.interaction.source.sha256===deepSource):null;
if(linked){definitions.linked={id:linked.id,title:linked.title||linked.case_id||'Recorded interaction',kind:'CURATED OBJECT PROGRAM',description:'Inspect generated parts and source-linked task structure from this recording.',caveat:'Generated geometry and estimated mechanisms require inspection; physical accuracy remains unverified.'};loadSample('linked');}else loadSample('jabra');processor();
})();
