/* Default native geometry inspection; original video evidence remains separate. */
(function(root){'use strict';
function selection(catalog,sourceSha,component,models=root.Sam3DModelReviews){
 const record=catalog?.records?.find(r=>r.source.sha256===sourceSha);
 return{record:record||null,review:record&&models?.forComponent(record,component)||null};
}
if(typeof module!=='undefined'&&module.exports)module.exports={selection};
if(typeof document==='undefined')return;
root.TWIN_NATIVE_DEFAULT=new URLSearchParams(location.search).get('legacy')!=='1';
if(!root.TWIN_NATIVE_DEFAULT)return;
document.documentElement.classList.add('sam3d-workspace');
let epoch=0,catalog=null,lastKey=null;
const $=id=>document.getElementById(id),host=document.querySelector('.scene-box');
if(!host)return;
const frame=document.createElement('iframe');frame.id='workspace-native-model';frame.title='Native SAM 3D component';frame.hidden=true;
frame.setAttribute('sandbox','allow-scripts allow-same-origin');
const state=document.createElement('div');state.id='workspace-native-status';state.setAttribute('role','status');state.textContent='Loading the SAM 3D component library…';
host.append(frame,state);
const foot=document.createElement('div');foot.className='native-workspace-foot';
const label=document.createElement('label');label.textContent='SAM 3D part ';
const picker=document.createElement('select');picker.id='workspace-native-part';picker.setAttribute('aria-label','Choose a SAM 3D component');label.append(picker);
const library=document.createElement('a');library.textContent='Open model library ↗';library.href='../showcase/library.html';
foot.append(label,library);host.after(foot);
const note=document.createElement('p');note.className='native-workspace-note';note.textContent='Generated geometry · independent model coordinates. Video alignment, common assembly and physical fit remain unverified.';foot.after(note);
const advanced=document.createElement('details');advanced.className='native-workspace-advanced';
const summary=document.createElement('summary');summary.textContent='Advanced · saved motion experiments';
const legacy=document.createElement('a');legacy.textContent='Open the earlier comparison viewer ↗';
advanced.append(summary,legacy);note.after(advanced);
async function refreshCatalog(){
 // Refresh the hash pin as data, never execute a downloaded script. The builder
 // emits this same bounded index beside its JS bootstrap for live upload refresh.
 const response=await fetch('/showcase/library-index.json',{cache:'no-store'});
 if(response.ok){const value=await response.json();if(value?.format!=='interaction-twin-library-index'||value.version!==1)throw Error('Library index changed.');root.INTERACTION_LIBRARY_INDEX=value;}
 else if(response.status!==404)throw Error('Could not refresh the model library.');
 return root.InteractionLibrary.loadCatalog();
}
function paint(active,component,data){
 const selected=selection(data,active.interaction.source.sha256,component),record=selected.record;
 const rows=record?root.Sam3DModelReviews.forRecording(record):[];
 picker.replaceChildren();
 for(const track of active.interaction.tracklets){const option=document.createElement('option');option.value=track.id;option.textContent=track.label+(rows.some(r=>r.component_id===track.id)?' · SAM 3D':' · geometry pending');picker.append(option);}
 picker.value=component||'';picker.disabled=!active.interaction.tracklets.length;
 for(const row of document.querySelectorAll('#component-list [data-track]')){const badge=row.querySelector('.component-meta small');if(badge)badge.textContent=rows.some(r=>r.component_id===row.dataset.track)?'SAM 3D available':'Geometry pending';}
 const href=new URL('../showcase/library.html',location.href);if(record)href.searchParams.set('case',record.id);href.hash='sam3d-models';library.href=href.href;
 const old=new URL(location.href);old.searchParams.set('legacy','1');legacy.href=old.href;
 if(!selected.review){frame.hidden=true;frame.removeAttribute('src');state.hidden=false;state.textContent=rows.length?'SAM 3D geometry is not available for this component yet. Choose another part above or inspect its video evidence.':'SAM 3D reconstruction is pending for this recording. Its source video and descriptive procedure remain available.';lastKey=null;return;}
 const key=active.interaction.source.sha256+'/'+component+'/'+selected.review.native_glb.sha256;
 if(key!==lastKey){frame.hidden=true;frame.removeAttribute('src');const url=new URL('../showcase/native-model.html',location.href);url.searchParams.set('source',active.interaction.source.sha256);url.searchParams.set('component',component);frame.src=url.href;lastKey=key;}
 state.hidden=true;frame.hidden=false;
}
async function sync(refresh=false){
 const current=++epoch,api=root.TwinWorkspace,active=api?.active,component=api?.selectedObject;
 frame.hidden=true;state.hidden=false;state.textContent='Verifying this recording’s SAM 3D models…';
 if(!active){frame.removeAttribute('src');return;}
 try{if(refresh||!catalog)catalog=await refreshCatalog();if(current!==epoch||root.TwinWorkspace.active?.id!==active.id||root.TwinWorkspace.selectedObject!==component)return;paint(active,component,catalog);}
 catch(error){if(current!==epoch)return;frame.removeAttribute('src');state.hidden=false;state.textContent=error.message+' Source evidence remains available; no earlier model is substituted.';}
}
picker.addEventListener('change',()=>root.TwinWorkspace?.selectObject(picker.value));
document.addEventListener('twin:casechange',()=>queueMicrotask(()=>sync(true)));
document.addEventListener('twin:componentchange',()=>queueMicrotask(()=>sync(false)));
queueMicrotask(()=>sync(true));
})(globalThis);
