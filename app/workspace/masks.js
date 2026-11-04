/* Inspect only the current procedure's actual component PNGs at exact source PTS.
 * This controller never infers, uploads, propagates, interpolates or holds masks.
 */
(function(root){'use strict';
 const EPSILON_S=1e-4,PALETTE=['#76dbc0','#b6a4ef','#e9bd80','#91bbea','#ea9bb4','#a7d37c','#edaa78'];
 function colorFor(id){let h=2166136261;for(const c of String(id)){h^=c.charCodeAt(0);h=Math.imul(h,16777619);}return PALETTE[(h>>>0)%PALETTE.length];}
 function exactEvidence(rows,time){if(!Number.isFinite(time))return null;const matches=(rows||[]).filter(e=>Number.isFinite(e.timestamp_s)&&Math.abs(e.timestamp_s-time)<=EPSILON_S);return matches.length===1?matches[0]:null;}
 function nearestEvidence(rows,time){const available=(rows||[]).filter(e=>Number.isFinite(e.timestamp_s)&&e.observations?.length);if(!available.length)return null;const t=Number.isFinite(time)?time:0;return available.reduce((a,b)=>Math.abs(b.timestamp_s-t)<Math.abs(a.timestamp_s-t)?b:a);}
 function safeLocalURL(path,workspace,location){
  if(typeof path!=='string'||!path.startsWith('data/')||path.includes('\\')||path.split('/').some(s=>!s||s==='.'||s==='..')||!/^[-a-zA-Z0-9_./]+$/.test(path))throw new Error('Component evidence must be a local data artifact.');
  const resolved=new URL(workspace.url(path),location.href);
  if(resolved.protocol!==location.protocol||resolved.origin!==location.origin||!['http:','https:','file:'].includes(resolved.protocol))throw new Error('External component evidence is not allowed.');
  return resolved.href;
 }
 function binaryColorPixels(pixels,width,height,color){
  if(pixels.length!==width*height*4)throw new Error('Mask pixel dimensions are inconsistent.');
  const mask=new Uint8Array(width*height);let area=0;
  for(let i=0;i<mask.length;i++){
   const p=i*4,a=pixels[p+3];if(a===0)continue;
   const v=pixels[p];if(a!==255||pixels[p+1]!==v||pixels[p+2]!==v||(v!==0&&v!==255))throw new Error('The saved PNG is not a binary component mask.');
   if(v===255){mask[i]=1;area++;}
  }
  const rgb=[1,3,5].map(i=>parseInt(color.slice(i,i+2),16)),result=new Uint8ClampedArray(pixels.length);
  for(let i=0;i<mask.length;i++)if(mask[i]){
   const x=i%width,y=Math.floor(i/width),edge=x===0||y===0||x===width-1||y===height-1||!mask[i-1]||!mask[i+1]||!mask[i-width]||!mask[i+width],p=i*4;
   result[p]=rgb[0];result[p+1]=rgb[1];result[p+2]=rgb[2];result[p+3]=edge?225:92;
  }
  return{pixels:result,area};
 }
 function auditLabel(value){if(value==='plausible_visible_mask')return'model: plausible';if(value==='reject_wrong_or_lost_mask')return'model: rejected';if(value==='not_model_audited'||!value)return'not audited';return'model: '+String(value).replaceAll('_',' ');}

 class Controller{
  constructor({document,workspace,video,toggle,canvas,caption,ImageClass=root.Image,location=root.location}){
   this.document=document;this.workspace=workspace;this.video=video;this.toggle=toggle;this.canvas=canvas;this.caption=caption;this.ImageClass=ImageClass;this.location=location;this.epoch=0;this.signature=null;this.cache=new Map();this.listeners=[];this.caseId=null;
   Object.assign(canvas.style,{position:'absolute',inset:'0',width:'100%',height:'100%',objectFit:'contain',pointerEvents:'none'});
   Object.assign(caption.style,{maxWidth:'calc(100% - 20px)',whiteSpace:'normal',lineHeight:'1.55'});
   canvas.setAttribute('aria-label','Actual saved component mask overlay at one exact source sample');canvas.setAttribute('aria-hidden','true');caption.setAttribute('role','status');caption.setAttribute('aria-live','polite');
   this.listen(toggle,'change',()=>this.enabled());
   for(const name of ['play','playing','seeking','emptied'])this.listen(video,name,()=>this.invalidate(name==='play'||name==='playing'?'Component masks are hidden during playback. Pause at a saved source sample.':'Waiting for an exact saved source sample.'));
   for(const name of ['pause','seeked','timeupdate','loadedmetadata','loadeddata'])this.listen(video,name,()=>this.render());
   this.listen(document,'twin:casechange',()=>this.caseChanged());
   const hand=document.getElementById('show-masks');if(hand)this.listen(hand,'change',()=>{if(hand.checked){toggle.checked=false;this.invalidate();}});
   this.caseChanged();
  }
  listen(target,name,fn){target.addEventListener(name,fn);this.listeners.push([target,name,fn]);}
  get active(){return this.workspace.active;}
  get rows(){return this.active?.interaction?.evidence||[];}
  isCurrent(epoch,active,row){return epoch===this.epoch&&this.toggle.checked&&this.active===active&&this.video.paused&&!this.video.seeking&&exactEvidence(this.rows,Number(this.video.currentTime))===row;}
  clear(){this.canvas.hidden=true;const c=this.canvas.getContext('2d');c?.clearRect(0,0,this.canvas.width,this.canvas.height);}
  invalidate(message=''){this.epoch++;this.signature=null;this.clear();this.caption.hidden=!(message&&this.toggle.checked);this.caption.textContent=message&&this.toggle.checked?message:'';}
  caseChanged(){this.caseId=this.active?.id||null;this.toggle.checked=false;this.toggle.disabled=!this.rows.some(e=>e.observations?.length);this.toggle.title=this.toggle.disabled?'No saved component masks exist for this recording.':'Pause and inspect the nearest actual source mask sample.';this.cache.clear();this.invalidate();}
  enabled(){
   if(!this.toggle.checked){this.invalidate();return;}
   const row=nearestEvidence(this.rows,Number(this.video.currentTime));if(!row){this.toggle.checked=false;this.invalidate();return;}
   const hand=this.document.getElementById('show-masks');if(hand?.checked){hand.checked=false;hand.dispatchEvent(new root.Event('change',{bubbles:true}));}
   this.video.pause();this.invalidate('Loading saved component masks at the nearest source sample…');this.workspace.seek(row.timestamp_s);this.render();
  }
  image(path){
   const url=safeLocalURL(path,this.workspace,this.location);if(this.cache.has(url))return this.cache.get(url);
   const pending=new Promise((resolve,reject)=>{const image=new this.ImageClass();image.onload=()=>resolve(image);image.onerror=()=>{this.cache.delete(url);reject(new Error('A saved source image or mask could not be loaded.'));};image.src=url;});
   this.cache.set(url,pending);while(this.cache.size>48)this.cache.delete(this.cache.keys().next().value);return pending;
  }
  async render(){
   if(!this.toggle.checked){if(!this.canvas.hidden)this.invalidate();return;}
   const active=this.active;if(!active||active.id!==this.caseId){this.caseChanged();return;}
   if(!this.video.paused||this.video.seeking){this.invalidate(!this.video.paused?'Component masks are hidden during playback. Pause at a saved source sample.':'Waiting for an exact saved source sample.');return;}
   const row=exactEvidence(this.rows,Number(this.video.currentTime));
   if(!row||!row.observations?.length){this.invalidate('No component masks at this exact source time. Toggle Component masks to jump to the nearest saved sample.');return;}
   if(!(this.video.videoWidth>0&&this.video.videoHeight>0)){this.invalidate('Waiting for the source video dimensions.');return;}
   const key=active.id+'|'+row.id+'|'+this.video.videoWidth+'x'+this.video.videoHeight;if(this.signature===key)return;
   this.invalidate('Loading actual saved component PNGs…');this.signature=key;const epoch=this.epoch;
   try{
    const frame=await this.image(row.frame_path);if(!this.isCurrent(epoch,active,row))return;
    const width=frame.naturalWidth,height=frame.naturalHeight;
    if(width!==this.video.videoWidth||height!==this.video.videoHeight)throw new Error('Source-frame and video dimensions differ. Masks are withheld to avoid an incorrect overlay.');
    const masks=await Promise.all(row.observations.map(async observation=>({observation,image:await this.image(observation.mask_path)})));if(!this.isCurrent(epoch,active,row))return;
    const scratch=this.document.createElement('canvas');scratch.width=width;scratch.height=height;const sc=scratch.getContext('2d',{willReadFrequently:true});if(!sc)throw new Error('Canvas mask inspection is unavailable.');
    const layers=[];
    for(const item of masks){
     if(item.image.naturalWidth!==width||item.image.naturalHeight!==height)throw new Error('A saved mask has different dimensions from its cited source frame.');
     sc.clearRect(0,0,width,height);sc.drawImage(item.image,0,0);const colored=binaryColorPixels(sc.getImageData(0,0,width,height).data,width,height,colorFor(item.observation.track_id));layers.push({...item,...colored});
    }
    if(!this.isCurrent(epoch,active,row))return;this.canvas.width=width;this.canvas.height=height;const target=this.canvas.getContext('2d');if(!target)throw new Error('Canvas mask display is unavailable.');target.clearRect(0,0,width,height);
    for(const layer of layers){const pixels=sc.createImageData(width,height);pixels.data.set(layer.pixels);sc.putImageData(pixels,0,0);target.drawImage(scratch,0,0);}
    if(!this.isCurrent(epoch,active,row)){this.clear();return;}this.canvas.hidden=false;this.caption.replaceChildren();
    const title=this.document.createElement('strong');title.textContent='Saved component masks · '+row.timestamp_s.toFixed(6)+' s';this.caption.append(title);
    for(const layer of layers){const badge=this.document.createElement('span');badge.textContent=layer.observation.track_id+' · '+auditLabel(layer.observation.audit_status)+(layer.area===0?' · empty mask':'');Object.assign(badge.style,{display:'block',color:colorFor(layer.observation.track_id)});this.caption.append(badge);}
    const note=this.document.createElement('small');note.textContent='One exact saved source sample · model audit is not measured accuracy';Object.assign(note.style,{display:'block',fontSize:'9px',color:'#a5b6c5'});this.caption.append(note);this.caption.hidden=false;
   }catch(error){if(epoch===this.epoch){this.signature=null;this.clear();this.caption.textContent='Component masks withheld: '+error.message;this.caption.hidden=!this.toggle.checked;}}
  }
  dispose(){for(const[t,n,f]of this.listeners)t.removeEventListener(n,f);this.listeners=[];this.cache.clear();this.invalidate();}
 }
 const api={Controller,exactEvidence,nearestEvidence,binaryColorPixels,colorFor,safeLocalURL,EPSILON_S};root.TwinComponentMasks=api;
 if(typeof module!=='undefined'&&module.exports)module.exports=api;
 const d=root.document,w=root.TwinWorkspace;if(d&&w){const video=d.getElementById('video'),toggle=d.getElementById('show-component-masks'),canvas=d.getElementById('sam-component-overlay'),caption=d.getElementById('sam-component-caption');if(video&&toggle&&canvas&&caption)api.controller=new Controller({document:d,workspace:w,video,toggle,canvas,caption});}
})(globalThis);
