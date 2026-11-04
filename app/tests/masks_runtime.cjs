/* Actual masks.js with DOM/canvas doubles: state, provenance and async contracts.
 * This is not a browser pixel test or model inference.
 */
'use strict';
const assert=require('node:assert/strict');
const M=require('../workspace/masks.js');
const location={href:'http://127.0.0.1:8765/workspace/index.html',origin:'http://127.0.0.1:8765',protocol:'http:'};
class Element{
 constructor(){this.listeners={};this.style={};this.hidden=true;this.checked=false;this.disabled=false;this.children=[];this.textContent='';}
 addEventListener(n,f){(this.listeners[n]??=[]).push(f);}
 removeEventListener(n,f){this.listeners[n]=(this.listeners[n]||[]).filter(x=>x!==f);}
 dispatchEvent(event){for(const f of this.listeners[event.type]||[])f(event);}
 emit(type){this.dispatchEvent({type});}
 setAttribute(n,v){this[n]=v;}
 append(...items){this.children.push(...items);}
 replaceChildren(...items){this.children=items;this.textContent='';}
}
class Canvas extends Element{
 constructor(){super();this.width=2;this.height=2;this.context={draws:0,data:new Uint8ClampedArray(16),clearRect:()=>{this.context.data=new Uint8ClampedArray(this.width*this.height*4);this.context.draws=0;},drawImage:image=>{this.context.draws++;this.context.data=new Uint8ClampedArray(image.context?.data||image.pixels);},getImageData:()=>({data:new Uint8ClampedArray(this.context.data)}),createImageData:(w,h)=>({data:new Uint8ClampedArray(w*h*4)}),putImageData:im=>{this.context.data=new Uint8ClampedArray(im.data);}};}
 getContext(){return this.context;}
}
function pixels(values){return new Uint8ClampedArray(values.flatMap(v=>[v,v,v,255]));}
function doc(id='case-a'){
 return{id,interaction:{evidence:[1,2].map(i=>({id:'source_'+i,timestamp_s:i,frame_path:`data/${id}/frame${i}.jpg`,observations:[{track_id:'part',mask_path:`data/${id}/mask${i}.png`,audit_status:i===1?'plausible_visible_mask':'reject_wrong_or_lost_mask'}]}))}};
}
function fixture(){
 const pending=[],images=new Map(),document=new Element(),video=new Element(),toggle=new Element(),canvas=new Canvas(),caption=new Element(),hand=new Element();
 video.paused=true;video.currentTime=1.4;video.seeking=false;video.videoWidth=2;video.videoHeight=2;video.pause=()=>{video.paused=true;video.emit('pause');};
 const elements={'show-masks':hand};document.getElementById=id=>elements[id]||null;document.createElement=tag=>tag==='canvas'?new Canvas():new Element();
 const workspace={active:doc(),url:p=>'/files/'+p,seek:t=>{video.currentTime=t;video.emit('seeked');}};
 class Image{set src(url){this.url=url;pending.push(this);}}
 const controller=new M.Controller({document,workspace,video,toggle,canvas,caption,ImageClass:Image,location});
 function provide(id='case-a'){for(const i of [1,2]){images.set('/files/data/'+id+'/frame'+i+'.jpg',{naturalWidth:2,naturalHeight:2,pixels:pixels([20,30,40,50])});images.set('/files/data/'+id+'/mask'+i+'.png',{naturalWidth:2,naturalHeight:2,pixels:pixels(i===1?[0,255,0,255]:[255,0,0,0])});}}
 provide();
 async function resolveAll(){for(let round=0;round<12;round++){while(pending.length){const im=pending.shift(),u=new URL(im.url);const def=images.get(u.pathname);if(!def){im.onerror();continue;}Object.assign(im,def);im.onload();}await Promise.resolve();}}
 return{controller,document,workspace,video,toggle,canvas,caption,hand,pending,images,provide,resolveAll};
}
function contents(e){return e.textContent+' '+e.children.map(contents).join(' ');}
(async()=>{
 let checks=0;
 assert.equal(M.exactEvidence([{timestamp_s:1}],1.00001).timestamp_s,1);assert.equal(M.exactEvidence([{timestamp_s:1}],1.001),null);assert.equal(M.exactEvidence([{timestamp_s:1},{timestamp_s:1.00001}],1),null);checks+=3;
 assert.equal(M.nearestEvidence([{timestamp_s:1,observations:[{}]},{timestamp_s:2,observations:[{}]}],1.6).timestamp_s,2);checks++;
 assert.throws(()=>M.safeLocalURL('https://bad.example/mask.png',{url:p=>p},location));assert.throws(()=>M.safeLocalURL('data/a/../secret.png',{url:p=>p},location));assert.throws(()=>M.safeLocalURL('data/a/mask.png',{url:()=> 'https://bad.example/mask.png'},location));checks+=3;
 const colored=M.binaryColorPixels(pixels([0,255,0,255]),2,2,M.colorFor('part'));assert.equal(colored.area,2);assert.equal(colored.pixels[3],0);assert.equal(colored.pixels[7],225);assert.equal(M.colorFor('part'),M.colorFor('part'));assert.throws(()=>M.binaryColorPixels(pixels([0,128,0,255]),2,2,'#ffffff'));checks+=5;
 {
  const f=fixture();assert.equal(f.toggle.disabled,false);assert.equal(f.toggle.checked,false);f.hand.checked=true;f.toggle.checked=true;f.toggle.emit('change');assert.equal(f.video.paused,true);assert.equal(f.video.currentTime,1);assert.equal(f.hand.checked,false);await f.resolveAll();assert.equal(f.canvas.hidden,false);assert.equal(f.canvas.context.draws,1);assert.match(contents(f.caption),/Saved component masks/);assert.match(contents(f.caption),/model: plausible/);assert.match(contents(f.caption),/not measured accuracy/);checks+=10;
  f.video.paused=false;f.video.emit('play');assert.equal(f.canvas.hidden,true);assert.match(f.caption.textContent,/hidden during playback/);f.video.paused=true;f.video.currentTime=1.5;f.video.emit('pause');await f.resolveAll();assert.equal(f.canvas.hidden,true);assert.match(f.caption.textContent,/No component masks at this exact source time/);checks+=4;
  f.video.currentTime=2;f.video.emit('seeked');await f.resolveAll();assert.equal(f.canvas.hidden,false);assert.match(contents(f.caption),/2.000000 s/);assert.match(contents(f.caption),/model: rejected/);checks+=3;
  f.hand.checked=true;f.hand.emit('change');assert.equal(f.toggle.checked,false);assert.equal(f.canvas.hidden,true);checks+=2;f.controller.dispose();
 }
 {
  const f=fixture();f.toggle.checked=true;f.toggle.emit('change');assert.equal(f.pending.length,1);f.video.seeking=true;f.video.currentTime=2;f.video.emit('seeking');f.video.seeking=false;f.video.emit('seeked');await f.resolveAll();assert.equal(f.canvas.hidden,false);assert.match(contents(f.caption),/2.000000 s/);assert.doesNotMatch(contents(f.caption),/1.000000 s/);checks+=4;f.controller.dispose();
 }
 {
  const f=fixture();f.toggle.checked=true;f.toggle.emit('change');f.workspace.active=doc('case-b');f.provide('case-b');f.document.emit('twin:casechange');await f.resolveAll();assert.equal(f.canvas.hidden,true);assert.equal(f.toggle.checked,false);assert.equal(f.caption.hidden,true);f.toggle.checked=true;f.toggle.emit('change');await f.resolveAll();assert.equal(f.canvas.hidden,false);checks+=4;f.controller.dispose();
 }
 {
  const f=fixture();f.toggle.checked=true;f.toggle.emit('change');f.video.paused=false;f.video.emit('play');await f.resolveAll();assert.equal(f.canvas.hidden,true);assert.match(f.caption.textContent,/hidden during playback/);checks+=2;f.controller.dispose();
 }
 for(const mismatch of ['source','mask','binary']){
  const f=fixture(),path='/files/data/case-a/'+(mismatch==='source'?'frame1.jpg':'mask1.png');
  f.images.set(path,mismatch==='binary'?{naturalWidth:2,naturalHeight:2,pixels:pixels([0,100,0,255])}:{naturalWidth:3,naturalHeight:2,pixels:pixels([0,255,0,255,0,0])});
  f.toggle.checked=true;f.toggle.emit('change');await f.resolveAll();assert.equal(f.canvas.hidden,true);assert.match(f.caption.textContent,/withheld/);checks+=2;f.controller.dispose();
 }
 {
  const f=fixture();f.workspace.active={id:'empty',interaction:{evidence:[{timestamp_s:0,observations:[]}]}};f.document.emit('twin:casechange');assert.equal(f.toggle.disabled,true);assert.equal(f.canvas.hidden,true);checks+=2;f.controller.dispose();
 }
 console.log(JSON.stringify({status:'passed',checks,actual_shipped_controller:true,browser_pixel_validation:false,network_or_model_calls:false}));
})().catch(error=>{console.error(error);process.exitCode=1;});
