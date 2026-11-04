/* Player navigation only: seek real media, preserve whether the viewer paused it. */
(function(root){'use strict';
function attach(document){
 const video=document.getElementById('project-film'),buttons=Array.from(document.querySelectorAll('[data-film-time]'));
 if(!video||!buttons.length)return null;
 const starts=buttons.map(b=>Number(b.dataset.filmTime));
 if(starts.some((n,i)=>!Number.isFinite(n)||n<0||n>=45||(i&&n<=starts[i-1])))throw Error('Invalid film chapter clock.');
 let pending=null;
 function update(){const time=Number(video.currentTime)||0;let active=0;for(let i=0;i<starts.length;i++)if(starts[i]<=time)active=i;buttons.forEach((b,i)=>{if(i===active)b.setAttribute('aria-current','true');else b.removeAttribute('aria-current');});}
 function seek(time){if(video.readyState<1||!Number.isFinite(video.duration)){pending=time;return;}pending=null;video.currentTime=Math.min(time,Math.max(0,video.duration-.001));update();}
 const listeners=buttons.map((button,i)=>{const click=()=>seek(starts[i]);button.addEventListener('click',click);return[button,click];});
 function metadata(){if(pending!==null)seek(pending);else update();}
 video.addEventListener('loadedmetadata',metadata);video.addEventListener('timeupdate',update);video.addEventListener('seeked',update);update();
 return()=>{for(const[b,fn]of listeners)b.removeEventListener('click',fn);video.removeEventListener('loadedmetadata',metadata);video.removeEventListener('timeupdate',update);video.removeEventListener('seeked',update);};
}
if(typeof module!=='undefined'&&module.exports)module.exports={attach};
if(typeof document!=='undefined'){const dispose=attach(document);if(dispose)root.addEventListener('pagehide',dispose,{once:true});}
})(typeof globalThis!=='undefined'?globalThis:this);
