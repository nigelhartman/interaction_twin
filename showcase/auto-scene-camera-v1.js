/* Presentation camera state only. Source poses, geometry and support are untouched. */
(function(root){'use strict';
const finite=n=>typeof n==='number'&&Number.isFinite(n);
function create({idle_ms=3000,return_ms=850}={}){
 if(!finite(idle_ms)||idle_ms<0||!finite(return_ms)||return_ms<=0)throw Error('Finite camera timings required.');
 let last=null;
 return Object.freeze({
  activity(now){if(!finite(now))throw Error('Finite camera clock required.');last=now;},
  reset(){last=null;},
  state(now){if(!finite(now))throw Error('Finite camera clock required.');if(last===null)return{phase:'source',progress:1};const delta=Math.max(0,now-last);if(delta<idle_ms)return{phase:'manual',progress:0};const t=Math.min(1,(delta-idle_ms)/return_ms);return{phase:t<1?'return':'source',progress:t*t*(3-2*t)};}
 });
}
const api=Object.freeze({create});root.AutoSceneCameraV1=api;if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(globalThis);
