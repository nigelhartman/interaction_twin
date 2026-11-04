/* A once-through illustration. No requests, inference or recorded run status. */
(function(){'use strict';
 const stage=document.querySelector('.workflow'),button=document.getElementById('workflow-control'),caption=document.getElementById('workflow-caption');if(!stage||!button||!caption)return;
 const nodes=Array.from(stage.querySelectorAll('[data-node]')),motion=window.matchMedia('(prefers-reduced-motion: reduce)'),phases=[
  {node:0,label:'Illustration · Muse proposes a source-linked draft.'},
  {node:1,label:'Illustration · Critic prompts and code checks review the draft.'},
  {node:0,label:'Illustration · A targeted repair can address a review finding.'},
  {node:1,label:'Illustration · The revised draft is reviewed again.'},
  {node:0,label:'Illustration · One more targeted repair is allowed, if needed.'},
  {node:1,label:'Illustration · Review compares the available evidence.'},
  {node:2,label:'Illustration · The owner decides whether to approve this exact revision.'}
 ];
 const stepMs=1700,duration=stepMs*phases.length;let elapsed=0,last=null,raf=null,playing=false,ended=false;
 function paint(){const index=Math.min(phases.length-1,Math.floor(elapsed/stepMs)),phase=phases[index];stage.dataset.phase=String(index);stage.dataset.playing=String(playing);nodes.forEach((node,i)=>{node.dataset.active=String(i===phase.node);});caption.textContent=phase.label;button.textContent=ended?'Replay illustration ↻':playing?'Pause illustration Ⅱ':elapsed?'Resume illustration ▶':'Play illustration ▶';button.setAttribute('aria-label',ended?'Replay illustration':playing?'Pause illustration':elapsed?'Resume illustration':'Play illustration');}
 function tick(now){if(!playing)return;if(last!==null)elapsed=Math.min(duration,elapsed+Math.max(0,now-last));last=now;if(elapsed>=duration){playing=false;ended=true;raf=null;paint();return;}paint();raf=requestAnimationFrame(tick);}
 function pause(){playing=false;last=null;if(raf!==null)cancelAnimationFrame(raf);raf=null;paint();}
 function play(){if(ended){elapsed=0;ended=false;}playing=true;last=null;paint();raf=requestAnimationFrame(tick);}
 button.hidden=false;button.addEventListener('click',()=>playing?pause():play());
 document.addEventListener('visibilitychange',()=>{if(document.hidden&&playing)pause();});
 const reduce=()=>{if(motion.matches&&playing)pause();};if(motion.addEventListener)motion.addEventListener('change',reduce);
 if(motion.matches){nodes.forEach(node=>{node.dataset.active='false';});button.textContent='Play illustration ▶';}else play();
 window.addEventListener('pagehide',()=>{if(raf!==null)cancelAnimationFrame(raf);});
})();
