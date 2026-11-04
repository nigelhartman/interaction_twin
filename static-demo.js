/* Static saved-baseline transport. No server, inference or model calls. */
(() => { 'use strict';
const base="/interaction_twin/"; let frozen;
async function read(path){const r=await fetch(base+path);if(!r.ok)throw Error('Saved demo file unavailable');return r.json();}
const index=()=>frozen||(frozen=read('showcase/demos/robotics/index.json'));
async function taskFor(params){const i=await index(),t=i.tasks.find(t=>t.twin_id===params.get('twin_id')&&t.instruction===params.get('instruction'));if(!t||params.get('kit_id')!==i.kit_id)throw Error('No saved baseline for this task');return t;}
window.StaticTwin={
 async robotJSON(url){const u=new URL(url,location.origin);if(u.pathname==='/api/library')return read('showcase/public/library-catalog.json');
 if(u.pathname==='/api/robotics/kit'){const i=await index();if(u.searchParams.get('kit_id')!==i.kit_id)throw Error('Unknown kit');return{kit:await read(i.kit.path),snapshot_mode:true};}
 if(u.pathname==='/api/robotics/plan'){const t=await taskFor(u.searchParams);return read(t.task.path);}throw Error('This static demo has no API');},
 bundleURL(params){const id=params.get('twin_id');return base+'showcase/demos/robotics/'+(id.includes('wallet')?'wallet':'jabra')+'/starter.zip';}
};
document.addEventListener('DOMContentLoaded',()=>{if(window.self!==window.top)return;const b=document.createElement('div');b.className='static-demo-note';b.textContent='Read-only demo · saved results';document.body.prepend(b);});
})();
