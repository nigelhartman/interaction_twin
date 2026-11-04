/* Static saved-baseline transport. No server, inference or model calls. */
(() => { 'use strict';
const base="/interaction_twin/"; let frozen,snapshot;
async function read(path){const r=await fetch(base+path);if(!r.ok)throw Error('Saved demo file unavailable');return r.json();}
const index=()=>frozen||(frozen=read('showcase/demos/robotics/index.json').then(i=>{snapshot=i;return i;}));
function selectTask(i,params){const matches=i.tasks.filter(t=>t.twin_id===params.get('twin_id')&&t.instruction===params.get('instruction'));if(matches.length!==1||params.get('kit_id')!==i.kit_id)throw Error('No saved baseline for this task');return matches[0];}
async function taskFor(params){return selectTask(await index(),params);}
window.StaticTwin={
 async robotCapabilities(){const i=await index();return{twin_ids:[...new Set(i.tasks.map(t=>t.twin_id))],scope:'saved_baselines'};},
 async robotJSON(url){const u=new URL(url,location.origin);if(u.pathname==='/api/library')return read('showcase/public/library-catalog.json');
 if(u.pathname==='/api/robotics/kit'){const i=await index();if(u.searchParams.get('kit_id')!==i.kit_id)throw Error('Unknown kit');return{kit:await read(i.kit.path),snapshot_mode:true};}
 if(u.pathname==='/api/robotics/plan'){const t=await taskFor(u.searchParams);return read(t.task.path);}throw Error('This static demo has no API');},
 bundleURL(params){if(!snapshot)throw Error('Load a saved baseline first');const t=selectTask(snapshot,params);if(!/^showcase\/demos\/robotics\/[a-z0-9_-]+\/starter\.zip$/.test(t.starter.path))throw Error('Invalid saved starter path');return base+t.starter.path;}
};
document.addEventListener('DOMContentLoaded',()=>{if(window.self!==window.top)return;const b=document.createElement('div');b.className='static-demo-note';b.textContent='Read-only demo · saved results';document.body.prepend(b);});
})();
