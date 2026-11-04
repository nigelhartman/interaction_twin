/* Immutable, hash-bound scene loading. No inference, geometry edits or remote URLs. */
(function(){'use strict';
const base=window.TWIN_CASES?.curated?new URL('/showcase/public/',location.origin):new URL('.',document.currentScript.src),cache=new Map(),MAX_BYTES=32*1024*1024;
async function load(c,{signal}={}){
 if(c.scene||!c.scene_ref)return c;
 const ref=c.scene_ref;
 if(!/^scenes\/[a-f0-9]{64}\.json$/.test(ref.path)||ref.path!==`scenes/${ref.sha256}.json`||!Number.isSafeInteger(ref.bytes)||ref.bytes<1||ref.bytes>MAX_BYTES)throw Error('Invalid saved scene identity.');
 const url=new URL(ref.path,base);
 if(url.origin!==location.origin||!url.pathname.startsWith(base.pathname+'scenes/'))throw Error('Scene must stay in the local workspace.');
 let pack=cache.get(ref.sha256);
 if(!pack){
  const response=await fetch(url.href,{cache:'force-cache',signal});
  if(!response.ok)throw Error('Saved scene unavailable; no replacement is shown.');
  const raw=await response.arrayBuffer();
  if(raw.byteLength!==ref.bytes||raw.byteLength>MAX_BYTES)throw Error('Saved scene length changed.');
  const digest=[...new Uint8Array(await crypto.subtle.digest('SHA-256',raw))].map(v=>v.toString(16).padStart(2,'0')).join('');
  if(digest!==ref.sha256)throw Error('Saved scene bytes changed.');
  pack=JSON.parse(new TextDecoder().decode(raw));
 }
 if(signal?.aborted)throw new DOMException('Source selection changed.','AbortError');
 if(pack.format!=='workspace-case-scene'||pack.version!==1||pack.case_id!==c.id||pack.interaction_sha256!==c.original_sha256||!pack.scene||!Array.isArray(pack.scene.objects))throw Error('Saved scene does not belong to this exact project.');
 cache.delete(ref.sha256);cache.set(ref.sha256,pack);
 while(cache.size>2)cache.delete(cache.keys().next().value);
 return {...c,scene:pack.scene};
}
window.TwinCaseData={load};
})();
