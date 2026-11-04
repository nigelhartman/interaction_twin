/* SAM 3D candidates come from the generated, hash-verified library catalog. */
(function(root){'use strict';
const SHA=/^[a-f0-9]{64}$/,ID=/^[A-Za-z0-9][A-Za-z0-9_.-]{0,180}$/;
function localRef(r){return r&&typeof r.path==='string'&&r.path.startsWith('data/')&&/^[A-Za-z0-9_./-]+$/.test(r.path)&&!r.path.split('/').some(p=>!p||p.startsWith('.'))&&SHA.test(r.sha256)&&Number.isSafeInteger(r.bytes)&&r.bytes>0&&r.bytes<64*1024*1024;}
function forRecording(record){
 if(!record?.source||!Array.isArray(record.components))return[];
 const ids=new Set(record.components.map(c=>c.id)),seen=new Set();
 return(record.sam3d_reviews||[]).filter(row=>{const key=row.component_id;if(!ID.test(key)||seen.has(key)||row.source_sha256!==record.source.sha256||!ids.has(key)||row.selected!==false||row.physical_accuracy_verified!==false||!localRef(row.review_page)||!localRef(row.source_image)||!localRef(row.source_mask)||!localRef(row.native_glb))return false;seen.add(key);return true;});
}
function forComponent(record,id){return forRecording(record).find(r=>r.component_id===id)||null;}
const api=Object.freeze({forRecording,forComponent,localRef});root.Sam3DModelReviews=api;if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(globalThis);
