/* Owner-only legacy panels: one guarded send, never an automatic mutation retry. */
(function(root){'use strict';
if(typeof root.fetch!=='function'||root.__twinOwnerFetch)return;
const original=root.fetch.bind(root);let authPending=null;
function ownerToken(){
 if(!authPending)authPending=original('/api/admin/auth',{credentials:'same-origin',cache:'no-store'}).then(async response=>{
  if(!response.ok)throw Error('Owner session unavailable. Sign in at /admin/.');
  const value=await response.json();
  if(value.authenticated!==true||typeof value.csrf_token!=='string'||!/^[A-Za-z0-9_-]{43}$/.test(value.csrf_token))throw Error('Owner sign-in required. Open /admin/ before making changes.');
  return value.csrf_token;
 }).finally(()=>{authPending=null;});
 return authPending;
}
root.fetch=async function(input,init){
 const url=new URL(typeof input==='string'||input instanceof URL?String(input):input.url,root.location.href);
 const method=String(init?.method||input?.method||'GET').toUpperCase();
 if(url.origin!==root.location.origin||!url.pathname.startsWith('/api/')||method!=='POST')return original(input,init);
 const token=await ownerToken(),headers=new Headers(init?.headers||input?.headers||undefined);
 headers.set('X-CSRF-Token',token);
 return original(input,{...init,headers,credentials:'same-origin'});
};
root.__twinOwnerFetch=true;
})(window);
