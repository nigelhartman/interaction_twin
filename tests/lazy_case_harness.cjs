/* Execute the shipped loader against changed bytes, foreign cases and eviction. */
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {webcrypto,createHash}=require('node:crypto');
const code=fs.readFileSync(require('node:path').join(__dirname,'../workspace/case-data.js'),'utf8');
const objects=new Map();let requests=0;
const hash=b=>createHash('sha256').update(b).digest('hex');
function record(id,value=1){const pack={format:'workspace-case-scene',version:1,case_id:id,interaction_sha256:'a'.repeat(64),scene:{objects:[{object_id:'part',vertices:[[value,2,3]],frames:[{timestamp_s:1.23,visible:false}]}]}};const raw=Buffer.from(JSON.stringify(pack)),digest=hash(raw),ref={path:'scenes/'+digest+'.json',sha256:digest,bytes:raw.length};objects.set(ref.path,raw);return{id,original_sha256:pack.interaction_sha256,scene:null,scene_ref:ref};}
const sandbox={URL,Map,Number,Uint8Array,TextDecoder,DOMException,crypto:webcrypto,
 document:{currentScript:{src:'http://127.0.0.1:8765/workspace/case-data.js'}},location:{origin:'http://127.0.0.1:8765'},
 fetch:async url=>{requests++;const path=new URL(url).pathname.replace('/workspace/',''),raw=objects.get(path);return{ok:!!raw,arrayBuffer:async()=>raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength)}}};
sandbox.window=sandbox;vm.runInNewContext(code,sandbox);const load=sandbox.TwinCaseData.load;
(async()=>{
 const a=record('case-a'),unchanged=JSON.stringify(a),result=await load(a);
 assert.equal(JSON.stringify(a),unchanged);assert.equal(result.scene.objects[0].vertices[0][0],1);
 assert.equal(JSON.stringify(result.scene),JSON.stringify(JSON.parse(objects.get(a.scene_ref.path)).scene));
 await load(a);assert.equal(requests,1,'same exact scene is cached');
 await assert.rejects(load({...a,id:'foreign-project'}),/does not belong/);
 await assert.rejects(load({...a,original_sha256:'b'.repeat(64)}),/does not belong/);
 const bad=record('changed-bytes',2),changed=Buffer.from(objects.get(bad.scene_ref.path));changed[changed.length-8]^=1;objects.set(bad.scene_ref.path,changed);
 await assert.rejects(load(bad),/bytes changed/);
 for(const path of ['../cases.json','https://example.com/scene.json','scenes/../../private.json'])await assert.rejects(load({...a,scene_ref:{...a.scene_ref,path}}),/identity/);
 const controller=new AbortController();controller.abort();await assert.rejects(load(a,{signal:controller.signal}),e=>e.name==='AbortError');
 const b=record('case-b',3),c=record('case-c',4);await load(b);await load(c);const before=requests;await load(a);assert.equal(requests,before+1,'cache is bounded to two scenes');
 assert.equal((await load({id:'empty',scene:null})).scene,null);
 const embedded={id:'offline',scene:{objects:[]}};assert.equal(await load(embedded),embedded);
 console.log(JSON.stringify({status:'passed',checks:14,requests,network:'in-memory fixture only',model_inference:false}));
})().catch(error=>{console.error(error);process.exitCode=1});
