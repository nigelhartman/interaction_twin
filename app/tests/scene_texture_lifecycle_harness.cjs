/* Actual Scene.batch/dispose lifecycle with synthetic GL/Image doubles only. */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const scenario=process.argv[2],images=[];let code=fs.readFileSync(path.join(__dirname,'../workspace/scene.js'),'utf8');
if(process.argv[3]==='drop_disposed_error_guard')code=code.replace("im.onerror=()=>{if(!this.disposed)this.onError('A generated texture could not be decoded.')};","im.onerror=()=>this.onError('A generated texture could not be decoded.');");
class ImageDouble{constructor(){images.push(this)}set src(value){this.source=value}}
const sandbox={console,Float32Array,Uint8Array,Image:ImageDouble};sandbox.globalThis=sandbox;vm.runInNewContext(code,sandbox);const Scene=sandbox.TwinScene.Scene;
const banner={source:'B',text:'B is ready'},callbacks=[];
function instance(label){
 let serial=0,uploads=0,draws=0;const deleted=[];
 const gl={createTexture(){return{kind:'texture',serial:serial++}},createBuffer(){return{kind:'buffer',serial:serial++}},bindTexture(){},texParameteri(){},pixelStorei(){},texImage2D(){uploads++},bindBuffer(){},bufferData(){},deleteTexture(x){deleted.push(x)},deleteBuffer(x){deleted.push(x)}};
 for(const name of ['TEXTURE_2D','TEXTURE_WRAP_S','CLAMP_TO_EDGE','TEXTURE_WRAP_T','TEXTURE_MIN_FILTER','LINEAR','TEXTURE_MAG_FILTER','RGBA','UNSIGNED_BYTE','ARRAY_BUFFER','STATIC_DRAW','UNPACK_FLIP_Y_WEBGL'])gl[name]=name;
 const scene=Object.assign(Object.create(Scene.prototype),{gl,disposed:false,resources:[],listeners:[],onError(message){callbacks.push(label);banner.source=label;banner.text=message},draw(){draws++}});
 const primitive={vertices:[[0,0,0],[1,0,0],[0,1,0]],faces:[[0,1,2]],colors:[[1,0,0,1],[0,1,0,1],[0,0,1,1]],uv:[[0,0],[1,0],[0,1]],texture:'data:image/png;base64,synthetic_fixture_not_decoded'};
 const batch=scene.batch(primitive),image=images.at(-1);return{scene,batch,image,deleted,get uploads(){return uploads},get draws(){return draws}};
}
const a=instance('A'),b=instance('B');a.scene.dispose();assert.equal(a.scene.disposed,true);assert.equal(a.deleted.length,5,'Actual dispose deletes texture and four buffers');
if(scenario==='late_error'){
 a.image.onerror();assert.equal(banner.source,'B','DisposedA error must not mark currentB as failed');assert.equal(banner.text,'B is ready');assert.deepEqual(callbacks,[]);
 b.image.onerror();assert.equal(banner.source,'B');assert.match(banner.text,/texture could not be decoded/);assert.deepEqual(callbacks,['B'],'LiveB error must still be visible');
}else if(scenario==='late_load'){
 const priorA=a.uploads,priorB=b.uploads;a.image.onload();assert.equal(a.uploads,priorA);assert.equal(a.draws,0);assert.equal(a.batch.ready,false);
 b.image.onload();assert.equal(b.uploads,priorB+1);assert.equal(b.draws,1);assert.equal(b.batch.ready,true);assert.deepEqual(callbacks,[]);
}else throw Error('Unknown lifecycle scenario');
console.log(JSON.stringify({status:'passed',scenario,actual_batch_and_dispose:true,synthetic_GL_and_Image:true,browser_pixels:false,models:0}));
