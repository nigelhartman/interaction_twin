/* Analytic transport contracts only; no browser, model, GPU or catalog fixture. */
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs');
globalThis.crypto??=require('node:crypto').webcrypto;
const C=require('../initial_native_review_v2/geometry.js');
const [descriptorPath,binaryPath,expectedPath]=process.argv.slice(2);
const original=JSON.parse(fs.readFileSync(descriptorPath,'utf8'));
const expected=JSON.parse(fs.readFileSync(expectedPath,'utf8'));
const raw=fs.readFileSync(binaryPath);const geometry=raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength);
const binding={sourceHash:original.source_glb_sha256,componentId:'software_only',imageSize:[64,48],meshStats:expected.scene.objects[0].mesh_stats};
const buffer=d=>new TextEncoder().encode(JSON.stringify(d)).buffer;
const copy=()=>JSON.parse(JSON.stringify(original));
const decode=(d=copy(),b=geometry.slice(0),bind=binding)=>C.decodeScene(buffer(d),b,bind);
async function alteredBinary(change){const d=copy(),b=geometry.slice(0);change(new DataView(b));d.geometry_sha256=await C.hash(b);return[d,b]}
test('Float64 and Uint32 transport round-trips exact JS numbers, signed zero, colours and UVs',async()=>{
 const d=await decode(),a=d.scene.objects[0].primitives[0],b=expected.scene.objects[0].primitives[0];
 for(const field of ['vertices','colors','uv','faces'])for(let i=0;i<a[field].length;i++)for(let j=0;j<a[field][i].length;j++)assert.ok(Object.is(a[field][i][j],b[field][i][j]),`${field}[${i}][${j}] exact`);
 assert.ok(Object.is(a.vertices[0][0],-0));assert.deepEqual(a.faces,b.faces);assert.equal(d.source_glb_sha256,binding.sourceHash);
});
test('binary hash rejects even a single changed bit before decoding',async()=>{const b=geometry.slice(0);new Uint8Array(b)[7]^=1;await assert.rejects(decode(copy(),b),/bytes\/hash differ/)});
test('source model identity cannot be changed independently',async()=>await assert.rejects(decode(copy(),geometry.slice(0),{...binding,sourceHash:'b'.repeat(64)}),/source identity differs/));
test('valid hash does not permit a NaN coordinate',async()=>{const[d,b]=await alteredBinary(v=>v.setFloat64(0,NaN,true));await assert.rejects(decode(d,b),/Nonfinite/)});
test('valid hash does not permit an infinite colour',async()=>{const ref=original.scene.objects[0].primitives[0].colors.binary;const[d,b]=await alteredBinary(v=>v.setFloat64(ref.offset,Infinity,true));await assert.rejects(decode(d,b),/Nonfinite/)});
test('valid hash does not permit an out-of-range triangle index',async()=>{const ref=original.scene.objects[0].primitives[0].faces.binary;const[d,b]=await alteredBinary(v=>v.setUint32(ref.offset,3,true));await assert.rejects(decode(d,b),/out-of-range/)});
test('overlapping or skipped offsets reject',async()=>{for(const delta of [-8,8]){const d=copy();d.scene.objects[0].primitives[0].colors.binary.offset+=delta;await assert.rejects(decode(d),/range, shape/)} });
test('wrong shapes, component types and counts reject',async()=>{
 for(const mutate of [p=>p.vertices.binary.columns=4,p=>p.vertices.binary.component_type='float32_le',p=>p.vertices.binary.bytes-=8,p=>p.colors.binary.rows=2]){
  const d=copy();mutate(d.scene.objects[0].primitives[0]);await assert.rejects(decode(d),/range|shape|count/);
 }
});
test('per-primitive and aggregate geometry caps remain enforced',async()=>{const d=copy();d.scene.objects[0].primitives[0].vertices.binary.rows=1000001;await assert.rejects(decode(d),/range, shape/)});
test('truncated payload or mismatched declared length rejects',async()=>{await assert.rejects(decode(copy(),geometry.slice(0,-1)),/bytes\/hash differ/);const d=copy();d.geometry_bytes++;await assert.rejects(decode(d),/bytes\/hash differ/)});
test('trailing payload is forbidden even with its new valid hash',async()=>{const b=new Uint8Array(geometry.byteLength+4);b.set(new Uint8Array(geometry));const d=copy();d.geometry_bytes=b.length;d.geometry_sha256=await C.hash(b.buffer);await assert.rejects(decode(d,b.buffer),/trailing/)});
test('64MiB combined scene cap applies to descriptor plus binary',async()=>{const b=new ArrayBuffer(C.MAX_SCENE_BYTES);await assert.rejects(decode(copy(),b),/Combined native scene/)});
test('exact bounds and actual counts must match pinned native statistics',async()=>{for(const key of ['faces','rendered_vertices','primitive_count']){const d=copy();d.scene.objects[0].mesh_stats[key]++;await assert.rejects(decode(d),/statistics differ/)}const[d,b]=await alteredBinary(v=>v.setFloat64(8,9000,true));await assert.rejects(decode(d,b),/exact bounds differ/)});
test('source poses, shared assembly or different component cannot enter standalone inspection',async()=>{
 for(const mutate of [d=>d.scene.timeline.push(1),d=>d.scene.physical_assembly_verified=true,d=>d.scene.objects[0].frames.push({timestamp_s:0}),d=>d.scene.objects[0].object_id='other']){const d=copy();mutate(d);await assert.rejects(decode(d),/poses|assembly|identity|placement/)}
});
test('unknown fields and remote textures reject',async()=>{const d=copy();d.scene.objects[0].primitives[0].extra='x';await assert.rejects(decode(d),/Unexpected/);const e=copy();e.scene.objects[0].primitives[0].texture='https://invalid.example/image.png';await assert.rejects(decode(e),/embedded/)});
test('nullable colour/UV arrays remain absent, never fabricated',async()=>{
 const d=copy(),p=d.scene.objects[0].primitives[0];const old=p.faces.binary;
 const faceBytes=new Uint8Array(geometry,old.offset,old.bytes),vertexBytes=new Uint8Array(geometry,0,p.vertices.binary.bytes);
 const b=new Uint8Array(vertexBytes.length+faceBytes.length);b.set(vertexBytes);b.set(faceBytes,vertexBytes.length);
 p.colors=null;p.uv=null;p.appearance_mode='material_color';p.faces.binary.offset=vertexBytes.length;
 d.geometry_bytes=b.length;d.geometry_sha256=await C.hash(b.buffer);
 const result=await decode(d,b.buffer);assert.equal(result.scene.objects[0].primitives[0].colors,null);assert.equal(result.scene.objects[0].primitives[0].uv,null);
});
