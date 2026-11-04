/* Lossless, bounded display transport. No fitting, rounding or model editing. */
(function(root){'use strict';
const MAX=64*1024*1024,VERTICES=1000000,INDICES=3000000,SHA=/^[a-f0-9]{64}$/;
const FIELDS=[['vertices',3,'float64_le'],['colors',4,'float64_le'],['uv',2,'float64_le'],['faces',3,'uint32_le']];
const PRIMITIVE=['node','geometry','material_index','vertices','faces','colors','uv','texture','base_color_factor','appearance_mode'];
function assert(v,m){if(!v)throw Error(m)}
function keys(value,expected,message){assert(value&&typeof value==='object'&&!Array.isArray(value)&&Object.keys(value).sort().join('|')===[...expected].sort().join('|'),message)}
function finiteList(a,n){return Array.isArray(a)&&a.length===n&&a.every(Number.isFinite)}
function layout(a){keys(a,['target','span','initial_distance'],'Invalid native display layout');assert(finiteList(a.target,3)&&Number.isFinite(a.span)&&a.span>0&&Number.isFinite(a.initial_distance)&&a.initial_distance>0,'Invalid native display bounds')}
function same(a,b){return JSON.stringify(a)===JSON.stringify(b)}
async function hash(raw){return Array.from(new Uint8Array(await root.crypto.subtle.digest('SHA-256',raw)),v=>v.toString(16).padStart(2,'0')).join('')}
async function decodeScene(descriptorRaw,geometryRaw,expected){
 assert(descriptorRaw instanceof ArrayBuffer&&geometryRaw instanceof ArrayBuffer,'Binary scene requires exact array buffers');
 assert(descriptorRaw.byteLength>0&&geometryRaw.byteLength>0&&descriptorRaw.byteLength+geometryRaw.byteLength<=MAX,'Combined native scene exceeds unchanged byte budget');
 const d=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(descriptorRaw));
 keys(d,['format','version','source_glb_sha256','geometry_sha256','geometry_bytes','scene'],'Unexpected binary descriptor fields');
 assert(d.format==='sam3d-native-binary-scene'&&d.version===1&&SHA.test(d.source_glb_sha256)&&d.source_glb_sha256===expected.sourceHash,'Native geometry source identity differs');
 assert(SHA.test(d.geometry_sha256)&&Number.isSafeInteger(d.geometry_bytes)&&d.geometry_bytes===geometryRaw.byteLength&&await hash(geometryRaw)===d.geometry_sha256,'Native geometry bytes/hash differ');
 const s=d.scene;
 keys(s,['objects','scene','native_only','timeline','source_projection_rows','image_size','physical_assembly_verified'],'Unexpected native scene fields');
 assert(s.native_only===true&&s.physical_assembly_verified===false&&s.source_projection_rows===null&&Array.isArray(s.timeline)&&s.timeline.length===0&&same(s.image_size,expected.imageSize)&&Array.isArray(s.objects)&&s.objects.length===1,'Native inspection cannot contain source poses or assembly');
 const o=s.objects[0];
 keys(o,['object_id','primitives','native_scene','mesh_stats','native_up','frames','max_hold_s','source_projection_rows'],'Unexpected native component fields');
 assert(o.object_id===expected.componentId&&o.native_up==='Y'&&Array.isArray(o.frames)&&o.frames.length===0&&o.max_hold_s===0&&o.source_projection_rows===null,'Native component identity or placement differs');
 assert(Array.isArray(o.primitives)&&o.primitives.length>0&&o.primitives.length<=256&&same(o.mesh_stats,expected.meshStats),'Native primitive statistics differ');
 layout(s.scene);layout(o.native_scene);assert(same(s.scene,o.native_scene),'Native scene display frames differ');
 const view=new DataView(geometryRaw);let offset=0,totalVertices=0,totalIndices=0;
 const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
 for(const p of o.primitives){
  keys(p,PRIMITIVE,'Unexpected native material/geometry fields');
  assert(typeof p.node==='string'&&typeof p.geometry==='string'&&(p.material_index===null||(Number.isSafeInteger(p.material_index)&&p.material_index>=0)),'Invalid native material identity');
  assert(finiteList(p.base_color_factor,4)&&p.base_color_factor.every(x=>x>=0&&x<=1)&&['uv_texture','vertex_rgb','material_color'].includes(p.appearance_mode),'Invalid native material values');
  assert(p.texture===null||(typeof p.texture==='string'&&/^data:image\/(png|jpeg);base64,[A-Za-z0-9+/]+=*$/.test(p.texture)),'Native texture must be embedded');
  let count=0;
  for(const[field,columns,type]of FIELDS){
   const ref=p[field];if(ref===null&&(field==='colors'||field==='uv'))continue;
   keys(ref,['binary'],'Invalid binary field reference');const b=ref.binary;
   keys(b,['offset','bytes','rows','columns','component_type'],'Invalid binary field shape');
   assert(Number.isSafeInteger(b.offset)&&b.offset===offset&&Number.isSafeInteger(b.rows)&&b.rows>0&&b.rows<=(field==='faces'?INDICES/3:VERTICES)&&b.columns===columns&&b.component_type===type,'Invalid binary range, shape or component type');
   const scalarBytes=type==='float64_le'?8:4,bytes=b.rows*columns*scalarBytes;
   assert(Number.isSafeInteger(bytes)&&b.bytes===bytes&&offset+bytes<=geometryRaw.byteLength,'Binary geometry exceeds payload range');
   if(field==='vertices'){count=b.rows;totalVertices+=count;assert(totalVertices<=VERTICES,'Aggregate native vertex budget exceeded')}
   else if(field==='faces'){totalIndices+=b.rows*3;assert(totalIndices<=INDICES,'Aggregate native index budget exceeded')}
   else assert(b.rows===count,'Native attribute count differs');
   const rows=new Array(b.rows);
   for(let i=0;i<b.rows;i++){
    const row=new Array(columns);
    for(let j=0;j<columns;j++){
     const at=offset+(i*columns+j)*scalarBytes,x=type==='float64_le'?view.getFloat64(at,true):view.getUint32(at,true);
     assert(Number.isFinite(x)&&(field!=='faces'||x<count),'Nonfinite native value or out-of-range triangle index');
     row[j]=x;if(field==='vertices'){lo[j]=Math.min(lo[j],x);hi[j]=Math.max(hi[j],x)}
    }
    rows[i]=row;
   }
   p[field]=rows;offset+=bytes;
  }
  assert((p.appearance_mode==='uv_texture')===Boolean(p.texture)&&(!p.texture||p.uv!==null),'Native appearance fields differ');
  assert(p.appearance_mode!=='vertex_rgb'||p.colors!==null,'Native colours missing');
 }
 assert(offset===geometryRaw.byteLength,'Unexpected trailing geometry payload');
 const stats=o.mesh_stats;
 assert(stats.rendered_vertices===totalVertices&&stats.faces*3===totalIndices&&stats.primitive_count===o.primitives.length&&same(stats.bounds,[lo,hi]),'Native counts or exact bounds differ');
 return d;
}
const api={decodeScene,hash,MAX_SCENE_BYTES:MAX};root.InitialNativeGeometry=api;
if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(globalThis);
