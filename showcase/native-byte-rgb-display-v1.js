/* Display the pipeline's historical byte RGB without a second transfer curve.
 * Only verified, opaque, untextured byte-colour SAM exports qualify. Native
 * buffers, geometry and downloaded GLBs are never modified by this adapter.
 */
(function(root){'use strict';
const POLICY='historical_uint8_vertex_rgb_interpolation_unchanged';
function inspect(raw){
 const no=reason=>({applicable:false,reason,policy:POLICY});
 try{
  const bytes=raw instanceof ArrayBuffer?new Uint8Array(raw):new Uint8Array(raw.buffer,raw.byteOffset,raw.byteLength),view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  if(bytes.length<28||view.getUint32(0,true)!==0x46546c67||view.getUint32(4,true)!==2||view.getUint32(8,true)!==bytes.length)return no('not_exact_glb2');
  const length=view.getUint32(12,true);if(length>2*1024*1024||20+length+8>bytes.length||view.getUint32(16,true)!==0x4e4f534a)return no('unsupported_json_chunk');
  const doc=JSON.parse(new TextDecoder().decode(bytes.subarray(20,20+length))),binStart=28+length,binLength=view.getUint32(20+length,true);
  if(view.getUint32(24+length,true)!==0x004e4942||binStart+binLength!==bytes.length||doc.extensionsUsed?.length||doc.extensionsRequired?.length||doc.buffers?.length!==1||doc.buffers[0].uri)return no('unsupported_native_container');
  let primitives=0,vertices=0;
  for(const mesh of doc.meshes||[])for(const p of mesh.primitives||[]){
   const c=doc.accessors?.[p.attributes?.COLOR_0],b=doc.bufferViews?.[c?.bufferView],channels=c?.type==='VEC4'?4:c?.type==='VEC3'?3:0,m=p.material===undefined?{}:doc.materials?.[p.material];
   if(!c||!b||!channels||c.componentType!==5121||c.normalized!==true||c.sparse||b.buffer!==0||!Number.isSafeInteger(c.count)||c.count<1||c.count>1000000||(p.mode!==undefined&&p.mode!==4)||!m)return no('not_legacy_byte_rgb');
   const factor=m.pbrMetallicRoughness?.baseColorFactor||[1,1,1,1];
   if((m.alphaMode||'OPAQUE')!=='OPAQUE'||m.extensions||m.pbrMetallicRoughness?.baseColorTexture||factor.length!==4||factor.some(v=>v!==1))return no('nontrivial_native_material');
   const base=b.byteOffset||0,relative=c.byteOffset||0,offset=base+relative,stride=b.byteStride||channels,end=relative+(c.count-1)*stride+channels;
   if(![base,relative,b.byteLength,doc.buffers[0].byteLength,stride].every(Number.isSafeInteger)||base<0||relative<0||b.byteLength<1||stride<channels||stride>252||end>b.byteLength||base+b.byteLength>doc.buffers[0].byteLength||doc.buffers[0].byteLength>binLength)return no('invalid_colour_buffer');
   if(channels===4)for(let i=0;i<c.count;i++)if(bytes[binStart+offset+i*stride+3]!==255)return no('nonopaque_vertex_alpha');
   primitives++;vertices+=c.count;
  }
  return primitives?{applicable:true,policy:POLICY,primitive_count:primitives,vertex_count:vertices,asset_bytes_changed:false}:no('no_meshes');
 }catch{return no('invalid_native_colour_metadata');}
}
function apply(model,raw,evidence){
 const policy=inspect(raw),allowed=new Set(['verified_sam3d_review_asset','verified_sam3d_scene_asset','software_colour_fixture']);
 if(!policy.applicable)return{...policy,materials:0};
 if(!evidence||!allowed.has(evidence.kind)||!/^[a-f0-9]{64}$/.test(evidence.asset_sha256))throw Error('Verified native colour provenance required.');
 let count=0;
 model.traverse(object=>{
  if(!object.isMesh)return;
  const colour=object.geometry?.attributes?.color;
  if(!colour||!(colour.array instanceof Uint8Array)||colour.normalized!==true)return;
  for(const material of Array.isArray(object.material)?object.material:[object.material]){
   if(!material?.isMeshBasicMaterial||material.map||material.transparent||material.opacity!==1||!material.vertexColors||!material.color||[material.color.r,material.color.g,material.color.b].some(v=>v!==1))continue;
   if(material.userData?.nativeByteRgbDisplayV1)continue;
   material.toneMapped=false;
   material.onBeforeCompile=shader=>{if(!shader.fragmentShader.includes('#include <colorspace_fragment>'))throw Error('Native RGB display shader is incompatible.');shader.fragmentShader=shader.fragmentShader.replace('#include <colorspace_fragment>','// Native historical RGB bytes already encode the display colour.');};
   material.customProgramCacheKey=()=>POLICY+'-v1';
   material.userData={...material.userData,nativeByteRgbDisplayV1:{...policy,evidence,display_only:true}};
   material.needsUpdate=true;count++;
  }
 });
 return{...policy,materials:count,evidence,display_only:true};
}
const api={inspect,apply,POLICY};root.NativeByteRgbDisplayV1=api;if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(globalThis);
