/* Saved-source RGB only. Caller hash-checks every referenced byte artifact. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.SourceAppearanceStatesV1=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
  'use strict';
  const FORMAT='interaction-twin-source-appearance-states';
  const POLICY={representation:'exact_source_RGB_vertex_patch_with_native_fallback',hidden_vertices:'unchanged_approved_native_appearance',patch_interpolation:'none',unknown_rows_are_barriers:true,maximum_nearest_observation_offset_s:.36,state_transition_gap:'unknown_native_fallback_not_electronic_state',restore_native_before_each_patch:true,electronics_simulation:false};
  function require(v,m){if(!v)throw new Error(m);}
  function same(a,b){if(a===b)return true;if(!a||!b||typeof a!=='object'||typeof b!=='object'||Array.isArray(a)!==Array.isArray(b))return false;
    const ka=Object.keys(a).sort(),kb=Object.keys(b).sort();return ka.length===kb.length&&ka.every((k,i)=>k===kb[i]&&same(a[k],b[k]));}
  function validate(d,previous,scene,program){
    require(d&&d.format===FORMAT&&d.version===1&&same(d.display_policy,POLICY),'Unknown source appearance contract');
    require(d.physical_accuracy_verified===false&&d.simulation_ready===false&&d.electronic_state_verified===false,'Unverified appearance cannot claim physics/electronics');
    require(same(d.source,previous.source)&&same(d.source,scene.source)&&Object.keys(program.source).every(k=>same(d.source[k],program.source[k]))&&same(d.scene_ref,previous.scene_ref)&&same(d.evidence_program,previous.evidence_program),'Source appearance lineage differs');
    const c=scene.components.find(x=>x.id===d.component_id),b=previous.component_assets.find(x=>x.component_id===d.component_id);
    require(c&&b&&same(d.baseline_asset_ref,b.native_glb)&&same(d.baseline_asset_ref,c.asset),'Source appearance asset differs');
    require(Number.isInteger(d.vertex_count)&&d.vertex_count>0&&d.vertex_count<=1000000&&d.keyframes.length>0&&d.keyframes.length<=8,'Bounded source patch required');
    const keys=new Map(),evidence=new Map(program.evidence.map(e=>[e.source_index,e]));
    for(const k of d.keyframes){const p=c.poses.find(x=>x.source_sample_index===k.source_sample_index),e=evidence.get(k.source_sample_index);
      require(!keys.has(k.id)&&p&&e&&p.status==='estimated'&&p.timestamp_s===k.timestamp_s&&e.timestamp_s===k.timestamp_s,'Invalid source appearance keyframe clock');
      require(k.source_rgb_ref.path===e.frame_path&&k.source_rgb_ref.sha256===e.frame_sha256&&same(k.source_mask_ref,p.source_mask)&&same(k.source_camera,scene.camera)&&same(k.camera_from_native_mesh,p.camera_from_native_mesh),'Source appearance projection differs');
      require(k.painted_vertex_count>0&&k.native_fallback_vertex_count===d.vertex_count-k.painted_vertex_count,'Source appearance coverage differs');keys.set(k.id,k);
    }
    require(d.observations.length===c.poses.length&&d.observations.length<=240,'All source observation barriers required');
    d.observations.forEach((r,i)=>{const p=c.poses[i],e=evidence.get(p.source_sample_index);require(r.source_sample_index===p.source_sample_index&&r.timestamp_s===p.timestamp_s&&e.timestamp_s===r.timestamp_s,'Source appearance observation clock differs');
      if(r.source_rgb_ref||r.source_mask_ref)require(r.source_rgb_ref&&r.source_rgb_ref.path===e.frame_path&&r.source_rgb_ref.sha256===e.frame_sha256&&same(r.source_mask_ref,p.source_mask),'Optional unknown-row RGB/SAM differs');
      if(r.status==='unknown'){require(r.keyframe_id===null,'Unknown observation cannot have a patch');return;}
      const k=keys.get(r.keyframe_id);require(r.status==='source_observed_appearance_estimate'&&p.status==='estimated'&&k&&r.appearance_label===k.appearance_label,'Unsupported appearance observation');
      require(r.source_rgb_ref.path===e.frame_path&&r.source_rgb_ref.sha256===e.frame_sha256&&same(r.source_mask_ref,p.source_mask),'Appearance observation RGB/SAM differs');
    });return d;
  }
  function decodePatch(buffer,expectedVertexCount){const bytes=buffer instanceof Uint8Array?buffer:new Uint8Array(buffer),v=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
    require(bytes.length>=16&&bytes.length<=8*1024*1024&&String.fromCharCode(...bytes.subarray(0,4))==='ITCP'&&v.getUint32(4,true)===1,'Invalid compact source RGB patch');
    const total=v.getUint32(8,true),count=v.getUint32(12,true);require(total>0&&total<=1000000&&count>0&&count<=total&&bytes.length===16+count*11&&(expectedVertexCount===undefined||total===expectedVertexCount),'Compact patch native vertex count differs');
    const ids=new Uint32Array(count),rgb=new Uint8Array(count*3),pixels=new Uint16Array(count*2);let last=-1;
    for(let i=0;i<count;i++){const o=16+i*11,id=v.getUint32(o,true);require(id>last&&id<total,'Sorted unique patch vertex IDs required');last=id;ids[i]=id;pixels[i*2]=v.getUint16(o+4,true);pixels[i*2+1]=v.getUint16(o+6,true);rgb.set(bytes.subarray(o+8,o+11),i*3);}
    return {vertexCount:total,ids,rgb,pixels};
  }
  function at(d,t){require(Number.isFinite(t),'Finite source clock required');const rows=d.observations;let nearest=rows[0],before=null,after=null;
    for(const r of rows){if(Math.abs(r.timestamp_s-t)<Math.abs(nearest.timestamp_s-t))nearest=r;if(r.timestamp_s<=t)before=r;if(after===null&&r.timestamp_s>=t)after=r;}
    if(!before||!after||nearest.status==='unknown'||Math.abs(nearest.timestamp_s-t)>.36||before.status==='unknown'||after.status==='unknown'||before.appearance_label!==after.appearance_label)return null;
    return d.keyframes.find(k=>k.id===nearest.keyframe_id)||null;
  }
  function applyPatch(attribute,baseline,patch){require(attribute&&attribute.itemSize>=3&&attribute.count===patch.vertexCount&&baseline.length===attribute.array.length,'Exact native color attribute required');
    const a=attribute.array;require(a instanceof Uint8Array||a instanceof Float32Array,'Known normalized-byte or float RGB attribute required');a.set(baseline);
    for(let i=0;i<patch.ids.length;i++){const o=patch.ids[i]*attribute.itemSize;for(let j=0;j<3;j++)a[o+j]=a instanceof Uint8Array?patch.rgb[i*3+j]:patch.rgb[i*3+j]/255;}attribute.needsUpdate=true;
  }
  return {FORMAT,POLICY,validate,decodePatch,at,applyPatch};
});
