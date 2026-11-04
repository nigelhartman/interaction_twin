/* Readable projection of verified saved data. No inference or capability grants. */
(function(root){'use strict';
const fail=message=>{throw Error(message);},finite=Number.isFinite;
const sameRef=(a,b)=>a&&b&&a.path===b.path&&a.sha256===b.sha256&&a.bytes===b.bytes;
const words=value=>String(value||'').replaceAll('_',' ');
const plain=value=>typeof value==='string'?value:'';
function summarize(record,aligned=null){
 if(!record||!Array.isArray(record.components)||!Array.isArray(record.steps)||!Array.isArray(record.relationships))fail('A verified twin is required.');
 const names=new Map(record.components.map(c=>[c.id,plain(c.label)||c.id])),repeated=new Map();
 for(const name of names.values())repeated.set(name,(repeated.get(name)||0)+1);
 const label=id=>{const name=names.get(id);if(!name)return null;return repeated.get(name)>1?name+' ('+id.replace(/^.*_/,'')+')':name;};
 const result={scope:'Demonstrated procedure',scope_note:'Source-linked model proposals. Physical fit and execution remain unverified.',parts:record.components.map(c=>label(c.id)),connections:[],steps:[],step_count:record.steps.length,evidence:[]};
 let mechanics=null,surface=null;
 if(aligned){
  const m=aligned.manifest;
  if(!m||m.source?.sha256!==record.source?.sha256||!sameRef(m.original_program,record.interaction))fail('Description belongs to another source or procedure.');
  mechanics=m.mechanics;surface=aligned.observed_surfaces?.document;
  if(surface){
   if(surface.source?.sha256!==record.source.sha256||!sameRef(surface.source_program,record.interaction)||!sameRef(surface.source_scene,aligned.reference)||surface.physical_accuracy_verified!==false||surface.native_SAM_contract_compatible!==false)fail('Observed description belongs to another saved scene.');
   result.scope='Source-visible surface preview';result.scope_note='Partial source surfaces with estimated depth. Original generated 3D drafts remain separate; hidden geometry and numerical joint parameters are unknown.';
  }else if(aligned.derived_assembly){
   const d=aligned.derived_assembly.document;
   if(d.source?.sha256!==record.source.sha256||!sameRef(d.evidence_program,record.interaction)||!sameRef(d.scene_ref,aligned.reference)||d.physical_accuracy_verified!==false)fail('Refined description belongs to another saved scene.');
   result.scope='Refined estimated assembly';result.scope_note='Declared geometry refinements and source-linked connections. Angles are estimates; physical stops, contact and seating are unverified.';
  }else{result.scope='Estimated connected assembly';result.scope_note='Original generated parts with source-linked connection estimates. Physical fit and execution remain unverified.';}
 }
 const covered=new Set(),pair=(a,b)=>[a,b].sort().join('\n');
 for(const joint of mechanics?.joints||[]){
  const parent=label(joint.parent_id),child=label(joint.child_id);if(!parent||!child)continue;
  covered.add(pair(joint.parent_id,joint.child_id));
  const row={title:parent+' → '+child,kind:surface?'Numerical joint unknown':joint.type==='revolute'?'Estimated hinge':words(joint.type)+' estimate',detail:'Axis, joint position and angle remain unknown.'};
  if(!surface){
   const range=joint.limits_rad,consistent=joint.angle_quality?.angle_visual_consistency!==false;
   const axis=Array.isArray(joint.axis)&&joint.axis.length===3&&joint.axis.every(finite)?joint.axis.map(n=>(Math.abs(n)<.005?0:n).toFixed(2)).join(', '):null;
   const validRange=Array.isArray(range)&&range.length===2&&range.every(finite)&&range[0]<=range[1];
   row.kind=consistent?row.kind:'Estimated configuration';
   row.detail=validRange?(consistent?'Source configuration range ':'Configuration range ')+range.map(n=>Math.round(n*180/Math.PI)||0).join('–')+'°.':'Configuration range unknown.';
   if(axis)row.detail+=' Axis ['+axis+'] in the joint frame.';
   row.detail+=consistent?' Physical stops are unverified.':' Opening angle is unresolved.';
  }
  result.connections.push(row);
 }
 for(const edge of mechanics?.unresolved_children||[]){
  const parent=label(edge.parent_id),child=label(edge.child_id);if(!parent||!child)continue;
  covered.add(pair(edge.parent_id,edge.child_id));
  const flexible=edge.reason==='rigid_motion_withheld_due_to_explicit_source_nonrigidity';
  result.connections.push({title:parent+' → '+child,kind:flexible?'Flexible connection':'Unresolved connection',detail:flexible?'Fold axis, joint position and angle remain unknown.':'Joint position, axis and angle remain unknown.'});
 }
 const related=[];
 for(const relation of record.relationships){
  const a=relation.subject_track_id,b=relation.object_track_id;if(covered.has(pair(a,b))||!label(a)||!label(b)||relation.predicate==='possibly_same_entity')continue;
  const removable=relation.predicate==='removable_from',connector=relation.predicate==='inserted_into'||relation.predicate==='connects_to';
  related.push({title:label(a)+' → '+label(b),kind:removable?'Removable relationship proposal':connector?'Connection proposal':words(relation.predicate)+' · proposal',detail:removable||connector?'Physical contact and seating are unverified.':'A source-linked model proposal; mechanical fit remains unverified.'});
 }
 result.connections.push(...related);result.connections=result.connections.slice(0,3);
 if(!result.connections.length)result.connections.push({title:'Connections',kind:'Not established',detail:'No source-supported mechanical connection is available.'});
 const indices=[0,Math.floor((record.steps.length-1)/2),record.steps.length-1];
 result.steps=[...new Set(indices)].filter(i=>i>=0&&record.steps[i]).map(i=>({index:i+1,title:plain(record.steps[i].title),start_s:record.steps[i].start_s,end_s:record.steps[i].end_s}));
 const citations=new Map();for(const item of [...record.relationships,...record.steps])for(const c of item.citations||[]){
  const f=c.frame;if(typeof c.id==='string'&&finite(c.timestamp_s)&&c.timestamp_s>=0&&c.timestamp_s<=record.duration_s&&f&&typeof f.path==='string'&&/^data\/[A-Za-z0-9_./-]+$/.test(f.path)&&!f.path.split('/').some(s=>!s||s.startsWith('.'))&&/^[a-f0-9]{64}$/.test(f.sha256)&&Number.isSafeInteger(f.bytes)&&f.bytes>0&&!citations.has(c.id))citations.set(c.id,{id:c.id,timestamp_s:c.timestamp_s,frame:{path:f.path,sha256:f.sha256,bytes:f.bytes}});
 }
 const rows=[...citations.values()].sort((a,b)=>a.timestamp_s-b.timestamp_s||a.id.localeCompare(b.id)),positions=[0,Math.floor((rows.length-1)/2),rows.length-1];result.evidence=[...new Set(positions)].filter(i=>i>=0&&rows[i]).map(i=>rows[i]);
 return result;
}
const api={summarize};if(typeof module!=='undefined'&&module.exports)module.exports=api;root.ObjectProgramSummaryV1=api;
})(globalThis);
