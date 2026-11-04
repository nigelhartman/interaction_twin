// Corrected, source-bound saved evidence only. No run endpoint or pose placement.
export const ASSEMBLY_INDEX_PATH='data/assembly-feasibility-20261101/v2/index.json';
export const ASSEMBLY_INDEX_SHA256='8b264a6d7d8ec38a378b9c866e54f8886a7cc53f86530f3e399de1730edbecd4';
const HASH=/^[a-f0-9]{64}$/;
export function artifactUrl(path){
  if(typeof path!=='string'||!/^data\/[A-Za-z0-9_./-]+$/.test(path)||path.split('/').some(p=>!p||p.startsWith('.')))throw new Error('Unsupported assembly artifact path');
  return `/files/${path}`;
}
async function checked(path,hash){
  if(!HASH.test(hash))throw new Error('Missing assembly artifact identity');
  const response=await fetch(artifactUrl(path),{cache:'no-store'});
  if(!response.ok)throw new Error(`Saved assembly artifact unavailable (${response.status})`);
  const bytes=await response.arrayBuffer();
  if(bytes.byteLength>2*1024*1024)throw new Error('Saved assembly artifact exceeds allowance');
  const actual=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(v=>v.toString(16).padStart(2,'0')).join('');
  if(actual!==hash)throw new Error('Saved assembly identity changed');
  return JSON.parse(new TextDecoder().decode(bytes));
}
function integer(value){if(!Number.isSafeInteger(value)||value<0)throw new Error('Invalid assembly count');return value;}
function positive(value){if(!Number.isFinite(value)||value<=0)throw new Error('Invalid assembly diagnostic');return value;}
export function normalizeAssembly(index,raw,physicsContract){
  if(index.format!=='interaction-twin-assembly-feasibility-index'||index.version!==2||index.read_only!==true||index.status!=='unsupported_common_scale')throw new Error('Only corrected V2 assembly evidence is supported');
  for(const flag of ['physical_scale_known','physical_assembly_established','joint_axis_established','contact_or_insertion_established','simulation_ready','automatic_promotion_allowed'])if(index[flag]!==false)throw new Error('Unsupported physical assembly claim');
  if(index.v1_excitation_criterion_untrusted_due_to_mixed_native_units!==true||index.supersedes_criterion_only?.old_results_preserved!==true)throw new Error('Superseded mixed-unit criterion must remain untrusted');
  if(index.interaction_sha256!==physicsContract.interaction.sha256||index.interaction_path!==physicsContract.interaction.path||index.source_video_sha256!==physicsContract.source_video.sha256||index.case_id!==physicsContract.case_id)throw new Error('Assembly and physics source identities differ');
  if(raw.format!=='co-rigid-scale-compatibility-report'||raw.version!==2||raw.protocol_sha256!==index.artifacts.corrected_protocol.sha256||raw.accepted_conditional_hypotheses!==0||raw.physical_assembly_established!==false||raw.metric_scale_known!==false)throw new Error('Unsupported corrected assembly report');
  const scale=index.scale_diagnostics,support=index.support;
  if(raw.policy?.version!==scale.corrected_policy.version||raw.policy.excitation_is_necessary_not_sufficient!==true)throw new Error('Corrected unit-invariant policy must remain explicit');
  if(scale.accepted_conditional_hypotheses!==0||scale.common_frame_transform_a_from_b!==null||scale.reserved_rows_used_to_fit!==false||scale.diagnostic_ratios_are_not_accepted_scales!==true||support.comparison_is_matched_input_accuracy_benchmark!==false)throw new Error('Rejected diagnostics cannot become an accepted transform or accuracy benchmark');
  if(!Array.isArray(raw.pairs)||raw.pairs.length>16)throw new Error('Invalid assembly pair inventory');
  for(const pair of raw.pairs){
    if(pair.status!=='unsupported'||pair.common_frame_transform_a_from_b!==null||pair.physical_assembly_established!==false||pair.physical_scale_known!==false||pair.joint_or_contact_established!==false||pair.reserved_used_to_fit!==false||pair.geometry_or_saved_poses_changed!==false||pair.interaction_sha256!==index.interaction_sha256||pair.source_video_sha256!==index.source_video_sha256)throw new Error('Assembly pair violates source or physical boundaries');
    for(const [component,hash]of [[pair.component_a,pair.mesh_a_sha256],[pair.component_b,pair.mesh_b_sha256]])if(!physicsContract.components.some(row=>row.id===component&&row.visual_assets.some(asset=>asset.sha256===hash)))throw new Error('Assembly native asset differs from the source-backed physics component');
  }
  const pairs=raw.pairs.filter(pair=>pair.component_a==='case_base'&&pair.component_b==='case_lid'&&pair.training_count===scale.paired.training_timestamps&&pair.diagnostic_fit?.column_normalized_condition===scale.paired.normalized_condition);
  if(pairs.length!==1)throw new Error('Corrected paired diagnostic must have one exact source record');
  const pair=pairs[0],fit=pair.diagnostic_fit;
  if(fit.fit_status!=='rejected_diagnostic_only'||fit.translation_excitation_over_anchor_depth!==scale.paired.translation_excitation_over_anchor_depth||fit.leave_one_training_out_relative_spread!==scale.paired.leave_one_training_out_relative_spread)throw new Error('Corrected paired metrics differ from raw report');
  const counts={dense:integer(support.paired_simultaneous_nonreference_supported),denseTotal:integer(support.paired_simultaneous_nonreference_planned),sparse:integer(support.sparse_simultaneous_nonreference_supported),sparseTotal:integer(support.sparse_simultaneous_nonreference_planned),withheld:integer(support.base_scope_rejected_withheld_frames)};
  if(counts.dense>counts.denseTotal||counts.sparse>counts.sparseTotal)throw new Error('Supported counts exceed planned counts');
  const condition=positive(fit.column_normalized_condition),conditionMax=positive(scale.gate_condition_max),spread=positive(fit.leave_one_training_out_relative_spread),spreadMax=positive(scale.gate_leave_one_training_out_relative_spread_max),excitation=positive(fit.translation_excitation_over_anchor_depth),excitationMin=positive(scale.gate_corrected_translation_excitation_min);
  if(condition<=conditionMax||spread<=spreadMax||excitation>=excitationMin)throw new Error('This saved unsupported result must retain all three failed gates');
  const accepted=raw.pairs.filter(pair=>pair.status==='supported').length;
  if(accepted!==0)throw new Error('No assembly transform is accepted in this saved experiment');
  return {counts,condition,conditionMax,spread,spreadMax,excitation,excitationMin,accepted:0,rawReference:index.artifacts.corrected_compatibility,humanReport:index.report,oldCriterion:index.supersedes_criterion_only,sourceProgramHash:index.interaction_sha256,policy:scale.corrected_policy};
}
function node(tag,text,className){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(className)el.className=className;return el;}
function link(label,path,className){const el=node('a',label,className);el.href=artifactUrl(path);return el;}
export async function mountAssemblyEvidence(physicsContract){
  const host=document.getElementById('assembly-content'),status=document.getElementById('assembly-status');
  try{
    const index=await checked(ASSEMBLY_INDEX_PATH,ASSEMBLY_INDEX_SHA256);
    const rawRef=index.artifacts?.corrected_compatibility;
    if(!rawRef)throw new Error('Corrected assembly report is absent');
    const raw=await checked(rawRef.path,rawRef.sha256),model=normalizeAssembly(index,raw,physicsContract);
    const stats=node('div',undefined,'assembly-stats');
    for(const [value,label,note]of [[`${model.counts.dense} / ${model.counts.denseTotal}`,'New paired source support','Non-reference simultaneous timestamps'],[`${model.counts.sparse} / ${model.counts.sparseTotal}`,'Earlier sparse source support','Different sampling density'],['0','Accepted common-frame transforms','No transform is available for assembly']]){
      const card=node('div',undefined,'assembly-stat');card.append(node('strong',value),node('span',label),node('small',note));stats.append(card);
    }
    host.append(stats,node('p','These counts describe source availability at different sampling densities. They are not a matched accuracy comparison; no physical or cadence accuracy improvement is established.','assembly-context'));
    const gates=node('div',undefined,'assembly-gates');
    const gateRows=[['Fit conditioning',`${model.condition.toFixed(2)} > ${model.conditionMax.toFixed(0)}`,'The scale-and-offset fit is ill-conditioned.'],['Scale stability',`${(model.spread*100).toFixed(2)}% > ${(model.spreadMax*100).toFixed(0)}%`,'The fitted scale changes too much when one TRAIN timestamp is omitted.'],['Translation evidence',`${(model.excitation*100).toFixed(3)}% < ${(model.excitationMin*100).toFixed(0)}%`,'Motion divided by anchor depth uses the same A-native units. This necessary gate is still below its minimum.']];
    for(const [title,value,description]of gateRows){const card=node('div',undefined,'assembly-gate');card.append(node('span','WITHHELD'),node('strong',title),node('b',value),node('p',description));gates.append(card);}
    host.append(gates,node('p',`${model.counts.withheld} scope-rejected frames remain withheld. Reference poses are unverified and excluded from the support counts. There is no common metric scale, recovered joint, contact or insertion result. Gaps are not bridged.`, 'assembly-context'));
    const history=node('details',undefined,'assembly-history');history.append(node('summary','Why V2 replaces the earlier criterion'));
    history.append(node('p','V1 divided motion and depth expressed in different native units. Its excitation criterion is superseded and untrusted. V2 uses A-native units in both terms and preserves the original records and unchanged source-support counts. The corrected checks still reject the assembly.'));
    const historyLinks=node('div',undefined,'assembly-links');historyLinks.append(link('Superseded criterion record · V1 ↗',model.oldCriterion.path));history.append(historyLinks);host.append(history);
    const links=node('div',undefined,'assembly-links');links.append(link('Corrected raw report · V2 ↗',model.rawReference.path),link('Read the assembly report ↗',model.humanReport.path));
    const reconstruction=node('a','Inspect existing video reconstruction ↗','assembly-capture');reconstruction.href='geometry-lab.html';links.append(reconstruction);host.append(links);
    status.textContent='Verified corrected V2 · common scale unsupported';status.classList.add('warn');host.dataset.integrityStatus='verified';host.dataset.indexSha256=ASSEMBLY_INDEX_SHA256;
  }catch(error){status.textContent='Saved assembly evidence unavailable or identity check failed';host.append(node('p',error.message,'assembly-context'));host.dataset.integrityStatus='failed';console.error('Assembly evidence:',error.message);}
}
