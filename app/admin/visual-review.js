/* Passive owner-only review presentation. No fetch, inference, mutation or proof loading. */
(function(root){'use strict';
const SHA=/^[a-f0-9]{64}$/, TARGETS=new Set(['reasoning','tracking','reconstruction','mechanics']);
const list=value=>Array.isArray(value)?value:[];
const human=value=>String(value||'unknown').replaceAll('_',' ');
const number=value=>typeof value==='number'&&Number.isFinite(value)?value:null;
const ratio=value=>number(value)!==null&&value>=0&&value<=1?value:null;
const count=value=>Number.isSafeInteger(value)&&value>=0?value:null;
const percent=value=>value===null?'Unavailable':(value*100).toFixed(1)+'%';
function sameRef(a,b){return a==null||b==null?a==null&&b==null:['path','sha256','bytes'].every(k=>a[k]===b[k]);}
function exactBinding(review,manifest){return review?.artifact_binding&&['source_ref','interaction_ref','object_program_ref','scene_ref'].every(k=>sameRef(review.artifact_binding[k],manifest[k]));}
function sourceEvidence(manifest,data,kind){
 // The controller calls this only after its existing byte/hash and scene/source
 // validation. Narrative timestamps and sample-ID suffixes are never seconds.
 const upload=manifest?.source_ref?.sha256,source=data?.source;
 if(!SHA.test(upload||'')||!source||!SHA.test(source.sha256||'')||![source.sha256,source.original_sha256].includes(upload))return null;
 let rows=[];
 if(kind==='scene'){
  if(!sameRef(data.original_program,manifest.interaction_ref))return null;
  for(const c of list(data.components))for(const p of list(c.poses))rows.push({sample_id:c.id+':'+p.source_sample_index,component_id:c.id,source_index:p.source_sample_index,timestamp_s:p.timestamp_s});
 }else if(kind==='program'){
  for(const c of list(data.tracklets))for(const e of list(data.evidence))rows.push({sample_id:c.id+':'+e.source_index,component_id:c.id,source_index:e.source_index,timestamp_s:e.timestamp_s});
 }else return null;
 const duration=number(source.duration_s),samples=new Map();
 for(const row of rows){
  if(typeof row.component_id!=='string'||!Number.isSafeInteger(row.source_index)||row.source_index<1||number(row.timestamp_s)===null||row.timestamp_s<0||(duration!==null&&row.timestamp_s>duration))continue;
  const old=samples.get(row.sample_id);
  if(old&&(old.timestamp_s!==row.timestamp_s||old.component_id!==row.component_id))return null;
  samples.set(row.sample_id,row);
 }
 return{source_sha256:source.sha256,interaction_ref:manifest.interaction_ref,scene_ref:kind==='scene'?manifest.scene_ref:null,samples};
}
function visualSummary(manifest={},evidence=null){
 const review=manifest.visual_review;
 const valid=review?.status==='complete'&&exactBinding(review,manifest);
 const unavailable=review?.status==='complete'&&!valid?'The visual report belongs to another revision. Its findings are withheld.':String(review?.diagnostic||review?.reason||'No saved visual evidence review is available for this revision.');
 if(!valid)return{available:false,status:'Unavailable',diagnostic:unavailable,issues:[],metrics:null,targets:[],report:null,evidence:null};
 const assessment=review.improvement_assessment,metrics=assessment?.measurement_kind==='deterministic_local_metrics'?assessment.metrics:null;
 const basis=evidence&&evidence.samples instanceof Map&&sameRef(evidence.interaction_ref,manifest.interaction_ref)&&(!evidence.scene_ref||sameRef(evidence.scene_ref,manifest.scene_ref))?evidence:null;
 return{available:true,status:'Critic completed',diagnostic:'',issues:list(review.issues).slice(0,8),
  metrics:metrics?{overlap:ratio(metrics.alignment_error)===null?null:1-metrics.alignment_error,coverage:ratio(metrics.coverage),missing:count(metrics.missing_components),blocking:count(metrics.blocking_issues)}:null,
  reviewed:count(review.metrics?.development_samples_reviewed),targets:list(review.repair_targets).filter(t=>TARGETS.has(t)),
  report:SHA.test(review.report_ref?.sha256||'')?review.report_ref:null,evidence:basis,
  automaticRepairSafe:assessment?.automatic_repair_safe===true};
}
const EXPLANATIONS={
 repair_candidate:'A measured repair was queued within the bounded policy. It must be reviewed again before you can approve it.',
 plateau_or_regression:'The repair did not meet the measured non-regression threshold. The previously retained best result was kept.',
 unknown_visual_quality:'Comparable visual measurements are unavailable. Automatic improvement stopped without guessing a quality score.',
 structural_review_failed:'Source-integrity or structural checks failed. Automatic improvement stopped; these checks cannot be bypassed.',
 review_incomplete:'Required hosted metadata reviews are incomplete. Outputs and earlier attempts were preserved without an automatic retry.',
 repair_requires_human_inspection:'The proposed repair was not supported by enough safe development evidence. Your inspection is required.',
 no_supported_repair_needed:'No supported repair target remained. This does not establish physical accuracy.',
 round_limit_reached:'The initial build and at most two repairs have been reached. The retained best result now needs your review.',
 budget_exhausted:'The estimated spending guard stopped additional model work. Existing results and reservations were preserved.',
 budget_or_usage_unknown:'Spending admission could not be established. Automatic improvement stopped; unknown charges were not treated as zero.',
 budget_admission_unavailable:'No authorized spending-admission check was available, so no additional repair was queued.',
 authorization_policy_changed:'The approved execution policy changed. Automatic improvement stopped for review.',
 human_or_external_decision_preempted:'A newer owner decision or another job superseded this round. Automatic improvement left that decision in place.',
 failed:'The attempt failed and its outputs were preserved. No automatic paid retry was made.',
 pending:'The pipeline stopped pending an explicit recovery decision. Completed paid outputs remain preserved.',
 interrupted:'The worker was interrupted. Preserved attempts must be inspected before recovery; they were not automatically resent.'
};
function roundInfo(row){
 let explanation=EXPLANATIONS[row?.decision||row?.state]||'The saved outcome needs inspection; no quality or physical success is inferred.';
 if(row?.assessment?.comparison_key&&row.best_comparison_key&&row.assessment.comparison_key!==row.best_comparison_key)explanation='The comparison basis changed. Measurements from different bases were not ranked; the prior best was retained.';
 const n=count(row?.round_number);
 return{number:n===null?null:n+1,decision:human(row?.decision||row?.state),explanation,diagnostic:typeof row?.diagnostic==='string'?row.diagnostic:'',best:row?.best_revision_id||null};
}
function outcomeSummary(detail={}){
 const rounds=list(detail.improvement_rounds),stops=list(detail.improvement_stops),events=[...rounds.map(row=>({row,type:'round'})),...stops.map(row=>({row,type:'stop'}))].sort((a,b)=>String(a.row.created_utc||'').localeCompare(String(b.row.created_utc||''))),latest=events.at(-1);
 const active=list(detail.jobs).find(j=>['queued','running'].includes(j.state)&&j.kind!=='review');
 const resumed=active&&(!latest||latest.row.next_job_id===active.id||String(active.created_utc||'')>=String(latest.row.created_utc||''));
 const best=detail.best_revision_id||latest?.row.best_revision_id||null,revision=list(detail.revisions).find(r=>r.id===best);
 const bestLabel=revision?'Revision '+revision.number:best?'Saved revision '+String(best).slice(-8):'No reviewed result yet';
 if(!latest)return null;
 if(resumed){const preceding=rounds.filter(r=>r.next_job_id===active.id).at(-1);return{title:(preceding?'Automatic improvement is ':'Private processing is ')+(active.state==='queued'?'queued':'running'),explanation:preceding?'The candidate remains private. Its next reviewed result will be compared on the same frozen development basis.':'An earlier stop remains in the history. The resumed candidate is private; its new result still needs review.',bestLabel,active:true,latest:preceding?roundInfo({...preceding,round_number:preceding.round_number+1}):null,events};}
 const current=roundInfo(latest.row);return{title:'Automatic improvement stopped',explanation:current.explanation,bestLabel,active:false,latest:current,events};
}
function node(container,tag,text,css){const e=container.ownerDocument.createElement(tag);if(text!==undefined)e.textContent=String(text);if(css)e.className=css;return e;}
function renderOutcome(container,detail={},options={}){
 container.replaceChildren();const summary=outcomeSummary(detail);container.hidden=!summary;if(!summary)return;
 const head=node(container,'div',undefined,'visual-review-top');head.append(node(container,'strong',summary.title),node(container,'span',summary.active?'In progress':'Owner review required','state-chip '+(summary.active?'active':'warning')));container.append(head,node(container,'p',summary.explanation));
 if(summary.latest?.number)container.append(node(container,'p','Round '+summary.latest.number+' of at most 3 · '+summary.latest.decision,'visual-review-note'));
 container.append(node(container,'p','Best retained by the automatic comparison: '+summary.bestLabel+'. Approval still targets the exact current revision.','visual-review-note'));
 if(summary.latest?.diagnostic)container.append(node(container,'p',summary.latest.diagnostic,'visual-review-diagnostic'));
 if(summary.events.length){const history=node(container,'details',undefined,'visual-round-history');history.append(node(container,'summary','Saved rounds and stops ('+summary.events.length+')'));const rows=node(container,'ol');for(const item of summary.events){const r=roundInfo(item.row),li=node(container,'li');li.append(node(container,'strong',(r.number?'Round '+r.number:'Saved attempt')+' · '+r.decision),node(container,'p',r.explanation));if(r.diagnostic)li.append(node(container,'p',r.diagnostic,'visual-review-diagnostic'));if(item.row.measured_improvement===true)li.append(node(container,'p','Accepted by the local development comparator; this is not physical acceptance.','visual-review-note'));rows.append(li);}history.append(rows);container.append(history);}
}
function renderVisual(container,manifest={},options={}){
 container.replaceChildren();const data=visualSummary(manifest,options.sourceEvidence),top=node(container,'div',undefined,'review-section-heading'),title=node(container,'div');title.append(node(container,'span','SOURCE FRAMES + NATIVE RENDERS','eyebrow'),node(container,'h3','Visual evidence review.'));top.append(title,node(container,'span',data.status,'state-chip '+(data.available?'':'warning')));container.append(top);
 if(!data.available){container.append(node(container,'p',data.diagnostic,'visual-review-diagnostic'),node(container,'p','No visual conclusion is inferred. Browsing this review does not run a model or retry an attempt.','visual-review-note'));return;}
 const reviewed=data.reviewed===null?'':data.reviewed+' sampled component comparisons. ';
 container.append(node(container,'p',reviewed+'Muse compared source images with rendered native meshes. Its findings remain model observations, separate from the text-only metadata roles.'));
 if(data.metrics){const grid=node(container,'dl',undefined,'visual-metrics');for(const [label,value]of[['Median mask overlap',percent(data.metrics.overlap)],['Usable development comparisons',percent(data.metrics.coverage)],['Missing components',data.metrics.missing===null?'Unavailable':data.metrics.missing]]){const pair=node(container,'div');pair.append(node(container,'dt',label),node(container,'dd',value));grid.append(pair);}container.append(grid);}
 else container.append(node(container,'p','Comparable local mask measurements are unavailable. No quality score has been invented.','visual-review-note'));
 container.append(node(container,'p','Mask mismatch can reflect hand occlusion, hidden surfaces or generated shape errors. These development diagnostics do not verify 3D pose, physical fit, contact or simulator readiness.','visual-review-note'));
 if(data.evidence)container.append(node(container,'p','Evidence source '+data.evidence.source_sha256.slice(0,16)+'… · Sample IDs are indices; cited times below come from the saved source clock. Critic prose may contain mistaken times.','visual-review-note'));
 else container.append(node(container,'p','The verified source clock is not available yet. Sample IDs are shown without guessed timestamps.','visual-review-note'));
 const issues=node(container,'div',undefined,'visual-issues');
 for(const issue of data.issues){const card=node(container,'article',undefined,'visual-issue'),heading=node(container,'div',undefined,'visual-review-top');heading.append(node(container,'strong',human(issue.category||'visual finding')),node(container,'span',issue.severity==='blocking'?'Blocking finding':'Review finding','state-chip '+(issue.severity==='blocking'?'error':'warning')));card.append(heading,node(container,'p',issue.visible_observation||issue.message||'No visible observation was supplied.'));if(issue.why_it_matters)card.append(node(container,'p',issue.why_it_matters,'visual-review-note'));
  const citations=node(container,'div',undefined,'visual-citations');for(const id of list(issue.sample_ids).slice(0,6)){const row=data.evidence?.samples.get(id),label=row?row.component_id+' · sample '+row.source_index+' · '+row.timestamp_s.toFixed(3)+' s':String(id)+' · source time unavailable';const el=node(container,row&&typeof options.onSeek==='function'?'button':'span',label,'visual-citation');if(el.tagName.toLowerCase()==='button'){el.type='button';el.setAttribute('aria-label','Inspect '+label+' in the source recording');el.addEventListener('click',()=>options.onSeek(row.timestamp_s));}citations.append(el);}if(list(issue.sample_ids).length)card.append(citations);issues.append(card);
 }
 if(data.issues.length)container.append(issues);else container.append(node(container,'p','No issue is listed in the saved critic report. This is not proof of physical accuracy.','visual-review-note'));
 if(data.targets.length)container.append(node(container,'p','Suggested bounded repair stages: '+data.targets.map(human).join(' · '),'visual-review-note'));
 if(data.report){const proof=node(container,'details',undefined,'visual-proof');proof.append(node(container,'summary','Private report reference'),node(container,'code',data.report.sha256));proof.append(node(container,'p','The exact report remains private. It is not loaded, edited or rerun by this view.','visual-review-note'));container.append(proof);}
}
const api={sameRef,sourceEvidence,visualSummary,outcomeSummary,renderVisual,renderOutcome};if(typeof module!=='undefined'&&module.exports)module.exports=api;root.InteractionVisualReview=api;
})(globalThis);
