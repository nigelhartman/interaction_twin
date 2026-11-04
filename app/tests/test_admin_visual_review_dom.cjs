/* Real passive review renderer with DOM doubles; these are not pixel QA. */
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const V=require('../admin/visual-review.js');
class Element{
 constructor(tag,doc){this.tagName=tag.toUpperCase();this.ownerDocument=doc;this.children=[];this.attributes={};this.events={};this.hidden=false;this._text='';}
 set textContent(value){this._text=String(value);this.children=[];}get textContent(){return this._text+this.children.map(e=>e.textContent).join('');}
 set innerHTML(value){throw Error('Unsafe HTML assignment: '+value);}
 append(...items){this.children.push(...items);}replaceChildren(...items){this.children=items;this._text='';}
 setAttribute(name,value){this.attributes[name]=String(value);}addEventListener(name,callback){this.events[name]=callback;}
}
function dom(){const doc={createElement(tag){return new Element(tag,doc);}};return doc.createElement('section');}
function nodes(root,tag){return[root,...root.children.flatMap(c=>nodes(c,tag))].filter(e=>!tag||e.tagName===tag.toUpperCase());}
const r=(path,letter='a')=>({path,sha256:letter.repeat(64),bytes:100});
function fixture(){
 const manifest={source_ref:r('data/uploads/source.mp4'),interaction_ref:r('data/fixture/procedure/interaction.json','b'),scene_ref:r('data/fixture/scene.json','c'),object_program_ref:r('data/fixture/object-program.json','d')};
 manifest.visual_review={status:'complete',artifact_binding:Object.fromEntries(['source_ref','interaction_ref','object_program_ref','scene_ref'].map(k=>[k,structuredClone(manifest[k])])),report_ref:r('.private/measured-visual-reviews/report.json','e'),
  issues:[{category:'mask_alignment',severity:'warning',component_ids:['earbud_a'],sample_ids:['earbud_a:61'],visible_observation:'At 61 seconds the generated earbud has a visible silhouette mismatch.',why_it_matters:'A hand may obscure the recorded boundary.'}],repair_targets:['tracking'],metrics:{development_samples_reviewed:6},improvement_assessment:{measurement_kind:'deterministic_local_metrics',metrics:{alignment_error:.25,coverage:.75,missing_components:0,blocking_issues:0},automatic_repair_safe:true}};
 const program={source:{sha256:'f'.repeat(64),original_sha256:'a'.repeat(64),duration_s:34},tracklets:[{id:'earbud_a'}],evidence:[{id:'source_000061',source_index:61,timestamp_s:20.001756}]};
 return{manifest,program,evidence:V.sourceEvidence(manifest,program,'program')};
}
function noNetwork(work){const previous=global.fetch;let calls=0;global.fetch=()=>{calls++;throw Error('Rendering must never send a request');};try{work();assert.equal(calls,0);}finally{global.fetch=previous;}}

test('complete visual review renders actual metrics and typed source times without reading private reports or sending requests',()=>{
 const{manifest,evidence}=fixture(),box=dom(),seek=[];
 noNetwork(()=>V.renderVisual(box,manifest,{sourceEvidence:evidence,onSeek:time=>seek.push(time)}));
 assert.match(box.textContent,/Visual evidence review/);assert.match(box.textContent,/Critic completed/);assert.match(box.textContent,/Median mask overlap75.0%/);assert.match(box.textContent,/Usable development comparisons75.0%/);
 assert.match(box.textContent,/sample 61 · 20.002 s/);assert.match(box.textContent,/61 seconds/,'The original model observation is retained; its prose time is not promoted into a timestamp.');assert.match(box.textContent,/prose may contain mistaken times/);assert.match(box.textContent,/hand occlusion/);assert.match(box.textContent,/do not verify 3D pose/);
 assert.equal(nodes(box,'a').length,0,'Private report references are not links or fetch targets.');const buttons=nodes(box,'button');assert.equal(buttons.length,1);assert.deepEqual(seek,[]);noNetwork(()=>buttons[0].events.click());assert.deepEqual(seek,[20.001756]);
});

test('unavailable review renders its saved reason safely and cannot invent metrics or trigger inference',()=>{
 const box=dom(),manifest={visual_review:{status:'unavailable',diagnostic:'Prior exact-state attempt is uncertain. <img src=x onerror=alert(1)>'}};
 noNetwork(()=>V.renderVisual(box,manifest));assert.match(box.textContent,/unavailable/i);assert.match(box.textContent,/<img src=x onerror=alert\(1\)>/);assert.match(box.textContent,/does not run a model or retry/);assert.equal(nodes(box,'img').length,0);assert.equal(nodes(box,'dl').length,0);assert.equal(nodes(box,'button').length,0);
});

test('model strings are text only, including malicious categories, observations and sample IDs',()=>{
 const{manifest,evidence}=fixture(),box=dom();manifest.visual_review.issues[0]={category:'<svg onload=alert(1)>',severity:'blocking',visible_observation:'<script>globalThis.pwned=true</script>',why_it_matters:'<a href="javascript:alert(1)">bad</a>',sample_ids:['javascript:alert(1)','<img src=x>']};
 noNetwork(()=>V.renderVisual(box,manifest,{sourceEvidence:evidence,onSeek:()=>assert.fail('Unknown citation must not seek')}));assert.match(box.textContent,/<script>globalThis.pwned=true<\/script>/);assert.match(box.textContent,/javascript:alert\(1\)/);assert.equal(nodes(box,'script').length,0);assert.equal(nodes(box,'a').length,0);assert.equal(nodes(box,'button').length,0);assert.equal(global.pwned,undefined);
});

test('unknown clocks stay unknown and an unbound review cannot borrow another revision’s evidence',()=>{
 const{manifest,evidence}=fixture(),box=dom();V.renderVisual(box,manifest);assert.match(box.textContent,/earbud_a:61 · source time unavailable/);assert.doesNotMatch(box.textContent,/20.002 s/);
 const changed=structuredClone(manifest);changed.interaction_ref=r('data/fixture/another.json','9');V.renderVisual(box,changed,{sourceEvidence:evidence});assert.match(box.textContent,/belongs to another revision/);assert.doesNotMatch(box.textContent,/visible silhouette mismatch/);assert.equal(nodes(box,'button').length,0);
});

test('scene citations require exact source/program binding and use source indices rather than narrative numbers',()=>{
 const{manifest}=fixture();const scene={source:{sha256:'f'.repeat(64),original_sha256:'a'.repeat(64),duration_s:34},original_program:manifest.interaction_ref,components:[{id:'earbud_a',poses:[{source_sample_index:61,timestamp_s:20.001756}]}]};
 const evidence=V.sourceEvidence(manifest,scene,'scene');assert.equal(evidence.samples.get('earbud_a:61').timestamp_s,20.001756);
 assert.equal(V.sourceEvidence(manifest,{...scene,original_program:r('data/wrong.json','0')},'scene'),null);assert.equal(V.sourceEvidence(manifest,{...scene,source:{sha256:'8'.repeat(64),duration_s:34}},'scene'),null);
 const duplicate=structuredClone(scene);duplicate.components[0].poses.push({source_sample_index:61,timestamp_s:21});assert.equal(V.sourceEvidence(manifest,duplicate,'scene'),null);
});

test('zero and missing measurements are distinct; invalid ranges are not converted into quality scores',()=>{
 const{manifest}=fixture(),box=dom();manifest.visual_review.improvement_assessment.metrics={alignment_error:0,coverage:0,missing_components:0,blocking_issues:0};V.renderVisual(box,manifest);assert.match(box.textContent,/Median mask overlap100.0%/);assert.match(box.textContent,/Usable development comparisons0.0%/);
 manifest.visual_review.improvement_assessment.metrics={alignment_error:NaN,coverage:1.1,missing_components:-1,blocking_issues:0};V.renderVisual(box,manifest);assert.match(box.textContent,/Median mask overlapUnavailable/);assert.match(box.textContent,/Usable development comparisonsUnavailable/);assert.doesNotMatch(box.textContent,/NaN|110.0%|\/10|physics confidence/i);
});

test('plateau and unknown stop explain why the best retained revision still requires owner approval',()=>{
 const detail={candidate:{current_revision_id:'revision_old'},best_revision_id:'revision_old',revisions:[{id:'revision_old',number:1},{id:'revision_new',number:2}],jobs:[],improvement_rounds:[{round_number:1,decision:'plateau_or_regression',best_revision_id:'revision_old',created_utc:'2026-11-03T01:00:00Z'}]},box=dom();
 noNetwork(()=>V.renderOutcome(box,detail));assert.match(box.textContent,/Automatic improvement stopped/);assert.match(box.textContent,/previously retained best result was kept/);assert.match(box.textContent,/Round 2 of at most 3/);assert.match(box.textContent,/Revision 1/);assert.match(box.textContent,/Approval still targets the exact current revision/);assert.equal(nodes(box,'button').length,0);
 detail.improvement_stops=[{round_number:2,state:'pending',diagnostic:'Model usage unavailable; preserved request has no retry.',created_utc:'2026-11-03T01:02:00Z'}];V.renderOutcome(box,detail);assert.match(box.textContent,/explicit recovery decision/);assert.match(box.textContent,/usage unavailable/);assert.match(box.textContent,/Round 3 of at most 3/);
});

test('queued repair is shown as active even when its immutable round record was written milliseconds later',()=>{
 const detail={revisions:[],jobs:[{id:'repair2',state:'queued',created_utc:'2026-11-03T01:00:00.001Z'}],improvement_rounds:[{round_number:0,decision:'repair_candidate',next_job_id:'repair2',created_utc:'2026-11-03T01:00:00.003Z'}]},box=dom();V.renderOutcome(box,detail);assert.match(box.textContent,/Automatic improvement is queued/);assert.doesNotMatch(box.textContent,/Automatic improvement stopped/);assert.match(box.textContent,/Round 2 of at most 3/);
});

test('resumed work does not present an old preserved stop as the current outcome',()=>{
 const detail={revisions:[],jobs:[{id:'resume',state:'running',created_utc:'2026-11-03T01:03:00Z'}],improvement_stops:[{round_number:0,state:'pending',created_utc:'2026-11-03T01:02:00Z'}]},box=dom();V.renderOutcome(box,detail);assert.match(box.textContent,/Private processing is running/);assert.match(box.textContent,/Saved rounds and stops/);assert.equal(box.hidden,false);V.renderOutcome(box,{});assert.equal(box.hidden,true);assert.equal(box.textContent,'');
});

test('the real owner page wires the passive section without changing approval or adding model routes',()=>{
 const html=fs.readFileSync(path.resolve(__dirname,'../admin/index.html'),'utf8'),js=fs.readFileSync(path.resolve(__dirname,'../admin/admin.js'),'utf8'),passive=fs.readFileSync(path.resolve(__dirname,'../admin/visual-review.js'),'utf8');
 assert.match(html,/id="visual-evidence-review"/);assert.match(html,/id="improvement-outcome"/);assert.ok(html.indexOf('src="visual-review.js"')<html.indexOf('src="admin.js"'));
 assert.match(js,/renderVisualEvidence\(current.manifest\)/);assert.match(js,/sourceEvidence\(manifest,scene,'scene'\)/);assert.match(js,/sourceEvidence\(manifest,program,'program'\)/);assert.match(js,/approvalPayload\(state.detail/);assert.doesNotMatch(passive,/\bfetch\s*\(|XMLHttpRequest|innerHTML|\/api\//);
});


test('a text-only review or unclassified first job is not mislabeled as an automatic improvement loop',()=>{
 const box=dom();V.renderOutcome(box,{jobs:[{id:'first',kind:'pipeline',state:'running'}]});assert.equal(box.hidden,true);
 const detail={jobs:[{id:'text',kind:'review',state:'running',created_utc:'2026-11-03T01:03:00Z'}],improvement_stops:[{round_number:0,state:'pending',created_utc:'2026-11-03T01:02:00Z'}]};V.renderOutcome(box,detail);assert.match(box.textContent,/Automatic improvement stopped/);assert.doesNotMatch(box.textContent,/improvement is running/);
});
