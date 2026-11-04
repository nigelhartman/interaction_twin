/* Execute the real review module against browser-shaped controls, without network. */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const original=JSON.parse(fs.readFileSync(process.argv[2],'utf8')),scenario=process.argv[3];
const copy=x=>JSON.parse(JSON.stringify(x));
function element(tag='div'){
 const e={tagName:tag.toUpperCase(),children:[],dataset:{},value:'',textContent:'',hidden:false,disabled:false,selected:false,multiple:false,
  append(...rows){this.children.push(...rows)},replaceChildren(...rows){this.children=rows;this.value=''},
  querySelectorAll(selector){const out=[];for(const child of this.children){if(selector==='[data-field]'?child.dataset.field:['INPUT','TEXTAREA','SELECT'].includes(child.tagName))out.push(child);out.push(...child.querySelectorAll(selector))}return out}};
 Object.defineProperty(e,'selectedOptions',{get(){return this.children.filter(c=>c.selected)}});return e;
}
function setup(protocol='file:',request=null){
 const nodes=new Map(),calls=[],changes=[],downloads=[],seeks=[];
 const get=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id)};
 const sandbox={console,Map,Set,Date,Number,Math,Object,JSON,location:{protocol},document:{createElement:element}};sandbox.window=sandbox;
 vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../workspace/review.js'),'utf8'),sandbox);
 const c={id:'sample--procedure-v1',original_sha256:'a'.repeat(64),interaction:copy(original)};
 const editor=new sandbox.TwinReview.Editor({get,url:x=>'/files/'+x,seek:t=>seeks.push(t),onChange:(document,review)=>changes.push({document:copy(document),review}),download:(data,name,type)=>downloads.push({data:copy(data),name,type}),request:async(route,body)=>{calls.push({route,body:copy(body??null)});if(request)return request(route,body);return{original_sha256:c.original_sha256,head_revision_id:null,revisions:[]}}});
 editor.setCase(c);
 return{get,editor,c,calls,changes,downloads,seeks,review:sandbox.TwinReview};
}
const flush=()=>new Promise(resolve=>setImmediate(resolve));
function field(env,key){return env.get('review-fields').querySelectorAll('[data-field]').find(e=>e.dataset.field===key)}
async function run(){
 if(scenario==='offline_review'){
  const e=setup();assert.equal(e.calls.length,0);e.editor.open('steps','visible_opening');
  e.get('review-decision').value='edited';field(e,'title').value='Human corrected title';e.editor.applyForm();
  assert.equal(e.editor.envelope().document.steps[0].title,'Human corrected title');assert.equal(e.editor.envelope().saved,false);
  assert.equal(e.editor.envelope().authorship,'user_review_via_local_workspace');assert.equal(e.get('review-save').disabled,true);
  assert.equal(e.c.interaction.steps[0].title,'Visible opening','Original payload remains unchanged');
  e.editor.exportReview();e.editor.exportGuide();assert.equal(e.downloads.length,2);assert.match(e.downloads[1].data,/UNSAVED DRAFT/);
  assert.match(e.downloads[1].data,/not physical simulation/);assert.match(e.downloads[1].data,/source_000001/);
  e.editor.open('assertions','open_view');e.get('review-decision').value='rejected';e.editor.applyForm();
  assert.equal(e.editor.envelope().document.assertions.length,1);
  e.get('review-evidence').children[0].onclick();assert.equal(e.seeks.at(-1),1);
  e.editor.setMode('original');assert.equal(e.changes.at(-1).document.assertions.length,2);e.editor.setMode('review');assert.equal(e.changes.at(-1).document.assertions.length,1);
 }else if(scenario==='pending_save'){
  let resolvePost,submitted;const e=setup('http:',async(route,body)=>{if(body){submitted=copy(body);return new Promise(resolve=>resolvePost=resolve)}return{original_sha256:'a'.repeat(64),head_revision_id:null,revisions:[]}});await flush();
  e.editor.open('steps','visible_opening');e.get('review-decision').value='edited';field(e,'title').value='Submitted title';e.editor.applyForm();
  const pending=e.editor.save();assert.equal(e.get('review-save').disabled,true);assert.equal(e.editor.session.saving,true);
  field(e,'title').value='Unsubmitted title';e.editor.applyForm();e.editor.resetItem();e.editor.setMode('original');
  assert.equal(e.editor.envelope().document.steps[0].title,'Submitted title');assert.equal(e.editor.session.mode,'review');
  const firstSession=e.editor.session;e.editor.setCase({...e.c,id:'different--procedure-v1'});await flush();const changeCount=e.changes.length;
  const revision='rev-000001-0123456789abcdef';resolvePost({revision_id:revision,document:e.review.apply(original,submitted.edits).document});await pending;
  assert.equal(e.changes.length,changeCount,'A completed save must not replace another project');
  assert.equal(firstSession.dirty,false);assert.equal(firstSession.parentId,revision);assert.equal(firstSession.saving,false);
  const restored=e.editor.setCase(e.c);assert.equal(restored.steps[0].title,'Submitted title');assert.equal(e.editor.envelope().saved,true);
  assert.equal(e.calls.filter(c=>c.body).length,1,'Pending save guards must not submit another request');
 }else if(scenario==='conflict_keeps_draft'){
  const e=setup('http:',async(_route,body)=>{if(body)throw Error('Another review was saved; load the latest revision');return{original_sha256:'a'.repeat(64),head_revision_id:null,revisions:[]}});await flush();
  e.editor.open('steps','visible_opening');e.editor.applyForm();await e.editor.save();
  assert.equal(e.editor.session.dirty,true);assert.equal(e.editor.session.saving,false);assert.equal(e.editor.envelope().saved,false);
  assert.match(e.get('review-message').textContent,/Another review/);assert.match(e.get('review-message').textContent,/export/);
 }else if(scenario==='certainty_and_evidence'){
  const e=setup(),edits=e.review.empty();edits.assertions.closed_view={decision:'edited',note:'Occlusion',patch:{status:'unknown'}};
  const out=e.review.apply(original,edits).document;assert.equal(out.assertions[0].value,'unknown');assert.equal(out.assertions[0].status,'unknown');
  assert.equal(out.steps[0].preconditions[0].value,'unknown');assert.equal(out.runtime_policy.physical_identity_confirmed,false);
  edits.assertions.closed_view.patch.value='open';assert.throws(()=>e.review.apply(original,edits),/Unknown status/);
  edits.assertions.closed_view.patch={evidence_id:'invented'};assert.throws(()=>e.review.apply(original,edits),/existing evidence/);
  edits.assertions.closed_view.patch={value:'physically_verified'};assert.throws(()=>e.review.apply(original,edits),/existing state value/);
 }else if(scenario==='stale_revision_failure'){
  let rejectRevision;const e=setup('http:',async(route)=>{if(/\/revisions\/rev-/.test(route))return new Promise((_resolve,reject)=>rejectRevision=reject);return{original_sha256:'a'.repeat(64),head_revision_id:null,revisions:[]}});await flush();
  const pending=e.editor.loadRevision('rev-000001-0123456789abcdef'),firstSession=e.editor.session;
  assert.equal(firstSession.loading,true);e.editor.setCase({...e.c,id:'second--procedure-v1'});await flush();
  rejectRevision(Error('Old project revision was unavailable'));await pending;
  assert.equal(firstSession.loading,false);assert.doesNotMatch(e.get('review-message').textContent,/Old project/);
  assert.equal(e.editor.case.id,'second--procedure-v1');assert.equal(e.changes.length,0);
 }else throw Error('Unknown scenario '+scenario);
 console.log(scenario+' passed');
}
run().catch(error=>{console.error(error);process.exitCode=1});
