function safeReview(){return{all_required_roles_completed:true,blocking_failures:0,roles:['evidence','geometry','mechanics','semantics'].map(role=>({role,status:role==='evidence'?'pass':'warning',issues:[]})),hosted_roles:[]};}
const test=require('node:test'),assert=require('node:assert/strict');
const A=require('../admin/admin.js'),ORIGIN='http://127.0.0.1:8766';
function fixture({status=200,error='Request rejected',throws=false}={}){const calls=[],client=A.createClient(async(path,init)=>{calls.push({path,init});if(throws)throw Error('Network response unknown');return{ok:status<400,status,json:async()=>status<400?{ok:true}:{error}};},ORIGIN);return{client,calls};}

test('an owner mutation requires a valid session token and stays same-origin',async()=>{
 const f=fixture();await assert.rejects(f.client.request('/api/admin/candidates',{method:'POST',body:{}}),/owner session/);assert.equal(f.calls.length,0);
 f.client.setSession({authenticated:true,csrf_token:'A'.repeat(43)});
 await assert.rejects(f.client.request('https://example.com/api/admin/candidates',{method:'POST',body:{}}),/same-origin/);
 await assert.rejects(f.client.request('/files/data/private/video.mp4',{method:'POST',body:{}}),/same-origin/);assert.equal(f.calls.length,0);
 await f.client.request('/api/admin/candidates',{method:'POST',body:{input_path:'data/uploads/clip.mp4',processing_consent:true}});
 assert.equal(f.calls.length,1);const request=f.calls[0];assert.equal(request.init.headers.get('X-CSRF-Token'),'A'.repeat(43));assert.equal(request.init.credentials,'same-origin');assert.equal(request.init.cache,'no-store');assert.deepEqual(JSON.parse(request.init.body),{input_path:'data/uploads/clip.mp4',processing_consent:true});
});

test('setup can use an anonymous session; a rotated token replaces it without browser credential storage',async()=>{
 const f=fixture();f.client.setSession({authenticated:false,csrf_token:'B'.repeat(43)});
 await f.client.request('/api/admin/setup',{method:'POST',body:{setup_code:'UNIT_TEST_ONLY_NOT_REAL',password:'UNIT_TEST_ONLY_NOT_REAL'}});
 f.client.setSession({authenticated:true,csrf_token:'C'.repeat(43)});await f.client.request('/api/admin/logout',{method:'POST',body:{}});
 assert.equal(f.calls[0].init.headers.get('X-CSRF-Token'),'B'.repeat(43));assert.equal(f.calls[1].init.headers.get('X-CSRF-Token'),'C'.repeat(43));
});

test('failed or ambiguous mutation responses are sent once, never retried',async()=>{
 for(const options of[{status:403},{status:409},{throws:true}]){const f=fixture(options);f.client.setSession({authenticated:true,csrf_token:'D'.repeat(43)});await assert.rejects(f.client.request('/api/admin/candidates/example/repair',{method:'POST',body:{processing_consent:true}}));assert.equal(f.calls.length,1);}
});

test('binary upload preserves the file body and uses an explicit bounded basename',async()=>{
 const f=fixture(),bytes=new Uint8Array([1,2,3]);f.client.setSession({authenticated:true,csrf_token:'E'.repeat(43)});
 await f.client.request('/api/upload',{method:'POST',raw:true,body:bytes,headers:{'Content-Type':'application/octet-stream','X-Filename':A.safeUploadName('coin capture.MOV')}});
 assert.equal(f.calls[0].init.body,bytes);assert.equal(f.calls[0].init.headers.get('Content-Type'),'application/octet-stream');assert.equal(f.calls[0].init.headers.get('X-Filename'),'coin capture.mov');
 for(const name of['thing.mp4','äö capture.MOV','../capture.webm','a'.repeat(180)+'.m4v'])assert.match(A.safeUploadName(name),/^[A-Za-z0-9][A-Za-z0-9_. -]{0,119}\.(?:mp4|mov|webm|m4v)$/);
 assert.throws(()=>A.safeUploadName('secret.txt'),/Choose an/);
});

test('approval always targets the exact current revision and immutable fingerprint',()=>{
 const old={id:'revision_'+'a'.repeat(32),fingerprint:'b'.repeat(64)},current={id:'revision_'+'c'.repeat(32),fingerprint:'d'.repeat(64),manifest:{review:safeReview()}},detail={candidate:{current_revision_id:current.id},revisions:[old,current]};
 assert.deepEqual(A.approvalPayload(detail),{revision_id:current.id,expected_fingerprint:current.fingerprint});
 assert.throws(()=>A.approvalPayload({...detail,revisions:[old]}),/exact current revision/);assert.throws(()=>A.approvalPayload({...detail,revisions:[{...current,fingerprint:'invalid'}]}),/fingerprint/);
});

test('private preview references reject arbitrary URLs, traversal, hidden state and changed-size contracts',()=>{
 const ref={path:'data/safe/scene.json',sha256:'a'.repeat(64),bytes:100};assert.equal(A.dataRef(ref),ref);
 for(const path of['.secrets/admin-setup-code','data/../.secrets/meta.env','https://example.com/scene.json','data//scene.json','data/.private/scene.json'])assert.throws(()=>A.dataRef({...ref,path}),/data reference/);
 assert.throws(()=>A.dataRef({...ref,bytes:9*1024*1024}),/data reference/);
});


test('hosted text-only roles retain blocking findings and repair suggestions without claiming visual review',()=>{
 const report=A.hostedReviewSummary({role:'mechanics',model:'muse-spark-1.3',verdict:'review_required',summary:'Insertion is not established.',issues:[{severity:'blocking',message:'Joint sample is unsupported.'}],repair_targets:['mechanics','reconstruction']});
 assert.equal(report.verdict,'Human review required');assert.equal(report.blocked,true);assert.deepEqual(report.issues,[{message:'Joint sample is unsupported.',blocking:true}]);assert.deepEqual(report.repairTargets,['mechanics','reconstruction']);assert.equal(A.hostedReviewSummary({verdict:'blocked'}).verdict,'Blocked');
});


test('partial hosted reviews show actual completed roles and unavailable diagnostics without retry claims',()=>{
 const review={hosted_roles_completed:3,hosted_roles_expected:4,hosted_review_enabled:true,hosted_roles:[{role:'geometry'},{role:'mechanics'},{role:'semantics'}],hosted_errors:[{role:'evidence',code:'transport_failed_charge_uncertain'}]};
 const result=A.hostedReviewProgress(review);assert.equal(result.label,'3/4 Muse roles completed · 1 unavailable');assert.equal(result.warning,true);assert.equal(result.completed,3);assert.deepEqual(result.errors,[{role:'evidence',code:'transport failed charge uncertain'}]);
 assert.equal(A.hostedReviewProgress({...review,hosted_roles_completed:4}).completed,3,'An aggregate count cannot invent a missing actual report.');
 assert.equal(A.hostedReviewProgress({hosted_roles:[],hosted_errors:[],hosted_review_enabled:false}).label,'Hosted reviews not requested');
 assert.equal(A.hostedReviewProgress({hosted_roles:[{},{},{},{}],hosted_roles_completed:4,hosted_roles_expected:4,hosted_review_enabled:true}).warning,false);
});

test('optional task collection hints are typed memory context, with automatic default null',()=>{
 const path='data/uploads/example.mp4';assert.deepEqual(A.newCandidatePayload(path),{input_path:path,processing_consent:true,object_family:null});
 assert.deepEqual(A.newCandidatePayload(path,'hinged_assemblies'),{input_path:path,processing_consent:true,object_family:'hinged_assemblies'});assert.throws(()=>A.newCandidatePayload(path,'made_up_geometry'),/supported optional task collection/);
});

test('budget displays the real snapshot remaining_estimated_usd key without claiming provider credit balance',()=>{
 assert.equal(A.budgetRemaining({remaining_estimated_usd:89.125}),89.125);assert.equal(A.budgetRemaining({remaining_estimated_usd:0}),0);assert.equal(A.budgetRemaining({remaining_estimate_usd:'88.25'}),88.25);
 for(const value of[null,undefined,NaN,Infinity,-1,''])assert.equal(A.budgetRemaining({remaining_estimated_usd:value}),null);
});


test('Nigel can acknowledge hosted findings for the exact research draft; structural failures cannot be overridden',()=>{
 const review=safeReview();review.blocking_failures=1;review.hosted_roles=[{role:'mechanics',verdict:'blocked',issues:[{severity:'blocking',message:'Physical seating is unverified.'}]}];
 const current={id:'revision_'+'a'.repeat(32),fingerprint:'c'.repeat(64),manifest:{review}},detail={candidate:{current_revision_id:current.id},revisions:[current]},options={overrideReason:'I inspected the visible draft and will share its limitations as research.',overrideAcknowledged:true};
 assert.equal(A.approvalReviewState(review).canOverride,true);assert.equal(A.approvalReviewState(review).canApprove,false);
 assert.throws(()=>A.approvalPayload(detail),/acknowledge/);assert.throws(()=>A.approvalPayload(detail,{...options,overrideAcknowledged:false}),/acknowledge/);assert.throws(()=>A.approvalPayload(detail,{...options,overrideReason:'  '}),/acknowledge/);
 assert.deepEqual(A.approvalPayload(detail,options),{revision_id:current.id,expected_fingerprint:current.fingerprint,review_override_reason:options.overrideReason});
 for(const broken of[{...review,all_required_roles_completed:false},{...review,roles:review.roles.map(role=>role.role==='evidence'?{...role,status:'fail'}:role)},{...review,blocking_failures:2}]){const blocked={...detail,revisions:[{...current,manifest:{review:broken}}]};assert.equal(A.approvalReviewState(broken).canOverride,false);assert.throws(()=>A.approvalPayload(blocked,options),/cannot be overridden/);}
 assert.equal(A.candidateGroup('queued_review'),'active');
});


test('legacy V1/V2 packets get one quiet notice; V3 does not and saved findings/counts remain unchanged',()=>{
 const element=()=>({children:[],ownerDocument:{createElement:()=>({textContent:'',className:''})},append(node){this.children.push(node);}});
 for(const protocol of['text_roles_v1_1000_total','text_roles_v2_8000_total','text_roles_v1','text_roles_v2']){
  const review={blocking_failures:3,hosted_roles:[{role:'semantics',review_protocol:protocol,summary:'Variables are empty.',issues:[{severity:'blocking',message:'No variables.'}]},{role:'mechanics',review_protocol:protocol}]},before=structuredClone(review),box=element();
  assert.equal(A.renderHostedLegacyNotice(box,review),true);assert.equal(box.children.length,1);assert.match(box.children[0].textContent,/compiled variable definitions were omitted/);assert.match(box.children[0].textContent,/downloaded object program/);assert.match(box.children[0].className,/small-note/);assert.deepEqual(review,before);
 }
 for(const review of[{}, {hosted_roles:[]}, {hosted_roles:[{review_protocol:'text_roles_v3_compiled_states_8000_total'}]}, {hosted_roles:[{summary:'Variables are empty.'}]}, {hosted_roles:[{review_protocol:'text_roles_v12_other'}]}]){const box=element();assert.equal(A.renderHostedLegacyNotice(box,review),false);assert.equal(box.children.length,0);}
 assert.ok(A.legacyHostedPacketNotice({hosted_roles:[{review_protocol:'text_roles_v3_compiled_states_8000_total'},{review_protocol:'text_roles_v2_8000_total'}]}));
 const source=require('node:fs').readFileSync(require('node:path').resolve(__dirname,'../admin/admin.js'),'utf8');assert.match(source,/renderHostedLegacyNotice\(hosted,review\)/,'The actual owner hosted-review section renders the protocol-specific notice.');
});
