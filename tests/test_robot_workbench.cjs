/* Actual curated planner → shared browser controller. No model calls or renderer/pixel claims. */
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process'),{createHash,webcrypto}=require('node:crypto');
const C=require('../showcase/robot-controller.js'),P=require('../showcase/robot-task-plan.js'),A=require('../showcase/aligned-scene-contract.js'),ROOT=path.resolve(__dirname,'..');
const fixture=JSON.parse(cp.execFileSync(path.join(ROOT,'.venv-sam/bin/python'),['-c',`import json
from pathlib import Path
from unittest.mock import patch
from robot_task_program import get_robot_kit,plan_robot_task
from robot_task_program import _context
r=next(r for r in json.loads(Path('showcase/public/library-catalog.json').read_text())['records']if {'case_base','case_lid','earbud_a','earbud_b'} <= {c['id']for c in r.get('components',[])})
texts=['Place the case on the table, open the lid, take out both earbuds and place them on the tray, then close it.','Open the lid and remove earbud_a to the tray.','Return earbud_a from the tray to the open case, then close it.','Place the case and open the lid.']
records=json.loads(Path('showcase/public/library-catalog.json').read_text())['records']
wallet=next(v for v in records if any(c['id']=='gold_coin'for c in v.get('components',[])))
camera=next(v for v in records if {'camera_body','battery_door','batteries'} <= {c['id']for c in v.get('components',[])})
camera_goal={'operation':'pick_place','component_ids':['batteries'],'open_container':False,'close_container':False,'destination':'tray'}
camera_plan=plan_robot_task(camera['id'],goal=camera_goal)
laptop=next(v for v in records if any(c['id']=='laptop_lid'for c in v.get('components',[])))
phone=next(v for v in records if any(c['id']=='phone_blue'for c in v.get('components',[])))
laptop_plan=plan_robot_task(laptop['id'],'Open the laptop_lid, then close it.')
phone_plan=plan_robot_task(phone['id'],'Move cable_connector to the tray.')
# Explicit isolated missing-geometry fixture, after exact real source/native
# validation. No file, public catalog or production planner is changed.
missing_context={**_context(camera['id'],root=Path.cwd(),catalog=None),'assets':{}}
with patch('robot_task_program._context',return_value=missing_context):
    missing_geometry=plan_robot_task(camera['id'],goal=camera_goal)
print(json.dumps({'kit':get_robot_kit(),'plans':[plan_robot_task(r['id'],t)for t in texts],'texts':texts,'wallet':plan_robot_task(wallet['id'],'Move gold_coin to the tray.'),'camera':camera_plan,'laptop':laptop_plan,'phone':phone_plan,'missing_geometry':missing_geometry}))`],{cwd:ROOT,encoding:'utf8',maxBuffer:2*1024*1024}));
const baseline=fixture.plans[0],clone=v=>JSON.parse(JSON.stringify(v));
const declaredDerivative=baseline.objects.some(o=>o.asset_scope==='curated_geometric_derivative');
const expectedActions=declaredDerivative?40:38;
function lastReleaseTarget(plan,id){const at=plan.actions.findIndex(a=>a.operation==='detach'&&a.object_id===id);assert.ok(at>=0);let move=at-1;while(move>=0&&plan.actions[move].operation!=='move_to')move--;assert.ok(move>=0);return plan.actions[move].position.slice();}
function compile(plan=baseline){const p=P.validate(plan,baseline,fixture.kit),t=C.compile(P.translate(p,fixture.kit),{joint_tool_position:(step,q,state)=>P.jointToolPosition(p,step,q,state)});P.checkOutcome(p,t);return t;}
function close(a,b,e=1e-8){assert.ok(C.distance(a,b)<e,JSON.stringify({a,b}));}

test('actual public Python kit and selected native/declared task execute in the same browser kinematics',()=>{
 assert.deepEqual(fixture.kit.robot,C.DEFAULT_ROBOT);assert.equal(baseline.status,'ready');assert.equal(baseline.objects.length,4);assert.equal(baseline.actions.length,expectedActions);
 const t=compile(),end=C.at(t,t.duration_s);assert.equal(t.phases.length,expectedActions);assert.ok(Math.abs(t.duration_s-(declaredDerivative?25.9:23.5))<1e-8);assert.equal(end.attached,null);const baseTarget=lastReleaseTarget(baseline,'case_base');baseTarget[1]-=.11;close(end.objects.case_base.position,baseTarget);close(end.objects.earbud_a.position,lastReleaseTarget(baseline,'earbud_a'));close(end.objects.earbud_b.position,lastReleaseTarget(baseline,'earbud_b'));if(!declaredDerivative){close(end.objects.earbud_a.position,[-.42,.08,.35]);close(end.objects.earbud_b.position,[-.28,.08,.35]);}assert.equal(Object.values(end.object_joints)[0],0);
 for(const o of baseline.objects){const bytes=fs.readFileSync(path.join(ROOT,o.asset_ref.path));assert.equal(bytes.length,o.asset_ref.bytes);assert.equal(createHash('sha256').update(bytes).digest('hex'),o.asset_ref.sha256);}
 for(let i=0;i<=200;i++){const s=C.at(t,t.duration_s*i/200);assert.equal(s.ik.reachable,true);assert.equal(s.ik.within_limits,true);assert.ok(s.ik.residual<.002);assert.ok(Math.abs(s.ik.tool_pitch+Math.PI/2)<1e-8);}
});

test('seeking is deterministic across reverse order, reset and pause boundaries; carried descendants follow the case',()=>{
 const t=compile(),times=[0,3,5.1,7.5,11.3,14,18.6,t.duration_s],expected=times.map(n=>C.at(t,n));for(const n of times.slice().reverse())assert.deepEqual(C.at(t,n),expected[times.indexOf(n)]);assert.deepEqual(C.at(t,0),expected[0]);
 const transfer=t.phases.find(p=>p.step.title==='Transfer case_base'),s=C.at(t,(transfer.start_s+transfer.end_s)/2),delta=s.objects.case_base.position.map((v,i)=>v-t.initial.objects.case_base.position[i]);for(const id of['earbud_a','earbud_b','case_lid'])close(s.objects[id].position,t.initial.objects[id].position.map((v,i)=>v+delta[i]));assert.equal(s.attached,'case_base');close(s.objects.case_base.position,s.tool.map((v,i)=>v+s.attachment_offset[i]));
 const released=t.phases.find(p=>p.step.title==='Retreat from earbud_a'),before=C.at(t,released.start_s),after=C.at(t,released.end_s);close(before.objects.earbud_a.position,after.objects.earbud_a.position);assert.notDeepEqual(before.tool,after.tool);
});

test('single removal, separate return and open-only are genuinely different supported programs',()=>{
 for(const plan of fixture.plans.slice(1)){const t=compile(plan);assert.ok(t.phases.length<expectedActions);assert.equal(t.final.attached,null);}assert.equal(fixture.plans[2].goal.operation,'return_components');assert.equal(fixture.plans[2].attachments.length,0);close(compile(fixture.plans[2]).final.objects.earbud_a.position,lastReleaseTarget(fixture.plans[2],'earbud_a'));if(!declaredDerivative)close(compile(fixture.plans[2]).final.objects.earbud_a.position,[-.22,.22,.05]);
 for(const [i,text]of fixture.texts.entries()){const p=P.compileInstruction(text,baseline,fixture.kit),t=compile(p);assert.equal(p.goal.operation,fixture.plans[i].goal.operation);const substantive=actions=>actions.filter(a=>a.operation!=='wait'&&a.title!=='Withdraw from the completed task').length;assert.equal(substantive(p.actions),substantive(fixture.plans[i].actions));assert.deepEqual(p.objects.map(o=>o.asset_ref),baseline.objects.map(o=>o.asset_ref));}
 for(const text of['Open the lid and wash the earbuds.','Open the lid and throw it.','Remove earbud_a then return earbud_a.','Connect the phone to a socket.','Open the lid and exec alert()'])assert.throws(()=>P.compileInstruction(text,baseline,fixture.kit));
});

test('invalid source identities, claims, joint frames and executable action fields stop before execution',()=>{
 const edits=[p=>p.objects[0].asset_ref.sha256='0'.repeat(64),p=>p.objects[0].asset_ref.path='data/../.secrets/meta.env',p=>p.source_binding.source_sha256='0'.repeat(64),p=>p.kit_sha256='0'.repeat(64),p=>p.claims.contact_or_seating_verified=true,p=>p.claims.physical_robot_execution_available=true,p=>p.joints[0].source_joint.axis=[0,1,0],p=>p.actions[0].code='alert(1)',p=>p.actions[0].value=1,p=>p.status='blocked',p=>p.missing_requirements=[{message:'Missing geometry'}],p=>p.objects[2].initial_transform.position[0]+=.01,p=>p.actions.find(a=>a.operation==='move_to').position[0]=NaN];
 for(const edit of edits){const p=clone(baseline);edit(p);assert.throws(()=>compile(p));}assert.equal(baseline.claims.physical_robot_execution_available,false);assert.equal(baseline.actions[0].title,declaredDerivative?'Closed reference · ready to begin':'Approach case_base');
});

test('edited tasks cannot pick through a closed lid, release without motion, open a held gripper or carry a hinged child',()=>{
 const mutations=[p=>p.actions=p.actions.filter(a=>a.operation!=='set_joint'),p=>p.actions.splice(5,0,{id:'bad_open',title:'Open while carrying',operation:'gripper',value:0,duration_s:.2}),p=>{p.actions=p.actions.filter((a,i)=>!(i>=5&&i<=7));},p=>{p.actions[4].object_id='case_lid';},p=>p.actions[0].position=[2,.5,2],p=>p.actions=p.actions.filter(a=>a.operation!=='detach')];for(const edit of mutations){const p=clone(baseline);edit(p);assert.throws(()=>compile(p));}
 assert.equal(C.solve([10,0,0]).reachable,false);assert.equal(C.solve([10,0,0]).within_limits,true,'An unreachable diagnostic retains bounded robot angles, never success.');
});

test('source program and descriptive joint references bind exact component identities and source matrices',()=>{
 const program=JSON.parse(fs.readFileSync(path.join(ROOT,baseline.source_binding.program_ref.path))),objectProgram=JSON.parse(fs.readFileSync(path.join(ROOT,baseline.source_binding.object_program_ref.path)));assert.equal(P.sourceBinding(baseline,program,objectProgram),true);const wrong=clone(program);wrong.source.sha256='0'.repeat(64);assert.throws(()=>P.sourceBinding(baseline,wrong,objectProgram));const bad=clone(objectProgram);bad.joints[0].axis=[0,1,0];assert.throws(()=>P.sourceBinding(baseline,program,bad));
 const j=baseline.joints[0].source_joint;for(const q of[0,.2,1,1.8,2.8]){const chain=A.multiply(A.multiply(j.parent_from_joint,A.axisRotation(j.axis,q)),j.joint_from_child_zero);A.similarity(chain);assert.ok(chain.flat().every(Number.isFinite));}const runtime=fs.readFileSync(path.join(ROOT,'showcase/robot-workbench.js'),'utf8');assert.match(runtime,/A\.multiply\(A\.multiply\(j\.source_joint\.parent_from_joint,A\.axisRotation\(j\.source_joint\.axis,q\)\),j\.source_joint\.joint_from_child_zero\)/);assert.doesNotMatch(runtime,/eval\(|new Function|\/api\/admin|\.private|\.secrets/);
});

test('current generic derivative binds its exact reviewed geometry and normalizes only the declared compact joint convention',()=>{
 const program=JSON.parse(fs.readFileSync(path.join(ROOT,baseline.source_binding.program_ref.path))),objectProgram=JSON.parse(fs.readFileSync(path.join(ROOT,baseline.source_binding.object_program_ref.path)));
 assert.equal(objectProgram.format,'interaction-twin-declared-geometry-derivative');assert.equal(baseline.actions.length,40);assert.equal(P.sourceBinding(baseline,program,objectProgram),true);
 const changed=baseline.objects.find(o=>o.asset_scope==='curated_geometric_derivative');assert(changed);const component=d=>d.component_assets.find(c=>c.component_id===changed.component_id);
 for(const mutate of[d=>d.claims.declared_geometry_derivative_used=false,d=>d.claims.manual_geometry_or_poses_used=true,d=>d.source_audit_summary.default_eligible=false,d=>d.source_audit_summary.geometric_gates_passed=false,d=>d.source_audit_summary.source_pixel_gates_passed=false,d=>d.physical_accuracy_verified=true,d=>d.evidence_program.sha256='0'.repeat(64),d=>component(d).native_glb.sha256='0'.repeat(64),d=>component(d).geometry_origin='manual_unbound_mesh',d=>component(d).original_native_glb=clone(component(d).native_glb),d=>d.component_assets.push(clone(d.component_assets[0])),d=>d.baseline_scene_ref.path='data/../private.json',d=>d.joints[0].matrix_convention='axis_in_camera_frame',d=>d.joints[0].axis_frame='camera_frame']){const changedDoc=clone(objectProgram);mutate(changedDoc);assert.throws(()=>P.sourceBinding(baseline,program,changedDoc));}
 const native=JSON.parse(fs.readFileSync(path.join(ROOT,objectProgram.base_object_program_ref.path)));assert.throws(()=>P.sourceBinding(baseline,program,native),'The old native object program cannot silently replace the approved derivative.');
 const badPlan=clone(baseline);badPlan.joints[0].source_joint.axis_frame='camera_frame';assert.throws(()=>P.sourceBinding(badPlan,program,objectProgram));
});

test('artifact transport checks exact full bytes/hash and local refs before any renderer parse',async()=>{
 const bytes=Buffer.from('complete native payload'),reference={path:'data/public/asset.glb',bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')},calls=[];
 const transport=async(url,options)=>{calls.push({url,options});return{ok:true,arrayBuffer:async()=>Uint8Array.from(bytes).buffer}};assert.equal((await P.verifiedArtifact(reference,transport,webcrypto.subtle)).byteLength,bytes.length);assert.equal(calls[0].url,'/files/data/public/asset.glb');assert.equal(calls[0].options.credentials,'omit');
 await assert.rejects(P.verifiedArtifact({...reference,bytes:bytes.length+1},transport,webcrypto.subtle),/length/);await assert.rejects(P.verifiedArtifact({...reference,sha256:'0'.repeat(64)},transport,webcrypto.subtle),/hash/);const count=calls.length;await assert.rejects(P.verifiedArtifact({...reference,path:'https://evil.invalid/model.glb'},transport,webcrypto.subtle));assert.equal(calls.length,count);
});

test('workbench page has usable controls, typed editor and standalone mode with no model route',()=>{
 const html=fs.readFileSync(path.join(ROOT,'showcase/robot-workbench.html'),'utf8'),js=fs.readFileSync(path.join(ROOT,'showcase/robot-workbench.js'),'utf8'),lab=fs.readFileSync(path.join(ROOT,'showcase/robotics-lab.html'),'utf8');for(const id of['task-prompt','run-task','task-time','play-task','task-json','apply-plan','download-plan','download-kit','joint-inspector'])assert.ok(html.includes('id="'+id+'"'));assert.match(lab,/mode-workbench/);assert.match(js,/P\.compileInstruction/);assert.match(js,/P\.sourceBinding/);assert.match(js,/P\.checkOutcome/);assert.match(js,/requestEpoch/);assert.match(js,/physics|physical simulation|No inference/);assert.doesNotMatch(js,/\/api\/(?:run|chat|upload|admin)|Authorization|api_key/);
});


test('typed native lid actuation couples the bounded robot tool to its source hinge arc without detaching',()=>{
 const t=compile(),phase=t.phases.find(p=>p.step.op==='joint');assert.equal(phase.step.actuation.kind,'authored_native_bbox_surface_cue_v1');const begin=C.at(t,phase.start_s),middle=C.at(t,phase.start_s+(phase.end_s-phase.start_s)*.5),end=C.at(t,phase.end_s);assert.equal(middle.attached,null);assert.equal(middle.objects.case_lid.parent_id,'case_base');assert.equal(middle.gripper,1);assert.ok(middle.object_joints['case_base:'+phase.step.joint_id]>0);assert.ok(C.distance(middle.tool,begin.tool)>.02);close(end.tool,phase.before.tool);assert.equal(end.gripper,0);
 for(let i=0;i<=120;i++){const s=C.at(t,phase.start_s+(phase.end_s-phase.start_s)*i/120);assert.equal(s.ik.reachable,true);assert.ok(s.ik.residual<.002);if(i/120>=.25&&i/120<=.8)close(s.tool,P.jointToolPosition(baseline,phase.step,s.object_joints['case_base:'+phase.step.joint_id],s));}
 for(const edit of[p=>p.actions.find(a=>a.operation==='set_joint').actuation.anchor_native[0]+=.01,p=>delete p.actions.find(a=>a.operation==='set_joint').actuation]){const p=clone(baseline);edit(p);assert.throws(()=>compile(p));}
 assert.throws(()=>C.compile(P.translate(baseline,fixture.kit),{joint_tool_position:()=>[3,3,3]}),/joint cue exceeds/);
});


test('wallet and camera use their own exact current native assets; an isolated missing-model fixture remains blocked',()=>{
 const p=P.validate(fixture.wallet,fixture.wallet,fixture.kit),t=C.compile(P.translate(p,fixture.kit));P.checkOutcome(p,t);assert.equal(p.objects.length,2);assert.equal(p.actions.length,12);assert.equal(p.joints.length,0);assert.equal(t.final.attached,null);close(t.final.objects.gold_coin.position,[-.42,.08,.35]);assert.ok(p.objects.every(o=>!o.id.includes('earbud')));for(const o of p.objects){const bytes=fs.readFileSync(path.join(ROOT,o.asset_ref.path));assert.equal(createHash('sha256').update(bytes).digest('hex'),o.asset_ref.sha256);}
 const alternative=P.compileInstruction('Move gold_coin to the table.',fixture.wallet,fixture.kit),a=C.compile(P.translate(alternative,fixture.kit));P.checkOutcome(alternative,a);assert.notDeepEqual(a.final.objects.gold_coin.position,t.final.objects.gold_coin.position);assert.equal(fixture.camera.status,'ready');assert.ok(fixture.camera.objects.some(o=>o.component_id==='batteries'));for(const o of fixture.camera.objects){const bytes=fs.readFileSync(path.join(ROOT,o.asset_ref.path));assert.equal(createHash('sha256').update(bytes).digest('hex'),o.asset_ref.sha256);}assert.equal(fixture.missing_geometry.status,'blocked');assert.equal(fixture.missing_geometry.objects.length,0);assert.ok(fixture.missing_geometry.missing_requirements.some(i=>i.code==='native_geometry_missing'));assert.throws(()=>P.validate(fixture.missing_geometry,fixture.missing_geometry,fixture.kit),/blocked/);
});


test('generic source-label vocabulary is accepted without inventing cabinet geometry or ignoring unknown clauses',()=>{
 const cabinet=clone(baseline);cabinet.objects.find(o=>o.role==='assembly_base').label='cabinet body';cabinet.objects.find(o=>o.role==='articulated_child').label='cabinet door';const p=P.compileInstruction('Open the cabinet door.',cabinet,fixture.kit);assert.equal(p.goal.operation,'open_close');assert.deepEqual(p.goal.component_ids,[]);assert.deepEqual(p.objects.map(o=>o.asset_ref),baseline.objects.map(o=>o.asset_ref),'Only synthetic vocabulary metadata changes in this grammar fixture; no new geometry exists.');assert.throws(()=>P.compileInstruction('Open the cabinet door.',baseline,fixture.kit),/vocabulary/);assert.throws(()=>P.compileInstruction('Open the cabinet door then wash it.',cabinet,fixture.kit),/vocabulary/);
 const openOnly=clone(fixture.plans[3]);openOnly.actions=openOnly.actions.filter(a=>a.operation!=='set_joint');assert.throws(()=>compile(openOnly),/opening command/);const falseGoal=clone(baseline);falseGoal.goal={operation:'open_close',component_ids:[],open_container:false,close_container:false,destination:'table'};assert.throws(()=>compile(falseGoal),/joint goal/);
 const alias=P.compileInstruction('Open the charging case and remove both earphones to the tray.',baseline,fixture.kit);assert.equal(alias.goal.component_ids.length,2);compile(alias);
});


test('standalone provenance links the included exact source before asynchronous model activation',()=>{
 const source=fs.readFileSync(path.join(ROOT,'showcase/robot-workbench.js'),'utf8'),helper=source.match(/^function setSourceLink\(initial\)\{[^\n]+\}/m)?.[0];assert.ok(helper);for(const bundle of[true,false]){const link={href:'library.html',textContent:''},context={P,bundle,$:id=>{assert.equal(id,'task-source');return link;},encodeURIComponent};require('node:vm').runInNewContext(helper+';setSourceLink('+JSON.stringify(baseline)+');',context);assert.equal(link.href,bundle?'/files/'+baseline.source_binding.program_ref.path:'library.html?case='+encodeURIComponent(baseline.twin_id));assert.equal(link.textContent,bundle?'Inspect included source program →':'Inspect original twin →');}
 assert.ok(source.indexOf('setSourceLink(initial);initialize();')<source.indexOf('await activate(initial,{reloadModels:true})'));assert.match(source,/\$\('task-source'\)\.removeAttribute\('href'\)/);
});


test('startup geometry failure withdraws broken starter and unsupported prompt actions without touching successful runs',()=>{
 const source=fs.readFileSync(path.join(ROOT,'showcase/robot-workbench.js'),'utf8'),helper=source.match(/^function showStartupFailure\(error\)\{[^\n]+\}/m)?.[0];assert.ok(helper);const nodes=new Map(),get=id=>{if(!nodes.has(id))nodes.set(id,{textContent:id==='scene-title'?'Camera & battery · generated components':'Preparing',value:'Remove both earbuds',disabled:false,hidden:true,href:'/api/robotics/bundle?bad=1',style:{},removeAttribute(k){delete this[k]},setAttribute(k,v){this[k]=v}});return nodes.get(id);},buttons=[{disabled:false},{disabled:false},{disabled:false}],hint={textContent:'earbud examples'},messages=[],context={baseline:fixture.missing_geometry,failed:false,ready:true,playing:true,$:get,document:{querySelectorAll:()=>buttons,querySelector:()=>hint},message:(...args)=>messages.push(args)};require('node:vm').runInNewContext(helper+';showStartupFailure(new Error("Curated native geometry is unavailable."));',context);assert.equal(context.ready,false);assert.equal(context.playing,false);assert.equal(get('download-kit').href,undefined);assert.equal(get('download-kit')['aria-disabled'],'true');assert.equal(get('current-step').textContent,'Geometry pending');assert.equal(get('reach-state').textContent,'Unavailable');assert.equal(get('task-prompt').value,'');assert.equal(get('task-prompt').disabled,true);assert.match(get('task-prompt').placeholder,/geometry is unavailable/);assert.equal(buttons.every(b=>b.disabled),true);for(const id of['run-task','play-task','task-time','apply-plan','download-plan','record-preview'])assert.equal(get(id).disabled,true);assert.equal(get('scene-title').textContent,'Camera & battery · generated components');assert.deepEqual(messages,[['Curated native geometry is unavailable.',true]]);assert.match(source,/catch\(e\)\{showStartupFailure\(e\);\}/);
});

test('source-oriented robot rest keeps exact raw observations and rejects altered scene, masks and claimed physical fit',()=>{
 const d=baseline.staging_reference;assert.ok(d);const scene=JSON.parse(fs.readFileSync(path.join(ROOT,d.scene_ref.path)));assert.equal(P.sourceStaging(baseline,scene),true);assert.deepEqual(d.components.map(r=>r.component_id).sort(),['earbud_a','earbud_b']);for(const row of d.components){const object=baseline.objects.find(o=>o.id===row.component_id);assert.ok(object.display_max_dimension>.095);assert.notDeepEqual(object.initial_transform.quaternion_xyzw,[0,0,0,1]);assert.equal(row.physical_contact_verified,false);}const unknown=clone(scene),selected=d.components[0];unknown.components.find(c=>c.id===selected.component_id).poses.find(p=>p.source_sample_index===selected.source_sample_index).status='unknown';assert.throws(()=>P.sourceStaging(baseline,unknown),/supported/);for(const edit of[p=>p.staging_reference.components[0].parent_from_child[0][3]+=.01,p=>p.staging_reference.components[0].child_source_mask_ref.sha256='0'.repeat(64),p=>p.staging_reference.visual_rest_constraints.physical_contact_verified=true]){const bad=clone(baseline);edit(bad);assert.throws(()=>P.sourceStaging(bad,scene));assert.throws(()=>compile(bad));}const rotated=clone(baseline);rotated.objects.find(o=>o.role==='removable').initial_transform.quaternion_xyzw=[0,0,0,1];assert.throws(()=>compile(rotated),/source-bound relative rotation/);
});


test('selected laptop, camera and phone tasks retain their own exact source chains and supported local actions',()=>{
 const examples=[[fixture.laptop,'Open the laptop_lid, then close it.'],[fixture.camera,'Move batteries to the tray.'],[fixture.phone,'Move cable_connector to the tray.']];
 for(const[plan,instruction]of examples){
  assert.equal(plan.status,'ready');const program=JSON.parse(fs.readFileSync(path.join(ROOT,plan.source_binding.program_ref.path))),objectProgram=JSON.parse(fs.readFileSync(path.join(ROOT,plan.source_binding.object_program_ref.path)));
  assert.equal(P.sourceBinding(plan,program,objectProgram),true);const t=C.compile(P.translate(P.validate(plan,plan,fixture.kit),fixture.kit),{joint_tool_position:(step,q,state)=>P.jointToolPosition(plan,step,q,state)});P.checkOutcome(plan,t);assert.equal(t.final.attached,null);
  const changed=P.compileInstruction(instruction,plan,fixture.kit);assert.equal(changed.goal.operation,plan.goal.operation);assert.deepEqual(changed.objects.map(o=>o.asset_ref),plan.objects.map(o=>o.asset_ref));assert.equal(plan.claims.contact_or_seating_verified,false);
 }
 const laptop=fixture.laptop,program=JSON.parse(fs.readFileSync(path.join(ROOT,laptop.source_binding.program_ref.path))),objectProgram=JSON.parse(fs.readFileSync(path.join(ROOT,laptop.source_binding.object_program_ref.path)));
 assert.equal(objectProgram.joints[0].matrix_convention,'parent_native_pose * parent_from_joint(inverse_parent_scale) * axis_rotation(q) * joint_from_child_zero(child_scale_once)');
 for(const invalid of['parent_native_pose * parent_from_joint(inverse_parent_scale) * axis_rotation(q) * joint_from_child_zero(child_scale_twice)','camera * axis_rotation(q)']){const bad=clone(objectProgram);bad.joints[0].matrix_convention=invalid;assert.throws(()=>P.sourceBinding(laptop,program,bad));}
});

test('reviewed B cap is an explicit minimal visual-rest delta and cannot be silently rebound or called physical',()=>{
 const descriptor=baseline.staging_reference,row=descriptor.components.find(r=>r.component_id==='earbud_b'),scene=JSON.parse(fs.readFileSync(path.join(ROOT,descriptor.scene_ref.path)));
 assert.equal(row.orientation_hypothesis.id,'source_matched_cap_tilt');assert.equal(descriptor.reviewed_visual_rest_profile.semantic_confidence,.78);assert.equal(row.orientation_hypothesis.authored_visual_rest_not_recovered_source_pose,true);assert.equal(P.sourceStaging(baseline,scene),true);
 for(const mutate of[
  r=>r.orientation_hypothesis.delta_rotation_parent[0][0]+=.1,
  r=>r.orientation_hypothesis.target_cap_direction_parent[1]-=.1,
  r=>r.orientation_hypothesis.physical_orientation_verified=true,
  r=>r.orientation_derived_parent_from_child[0][3]+=.02,
  r=>r.source_quaternion_xyzw=[0,0,0,1],
  r=>r.orientation_hypothesis.semantic_review_sha256='0'.repeat(64),
  r=>r.source_visible_rim_protrusion.bounded_visible_height_fraction=.49
 ]){const plan=clone(baseline),target=plan.staging_reference.components.find(r=>r.component_id==='earbud_b');mutate(target);assert.throws(()=>P.sourceStaging(plan,scene));assert.throws(()=>P.validate(plan,baseline,fixture.kit));}
 const changed=clone(baseline);changed.staging_reference.reviewed_visual_rest_profile.components[0].source_sample_index=78;assert.throws(()=>P.sourceStaging(changed,scene));
 assert.deepEqual(descriptor.components.find(r=>r.component_id==='earbud_a').source_sample_index,77);assert.equal(descriptor.visual_rest_constraints.physical_contact_verified,false);assert.equal(descriptor.visual_rest_constraints.actual_cavity_surface_or_lid_clearance_verified,false);
});
