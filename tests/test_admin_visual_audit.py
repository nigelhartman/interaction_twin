"""Fixed visual adapter admission and development-only measured experiment gates."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import admin_visual_audit as V
import admin_machine_context as C
from admin_memory_context import contextual_client
from agent_memory import FeedbackMemory
from object_program import ref

class VisualAuditPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
    def fixtures(self):
        samples=[{'role':'development','source_index':index,'component_id':part,'status':'diagnostic_available','metrics':{'mask_iou':.4}}
                 for index in [1,2,3]for part in ['base','lid']]
        samples+=[{'role':'final_test','source_index':4,'component_id':'base','status':'diagnostic_available','metrics':{'mask_iou':1.}}]
        audit={'source':{'sha256':'a'*64},'samples':samples}
        scene={'components':[{'id':p,'asset':{'sha256':p}}for p in ['base','lid']]}
        program={'tracklets':[{'id':p}for p in ['base','lid']], 'assets':[{'track_id':p,'sha256':p}for p in ['base','lid']]}
        critic={'repair_proposals':[{'target':'mechanics'}]}
        return audit,scene,program,critic
    def test_activation_is_absent_by_default_and_code_changes_require_new_root_review(self):
        self.assertIsNone(V.activation(root=self.root))
        for name in V.CODE:
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('exact fixed code')
        path=self.root/V.ACTIVATION;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(V.activation_template(root=self.root)))
        self.assertEqual(V.activation(root=self.root),ref(path,self.root))
        (self.root/V.CODE[0]).write_text('changed code')
        with self.assertRaisesRegex(ValueError,'changed'):V.activation(root=self.root)
    def test_disabled_adapter_never_builds_a_packet_or_contacts_a_model(self):
        with patch('native_scene_audit.build_audit',side_effect=AssertionError('No packet before activation')):
            result=V.build_reviewer({},root=self.root)({})
        self.assertEqual(result['status'],'unavailable')
    def test_exact_state_dispatch_intent_and_uncertain_plan_survive_code_output_changes(self):
        manifest={'interaction_ref':{'path':'data/p.json','sha256':'a'*64,'bytes':3},'scene_ref':{'path':'data/s.json','sha256':'b'*64,'bytes':4}}
        V.claim_dispatch(manifest,{'path':'.private/plan.json','sha256':'c'*64,'bytes':5},root=self.root)
        with self.assertRaisesRegex(ValueError,'already claimed'):
            V.claim_dispatch(manifest,{'path':'.private/different-version.json','sha256':'d'*64,'bytes':5},root=self.root)
        folder=self.root/'.private/native-visual-reviews/old-output';(folder/'attempt').mkdir(parents=True)
        (folder/'plan.json').write_text(json.dumps({'role':'critic','context':manifest,'attempt_namespace':'uncertain'}))
        found,unknown=V.prior_critic(manifest,root=self.root);self.assertIsNone(found);self.assertTrue(unknown)
    def test_final_test_scores_do_not_change_any_development_assessment(self):
        args=self.fixtures();first=V.local_assessment(*args)
        args[0]['samples'][-1]['metrics']['mask_iou']=0.;args[0]['samples'][-1]['status']='insufficient_source_observation'
        self.assertEqual(V.local_assessment(*args),first)
        self.assertAlmostEqual(first['metrics']['alignment_error'],.6);self.assertEqual(first['metrics']['coverage'],1.)
        self.assertTrue(first['automatic_repair_safe']);self.assertFalse(first['final_test_evidence_used'])
    def test_no_muse_score_can_override_missing_or_ambiguous_local_measurements(self):
        audit,scene,program,critic=self.fixtures();critic['confidence']=1.
        for row in audit['samples']:
            if row['role']=='development':row['status']='insufficient_source_observation'
        self.assertIsNone(V.local_assessment(audit,scene,program,critic))
    def test_compiled_asset_omission_requests_compile_reasoning_instead_of_purchasing_geometry(self):
        audit,scene,program,critic=self.fixtures();program['assets']=program['assets'][:1]
        result=V.local_assessment(audit,scene,program,critic)
        self.assertEqual(result['repair_targets'],['reasoning']);self.assertEqual(result['metrics']['blocking_issues'],1)
        self.assertTrue(result['automatic_repair_safe'])
    def test_missing_scene_component_reduces_coverage_and_uses_reconstruction_adapter(self):
        audit,scene,program,critic=self.fixtures();program['tracklets'].append({'id':'earbud'})
        result=V.local_assessment(audit,scene,program,critic)
        self.assertEqual(result['repair_targets'],['reconstruction']);self.assertEqual(result['metrics']['missing_components'],1)
        self.assertAlmostEqual(result['metrics']['coverage'],2/3)
    def measured(self):
        body={'artifact_binding':{'source_ref':{'sha256':'a'*64}},'issues':[{'message':'fallible machine issue'}],
            'repair_proposals':[{'target':'mechanics'}],'improvement_assessment':{'metrics':{'alignment_error':.6}}}
        path=self.root/'.private/measured/review.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(body));path.chmod(0o600)
        return ref(path,self.root)
    def test_machine_context_is_private_source_bound_and_separate_from_human_memory(self):
        reference=C.snapshot(self.measured(),source_sha256='a'*64,root=self.root)
        context=C.load(reference,source_sha256='a'*64,root=self.root)
        self.assertFalse(context['human_feedback_authored']);self.assertEqual(FeedbackMemory(self.root).entries(),[])
        with self.assertRaisesRegex(ValueError,'source'):C.load(reference,source_sha256='b'*64,root=self.root)
        path=self.root/reference['path'];path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'changed'):C.load(reference,source_sha256='a'*64,root=self.root)
    def test_machine_context_reaches_effective_prompt_with_own_provenance_not_a_human_lesson(self):
        machine=C.snapshot(self.measured(),source_sha256='a'*64,root=self.root)
        context=FeedbackMemory(self.root).retrieve(source_sha256='a'*64)
        path=self.root/'.private/context.json';path.write_text(json.dumps(context));path.chmod(0o600)
        reference=dict(ref(path,self.root),context_sha256=context['context_sha256'])
        captured=[]
        class Offline:
            def json_result(inner,prompt,*args,**kwargs):captured.append(prompt);return {'ok':True},{'estimated_cost_usd':'0'}
        client=contextual_client(reference,source_sha256='a'*64,script='stage5_meta_propose.py',root=self.root,base_class=Offline,machine_review_ref=machine)()
        out=self.root/'data/result';out.mkdir(parents=True)
        _,receipt=client.json_result('original prompt',{},[],output=out)
        self.assertIn('MACHINE REVIEW GUIDANCE',captured[0]);self.assertNotIn('OWNER FEEDBACK CONTEXT',captured[0])
        provenance=receipt['owner_annotation_context'];self.assertEqual(provenance['annotation_count'],0);self.assertEqual(provenance['machine_review_ref'],machine)
        self.assertEqual(FeedbackMemory(self.root).entries(),[])
    def test_nested_overnight_admin_argv_injection_precedes_script_and_preserves_fence(self):
        from admin_pipeline import AdminRunner
        import pipeline
        runner=AdminRunner.__new__(AdminRunner);runner.execution_lease=Mock();runner.boundary=None;runner.machine_review_ref={'path':'private exact data'}
        args=['python','run_overnight_stage.py','--task-budget-plan','fixed-plan','run_admin_stage.py','--memory-context-ref','{}','--source-sha256','a'*64,'stage5_meta_propose.py','--case','fixed-case']
        with patch.object(pipeline.Runner,'command')as command:runner.command(args,30)
        actual=command.call_args.args[0];index=actual.index('run_admin_stage.py')
        self.assertEqual(actual[index+1],'--machine-review-ref');self.assertLess(actual.index('--machine-review-ref'),actual.index('stage5_meta_propose.py'))
        self.assertEqual(actual[-2:],['--case','fixed-case']);self.assertEqual(runner.execution_lease.guard.call_count,2)

    def test_real_cpu_audit_prepare_critic_measurement_bridge_reuses_one_offline_send(self):
        import shutil
        from PIL import Image
        from tests import test_native_scene_audit as audit_fixtures
        from tests import test_native_visual_review as visual_fixtures
        import native_visual_review as visual
        from admin_execution_policy import improvement_policy,review_assessment,TASK_PLAN
        harness=audit_fixtures.AuditArtifactTests('test_end_to_end_audit_freezes_real_triangle_rasters_code_and_original_bytes');harness.setUp()
        self.addCleanup(harness.doCleanups);root=harness.root
        for name in V.CODE:
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(V.ROOT/name,path)
        budget=root/TASK_PLAN;budget.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(V.ROOT/TASK_PLAN,budget)
        source=harness.folder/'video.mp4';source.write_bytes(b'local fixture source; saved frames used for raster tests')
        source_ref=ref(source,root);program=json.loads(harness.program.read_text())
        program['source'].update(path=source_ref['path'],sha256=source_ref['sha256'],original_sha256=source_ref['sha256'])
        for evidence in program['evidence']:
            rgb=root/evidence['frame_path']
            with Image.open(rgb)as im:im.resize((32,32)).save(rgb)
            evidence['frame_sha256']=hashlib.sha256(rgb.read_bytes()).hexdigest()
            for obs in evidence['observations']:
                mask=root/obs['mask_path']
                with Image.open(mask)as im:im.resize((32,32),Image.Resampling.NEAREST).save(mask)
                obs['mask_sha256']=hashlib.sha256(mask.read_bytes()).hexdigest()
        harness.program.write_text(json.dumps(program));scene=json.loads(harness.scene.read_text())
        scene['source'].update(sha256=source_ref['sha256'],height=32,width=32)
        scene['evaluation_protocol']['source_sha256']=source_ref['sha256'];scene['original_program']=ref(harness.program,root)
        harness.scene.write_text(json.dumps(scene))
        active=root/V.ACTIVATION;active.write_text(json.dumps(V.activation_template(root=root)))
        manifest=dict(source_ref=source_ref,interaction_ref=ref(harness.program,root),scene_ref=ref(harness.scene,root),object_program_ref=None)
        lease=Mock();lease.guard=Mock(side_effect=lambda stage:None)
        job={'candidate_id':'candidate_'+'a'*32,'source_ref':source_ref,'improvement_policy':improvement_policy(root),'improvement_round':0,'execution_lease':lease};client=visual_fixtures.OfflineClient(root)
        original=visual.build_admin_reviewer
        with patch.object(visual,'build_admin_reviewer',side_effect=lambda *args,**kwargs:original(*args,**kwargs,client=client)):
            result=V.build_reviewer(job,root=root)(manifest)
            second=V.build_reviewer(job,root=root)(manifest)
            implementation=root/V.CODE[0];implementation.write_bytes(implementation.read_bytes()+b'\n# fixture reviewed next implementation\n')
            active.write_text(json.dumps(V.activation_template(root=root)))
            third=V.build_reviewer(job,root=root)(manifest)
        self.assertEqual(result['status'],'complete');self.assertEqual(len(client.calls),1)
        self.assertEqual(result['report_ref'],second['report_ref'])
        self.assertTrue(third['raw_critic_reused'])
        measured=review_assessment(result,manifest,root=root)
        self.assertIsNotNone(measured);self.assertFalse(measured['automatic_repair_safe'])
        third_measured=review_assessment(third,manifest,root=root)
        self.assertEqual(measured['evaluation_plan_ref'],third_measured['evaluation_plan_ref'])
        self.assertEqual(measured['comparison_key'],third_measured['comparison_key'])
        self.assertFalse(any(path.name=='2.png'or path.name.startswith('000002-')for path in client.calls[0]['images']))
        self.assertIn('native_visual_request',[call.args[0]for call in lease.guard.call_args_list])
        guidance=C.snapshot(result['report_ref'],source_sha256=source_ref['sha256'],root=root)
        context=C.load(guidance,source_sha256=source_ref['sha256'],root=root)
        self.assertEqual([r['source_index']for r in context['typed_citations']],[1,3])
        self.assertEqual([r['timestamp_s']for r in context['typed_citations']],[0.,.6])
        changed_program=copy.deepcopy(program);changed_program['tracklets'].append({'id':'new_part'})
        new_program=harness.folder/'new-program.json';new_program.write_text(json.dumps(changed_program))
        changed_scene=copy.deepcopy(scene);changed_scene['original_program']=ref(new_program,root)
        new_scene=harness.folder/'new-scene.json';new_scene.write_text(json.dumps(changed_scene))
        later_job=dict(job,improvement_round=1,evaluation_plan_ref=measured['evaluation_plan_ref'])
        later_manifest=dict(manifest,interaction_ref=ref(new_program,root),scene_ref=ref(new_scene,root))
        blocked=V.build_reviewer(later_job,root=root)(later_manifest)
        self.assertEqual(blocked['status'],'unavailable');self.assertIn('Frozen evaluation basis changed',blocked['diagnostic'])
        self.assertEqual(len(client.calls),1)

if __name__=='__main__':unittest.main()
