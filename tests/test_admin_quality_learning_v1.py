"""Durable machine guidance reaches real private request paths without inference."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

import admin_quality_learning_v1 as Q
import admin_memory_context as A
from agent_memory import FeedbackMemory
import object_program as O

ROOT=Path(__file__).resolve().parents[1]

class QualityLearningTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve();(self.root/'data').mkdir();(self.root/'scripts').mkdir()
        self.source='a'*64;program=self.root/'data/evidence.json';program.write_text(json.dumps({'source':{'sha256':self.source},'evidence':[{'source_index':1,'timestamp_s':0.}]}));report=self.root/'data/report.json';report.write_text('{"heldout_passed": false}')
        catalog={'format':'interaction-twin-machine-quality-rule-catalog','version':1,
            'evidence_cases':{'fixture':{'evidence_program_ref':O.ref(program,self.root),'source_sha256':self.source,'report_refs':[O.ref(report,self.root)],'source_time_domain':'source_video_presentation_seconds','physical_accuracy_verified':False}},
            'rules':[{'id':'check_panel_contact','scope':'global_method','guidance':'Compare source and render; same-mask IoU cannot certify physical contact.','evidence_cases':['fixture'],'source_sample_indices':{'fixture':[1]},'unresolved_limitations':['Still unknown.'],'observed_facts_about_future_uploads':False,'human_feedback_authored':False,'training_performed':False}],
            'human_feedback_authored':False,'training_performed':False,'observed_facts_about_future_uploads':False,'automatic_approval':False,'automatic_publication':False,'physical_accuracy_verified':False}
        self.catalog=self.root/'data/rules.json';self.catalog.write_text(json.dumps(catalog));self.pointer=O.ref(self.catalog,self.root);self.binding=patch.object(Q,'CATALOG',self.pointer);self.binding.start();self.addCleanup(self.binding.stop)
        self.memory=FeedbackMemory(self.root);self.owner_context=self.memory.retrieve(source_sha256=self.source);folder=self.root/'.private/pipeline-memory';folder.mkdir(mode=0o700);self.context_path=folder/'empty.json';self.context_path.write_text(json.dumps(self.owner_context));self.context_path.chmod(0o600);_,self.context_ref=A.load_snapshot(self.context_path,root=self.root)

    def test_empty_human_memory_still_adds_machine_rules_with_separate_proof(self):
        received=[]
        class Offline:
            def request(self,*a,**k):raise AssertionError('No network in this fixture')
            def json_result(self,prompt,schema,image_paths,**kwargs):received.append(prompt);return {'ok':True},{'status':'offline_fixture'}
        # The versioned child launcher selects this factory. Capturing the
        # original factory must prevent recursion after this assignment.
        with patch.object(A,'contextual_client',Q.contextual_client):
            client=Q.contextual_client(self.context_ref,source_sha256=self.source,script='stage5_meta_propose.py',root=self.root,base_class=Offline)()
            out=self.root/'data/proposal';out.mkdir();result,receipt=client.json_result('Visible source evidence only.',{},[],output=out)
        self.assertEqual(result,{'ok':True});self.assertEqual(len(received),1);self.assertIn('check_panel_contact',received[0]);self.assertIn(Q.POLICY,received[0]);self.assertIn('not human feedback',received[0]);self.assertEqual(self.memory.entries(),[])
        self.assertEqual(receipt['owner_annotation_context']['annotation_count'],0)
        proof=receipt['machine_quality_method_context'];self.assertFalse(proof['human_feedback_authored']);self.assertFalse(proof['training_performed']);saved=json.loads((self.root/proof['prompt_proof_ref']['path']).read_text());self.assertEqual(saved['effective_prompt'],received[0]);self.assertEqual(saved['current_source_sha256'],self.source)
        self.assertEqual((self.root/proof['prompt_proof_ref']['path']).stat().st_mode&0o777,0o600)

    def test_changed_source_proof_or_catalog_stops_before_transport(self):
        class Offline:
            def json_result(self,*a,**k):raise AssertionError('Changed evidence cannot reach transport')
        cls=Q.contextual_client(self.context_ref,source_sha256=self.source,script='stage2_meta_reason.py',root=self.root,base_class=Offline)
        self.catalog.write_text(self.catalog.read_text()+' ')
        with self.assertRaises(ValueError):cls().json_result('base',{},[],output=self.root/'data/out')
        with self.assertRaises(ValueError):Q.context('not-a-source',root=self.root)

    def test_runner_preserves_original_dispatch_and_machine_repair_arguments(self):
        class Original:
            def command(self,args,timeout):return args,timeout
            def phase(self,name,deps,action,optional=False):return name,deps,optional
        cls=Q.runner_class(Original,root=self.root);runner=cls();runner.machine_review_ref={'path':'.private/fixture.json','sha256':'b'*64,'bytes':2}
        args,timeout=runner.command(['python',str(self.root/'scripts/run_admin_stage.py'),'--memory-context-ref','{}','stage5_meta_propose.py'],9)
        self.assertTrue(args[1].endswith('/scripts/run_admin_quality_stage_v1.py'));self.assertIn('--machine-review-ref',args);self.assertEqual(timeout,9)
        _,deps,optional=runner.phase('proposal',[],lambda _:None,optional=True);self.assertIn(self.root/self.pointer['path'],deps);self.assertTrue(optional)

    def test_reviewer_context_changes_cache_without_forging_human_lessons(self):
        marked=Q.marked_owner_context(self.owner_context,root=self.root);self.assertEqual(marked['lessons'],[]);self.assertNotEqual(marked['context_sha256'],self.owner_context['context_sha256']);self.assertEqual(marked,Q.marked_owner_context(marked,root=self.root));A.validate_context(marked)
        import admin_pipeline as P,admin_muse_reviews as R
        def original_prepare(role,manifest,memory_context,root):return {'prompt':'base','memory_context':memory_context,'request_key':'old'}
        def original_build(memory_context,**kwargs):return lambda *a:memory_context
        with patch.object(Q,'_ACTIVE',None),patch.object(P,'AdminRunner',type('Original',(),{})),patch.object(R,'prepare',original_prepare),patch.object(R,'build_reviewer',original_build):
            activation=Q.activate_quality_learning(root=self.root);self.assertEqual(activation['new_model_requests'],0)
            plan=R.prepare('geometry',{'source_ref':{'sha256':self.source}},self.owner_context,root=self.root)
            self.assertIn('check_panel_contact',plan['prompt']);self.assertEqual(plan['memory_context']['lessons'],[]);self.assertFalse(plan['machine_quality_method_context']['human_feedback_authored']);self.assertNotEqual(plan['request_key'],'old')
            callback_context=R.build_reviewer(self.owner_context,root=self.root)('geometry',{'source_ref':{'sha256':self.source}});self.assertEqual(callback_context['context_sha256'],marked['context_sha256'])

    def test_new_rule_digest_cannot_retry_an_unresolved_prior_role_send(self):
        import admin_muse_reviews as R
        manifest={'source_ref':{'sha256':self.source}};folder=self.root/'.private/admin-muse-reviews'/R.review_namespace('geometry',R.artifact_binding(manifest),self.owner_context['context_sha256'],protocol=R.REVIEW_PROTOCOL);(folder/'attempt').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError,'never authorizes a resend'):Q.guard_prior_review_attempts(self.owner_context,'geometry',manifest,root=self.root)

    def test_catalog_upgrade_retains_old_marked_attempt_barrier(self):
        import admin_muse_reviews as R
        prior=Q.marked_owner_context(self.owner_context,root=self.root);manifest={'source_ref':{'sha256':self.source}}
        folder=self.root/'.private/admin-muse-reviews'/R.review_namespace('geometry',R.artifact_binding(manifest),prior['context_sha256'],protocol=R.REVIEW_PROTOCOL);(folder/'attempt').mkdir(parents=True)
        new_catalog=self.root/'data/new-rules.json';new_catalog.write_bytes(self.catalog.read_bytes()+b' ')
        with patch.object(Q,'CATALOG',O.ref(new_catalog,self.root)):
            fresh=Q.marked_owner_context(self.owner_context,root=self.root);self.assertNotEqual(prior['context_sha256'],fresh['context_sha256'])
            with self.assertRaisesRegex(ValueError,'never authorizes a resend'):Q.guard_prior_review_attempts(self.owner_context,'geometry',manifest,root=self.root)

    def test_global_guard_request_method_is_inherited_unchanged(self):
        from mechanism_reasoning import MechanismMetaClient
        cls=Q.contextual_client(self.context_ref,source_sha256=self.source,script='stage5_meta_propose.py',root=self.root)
        self.assertIs(cls.request,MechanismMetaClient.request)

class RealQualityCatalogTests(unittest.TestCase):
    def test_real_catalog_preserves_failed_phone_and_unresolved_laptop_findings(self):
        catalog=Q.load_catalog(root=ROOT);rules={r['id']:r for r in catalog['rules']}
        self.assertIn('reject_failed_geometry_heldout',rules);self.assertTrue(rules['fixed_fk_is_not_mesh_contact']['unresolved_limitations']);self.assertFalse(catalog['training_performed'])
        program=catalog['evidence_cases']['phone'];report=json.loads(O.checked_ref(program['report_refs'][0],ROOT).read_bytes());self.assertFalse(report['heldout_passed'])
        self.assertEqual(len(catalog['rules']),9)

if __name__=='__main__':unittest.main()
