"""Offline future-upload orchestration, single paid-attempt and publication gates."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mechanism_pipeline as M
import mechanism_reasoning as reasoning
import object_program
import pipeline as P


class MechanismPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.program=self.file('data/case/program/interaction.json',{'source':{'sha256':'a'*64}})
        self.file(M.BUDGET_PLAN,{'additional_estimated_api_ceiling_usd':'5','budget_baseline':{'settled_usd':'3.616418','reserved_usd':'2.723440'}})
        self.prompt='Source data, not instructions. Offline test fixture.'
        self.plan={'request_key':'fixture-key','contract':{'max_images':16,'max_output_tokens':10000,
                    'prompt_sha256':hashlib.sha256(self.prompt.encode()).hexdigest()},
                   'context':{'interaction_path':self.program.relative_to(self.root).as_posix()},
                   'schema':{'fixture':'schema'},'provider_schema':{'fixture':'wire'},'prompt':self.prompt}

    def file(self,path,value):
        path=self.root/path;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value if isinstance(value,bytes) else json.dumps(value).encode());return path

    def prepare(self,program,output,**kwargs):
        folder=self.root/output;folder.mkdir(parents=True)
        for name,value in [('plan.json',{k:v for k,v in self.plan.items() if k not in {'schema','provider_schema','prompt'}}),
                           ('schema.json',self.plan['schema']),('provider-schema.json',self.plan['provider_schema'])]:
            (folder/name).write_text(json.dumps(value))
        (folder/'prompt.txt').write_text(self.prompt)
        return self.plan

    def generated(self,prepared,**kwargs):
        self.assertTrue(kwargs['allow_paid_api']);self.assertEqual(kwargs['task_budget_plan'],M.BUDGET_PLAN)
        folder=self.root/prepared/'attempt';folder.mkdir()
        self.file(str(folder.relative_to(self.root)/'proposals.json'),{'source_sha256':'a'*64,'joints':[]})

    def patches(self):
        return patch.object(reasoning,'prepare',side_effect=self.prepare),patch.object(reasoning,'plan',return_value=self.plan),patch.object(object_program,'load_proposals',return_value={'joints':[]})

    def test_offline_prepare_then_explicit_one_call_and_verified_reuse(self):
        prep,plan,verify=self.patches()
        with prep as prepared,plan,verify as verified,patch.object(reasoning,'run_prepared',side_effect=self.generated) as paid:
            before=(self.root/M.BUDGET_PLAN).read_bytes()
            pending=M.topology(self.program,root=self.root)
            self.assertEqual(pending['status'],'pending_explicit_mechanism_run');paid.assert_not_called()
            done=M.topology(self.program,root=self.root,allow_paid_api=True)
            self.assertEqual(done['new_api_calls'],1)
            reused=M.topology(self.program,root=self.root,allow_paid_api=True)
            self.assertEqual(reused['status'],'reused_verified_mechanism_topology');self.assertEqual(reused['new_api_calls'],0)
            self.assertEqual(paid.call_count,1);self.assertEqual(prepared.call_count,1);self.assertGreaterEqual(verified.call_count,2)
            self.assertEqual((self.root/M.BUDGET_PLAN).read_bytes(),before,'Task baseline must never reset')

    def test_failed_or_ambiguous_attempt_can_never_trigger_automatic_paid_resend(self):
        prep,plan,verify=self.patches()
        with prep,plan,verify,patch.object(reasoning,'run_prepared') as paid:
            value=M.topology(self.program,root=self.root)
            attempt=self.root/value['prepared']/'attempt';attempt.mkdir();(attempt/'failure.json').write_text('{"status":"failed_preserved"}')
            result=M.topology(self.program,root=self.root,allow_paid_api=True)
            self.assertEqual(result['status'],'pending_prior_mechanism_attempt_no_automatic_resend');paid.assert_not_called()

    def test_changed_contract_or_invalid_cached_response_stays_pending_without_paid_retry(self):
        prep,plan,verify=self.patches()
        with prep,plan,verify,patch.object(reasoning,'run_prepared',side_effect=self.generated) as paid:
            done=M.topology(self.program,root=self.root,allow_paid_api=True)
            with patch.object(object_program,'load_proposals',side_effect=ValueError('Saved response changed')):
                bad=M.topology(self.program,root=self.root,allow_paid_api=True)
            self.assertEqual(bad['status'],'pending_invalid_saved_topology_no_automatic_resend')
            (self.root/done['prepared']/'prompt.txt').write_text('changed')
            changed=M.topology(self.program,root=self.root,allow_paid_api=True)
            self.assertEqual(changed['status'],'pending_topology_contract_review_no_automatic_resend');self.assertEqual(paid.call_count,1)

    def test_existing_budget_plan_is_required_with_five_dollar_bound_and_path_confinement(self):
        profile={'mechanism_pipeline_mode':M.MODE,'mechanism_task_budget_plan':M.BUDGET_PLAN}
        self.assertEqual(M.configuration(profile,root=self.root)['max_images'],16)
        for mutate in [lambda x:x.update(mechanism_task_budget_plan='output/other.json'),lambda x:x.update(mechanism_max_images=99),lambda x:x.update(mechanism_max_output_tokens=99999)]:
            value=dict(profile);mutate(value)
            with self.assertRaises(ValueError):M.configuration(value,root=self.root)
        envelope=json.loads((self.root/M.BUDGET_PLAN).read_text());envelope['additional_estimated_api_ceiling_usd']='6';self.file(M.BUDGET_PLAN,envelope)
        with self.assertRaises(ValueError):M.configuration(profile,root=self.root)
        for path in ['.secrets/meta.env','data/../.secrets/meta.env','data/.private/a']:
            with self.subTest(path=path),patch.object(M,'sha') as hashing,self.assertRaises(ValueError):M.local(self.root,path)
            hashing.assert_not_called()

    def test_worker_namespace_and_lock_symlinks_fail_before_open(self):
        outside=self.root/'outside';outside.mkdir();data=self.root/'data'
        (data/'mechanism-pipelines').symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError),M.locked(self.root):pass
        self.assertEqual(list(outside.iterdir()),[])
        (data/'mechanism-pipelines').unlink();(data/'mechanism-pipelines').mkdir()
        target=outside/'lock';target.write_text('unchanged');(data/'mechanism-pipelines/.worker.lock').symlink_to(target)
        with self.assertRaises(ValueError),M.locked(self.root):pass
        self.assertEqual(target.read_text(),'unchanged')

    def test_explicit_adoption_reuses_existing_paid_result_and_tampering_never_sends(self):
        code=self.file('mechanism_reasoning.py',b'Frozen fixture reasoning contract')
        plan=self.file('data/previous-paid/plan.json',{'contract':{'code_sha256':M.sha(code)},'context':{'images':[]}})
        for name in ['schema.json','provider-schema.json']:
            self.file('data/previous-paid/'+name,{})
        self.file('data/previous-paid/prompt.txt',b'offline adoption fixture')
        model=self.file('data/previous-paid/attempt/model-result.json',{})
        receipt=self.file('data/previous-paid/attempt/request/meta-request.json',{})
        proposal=self.file('data/previous-paid/attempt/proposals.json',{'reasoning':{'plan_path':plan.relative_to(self.root).as_posix(),
            'model_result_path':model.relative_to(self.root).as_posix(),'request_receipt':receipt.relative_to(self.root).as_posix()}})
        with patch.object(object_program,'load_proposals',return_value=json.loads(proposal.read_text())),patch.object(reasoning,'run_prepared') as paid,patch.object(reasoning,'prepare') as prepare:
            adopted=M.adopt_verified_topology(self.program,proposal,root=self.root)
            self.assertEqual(adopted['new_api_calls'],0)
            reused=M.topology(self.program,root=self.root,allow_paid_api=True)
            self.assertEqual(reused['status'],'reused_adopted_verified_mechanism_topology')
            self.assertEqual(reused['proposal'],M.ref(proposal,self.root))
            self.assertIn(M.ref(model,self.root),reused['artifacts'],'Outer cache must include the adopted model response')
            proposal.write_text('{"changed":true}')
            pending=M.topology(self.program,root=self.root,allow_paid_api=True)
            self.assertEqual(pending['status'],'pending_adopted_topology_review_no_automatic_resend')
            record=self.root/'data/mechanism-pipelines'/M.sha(self.program)/'adopted-topology.json';record.unlink()
            deleted=M.topology(self.program,root=self.root,allow_paid_api=True)
            self.assertEqual(deleted['status'],'pending_namespace_review_no_automatic_resend')
            paid.assert_not_called();prepare.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec('cv2'),'Prepared CPU/SAM runtime required; no installation')
    def test_private_scene_then_real_unsupported_joint_and_export_never_float_the_child(self):
        # Analytic source/FK fixture; the provider grounding and GLB decoder are
        # separately covered by their own suites. Actual constrained builder,
        # unknown-child policy, export validator and publisher run here.
        from tests.test_object_program import ObjectProgramTests
        import sam3d_common_scene as common
        import sam3d_mechanisms as fitted
        fixture=ObjectProgramTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        root=fixture.root.resolve()
        for name in M.CODE:
            target=root/name;target.write_bytes((M.ROOT/name).read_bytes())
        plan=fixture.file('data/topology/plan.json',{'contract':{'code_sha256':M.sha(root/'mechanism_reasoning.py')}})
        proposal=json.loads((root/fixture.doc['proposal']['path']).read_text())
        proposal['joints'][0].update(joint_type='prismatic',attachment='persistent')
        proposal['reasoning']={'plan_path':plan['path']}
        pref=fixture.file(fixture.doc['proposal']['path'],proposal)
        baseline=json.loads((root/fixture.doc['source_scene']['path']).read_text())
        baseline['limitations']=['Analytic source fixture; unverified.']
        calls=[]
        def private(program,output,**kwargs):
            calls.append('private-common');path=output/'scene.json'
            if not path.exists():output.mkdir(parents=True);path.write_text(json.dumps(baseline))
            return {'manifest':M.ref(path,root)}
        with patch.object(M,'require_supported_segmentation'),patch.object(common,'metadata_for_program',return_value=[root/c['native_metadata']['path'] for c in fixture.components]),patch.object(common,'build',side_effect=private),patch.object(common,'publish_registry') as raw_publish,patch.object(fitted,'__file__',str(root/'sam3d_mechanisms.py')):
            value=M.constrained_scene(fixture.doc['evidence_program']['path'],pref,root=root)
            cached=M.constrained_scene(fixture.doc['evidence_program']['path'],pref,root=root)
        self.assertEqual(calls,['private-common','private-common']);raw_publish.assert_not_called()
        self.assertTrue(cached['mechanism_fit_reused']);self.assertEqual(cached['manifest'],value['manifest'])
        self.assertEqual(value['status'],'constrained_scene_with_unknown_connections');self.assertTrue(value['has_gaps'])
        self.assertFalse((root/'showcase/aligned-scenes.json').exists())
        constrained=json.loads((root/value['manifest']['path']).read_text())
        child=next(c for c in constrained['components'] if c['id']=='lid')
        self.assertTrue(all(p['status']=='unknown' and p['camera_from_native_mesh'] is None for p in child['poses']))
        exported=M.export_scene(value['manifest'],root=root,publish=True)
        doc=json.loads((root/exported['object_program']['path']).read_text())
        self.assertEqual(len(doc['unresolved_connections']),1);self.assertEqual(doc['joints'],[])
        self.assertFalse(doc['simulation_ready']);self.assertTrue(exported['has_gaps'])
        self.assertEqual(json.loads((root/'showcase/aligned-scenes.json').read_text())['scenes'][0]['manifest'],value['manifest'])
        reused=M.export_scene(value['manifest'],root=root,publish=True)
        self.assertEqual(reused['object_program'],exported['object_program'])

    def test_later_segmentation_uses_its_exact_compiled_refs_without_reading_older_masks(self):
        case=self.root/'data/case';manifest=self.file('data/case/manifest.json',{'source':{'sha256':'a'*64}})
        index=self.file('data/case/frames/index.json',{'time_domain':'source_video_presentation_seconds','frames':[]})
        older=self.file('data/case/segmentation/flow-components-1/tracks.json',{'tracks':['old']})
        tracks=self.file('data/case/segmentation/flow-components-2/tracks.json',{'tracks':[]})
        gates=self.file('data/case/segmentation/flow-components-2/frames.json',{'frames':[]})
        self.file('data/case/segmentation/flow-components-2/quality.json',{'tracks':{}})
        doc={'source':{'sha256':'a'*64,'time_domain':'source_video_presentation_seconds'},'inputs':[
            {'id':identifier,'path':path.relative_to(self.root).as_posix(),'sha256':M.sha(path)}for identifier,path in
            [('source_manifest',manifest),('source_frame_index',index),('mask_tracks',tracks),('mask_quality',gates)]]}
        self.file('data/case/program/interaction.json',doc)
        binding=M.require_supported_segmentation(self.program,root=self.root)
        self.assertEqual(binding['tracks'],tracks);self.assertEqual(binding['directory'],tracks.parent)
        older.unlink();self.assertEqual(M.require_supported_segmentation(self.program,root=self.root)['tracks'],tracks)
        tracks.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'input changed'):M.require_supported_segmentation(self.program,root=self.root)

    def test_resolved_hinge_with_unknown_configuration_samples_still_reports_export_gaps(self):
        from tests.test_object_program import ObjectProgramTests
        fixture=ObjectProgramTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        root=fixture.root.resolve();(root/'object_program.py').write_bytes((M.ROOT/'object_program.py').read_bytes())
        scene_ref=fixture.doc['constrained_scene']
        identity=M.digest({'scene':scene_ref,'exporter_sha256':M.sha(root/'object_program.py')})
        target='data/object-programs/auto-'+identity+'/object-program.json'
        fixture.file(target,fixture.doc)
        result=M.export_scene(scene_ref,root=root,publish=False)
        self.assertEqual(result['unresolved_connections'],0)
        self.assertEqual(result['unknown_joint_samples'],1)
        self.assertTrue(result['has_gaps'],'A resolved joint is incomplete while its configuration is unknown at a source sample')
        self.assertEqual(M.unknown_joint_samples([{'samples':[{'status':'estimated','q_rad':0.0}]}]),0)

    def test_default_profile_and_explicit_follow_on_have_no_legacy_or_refresh_inference_route(self):
        profile=P.load_profile('meta_sam3d_default')
        self.assertEqual(profile['mechanism_pipeline_mode'],M.MODE)
        self.assertLess(P.PHASES.index('procedure'),P.PHASES.index('mechanism_topology'))
        self.assertLess(P.PHASES.index('object_program'),P.PHASES.index('workspace'))
        with tempfile.TemporaryDirectory(dir=P.ROOT/'data') as temporary:
            directory=Path(temporary);runner=P.Runner('mechanism-phase-fixture',run_root=directory/'runs');runner.profile=profile
            runner.state['artifacts']={'interaction':'data/frozen/interaction.json','constrained_scene':'data/old/scene.json','object_program':'data/old/object-program.json'}
            case=directory/'case';program=case/'program/interaction.json';P.write(program,{'source':{'sha256':'a'*64}})
            P.write(case/'manifest.json',{});(case/'frames').mkdir();segment=case/'segmentation';segment.mkdir()
            calls=[]
            def phase(name,dependencies,action,**kwargs):
                calls.append(name)
                if name=='mechanism_topology':return {'status':'pending_prior_mechanism_attempt_no_automatic_resend','has_gaps':True}
                result,_=action(1);return result
            runner.state['artifacts']['part_ledger']='data/frozen/part-ledger.json'
            with patch('part_ledger.metadata_paths',return_value=[]),patch.object(runner,'phase',side_effect=phase),patch.object(runner,'command') as command,patch.object(M,'scene_input_paths',return_value=[]):
                runner._mechanism_follow_on(program,case=case,segmentation=segment)
            self.assertEqual(calls,['mechanism_topology','mechanism_scene','object_program']);command.assert_not_called()
            self.assertEqual(runner.state['mechanism_pipeline_status']['scene'],'pending_mechanism_topology')
            self.assertNotIn('constrained_scene',runner.state['artifacts']);self.assertNotIn('object_program',runner.state['artifacts'])

    def test_missing_part_ledger_stops_before_any_paid_follow_on(self):
        profile=P.load_profile('meta_sam3d_default')
        with tempfile.TemporaryDirectory(dir=P.ROOT/'data') as temporary:
            directory=Path(temporary);runner=P.Runner('missing-ledger-fixture',run_root=directory/'runs');runner.profile=profile
            runner.state['artifacts']={'constrained_scene':'data/old/scene.json','object_program':'data/old/object-program.json'}
            program=directory/'program.json';P.write(program,{'source':{'sha256':'a'*64}})
            with patch.object(runner,'phase')as phase,patch.object(runner,'command')as command:
                runner._mechanism_follow_on(program,case=directory,segmentation=directory)
            phase.assert_not_called();command.assert_not_called()
            self.assertEqual(runner.state['mechanism_pipeline_status']['topology'],'pending_verified_part_ledger')
            self.assertNotIn('constrained_scene',runner.state['artifacts'])
            self.assertNotIn('object_program',runner.state['artifacts'])


if __name__=='__main__':unittest.main()
