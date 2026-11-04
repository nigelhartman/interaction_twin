"""Static original-parser attestation survives the actual nested private wrappers."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

import admin_machine_context as machine
from agent_memory import FeedbackMemory
import meta_api
import native_visual_recovery as historical
import native_visual_recovery_v2 as recovery
import native_visual_review as visual
import mechanism_pass_budget as mechanism_budget
from object_program import ref
from tests import test_native_visual_recovery as fixtures


class StaticParserRecoveryTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.NativeVisualRecoveryTests()
        # The actual nested admin wrapper now admits the exact approved nested
        # budget before constructing its contextual client.  Seed immutable
        # companion bytes before the visual plan freezes its parent-plan ref;
        # never replace a plan after its evidence/report was prepared.
        original_prepare = fixtures.fixtures.NativeVisualReviewTests.prepare

        def prepare_with_approved_budget(harness, **kwargs):
            for relative in [mechanism_budget.PARENT_PLAN, mechanism_budget.HISTORICAL_PLAN,
                             mechanism_budget.PLAN]:
                destination = harness.root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((mechanism_budget.ROOT / relative).read_bytes())
            return original_prepare(harness, **kwargs)

        with patch.object(fixtures.fixtures.NativeVisualReviewTests, 'prepare', prepare_with_approved_budget):
            fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture; self.root = fixture.root
        self.historical = fixture.recover()
        self.historical_path = self.root / fixture.output / 'report.json'
        self.historical_bytes = self.historical_path.read_bytes()
        self.historical_helper_bytes = Path(historical.__file__).read_bytes()
        self.output = '.private/native-visual-recoveries-v2/fixture'

    def recover(self):
        return recovery.recover(self.fixture.prepared, self.output, root=self.root)

    def test_static_method_matches_original_exact_hash_and_refuses_ambiguous_structure(self):
        raw = Path(meta_api.__file__).read_bytes()
        method = recovery._static_parser_method(raw)
        self.assertEqual(hashlib.sha256(method).hexdigest(), recovery.PARSER_METHOD_SHA256)
        for candidate in [raw + b'\nclass MetaClient:\n    pass\n',
                          b'class MetaClient:\n    def json_result(self): pass\n    def json_result(self): pass\n',
                          b'class MetaClient:\n    @staticmethod\n    def json_result(): pass\n',
                          b'class MetaClient:\n    async def json_result(self): pass\n']:
            with self.subTest(source=candidate[-60:]), self.assertRaises(ValueError):
                recovery._static_parser_method(candidate)

    def test_v2_is_an_exact_offline_omission_and_historical_v1_stays_verifiable(self):
        before = self.fixture.client.ledger.status()
        report = self.recover()
        self.assertEqual(report['format'], recovery.FORMAT); self.assertEqual(report['version'], 2)
        self.assertTrue(recovery.verify_any_report(report, root=self.root))
        self.assertTrue(recovery.verify_any_report(self.historical, root=self.root))
        self.assertEqual(report['repair_proposals'], self.fixture.result['repair_proposals'][1:])
        self.assertEqual(report['issues'], self.fixture.result['issues'])
        self.assertEqual(report['verdict'], self.fixture.result['verdict'])
        self.assertEqual(report['new_api_calls'], 0); self.assertFalse(report['automatic_repair_safe'])
        self.assertFalse(report['parser_attestation']['live_client_class_inspected'])
        self.assertFalse(report['parser_attestation']['provider_envelope_preserved'])
        self.assertEqual(self.fixture.client.ledger.status(), before)
        self.assertEqual(self.historical_path.read_bytes(), self.historical_bytes)
        self.assertEqual(Path(historical.__file__).read_bytes(), self.historical_helper_bytes)
        self.assertEqual(len(self.fixture.calls), 1)

    def test_wrapped_live_class_does_not_change_v2_original_source_proof(self):
        report = self.recover()
        class ReboundPrivateClient:
            def json_result(self, *args, **kwargs):
                raise AssertionError('This proof must never call or inspect the live client method')
        with patch.object(meta_api, 'MetaClient', ReboundPrivateClient):
            self.assertTrue(recovery.verify_report(report, root=self.root))
            self.assertEqual(recovery._parser_evidence()['original-json-result.py.txt'],
                recovery._static_parser_method(Path(meta_api.__file__).read_bytes()))
        self.assertTrue(historical.verify_report(self.historical, root=self.root))

    def test_changed_module_bytes_are_rejected_even_with_a_matching_live_class(self):
        path = self.root / 'changed-meta-api.py'
        path.write_bytes(Path(meta_api.__file__).read_bytes() + b'\n# changed source\n')
        with patch.object(meta_api, '__file__', str(path)), self.assertRaisesRegex(ValueError, 'source changed'):
            self.recover()

    def test_actual_overnight_admin_child_rebinding_reloads_typed_guidance_before_offline_dispatch(self):
        report = self.recover()
        report_path = self.root / self.output / 'report.json'
        context = self.fixture.plan['context']
        measured = {'artifact_binding': {'source_ref': {'sha256': 'b' * 64},
                    'interaction_ref': context['interaction_ref'], 'scene_ref': context['scene_ref'],
                    'object_program_ref': None},
                    'raw_critic_ref': ref(report_path, self.root), 'issues': report['issues'],
                    'repair_proposals': report['repair_proposals'],
                    'improvement_assessment': {'metrics': {'alignment_error': .6}}}
        measured_path = self.root / '.private/measured/review.json'
        measured_path.parent.mkdir(parents=True); visual.write_new(measured_path, measured)
        guidance = machine.snapshot(ref(measured_path, self.root), source_sha256='b' * 64, root=self.root)
        memory = FeedbackMemory(self.root).retrieve(source_sha256='b' * 64)
        memory_path = self.root / '.private/memory.json'; visual.write_new(memory_path, memory)
        memory_ref = dict(ref(memory_path, self.root), context_sha256=memory['context_sha256'])
        project = Path(recovery.__file__).resolve().parent
        scripts = self.root / 'scripts'; scripts.mkdir()
        for name in ['run_overnight_stage.py', 'run_admin_stage.py']:
            shutil.copyfile(project / 'scripts' / name, scripts / name)
        shutil.copyfile(project / 'mechanism_pass_budget.py', self.root / 'mechanism_pass_budget.py')
        bootstrap = r'''
import functools,json,runpy,sys
from pathlib import Path
import admin_memory_context as memory
import meta_api
root=Path(sys.argv[1]);reference=json.loads(sys.argv[2]);guidance=json.loads(sys.argv[3])
proof={'wrappers':[],'dispatches':0,'network_attempts':0}
# Replace only the terminal inference method on the original live class.  Both
# real wrapper scripts and the real fixed-plan scoped client remain selected;
# static parser attestation must ignore this deliberate runtime replacement.
def offline_result(self,prompt,schema,images,*,output,**kwargs):
    proof['dispatches']+=1
    proof['machine_guidance_present']='MACHINE REVIEW GUIDANCE'in prompt
    proof['typed_citations_present']='typed_citations'in prompt
    Path(output).mkdir(parents=True,exist_ok=True)
    return {'offline_fixture':True},{'estimated_cost_usd':'0.000000','offline_test_no_transport':True}
meta_api.MetaClient.json_result=offline_result
memory.contextual_client=functools.partial(memory.contextual_client,root=root)
original=runpy.run_path
def traced(path,*args,**kwargs):
    name=Path(path).name
    if name in {'run_overnight_stage.py','run_admin_stage.py'}:
        proof['wrappers'].append(name)
        return original(path,*args,**kwargs)
    assert name=='stage5_meta_propose.py',name
    proof['rebound_class']=meta_api.MetaClient.__name__
    proof['actual_nested_scope_present']='MechanismPassClient' in [c.__name__ for c in meta_api.MetaClient.__mro__]
    def no_network(*args,**kwargs):
        proof['network_attempts']+=1
        raise AssertionError('Network is forbidden in the nested wrapper fixture')
    result,receipt=meta_api.MetaClient(ledger_path=root/'.private/offline-child.sqlite3',transport=no_network).json_result(
        'Inert base prompt',{},[],output=root/'data/offline-child')
    assert result=={'offline_fixture':True}
    assert receipt['owner_annotation_context']['machine_review_ref']==guidance
    (root/'child-proof.json').write_text(json.dumps(proof))
    return {}
runpy.run_path=traced
target=root/'scripts/run_overnight_stage.py'
sys.argv=[str(target),'--task-budget-plan','output/overnight-pipeline-20261102/task-budget-plan.json',
          'run_admin_stage.py','--memory-context-ref',json.dumps(reference),'--source-sha256','b'*64,
          '--machine-review-ref',json.dumps(guidance),'stage5_meta_propose.py']
runpy.run_path(str(target),run_name='__main__')
'''
        before = self.fixture.client.ledger.status()
        child = subprocess.run([sys.executable, '-c', bootstrap, str(self.root), json.dumps(memory_ref),
                                json.dumps(guidance)], cwd=project, capture_output=True, text=True, timeout=30)
        self.assertEqual(child.returncode, 0, child.stderr)
        proof = json.loads((self.root / 'child-proof.json').read_bytes())
        self.assertEqual(proof['wrappers'], ['run_overnight_stage.py', 'run_admin_stage.py'])
        self.assertEqual(proof['rebound_class'], 'AdminContextMetaClient')
        self.assertEqual(proof['dispatches'], 1)
        self.assertEqual(proof['network_attempts'], 0)
        self.assertTrue(proof['actual_nested_scope_present'])
        self.assertTrue(proof['machine_guidance_present']); self.assertTrue(proof['typed_citations_present'])
        self.assertEqual(self.fixture.client.ledger.status(), before)
        self.assertEqual(self.historical_path.read_bytes(), self.historical_bytes)
        self.assertEqual(Path(historical.__file__).read_bytes(), self.historical_helper_bytes)
        self.assertEqual(len(self.fixture.calls), 1)


if __name__ == '__main__':
    unittest.main()
