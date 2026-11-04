"""Local-only exact supplied aliases; original paid response is never repaired."""
import copy,json,unittest
from unittest.mock import patch

import generation_visual_alias_v1 as A
import generation_visual_judge as J
from tests import test_generation_visual_judge as F


class AliasDerivativeTests(unittest.TestCase):
    def setUp(self):
        self.f=F.GenerationVisualJudgeTests(methodName='runTest');self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.f.prepare();self.f.execute();self.root=self.f.root;self.packet=self.f.packet
        self.plan,_,_=J.verify_plan(self.packet,root=self.root)
        self.raw_path=self.root/self.packet/'attempt/model-result.json'
        self.pending=self.root/self.packet/'attempt/request/response.pending.json'
        self.raw=json.loads(self.raw_path.read_text());self.output='output/alias/one'

    def save(self,result):
        for p in [self.raw_path,self.pending]:p.write_text(json.dumps(result))

    def test_same_sample_aliases_only_and_no_client_or_original_mutation(self):
        anchor=self.plan['context']['anchor'];candidate=self.plan['context']['candidate_anchors'][0]
        self.raw['evidence'][0]['evidence_ids']=[anchor['evidence_id'],'orbit:0']
        self.raw['anchor_suggestions'][0]['candidate_evidence_id']=candidate['evidence_id'];self.save(self.raw)
        original=self.raw_path.read_bytes();receipt=(self.pending.parent/'meta-request.json').read_bytes()
        with patch.object(J,'OvernightMetaClient',side_effect=AssertionError('No provider')):
            reference=A.build(self.packet,self.output,root=self.root);status=A.verify(reference['path'],root=self.root)
        self.assertEqual(status['status'],'valid_fallible_offline_derivative')
        proof=json.loads((self.root/reference['path']).read_text());self.assertEqual(len(proof['normalization_edits']),2)
        normalized=json.loads((self.root/proof['normalized_result_ref']['path']).read_text())
        self.assertEqual(normalized['evidence'][0]['evidence_ids'],[anchor['id'],'orbit:0'])
        self.assertEqual(normalized['anchor_suggestions'][0]['candidate_evidence_id'],candidate['id'])
        self.assertEqual(self.raw_path.read_bytes(),original);self.assertEqual((self.pending.parent/'meta-request.json').read_bytes(),receipt)

    def test_contradictory_faithful_is_still_rejected_without_verdict_rewrite(self):
        self.raw['verdict']='faithful';self.save(self.raw)
        reference=A.build(self.packet,self.output,root=self.root);status=A.verify(reference['path'],root=self.root)
        self.assertEqual(status['status'],'rejected');self.assertIn('contradicts',status['validation_reason'])
        self.assertEqual(status['verdict'],'faithful')

    def test_foreign_or_anchor_only_suggestions_not_invented(self):
        for bad in ['source_999999',self.plan['context']['anchor']['evidence_id']]:
            raw=copy.deepcopy(self.raw);raw['anchor_suggestions'][0]['candidate_evidence_id']=bad
            with self.assertRaisesRegex(ValueError,'unsupplied'):A.normalize_result(raw,self.plan)

    def test_non_bijective_or_foreign_component_aliases_fail(self):
        plan=copy.deepcopy(self.plan);plan['context']['candidate_anchors'][0]['evidence_id']=plan['context']['anchor']['evidence_id']
        with self.assertRaisesRegex(ValueError,'bijective'):A.aliases(plan)
        plan=copy.deepcopy(self.plan);plan['context']['anchor']['id']='other:2'
        with self.assertRaisesRegex(ValueError,'same-component'):A.aliases(plan)

    def test_changed_raw_parser_or_accounting_blocks_before_output(self):
        self.pending.write_text('{}')
        with self.assertRaisesRegex(ValueError,'parser output'):A.build(self.packet,self.output,root=self.root)
        self.assertFalse((self.root/self.output).exists())
        self.save(self.raw);path=self.pending.parent/'meta-request.json';receipt=json.loads(path.read_text());receipt['payload_sha256']='0'*64;path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError,'accounted'):A.build(self.packet,self.output,root=self.root)

    def test_result_or_claim_tamper_rejected_and_outputs_never_overwritten(self):
        reference=A.build(self.packet,self.output,root=self.root)
        with self.assertRaisesRegex(ValueError,'fresh immutable'):A.build(self.packet,self.output,root=self.root)
        path=self.root/reference['path'];proof=json.loads(path.read_text());proof['automatic_approval']=True;path.write_text(json.dumps(proof))
        with self.assertRaisesRegex(ValueError,'policy'):A.verify(path,root=self.root)

    def test_unknown_evidence_or_manual_geometry_claim_cannot_be_normalized(self):
        raw=copy.deepcopy(self.raw);raw['evidence'][0]['evidence_ids']=['other:1']
        with self.assertRaisesRegex(ValueError,'unsupplied'):A.normalize_result(raw,self.plan)
        raw=copy.deepcopy(self.raw);raw['geometry_modified']=True
        with self.assertRaises(Exception):A.normalize_result(raw,self.plan)

    def test_rehashed_derivative_cannot_change_semantic_text(self):
        reference=A.build(self.packet,self.output,root=self.root)
        path=self.root/reference['path'];proof=json.loads(path.read_text())
        result_path=self.root/proof['normalized_result_ref']['path'];result=json.loads(result_path.read_text())
        result['summary']='Altered scientific claim.';result_path.write_text(json.dumps(result))
        proof['normalized_result_ref']=A.S.file_ref(result_path,self.root);path.write_text(json.dumps(proof))
        with self.assertRaisesRegex(ValueError,'exact permitted'):A.verify(path,root=self.root)


if __name__=='__main__':unittest.main()
