import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import local_candidates as C


class CandidateDiagnostics(unittest.TestCase):
    def test_three_second_blocks_are_disjoint_and_deterministic(self):
        timestamps=[0.,2.99,3.,5.99,6.,8.99,9.,12.,15.]
        self.assertEqual([C.split_name(x) for x in timestamps],['selection','selection','validation','validation','final_test','final_test','selection','validation','final_test'])

    def test_references_and_missing_are_explicit_not_supported(self):
        pack={'frames':[
            {'sample_index':1,'status':'reference_candidate_unverified_geometry_and_pose'},
            {'sample_index':2,'status':'supported_conditional_motion_candidate','raw_full_projection_iou_with_sam':.8,'heldout_median_px':2.},
            {'sample_index':3,'status':'unsupported'},
        ]}
        actual=C.score_motion(pack,[1,2,3,4])
        self.assertEqual((actual['denominator'],actual['supported'],actual['reference_excluded'],actual['unsupported'],actual['missing']),(4,1,1,1,1))
        self.assertEqual(actual['coverage'],.25)
        self.assertEqual(actual['median_supported_heldout_px'],2.)

    def test_winner_uses_validation_only_even_if_final_test_reverses(self):
        rows=[{'id':'a','status':'complete','validation':{'denominator':2,'supported':2,'median_supported_iou':.6,'median_supported_heldout_px':2.},'final_test':{'supported':0}},
              {'id':'b','status':'complete','validation':{'denominator':2,'supported':1,'median_supported_iou':.99,'median_supported_heldout_px':.1},'final_test':{'supported':100}},
              {'id':'c','status':'failed','validation':{'denominator':2,'supported':999}}]
        self.assertEqual(C.select_winner(rows),'a')
        rows[0]['final_test']['supported']=9999
        self.assertEqual(C.select_winner(rows),'a')

    def test_zero_denominator_cannot_win(self):
        self.assertIsNone(C.select_winner([{'id':'x','status':'complete','validation':{'denominator':0}}]))

    def test_partial_stage_uses_fresh_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'model').mkdir();(root/'model'/'receipt').write_text('preserved')
            target=C.fresh_stage(root,'model');self.assertEqual(target.name,'model-attempt-02')
            target.mkdir();self.assertEqual(C.fresh_stage(root,'model').name,'model-attempt-03')
            self.assertEqual((root/'model'/'receipt').read_text(),'preserved')

    def test_recover_model_only_after_every_saved_output_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);asset=root/'asset.glb';asset.write_bytes(b'actual-asset-placeholder')
            value={'status':'candidate_generated_pending_visual_review','learned_inference_executed':True,
                   'source':{'selection_manifest_sha256':'selected'},'density_grid_resolution':96,'runtime':{'device':'mps'},
                   'outputs':{'glb_path':'asset.glb'},'artifact_files':[{'file':'asset.glb','bytes':asset.stat().st_size,'sha256':C.sha(asset)}]}
            metadata=root/'metadata.json';C.write(metadata,value)
            self.assertTrue(C.complete_model(metadata,'selected'))
            self.assertFalse(C.complete_model(metadata,'changed-selection'))
            asset.write_bytes(b'changed');self.assertFalse(C.complete_model(metadata,'selected'))

    def test_migration_cannot_override_immutable_scripts_or_changed_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'experiment';out.mkdir()
            wrapper=root/'local_candidates.py';wrapper.write_text('old wrapper')
            original=C.sha(wrapper);snapshot=out/'original.py';snapshot.write_bytes(wrapper.read_bytes())
            modelscript=root/'model.py';modelscript.write_text('immutable model')
            plan={'code_inputs':[{'path':'local_candidates.py','sha256':original},{'path':'model.py','sha256':C.sha(modelscript)}],'input_receipts':[]}
            C.write(out/'plan.json',plan);state={'plan_sha256':C.sha(out/'plan.json')}
            wrapper.write_text('recovery wrapper')
            migration={'plan_sha256':state['plan_sha256'],'original_sha256':original,'original_snapshot':'experiment/original.py','replacement_sha256':C.sha(wrapper),'policy_or_model_changes':False}
            C.write(out/'orchestration-migration.json',migration)
            with patch.object(C,'ROOT',root):
                C.verify_plan(out,plan,state)
                modelscript.write_text('changed model')
                with self.assertRaisesRegex(ValueError,'Frozen code/input changed'):C.verify_plan(out,plan,state)
                C.write(out/'plan.json',{'changed':'split'})
                with self.assertRaisesRegex(ValueError,'Frozen plan changed'):C.verify_plan(out,plan,state)


if __name__=='__main__':unittest.main()
