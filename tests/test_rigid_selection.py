"""Exact seed lineage checks; saved source fixtures are read without inference."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from rigid_selection import validate_config, sha
from stage3_rigid_select_track_views import collect_track

CASE=ROOT/'data/flow-meta-dense-jabra-20261030-v1'
RUNS=['rigid-parts-video-v1-20261031','rigid-parts-video-v2-20261031']


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def load(path):return json.loads(path.read_text())


class RigidSelectionTests(unittest.TestCase):
    def fixture(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        case=Path(temporary.name)/'case';case.mkdir()
        for relative in ['manifest.json','frames/index.json']:
            target=case/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(CASE/relative,target)
        cfg=load(CASE/'segmentation'/RUNS[0]/'annotations.json')
        discovery=next(p for p in (CASE/'reasoning').glob('*/discovery.json') if sha(p)==cfg['discovery_record_sha256'])
        target=case/discovery.relative_to(CASE);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(discovery,target)
        for run in RUNS:
            for relative in ['metadata.json','annotations.json','tracks.json','model-audit-v1/audit.json']:
                target=case/'segmentation'/run/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(CASE/'segmentation'/run/relative,target)
        return case

    def rebind_config_receipt(self,run):
        metadata=load(run/'metadata.json');metadata['config_sha256']=sha(run/'annotations.json')
        save(run/'metadata.json',metadata)

    def test_actual_v1_v2_all_parts_have_source_audited_candidates(self):
        for run in RUNS:
            for tid in ['case_base','case_lid','earbud_a','earbud_b']:
                with self.subTest(run=run,track=tid):
                    candidates,_,_,automatic=collect_track(CASE,run,tid,'model-audit-v1')
                    self.assertTrue(automatic);self.assertGreater(len(candidates),0)
                    for row in candidates:
                        self.assertTrue(row['mask_provenance']['exact_rigid_config_lineage_verified'])
                        self.assertFalse(row['mask_provenance']['manual_seed_boxes_used'])
                        self.assertEqual(row['mask_provenance']['rigid_selection']['omission_only_derivative'],run==RUNS[1])

    def test_v1_extra_policy_is_explicit_and_unknown_policy_fails(self):
        case=self.fixture();run=case/'segmentation'/RUNS[0]
        cfg=load(run/'annotations.json');cfg['component_policy']='arbitrary_unknown_policy'
        save(run/'annotations.json',cfg);self.rebind_config_receipt(run)
        with self.assertRaisesRegex(ValueError,'canonical'):validate_config(case,run)

    def test_v1_changed_model_box_rejected_even_with_updated_local_config_receipt(self):
        case=self.fixture();run=case/'segmentation'/RUNS[0]
        cfg=load(run/'annotations.json');cfg['tracks'][0]['prompts'][0]['box_xyxy'][0]+=1
        save(run/'annotations.json',cfg);self.rebind_config_receipt(run)
        with self.assertRaisesRegex(ValueError,'canonical'):validate_config(case,run)

    def test_v2_changed_box_and_omission_trace_each_fail_closed(self):
        for mutation in ['box','trace']:
            with self.subTest(mutation=mutation):
                case=self.fixture();run=case/'segmentation'/RUNS[1]
                cfg=load(run/'annotations.json')
                if mutation=='box':cfg['tracks'][0]['prompts'][0]['box_xyxy'][0]+=1
                else:cfg['seed_review']['changes'][0]['operation']='retain_identical_model_seed'
                save(run/'annotations.json',cfg);self.rebind_config_receipt(run)
                with self.assertRaisesRegex(ValueError,'Omission derivative'):validate_config(case,run)

    def test_parent_config_tracks_or_audit_bytes_cannot_be_substituted(self):
        for relative in ['annotations.json','tracks.json','model-audit-v1/audit.json']:
            with self.subTest(relative=relative):
                case=self.fixture();parent=case/'segmentation'/RUNS[0]
                p=parent/relative;p.write_bytes(p.read_bytes()+b' ')
                with self.assertRaisesRegex(ValueError,'parent'):validate_config(case,case/'segmentation'/RUNS[1])

    def test_current_config_must_match_original_sam_receipt(self):
        case=self.fixture();run=case/'segmentation'/RUNS[1]
        p=run/'annotations.json';p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'receipt/config'):validate_config(case,run)

    def test_parent_and_current_source_identity_required(self):
        for run_name in RUNS:
            with self.subTest(run=run_name):
                case=self.fixture();run=case/'segmentation'/run_name
                meta=load(run/'metadata.json');meta['source_sha256']='0'*64;save(run/'metadata.json',meta)
                with self.assertRaisesRegex(ValueError,'source'):validate_config(case,case/'segmentation'/RUNS[1])


if __name__=='__main__':unittest.main()
