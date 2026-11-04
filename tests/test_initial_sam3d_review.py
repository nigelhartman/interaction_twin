"""Initial import review boundaries; no retained or fake model review pages."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import initial_sam3d_import as N
import sam3d_milestone as M
import upgrade_bundles as U
import tests.test_initial_sam3d_import as F

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('initial_native_review_builder',ROOT/'scripts/build_initial_sam3d_review.py')
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)


class InitialNativeReviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture=F.InitialNativeImportTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root;self.imported=self.root/'data/initial-native-v1'
        N.import_output(self.fixture.handoff,self.fixture.output,self.imported,root=self.root)
        self.output=self.root/'data/review-never-created'

    def rebind_import_inventory(self):
        record=U.read(self.imported/'import.json')
        record['import_files']=M._records({k:v for k,v in M._tree(self.imported).items() if k!='import.json'})
        U.write(self.imported/'import.json',record)

    def test_readonly_revalidation_uses_initial_input_and_preserves_every_byte(self):
        before=M._records(M._tree(self.imported))
        with patch.object(U,'context',side_effect=AssertionError('No baseline context')):
            directory,receipt,source,native,model=V.verified_initial_import(self.imported,root=self.root)
        self.assertEqual(directory,self.imported);self.assertFalse(receipt['baseline_required']);self.assertFalse(model['selected'])
        self.assertEqual(native['files']['object.glb'],(self.imported/'asset.glb').read_bytes())
        self.assertEqual(M._records(M._tree(self.imported)),before);self.assertFalse(self.output.exists())

    def test_explicit_fixture_receipt_never_creates_page(self):
        record=U.read(self.imported/'import.json');record['fixture']=True;U.write(self.imported/'import.json',record)
        with self.assertRaisesRegex(ValueError,'fixture'):V.build(self.imported,self.output,root=self.root)
        self.assertFalse(self.output.exists())

    def test_changed_inventory_stops_before_output_or_model_revalidation(self):
        (self.imported/'unexpected.txt').write_bytes(b'inert')
        with patch.object(N,'inspect_output',side_effect=AssertionError('Must not reach model check')):
            with self.assertRaisesRegex(ValueError,'inventory/hash'):V.build(self.imported,self.output,root=self.root)
        self.assertFalse(self.output.exists())

    def test_rehashed_selection_or_baseline_lineage_promotion_still_rejected(self):
        for key in ['selected','camera_pose_estimated','physical_accuracy_verified']:
            model=U.read(self.imported/'metadata.json');model[key]=True;U.write(self.imported/'metadata.json',model);self.rebind_import_inventory()
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'provenance or selection'):
                V.verified_initial_import(self.imported,root=self.root)
            model[key]=False;U.write(self.imported/'metadata.json',model)
        model=U.read(self.imported/'metadata.json');model['initial_reconstruction_lineage']['old_poses_read']=True
        U.write(self.imported/'metadata.json',model);self.rebind_import_inventory()
        with self.assertRaisesRegex(ValueError,'lineage'):V.verified_initial_import(self.imported,root=self.root)

    def test_rehashed_handoff_copy_or_source_time_cannot_be_substituted(self):
        copied=self.imported/'initial-handoff/README.txt';copied.write_text('changed');self.rebind_import_inventory()
        with self.assertRaisesRegex(ValueError,'handoff copy'):V.verified_initial_import(self.imported,root=self.root)

    def test_rehashed_normalized_source_does_not_override_real_input_timestamp(self):
        model=U.read(self.imported/'metadata.json');model['source']['timestamp_s']=9.
        U.write(self.imported/'metadata.json',model);self.rebind_import_inventory()
        with self.assertRaisesRegex(ValueError,'source RGB/mask/time'):V.verified_initial_import(self.imported,root=self.root)


if __name__=='__main__':unittest.main()
