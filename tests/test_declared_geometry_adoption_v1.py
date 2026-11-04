"""Private derivative adoption keeps queue/approval and provenance boundaries."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import declared_geometry_adoption_v1 as A
from tests import test_jabra_cached_assembly_adoption_20261103 as Previous


class GeometryAdoptionIsolationTests(Previous.DeclaredAssemblyAdoptionTests):
    def setUp(self):
        super().setUp()
        self.plan['result']['repair_execution'][0]['strategy'] = A.STRATEGY

    def adopt(self, plan=None):
        with patch.object(A, 'build', return_value=copy.deepcopy(self.plan)):
            return A.adopt(copy.deepcopy(self.plan) if plan is None else plan,
                           root=self.root, contract=self.plan['contract'])


class GeometryAdoptionContractTests(unittest.TestCase):
    def test_every_prior_hosted_retention_record_is_preserved_without_resolution(self):
        old = [{'finding': 'gap remains'}, {'finding': 'appearance wrong'}]
        base = {'id': 'revision_one', 'fingerprint': 'f' * 64,
                'hosted_blocking_findings': 3,
                'manifest': {'review': {'hosted_roles_completed': 2,
                                       'hosted_review_status': 'needs_repair'},
                             'repair_execution': [{'retained_prior_hosted_findings': x}
                                                  for x in old]}}
        retained = A.retained_findings(base)
        self.assertEqual(retained['prior_retention_records'], old)
        self.assertFalse(retained['new_hosted_visual_review_performed_by_adoption'])
        self.assertTrue(retained['objections_are_not_resolved_by_structural_review'])
        retained['prior_retention_records'][0]['finding'] = 'changed copy'
        self.assertEqual(old[0]['finding'], 'gap remains')

    def test_non_data_reference_cannot_cause_a_credential_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'data'
            folder.mkdir()
            path = folder / 'native-program.json'
            raw = json.dumps({'implementation_provenance': []}).encode()
            path.write_bytes(raw)
            pointer = {'path': 'data/native-program.json', 'bytes': len(raw),
                       'sha256': hashlib.sha256(raw).hexdigest()}
            document = {'base_object_program_ref': pointer,
                        'untrusted': {'path': '.secrets/meta.env', 'bytes': 10,
                                      'sha256': 'a' * 64}}
            with self.assertRaisesRegex(ValueError, 'Non-data reference lacks exact original code archive'):
                A.artifact_inventory(root, {'artifacts': []}, document)
            self.assertFalse((root / '.secrets').exists())

    def test_cached_or_structural_only_validation_cannot_skip_real_source_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            doc = root / 'derivative.json'
            doc.write_text('{}')
            contract = {'candidate_id': 'c', 'base_revision_id': 'r',
                        'base_fingerprint': 'a' * 64,
                        'derived_assembly_ref': {'path': 'data/d.json',
                                                 'bytes': 2, 'sha256': 'b' * 64}}
            with patch.object(A, 'readonly_base', return_value=({}, {}, {})), \
                    patch.object(A, 'verified_ref', return_value=doc), \
                    patch.object(A.D, 'validate_files', return_value={'source_pixels_recomputed': False}):
                with self.assertRaisesRegex(ValueError, 'Full source reproduction required'):
                    A.build(contract=contract, root=root)


if __name__ == '__main__':
    unittest.main()
