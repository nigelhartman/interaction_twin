import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import jsonschema
import laptop_hinge_relocalization_v1 as L
from tests import test_laptop_hinge_omission_recovery as F


class SourceOnlyRelocalization(unittest.TestCase):
    def result(self):
        raw, _ = F.HingeOmissionRecovery().fixture()
        joint = raw['joints'][0]
        joint.update(id='j_laptop_base_laptop_lid', parent_id='laptop_base', child_id='laptop_lid')
        return dict(raw, joints=[joint])

    def test_schema_only_same_laptop_pair_not_cable_or_other_relations(self):
        result = self.result()
        jsonschema.validate(result, L.schema())
        for key, value in [('child_id', 'power_cable'), ('parent_id', 'elsewhere'), ('joint_type', 'removable')]:
            bad = copy.deepcopy(result)
            bad['joints'][0][key] = value
            with self.subTest(key=key), self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(bad, L.schema())

    def test_unknown_is_allowed_without_forcing_a_hinge(self):
        result = self.result()
        result['joints'][0].update(joint_type='unknown', attachment='uncertain')
        jsonschema.validate(result, L.schema())

    def test_explicit_plan_hash_required_before_client_dispatch(self):
        client = Mock()
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name).resolve()
            (folder / 'plan.json').write_text('{}')
            with patch.object(L, 'verified_plan', return_value=({}, {}, folder)):
                with self.assertRaisesRegex(ValueError, 'exact packet authorization'):
                    L.run(authorized_plan_sha256='wrong', allow_paid_api=True, client=client)
        client.json_result.assert_not_called()

    def test_existing_attempt_blocks_any_repeated_call_even_with_new_authorization(self):
        client = Mock()
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name).resolve()
            (folder / 'plan.json').write_text('{}')
            (folder / 'attempt').mkdir()
            with patch.object(L, 'verified_plan', return_value=({}, {}, folder)):
                with self.assertRaisesRegex(ValueError, 'Once-only'):
                    L.run(authorized_plan_sha256=L.M.sha(folder / 'plan.json'), allow_paid_api=True, client=client)
        client.json_result.assert_not_called()


if __name__ == '__main__':
    unittest.main()
