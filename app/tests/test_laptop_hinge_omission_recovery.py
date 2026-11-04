import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import laptop_hinge_omission_recovery_v1 as R


class HingeOmissionRecovery(unittest.TestCase):
    def fixture(self):
        joint = {'id': 'hinge', 'parent_id': 'base', 'child_id': 'lid', 'joint_type': 'revolute',
                 'attachment': 'persistent', 'confidence': 'medium', 'evidence_ids': ['e1'],
                 'hinge_hints': [], 'state_hints': [], 'rationale': 'Observed attached rotating part.',
                 'limitations': ['Estimated, not calibrated.']}
        cable = dict(copy.deepcopy(joint), id='cable', child_id='cable', joint_type='removable',
                     attachment='removable', evidence_ids=['e2'])
        raw = {'title': 'Software contract fixture', 'joints': [joint, cable],
               'limitations': ['Fallible visual proposal.'], 'physical_accuracy_verified': False}
        context = {'tracklets': [{'id': 'base', 'source_sample_range': [1, 20]},
                                 {'id': 'lid', 'source_sample_range': [1, 20]},
                                 {'id': 'cable', 'source_sample_range': [30, 40]}],
                   'images': [{'evidence_id': 'e1', 'role': 'source_rgb'}, {'evidence_id': 'e2', 'role': 'source_rgb'}],
                   'evidence': [{'evidence_id': 'e1', 'source_index': 5, 'observations': [{'track_id': 'base'}, {'track_id': 'lid'}]},
                                {'evidence_id': 'e2', 'source_index': 35, 'observations': [{'track_id': 'cable'}]}]}
        return raw, context

    def test_exact_whole_joint_omission_no_other_value_or_source_changes(self):
        raw, context = self.fixture()
        before = copy.deepcopy((raw, context))
        projected, kept, removed = R.minus_only(raw, context)
        self.assertEqual(kept, [0])
        self.assertEqual(removed[0]['original_index'], 1)
        self.assertEqual(removed[0]['joint'], raw['joints'][1])
        self.assertEqual(projected, dict(raw, joints=[raw['joints'][0]]))
        self.assertEqual(before, (raw, context))

    def test_remaining_joint_text_state_or_hint_mutation_rejected(self):
        raw, context = self.fixture()
        for key, value in [('rationale', 'Invented improvement'), ('confidence', 'high'),
                           ('state_hints', [{'evidence_id': 'e1', 'state': 'open'}])]:
            result = copy.deepcopy(raw)
            result['joints'] = result['joints'][:1]
            result['joints'][0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'exact minus-only'):
                R.verify_derivative(raw, result, context)

    def test_no_new_joint_or_top_level_copy_edit(self):
        raw, context = self.fixture()
        for field, value in [('title', 'New title'), ('limitations', ['Changed limits.'])]:
            result = dict(copy.deepcopy(raw), joints=[copy.deepcopy(raw['joints'][0])])
            result[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'exact minus-only'):
                R.verify_derivative(raw, result, context)
        with self.assertRaisesRegex(ValueError, 'exact minus-only'):
            R.verify_derivative(raw, raw, context)

    def test_foreign_evidence_or_other_invalid_relationship_not_salvaged(self):
        raw, context = self.fixture()
        raw['joints'][0]['evidence_ids'] = ['unknown']
        with self.assertRaisesRegex(ValueError, 'known scope-only'):
            R.minus_only(raw, context)

    def test_extending_tracklet_scope_does_not_count_as_omission_recovery(self):
        raw, context = self.fixture()
        context['tracklets'][0]['source_sample_range'] = [1, 40]
        context['evidence'][1]['observations'].append({'track_id': 'base'})
        with self.assertRaisesRegex(ValueError, 'Valid source responses'):
            R.minus_only(raw, context)

    def test_changed_fixed_original_bytes_stop_before_accounting_or_client(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            prepared = root / R.PREPARED
            prepared.mkdir(parents=True)
            (prepared / 'plan.json').write_text('{}')
            with patch.object(R, 'settled_proof') as ledger:
                with self.assertRaisesRegex(ValueError, 'original request/result bytes'):
                    R.original(root)
            ledger.assert_not_called()

    def test_uncertain_receipt_cannot_be_omitted_into_success(self):
        with self.assertRaisesRegex(ValueError, 'accounted original call'):
            R.settled_proof({'status': 'sent_charge_pending'}, Path('/unused'))

    def test_symlink_or_traversal_paths_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root / 'data').mkdir()
            (root / 'data/a').symlink_to(root / 'data')
            for relative in ['data/../thing', 'data/a/missing.json']:
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    R.local(relative, root, exists=False)


if __name__ == '__main__':
    unittest.main()
