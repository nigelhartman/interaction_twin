"""Prospective native-unit invariance and unchanged rejection-gate checks."""
import copy
import unittest
import numpy as np
from tests.test_assembly_scale_feasibility import rows
import assembly_scale_feasibility as old
import assembly_scale_feasibility_v2 as current


class GaugeInvariantAssemblyTests(unittest.TestCase):
    def test_only_excitation_definition_changes_the_fixed_gate_values(self):
        old_policy, new_policy = copy.deepcopy(old.POLICY), copy.deepcopy(current.POLICY)
        old_policy.pop('version'); new_policy.pop('version')
        new_policy.pop('excitation_definition'); new_policy.pop('excitation_is_necessary_not_sufficient')
        self.assertEqual(old_policy, new_policy)

    def test_rescaling_only_b_native_units_preserves_gate_and_physical_hypothesis(self):
        original, scale, offset, rotation = rows()
        before = current.fit_compatibility(original)
        for factor in [.01, 100.]:
            changed = copy.deepcopy(original)
            for row in changed: row['b_t'] = row['b_t'] * factor
            after = current.fit_compatibility(changed)
            self.assertEqual(before['status'], after['status']); self.assertEqual(before['reasons'], after['reasons'])
            self.assertAlmostEqual(before['diagnostic_fit']['translation_excitation_over_anchor_depth'], after['diagnostic_fit']['translation_excitation_over_anchor_depth'], places=12)
            self.assertAlmostEqual(after['diagnostic_fit']['scale_b_in_a_native_units'], scale / factor, places=10)
            np.testing.assert_allclose(after['diagnostic_fit']['offset_b_origin_in_a_native_units'], offset, atol=1e-10)
            transform = np.array(after['common_frame_transform_a_from_b'])
            np.testing.assert_allclose(transform[:3, :3] * factor, scale * rotation, atol=1e-10)

    def test_rescaling_only_a_native_units_preserves_gate_and_transform_equivalence(self):
        original, scale, offset, rotation = rows(); before = current.fit_compatibility(original)
        for factor in [.01, 100.]:
            changed = copy.deepcopy(original)
            for row in changed: row['a_t'] = row['a_t'] * factor
            after = current.fit_compatibility(changed)
            self.assertEqual(before['status'], after['status']); self.assertEqual(before['reasons'], after['reasons'])
            self.assertAlmostEqual(before['diagnostic_fit']['translation_excitation_over_anchor_depth'], after['diagnostic_fit']['translation_excitation_over_anchor_depth'], places=12)
            self.assertAlmostEqual(after['diagnostic_fit']['scale_b_in_a_native_units'], scale * factor, places=10)
            np.testing.assert_allclose(after['diagnostic_fit']['offset_b_origin_in_a_native_units'], offset * factor, atol=1e-10)
            np.testing.assert_allclose(np.array(after['common_frame_transform_a_from_b'])[:3, :3] / factor, scale * rotation, atol=1e-10)

    def test_low_excitation_stays_rejected_in_either_native_gauge(self):
        original, *_ = rows()
        for i, r in enumerate(original):
            r['a_R'] = r['b_R'] = np.eye(3)
            r['a_t'] = np.array([.001 * i, 0., 3.])
            r['b_t'] = (r['a_t'] + [.2, 0., .1]) / .35
        before = current.fit_compatibility(original)
        self.assertIn('insufficient_translation_excitation', before['reasons'])
        for a_scale, b_scale in [(100., 1.), (1., 100.), (.01, 100.)]:
            changed = copy.deepcopy(original)
            for r in changed: r['a_t'] *= a_scale; r['b_t'] *= b_scale
            after = current.fit_compatibility(changed)
            self.assertEqual(before['reasons'], after['reasons'])
            self.assertIsNone(after['common_frame_transform_a_from_b'])

    def test_reserved_values_do_not_change_corrected_excitation_or_fit(self):
        original, *_ = rows(); before = current.fit_compatibility(original)
        for row in original:
            if row['reserved']: row['a_t'] += [2., .3, 0.]
        after = current.fit_compatibility(original)
        a, b = copy.deepcopy(before['diagnostic_fit']), copy.deepcopy(after['diagnostic_fit'])
        a.pop('fit_status'); b.pop('fit_status'); self.assertEqual(a, b)
        self.assertIn('reserved_relative_translation_inconsistent', after['reasons'])


if __name__ == '__main__': unittest.main()
