import unittest
from decimal import Decimal
from pathlib import Path
from increment_budget import remaining_increment
import pipeline
import rigid_discovery


class RigidDiscoveryTests(unittest.TestCase):
    def test_increment_counts_uncertainty_and_does_not_reset_baseline(self):
        plan = {'budget_baseline': {'settled_usd': '1.98', 'reserved_usd': '2.72'},
                'additional_estimated_api_ceiling_usd': '20'}
        self.assertEqual(remaining_increment(plan, {'settled_usd': '3.00', 'reserved_usd': '4.00'}), Decimal('17.70'))
        with self.assertRaises(ValueError):
            remaining_increment(plan, {'settled_usd': '0', 'reserved_usd': '0'})

    def test_rigid_prompt_keeps_category_selection_in_model(self):
        text = rigid_discovery.prompt_text([{'index': 1, 'timestamp_s': 0, 'width': 960, 'height': 540}])
        self.assertIn('visibly move relative', text)
        self.assertIn('Deformable', text)
        self.assertIn('actual sample metadata', text)
        self.assertNotIn('Jabra', text)
        self.assertNotIn('earbud', text)

    def test_new_profile_dispatch_is_explicit_and_old_profile_unchanged(self):
        original = pipeline.load_profile('meta_sam31_video_low')
        self.assertEqual(pipeline.reasoning_dispatch(original, 'discover')[0], 'stage2_meta_reason.py')
        rigid = pipeline.load_profile('meta_sam31_parts')
        self.assertEqual(pipeline.reasoning_dispatch(rigid, 'discover')[0], 'discover_rigid_components.py')
        self.assertEqual(pipeline.reasoning_dispatch(rigid, 'audit')[0], 'stage2_meta_reason.py')

    def test_every_paid_new_profile_stage_uses_increment_envelope(self):
        profile = pipeline.load_profile('meta_sam31_parts')
        paid = ['discover_rigid_components.py', 'stage2_meta_reason.py', 'stage2_meta_video_segment.py',
                'stage3_meta_select_track_views.py', 'stage5_meta_propose.py']
        for script in paid:
            argv = pipeline.stage_argv(profile, 'python', script, ['--case', 'example'])
            self.assertEqual(Path(argv[1]).name, 'run_budgeted_stage.py')
            self.assertEqual(argv[2:], [script, '--case', 'example'])
        self.assertEqual(Path(pipeline.stage_argv(profile, 'python', 'stage3_triposr.py', [])[1]).name, 'stage3_triposr.py')
        self.assertEqual(Path(pipeline.stage_argv(pipeline.load_profile('meta_sam31_video_low'), 'python', paid[0], [])[1]).name, paid[0])


if __name__ == '__main__':
    unittest.main()
