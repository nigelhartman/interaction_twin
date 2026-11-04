"""The rehearsal may only clone explicit non-secret evidence into its temp root."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import admin_workflow_rehearsal_20261103 as R


class AdminRehearsalSafetyTests(unittest.TestCase):
    def test_hidden_external_and_traversal_inputs_refused_before_any_hash(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            with patch.object(R, '_sha', side_effect=AssertionError('No forbidden file may be hashed')):
                for relative in ['.secrets/meta.env', '.private/admin-workflow/state.json',
                                 '/etc/passwd', 'data/../outside.txt', 'data/.private/secret',
                                 'data\\unexpected', 'arbitrary.py', 'annotations/secret.env']:
                    with self.subTest(relative=relative), self.assertRaises(ValueError):
                        R.confined_evidence(root, relative)

    def test_declared_ordinary_paths_and_exact_compiler_inputs_are_allowed(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            for relative in ['data/capture/source.mp4', 'annotations/proposal.json',
                             'stage5.py', 'scripts/stage5_compile.py', 'schemas/interaction.schema.json']:
                self.assertEqual(R.confined_evidence(root, relative), root / relative)

    def test_symlink_evidence_is_refused(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            (root / 'data').mkdir()
            (root / 'data/linked').symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                R.confined_evidence(root, 'data/linked/evidence.json')

    def test_deferred_scheduler_cannot_dispatch_model_work(self):
        scheduled = R.DeferredScheduler()
        with patch.object(R, '_sha', side_effect=AssertionError('Never execute deferred callbacks')) as callback:
            scheduled.submit(callback, 'argument')
            scheduled.shutdown(wait=True)
            callback.assert_not_called()
        self.assertEqual(len(scheduled.calls), 1)


if __name__ == '__main__': unittest.main()
