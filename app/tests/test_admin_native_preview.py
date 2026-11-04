"""Owner-only native inspection contracts and real desk asynchronous selection."""
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeOwnerPreviewTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Existing Node runtime required; no installation')
    def test_private_ref_revision_hash_and_async_selection_contracts(self):
        result = subprocess.run([shutil.which('node'), '--test', str(ROOT / 'tests/test_admin_native_preview.cjs')],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
