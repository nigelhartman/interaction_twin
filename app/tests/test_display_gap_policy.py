"""Run the actual JavaScript display policy; recorded-state samplers stay separate."""
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('node'), 'Existing Node runtime required; no install')
class DisplayGapPolicyTests(unittest.TestCase):
    def test_display_math_and_fail_closed_evidence_boundaries(self):
        result = subprocess.run([shutil.which('node'), '--test', 'tests/test_display_gap_policy.cjs'],
                                cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
