"""Run actual owner-only legacy fetch wrapper trust/no-resend behavior."""
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]

class AdminFetchTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'),'Existing Node runtime required; no installation')
    def test_actual_csrf_wrapper_origin_owner_and_single_send(self):
        result=subprocess.run([shutil.which('node'),'--test',str(ROOT/'tests/test_admin_fetch.cjs')],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_owner_workspace_panels_load_the_helper(self):
        for name in ['index.html','geometry-lab.html','motion-lab.html','upgrades.html','calibration-lab.html','robotics.html']:
            self.assertIn('admin-fetch.js',(ROOT/'workspace'/name).read_text(),name)

if __name__=='__main__':unittest.main()
