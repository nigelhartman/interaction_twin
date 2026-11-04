import json,subprocess,unittest
from pathlib import Path

class ComponentSelectionTests(unittest.TestCase):
 def test_actual_workspace_component_identity_and_permalink(self):
  root=Path(__file__).resolve().parents[1]
  run=subprocess.run(['node',str(root/'tests/component_selection_harness.cjs')],cwd=root,capture_output=True,text=True,timeout=20)
  self.assertEqual(run.returncode,0,run.stdout+run.stderr)
  self.assertEqual(json.loads(run.stdout)['status'],'passed')

if __name__=='__main__':unittest.main()
