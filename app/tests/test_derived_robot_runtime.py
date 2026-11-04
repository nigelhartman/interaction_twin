"""MCP robot reads keep exact reviewed staging without image/CV packages."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DerivedRobotRuntimeTests(unittest.TestCase):
    def test_light_runtime_validates_historical_plans_against_their_own_frozen_bindings(self):
        fixture = ROOT/'output/submission-finalization-20261103/jabra/robot-plans-private-probe-v3.json'
        snapshot = ROOT/'output/submission-finalization-20261103/judge-preview-final-v1/Interaction-Twin-judge-preview'
        source_archive = ROOT/'output/submission-polish-20261104-v1/readiness/source-review-v7/final-v7/Interaction-Twin-source-review-20261104-v7.zip'
        if not fixture.is_file() or not (snapshot/'bundle-manifest.json').is_file() or not source_archive.is_file():
            self.skipTest('Reviewed local historical plan/snapshot fixture is unavailable')
        interpreter = ROOT/'.venv-mcp/bin/python'
        script = r'''import builtins,hashlib,json,time,types,sys,zipfile
from pathlib import Path
original = builtins.__import__
def guarded(name,*args,**kwargs):
 if name.split('.')[0] in {'cv2','PIL','scipy','native_appearance','refined_assembly_contract'}:
  raise RuntimeError('Unexpected CV/appearance dependency: '+name)
 return original(name,*args,**kwargs)
builtins.__import__ = guarded
from curated_twin_catalog import CuratedTwinCatalog
# Historical plans execute their original source runtime. They are not silently
# rebound to a new staging adapter or migrated to today's task schema.
source_archive=Path('output/submission-polish-20261104-v1/readiness/source-review-v7/final-v7/Interaction-Twin-source-review-20261104-v7.zip')
archive_bytes=source_archive.read_bytes()
assert hashlib.sha256(archive_bytes).hexdigest()=='3d7af8f1b2711972798ae1961d0faff1842b619fb165c9bf20855a41cf28c60b'
with zipfile.ZipFile(source_archive)as source_zip:
 for name,expected in [('robot_task_geometry','7870cd81400f02f54a673fe27b2ab0a9f2f8ebcf4139971498e34a17bddd1133'),('robot_task_program','999eec6a875484064113a80d73078b82751d12ff107a8af7482ad6e894eaf01f')]:
  payload=source_zip.read('Interaction-Twin-source-review/'+name+'.py');assert hashlib.sha256(payload).hexdigest()==expected
  module=types.ModuleType(name);module.__file__=str(Path.cwd()/(name+'.py'));sys.modules[name]=module
  exec(compile(payload,module.__file__,'exec'),module.__dict__)
R=sys.modules['robot_task_program']
rows = json.loads(Path('output/submission-finalization-20261103/jabra/robot-plans-private-probe-v3.json').read_text())['plans']
archive=Path('output/submission-finalization-20261103/judge-preview-final-v1/Interaction-Twin-judge-preview')
raw=(archive/'bundle-manifest.json').read_bytes()
assert len(raw)==927114 and hashlib.sha256(raw).hexdigest()=='2f000030ae7bfdf3ee1175be4190ebe79193a8a91c030ad8652fc0bf191f4541', 'Historical snapshot manifest changed'
bindings={r['path']:r for r in json.loads(raw)['current_curated_snapshot_refs']}
class FrozenCatalog(CuratedTwinCatalog):
 def _json(self,relative,maximum=8*1024*1024):
  pointer=bindings[relative];raw=(archive/relative).read_bytes()
  assert len(raw)<=maximum and len(raw)==pointer['bytes'] and hashlib.sha256(raw).hexdigest()==pointer['sha256'], 'Historical catalog binding changed'
  return json.loads(raw),raw
catalog = FrozenCatalog();counts=[];start=time.perf_counter()
for row in rows:
 historical=row['plan']
 assert historical['status']=='ready' and R.validate_robot_task(historical,catalog=catalog)['valid'], 'Historical source-bound program no longer validates'
 assert historical['source_binding']['object_program_ref']==catalog.derived_assembly(historical['twin_id'])['artifact'], 'Historical plan silently rebound to another derivative'
 counts.append(len(historical['actions']))
print(json.dumps({'action_counts':counts,'seconds':time.perf_counter()-start}))
'''
        completed = subprocess.run([str(interpreter) if interpreter.is_file() else sys.executable,
                                    '-c', script], cwd=ROOT, check=True,
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(json.loads(completed.stdout)['action_counts'], [40, 27, 29, 16])

    def test_current_curated_generic_derivative_builds_a_separate_ready_forty_action_plan(self):
        if not (ROOT/'showcase/public/library-catalog.json').is_file():
            self.skipTest('Current curated library is unavailable')
        interpreter = ROOT/'.venv-mcp/bin/python'
        script = r'''import builtins,json
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name.split('.')[0] in {'cv2','PIL','scipy','native_appearance','refined_assembly_contract'}:
  raise RuntimeError('Unexpected CV/appearance dependency: '+name)
 return original(name,*args,**kwargs)
builtins.__import__=guarded
from robot_task_catalog import RobotTaskCatalog
c=RobotTaskCatalog();rows,_=c.catalog._snapshot()
r=next(r for r in rows if {'case_base','case_lid','earbud_a','earbud_b'}<={p['id']for p in r['components']})
d=c.catalog.derived_assembly(r['id'])
assert d['document']['format']=='interaction-twin-declared-geometry-derivative'
p=c.plan(r['id'],'Place the case on the table, open the lid, take out both earbuds and place them on the tray, then close it.')
assert p['status']=='ready' and len(p['actions'])==40 and c.validate(p)['valid']
assert p['source_binding']['object_program_ref']==d['artifact']
print(json.dumps({'status':p['status'],'action_count':len(p['actions']),'current_derived_ref':d['artifact']}))
'''
        completed = subprocess.run([str(interpreter) if interpreter.is_file() else sys.executable,
                                    '-c', script], cwd=ROOT, check=True,
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(json.loads(completed.stdout)['action_count'], 40)


if __name__ == '__main__':
    unittest.main()
