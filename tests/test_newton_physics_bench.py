"""Independent ballistic/contact numerical checks of the actual engine replay."""
import hashlib
import json
from pathlib import Path
import unittest
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'showcase/public/physics/newton-contact-20261104-v1.json'

class NewtonPhysicsBenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not RESULT.exists():raise unittest.SkipTest('Optional local physics execution artifact unavailable')
        cls.d=json.loads(RESULT.read_text())
    def test_actual_cpu_engine_timing_and_unsupported_claims(self):
        d=self.d
        self.assertEqual(d['engine']['name'],'Newton')
        self.assertEqual(d['engine']['device'],'cpu')
        self.assertEqual(d['engine']['steps'],2400)
        self.assertEqual(len(d['frames']),241)
        self.assertTrue(d['claims']['real_physics_engine_execution'])
        for k in ['metric_scale_measured','material_parameters_measured','physical_accuracy_verified','robot_grasp_validated','simulation_ready']:
            self.assertFalse(d['claims'][k])
    def test_initial_free_fall_matches_gravity_before_any_contact(self):
        frames=self.d['frames']
        self.assertEqual(frames[5]['contact_count'],0)
        elapsed=frames[5]['time_s'];dt=self.d['engine']['timestep_s']
        z0=np.asarray(frames[0]['poses_xyz_xyzw'])[:,2]
        z5=np.asarray(frames[5]['poses_xyz_xyzw'])[:,2]
        expected=z0-.5*9.81*elapsed*(elapsed+dt)
        np.testing.assert_allclose(z5,expected,atol=2e-7)
        np.testing.assert_allclose(np.asarray(frames[5]['velocities'])[:,2],-9.81*elapsed,atol=2e-6)
    def test_every_transformed_collision_vertex_stays_above_plane(self):
        lowest=1.
        for f in self.d['frames']:
            for o,p in zip(self.d['objects'],f['poses_xyz_xyzw']):
                x,y,z,w=p[3:]
                R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
                v=np.asarray(o['collision_proxy']['vertices'])@R.T+np.array(p[:3])
                self.assertTrue(np.isfinite(v).all());lowest=min(lowest,float(v[:,2].min()))
        self.assertGreaterEqual(lowest,-.0005)
        self.assertAlmostEqual(lowest,self.d['execution']['minimum_collision_vertex_height_m'],places=8)
    def test_contacts_and_settling_are_computed_for_every_body(self):
        self.assertGreater(self.d['execution']['contact_steps'],2000)
        f=self.d['frames'][-1]
        self.assertGreater(f['contact_count'],0)
        self.assertLess(np.linalg.norm(np.asarray(f['velocities'])[:,:3],axis=1).max(),.001)
        self.assertTrue(all(p[2]<.03 for p in f['poses_xyz_xyzw']))
        self.assertEqual(len(f['contact_points_xyz']),f['contact_count'])
    def test_native_assets_and_starter_bytes_are_unchanged(self):
        for o in self.d['objects']:
            r=o['asset_ref'];p=ROOT/r['path']
            self.assertEqual(p.stat().st_size,r['bytes'])
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),r['sha256'])
        archive=ROOT/'showcase/media/interaction-twin-newton-starter-20261104-v1.zip'
        with zipfile.ZipFile(archive)as z:
            m=json.loads(z.read('Interaction-Twin-Newton/manifest.json'))
            for r in m['files']:
                raw=z.read('Interaction-Twin-Newton/'+r['path'])
                self.assertEqual(hashlib.sha256(raw).hexdigest(),r['sha256'])
                self.assertEqual(len(raw),r['bytes'])
            self.assertTrue(all(not any(x.startswith('.')or x=='..' for x in p.split('/'))for p in z.namelist()))
