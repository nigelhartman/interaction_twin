import importlib.util,tempfile,unittest
from pathlib import Path
import numpy as np

@unittest.skipUnless(importlib.util.find_spec('trimesh') and importlib.util.find_spec('cv2'), 'Requires local geometry dependencies')
class BatteryCompletionTests(unittest.TestCase):
    def test_closed_asset_has_correct_units_volume_and_reloadable_topology(self):
        import trimesh
        from scripts.stage3_battery_fit import closed_battery_mesh
        for length in [2.5,4.2]:
            mesh=closed_battery_mesh(.5,length)
            self.assertTrue(mesh.is_watertight)
            self.assertTrue(mesh.is_winding_consistent)
            self.assertEqual(mesh.euler_number,2)
            np.testing.assert_allclose(mesh.extents,[1,1,length])
            self.assertLess(abs(mesh.volume/(np.pi*.5**2*length)-1),.01)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'battery.glb';mesh.export(path)
                loaded=trimesh.load(path,force='mesh')
                self.assertTrue(loaded.is_watertight)
                np.testing.assert_allclose(loaded.extents,mesh.extents,rtol=1e-6)

if __name__=='__main__':unittest.main()
