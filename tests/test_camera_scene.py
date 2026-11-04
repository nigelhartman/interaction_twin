import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from alignment_render import ray_mesh_points
from stage4_camera_scene import shell, BAY_CENTRES, BAY_FLOOR_Y, DIMENSIONS, plane_pose, SCALE


class CameraSceneTest(unittest.TestCase):
    def test_blind_battery_wells_are_real_open_geometry(self):
        mesh=shell()
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_winding_consistent)
        self.assertEqual(mesh.euler_number,2)
        r=Rotation.from_euler('x',-90,degrees=True).as_matrix()
        # Centre the viewing ray over the first well so it reaches the floor
        # rather than hitting the physically correct near well wall.
        t=np.array([-BAY_CENTRES[0],0.,10.])
        k=np.array([[300.,0,200],[0,300.,200],[0,0,1]])
        xyz=np.array([[BAY_CENTRES[0],DIMENSIONS[1]/2,0],[0,DIMENSIONS[1]/2,0]])
        camera=xyz@r.T+t
        uv=camera@k.T;uv=uv[:,:2]/uv[:,2:]
        hit,valid=ray_mesh_points(uv,mesh.vertices,mesh.faces,k,r,t)
        self.assertTrue(valid.all())
        self.assertAlmostEqual(hit[0,1],BAY_FLOOR_Y,places=7)
        self.assertAlmostEqual(hit[1,1],DIMENSIONS[1]/2,places=7)

    def test_known_rear_plane_pose_reprojects_after_positive_depth_fit(self):
        # A rear plane offset from the origin catches a poor IPPE-only local
        # minimum observed in real rear-face keyframes on this OpenCV version.
        xyz=np.array([[3.,-2.,1.],[-3.,-2.,1.],[-3.,2.,1.],[3.,2.,1.]])
        r=Rotation.from_euler('xyz',[.13,2.9,.08]).as_matrix();t=np.array([-.02,.01,.4])
        k=np.array([[1027.,0,479.5],[0,1027.,269.5],[0,0,1.]])
        camera=xyz*SCALE@r.T+t
        uv=camera@k.T;uv=uv[:,:2]/uv[:,2:]
        best,_=plane_pose(xyz,uv,k)
        self.assertLess(best[0],1e-5)
        self.assertTrue(np.all((xyz*SCALE@best[1].T+best[2])[:,2]>0))


if __name__=='__main__':
    unittest.main()
