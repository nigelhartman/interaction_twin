import unittest
import numpy as np
import trimesh
from native_appearance import load_native_glb
from native_compact_subset_v1 import rewrite_compact_subset
from charger_contact_gauge_20261104_v1 import scaled_pose


class CompactSubsetTests(unittest.TestCase):
    def test_real_glb_subset_retains_exact_geometry_color_winding_and_compacts_bounds(self):
        mesh=trimesh.creation.box();mesh.visual.vertex_colors=np.c_[np.arange(len(mesh.vertices))*20,np.full(len(mesh.vertices),80),np.full(len(mesh.vertices),130),np.full(len(mesh.vertices),255)]
        raw=mesh.export(file_type='glb');original=load_native_glb(raw);chosen=np.array([0,1,2],int)
        derived_raw,vertexmap,proof=rewrite_compact_subset(raw,chosen);derived=load_native_glb(derived_raw)
        self.assertTrue(np.array_equal(derived.vertices,original.vertices[vertexmap]));self.assertTrue(np.array_equal(derived.vertex_colors,original.vertex_colors[vertexmap]));self.assertEqual(len(derived.faces),3)
        self.assertTrue(np.array_equal(derived.vertices[derived.faces],original.vertices[original.faces[chosen]]));self.assertEqual(proof['added_vertices'],0);self.assertTrue(proof['native_images_and_nonattribute_binary_preserved'])

    def test_subset_cannot_duplicate_reorder_or_select_new_faces(self):
        raw=trimesh.creation.box().export(file_type='glb')
        for ids in ([1,1],[2,1],[-1,0],[0,999],[],[1.5,2.5]):
            with self.subTest(ids=ids),self.assertRaises(ValueError):rewrite_compact_subset(raw,np.asarray(ids))

    def test_camera_origin_gauge_changes_depth_and_size_without_altering_any_projection(self):
        vertices=np.array([[-.3,-.1,.1],[.1,.2,-.2],[.3,-.3,.2]])
        transform=np.eye(4);transform[:3,:3]*=.213;transform[:3,3]=[-.1,.3,1.7]
        before=vertices@transform[:3,:3].T+transform[:3,3];scale=1.2599428434865454;changed=scaled_pose(transform,scale)
        after=(vertices*scale)@changed[:3,:3].T+changed[:3,3]
        self.assertTrue(np.allclose(before[:,:2]/before[:,2:],after[:,:2]/after[:,2:],atol=1e-14));self.assertFalse(np.allclose(before,after))


if __name__=='__main__':unittest.main()
