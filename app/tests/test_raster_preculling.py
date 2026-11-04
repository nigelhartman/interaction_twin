"""Accelerated native rasterization must match the original all-face loop exactly."""
import unittest
from unittest import mock
import numpy as np
from scipy.spatial.transform import Rotation

import alignment_render as R
import native_appearance as N
from tests.test_native_appearance import fixture


def all_faces(points,faces,*args,**kwargs):return np.arange(len(faces))


class RasterPrecullingTests(unittest.TestCase):
    def equal(self,fast,original):
        for name in ['mask','depth','rgb','face_index']:
            if fast[name]is None:self.assertIsNone(original[name])
            else:np.testing.assert_array_equal(fast[name],original[name],err_msg=name+' changed')

    def test_random_perspective_near_clipping_and_rotated_views_remain_pixel_exact(self):
        rng=np.random.default_rng(431);vertices=rng.uniform([-3,-3,-.5],[3,3,4],size=(90,3));faces=np.arange(90).reshape(-1,3)
        colors=rng.integers(0,256,size=(90,3),dtype=np.uint8)
        for width,height in [(17,13),(64,48),(129,191)]:
            k=np.asarray([[width*.7,width*.03,width*.51],[0,height*.8,height*.47],[0,0,1.]])
            for angles in [(0,0,0),(.3,-.5,.1),(-.6,.2,1.)]:
                rotation=Rotation.from_euler('xyz',angles).as_matrix();translation=np.asarray([.05,-.1,.7])
                fast=R.render_mesh(vertices,faces,k,rotation,translation,width,height,scale=.7,vertex_colors=colors)
                with mock.patch.object(R,'raster_triangle_candidates',side_effect=all_faces):
                    original=R.render_mesh(vertices,faces,k,rotation,translation,width,height,scale=.7,vertex_colors=colors)
                self.equal(fast,original)

    def test_native_textures_culling_and_original_face_offsets_remain_exact(self):
        parts=[{'vertices':[[100,100,1],[101,100,1],[100,101,1]],'uv':[[0,0],[1,0],[0,1]],'material':0},
            {'vertices':[[.1,.1,1],[.9,.1,1],[.1,.9,1]],'uv':[[0,0],[1,0],[0,1]],'material':0},
            {'vertices':[[0,0,1],[8,0,1],[0,8,1]],'uv':[[0,0],[1,0],[0,1]],'material':0}]
        mesh=N.load_native_glb(fixture(parts));fast=N.render_native_mesh(mesh,np.eye(3),np.eye(3),np.zeros(3),12,12)
        with mock.patch.object(N,'raster_triangle_candidates',side_effect=all_faces):
            original=N.render_native_mesh(mesh,np.eye(3),np.eye(3),np.zeros(3),12,12)
        self.equal(fast,original);self.assertEqual(fast['face_index'][2,2],2)

    def test_integer_boundaries_subpixel_faces_and_depth_ties_keep_original_samples(self):
        vertices=np.asarray([[.1,.1,1],[.9,.1,1],[.1,.9,1],[-1e-10,0,1],[4,0,1],[0,4,1],[-1,-1,-1],[2,-1,1],[-1,2,1]],float)
        faces=np.asarray([[0,1,2],[3,4,5],[3,4,5],[6,7,8]])
        k=np.eye(3);fast=R.render_mesh(vertices,faces,k,np.eye(3),np.zeros(3),8,8)
        with mock.patch.object(R,'raster_triangle_candidates',side_effect=all_faces):original=R.render_mesh(vertices,faces,k,np.eye(3),np.zeros(3),8,8)
        self.equal(fast,original)
        self.assertNotIn(0,R.raster_triangle_candidates(vertices,faces,k,8,8));self.assertIn(3,R.raster_triangle_candidates(vertices,faces,k,8,8))

    def test_large_face_batches_keep_original_global_indices(self):
        points=np.asarray([[.1,.1,1],[.9,.1,1],[.1,.9,1],[0,0,1],[4,0,1],[0,4,1.]])
        faces=np.repeat([[0,1,2]],70001,axis=0);faces[65536]=[3,4,5];faces[-1]=[3,4,5]
        selected=R.raster_triangle_candidates(points,faces,np.eye(3),8,8)
        np.testing.assert_array_equal(selected,[65536,70000])


if __name__=='__main__':unittest.main()
