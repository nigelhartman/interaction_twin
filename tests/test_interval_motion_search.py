import unittest
import numpy as np
import cv2
from native_appearance import load_native_glb, render_native_mask
from tests.test_native_appearance import fixture
from interval_motion_search import silhouette


class IntervalSearchProxyTests(unittest.TestCase):
    def compare(self,raw,k=None,size=(32,32)):
        mesh=load_native_glb(raw);before=mesh.vertices.copy();k=np.eye(3) if k is None else np.asarray(k,float)
        proxy=silhouette(mesh,np.eye(3),np.zeros(3),k,size)
        exact=render_native_mask(mesh,k,np.eye(3),np.zeros(3),*size)['mask']
        np.testing.assert_array_equal(mesh.vertices,before)
        return proxy,exact

    def test_full_triangle_proxy_is_bounded_near_exact_boundary(self):
        raw=fixture(parts=[{'vertices':[[3.3,4.2,1],[26.4,5.1,1],[5.2,25.5,1]],'uv':[[0,0],[1,0],[0,1]],'material':0}])
        proxy,exact=self.compare(raw)
        self.assertTrue(proxy.any());self.assertTrue(exact.any())
        # A sharp vertex can lie over two pixels from the nearest *interior
        # raster sample*. Check the known continuous triangle boundary instead.
        polygon=np.array([[3.3,4.2],[26.4,5.1],[5.2,25.5]],np.float32)
        ys,xs=np.where(proxy)
        outside=max(-cv2.pointPolygonTest(polygon,(float(x),float(y)),True) for x,y in zip(xs,ys))
        self.assertLessEqual(outside,1.5)
        self.assertTrue(cv2.dilate(proxy.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)[exact].all())

    def test_near_plane_intersection_is_clipped_not_dropped(self):
        raw=fixture(parts=[{'vertices':[[0,0,-1],[1,0,1],[0,1,1]],'uv':[[0,0],[1,0],[0,1]],'material':0}])
        proxy,exact=self.compare(raw,k=[[8,0,8],[0,8,8],[0,0,1]],size=(16,16))
        self.assertTrue(proxy.any());self.assertTrue(exact.any())

    def test_native_backface_culling_and_double_sided_are_preserved(self):
        part={'vertices':[[3,3,1],[20,3,1],[3,20,1]],'uv':[[0,0],[1,0],[0,1]],'material':0}
        raw=fixture(parts=[part],materials=[{'doubleSided':False,'pbrMetallicRoughness':{'baseColorTexture':{'index':0}}}])
        proxy,exact=self.compare(raw);self.assertFalse(proxy.any());self.assertFalse(exact.any())
        part['indices']=[[2],[1],[0]];raw=fixture(parts=[part],materials=[{'doubleSided':False,'pbrMetallicRoughness':{'baseColorTexture':{'index':0}}}])
        proxy,exact=self.compare(raw);self.assertTrue(proxy.any());self.assertTrue(exact.any())

    def test_overlapping_triangles_form_union_without_parity_holes(self):
        part={'vertices':[[3,3,1],[20,3,1],[3,20,1]],'uv':[[0,0],[1,0],[0,1]],'material':0}
        proxy,exact=self.compare(fixture(parts=[part,part]))
        self.assertTrue(proxy[7,7]);self.assertTrue(exact[7,7])


if __name__=='__main__':unittest.main()
