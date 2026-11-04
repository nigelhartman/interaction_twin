import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

import native_assembly_refinement_v1 as R
from native_appearance import load_native_glb
from tests.test_native_appearance import fixture


class GeometricMatingTests(unittest.TestCase):
    def shell(self,a=.5,b=.25,depth=.3):
        theta=np.arange(128*12)*2*np.pi/(128*12)
        height=np.linspace(-depth/2,depth/2,15)
        return np.array([[a*np.cos(t),b*np.sin(t),h]for t in theta for h in height])

    def test_whole_rim_fit_improves_independent_shells_and_is_bounded(self):
        parent=self.shell();child=self.shell(b=.225,depth=.2)
        pf=R.opening_frame(parent,[0,0,1]);cf=R.opening_frame(child,[0,0,1])
        pp=R.rim_profile(parent,pf);cp=R.rim_profile(child,cf)
        original=np.eye(4);original[:3,:3]=np.diag([1,-1,-1]);original[:3,3]=[.04,.02,.3]
        closed=R.similarity_closed(pf,cf,pp,cp,original)
        vertices,receipt=R.derive_warp(parent,child,pf,cf,pp,cp,closed)
        before=R.rim_metrics(pp['points'],R.transform(cp['points'],original),1.)
        after=R.rim_metrics(pp['points'],R.transform(vertices[[r['vertex_id']for r in cp['rows']]],closed),1.)
        self.assertLess(after['symmetric_p95_distance'],before['symmetric_p95_distance']*.25)
        self.assertLess(receipt['maximum_displacement_fraction_parent_width'],R.MAX_DISPLACEMENT_FRACTION)
        self.assertEqual(receipt['vertex_count_unchanged'],len(child))
        np.testing.assert_allclose(child,self.shell(b=.225,depth=.2))

    def test_whole_rim_metric_cannot_be_satisfied_by_one_anchor(self):
        t=np.linspace(0,2*np.pi,128,endpoint=False);a=np.c_[np.cos(t),np.sin(t),t*0]
        b=a.copy();b[1:,2]=.08
        self.assertEqual(np.linalg.norm(a[0]-b[0]),0)
        metric=R.rim_metrics(a,b,2.)
        self.assertGreater(metric['symmetric_p95_distance'],.079)
        self.assertEqual(metric['parent_sample_count'],128)

    def test_two_endpoint_test_detects_wrong_axis_despite_one_fixed_anchor(self):
        parent=np.array([[-.3,0,0],[.3,0,0],[0,.1,0]])
        child=parent.copy();closed=np.eye(4)
        segment=[parent[0],parent[1]]
        bad=R.seam_metrics(parent,child,closed,parent[0],[0,0,1],[0,1.2],segment)
        good=R.seam_metrics(parent,child,closed,[0,0,0],[1,0,0],[0,1.2],segment)
        self.assertLess(bad['samples'][1]['endpoint_distances'][0],1e-12)
        self.assertGreater(bad['samples'][1]['endpoint_distances'][1],.5)
        self.assertLess(good['maximum_across_angles_and_both_endpoints'],1e-12)

    def test_missing_opening_support_or_extreme_shells_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Resolved'):R.opening_frame(np.zeros((20,3)),[0,1,0])
        parent=self.shell();child=self.shell(a=.2,b=.1)
        pf=R.opening_frame(parent,[0,0,1]);cf=R.opening_frame(child,[0,0,1])
        with self.assertRaisesRegex(ValueError,'scale mismatch'):
            R.similarity_closed(pf,cf,R.rim_profile(parent,pf),R.rim_profile(child,cf),np.eye(4))

    def test_triangle_orientation_gate_measures_real_deformation(self):
        a=np.array([[0,0,0],[1,0,0],[0,1,0]],float);f=np.array([[0,1,2]])
        self.assertEqual(R.triangle_quality(a,a,f)['orientation_reversals'],0)
        b=a.copy();b[2,1]=-1
        self.assertEqual(R.triangle_quality(a,b,f)['orientation_reversals'],1)


class GLBDerivativeTests(unittest.TestCase):
    def raw(self):
        return fixture([{'vertices':[[0,0,0],[1,0,0],[0,1,0]],'material':0}],materials=[{'doubleSided':True}],images=[])

    def test_position_only_writer_preserves_topology_and_original_bytes(self):
        raw=self.raw();before=bytes(raw);mesh=load_native_glb(raw)
        v=mesh.vertices.copy();v[:,0]*=1.05
        changed,receipt=R.rewrite_positions(raw,v)
        derivative=load_native_glb(changed)
        self.assertEqual(raw,before);self.assertNotEqual(changed,raw)
        np.testing.assert_array_equal(mesh.faces,derivative.faces)
        np.testing.assert_array_equal(mesh.vertex_colors,derivative.vertex_colors)
        np.testing.assert_allclose(derivative.vertices,v,atol=1e-7)
        self.assertEqual(receipt['original_nonposition_binary_sha256'],receipt['unchanged_nonposition_binary_sha256'])
        self.assertTrue(receipt['topology_color_uv_material_image_bytes_preserved'])
        self.assertEqual(len(raw),len(before))

    def test_reference_tamper_and_path_escape_fail_closed(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);p=root/'data/a.json';p.parent.mkdir();p.write_text('{}')
            reference=R.ref(p,root);self.assertEqual(R.checked(reference,root),b'{}')
            p.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'hash or length'):R.checked(reference,root)
            with self.assertRaisesRegex(ValueError,'outside'):
                R.checked({'path':'../outside','sha256':'0'*64,'bytes':1},root)

    def test_writer_rejects_geometry_count_change(self):
        with self.assertRaisesRegex(ValueError,'positions'):R.rewrite_positions(self.raw(),np.zeros((4,3)))


if __name__=='__main__':unittest.main()
