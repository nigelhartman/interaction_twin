"""Analytic camera/layout diagnostics; no model inference or physical claims."""
import unittest
import importlib.util
from pathlib import Path
import tempfile
import numpy as np
from sam3d_source_projection import grid_observations, recover_estimated_camera, raw_layout_projection


class SourceProjectionTests(unittest.TestCase):
    def pointmap(self):
        yy,xx=np.mgrid[:80,:80];z=1+xx*.001+yy*.002
        cv=np.stack([(xx-39.2)*z/150,(yy-41.3)*z/160,z],-1)
        return cv*[-1,-1,1]

    def test_fixed_disjoint_grids_recover_known_analytic_camera(self):
        original=self.pointmap();before=original.copy();result=recover_estimated_camera(original)
        np.testing.assert_allclose(result['K_estimated'],[[150,0,39.2],[0,160,41.3],[0,0,1]],atol=1e-10)
        self.assertLess(result['heldout_grid']['maximum_px'],1e-10)
        _,a,_=grid_observations(original,offset=0);_,b,_=grid_observations(original,offset=4)
        self.assertFalse(set(map(tuple,a))&set(map(tuple,b)));np.testing.assert_array_equal(original,before)
        self.assertFalse(result['independent_camera_calibration'])

    def test_missing_points_are_excluded_without_filling_or_changing_original(self):
        points=self.pointmap();points[0,0]=np.nan;points[4,4,2]=-1;before=points.copy()
        result=recover_estimated_camera(points);self.assertEqual(result['training_grid']['support'],99)
        self.assertEqual(result['heldout_grid']['support'],99);np.testing.assert_array_equal(points,before)

    def test_degenerate_or_too_small_pointmap_is_not_assumed_camera(self):
        for points in [np.zeros((80,80,3)),np.ones((80,80,3)),self.pointmap()[:10,:10],np.ones((80,80))]:
            with self.assertRaises(ValueError):recover_estimated_camera(points)

    def test_layout_projection_divides_translation_not_mesh_or_scale_twice(self):
        result=raw_layout_projection([1,0,0,0],[.2,.3,2.],[.16,.16,.16])
        vertices=np.array([[0,0,0],[.1,.3,-.2],[-.2,.1,.2]])
        matrix=np.array(result['layout']['camera_from_glb']);full=vertices@matrix[:3,:3].T+matrix[:3,3]
        equivalent=vertices@np.array(result['R']).T+result['t']
        np.testing.assert_allclose(full[:,:2]/full[:,2:],equivalent[:,:2]/equivalent[:,2:],atol=1e-12)
        self.assertFalse(result['accepted_source_video_pose']);self.assertFalse(result['optimized'])

    def test_only_diagnostic_console_log_may_be_empty(self):
        path=Path(__file__).resolve().parents[1]/'scripts/raw_sam3d_source_projection.py'
        spec=importlib.util.spec_from_file_location('raw_projection_fixture',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);(directory/'worker.log').write_bytes(b'');(directory/'render.png').write_bytes(b'inert')
            rows=module.diagnostic_records(directory)
            self.assertEqual(next(r['bytes'] for r in rows if r['path']=='worker.log'),0)
            (directory/'empty-render.png').write_bytes(b'')
            with self.assertRaises(ValueError):module.diagnostic_records(directory)


if __name__=='__main__':unittest.main()
