"""Analytic similarity and native-support tests, no saved footage or inference."""
import itertools
import math
import unittest

from robot_source_staging_v1 import cap_tilt_from_supported_reference, component_display, constrain_rest, inverse, multiply, quaternion, similarity, transform
from robot_task_geometry import rotate_quaternion


class SourceStagingMathTests(unittest.TestCase):
    def test_reviewed_cap_tilt_is_minimal_proper_and_preserves_source_yaw_scale_and_native_center(self):
        def pose(angle,scale=1):
            c=math.cos(angle);s=math.sin(angle)
            return [[scale,0,0,.4],[0,scale*c,-scale*s,.2],[0,scale*s,scale*c,-.3],[0,0,0,1]]
        source=pose(-math.pi/6,.7);reference=pose(-math.pi*.4,.9);center=[.2,-.1,.5]
        refined,proof=cap_tilt_from_supported_reference(source,center,[0,0,1],reference,[0,0,1])
        scale,rotation=similarity(refined)
        self.assertAlmostEqual(scale,.7)
        self.assertLess(math.dist(transform(source,center),transform(refined,center)),1e-12)
        self.assertAlmostEqual(proof['minimal_proper_rotation_rad'],math.pi*.4-math.pi/6)
        self.assertAlmostEqual(rotation[1][2],math.sin(math.pi*.4))
        self.assertAlmostEqual(rotation[0][2],0)
        self.assertGreater(rotation[2][2],0)
        self.assertTrue(proof['source_yaw_preserved']);self.assertFalse(proof['physical_orientation_verified'])
        with self.assertRaises(ValueError):cap_tilt_from_supported_reference(source,center,[0,0,2],reference,[0,0,1])
        with self.assertRaises(ValueError):cap_tilt_from_supported_reference(source,center,[0,0,1],pose(math.pi/6),[0,0,1])

    def test_one_parent_normalization_reproduces_every_oriented_child_corner(self):
        parent = [[-2, -1, -1], [2, 1, 1]]
        child = [[-1, -.5, -.25], [1, .5, .25]]
        angle = .8;s = .7;c = math.cos(angle);sn = math.sin(angle)
        relative = [[s*c, -s*sn, 0, .2], [s*sn, s*c, 0, .1], [0, 0, s, -.3], [0, 0, 0, 1]]
        result = component_display(parent, child, relative)
        span = max(child[1][i]-child[0][i] for i in range(3))
        for point in itertools.product(*zip(*child)):
            actual = rotate_quaternion([v * result['display_max_dimension']/span for v in point], result['quaternion_xyzw'])
            actual = [actual[i]+result['center_offset'][i] for i in range(3)]
            expected = [v*.38/4 for v in transform(relative, point)]
            self.assertLess(math.dist(actual, expected), 1e-12)
        self.assertAlmostEqual(result['half_height'], (.7*sn+.35*c)*.38/4)

    def test_camera_gauge_cancels_before_common_native_scale_and_rotation(self):
        p = [[0, -.6, 0, 3], [.6, 0, 0, -2], [0, 0, .6, 7], [0, 0, 0, 1]]
        relative = [[.4, 0, 0, .2], [0, .4, 0, -.1], [0, 0, .4, .5], [0, 0, 0, 1]]
        camera_child = multiply(p, relative)
        recovered = multiply(inverse(p), camera_child)
        for actual, expected in zip(sum(recovered, []), sum(relative, [])):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_actual_vertex_support_avoids_empty_oriented_box_corner_clearance(self):
        bounds = [[-1, -1, 0], [1, 1, 0]]
        sparse_cross = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]]
        c = 2**-.5;m = [[c,-c,0,0], [c,c,0,0], [0,0,1,0], [0,0,0,1]]
        box = component_display([[-2,-2,-2],[2,2,2]], bounds, m)
        exact = component_display([[-2,-2,-2],[2,2,2]], bounds, m, vertices=sparse_cross)
        self.assertAlmostEqual(box['half_height'], exact['half_height']*2)

    def test_nonuniform_reflected_nonfinite_and_nonhomogeneous_poses_fail_closed(self):
        for m in [[[1,0,0,0],[0,2,0,0],[0,0,1,0],[0,0,0,1]],
                  [[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                  [[1,0,0,float('nan')],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                  [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,1,0,1]]]:
            with self.subTest(matrix=m), self.assertRaises(ValueError):similarity(m)

    def test_quaternion_pi_branches_remain_unit_and_rotate_the_expected_axis(self):
        for rotation in [[[1,0,0],[0,-1,0],[0,0,-1]],
                         [[-1,0,0],[0,1,0],[0,0,-1]],
                         [[-1,0,0],[0,-1,0],[0,0,1]]]:
            q = quaternion(rotation)
            self.assertAlmostEqual(sum(v*v for v in q), 1)
            p = [1,2,3];actual = rotate_quaternion(p,q)
            expected = [sum(rotation[i][j]*p[j] for j in range(3)) for i in range(3)]
            self.assertLess(math.dist(actual, expected), 1e-12)

    def test_visual_rest_retains_raw_source_rotation_and_ratios_but_reports_envelope_corrections(self):
        parent = [[-1,-.4,-.5],[1,.4,.5]]
        rows = []
        for index, scale in enumerate([1., .8]):
            m = [[scale,0,0,-.8+index*.1],[0,scale,0,1.2],[0,0,scale,0],[0,0,0,1]]
            display = component_display(parent,[[-.4,-.5,-.3],[.4,.5,.3]],m)
            rows.append({**display,'component_id':'part'+str(index)})
        descriptor = {'components':rows,'parent_native_bounds':parent,'base_display_max_dimension':.38}
        before = [row['quaternion_xyzw'][:]for row in rows]
        result = constrain_rest(descriptor,{'body_bottom_offset_y':-.076,'opening_rim_offset_y':.07})
        factor = result['visual_rest_constraints']['shared_child_scale_multiplier']
        self.assertLess(factor,1)
        self.assertEqual([row['quaternion_xyzw']for row in result['components']],before)
        for raw,row in zip(rows,result['components']):
            self.assertAlmostEqual(row['display_max_dimension']/raw['display_max_dimension'],factor)
            self.assertEqual(row['raw_source_center_offset'],raw['center_offset'])
            self.assertGreaterEqual(row['center_offset'][1]-row['lower_support_extents'][1],-.072-1e-12)
            self.assertLessEqual(row['center_offset'][1]+row['upper_support_extents'][1],.066+1e-12)
        self.assertFalse(result['visual_rest_constraints']['physical_contact_verified'])
        self.assertFalse(result['visual_rest_constraints']['actual_cavity_surface_or_lid_clearance_verified'])
        self.assertNotIn('raw_source_center_offset',rows[0],'The source descriptor must remain immutable')

    def test_source_visible_upper_region_can_protrude_without_burying_the_whole_part(self):
        parent=[[-1,-.4,-.5],[1,.4,.5]]
        display=component_display(parent,[[-.2,-.8,-.2],[.2,.8,.2]],[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        row={**display,'component_id':'part','source_sample_index':24,'parent_source_mask_ref':{'sha256':'a'*64},'child_source_mask_ref':{'sha256':'b'*64}}
        row['source_visible_rim_protrusion']={'method':'adjacent_parent_mask_top_median_vs_child_top','source_sample_index':24,'parent_mask_ref':row['parent_source_mask_ref'],'child_mask_ref':row['child_source_mask_ref'],'bounded_visible_height_fraction':.35,'physical_seating_verified':False}
        result=constrain_rest({'components':[row],'parent_native_bounds':parent,'base_display_max_dimension':.38},{'body_bottom_offset_y':-.076,'opening_rim_offset_y':.07})
        staged=result['components'][0];height=2*staged['half_height'];rim=result['visual_rest_constraints']['opening_rim_offset_y']
        self.assertAlmostEqual(staged['center_offset'][1]+staged['upper_support_extents'][1],rim+.35*height)
        self.assertGreaterEqual(staged['center_offset'][1]-staged['lower_support_extents'][1],-.072-1e-12)
        self.assertTrue(result['visual_rest_constraints']['source_visible_child_upper_region_above_rim'])
        row['source_visible_rim_protrusion']['source_sample_index']=25
        with self.assertRaises(ValueError):constrain_rest({'components':[row],'parent_native_bounds':parent,'base_display_max_dimension':.38},{'body_bottom_offset_y':-.076,'opening_rim_offset_y':.07})


if __name__ == '__main__':unittest.main()
