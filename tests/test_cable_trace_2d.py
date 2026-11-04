"""Local observed paths never bridge mask gaps or become 3D/metric claims."""
import copy
import io
import unittest

import numpy as np
from PIL import Image, ImageDraw

import cable_trace_2d as C
import semantic_asset_companion as S
from tests import test_twin_catalog as fixtures


def mask_image(lines, *, size=(120, 100), width=5):
    image = Image.new('L', size)
    draw = ImageDraw.Draw(image)
    for line in lines:
        draw.line(line, fill=255, width=width)
    return np.asarray(image)


def png(array):
    output = io.BytesIO()
    Image.fromarray(array).save(output, format='PNG')
    return output.getvalue()


class ObservedCablePathTests(unittest.TestCase):
    def assert_observed_segments(self, mask, result):
        self.assertEqual(result['status'], 'observed_2d_path_candidate')
        self.assertGreater(len(result['points_px']), 2)
        for a, b in zip(result['points_px'], result['points_px'][1:]):
            for x, y in C._line(a, b):
                self.assertTrue(mask[y, x], f'Path invented foreground at {(x,y)}')

    def test_curved_visible_path_stays_inside_saved_mask_without_metric_endpoints(self):
        mask = mask_image([[(15, 18), (25, 40), (70, 45), (95, 80)]])
        result = C.trace_mask(mask)
        self.assert_observed_segments(mask, result)
        self.assertGreater(result['path_length_px'], 100)
        self.assertEqual(result['observed_endpoint_count'], 2)
        self.assertEqual(result['path_method'], 'exact_tree_geodesic_diameter')
        self.assertTrue(result['endpoint_uncertain'])
        self.assertIn('not_connector_identity', result['endpoint_identity'])

    def test_disconnected_occluded_spans_are_never_stitched(self):
        mask = mask_image([[(10, 50), (60, 50)], [(74, 50), (110, 50)]], width=3)
        result = C.trace_mask(mask)
        self.assert_observed_segments(mask, result)
        self.assertEqual(result['source_component_count'], 2)
        self.assertIn('disconnected_source_mask_spans_not_joined', result['flags'])
        self.assertTrue(all(x < 65 for x, _ in result['points_px']))
        self.assertFalse(result['path_is_complete_mask'])

    def test_short_spur_pruning_preserves_observed_long_cable_but_keeps_branch_warning(self):
        mask = mask_image([[(10, 50), (110, 50)], [(60, 50), (60, 46)]], width=1)
        result = C.trace_mask(mask, prune_spur_px=6)
        self.assert_observed_segments(mask, result)
        self.assertGreater(result['pruned_spur_nodes'], 0)
        self.assertEqual(result['observed_endpoint_count'], 2)
        branched = C.trace_mask(mask_image([[(10, 50), (110, 50)], [(60, 50), (60, 15)]], width=1))
        self.assertGreater(branched['branch_nodes'], 0)
        self.assertIn('branch_or_self_overlap_endpoint_assignment_ambiguous', branched['flags'])
        self.assertLess(branched['principal_path_fraction_of_retained_skeleton'], 1)

    def test_closed_projection_does_not_invent_physical_open_ends(self):
        image = Image.new('L', (120, 100))
        ImageDraw.Draw(image).ellipse((20, 15, 100, 85), outline=255, width=3)
        mask = np.asarray(image)
        result = C.trace_mask(mask)
        self.assert_observed_segments(mask, result)
        self.assertEqual(result['observed_endpoint_count'], 0)
        self.assertEqual(result['path_method'], 'bounded_double_sweep_geodesic_approximation')
        self.assertTrue(result['endpoint_uncertain'])

    def test_bounded_reduction_cannot_connect_separate_source_components(self):
        mask = mask_image([[(15, 80), (90, 80)], [(93, 80), (220, 80)]], size=(240, 160), width=2)
        result = C.trace_mask(mask, max_long_edge=64)
        self.assert_observed_segments(mask, result)
        self.assertEqual(result['working_size_px'], [64, 43])
        self.assertEqual(result['source_component_count'], 2)
        self.assertTrue(all(x > 90 for x, _ in result['points_px']))
        self.assertIn('nearest_pixel_reduction_may_omit_thin_visible_spans', result['flags'])

    def test_invalid_probabilities_bulk_masks_and_complexity_fail_closed(self):
        with self.assertRaises(ValueError): C.trace_mask(np.full((100, 100), .5))
        with self.assertRaises(ValueError): C.trace_mask(np.full((100, 100), 127, dtype=np.uint8))
        with self.assertRaises(ValueError): C.trace_mask(np.zeros((1, 1, 1), dtype=bool))
        self.assertEqual(C.trace_mask(np.ones((100, 100), dtype=np.uint8))['status'], 'withheld_bulk_or_full_frame_mask')
        sparse = np.zeros((960, 960), dtype=np.uint8)
        sparse[::3, ::3] = 255
        self.assertEqual(C.trace_mask(sparse)['status'], 'withheld_skeleton_complexity')
        with self.assertRaises(ValueError): C.trace_mask(mask_image([[(10, 50), (100, 50)]]), max_long_edge=True)


class CableSourceContractTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TwinCatalogTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.program = 'data/case/procedure/interaction.json'
        self.mask = mask_image([[(10, 40), (50, 65), (105, 30)]])
        self.fixture.doc['tracklets'][0]['label'] = 'flexible charging cable'
        for frame in self.fixture.doc['evidence']:
            frame['frame_sha256'] = self.fixture.file(frame['frame_path'], png(np.zeros((100,120,3), dtype=np.uint8)))
            if frame['source_index'] != 2:
                path = f"data/case/masks/{frame['source_index']}.png"
                digest = self.fixture.file(path, png(self.mask))
                frame['observations'] = [{'track_id':'part', 'mask_path':path, 'mask_sha256':digest,
                    'gate_status':'candidate_geometric_gate_only' if frame['source_index'] == 1 else 'withheld_geometric_gate',
                    'audit_status':'not_reviewed'}]
        self.fixture.publish()

    def test_exact_pts_masks_and_gaps_are_preserved_without_temporal_fill(self):
        result = C.build(self.program, 'part', root=self.root)
        self.assertEqual([s['timestamp_s'] for s in result['samples']], [0.,1.,2.])
        self.assertEqual([s['source_index'] for s in result['samples']], [1,2,3])
        self.assertEqual([s['status'] for s in result['samples']], ['observed_2d_path_candidate','withheld_no_saved_track_mask','withheld_saved_observation_gate'])
        self.assertIsNone(result['samples'][1]['trace'])
        self.assertIsNone(result['samples'][2]['trace'])
        self.assertEqual(result['claims'], C.CLAIMS)
        self.assertFalse(result['claims']['occluded_spans_filled'])
        self.assertFalse(result['claims']['three_dimensional_cable_recovered'])
        self.assertTrue(C.validate(result, root=self.root))

    def test_content_addressed_write_rebuild_tamper_and_source_changes(self):
        result = C.build(self.program, 'part', root=self.root)
        first = C.write(result, 'output/cable-review', root=self.root)
        self.assertEqual(C.write(result, 'output/cable-review', root=self.root), first)
        self.assertEqual(C.load(first['path'], root=self.root), result)
        altered = copy.deepcopy(result)
        altered['samples'][0]['trace']['points_px'][0] = [0,0]
        with self.assertRaisesRegex(ValueError, 'differs'): C.validate(altered, root=self.root)
        altered = copy.deepcopy(result)
        altered['claims']['depth_recovered'] = True
        with self.assertRaises(ValueError): C.validate(altered, root=self.root)
        self.fixture.file(result['samples'][0]['mask_ref']['path'], png(np.zeros_like(self.mask)))
        with self.assertRaises(ValueError): C.load(first['path'], root=self.root)

    def test_explicit_selection_and_bounded_quantiles_never_use_foreign_samples(self):
        result = C.build(self.program, 'part', max_frames=2, root=self.root)
        self.assertEqual([s['source_index'] for s in result['samples']], [1,3])
        selected = C.build(self.program, 'part', evidence_ids=['source_000003','source_000001'], root=self.root)
        self.assertEqual([s['source_index'] for s in selected['samples']], [1,3])
        self.assertEqual(selected['selection']['method'], 'explicit_evidence_ids')
        self.assertTrue(C.validate(selected, root=self.root))
        for ids in [['foreign'],['source_000001','source_000001']]:
            with self.assertRaises(ValueError): C.build(self.program,'part', evidence_ids=ids, root=self.root)
        with self.assertRaises(ValueError): C.build(self.program, 'foreign', root=self.root)
        with self.assertRaises(ValueError): C.build(self.program, 'part', max_frames=91, root=self.root)
        variable_pts={'evidence':[{'id':str(i),'timestamp_s':t}for i,t in enumerate([0.,.1,.2,.3,3.])]}
        self.assertEqual([e['timestamp_s']for e in C._selected_frames(variable_pts,None,3)],[0.,.3,3.])

    def test_dimension_mismatch_or_duplicate_track_masks_never_invents_transform(self):
        frame = self.fixture.doc['evidence'][0]
        observation = frame['observations'][0]
        observation['mask_sha256'] = self.fixture.file(observation['mask_path'], png(self.mask[:50]))
        self.fixture.publish()
        result = C.build(self.program, 'part', root=self.root)
        self.assertEqual(result['samples'][0]['status'], 'withheld_frame_mask_dimension_mismatch')
        self.assertIsNone(result['samples'][0]['trace'])
        frame['observations'].append(copy.deepcopy(observation))
        self.fixture.publish()
        with self.assertRaisesRegex(ValueError, 'duplicate'): C.build(self.program, 'part', root=self.root)


if __name__ == '__main__':
    unittest.main()
