"""Nonrigid runtime readers obey the curated bounded callback contract."""
import copy
import hashlib
import json
import unittest

import nonrigid_observations_v1 as N


class NonrigidObservationReaderTests(unittest.TestCase):
    def setUp(self):
        self.raw = {}
        def artifact(name, value):
            raw = json.dumps(value, sort_keys=True).encode()
            self.raw[name] = raw
            return {'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        self.source = {'sha256': 'a' * 64}
        program = {'source': self.source,
                   'tracklets': [{'id': 'cable', 'limitations': ['Observed deformable structure']}],
                   'evidence': [{'source_index': 1, 'timestamp_s': 0.0, 'id': 'e1',
                                 'frame_path': 'data/frame.jpg', 'frame_sha256': 'b' * 64,
                                 'observations': []}]}
        program_ref = artifact('data/program.json', program)
        self.trace = {'format': 'interaction-twin-cable-trace-2d', 'version': 1,
                      'program_ref': program_ref, 'source_sha256': 'a' * 64, 'track_id': 'cable',
                      'samples': [{'source_index': 1, 'timestamp_s': 0.0, 'evidence_id': 'e1',
                                   'frame_ref': {'path': 'data/frame.jpg', 'sha256': 'b' * 64},
                                   'mask_ref': None, 'trace': None}]}
        trace_ref = artifact('data/trace.json', self.trace)
        self.doc = {'format': N.FORMAT, 'version': 1, 'source': self.source,
                    'component_id': 'cable', 'evidence_program': program_ref, 'trace_ref': trace_ref,
                    'depth': 'undefined', 'physical_accuracy_verified': False, 'simulation_ready': False,
                    'sample_count': 1, 'source_time_domain': 'source_video_presentation_seconds',
                    'display_policy': {'rigid_native_pose_replay': 'withheld_for_known_nonrigid_component',
                        'native_static_asset_inspection_retained': True,
                        'viewer_representation': 'source_pixel_polyline_overlay_video_camera_only',
                        'temporal_interpolation': 'none', 'maximum_nearest_sample_offset_s': .36,
                        'unknown_rows_are_barriers': True,
                        'display_between_samples': 'nearest_saved_annotation_estimate_not_motion_measurement'}}

    def test_required_keyword_size_bound_is_supplied_to_curated_callback(self):
        calls = []
        def read_bytes(reference, *, maximum):
            calls.append((reference['path'], maximum))
            return self.raw[reference['path']]
        self.assertEqual(N.validate(self.doc, read_bytes), self.trace)
        self.assertEqual(calls, [('data/program.json', N.MAX_BYTES), ('data/trace.json', N.MAX_BYTES)])

    def test_changed_observation_bytes_fail_closed(self):
        def read_bytes(reference, *, maximum):
            return self.raw[reference['path']] + b' '
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            N.validate(self.doc, read_bytes)

    def test_source_pixel_observations_cannot_be_upgraded_to_3d(self):
        value = copy.deepcopy(self.doc)
        value['depth'] = 'estimated_3d'
        def read_bytes(reference, *, maximum):
            raise AssertionError('Invalid scope must reject before any artifact read')
        with self.assertRaisesRegex(ValueError, 'cannot claim depth'):
            N.validate(value, read_bytes)


if __name__ == '__main__':
    unittest.main()
