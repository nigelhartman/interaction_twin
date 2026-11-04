import importlib.util
import unittest
import tempfile
import json
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('generated_observer', Path(__file__).parents[1]/'scripts/stage4_generated_observer.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class GeneratedObserverTests(unittest.TestCase):
    def test_portrait_gl_projection_matches_original_integer_pixel_coordinates(self):
        width, height = 540, 960
        k = np.array([[467.65, 0, 269.5], [0, 467.65, 479.5], [0, 0, 1.]])
        xyz = np.array([[.2, .5, 2.], [-.7, .6, 3.], [0., 0, 4.]])
        cv_pixels = (xyz@k.T)[:, :2]/xyz[:, 2:]
        cv_to_gl = np.diag([1., -1, -1, 1])
        clip = np.c_[xyz, np.ones(len(xyz))]@(M.source_projection(k, width, height, .01, 100)@cv_to_gl).T
        ndc = clip[:, :2]/clip[:, 3:]
        pixels = np.c_[(ndc[:, 0]+1)*width/2-.5, (1-ndc[:, 1])*height/2-.5]
        np.testing.assert_allclose(pixels, cv_pixels, atol=1e-10)

    def test_scene_bounds_and_path_never_bridge_withheld_intervals(self):
        vertices = np.array([[-.5, -.5, -.5], [.5, .5, .5]])
        frames = []
        for sample, status in enumerate(['supported_conditional_motion_candidate', 'supported_conditional_motion_candidate', 'unsupported', 'supported_conditional_motion_candidate', 'supported_conditional_motion_candidate'], 1):
            frames.append({'sample_index': sample, 'timestamp_s': float(sample), 'status': status,
                           'candidate_rotation': np.eye(3).tolist(), 'candidate_translation': [sample*.1, .2, 4.]})
        layout = M.scene_layout(vertices, frames)
        self.assertEqual([[p['sample_index'] for p in s] for s in layout['path_segments']], [[1, 2], [4, 5]])
        np.testing.assert_allclose(layout['bounds'], [[-.4, -.3, 3.5], [1., .7, 4.5]])
        self.assertLess(layout['source_near'], 3.5)
        self.assertGreater(layout['source_far'], 4.5)

    def test_malformed_pose_cannot_enter_live_scene(self):
        with self.assertRaises(ValueError):
            M.pose_matrix({'candidate_rotation': (np.eye(3)*2).tolist(), 'candidate_translation': [0, 0, 3]})

    def test_dense_motion_case_takes_precedence_and_wrong_hash_never_falls_back(self):
        with tempfile.TemporaryDirectory() as temp:
            old=Path(temp)/'old';dense=Path(temp)/'dense';old.mkdir();dense.mkdir()
            for path in [old,dense]:(path/'manifest.json').write_text(json.dumps({'source':{'sha256':'same-video'}}))
            metadata={'source':{'selection_manifest_path':str(old/'selection.json')}}
            pack={'case_path':str(dense),'provenance':{'source_video_sha256':'same-video'}}
            self.assertEqual(M.source_case(pack,metadata),dense.resolve())
            (dense/'manifest.json').write_text(json.dumps({'source':{'sha256':'other-video'}}))
            with self.assertRaises(ValueError):M.source_case(pack,metadata)
            pack.pop('case_path')
            self.assertEqual(M.source_case(pack,metadata),old)


if __name__ == '__main__':
    unittest.main()
