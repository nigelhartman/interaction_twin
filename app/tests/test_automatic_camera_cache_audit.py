"""Independent portability and invalid-cache checks; fake weights only."""
import shutil
import unittest
from unittest.mock import patch

import observed_pipeline as pipeline
import tests.test_observed_camera_workflow as workflow_fixtures


class IndependentCameraCacheTests(unittest.TestCase):
    def fixture(self):
        fixture = workflow_fixtures.CameraWorkflowTests('test_complete_camera_cache_pins_every_nested_receipt_and_reuses_without_inference')
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        folder = fixture.make_learned_cache()
        return fixture, folder

    def test_portable_cache_does_not_require_live_model_weights_or_provenance(self):
        fixture, folder = self.fixture()
        shutil.rmtree(fixture.root / 'models')
        with patch('automatic_camera_prior.CameraPrior._predict', side_effect=AssertionError('Cache verification must never infer')):
            protocol, viewer, binding = pipeline._candidate_binding(fixture.root, folder)
            caches = pipeline.sparse_caches(fixture.root)
        self.assertEqual(len(caches), 1)
        self.assertIn('camera-prior/model-provenance.json', caches[0]['files'])
        self.assertTrue(binding['camera_artifact_sha256'])
        self.assertTrue(viewer['learned_camera_prior_used'])

    def test_same_bytes_via_model_snapshot_symlink_are_not_a_verified_cache(self):
        fixture, folder = self.fixture()
        snapshot = folder / 'camera-prior/model-implementation.py'
        duplicate = fixture.root / 'data/same-bytes.py'; duplicate.write_bytes(snapshot.read_bytes())
        snapshot.unlink(); snapshot.symlink_to(duplicate)
        with self.assertRaises(ValueError):
            pipeline._candidate_binding(fixture.root, folder)
        self.assertEqual(pipeline.sparse_caches(fixture.root), [])

    def test_read_only_catalog_never_constructs_model_or_infers(self):
        fixture, folder = self.fixture()
        before = {str(path.relative_to(folder)): path.read_bytes() for path in folder.rglob('*') if path.is_file()}
        with patch('automatic_camera_prior.CameraPrior', side_effect=AssertionError('No model construction during catalog')):
            self.assertEqual(len(pipeline.sparse_caches(fixture.root)), 1)
        after = {str(path.relative_to(folder)): path.read_bytes() for path in folder.rglob('*') if path.is_file()}
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
