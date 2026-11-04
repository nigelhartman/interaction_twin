import unittest
import numpy as np
from observed_correspondence_review import source_scopes, project, capture_guidance


class CorrespondenceReviewTests(unittest.TestCase):
    def test_any_heldout_image_in_training_call_is_rejected(self):
        frames=[{'id':'a','split':'train','image_sha256':'aa'},{'id':'b','split':'heldout','image_sha256':'bb'}]
        bad=[{'scope':'train_geometry_forward','source_frame_ids':['a','b'],'image_sha256':['aa','bb'],'status':'complete'}]
        self.assertTrue(source_scopes(bad,frames))
        good=[{'scope':'heldout_isolated_ref_test_forward','source_frame_ids':['a','b'],'image_sha256':['aa','bb'],'status':'complete'}]
        self.assertFalse(source_scopes(good,frames))

    def test_source_hash_or_multiple_test_images_rejected(self):
        frames=[{'id':'a','split':'train','image_sha256':'aa'},{'id':'b','split':'heldout','image_sha256':'bb'}]
        bad=[{'scope':'heldout_forward','source_frame_ids':['b','b'],'image_sha256':['bb','bb'],'status':'complete'}]
        self.assertTrue(source_scopes(bad,frames))
        bad=[{'scope':'train_geometry_forward','source_frame_ids':['a'],'image_sha256':['wrong'],'status':'complete'}]
        self.assertTrue(source_scopes(bad,frames))

    def test_known_camera_projection_and_behind_camera(self):
        camera={'R':np.eye(3),'t':[1,0,2],'K':[[200,0,100],[0,300,50],[0,0,1]]}
        uv,z=project([1,1,2],camera)
        np.testing.assert_allclose(uv,[200,125]);self.assertEqual(z,4)
        _,z=project([1,1,-3],camera);self.assertLess(z,0)

    def test_failed_geometry_stays_failed_in_capture_guidance(self):
        rows=capture_guidance({'feature_tracks':2,'recovered_points':0})
        by_code={r['code']:r for r in rows}
        self.assertFalse(by_code['no_observed_3d']['automatic_repair_established'])
        self.assertFalse(by_code['metric_assembly_unresolved']['physical_contact_or_assembly_established'])


if __name__=='__main__':unittest.main()
