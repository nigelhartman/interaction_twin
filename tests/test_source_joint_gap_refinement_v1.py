import copy
import unittest
import source_joint_gap_refinement_v1 as J


class SourceJointGapGateTests(unittest.TestCase):
    def setUp(self):
        self.metrics={'parent':{'mask_iou':.80,'source_foreground_coverage':.9},'child':{'mask_iou':.8,'source_foreground_coverage':.9},'union':{'mask_iou':.8}}
        self.fit={'interior_facing_cosine':{'parent':.7,'child':.7},'projected_hinge_seam_distance_px':4.}

    def test_silhouette_fit_cannot_override_wrong_visible_face(self):
        self.assertTrue(J.gate(self.metrics,self.fit))
        self.fit['interior_facing_cosine']['child']=-.7
        self.assertFalse(J.gate(self.metrics,self.fit))

    def test_union_overlap_cannot_hide_missing_child(self):
        self.metrics['child']={'mask_iou':.35,'source_foreground_coverage':.40}
        self.assertFalse(J.gate(self.metrics,self.fit))

    def test_separated_projected_seam_is_withheld(self):
        self.fit['projected_hinge_seam_distance_px']=J.POLICY['maximum_projected_hinge_seam_distance_px']+1
        self.assertFalse(J.gate(self.metrics,self.fit))


if __name__=='__main__':unittest.main()
