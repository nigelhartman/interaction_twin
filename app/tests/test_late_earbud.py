import unittest
import numpy as np
from scripts.stage4_late_earbud import choose_one,excluded_mask,conservative_verdict


class LateEarbudTests(unittest.TestCase):
    def candidate(self,t,score=10.,**kw):
        return dict(timestamp_s=t,scope_safe=True,mask_verdict='plausible_visible_mask',reconstruction_view_usable=True,quality_score=score,**kw)

    def test_integer_time_never_becomes_reference(self):
        self.assertEqual(choose_one([self.candidate(21.001,1000),self.candidate(21.4)])['timestamp_s'],21.4)

    def test_failed_scope_and_uncertain_evidence_not_selected(self):
        bad=self.candidate(21.4,100);bad['scope_safe']=False
        uncertain=self.candidate(21.8,200);uncertain['mask_verdict']='uncertain'
        self.assertEqual(choose_one([bad,uncertain,self.candidate(22.6)])['timestamp_s'],22.6)

    def test_no_usable_view_is_explicit_failure(self):
        with self.assertRaises(ValueError):choose_one([self.candidate(21.001)])

    def test_hand_subtraction_requires_exact_positive_audit_and_adds_no_pixels(self):
        component=np.array([[True,False],[True,True]]);hand=np.array([[True,True],[False,False]])
        self.assertTrue(np.array_equal(excluded_mask(component,hand,'unassessed'),component))
        result=excluded_mask(component,hand,'plausible_visible_mask')
        self.assertFalse(np.any(result&~component));self.assertEqual(int(result.sum()),2)

    def test_prior_negative_or_uncertain_mask_never_becomes_positive(self):
        self.assertEqual(conservative_verdict('uncertain','plausible_visible_mask'),'uncertain')
        self.assertEqual(conservative_verdict('reject_wrong_or_lost_mask','plausible_visible_mask'),'reject_wrong_or_lost_mask')
        self.assertEqual(conservative_verdict('plausible_visible_mask','uncertain'),'uncertain')

if __name__=='__main__':unittest.main()
