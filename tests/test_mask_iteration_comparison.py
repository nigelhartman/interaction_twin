import unittest
import numpy as np
from mask_iteration_comparison import mask_agreement, _unique


class ExactMaskComparisonTests(unittest.TestCase):
    def test_empty_masks_are_unknown_and_never_perfect_agreement(self):
        a=np.zeros((2,3),bool)
        self.assertEqual(mask_agreement(a,a)['state'],'both_empty')
        self.assertIsNone(mask_agreement(a,a)['iou_agreement'])

    def test_missing_mask_does_not_receive_numeric_accuracy(self):
        a=np.zeros((2,3),bool);b=a.copy();b[0,0]=True
        self.assertEqual(mask_agreement(a,b)['state'],'after_only_nonempty')
        self.assertIsNone(mask_agreement(a,b)['iou_agreement'])

    def test_actual_nonempty_iou_agreement(self):
        a=np.array([[1,1,0],[0,0,0]],bool)
        b=np.array([[0,1,1],[0,0,0]],bool)
        self.assertAlmostEqual(mask_agreement(a,b)['iou_agreement'],1/3)

    def test_no_silent_resize_or_duplicate_keys(self):
        with self.assertRaises(ValueError):mask_agreement(np.zeros((2,3)),np.zeros((3,2)))
        with self.assertRaises(ValueError):_unique([{'id':1},{'id':1}],lambda r:r['id'])


if __name__=='__main__':unittest.main()
