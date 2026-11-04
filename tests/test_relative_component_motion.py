import unittest
import cv2
import numpy as np
from relative_component_motion import compare_views


def view(orientations):
    K=[[400.,0,320],[0,400.,240],[0,0,1.]]
    frames=[{'id':str(i),'timestamp_s':float(i),'width':640,'height':480} for i in range(len(orientations))]
    return {'id':'test','frames':frames,'cameras':[{'frame_id':str(i),'R':R.tolist(),'t':[0,0,1],
             'K':K,'split':'train'} for i,R in enumerate(orientations)]}


class RelativeMotionTests(unittest.TestCase):
    def test_common_moving_camera_rotation_cancels(self):
        camera=[cv2.Rodrigues(np.array([0,.2*i,.1*i]))[0] for i in range(5)]
        report=compare_views(view(camera),view(camera),{'sha256':'same'},{'sha256':'same'})
        self.assertLess(report['maximum_conditional_rotation_change_deg'],1e-5)
        self.assertFalse(report['physical_assembly_established'])
        self.assertIsNone(report['relative_translation'])

    def test_relative_angle_is_invariant_to_native_rotation_gauges(self):
        camera=[cv2.Rodrigues(np.array([.04*i,.15*i,.03*i]))[0] for i in range(5)]
        part=[cv2.Rodrigues(np.array([0,0,.12*i]))[0] for i in range(5)]
        ga=cv2.Rodrigues(np.array([.7,.1,-.2]))[0]
        gb=cv2.Rodrigues(np.array([-.2,.3,.4]))[0]
        first=view([R@ga for R in camera]);second=view([R@P@gb for R,P in zip(camera,part)])
        report=compare_views(first,second,{'sha256':'same'},{'sha256':'same'})
        self.assertAlmostEqual(report['maximum_conditional_rotation_change_deg'],np.degrees(.48),places=8)
        self.assertFalse(report['joint_axis_established'])

    def test_missing_simultaneous_pose_stays_missing(self):
        first=view([np.eye(3)]*3);second=view([np.eye(3)]*3)
        second['cameras'].pop(1)
        report=compare_views(first,second,{'sha256':'same'},{'sha256':'same'})
        self.assertEqual(report['rows'][1]['status'],'missing_simultaneous_component_pose')
        self.assertIsNone(report['rows'][1]['conditional_relative_rotation_change_deg'])

    def test_different_videos_or_intrinsics_are_not_combined(self):
        first=view([np.eye(3)]);second=view([np.eye(3)])
        report=compare_views(first,second,{'sha256':'a'},{'sha256':'b'})
        self.assertEqual(report['simultaneous_pose_pairs'],0)
        second['cameras'][0]['K'][0][0]=450
        report=compare_views(first,second,{'sha256':'same'},{'sha256':'same'})
        self.assertEqual(report['simultaneous_pose_pairs'],0)


if __name__=='__main__':unittest.main()
