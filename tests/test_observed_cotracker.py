"""Synthetic correspondence/gate tests; these never load a learned model."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import observed_geometry as G
import observed_cotracker as C


def frames():
    rng=np.random.default_rng(84)
    image=rng.integers(0,256,(100,130,3),dtype=np.uint8)
    mask=np.zeros((100,130),np.uint8);mask[15:85,20:110]=255
    return [{'index':i,'id':f'frame_{i}','source_index':i,'timestamp_s':i*.1,
             'image':image.copy(),'mask':mask.copy(),'split':'heldout' if i%5==2 else 'train',
             'image_sha256':str(i),'image_url':'/files/data/frame.jpg','mask_url':'/files/data/mask.png',
             'audit_status':'not_model_audited'} for i in range(8)]


class CoTrackerGeometryTests(unittest.TestCase):
    def test_training_crop_cannot_depend_on_heldout_mask(self):
        rows=frames();before=C.training_crop(rows)
        rows[2]['mask'][:]=255;rows[7]['mask'][:]=255
        self.assertEqual(C.training_crop(rows),before)

    def test_queries_are_automatic_textured_mask_interior_only(self):
        row=frames()[0]
        points,margin=C.automatic_queries(row['image'],row['mask'],80)
        self.assertGreater(len(points),25)
        core,_=C.core_mask(row['mask'],3,.08)
        xy=np.rint(points).astype(int)
        self.assertTrue(core[xy[:,1],xy[:,0]].all())
        self.assertGreaterEqual(margin,3)
        blank,_=C.automatic_queries(np.ones_like(row['image'])*120,row['mask'],80)
        self.assertEqual(len(blank),0)

    def test_mask_visibility_and_nonfinite_predictions_are_rejected(self):
        rows=frames()[:2]
        uv=np.array([[[50,50],[1,1],[60,50]],[[50,50],[50,50],[np.nan,50]]],float)
        visible=np.array([[True,True,False],[False,True,True]])
        support=C.mask_support(uv,visible,rows)
        self.assertEqual(support.tolist(),[[True,False,False],[False,True,False]])

    def test_loss_does_not_reacquire_identity(self):
        support=np.ones((8,2),bool)
        support[1,0]=False;support[5,0]=False
        accepted=C.no_reacquisition(support,3)
        self.assertEqual(accepted[:,0].tolist(),[False,False,True,True,True,False,False,False])
        self.assertTrue(accepted[:,1].all())

    def test_reverse_cycle_endpoint_cannot_use_later_reappearance(self):
        support=np.ones((10,2),bool)
        support[5,0]=False
        endpoints=C.cycle_endpoints(support,3)
        self.assertEqual(endpoints[0],0)  # backward distance3, forward contiguous end4 distance1
        self.assertEqual(endpoints[1],9)
        support[:3,0]=False
        self.assertEqual(C.cycle_endpoints(support,3)[0],4)

    def test_train_tracker_never_receives_heldout_images(self):
        rows=frames()
        provider=object.__new__(C.CorrespondenceProvider)
        provider.training=[];provider.tracks={};provider.calls=[];provider.heldout=[]
        provider.model_provenance={};provider.crop=C.training_crop(rows)
        seen=[]
        def infer(sequence,queries,scope):
            self.assertTrue(all(row['split']=='train' for row in sequence))
            seen.append(scope)
            queries=np.array(queries,float);result=[]
            for frame in sequence:
                locations=[]
                for q in queries:
                    anchor=sequence[int(q[0])]['index']
                    locations.append(q[1:]+np.array([.1,0])*(frame['index']-anchor))
                result.append(locations)
            return np.array(result),np.ones((len(sequence),len(queries)),bool)
        provider._infer=infer
        with tempfile.TemporaryDirectory() as tmp:
            tracks,_,_,conflicts=provider.build_tracks(rows,Path(tmp))
        self.assertEqual(conflicts,0)
        self.assertEqual(len(seen),2)
        self.assertGreaterEqual(len(tracks),25)
        self.assertTrue(all(rows[fi]['split']=='train' for obs in tracks.values() for fi in obs))
        self.assertEqual(len(rows[2]['xy']),0)
        self.assertEqual(len(rows[7]['xy']),0)

    def test_heldout_matcher_hook_keeps_anchor_score_sets_disjoint(self):
        K=G.camera_matrix(130,100)
        points={f'ct_{i:06d}':np.array([((i%10)-5)*.08,((i//10)-4)*.08,4.]) for i in range(90)}
        obs={key:G.project(xyz[None],(np.eye(3),np.zeros(3)),K)[0][0] for key,xyz in points.items()}
        rows=[{'index':0,'id':'train','split':'train'}, {'index':1,'id':'test','split':'heldout','timestamp_s':1.,'mask':np.ones((100,130),np.uint8)}]
        model={'points':points,'poses':{0:(np.eye(3),np.zeros(3))}}
        def solve(geometry,anchors,intrinsics):
            self.assertTrue(anchors)
            self.assertTrue(all(not G.score_track(tid) for tid in anchors))
            return (np.eye(3),np.zeros(3)),list(anchors)
        with patch.object(G,'solve_pose',side_effect=solve),patch.object(G,'match_heldout',side_effect=AssertionError('SIFT matcher must not run')):
            result,predictions,errors=G.evaluate_heldout(model,{},rows,{},K,observation_provider=lambda *args:obs)
        self.assertEqual(len(errors),sum(G.score_track(tid) for tid in points))
        self.assertTrue(all(error<1e-9 for error in errors))
        self.assertEqual(set(predictions[1]['roles'].values()),{'heldout','pose_anchor'})


if __name__=='__main__':
    unittest.main()
