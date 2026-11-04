"""Synthetic mathematical checks; these fixtures are NOT real reconstructions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

import observed_geometry as G


def synthetic_scene(count=100,views=15,pure_rotation=False,noise=.08,planar=False):
    rng=np.random.default_rng(20261031)
    xyz=np.column_stack([rng.uniform(-.9,.9,count),rng.uniform(-.65,.65,count),
                         np.full(count,4.) if planar else rng.uniform(3.2,5.,count)])
    descriptors=rng.uniform(0,200,(count,128)).astype(np.float32)
    K=G.camera_matrix(640,480)
    frames=[];tracks={f'point_{i:06d}':{} for i in range(count)};nodes={};truth={}
    for fi in range(views):
        R=cv2.Rodrigues(np.array([.003*fi,-.012*fi,.001*fi]))[0]
        centre=np.array([0.,0.,0.]) if pure_rotation else np.array([-.065*fi,.015*np.sin(fi),0.])
        pose=(R,-R@centre);uv,z=G.project(xyz,pose,K);uv+=rng.normal(0,noise,uv.shape)
        frame={'index':fi,'id':f'fixture_{fi:03d}','timestamp_s':float(fi),'split':'heldout' if fi%5==2 else 'train',
               'image':np.zeros((480,640,3),np.uint8),'mask':np.ones((480,640),np.uint8)*255,
               'xy':uv,'des':descriptors.copy()}
        frames.append(frame);truth[fi]=pose
        if frame['split']=='train':
            for i,tid in enumerate(tracks): tracks[tid][fi]=uv[i].copy();nodes[(fi,i)]=tid
    return xyz,frames,tracks,nodes,K,truth


class ObservedGeometryTests(unittest.TestCase):
    def test_triangulation_recovers_nonplanar_points(self):
        xyz,frames,_,_,K,truth=synthetic_scene(noise=0.)
        recovered,angles,errors,positive=G.triangulate(frames[0]['xy'],frames[14]['xy'],truth[0],truth[14],K)
        self.assertTrue(positive.all())
        self.assertLess(np.max(np.linalg.norm(recovered-xyz,axis=1)),1e-10)
        self.assertGreater(np.median(angles),5.)
        self.assertLess(np.max(errors),1e-8)

    def test_pure_rotation_has_no_acceptable_parallax(self):
        _,frames,tracks,_,K,_=synthetic_scene(pure_rotation=True,noise=0.)
        model,logs=G.find_seed(tracks,frames,K)
        self.assertIsNone(model)
        self.assertFalse(any(r['passes_seed_gate'] for r in logs))

    def test_train_reconstruction_predicts_disjoint_heldout_tracks(self):
        _,frames,tracks,nodes,K,_=synthetic_scene()
        model,_=G.reconstruct(tracks,frames,K,max_nfev=8)
        self.assertIsNotNone(model)
        heldout,predictions,_=G.evaluate_heldout(model,tracks,frames,nodes,K)
        self.assertEqual(len(heldout),3)
        self.assertGreaterEqual(sum(r['passes'] for r in heldout),3)
        self.assertTrue(all(r['score_median_px']<1. for r in heldout))
        self.assertFalse(set(model['poses']) & {f['index'] for f in frames if f['split']=='heldout'})
        for info in predictions.values():
            anchors={t for t,r in info['roles'].items() if r=='pose_anchor'}
            scored={t for t,r in info['roles'].items() if r=='heldout'}
            self.assertFalse(anchors & scored)
            self.assertTrue(all(G.score_track(t) for t in scored))

    def test_corrupt_heldout_score_pixels_cannot_change_training_geometry(self):
        _,frames,tracks,nodes,K,_=synthetic_scene()
        model,_=G.reconstruct(tracks,frames,K,max_nfev=4)
        before=json.dumps({t:p.tolist() for t,p in model['points'].items()},sort_keys=True)
        normal,_,_=G.evaluate_heldout(model,tracks,frames,nodes,K)
        changed=copy.deepcopy(frames)
        for f in changed:
            if f['split']=='heldout':
                for i in range(len(f['xy'])):
                    if G.score_track(f'point_{i:06d}'): f['xy'][i]+=np.array([40.,-25.])
        corrupt,_,_=G.evaluate_heldout(model,tracks,changed,nodes,K)
        after=json.dumps({t:p.tolist() for t,p in model['points'].items()},sort_keys=True)
        self.assertEqual(before,after)
        refitted,_=G.reconstruct(tracks,changed,K,max_nfev=4)
        rebuilt=json.dumps({t:p.tolist() for t,p in refitted['points'].items()},sort_keys=True)
        self.assertEqual(before,rebuilt)
        for fi,pose in model['poses'].items():
            np.testing.assert_array_equal(pose[0],refitted['poses'][fi][0])
            np.testing.assert_array_equal(pose[1],refitted['poses'][fi][1])
        self.assertTrue(all(r['passes'] for r in normal))
        self.assertFalse(any(r['passes'] for r in corrupt))
        self.assertTrue(all(r['score_median_px']>30. for r in corrupt))

    def test_planar_and_collinear_shape_diagnostics(self):
        points={str(i):np.array([x,y,1.]) for i,(x,y) in enumerate([(0,0),(1,0),(0,1),(1,1),(2,2)])}
        self.assertLess(G.shape_metrics(points)['planarity_ratio'],1e-12)
        line={str(i):np.array([i,0.,0.]) for i in range(10)}
        self.assertLess(G.shape_metrics(line)['linearity_ratio'],1e-12)

    def test_track_merge_rejects_two_features_from_same_frame(self):
        u=G.TrackUnion()
        self.assertTrue(u.join((0,0),(1,0)))
        self.assertTrue(u.join((0,1),(2,0)))
        self.assertFalse(u.join((1,0),(2,0)))
        for track in u.tracks(min_length=2):
            self.assertEqual(len(track),len({f for f,k in track}))

    def test_ransac_rejects_bad_pose_correspondences(self):
        xyz,frames,_,_,K,truth=synthetic_scene(noise=0.)
        points={f'p{i}':x for i,x in enumerate(xyz)}
        observations={f'p{i}':uv.copy() for i,uv in enumerate(frames[10]['xy'])}
        for i in range(15): observations[f'p{i}']+=np.array([80.,-70.])
        pose,inliers=G.solve_pose(points,observations,K)
        self.assertGreater(len(inliers),75)
        self.assertLess(np.linalg.norm(pose[0]-truth[10][0]),1e-5)
        self.assertFalse(any(f'p{i}' in inliers for i in range(15)))

    def test_similarity_comparison_removes_only_global_gauge(self):
        rng=np.random.default_rng(5);a={f'p{i}':p for i,p in enumerate(rng.normal(size=(30,3)))}
        R=cv2.Rodrigues(np.array([.2,.1,-.3]))[0]
        b={t:2.5*p@R+np.array([4.,3.,2.]) for t,p in a.items()}
        self.assertLess(G.similarity_difference(a,b)['normalized_rms'],1e-10)
        b['p0']+=np.array([4.,0.,0.])
        self.assertGreater(G.similarity_difference(a,b)['normalized_rms'],.1)

    def test_missing_holdout_support_is_inconclusive_not_zero_error(self):
        xyz,frames,tracks,nodes,K,truth=synthetic_scene(count=8)
        model={'points':{t:p for t,p in zip(tracks,xyz)},'poses':{i:p for i,p in truth.items() if frames[i]['split']=='train'}}
        rows,_,errors=G.evaluate_heldout(model,tracks,frames,nodes,K)
        self.assertFalse(any(r['passes'] for r in rows))
        self.assertTrue(all(not r['pose_registered'] for r in rows))
        self.assertEqual(errors,[])

    def test_source_hash_and_path_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'data').mkdir();p=root/'data'/'fixture.txt';p.write_text('synthetic fixture')
            self.assertEqual(G.safe_file(root,'data/fixture.txt',G.digest(p)),p.resolve())
            with self.assertRaises(ValueError): G.safe_file(root,'data/fixture.txt','0'*64)
            (root/'outside.txt').write_text('fixture')
            with self.assertRaises(ValueError): G.safe_file(root,'outside.txt')

    def test_feature_core_excludes_thin_appendage_without_manual_points(self):
        rng=np.random.default_rng(8)
        mask=np.zeros((480,640),np.uint8)
        mask[100:380,180:460]=255
        mask[220:232,30:190]=255
        image=rng.integers(0,256,(480,640,3),dtype=np.uint8)
        frame={'image':image,'mask':mask}
        G.extract_features(frame)
        self.assertGreater(len(frame['xy']),10)
        distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
        margin=frame['feature_filter']['core_margin_px']
        self.assertGreater(margin,10.)
        for x,y in frame['xy']:
            self.assertGreaterEqual(distance[int(round(y)),int(round(x))],margin)
            self.assertGreater(x,180.)

    def test_empty_feature_mask_remains_empty(self):
        frame={'image':np.zeros((100,120,3),np.uint8),'mask':np.zeros((100,120),np.uint8)}
        G.extract_features(frame)
        self.assertEqual(len(frame['xy']),0)
        self.assertEqual(frame['des'].shape,(0,128))
        self.assertEqual(frame['feature_filter']['kept_unique_features'],0)


if __name__=='__main__': unittest.main()
