"""Intrinsics adapter contracts with synthetic sources and mocked providers."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import observed_geometry as G
import observed_cotracker as O


POLICY={'version':'synthetic-camera-fixture-only','heldout_images_used':False,
    'depth_or_point_maps_used':False,'camera_calibrated':False,'physical_scale_known':False,
    'physical_accuracy_verified':False,'downloads_allowed':False,'new_api_calls':0,
    'minimum_train_images':2,'maximum_train_images':3}


class Provider:
    def __init__(self,root,K=None,mutate=None):
        self.root=root;self.K=np.array(K if K is not None else [[75.,0.,31.5],[0.,80.,23.5],[0.,0.,1.]])
        self.mutate=mutate;self.seen=[];self.called=False

    def describe(self):
        return copy.deepcopy(POLICY)

    def estimate(self,frames,output):
        self.called=True;self.seen=frames
        frozen=json.loads((output/'protocol.json').read_text())
        assert frozen['camera_provider']['declaration']==POLICY
        assert all(f['split']=='train' for f in frames)
        assert all(not f['image'].flags.writeable for f in frames)
        directory=output/'camera-prior';directory.mkdir()
        inputs=[{k:f[k] for k in ['id','timestamp_s','image_url','image_sha256','split']} for f in frames[:3]]
        report={'format':'ordinary-video-camera-prior','version':1,'status':'unvalidated_learned_camera_assumption',
            'native_size':[64,48],'K':self.K.tolist(),'policy':self.describe(),'input_views':inputs,
            'camera_calibrated':False,'physical_scale_known':False,'physical_accuracy_verified':False,
            'prior_geometry_or_poses_used':False,'learned_depth_or_point_maps_used':False,
            'new_api_calls':0,'geometry_must_be_reconstructed_with_this_assumption':True,
            'actual_model_calls':len(inputs),'completed_model_calls':len(inputs),
            'input_integrity':{'unchanged':True}}
        if self.mutate:self.mutate(report)
        path=directory/'camera-prior.json';path.write_text(json.dumps(report))
        return self.K,{'K':self.K.tolist(),'report_path':str(path.relative_to(self.root)),
            'kind':report['status'],'policy':self.describe(),'input_views':inputs}


class ObservedCameraProviderTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
        data=self.root/'data';data.mkdir();self.frames=[]
        for i in range(5):
            image=np.full((48,64,3),i*20,np.uint8)
            path=data/f'frame-{i}.png';cv2.imwrite(str(path),image)
            maskpath=data/f'mask-{i}.png';cv2.imwrite(str(maskpath),np.full((48,64),255,np.uint8))
            self.frames.append({'id':f'f{i}','index':i,'source_index':i,'timestamp_s':float(i),
                'split':'heldout' if i==2 else 'train','image':image,'mask':np.full((48,64),255,np.uint8),
                'image_url':'/files/data/'+path.name,'image_sha256':G.digest(path),
                'mask_url':'/files/data/'+maskpath.name,'mask_sha256':G.digest(maskpath),
                'audit_status':None,'subtracted_components':[]})
        self.program=data/'program.json';self.program.write_text(json.dumps({'case_id':'fixture-only',
            'tracklets':[{'id':'part','label':'Synthetic fixture part'}],
            'source':{'path':'data/no-real-video','sha256':'0'*64},'evidence':[]}))

    def tearDown(self):self.tmp.cleanup()

    @staticmethod
    def features(frame):
        frame['xy']=np.empty((0,2));frame['des']=np.empty((0,128));frame['feature_filter']={'fixture':True}

    def run_fixture(self,provider=None,**kwargs):
        output=self.root/'data'/'result'
        with patch.object(G,'select_views',return_value=(copy.deepcopy(self.frames),[])),\
             patch.object(G,'extract_features',side_effect=self.features),\
             patch.object(G,'build_tracks',return_value=({}, {}, [], 0)),\
             patch.object(G,'reconstruct',return_value=(None,[])) as reconstruct:
            result=G.run(self.program,'part',output,root=self.root,sensitivity=False,
                         camera_provider=provider,**kwargs)
        return result,output,reconstruct

    def test_engine_passes_only_readonly_TRAIN_and_K_before_reconstruction(self):
        provider=Provider(self.root)
        result,output,reconstruct=self.run_fixture(provider)
        self.assertEqual([x['id'] for x in provider.seen],['f0','f1','f3','f4'])
        np.testing.assert_array_equal(reconstruct.call_args.args[2],provider.K)
        self.assertTrue(result['learned_camera_prior_used'])
        self.assertFalse(result['prior_geometry_or_depth_used'])
        assumption=result['camera_assumption']
        self.assertEqual(assumption['source_binding']['case_id'],'fixture-only')
        self.assertEqual(assumption['native_size'],[64,48])
        self.assertEqual(G.digest(output/'camera-prior/camera-prior.json'),assumption['report_sha256'])
        self.assertEqual(G.digest(output/'camera-assumption.json'),assumption['result_record']['sha256'])
        for flag in ['camera_calibrated','physical_scale_known','physical_accuracy_verified',
                     'prior_geometry_or_poses_used','learned_depth_or_point_maps_used']:
            self.assertFalse(assumption[flag])

    def test_default_mode_retains_assumed_K_without_prior_metadata(self):
        result,output,reconstruct=self.run_fixture()
        np.testing.assert_array_equal(reconstruct.call_args.args[2],G.camera_matrix(64,48))
        self.assertNotIn('camera_assumption',result)
        self.assertNotIn('learned_camera_prior_used',result)
        self.assertNotIn('camera_provider',json.loads((output/'protocol.json').read_text()))

    def test_invalid_K_fails_before_geometry(self):
        provider=Provider(self.root,K=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.000001]])
        with self.assertRaises(ValueError):self.run_fixture(provider)
        self.assertFalse((self.root/'data/result/geometry.json').exists())

    def test_claimed_calibration_rejected(self):
        provider=Provider(self.root,mutate=lambda r:r.update(camera_calibrated=True))
        with self.assertRaises(ValueError):self.run_fixture(provider)
        self.assertFalse((self.root/'data/result/camera-assumption.json').exists())

    def test_wrong_native_dimensions_rejected(self):
        provider=Provider(self.root,mutate=lambda r:r.update(native_size=[512,384]))
        with self.assertRaises(ValueError):self.run_fixture(provider)

    def test_wrong_TRAIN_source_hash_rejected(self):
        provider=Provider(self.root,mutate=lambda r:r['input_views'][0].update(image_sha256='f'*64))
        with self.assertRaises(ValueError):self.run_fixture(provider)

    def test_reserved_source_rejected_even_if_provider_reports_it(self):
        def mutate(report):report['input_views'][0]['split']='heldout'
        with self.assertRaises(ValueError):self.run_fixture(Provider(self.root,mutate=mutate))

    def test_learned_depth_claim_rejected(self):
        with self.assertRaises(ValueError):
            self.run_fixture(Provider(self.root,mutate=lambda r:r.update(learned_depth_or_point_maps_used=True)))

    def test_incomplete_call_receipts_rejected(self):
        with self.assertRaises(ValueError):
            self.run_fixture(Provider(self.root,mutate=lambda r:r.update(completed_model_calls=1)))

    def test_changed_input_integrity_rejected(self):
        with self.assertRaises(ValueError):
            self.run_fixture(Provider(self.root,mutate=lambda r:r.update(input_integrity={'unchanged':False})))

    def test_sensitivity_mix_rejected_before_inference_or_output(self):
        provider=Provider(self.root)
        with self.assertRaises(ValueError):
            G.run(self.program,'part',self.root/'data/result',root=self.root,camera_provider=provider)
        self.assertFalse(provider.called)
        self.assertFalse((self.root/'data/result').exists())

    def test_mixed_dimensions_rejected_before_provider(self):
        self.frames[-1]['image']=np.zeros((24,32,3),np.uint8)
        provider=Provider(self.root)
        with self.assertRaises(ValueError):self.run_fixture(provider)
        self.assertFalse(provider.called)

    def test_correspondence_wrapper_forwards_camera_provider(self):
        camera=object();correspondence=object()
        with patch.object(O,'CorrespondenceProvider',return_value=correspondence),patch.object(G,'run',return_value={'fixture':True}) as run:
            result=O.run(self.program,'part',self.root/'data/result',root=self.root,sensitivity=False,camera_provider=camera)
        self.assertEqual(result,{'fixture':True})
        self.assertIs(run.call_args.kwargs['camera_provider'],camera)
        self.assertIs(run.call_args.kwargs['correspondence_provider'],correspondence)

    def test_provider_cannot_rebind_mutated_input_dicts(self):
        outer=self
        class MutatingProvider(Provider):
            def estimate(self,frames,output):
                # Attempt to substitute reserved bytes under a TRAIN label.
                frames[0].update(image_url=outer.frames[2]['image_url'],
                    image_sha256=outer.frames[2]['image_sha256'])
                return super().estimate(frames,output)
        with self.assertRaises(ValueError):self.run_fixture(MutatingProvider(self.root))

    def test_noncentral_nonsquare_full_K_reconstructs_math_fixture(self):
        from tests.test_observed_geometry import synthetic_scene
        xyz,frames,tracks,nodes,old,truth=synthetic_scene(count=60,views=10,noise=0.)
        K=np.array([[710.,0.,301.5],[0.,730.,231.5],[0.,0.,1.]])
        for fi,frame in enumerate(frames):
            uv,_=G.project(xyz,truth[fi],K);frame['xy']=uv
            if frame['split']=='train':
                for j,tid in enumerate(tracks):tracks[tid][fi]=uv[j]
        model,_=G.reconstruct(tracks,frames,K,max_nfev=4)
        self.assertIsNotNone(model)
        rows,predictions,errors=G.evaluate_heldout(model,tracks,frames,nodes,K)
        self.assertEqual(sum(r['passes'] for r in rows),2)
        self.assertLess(max(errors),.01)


if __name__=='__main__':unittest.main()
