import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

spec=importlib.util.spec_from_file_location('dense_jabra',Path(__file__).parents[1]/'scripts/stage4_dense_jabra.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


class DenseJabraTests(unittest.TestCase):
    def test_crop_transform_preserves_source_motion_and_round_trips(self):
        masks=[]
        for x,y in [(20,30),(65,105)]:
            mask=np.zeros((200,150),bool);mask[y:y+20,x:x+12]=True;masks.append(mask)
        transforms=[M.crop_transform(mask) for mask in masks]
        full=np.array([[25.,35.],[70.,110.]])
        local=[M.transform_uv(full[i:i+1],tr['source_to_crop']) for i,tr in enumerate(transforms)]
        recovered=np.vstack([M.transform_uv(points,tr['crop_to_source']) for points,tr in zip(local,transforms)])
        np.testing.assert_allclose(recovered,full,atol=1e-10)
        np.testing.assert_allclose(recovered[1]-recovered[0],[45,75],atol=1e-10)
        np.testing.assert_allclose(local[1]-local[0],0,atol=1e-10)

    def test_crop_padding_keeps_edge_objects_and_refuses_empty_mask(self):
        mask=np.zeros((60,50),bool);mask[:6,:4]=True
        tr=M.crop_transform(mask)
        self.assertLess(tr['source_bounds_xyxy'][0],0)
        self.assertLess(tr['source_bounds_xyxy'][1],0)
        uv=M.transform_uv(np.array([[0.,0.],[3.,5.]]),tr['source_to_crop'])
        self.assertTrue(((uv>=0)&(uv<384)).all())
        with self.assertRaises(ValueError):M.crop_transform(np.zeros((60,50),bool))

    def test_adaptive_crop_does_not_change_native_image_coordinates(self):
        mask=np.zeros((960,540),bool);mask[400:500,200:300]=True
        tr=M.crop_transform(mask)
        self.assertEqual(tr['source_bbox_xyxy'],[200,400,300,500])
        self.assertFalse(tr['new_image_detail_created'])
        self.assertEqual(tr['output_size'],[384,384])
        a=np.array(tr['source_to_crop']);b=np.array(tr['crop_to_source'])
        np.testing.assert_allclose(a@b,np.eye(3),atol=1e-12)
        np.testing.assert_allclose(M.transform_uv(np.array([[249.5,449.5]]),a),[[191.5,191.5]],atol=1e-10)

    def test_timestamp_transfer_keeps_model_coordinates_and_rejects_manual_seeds(self):
        src=[{'index':1,'timestamp_s':1.,'width':540,'height':960},{'index':2,'timestamp_s':2.,'width':540,'height':960}]
        dense=[{'index':i+1,'timestamp_s':1+i*.2,'width':540,'height':960} for i in range(6)]
        cfg={'manual_seed_boxes_used':False,'tracks':[{'sample_range':[1,2],'prompts':[{'sample_index':2,'box_xyxy':[12,30,55,60]}]}]}
        with patch.object(M,'sha',return_value='hash'):
            result=M.transfer_config(cfg,src,dense)
        self.assertEqual(result['tracks'][0]['sample_range'],[1,6])
        self.assertEqual(result['tracks'][0]['prompts'][0]['box_xyxy'],[12,30,55,60])
        self.assertEqual(cfg['tracks'][0]['prompts'][0]['sample_index'],2)
        cfg['manual_seed_boxes_used']=True
        with self.assertRaises(ValueError):M.transfer_config(cfg,src,dense)
        with self.assertRaises(ValueError):M.nearest_sample(dense,40.)

    def test_supported_time_compares_seconds_not_dense_frame_counts(self):
        def rec(t):return {'timestamp_s':t,'status':'supported_conditional_motion_candidate' if t<2 else 'unsupported'}
        coarse=[rec(t) for t in [0.,1.,2.,3.]]
        dense=[rec(t) for t in np.linspace(0,3,16)]
        a=M.supported_time(coarse,0,3);b=M.supported_time(dense,0,3)
        self.assertAlmostEqual(a['supported_time_s'],1.5)
        self.assertAlmostEqual(b['supported_time_s'],1.9)
        self.assertLess(abs(a['supported_time_s']-b['supported_time_s']),.5)
        none=M.supported_time([dict(r,status='unsupported') for r in dense],0,3)
        self.assertEqual(none['supported_time_s'],0)


if __name__=='__main__':unittest.main()
