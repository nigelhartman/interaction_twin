"""Offline evidence/display contracts for the new source-grounded explainer."""
import importlib.util
import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('brand_explainer',ROOT/'scripts/build_brand_explainer_20261103_v1.py')
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)

class BrandExplainerTests(unittest.TestCase):
    def test_fixed_crop_includes_temporal_extremes_with_constant_padding(self):
        with tempfile.TemporaryDirectory()as tmp:
            paths=[]
            for n,rect in enumerate([(30,50,80,100),(500,400,550,450)]):
                im=Image.new('RGB',(1920,1080),B.BG);ImageDraw.Draw(im).rectangle(rect,fill='#222222')
                path=Path(tmp)/f'{n}.png';im.save(path);paths.append(path)
            # One union over the whole shot, not per-frame tracking or zoom.
            self.assertEqual(B.fixed_union_bbox(paths,(0,0,960,540),padding=18),[12,32,569,469])

    def test_quadrant_crop_cannot_absorb_another_component(self):
        with tempfile.TemporaryDirectory()as tmp:
            im=Image.new('RGB',(1920,1080),B.BG);d=ImageDraw.Draw(im)
            d.rectangle((900,500,959,539),fill='black');d.rectangle((1100,50,1300,300),fill='black')
            p=Path(tmp)/'frame.png';im.save(p)
            self.assertEqual(B.fixed_union_bbox([p],(0,0,960,540),padding=18),[882,482,960,540])

    def test_blank_or_wrong_resolution_cannot_become_a_native_crop(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'frame.png';Image.new('RGB',(1920,1080),B.BG).save(p)
            with self.assertRaisesRegex(ValueError,'No native geometry'):B.fixed_union_bbox([p],(0,0,960,540))
            Image.new('RGB',(1280,720),'black').save(p)
            with self.assertRaisesRegex(ValueError,'1080p'):B.fixed_union_bbox([p],(0,0,960,540))

    def fixture(self):
        return {'_object_program':{'joints':[{'parent_id':'base_exact','child_id':'lid_exact','type':'revolute','status':'estimated',
          'axis':[.1234,-.5678,.9],'samples':[{'source_sample_index':21,'timestamp_s':6.667256,'q_rad':1.23456}],
          'evidence_ids':['source_000021']}]}}

    def test_readable_excerpt_uses_saved_ids_angles_and_typed_time(self):
        excerpt=json.loads('\n'.join(B.descriptive_lines(self.fixture())))
        self.assertEqual(excerpt['parent_id'],'base_exact');self.assertEqual(excerpt['child_id'],'lid_exact')
        self.assertEqual(excerpt['axis'],[.12,-.57,.9]);self.assertEqual(excerpt['q_rad'],1.23)
        self.assertEqual(excerpt['timestamp_s'],6.67);self.assertNotEqual(excerpt['timestamp_s'],21)
        self.assertEqual(excerpt['evidence_ids'],['source_000021']);self.assertIs(excerpt['physical_accuracy_verified'],False)

    def test_absent_citation_is_not_invented_for_trailer(self):
        p=self.fixture();p['_object_program']['joints'][0]['evidence_ids']=['source_000042']
        with self.assertRaisesRegex(ValueError,'citation is absent'):B.descriptive_lines(p)

    def test_story_uses_contiguous_short_real_scopes_and_no_physics_claim(self):
        self.assertEqual(B.CHAPTERS[0][0],0);self.assertEqual(B.CHAPTERS[-1][1],55)
        for before,after in zip(B.CHAPTERS,B.CHAPTERS[1:]):self.assertEqual(before[1],after[0])
        self.assertIn('no physics or trained policy',B.CHAPTERS[-2][3])
        self.assertIn('estimated',B.CHAPTERS[4][3])
        self.assertEqual(B.DURATION*B.FPS,1320)

    def annotation_fixture(self):
        identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];times=[15,16,17,18,19,20,31,32]
        artifact={'path':'data/synthetic/fixture.glb','sha256':'b'*64,'bytes':100}
        scene={'format':'sam3d-common-video-scene','version':1,'original_program':artifact,
          'source':{'sha256':'a'*64,'width':540,'height':960,'duration_s':33,'time_domain':'source_video_presentation_seconds'},
          'camera':{'K':[[800,0,269.5],[0,810,479.5],[0,0,1]],'width':540,'height':960,'world_from_camera':identity},
          'common_frame':{'id':'current_source_camera_cv','coordinate_convention':'+Xright+Ydown+Zforward','metric_scale_verified':False,'assembly_verified':False},
          'physical_accuracy_verified':False,'accepted_motion_poses':0,'timing':{'max_hold_s':0,'max_interpolation_gap_s':1,'max_estimated_gap_s':1},
          'components':[{'id':name,'label':name,'asset':artifact,'poses':[{'timestamp_s':t,'status':'estimated','camera_from_native_mesh':identity}for t in times]}for name in ['base','lid']],
          'mechanics':{'format':'sam3d-constrained-mechanisms','version':1,'physical_accuracy_verified':False,'joints':[{
            'id':'hinge','type':'revolute','status':'estimated','parent_id':'base','child_id':'lid','parent_from_joint':[[2,0,0,.1],[0,2,0,.2],[0,0,2,1],[0,0,0,1]],
            'joint_from_child_zero':identity,'axis':[1,0,0],'limits_rad':[0,3],'samples':[{'timestamp_s':t,'status':'estimated','q_rad':.5}for t in times]}]}}
        doc={'config':{'source_sha256':'a'*64},'joint_annotation':{'joint_id':'hinge','axis_length_world':.14},
          'capture_camera':{'matrix_world':identity,'projection_matrix':[[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,-1,0]]},'projection_trace':[]}
        for i in range(241):
            capture=7+i/24;q=capture-7;source=15+q*5/6 if q<=6 else 31+(q-6)/4
            doc['projection_trace'].append({'capture_time_s':capture,'source_time_s':source,'joint_id':'hinge','q_rad':.5,'pivot_world':[.1,-.2,-1],
              'axis_world':[1,0,0],'pivot_canvas':[1056,648,1],'axis_canvas':[[988.8,648,1],[1123.2,648,1]],'annotation_only':True})
        return doc,scene

    def test_diagnostic_annotation_recomputes_similarity_axis_and_projection(self):
        doc,scene=self.annotation_fixture()
        self.assertEqual(B.validate_annotation_trace(doc,scene),{'verified':True,'rows':241})

    def test_wrong_source_clock_pivot_axis_or_projection_cannot_be_accepted(self):
        original,scene=self.annotation_fixture()
        for field,value in [('source_time_s',18),('pivot_world',[0,0,0]),('axis_world',[0,1,0]),('pivot_canvas',[0,0,0]),('axis_canvas',[[0,0,0],[1,1,1]]),('q_rad',2)]:
            with self.subTest(field=field):
                doc=copy.deepcopy(original);doc['projection_trace'][20][field]=value
                with self.assertRaisesRegex(ValueError,'annotation evidence failed'):B.validate_annotation_trace(doc,scene)

if __name__=='__main__':unittest.main()
