import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('flagship_contract', ROOT/'scripts/flagship_capture_contract_v1.py')
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)


class CaptureContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'data').mkdir()
        self.video = self.root/'capture.webm'
        self.video.write_bytes(b'not decoded in transport-only fixtures')
        self.catalog = {'path':'catalog.json','sha256':'a'*64,'bytes':1}
        self.asset = self.root/'data/shape.glb'
        self.asset.write_bytes(b'exact fixture asset')
        asset_ref = F.file_ref(self.asset,self.root)
        scene_ref = {'path':'data/scene.json','sha256':'b'*64,'bytes':1}
        program_ref = {'path':'data/program.json','sha256':'c'*64,'bytes':1}
        binding = {'record':{'id':'fixture-record','source':{'sha256':'d'*64},'interaction':program_ref},'entry':{'manifest':scene_ref},
                   'scene':{'original_program':program_ref,'components':[{'id':'power_cable','asset':asset_ref},{'id':'adapter_early','asset':asset_ref}]},
                   'observed':{'samples':[{'source_sample_index':5,'timestamp_s':2.7962,'asset':asset_ref}]}}
        self.context = {'catalog':self.catalog,'records':{'phone':binding,'charger':binding}}
        roles = ['phone','cable','charger']
        models = [{'role':role,'asset':asset_ref,'source_sha256':'d'*64,'scene_ref':scene_ref,'source_program':program_ref,
                   'representation':'source_visible_2.5D_estimated_depth' if role=='phone' else 'saved_native_or_declared_derived_mesh',
                   'observation':{'source_sample_index':5,'timestamp_s':2.7962} if role=='phone' else None} for role in roles]
        trace = [{'time_s':i*.08,'views':[{'role':role,'projected_bounds':[[n*640+100,100,0]]*8} for n,role in enumerate(roles)]}for i in range(101)]
        self.doc = {'format':'interaction-twin-flagship-multiple-model-capture','version':1,
                    'current_catalog':self.catalog,'roles':roles,'models':models,'duration_s':8,'canvas':[1920,1080],
                    'bytes':self.video.stat().st_size,'video_sha256':hashlib.sha256(self.video.read_bytes()).hexdigest(),
                    'trace':trace,'geometry_loaded_actual_WebGL':True,'source_assets_modified':False,
                    'charging_or_insertion_executed':False,'physical_accuracy_verified':False,'new_API_calls':0,'new_GPU_calls':0}

    def accept(self, doc=None):
        return F.validate_receipt('multiple_models',doc or self.doc,self.video,self.context,self.root)

    def test_complete_typed_fixture_accepts_transport_only(self):
        result = self.accept()
        self.assertTrue(result['lineage_and_clock_passed'])
        self.assertFalse(result['physical_accuracy_verified'])

    def test_stale_catalog_is_rejected(self):
        doc=copy.deepcopy(self.doc);doc['current_catalog']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'stale'):self.accept(doc)

    def historical_capture(self):
        doc=copy.deepcopy(self.doc)
        raw=(json.dumps({'records':[self.context['records']['phone']['record']]},sort_keys=True)+'\n').encode()
        sha=hashlib.sha256(raw).hexdigest()
        path=self.root/'showcase/public'/('library-catalog-'+sha+'.json')
        path.parent.mkdir(parents=True);path.write_bytes(raw)
        doc['current_catalog']=F.file_ref(path,self.root)
        return doc,path

    def test_additive_catalog_revision_keeps_exact_models_eligible(self):
        doc,path=self.historical_capture()
        result=self.accept(doc)
        self.assertFalse(result['capture_catalog_equals_current'])
        self.assertTrue(result['lineage_and_clock_passed'])

    def test_historical_catalog_cannot_bypass_source_program_change(self):
        doc,path=self.historical_capture()
        self.context['records']['phone']['record']['interaction']=dict(self.context['records']['phone']['record']['interaction'],sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'Historical capture source/program changed'):self.accept(doc)

    def test_historical_catalog_bytes_must_remain_exact(self):
        doc,path=self.historical_capture();path.write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'bytes changed'):self.accept(doc)

    def test_historical_catalog_cannot_bypass_current_surface_change(self):
        doc,path=self.historical_capture()
        binding=self.context['records']['phone']
        binding['observed']['samples'][0]['asset']=dict(binding['observed']['samples'][0]['asset'],sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'current supported surface'):self.accept(doc)

    def test_duplicate_role_is_not_a_third_model(self):
        doc=copy.deepcopy(self.doc);doc['roles']=['phone','cable','charger','charger']
        with self.assertRaisesRegex(ValueError,'duplicate'):self.accept(doc)

    def test_wrong_model_geometry_is_rejected(self):
        doc=copy.deepcopy(self.doc);doc['models'][1]['asset']=dict(doc['models'][1]['asset'],sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'current native'):self.accept(doc)

    def test_unknown_phone_cannot_fallback_to_original_native(self):
        doc=copy.deepcopy(self.doc);doc['models'][0]['observation']['source_sample_index']=39
        with self.assertRaisesRegex(ValueError,'supported surface'):self.accept(doc)

    def test_partial_phone_may_not_be_called_native(self):
        doc=copy.deepcopy(self.doc);doc['models'][0]['representation']='SAM3D repaired'
        with self.assertRaisesRegex(ValueError,'mislabeled'):self.accept(doc)

    def test_missing_real_view_is_rejected(self):
        doc=copy.deepcopy(self.doc);doc['trace'][50]['views'].pop()
        with self.assertRaisesRegex(ValueError,'Missing model'):self.accept(doc)

    def test_bounds_outside_own_viewport_are_rejected(self):
        doc=copy.deepcopy(self.doc);doc['trace'][50]['views'][1]['projected_bounds'][0]=[630,100,0]
        with self.assertRaisesRegex(ValueError,'clipped'):self.accept(doc)

    def test_trace_ending_early_is_rejected(self):
        doc=copy.deepcopy(self.doc);doc['trace']=doc['trace'][:-2]
        with self.assertRaisesRegex(ValueError,'sparse|complete'):self.accept(doc)

    def test_reversed_clock_is_rejected(self):
        doc=copy.deepcopy(self.doc);doc['trace'][50]['time_s']=0
        with self.assertRaisesRegex(ValueError,'reversed'):self.accept(doc)

    def test_receipt_cannot_certify_other_video(self):
        self.video.write_bytes(b'different clip')
        with self.assertRaisesRegex(ValueError,'does not match'):self.accept()

    def test_path_does_not_admit_parent_or_private(self):
        for name in ['../elsewhere','data/../capture.webm','.secrets/meta.env','/tmp/escape']:
            with self.subTest(name=name),self.assertRaises(ValueError):F.read_ref({'path':name},self.root)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'Optional local video tools absent.')
    def test_complete_real_video_with_blank_scene_is_not_accepted(self):
        video = self.root/'blank.mp4'
        subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i',
                        'color=c=white:s=160x90:d=1:r=30','-c:v','libx264',str(video)],check=True)
        with self.assertRaisesRegex(ValueError,'blank/cleared'):
            F.probe_and_pixels(video,1,[160,90])


class EditorialClockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('flagship_builder_clock',ROOT/'scripts/build_flagship_trailer_20261104_v1.py')
        cls.film=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.film)

    def plan(self):
        return {'goal':{'component_ids':['a','b']},'actions':[
            {'operation':'wait','duration_s':2},
            {'operation':'move_to','duration_s':1},
            {'operation':'set_joint','value':1,'duration_s':2},
            {'operation':'detach','object_id':'a','duration_s':1},
            {'operation':'move_to','duration_s':1},
            {'operation':'detach','object_id':'b','duration_s':1},
            {'operation':'move_to','duration_s':1},
            {'operation':'set_joint','value':0,'duration_s':2},
            {'operation':'wait','duration_s':1}]}

    def test_retiming_keeps_every_action_order_and_complete_end(self):
        knots=self.film.robot_editorial_knots(self.plan())
        self.assertEqual(knots,[(0,0),(2,2),(4.5,5),(8,7),(11.5,9),(15,12)])
        times=[self.film.interpolate_clock(knots,i/30)for i in range(451)]
        self.assertEqual(times[0],0);self.assertEqual(times[-1],12)
        self.assertTrue(all(a<b for a,b in zip(times,times[1:])))

    def test_each_release_and_retreat_have_exact_editorial_boundary(self):
        knots=self.film.robot_editorial_knots(self.plan())
        self.assertEqual(self.film.interpolate_clock(knots,8),7)
        self.assertEqual(self.film.interpolate_clock(knots,11.5),9)

    def test_late_or_early_request_does_not_extend_or_reverse_clock(self):
        knots=self.film.robot_editorial_knots(self.plan())
        self.assertEqual(self.film.interpolate_clock(knots,-1),0)
        self.assertEqual(self.film.interpolate_clock(knots,99),12)

    def test_reversed_goal_order_cannot_retime_backwards(self):
        plan=self.plan();plan['goal']['component_ids']=['b','a']
        with self.assertRaises(AssertionError):self.film.robot_editorial_knots(plan)


if __name__ == '__main__':unittest.main()
