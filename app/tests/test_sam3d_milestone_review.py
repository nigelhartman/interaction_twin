"""Refusal/source/native-camera contracts only. Never builds a fake model page."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import upgrade_bundles as U

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sam3d_milestone_review',ROOT/'scripts/build_sam3d_milestone_review.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)


class NativeReviewBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.source=self.root/'data/import-v1';self.source.mkdir(parents=True)
        self.destination=self.root/'data/review-v1'

    def test_missing_actual_import_never_creates_a_page(self):
        with self.assertRaisesRegex(ValueError,'actual validated native import'):R.build(self.source,self.destination,root=self.root)
        self.assertFalse(self.destination.exists())

    def test_explicit_fixture_import_never_creates_a_page(self):
        U.write(self.source/'import.json',{'format':'sam3d-native-milestone-readiness','version':'1','fixture':True,'status':'validated_native_candidate_pending_review','import_directory':'data/import-v1'})
        with self.assertRaisesRegex(ValueError,'fixture'):R.build(self.source,self.destination,root=self.root)
        self.assertFalse(self.destination.exists())

    def test_changed_import_inventory_rejects_before_model_access(self):
        U.write(self.source/'import.json',{'format':'sam3d-native-milestone-readiness','version':'1','fixture':False,'status':'validated_native_candidate_pending_review','import_directory':'data/import-v1','import_files':[]})
        (self.source/'unexpected.json').write_text('{"software_contract":true}')
        with self.assertRaisesRegex(ValueError,'inventory/hash'):R.build(self.source,self.destination,root=self.root)
        self.assertFalse(self.destination.exists())

    def test_native_scene_has_no_pose_or_common_gauge_and_preserves_primitives(self):
        primitives=[{'software_contract':True}]
        decoded={'source_glb_sha256':'a'*64,'primitives':primitives,'native_scene':{'target':[0,0,0],'span':1,'initial_distance':2},'mesh_stats':{'software_contract':True}}
        value=R.native_scene(decoded,'part','Y',[540,960])
        self.assertIs(value['scene']['objects'][0]['primitives'],primitives)
        self.assertEqual(value['scene']['objects'][0]['frames'],[])
        self.assertIsNone(value['scene']['source_projection_rows']);self.assertEqual(value['scene']['timeline'],[])
        self.assertFalse(value['scene']['physical_assembly_verified'])

    def test_compact_scene_preserves_exact_json_values_and_existing_browser_budget(self):
        value={'coordinates':[[.123456789,2.,3.],[4.,5.,6.]],'faces':[[0,1,2]]}
        compact=R.scene_json_bytes(value)
        self.assertEqual(json.loads(compact),value)
        self.assertLess(len(compact),len(R.json_bytes(value)))
        with patch.object(R,'MAX_SCENE_JSON',len(compact)):
            self.assertEqual(R.scene_json_bytes(value),compact)
        with patch.object(R,'MAX_SCENE_JSON',len(compact)-1):
            with self.assertRaisesRegex(ValueError,'unchanged browser JSON budget'):R.scene_json_bytes(value)

    def test_camera_patch_refuses_changed_renderer_without_touching_shared_file(self):
        path=ROOT/'workspace/scene.js';before=path.read_bytes()
        patched=R.camera_only_renderer(before.decode())
        self.assertEqual(path.read_bytes(),before)
        self.assertIn("this.object().native_up==='Y'",patched)
        with self.assertRaisesRegex(ValueError,'renderer changed'):R.camera_only_renderer('unexpected new renderer')

    def test_camera_up_changes_view_vectors_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'camera-only.js';path.write_text(R.camera_only_renderer((ROOT/'workspace/scene.js').read_text()))
            code="const fs=require('node:fs'),vm=require('node:vm');const context={module:{exports:{}}};vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),context);const f=context.module.exports.Scene.prototype.basis;const a=f.call({yaw:0,pitch:0,inspection:'native',object:()=>({native_up:'Y'})});const b=f.call({yaw:0,pitch:0,inspection:'native',object:()=>({native_up:'Z'})});console.log(JSON.stringify({Y:a,Z:b}));"
            result=subprocess.run(['node','-e',code,str(path)],capture_output=True,text=True,check=True)
        data=json.loads(result.stdout)
        self.assertEqual(data['Y']['direction'],[0,0,1]);self.assertEqual(data['Y']['up'],[0,1,0])
        self.assertEqual(data['Z']['direction'],[0,-1,0]);self.assertEqual(data['Z']['up'],[0,0,1])

    def test_neutral_background_changes_only_isolated_clear_colour(self):
        path=ROOT/'workspace/scene.js';before=path.read_bytes()
        dark=R.camera_only_renderer(before.decode())
        neutral=R.camera_only_renderer(before.decode(),background='neutral')
        self.assertEqual(neutral,dark.replace('g.clearColor(.031,.047,.075,1);','g.clearColor(.91,.93,.90,1);'))
        self.assertEqual(path.read_bytes(),before)
        self.assertIn('gl_FragColor=mix(vColor,texture2D(uTexture,vUV),uHasTexture)*uFactor;',neutral)
        with self.assertRaisesRegex(ValueError,'Unsupported review display background'):
            R.camera_only_renderer(before.decode(),background='unsupported')
        with self.assertRaisesRegex(ValueError,'background changed'):
            R.camera_only_renderer(before.decode().replace('g.clearColor(.031,.047,.075,1);','g.clearColor(0,0,0,1);'),background='neutral')

    def test_browser_source_integrity_and_native_ui_contracts(self):
        result=subprocess.run(['node','--test',str(ROOT/'tests/native_review_harness.cjs')],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
