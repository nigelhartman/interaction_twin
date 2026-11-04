"""Meaningful software-review controls and fixture/native separation checks."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('appearance_builder', ROOT/'scripts/build_native_appearance_review.py')
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)


class AppearanceReviewTests(unittest.TestCase):
    def test_known_fixture_preserves_uvs_and_separate_faces(self):
        texture = B.calibration_texture()
        raw = B.fixture_glb(B.cube_parts(), texture, B.cube_materials())
        mesh = B.load_native_glb(raw)
        self.assertFalse(mesh.receipt['legacy_vertex_compatibility'])
        self.assertIsNone(mesh.vertex_colors)
        self.assertEqual(len(mesh.vertices), 24)
        self.assertEqual(len(np.unique(mesh.vertices, axis=0)), 8)
        self.assertEqual(len(mesh.faces), 12)
        self.assertFalse(mesh.receipt['uv_to_vertex_color_conversion'])
        self.assertIn(b'SOFTWARE TEST FIXTURE', raw)

    def test_baseline_comparison_detects_real_changes(self):
        left = {'rgb': np.zeros((2, 2, 3), np.uint8), 'mask': np.ones((2, 2), bool),
                'depth': np.ones((2, 2), np.float32), 'face_index': np.zeros((2, 2), np.int32)}
        right = {k: v.copy() for k, v in left.items()}
        self.assertTrue(B.comparison_metrics(left, right)['pixel_equality'])
        right['rgb'][0, 1, 2] = 7
        result = B.comparison_metrics(left, right)
        self.assertEqual(result['changed_rgb_pixels'], 1)
        self.assertEqual(result['rgb_max_absolute_difference_uint8'], 7)
        self.assertFalse(result['pixel_equality'])

    @unittest.skipUnless(shutil.which('node'), 'Node runtime not available')
    def test_controller_scrub_mode_play_and_fixture_guard(self):
        script = r'''
const assert=require('assert'),A=require(process.argv[1]);
const frame=i=>({index:i,fixture:true,known_pose:true,images:{rgb:'data/review/f'+i+'.png',depth:'data/review/d'+i+'.png',faces:'data/review/p'+i+'.png'}});
const data={format:'interaction-twin-native-appearance-software-review',fixture:true,physical_accuracy_verified:false,frames:[frame(0),frame(1),frame(2)],examples:[],baseline:{fixture:false}};
let latest, tick, cleared=false;
const control=A.createController(data,s=>latest=s,{setInterval(fn){tick=fn;return 23},clearInterval(id){assert.equal(id,23);cleared=true;}});
assert.equal(latest.index,0);control.seek(99);assert.equal(latest.index,2);control.step(-1);assert.equal(latest.index,1);
control.seek(-5);assert.equal(latest.index,0);control.mode('depth');assert.equal(latest.mode,'depth');control.mode('invalid');assert.equal(latest.mode,'depth');
control.play();assert.equal(latest.playing,true);tick();assert.equal(latest.index,1);tick();tick();assert.equal(latest.index,0);control.pause();assert.equal(latest.playing,false);assert(cleared);
control.difference();assert(latest.difference);control.difference();assert(!latest.difference);
assert.equal(A.assetURL('data/review/file.png','file:'),'../data/review/file.png');assert.equal(A.assetURL('data/review/file.png','http:'),'/files/data/review/file.png');
for(const path of ['https://example.com/a','data/review/../secret','data/review/.hidden','data/review/a.png?x','data/review/a%2fb','data//a'])assert.equal(A.assetURL(path,'http:'),null);
assert.throws(()=>A.createController({...data,fixture:false},()=>{}));assert.throws(()=>A.createController({...data,physical_accuracy_verified:true},()=>{}));
assert.throws(()=>A.createController({...data,frames:[{...frame(0),known_pose:false},frame(1)]},()=>{}));
control.destroy();console.log('Appearance controls passed');
'''
        result = subprocess.run(['node', '-e', script, str(ROOT/'workspace/appearance.js')], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Appearance controls passed', result.stdout)

    def test_page_has_no_external_or_processing_dependencies(self):
        page = (ROOT/'workspace/appearance.html').read_text()
        controller = (ROOT/'workspace/appearance.js').read_text()
        self.assertIn('SOFTWARE TEST FIXTURE', page)
        self.assertIn('appearance-review-data.js', page)
        self.assertNotIn('https://', page)
        self.assertNotIn('fetch(', controller)
        self.assertNotIn('/api/', controller)
        for name in ['frame-slider', 'play', 'previous', 'next', 'difference', 'manifest-link']:
            self.assertIn('id="'+name+'"', page)


if __name__ == '__main__':
    unittest.main()
