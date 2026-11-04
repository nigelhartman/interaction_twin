"""Independent-unit models can be inspected without inventing an assembly."""
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ComponentInspectionTests(unittest.TestCase):
    def test_selected_motion_never_draws_a_sibling_or_borrows_its_camera(self):
        source = r'''
const assert=require('node:assert/strict');
const {Scene,recordAt}=require('./workspace/scene.js');
const a={object_id:'a',native_scene:{span:1},motion_scene:{span:2},source_projection_rows:[[111]],frames:[{timestamp_s:0,visible:true}],max_hold_s:.2};
const b={object_id:'b',native_scene:{span:3},motion_scene:{span:4},source_projection_rows:[[222]],frames:[{timestamp_s:1,visible:true}],max_hold_s:.2};
const s=Object.create(Scene.prototype);s.data={objects:[a,b],native_only:true,scene:{span:999}};s.selected='b';s.inspection='component';s.draw=()=>{};s.reset=()=>{};
assert.deepEqual(s.objectsInView().map(o=>o.object_id),['b']);
assert.equal(s.layout().span,4);assert.deepEqual(s.sourceProjection,[[222]]);
assert.deepEqual(s.setTime(1).map(r=>r.id),['b']);
assert.equal(s.setTime(0)[0].visible,false);
assert.throws(()=>s.inspect('assembly'),/Independent component scales/);
assert.throws(()=>s.inspect('component','invented'),/Unknown component/);
s.inspect('native','a');assert.deepEqual(s.objectsInView().map(o=>o.object_id),['a']);
assert.equal(s.layout().span,1);assert.equal(s.sourceProjection,null);
assert.equal(recordAt(a,.21).visible,false);
console.log(JSON.stringify({status:'passed'}));
'''
        result = subprocess.run(['node', '-e', source], cwd=ROOT, text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout)['status'], 'passed')


if __name__ == '__main__':
    unittest.main()
