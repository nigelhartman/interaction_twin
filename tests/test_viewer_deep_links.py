"""Actual viewer JavaScript time parsing, with unchanged pose gates."""
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT=Path(__file__).parents[1]


@unittest.skipUnless(shutil.which('node'),'Node is required for actual JavaScript checks')
class ViewerDeepLinkTests(unittest.TestCase):
    def test_finite_in_range_source_time_and_existing_default_on_invalid_input(self):
        code=r'''
const assert=require('assert');
for(const file of ['generated_motion_observer.js','jabra_assembly_observer.js']){
  const api=require('./'+file),duration=33.636278,fallback=13.001144;
  for(const [query,wanted] of [['?t=0',0],['?t=12.401089',12.401089],['?t=33.636278',duration],['?other=1&t=2.5',2.5]])
    assert.strictEqual(api.initialSourceTime(query,duration,fallback),wanted,file+' '+query);
  for(const query of ['', '?x=1','?t=','?t=%20','?t=-.1','?t=Infinity','?t=NaN','?t=33.7','?t=words'])
    assert.strictEqual(api.initialSourceTime(query,duration,fallback),fallback,file+' '+query);
}
'''
        subprocess.run([shutil.which('node'),'-e',code],cwd=ROOT,check=True,capture_output=True,text=True)

    def test_deep_link_keeps_unsupported_pose_hidden_and_does_not_interpolate(self):
        code=r'''
const assert=require('assert'),g=require('./generated_motion_observer.js'),a=require('./jabra_assembly_observer.js');
const rows=[{timestamp_s:1,visible:true},{timestamp_s:2,visible:false},{timestamp_s:3,visible:true}];
let time=g.initialSourceTime('?t=2.5',5,1);assert.strictEqual(time,2.5);
assert.strictEqual(g.selectRecord(rows,time).visible,false);
assert.strictEqual(a.selectRecord({frames:rows,max_hold_s:1.05},time).visible,false);
assert.strictEqual(g.selectRecord(rows,1.5).row.timestamp_s,1);
assert.strictEqual(a.selectRecord({frames:rows,max_hold_s:.2},1.5).visible,false);
assert.strictEqual(g.selectRecord(rows,0).visible,false);
assert.strictEqual(a.selectRecord({frames:rows,max_hold_s:1},0).visible,false);
'''
        subprocess.run([shutil.which('node'),'-e',code],cwd=ROOT,check=True,capture_output=True,text=True)
