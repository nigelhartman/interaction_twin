"""Reload links preserve exact camera/source identity or fail without a substitute."""
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]

@unittest.skipUnless(shutil.which('node'),'Node is needed for the browser URL helper')
class CameraPermalinkTests(unittest.TestCase):
    def test_exact_source_roundtrip_normal_clear_and_invalid_links(self):
        script=r"""
import assert from 'node:assert/strict';
import {readCameraPermalink,writeCameraPermalink} from './workspace/camera-permalink.mjs';
const binding={caseId:'camera-case-v1',trackId:'camera_body',sourceSha256:'a'.repeat(64)};
const expected={...binding,hypothesisId:'moge-prior-v1'};
const base=new URL('http://127.0.0.1:8765/workspace/geometry-lab.html?keep=1#workbench');
assert.equal(readCameraPermalink(base),null);
const link=writeCameraPermalink(base,{id:'diagnostic-1',cameraBinding:binding,cameraHypothesisId:expected.hypothesisId});
assert.deepEqual(readCameraPermalink(link),expected);
assert.equal(link.searchParams.get('run'),'diagnostic-1');
assert.equal(link.searchParams.get('keep'),'1');assert.equal(link.hash,'#workbench');
const normal=writeCameraPermalink(link,{id:'normal-run'});
assert.equal(readCameraPermalink(normal),null);assert.equal(normal.searchParams.get('run'),'normal-run');
assert.equal(base.searchParams.has('run'),false);
for(const bad of ['?camera_case=camera-case-v1','?camera_hypothesis=../viewer.json',
'?camera_case=x&camera_track=y&camera_source='+('a'.repeat(64))+'&camera_hypothesis=z&camera_track=y',
'?camera_case=x&camera_track=y&camera_source=not-a-source&camera_hypothesis=z']){
 assert.throws(()=>readCameraPermalink(new URL(bad,base)),/No other reconstruction was substituted/);
}
assert.throws(()=>writeCameraPermalink(base,{id:'x',cameraBinding:binding}));
console.log('camera permalink source identity checks passed');
"""
        result=subprocess.run(['node','--input-type=module','-e',script],cwd=ROOT,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

if __name__=='__main__':unittest.main()
