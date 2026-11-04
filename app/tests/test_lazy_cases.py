"""Exact scene derivatives and shipped loader; no model or hosted inference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('lazy_case_builder_tests',ROOT/'scripts/build_workspace.py')
builder=importlib.util.module_from_spec(spec)
_search_path=list(sys.path)
try:spec.loader.exec_module(builder)
finally:sys.path[:]=_search_path


class LazySceneTests(unittest.TestCase):
    def test_derivative_keeps_exact_scene_and_original_source_binding(self):
        data={'cases':[{'id':'synthetic-case','original_sha256':'a'*64,'scene':{'objects':[{'vertices':[[1.,2.,3.]],'frames':[{'timestamp_s':.123456,'visible':False}]}]},'interaction':{'explicit_test_fixture':True}},
                       {'id':'empty-fixture','original_sha256':'b'*64,'scene':None}], 'physical_accuracy_verified':False}
        original=json.dumps(data)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);full=out/'cases.json';full.write_text(original)
            builder.lazy_payload(data,out)
            index=json.loads((out/'cases-index.json').read_text());row=index['cases'][0]
            self.assertIsNone(row['scene']);self.assertNotIn('scene_ref',index['cases'][1])
            path=out/row['scene_ref']['path'];raw=path.read_bytes();pack=json.loads(raw)
            self.assertEqual(hashlib.sha256(raw).hexdigest(),row['scene_ref']['sha256'])
            self.assertEqual(pack['scene'],data['cases'][0]['scene'])
            self.assertEqual(pack['interaction_sha256'],row['original_sha256'])
            self.assertEqual(pack['case_id'],row['id'])
            self.assertEqual(full.read_text(),original);self.assertEqual(json.dumps(data),original)
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError,'immutable'):builder.lazy_payload(data,out)

    @unittest.skipUnless(shutil.which('node'),'Node required; no installation attempted')
    def test_actual_browser_loader_rejects_changed_or_foreign_data_and_bounds_cache(self):
        result=subprocess.run(['node',str(ROOT/'tests/lazy_case_harness.cjs')],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'],'passed')

    def test_current_index_and_every_scene_equal_the_preserved_full_catalog(self):
        full=json.loads((ROOT/'workspace/cases.json').read_text())
        index=json.loads((ROOT/'workspace/cases-index.json').read_text())
        self.assertEqual(len(full['cases']),len(index['cases']))
        self.assertLess((ROOT/'workspace/cases-index.js').stat().st_size,(ROOT/'workspace/cases.js').stat().st_size/2)
        for old,new in zip(full['cases'],index['cases']):
            with self.subTest(case=old['id']):
                restored=dict(new);ref=restored.pop('scene_ref',None)
                if ref:
                    raw=(ROOT/'workspace'/ref['path']).read_bytes()
                    self.assertEqual(hashlib.sha256(raw).hexdigest(),ref['sha256'])
                    self.assertEqual(len(raw),ref['bytes'])
                    pack=json.loads(raw);self.assertEqual(pack['case_id'],old['id'])
                    self.assertEqual(pack['interaction_sha256'],old['original_sha256'])
                    restored['scene']=pack['scene']
                self.assertEqual(restored,old)


if __name__=='__main__':unittest.main()
