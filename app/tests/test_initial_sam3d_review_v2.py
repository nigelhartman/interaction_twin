"""Lossless display transport contracts. Synthetic arrays never become a model page."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import tests.test_initial_sam3d_import as F
import initial_sam3d_import as N

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('initial_native_review_v2_builder',ROOT/'scripts/build_initial_sam3d_review_v2.py')
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)


def analytic_scene():
    points=[[-0.0,.12345678912345678,1.0000000000000002],[2.,-3.,1e-20],[1.,2.,3.]]
    bounds=[[min(p[j] for p in points) for j in range(3)],[max(p[j] for p in points) for j in range(3)]]
    stats={'rendered_vertices':3,'faces':1,'primitive_count':1,'bounds':bounds}
    primitive={'node':'software_only','geometry':'analytic_triangle','material_index':None,
        'vertices':points,'faces':[[0,1,2]],'colors':[[.12345678912345678,.2,.3,1.],[.4,.5,.6,1.],[.7,.8,.9,1.]],
        'uv':[[-0.,.1],[.3333333333333333,.5],[1.,1.]],'texture':None,'base_color_factor':[1.,1.,1.,1.],'appearance_mode':'vertex_rgb'}
    decoded={'source_glb_sha256':'a'*64,'primitives':[primitive],'mesh_stats':stats,
        'native_scene':{'target':[1.,-.5,1.5],'span':6.,'initial_distance':10.8}}
    return V.SHARED.native_scene(decoded,'software_only','Y',[64,48])


class BinaryNativeReviewTests(unittest.TestCase):
    def test_encoding_preserves_input_arrays_and_uses_float64(self):
        value=analytic_scene();original=copy.deepcopy(value);descriptor,geometry=V.encode_scene(value)
        self.assertEqual(value,original)
        data=json.loads(descriptor);self.assertEqual(data['geometry_bytes'],len(geometry))
        p=data['scene']['objects'][0]['primitives'][0]
        self.assertEqual(p['vertices']['binary']['component_type'],'float64_le')
        self.assertEqual(p['faces']['binary']['component_type'],'uint32_le')
        self.assertEqual(len(geometry),3*(3+4+2)*8+3*4)

    def test_exact_cross_language_roundtrip_and_browser_refusal_contracts(self):
        value=analytic_scene();descriptor,geometry=V.encode_scene(value)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'descriptor.json').write_bytes(descriptor);(p/'geometry.bin').write_bytes(geometry)
            (p/'expected.json').write_text(json.dumps(value,allow_nan=False))
            result=subprocess.run(['node',str(ROOT/'tests/initial_native_binary_harness.cjs'),str(p/'descriptor.json'),str(p/'geometry.bin'),str(p/'expected.json')],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_combined_byte_budget_never_expands(self):
        descriptor,geometry=V.encode_scene(analytic_scene());limit=len(descriptor)+len(geometry)
        with patch.object(V,'MAX_SCENE_BYTES',limit):self.assertEqual(V.encode_scene(analytic_scene()),(descriptor,geometry))
        with patch.object(V,'MAX_SCENE_BYTES',limit-1):
            with self.assertRaisesRegex(ValueError,'unchanged browser budget'):V.encode_scene(analytic_scene())

    def test_invalid_arrays_never_encode(self):
        changes=[lambda p:p['vertices'][0].__setitem__(0,float('nan')),lambda p:p['faces'][0].__setitem__(0,3),
            lambda p:p['faces'][0].__setitem__(0,True),lambda p:p['uv'].pop(),lambda p:p['colors'][0].pop()]
        for mutate in changes:
            value=analytic_scene();mutate(value['scene']['objects'][0]['primitives'][0])
            with self.subTest(change=mutate),self.assertRaises(ValueError):V.encode_scene(value)

    def test_statistics_pose_and_aggregate_caps_never_bypass(self):
        value=analytic_scene();value['scene']['objects'][0]['mesh_stats']['faces']=2
        with self.assertRaisesRegex(ValueError,'statistics'):V.encode_scene(value)
        value=analytic_scene();value['scene']['objects'][0]['frames']=[{'manual_pose':True}]
        with self.assertRaisesRegex(ValueError,'source poses'):V.encode_scene(value)
        with patch.object(V,'MAX_RENDERED_VERTICES',2):
            with self.assertRaisesRegex(ValueError,'vertex budget'):V.encode_scene(analytic_scene())
        with patch.object(V,'MAX_TRIANGLE_INDICES',2):
            with self.assertRaisesRegex(ValueError,'index budget'):V.encode_scene(analytic_scene())

    def test_v2_refuses_fixture_import_before_creating_output(self):
        fixture=F.InitialNativeImportTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        imported=fixture.root/'data/import-v1';N.import_output(fixture.handoff,fixture.output,imported,root=fixture.root)
        receipt=V.U.read(imported/'import.json');receipt['fixture']=True;V.U.write(imported/'import.json',receipt)
        output=fixture.root/'data/no-fixture-page'
        with self.assertRaisesRegex(ValueError,'fixture'):V.build(imported,output,root=fixture.root)
        self.assertFalse(output.exists())

    def test_postvalidation_metadata_or_receipt_bytes_cannot_be_substituted(self):
        for name in ('metadata.json','import.json'):
            fixture=F.InitialNativeImportTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
            imported=fixture.root/'data/import-v1';N.import_output(fixture.handoff,fixture.output,imported,root=fixture.root)
            output=fixture.root/'data/no-race-page';verify=V.V1.verified_initial_import
            def changed(*args,**kwargs):
                value=verify(*args,**kwargs)
                p=imported/name;p.write_bytes(p.read_bytes()+b' ')
                return value
            with self.subTest(name=name),patch.object(V.V1,'verified_initial_import',side_effect=changed):
                with self.assertRaisesRegex(ValueError,'changed'):V.build(imported,output,root=fixture.root)
            self.assertFalse(output.exists())

    def test_combined_geometry_failure_preserves_original_and_creates_no_page(self):
        fixture=F.InitialNativeImportTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        imported=fixture.root/'data/import-v1';N.import_output(fixture.handoff,fixture.output,imported,root=fixture.root)
        before=V.M._records(V.M._tree(imported));output=fixture.root/'data/no-oversized-page'
        with patch.object(V,'MAX_SCENE_BYTES',1):
            with self.assertRaisesRegex(ValueError,'unchanged browser budget'):V.build(imported,output,root=fixture.root)
        self.assertFalse(output.exists());self.assertEqual(before,V.M._records(V.M._tree(imported)))


if __name__=='__main__':unittest.main()
