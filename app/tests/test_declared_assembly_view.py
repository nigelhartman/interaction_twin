"""A read-only selected derivative needs no CV runtime or geometry projection."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

import declared_assembly_view as V

ROOT=Path(__file__).resolve().parents[1]


class DeclaredAssemblyViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=ROOT/'data/assembly-refinements/jabra-source-refit-20261103-v2/derived-assembly.json'
        if not path.is_file():raise unittest.SkipTest('Actual approved V2 derivative not present')
        cls.document=json.loads(path.read_text())
        cls.allowlist={x['path']:x for x in json.loads((ROOT/'showcase/public/public-assets.json').read_text())['assets']}

    def setUp(self):
        self.doc=copy.deepcopy(self.document);self.overrides={};self.calls=[]

    def read(self,ref,*,maximum):
        V.reference(ref,maximum)
        if ref['path'] in self.overrides:
            pointer,raw=self.overrides[ref['path']]
            self.assertEqual(pointer,ref)
        else:
            pointer=self.allowlist[ref['path']]
            self.assertEqual(pointer,ref)
            raw=(ROOT/ref['path']).read_bytes()
        self.assertLessEqual(len(raw),maximum);self.assertEqual(len(raw),ref['bytes'])
        self.assertEqual(hashlib.sha256(raw).hexdigest(),ref['sha256'])
        self.calls.append(ref['path']);return raw

    def put(self,label,document):
        raw=(json.dumps(document,indent=2)+'\n').encode()
        ref={'path':'data/explicit-test-fixture/'+label+'.json','sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
        self.overrides[ref['path']]=(ref,raw);return ref

    def audit_edit(self,edit):
        audit=json.loads(self.read(self.doc['source_alignment_audit_ref'],maximum=V.MAX_JSON))
        edit(audit);ref=self.put('audit',audit)
        scene=json.loads(self.read(self.doc['scene_ref'],maximum=V.MAX_JSON))
        scene['assembly_refinement']['source_audit']=ref
        self.doc['source_alignment_audit_ref']=ref;self.doc['scene_ref']=self.put('scene',scene)

    def scene_edit(self,edit):
        scene=json.loads(self.read(self.doc['scene_ref'],maximum=V.MAX_JSON))
        edit(scene);self.doc['scene_ref']=self.put('scene',scene)

    def test_actual_selected_view_checks_seven_jsons_no_images_or_meshes(self):
        self.assertIs(V.validate(self.doc,self.read),self.doc)
        self.assertEqual(len(self.calls),7)
        self.assertTrue(all(name.endswith('.json')for name in self.calls))
        self.assertEqual(self.doc['source_audit_summary']['retained_samples'],49)
        self.assertEqual(self.doc['source_audit_summary']['total_unknown_samples'],52)

    def test_module_loads_with_computer_vision_and_model_imports_forbidden(self):
        script=r'''import builtins,importlib.util,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name.split('.')[0] in {'cv2','numpy','scipy','PIL','native_appearance','refined_assembly_contract'}:raise RuntimeError('CV dependency imported')
 return original(name,*args,**kwargs)
builtins.__import__=guarded
spec=importlib.util.spec_from_file_location('pure_view',sys.argv[1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert callable(module.validate)
'''
        subprocess.run([sys.executable,'-I','-c',script,str(ROOT/'declared_assembly_view.py')],check=True,capture_output=True,text=True,timeout=10)

    def test_v1_silhouette_only_derivative_cannot_be_selected(self):
        old=json.loads((ROOT/'data/assembly-refinements/jabra-source-refit-20261103-v1/derived-assembly.json').read_text())
        for field in ['scene_ref','source_alignment_audit_ref']:
            pointer=old[field];self.overrides[pointer['path']]=(pointer,(ROOT/pointer['path']).read_bytes())
        with self.assertRaisesRegex(ValueError,'facing/unknown policy'):V.validate(old,self.read)

    def test_cross_source_and_physical_or_native_relabelling_are_rejected(self):
        for field in ['physical_accuracy_verified','simulation_ready']:
            changed=copy.deepcopy(self.doc);changed[field]=True
            with self.assertRaises(ValueError):V.validate(changed,self.read)
        changed=copy.deepcopy(self.doc);changed['source']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'source'):V.validate(changed,self.read)
        changed=copy.deepcopy(self.doc);changed['component_assets'][1]['geometry_origin']='unchanged_SAM_3D_inference'
        with self.assertRaisesRegex(ValueError,'relabelled'):V.validate(changed,self.read)

    def test_callback_cannot_substitute_changed_json_even_if_length_is_same(self):
        def bad(ref,*,maximum):
            raw=self.read(ref,maximum=maximum)
            if ref==self.doc['base_object_program_ref']:return b' '+raw[1:]
            return raw
        with self.assertRaisesRegex(ValueError,'hash'):V.validate(self.doc,bad)

    def test_facing_rejected_sample_cannot_be_restored_by_a_perfect_iou(self):
        def edit(audit):
            row=next(r for r in audit['rows']if r['source_sample_index']==23)
            self.assertTrue(row['facing_ambiguity_rejected'])
            row['mask_iou']={'parent':1.,'child':1.,'union':1.};row['retained']=True
        self.audit_edit(edit)
        with self.assertRaisesRegex(ValueError,'visibility gate'):V.validate(self.doc,self.read)

    def test_facing_sign_or_open_bracket_cannot_be_relabelled(self):
        self.audit_edit(lambda a:a['rows'][0]['cavity_facing_cosine'].update(child=-1.))
        with self.assertRaisesRegex(ValueError,'ambiguity check'):V.validate(self.doc,self.read)
        self.setUp()
        self.audit_edit(lambda a:a['rows'][0].update(source_state_bracket={'state':'closed','source_sample_indices':[1,101]}))
        with self.assertRaisesRegex(ValueError,'ambiguity check'):V.validate(self.doc,self.read)

    def test_original_unknown_or_fixed_joint_cannot_change_in_a_new_scene_reference(self):
        def fill(scene):
            row=next(r for r in scene['components'][1]['poses']if r['camera_from_native_mesh']is None)
            row['camera_from_native_mesh']=[[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1]];row['status']='estimated'
        self.scene_edit(fill)
        with self.assertRaisesRegex(ValueError,'original unknown'):V.validate(self.doc,self.read)
        self.setUp();self.scene_edit(lambda s:s['mechanics']['joints'][0].update(axis=[0,1,0]))
        with self.assertRaisesRegex(ValueError,'geometric joint changed'):V.validate(self.doc,self.read)

    def test_private_reference_and_mismatched_source_audit_are_not_read(self):
        self.doc['base_object_program_ref']['path']='.secrets/meta.env'
        with self.assertRaisesRegex(ValueError,'ordinary declared data'):V.validate(self.doc,self.read)
        self.assertEqual(self.calls,[])
        self.setUp();self.doc['source_alignment_audit_ref']['sha256']='0'*64
        with self.assertRaises((ValueError,AssertionError)):V.validate(self.doc,self.read)


if __name__=='__main__':unittest.main()
