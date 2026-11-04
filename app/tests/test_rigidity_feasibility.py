"""Conservative source-declaration withholding; software-only, no inference."""
import copy,json,shutil,unittest
from unittest import mock
import numpy as np
import rigidity_feasibility as R
import object_program as O
import sam3d_common_scene as S


class DeclarationTests(unittest.TestCase):
    def proposal(self,text,**extra):return {'joints':[{'id':'link','parent_id':'holder','child_id':'moving','joint_type':'revolute','attachment':'persistent','limitations':[text],**extra}]}

    def test_only_positive_complete_declarations_trigger(self):
        for text in ['not rigid','Non-rigid.','deformable','The child is not rigid','the moving component remains deformable','deformable leather, shape changes with folding, not rigid']:
            with self.subTest(text=text):
                rows=R.declarations(self.proposal(text));self.assertEqual(len(rows),1);self.assertEqual(rows[0]['declarations'][0]['source_limitation'],text)
        for text in ['not deformable','no non-rigid motion observed','may be deformable','deformable object nearby','rigid holder near deformable cable','not rigidly fixed to parent','rigidity unknown','']:
            with self.subTest(text=text):self.assertEqual(R.declarations(self.proposal(text)),[])
        self.assertEqual(R.declarations(self.proposal('not rigid',attachment='removable',joint_type='removable')),[])
        p=self.proposal('rigidity unknown');p['joints'][0]['child_id']='deformable_leather';self.assertEqual(R.declarations(p),[])

    def scene(self):
        def part(identifier):
            matrix=np.eye(4);matrix[2,3]=2.
            return {'id':identifier,'asset':{'path':f'data/{identifier}.glb','sha256':'a'*64,'bytes':10},'native_uniform_scale':1.,'poses':[
                {'source_sample_index':i,'timestamp_s':i-1.,'status':'estimated'if i!=3 else'unknown','camera_from_native_mesh':matrix.tolist()if i!=3 else None,'accepted_motion_evidence':False}for i in [1,2,3]]}
        def proposal(identifier,parent,child,attachment='persistent',text='rigidity unknown'):
            return {'id':identifier,'parent_id':parent,'child_id':child,'joint_type':'revolute'if attachment=='persistent'else'removable','attachment':attachment,'limitations':[text]}
        proposals={'joints':[proposal('flap','holder','moving',text='deformable leather, shape changes with folding, not rigid'),proposal('nested','moving','tip'),proposal('contents','holder','coin','removable')]}
        scene={'components':[part(i)for i in ['holder','moving','tip','coin']],'mechanics':{'joints':[{'id':'flap'},{'id':'nested'}],'unresolved_children':[],'structural_connectivity_verified':True},'limitations':[]}
        return scene,proposals

    def test_unknown_descendants_with_original_parent_removable_and_assets_preserved(self):
        scene,proposal=self.scene();before=copy.deepcopy(scene);original_proposal=copy.deepcopy(proposal);result,assessment=R.apply(scene,proposal)
        self.assertEqual(scene,before);self.assertEqual(proposal,original_proposal);self.assertEqual(assessment['withheld_joint_ids'],['flap','nested'])
        parts={p['id']:p for p in result['components']}
        for identifier in ['holder','coin']:self.assertEqual(parts[identifier],next(p for p in scene['components']if p['id']==identifier))
        for identifier in ['moving','tip']:
            self.assertTrue(all(r['status']=='unknown'and r['camera_from_native_mesh']is None for r in parts[identifier]['poses']))
            self.assertEqual(parts[identifier]['asset'],next(p for p in scene['components']if p['id']==identifier)['asset'])
        self.assertEqual(result['mechanics']['joints'],[]);self.assertFalse(result['mechanics']['structural_connectivity_verified'])
        self.assertEqual(result['mechanics']['unresolved_children'][1]['reason'],'rigid_motion_withheld_due_to_unsupported_ancestor')

    def test_no_claim_is_not_a_rigidity_certificate_and_original_scene_is_equal(self):
        scene,proposal=self.scene();proposal['joints'][0]['limitations']=['unknown'];result,assessment=R.apply(scene,proposal)
        self.assertEqual(result,scene);self.assertFalse(assessment['rigidity_established']);self.assertEqual(assessment['findings'],[])

    def test_existing_unresolved_child_is_annotated_once_without_free_pose(self):
        scene,proposal=self.scene();scene['mechanics']['joints']=[{'id':'nested'}];scene['mechanics']['unresolved_children']=[{'id':'flap'}]
        result,_=R.apply(scene,proposal);self.assertEqual([r['id']for r in result['mechanics']['unresolved_children']],['flap','nested'])


class GuardArtifactTests(unittest.TestCase):
    def fixture(self,*,contradiction=True):
        from tests.test_object_program import ObjectProgramTests
        f=ObjectProgramTests();f.setUp();self.addCleanup(f.doCleanups);root=f.root.resolve()
        for name in O.CODE:shutil.copyfile(O.ROOT/name,root/name)
        program=O.read_json(root/f.doc['evidence_program']['path']);program['evidence']=[{'id':f'source_{i}','source_index':i,'timestamp_s':i-1.}for i in [1,2,3]]
        pref=f.file(f.doc['evidence_program']['path'],program);source=O.read_json(root/f.doc['source_scene']['path']);source['original_program']=pref;source['dependencies']=[S.source_ref(root/name,root)for name in S.DEPENDENCIES]
        sref=f.file(f.doc['source_scene']['path'],source);proposal=O.read_json(root/f.doc['proposal']['path']);proposal['interaction_sha256']=pref['sha256'];proposal['joints'][0]['limitations']=['deformable leather, shape changes with folding, not rigid']if contradiction else['rigidity unknown']
        plan=f.file('data/topology/plan.json',{'contract':{'code_sha256':O.sha(root/'mechanism_reasoning.py')}});proposal['reasoning']={'plan_path':plan['path']};proposal_ref=f.file(f.doc['proposal']['path'],proposal)
        f.mechanics.update(source_scene=sref,proposal=proposal_ref,implementation=S.source_ref(root/'sam3d_mechanisms.py',root));f.scene['original_program']=pref;f.update_mechanics();return f

    def test_source_bound_build_then_actual_descriptive_export_preserve_originals(self):
        f=self.fixture();root=f.root;before={r['path']:(root/r['path']).read_bytes()for r in [f.doc['evidence_program'],f.doc['constrained_scene'],f.doc['mechanics'],f.doc['proposal']]}
        result=R.build(f.doc['constrained_scene']['path'],'data/rigidity-feasibility/unit',root=root);self.assertEqual(result['status'],'unsupported_rigid_child_motion_withheld')
        guarded=O.read_json(root/result['scene_ref']['path']);self.assertEqual(guarded['components'][0],f.scene['components'][0]);self.assertEqual(guarded['mechanics']['joints'],[])
        self.assertEqual(guarded['mechanics']['rigidity_feasibility']['findings'][0]['declarations'][0]['source_limitation'],'deformable leather, shape changes with folding, not rigid')
        exported=O.build(result['scene_ref']['path'],'data/object-programs/guarded-unit/object-program.json',root=root);doc=O.read_json(root/exported['path']);self.assertTrue(O.validate(doc,root=root))
        self.assertEqual(doc['joints'],[]);self.assertEqual(doc['unresolved_connections'][0]['status'],'unknown');self.assertEqual(len(doc['component_assets']),2)
        archived={r['original_path']:r for r in doc['implementation_provenance']};self.assertIn('rigidity_feasibility.py',archived);self.assertIn('sam3d_mechanisms.py',archived)
        for path,payload in before.items():self.assertEqual((root/path).read_bytes(),payload)

    def test_no_declaration_returns_base_identity_and_no_rewritten_scene(self):
        f=self.fixture(contradiction=False);result=R.build(f.doc['constrained_scene']['path'],'data/rigidity-feasibility/no-claim',root=f.root)
        self.assertEqual(result['scene_ref'],f.doc['constrained_scene']);self.assertFalse((f.root/'data/rigidity-feasibility/no-claim/scene.json').exists())

    def test_changed_proposal_source_binding_or_reused_output_rejected(self):
        f=self.fixture();R.build(f.doc['constrained_scene']['path'],'data/rigidity-feasibility/unit',root=f.root)
        with self.assertRaisesRegex(ValueError,'Fresh ordinary'):R.build(f.doc['constrained_scene']['path'],'data/rigidity-feasibility/unit',root=f.root)
        (f.root/f.doc['proposal']['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'bytes changed'):R.build(f.doc['constrained_scene']['path'],'data/rigidity-feasibility/changed',root=f.root)


if __name__=='__main__':unittest.main()
