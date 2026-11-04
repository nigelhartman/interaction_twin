"""Automatic correction contracts; no inference or manual source-object poses."""
import copy,json,shutil,tempfile,unittest
from pathlib import Path
from unittest import mock
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import native_pose_refinement as P
import sam3d_mechanisms as M
import sam3d_common_scene as S
from native_appearance import load_native_glb,render_native_mask
from tests.test_native_appearance import fixture


class PoseCorrectionMathTests(unittest.TestCase):
    def test_occlusion_discount_preserves_raw_and_does_not_invent_hand_masks(self):
        source=np.zeros((8,8),bool);source[1:3,1:3]=True
        render=source.copy();render[5,5:7]=True;other=np.zeros_like(source);other[5,5]=True
        row=P.occlusion_metrics(source,render,other)
        self.assertAlmostEqual(row['raw_mask_iou'],4/6);self.assertAlmostEqual(row['occluder_accounted_mask_iou'],4/5)
        self.assertEqual(row['known_other_mask_discounted_pixels'],1);self.assertFalse(row['unmodeled_hand_occlusion_discounted'])

    def rows(self,raw=.5,accounted=.5):
        return [{'source_index':i,'raw_mean_iou':raw,'accounted_mean_iou':accounted,
                 'components':{'body':{'raw_mask_iou':raw},'lid':{'raw_mask_iou':raw}}}for i in [2,4,5]]

    def test_fixed_development_gate_rejects_hidden_component_regression_and_no_gain(self):
        baseline=self.rows();self.assertFalse(P.correction_gate(baseline,self.rows())['accepted'])
        self.assertTrue(P.correction_gate(baseline,self.rows(.54,.56))['accepted'])
        changed=self.rows(.54,.56);changed[0]['components']['lid']['raw_mask_iou']=.40
        self.assertFalse(P.correction_gate(baseline,changed)['accepted'])
        changed=self.rows();changed[0]['source_index']=3
        with self.assertRaisesRegex(ValueError,'selection changed'):P.correction_gate(baseline,changed)

    def test_constant_native_delta_rebuilds_unordered_descendants_and_preserves_unknowns(self):
        def rows(matrix):return [{'source_sample_index':i,'status':'estimated'if i!=3 else'unknown',
                                 'camera_from_native_mesh':matrix.tolist()if i!=3 else None}for i in [1,2,3]]
        parent=np.eye(4);parent[:3,:3]*=.4;parent[2,3]=3
        a,b=M.joint_frames(.4,.2,[.1,0,0],[0,0,.1],np.eye(3));child=M.forward(parent,a,[0,1,0],.3,b)
        c,d=M.joint_frames(.2,.1,[0,.1,0],[0,0,0],np.eye(3));grandchild=M.forward(child,c,[1,0,0],.2,d)
        def joint(identifier,pid,cid,pa,ch,q,axis):return {'id':identifier,'parent_id':pid,'child_id':cid,'axis':axis,'parent_from_joint':pa.tolist(),'joint_from_child_zero':ch.tolist(),
            'samples':[{'source_sample_index':i,'status':'estimated'if i!=3 else'unknown','q_rad':q if i!=3 else None}for i in [1,2,3]]}
        scene={'components':[{'id':i,'poses':rows(m)}for i,m in [('body',parent),('lid',child),('tip',grandchild)]],
               'mechanics':{'joints':[joint('nested','lid','tip',c,d,.2,[1,0,0]),joint('main','body','lid',a,b,.3,[0,1,0])]}}
        original=copy.deepcopy(scene);delta=np.eye(4);delta[:3,:3]=Rotation.from_euler('x',12,degrees=True).as_matrix();delta[0,3]=.03
        result=P.apply_correction(scene,'body',delta);parts={c['id']:c for c in result['components']}
        np.testing.assert_allclose(parts['body']['poses'][0]['camera_from_native_mesh'],parent@delta)
        expected=M.forward(parent@delta,a,[0,1,0],.3,b);np.testing.assert_allclose(parts['lid']['poses'][0]['camera_from_native_mesh'],expected)
        np.testing.assert_allclose(parts['tip']['poses'][0]['camera_from_native_mesh'],M.forward(expected,c,[1,0,0],.2,d))
        self.assertEqual(scene,original);self.assertEqual([c['poses'][2]['status']for c in result['components']],['unknown']*3)
        self.assertEqual(result['mechanics']['joints'],scene['mechanics']['joints'])
        delta[:3,:3]*=2
        with self.assertRaisesRegex(ValueError,'native scale'):P.apply_correction(scene,'body',delta)


class PoseCorrectionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve();folder=self.root/'data/fixture';folder.mkdir(parents=True)
        for name in P.CODE:shutil.copyfile(P.ROOT/name,self.root/name)
        asset=folder/'asset.glb';asset.write_bytes(fixture([{'vertices':[[-.4,-.4,0],[.4,-.4,0],[-.4,.4,0]],'material':0}],materials=[{'doubleSided':True}],images=[]))
        mesh=load_native_glb(asset);k=np.array([[100.,0,64],[0,100,64],[0,0,1]])
        parent=np.eye(4);parent[2,3]=4.;a,b=M.joint_frames(1.,1.,[.2,.1,0],[0,0,0],np.eye(3));child=M.forward(parent,a,[0,1,0],0.,b)
        poses=lambda m:[{'source_sample_index':i,'status':'unknown'if i==3 else'estimated','camera_from_native_mesh':None if i==3 else m.tolist()}for i in range(1,6)]
        evidence=[]
        for i in range(1,6):
            observations=[]
            for identifier,m in [('body',parent),('lid',child)]:
                path=folder/f'{i}-{identifier}.png';mask=render_native_mask(mesh,k,m[:3,:3],m[:3,3],128,128)['mask'];Image.fromarray(mask.astype(np.uint8)*255).save(path)
                observations.append({'track_id':identifier,'gate_status':'candidate_geometric_gate_only','audit_status':'not_model_audited','mask_path':S.ref(path,self.root)['path'],'mask_sha256':S.U.sha(path)})
            evidence.append({'source_index':i,'timestamp_s':(i-1)*.3,'observations':observations})
        program=folder/'interaction.json';S.write(program,{'evidence':evidence})
        scene=folder/'scene.json';S.write(scene,{'original_program':S.ref(program,self.root),'components':[{'id':i,'asset':S.ref(asset,self.root),'poses':poses(m)}for i,m in [('body',parent),('lid',child)]],
            'source':{'height':128,'width':128},'camera':{'K':k.tolist()},'evaluation_protocol':{'held_out_sample_indices':[3]},'mechanics':{'joints':[{'id':'hinge','parent_id':'body','child_id':'lid','axis':[0,1,0],
            'parent_from_joint':a.tolist(),'joint_from_child_zero':b.tolist(),'samples':[{'source_sample_index':i,'status':'unknown'if i==3 else'estimated','q_rad':None if i==3 else 0.}for i in range(1,6)]}]}})
        self.scene_ref=S.ref(scene,self.root);audit=folder/'audit.json';S.write(audit,{'scene_ref':self.scene_ref,'interaction_ref':S.ref(program,self.root),
             'samples':[{'source_index':i,'role':'final_test'if i==3 else'development'}for i in range(1,6)]})
        self.audit_ref=S.ref(audit,self.root);self.audit=S.U.read(audit)

    def test_actual_triangle_search_preserves_baseline_and_never_opens_final_targets(self):
        original=Image.open;opened=[]
        def guarded(path,*args,**kwargs):
            self.assertFalse(Path(path).name.startswith('3-'),'Final-test target loaded by correction search');opened.append(Path(path).name);return original(path,*args,**kwargs)
        with mock.patch('native_scene_audit.validate_manifest',return_value=self.audit),mock.patch.object(Image,'open',side_effect=guarded):
            result=P.propose(self.scene_ref,self.audit_ref,'data/correction',parent_id='body',training_samples=[1],development_samples=[2,4,5],root=self.root,max_dimension=128)
        self.assertEqual(result['decision'],'baseline_retained');self.assertFalse((self.root/'data/correction/scene.json').exists())
        plan=S.U.read(self.root/'data/correction/selection-plan.json');self.assertFalse(plan['final_test_target_pixels_used']);self.assertEqual(len(opened),8)
        self.assertTrue(all('/3-'not in row['path']for row in plan['inputs']))

    def test_final_test_and_overlapping_selection_rejected_before_target_pixels(self):
        with mock.patch('native_scene_audit.validate_manifest',return_value=self.audit),mock.patch.object(Image,'open',side_effect=AssertionError('Target opened')):
            with self.assertRaisesRegex(ValueError,'Final-test'):P.propose(self.scene_ref,self.audit_ref,'data/rejected',parent_id='body',training_samples=[3],development_samples=[2,4,5],root=self.root,max_dimension=128)
            with self.assertRaisesRegex(ValueError,'disjoint'):P.propose(self.scene_ref,self.audit_ref,'data/rejected',parent_id='body',training_samples=[1],development_samples=[1,4,5],root=self.root,max_dimension=128)
        self.assertFalse((self.root/'data/rejected').exists())

if __name__=='__main__':unittest.main()
