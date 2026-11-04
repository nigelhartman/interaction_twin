"""Analytic CPU audit fixtures; no inference or measured physical claims."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
from PIL import Image

import native_scene_audit as A
import native_audit_protocol as E
import sam3d_common_scene as S
from native_appearance import load_native_glb,render_native_mask
from tests.test_native_appearance import fixture


class NativeAuditMathTests(unittest.TestCase):
    def scene(self):
        def pose(i,status='estimated'):
            m=np.eye(4);m[0,3]=i*.1
            return {'source_sample_index':i,'timestamp_s':(i-1)*.3,'status':status,'camera_from_native_mesh':m.tolist()if status=='estimated'else None}
        return {'components':[{'id':'body','poses':[pose(1),pose(2,'unknown'),pose(3)]}],
            'timing':{'max_interpolation_gap_s':.8},'evaluation_protocol':{'held_out_sample_indices':[2]}}

    def test_holdout_prediction_never_reads_or_uses_the_target_pose_or_mask(self):
        scene=self.scene();before=copy.deepcopy(scene)
        with mock.patch.object(Image,'open',side_effect=AssertionError('Target pixels reached prediction')):
            _,predicted=A.predict_poses(scene,2)
        self.assertEqual(predicted['body']['neighbors'],[1,3]);self.assertAlmostEqual(predicted['body']['matrix'][0,3],.2)
        scene['components'][0]['poses'][1].update(status='estimated',camera_from_native_mesh=(np.eye(4)*99).tolist())
        _,poisoned=A.predict_poses(scene,2)
        np.testing.assert_allclose(poisoned['body']['matrix'],predicted['body']['matrix']);self.assertEqual(before['components'][0]['poses'][1]['status'],'unknown')

    def test_no_prediction_across_rejected_neighbors_or_extrapolation(self):
        scene=self.scene();scene['components'][0]['poses'][0].update(status='unknown',camera_from_native_mesh=None)
        self.assertIsNone(A.predict_poses(scene,2)[1]['body']['matrix'])
        scene=self.scene();scene['timing']['max_interpolation_gap_s']=.4
        self.assertIsNone(A.predict_poses(scene,2)[1]['body']['matrix'])

    def test_all_native_triangles_preserve_a_hole_that_convex_hull_validation_would_hide(self):
        vertices=[[1,1,1],[9,1,1],[9,9,1],[1,9,1],[3,3,1],[7,3,1],[7,7,1],[3,7,1]]
        faces=[[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
        raw=fixture([{'vertices':vertices,'indices':np.asarray(faces).reshape(-1,1).tolist(),'material':0}],materials=[{'doubleSided':True}],images=[])
        mesh=load_native_glb(raw);render=render_native_mask(mesh,np.eye(3),np.eye(3),np.zeros(3),11,11)
        self.assertFalse(render['mask'][5,5]);self.assertTrue(render['mask'][2,5]);self.assertEqual(len(mesh.faces),8)
        actual=A.mask_metrics(render['mask'],render['mask']);self.assertEqual(actual['mask_iou'],1.)
        hull=np.zeros((11,11),np.uint8);cv2.fillConvexPoly(hull,np.asarray([[1,1],[9,1],[9,9],[1,9]]),1)
        self.assertLess(A.mask_metrics(render['mask'],hull>0)['mask_iou'],1.)

    def test_occlusion_counts_are_explicit_and_do_not_erase_unexplained_pixels(self):
        observed=np.zeros((8,8),bool);observed[2:4,2:4]=True
        predicted=observed.copy();predicted[4:6,4:6]=True;other=np.zeros_like(observed);other[4,4]=True
        metric=A.mask_metrics(observed,predicted,other_observed=other)
        self.assertEqual(metric['known_other_source_overlap_pixels'],1);self.assertEqual(metric['unexplained_render_pixels'],3)
        self.assertEqual(metric['mask_iou'],.5);self.assertTrue(metric['unmodeled_hand_occlusion_possible'])

    def test_predeclared_split_excludes_anchors_and_endpoints(self):
        held=E.choose_holdouts(list(range(1,22)),protected_sample_indices=[10],stride=5)
        self.assertEqual(held,[5,15,20]);protocol=E.make_protocol('a'*64,list(range(1,22)),held,protected_sample_indices=[10])
        self.assertFalse(protocol['held_out_masks_used_for_pose_or_joint_fit']);self.assertFalse(set(protocol['fit_sample_indices'])&set(held))
        for invalid in [[1],[21],[10],[5,5]]:
            with self.assertRaises(ValueError):E.make_protocol('a'*64,list(range(1,22)),invalid,protected_sample_indices=[10])


class ExactSegmentationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        self.case=self.root/'data/case';self.program=self.case/'program/interaction.json';self.program.parent.mkdir(parents=True)
        def save(relative,value):
            path=self.case/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path
        self.save=save;manifest=save('manifest.json',{'source':{'sha256':'a'*64}});index=save('frames/index.json',{'time_domain':S.U.TIME_DOMAIN})
        tracks=save('segmentation/flow-components-2/tracks.json',{'tracks':[]});gates=save('segmentation/flow-components-2/frames.json',{'frames':[]});save('segmentation/flow-components-2/quality.json',{})
        self.doc={'source':{'sha256':'a'*64,'time_domain':S.U.TIME_DOMAIN},'inputs':[{'id':i,'path':p.relative_to(self.root).as_posix(),'sha256':S.U.sha(p)}for i,p in [('source_manifest',manifest),('source_frame_index',index),('mask_tracks',tracks),('mask_quality',gates)]]}
        self.write()
    def write(self):self.program.write_text(json.dumps(self.doc))
    def test_exact_later_attempt_resolves_without_any_first_attempt(self):
        result=S.segmentation_for_program(self.program,root=self.root)
        self.assertEqual(result['directory'],self.case/'segmentation/flow-components-2')
    def test_program_location_and_unrelated_ancestor_cannot_override_exact_compiled_source(self):
        moved=self.root/'data/owner-candidates/camera/procedure/interaction.json';moved.parent.mkdir(parents=True)
        moved.write_bytes(self.program.read_bytes())
        (moved.parent.parent/'manifest.json').write_text(json.dumps({'source':{'sha256':'b'*64}}))
        result=S.segmentation_for_program(moved,root=self.root)
        self.assertEqual(result['case'],self.case);self.assertEqual(result['manifest'],self.case/'manifest.json')
        self.assertEqual(result['directory'],self.case/'segmentation/flow-components-2')
    def test_compiled_manifest_requires_the_canonical_source_case_filename(self):
        alternate=self.save('custom-manifest.json',{'source':{'sha256':'a'*64}})
        self.doc['inputs'][0].update(path=alternate.relative_to(self.root).as_posix(),sha256=S.U.sha(alternate));self.write()
        with self.assertRaisesRegex(ValueError,'canonical case filename'):S.segmentation_for_program(self.program,root=self.root)
    def test_cross_attempt_gates_duplicate_references_and_wrong_source_are_rejected(self):
        original=copy.deepcopy(self.doc)
        foreign=self.save('segmentation/flow-components-1/frames.json',{})
        self.doc['inputs'][-1].update(path=foreign.relative_to(self.root).as_posix(),sha256=S.U.sha(foreign));self.write()
        with self.assertRaisesRegex(ValueError,'different case or SAM attempt'):S.segmentation_for_program(self.program,root=self.root)
        self.doc=copy.deepcopy(original);self.doc['inputs'].append(self.doc['inputs'][-1]);self.write()
        with self.assertRaisesRegex(ValueError,'One exact compiled'):S.segmentation_for_program(self.program,root=self.root)
        self.doc=copy.deepcopy(original);self.doc['source']['sha256']='b'*64;self.write()
        with self.assertRaisesRegex(ValueError,'source recording'):S.segmentation_for_program(self.program,root=self.root)
    def test_changed_track_bytes_never_read_a_convenient_old_directory(self):
        self.save('segmentation/flow-components-1/tracks.json',{'tracks':['valid-old']})
        self.save('segmentation/flow-components-2/tracks.json',{'tracks':['changed']})
        with self.assertRaisesRegex(ValueError,'input changed'):S.segmentation_for_program(self.program,root=self.root)

class AuditArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        for name in A.CODE:shutil.copyfile(A.ROOT/name,self.root/name)
        folder=self.root/'data/fixture';folder.mkdir(parents=True);self.folder=folder
        self.glb=folder/'asset.glb';self.glb.write_bytes(fixture([{'vertices':[[1,1,1],[10,1,1],[1,10,1]],'material':0}],materials=[{'doubleSided':True}],images=[]))
        mask=render_native_mask(load_native_glb(self.glb),np.eye(3),np.eye(3),np.zeros(3),12,12)['mask']
        evidence=[];poses=[]
        for i in range(1,4):
            image=folder/f'{i}.png';Image.fromarray(np.full((12,12,3),180,np.uint8)).save(image)
            target=folder/f'{i}-mask.png';Image.fromarray(mask.astype(np.uint8)*255).save(target)
            evidence.append({'id':f'source_{i:06d}','source_index':i,'timestamp_s':(i-1)*.3,'frame_path':S.ref(image,self.root)['path'],'frame_sha256':S.U.sha(image),
                'observations':[{'track_id':'body','mask_path':S.ref(target,self.root)['path'],'mask_sha256':S.U.sha(target),'gate_status':'candidate_geometric_gate_only','audit_status':'plausible'}]})
            poses.append({'source_sample_index':i,'timestamp_s':(i-1)*.3,'status':'unknown'if i==2 else'estimated','camera_from_native_mesh':None if i==2 else np.eye(4).tolist()})
        self.program=folder/'interaction.json';self.program.write_text(json.dumps({'source':{'sha256':'a'*64,'time_domain':S.U.TIME_DOMAIN},'tracklets':[{'id':'body'}],'assets':[{'track_id':'body','sha256':S.U.sha(self.glb)}],'evidence':evidence}))
        self.scene=folder/'scene.json';self.scene.write_text(json.dumps({'format':S.FORMAT,'source':{'sha256':'a'*64,'time_domain':S.U.TIME_DOMAIN,'height':12,'width':12},'original_program':S.ref(self.program,self.root),
            'components':[{'id':'body','asset':S.ref(self.glb,self.root),'poses':poses}],'camera':{'K':np.eye(3).tolist()},'timing':{'max_interpolation_gap_s':.8},'accepted_motion_poses':0,'physical_accuracy_verified':False,
            'evaluation_protocol':E.make_protocol('a'*64,[1,2,3],[2],protected_sample_indices=[1])}))
        self.output=self.root/'data/audit'
    def build(self):return A.build_audit(self.scene,self.output,root=self.root,max_frames=4,max_dimension=128)
    def test_end_to_end_audit_freezes_real_triangle_rasters_code_and_original_bytes(self):
        before=self.glb.read_bytes();result=self.build();doc=A.validate_manifest(self.output/'audit.json',root=self.root)
        self.assertEqual(doc['evaluation_protocol']['mode'],'predeclared_mask_holdout');self.assertTrue(doc['component_inventory']['consistent'])
        self.assertEqual(len(doc['implementation_snapshots']),len(A.CODE));self.assertEqual(self.glb.read_bytes(),before)
        self.assertEqual(doc['development_summary']['component_samples'],2);self.assertEqual(len([s for s in doc['samples']if s['role']=='final_test']),1)
        self.assertFalse(doc['summary']['physical_accuracy_verified']);self.assertFalse(result['reused_verified_cache'])
        with mock.patch.object(A,'render_native_mesh',side_effect=AssertionError('Verified reuse must not rerender')):
            self.assertTrue(self.build()['reused_verified_cache'])
    def test_changed_target_pixels_or_panels_are_not_silently_reused(self):
        self.build();manifest=self.output/'audit.json';saved=manifest.read_bytes()
        panel=next(self.output.glob('*comparison.png'));panel.write_bytes(b'changed output')
        with self.assertRaisesRegex(ValueError,'input bytes changed'):self.build()
        self.assertEqual(manifest.read_bytes(),saved)
    def test_component_id_cannot_become_a_traversing_output_filename(self):
        scene=json.loads(self.scene.read_text());scene['components'][0]['id']='body/../../escape';self.scene.write_text(json.dumps(scene))
        with self.assertRaisesRegex(ValueError,'component IDs'):self.build()
        self.assertFalse(self.output.exists())
    def test_audit_rejects_fitted_holdout_pose_instead_of_claiming_validation(self):
        scene=json.loads(self.scene.read_text());scene['components'][0]['poses'][1].update(status='estimated',camera_from_native_mesh=np.eye(4).tolist());self.scene.write_text(json.dumps(scene))
        with self.assertRaisesRegex(ValueError,'declared heldout pose'):self.build()

class LedgerFitterBindingTests(unittest.TestCase):
    def fixture(self):
        from tests.test_sam3d_common_scene import SceneCacheTests
        case=SceneCacheTests();case.setUp();self.addCleanup(case.doCleanups)
        ledger=case.case/'part-ledger.json';ledger.write_text('{"analytic_ledger_fixture":true}')
        case.scene['part_ledger']=S.ref(ledger,case.root);case.scene['inputs'].append(S.ref(ledger,case.root))
        case.scene['input_signature']=S.U.digest(S.U.encoded({'inputs':case.scene['inputs'],'dependencies':case.scene['dependencies']}))
        S.U.write(case.out/'scene.json',case.scene);S.U.write(case.out/'build.json',{'format':'sam3d-common-scene-build','manifest':S.ref(case.out/'scene.json',case.root)})
        return case,ledger
    def test_ledger_path_resolves_authoritative_metadata_without_review_discovery(self):
        import part_ledger
        case,ledger=self.fixture()
        with mock.patch.object(part_ledger,'metadata_paths',return_value=[case.model])as lookup,mock.patch.object(S,'metadata_for_program',side_effect=AssertionError('Ledger mode must not discover convenient reviews')):
            result=S.build(case.program,case.out,root=case.root,part_ledger_path=ledger)
        self.assertEqual(result['status'],'reused_verified_estimated_scene');lookup.assert_called_once_with(ledger,program_path=case.program,root=case.root)
    def test_explicit_geometry_cannot_disagree_with_authoritative_ledger(self):
        import part_ledger
        case,ledger=self.fixture()
        with mock.patch.object(part_ledger,'metadata_paths',return_value=[case.model]):
            with self.assertRaisesRegex(ValueError,'authoritative part ledger'):
                S.build(case.program,case.out,root=case.root,part_ledger_path=ledger,metadata_paths=[case.mask])


class FreshObjectProgramExportTests(unittest.TestCase):
    def test_fresh_export_archives_ledger_and_holdout_helpers_without_changing_legacy_documents(self):
        import object_program as O
        from tests.test_object_program import ObjectProgramTests
        fixture=ObjectProgramTests();fixture.setUp();self.addCleanup(fixture.doCleanups);root=fixture.root.resolve()
        for name in O.CODE:shutil.copyfile(O.ROOT/name,root/name)
        before=(root/fixture.doc['evidence_program']['path']).read_bytes()
        source=O.read_json(root/fixture.doc['source_scene']['path']);source['dependencies']=[S.source_ref(root/name,root)for name in S.DEPENDENCIES]
        source_ref=fixture.file(fixture.doc['source_scene']['path'],source)
        fixture.mechanics['source_scene']=source_ref
        fixture.mechanics['implementation']=S.source_ref(root/'native_pose_refinement.py',root)
        fixture.mechanics['additional_implementations']=[S.source_ref(root/name,root)for name in ['sam3d_mechanisms.py','alignment_render.py','native_appearance.py','native_scene_audit.py']]
        proposal=O.read_json(root/fixture.doc['proposal']['path'])
        plan=fixture.file('data/topology/plan.json',{'contract':{'code_sha256':O.sha(root/'mechanism_reasoning.py')}})
        proposal['reasoning']={'plan_path':plan['path']};proposal_ref=fixture.file(fixture.doc['proposal']['path'],proposal)
        fixture.mechanics['proposal']=proposal_ref;fixture.update_mechanics()
        result=O.build(fixture.doc['constrained_scene']['path'],'data/object-programs/fresh-ledger-bound/object-program.json',root=root)
        document=O.read_json(root/result['path']);self.assertTrue(O.validate(document,root=root))
        archived={r['original_path']:r for r in document['implementation_provenance']}
        for name in ['native_audit_protocol.py','part_ledger.py','native_pose_refinement.py','sam3d_mechanisms.py','alignment_render.py','native_scene_audit.py']:
            self.assertIn(name,archived);self.assertEqual((root/archived[name]['archived_file']['path']).read_bytes(),(root/name).read_bytes())
        self.assertEqual((root/fixture.doc['evidence_program']['path']).read_bytes(),before)


if __name__=='__main__':unittest.main()
