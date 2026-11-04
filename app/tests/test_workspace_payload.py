"""Evidence/geometry contracts for Stage 6, independently of browser controls."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('workspace_payload_builder',ROOT/'scripts/build_workspace.py')
W=importlib.util.module_from_spec(spec);spec.loader.exec_module(W)


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data))


def triangle():
    return [{'node':'actual_node','geometry':'actual_geometry','material_index':0,
             'vertices':[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],'faces':[[0,1,2]],
             'base_color_factor':[.2,.3,.4,1.],'appearance_mode':'vertex_rgb',
             'colors':[[1.,0.,0.,1.],[0.,1.,0.,1.],[0.,0.,1.,1.]],'uv':None,'texture':None}]


class WorkspacePayloadTests(unittest.TestCase):
    def test_static_sam3d_parts_never_decode_into_an_aggregate_scene_or_reuse_an_old_link(self):
        doc={'assets':[{'track_id':'lid','path':'data/native/lid.glb','sha256':'a'*64,'backend':'Meta SAM 3D Objects'}],
             'motion':[], 'links':{'scene_viewer':'data/old-comparison/index.html'}}
        original=copy.deepcopy(doc)
        with patch.object(W,'mesh_reader',side_effect=AssertionError('Static native geometry must stay in its original GLB')),\
             patch.object(W,'read',side_effect=AssertionError('No old aggregate scene may be substituted')):
            self.assertIsNone(W.generic_scene(doc))
            self.assertIsNone(W.scene_for(doc))
        self.assertEqual(doc,original)
        doc['motion']=[{'track_id':'lid','status':'reference_candidate_unverified_geometry_and_pose',
                        'camera_from_native_mesh':[[1,0,0,0],[0,1,0,0],[0,0,1,3],[0,0,0,1]]}]
        self.assertTrue(W.static_sam3d_assets(doc),'An unverified reference is not supported native motion')
        doc['motion'][0]['status']='supported_conditional_motion_candidate'
        self.assertFalse(W.static_sam3d_assets(doc),'Supported native motion retains its existing exact scene path')
        doc['assets'][0]['backend']='TripoSR'
        doc['motion']=[]
        self.assertFalse(W.static_sam3d_assets(doc),'Historical comparison scenes remain unchanged')

    def test_real_static_native_program_preserves_all_assets_exports_and_source_without_decoding(self):
        path=ROOT/'data/flow-sam3d-default-e2e-20261102-v1/flow-procedure-1/interaction.json'
        if not path.exists():self.skipTest('Fresh real native upload is not available in this checkout')
        doc=W.read(path)
        self.assertTrue(W.static_sam3d_assets(doc))
        with patch.object(W,'mesh_reader',side_effect=AssertionError('No native GLB decoding in the workspace payload')):
            payload=W.payload(path)
        self.assertIsNone(payload['scene'])
        self.assertEqual(payload['interaction'],doc)
        self.assertEqual(payload['labels']['assets'],len(doc['assets']))
        expected={a['path'] for a in doc['assets']}
        self.assertEqual({e['path'] for e in payload['exports'] if e['filename'].endswith('.glb')},expected)
        for asset in doc['assets']:self.assertEqual(W.stage5.sha(ROOT/asset['path']),asset['sha256'])
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);report=W.lazy_payload({'cases':[payload]},directory)
            saved=W.read(directory/'cases-index.json')['cases'][0]
            self.assertIsNone(saved['scene']);self.assertIsNone(saved.get('scene_ref'))
            self.assertEqual(saved['interaction']['assets'],doc['assets'])
            self.assertEqual(report['scene_files'],0)
            self.assertEqual(list((directory/'scenes').iterdir()),[])
            self.assertLess((directory/'cases-index.json').stat().st_size,4*1024*1024)

    def test_nested_upgrade_results_have_distinct_run_ids_without_changing_legacy_ids(self):
        source=ROOT/'data/jabra-stage6-v1/procedure-v1/interaction.json'
        if not source.exists():self.skipTest('Actual Jabra program unavailable')
        doc=W.read(source)
        with patch.object(W,'read',return_value=doc),patch.object(W.stage5,'validate_interaction'),patch.object(W,'scene_for',return_value=None),patch.object(W.stage5,'sha',return_value='a'*64):
            first=W.payload(ROOT/'data/model-upgrades/runs/upgrade-first/procedure-attempt-1/procedure/interaction.json')
            second=W.payload(ROOT/'data/model-upgrades/runs/upgrade-second/procedure-attempt-1/procedure/interaction.json')
            legacy=W.payload(source)
        self.assertEqual(first['id'],'upgrade-first--procedure-attempt-1--procedure')
        self.assertNotEqual(first['id'],second['id'])
        self.assertEqual(legacy['id'],'jabra-stage6-v1--procedure-v1')

    def test_additional_case_registry_is_explicit_and_confined_to_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'workspace').mkdir();(root/'data/study').mkdir(parents=True)
            program=root/'data/study/interaction.json';program.write_text('{}')
            rows=[{'interaction':'data/study/interaction.json','title':'Candidate study'},
                  {'interaction':'data/../workspace/interaction.json','title':'Escaped'},
                  {'interaction':'data/.private/interaction.json','title':'Hidden'},
                  {'interaction':'data/study/interaction.json','title':''}]
            write(root/'workspace/registered-cases.json',{'cases':rows})
            candidates,errors=W.registered_candidates(root)
            self.assertEqual(candidates,[(program.resolve(),'Candidate study')])
            self.assertEqual(len(errors),3)

    def test_independent_native_unit_assets_never_become_an_assembly(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write(root/'frames.json',{'frames':[{'width':540,'height':960}]})
            doc={'assets':[{'track_id':t,'path':t+'.glb','sha256':'a'*64} for t in ['part_a','part_b']],
                 'tracklets':[{'id':t,'label':t} for t in ['part_a','part_b']],
                 'motion':[{'track_id':t,'source_sample_index':i+1,'timestamp_s':float(i),
                            'status':'supported_conditional_motion_candidate','reasons':[],
                            'camera_from_native_mesh':[[1,0,0,0],[0,1,0,0],[0,0,1,3],[0,0,0,1]]}
                           for t in ['part_a','part_b'] for i in [0,1]],
                 'inputs':[{'kind':'source_frame_index','path':'frames.json'}],
                 'source':{'duration_s':2.},'evidence':[{'timestamp_s':0.}]}
            reader=SimpleNamespace(mesh_primitives=lambda p:(triangle(),{'appearance_modes':['vertex_rgb']}))
            with patch.object(W,'ROOT',root),patch.object(W,'mesh_reader',return_value=reader):
                scene=W.generic_scene(doc)
            self.assertTrue(scene['native_only'])
            self.assertIsNone(scene['source_projection_rows'])
            self.assertEqual(scene['objects'][0]['primitives'],triangle())
            self.assertFalse(scene['physical_assembly_verified'])

    def test_actual_coin_primitives_and_colors_are_preserved_without_dynamic_claim(self):
        path=ROOT/'data/wallet-stage6-v1/procedure-fresh-v1/interaction.json'
        if not path.exists():self.skipTest('Actual wallet artifact not available')
        doc=W.read(path)
        scene=W.generic_scene(doc)
        self.assertEqual(doc['motion'],[])
        self.assertTrue(scene['native_only'])
        self.assertIsNone(scene['source_projection_rows'])
        self.assertTrue(all(not obj['frames'] for obj in scene['objects']))
        self.assertFalse(scene['physical_assembly_verified'])
        original,_=W.mesh_reader().mesh_primitives(ROOT/doc['assets'][0]['path'])
        self.assertEqual(scene['objects'][0]['primitives'],original)
        self.assertTrue(any(p['colors'] or p['texture'] for p in original))

    def test_independent_parts_keep_separate_motion_bounds_and_matching_cameras(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write(root/'frames.json',{'frames':[{'width':540,'height':960}]})
            assets=[{'track_id':t,'path':t+'.glb','sha256':str(i+1)*64} for i,t in enumerate(['part_a','part_b'])]
            inputs=[{'kind':'source_frame_index','path':'frames.json'}]
            motion=[]
            for i,asset in enumerate(assets):
                tid=asset['track_id']
                write(root/(tid+'.json'),{'object_id':tid,'provenance':{'mesh_sha256':asset['sha256']},
                      'intrinsics':{'matrix':[[400+i*100,0,269.5],[0,400+i*100,479.5],[0,0,1]]}})
                inputs.insert(0,{'kind':'model_conditional_motion_experiment','path':tid+'.json'})
                motion.append({'track_id':tid,'source_sample_index':1,'timestamp_s':0.,
                               'status':'supported_conditional_motion_candidate','reasons':[],
                               'camera_from_native_mesh':[[1,0,0,i*20],[0,1,0,0],[0,0,1,3],[0,0,0,1]]})
            doc={'assets':assets,'tracklets':[{'id':a['track_id'],'label':a['track_id']} for a in assets],
                 'inputs':inputs,'motion':motion,'source':{'duration_s':2.},'evidence':[{'timestamp_s':0.}]}
            reader=SimpleNamespace(mesh_primitives=lambda p:(triangle(),{}))
            with patch.object(W,'ROOT',root),patch.object(W,'mesh_reader',return_value=reader):
                scene=W.generic_scene(doc)
            self.assertTrue(scene['native_only'])
            self.assertIsNone(scene['source_projection_rows'])
            a,b=scene['objects']
            self.assertEqual(b['motion_scene']['target'][0]-a['motion_scene']['target'][0],20)
            self.assertNotEqual(a['source_projection_rows'],b['source_projection_rows'])
            self.assertAlmostEqual(a['source_projection_rows'][0][0],800/540)
            self.assertAlmostEqual(b['source_projection_rows'][0][0],1000/540)

    def saved_scene(self,root,gauge=True):
        source='1'*64
        assets=[];objects=[];inputs=[]
        for i,tid in enumerate(['part_a','part_b']):
            asset=root/(tid+'.glb');asset.write_bytes(tid.encode())
            sha=W.stage5.sha(asset)
            pack={'object_id':tid,'provenance':{'source_video_sha256':source,'mesh_sha256':sha,'mesh_file':str(asset),
                                              'manual_geometry_used':False,'manual_pose_used':False,'manual_corners_used':False},
                  'frames':[{'sample_index':1,'timestamp_s':0.,'status':'supported_conditional_motion_candidate',
                             'candidate_rotation':[[1,0,0],[0,1,0],[0,0,1]],'candidate_translation':[0,0,3]}]}
            exp=root/(tid+'-experiment.json');write(exp,pack)
            assets.append({'track_id':tid,'path':asset.name,'sha256':sha})
            inputs.append({'id':'motion_'+str(i+1),'path':exp.name,'sha256':W.stage5.sha(exp),'kind':'model_conditional_motion_experiment'})
            objects.append({'object_id':tid,'provenance':{'experiment_file':str(exp),'experiment_sha256':W.stage5.sha(exp),
                                                        'mesh_file':str(asset),'mesh_sha256':sha,
                                                        'geometry_modified':False,'manual_geometry_used':False,'manual_pose_used':False},
                            'fixed_depth_gauge_scale':1.,'frames':[{'sample_index':1,'timestamp_s':0.,
                            'status':'supported_conditional_motion_candidate','visible':True,
                            'camera_from_native_mesh':[[1,0,0,0],[0,1,0,0],[0,0,1,3],[0,0,0,1]]}]})
        scene={'objects':objects,'physical_assembly_verified':False}
        if gauge:scene['depth_gauge']={'method':'same_frame_relative_depth','physical_scale_verified':False,
                                     'selected_scales':{'part_a':1.,'part_b':1.}}
        write(root/'review/scene.json',scene)
        doc={'source':{'sha256':source},'assets':assets,'inputs':inputs,'links':{'scene_viewer':'review/index.html'}}
        return doc,scene

    def rejected_scene(self,mutate):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);doc,scene=self.saved_scene(root)
            mutate(root,doc,scene)
            write(root/'review/scene.json',scene)
            fallback={'native_only':True,'reason':'independent_asset_inspection'}
            with patch.object(W,'ROOT',root),patch.object(W,'generic_scene',return_value=fallback):
                self.assertEqual(W.scene_for(doc),fallback)

    def test_saved_scene_rejects_object_to_mesh_swap_even_when_hashes_are_in_asset_set(self):
        def mutate(root,doc,scene):
            a,b=scene['objects']
            a['provenance']['mesh_sha256']=b['provenance']['mesh_sha256']
            a['provenance']['mesh_file']=b['provenance']['mesh_file']
        self.rejected_scene(mutate)

    def test_valid_source_track_mesh_pose_join_is_not_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);doc,scene=self.saved_scene(root)
            with patch.object(W,'ROOT',root):
                self.assertTrue(W.scene_lineage_valid(scene,doc))

    def test_accepted_scene_reloads_geometry_and_colors_from_actual_asset_reader(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);doc,scene=self.saved_scene(root)
            scene['objects'][0]['primitives']=[{'vertices':[[999,999,999]],'colors':[[0,0,0,0]]}]
            write(root/'review/scene.json',scene)
            reader=SimpleNamespace(mesh_primitives=lambda p:(triangle(),{'appearance_modes':['vertex_rgb']}))
            with patch.object(W,'ROOT',root),patch.object(W,'mesh_reader',return_value=reader):
                accepted=W.scene_for(doc)
            self.assertEqual(accepted['objects'][0]['primitives'],triangle())

    def test_saved_scene_rejects_pose_not_in_upstream_experiment(self):
        def mutate(root,doc,scene):scene['objects'][0]['frames'][0]['camera_from_native_mesh'][0][3]=.25
        self.rejected_scene(mutate)

    def test_saved_scene_rejects_scale_different_from_shared_gauge(self):
        def mutate(root,doc,scene):scene['objects'][0]['fixed_depth_gauge_scale']=2.
        self.rejected_scene(mutate)

    def test_saved_scene_rejects_source_mismatch_even_with_matching_mesh(self):
        def mutate(root,doc,scene):
            obj=scene['objects'][0];p=Path(obj['provenance']['experiment_file']);pack=W.read(p)
            pack['provenance']['source_video_sha256']='2'*64;write(p,pack)
            obj['provenance']['experiment_sha256']=W.stage5.sha(p)
            doc['inputs'][0]['sha256']=W.stage5.sha(p)
        self.rejected_scene(mutate)

    def test_saved_scene_rejects_stale_experiment_hash(self):
        def mutate(root,doc,scene):
            obj=scene['objects'][0];p=Path(obj['provenance']['experiment_file']);pack=W.read(p)
            pack['unverified_new_field']='changed_after_scene_generation';write(p,pack)
        self.rejected_scene(mutate)

    def test_multiple_saved_assets_need_explicit_common_gauge(self):
        self.rejected_scene(lambda root,doc,scene:scene.pop('depth_gauge'))

    def test_actual_payloads_keep_fresh_semantics_and_zero_pose_wallet(self):
        expected=[('data/jabra-stage6-v1/procedure-v1/interaction.json',14),
                  ('data/wallet-stage6-v1/procedure-fresh-v1/interaction.json',10)]
        for name,count in expected:
            path=ROOT/name
            if not path.exists():self.skipTest('Actual artifact unavailable: '+name)
            with self.subTest(path=name):
                payload=W.payload(path)
                self.assertEqual(payload['labels']['steps'],count)
                self.assertFalse(payload['interaction']['runtime_policy']['simulation_ready'])
                if 'wallet' in name:
                    self.assertEqual(payload['labels']['conditional_poses'],0)
                    self.assertTrue(payload['scene']['native_only'])
                self.assertTrue(payload['exports'])

    def test_payload_rejects_corrupted_cached_input_hash(self):
        source=ROOT/'data/jabra-stage6-v1/procedure-v1/interaction.json'
        if not source.exists():self.skipTest('Actual Jabra artifact unavailable')
        doc=W.read(source);doc['inputs'][0]['sha256']='0'*64
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path=Path(temp)/'interaction.json';write(path,doc)
            with self.assertRaisesRegex(ValueError,'changed evidence'):
                W.payload(path)

    def test_completed_smoke_procedures_join_actual_source_and_assets(self):
        paths=list((ROOT/'data').glob('*/flow-procedure-*/interaction.json'))
        if not paths:self.skipTest('Fresh smoke procedure is still running')
        for path in paths:
            with self.subTest(path=str(path)):
                payload=W.payload(path)
                self.assertEqual(payload['labels']['source_samples'],len(payload['interaction']['evidence']))
                self.assertFalse(payload['interaction']['runtime_policy']['physical_identity_confirmed'])


if __name__=='__main__':unittest.main()
