"""Numeric FK/scale/unknown-time and confined companion format regression."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import object_program as O


class ObjectProgramTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        def file(name, value):
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(value if isinstance(value,bytes) else json.dumps(value).encode())
            return O.ref(path,self.root)
        self.file=file
        video=file('data/case/video.mp4',b'contract-video')
        self.source={'path':video['path'],'sha256':video['sha256'],'original_sha256':video['sha256'],'duration_s':2.,'time_domain':'source_video_presentation_seconds'}
        program={'source':self.source,'tracklets':[{'id':'body'},{'id':'lid'}],
                 'evidence':[{'id':'source_a','source_index':1,'timestamp_s':0.}]}
        pref=file('data/case/program.json',program)
        self.components=[]
        for cid, scale in [('body',2.),('lid',3.)]:
            self.components.append({'id':cid,'asset':file(f'data/{cid}/asset.glb',b'glTFfixture'),
                                    'native_metadata':file(f'data/{cid}/metadata.json',{'object_id':cid,'source':{'video_sha256':video['sha256']}}),
                                    'native_uniform_scale':scale,'poses':[]})
        a=np.eye(4);a[:3,:3]/=2
        b=np.eye(4);b[:3,:3]*=3
        self.joint={'id':'hinge','type':'revolute','parent_id':'body','child_id':'lid','status':'estimated',
                    'axis':[0.,1.,0.],'parent_from_joint':a.tolist(),'joint_from_child_zero':b.tolist(),
                    'parent_anchor_native':[0.,0.,0.],'child_anchor_native':[0.,0.,0.],
                    'parent_anchor_vertex_id':0,'child_anchor_vertex_id':0,'parent_native_scale':2.,'child_native_scale':3.,
                    'limits_rad':[0.,.5],'limits_kind':'observed_estimated_range_not_verified_physical_stops','physical_limits_rad':None,
                    'zero_reference':{'source_sample_index':1,'timestamp_s':0.,'state':'model_reference_state_unknown'},
                    'samples':[],'evidence_ids':['source_a'],'matrix_convention':O.CONVENTION,
                    'physical_accuracy_verified':False,'simulation_ready':False,
                    'angle_quality':{'angle_visual_consistency':False,'test':'all_cited_supported_open_states_separated_from_cited_closed_zero_by_more_than_declared_threshold',
                        'declared_separation_threshold_rad':np.pi/12,'state_comparisons':[],
                        'angle_interpretation':'model_configuration_parameter_not_verified_opening_angle','confidence':'low',
                        'physical_angle_verified':False,'zero_state_alignment_verified':False,'source_alignment_verified':False,
                        'quality_flags':['No supported closed/open comparison.']}}
        for index,q in [(1,0.),(2,.5),(3,None)]:
            p=np.eye(4);p[:3,:3]*=2;p[2,3]=5
            r=np.eye(4);r[:3,:3]=O.axis_rotation([0,1,0],q or 0)
            child=p@a@r@b
            self.components[0]['poses'].append({'source_sample_index':index,'timestamp_s':index-1.,'status':'estimated','camera_from_native_mesh':p.tolist()})
            self.components[1]['poses'].append({'source_sample_index':index,'timestamp_s':index-1.,'status':'estimated' if q is not None else 'unknown',
                                               'camera_from_native_mesh':child.tolist() if q is not None else None})
            self.joint['samples'].append({'source_sample_index':index,'timestamp_s':index-1.,'status':'estimated' if q is not None else 'unknown','q_rad':q})
        for component in self.components:
            component['native_metadata']=file(f"data/{component['id']}/metadata.json",{'object_id':component['id'],'source':{'video_sha256':video['sha256']},
                'artifact_files':[{'file':'asset.glb','sha256':component['asset']['sha256']}],
                'geometry_reconstructed_by_model':True,'manual_geometry_fallback':False,'manual_pose_used':False})
        source_scene={'format':'sam3d-common-video-scene','source':{**self.source,'width':540,'height':960},'original_program':pref,
                      'components':copy.deepcopy(self.components),'inputs':[],'dependencies':[],
                      'physical_accuracy_verified':False,'simulation_ready':False,'manual_geometry_or_poses_used':False,
                      'old_backend_geometry_or_poses_read':False,'accepted_motion_poses':0}
        sref=file('data/scene/source.json',source_scene)
        proposal={'format':'sam3d-mechanism-proposals','version':1,'source_sha256':video['sha256'],'interaction_sha256':pref['sha256'],
                  'physical_accuracy_verified':False,'joints':[{'id':'hinge','parent_id':'body','child_id':'lid','joint_type':'revolute',
                    'attachment':'persistent','evidence_ids':['source_a']}]}
        proposal_ref=file('data/proposal/proposals.json',proposal)
        code=file('data/object-programs/unit/provenance/sam3d_mechanisms.py.txt',b'unit fixture code')
        self.mechanics={'format':'sam3d-constrained-mechanisms','version':1,'source_scene':sref,'proposal':proposal_ref,
                        'joints':[self.joint],'unresolved_children':[],'implementation':{'path':'sam3d_mechanisms.py','sha256':code['sha256'],'bytes':code['bytes']},
                        'physical_accuracy_verified':False,'simulation_ready':False,'manual_joint_coordinates_used':False,'free_floating_child_fallback':False}
        mref=file('data/scene/mechanics.json',self.mechanics)
        self.scene={**source_scene,'components':self.components,'mechanics':self.mechanics}
        cref=file('data/scene/scene.json',self.scene)
        self.doc={'format':O.FORMAT,'version':O.VERSION,'created_utc':'2026-11-02T00:00:00+00:00','evidence_program':pref,
                  'source':source_scene['source'],'source_scene':sref,'constrained_scene':cref,'mechanics':mref,'proposal':proposal_ref,
                  'component_assets':[{'component_id':c['id'],'native_glb':c['asset'],'native_metadata':c['native_metadata'],
                    'native_uniform_scale':c['native_uniform_scale']} for c in self.components],
                  'joints':[O.projected_joint(self.joint)],'unresolved_connections':[],'removable_relationships':[],
                  'coordinate_system':{'camera_frame':'opencv_x_right_y_down_z_forward','length_units':'inferred_arbitrary_scene_units_no_metric_scale',
                    'angle_units':'radians','unknown_sample_policy':'withhold_joint_and_child_when_parent_or_joint_unknown','physical_stops':'unknown'},
                  'implementation_provenance':[{'original_path':'sam3d_mechanisms.py','sha256':code['sha256'],'archived_file':code}],
                  'physical_accuracy_verified':False,'simulation_ready':False,'robot_execution_available':False,'manual_joint_coordinates_used':False,
                  'limitations':['Fixture estimates have no physical verification.']}
        # These fixtures test actual reference/FK/scale/timing validation, while
        # replacing just the already separately tested provider grounding and
        # GLB decoding. Neither replacement runs inference or invents poses.
        self.p1=patch.object(O,'_verify_proposal');self.p1.start();self.addCleanup(self.p1.stop)
        self.p2=patch.object(O,'native_vertices',return_value=np.asarray([[0.,0.,0.],[1.,0.,0.]]))
        self.p2.start();self.addCleanup(self.p2.stop)

    def update_mechanics(self):
        self.scene['mechanics']=self.mechanics
        self.doc['mechanics']=self.file('data/scene/mechanics.json',self.mechanics)
        self.doc['constrained_scene']=self.file('data/scene/scene.json',self.scene)
        self.doc['joints']=[O.projected_joint(j) for j in self.mechanics['joints']]
        self.doc['unresolved_connections']=self.mechanics['unresolved_children']

    def test_exact_scaled_fk_and_all_unknown_samples_validate(self):
        self.assertTrue(O.validate(self.doc,root=self.root))
        self.assertEqual(self.doc['joints'][0]['samples'][-1]['q_rad'],None)
        self.assertEqual(self.doc['coordinate_system']['angle_units'],'radians')

    def test_optional_correction_implementation_is_exactly_archived_and_legacy_stays_valid(self):
        before=copy.deepcopy(self.doc);self.assertTrue(O.validate(before,root=self.root))
        archive=self.file('data/object-programs/unit/provenance/native_pose_refinement.py.txt',b'analytic correction implementation')
        self.mechanics['additional_implementations']=[{'path':'native_pose_refinement.py','sha256':archive['sha256'],'bytes':archive['bytes']}]
        self.update_mechanics()
        with self.assertRaisesRegex(ValueError,'Missing historical'):O.validate(self.doc,root=self.root)
        self.doc['implementation_provenance'].append({'original_path':'native_pose_refinement.py','sha256':archive['sha256'],'archived_file':archive})
        self.assertTrue(O.validate(self.doc,root=self.root));self.assertEqual(before['implementation_provenance'][0]['original_path'],'sam3d_mechanisms.py')
        self.mechanics['additional_implementations'][0]['bytes']+=1;self.update_mechanics()
        with self.assertRaisesRegex(ValueError,'Missing historical'):O.validate(self.doc,root=self.root)

    def test_conflicting_or_escaping_optional_implementation_is_rejected(self):
        reference=copy.deepcopy(self.mechanics['implementation']);reference['sha256']='a'*64
        self.mechanics['additional_implementations']=[reference];self.update_mechanics()
        with self.assertRaisesRegex(ValueError,'Conflicting'):O.validate(self.doc,root=self.root)
        reference['path']='../private.py';self.update_mechanics()
        with self.assertRaisesRegex(ValueError,'allowed implementation'):O.validate(self.doc,root=self.root)

    def test_rejects_child_pose_that_free_floats_away_from_declared_joint(self):
        self.scene['components'][1]['poses'][1]['camera_from_native_mesh'][0][3]+=.1
        self.doc['constrained_scene']=self.file('data/scene/scene.json',self.scene)
        with self.assertRaisesRegex(ValueError,'forward kinematics'):
            O.validate(self.doc,root=self.root)

    def test_rejects_double_scale_and_false_physical_stops(self):
        self.mechanics['joints'][0]['joint_from_child_zero'][0][0]*=3
        self.update_mechanics()
        with self.assertRaisesRegex(ValueError,'similarity'):
            O.validate(self.doc,root=self.root)
        self.doc['joints'][0]['physical_limits_rad']=[0,1]
        with self.assertRaises(Exception): O.validate(self.doc,root=self.root)

    def test_rejects_filled_unknown_and_invented_source_time(self):
        self.mechanics['joints'][0]['samples'][-1].update(status='estimated',q_rad=.5)
        self.update_mechanics()
        with self.assertRaisesRegex(ValueError,'withheld'):
            O.validate(self.doc,root=self.root)
        self.mechanics['joints'][0]['samples'][-1].update(status='unknown',q_rad=None,timestamp_s=1.9)
        self.update_mechanics()
        with self.assertRaisesRegex(ValueError,'time mismatch'):
            O.validate(self.doc,root=self.root)

    def test_rejects_mismatched_asset_undelared_vertex_and_observed_range(self):
        changed=copy.deepcopy(self.doc)
        changed['component_assets'][0]['native_glb']['sha256']='f'*64
        with self.assertRaises(ValueError): O.validate(changed,root=self.root)
        self.mechanics['joints'][0]['parent_anchor_vertex_id']=500
        self.update_mechanics()
        with self.assertRaisesRegex(ValueError,'vertex'):
            O.validate(self.doc,root=self.root)
        self.mechanics['joints'][0]['parent_anchor_vertex_id']=0
        self.mechanics['joints'][0]['limits_rad']=[0,3.14]
        self.update_mechanics()
        with self.assertRaisesRegex(ValueError,'observed'):
            O.validate(self.doc,root=self.root)

    def test_zero_fitted_joints_exports_removable_only_graph_and_preserves_unknown_topology(self):
        proposal_ref=self.doc['proposal']
        proposal=json.loads((self.root/proposal_ref['path']).read_text())
        j=proposal['joints'][0];j.update(joint_type='removable',attachment='removable',confidence='medium')
        self.doc['proposal']=self.file(proposal_ref['path'],proposal)
        self.mechanics['proposal']=self.doc['proposal']
        self.mechanics['joints']=[]
        self.doc['removable_relationships']=[{'id':'hinge','parent_id':'body','child_id':'lid','joint_type':'removable',
            'attachment':'removable','confidence':'medium','evidence_ids':['source_a'],'physical_accuracy_verified':False}]
        self.update_mechanics()
        self.assertTrue(O.validate(self.doc,root=self.root))
        j.update(joint_type='prismatic',attachment='persistent')
        self.doc['proposal']=self.file(proposal_ref['path'],proposal);self.mechanics['proposal']=self.doc['proposal']
        self.mechanics['unresolved_children']=[{'id':'hinge','parent_id':'body','child_id':'lid','type':'prismatic',
            'status':'unknown','reason':'not_supported'}]
        self.doc['removable_relationships']=[]
        self.update_mechanics()
        with self.assertRaisesRegex(ValueError,'must remain absent'):
            O.validate(self.doc,root=self.root)
        for row in self.scene['components'][1]['poses']:
            row.update(status='unknown',camera_from_native_mesh=None)
        self.update_mechanics()
        self.assertTrue(O.validate(self.doc,root=self.root))

    def test_inconsistent_angle_cannot_be_promoted_to_verified_or_visual_opening_angle(self):
        self.mechanics['joints'][0]['angle_quality'].update(angle_visual_consistency=True,
            angle_interpretation='estimated_opening_configuration_not_physically_calibrated')
        self.update_mechanics()
        with self.assertRaisesRegex(ValueError,'cannot conceal'):
            O.validate(self.doc,root=self.root)
        self.doc['joints'][0]['angle_quality']['physical_angle_verified']=True
        with self.assertRaises(Exception):O.validate(self.doc,root=self.root)

    def test_catalog_exposes_exact_joint_program_without_api_or_robot_execution(self):
        from twin_catalog import TwinCatalog
        target='data/object-programs/unit/object-program.json'
        self.file(target,self.doc)
        catalog=TwinCatalog(self.root)
        program={'source':self.source}
        with patch.object(catalog,'load',return_value=({},program,self.doc['evidence_program']['sha256'])) as loaded:
            value=catalog.kinematics('saved-twin')
            self.assertEqual(value['object_program']['joints'][0]['type'],'revolute')
            self.assertEqual(value['artifact']['uri'],'twin://saved-twin/kinematics')
            self.assertFalse(value['policy']['simulation_ready'])
            self.assertTrue(value['original_evidence_program_unchanged'])
            raw=catalog.kinematics_artifact('saved-twin')
            self.assertEqual(__import__('hashlib').sha256(raw).hexdigest(),value['artifact']['sha256'])
            self.assertEqual(catalog.kinematics_artifact('saved-twin','mechanics'),(self.root/'data/scene/mechanics.json').read_bytes())
            loaded.reset_mock()
            with self.assertRaises(ValueError):catalog.kinematics_artifact('saved-twin','../../private')
            loaded.assert_not_called()

    def test_ref_confinement_precedes_hashing_hidden_or_external_paths(self):
        for path in ['.secrets/meta.env','data/../.secrets/meta.env','/tmp/private','data/.hidden/value']:
            with self.subTest(path=path),patch.object(O,'sha') as hashing,self.assertRaises(ValueError):
                O.checked_ref({'path':path,'sha256':'f'*64,'bytes':3},self.root)
            hashing.assert_not_called()
        link=self.root/'data/escape';link.symlink_to(self.root)
        with self.assertRaises(ValueError): O.data_path(self.root,'data/escape/example.json')

    def test_discovery_is_exact_program_bound_and_skips_invalid_optional_companions(self):
        target='data/object-programs/unit/object-program.json'
        self.file(target,self.doc)
        found=O.discover(self.root,self.doc['evidence_program']['sha256'],self.source['sha256'])
        self.assertEqual(found['artifact']['path'],target)
        self.assertIsNone(O.discover(self.root,'e'*64,self.source['sha256']))
        self.file('data/scene/scene.json',b'changed')
        self.assertIsNone(O.discover(self.root,self.doc['evidence_program']['sha256'],self.source['sha256']))

if __name__=='__main__':unittest.main()
