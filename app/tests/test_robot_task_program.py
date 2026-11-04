"""Pure rehearsal contracts and one read-only actual-curated integration probe."""
import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from curated_twin_catalog import CuratedTwinCatalog
import robot_task_program as R


class FixtureCatalog(CuratedTwinCatalog):
    """Explicit mathematical fixture; no generated fixture enters a catalog."""
    def __init__(self, root):
        self.root = Path(root).resolve();self.checked = []
        self.doc = {'source': {'sha256': 'a'*64}, 'tracklets': [
            {'id': 'container', 'label': 'charging case base'}, {'id': 'lid', 'label': 'charging case lid'},
            {'id': 'part_a', 'label': 'wireless earbud'}, {'id': 'part_b', 'label': 'wireless earbud'}],
            'assets': [], 'relationships': []}
        def ref(name, sha):return {'path': 'data/fixture/'+name, 'sha256': sha*64, 'bytes': 200}
        self.refs = {cid: ref(cid+'.glb', str(index+1)) for index, cid in enumerate(['container', 'lid', 'part_a', 'part_b'])}
        self.record = {'id': 'fixture', 'interaction': ref('interaction.json', 'b'), 'sam3d_reviews': [
            {'component_id': cid, 'source_sha256': 'a'*64, 'backend': 'Meta SAM 3D Objects', 'native_glb': pointer}
            for cid, pointer in self.refs.items()]}
        identity = [[1 if i == j else 0 for j in range(4)] for i in range(4)]
        joint = {'id': 'lid_joint', 'parent_id': 'container', 'child_id': 'lid', 'type': 'revolute', 'status': 'estimated',
            'axis': [1, 0, 0], 'parent_from_joint': identity, 'joint_from_child_zero': identity,
            'parent_native_scale': 1, 'child_native_scale': 1, 'limits_rad': [0, 2.5],
            'limits_kind': 'observed_estimated_range_not_verified_physical_stops',
            'zero_reference': {'state': 'estimated_closed', 'source_sample_index': 1, 'timestamp_s': 0},
            'axis_frame': 'joint_frame_after_parent_from_joint', 'matrix_convention': 'fixture exact frame chain',
            'evidence_ids': ['sample_1'], 'physical_accuracy_verified': False, 'simulation_ready': False}
        self.obj = {'component_assets': [{'component_id': cid, 'native_glb': copy.deepcopy(pointer)} for cid, pointer in self.refs.items()],
                    'joints': [joint], 'removable_relationships': [
                        {'id': 'removable_'+cid, 'parent_id': 'container', 'child_id': cid, 'joint_type': 'removable'}
                        for cid in ['part_a', 'part_b']]}
        self.artifact = {**ref('object-program.json', 'c'), 'uri': 'twin://fixture/kinematics', 'mime_type': 'application/json'}

    def _record(self, twin_id):
        if twin_id != 'fixture':raise ValueError('Not curated')
        return copy.deepcopy(self.record), {}

    def load(self, twin_id):
        self._record(twin_id)
        return {}, copy.deepcopy(self.doc), 'b'*64

    def _checked(self, reference, **kwargs):
        self.checked.append(copy.deepcopy(reference));return self.root/reference['path']

    def _bytes(self, reference, **kwargs):
        # Explicit synthetic cuboid geometry only in this mathematical fixture.
        import trimesh
        return trimesh.creation.box(extents=[1, .3, .5]).export(file_type='glb')

    def kinematics(self, twin_id):
        self._record(twin_id)
        return {'object_program': copy.deepcopy(self.obj), 'artifact': copy.deepcopy(self.artifact)}


class RobotTaskProgramTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve();self.catalog = FixtureCatalog(self.root)

    def plan(self, instruction=None, **kwargs):
        return R.plan_robot_task('fixture', instruction, root=self.root, catalog=self.catalog, **kwargs)

    def validate(self, program):
        return R.validate_robot_task(program, root=self.root, catalog=self.catalog)

    def codes(self, result):return {e['code'] for e in result['errors']}

    def test_default_places_case_opens_transfers_both_parts_and_closes(self):
        p = self.plan();self.assertEqual(p['status'], 'ready');self.assertEqual(len(p['actions']), 38)
        self.assertTrue(self.validate(p)['valid']);self.assertEqual(p['goal']['component_ids'], ['part_a', 'part_b'])
        attached = [a['object_id'] for a in p['actions'] if a['operation'] == 'attach']
        self.assertEqual(attached, ['container', 'part_a', 'part_b'])
        angles = [a['value'] for a in p['actions'] if a['operation'] == 'set_joint']
        self.assertEqual(angles, [1.8, 0]);self.assertFalse(p['claims']['physical_accuracy_verified'])
        self.assertEqual(set(p['source_binding']['object_program_ref']), {'path', 'sha256', 'bytes'})
        self.assertTrue(all(o['placement_origin'] == 'authored_staging_not_recovered_video_pose' for o in p['objects']))

    def test_local_english_synonyms_and_charging_case_noun_preserve_both_parts(self):
        for text in ['Place the case on the table, open the lid, take out both earbuds and place them on the tray, then close it.',
                     'Open the charging case and take out both earphones to the tray, then close it.',
                     'Open the case and remove the headset to the tray, then close it.']:
            with self.subTest(text=text):
                p = self.plan(text);self.assertEqual(p['status'], 'ready');self.assertTrue(self.validate(p)['valid'])
                self.assertEqual(p['goal']['component_ids'], ['part_a', 'part_b'])

    def test_unknown_or_unsupported_clause_blocks_entire_request_without_partial_actions(self):
        for text in ['Open the case and wash the earphones.', 'Open the lid then throw both earbuds.',
                     'Open the case and plug in the phone.', 'Open the lid and connect USB C.',
                     'Open the case and execute code.', 'Remove both earbuds then return them.']:
            with self.subTest(text=text):
                p = self.plan(text);self.assertEqual(p['status'], 'blocked');self.assertEqual(p['actions'], [])
                self.assertFalse(self.validate(p)['valid']);self.assertTrue(p['missing_requirements'])

    def test_singular_or_physical_side_is_not_fabricated_into_identity(self):
        for text in ['Take out the earbud.', 'Take out the left earbud.', 'Take out the right earbud.']:
            p = self.plan(text);self.assertEqual(p['status'], 'blocked');self.assertEqual(p['actions'], [])

    def test_return_is_explicit_separate_initial_staging_and_no_seating_claim(self):
        p = self.plan('Return part_a from the tray to the open case, then close it.')
        self.assertEqual(p['status'], 'ready');self.assertTrue(self.validate(p)['valid'])
        self.assertEqual(p['goal']['operation'], 'return_components');self.assertEqual(p['attachments'], [])
        self.assertFalse(p['claims']['contact_or_seating_verified'])

    def test_typed_goal_unknown_fields_foreign_component_and_conflicting_inputs_block(self):
        goal = {'operation': 'remove_components', 'component_ids': ['part_b'], 'open_container': True,
                'close_container': False, 'destination': 'table'}
        self.assertEqual(self.plan(goal=goal)['status'], 'ready')
        for changed in [{**goal, 'shell': 'anything'}, {**goal, 'component_ids': ['foreign']},
                        {**goal, 'open_container': 'yes'}]:
            self.assertEqual(self.plan(goal=changed)['status'], 'blocked')
        self.assertEqual(self.plan('Open the lid', goal=goal)['status'], 'blocked')

    def test_missing_geometry_joint_or_mismatched_joint_asset_blocks_no_fallback(self):
        self.catalog.obj['component_assets'][0]['native_glb']['sha256'] = 'f'*64
        self.assertEqual(self.plan()['status'], 'blocked')
        self.catalog.obj['joints'] = []
        self.assertEqual(self.plan()['status'], 'blocked')

    def test_native_refs_joint_and_inventory_cannot_be_edited_or_secret_paths_read(self):
        original = self.plan()
        variants = []
        p = copy.deepcopy(original);p['objects'][0]['asset_ref']['path'] = '.secrets/meta.env';variants.append(p)
        p = copy.deepcopy(original);p['objects'].pop();variants.append(p)
        p = copy.deepcopy(original);p['joints'] = [];variants.append(p)
        p = copy.deepcopy(original);p['joints'][0]['source_joint']['axis'] = [0, 1, 0];variants.append(p)
        p = copy.deepcopy(original);p['claims']['simulation_ready'] = True;variants.append(p)
        for p in variants:self.assertFalse(self.validate(p)['valid'])
        self.assertFalse(any('.secrets' in ref['path'] for ref in self.catalog.checked))

    def test_closed_lid_pick_and_close_while_holding_are_invalid(self):
        p = self.plan();p['actions'] = [a for a in p['actions'] if not (a['operation'] == 'set_joint' and a['value'] > 0)]
        self.assertIn('container_closed', self.codes(self.validate(p)))
        p = self.plan();at = next(i for i,a in enumerate(p['actions']) if a['operation'] == 'attach' and a['object_id'] == 'part_a')
        actuation = next(a['actuation'] for a in p['actions'] if a['operation'] == 'set_joint')
        p['actions'].insert(at+1, {'id': 'bad_close', 'operation': 'set_joint', 'title': 'Invalid close while held',
            'object_id': 'container', 'joint_id': 'lid_joint', 'value': 0, 'duration_s': .5, 'actuation': actuation})
        self.assertIn('invalid_joint_target', self.codes(self.validate(p)))

    def test_immediate_release_and_open_gripper_while_held_are_invalid(self):
        p = self.plan();at = next(i for i,a in enumerate(p['actions']) if a['operation'] == 'attach')
        p['actions'].insert(at+1, {'id': 'bad_detach', 'operation': 'detach', 'object_id': 'container',
                                 'title': 'Invalid immediate release', 'duration_s': .2})
        self.assertIn('release_without_movement', self.codes(self.validate(p)))
        p = self.plan();p['actions'].insert(at+1, {'id': 'bad_open', 'operation': 'gripper', 'value': 0,
                                                'title': 'Invalid open while held', 'duration_s': .2})
        self.assertIn('gripper_open_while_holding', self.codes(self.validate(p)))

    def test_unreachable_nonfinite_unknown_action_or_ignored_goal_is_rejected(self):
        p = self.plan();p['actions'][0]['position'] = [2, 2, 2]
        self.assertIn('unreachable_path', self.codes(self.validate(p)))
        p = self.plan();p['actions'][0]['position'][0] = float('nan')
        self.assertFalse(self.validate(p)['valid'])
        p = self.plan();p['actions'][0]['operation'] = 'exec'
        self.assertFalse(self.validate(p)['valid'])
        p = self.plan();p['actions'] = [a for a in p['actions'] if 'part_b' not in a['title'] and a.get('object_id') != 'part_b']
        self.assertIn('goal_not_reached', self.codes(self.validate(p)))

    def test_schema_bindings_and_actual_five_dof_kit_are_bounded_and_deterministic(self):
        self.assertEqual(R.get_robot_kit(), R.get_robot_kit());kit = R.get_robot_kit()
        self.assertEqual(len(kit['robot']['joint_limits']), 5);self.assertFalse(kit['units']['metric_scale_known'])
        self.assertEqual(len(R.task_bindings(self.plan())), 6)
        with self.assertRaises(ValueError):R.get_robot_kit('pretend_physical_robot')
        with self.assertRaises(ValueError):R.plan_robot_task('fixture', root=self.root, catalog=object())

    def test_generic_saved_cabinet_hinge_can_open_without_removable_parts(self):
        self.catalog.doc['tracklets'][0]['label'] = 'cabinet'
        self.catalog.obj['removable_relationships'] = []
        p = self.plan('Open the cabinet, then close it.')
        self.assertEqual(p['status'], 'ready');self.assertEqual(p['goal']['operation'], 'open_close')
        self.assertEqual(p['goal']['component_ids'], []);self.assertTrue(self.validate(p)['valid'])
        blocked = self.plan('Place the case on the shelf.')
        self.assertEqual(blocked['status'], 'blocked')

    def test_actuation_descriptor_is_source_bound_and_cannot_be_removed_or_retargeted(self):
        p = self.plan();joint_action = next(a for a in p['actions'] if a['operation'] == 'set_joint')
        cue = joint_action['actuation']
        self.assertEqual(cue['kind'], 'authored_native_bbox_surface_cue_v1')
        self.assertEqual(cue['asset_sha256'], self.catalog.refs['lid']['sha256'])
        self.assertEqual(cue['sample_count'], 41);self.assertEqual(cue['tool_tracking'], 'position_only')
        for key, value in [('anchor_native', [20, 0, 0]), ('asset_sha256', 'f'*64),
                           ('parent_native_bounds', [[-1, -1, -1], [1, 1, 1]])]:
            changed = copy.deepcopy(p);next(a for a in changed['actions'] if a['operation'] == 'set_joint')['actuation'][key] = value
            self.assertFalse(self.validate(changed)['valid'])
        changed = copy.deepcopy(p);del next(a for a in changed['actions'] if a['operation'] == 'set_joint')['actuation']
        self.assertIn('invalid_action_fields', self.codes(self.validate(changed)))

    def test_source_cue_position_follows_hinge_but_not_rigid_grasp_orientation(self):
        from robot_task_geometry import cue_state
        p = self.plan();joint = p['joints'][0];parent = next(o for o in p['objects'] if o['id'] == joint['parent_id'])
        cue = next(a['actuation'] for a in p['actions'] if a['operation'] == 'set_joint')
        tool = [-.15, .41, -.03];position = [-.15, .13, .05]
        start = cue_state(cue, joint, parent, position, tool, 0, 1.8, 0)
        gripping = cue_state(cue, joint, parent, position, tool, 0, 1.8, .5)
        end = cue_state(cue, joint, parent, position, tool, 0, 1.8, 1)
        self.assertEqual(start['position'], tool);self.assertEqual(end['position'], tool)
        self.assertEqual(start['gripper'], 0);self.assertEqual(gripping['gripper'], 1);self.assertEqual(end['gripper'], 0)
        self.assertGreater(gripping['angle_rad'], 0);self.assertEqual(end['angle_rad'], 1.8)
        self.assertFalse(p['claims']['contact_or_seating_verified'])

    def test_edit_cannot_silently_remove_the_only_requested_open_action(self):
        p = self.plan('Open the lid, then close it.')
        p['actions'] = [a for a in p['actions'] if not (a['operation'] == 'set_joint' and a['value'] > 0)]
        self.assertIn('open_goal_not_reached', self.codes(self.validate(p)))
        goal = {'operation': 'open_close', 'component_ids': [], 'open_container': False,
                'close_container': False, 'destination': 'table'}
        self.assertEqual(self.plan(goal=goal)['status'], 'blocked')

    @unittest.skipUnless(shutil.which('node'), 'Node is needed for cross-language mathematical parity')
    def test_python_and_browser_kinematics_agree_for_reachable_and_unreachable_targets(self):
        targets = [[-.5, .24, -.2], [-.15, .42, .05], [-.35, .08, .35], [2, 2, 2], [-2, .1, -1]]
        script = "const C=require('./showcase/robot-controller.js');const t=JSON.parse(process.argv[1]);process.stdout.write(JSON.stringify({robot:C.DEFAULT_ROBOT,results:t.map(x=>C.solve(x))}));"
        child = subprocess.run(['node', '-e', script, json.dumps(targets)], cwd=R.ROOT, capture_output=True, text=True, timeout=10, check=True)
        result = json.loads(child.stdout);self.assertEqual(result['robot'], R.ROBOT)
        for target, js in zip(targets, result['results']):
            py = R.inverse_kinematics(target);self.assertEqual(py['reachable'], js['reachable'])
            self.assertAlmostEqual(py['residual'], js['residual'], places=7)


class ActualCuratedRobotPlanTests(unittest.TestCase):
    def test_compiled_cap_profile_requires_exact_scene_native_assets_masks_and_source_protrusion(self):
        from robot_native_cap_profile_v1 import PROFILE
        import robot_source_staging_v1 as S
        context=R._context('flow-sam3d-default-e2e-20261102-v1--flow-procedure-1',root=R.ROOT,catalog=None)
        raw=S.build(context,'case_base',['earbud_a','earbud_b'])
        refined=S.apply_reviewed_cap_profile(raw,context,PROFILE)
        before=next(r for r in raw['components']if r['component_id']=='earbud_b')
        after=next(r for r in refined['components']if r['component_id']=='earbud_b')
        self.assertEqual(after['parent_from_child'],before['parent_from_child'])
        self.assertEqual(after['source_quaternion_xyzw'],before['quaternion_xyzw'])
        self.assertNotEqual(after['quaternion_xyzw'],before['quaternion_xyzw'])
        self.assertEqual(after['orientation_hypothesis']['semantic_confidence'],.78)
        self.assertFalse(after['orientation_hypothesis']['physical_orientation_verified'])
        mutations=[lambda p:p['scene_ref'].update(sha256='0'*64),
                   lambda p:p['evidence_program_ref'].update(sha256='0'*64),
                   lambda p:p['components'][0]['native_asset_ref'].update(sha256='0'*64),
                   lambda p:p['components'][1].update(source_sample_index=25),
                   lambda p:p['components'][1]['child_source_mask_ref'].update(sha256='0'*64),
                   lambda p:p.update(physical_orientation_verified=True),
                   lambda p:p['source_visible_rim_protrusion'].update(bounded_visible_height_fraction=.4)]
        for mutate in mutations:
            profile=copy.deepcopy(PROFILE);mutate(profile)
            with self.assertRaises(ValueError):S.apply_reviewed_cap_profile(raw,context,profile)
        self.assertNotIn('orientation_hypothesis',before,'The original source observation stays immutable')

    def test_selected_laptop_camera_and_phone_use_their_own_tasks_and_measured_parent_table_height(self):
        from robot_task_geometry import native_bounds
        catalog = CuratedTwinCatalog(R.ROOT)
        profiles = [
            ('flow-capture-laptop-20261103-v1--flow-procedure-1', 'Open the laptop_lid, then close it.', 'open_close', []),
            ('flow-meta-video-camera-20261030-v1--flow-procedure-1', 'Move batteries to the tray.', 'pick_place', ['batteries']),
            ('flow-capture-phone-20261103-v1--flow-procedure-1', 'Move cable_connector to the tray.', 'pick_place', ['cable_connector']),
        ]
        for twin_id, instruction, operation, component_ids in profiles:
            with self.subTest(twin_id=twin_id):
                plan = R.plan_robot_task(twin_id, instruction)
                self.assertEqual(plan['status'], 'ready', plan['missing_requirements'])
                self.assertTrue(R.validate_robot_task(plan)['valid'])
                self.assertEqual(plan['goal']['operation'], operation)
                self.assertEqual(plan['goal']['component_ids'], component_ids)
                self.assertFalse(plan['claims']['contact_or_seating_verified'])
                parent = next(o for o in plan['objects'] if o['role'] == 'assembly_base')
                bounds = native_bounds(parent['asset_ref'], catalog)
                span = max(bounds[1][i]-bounds[0][i] for i in range(3))
                expected = parent['display_max_dimension'] * (bounds[1][1]-bounds[0][1]) / (2*span)
                self.assertAlmostEqual(parent['initial_transform']['position'][1], expected)

    def test_source_oriented_staging_retains_exact_scene_masks_and_rejects_provenance_or_rotation_edits(self):
        plan = R.plan_robot_task('flow-sam3d-default-e2e-20261102-v1--flow-procedure-1')
        staging = plan['staging_reference']
        self.assertEqual(staging['method'],'source_relative_component_display_v1')
        self.assertEqual({row['component_id']:row['source_sample_index']for row in staging['components']},{'earbud_a':77,'earbud_b':24})
        self.assertFalse(staging['physical_contact_verified'])
        self.assertFalse(staging['visual_rest_constraints']['actual_cavity_surface_or_lid_clearance_verified'])
        refs = R.task_bindings(plan)
        self.assertIn(staging['scene_ref'],refs)
        for row in staging['components']:
            self.assertIn(row['parent_source_mask_ref'],refs);self.assertIn(row['child_source_mask_ref'],refs)
            obj=next(o for o in plan['objects']if o['id']==row['component_id'])
            self.assertGreater(obj['display_max_dimension'],.095)
            self.assertNotEqual(obj['initial_transform']['quaternion_xyzw'],[0,0,0,1])
            self.assertEqual(obj['initial_transform']['quaternion_xyzw'],row['quaternion_xyzw'])
            self.assertAlmostEqual(row['display_max_dimension']/row['raw_source_display_max_dimension'],staging['visual_rest_constraints']['shared_child_scale_multiplier'])
        for kind in ['reference','quaternion','scale']:
            changed=copy.deepcopy(plan)
            if kind=='reference':changed['staging_reference']['scene_ref']['sha256']='0'*64
            elif kind=='quaternion':changed['objects'][2]['initial_transform']['quaternion_xyzw']=[0,0,0,1]
            else:changed['objects'][2]['initial_transform']['uniform_scale']=.9
            self.assertFalse(R.validate_robot_task(changed)['valid'],kind)

    def test_real_curated_earbud_plan_is_ready_and_all_four_native_refs_are_approved(self):
        twin_id = 'flow-sam3d-default-e2e-20261102-v1--flow-procedure-1'
        plan = R.plan_robot_task(twin_id, 'Place the case on the table, open the lid, take out both earbuds and place them on the tray, then close it.')
        self.assertEqual(plan['status'], 'ready');self.assertTrue(R.validate_robot_task(plan)['valid'])
        self.assertEqual({o['component_id'] for o in plan['objects']}, {'case_base', 'case_lid', 'earbud_a', 'earbud_b'})
        catalog = CuratedTwinCatalog(R.ROOT)
        for reference in R.task_bindings(plan):catalog._checked(reference)
        self.assertEqual(plan['goal']['component_ids'], ['earbud_a', 'earbud_b'])

    def test_real_coin_pick_place_starts_away_from_its_tray_destination(self):
        plan = R.plan_robot_task('flow-rigid-parts-wallet-20261031-v1--flow-procedure-1', 'Move gold_coin to the tray.')
        self.assertEqual(plan['status'], 'ready');self.assertTrue(R.validate_robot_task(plan)['valid'])
        coin = next(o for o in plan['objects'] if o['id'] == 'gold_coin')
        self.assertLess(coin['initial_transform']['position'][2], 0)
        self.assertEqual(plan['goal']['operation'], 'pick_place')

    @unittest.skipUnless(shutil.which('node'), 'Node is needed for browser cue parity')
    def test_python_and_browser_source_chain_and_three_phase_cue_match(self):
        from robot_task_geometry import cue_state
        plan = R.plan_robot_task('flow-sam3d-default-e2e-20261102-v1--flow-procedure-1')
        step = next(a for a in plan['actions'] if a['operation'] == 'set_joint')
        joint = plan['joints'][0];parent = next(o for o in plan['objects'] if o['id'] == joint['parent_id'])
        # Non-identity authored transforms expose quaternion/scale/order bugs;
        # the original native geometry and estimated joint frames stay exact.
        parent['initial_transform']['quaternion_xyzw'] = [0, math.sin(.15), 0, math.cos(.15)]
        parent['initial_transform']['uniform_scale'] = .8
        positions = [0, .1, .249, .25, .5, .799, .8, .9, 1]
        payload = {'plan': plan, 'step': step, 'position': [-.2, .15, .02],
                   'tool': [-.2, .5, -.05], 'progress': positions}
        script = r'''globalThis.AlignedSceneContract=require('./showcase/aligned-scene-contract.js');
const C=require('./showcase/robot-controller.js'),P=require('./showcase/robot-task-plan.js'),v=JSON.parse(process.argv[1]);
const state={tool:v.tool,gripper:0,objects:{[v.step.object_id]:{position:v.position}}};
process.stdout.write(JSON.stringify(v.progress.map(t=>C.jointCue(v.step,0,1.8,t,state,(s,q,b)=>P.jointToolPosition(v.plan,s,q,b)))));'''
        child = subprocess.run(['node', '-e', script, json.dumps(payload)], cwd=R.ROOT,
                               capture_output=True, text=True, timeout=10, check=True)
        for progress, js in zip(positions, json.loads(child.stdout)):
            py = cue_state(step['actuation'], joint, parent, payload['position'], payload['tool'], 0, 1.8, progress)
            self.assertAlmostEqual(py['angle_rad'], js['q'], places=10)
            self.assertAlmostEqual(py['gripper'], js['gripper'], places=10)
            for a, b in zip(py['position'], js['tool']):self.assertAlmostEqual(a, b, places=10)


if __name__ == '__main__':unittest.main()
