"""Pure Node checks for the saved assembly UI's source/claim/count boundary."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = (ROOT / 'workspace/physics-assembly.js').as_uri()


def fixture():
    source, program, protocol = '1' * 64, '2' * 64, '3' * 64
    condition, spread, excitation = 125.76532151321784, .1477395902714905, .009141777341320377
    physics = {'case_id': 'synthetic', 'interaction': {'path': 'data/synthetic/interaction.json', 'sha256': program},
               'source_video': {'sha256': source},
               'components': [{'id': name, 'visual_assets': [{'sha256': h * 64}]} for name, h in [('case_base', 'a'), ('case_lid', 'b')]]}
    index = {'format': 'interaction-twin-assembly-feasibility-index', 'version': 2, 'read_only': True,
             'status': 'unsupported_common_scale', 'case_id': 'synthetic', 'source_video_sha256': source,
             'interaction_path': physics['interaction']['path'], 'interaction_sha256': program,
             'physical_scale_known': False, 'physical_assembly_established': False, 'joint_axis_established': False,
             'contact_or_insertion_established': False, 'simulation_ready': False, 'automatic_promotion_allowed': False,
             'v1_excitation_criterion_untrusted_due_to_mixed_native_units': True,
             'supersedes_criterion_only': {'old_results_preserved': True},
             'artifacts': {'corrected_protocol': {'sha256': protocol}, 'corrected_compatibility': {'path': 'data/synthetic/corrected.json'}},
             'support': {'paired_simultaneous_nonreference_supported': 11, 'paired_simultaneous_nonreference_planned': 23,
                         'sparse_simultaneous_nonreference_supported': 2, 'sparse_simultaneous_nonreference_planned': 2,
                         'base_scope_rejected_withheld_frames': 12, 'comparison_is_matched_input_accuracy_benchmark': False},
             'scale_diagnostics': {'accepted_conditional_hypotheses': 0, 'common_frame_transform_a_from_b': None,
                                   'reserved_rows_used_to_fit': False, 'diagnostic_ratios_are_not_accepted_scales': True,
                                   'paired': {'training_timestamps': 12, 'normalized_condition': condition,
                                              'translation_excitation_over_anchor_depth': excitation,
                                              'leave_one_training_out_relative_spread': spread},
                                   'gate_condition_max': 100, 'gate_leave_one_training_out_relative_spread_max': .1,
                                   'gate_corrected_translation_excitation_min': .01,
                                   'corrected_policy': {'version': 'synthetic_unit_invariant'}}}
    raw = {'format': 'co-rigid-scale-compatibility-report', 'version': 2, 'protocol_sha256': protocol,
           'accepted_conditional_hypotheses': 0, 'physical_assembly_established': False, 'metric_scale_known': False,
           'policy': {'version': 'synthetic_unit_invariant', 'excitation_is_necessary_not_sufficient': True},
           'pairs': [{'status': 'unsupported', 'component_a': 'case_base', 'component_b': 'case_lid',
                      'mesh_a_sha256': 'a' * 64, 'mesh_b_sha256': 'b' * 64,
                      'interaction_sha256': program, 'source_video_sha256': source,
                      'training_count': 12, 'common_frame_transform_a_from_b': None,
                      'physical_assembly_established': False, 'physical_scale_known': False,
                      'joint_or_contact_established': False, 'reserved_used_to_fit': False,
                      'geometry_or_saved_poses_changed': False,
                      'diagnostic_fit': {'column_normalized_condition': condition,
                                         'translation_excitation_over_anchor_depth': excitation,
                                         'leave_one_training_out_relative_spread': spread,
                                         'fit_status': 'rejected_diagnostic_only'}}]}
    return {'index': index, 'raw': raw, 'physics': physics}


@unittest.skipUnless(shutil.which('node'), 'Node is required for the pure saved-UI contract checks')
class PhysicsAssemblyUITests(unittest.TestCase):
    def node(self, value, expression):
        script = f"import {{normalizeAssembly,artifactUrl}} from {json.dumps(MODULE)};" + \
                 "let text='';for await(const chunk of process.stdin)text+=chunk;const d=JSON.parse(text);" + expression
        return subprocess.run(['node', '--input-type=module', '-e', script], input=json.dumps(value),
                              text=True, capture_output=True, timeout=10)

    def test_actual_percentage_units_counts_and_rejected_transform(self):
        result = self.node(fixture(), "const m=normalizeAssembly(d.index,d.raw,d.physics);console.log(JSON.stringify(m));")
        self.assertEqual(result.returncode, 0, result.stderr)
        model = json.loads(result.stdout)
        self.assertEqual(model['counts'], {'dense': 11, 'denseTotal': 23, 'sparse': 2, 'sparseTotal': 2, 'withheld': 12})
        self.assertEqual(model['accepted'], 0)
        self.assertEqual(f"{model['excitation'] * 100:.3f}%", '0.914%')
        self.assertEqual(f"{model['spread'] * 100:.2f}%", '14.77%')

    def test_superseded_policy_false_accuracy_and_physical_promotions_rejected(self):
        for change in [lambda d: d['index'].update(version=1),
                       lambda d: d['index'].update(simulation_ready=True),
                       lambda d: d['index'].update(v1_excitation_criterion_untrusted_due_to_mixed_native_units=False),
                       lambda d: d['index']['support'].update(comparison_is_matched_input_accuracy_benchmark=True),
                       lambda d: d['raw']['policy'].update(version='old_mixed_units'),
                       lambda d: d['raw']['pairs'][0].update(status='supported'),
                       lambda d: d['raw']['pairs'][0].update(common_frame_transform_a_from_b=[[1]]),
                       lambda d: d['raw']['pairs'][0].update(reserved_used_to_fit=True)]:
            value = fixture(); change(value)
            result = self.node(value, 'normalizeAssembly(d.index,d.raw,d.physics);')
            self.assertNotEqual(result.returncode, 0)

    def test_source_program_and_exact_native_assets_cannot_be_swapped(self):
        for change in [lambda d: d['index'].update(case_id='another'),
                       lambda d: d['index'].update(interaction_sha256='9' * 64),
                       lambda d: d['raw']['pairs'][0].update(source_video_sha256='9' * 64),
                       lambda d: d['raw']['pairs'][0].update(mesh_a_sha256='9' * 64)]:
            value = fixture(); change(value)
            result = self.node(value, 'normalizeAssembly(d.index,d.raw,d.physics);')
            self.assertNotEqual(result.returncode, 0)

    def test_counts_and_diagnostic_facts_must_match_raw_and_remain_rejected(self):
        for change in [lambda d: d['index']['support'].update(paired_simultaneous_nonreference_supported=24),
                       lambda d: d['index']['support'].update(paired_simultaneous_nonreference_supported=True),
                       lambda d: d['raw']['pairs'][0]['diagnostic_fit'].update(translation_excitation_over_anchor_depth=.1),
                       lambda d: d['index']['scale_diagnostics'].update(gate_condition_max=200),
                       lambda d: d['raw']['pairs'].append(copy.deepcopy(d['raw']['pairs'][0]))]:
            value = fixture(); change(value)
            result = self.node(value, 'normalizeAssembly(d.index,d.raw,d.physics);')
            self.assertNotEqual(result.returncode, 0)

    def test_artifact_links_cannot_escape_data_or_use_hidden_paths(self):
        for path in ['/etc/passwd', '../private', 'data/../private', 'data/.hidden/file', 'data/a?x=1', 'https://example.test/a']:
            result = self.node({'path': path}, 'artifactUrl(d.path);')
            self.assertNotEqual(result.returncode, 0)

    def test_current_frozen_real_inputs_pass_without_mutation_when_available(self):
        index_path = ROOT / 'data/assembly-feasibility-20261101/v2/index.json'
        report_path = ROOT / 'data/physics-validation-20261101/v2/report.json'
        if not index_path.is_file() or not report_path.is_file():
            self.skipTest('Private saved examples are absent in source-only distribution')
        index = json.loads(index_path.read_text())
        value = {'index': index, 'raw': json.loads((ROOT / index['artifacts']['corrected_compatibility']['path']).read_text()),
                 'physics': json.loads(report_path.read_text())['source_case']['contract']}
        result = self.node(value, 'const before=JSON.stringify(d);const m=normalizeAssembly(d.index,d.raw,d.physics);if(before!==JSON.stringify(d))throw Error("mutated");console.log(m.accepted);')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), '0')


if __name__ == '__main__':
    unittest.main()
