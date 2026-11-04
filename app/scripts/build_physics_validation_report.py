"""Build one source-bound, read-only physics-preparation report. No engine run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import physics_validation as P


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interaction', required=True)
    parser.add_argument('--version-id', action='append', default=[])
    parser.add_argument('--environment', required=True)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    if (not args.output.startswith('data/') or output.exists() or
            output.resolve() != output or any(char in args.output for char in ['\\', ':', '?', '#']) or
            any(not part or part.startswith('.') for part in args.output.split('/'))):
        parser.error('Use a fresh canonical data report path')
    program = P.data_path(ROOT, args.interaction, maximum=8 * 1024 * 1024)
    environment_path = P.data_path(ROOT, args.environment, maximum=1024 * 1024)
    fixture_path = P.data_path(ROOT, args.fixture, maximum=1024 * 1024)
    environment, fixture = P.read_json(environment_path), P.read_json(fixture_path)
    if (environment.get('format') != 'interaction-twin-local-physics-environment' or
            fixture.get('format') != 'blender-bullet-synthetic-software-validation' or
            fixture.get('real_reconstructed_assets_used') is not False or fixture.get('fixture_only') is not True):
        raise ValueError('Expected separate software-only availability/fixture receipts')
    if environment['blender_cpu_execution']['receipt_sha256'] != P.sha(fixture_path):
        raise ValueError('Environment and fixture receipt identities differ')
    floor_recheck = P.synthetic_box_floor_recheck(fixture)
    checks = dict(fixture['checks'])
    checks['floor_penetration_below_5mm'] = floor_recheck['floor_penetration_below_5mm']
    report = {'format': 'interaction-twin-physics-validation-report', 'version': 1,
              'utc': datetime.now(timezone.utc).isoformat(), 'status': 'physical_inputs_missing',
              'source_case': P.audit_program(ROOT, {'path': args.interaction, 'sha256': P.sha(program)}),
              'reconstruction_versions': [P.audit_reconstruction_version(ROOT, item) for item in args.version_id],
              'backend': {'name': 'installed_blender_bullet_cpu',
                          'environment': {'path': args.environment, 'sha256': P.sha(environment_path)},
                          'software_fixture': {'path': args.fixture, 'sha256': P.sha(fixture_path)},
                          'software_execution_verified': environment['blender_cpu_execution']['execution_verified_by_saved_samples'],
                          'fixture_status': 'passed' if all(checks.values()) else 'failed',
                          'original_fixture_status': fixture['status'], 'fixture_checks': checks,
                          'rotation_aware_box_floor_recheck': floor_recheck,
                          'all_fixture_checks_passed': all(checks.values()),
                          'real_object_dynamics_validated': False},
              'claim_boundaries': {'actual_reconstructed_object_dynamics_run': False,
                                   'common_assembly_recovered': False, 'hinge_axis_recovered': False,
                                   'contact_or_insertion_validated': False, 'robot_simulation_run': False,
                                   'metric_shape_accuracy_measured': False, 'physical_accuracy_verified': False,
                                   'simulation_ready': False, 'new_api_calls': 0, 'model_inference': False,
                                   'installs_or_downloads': False, 'original_assets_modified': False},
              'recommended_next_action': {'title': 'Capture one rigid assembly with calibrated camera and measured dimensions',
                    'benefit': 'Establish metric scale and common frames before testing cavities, joints or contacts.',
                    'additional_capture': True, 'api_required': False, 'gpu_required': False},
              'priorities': [
                  {'id': 'metric_common_frame', 'title': 'Calibrate scale, intrinsics and shared component frames',
                   'benefit': 'Makes clearance and relative placement meaningful in meters.',
                   'additional_capture_or_measurement': True, 'api_required': False, 'gpu_required': False},
                  {'id': 'collision_geometry', 'title': 'Review cavity-preserving collision geometry',
                   'benefit': 'Avoids sealed convex-hull cavities and unobserved wall-thickness assumptions.',
                   'additional_capture_or_measurement': True, 'api_required': False, 'gpu_required': False},
                  {'id': 'physical_parameters', 'title': 'Measure mass and document COM, inertia, contact and joint uncertainty',
                   'benefit': 'Permits an explicit, reviewable passive dynamics trial without hidden defaults.',
                   'additional_capture_or_measurement': True, 'api_required': False, 'gpu_required': False},
                  {'id': 'frozen_simulator_trial', 'title': 'Investigate the synthetic lateral-drift failure, then freeze a simple real-object trial',
                   'benefit': 'Separates solver behavior from reconstruction error before robot grasp/insertion.',
                   'additional_capture_or_measurement': False, 'api_required': False, 'gpu_required': False}],
              'code': {'physics_validation_sha256': P.LOADED_SOURCE_SHA256,
                       'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              'limitations': ['Installed software execution is not physical reconstruction validation.',
                              'A watertight mesh can still have an incorrect cavity, shape or wall thickness.',
                              'Observed patches and inferred completions remain separate diagnostic versions.',
                              'The illustrative browser robot remains illustrative; no real robot, controller or training environment ran.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': args.output, 'status': report['status'],
                      'components': len(report['source_case']['contract']['components']),
                      'reconstruction_versions': len(report['reconstruction_versions']),
                      'software_fixture_status': fixture['status']}))


if __name__ == '__main__':
    main()
