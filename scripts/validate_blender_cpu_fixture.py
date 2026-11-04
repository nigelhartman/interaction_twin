"""Blender-only, fixed synthetic free-fall/contact test. No captured-object data.

Run with already-installed Blender --background --factory-startup
--disable-autoexec --python-exit-code1 --python THIS -- --output FRESH.json.
The synthetic dimensions/mass/material are TEST FIXTURE definitions and must
never be transferred into the source-backed reconstruction contract.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time


def main():
    import bpy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.output.exists():
        raise ValueError('Synthetic fixture receipt must be fresh')
    if not bpy.app.build_options.bullet:
        raise RuntimeError('Installed Blender lacks Bullet')
    start = time.monotonic()
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.0
    scene.gravity = (0, 0, -9.81)
    scene.use_gravity = True
    scene.render.fps = 240
    scene.render.fps_base = 1.0
    scene.frame_start = 1
    scene.frame_end = 481
    # These are analytic fixtures, not models or assistant reconstruction.
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, -.1))
    floor = bpy.context.object
    floor.name = 'SYNTHETIC_TEST_FLOOR'
    floor.dimensions = (10, 10, .2)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.rigidbody.object_add()
    floor.rigid_body.type = 'PASSIVE'
    floor.rigid_body.collision_shape = 'BOX'
    floor.rigid_body.use_margin = False
    floor.rigid_body.friction = .5
    floor.rigid_body.restitution = 0
    bpy.ops.mesh.primitive_cube_add(size=.2, location=(0, 0, 2))
    cube = bpy.context.object
    cube.name = 'SYNTHETIC_TEST_BOX'
    bpy.ops.rigidbody.object_add()
    body = cube.rigid_body
    body.type = 'ACTIVE'
    body.mass = 1.0
    body.collision_shape = 'BOX'
    body.use_margin = False
    body.friction = .5
    body.restitution = 0
    body.linear_damping = 0
    body.angular_damping = 0
    body.use_deactivation = False
    world = scene.rigidbody_world
    world.time_scale = 1
    world.substeps_per_frame = 10
    world.solver_iterations = 20
    world.point_cache.frame_start = 1
    world.point_cache.frame_end = 481
    records = []
    for frame in range(1, 482):
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        matrix = cube.evaluated_get(depsgraph).matrix_world.copy()
        loc = matrix.translation
        records.append({'frame': frame, 't_s': (frame - 1) / 240,
                        'position_m': list(loc), 'rotation_quaternion_wxyz': list(matrix.to_quaternion()),
                        'lowest_box_corner_z_m': loc.z - .1 * sum(abs(matrix[2][axis]) for axis in range(3))})
    samples = [records[60], records[120]]
    freefall = []
    for sample in samples:
        expected = 2 - .5 * 9.81 * sample['t_s'] ** 2
        freefall.append({'t_s': sample['t_s'], 'measured_z_m': sample['position_m'][2],
                         'analytic_z_m': expected, 'absolute_error_m': abs(sample['position_m'][2] - expected)})
    last = records[-1]['position_m']
    minimum_bottom = min(row['lowest_box_corner_z_m'] for row in records)
    drift = math.hypot(last[0], last[1])
    # Frozen tolerances are numerical fixture checks, not measurement accuracy.
    checks = {'freefall_error_below_2cm': all(row['absolute_error_m'] < .02 for row in freefall),
              'final_rest_height_within_5mm': abs(last[2] - .1) < .005,
              'floor_penetration_below_5mm': minimum_bottom > -.005,
              'horizontal_drift_below_5mm': drift < .005,
              'all_finite': all(math.isfinite(x) for row in records for x in row['position_m'])}
    result = {'format': 'blender-bullet-synthetic-software-validation', 'version': 1,
              'status': 'passed' if all(checks.values()) else 'failed',
              'engine': {'name': 'Blender Bullet rigid bodies', 'compute': 'CPU',
                         'blender_version': bpy.app.version_string, 'build_hash': bpy.app.build_hash.decode()},
              'fixture_only': True, 'real_reconstructed_assets_used': False,
              'real_object_dynamics_validated': False, 'simulation_ready': False,
              'physical_accuracy_verified': False,
              'fixture_definition': {'box_side_m': .2, 'box_initial_center_m': [0, 0, 2],
                                     'box_mass_kg': 1, 'floor_top_z_m': 0,
                                     'friction': .5, 'restitution': 0, 'gravity_m_s2': [0, 0, -9.81],
                                     'linear_damping': 0, 'angular_damping': 0,
                                     'fps': 240, 'simulation_duration_s': 2,
                                     'substeps_per_frame': 10, 'solver_iterations': 20},
              'freefall_checks': freefall, 'minimum_box_bottom_z_m': minimum_bottom,
              'final_position_m': last, 'horizontal_drift_m': drift, 'checks': checks,
              'samples': records, 'wall_elapsed_s': time.monotonic() - start,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limitations': ['Only proves this installed solver can execute an analytic synthetic fixture.',
                              'No real object dimensions, mass, contact parameters, joints, grasps or insertion were validated.',
                              'No renderer, GPU model, external assets, robot or network was used.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'checks': checks,
                      'output': str(args.output), 'wall_elapsed_s': result['wall_elapsed_s']}))
    if result['status'] != 'passed':
        raise RuntimeError('Frozen synthetic fixture checks failed; preserve receipt')


if __name__ == '__main__':
    main()
