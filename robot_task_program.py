"""Read-only authored robot rehearsal plans bound to exact curated twin assets.

The kit is a mathematical five-DOF display arm, not a physical URDF.  Planning
uses saved semantic relationships, never an API, shell, inference or file write.
Authored staging/grasp points remain distinct from recovered twin articulation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re

from curated_twin_catalog import CuratedTwinCatalog
from library_query import backend_key

ROOT = Path(__file__).resolve().parent
KIT_ID = 'generic5dof_tabletop_v1'
FORMAT = 'interaction-twin-robot-task-program'
ID = r'^[A-Za-z][A-Za-z0-9_-]{0,80}$'
OPERATIONS = ('remove_components', 'return_components', 'open_close', 'pick_place')
CLAIMS = {'authored_rehearsal': True, 'observed_robot_demonstration': False,
          'physical_accuracy_verified': False, 'simulation_ready': False,
          'physical_robot_execution_available': False, 'contact_or_seating_verified': False,
          'new_api_calls': 0, 'generated_code_executed': False}
ROBOT = {'id': 'generic_5dof_arm', 'base': [-.95, 0, -.55], 'shoulder_height': .34,
         'link_lengths': [.68, .62], 'tool_length': .14,
         'home': [0, 1.05, -1.75, -.8707963267948966, 0],
         'joint_names': ['base_yaw', 'shoulder', 'elbow', 'wrist_pitch', 'wrist_roll'],
         'joint_limits': [[-2.96706, 2.96706], [-.523599, 2.268928], [-2.617994, 2.617994],
                          [-2.96706, 2.96706], [-math.pi, math.pi]],
         'gripper_open': .12, 'gripper_closed': .028}


def _clone(value):
    return copy.deepcopy(value)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _exact_ref(value):
    return {key: _clone(value[key]) for key in ['path', 'sha256', 'bytes']}


def get_robot_kit(kit_id=KIT_ID, *, root=ROOT):
    if kit_id != KIT_ID:
        raise ValueError('Unknown authored robot kit')
    return {'format': 'interaction-twin-robot-kit', 'version': 1, 'id': KIT_ID,
            'title': 'Five-DOF tabletop rehearsal arm', 'robot': _clone(ROBOT),
            'units': {'length': 'authored_display_unit', 'angle': 'radian', 'time': 'second',
                      'metric_scale_known': False},
            'coordinate_system': {'up': 'Y', 'handedness': 'right',
                'base_yaw_radial': '[cos(q0), 0, sin(q0)]',
                'tool_orientation': 'downward pitch; wrist roll independently bounded'},
            'joint_types': ['revolute'] * 5,
            'kinematic_convention': 'two serial pitch links; wrist pitch sets downward tool; roll about tool axis',
            'table': {'top_y': 0, 'width': 2.8, 'depth': 1.8},
            'tray': {'position': [-.35, .08, .35], 'width': .35, 'depth': .25},
            'solver': {'id': 'analytic_2link_downward_tool_with_bounded_5dof_joints',
                       'maximum_position_residual': .002, 'path_check_intervals': 20},
            'joint_actuation': {'kind': 'authored_native_bbox_surface_cue_v1',
                'anchor_rule': 'bbox_face_center_farthest_from_joint_axis', 'tool_tracking': 'position_only',
                'approach_fraction': .25, 'release_fraction': .8, 'sample_count': 41,
                'physical_grasp_or_orientation_verified': False},
            'limits': {'maximum_objects': 16, 'maximum_actions': 160, 'maximum_duration_s': 300,
                       'attachment_tolerance': .18, 'maximum_json_bytes': 262144},
            'capabilities': ['authored_pick_place', 'estimated_revolute_joint_rehearsal',
                             'deterministic_seek', 'joint_limited_inverse_kinematics'],
            'claims': _clone(CLAIMS),
            'limitations': ['Display units are not calibrated metres.',
                'No physical collision, force, grasp, insertion or seating is validated.',
                'Not a URDF, learned policy, physics simulator or robot driver.']}


def list_robot_kits(*, root=ROOT):
    kit = get_robot_kit(root=root)
    return {'format': 'interaction-twin-robot-kit-index', 'version': 1,
            'kits': [{'id': kit['id'], 'title': kit['title'], 'degrees_of_freedom': 5,
                      'parallel_gripper': True, 'kit_sha256': _digest(kit), 'claims': _clone(CLAIMS)}]}


def _object(properties, required=None):
    return {'type': 'object', 'additionalProperties': False, 'properties': properties,
            'required': list(properties) if required is None else required}


def task_schema():
    number = {'type': 'number'}
    vector = {'type': 'array', 'items': number, 'minItems': 3, 'maxItems': 3}
    ident = {'type': 'string', 'pattern': ID}
    string = {'type': 'string', 'maxLength': 500}
    reference = _object({'path': {'type': 'string', 'maxLength': 600},
                         'sha256': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'},
                         'bytes': {'type': 'integer', 'minimum': 1, 'maximum': 32*1024*1024}})
    nullable_ref = {'anyOf': [reference, {'type': 'null'}]}
    goal = _object({'operation': {'enum': list(OPERATIONS)},
                    'component_ids': {'type': 'array', 'items': ident, 'uniqueItems': True, 'maxItems': 16},
                    'open_container': {'type': 'boolean'}, 'close_container': {'type': 'boolean'},
                    'destination': {'enum': ['table', 'tray']}})
    transform = _object({'position': vector,
        'quaternion_xyzw': {'type': 'array', 'items': number, 'minItems': 4, 'maxItems': 4},
        'uniform_scale': {'type': 'number', 'minimum': .001, 'maximum': 10}})
    obj = _object({'id': ident, 'component_id': ident, 'label': string,
        'role': {'enum': ['assembly_base', 'articulated_child', 'removable']}, 'asset_ref': reference,
        'asset_scope': {'enum': ['saved_program_asset', 'curated_native_candidate', 'curated_geometric_derivative']},
        'initial_transform': transform, 'display_max_dimension': {'type': 'number', 'minimum': .02, 'maximum': 1},
        'placement_origin': {'const': 'authored_staging_not_recovered_video_pose'}})
    actuation = _object({'kind': {'const': 'authored_native_bbox_surface_cue_v1'}, 'child_id': ident,
        'asset_sha256': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'}, 'anchor_native': vector,
        'anchor_rule': {'const': 'bbox_face_center_farthest_from_joint_axis'},
        'parent_native_bounds': {'type': 'array', 'items': vector, 'minItems': 2, 'maxItems': 2},
        'tool_tracking': {'const': 'position_only'}, 'approach_fraction': {'const': .25},
        'release_fraction': {'const': .8}, 'sample_count': {'const': 41}})
    action = _object({'id': ident, 'operation': {'enum': ['move_to', 'gripper', 'attach', 'detach', 'set_joint', 'wait']},
        'title': string, 'duration_s': {'type': 'number', 'minimum': .05, 'maximum': 30},
        'position': vector, 'object_id': ident, 'joint_id': ident, 'value': number, 'actuation': actuation},
        ['id', 'operation', 'title', 'duration_s'])
    joint = _object({'id': ident, 'parent_id': ident, 'child_id': ident, 'type': {'const': 'revolute'},
        'source_joint': {'type': 'object'}, 'initial_rad': number,
        'source': {'const': 'estimated_curated_descriptive_twin'}})
    attachment = _object({'parent_id': ident, 'child_id': ident, 'offset': vector,
        'type': {'const': 'authored_rigid_staging'}, 'releasable': {'const': True},
        'source_relationship_id': ident})
    issue = _object({'code': ident, 'message': string, 'component_ids': {'type': 'array', 'items': ident, 'maxItems': 16}})
    return _object({'format': {'const': FORMAT}, 'version': {'const': 1},
        'status': {'enum': ['ready', 'blocked']}, 'kit_id': {'const': KIT_ID},
        'kit_sha256': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'}, 'twin_id': {'type': 'string', 'minLength': 1, 'maxLength': 200},
        'source_binding': _object({'source_sha256': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'},
                                  'program_ref': reference, 'object_program_ref': nullable_ref}),
        'staging_reference': {'anyOf': [{'type': 'null'}, {'type': 'object', 'maxProperties': 12}]},
        'goal': goal, 'objects': {'type': 'array', 'items': obj, 'maxItems': 16},
        'joints': {'type': 'array', 'items': joint, 'maxItems': 16},
        'attachments': {'type': 'array', 'items': attachment, 'maxItems': 16},
        'actions': {'type': 'array', 'items': action, 'maxItems': 160},
        'missing_requirements': {'type': 'array', 'items': issue, 'maxItems': 32},
        'claims': _object({key: {'const': value} for key, value in CLAIMS.items()}),
        'limitations': {'type': 'array', 'items': string, 'maxItems': 16}})


def _catalog(root, catalog):
    root = Path(root).resolve()
    if catalog is None:
        return CuratedTwinCatalog(root)
    if not isinstance(catalog, CuratedTwinCatalog) or catalog.root != root:
        raise ValueError('Robot planning requires the current curated catalog')
    return catalog


JOINT_FIELDS = ('axis', 'parent_from_joint', 'joint_from_child_zero', 'parent_native_scale',
                'child_native_scale', 'limits_rad', 'limits_kind', 'zero_reference', 'axis_frame',
                'matrix_convention', 'evidence_ids', 'physical_accuracy_verified', 'simulation_ready')


def _context(twin_id, *, root, catalog):
    catalog = _catalog(root, catalog)
    record, _ = catalog._record(twin_id)
    _, doc, program_sha = catalog.load(twin_id)
    assets = {}
    for row in record.get('sam3d_reviews', []):
        cid = row.get('component_id')
        if cid not in {t['id'] for t in doc['tracklets']} or cid in assets:
            raise ValueError('Ambiguous curated robot component inventory')
        if row.get('source_sha256') != doc['source']['sha256'] or backend_key(row.get('backend', '')) != 'sam3d':
            raise ValueError('Robot asset belongs to another source or backend')
        ref = row['native_glb']; catalog._checked(ref, maximum=16*1024*1024)
        selected = any(a['track_id'] == cid and a['sha256'] == ref['sha256'] for a in doc['assets'])
        assets[cid] = {'asset_ref': _exact_ref(ref), 'asset_scope': 'saved_program_asset' if selected else 'curated_native_candidate'}
    # Saved SAM assets remain eligible when no separate native review is pinned.
    for asset in doc['assets']:
        if asset['track_id'] not in assets and backend_key(asset.get('backend', '')) == 'sam3d':
            ref = catalog._assets().get(asset['path'])
            if ref is None or ref['sha256'] != asset['sha256']:
                raise ValueError('Saved robot geometry is not the curated asset revision')
            catalog._checked(ref, maximum=16*1024*1024)
            assets[asset['track_id']] = {'asset_ref': _exact_ref(ref), 'asset_scope': 'saved_program_asset'}
    kin = catalog.kinematics(twin_id)
    obj = kin.get('object_program')
    descriptive_ref = kin.get('artifact') if obj else None
    derived = None
    if record.get('derived_assembly'):
        # An explicit curated pointer is required. Never discover a convenient
        # newer derivative merely because a research folder exists on disk.
        derived = catalog.derived_assembly(twin_id)
        value = derived['document']
        if (value['base_object_program_ref'] != _exact_ref(descriptive_ref) or
                value['evidence_program'] != _exact_ref(record['interaction']) or
                value['source']['sha256'] != doc['source']['sha256']):
            raise ValueError('Declared geometric derivative belongs to a different curated native program')
        for component in value['component_assets']:
            cid = component['component_id']
            if cid not in assets or any(assets[cid]['asset_ref'][key] != component['original_native_glb'][key] for key in ['sha256','bytes']):
                raise ValueError('Declared derivative changes the exact source component inventory')
            # Curated native reviews can store an immutable duplicate path.
            # Both copies still require the same exact content hash and size.
            catalog._checked(component['original_native_glb'], maximum=16*1024*1024)
            catalog._checked(component['native_glb'], maximum=16*1024*1024)
            changed = component['native_glb']['sha256'] != component['original_native_glb']['sha256']
            assets[cid] = {'asset_ref': _exact_ref(component['native_glb']),
                           'asset_scope': 'curated_geometric_derivative' if changed else assets[cid]['asset_scope']}
        obj, descriptive_ref = value, derived['artifact']
    joints = []
    if obj:
        source_assets = {a['component_id']: a['native_glb']['sha256'] for a in obj['component_assets']}
        for j in obj['joints']:
            if obj.get('format')=='interaction-twin-declared-geometry-derivative':
                # The generic reader verifies exactly P * A * R(axis,q) * B.
                # Its compact projection omits the redundant frame label.
                convention='parent_native_pose * parent_from_joint * axis_rotation(q) * joint_from_child_zero'
                scaled_convention = 'parent_native_pose * parent_from_joint(inverse_parent_scale) * axis_rotation(q) * joint_from_child_zero(child_scale_once)'
                if not (j.get('matrix_convention','').startswith(convention) or j.get('matrix_convention') == scaled_convention):
                    raise ValueError('Unsupported compact joint matrix convention')
                j={**j,'axis_frame':'joint_frame_after_parent_from_joint'}
            if (j['type'] == 'revolute' and j['status'] == 'estimated' and
                all(p in assets and source_assets.get(p) == assets[p]['asset_ref']['sha256']
                    for p in [j['parent_id'], j['child_id']])):
                joints.append({'id': j['id'], 'parent_id': j['parent_id'], 'child_id': j['child_id'],
                    'type': 'revolute', 'source_joint': {k: _clone(j[k]) for k in JOINT_FIELDS},
                    'initial_rad': 0, 'source': 'estimated_curated_descriptive_twin'})
    removable = []
    if obj:
        removable = [{'id': r['id'], 'parent_id': r['parent_id'], 'child_id': r['child_id']}
                     for r in obj['removable_relationships'] if r['joint_type'] == 'removable']
    if not removable:
        removable = [{'id': r['id'], 'parent_id': r['object_track_id'], 'child_id': r['subject_track_id']}
                     for r in doc.get('relationships', []) if r['predicate'] == 'removable_from']
    from robot_task_geometry import actuation_descriptor
    actuations = {};actuation_errors = []
    for joint in joints:
        try:actuations[joint['id']] = actuation_descriptor(joint, assets, catalog)
        except (ValueError, KeyError, OSError):actuation_errors.append(joint['id'])
    return {'catalog': catalog, 'record': record, 'doc': doc, 'assets': assets, 'joints': joints,
            'actuations': actuations, 'actuation_errors': actuation_errors,
            'derived_assembly': derived,
            'removable': removable, 'labels': {t['id']: t['label'] for t in doc['tracklets']},
            'binding': {'source_sha256': doc['source']['sha256'], 'program_ref': _exact_ref(record['interaction']),
                        'object_program_ref': _exact_ref(descriptive_ref) if obj else None}}


def _issue(code, message, ids=()):
    return {'code': code, 'message': message, 'component_ids': list(ids)}


def _goal(context, instruction, goal):
    removable = [r['child_id'] for r in context['removable']]
    default = {'operation': 'remove_components', 'component_ids': removable,
               'open_container': True, 'close_container': True, 'destination': 'tray'}
    if instruction is not None and goal is not None:
        return default, [_issue('conflicting_goal_inputs', 'Supply either one structured goal or one local English instruction.')]
    if goal is not None:
        if not isinstance(goal, dict):
            return default, [_issue('unsupported_goal', 'A typed goal object is required.')]
        required = set(default)
        if set(goal) != required:
            return default, [_issue('unsupported_goal_fields', 'Use operation, component_ids, open_container, close_container and destination.')]
        try:
            import jsonschema
            jsonschema.validate(goal, task_schema()['properties']['goal'])
        except Exception:
            return default, [_issue('unsupported_goal', 'Goal fields or values are invalid.')]
        return _clone(goal), []
    if instruction is None:
        return default, []
    if not isinstance(instruction, str) or not 1 <= len(instruction) <= 2000:
        return default, [_issue('instruction_budget', 'Use one short English rehearsal instruction.')]
    text = instruction.lower().strip()
    if re.search(r'\b(?:usb(?:c)?|phones?|sockets?|outlets?|mains|connect(?:ing|ed)?|plug(?:ging|ged)?|unplug|charge|power|screw|drill|cut|exec|eval)\b|https?://', text):
        return default, [_issue('unsupported_capability', 'This local planner supports authored pick/place and an existing estimated hinge; it cannot connect or operate electrical equipment.')]
    vocabulary = set('a an the please case charging lid open close shut place put on onto in into out of from to and then both all two one each earbuds earbud earphones earphone headphones headphone headset components component parts part table tray pick up take remove return reinsert move it them back first second coin battery batteries wireless black'.split())
    vocabulary.update(cid.lower() for cid in context['labels'])
    for label in context['labels'].values():vocabulary.update(re.findall(r'[a-z0-9_]+', label.lower()))
    if any(token not in vocabulary for token in re.findall(r'[a-z0-9_]+', text)):
        return default, [_issue('unsupported_instruction_clause', 'The complete instruction must use the supported pick/place/open/close vocabulary; no unknown clause is executed.')]
    removes = bool(re.search(r'take\s+out|remove', text));returns = bool(re.search(r'put\s+back|return|reinsert', text))
    if removes and returns:
        return default, [_issue('unsupported_round_trip', 'Use separate remove and return plans; this version does not collapse a round trip into one direction.')]
    destination = 'tray' if 'tray' in text else 'table'
    if re.search(r'\b(?:on|onto|to|into)\s+(?:the\s+)?(?:floor|shelf|cabinet|sink|oven|wall)\b', text):
        return default, [_issue('unsupported_destination', 'Only the authored table or tray is available.')]
    chosen = [cid for cid in removable if cid.lower() in text]
    if not chosen and re.search(r'earbuds?|earphones?|headphones?|headset', text):
        chosen = [cid for cid in removable if re.search(r'earbud|headphone', context['labels'].get(cid, ''), re.I)]
        if re.search(r'\b(earbud|earphone|headphone)\b', text) and not re.search(r'\b(both|all|two|each)\b', text) and len(chosen) > 1:
            return default, [_issue('ambiguous_component', 'Specify both earbuds or an exact component ID.')]
    if not chosen and re.search(r'\b(coin|batter(?:y|ies))\b', text):
        chosen = [cid for cid in removable if re.search(r'coin|batter', context['labels'].get(cid, ''), re.I)]
    if not chosen and re.search(r'\b(all|both|each|components|parts)\b', text):
        chosen = removable
    opening = bool(re.search(r'\bopen\b', text));closing = bool(re.search(r'\b(close|shut)\b', text))
    if returns:
        operation = 'return_components'
    elif removes:
        operation = 'remove_components'
    elif re.search(r'pick|place|move', text) and chosen:
        operation = 'pick_place'
    elif opening or closing:
        operation = 'open_close';chosen = []
    else:
        return default, [_issue('unsupported_instruction', 'Try opening the case, removing both earbuds to the tray, then closing it.')]
    if operation != 'open_close' and not chosen:
        return default, [_issue('component_not_recognized', 'No selected removable component matches this curated twin.')]
    return {'operation': operation, 'component_ids': chosen, 'open_container': opening or operation in {'remove_components', 'return_components'},
            'close_container': closing, 'destination': destination}, []


def _forward(q, robot=ROBOT):
    yaw, a, b, w, _ = q
    tip = [robot['base'][0], robot['base'][1] + robot['shoulder_height'], robot['base'][2]]
    for length, pitch in zip([*robot['link_lengths'], robot['tool_length']], [a, a+b, a+b+w]):
        tip = [tip[0]+length*math.cos(pitch)*math.cos(yaw), tip[1]+length*math.sin(pitch),
               tip[2]+length*math.cos(pitch)*math.sin(yaw)]
    return tip


def inverse_kinematics(target):
    """Same bounded downward-tool analytic positional solver as the browser."""
    base = ROBOT['base'];yaw = math.atan2(target[2]-base[2], target[0]-base[0])
    radial = math.hypot(target[0]-base[0], target[2]-base[2])
    dy = target[1]+ROBOT['tool_length']-base[1]-ROBOT['shoulder_height'];l1, l2 = ROBOT['link_lengths']
    c = (radial*radial+dy*dy-l1*l1-l2*l2)/(2*l1*l2)
    candidates = []
    for sign in [-1, 1]:
        elbow = sign*math.acos(max(-1, min(1, c)))
        shoulder = math.atan2(dy, radial)-math.atan2(l2*math.sin(elbow), l1+l2*math.cos(elbow))
        q = [yaw, shoulder, elbow, -math.pi/2-shoulder-elbow, 0]
        q = [max(bounds[0], min(bounds[1], value)) for value, bounds in zip(q, ROBOT['joint_limits'])]
        residual = math.dist(_forward(q), target)
        candidates.append({'joints': q, 'residual': residual,
                           'reachable': -1-1e-9 <= c <= 1+1e-9 and residual < .002})
    return min(candidates, key=lambda row: row['residual'])


def _stage(context, goal):
    joints = _clone(context['joints'][:1]);joint = joints[0] if joints else None
    parent = joint['parent_id'] if joint else (context['removable'][0]['parent_id'] if context['removable'] else None)
    child_ids = [r['child_id'] for r in context['removable'] if r['parent_id'] == parent]
    ids = ([parent, joint['child_id']] if joint else ([parent] if parent else [])) + child_ids
    objects = [];attachments = []
    geometry = (context.get('derived_display_geometry') or _derived_display_geometry(context)) if context.get('derived_assembly') else None
    if 'source_staging_reference' not in context:
        from robot_source_staging_v1 import apply_reviewed_cap_profile, build, constrain_rest
        from robot_native_cap_profile_v1 import profile_for_source
        raw = build(context, parent, child_ids) if parent and child_ids and geometry else None
        if raw:raw = apply_reviewed_cap_profile(raw, context, profile_for_source(context['binding']['source_sha256']))
        context['source_staging_reference'] = constrain_rest(raw, geometry)
    source_rows = {row['component_id']: row for row in (context.get('source_staging_reference') or {}).get('components', [])}
    if geometry and source_rows:
        geometry = _clone(geometry)
        # Existing callers use this dictionary as the distance from the staged
        # native AABB center to the floor, which can be asymmetric after rotation.
        geometry['removable_half_heights'].update({cid: row['lower_support_extents'][1] for cid, row in source_rows.items()})
        geometry['removable_offsets'] = {cid: row['center_offset'] for cid, row in source_rows.items()}
    context['derived_display_geometry'] = geometry
    base_height = geometry['base_half_height'] if geometry else .13
    if not geometry and parent in context['assets']:
        from robot_task_geometry import native_bounds
        bounds = native_bounds(context['assets'][parent]['asset_ref'], context['catalog'])
        base_height = .38 * (bounds[1][1]-bounds[0][1]) / (2*max(bounds[1][i]-bounds[0][i] for i in range(3)))
    for cid in dict.fromkeys(ids):
        if cid not in context['assets']:
            continue
        role = 'assembly_base' if cid == parent else 'articulated_child' if joint and cid == joint['child_id'] else 'removable'
        offset = [-.07 if child_ids.index(cid) % 2 == 0 else .07, .09, .04*(child_ids.index(cid)//2)] if role == 'removable' else [0, 0, 0]
        source_row = source_rows.get(cid) if role == 'removable' else None
        if source_row:
            offset = source_row['center_offset'][:]
        elif geometry and role == 'removable':
            half_height = geometry['removable_half_heights'][cid]
            offset[1] = geometry['opening_rim_offset_y']-.004-half_height
            if offset[1]-half_height < geometry['body_bottom_offset_y']+.004:
                raise ValueError('Native bounds do not fit the declared container staging height')
        position = [-.5+offset[0], base_height+offset[1], -.2+offset[2]]
        if goal['operation'] == 'return_components' and role == 'removable':
            position = [-.35+offset[0], .08, .35+offset[2]]
            if geometry:
                position[1] = .009 + geometry['removable_half_heights'][cid]
        elif goal['operation'] == 'pick_place' and role == 'removable':
            position = [-.5+offset[0], .08, -.2+offset[2]]
            if geometry:
                position[1] = geometry['removable_half_heights'][cid]
        objects.append({'id': cid, 'component_id': cid, 'label': context['labels'][cid], 'role': role,
            **_clone(context['assets'][cid]), 'initial_transform': {'position': position, 'quaternion_xyzw': source_row['quaternion_xyzw'][:] if source_row else [0, 0, 0, 1], 'uniform_scale': 1},
            'display_max_dimension': source_row['display_max_dimension'] if source_row else .38 if role == 'assembly_base' else .3 if role == 'articulated_child' else .095,
            'placement_origin': 'authored_staging_not_recovered_video_pose'})
        if role == 'removable' and goal['operation'] not in {'return_components', 'pick_place'}:
            relation = next(r for r in context['removable'] if r['child_id'] == cid and r['parent_id'] == parent)
            attachments.append({'parent_id': parent, 'child_id': cid, 'offset': offset, 'type': 'authored_rigid_staging',
                                'releasable': True, 'source_relationship_id': relation['id']})
    return objects, joints, attachments


def _derived_display_geometry(context):
    """Measured native bounds for authored visual staging, never grasp/fit truth."""
    if not context['joints'] or not context['removable']:
        return None
    from portable_twin import decode_glb_bytes
    from robot_task_geometry import native_bounds
    catalog = context['catalog']; descriptor = context['derived_assembly']['document']
    report = json.loads(catalog._bytes(descriptor['refinement_report_ref'], maximum=6*1024*1024))
    if 'rim_support' not in report:
        parent_ref=descriptor.get('parent_derived_assembly_ref')
        if not parent_ref:return None
        parent_document=json.loads(catalog._bytes(parent_ref,maximum=6*1024*1024))
        report=json.loads(catalog._bytes(parent_document['refinement_report_ref'],maximum=6*1024*1024))
        if 'rim_support' not in report:return None
    joint = context['joints'][0]
    parent = joint['parent_id']
    bounds = {cid: native_bounds(row['asset_ref'], catalog) for cid, row in context['assets'].items()}
    # The structural decoder preserves native selected-scene primitive/vertex
    # order. Appearance/texture decoding is irrelevant to support measurements
    # and must not become a dependency of a read-only MCP robot plan.
    decoded = decode_glb_bytes(catalog._bytes(context['assets'][parent]['asset_ref'], maximum=16*1024*1024))
    vertices = [point for primitive in decoded['primitives'] for point in primitive['vertices']]
    span = max(bounds[parent][1][i]-bounds[parent][0][i] for i in range(3)); scale = .38/span
    ids = report['rim_support']['parent_vertex_ids']
    if not len(ids) >= 32 or any(type(i) is not int or not 0 <= i < len(vertices) for i in ids):
        raise ValueError('Declared container needs exact measured rim support')
    heights = sorted(vertices[i][1] for i in ids)
    rank = (len(heights)-1)*.05; low = math.floor(rank); high = math.ceil(rank)
    rim_y = heights[low]+(heights[high]-heights[low])*(rank-low)
    center_y = (bounds[parent][0][1]+bounds[parent][1][1])/2
    return {'opening_rim_offset_y': float((rim_y-center_y)*scale),
            'body_bottom_offset_y': float((bounds[parent][0][1]-center_y)*scale),
            'base_half_height': float((bounds[parent][1][1]-bounds[parent][0][1])/2*scale),
            'removable_half_heights': {r['child_id']: float((bounds[r['child_id']][1][1]-bounds[r['child_id']][0][1])/2*.095/max(bounds[r['child_id']][1][i]-bounds[r['child_id']][0][i] for i in range(3)))
                                      for r in context['removable'] if r['parent_id'] == parent and r['child_id'] in bounds},
            'method': 'native_rim_and_bbox_authored_display_staging', 'physical_seating_verified': False}


def _actions(objects, joints, goal, actuations, display_geometry=None):
    actions = [];positions = {o['id']: o['initial_transform']['position'][:] for o in objects}
    base = next((o['id'] for o in objects if o['role'] == 'assembly_base'), None)
    def add(op, title, duration=.5, **values):
        actions.append({'id': 'action_%03d' % (len(actions)+1), 'operation': op, 'title': title, 'duration_s': duration, **values})
    def move(p, title, duration=.8):add('move_to', title, duration, position=p)
    if display_geometry:
        add('wait', 'Closed reference · ready to begin', .6)
    def carry(cid, destination, *, grasp_height=0):
        start = positions[cid][:];grasp = [start[0], start[1]+grasp_height, start[2]];above = [grasp[0], max(.42, grasp[1]+.18), grasp[2]]
        move(above, 'Approach '+cid);add('gripper', 'Open gripper', .3, value=0)
        move(grasp, 'Lower to authored grasp anchor', .6);add('gripper', 'Close gripper', .3, value=1)
        add('attach', 'Attach authored '+cid, .2, object_id=cid);move(above, 'Lift '+cid)
        target = [destination[0], destination[1]+grasp_height, destination[2]]
        move([target[0], max(.42, target[1]+.18), target[2]], 'Transfer '+cid, 1.2)
        move(target, 'Lower '+cid, .6);add('detach', 'Release '+cid, .2, object_id=cid)
        add('gripper', 'Open gripper', .3, value=0)
        move([target[0], max(.42, target[1]+.18), target[2]], 'Retreat from '+cid)
        delta = [destination[i]-start[i] for i in range(3)];positions[cid] = destination[:]
        if cid == base:
            for obj in objects:
                if obj['id'] != cid and goal['operation'] not in {'return_components', 'pick_place'}:
                    positions[obj['id']] = [positions[obj['id']][i]+delta[i] for i in range(3)]
    if base and goal['operation'] != 'pick_place':
        carry(base, [-.15, display_geometry['base_half_height'] if display_geometry else positions[base][1], .05], grasp_height=.11)
    def set_lid(opened):
        joint = joints[0];bounds = joint['source_joint']['limits_rad']
        angle = max(bounds[0], min(bounds[1], 1.8)) if opened else 0
        p = positions[joint['parent_id']]
        move([p[0], p[1]+.28, p[2]-.08], 'Approach authored lid actuation cue')
        add('set_joint', 'Open estimated lid' if opened else 'Close estimated lid', 1.4,
            object_id=joint['parent_id'], joint_id=joint['id'], value=angle,
            actuation=_clone(actuations[joint['id']]))
    if goal['open_container'] and joints:set_lid(True)
    for index, cid in enumerate(goal['component_ids']):
        if goal['operation'] == 'return_components':
            parent = positions[base];destination = [parent[0]+(-.07 if index % 2 == 0 else .07), parent[1]+.09, parent[2]]
            if display_geometry and cid in display_geometry.get('removable_offsets', {}):
                destination = [parent[i]+display_geometry['removable_offsets'][cid][i] for i in range(3)]
            elif display_geometry:
                destination[1] = parent[1] + display_geometry['opening_rim_offset_y']-.004-display_geometry['removable_half_heights'][cid]
        else:
            destination = [-.35+(-.07 if index % 2 == 0 else .07), .08, .35] if goal['destination'] == 'tray' else [.02+index*.12, .08, .22]
            if display_geometry:
                # Match the actual drawn table/tray surface using unchanged
                # native bounds, instead of leaving small parts floating.
                destination[1] = (.009 if goal['destination'] == 'tray' else 0.) + display_geometry['removable_half_heights'][cid]
        carry(cid, destination)
    if goal['close_container'] and joints:set_lid(False)
    if display_geometry:
        move(_forward(ROBOT['home']), 'Withdraw from the completed task', 1.2)
    add('wait', 'Inspect the authored rehearsal', 1.4 if display_geometry else .8)
    return actions


def plan_robot_task(twin_id, instruction=None, *, goal=None, kit_id=KIT_ID, root=ROOT, catalog=None):
    kit = get_robot_kit(kit_id, root=root);context = _context(twin_id, root=root, catalog=catalog)
    chosen, missing = _goal(context, instruction, goal)
    if chosen['operation'] == 'open_close' and (chosen['component_ids'] or not (chosen['open_container'] or chosen['close_container'])):
        missing.append(_issue('invalid_articulation_goal', 'Choose opening or closing and no removable selection for an open/close goal.'))
    ids = {r['child_id'] for r in context['removable']}
    for cid in chosen['component_ids']:
        if cid not in ids:missing.append(_issue('unsupported_component', 'Component has no saved removable relationship.', [cid]))
        elif cid not in context['assets']:missing.append(_issue('native_geometry_missing', 'Curated native geometry is unavailable.', [cid]))
    needs_joint = chosen['open_container'] or chosen['close_container'] or chosen['operation'] == 'open_close'
    active_parent = context['joints'][0]['parent_id'] if context['joints'] else (context['removable'][0]['parent_id'] if context['removable'] else None)
    selected_parents = {r['parent_id'] for r in context['removable'] if r['child_id'] in chosen['component_ids']}
    if len(selected_parents) > 1 or selected_parents and selected_parents != {active_parent}:
        missing.append(_issue('multiple_assemblies_require_selection', 'This starter needs one unambiguous source assembly; a different assembly cannot be silently substituted.'))
    if needs_joint and len(context['joints']) > 1:
        missing.append(_issue('multiple_joints_require_selection', 'Select one articulated assembly before planning; this starter does not silently choose among several joints.'))
    if needs_joint and not context['joints']:
        missing.append(_issue('estimated_joint_missing', 'No source/asset-bound estimated revolute joint is available.'))
    if needs_joint and context['actuation_errors']:
        missing.append(_issue('native_actuation_cue_unavailable', 'The exact native geometry cannot supply the authored position-only actuation cue.'))
    if not chosen['component_ids'] and chosen['operation'] != 'open_close':
        missing.append(_issue('removable_component_missing', 'Select at least one source-described removable component.'))
    if context['joints'] and needs_joint:
        j = context['joints'][0]['source_joint']
        allowed_zero = {'estimated_closed', 'estimated_geometric_closed'} if context.get('derived_assembly') else {'estimated_closed'}
        if not j['limits_rad'][0] <= 0 <= j['limits_rad'][1] or j['zero_reference'].get('state') not in allowed_zero:
            missing.append(_issue('closed_reference_missing', 'The saved joint does not supply a closed zero inside its estimated range.'))
    objects, joints, attachments = _stage(context, chosen)
    plan = {'format': FORMAT, 'version': 1, 'status': 'blocked' if missing else 'ready',
            'kit_id': kit_id, 'kit_sha256': _digest(kit), 'twin_id': twin_id, 'source_binding': context['binding'],
            'staging_reference': _clone(context.get('source_staging_reference')),
            'goal': chosen, 'objects': objects, 'joints': joints, 'attachments': attachments,
            'actions': [] if missing else _actions(objects, joints, chosen, context['actuations'], context.get('derived_display_geometry')), 'missing_requirements': missing,
            'claims': _clone(CLAIMS), 'limitations': ['Robot poses, grasp anchors and staging are authored display choices, not observed robot motion.',
                'Native geometry and estimated object articulation retain their saved uncertainty.',
                'Joint actuation follows an authored bounding-box cue in position only; rigid grasp orientation, collision and contact are unverified.',
                'Earbud/component physical identity, insertion and seating remain unverified.']}
    if context.get('derived_assembly'):
        plan['limitations'].append('This task uses the explicitly approved geometry derivative. Source fitting and authored staging do not establish physical contact. Original native meshes remain available separately.')
    result = _validate(plan, context, kit)
    if not result['valid'] and not missing:
        plan['status'] = 'blocked';plan['actions'] = []
        plan['missing_requirements'] = result['errors'][:32]
    return plan


def _validate(plan, context, kit):
    errors = []
    try:
        import jsonschema
        if len(json.dumps(plan, allow_nan=False).encode()) > 262144:raise ValueError('Task JSON exceeds its bounded size')
        jsonschema.validate(plan, task_schema())
    except Exception:
        return {'valid': False, 'status': 'invalid', 'errors': [_issue('invalid_task_schema', 'Task JSON does not match the bounded version1 contract.')], **_clone(CLAIMS)}
    if plan['kit_sha256'] != _digest(kit) or plan['source_binding'] != context['binding']:
        errors.append(_issue('task_binding_changed', 'Kit or curated source/program binding changed.'))
    known = context['assets'];objects = {o['id']: o for o in plan['objects']}
    baseline_objects, baseline_joints, baseline_attachments = _stage(context, plan['goal'])
    if plan.get('staging_reference') != context.get('source_staging_reference'):
        errors.append(_issue('staging_reference_changed', 'Keep the exact saved source/native visual staging provenance.'))
    baseline_objects = {o['id']: o for o in baseline_objects}
    if set(objects) != set(baseline_objects):
        errors.append(_issue('component_inventory_changed', 'Keep the complete source-bound assembly component inventory.'))
    if len(objects) != len(plan['objects']):errors.append(_issue('duplicate_object', 'Object IDs must be unique.'))
    for cid, obj in objects.items():
        if obj['component_id'] != cid or cid not in known or any(obj[k] != known[cid][k] for k in ['asset_ref', 'asset_scope']):
            errors.append(_issue('unapproved_asset', 'Object is not the exact curated native component.', [cid]))
        if cid in baseline_objects and any(obj[k] != baseline_objects[cid][k] for k in ['label', 'role', 'display_max_dimension', 'placement_origin']):
            errors.append(_issue('component_role_changed', 'Native component roles and display normalization are immutable.', [cid]))
        p = obj['initial_transform']['position'];q = obj['initial_transform']['quaternion_xyzw']
        if (context.get('source_staging_reference') and
            cid in {r['component_id'] for r in context['source_staging_reference']['components']} and
            (q != baseline_objects[cid]['initial_transform']['quaternion_xyzw'] or
             obj['initial_transform']['uniform_scale'] != baseline_objects[cid]['initial_transform']['uniform_scale'])):
            errors.append(_issue('source_normalization_changed', 'Keep the source-bound relative rotation and measured display normalization.', [cid]))
        if any(abs(v) > 2.5 for v in p) or p[1] < 0 or abs(sum(v*v for v in q)-1) > 1e-6:
            errors.append(_issue('invalid_staging_transform', 'Staging transform is outside the authored workspace.', [cid]))
    source_joints = {j['id']: j for j in context['joints']};joints = {}
    if plan['joints'] != baseline_joints:
        errors.append(_issue('joint_inventory_changed', 'Keep the exact estimated articulation associated with these native parts.'))
    for j in plan['joints']:
        if j['id'] in joints or j != source_joints.get(j['id']) or j['parent_id'] not in objects or j['child_id'] not in objects:
            errors.append(_issue('joint_binding_changed', 'Object joint must exactly match the saved descriptive twin.'))
        joints[j['id']] = j
    relations = {(r['parent_id'], r['child_id'], r['id']) for r in context['removable']};parents = {}
    if plan['attachments'] != baseline_attachments:
        errors.append(_issue('staging_attachment_changed', 'Keep the authored assembly membership; edit placements consistently instead.'))
    for a in plan['attachments']:
        if ((a['parent_id'], a['child_id'], a['source_relationship_id']) not in relations or a['child_id'] in parents or
                a['parent_id'] not in objects or a['child_id'] not in objects or a['parent_id'] == a['child_id']):
            errors.append(_issue('unsupported_staging_attachment', 'Staging attachment needs an exact saved removable relationship.'))
        parents[a['child_id']] = a['parent_id']
        if a['parent_id'] in objects and a['child_id'] in objects:
            parent_position = objects[a['parent_id']]['initial_transform']['position']
            child_position = objects[a['child_id']]['initial_transform']['position']
            if any(abs(child_position[i]-parent_position[i]-a['offset'][i]) > 1e-7 for i in range(3)):
                errors.append(_issue('inconsistent_staging_attachment', 'Move staged children with their parent or update the complete authored setup.'))
    for child in parents:
        seen = set();current = child
        while current in parents:
            if current in seen:errors.append(_issue('attachment_cycle', 'Attachment graph must be acyclic.'));break
            seen.add(current);current = parents[current]
    if plan['status'] == 'blocked':
        if plan['actions'] or not plan['missing_requirements']:errors.append(_issue('blocked_task_has_actions', 'Blocked tasks must contain requirements and no actions.'))
        return {'valid': False, 'status': 'blocked' if not errors else 'invalid', 'errors': errors or _clone(plan['missing_requirements']), **_clone(CLAIMS)}
    if plan['missing_requirements'] or not plan['actions']:
        errors.append(_issue('ready_task_incomplete', 'Ready tasks require actions and no missing requirements.'))
    needed_joint = plan['goal']['open_container'] or plan['goal']['close_container'] or plan['goal']['operation'] == 'open_close'
    if needed_joint and len(context['joints']) != 1:
        errors.append(_issue('unambiguous_joint_required', 'An articulated task requires exactly one source-bound joint in this starter.'))
    if ((plan['goal']['operation'] == 'open_close' and plan['goal']['component_ids']) or
        (plan['goal']['operation'] != 'open_close' and not plan['goal']['component_ids'])):
        errors.append(_issue('invalid_goal_selection', 'Open/close has no moving selection; transfer goals require selected removable parts.'))
    if plan['goal']['operation'] == 'open_close' and not (plan['goal']['open_container'] or plan['goal']['close_container']):
        errors.append(_issue('invalid_articulation_goal', 'An open/close goal requires at least one articulation command.'))
    positions = {cid: o['initial_transform']['position'][:] for cid, o in objects.items()}
    initial_positions = _clone(positions)
    tool = _forward(ROBOT['home']);gripper = 0;held = None;offset = None;duration = 0;action_ids = set();carried_distance = 0
    child_joints = {j['child_id'] for j in plan['joints']}
    joint_values = {j['id']: j['initial_rad'] for j in plan['joints']}
    opened_joints = set()
    for action in plan['actions']:
        if action['id'] in action_ids:errors.append(_issue('duplicate_action', 'Action IDs must be unique.'))
        action_ids.add(action['id']);duration += action['duration_s'];op = action['operation'];cid = action.get('object_id')
        fields = {'move_to': {'position'}, 'gripper': {'value'}, 'attach': {'object_id'}, 'detach': {'object_id'},
                  'set_joint': {'object_id', 'joint_id', 'value', 'actuation'}, 'wait': set()}[op]
        if set(action)-{'id', 'operation', 'title', 'duration_s'} != fields:
            errors.append(_issue('invalid_action_fields', 'Action fields must exactly match its operation.'));continue
        if op == 'move_to':
            p = action['position']
            if p[1] < 0 or any(abs(v) > 2.5 for v in p) or any(not inverse_kinematics([tool[i]+(p[i]-tool[i])*k/20 for i in range(3)])['reachable'] for k in range(21)):
                errors.append(_issue('unreachable_path', 'A move exceeds joint limits or the position residual bound.'))
            if held:
                old = positions[held][:];positions[held] = [p[i]+offset[i] for i in range(3)];delta = [positions[held][i]-old[i] for i in range(3)]
                carried_distance += math.dist(old, positions[held])
                for child in parents:
                    ancestor = parents.get(child);seen = set()
                    while ancestor in parents and ancestor not in seen:seen.add(ancestor);ancestor = parents[ancestor]
                    if ancestor == held:positions[child] = [positions[child][i]+delta[i] for i in range(3)]
            tool = p[:]
        elif op == 'gripper':
            if not 0 <= action['value'] <= 1:errors.append(_issue('invalid_gripper', 'Gripper value must be between zero and one.'))
            if held and action['value'] < .8:errors.append(_issue('gripper_open_while_holding', 'Detach the carried object before opening the gripper.'))
            gripper = action['value']
        elif op == 'attach':
            if cid not in objects or held or gripper < .8 or cid in child_joints:
                errors.append(_issue('invalid_grasp_transition', 'Close the gripper before attaching one free, non-hinged object.'));continue
            parent = parents.get(cid)
            if parent and any(j['parent_id'] == parent and abs(joint_values[j['id']]) <= .2617993877991494 for j in plan['joints']):
                errors.append(_issue('container_closed', 'Open the estimated lid before picking an authored contained object.', [cid]))
            offset = [positions[cid][i]-tool[i] for i in range(3)]
            if math.sqrt(sum(v*v for v in offset)) > .18:errors.append(_issue('grasp_anchor_too_far', 'Tool is too far from the authored grasp anchor.', [cid]))
            parents.pop(cid, None);held = cid;carried_distance = 0
        elif op == 'detach':
            if held != cid:errors.append(_issue('invalid_release', 'Release must name the carried object.'))
            if held == cid and carried_distance < .001:errors.append(_issue('release_without_movement', 'Carry the attached object before releasing it.', [cid]))
            held = None;offset = None
        elif op == 'set_joint':
            j = joints.get(action['joint_id'])
            if not j or cid != j['parent_id'] or held or not j['source_joint']['limits_rad'][0] <= action['value'] <= j['source_joint']['limits_rad'][1]:
                errors.append(_issue('invalid_joint_target', 'Joint command is not within the exact estimated object joint limits.'))
            elif action['actuation'] != context['actuations'].get(j['id']):
                errors.append(_issue('actuation_binding_changed', 'Actuation cue must match the exact native geometry and source joint.'))
            else:
                from robot_task_geometry import cue_state
                boundaries = [0, action['actuation']['approach_fraction'], action['actuation']['release_fraction'], 1]
                reachable = True
                for low, high in zip(boundaries, boundaries[1:]):
                    for k in range(action['actuation']['sample_count']):
                        progress = low+(high-low)*k/(action['actuation']['sample_count']-1)
                        cue = cue_state(action['actuation'], j, objects[cid], positions[cid], tool,
                                        joint_values[j['id']], action['value'], progress)
                        if cue['position'][1] < 0 or not inverse_kinematics(cue['position'])['reachable']:
                            reachable = False;break
                    if not reachable:break
                if not reachable:
                    errors.append(_issue('unreachable_joint_actuation', 'The position-only native hinge cue exceeds the arm joint limits.'))
                joint_values[j['id']] = action['value'];gripper = 0
                if abs(action['value']) > .2617993877991494:opened_joints.add(j['id'])
    if duration > 300:errors.append(_issue('duration_budget', 'Task duration exceeds five minutes.'))
    if held:errors.append(_issue('unfinished_grasp', 'Task must finish with the object released.'))
    for cid in plan['goal']['component_ids']:
        if cid not in objects:
            errors.append(_issue('goal_component_missing', 'A requested component is absent from the task.', [cid]));continue
        if plan['goal']['operation'] in {'remove_components', 'pick_place'}:
            if cid in parents or math.dist(positions[cid], initial_positions[cid]) < .03:
                errors.append(_issue('goal_not_reached', 'The requested component was not moved and released.', [cid]))
            if plan['goal']['destination'] == 'tray' and (abs(positions[cid][0]+.35) > .2 or abs(positions[cid][2]-.35) > .2 or not 0 <= positions[cid][1] <= .2):
                errors.append(_issue('destination_not_reached', 'The requested component did not reach the authored tray region.', [cid]))
        elif plan['goal']['operation'] == 'return_components':
            relation = next((r for r in context['removable'] if r['child_id'] == cid), None)
            if not relation or relation['parent_id'] not in positions or math.dist(positions[cid], positions[relation['parent_id']]) > .2:
                errors.append(_issue('return_target_not_reached', 'The component did not reach the authored container region; seating remains unverified.', [cid]))
    if plan['goal']['close_container'] and any(abs(joint_values[j['id']]) > 1e-6 for j in plan['joints']):
        errors.append(_issue('close_goal_not_reached', 'The estimated lid was not returned to its closed reference.'))
    if plan['goal']['open_container'] and any(j['id'] not in opened_joints for j in plan['joints']):
        errors.append(_issue('open_goal_not_reached', 'The requested opening command was not performed.'))
    return {'valid': not errors, 'status': 'valid' if not errors else 'invalid', 'errors': errors[:32],
            'duration_s': duration, 'kinematics_scope': 'joint-limited position diagnostics; no collision or contact validation', **_clone(CLAIMS)}


def validate_robot_task(program, *, root=ROOT, catalog=None):
    if not isinstance(program, dict) or not isinstance(program.get('twin_id'), str):
        return {'valid': False, 'status': 'invalid', 'errors': [_issue('invalid_task_schema', 'A version1 robot task object is required.')], **_clone(CLAIMS)}
    try:
        context = _context(program['twin_id'], root=root, catalog=catalog)
        kit = get_robot_kit(program.get('kit_id'), root=root)
        return _validate(program, context, kit)
    except (ValueError, KeyError, TypeError, OSError):
        return {'valid': False, 'status': 'invalid', 'errors': [_issue('curated_binding_unavailable', 'The exact curated twin, kit or source artifacts are unavailable.')], **_clone(CLAIMS)}


def task_bindings(program):
    """Only schema-declared refs; adapters must additionally validate and allowlist."""
    import jsonschema
    jsonschema.validate(program, task_schema())
    refs = [program['source_binding']['program_ref'], program['source_binding']['object_program_ref']]
    staging = program.get('staging_reference')
    if staging:
        refs.append(staging['scene_ref'])
        for row in staging['components']:
            refs += [row['parent_source_mask_ref'], row['child_source_mask_ref']]
    refs += [obj['asset_ref'] for obj in program['objects']]
    found = {}
    for item in refs:
        if item is not None:
            found[(item['path'], item['sha256'])] = _clone(item)
    return list(found.values())
