"""First-class, source-bound descriptive kinematics beside immutable v0.1 evidence.

This format makes estimated joints/frames/angle curves available to agent clients.
It does not certify geometry, metric scale, physical stops, contact or simulation.
Only saved source-bound artifacts are read; no inference or robot execution.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import jsonschema
import numpy as np

ROOT = Path(__file__).resolve().parent
FORMAT = 'interaction-twin-object-program'
VERSION = '0.2'
MAX_JSON = 6*1024*1024
CODE = {'sam3d_common_scene.py', 'sam3d_coordinates.py', 'sam3d_source_projection.py', 'native_appearance.py',
        'sam3d_review_catalog.py', 'sam3d_mechanisms.py', 'mechanism_reasoning.py', 'object_program.py',
        'native_audit_protocol.py', 'part_ledger.py', 'native_pose_refinement.py', 'alignment_render.py', 'native_scene_audit.py', 'rigidity_feasibility.py'}
CONVENTION = 'parent_native_pose * parent_from_joint(inverse_parent_scale) * axis_rotation(q) * joint_from_child_zero(child_scale_once)'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def implementation_refs(source_scene, mechanics):
    """Exact optional refinement dependencies; legacy mechanics need no change."""
    additional=mechanics.get('additional_implementations',[])
    require(isinstance(additional,list) and len(additional)<=8,'Bounded additional implementation references required')
    ordered={}
    for item in [*source_scene['dependencies'],mechanics['implementation'],*additional]:
        require(isinstance(item,dict) and set(item)=={'path','sha256','bytes'} and item['path']in CODE and
                isinstance(item['sha256'],str) and re.fullmatch('[a-f0-9]{64}',item['sha256']) and
                type(item['bytes'])is int and 0<item['bytes']<=500000,'Exact allowed implementation reference required')
        require(item['path']not in ordered or ordered[item['path']]==item,'Conflicting implementation references')
        ordered[item['path']]=item
    return list(ordered.values())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def data_path(root, relative):
    root = Path(root).resolve()
    require(isinstance(relative, str) and relative.startswith('data/') and
            not any(p.startswith('.') for p in Path(relative).parts), 'Only ordinary declared data artifacts are available')
    path = root/relative
    require((root/'data').resolve() == root/'data' and path.resolve().is_relative_to(root/'data') and
            not any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(root/'data')),
            'Symlink or escaping object-program artifact')
    return path


def ref(path, root):
    path = Path(path)
    return {'path': str(path.relative_to(root)), 'sha256': sha(path), 'bytes': path.stat().st_size}


def checked_ref(value, root, maximum=MAX_JSON):
    require(isinstance(value, dict) and set(value) == {'path', 'sha256', 'bytes'}, 'Exact object-program artifact reference required')
    require(isinstance(value['sha256'], str) and re.fullmatch('[0-9a-f]{64}', value['sha256']) and
            type(value['bytes']) is int and 0 < value['bytes'] <= maximum, 'Invalid object-program artifact hash/size')
    path = data_path(root, value['path'])
    require(path.is_file() and path.stat().st_size == value['bytes'] and sha(path) == value['sha256'],
            'Missing or changed object-program artifact')
    return path


def read_json(path, maximum=MAX_JSON):
    require(path.is_file() and 0 < path.stat().st_size <= maximum, 'Missing or oversized object-program JSON')
    value = json.loads(path.read_text())
    # Reject non-finite JSON extension values rather than passing them to consumers.
    json.dumps(value, allow_nan=False)
    return value


def _obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def schema():
    """Typed public interchange schema; geometric/reference checks are additional."""
    s = {'type': 'string', 'minLength': 1}
    n = {'type': 'number'}
    v3 = {'type': 'array', 'items': n, 'minItems': 3, 'maxItems': 3}
    matrix = {'type': 'array', 'items': {'type': 'array', 'items': n, 'minItems': 4, 'maxItems': 4}, 'minItems': 4, 'maxItems': 4}
    artifact = _obj({'path': s, 'sha256': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'}, 'bytes': {'type': 'integer', 'minimum': 1}})
    sample = _obj({'source_sample_index': {'type': 'integer', 'minimum': 1}, 'timestamp_s': {'type': 'number', 'minimum': 0},
                   'status': {'enum': ['estimated', 'unknown']}, 'q_rad': {'type': ['number', 'null']}})
    joint = _obj({'id': s, 'type': {'const': 'revolute'}, 'parent_id': s, 'child_id': s, 'status': {'const': 'estimated'},
                  'axis': v3, 'axis_frame': {'const': 'joint_frame_after_parent_from_joint'},
                  'parent_from_joint': matrix, 'joint_from_child_zero': matrix,
                  'parent_anchor_native': v3, 'child_anchor_native': v3,
                  'parent_anchor_vertex_id': {'type': 'integer', 'minimum': 0}, 'child_anchor_vertex_id': {'type': 'integer', 'minimum': 0},
                  'parent_native_scale': {'type': 'number', 'exclusiveMinimum': 0}, 'child_native_scale': {'type': 'number', 'exclusiveMinimum': 0},
                  'limits_rad': {'type': 'array', 'items': n, 'minItems': 2, 'maxItems': 2},
                  'limits_kind': {'const': 'observed_estimated_range_not_verified_physical_stops'},
                  'physical_limits_rad': {'type': 'null'},
                  'angle_quality': _obj({'angle_visual_consistency':{'type':'boolean'},
                      'test':{'const':'all_cited_supported_open_states_separated_from_cited_closed_zero_by_more_than_declared_threshold'},
                      'declared_separation_threshold_rad':{'type':'number','exclusiveMinimum':0},
                      'state_comparisons':{'type':'array','maxItems':24,'items':_obj({'source_sample_index':{'type':'integer','minimum':1},
                          'timestamp_s':{'type':'number','minimum':0},'cited_state':{'enum':['open','closed']},'q_rad':n,
                          'distance_from_closed_reference_rad':{'type':'number','minimum':0}})},
                      'angle_interpretation':{'enum':['model_configuration_parameter_not_verified_opening_angle','estimated_opening_configuration_not_physically_calibrated']},
                      'confidence':{'const':'low'},'physical_angle_verified':{'const':False},'zero_state_alignment_verified':{'const':False},
                      'source_alignment_verified':{'const':False},'quality_flags':{'type':'array','items':s,'maxItems':12}}),
                  'zero_reference': _obj({'source_sample_index':{'type':'integer','minimum':1},'timestamp_s':{'type':'number','minimum':0},
                      'state':{'enum':['estimated_closed','model_reference_state_unknown']}}), 'samples': {'type': 'array', 'items': sample, 'minItems': 1, 'maxItems': 10000},
                  'evidence_ids': {'type': 'array', 'items': s, 'maxItems': 16},
                  'matrix_convention': {'const': CONVENTION}, 'physical_accuracy_verified': {'const': False}, 'simulation_ready': {'const': False}})
    return _obj({'format': {'const': FORMAT}, 'version': {'const': VERSION}, 'created_utc': s,
                 'evidence_program': artifact, 'source': {'type': 'object'},
                 'source_scene': artifact, 'constrained_scene': artifact, 'mechanics': artifact, 'proposal': artifact,
                 'component_assets': {'type': 'array', 'minItems': 1, 'maxItems': 32,
                     'items': _obj({'component_id': s, 'native_glb': artifact, 'native_metadata': artifact,
                                    'native_uniform_scale': {'type': 'number', 'exclusiveMinimum': 0}})},
                 'joints': {'type': 'array', 'items': joint, 'minItems': 0, 'maxItems': 12},
                 'unresolved_connections': {'type':'array','maxItems':24,'items':{
                     'type':'object','properties':{'id':s,'parent_id':s,'child_id':s,
                         'type':{'enum':['fixed','revolute','prismatic','unknown']},'status':{'const':'unknown'},
                         'reason':s,'diagnostic':{'type':'string','maxLength':500}},
                     'required':['id','parent_id','child_id','type','status','reason'],'additionalProperties':False}},
                 'removable_relationships': {'type': 'array', 'items': _obj({'id': s, 'parent_id': s, 'child_id': s,
                     'joint_type': {'const': 'removable'}, 'attachment': {'const': 'removable'},
                     'confidence': {'enum': ['low', 'medium', 'high']}, 'evidence_ids': {'type': 'array', 'items': s, 'minItems': 1},
                     'physical_accuracy_verified': {'const': False}}), 'maxItems': 24},
                 'coordinate_system': _obj({'camera_frame': {'const': 'opencv_x_right_y_down_z_forward'},
                     'length_units': {'const': 'inferred_arbitrary_scene_units_no_metric_scale'}, 'angle_units': {'const': 'radians'},
                     'unknown_sample_policy': {'const': 'withhold_joint_and_child_when_parent_or_joint_unknown'},
                     'physical_stops': {'const': 'unknown'}}),
                 'implementation_provenance': {'type': 'array', 'minItems': 1, 'maxItems': len(CODE),
                     'items': _obj({'original_path': s, 'sha256': s, 'archived_file': artifact})},
                 'physical_accuracy_verified': {'const': False}, 'simulation_ready': {'const': False},
                 'robot_execution_available': {'const': False}, 'manual_joint_coordinates_used': {'const': False},
                 'limitations': {'type': 'array', 'items': s, 'minItems': 1, 'maxItems': 32}})


JOINT_FIELDS = ['id', 'type', 'parent_id', 'child_id', 'status', 'axis', 'parent_from_joint', 'joint_from_child_zero',
                'parent_anchor_native', 'child_anchor_native', 'parent_anchor_vertex_id', 'child_anchor_vertex_id',
                'parent_native_scale', 'child_native_scale', 'limits_rad', 'limits_kind', 'physical_limits_rad',
                'zero_reference', 'angle_quality', 'evidence_ids', 'matrix_convention', 'physical_accuracy_verified', 'simulation_ready']


def projected_joint(joint):
    return {**{k: joint[k] for k in JOINT_FIELDS}, 'axis_frame': 'joint_frame_after_parent_from_joint',
            'samples': [{k: sample[k] for k in ['source_sample_index', 'timestamp_s', 'status', 'q_rad']} for sample in joint['samples']]}


def axis_rotation(axis, angle):
    x,y,z = np.asarray(axis,float)
    skew = np.asarray([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+math.sin(angle)*skew+(1-math.cos(angle))*(skew@skew)


def _similarity(matrix, scale):
    matrix = np.asarray(matrix, float)
    require(matrix.shape == (4, 4) and np.isfinite(matrix).all() and np.allclose(matrix[3], [0, 0, 0, 1]) and
            np.allclose(matrix[:3, :3].T@matrix[:3, :3], np.eye(3)*scale**2, atol=1e-7) and
            np.isclose(np.linalg.det(matrix[:3, :3]), scale**3, atol=1e-7), 'Invalid proper joint-frame similarity')
    return matrix


def _verify_proposal(proposal, program, root):
    """Check saved model JSON and deterministic UV export, never call inference.

    This reader deliberately has no hosted-client, image-library or learned-model
    dependency, so the read-only MCP environment needs no inference runtime.
    """
    def digest(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    reasoning = proposal['reasoning']
    require(reasoning['provider'] == 'meta_api' and reasoning['model'] == 'muse-spark-1.3' and
            reasoning['geometric_joint_fit_performed'] is False, 'Unknown mechanism reasoning provenance')
    plan_path = data_path(root, reasoning['plan_path'])
    require(plan_path.is_file() and sha(plan_path) == reasoning['plan_sha256'], 'Mechanism plan changed')
    plan = read_json(plan_path)
    context = plan['context']
    require(digest(context) == plan['contract']['context_sha256'] and context['tracklets'] == program['tracklets'] and
            context['source_sha256'] == program['source']['sha256'] and
            context['interaction_sha256'] == proposal['interaction_sha256'], 'Mechanism source context mismatch')
    evidence = {e['id']:e for e in program['evidence']}
    for row in context['evidence']:
        original = evidence.get(row['evidence_id'])
        require(original and all(original[k] == row[k] for k in ['source_index','timestamp_s','frame_path','frame_sha256']),
                'Mechanism source citation changed')
    images = {i['evidence_id']:i for i in context['images'] if i['role']=='source_rgb'}
    for image in context['images']:
        path = data_path(root, image['path'])
        require(path.is_file() and path.stat().st_size <= 8*1024*1024 and sha(path) == image['sha256'], 'Mechanism source image changed')
    model_path = data_path(root, reasoning['model_result_path'])
    require(model_path.is_file() and sha(model_path) == reasoning['model_result_sha256'], 'Mechanism model response changed')
    model = read_json(model_path)
    schema_path = data_path(root,str((plan_path.parent/'schema.json').relative_to(root)))
    model_schema = read_json(schema_path)
    require(digest(model_schema)==plan['contract']['schema_sha256'],'Mechanism response schema changed')
    jsonschema.Draft202012Validator.check_schema(model_schema)
    jsonschema.validate(model,model_schema)
    def source(ref):
        original,image=evidence[ref],images[ref]
        require(image['path']==original['frame_path'] and image['sha256']==original['frame_sha256'] and
                type(image['width']) is int and type(image['height']) is int and
                1<image['width']<=8192 and 1<image['height']<=8192,'Invalid source image coordinate binding')
        return {'evidence_id':ref,'source_sample_index':original['source_index'],'timestamp_s':original['timestamp_s'],
                'image_width':image['width'],'image_height':image['height']}
    expected_joints=[]
    for joint in model['joints']:
        require(set(joint['evidence_ids']) <= set(images),'Mechanism cites an unsupplied RGB frame')
        hints=[]
        for hint in joint['hinge_hints']:
            require(hint['evidence_id'] in joint['evidence_ids'],'Hinge hint lacks joint evidence')
            e=source(hint['evidence_id'])
            endpoints=[[uv[0]*(e['image_width']-1),uv[1]*(e['image_height']-1)]
                       for uv in [hint['endpoint_a_normalized'],hint['endpoint_b_normalized']]]
            hints.append({**e,'endpoints_px':endpoints,'uncertainty_px':hint['uncertainty_normalized']*max(e['image_width'],e['image_height']),
                          'confidence':hint['confidence'],'normalized_hint':hint})
        states=[]
        for state in joint['state_hints']:
            require(state['evidence_id'] in joint['evidence_ids'],'State hint lacks joint evidence')
            states.append({**source(state['evidence_id']),'state':state['state']})
        expected_joints.append({**joint,'hinge_hints':hints,'state_hints':states})
    require(proposal['joints']==expected_joints and proposal['title']==model['title'] and proposal['limitations']==model['limitations'] and
            proposal['coordinate_system']=='source_image_pixels_origin_upper_left_y_down',
            'Mechanism proposal differs from saved model/coordinate conversion')
    receipt = read_json(data_path(root,reasoning['request_receipt']), 200000)
    require(receipt.get('status') == 'complete_accounted' and receipt.get('model') == reasoning['model'] and
            receipt.get('usage') == reasoning['usage'] and receipt.get('estimated_cost_usd') == reasoning['estimated_cost_usd'],
            'Mechanism saved accounting receipt mismatch')


def load_proposals(path, program_path, *, root=ROOT):
    """Verify and reuse a completed source-bound topology request, no paid resend."""
    root=Path(root).resolve()
    program_file=data_path(root,str(program_path))
    program=read_json(program_file)
    proposal=read_json(data_path(root,str(path)))
    require(proposal.get('format')=='sam3d-mechanism-proposals' and proposal.get('version')==1 and
            proposal.get('source_sha256')==program['source']['sha256'] and
            proposal.get('interaction_sha256')==sha(program_file) and proposal.get('physical_accuracy_verified') is False,
            'Cached mechanism proposal source/program mismatch')
    _verify_proposal(proposal,program,root)
    return proposal


def native_vertices(path):
    # The bounded embedded-only decoder preserves source vertex order for these
    # SAM meshes. A different ordering fails the explicit vertex-ID check rather
    # than silently snapping an exported joint to another point.
    from portable_twin import decode_glb_bytes
    decoded=decode_glb_bytes(path.read_bytes())
    return np.concatenate([np.asarray(p['vertices'],float) for p in decoded['primitives']])


def validate(document, *, root=ROOT, verify_files=True):
    root = Path(root).resolve()
    json.dumps(document, allow_nan=False)
    jsonschema.validate(document, schema())
    created=datetime.fromisoformat(document['created_utc'].replace('Z','+00:00'))
    require(created.tzinfo is not None,'Object-program creation time requires a timezone')
    if not verify_files:
        raise ValueError('Object programs require source/artifact verification')
    refs = {key: checked_ref(document[key], root) for key in ['evidence_program', 'source_scene', 'constrained_scene', 'mechanics', 'proposal']}
    program, source_scene, scene, mechanics, proposal = [read_json(refs[key]) for key in ['evidence_program', 'source_scene', 'constrained_scene', 'mechanics', 'proposal']]
    require(program['source'] == {k:v for k,v in document['source'].items() if k not in {'width','height'}} and
            document['source'] == scene['source'] == source_scene['source'], 'Object-program source mismatch')
    require(all(s.get('physical_accuracy_verified') is False and s.get('simulation_ready') is False and
                s.get('manual_geometry_or_poses_used') is False and s.get('old_backend_geometry_or_poses_read') is False and
                s.get('accepted_motion_poses') == 0 for s in [scene,source_scene]), 'Scene estimates cannot be promoted to physical or accepted motion')
    require(scene['original_program'] == source_scene['original_program'] == document['evidence_program'], 'Object-program evidence binding mismatch')
    require(scene['mechanics'] == mechanics and mechanics['source_scene'] == document['source_scene'] and
            mechanics['proposal'] == document['proposal'], 'Object-program mechanics provenance mismatch')
    require(mechanics.get('format') == 'sam3d-constrained-mechanisms' and mechanics.get('version') == 1 and
            mechanics.get('physical_accuracy_verified') is False and mechanics.get('simulation_ready') is False and
            mechanics.get('manual_joint_coordinates_used') is False and mechanics.get('free_floating_child_fallback') is False,
            'Unsupported or promoted mechanics contract')
    require(proposal.get('format') == 'sam3d-mechanism-proposals' and proposal.get('version') == 1 and
            proposal.get('source_sha256') == program['source']['sha256'] and
            proposal.get('interaction_sha256') == document['evidence_program']['sha256'] and
            proposal.get('physical_accuracy_verified') is False, 'Object-program proposal source mismatch')
    video = data_path(root, program['source']['path'])
    require(video.is_file() and video.stat().st_size <= 1024*1024*1024 and sha(video) == program['source']['sha256'],
            'Missing or changed source recording')
    _verify_proposal(proposal, program, root)
    evidence = {e['id']: e for e in program['evidence']}
    tracks = {t['id'] for t in program['tracklets']}
    components = {c['id']: c for c in scene['components']}
    require(set(components) <= tracks and len(components) == len(scene['components']), 'Unknown or duplicate scene component')
    original_components = {c['id']: c for c in source_scene['components']}
    require(set(components) == set(original_components), 'Constrained scene changes source components')
    expected_assets = [{'component_id': c['id'], 'native_glb': c['asset'], 'native_metadata': c['native_metadata'],
                        'native_uniform_scale': c['native_uniform_scale']} for c in scene['components']]
    require(document['component_assets'] == expected_assets, 'Object-program asset binding mismatch')
    for component in document['component_assets']:
        asset_path = checked_ref(component['native_glb'], root, 16*1024*1024)
        require(asset_path.read_bytes()[:4] == b'glTF', 'Invalid native GLB signature')
        metadata = read_json(checked_ref(component['native_metadata'], root))
        require(metadata['object_id'] == component['component_id'] and metadata['source']['video_sha256'] == program['source']['sha256'],
                'Native object/source metadata mismatch')
        artifact = next((a for a in metadata.get('artifact_files',[]) if a.get('file')=='asset.glb'),None)
        require(artifact and artifact['sha256']==component['native_glb']['sha256'] and metadata.get('geometry_reconstructed_by_model') is True and
                metadata.get('manual_geometry_fallback') is False and metadata.get('manual_pose_used') is False, 'Native asset lacks genuine automatic generation provenance')
        c, old = components[component['component_id']], original_components[component['component_id']]
        require(c['asset'] == old['asset'] and c['native_metadata'] == old['native_metadata'] and
                c['native_uniform_scale'] == old['native_uniform_scale'], 'Mechanics changed native asset or scale')
    archived = {};archived_sizes={}
    for row in document['implementation_provenance']:
        require(row['original_path'] in CODE and row['original_path'] not in archived, 'Unexpected or duplicate implementation archive')
        checked_ref(row['archived_file'], root, 500000)
        require(row['sha256'] == row['archived_file']['sha256'], 'Implementation archive hash mismatch')
        archived[row['original_path']] = row['sha256']
        archived_sizes[row['original_path']]=row['archived_file']['bytes']
    required_code = implementation_refs(source_scene,mechanics)
    # Every saved input is confined and hashed; current code is not executed or
    # required to remain byte-identical because its exact source is archived.
    for item in source_scene['inputs']:
        checked_ref(item, root, 64*1024*1024)
    for item in required_code:
        require(archived.get(item['path']) == item['sha256'] and archived_sizes.get(item['path'])==item['bytes'], 'Missing historical implementation archive')
    require(document['unresolved_connections'] == mechanics['unresolved_children'], 'Unresolved topology differs from geometric fitter')
    require(document['joints'] == [projected_joint(j) for j in mechanics['joints']], 'Object-program joint definition differs from fitter')
    parents, seen = {}, set()
    for joint in document['joints']:
        p, c = joint['parent_id'], joint['child_id']
        require(p in components and c in components and p != c and c not in parents and joint['id'] not in seen,
                'Unknown, duplicate or ambiguous mechanism parent')
        parents[c] = p; seen.add(joint['id'])
        ancestor, visited = c, set()
        while ancestor in parents:
            require(ancestor not in visited, 'Cyclic persistent mechanism graph')
            visited.add(ancestor); ancestor = parents[ancestor]
        require(joint['evidence_ids'] and set(joint['evidence_ids']) <= set(evidence), 'Unknown or empty mechanism evidence IDs')
        proposal_joint = next((j for j in proposal['joints'] if j['id'] == joint['id']), None)
        require(proposal_joint and proposal_joint['evidence_ids'] == joint['evidence_ids'] and proposal_joint['parent_id'] == p and proposal_joint['child_id'] == c and
                proposal_joint['joint_type'] == 'revolute' and proposal_joint['attachment'] == 'persistent', 'Joint not grounded in semantic proposal')
        axis = np.asarray(joint['axis'])
        require(np.isclose(np.linalg.norm(axis), 1.), 'Joint axis must be a finite unit vector')
        sp, sc = joint['parent_native_scale'], joint['child_native_scale']
        require(np.isclose(sp, components[p]['native_uniform_scale']) and np.isclose(sc, components[c]['native_uniform_scale']), 'Joint native scale mismatch')
        a = _similarity(joint['parent_from_joint'], 1/sp)
        b = _similarity(joint['joint_from_child_zero'], sc)
        require(np.allclose(a[:3,3], joint['parent_anchor_native'], atol=1e-8) and
                np.allclose(b@np.r_[joint['child_anchor_native'],1.], [0,0,0,1], atol=1e-8),
                'Joint frames do not map declared native anchors to their shared origin')
        for cid, field in [(p,'parent'),(c,'child')]:
            vertices = native_vertices(root/components[cid]['asset']['path'])
            vertex_id = joint[field+'_anchor_vertex_id']
            require(vertex_id < len(vertices) and np.allclose(vertices[vertex_id], joint[field+'_anchor_native'], atol=1e-8),
                    'Joint anchor is not its declared unchanged native mesh vertex')
        poses = {r['source_sample_index']: r for r in components[c]['poses']}
        parent_poses = {r['source_sample_index']: r for r in components[p]['poses']}
        expected_rows = source_scene['components'][next(i for i,c0 in enumerate(source_scene['components']) if c0['id'] == c)]['poses']
        require(len(joint['samples']) == len(expected_rows), 'Angle curve must preserve all source samples')
        angles, previous = [], -1.
        for sample, original in zip(joint['samples'], expected_rows):
            index, time = sample['source_sample_index'], sample['timestamp_s']
            require(index == original['source_sample_index'] and time == original['timestamp_s'] and previous < time <= program['source']['duration_s'], 'Angle source sample/time mismatch')
            previous = time; child, parent = poses[index], parent_poses[index]
            if sample['status'] == 'unknown':
                require(sample['q_rad'] is None and child['camera_from_native_mesh'] is None and child['status'] == 'unknown', 'Unknown angle must withhold child pose')
                continue
            require(original['status'] == 'estimated', 'Joint cannot fill a withheld source estimate')
            require(type(sample['q_rad']) in (int, float) and math.isfinite(sample['q_rad']) and parent['status'] == child['status'] == 'estimated', 'Invalid estimated angle or unavailable parent')
            angles.append(sample['q_rad'])
            rotation = np.eye(4); rotation[:3,:3] = axis_rotation(axis,sample['q_rad'])
            expected = np.asarray(parent['camera_from_native_mesh'])@a@rotation@b
            require(np.allclose(child['camera_from_native_mesh'], expected, atol=1e-8), 'Child pose differs from declared forward kinematics')
            pa = np.asarray(parent['camera_from_native_mesh'])@np.r_[joint['parent_anchor_native'],1.]
            ca = expected@np.r_[joint['child_anchor_native'],1.]
            require(np.allclose(pa, ca, atol=1e-8), 'Persistent mesh anchors separate')
        require(angles and np.allclose(joint['limits_rad'], [min(angles),max(angles)]), 'Limits must be observed estimated angle range')
        zero=joint['zero_reference']
        zero_sample=next((s for s in joint['samples'] if s['source_sample_index']==zero['source_sample_index']),None)
        require(zero_sample and zero_sample['timestamp_s']==zero['timestamp_s'] and zero_sample['status']=='estimated' and
                np.isclose(zero_sample['q_rad'],0.,atol=1e-8),'Joint zero reference differs from source angle curve')
        if zero['state']=='estimated_closed':
            require(any(s['source_sample_index']==zero['source_sample_index'] and s['timestamp_s']==zero['timestamp_s'] and
                        s['state']=='closed' for s in proposal_joint.get('state_hints',[])),
                    'Closed zero reference lacks cited model state evidence')
        quality=joint['angle_quality']
        require(np.isclose(quality['declared_separation_threshold_rad'],math.pi/12),'Unknown angle interpretation test threshold')
        samples_by_index={s['source_sample_index']:s for s in joint['samples']}
        expected_comparisons=[]
        for hint in proposal_joint.get('state_hints',[]):
            sample=samples_by_index.get(hint['source_sample_index'])
            if hint['state'] not in {'open','closed'} or not sample or sample['status']!='estimated':continue
            require(hint['timestamp_s']==sample['timestamp_s'],'Angle interpretation state/source time mismatch')
            expected_comparisons.append({'source_sample_index':sample['source_sample_index'],'timestamp_s':sample['timestamp_s'],
                'cited_state':hint['state'],'q_rad':sample['q_rad'],'distance_from_closed_reference_rad':abs(sample['q_rad'])})
        require(quality['state_comparisons']==expected_comparisons,'Angle interpretation comparisons differ from source curve/state citations')
        open_samples=[s for s in expected_comparisons if s['cited_state']=='open']
        consistent=bool(zero['state']=='estimated_closed' and open_samples and all(s['distance_from_closed_reference_rad']>math.pi/12 for s in open_samples))
        expected_interpretation='estimated_opening_configuration_not_physically_calibrated' if consistent else 'model_configuration_parameter_not_verified_opening_angle'
        require(quality['angle_visual_consistency'] is consistent and quality['angle_interpretation']==expected_interpretation,
                'Angle interpretation cannot conceal contradictory cited open/closed states')
    for pending in document['unresolved_connections']:
        p,c=pending['parent_id'],pending['child_id']
        require(p in tracks and c in tracks and p!=c and c not in parents and pending['id'] not in seen,
                'Unknown, duplicate or ambiguous unresolved connection')
        parents[c]=p;seen.add(pending['id'])
        original=next((j for j in proposal['joints'] if j['id']==pending['id']),None)
        require(original and original['attachment']=='persistent' and original['parent_id']==p and original['child_id']==c and
                original['joint_type']==pending['type'],'Unresolved connection lacks exact semantic proposal')
        if c in components:
            rows=components[c]['poses']; original_rows=original_components[c]['poses']
            require(len(rows)==len(original_rows) and all(r['status']=='unknown' and r['camera_from_native_mesh'] is None and
                r['source_sample_index']==old['source_sample_index'] and r['timestamp_s']==old['timestamp_s'] for r,old in zip(rows,original_rows)),
                'Unresolved persistent child must remain absent at every source sample')
    persistent={j['id'] for j in proposal['joints'] if j['attachment']=='persistent'}
    require(seen==persistent,'Persistent hypotheses must be fitted or explicitly unresolved')
    for node in parents:
        visited=set()
        while node in parents:
            require(node not in visited,'Cyclic fitted/unresolved mechanism graph')
            visited.add(node);node=parents[node]
    expected_removable = [{'id': j['id'], 'parent_id': j['parent_id'], 'child_id': j['child_id'], 'joint_type': 'removable',
                          'attachment': 'removable', 'confidence': j['confidence'], 'evidence_ids': j['evidence_ids'],
                          'physical_accuracy_verified': False} for j in proposal['joints'] if j['joint_type'] == 'removable']
    require(document['removable_relationships'] == expected_removable, 'Removable relationships differ from semantic proposal')
    return True


def build(scene_path, output, *, root=ROOT):
    """Create a separately versioned descriptive export; original programs stay intact."""
    root = Path(root).resolve()
    path = data_path(root, scene_path)
    scene = read_json(path)
    mechanics_path = path.parent/'mechanics.json'
    mechanics = read_json(mechanics_path)
    require(scene.get('mechanics') == mechanics, 'Scene/mechanics companion mismatch')
    proposal = read_json(checked_ref(mechanics['proposal'], root))
    out = data_path(root, output)
    require(out.name == 'object-program.json' and out.parent.parent == root/'data/object-programs' and not out.parent.exists(),
            'Fresh data/object-programs/<version>/object-program.json output required')
    out.parent.mkdir(parents=True)
    provenance = []
    reason_plan = read_json(data_path(root, proposal['reasoning']['plan_path']))
    for item in [*implementation_refs(read_json(checked_ref(mechanics['source_scene'], root)),mechanics),
                 {'path': 'object_program.py', 'sha256': sha(__file__), 'bytes': Path(__file__).stat().st_size},
                 {'path': 'mechanism_reasoning.py', 'sha256': reason_plan['contract']['code_sha256'], 'bytes': (root/'mechanism_reasoning.py').stat().st_size}]:
        require(item['path'] in CODE and not (root/item['path']).is_symlink() and (root/item['path']).is_file() and
                sha(root/item['path']) == item['sha256'] and (root/item['path']).stat().st_size==item['bytes'], 'Current reviewed implementation differs from saved provenance')
        target = out.parent/'provenance'/(item['path']+'.txt')
        target.parent.mkdir(exist_ok=True)
        with target.open('xb') as stream:
            stream.write((root/item['path']).read_bytes())
        provenance.append({'original_path': item['path'], 'sha256': item['sha256'], 'archived_file': ref(target,root)})
    document = {'format': FORMAT, 'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'evidence_program': scene['original_program'], 'source': scene['source'],
                'source_scene': mechanics['source_scene'], 'constrained_scene': ref(path,root), 'mechanics': ref(mechanics_path,root),
                'proposal': mechanics['proposal'], 'component_assets': [{'component_id':c['id'], 'native_glb':c['asset'],
                    'native_metadata':c['native_metadata'], 'native_uniform_scale':c['native_uniform_scale']} for c in scene['components']],
                'joints': [projected_joint(j) for j in mechanics['joints']], 'unresolved_connections':mechanics['unresolved_children'],
                'removable_relationships': [{'id':j['id'], 'parent_id':j['parent_id'], 'child_id':j['child_id'], 'joint_type':'removable',
                    'attachment':'removable', 'confidence':j['confidence'], 'evidence_ids':j['evidence_ids'], 'physical_accuracy_verified':False}
                    for j in proposal['joints'] if j['joint_type']=='removable'],
                'coordinate_system': {'camera_frame':'opencv_x_right_y_down_z_forward', 'length_units':'inferred_arbitrary_scene_units_no_metric_scale',
                    'angle_units':'radians', 'unknown_sample_policy':'withhold_joint_and_child_when_parent_or_joint_unknown', 'physical_stops':'unknown'},
                'implementation_provenance':provenance, 'physical_accuracy_verified':False, 'simulation_ready':False,
                'robot_execution_available':False, 'manual_joint_coordinates_used':False,
                'limitations':['Joint axes, anchors and angles are inferred estimates, not measured mechanism parameters.',
                    'Angle limits are the estimated observed clip range; physical stops, contact and insertion remain unknown.',
                    'Removable relations are fallible Muse hypotheses, not persistent joints or verified identity.',
                    'The camera-relative frame has inferred nonmetric depth and is not a stationary calibrated world.',
                    'This is an agent-readable descriptive program, not verified physics or robot execution.']}
    validate(document,root=root)
    with out.open('x') as stream:
        json.dump(document,stream,indent=2,allow_nan=False);stream.write('\n')
    return ref(out,root)


def discover(root, program_sha256, source_sha256):
    """Inspect only the declared companion namespace, never run a fitter/model."""
    root = Path(root).resolve()
    base = root/'data/object-programs'
    if not base.exists():
        return None
    paths = sorted(base.glob('*/object-program.json'))
    require(len(paths) <= 128, 'Object-program catalog exceeds bounded inspection budget')
    found = []
    for path in paths:
        try:
            doc = read_json(data_path(root,str(path.relative_to(root))))
            if doc.get('evidence_program',{}).get('sha256') != program_sha256 or doc.get('source',{}).get('sha256') != source_sha256:
                continue
            validate(doc,root=root)
            date = datetime.fromisoformat(doc['created_utc'].replace('Z','+00:00'))
            require(date.tzinfo is not None, 'Object-program timestamp must include timezone')
            found.append((date.timestamp(),str(path.relative_to(root)),doc))
        except (ValueError, OSError, KeyError, TypeError, jsonschema.ValidationError):
            continue
    if not found:
        return None
    _,relative,doc = sorted(found,key=lambda row:(row[0],row[1]))[-1]
    return {'artifact':ref(root/relative,root),'document':doc}
