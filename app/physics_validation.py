"""Source-bound physical-input preflight; never a physical certification.

This module uses only installed software. Unknown physical values remain null.
Mesh topology is inspected without repair/welding and never converted into mass,
collision acceptance, joint axes or measured scale. The Blender synthetic smoke
is a separate software test, not evidence about the reconstructed objects.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct

FORMAT = 'interaction-twin-physical-validation-input'
VERSION = 1
HASH = re.compile(r'[a-f0-9]{64}\Z')
ID = re.compile(r'[a-zA-Z][a-zA-Z0-9_-]{0,159}\Z')
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_MESH_BYTES = 16 * 1024 * 1024
MAX_VERTICES = 500_000
MAX_FACES = 1_000_000
LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Non-finite physical input')
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def _confined_path(root, relative, *, prefix, maximum):
    root = Path(root).resolve()
    if (not isinstance(relative, str) or not relative.startswith(prefix) or
            any(char in relative for char in ['\\', '?', '#', '\x00', ':']) or
            any(not part or part.startswith('.') for part in relative.split('/'))):
        raise ValueError('Only canonical non-hidden data paths may be opened')
    candidate = root / relative
    if (candidate.resolve() != candidate or not candidate.is_file() or
            not candidate.is_relative_to(root) or candidate.stat().st_size > maximum):
        raise ValueError('Data artifact is missing, symlinked or oversized')
    return candidate


def data_path(root, relative, *, maximum=MAX_ARTIFACT_BYTES):
    """No absolute, hidden, symlink, URL or traversing paths; data only."""
    return _confined_path(root, relative, prefix='data/', maximum=maximum)


def program_artifact_path(root, item):
    """Only source-declared data, annotations and exact compiler dependencies."""
    relative = item.get('path')
    if isinstance(relative, str) and relative.startswith('data/'):
        return data_path(root, relative)
    if (item.get('kind') == 'model_tracklet_proposals' and isinstance(relative, str)
            and relative.startswith('annotations/') and relative.endswith('.json')):
        return _confined_path(root, relative, prefix='annotations/', maximum=1024 * 1024)
    allowed = {'deterministic_compiler_code': 'stage5.py',
               'deterministic_compiler_orchestration_code': 'scripts/stage5_compile.py',
               'schema': 'schemas/interaction.schema.json'}
    if relative == allowed.get(item.get('kind')) and relative is not None:
        return _confined_path(root, relative, prefix=relative, maximum=1024 * 1024)
    raise ValueError('Unsupported program dependency; arbitrary filesystem reads are prohibited')


def verified(root, reference, *, maximum=MAX_ARTIFACT_BYTES):
    if not isinstance(reference, dict) or not HASH.fullmatch(reference.get('sha256', '')):
        raise ValueError('A pinned artifact hash is required')
    path = data_path(root, reference.get('path'), maximum=maximum)
    if sha(path) != reference['sha256']:
        raise ValueError('Artifact hash mismatch')
    return path


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError('Non-finite JSON constant')
    result = json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=bad_constant)
    finite(result)
    return result


def program_artifacts(document):
    """Explicit identity graph only; never interpret labels as paths or commands."""
    yield document['source']
    yield from document.get('inputs', [])
    yield from document['assets']
    for frame in document['evidence']:
        yield {'path': frame['frame_path'], 'sha256': frame['frame_sha256']}
        for observation in frame['observations']:
            yield {'path': observation['mask_path'], 'sha256': observation['mask_sha256']}


def load_program(root, reference):
    path = verified(root, reference, maximum=8 * 1024 * 1024)
    document = read_json(path)
    # Check paths BEFORE stage5's structural validator. This branch intentionally
    # does not recursively open arbitrary model-output paths in upstream JSON.
    refs = list(program_artifacts(document))
    for item in refs:
        program_artifact_path(root, item)
    from stage5 import validate_interaction
    validate_interaction(document, root=root, verify_files=False)
    seen = set()
    for item in refs:
        key = (item['path'], item['sha256'])
        if key not in seen:
            if not HASH.fullmatch(item.get('sha256', '')) or sha(program_artifact_path(root, item)) != item['sha256']:
                raise ValueError('Program artifact hash mismatch')
            seen.add(key)
    return document, len(seen)


def glb_document(payload):
    """Reject external resource resolution before any optional mesh library load."""
    if len(payload) < 20 or len(payload) > MAX_MESH_BYTES:
        raise ValueError('GLB outside bounded size')
    magic, version, length = struct.unpack_from('<4sII', payload)
    if magic != b'glTF' or version != 2 or length != len(payload):
        raise ValueError('Malformed GLB header')
    offset, document, binary_chunks = 12, None, 0
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise ValueError('Truncated GLB chunk header')
        size, kind = struct.unpack_from('<II', payload, offset)
        offset += 8
        if size % 4 or offset + size > len(payload):
            raise ValueError('Invalid GLB chunk length')
        chunk = payload[offset:offset + size]
        if kind == 0x4E4F534A:
            if document is not None or offset != 20:
                raise ValueError('GLB requires exactly one leading JSON chunk')
            document = json.loads(chunk.decode('utf-8'))
        elif kind == 0x004E4942:
            binary_chunks += 1
        else:
            raise ValueError('Unexpected GLB chunk')
        offset += size
    if document is None or binary_chunks != 1:
        raise ValueError('Embedded GLB geometry required')
    finite(document)
    # URI could otherwise trigger local reads/network in an importer. Data URLs
    # are rejected too: bounded, wholly embedded bufferViews only.
    def scan(value):
        if isinstance(value, dict):
            if 'uri' in value:
                raise ValueError('External or URI GLB resources are not allowed')
            for item in value.values():
                scan(item)
        elif isinstance(value, list):
            for item in value:
                scan(item)
    scan(document)
    return document


def inspect_mesh(root, reference):
    """Computational raw-index topology, not measured physical shape accuracy."""
    import numpy as np
    import trimesh
    path = verified(root, reference, maximum=MAX_MESH_BYTES)
    payload = path.read_bytes()
    document = glb_document(payload)
    for accessor in document.get('accessors', []):
        count = accessor.get('count')
        if type(count) is not int or count < 0 or count > MAX_FACES * 3:
            raise ValueError('Unbounded GLB accessor')
    scene = trimesh.load(io.BytesIO(payload), file_type='glb', force='scene',
                         process=False, skip_materials=True)
    # to_mesh applies original GLB node transforms on a COPY. No input mutation,
    # mesh repair, normal fixing, welding, hull generation or inferred thickness.
    mesh = scene.to_mesh()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or
            faces.ndim != 2 or faces.shape[1:] != (3,) or
            not 0 < len(vertices) <= MAX_VERTICES or not 0 < len(faces) <= MAX_FACES or
            not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('Invalid or oversized native triangle mesh')
    bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)])
    extent = bounds[1] - bounds[0]
    diagonal = float(np.linalg.norm(extent))
    if diagonal <= 0:
        raise ValueError('Collapsed native mesh')
    areas2 = np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                     vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1)
    area_threshold = 1e-12 * diagonal ** 2
    degenerate = int((areas2 <= area_threshold).sum())
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    parent = np.arange(len(vertices))
    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item
    for a, b in unique_edges:
        ra, rb = find(int(a)), find(int(b))
        parent[ra] = rb
    used = np.unique(faces)
    connected = len({int(find(int(item))) for item in used})
    duplicate_faces = len(faces) - len(np.unique(np.sort(faces, axis=1), axis=0))
    result = {
        'artifact': copy.deepcopy(reference), 'vertices': len(vertices), 'triangles': len(faces),
        'node_count': len(scene.graph.nodes_geometry), 'native_bounds': bounds.tolist(),
        'native_extents': extent.tolist(), 'units': 'unknown_native_units',
        'triangle_area2_degeneracy_threshold_native_units2': area_threshold,
        'degenerate_triangle_count': degenerate, 'duplicate_index_triangle_count': int(duplicate_faces),
        'raw_index_boundary_edge_count': int((counts == 1).sum()),
        'raw_index_nonmanifold_edge_count': int((counts > 2).sum()),
        'raw_index_connected_components': connected,
        'raw_index_watertight': bool(np.all(counts == 2)),
        'winding_consistent': bool(mesh.is_winding_consistent),
        'input_unchanged': sha(path) == reference['sha256'],
        'processing': {'original_glb_node_transforms_applied_to_copy': True,
                       'repair': False, 'weld': False, 'convex_hull': False, 'scale_estimation': False},
        'physical_accuracy_verified': False, 'collision_geometry_accepted': False,
        'self_intersection_tested': False, 'cavity_clearance_tested': False,
        'limitations': ['Raw index seams may appear as boundaries; no vertices were welded.',
                        'Watertight topology does not establish correct cavities, wall thickness or real dimensions.',
                        'No inferred face is automatically accepted as collision geometry.'],
    }
    if not result['input_unchanged']:
        raise ValueError('Mesh changed during inspection')
    return result


def build_contract(root, interaction_reference):
    document, count = load_program(root, interaction_reference)
    by_track = {}
    for asset in document['assets']:
        by_track.setdefault(asset['track_id'], []).append(asset)
    components = []
    for track in document['tracklets']:
        assets = by_track.get(track['id'], [])
        components.append({
            'id': track['id'], 'label': track['label'], 'identity_status': track['identity_status'],
            'visual_assets': [copy.deepcopy(a) for a in assets],
            'meters_per_native_unit': None, 'body_from_native_mesh': None,
            'world_from_body_m': None, 'collision_geometry': None,
            'mass_kg': None, 'center_of_mass_m': None, 'inertia_kg_m2': None,
            'contact_material': None, 'physical_input_evidence': [],
        })
    return {
        'format': FORMAT, 'version': VERSION, 'scope': 'passive_rigid_body_preflight',
        'case_id': document['case_id'], 'interaction': copy.deepcopy(interaction_reference),
        'source_video': {k: document['source'][k] for k in ['path', 'sha256']},
        'components': components,
        'assembly': {'world_axis_convention': None, 'common_frame_evidence': None,
                     'joint_definitions': None, 'initial_state_evidence': None},
        'trial': {'gravity_m_s2': None, 'timestep_s': None, 'steps': None,
                  'environment_collision_geometry': None, 'success_criteria': None,
                  'failure_criteria': None},
        'proposed_relationships': copy.deepcopy(document['relationships']),
        'source_identity_verification': {'declared_artifact_hashes_verified': count,
                                         'upstream_nested_model_outputs_recursively_validated': False},
        'simulation_ready': False, 'physical_accuracy_verified': False,
        'physical_execution_authorized': False,
    }


def _number(value, *, positive=False, nonnegative=False):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError('Expected a finite physical number, not a boolean/string')
    if (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError('Physical value is outside its admissible range')


def _vector(value, n):
    if not isinstance(value, list) or len(value) != n:
        raise ValueError('Invalid physical vector shape')
    for item in value:
        _number(item)


def _rigid(value):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError('Transform must be a 4x4 matrix')
    for row in value:
        _vector(row, 4)
    if any(abs(a - b) > 1e-9 for a, b in zip(value[3], [0, 0, 0, 1])):
        raise ValueError('Invalid homogeneous transform last row')
    r = [row[:3] for row in value[:3]]
    for i in range(3):
        for j in range(3):
            if abs(sum(r[k][i] * r[k][j] for k in range(3)) - (i == j)) > 1e-6:
                raise ValueError('Transform rotation must be orthonormal; scale is separate')
    determinant = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) -
                   r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) +
                   r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
    if abs(determinant - 1) > 1e-6:
        raise ValueError('Physical rotation must have positive unit determinant')


def _inertia(value):
    """SPD alone is insufficient: principal moments need triangle inequalities."""
    import numpy as np
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError('Expected 3x3 inertia tensor in kg m^2')
    for row in value:
        _vector(row, 3)
    matrix = np.asarray(value, float)
    scale = float(np.max(np.abs(matrix)))
    if not np.allclose(matrix, matrix.T, rtol=0, atol=scale * 1e-10):
        raise ValueError('Inertia must be symmetric')
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[0] <= 0 or eigenvalues[2] > eigenvalues[:2].sum() + scale * 1e-10:
        raise ValueError('Inertia must be positive and physically realizable')


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(label + ' has missing or unsupported fields')


def validate_contract(root, contract):
    """Bind exact source identities; validate supplied values; list absent inputs.

    A structurally complete proposal still needs a reviewed experiment adapter.
    This never executes Blender or certifies physical truth.
    """
    finite(contract)
    _keys(contract, ['format', 'version', 'scope', 'case_id', 'interaction', 'source_video',
                     'components', 'assembly', 'trial', 'proposed_relationships',
                     'source_identity_verification', 'simulation_ready',
                     'physical_accuracy_verified', 'physical_execution_authorized'], 'Contract')
    if (contract.get('format') != FORMAT or type(contract.get('version')) is not int or contract.get('version') != VERSION or
            contract.get('scope') != 'passive_rigid_body_preflight'):
        raise ValueError('Unsupported physical-input contract')
    for flag in ['simulation_ready', 'physical_accuracy_verified', 'physical_execution_authorized']:
        if contract.get(flag) is not False:
            raise ValueError('Preflight cannot promote physical readiness or execution permission')
    document, artifact_count = load_program(root, contract['interaction'])
    verification = contract.get('source_identity_verification')
    _keys(verification, ['declared_artifact_hashes_verified',
                         'upstream_nested_model_outputs_recursively_validated'], 'Source verification')
    if (type(verification['declared_artifact_hashes_verified']) is not int or
            verification['declared_artifact_hashes_verified'] != artifact_count or
            verification['upstream_nested_model_outputs_recursively_validated'] is not False):
        raise ValueError('Source verification counts or transitive claims changed')
    if (contract.get('case_id') != document['case_id'] or
            contract.get('source_video') != {k: document['source'][k] for k in ['path', 'sha256']}):
        raise ValueError('Physical contract source/program binding differs')
    rows = contract.get('components')
    if not isinstance(rows, list) or len(rows) > 32 or not all(isinstance(row, dict) for row in rows):
        raise ValueError('Expected bounded components')
    expected = {t['id']: t for t in document['tracklets']}
    if len({c.get('id') for c in rows}) != len(rows) or {c.get('id') for c in rows} != set(expected):
        raise ValueError('Physical component IDs must exactly match source tracklets')
    gaps = []
    def missing(component, field, detail):
        gaps.append({'component_id': component, 'field': field, 'reason': detail})
    for row in rows:
        _keys(row, ['id', 'label', 'identity_status', 'visual_assets', 'meters_per_native_unit',
                    'body_from_native_mesh', 'world_from_body_m', 'collision_geometry', 'mass_kg',
                    'center_of_mass_m', 'inertia_kg_m2', 'contact_material', 'physical_input_evidence'], 'Component')
        key = row['id']
        expected_assets = [a for a in document['assets'] if a['track_id'] == key]
        if (row.get('visual_assets') != expected_assets or row.get('label') != expected[key]['label'] or
                row.get('identity_status') != expected[key]['identity_status']):
            raise ValueError('Visual asset or component identity was silently changed')
        if not expected_assets:
            missing(key, 'visual_assets', 'No generated visual asset exists for this source tracklet.')
        missing(key, 'physical_identity', 'Source tracklet identity has not been physically verified.')
        for field in ['meters_per_native_unit', 'mass_kg']:
            if row.get(field) is not None:
                _number(row[field], positive=True)
        for field in ['body_from_native_mesh', 'world_from_body_m']:
            if row.get(field) is not None:
                _rigid(row[field])
        if row.get('center_of_mass_m') is not None:
            _vector(row['center_of_mass_m'], 3)
        if row.get('inertia_kg_m2') is not None:
            _inertia(row['inertia_kg_m2'])
        material = row.get('contact_material')
        if material is not None:
            if set(material) != {'friction', 'restitution'}:
                raise ValueError('Contact material requires explicit friction and restitution')
            _number(material['friction'], nonnegative=True)
            _number(material['restitution'], nonnegative=True)
            if material['restitution'] > 1:
                raise ValueError('Restitution must lie between zero and one')
        collision = row.get('collision_geometry')
        if collision is not None:
            if not isinstance(collision, dict) or set(collision) != {'artifact', 'representation', 'acceptance_evidence'}:
                raise ValueError('Collision shape needs explicit artifact/representation/evidence')
            collision_path = verified(root, collision['artifact'], maximum=MAX_MESH_BYTES)
            if collision_path.suffix.lower() != '.glb':
                raise ValueError('Prepared collision representation requires an embedded GLB')
            glb_document(collision_path.read_bytes())
            if collision['representation'] not in {'reviewed_convex_pieces', 'reviewed_triangle_surface'}:
                raise ValueError('Do not silently hull a cavity or use a visual asset as a collider')
            verified(root, collision['acceptance_evidence'], maximum=1024 * 1024)
            missing(key, 'collision_physical_review', 'Evidence identity is checked; collision adequacy is not independently established.')
        fields = ['meters_per_native_unit', 'body_from_native_mesh', 'world_from_body_m',
                  'collision_geometry', 'mass_kg', 'center_of_mass_m', 'inertia_kg_m2', 'contact_material']
        for field in fields:
            if row.get(field) is None:
                missing(key, field, 'Missing; no simulator default or video-derived guess is substituted.')
        evidence = row.get('physical_input_evidence')
        if not isinstance(evidence, list) or len(evidence) > 32:
            raise ValueError('Physical evidence must be an explicit bounded array')
        if not evidence:
            missing(key, 'physical_input_evidence', 'No measurements or calibrated physical parameter evidence.')
        for item in evidence:
            if not isinstance(item, dict) or set(item) != {'field', 'method', 'artifact'}:
                raise ValueError('Physical evidence requires field/method/pinned artifact')
            if item['field'] not in fields or item['method'] not in {'measured', 'calibrated', 'declared_estimate_unverified'}:
                raise ValueError('Unsupported physical evidence claim')
            verified(root, item['artifact'], maximum=1024 * 1024)
        supplied = {field for field in fields if row.get(field) is not None}
        covered = {item['field'] for item in evidence}
        for field in sorted(supplied - covered):
            missing(key, field + '_provenance', 'Supplied number has no pinned measurement/calibration/estimate record.')
    assembly = contract.get('assembly')
    trial = contract.get('trial')
    _keys(assembly, ['world_axis_convention', 'common_frame_evidence', 'joint_definitions',
                     'initial_state_evidence'], 'Assembly')
    _keys(trial, ['gravity_m_s2', 'timestep_s', 'steps', 'environment_collision_geometry',
                 'success_criteria', 'failure_criteria'], 'Trial')
    if assembly.get('world_axis_convention') not in (None, 'right_handed_z_up_meters'):
        raise ValueError('Only explicit metric right-handed Z-up adapter convention is supported')
    # Version1 deliberately does not accept a user-filled joint block yet. A
    # null block is unknown, [] explicitly proposes no joints; neither defines
    # physical relations for our existing multi-part program.
    joints = assembly.get('joint_definitions')
    if joints not in (None, []):
        raise ValueError('Joint execution is unsupported until a measured joint adapter is implemented')
    for field in ['common_frame_evidence', 'initial_state_evidence']:
        if assembly.get(field) is not None:
            verified(root, assembly[field], maximum=1024 * 1024)
    for field in ['world_axis_convention', 'common_frame_evidence', 'joint_definitions', 'initial_state_evidence']:
        if assembly.get(field) is None:
            missing(None, 'assembly.' + field, 'Common assembly and joint status must be explicitly established.')
    if document['relationships']:
        missing(None, 'assembly.mechanism_validation', 'Visual attached/removable relations do not establish joints, seating or common assembly.')
    if trial.get('gravity_m_s2') is not None:
        _vector(trial['gravity_m_s2'], 3)
    if trial.get('timestep_s') is not None:
        _number(trial['timestep_s'], positive=True)
        if not 1 / 2000 <= trial['timestep_s'] <= 1 / 30:
            raise ValueError('Timestep outside bounded adapter allowance')
    if trial.get('steps') is not None and (type(trial['steps']) is not int or not 1 <= trial['steps'] <= 2000):
        raise ValueError('Trial requires at most 2000 integer steps')
    if trial.get('environment_collision_geometry') is not None:
        environment = verified(root, trial['environment_collision_geometry'], maximum=MAX_MESH_BYTES)
        if environment.suffix.lower() != '.glb':
            raise ValueError('Environment collision candidate requires an embedded GLB')
        glb_document(environment.read_bytes())
    for group in ['success_criteria', 'failure_criteria']:
        criteria = trial.get(group)
        if criteria is not None:
            if not isinstance(criteria, list) or not 1 <= len(criteria) <= 8:
                raise ValueError('Trial criteria must be a nonempty bounded list')
            for criterion in criteria:
                _keys(criterion, ['metric', 'operator', 'threshold'], 'Criterion')
                if criterion['metric'] not in {'maximum_penetration_m', 'linear_speed_m_s', 'rest_height_error_m'}:
                    raise ValueError('Unsupported passive-trial metric')
                if criterion['operator'] not in {'less_than', 'greater_than'}:
                    raise ValueError('Unsupported trial criterion operator')
                _number(criterion['threshold'], nonnegative=True)
    for field in ['gravity_m_s2', 'timestep_s', 'steps', 'environment_collision_geometry',
                  'success_criteria', 'failure_criteria']:
        if trial.get(field) is None:
            missing(None, 'trial.' + field, 'Explicit trial input absent.')
    if contract.get('proposed_relationships') != document['relationships']:
        raise ValueError('Source semantic relationship proposals must remain unchanged')
    return {'format': 'interaction-twin-physical-input-preflight', 'version': 1,
            'case_id': document['case_id'], 'status': 'blocked', 'blocking_inputs': gaps,
            'declared_artifact_hashes_verified': artifact_count,
            'contract_values_checked': True, 'simulator_executed_for_components': False,
            'physical_accuracy_verified': False, 'simulation_ready': False,
            'adapter': {'backend_candidate': 'installed_blender_bullet_cpu',
                        'actual_object_execution_implemented': False,
                        'status': 'preparation_only', 'maximum_steps': 2000,
                        'automatic_parameter_defaults': False,
                        'joint_execution_supported': False,
                        'reason': 'Review measured inputs and implement a task-specific frozen trial before running real objects.'},
            'limitations': ['Schema/hash/numeric checks do not validate measurement truth.',
                            'Software fixture success cannot validate these physical objects.',
                            'Conditional video projection agreement is not a contact or physics accuracy test.']}


def audit_program(root, reference):
    contract = build_contract(root, reference)
    preflight = validate_contract(root, contract)
    geometry = []
    for row in contract['components']:
        for asset in row['visual_assets']:
            geometry.append({'component_id': row['id'], 'geometry_role': 'visual_only',
                             **inspect_mesh(root, {'path': asset['path'], 'sha256': asset['sha256']})})
    return {'contract': contract, 'preflight': preflight, 'native_mesh_diagnostics': geometry,
            'module_sha256': LOADED_SOURCE_SHA256,
            'new_api_calls': 0, 'model_inference': False, 'physical_accuracy_verified': False,
            'simulation_ready': False, 'original_assets_modified': False}


def audit_reconstruction_version(root, version_id):
    """Use the existing strict read-only catalog; preserve every layer's role."""
    from reconstruction_catalog import ReconstructionCatalog
    catalog = ReconstructionCatalog(root)
    row = next((item for item in catalog.rows() if item['version_id'] == version_id), None)
    if row is None:
        raise ValueError('Unknown reconstruction version')
    program_path = data_path(root, row['interaction_path'], maximum=8 * 1024 * 1024)
    document = read_json(program_path)
    program_hash = sha(program_path)
    version = catalog.load(version_id, interaction_path=row['interaction_path'],
                           document=document, program_sha256=program_hash)
    geometry = inspect_mesh(root, version['artifacts']['glb'])
    return {'version_id': version_id, 'component_id': version['component_id'],
            'case_id': version['case_id'], 'kind': version['kind'],
            'descriptor': {'path': row['descriptor_path'], 'sha256': row['descriptor_sha256']},
            'interaction': {'path': row['interaction_path'], 'sha256': program_hash},
            'source_video_sha256': version['source_video_sha256'],
            'catalog_source_and_artifact_identities_verified': True,
            'quality': copy.deepcopy(version['artifacts']['quality']),
            'gauge': copy.deepcopy(version['gauge']), 'acceptance': version['acceptance'],
            'geometry_role': 'diagnostic_inferred_layer' if version['kind'] == 'inferred' else 'diagnostic_observed_patch',
            'native_mesh_diagnostics': geometry, 'physical_accuracy_verified': False,
            'collision_geometry_accepted': False, 'simulation_ready': False}


def synthetic_box_floor_recheck(fixture):
    """Recompute true rotated-box lowest corners from preserved synthetic poses.

    This is a derived geometry check, not another physics-engine execution.
    The original center-only diagnostic/receipt remains unchanged.
    """
    if (fixture.get('fixture_only') is not True or fixture.get('real_reconstructed_assets_used') is not False or
            fixture.get('format') != 'blender-bullet-synthetic-software-validation'):
        raise ValueError('Expected separate synthetic fixture receipt')
    definition = fixture['fixture_definition']
    side, floor = definition['box_side_m'], definition['floor_top_z_m']
    _number(side, positive=True)
    _number(floor)
    samples = fixture.get('samples')
    if not isinstance(samples, list) or not 1 <= len(samples) <= 2001:
        raise ValueError('Expected bounded saved synthetic poses')
    bottoms = []
    for row in samples:
        _vector(row['position_m'], 3)
        _vector(row['rotation_quaternion_wxyz'], 4)
        q = row['rotation_quaternion_wxyz']
        norm = math.sqrt(sum(a * a for a in q))
        if abs(norm - 1) > 1e-5:
            raise ValueError('Non-unit saved synthetic quaternion')
        w, x, y, z = [a / norm for a in q]
        rotation_row_z = [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
        # Minimum over all eight box corners: subtract the absolute projected
        # half extents. This remains valid while the cube tilts during contact.
        bottoms.append(row['position_m'][2] - side / 2 * sum(abs(a) for a in rotation_row_z) - floor)
    minimum = min(bottoms)
    return {'method': 'all_eight_rotated_box_corners_from_saved_rigid_poses',
            'derived_only_no_solver_rerun': True, 'samples_checked': len(samples),
            'quaternion_normalization': 'unit_length_numeric_conversion_only',
            'minimum_corner_clearance_above_floor_m': minimum,
            'maximum_penetration_m': max(0, -minimum), 'frozen_tolerance_m': .005,
            'floor_penetration_below_5mm': minimum > -.005,
            'supersedes_original_center_only_minimum_m': fixture['minimum_box_bottom_z_m'],
            'physical_accuracy_verified': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--interaction', required=True, help='Canonical data/.../interaction.json')
    parser.add_argument('--output', required=True, type=Path, help='Fresh diagnostic JSON; never overwrite')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output exists; use a new version')
    source = data_path(args.root, args.interaction, maximum=8 * 1024 * 1024)
    report = audit_program(args.root, {'path': args.interaction, 'sha256': sha(source)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
