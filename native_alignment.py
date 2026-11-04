"""Receipts shared by the versioned native-material alignment adapters.

No reconstruction, recolouring, remeshing or pose correction happens here. The
same flattened native scene interpretation is bound across the reference fit and
tracking so material/instance ordering cannot silently change a ray query.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CONTRACT = 'native-base-color-v1'
LEGACY_ORIGINS = {
    'stage4_native_pose.py': 'stage4_generated_pose.py',
    'stage4_native_track.py': 'stage4_track_generated.py',
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_representation_receipt(mesh):
    """Bind native decoded geometry ordering and loader bytes, not just a GLB."""
    vertices = np.ascontiguousarray(mesh.vertices, dtype='<f8')
    faces = np.ascontiguousarray(mesh.faces, dtype='<i8')
    return dict(mesh.receipt) | {
        'loader_source_sha256': _sha(ROOT/'native_appearance.py'),
        'decoder_source_sha256': _sha(ROOT/'portable_twin.py'),
        'decoder_contract_source_sha256': _sha(ROOT/'stage5.py'),
        'geometry_renderer_source_sha256': _sha(ROOT/'alignment_render.py'),
        'receipt_source_sha256': _sha(Path(__file__)),
        'flattened_vertices_sha256': hashlib.sha256(vertices.tobytes()).hexdigest(),
        'flattened_faces_sha256': hashlib.sha256(faces.tobytes()).hexdigest(),
        'vertex_count': len(vertices), 'face_count': len(faces),
        'vertex_conversion_performed': False,
        'native_origin_preserved': True,
    }


def native_model_provenance(metadata):
    """Use the declared model backend, independently of the alignment backend."""
    backend = metadata.get('backend')
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError('Actual reconstruction backend is required in model metadata')
    return {'reconstruction_backend': backend,
            'appearance': metadata.get('appearance', 'source-native material appearance'),
            'sam3d_output': backend == 'Meta SAM 3D Objects'}


def native_execution_receipt(script_name):
    """Identify the native adapter and every project executable it relies on."""
    if script_name not in LEGACY_ORIGINS:
        raise ValueError('Unknown native alignment adapter')
    dependencies = [ROOT/'scripts'/script_name, Path(__file__), ROOT/'native_appearance.py', ROOT/'portable_twin.py', ROOT/'stage5.py', ROOT/'alignment_render.py']
    if script_name == 'stage4_native_track.py':
        dependencies.append(ROOT/'alignment_fit.py')
    return {'contract_version': CONTRACT,
            'backend': 'native-material bounded reference fit' if script_name.endswith('pose.py') else 'native-material conditional CoTracker/PnP',
            'project_sources': [{'file': str(path.relative_to(ROOT)), 'sha256': _sha(path)} for path in dependencies],
            'algorithm_origin': {'file': 'scripts/'+LEGACY_ORIGINS[script_name],
                                 'sha256': _sha(ROOT/'scripts'/LEGACY_ORIGINS[script_name]),
                                 'original_file_modified': False},
            'search_and_final_rasterizer': 'native_appearance.render_native_mesh',
            'manual_geometry_or_pose_used': False,
            'physical_accuracy_verified': False}


def verify_native_reference(reference, mesh):
    if reference.get('native_appearance') != native_representation_receipt(mesh):
        raise ValueError('Native appearance/geometry interpretation changed after reference fit')
