#!/usr/bin/env python3
"""Index attested local diagnostic reconstructions without promoting assets."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import geometry_intelligence as gi
from reconstruction_catalog import INDEX, ReconstructionCatalog
from twin_catalog import TwinCatalog


def immutable_json(path, document):
    data = (json.dumps(document, indent=2, allow_nan=False) + '\n').encode()
    if path.exists() and path.read_bytes() != data:
        raise ValueError('Immutable reconstruction descriptor differs; create a new version ID')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)


def artifact(path, mime, expected=None):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / 'data') or not path.is_file():
        raise ValueError('Declared reconstruction artifact is outside local data')
    actual = gi.sha256(path)
    if expected and actual != expected:
        raise ValueError('Source export hash mismatch')
    return {'path': str(path.relative_to(ROOT)), 'sha256': actual, 'mime': mime}


def from_url(url):
    return gi.local_path(ROOT, url)


def build():
    base = ROOT / 'data/reconstruction-catalog-20261031'
    catalog, rows = TwinCatalog(ROOT), []
    bindings = {}
    for path in gi.SOURCE_PROGRAMS.values():
        entry = next(r for r in catalog.entries() if r['interaction_path'] == path)
        row, document, digest = catalog.load(entry['id'])
        bindings[path] = (row, document, digest)

    def version(version_id, label, interaction_path, component_id, kind, glb, source_report, evidence,
                quality, *, parent_track_id=None, advisory=None, extra_artifacts=None):
        row, document, digest = bindings[interaction_path]
        directory = base / 'versions' / version_id
        quality_path = directory / 'quality-summary.json'
        immutable_json(quality_path, {'format': 'reconstruction-inspection-quality', 'version': 1,
                       'version_id': version_id, 'kind': kind, 'physical_accuracy_verified': False,
                       'simulation_ready': False, 'summary': quality,
                       'limitations': quality['limitations'], 'source_report_sha256': source_report['sha256']})
        artifacts = {'glb': glb, 'quality': artifact(quality_path, 'application/json'), 'source_report': source_report}
        for i, frame in enumerate(evidence):
            artifacts['source-image-' + str(i)] = artifact(ROOT / frame['image_path'], 'image/jpeg')
            if frame.get('mask_path'):
                artifacts['source-mask-' + str(i)] = artifact(ROOT / frame['mask_path'], 'image/png')
        artifacts.update(extra_artifacts or {})
        descriptor = {'format': 'interaction-reconstruction-version', 'version': 1, 'version_id': version_id,
                      'label': label, 'case_id': document['case_id'], 'interaction_path': interaction_path,
                      'interaction_sha256': digest, 'source_video_sha256': document['source']['sha256'],
                      'component_id': component_id, 'kind': kind, 'artifacts': artifacts,
                      'source_evidence': evidence,
                      'gauge': {'metric': False, 'units': 'arbitrary_model_native' if kind == 'single_image_prior' else 'arbitrary_reconstruction_gauge',
                                'common_assembly_established': False, 'meters_per_unit': None},
                      'acceptance': 'unreviewed', 'physical_accuracy_verified': False, 'simulation_ready': False,
                      'quality_summary': {k: v for k, v in quality.items() if k != 'limitations'},
                      'limitations': quality['limitations'], 'production_assets_replaced': False}
        if parent_track_id:
            descriptor.update(parent_track_id=parent_track_id, proposal_advisory=advisory,
                              component_identity_status='model_proposed_subpart_not_verified_physical_identity')
        if kind == 'inferred':
            descriptor.update(inferred_layer_optional=True, contains_observed_layer=True,
                              inferred_faces_confidence='unknown', inferred_is_observed=False)
        path = directory / 'reconstruction-version.json'
        immutable_json(path, descriptor)
        rows.append({'version_id': version_id, 'interaction_path': interaction_path,
                     'descriptor_path': str(path.relative_to(ROOT)), 'descriptor_sha256': gi.sha256(path),
                     'label': label, 'component_id': component_id, 'kind': kind,
                     'acceptance': 'unreviewed', 'twin_id': row['id']})

    camera_path = gi.SOURCE_PROGRAMS['camera']
    camera_program = bindings[camera_path][1]
    surface_dir = ROOT / 'data/observed-surface-experiment-20261031/camera-body-stereo-v1'
    sparse_protocol = gi.read(ROOT / 'data/observed-geometry-experiment-20261031/camera-body-core-v5/protocol.json')
    dense_row = next(f for f in sparse_protocol['input_views'] if f['id'] == 'dense_source_000024')
    evidence = [{'id': dense_row['id'], 'timestamp_s': dense_row['timestamp_s'],
                 'source_video_sha256': camera_program['source']['sha256'],
                 'image_path': str(from_url(dense_row['image_url']).relative_to(ROOT)), 'image_sha256': dense_row['image_sha256'],
                 'mask_path': str(from_url(dense_row['mask_url']).relative_to(ROOT)), 'mask_sha256': dense_row['mask_sha256'],
                 'sampling_domain': 'additional_exact_native_PTS_dense_video_sample',
                 'index_domain': 'dense_camera_input_v2_not_program_1fps_index'}]
    exported = gi.read(ROOT / 'data/observed-surface-exports-20261031/manifest.json')
    for row in exported['exports']:
        report_path = from_url(row['metadata_url'])
        report = gi.read(report_path)
        validation_dir = ROOT / 'data/observed-surface-validation-20261031' / row['id']
        validation = gi.read(validation_dir / 'summary.json')
        version(row['id'], row['label'], camera_path, 'camera_body', 'observed',
                artifact(from_url(row['glb_url']), 'model/gltf-binary', row['glb_sha256']),
                artifact(report_path, 'application/json', row['metadata_sha256']), evidence,
                {'representation': 'observed_open_partial_patch', 'vertices': row['vertices'], 'triangles': row['triangles'],
                 'intrinsics': 'ambiguous_unmeasured', 'physical_scale': 'unknown',
                 'heldout_photometric_diagnostic': validation['metrics'],
                 'independent_geometry_accuracy': 'unmeasured', 'limitations': report['limitations']},
                extra_artifacts={'heldout_diagnostic': artifact(validation_dir / 'summary.json', 'application/json')})
    completions = gi.read(ROOT / 'data/observed-surface-exports-20261031/completion-exports.json')
    for row in completions['exports']:
        report_path = from_url(row['metadata_url'])
        report = gi.read(report_path)
        version(row['id'], row['label'], camera_path, 'camera_body', 'inferred',
                artifact(from_url(row['glb_url']), 'model/gltf-binary', row['glb_sha256']),
                artifact(report_path, 'application/json'), evidence,
                {'representation': 'observed_plus_optional_inferred_separate_GLTF_nodes',
                 'observed_triangles': row['observed_triangles'], 'inferred_triangles': row['inferred_triangles'],
                 'inferred_provider': row['inferred_provider'], 'physical_confidence': 'unknown',
                 'intrinsics': 'ambiguous_unmeasured', 'physical_scale': 'unknown',
                 'limitations': report['limitations']})
    parts_path = ROOT / 'data/moving-parts-reconstruction-20261031/jabra-parts-triposr-v1/display-manifest.json'
    parts = gi.read(parts_path)
    advisory_path = ROOT / 'data/geometry-intelligence-20261031/reconstruction-review-v3/review.json'
    for part in parts['components']:
        report_path = from_url(part['metadata_url'])
        report = gi.read(report_path)
        frame = next(e for e in bindings[gi.SOURCE_PROGRAMS['jabra']][1]['evidence'] if e['id'] == 'source_000023')
        e = [{'id': frame['id'], 'program_evidence_id': frame['id'], 'timestamp_s': frame['timestamp_s'],
              'source_video_sha256': bindings[gi.SOURCE_PROGRAMS['jabra']][1]['source']['sha256'],
              'image_path': str(from_url(part['source_image_url']).relative_to(ROOT)), 'image_sha256': frame['frame_sha256'],
              'mask_path': str(from_url(part['mask_url']).relative_to(ROOT)), 'mask_sha256': gi.sha256(from_url(part['mask_url'])),
              'sampling_domain': 'exact_program_source_frame'}]
        version(part['proposal_id'].replace('_', '-') + '-triposr-v1', part['label'] + ' · independent model preview',
                gi.SOURCE_PROGRAMS['jabra'], part['proposal_id'], 'single_image_prior',
                artifact(from_url(part['glb_url']), 'model/gltf-binary', part['glb_sha256']),
                artifact(report_path, 'application/json', part['metadata_sha256']), e,
                {'representation': 'single_image_model_prior_unvalidated', 'provider': part['model_provider'],
                 'vertices': part['vertices'], 'faces': part['faces'], 'hidden_shape': 'model_inferred_unobserved',
                 'common_assembly': 'absent_independent_native_gauges', 'mask_audit': 'not_model_audited',
                 'limitations': report['limitations']}, parent_track_id='charging_case',
                advisory=artifact(advisory_path, 'application/json'))
    # Newly finished application workflows declare their exact immutable version
    # paths. No arbitrary uploaded filename or URL is scanned/interpreted.
    for name in ['reconstruction-version.json', 'inferred-version.json']:
        for path in sorted((ROOT / 'data/observed-app-runs-20261031').glob('*-workflow/' + name)):
            descriptor = gi.read(path)
            if descriptor['version_id'] in {r['version_id'] for r in rows}:
                raise ValueError('Workflow reconstruction version id collides with an existing version')
            rows.append({'version_id': descriptor['version_id'], 'interaction_path': descriptor['interaction_path'],
                         'descriptor_path': str(path.relative_to(ROOT)), 'descriptor_sha256': gi.sha256(path),
                         'label': descriptor['label'], 'component_id': descriptor['component_id'],
                         'kind': descriptor['kind'], 'acceptance': descriptor['acceptance']})
    index = {'format': 'interaction-reconstruction-catalog', 'version': 1, 'versions': rows,
             'policy': ReconstructionCatalog.policy(), 'new_api_calls': 0, 'new_model_inference': False}
    # Validate all entries against their actual registered program before making
    # this rebuilt local index available to the MCP server/frontend.
    target = ROOT / INDEX
    temporary = base / 'index-candidate.json'
    gi.write(temporary, index)
    reader = ReconstructionCatalog(ROOT, str(temporary.relative_to(ROOT)))
    entries = catalog.entries()
    for row in rows:
        entry = next(r for r in entries if r['interaction_path'] == row['interaction_path'])
        _, doc, digest = catalog.load(entry['id'])
        reader.load(row['version_id'], interaction_path=row['interaction_path'], document=doc, program_sha256=digest)
    temporary.replace(target)
    return index


if __name__ == '__main__':
    result = build()
    print({'status': 'indexed_diagnostic_versions', 'versions': len(result['versions']), 'index': '/files/' + INDEX})
