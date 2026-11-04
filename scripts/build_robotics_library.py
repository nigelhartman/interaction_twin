#!/usr/bin/env python3
"""Derive a small local library from genuine, validated automatic Meta programs.

No inference, network, model selection, input modifications or publication.
Source families collapse saved experiments rather than inflating object counts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twin_catalog import TwinCatalog
from sim_readiness import audit_simulation_readiness
from library_query import CONTRACT as QUERY_CONTRACT, derive_index, backend_key


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path, data):
    """Publish complete local derivatives; interrupted writes preserve old bytes."""
    target = Path(path)
    staging = target.with_name('.' + target.name + '-' + uuid.uuid4().hex + '.tmp')
    try:
        staging.write_bytes(data)
        os.replace(staging, target)
    finally:
        staging.unlink(missing_ok=True)


def require(value, message):
    if not value:
        raise ValueError(message)


def local(root, name, prefix='data'):
    require(isinstance(name, str) and name.startswith(prefix + '/') and '\\' not in name,
            'Expected a declared local artifact')
    parts = Path(name).parts
    require(all(not p.startswith('.') for p in parts), 'Hidden or traversing artifacts are excluded')
    p = root
    for part in parts:
        p = p / part
        require(not p.is_symlink(), 'Symlink artifacts are excluded')
    require(p.is_file() and p.resolve().is_relative_to(root / prefix), 'Artifact outside allowed tree')
    return p


def record_file(root, path, expected=None, prefix='data'):
    p = local(root, path, prefix)
    actual = digest(p)
    require(expected is None or actual == expected, 'Artifact digest differs')
    return {'path': path, 'sha256': actual, 'bytes': p.stat().st_size}


def object_json(path):
    value = json.loads(Path(path).read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    require(isinstance(value, dict), 'JSON object required')
    return value


def genuine_asset(root, asset, doc):
    """Bind generated asset and full-frame anchor to actual saved model metadata."""
    path = Path(asset['path'])
    metadata_name = (path.parent / 'metadata.json').as_posix()
    metadata = object_json(local(root, metadata_name))
    require(metadata.get('fixture') in (None, False) and metadata.get('learned_inference_executed') is True
            and metadata.get('geometry_reconstructed_by_model') is True
            and metadata.get('manual_geometry_fallback') is False and metadata.get('manual_pose_used') is False
            and metadata.get('object_id') == asset['track_id'] and metadata.get('backend') == asset['backend']
            and metadata.get('metric_scale_verified') is False, 'A genuine automatic model draft is required')
    source = metadata.get('source')
    require(isinstance(source, dict) and source.get('video_sha256') == doc['source']['sha256'],
            'Model source differs from program source')
    mask = source.get('mask_provenance')
    require(isinstance(mask, dict) and mask.get('manual_seed_boxes_used') is False
            and mask.get('prompt_origin') == 'model_generated_prompt'
            and mask.get('model') == 'sam-3.1', 'Automatic Meta segmentation provenance required')
    files = metadata.get('artifact_files')
    require(isinstance(files, list), 'Actual generated file inventory required')
    glb = next((f for f in files if isinstance(f, dict) and f.get('file') == path.name), None)
    require(glb is not None and glb.get('sha256') == asset['sha256'], 'Native GLB and metadata differ')
    asset_ref = record_file(root, asset['path'], asset['sha256'])
    refs = {}
    for key, role in [('image_path', 'image'), ('mask_path', 'mask')]:
        name = source.get(key)
        # Native SAM imports preserve the official inputs under this exact
        # adapter-owned pair. Keep the historical direct-file rule otherwise;
        # do not permit arbitrary nested metadata paths.
        native_name = 'raw/inputs/' + ('image-source.png' if role == 'image' else 'mask-source.png')
        native = metadata.get('sam3d_output') is True and backend_key(metadata.get('backend', '')) == 'sam3d'
        require(isinstance(name, str) and (Path(name).name == name or native and name == native_name),
                'Model anchor must be a direct local file or exact native SAM input')
        expected = source.get(role + '_sha256')
        require(isinstance(expected, str) and re.fullmatch('[a-f0-9]{64}', expected), 'Model input digest is required')
        refs[role] = record_file(root, (path.parent / name).as_posix(), expected)
        if name == native_name:
            declared = next((f for f in files if isinstance(f, dict) and f.get('file') == name), None)
            require(declared is not None and declared.get('sha256') == refs[role]['sha256']
                    and declared.get('bytes') == refs[role]['bytes'], 'Native SAM input inventory differs')
    require(type(source.get('sample_index')) is int and type(source.get('timestamp_s')) in (int, float),
            'Exact source anchor is required')
    require(source['sample_index'] > 0 and math.isfinite(source['timestamp_s'])
            and 0 <= source['timestamp_s'] <= doc['source']['duration_s'] + .1
            and source.get('time_domain') == 'source_video_presentation_seconds', 'Model source time is invalid')
    model = metadata.get('model_provenance')
    require(model is None or isinstance(model, dict), 'Invalid model provenance')
    model = model or {}
    return {'id': asset['id'], 'track_id': asset['track_id'], **asset_ref, 'backend': asset['backend'],
            'metadata': record_file(root, metadata_name), 'appearance': metadata.get('appearance', 'Model-generated appearance'),
            'hidden_surfaces': metadata.get('hidden_surfaces', 'Inferred by the reconstruction model'),
            'model_revision': model.get('revision'),
            'code_commit': model.get('code_commit') or (model.get('code') or {}).get('commit'),
            'anchor': {'sample_index': source['sample_index'], 'timestamp_s': source['timestamp_s'],
                       'time_domain': source.get('time_domain'), **refs},
            'mask_model': mask['model'], 'prompt_model': mask.get('prompt_model'),
            'manual_geometry_used': False, 'manual_pose_used': False, 'physical_accuracy_verified': False,
            'source_video_pose_established_by_this_library': False}


def date_key(row):
    dates = re.findall(r'20[0-9]{6}', row['id'])
    return (max(dates, default='00000000'), row['id'])


def display_title(tracks):
    labels = list(dict.fromkeys(t['label'] for t in tracks))
    return ' + '.join(labels[:3]).capitalize()


def offline_queue(root, case_id, program_sha, source, tracks):
    """Read existing prepared queues; never prepare, submit or infer anything."""
    from component_asset_queue import inspect as inspect_queue
    found = []
    for queue_path in sorted((root / 'data/component-asset-queues').glob('*/queue.json')):
        try:
            verified = inspect_queue(queue_path, root=root)
            require(verified.get('status') == 'verified_offline_queue', 'Queue inspection did not pass')
        except (OSError, ValueError, KeyError, TypeError):
            # An optional stale handoff must not hide an otherwise valid twin.
            continue
        queue = object_json(local(root, queue_path.relative_to(root).as_posix()))
        plan_ref = queue.get('plan', {})
        plan = object_json(local(root, plan_ref.get('path', '')))
        require(digest(root / plan_ref['path']) == plan_ref.get('sha256'), 'Queue plan changed')
        if plan.get('twin_id') != case_id:
            continue
        require(plan.get('format') == 'interaction-twin-component-asset-queue' and plan.get('version') == 1
                and plan.get('program', {}).get('sha256') == program_sha and plan.get('source') == source,
                'Queue does not belong to this exact source program')
        for record in (plan, queue):
            require(record.get('no_inference') is True and record.get('remote_job_submitted') is False
                    and record.get('preferred_assets_changed') is False
                    and record.get('physical_accuracy_verified') is False and record.get('simulation_ready') is False,
                    'Queue must remain an offline draft')
        require(plan.get('read_only_source') is True and queue.get('status') == 'prepared_offline_not_executed',
                'Only saved offline queues are included')
        require({c['track_id'] for c in queue['components']} == {t['id'] for t in tracks}, 'Queue components differ')
        jobs = []
        for component in queue['components']:
            attachment = component.get('attachment', {})
            require(attachment.get('existing_poses_reused') is False and attachment.get('fresh_reference_fit_required') is True
                    and attachment.get('simulation_ready') is False, 'New geometry cannot inherit old poses')
            job = component.get('job')
            if job:
                require(job.get('state') == 'prepared_not_submitted', 'Only unsubmitted bundles are listed')
                jobs.append({'track_id': component['track_id'], 'manifest': record_file(root, job['manifest']['path'], job['manifest']['sha256']),
                             'zip': record_file(root, job['zip']['path'], job['zip']['sha256'])})
        found.append({'queue': record_file(root, queue_path.relative_to(root).as_posix()),
                      'plan': record_file(root, plan_ref['path'], plan_ref['sha256']),
                      'prepared_bundles': jobs, 'unavailable_components': [c['track_id'] for c in queue['components'] if c['state'] == 'unavailable'],
                      'remote_submitted': False, 'inference_executed': False, 'existing_poses_reused': False})
    return found[-1] if found else None


def cached_insights(root, interaction):
    from task_intelligence import cached_analyses
    return [{'analysis': record_file(root, row['path']), 'request_key': row['record']['request_key'],
             'question': row['record']['contract'].get('question', ''), 'model': row['record']['model'],
             'human_reviewed': False, 'robot_commands': False}
            for row in cached_analyses(interaction, root=root) if row['record']['mode'] == 'ask']


def initial_inputs(root, source, tracks):
    """List verified first-reconstruction inputs, separately from real assets."""
    from initial_sam3d_handoff import inspect as inspect_initial
    ids = {t['id'] for t in tracks}
    prepared = []
    for path in sorted((root / 'data/initial-sam3d-handoffs').glob('*/handoff.json')):
        try:
            record = object_json(local(root, path.relative_to(root).as_posix()))
            binding = object_json(local(root, (path.parent / 'source-binding.json').relative_to(root).as_posix()))
            if record.get('component_id') not in ids or binding.get('source', {}).get('video_sha256') != source['sha256']:
                continue
            verified = inspect_initial(path, root=root)
            require(verified.get('status') == 'verified_initial_reconstruction_prepared'
                    and verified.get('component_id') == record['component_id']
                    and verified.get('source') == binding['source'], 'Initial source input binding differs')
            prepared.append({'component_id': record['component_id'], 'handoff': record_file(root, path.relative_to(root).as_posix()),
                             'source': copy.deepcopy(verified['source']), 'status': 'initial_reconstruction_prepared',
                             'baseline_required': False, 'no_inference': True, 'remote_job_submitted': False,
                             'simulation_ready': False})
        except (OSError, ValueError, KeyError, TypeError):
            # Stale optional preparation must not hide the existing real twin.
            continue
    return prepared


def build_record(root, row, doc, program_sha, index, reviews=None):
    require(doc.get('reasoning', {}).get('provider') == 'meta_api'
            and doc['reasoning'].get('manual_semantic_story_used') is False
            and doc['reasoning'].get('human_reviewed') is False, 'Current automatic Meta program required')
    require(doc.get('fixture') in (None, False), 'Fixture programs are excluded')
    assets = [genuine_asset(root, a, doc) for a in doc['assets']]
    asset_map = {a['track_id']: a for a in assets}
    indexed = index.get(row['id'])
    require(indexed is not None and indexed.get('original_sha256') == program_sha, 'Workspace scene identity differs')
    ref = indexed.get('scene_ref')
    if ref is not None:
        require(isinstance(ref, dict) and ref.get('path') == 'scenes/' + str(ref.get('sha256')) + '.json',
                'A hash-bound saved native scene is required')
        scene_path = local(root, 'workspace/' + ref['path'], 'workspace')
        require(scene_path.stat().st_size == ref.get('bytes') and digest(scene_path) == ref['sha256'], 'Saved scene changed')
        pack = object_json(scene_path)
        require(pack.get('format') == 'workspace-case-scene' and pack.get('case_id') == row['id']
                and pack.get('interaction_sha256') == program_sha and pack.get('scene', {}).get('physical_assembly_verified') is False,
                'Saved native scene belongs to another program')
        for obj in pack['scene']['objects']:
            expected = asset_map.get(obj.get('object_id'))
            require(expected is not None and obj.get('provenance', {}).get('mesh_sha256') == expected['sha256']
                    and obj['provenance'].get('mesh_file') == expected['path'], 'Native scene/model binding differs')
        require({o['object_id'] for o in pack['scene']['objects']} == set(asset_map), 'Saved scene component set differs')
    else:
        # Fresh SAM 3D components have independent native gauges and no accepted
        # video motion yet. Their unchanged GLBs and source anchors are still
        # validated above, while independent native viewers/MCP serve geometry
        # without materializing a huge, misleading aggregate scene.
        native_unaligned = bool(assets) and all(backend_key(a['backend']) == 'sam3d' for a in assets)
        native_unaligned = native_unaligned and not any(m.get('status') == 'supported_conditional_motion_candidate'
            and m.get('camera_from_native_mesh') is not None for m in doc['motion'])
        require(not assets or native_unaligned, 'Generated assets require a hash-bound scene unless all are unaligned native SAM 3D')
    evidence = {e['id']: e for e in doc['evidence']}
    def citation(eid):
        e = evidence[eid]
        return {'id': eid, 'timestamp_s': e['timestamp_s'], 'source_index': e['source_index'],
                'frame': record_file(root, e['frame_path'], e['frame_sha256'])}
    tracks = [{**copy.deepcopy(t), 'asset': asset_map.get(t['id'])} for t in doc['tracklets']]
    relations = [{**copy.deepcopy(r), 'citations': [citation(e) for e in r['evidence_ids']]} for r in doc['relationships']]
    steps = [{**copy.deepcopy(s), 'citations': [citation(e) for e in s['evidence_ids']]} for s in doc['steps']]
    read = audit_simulation_readiness(doc)
    require(read['simulation_ready'] is False and read['simulation_runnable'] is False, 'Physical readiness cannot be promoted')
    predicates = {r['predicate'] for r in relations}
    tags = []
    if 'removable_from' in predicates: tags.append('Removable parts')
    if 'attached_to' in predicates: tags.append('Connected parts')
    if any('occlu' in text.lower() for t in tracks for text in t.get('limitations', [])): tags.append('Occlusion')
    if not tags: tags.append('Object handling')
    result = {'id': row['id'], 'title': display_title(doc['tracklets']), 'summary': doc['summary'],
            'interaction': record_file(root, row['interaction_path'], program_sha),
            'source': copy.deepcopy(doc['source']), 'duration_s': doc['source']['duration_s'],
            'thumbnail': assets[0]['anchor']['image'] if assets else citation(doc['evidence'][0]['id'])['frame'],
            'components': tracks, 'relationships': relations,
            'steps': steps, 'tags': tags, 'reasoning': copy.deepcopy(doc['reasoning']),
            'scene_case': {'id': row['id'], 'original_sha256': program_sha, 'scene_ref': copy.deepcopy(ref)},
            'readiness': read, 'limitations': copy.deepcopy(doc['limitations']),
            'counts': {'components': len(tracks), 'assets': len(assets), 'steps': len(steps),
                       'relationships': len(relations), 'evidence_samples': len(evidence)},
            'physical_accuracy_verified': False, 'simulation_ready': False,
            'supported_motion_samples': sum(m.get('status') == 'supported_conditional_motion_candidate' for m in doc['motion']),
            'native_supported_motion_samples': sum(m.get('status') == 'supported_conditional_motion_candidate'
                and backend_key(asset_map.get(m.get('track_id'), {}).get('backend', '')) == 'sam3d' for m in doc['motion']),
            'native_motion_component_ids': sorted({m['track_id'] for m in doc['motion'] if m.get('status') == 'supported_conditional_motion_candidate'
                and backend_key(asset_map.get(m.get('track_id'), {}).get('backend', '')) == 'sam3d'}),
            'source_video_alignment_added': False, 'preferred_model_changed': False,
            'offline_queue': offline_queue(root, row['id'], program_sha, doc['source'], tracks),
            'cached_insights': cached_insights(root, row['interaction_path'])}
    if reviews is None:
        from sam3d_review_catalog import reviews_for_record
        result['sam3d_reviews'] = reviews_for_record(root, result)
    else:
        from sam3d_review_catalog import match_reviews
        result['sam3d_reviews'] = match_reviews(reviews, result)
    result['search_index'] = derive_index(result)
    return result


def build(root=ROOT, out=None):
    root = Path(root).resolve()
    out = Path(out or root / 'showcase').resolve()
    require(out.is_relative_to(root / 'showcase'), 'Library output must remain inside showcase')
    catalog = TwinCatalog(root)
    from sam3d_review_catalog import discover_reviews
    reviews = discover_reviews(root)
    index_path = root / 'workspace/cases-index.json'
    index = {r['id']: r for r in object_json(index_path)['cases']}
    eligible, excluded = [], []
    for entry in catalog.entries():
        if entry.get('interaction', {}).get('reasoning', {}).get('provider') != 'meta_api': continue
        try:
            row, doc, program_sha = catalog.load(entry['id'])
            eligible.append((row, build_record(root, row, doc, program_sha, index, reviews)))
        except (OSError, ValueError, KeyError, TypeError) as error:
            excluded.append({'id': entry['id'], 'status': 'excluded_unverified_record', 'exception_type': type(error).__name__})
    grouped = {}
    for row, record in eligible:
        grouped.setdefault(record['source']['original_sha256'], []).append((row, record))
    records = []
    for _, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: date_key(item[0]), reverse=True)
        selected_row, selected = ordered[0]
        selected['initial_reconstruction_inputs'] = initial_inputs(root, selected['source'], selected['components'])
        selected['other_saved_versions'] = [{'id': r['id'], 'interaction': value['interaction'], 'counts': value['counts']}
                                            for r, value in ordered[1:]]
        records.append(selected)
    records.sort(key=lambda r: (r['title'].casefold(), r['id']))
    unique_assets = {c['asset']['sha256'] for r in records for c in r['components'] if c['asset']}
    document = {'format': 'interaction-twin-object-program-library', 'version': 1,
        'selection_policy': 'Latest dated eligible automatic Meta program per exact original-source SHA256; ties by case ID. Display grouping only; original/preferred models unchanged.',
        'read_only': True, 'no_inference': True, 'published': False,
        'physical_accuracy_verified': False, 'simulation_ready': False,
        'summary': {'recordings': len(records), 'native_asset_drafts': len(unique_assets),
                    'components': sum(r['counts']['components'] for r in records),
                    'steps': sum(r['counts']['steps'] for r in records),
                    'eligible_saved_versions': len(eligible)},
        'records': records, 'excluded_records': excluded,
        'query_contract': QUERY_CONTRACT,
        'provenance': {'builder_sha256': digest(Path(__file__)), 'catalog_code_sha256': digest(root / 'twin_catalog.py'),
                       'workspace_index': {'path': 'workspace/cases-index.json', 'sha256': digest(index_path)},
                       'method': 'Actual TwinCatalog validation, native model metadata and file hashes; no model execution'}}
    encoded = (json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode()
    catalog_sha = hashlib.sha256(encoded).hexdigest()
    manifest = {'format': 'interaction-twin-library-index', 'version': 1,
                'catalog': {'path': 'library-catalog-' + catalog_sha + '.json', 'sha256': catalog_sha, 'bytes': len(encoded)},
                'summary': document['summary']}
    out.mkdir(parents=True, exist_ok=True)
    # Immutable content first: both old and new manifests remain usable during
    # an upload refresh, without an old-index/new-catalog hash race.
    immutable = out / manifest['catalog']['path']
    if immutable.exists():
        require(immutable.read_bytes() == encoded, 'Preserve immutable library catalog')
    else:
        atomic_write(immutable, encoded)
    # Preserve the conventional current alias for existing offline consumers.
    atomic_write(out / 'library-catalog.json', encoded)
    atomic_write(out / 'library-index.json', (json.dumps(manifest, separators=(',', ':'), allow_nan=False) + '\n').encode())
    atomic_write(out / 'library-index.js', ('window.INTERACTION_LIBRARY_INDEX=' + json.dumps(manifest, separators=(',', ':')).replace('<', '\\u003c') + ';\n').encode())
    print(json.dumps({'status': 'local_library_built', **document['summary'], 'excluded': excluded,
                      'catalog_sha256': manifest['catalog']['sha256'], 'new_api_calls': 0, 'new_model_calls': 0}))
    return document


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    build(out=args.out)
