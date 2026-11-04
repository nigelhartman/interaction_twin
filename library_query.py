"""Shared, inference-free discovery contract for the human and agent library.

Facets describe saved model proposals, not confirmed physical properties. Actions
are saved procedure IDs/titles: this module never guesses a new action taxonomy.
"""
from __future__ import annotations

import copy
import re

FIELDS = ('parts', 'actions', 'relations', 'backends', 'readiness', 'task_classes')
SORTS = ('title', 'id', 'duration', 'components', 'assets', 'steps', 'version')
SHA = re.compile(r'^[a-f0-9]{64}$')
CONTRACT = {
    'version': 1,
    'filters': list(FIELDS),
    'matching': 'case_insensitive_exact_OR_within_field_AND_between_fields; query_requires_all_words',
    'action_values': 'saved_step_ids; titles_are_searchable; no_inferred_action_taxonomy',
    'sorts': list(SORTS), 'directions': ['asc', 'desc'],
    'groups': ['recording', 'variant'],
    'recording_group': 'exact_original_source_sha256; latest_saved_case_date_token_then_id',
    'version_sort': 'date_token_in_saved_case_id_not_capture_time',
    'pagination': {'limit_min': 1, 'limit_max': 25, 'offset_max': 10000},
    'facet_counts': 'global_requested_grouping_before_filters; count_each_record_once',
    'readiness_values': ['semantic_only', 'geometry_draft', 'motion_draft'],
    'readiness_scope': 'SAM3D_geometry_when_available; otherwise_saved_program_assets; legacy_motion_count_separate',
    'task_class_values': 'typed_saved_articulation_relations_or_explicit_task_annotations; no_prose_or_physics_inference',
    'physics_ready_available': False,
}


def normalized(value):
    # Match the browser's Unicode toLowerCase contract; full casefold would
    # make e.g. German sharp-S searches disagree between humans and MCP.
    return ' '.join(str(value).strip().lower().split())


def backend_key(value):
    text = normalized(value)
    compact = re.sub('[^a-z0-9]', '', text)
    if 'sam3d' in compact:
        return 'sam3d'
    if 'triposr' in compact:
        return 'triposr'
    return text


def version_key(record):
    dates = re.findall(r'20[0-9]{6}', record.get('id', record.get('twin_id', '')))
    return max(dates, default='00000000')


def derive_index(record):
    """Derive query metadata from saved program/library fields, never loose tags."""
    components = record.get('components', [])
    steps = record.get('steps', [])
    relationships = record.get('relationships', [])
    selected = [c['asset'] for c in components if c.get('asset')]
    selected += record.get('assets', [])
    reviews = record.get('sam3d_reviews', [])
    source_sha = record.get('source', {}).get('sha256')
    ids = {c.get('id') for c in components}
    # A future/mismatched candidate must not make this recording match SAM 3D.
    candidates = [r for r in reviews if r.get('source_sha256') == source_sha
                  and r.get('component_id') in ids and r.get('selected') is False]
    selected_backends = sorted({backend_key(a.get('backend', '')) for a in selected} - {''})
    candidate_backends = sorted({backend_key(r.get('backend', 'SAM 3D Objects')) for r in candidates} - {''})
    pose_count = record.get('supported_motion_samples', 0)
    if not pose_count:
        pose_count = sum(m.get('status') == 'supported_conditional_motion_candidate' for m in record.get('motion', []))
    native_ids = {c['id'] for c in components if c.get('asset') and backend_key(c['asset'].get('backend', '')) == 'sam3d'}
    native_rows = [m for m in record.get('motion', []) if m.get('track_id') in native_ids
                   and m.get('status') == 'supported_conditional_motion_candidate']
    native_pose_count = len(native_rows)
    native_motion_ids = sorted({m['track_id'] for m in native_rows})
    if 'motion' not in record and native_ids:
        # The human library omits full pose matrices. Its builder derives these
        # two summaries from the same hash-validated program and selected meshes.
        count = record.get('native_supported_motion_samples', 0)
        declared_ids = record.get('native_motion_component_ids', [])
        if (type(count) is int and count >= 0 and isinstance(declared_ids, list)
                and set(declared_ids) <= native_ids and bool(count) == bool(declared_ids)
                and count >= len(set(declared_ids))):
            native_pose_count = count
            native_motion_ids = sorted(set(declared_ids))
    has_native = 'sam3d' in selected_backends or 'sam3d' in candidate_backends
    readiness = ('motion_draft' if native_pose_count else 'geometry_draft') if has_native else (
        'motion_draft' if pose_count else 'geometry_draft' if selected else 'semantic_only')
    native_status = ('conditional_native_program_motion_unverified' if native_pose_count else
                     'candidate_not_aligned' if candidates else 'no_native_motion' if native_ids else 'not_generated')
    from task_taxonomy import TASK_CLASSES
    declared_classes=record.get('task_classes',[])
    task_classes=sorted(set(declared_classes)) if isinstance(declared_classes,list) and all(isinstance(c,str) and c in TASK_CLASSES for c in declared_classes) else[]
    fields = {
        'parts': sorted({normalized(c.get('id', '')) for c in components} - {''}),
        'actions': sorted({normalized(s.get('id', '')) for s in steps} - {''}),
        'relations': sorted({normalized(r.get('predicate', '')) for r in relationships} - {''}),
        'backends': sorted(set(selected_backends + candidate_backends)),
        'readiness': [readiness],
        'task_classes': task_classes,
    }
    labels = {
        'parts': {normalized(c['id']): c.get('label', c['id']) for c in components if c.get('id')},
        'actions': {normalized(s['id']): s.get('title', s['id']) for s in steps if s.get('id')},
        'task_classes': {key:TASK_CLASSES[key] for key in task_classes},
    }
    text = [record.get('id', record.get('twin_id', '')), record.get('title', ''),
            record.get('public_title', ''), record.get('summary', '')]
    text += [c.get('label', '') for c in components]
    text += [s.get('title', '') + ' ' + s.get('description', '') for s in steps]
    text += [r.get('rationale', '') for r in relationships]
    text += [v for values in fields.values() for v in values]
    return {**fields, 'labels': labels, 'text': normalized(' '.join(text)),
            'selected_backends': selected_backends, 'candidate_backends': candidate_backends,
            'candidate_component_ids': sorted({r['component_id'] for r in candidates}),
            'selected_asset_component_ids': sorted({c['id'] for c in components if c.get('asset')}),
            'readiness_scope': CONTRACT['readiness_scope'],
            'selected_motion_samples': pose_count,
            'native_supported_motion_samples': native_pose_count,
            'native_motion_component_ids': native_motion_ids,
            'native_motion_status': native_status,
            'candidate_motion_inherited': False,
            'simulation_ready': False, 'physical_accuracy_verified': False,
            'derived_from': 'saved_components_steps_relationships_and_source_bound_asset_candidates'}


def _validated(query, filters, sort, direction, offset, limit, group_by):
    if not isinstance(query, str) or len(query) > 200:
        raise ValueError('Query must be at most 200 characters')
    if sort not in SORTS or direction not in ('asc', 'desc') or group_by not in ('recording', 'variant'):
        raise ValueError('Unsupported library sort, direction or grouping')
    for name, value, maximum in [('offset', offset, 10000), ('limit', limit, 25)]:
        if type(value) is not int or not (0 if name == 'offset' else 1) <= value <= maximum:
            raise ValueError(f'{name} outside bounded library pagination')
    result = {}
    for name, values in filters.items():
        if values is None:
            values = []
        if not isinstance(values, list) or len(values) > 25 or any(not isinstance(v, str) or not 1 <= len(v.strip()) <= 200 for v in values):
            raise ValueError('Filters require at most 25 nonempty strings per field')
        result[name] = sorted({backend_key(v) if name == 'backends' else normalized(v) for v in values})
    return result


def query_records(records, query='', *, parts=None, actions=None, relations=None,
                  backends=None, readiness=None, task_classes=None, sort='title', direction='asc',
                  offset=0, limit=25, group_by='recording'):
    filters = _validated(query, dict(parts=parts, actions=actions, relations=relations,
                                    backends=backends, readiness=readiness, task_classes=task_classes),
                         sort, direction, offset, limit, group_by)
    rows = []
    for original in records:
        row = copy.deepcopy(original)
        # Always recompute; a stale/manually edited search_index is not authority.
        row['search_index'] = derive_index(row)
        rows.append(row)
    if group_by == 'recording':
        grouped = {}
        for row in rows:
            sha = row.get('source', {}).get('original_sha256')
            key = sha if isinstance(sha, str) and SHA.fullmatch(sha) else 'id:' + row['id']
            grouped.setdefault(key, []).append(row)
        selected = []
        for versions in grouped.values():
            versions.sort(key=lambda r: (version_key(r), r['id']), reverse=True)
            primary = versions[0]
            primary['saved_variant_ids'] = [r['id'] for r in versions]
            selected.append(primary)
        rows = selected
    facets = {}
    for field in FIELDS:
        counts, labels = {}, {}
        for row in rows:
            index = row['search_index']
            for value in set(index[field]):
                counts[value] = counts.get(value, 0) + 1
                labels.setdefault(value, index.get('labels', {}).get(field, {}).get(value, value))
        facets[field] = [{'value': value, 'label': labels[value], 'count': count}
                         for value, count in sorted(counts.items(), key=lambda p: (normalized(labels[p[0]]), p[0]))]
    terms = normalized(query).split()
    matched = [r for r in rows if all(t in r['search_index']['text'] for t in terms)
               and all(not values or set(values).intersection(r['search_index'][field]) for field, values in filters.items())]
    def key(row):
        counts = row.get('counts', {})
        if sort == 'title': return normalized(row.get('public_title') or row.get('title', ''))
        if sort == 'id': return row['id']
        if sort == 'duration': return row.get('duration_s', row.get('source', {}).get('duration_s', 0))
        if sort == 'version': return version_key(row)
        return counts.get(sort, 0)
    # Stable tie order remains ID ascending in either direction.
    matched.sort(key=lambda r: r['id'])
    matched.sort(key=key, reverse=direction == 'desc')
    page = matched[offset:offset + limit]
    more = offset + len(page) < len(matched)
    return {'records': page, 'total': len(matched), 'offset': offset, 'limit': limit,
            'has_more': more, 'next_offset': offset + len(page) if more else None,
            'facets': facets, 'query_contract': copy.deepcopy(CONTRACT),
            'applied': {'query': query, **filters, 'sort': sort, 'direction': direction, 'group_by': group_by},
            'simulation_ready': False, 'physical_accuracy_verified': False}
