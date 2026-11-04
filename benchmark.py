"""Saved-evidence benchmarks. Counts and source linkage are not physical accuracy.

This module performs no inference, network requests, model loading or secret reads.
Human judgments are accepted only from an explicitly supplied review sidecar.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

VERSION = '1.0'
ROOT = Path(__file__).resolve().parent
SUPPORTED = 'supported_conditional_motion_candidate'
REFERENCE = 'reference_candidate_unverified_geometry_and_pose'
# Matches the saved procedure replay's original-PTS equality tolerance. This is
# rounding allowance, never nearest-frame interpolation or resampling.
SOURCE_TIME_TOLERANCE_S = 1e-7
CODE_KINDS = {'deterministic_compiler_code', 'deterministic_compiler_orchestration_code',
              'dense_semantic_compiler_adapter', 'schema'}


def path_in(root, value):
    """Permit project artifacts, never credential/hidden directories or traversal."""
    root = Path(root).resolve()
    value = Path(value)
    if value.is_absolute() or any(p in {'..', '.'} or p.startswith('.') for p in value.parts):
        raise ValueError('Benchmark paths must be visible project-relative artifact paths')
    if value.suffix.lower() in {'.env', '.pem', '.key'}:
        raise ValueError('Credential files are not benchmark artifacts')
    path = (root / value).resolve()
    if not path.is_relative_to(root) or any(p.startswith('.') for p in path.relative_to(root).parts):
        raise ValueError('Benchmark artifact escapes project or enters a hidden directory')
    return path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(root, path):
    return json.loads(path_in(root, path).read_text())


def ratio(numerator, denominator):
    return {'numerator': numerator, 'denominator': denominator,
            'fraction': numerator / denominator if denominator else None}


def artifact(root, path, expected=None, kind=None):
    target = path_in(root, path)
    item = {'path': str(path), 'kind': kind, 'expected_sha256': expected}
    if not target.is_file():
        return dict(item, status='missing', sha256=None, bytes=None)
    actual = digest(target)
    return dict(item, status='verified' if expected == actual else 'changed' if expected else 'snapshotted',
                sha256=actual, bytes=target.stat().st_size)


def elapsed(start, end):
    if not start or not end:
        return None
    try:
        a = datetime.fromisoformat(start.replace('Z', '+00:00'))
        b = datetime.fromisoformat(end.replace('Z', '+00:00'))
        if a.tzinfo is None or b.tzinfo is None:
            return None
        seconds = (b-a).total_seconds()
        return seconds if math.isfinite(seconds) and seconds >= 0 else None
    except (TypeError, ValueError):
        return None


def runtime(root, entry, interaction_path, interaction_hash):
    """Evidence verification/preflight duration is never inference runtime."""
    if not entry.get('run'):
        return {'status': 'unknown', 'elapsed_s': None, 'reason': 'No inference-run receipt is bound to this program', 'phases': []}
    run = load(root, entry['run'])
    same_program = run.get('artifacts', {}).get('interaction') == interaction_path
    recorded = [f for p in run.get('phases', []) for f in p.get('files', [])
                if f.get('path') == interaction_path]
    matching_hash = bool(recorded) and all(f.get('sha256') == interaction_hash for f in recorded)
    inference = run.get('profile', {}).get('mode') == 'local_inference'
    seconds = elapsed(run.get('created_at'), run.get('completed_at'))
    phases = [{'name': p['name'], 'status': p['status'], 'attempt': p.get('attempt'),
               'elapsed_s': elapsed(p.get('started_at'), p.get('completed_at')),
               'reused': p['status'] == 'reused_verified', 'error': p.get('error'),
               'result_failures': p.get('result', {}).get('failures', []),
               'has_gaps': p.get('result', {}).get('has_gaps', p['status'] == 'complete_with_gaps'),
               'components': [{'track_id': c.get('track_id'), 'status': c.get('status'),
                               'error': c.get('error'), 'output': c.get('output')}
                              for c in p.get('result', {}).get('components', [])]}
              for p in run.get('phases', [])]
    trusted = same_program and matching_hash and inference and seconds is not None
    return {'status': 'recorded' if trusted else 'unknown', 'elapsed_s': seconds if trusted else None,
            'reason': None if trusted else 'Receipt is not a matching completed timed inference run for this exact program',
            'run_id': run.get('id'), 'run_status': run.get('status'), 'run_receipt': entry['run'],
            'run_receipt_sha256': digest(path_in(root, entry['run'])),
            'same_program': same_program, 'program_hash_matches_receipt': matching_hash,
            'profile': run.get('profile'), 'phases': phases,
            'reused_phases': sum(p['reused'] for p in phases),
            'failed_phases': sum(p['status'] == 'failed' for p in phases),
            'phases_with_gaps': sum(bool(p['has_gaps']) for p in phases)}


def human_scores(document, review, interaction_hash):
    """Never convert model agreement, provenance or UI acceptance into correctness."""
    statements = {kind + ':' + item['id'] for kind, key in [('step', 'steps'), ('assertion', 'assertions'),
                                                           ('relationship', 'relationships')]
                  for item in document.get(key, [])}
    identities = {t['id'] for t in document['tracklets']}
    review = review or {}
    if review and (review.get('source') != 'human' or not review.get('reviewer', '').strip() or
                   review.get('interaction_sha256') != interaction_hash):
        raise ValueError('Human review must identify its human reviewer and bind the exact interaction hash')
    count = review.get('physical_part_count')
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
        raise ValueError('Human physical part count must be a nonnegative integer or null')
    def score(rows, targets, category):
        seen = set()
        for row in rows:
            if row['id'] not in targets or row['id'] in seen or row['verdict'] not in {'correct', 'incorrect', 'unknown'}:
                raise ValueError('Invalid, duplicate or unknown human ' + category + ' judgment')
            if not row.get('note', '').strip():
                raise ValueError('Human judgments require a review note')
            seen.add(row['id'])
        counts = Counter(r['verdict'] for r in rows)
        reviewed = counts['correct'] + counts['incorrect']
        return {'status': 'reviewed' if reviewed else 'pending_human_review', 'total_targets': len(targets),
                'correct': counts['correct'], 'incorrect': counts['incorrect'],
                'explicitly_unknown': counts['unknown'], 'unreviewed': len(targets)-len(rows),
                'accuracy_on_decided_only': ratio(counts['correct'], reviewed),
                'rows': rows}
    return {'source': 'human_only', 'reviewer': review.get('reviewer'),
            'statement_correctness': score(review.get('statements', []), statements, 'statement'),
            'tracklet_identity_correctness': score(review.get('identities', []), identities, 'identity'),
            'physical_part_count': review.get('physical_part_count')}


def metrics(doc):
    evidence = doc['evidence']
    samples = {e['source_index']: e for e in evidence}
    if len(samples) != len(evidence):
        raise ValueError('Duplicate source sample IDs')
    tracks = {t['id']: t for t in doc['tracklets']}
    if len(tracks) != len(doc['tracklets']):
        raise ValueError('Duplicate tracklet IDs')
    generated = {a['track_id'] for a in doc['assets']}
    masked = {o['track_id'] for e in evidence for o in e.get('observations', [])}
    plausible = {o['track_id'] for e in evidence for o in e.get('observations', [])
                 if o.get('audit_status') == 'plausible_visible_mask' and o.get('gate_status') != 'withheld_geometric_gate'}
    if (generated | masked) - tracks.keys():
        raise ValueError('Generated or observed tracklet is not declared')
    per_part = []
    motion = doc['motion']
    seen = set()
    linked_motion = []
    for m in motion:
        # source_sample_index belongs to the referenced motion experiment,
        # whose cadence may differ from the procedure's evidence sampling.
        # Upstream index/matrix integrity is checked by validate_interaction;
        # source PTS is the common coordinate for coverage in this document.
        matched = [e for e in evidence
                   if abs(m['timestamp_s'] - e['timestamp_s']) <= SOURCE_TIME_TOLERANCE_S]
        if len(matched) != 1:
            raise ValueError('Motion time does not identify one exact source sample')
        source_index = matched[0]['source_index']
        key = (m['track_id'], source_index)
        if key in seen or key[0] not in tracks:
            raise ValueError('Duplicate or unlinked motion sample')
        seen.add(key)
        linked_motion.append((m, source_index))
    for tid in sorted(generated):
        linked_rows = [(m, i) for m, i in linked_motion if m['track_id'] == tid]
        rows = [m for m, _ in linked_rows]
        reference = {i for m, i in linked_rows if m['status'] == REFERENCE}
        supported = {i for m, i in linked_rows if m['status'] == SUPPORTED and m['camera_from_native_mesh'] is not None}
        lo, hi = tracks[tid]['source_sample_range']
        scope = {i for i in samples if lo <= i <= hi}
        if supported - scope or reference - scope:
            raise ValueError('Motion sample lies outside its declared tracklet range')
        full = set(samples) - reference
        scoped = scope - reference
        per_part.append({'track_id': tid, 'label': tracks[tid]['label'], 'reference_samples': len(reference),
                         'supported_nonreference_samples': len(supported),
                         'full_source_coverage': ratio(len(supported), len(full)),
                         'tracklet_range_coverage': ratio(len(supported), len(scoped)),
                         'withheld_motion_samples': sum(m['status'] not in {SUPPORTED, REFERENCE} for m in rows),
                         'unprocessed_or_absent_in_tracklet': len(scope - {i for _, i in linked_rows}),
                         'motion_status_counts': dict(Counter(m['status'] for m in rows)),
                         'physical_pose_accuracy': None})
    return {'sample_count': len(evidence), 'declared_tracklets': len(tracks),
            'mask_observed_tracklets': len(masked), 'model_audited_plausible_tracklets': len(plausible),
            'generated_asset_count': len(doc['assets']), 'generated_tracklets': len(generated),
            'generated_track_ids': sorted(generated), 'audited_track_ids': sorted(plausible),
            'unverified_physical_identity_tracklets': sum('unverified' in t['identity_status'] for t in tracks.values()),
            'physical_distinct_part_count': None,
            'parts': per_part, 'step_count': len(doc['steps']), 'assertion_count': len(doc['assertions']),
            'relationship_count': len(doc['relationships']),
            'statement_count': sum(len(doc[k]) for k in ['steps', 'assertions', 'relationships']),
            'physical_accuracy_verified': False}


def case_result(root, entry):
    target = entry['interaction']
    doc = load(root, target)
    result = {'id': entry['id'], 'title': entry.get('title', entry['id']), 'interaction': target,
              'cohort': entry.get('cohort', 'unspecified'), 'error': None}
    ihash = digest(path_in(root, target))
    result['interaction_sha256'] = ihash
    if entry.get('expected_sha256') and entry['expected_sha256'] != ihash:
        result['pinned_program_changed'] = True
    else:
        result['pinned_program_changed'] = False
    # Contract/schema validation checks references and source-time joins, not semantic truth.
    import stage5
    try:
        stage5.validate_interaction(doc, root=root, verify_files=False)
        result['structural_validation'] = {'status': 'valid', 'error': None}
    except (ValueError, KeyError, TypeError) as exc:
        result['structural_validation'] = {'status': 'invalid', 'error': str(exc)}
    receipts = [artifact(root, target, entry.get('expected_sha256'), 'interaction_program'),
                artifact(root, doc['source']['path'], doc['source']['sha256'], 'source_video')]
    receipts += [artifact(root, i['path'], i['sha256'], i['kind']) for i in doc['inputs']]
    receipts += [artifact(root, a['path'], a['sha256'], 'generated_asset') for a in doc['assets']]
    for e in doc['evidence']:
        receipts.append(artifact(root, e['frame_path'], e['frame_sha256'], 'source_frame'))
        for o in e.get('observations', []):
            receipts.append(artifact(root, o['mask_path'], o['mask_sha256'], 'source_mask'))
    # Deduplicate identical references while preserving different expected versions.
    result['artifacts'] = list({(r['path'], r['expected_sha256']): r for r in receipts}.values())
    bad = [r for r in result['artifacts'] if r['status'] in {'changed', 'missing'}]
    if bad:
        upstream = {'status': 'not_verified', 'error': 'Artifact bytes are missing or changed; full upstream joins withheld'}
    else:
        try:
            # Beyond matching bytes: verify source manifest/PTS and exact same-track
            # upstream pose matrices against their saved model experiments.
            stage5.validate_interaction(doc, root=root, verify_files=True)
            upstream = {'status': 'valid', 'error': None}
        except Exception as exc:
            upstream = {'status': 'invalid', 'error': type(exc).__name__ + ': ' + str(exc)}
    result['upstream_join_validation'] = upstream
    result['source_linkage'] = {'status': 'verified' if not bad and upstream['status'] == 'valid' else 'needs_review',
                               'checked_artifacts': len(result['artifacts']),
                               'changed_or_missing': len(bad),
                               'code_or_schema_drift': sum(r['kind'] in CODE_KINDS for r in bad),
                               'evidence_or_asset_drift': sum(r['kind'] not in CODE_KINDS for r in bad),
                               'bad_artifacts': bad,
                               'meaning': 'Valid references and matching bytes do not establish semantic correctness.'}
    result['metrics'] = metrics(doc)
    result['runtime'] = runtime(root, entry, target, ihash)
    result['models'] = {'reasoning': doc['reasoning'], 'assets': []}
    for inp in doc['inputs']:
        if inp['kind'] != 'learned_asset_metadata':
            continue
        meta = load(root, inp['path'])
        result['models']['assets'].append({'object_id': meta.get('object_id'), 'backend': meta.get('backend'),
                                          'model_provenance': meta.get('model_provenance'),
                                          'runtime': meta.get('runtime'),
                                          'settings': {k: meta[k] for k in ['inference', 'mesh', 'settings', 'preprocessing', 'seed',
                                                                          'density_grid_resolution', 'density_isovalue'] if k in meta},
                                          'sam3d_output': meta.get('sam3d_output'),
                                          'mask_provenance': meta.get('source', {}).get('mask_provenance')})
    result['human_review'] = human_scores(doc, load(root, entry['human_review']) if entry.get('human_review') else None, ihash)
    result['status'] = 'valid_saved_evidence' if result['structural_validation']['status'] == 'valid' and not bad and upstream['status'] == 'valid' else 'needs_review'
    return result


def build(root, config_path):
    root = Path(root).resolve()
    config = load(root, config_path)
    if config.get('version') != VERSION or not isinstance(config.get('cases'), list):
        raise ValueError('Unsupported benchmark configuration')
    ids = [e['id'] for e in config['cases']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate benchmark case IDs')
    cases = []
    for entry in config['cases']:
        try:
            cases.append(case_result(root, entry))
        except Exception as exc:
            cases.append({'id': entry['id'], 'title': entry.get('title', entry['id']), 'status': 'failed',
                          'interaction': entry['interaction'], 'error': type(exc).__name__ + ': ' + str(exc)})
    failures = []
    for run_path in config.get('failure_runs', []):
        try:
            run = load(root, run_path)
            failures.append({'path': run_path, 'sha256': digest(path_in(root, run_path)),
                             'id': run.get('id'), 'status': run.get('status'), 'error': run.get('error'),
                             'failed_phases': [{'name': p['name'], 'error': p.get('error')} for p in run.get('phases', []) if p['status'] == 'failed']})
        except Exception as exc:
            failures.append({'path': run_path, 'status': 'unreadable', 'error': str(exc)})
    return {'format': 'interaction-twin-saved-evidence-benchmark', 'version': VERSION,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'suite': config.get('id'), 'config': config_path, 'config_sha256': digest(path_in(root, config_path)),
            'implementation_sha256': digest(Path(__file__)), 'cases': cases, 'preserved_failures': failures,
            'policy': {'new_inference': False, 'network': False, 'physical_accuracy_measured': False,
                       'correctness_requires_human_review': True,
                       'identity_counts_are_tracklets_not_confirmed_physical_parts': True,
                       'cross_sampling_coverage_comparison_is_not_accuracy': True}}


def render(report, root_prefix='../../'):
    esc = lambda value: html.escape(str(value), quote=True)
    def cell(value):
        return '<td>' + esc(value) + '</td>'
    rows, details = [], []
    for case in report['cases']:
        if 'metrics' not in case:
            rows.append('<tr>' + cell(case['title']) + cell(case['status']) + '<td colspan="7">' + esc(case['error']) + '</td></tr>')
            continue
        m, rt = case['metrics'], case['runtime']
        coverage = '; '.join(p['track_id'] + ': ' + str(p['full_source_coverage']['numerator']) + '/' + str(p['full_source_coverage']['denominator']) for p in m['parts']) or 'No generated assets'
        seconds = f"{rt['elapsed_s']:.1f}s" if rt['elapsed_s'] is not None else 'Unknown'
        if rt.get('run_status') == 'complete_with_gaps':
            seconds += ' · with gaps'
        human = case['human_review']['statement_correctness']
        human_label = (f"{human['correct']}/{human['accuracy_on_decided_only']['denominator']} correct among decided; {human['unreviewed']} unreviewed"
                       if human['accuracy_on_decided_only']['denominator'] else f"Pending · {human['unreviewed']} unreviewed")
        rows.append('<tr>' + ''.join(cell(v) for v in [case['title'], case['cohort'], m['sample_count'],
                    f"{m['model_audited_plausible_tracklets']}/{m['mask_observed_tracklets']}",
                    m['generated_tracklets'], coverage, seconds, case['source_linkage']['status'], human_label]) + '</tr>')
        parts = ''.join('<tr>' + ''.join(cell(v) for v in [p['track_id'], p['reference_samples'],
                 f"{p['full_source_coverage']['numerator']}/{p['full_source_coverage']['denominator']}",
                 f"{p['tracklet_range_coverage']['numerator']}/{p['tracklet_range_coverage']['denominator']}",
                 p['withheld_motion_samples'], p['unprocessed_or_absent_in_tracklet']]) + '</tr>' for p in m['parts'])
        artifacts = ''.join('<li>' + esc(a['status'] + ': ' + a['path']) + '</li>' for a in case['source_linkage']['bad_artifacts']) or '<li>All referenced artifact bytes match.</li>'
        details.append('<details><summary>' + esc(case['title']) + ' · denominators, provenance and review</summary><p><a href="' + esc(root_prefix + case['interaction']) + '">Interaction JSON</a> · <a href="' + esc(root_prefix + 'workspace/index.html?case=' + Path(case['interaction']).parent.parent.name + '--' + Path(case['interaction']).parent.name) + '">Open in workspace</a></p>'
                       '<table><thead><tr><th>Generated tracklet</th><th>Reference</th><th>Full source, non-reference</th><th>Declared range, non-reference</th><th>Withheld rows</th><th>Absent rows in range</th></tr></thead><tbody>' + parts + '</tbody></table><p>References are excluded from supported coverage. No interpolation is counted. Coverage measures retained conditional candidates, not physical pose accuracy.</p><ul>' + artifacts + '</ul><pre>' + esc(json.dumps({'structural_validation': case['structural_validation'], 'upstream_join_validation': case['upstream_join_validation'], 'runtime': rt, 'human_review': case['human_review'], 'models': case['models']}, indent=2)) + '</pre></details>')
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Interaction Twin · evidence benchmark</title><style>body{font:16px/1.55 system-ui;background:#f3f5f8;color:#172333;max-width:1500px;margin:40px auto;padding:0 24px}h1{font-size:36px;line-height:1.1}table{width:100%;border-collapse:collapse;background:white;font-size:14px}td,th{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid #dce3ed}th{background:#e7edf5}main{overflow-x:auto}details{background:white;border:1px solid #dce3ed;padding:18px;margin:16px 0}summary{cursor:pointer;font-weight:650}pre{max-height:420px;overflow:auto;font-size:12px;background:#f5f7fa;padding:16px}a{color:#155cb5}.notice{padding:18px;border-left:4px solid #daa220;background:#fff7df}.meta{color:#526275;font-size:14px}</style><h1>Saved-evidence benchmark</h1><p class="meta">' + esc(report['suite']) + ' · generated ' + esc(report['created_at']) + ' · <a href="report.json">Full JSON and hashes</a></p><p class="notice">Counts are model-proposed tracklets, not confirmed physical parts. Conditional pose coverage is not physical accuracy. Statement and identity correctness stay pending until a human reviews the source. Different sampling rates and scopes make these diagnostic comparisons, not an accuracy ranking.</p><main><table><thead><tr><th>Case</th><th>Cohort</th><th>Samples</th><th>Audited / masked tracklets</th><th>Generated tracklets</th><th>Supported / non-reference source samples</th><th>Recorded run time</th><th>Source linkage</th><th>Human statement review</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></main>' + ''.join(details) + '<details><summary>Preserved failures</summary><pre>' + esc(json.dumps(report['preserved_failures'], indent=2)) + '</pre></details><p>No new model inference, media upload or API usage was performed to generate this report.</p></html>'
