"""Auditable local video-to-evidence orchestration; no synthetic success fallbacks.

The subprocess argument vectors are built here, never supplied by the browser.
Phase receipts hash inputs, relevant code, dependencies and all produced files.
"""
from __future__ import annotations

import hashlib
import concurrent.futures
import fcntl
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
LOADED_SOURCE = Path(__file__).read_bytes()
LOADED_SOURCE_SHA256 = hashlib.sha256(LOADED_SOURCE).hexdigest()
PHASES = ['prepare', 'discover', 'segment', 'audit', 'select', 'reconstruct', 'pose', 'motion', 'procedure_proposal', 'procedure',
          'part_ledger', 'mechanism_topology', 'mechanism_scene', 'object_program', 'workspace']
SLUG = re.compile(r'[a-z0-9][a-z0-9-]{0,63}\Z')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def confined(path, root=ROOT):
    result = Path(path).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError('Path escapes the permitted directory')
    return result


def rel(path):
    return str(confined(path).relative_to(ROOT))


def public_url(path):
    return '/files/' + quote(rel(path), safe='/')


def file_receipts(paths):
    files = set()
    for candidate in paths:
        candidate = confined(candidate)
        if candidate.is_dir():
            files.update(p for p in candidate.rglob('*') if p.is_file())
        elif candidate.is_file():
            files.add(candidate)
        else:
            raise ValueError('Artifact is missing: ' + rel(candidate))
    # Resolve every child as well: a nested symlink may not escape confinement.
    return [{'path': rel(p), 'sha256': sha(confined(p)), 'bytes': p.stat().st_size} for p in sorted(files)]


def verify_receipts(receipts):
    if not receipts:
        return False
    try:
        return all((p := confined(ROOT / r['path'])).is_file() and p.stat().st_size == r['bytes'] and sha(p) == r['sha256'] for r in receipts)
    except (OSError, ValueError, KeyError):
        return False


def load_profile(name):
    if not SLUG.fullmatch(name.replace('_', '-')):
        raise ValueError('Invalid profile identifier')
    profile = read(ROOT / 'profiles' / (name + '.json'))
    if profile.get('id') != name or profile.get('mode') not in {'local_inference', 'verified_evidence', 'preflight'}:
        raise ValueError('Invalid profile contract')
    if profile['mode'] == 'local_inference':
        for key, lo, hi in [('fps', .1, 10), ('max_dimension', 128, 1920), ('max_frames', 1, 1200),
                            ('max_duration_s', 1, 1800), ('phase_timeout_s', 1, 3600), ('reasoning_timeout_s', 1, 900),
                            ('mesh_resolution', 32, 256), ('reconstruct_limit', 1, 8)]:
            value = profile[key]
            if isinstance(value, bool) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError('Profile has an unsafe resource bound: ' + key)
        if type(profile['reconstruct_limit']) is not int or (profile.get('reconstruction_provider', 'triposr_mps') != 'sam3d_native' and profile['reconstruct_limit'] > 3):
            raise ValueError('Component budget exceeds its fixed reconstruction provider bound')
        validate_providers(profile)
    return profile


def validate_providers(profile):
    """Only reviewed provider adapters are executable, with explicit bounds."""
    reasoning = profile.get('reasoning_provider', 'codex_cli')
    segmentation = profile.get('segmentation_provider', 'sam2_mps')
    if reasoning not in {'codex_cli', 'meta_api'}:
        raise ValueError('Unsupported reasoning provider; no implicit fallback')
    if segmentation not in {'sam2_mps', 'meta_sam31_api'}:
        raise ValueError('Unsupported segmentation provider; no implicit fallback')
    reconstruction = profile.get('reconstruction_provider', 'triposr_mps')
    if reconstruction not in {'triposr_mps', 'sam3d_native'}:
        raise ValueError('Unsupported reconstruction provider; no implicit fallback')
    if reconstruction == 'sam3d_native' and (reasoning != 'meta_api' or segmentation != 'meta_sam31_api'
                                               or profile.get('native_motion_mode') != 'pending_fresh_alignment'):
        raise ValueError('SAM 3D upload profile requires reviewed Meta perception and explicit native motion boundary')
    if 'observed_camera_mode' in profile and profile['observed_camera_mode'] not in {'assumed', 'moge_prior'}:
        raise ValueError('Unsupported automatic camera mode')
    segment_mode = profile.get('segmentation_mode', 'images')
    if segment_mode not in {'images', 'video_boxes'} or (segment_mode == 'video_boxes' and segmentation != 'meta_sam31_api'):
        raise ValueError('Unsupported segmentation mode; no implicit image/video fallback')
    if reasoning == 'meta_api':
        if profile.get('reasoning_model') not in {'muse-spark-1.3'}:
            raise ValueError('Meta reasoning model is not in the reviewed profile allowlist')
        tokens = profile.get('max_output_tokens')
        if isinstance(tokens, bool) or not isinstance(tokens, int) or not 256 <= tokens <= 32768:
            raise ValueError('Meta profile needs a bounded max_output_tokens')
    mechanism_mode = profile.get('mechanism_pipeline_mode')
    if mechanism_mode is not None:
        if mechanism_mode != 'muse_constrained_v1' or reconstruction != 'sam3d_native':
            raise ValueError('Constrained mechanism pipeline requires the reviewed SAM 3D provider')
        import mechanism_pipeline
        mechanism_pipeline.configuration(profile, root=ROOT)
    return reasoning, segmentation


def reasoning_dispatch(profile, phase):
    """Return a fixed script/option vector, never an arbitrary browser command."""
    reasoning, _ = validate_providers(profile)
    if phase not in {'discover', 'audit', 'procedure_proposal'}:
        raise ValueError('Not a reasoning phase')
    if reasoning == 'meta_api':
        script = ('stage5_meta_propose.py' if phase == 'procedure_proposal' else
                  'discover_rigid_components.py' if phase == 'discover' and profile.get('discovery_mode') == 'rigid_components_v1' else
                  'stage2_meta_reason.py')
        return script, ['--model', profile['reasoning_model'], '--max-output-tokens', str(profile['max_output_tokens'])]
    return ('stage5_compile.py' if phase == 'procedure_proposal' else 'stage2_discover.py'), []


def stage_argv(profile, python, script, arguments):
    """A fixed additional estimate envelope for this approved new profile."""
    scoped = {'discover_rigid_components.py', 'stage2_meta_reason.py', 'stage2_meta_video_segment.py',
              'stage2_meta_segment.py', 'stage3_meta_select_track_views.py', 'stage5_meta_propose.py'}
    if profile.get('overnight_task_budget_plan') and script in scoped:
        from overnight_budget import PLAN
        if profile['overnight_task_budget_plan'] != PLAN:
            raise ValueError('Unrecognized overnight estimate scope')
        target_args = list(arguments)
        target_script = script
        if profile.get('admin_memory_context'):
            context = profile['admin_memory_context']
            target_args = ['--memory-context-ref', json.dumps({k:v for k,v in context.items() if k != 'source_sha256'}, separators=(',', ':')),
                           '--source-sha256', context['source_sha256'], script, *target_args]
            target_script = 'run_admin_stage.py'
        return [python, ROOT/'scripts/run_overnight_stage.py', '--task-budget-plan', PLAN, target_script, *target_args]
    if profile.get('admin_memory_context') and script in scoped:
        context = profile['admin_memory_context']
        return [python, ROOT / 'scripts/run_admin_stage.py', '--memory-context-ref',
                json.dumps({k:v for k,v in context.items() if k != 'source_sha256'}, separators=(',', ':')),
                '--source-sha256', context['source_sha256'], script, *arguments]
    if profile.get('increment_budget_scope') == 'autonomous_20261031' and script in scoped:
        return [python, ROOT / 'scripts/run_budgeted_stage.py', script, *arguments]
    return [python, ROOT / 'scripts' / script, *arguments]


def provider_dependencies(profile, *, segmentation=False):
    """Fingerprint adapter code, but never credentials or the mutable ledger."""
    reasoning, segment = validate_providers(profile)
    if (segmentation and segment != 'meta_sam31_api') or (not segmentation and reasoning != 'meta_api'):
        return []
    dependencies = sorted(path for path in ROOT.glob('meta_*.py') if path.is_file())
    if profile.get('increment_budget_scope') == 'autonomous_20261031':
        dependencies += [ROOT / 'increment_budget.py', ROOT / 'scripts/run_budgeted_stage.py',
                         ROOT / 'output/autonomous-improvements-20261031/plan.json']
    if profile.get('admin_memory_context'):
        dependencies += [ROOT / 'admin_memory_context.py', ROOT / 'agent_memory.py',
                         ROOT / 'task_taxonomy.py', ROOT / 'increment_budget.py', ROOT / 'mechanism_reasoning.py', ROOT / 'scripts/run_admin_stage.py',
                         *[ROOT/n for n in ['admin_machine_context.py', 'admin_execution_policy.py', 'admin_visual_audit.py'] if (ROOT/n).is_file()],
                         ROOT / profile['admin_memory_context']['path'],
                         ROOT / 'output/constrained-kinematics-20261102/task-budget-plan.json',
                         ROOT / 'output/autonomous-improvements-20261031/plan.json']
    if profile.get('overnight_task_budget_plan'):
        from mechanism_pass_budget import PLAN as MECHANISM_PASS_PLAN
        dependencies += [ROOT/'overnight_budget.py', ROOT/'scripts/run_overnight_stage.py', ROOT/profile['overnight_task_budget_plan'],
                         ROOT/'mechanism_pass_budget.py',ROOT/MECHANISM_PASS_PLAN]
    return dependencies


def provider_disclosure(profile):
    reasoning, segmentation = validate_providers(profile)
    reasoning_name = 'Meta Model API' if reasoning == 'meta_api' else 'OpenAI/Codex'
    return ('This profile sends sampled frames to ' + reasoning_name + ' for hosted reasoning' +
            (' including one guarded connection-graph request' if profile.get('mechanism_pipeline_mode') == 'muse_constrained_v1' else '') +
            (' and to Meta for SAM 3.1 segmentation' if segmentation == 'meta_sam31_api' else '') +
            (' and transfers selected RGB/mask inputs to the configured SAM 3D GPU worker'
             if profile.get('reconstruction_provider') == 'sam3d_native' else '') +
            '; explicit Run consent is required')


def label_generated_meta_viewer(path, program):
    """Correct a new generated viewer label without modifying its frozen compiler."""
    if program.get('reasoning', {}).get('provider') != 'meta_api':
        return False
    path = confined(path)
    text = path.read_text()
    old = '; temporary Meta reasoning substitute. No hand-authored semantic story.'
    new = '; official Meta Model API inference. No hand-authored semantic story.'
    if text.count(old) != 1:
        raise ValueError('Generated viewer provider label changed; inspect before publishing')
    path.write_text(text.replace(old, new, 1))
    return True


def validate_source(path, profile):
    if path is None:
        raise ValueError('Choose an input video or saved source case')
    path = Path(path).resolve()
    if not path.is_file() or path.suffix.lower() not in {'.mp4', '.mov', '.m4v', '.webm'}:
        raise ValueError('A readable MP4, MOV, M4V or WebM video is required')
    if path.stat().st_size > 1024 * 1024 * 1024:
        raise ValueError('Video exceeds the one GiB local input limit')
    probe = subprocess.run(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)],
                           check=True, capture_output=True, text=True, timeout=30)
    data = json.loads(probe.stdout)
    streams = [s for s in data['streams'] if s['codec_type'] == 'video']
    duration = float(data['format'].get('duration', 0))
    if not streams or not math.isfinite(duration) or duration <= 0:
        raise ValueError('The video has no finite positive duration')
    if profile['mode'] == 'local_inference' and (duration > profile['max_duration_s'] or math.ceil(duration * profile['fps']) > profile['max_frames']):
        raise ValueError('Video exceeds this profile duration/frame budget; choose a shorter clip')
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size, 'duration_s': duration}


def component_category(label):
    """A conservative budget-diversity key, never a physical identity claim.

    Only observation/position/ordinal modifiers are removed from model labels.
    Structural words such as base, lid, upper and lower remain distinct.
    """
    modifiers = {'a', 'an', 'the', 'in', 'on', 'of', 'at', 'to', 'from', 'and',
                 'first', 'second', 'third', 'fourth', 'left', 'right', 'visible',
                 'loose', 'handled', 'seated', 'inside', 'outside', 'small', 'large',
                 'another', 'other', 'closed', 'open', 'candidate', 'observed',
                 # These describe when/where a tracklet is observed, not a new
                 # component family. Keep its separate physical candidate ID.
                 'initial', 'initially', 'later', 'late', 'earlier', 'early',
                 'final', 'finally', 'subsequent', 'previous', 'reappearing',
                 'image', 'video', 'screen', 'frame', 'leftmost', 'rightmost',
                 'foreground', 'background', 'near', 'far', 'recessed',
                 'moving', 'movable', 'stationary'}
    tokens = [word for word in re.findall(r'[a-z]+', str(label).lower()) if word not in modifiers]
    return ' '.join(tokens) or 'unspecified'


def choose_components(case, records, annotations, tracks, limit, *, segmentation=None):
    """Select bounded distinct observations; preserve unresolved tracklet identities.

    Repeated labels are deferred until label diversity has had a turn. A track is
    suppressed as a redundant observation only for an exact duplicate candidate
    ID or same-category near-identical masks at at least two shared source PTS.
    This does not merge physical identities or silently rewrite discovery.
    """
    if type(limit) is not int or not 1 <= limit <= 8:
        raise ValueError('Component selection budget must be an integer between one and eight')
    case = confined(case)
    specs = {row['id']: row for row in annotations['tracks']}
    observed = {row['track_id']: {f['sample_index']: f for f in row.get('frames', [])
                               if f.get('status') == 'candidate_visible'} for row in tracks['tracks']}
    decisions, candidates = [], []
    for row in records:
        tid = row['track_id']
        spec = specs[tid]
        decision = {'track_id': tid, 'model_label': spec.get('label', tid),
                    'category_key': component_category(spec.get('label', tid)),
                    'identity_status': spec.get('identity_status', 'unverified'),
                    'candidate_entity_ids': spec.get('candidate_entity_ids', []),
                    'physical_identity_merged': False}
        decisions.append(decision)
        if not row.get('selection'):
            decision.update(status='skipped_no_eligible_view', reason=row.get('status', 'No eligible model-audited view'))
            continue
        selection = confined(case / row['selection'], case)
        quality = read(selection)['selected'][0]['score']
        if isinstance(quality, bool) or not isinstance(quality, (int, float)) or not math.isfinite(quality):
            raise ValueError('Nonfinite component quality score')
        decision.update(selection=rel(selection), score=quality)
        candidates.append(decision)
    candidates.sort(key=lambda row: (-row['score'], row['track_id']))
    unique, categories = [], set()
    for candidate in candidates:
        duplicate = None
        for earlier in unique:
            if candidate['track_id'] == earlier['track_id']:
                duplicate = {'track_id': earlier['track_id'], 'reason': 'repeated_identical_track_id'}
                break
            if candidate['category_key'] != earlier['category_key']:
                continue
            left, right = observed.get(candidate['track_id'], {}), observed.get(earlier['track_id'], {})
            shared = sorted(set(left) & set(right))
            # Bounded comparison: at most three evenly spaced shared source frames.
            sample_ids = sorted({shared[round(j * (len(shared)-1) / 2)] for j in range(3)}) if len(shared) >= 2 else []
            checks = []
            for sample in sample_ids:
                a, b = left[sample], right[sample]
                if abs(a['timestamp_s'] - b['timestamp_s']) > 1e-6:
                    raise ValueError('Overlapping observations have different source timestamps')
                from PIL import Image, ImageChops
                if segmentation is None:
                    raise ValueError('Shared-mask comparison needs the source segmentation directory')
                segmentation = confined(segmentation, case)
                with Image.open(confined(segmentation / a['raw_mask_file'], segmentation)) as raw:
                    am = raw.convert('L').point(lambda x: 255 if x else 0)
                with Image.open(confined(segmentation / b['raw_mask_file'], segmentation)) as raw:
                    bm = raw.convert('L').point(lambda x: 255 if x else 0)
                if am.size != bm.size:
                    raise ValueError('Duplicate comparison masks have different source dimensions')
                intersection = ImageChops.darker(am, bm).histogram()[255]
                union = ImageChops.lighter(am, bm).histogram()[255]
                checks.append({'sample_index': sample, 'timestamp_s': a['timestamp_s'],
                               'mask_iou': intersection / union if union else 0})
            if len(checks) >= 2 and all(row['mask_iou'] >= .92 for row in checks):
                duplicate = {'track_id': earlier['track_id'], 'reason': 'same_model_label_category_and_near_identical_shared_masks', 'checks': checks}
                break
        if duplicate:
            candidate.update(status='skipped_redundant_observation', redundant_with=duplicate,
                             reason='Budget excludes near-duplicate observations; no physical identity merge is asserted')
        else:
            candidate['diversity_round'] = 2 if candidate['category_key'] in categories else 1
            categories.add(candidate['category_key'])
            unique.append(candidate)
    ranked = sorted(unique, key=lambda row: (row['diversity_round'], -row['score'], row['track_id']))
    selected = ranked[:limit]
    for index, row in enumerate(ranked):
        row.update(status='selected' if index < limit else 'skipped_budget',
                   reason=('Distinct model-label category receives budget priority' if row['diversity_round'] == 1
                           else 'Same-category tracklet retained separately; physical correspondence remains unresolved'))
        if index < limit:
            row['selection_order'] = index + 1
    return selected, {'schema_version': '0.1', 'budget': limit, 'decisions': decisions,
                      'method': 'quality_then_model_label_diversity_with_bounded_shared_mask_duplicate_check',
                      'physical_identities_merged': False, 'max_duplicate_mask_comparisons_per_pair': 3,
                      'limitations': ['Model labels are fallible category hints, not a taxonomy or verified identity.',
                                      'Different-category masks are never merged; occluded reappearances retain distinct IDs.',
                                      'Redundant observation suppression does not establish physical identity.']}


class Cancelled(RuntimeError):
    pass


class Runner:
    """One synchronous worker. HTTP invokes it from a single executor thread."""
    def __init__(self, run_id=None, *, run_root=None):
        self.base = confined(run_root or ROOT / 'data/flow-runs')
        self.base.mkdir(parents=True, exist_ok=True)
        self.id = run_id or ('run-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-' + os.urandom(3).hex())
        if not SLUG.fullmatch(self.id):
            raise ValueError('Invalid run identifier')
        self.directory = confined(self.base / self.id, self.base)
        self.directory.mkdir(exist_ok=True)
        self.path = self.directory / 'run.json'
        self.code_snapshot = self.directory / ('pipeline-source-' + LOADED_SOURCE_SHA256[:16] + '.py')
        if not self.code_snapshot.exists():
            self.code_snapshot.write_bytes(LOADED_SOURCE)
        log = self.directory / 'pipeline.log'
        if not log.exists():
            log.write_text('Interaction Twin local pipeline. Subprocess commands, if any, are appended below. Readiness/evidence-only runs perform no new learned inference.\n')
        self.cancel_event = threading.Event()
        self.termination_lock = threading.Lock()
        self.process = None
        self.state = read(self.path) if self.path.exists() else {'id': self.id, 'created_at': now(), 'status': 'created', 'phase': None, 'progress': 0,
                                                              'phases': [], 'artifacts': {}, 'logs_url': public_url(self.directory / 'pipeline.log')}

    def save(self):
        self.state['updated_at'] = now()
        write(self.path, self.state)

    def cancel(self):
        self.cancel_event.set()
        self._terminate()

    def _terminate(self):
        # The API caller and worker can observe cancellation simultaneously.
        # Reap/signalling must have a single owner, even after the group exits.
        with self.termination_lock:
            process = self.process
            if process and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=4)
                except ProcessLookupError:
                    pass

    def command(self, args, timeout):
        if self.cancel_event.is_set():
            raise Cancelled('Cancelled by user')
        with (self.directory / 'pipeline.log').open('a') as log:
            log.write('\n' + now() + ' ' + json.dumps([str(x) for x in args]) + '\n')
            log.flush()
            self.process = subprocess.Popen([str(x) for x in args], cwd=ROOT, stdin=subprocess.DEVNULL,
                                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            started = time.monotonic()
            while self.process.poll() is None:
                if self.cancel_event.wait(.2):
                    self._terminate()
                    raise Cancelled('Cancelled by user; subprocess group terminated')
                if time.monotonic() - started > timeout:
                    self._terminate()
                    raise TimeoutError(f'Phase command exceeded {timeout} seconds')
            if self.cancel_event.is_set():
                raise Cancelled('Cancelled by user; subprocess group terminated')
            if self.process.returncode:
                raise RuntimeError(f'Phase command failed with exit status {self.process.returncode}; inspect its log')
            self.process = None

    def phase(self, name, dependencies, action, *, optional=False):
        if self.cancel_event.is_set():
            raise Cancelled('Cancelled by user')
        self.state.update(phase=name, progress=round(PHASES.index(name) / len(PHASES), 3))
        previous = next((p for p in self.state['phases'] if p['name'] == name), None)
        try:
            inputs = file_receipts(dependencies)
        except (OSError, ValueError) as exc:
            receipt = {'name': name, 'status': 'failed', 'error': 'Dependency unavailable: ' + str(exc),
                       'attempt': (previous or {}).get('attempt', 0) + 1, 'completed_at': now()}
            self.state['phases'] = [p for p in self.state['phases'] if p['name'] != name] + [receipt]
            self.save()
            if not optional:
                raise
            return {'unavailable': receipt['error']}
        identity = fingerprint({'profile': self.profile, 'source': self.state.get('source'), 'inputs': inputs, 'runner_code': LOADED_SOURCE_SHA256})
        pending_native = name == 'reconstruct' and bool((previous or {}).get('result', {}).get('pending_components'))
        if previous and not pending_native and previous.get('signature') == identity and previous['status'] in {'complete', 'complete_with_gaps', 'reused_verified', 'reused_verified_with_gaps'} and verify_receipts(previous.get('files')):
            previous['status'] = 'reused_verified_with_gaps' if previous.get('result', {}).get('has_gaps') else 'reused_verified'
            previous['reverified_at'] = now()
            self.save()
            return previous.get('result', {})
        attempt = (previous or {}).get('attempt', 0) + 1
        receipt = {'name': name, 'status': 'running', 'started_at': now(), 'signature': identity, 'inputs': inputs, 'attempt': attempt}
        self.state['phases'] = [p for p in self.state['phases'] if p['name'] != name] + [receipt]
        self.save()
        try:
            result, artifacts = action(attempt)
            receipt.update(status='complete_with_gaps' if result.get('has_gaps') else 'complete',
                           result=result, files=file_receipts(artifacts), completed_at=now())
            self.save()
            return result
        except (Cancelled, KeyboardInterrupt):
            receipt.update(status='cancelled', completed_at=now())
            self.save()
            raise
        except Exception as exc:
            receipt.update(status='failed', error=str(exc), completed_at=now())
            self.save()
            if not optional:
                raise
            return {'unavailable': str(exc)}

    def component_batch(self, name, attempt, case, components, action):
        """Run independent component operations with durable outcomes and logs.

        A model can fail one component without discarding its siblings. Cancellation
        is never converted into an optional component failure or model success.
        """
        if not components:
            raise ValueError('No eligible component inputs for ' + name)
        receipt_dir = case / f'flow-{name}-receipts-{attempt}'
        receipt_dir.mkdir()
        report_path = receipt_dir / 'components.json'
        report = {'phase': name, 'components': [], 'physical_accuracy_verified': False}
        directories = [receipt_dir]
        for index, component in enumerate(components):
            if self.cancel_event.is_set():
                raise Cancelled('Cancelled before next component')
            directory = case / f'flow-{name}-{attempt}-{index+1}'
            record = {'track_id': component['track_id'], 'input': component,
                      'directory': rel(directory), 'status': 'running', 'started_at': now()}
            report['components'].append(record)
            write(report_path, report)
            offset = (self.directory / 'pipeline.log').stat().st_size
            try:
                output = action(component, directory)
                if self.cancel_event.is_set():
                    raise Cancelled('Cancelled during component operation')
                record.update(output=output, status='complete_with_gaps' if output.get('has_gaps') else 'complete')
            except (Cancelled, KeyboardInterrupt):
                record.update(status='cancelled', completed_at=now())
                write(report_path, report)
                raise
            except Exception as exc:
                record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            finally:
                log_path = receipt_dir / f'component-{index+1}.log'
                with (self.directory / 'pipeline.log').open('rb') as log:
                    log.seek(offset)
                    log_path.write_bytes(log.read())
                record.update(log=rel(log_path), completed_at=now())
                if directory.exists():
                    directories.append(directory)
                write(report_path, report)
        successes = [record for record in report['components'] if record['status'] in {'complete', 'complete_with_gaps'}]
        failures = [record for record in report['components'] if record['status'] == 'failed']
        report.update(completed_at=now(), successful_components=len(successes), failed_components=len(failures),
                      has_gaps=any(record['status'] != 'complete' for record in report['components']))
        write(report_path, report)
        return {'components': report['components'], 'failures': failures, 'has_gaps': report['has_gaps'],
                'receipt': rel(report_path)}, directories

    def native_component_batch(self, attempt, case, components, action):
        """At most two independent bridge calls; state writes have one lock.

        These actions never call Runner.command (whose child process is single
        owner). The bridge owns per-job logs, attempts and cluster cancellation.
        """
        if not components:
            raise ValueError('No eligible component inputs for SAM 3D')
        receipt_dir = case / f'flow-model-receipts-{attempt}'
        receipt_dir.mkdir()
        path = receipt_dir / 'components.json'
        records = [{'track_id': c['track_id'], 'input': c,
                    'directory': rel(case / f'flow-model-{attempt}-{i + 1}'), 'status': 'queued'}
                   for i, c in enumerate(components)]
        report = {'phase': 'model', 'provider': 'sam3d_native', 'maximum_parallel_workers': 2,
                  'components': records, 'physical_accuracy_verified': False}
        state_lock = threading.Lock()
        write(path, report)
        def worker(index):
            row = records[index]
            with state_lock:
                row.update(status='running', started_at=now())
                write(path, report)
            try:
                if self.cancel_event.is_set():
                    raise Cancelled('Cancelled before SAM 3D component dispatch')
                output = action(components[index], ROOT / row['directory'])
                if self.cancel_event.is_set():
                    raise Cancelled('Cancelled during SAM 3D component dispatch; worker evidence retained')
                with state_lock:
                    row.update(output=output, status='complete_with_gaps' if output.get('has_gaps') else 'complete')
            except (Cancelled, KeyboardInterrupt):
                with state_lock:
                    row['status'] = 'cancelled'
                self.cancel_event.set()
                raise
            except Exception as exc:
                with state_lock:
                    row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            finally:
                with state_lock:
                    log = receipt_dir / f'component-{index + 1}.log'
                    log.write_text('SAM 3D worker evidence is linked from this component output.\n'
                                   + json.dumps(row.get('output', row.get('error', {'status': row['status']})), allow_nan=False) + '\n')
                    row.update(log=rel(log), completed_at=now())
                    write(path, report)
        cancelled = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix='sam3d-part') as executor:
            futures = [executor.submit(worker, i) for i in range(len(components))]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except (Cancelled, KeyboardInterrupt) as exc:
                    cancelled = exc
                    self.cancel_event.set()
        if cancelled is not None or self.cancel_event.is_set():
            raise Cancelled('Cancelled SAM 3D batch; partial native evidence retained') from cancelled
        failures = [r for r in records if r['status'] == 'failed']
        report.update(completed_at=now(), successful_components=sum(r['status'] in {'complete', 'complete_with_gaps'} for r in records),
                      failed_components=len(failures), has_gaps=any(r['status'] != 'complete' for r in records))
        write(path, report)
        directories = [receipt_dir] + [ROOT / r['directory'] for r in records if (ROOT / r['directory']).is_dir()]
        return {'components': records, 'failures': failures, 'has_gaps': report['has_gaps'], 'receipt': rel(path)}, directories

    def execute(self, profile_name, *, input_path=None, case_id=None, memory_context_path=None, publication_mode=None, overnight_task_budget_plan=None):
        self.profile = load_profile(profile_name)
        request = {'profile': profile_name, 'input_path': str(Path(input_path).resolve()) if input_path else None, 'case_id': case_id}
        if overnight_task_budget_plan is not None:
            from overnight_budget import PLAN, plan
            if overnight_task_budget_plan != PLAN or publication_mode != 'admin_review':
                raise ValueError('Overnight processing requires the fixed private approved policy')
            plan(ROOT)
            self.profile['overnight_task_budget_plan'] = PLAN
            from mechanism_pass_budget import PLAN as MECHANISM_PASS_PLAN,validate_profile
            self.profile['mechanism_task_budget_plan']=MECHANISM_PASS_PLAN
            validate_profile(self.profile,root=ROOT)
            request['overnight_task_budget_plan'] = PLAN
        if publication_mode not in {None, 'admin_review'}:
            raise ValueError('Unsupported private publication mode')
        if publication_mode is not None:
            self.profile['publication_mode'] = publication_mode
            request['publication_mode'] = publication_mode
        if memory_context_path is not None:
            if publication_mode != 'admin_review' or self.profile.get('reasoning_provider') != 'meta_api' or self.profile['mode'] != 'local_inference':
                raise ValueError('Owner feedback requires an explicit private Meta inference run')
            from admin_memory_context import load_snapshot
            _, memory_ref = load_snapshot(memory_context_path, root=ROOT)
            self.profile['admin_memory_context'] = memory_ref
            request['admin_memory_context'] = memory_ref
        if self.state.get('request') and self.state['request'] != request:
            raise ValueError('Resume requires the identical input/case/profile request')
        self.state.update(request=request, profile=self.profile, status='running', error=None,
                          provenance={'manual_geometry_used': False, 'manual_pose_used': False, 'manual_boxes_used': False,
                                      'cluster_contacted': False, 'gated_weights_downloaded': False,
                                      'owner_annotation_context': self.profile.get('admin_memory_context'),
                                      'owner_annotations_are_observed_facts': False,
                                      'model_training_performed': False,
                                      'orchestrator_source': {'path': rel(self.code_snapshot), 'sha256': LOADED_SOURCE_SHA256}})
        self.save()
        try:
            if self.profile['mode'] == 'preflight':
                self._preflight()
            elif self.profile['mode'] == 'verified_evidence':
                self._saved(case_id)
            else:
                if case_id:
                    if not SLUG.fullmatch(case_id):
                        raise ValueError('Invalid case identifier')
                    case = confined(ROOT / 'data' / case_id, ROOT / 'data')
                    manifest = read(case / 'manifest.json')
                    input_path = case / manifest['source'].get('original_file', manifest['source']['file'])
                source = validate_source(input_path, self.profile)
                if memory_context_path is not None:
                    # Retrieval is tied to the ORIGINAL upload, not a normalized
                    # video whose byte identity changes during preparation.
                    load_snapshot(memory_context_path, root=ROOT, expected_sha256=memory_ref['sha256'], source_sha256=source['sha256'])
                    self.profile['admin_memory_context'] = dict(memory_ref, source_sha256=source['sha256'])
                self.state['source'] = source
                self.save()
                lock_path = self.base / '.model-worker.lock'
                with lock_path.open('a+') as lock:
                    try:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        raise RuntimeError('Another pipeline owns the local model worker; retry after it finishes')
                    self._infer(Path(input_path), source)
            self.state.update(status='complete_with_gaps' if self.state.get('observed_geometry_has_gaps') or self.state.get('native_alignment_has_gaps') or any(p['status'] in {'failed', 'complete_with_gaps', 'reused_verified_with_gaps'} for p in self.state['phases']) else 'complete',
                              phase=None, progress=1, completed_at=now())
        except Cancelled as exc:
            self.state.update(status='cancelled', error=str(exc), completed_at=now())
        except KeyboardInterrupt:
            self.cancel_event.set()
            self.state.update(status='cancelled', error='Interrupted by user; subprocess group terminated', completed_at=now())
        except Exception as exc:
            self.state.update(status='failed', error=str(exc), completed_at=now())
        finally:
            self._terminate()
            self.save()
        return self.state

    def _preflight(self):
        checks = {'profile': self.profile['id'], 'status': 'readiness_report_only_no_inference', 'checked_at': now(),
                  'cluster_contacted': False, 'credentials_read': False, 'downloads': False,
                  'sam31_adapter_present': (ROOT / 'scripts/sam31_segment.py').is_file(),
                  'sam3d_adapter_present': (ROOT / 'scripts/stage3_sam3d.py').is_file(),
                  'required_for_sam31': ['separate CUDA runtime', 'approved downloaded weights', 'actual inference and regression comparison'],
                  'required_for_sam3d': ['Linux NVIDIA CUDA environment', 'approved downloaded weights', 'native coordinate validation on actual output'],
                  'meta_reasoning_status': 'Budgeted Meta API profiles are available when configured; this preflight performs no authentication or inference',
                  'ready_for_gated_inference': False}
        report = self.directory / 'preflight.json'
        write(report, checks)
        self.state['artifacts']['preflight'] = rel(report)
        self.save()

    def _saved(self, case_id):
        selected = next((x for x in list_cases() if x['id'] == case_id), None)
        if not selected:
            raise ValueError('Select a saved case with a compiled interaction artifact')
        path = confined(ROOT / selected['interaction_path'])
        import stage5
        stage5.validate_interaction(read(path))
        program = read(path)
        receipts = file_receipts([path, ROOT / program['source']['path']] + [ROOT / item['path'] for item in program['inputs']])
        report = self.directory / 'verified-evidence.json'
        write(report, {'status': 'saved_evidence_verified_no_new_inference', 'case_id': case_id,
                       'program': rel(path), 'files': receipts, 'fresh_model_inference': False})
        self.state['artifacts'].update(interaction=rel(path), verification=rel(report), case=program['case_id'])
        self.state['provenance']['fresh_model_inference'] = False
        self.save()

    def _infer(self, input_path, source):
        python = ROOT / '.venv-sam/bin/python'
        if not python.is_file():
            raise RuntimeError('Prepared .venv-sam runtime is unavailable; no automatic installation or fallback')
        p = self.profile
        timeout = p['phase_timeout_s']
        reasoning_provider, segmentation_provider = validate_providers(p)
        native_constrained = (p.get('reconstruction_provider') == 'sam3d_native' and
                              p.get('mechanism_pipeline_mode') == 'muse_constrained_v1')
        if native_constrained:
            # Completing the routing stages is not evidence that the destination
            # can align this upload. Resolve that separately after its real run.
            self.state['native_alignment_has_gaps'] = True
        def cli(script, *args):
            self.command(stage_argv(p, python, script, args), timeout)
        case_name = ('flow-' + self.id.removeprefix('run-'))[:60]
        case = ROOT / 'data' / case_name
        def prepare(attempt):
            # A failed partial preparation is retained with a new case suffix.
            selected = case if attempt == 1 else ROOT / 'data' / (case_name + '-' + str(attempt))
            ffmpeg = ROOT / '.venv/lib/python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1'
            args = ['--case', selected.name, '--file', input_path, '--fps', p['fps'], '--max-dimension', p['max_dimension']]
            if ffmpeg.is_file():
                args += ['--ffmpeg', ffmpeg]
            cli('prepare_recording.py', *args)
            manifest = read(selected / 'manifest.json')
            if manifest['source']['original_sha256'] != source['sha256']:
                raise ValueError('Prepared source differs from requested original')
            return {'case': rel(selected)}, [selected / 'manifest.json', selected / 'normalization.json', selected / 'source', selected / 'frames', selected / 'contact-sheet-01.jpg']
        prepared = self.phase('prepare', [ROOT / 'scripts/prepare_recording.py', ROOT / 'twin.py'], prepare)
        case = ROOT / prepared['case']
        self.state['artifacts']['case'] = case.name
        self.save()
        def discover(attempt):
            run = f'flow-discovery-{attempt}'
            script, options = reasoning_dispatch(p, 'discover')
            cli(script, 'discover', '--case', case.name, '--run', run, '--timeout-s', p['reasoning_timeout_s'], *options)
            directory = case / 'reasoning' / run
            config = ROOT / read(directory / 'metadata.json')['tracking_config']
            return {'directory': rel(directory), 'discovery': rel(directory / 'discovery.json'), 'annotations': rel(config)}, [directory, config]
        discovery_script, _ = reasoning_dispatch(p, 'discover')
        disc = self.phase('discover', [case / 'manifest.json', case / 'frames', ROOT / 'scripts/stage2_discover.py',
                                      ROOT / 'scripts' / discovery_script, ROOT / 'schemas/discovery.schema.json',
                                      *([ROOT / 'rigid_discovery.py', ROOT / 'discovery_repair.py'] if p.get('discovery_mode') == 'rigid_components_v1' else []),
                                      *provider_dependencies(p)], discover)
        config = ROOT / disc['annotations']
        def segment(attempt):
            run = f'flow-components-{attempt}'
            if segmentation_provider == 'meta_sam31_api':
                segment_script = 'stage2_meta_video_segment.py' if p.get('segmentation_mode') == 'video_boxes' else 'stage2_meta_segment.py'
                cli(segment_script, '--case', case.name, '--config', config, '--run', run)
                cli('stage2_meta_reason.py', 'review', '--case', case.name, '--run', run)
            else:
                cli('stage2_track.py', '--case', case.name, '--config', config, '--run', run, '--device', 'mps')
                cli('stage2_discover.py', 'review', '--case', case.name, '--run', run)
            directory = case / 'segmentation' / run
            return {'directory': rel(directory), 'run': run}, [directory]
        segment_dependencies = [config, ROOT / disc['discovery'], case / 'frames', ROOT / 'scripts/stage2_discover.py']
        if segmentation_provider == 'meta_sam31_api':
            segment_dependencies += [ROOT / 'scripts/stage2_meta_segment.py', ROOT / 'scripts/stage2_meta_reason.py',
                                     *([ROOT / 'scripts/stage2_meta_video_segment.py'] if p.get('segmentation_mode') == 'video_boxes' else []),
                                     *provider_dependencies(p, segmentation=True)]
        else:
            segment_dependencies += [ROOT / 'scripts/stage2_track.py', ROOT / 'requirements-sam.lock.txt', ROOT / 'models/sam2.1_hiera_tiny.pt']
        seg = self.phase('segment', segment_dependencies, segment)
        segment_dir = ROOT / seg['directory']
        # Freeze segmentation receipts before audit adds files to that directory.
        def audit(attempt):
            if attempt > 1 and (segment_dir / 'model-audit-v1').exists():
                (segment_dir / 'model-audit-v1').rename(segment_dir / f'model-audit-previous-{attempt-1}')
            script, options = reasoning_dispatch(p, 'audit')
            cli(script, 'audit', '--case', case.name, '--run', seg['run'], '--timeout-s', p['reasoning_timeout_s'], *options)
            directory = segment_dir / 'model-audit-v1'
            return {'directory': rel(directory)}, [directory]
        audit_script, _ = reasoning_dispatch(p, 'audit')
        audit_result = self.phase('audit', [segment_dir / 'tracks.json', segment_dir / 'annotations.json', case / 'frames',
                                          ROOT / 'scripts/stage2_discover.py', ROOT / 'scripts' / audit_script,
                                          *provider_dependencies(p)], audit)
        def select(attempt):
            output = f'flow-inputs-{attempt}'
            selector = ('stage3_rigid_select_track_views.py' if p.get('discovery_mode') == 'rigid_components_v1' else
                        'stage3_meta_select_track_views.py' if reasoning_provider == 'meta_api' else 'stage3_select_track_views.py')
            cli(selector, '--case', case.name, '--run', seg['run'], '--output', output, '--top-k', '1')
            directory = case / output
            records = read(directory / 'index.json')['tracks']
            components, receipt = choose_components(case, records, read(config), read(segment_dir / 'tracks.json'),
                                                    p['reconstruct_limit'], segmentation=segment_dir)
            selection_receipt = directory / 'component-selection.json'
            write(selection_receipt, receipt)
            selections = [row['selection'] for row in components]
            bundles = []
            for j, selection in enumerate(selections):
                bundle = case / f'flow-portable-input-{attempt}-{j+1}'
                cli('stage3_reconstruction_bundle.py', '--selection', ROOT / selection, '--output', bundle)
                bundles.append(bundle)
            return {'selections': selections, 'components': components, 'selection_receipt': rel(selection_receipt),
                    'directory': rel(directory), 'bundles': [rel(b) for b in bundles]}, [directory, *bundles]
        selector = ('stage3_rigid_select_track_views.py' if p.get('discovery_mode') == 'rigid_components_v1' else
                    'stage3_meta_select_track_views.py' if reasoning_provider == 'meta_api' else 'stage3_select_track_views.py')
        selected = self.phase('select', [config, segment_dir / 'tracks.json', segment_dir / 'frames.json', ROOT / disc['discovery'],
                                       ROOT / audit_result['directory'], ROOT / 'scripts' / selector,
                                       ROOT / 'scripts/stage3_select_views.py', ROOT / 'scripts/stage3_reconstruction_bundle.py',
                                       *([ROOT / 'rigid_selection.py', ROOT / 'component_seed_review.py', ROOT / 'schemas/discovery.schema.json'] if p.get('discovery_mode') == 'rigid_components_v1' else []),
                                       *([ROOT / 'scripts/stage2_meta_reason.py'] if reasoning_provider == 'meta_api' else [])], select)
        def reconstruct(attempt):
            if not selected['selections']:
                raise ValueError('No model-audited view passed resolution/geometry gates; no replacement mesh was invented')
            def reconstruct_one(component, directory):
                if p.get('reconstruction_provider') == 'sam3d_native':
                    import sam3d_uploads
                    index = next(i for i, row in enumerate(selected['components']) if row['track_id'] == component['track_id'])
                    try:
                        output = sam3d_uploads.reconstruct_component(component, ROOT / selected['bundles'][index], directory,
                                                                    cancel_event=self.cancel_event)
                        if output['status'] == 'failed_no_retry':
                            raise RuntimeError('SAM 3D component failed without retry; original GPU evidence retained: '
                                               + str(output.get('receipt', output.get('error', 'inspect the upload job'))))
                        return output
                    except InterruptedError as exc:
                        raise Cancelled(str(exc)) from exc
                cli('stage3_reconstruct_local.py', '--selection', ROOT / component['selection'], '--output', directory, '--resolution', p['mesh_resolution'])
                metadata = directory / 'metadata.json'
                model = read(metadata)
                if model.get('status') != 'candidate_generated_pending_visual_review' or model.get('object_id') != component['track_id']:
                    raise ValueError('Reconstruction did not produce the expected component candidate')
                native = confined(directory / model['outputs']['glb_path'], directory)
                if not native.is_file():
                    raise ValueError('Reconstruction did not produce its native learned mesh')
                return {'model': rel(metadata), 'native_asset': rel(native), 'physical_shape_verified': False}
            if p.get('reconstruction_provider') == 'sam3d_native':
                result, directories = self.native_component_batch(attempt, case, selected['components'], reconstruct_one)
            else:
                result, directories = self.component_batch('model', attempt, case, selected['components'], reconstruct_one)
            result['models'] = [row['output']['model'] for row in result['components'] if row.get('output', {}).get('model')]
            if p.get('reconstruction_provider') == 'sam3d_native':
                result['pending_components'] = [row['track_id'] for row in result['components']
                                                if row.get('output', {}).get('status') == 'awaiting_gpu_worker']
                result['native_alignment_required'] = bool(result['models'])
                # The bridge owns a global source-bound result directory so a
                # resumed request cannot trigger a second model through a fresh
                # phase directory. Hash those actual outputs, not only our log.
                for row in result['components']:
                    output = row.get('output', {})
                    if output.get('handoff', {}).get('path'):
                        directories.append(confined(ROOT / output['handoff']['path']).parent)
                    if output.get('model'):
                        directories.append(confined(ROOT / output['model']).parent)
                    if output.get('review', {}).get('path'):
                        directories.append(confined(ROOT / output['review']['path']).parent)
                    receipt = output.get('receipt')
                    if isinstance(receipt, dict):
                        receipt = receipt.get('path')
                    if isinstance(receipt, str):
                        directories.append(confined(ROOT / receipt))
            return result, directories
        if p.get('reconstruction_provider') == 'sam3d_native':
            recon_inputs = [ROOT / x for x in selected['selections'] + selected.get('bundles', [])] + [
                ROOT / 'sam3d_uploads.py', ROOT / 'sam3d_cluster_bridge.py', ROOT / 'initial_sam3d_handoff.py',
                ROOT / 'sam3d_cluster_remote.py', ROOT / 'initial_sam3d_import.py', ROOT / 'scripts/stage3_sam3d.py',
                ROOT / 'scripts/build_initial_sam3d_review_v2.py', ROOT / 'scripts/build_initial_sam3d_review.py',
                ROOT / 'scripts/build_sam3d_milestone_review.py', ROOT / 'workspace/scene.js', ROOT / 'initial_native_review_v2']
            if (ROOT / 'config/sam3d-cluster.json').is_file():
                recon_inputs.append(ROOT / 'config/sam3d-cluster.json')
        else:
            recon_inputs = [ROOT / x for x in selected['selections']] + [ROOT / 'scripts/stage3_reconstruct_local.py', ROOT / 'requirements-local-geometry.lock.txt',
                            ROOT / 'models/triposr-checkpoint/provenance.json', ROOT / 'models/triposr-checkpoint/model.ckpt', ROOT / 'models/triposr-checkpoint/config.yaml']
            recon_inputs += sorted((ROOT / 'models/triposr-code/tsr').rglob('*.py'))
        recon = self.phase('reconstruct', recon_inputs, reconstruct, optional=True)
        def deferred_alignment(name, attempt):
            result = {'status': 'deferred_to_mechanism_scene', 'phase': name,
                      'destination_stage': 'mechanism_scene',
                      'destination_mode': 'muse_constrained_v1',
                      'original_source_sha256': source['sha256'],
                      'orchestrator_source_sha256': LOADED_SOURCE_SHA256,
                      'native_models': recon.get('models', []),
                      'references': [], 'experiments': [], 'viewers': [],
                      'old_poses_reused': False, 'comparison_backend_used': False,
                      'alignment_output_available': False,
                      'physical_accuracy_verified': False, 'has_gaps': False}
            report = self.directory / f'native-{name}-route-{attempt}.json'
            write(report, dict(format='interaction-twin-native-alignment-route', version=1, **result))
            return dict(result, routing_receipt=rel(report)), [report]
        def poses(attempt):
            if native_constrained:
                return deferred_alignment('pose', attempt)
            if p.get('reconstruction_provider') == 'sam3d_native':
                raise ValueError('Fresh SAM 3D native alignment remains required; old mesh poses and the comparison-backend pose adapter are not reused')
            if not recon.get('models'):
                raise ValueError('No learned asset available for pose fitting')
            components = [{'track_id': row['track_id'], 'model': row['output']['model']}
                          for row in recon['components'] if 'output' in row]
            def pose_one(component, directory):
                cli('stage4_generated_pose.py', '--manifest', ROOT / component['model'], '--out', directory,
                    '--search-width', p['pose_search_width'], '--refine-count', p['pose_refine_count'], '--max-evaluations', p['pose_max_evaluations'], '--appearance-weight', '.25')
                reference = directory / 'experiment.json'
                record = read(reference)
                if record.get('object_id') != component['track_id'] or not record.get('candidate'):
                    raise ValueError('Pose fitting did not produce the expected reference candidate')
                accepted = record.get('usable_pose_gate', {}).get('accepted') is True
                return {'reference': rel(reference), 'pose_status': record.get('status'),
                        'physical_pose_accepted': accepted, 'has_gaps': not accepted,
                        'native_asset_retained': True, 'conditional_motion_may_be_tested': True}
            result, directories = self.component_batch('pose', attempt, case, components, pose_one)
            result['references'] = [row['output']['reference'] for row in result['components'] if 'output' in row]
            return result, directories
        pose_inputs = [ROOT / x for x in recon.get('models', [])] + (
            [self.code_snapshot] if native_constrained else
            [ROOT / 'scripts/stage4_generated_pose.py', ROOT / 'alignment_fit.py', ROOT / 'alignment_render.py'])
        pose = self.phase('pose', pose_inputs, poses, optional=True)
        def motions(attempt):
            if native_constrained:
                return deferred_alignment('motion', attempt)
            if not pose.get('references'):
                raise ValueError('No reference pose available; motion remains unknown')
            components = [{'track_id': row['track_id'], 'reference': row['output']['reference']}
                          for row in pose['components'] if 'output' in row]
            def motion_one(component, directory):
                cli('stage4_track_generated.py', '--reference', ROOT / component['reference'], '--segmentation-run', seg['run'], '--output', directory)
                experiment = directory / 'experiment.json'
                record = read(experiment)
                if record.get('status') != 'complete_conditional_tracking_experiment' or record.get('object_id') != component['track_id']:
                    raise ValueError('Motion model did not complete conditional tracking for this component: ' + str(record.get('status')))
                count = record.get('supported_nonreference_frames', 0)
                result = {'experiment': rel(experiment), 'supported_nonreference_frames': count,
                          'physical_motion_verified': False, 'has_gaps': count == 0}
                try:
                    cli('stage4_generated_observer.py', '--experiment', experiment, '--out', directory / 'observer-3d.html')
                    result['viewer'] = rel(directory / 'observer-3d.html')
                except Cancelled:
                    raise
                except Exception as exc:
                    # A presentation failure must not discard actual completed tracking.
                    result.update(has_gaps=True, viewer_error=f'{type(exc).__name__}: {exc}')
                return result
            result, directories = self.component_batch('motion', attempt, case, components, motion_one)
            result['experiments'] = [row['output']['experiment'] for row in result['components'] if 'output' in row]
            result['viewers'] = [row['output']['viewer'] for row in result['components'] if row.get('output', {}).get('viewer')]
            return result, directories
        if native_constrained:
            motion_inputs = [self.code_snapshot]
            if pose.get('routing_receipt'):
                motion_inputs.append(ROOT / pose['routing_receipt'])
        else:
            motion_inputs = [ROOT / x for x in pose.get('references', [])] + [segment_dir / 'tracks.json', segment_dir / 'model-audit-v1/audit.json', ROOT / 'scripts/stage4_track_generated.py', ROOT / 'scripts/stage4_generated_observer.py', ROOT / 'alignment_fit.py', ROOT / 'alignment_render.py', ROOT / 'generated_motion_observer.js', ROOT / 'models/cotracker3/scaled_offline.pth', ROOT / 'models/cotracker3/provenance.json', ROOT / 'requirements-local-geometry.lock.txt']
        motion = self.phase('motion', motion_inputs, motions, optional=True)
        common = ['--case', case, '--discovery', ROOT / disc['discovery'], '--annotations', config]
        def propose(attempt):
            directory = case / f'flow-procedure-proposal-{attempt}'
            script, options = reasoning_dispatch(p, 'procedure_proposal')
            cli(script, 'propose', *common, '--out', directory, '--timeout-s', p['reasoning_timeout_s'], *options)
            return {'proposal': rel(directory / 'proposal.json')}, [directory]
        proposal_script, _ = reasoning_dispatch(p, 'procedure_proposal')
        proposal = self.phase('procedure_proposal', [ROOT / disc['discovery'], config, ROOT / 'scripts/stage5_compile.py',
                                                    ROOT / 'scripts' / proposal_script, ROOT / 'stage5.py',
                                                    ROOT / 'schemas/interaction-proposal.schema.json', *provider_dependencies(p)], propose, optional=True)
        def compile_procedure(attempt):
            if not proposal.get('proposal'):
                raise ValueError('No valid model semantic proposal; no hand-written procedure fallback')
            directory = case / f'flow-procedure-{attempt}'
            args = ['build', *common, '--proposal', ROOT / proposal['proposal'], '--segmentation', segment_dir, '--out', directory]
            for model in recon.get('models', []):
                args += ['--model-metadata', ROOT / model]
            for experiment in motion.get('experiments', []):
                args += ['--motion', ROOT / experiment]
            if motion.get('viewers'):
                args += ['--scene-viewer', ROOT / motion['viewers'][0]]
            cli('stage5_compile.py', *args)
            if reasoning_provider == 'meta_api':
                label_generated_meta_viewer(directory / 'index.html', read(directory / 'interaction.json'))
            cli('stage5_compile.py', 'validate', directory / 'interaction.json')
            return {'interaction': rel(directory / 'interaction.json'), 'viewer': rel(directory / 'index.html')}, [directory]
        procedure_inputs = [ROOT / 'stage5.py', ROOT / 'scripts/stage5_compile.py', ROOT / 'schemas/interaction.schema.json',
                            case / 'manifest.json', case / 'frames', segment_dir / 'tracks.json', segment_dir / 'frames.json',
                            config, ROOT / disc['discovery'], ROOT / audit_result['directory']]
        procedure_inputs += [ROOT / x for x in recon.get('models', []) + motion.get('experiments', [])]
        if proposal.get('proposal'):
            procedure_inputs.append(ROOT / proposal['proposal'])
        procedure = self.phase('procedure', procedure_inputs, compile_procedure, optional=True)
        self.state['artifacts'].update({k: v for k, v in procedure.items() if k in {'interaction', 'viewer'}})
        self.state['artifacts'].update(models=recon.get('models', []), motion=motion.get('experiments', []), scene_viewers=motion.get('viewers', []))
        self.state['artifacts'].update(component_selection=selected.get('selection_receipt'),
                                       component_receipts={name: result['receipt'] for name, result in
                                                           [('reconstruct', recon), ('pose', pose), ('motion', motion)] if result.get('receipt')})
        self.state['component_outcomes'] = {name: {'components': result.get('components', []), 'failures': result.get('failures', []),
                                                  'has_gaps': result.get('has_gaps', bool(result.get('unavailable')))}
                                           for name, result in [('reconstruct', recon), ('pose', pose), ('motion', motion)]}
        self.state['provenance'].update(fresh_model_inference=True,
                                        reasoning='hosted_Meta_Model_API' if reasoning_provider == 'meta_api' else 'hosted_OpenAI_via_local_Codex_CLI',
                                        reasoning_model=p.get('reasoning_model') if reasoning_provider == 'meta_api' else 'configured_Codex_default',
                                        segmentation='SAM3.1_Meta_API' if segmentation_provider == 'meta_sam31_api' else 'SAM2.1Tiny_MPS',
                                        segmentation_mode=p.get('segmentation_mode', 'images') if segmentation_provider == 'meta_sam31_api' else 'SAM2_video_propagation',
                                        reconstruction='Meta_SAM3D_Objects_CUDA' if p.get('reconstruction_provider') == 'sam3d_native' else 'TripoSR_MPS_non_Meta_comparison',
                                        motion='deferred_to_constrained_native_scene' if native_constrained else 'pending_fresh_native_alignment' if p.get('reconstruction_provider') == 'sam3d_native' else 'CoTracker3_MPS',
                                        reconstruction_output_available=bool(recon.get('models')), motion_output_available=bool(motion.get('experiments')))
        if p.get('reconstruction_provider') == 'sam3d_native':
            self.state['provenance']['cluster_contacted'] = any(
                row.get('output', {}).get('cluster_contacted') is True for row in recon.get('components', []))
            self.state['sam3d_reconstruction_status'] = {
                'provider': 'sam3d_native', 'generated_components': len(recon.get('models', [])),
                'pending_component_ids': recon.get('pending_components', []),
                'native_motion_status': 'fresh_native_alignment_required',
                'old_poses_reused': False, 'fallback_backend_used': False, 'physical_accuracy_verified': False}
        self.save()
        self.state['artifacts'].pop('part_ledger', None)
        if p.get('reconstruction_provider') == 'sam3d_native' and procedure.get('interaction'):
            import part_ledger
            def build_ledger(attempt):
                result = part_ledger.build(ROOT/procedure['interaction'],
                    selection_receipt=selected.get('selection_receipt'), reconstruction_receipt=recon.get('receipt'))
                value = part_ledger.read(ROOT/result['path'])
                return {'ledger': result, 'summary': value['summary'], 'has_gaps': bool(value['summary']['pending'])}, [ROOT/result['path']]
            deps = [ROOT/procedure['interaction'], ROOT/'part_ledger.py'] + [ROOT/x for x in recon.get('models', [])]
            for metadata in recon.get('models', []):
                model_path = ROOT/metadata; model_doc = read(model_path)
                deps += [model_path.parent/'asset.glb', model_path.parent/model_doc['source']['image_path'], model_path.parent/model_doc['source']['mask_path']]
            for receipt in [selected.get('selection_receipt'), recon.get('receipt')]:
                if receipt: deps.append(ROOT/receipt)
            ledger_result = self.phase('part_ledger', deps, build_ledger, optional=True)
            if ledger_result.get('ledger'):
                self.state['artifacts']['part_ledger'] = ledger_result['ledger']['path']
                self.state['part_inventory'] = ledger_result['summary']
                self.save()
        if p.get('mechanism_pipeline_mode') == 'muse_constrained_v1' and procedure.get('interaction'):
            self._mechanism_follow_on(ROOT / procedure['interaction'], case=case, segmentation=segment_dir)
        if native_constrained:
            self._resolve_native_alignment_route(pose, motion, procedure)
        def workspace(attempt):
            build = ROOT / 'scripts/build_workspace.py'
            indexed = []
            if build.is_file():
                self.command([python, build, *(['--no-replays'] if p.get('publication_mode') == 'admin_review' else [])], 60)
                indexed = [path for path in [ROOT / 'workspace/cases.js', ROOT / 'workspace/cases.json'] if path.is_file()]
            library_builder = ROOT / 'scripts/build_robotics_library.py'
            if library_builder.is_file() and self.state['artifacts'].get('interaction'):
                self.command([python, library_builder], 90)
            report = self.directory / f'workspace-artifact-{attempt}.json'
            write(report, {'case_id': case.name, 'source_sha256': read(case / 'manifest.json')['source']['sha256'],
                           'artifacts': self.state['artifacts'], 'run_id': self.id,
                           'physical_accuracy_verified': False, 'hidden_motion_or_seating_generated': False})
            return {'manifest': rel(report)}, [report, *indexed]
        deps = [ROOT / v for k, v in self.state['artifacts'].items() if k in {'interaction', 'viewer', 'constrained_scene', 'object_program'}]
        if (ROOT / 'scripts/build_workspace.py').is_file():
            deps.append(ROOT / 'scripts/build_workspace.py')
        self.phase('workspace', deps or [case / 'manifest.json'], workspace)
        if p.get('observed_follow_on') == 'gated_components_v1' and self.state['artifacts'].get('interaction'):
            # _infer remains inside execute's process-wide model-worker flock.
            # The child helper preserves its own source/code/artifact receipts.
            import observed_jobs
            registered = next((item for item in list_cases() if item['interaction_path'] == self.state['artifacts']['interaction']), None)
            if registered is None:
                raise ValueError('The compiled result was not registered for local geometry checks')
            self.state.update(phase='geometry_checks', progress=.96)
            self.save()
            def child_state(value):
                running = value.get('status') in {'created', 'queued', 'running'}
                self.state['observed_child_job_id'] = value['id'] if running else None
                self.state['observed_child_status'] = {key: value.get(key) for key in
                    ['id', 'track_id', 'status', 'workflow_status', 'phase', 'stages', 'capture_diagnostics']}
                self.save()
            outcome = observed_jobs.run_follow_on_components(registered['id'], root=ROOT,
                cancel_event=self.cancel_event, on_state=child_state, parent_worker_lock_held=True,
                feature_provider='cotracker', completion='smooth_gaps', max_components=3,
                **({'camera_mode': p['observed_camera_mode']} if p.get('observed_camera_mode') else {}))
            report = self.directory / 'follow-on-geometry.json'
            write(report, outcome)
            self.state['artifacts']['observed_reconstruction_report'] = rel(report)
            self.state['artifacts']['observed_reconstructions'] = outcome['components']
            self.state['observed_child_job_id'] = None
            self.state['observed_geometry_has_gaps'] = any(
                item.get('workflow_status') != 'completed' for item in outcome['components']) or bool(outcome.get('unstarted_component_ids'))
            self.save()
            if outcome.get('cancelled') or self.cancel_event.is_set():
                raise Cancelled('Cancelled during local component geometry checks; partial results remain saved')
            catalog_builder = ROOT / 'scripts/build_reconstruction_catalog.py'
            if catalog_builder.is_file():
                self.command([python, catalog_builder], 60)

    def _resolve_native_alignment_route(self, pose, motion, procedure):
        """A successful route cannot stand in for the destination's real output."""
        destination = next((phase for phase in self.state['phases'] if phase['name'] == 'mechanism_scene'), {})
        result = destination.get('result', {})
        manifest = result.get('manifest')
        program = procedure.get('interaction')
        program_bound = bool(program) and any(
            item.get('path') == program and verify_receipts([item]) for item in destination.get('inputs', []))
        available = (destination.get('status') in {'complete', 'complete_with_gaps', 'reused_verified', 'reused_verified_with_gaps'} and
                     isinstance(manifest, dict) and bool(manifest.get('path')) and
                     program_bound and verify_receipts([manifest]) and
                     self.state['artifacts'].get('constrained_scene') == manifest['path'])
        report = {'format': 'interaction-twin-native-alignment-resolution', 'version': 1,
                  'status': 'estimated_native_alignment_available' if available else 'pending_downstream_alignment',
                  'destination_stage': 'mechanism_scene',
                  'interaction': program,
                  'destination_status': result.get('status', self.state.get('mechanism_pipeline_status', {}).get('scene', 'not_run')),
                  'destination_phase_status': destination.get('status', 'not_run'),
                  'destination_diagnostic': destination.get('error') or self.state.get('mechanism_pipeline_status', {}).get('diagnostic'),
                  'route_receipts': {name: value.get('routing_receipt') for name, value in [('pose', pose), ('motion', motion)]},
                  'scene_manifest': manifest if available else None,
                  'alignment_output_available': available,
                  'conditional_motion_experiments': [],
                  'old_poses_reused': False, 'comparison_backend_used': False,
                  'physical_accuracy_verified': False, 'simulation_ready': False,
                  'has_gaps': not available or bool(result.get('has_gaps')) or
                              destination.get('status') in {'complete_with_gaps', 'reused_verified_with_gaps'}}
        path = self.directory / ('native-alignment-resolution-' + fingerprint(report)[:24] + '.json')
        if path.exists():
            if read(path) != report:
                raise ValueError('Saved native alignment resolution differs from its identity')
        else:
            write(path, report)
        self.state['artifacts']['native_alignment_resolution'] = rel(path)
        self.state['native_alignment_has_gaps'] = report['has_gaps']
        self.state['native_alignment_status'] = report
        self.state['provenance']['native_scene_alignment_output_available'] = available
        self.state['sam3d_reconstruction_status']['native_motion_status'] = report['status']
        self.save()

    def _mechanism_follow_on(self, program, *, case, segmentation):
        """Explicit Run stages only; workspace/read paths never invoke inference."""
        import mechanism_pipeline as mechanisms
        profile = self.profile
        options = mechanisms.configuration(profile, root=ROOT)
        python = ROOT / '.venv-sam/bin/python'
        script = ROOT / 'scripts/run_mechanism_pipeline.py'
        code = [ROOT / name for name in mechanisms.CODE] + [script]
        relative_program = rel(program)
        for key in ['constrained_scene', 'object_program', 'rigidity_feasibility']:
            self.state['artifacts'].pop(key, None)
        import part_ledger
        ledger_path = self.state['artifacts'].get('part_ledger')
        try:
            if not ledger_path:
                raise ValueError('No current authoritative part ledger is available')
            part_ledger.metadata_paths(ledger_path, program_path=program, root=ROOT)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.state['mechanism_pipeline_status'] = {'topology': 'pending_verified_part_ledger',
                'scene': 'pending_verified_part_ledger', 'object_program': 'pending_verified_part_ledger',
                'diagnostic': str(exc)[:500], 'independent_scene_fallback': False,
                'physical_accuracy_verified': False, 'simulation_ready': False}
            self.save()
            return

        def execute_phase(name, attempt, arguments, timeout):
            report = self.directory / f'{name}-{attempt}.json'
            argv = [python, script, name, '--report', report, *arguments]
            if profile.get('overnight_task_budget_plan'):
                argv = [python, ROOT/'scripts/run_overnight_stage.py', '--task-budget-plan',
                        profile['overnight_task_budget_plan'], 'run_mechanism_pipeline.py', name, '--report', report, *arguments]
            self.command(argv, timeout)
            result = read(report)
            artifacts = [report] + [ROOT / item['path'] for item in result.get('artifacts', [])]
            return result, artifacts

        def topology(attempt):
            return execute_phase('topology', attempt, ['--program', relative_program, '--allow-paid-api',
                '--task-budget-plan', options['task_budget_plan'], '--max-images', str(options['max_images']),
                '--max-output-tokens', str(options['max_output_tokens']),
                *(['--overnight-task-budget-plan',profile['overnight_task_budget_plan']]if profile.get('overnight_task_budget_plan')else[])], profile['reasoning_timeout_s'] + 60)

        topology_result = self.phase('mechanism_topology', [program, case / 'frames', segmentation,
                                   *code, ROOT / options['task_budget_plan'],
                                   *([ROOT/'mechanism_pass_budget.py']if profile.get('overnight_task_budget_plan')else[]),
                                   *provider_dependencies(profile)], topology, optional=True)

        def scene(attempt):
            if not topology_result.get('proposal'):
                report = self.directory / f'mechanism-scene-pending-{attempt}.json'
                result = {'status': 'pending_mechanism_topology', 'has_gaps': True,
                          'independent_scene_fallback': False, 'new_api_calls': 0, 'new_gpu_calls': 0}
                write(report, result)
                return result, [report]
            result, artifacts = execute_phase('scene', attempt, ['--program', relative_program, '--proposal-ref',
                json.dumps(topology_result['proposal'], separators=(',', ':')), '--part-ledger', ledger_path], min(profile['phase_timeout_s'], 300))
            if not result.get('manifest'):
                return result, artifacts
            # A fitted rigid joint is only a hypothesis. An explicit source
            # nonrigidity declaration must withhold it before any export. Keep
            # the raw fit/report immutable; failure here never exports that fit.
            import rigidity_feasibility
            base_manifest = result['manifest']
            output = ROOT / 'data/rigidity-feasibility' / ('flow-' + self.id) / (
                f'attempt-{attempt}-' + base_manifest['sha256'][:24])
            feasibility = rigidity_feasibility.build(base_manifest['path'], output, root=ROOT)
            checked_scene = read(ROOT / feasibility['scene_ref']['path'])
            mechanics = checked_scene['mechanics']
            unresolved = len(mechanics.get('unresolved_children', []))
            unknown_samples = mechanisms.unknown_joint_samples(mechanics.get('joints', []))
            result = {**result, 'base_manifest': base_manifest,
                      'manifest': feasibility['scene_ref'], 'mechanics': feasibility['mechanics_ref'],
                      'rigidity_feasibility': feasibility, 'unresolved_connections': unresolved,
                      'unknown_joint_samples': unknown_samples,
                      'has_gaps': bool(result.get('has_gaps')) or bool(unresolved) or unknown_samples > 0}
            if feasibility['withheld_joint_ids']:
                result['status'] = 'constrained_scene_with_unknown_connections'
            checked_report = self.directory / f'scene-feasibility-{attempt}.json'
            write(checked_report, result)
            return result, [*artifacts, checked_report, output]

        import rigidity_feasibility
        scene_deps = [program, ROOT/ledger_path, ROOT/'part_ledger.py', case / 'manifest.json', case / 'frames', segmentation,
                      *code, *[ROOT / name for name in rigidity_feasibility.CODE]]
        try:
            scene_deps += mechanisms.scene_input_paths(program, root=ROOT, part_ledger_path=ledger_path)
        except (OSError, ValueError, KeyError):
            # The bounded scene stage records unavailable native inputs as a gap.
            # It may never replace them with a different reconstruction backend.
            pass
        if topology_result.get('proposal'):
            scene_deps.append(ROOT / topology_result['proposal']['path'])
        scene_result = self.phase('mechanism_scene', scene_deps, scene, optional=True)

        def export(attempt):
            if not scene_result.get('manifest'):
                report = self.directory / f'object-program-pending-{attempt}.json'
                result = {'status': 'pending_constrained_scene', 'has_gaps': True,
                          'independent_scene_fallback': False, 'new_api_calls': 0, 'new_gpu_calls': 0}
                write(report, result)
                return result, [report]
            return execute_phase('export', attempt, ['--scene-ref', json.dumps(scene_result['manifest'], separators=(',', ':')),
                *(['--private'] if profile.get('publication_mode') == 'admin_review' else [])], 90)

        export_deps = [program, *code, ROOT / 'scene_replay_publication.py']
        if scene_result.get('manifest'):
            export_deps.append(ROOT / scene_result['manifest']['path'])
        export_result = self.phase('object_program', export_deps, export, optional=True)
        self.state['mechanism_pipeline_status'] = {
            'topology': topology_result.get('status', 'unavailable'),
            'scene': scene_result.get('status', 'unavailable'),
            'rigidity_feasibility': scene_result.get('rigidity_feasibility', {}).get('status', 'not_completed'),
            'object_program': export_result.get('status', 'unavailable'),
            'independent_scene_fallback': False, 'physical_accuracy_verified': False,
            'simulation_ready': False}
        if scene_result.get('manifest'):
            self.state['artifacts']['constrained_scene'] = scene_result['manifest']['path']
        if scene_result.get('rigidity_feasibility', {}).get('assessment_ref'):
            self.state['artifacts']['rigidity_feasibility'] = scene_result['rigidity_feasibility']['assessment_ref']['path']
        if export_result.get('object_program'):
            self.state['artifacts']['object_program'] = export_result['object_program']['path']
        self.save()


def list_runs():
    base = ROOT / 'data/flow-runs'
    return sorted([read(p) for p in base.glob('*/run.json')], key=lambda x: x['created_at'], reverse=True) if base.exists() else []


def list_cases():
    rows = []
    candidates = list((ROOT / 'data').glob('*/procedure-*/interaction.json')) + list((ROOT / 'data').glob('*/flow-procedure-*/interaction.json'))
    for path in candidates:
        try:
            program = read(path)
            video = confined(ROOT / program['source']['path'], ROOT / 'data')
            if not video.is_file():
                continue
            # Keep historical variants selectable; never pretend newest means best.
            identifier = str(path.parent.parent.name + '--' + path.parent.name)
            scene_path = program.get('links', {}).get('scene_viewer')
            scene = confined(ROOT / scene_path, ROOT / 'data') if scene_path else None
            rows.append({'id': identifier, 'case_id': program['case_id'], 'title': path.parent.parent.name + ' / ' + path.parent.name,
                         'summary': program.get('summary', ''), 'video_url': public_url(video), 'interaction_url': public_url(path),
                         'interaction_path': rel(path), 'scene_url': public_url(scene) if scene and scene.is_file() else None,
                         'workspace_case_url': '/workspace/index.html?case=' + quote(identifier),
                         'duration_s': program['source']['duration_s'], 'source_sha256': program['source']['sha256'],
                         'physical_accuracy_verified': False})
        except (OSError, ValueError, KeyError):
            continue
    return sorted(rows, key=lambda x: x['id'])
