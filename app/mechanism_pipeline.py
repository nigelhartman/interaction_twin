"""Explicit future-upload mechanism stages; readers never execute these stages.

One paid topology attempt is persistent per exact original program. CPU scene
fitting stays private until a verified constrained object-program export exists.
No independent scene is published as a substitute for a missing joint graph.
"""
from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODE = 'muse_constrained_v1'
BUDGET_PLAN = 'output/constrained-kinematics-20261102/task-budget-plan.json'
CODE = ['mechanism_pipeline.py', 'sam3d_common_scene.py', 'sam3d_mechanisms.py',
        'sam3d_coordinates.py', 'sam3d_source_projection.py', 'native_appearance.py',
        'sam3d_review_catalog.py', 'object_program.py', 'mechanism_reasoning.py', 'native_audit_protocol.py', 'part_ledger.py']


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def local(root, path, *, data=True, exists=True):
    root = Path(root).resolve(); path = Path(path)
    candidate = path if path.is_absolute() else root/path
    require(not any(part.startswith('.') for part in candidate.relative_to(root).parts), 'Hidden mechanism paths refused')
    require(candidate.resolve() == candidate and candidate.is_relative_to(root/('data' if data else 'output')),
            'Mechanism artifact must stay in its ordinary project namespace')
    require(not any(p.is_symlink() for p in [candidate, *candidate.parents]), 'Symlink mechanism artifacts refused')
    if exists:
        require(candidate.is_file(), 'Saved mechanism artifact unavailable')
    return candidate


def ref(path, root):
    path = local(root, path)
    return {'path': path.relative_to(root).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size}


def checked(value, root):
    path = local(root, value['path'])
    require(ref(path, root) == value, 'Saved mechanism artifact changed')
    return path


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


@contextmanager
def locked(root):
    folder = local(root, Path(root)/'data/mechanism-pipelines', exists=False)
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder/'.worker.lock'
    require(not lock.is_symlink(), 'Symlink mechanism worker lock refused')
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'a+') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def configuration(profile, *, root=ROOT):
    """Validate a fixed existing budget baseline; never reset or create one."""
    require(profile.get('mechanism_pipeline_mode') == MODE, 'Unknown mechanism pipeline mode')
    budget = profile.get('mechanism_task_budget_plan')
    if budget != BUDGET_PLAN:
        from mechanism_pass_budget import validate_profile
        validate_profile(profile,root=root)
    path = local(root, budget, data=False)
    plan = json.loads(path.read_text()); maximum = Decimal(plan['additional_estimated_api_ceiling_usd'])
    require(maximum.is_finite() and Decimal('0') < maximum <= Decimal('5'), 'Mechanism task estimate ceiling exceeds five dollars')
    for key in ['settled_usd', 'reserved_usd']:
        amount = Decimal(plan['budget_baseline'][key]); require(amount.is_finite() and amount >= 0, 'Invalid persistent mechanism baseline')
    images = profile.get('mechanism_max_images', 16); tokens = profile.get('mechanism_max_output_tokens', 10000)
    require(type(images) is int and 4 <= images <= 16 and type(tokens) is int and 1000 <= tokens <= 12000,
            'Mechanism evidence/output bounds exceeded')
    return {'task_budget_plan': budget, 'max_images': images, 'max_output_tokens': tokens}


def outcome(status, *, artifacts=(), **fields):
    return {'status': status, 'artifacts': list(artifacts), 'new_gpu_calls': 0,
            'independent_scene_fallback': False, 'physical_accuracy_verified': False,
            'simulation_ready': False, **fields}


def unknown_joint_samples(joints):
    return sum(sample.get('status') == 'unknown' or sample.get('q_rad') is None
               for joint in joints for sample in joint.get('samples', []))


def scene_input_paths(program_path, *, root=ROOT, part_ledger_path=None):
    """Actual native inputs included in phase receipts, without learned inference."""
    import sam3d_common_scene as common
    root = Path(root).resolve(); program_path = local(root, program_path)
    program = json.loads(program_path.read_text()); paths = []
    if part_ledger_path is not None:
        import part_ledger
        paths.append(local(root,part_ledger_path));models=part_ledger.metadata_paths(part_ledger_path,program_path=program_path,root=root)
    else:models=common.metadata_for_program(program,root)
    for metadata in models:
        metadata = local(root, metadata); model = json.loads(metadata.read_text())
        paths.extend([metadata, metadata.parent/'asset.glb', metadata.parent/'raw/metadata.json',
                      metadata.parent/'raw/layout-and-pointmap.npz',
                      metadata.parent/model['source']['image_path'], metadata.parent/model['source']['mask_path']])
    return [local(root, path) for path in sorted(set(paths))]


def require_supported_segmentation(program_path, *, root=ROOT):
    """Require the exact source-bound compiled SAM attempt, including repairs."""
    import sam3d_common_scene as common
    return common.segmentation_for_program(program_path,root=root)


def topology_artifacts(proposal_path, *, root=ROOT):
    """Include adopted response/plan/evidence in outer phase cache receipts."""
    root = Path(root).resolve(); proposal_path = local(root, proposal_path)
    proposal = json.loads(proposal_path.read_text()); reason = proposal['reasoning']
    plan_path = local(root, reason['plan_path']); plan = json.loads(plan_path.read_text())
    paths = [proposal_path, plan_path, local(root, reason['model_result_path']), local(root, reason['request_receipt'])]
    paths += [plan_path.parent/name for name in ['schema.json','provider-schema.json','prompt.txt']]
    paths += [local(root, image['path']) for image in plan['context']['images']]
    return [ref(path, root) for path in sorted(set(paths))]


def adopt_verified_topology(program_path, proposal_path, *, root=ROOT):
    """Explicitly adopt a completed paid result without replaying any API phase.

    The immutable adoption is exact-program/source/response bound. A later
    missing or modified adoption stays pending and never falls through to a
    fresh hosted request.
    """
    import object_program
    root = Path(root).resolve(); program_path = local(root, program_path); proposal_path = local(root, proposal_path)
    original, proposal_ref = ref(program_path, root), ref(proposal_path, root)
    proposal = object_program.load_proposals(proposal_ref['path'], original['path'], root=root)
    plan = json.loads(local(root, proposal['reasoning']['plan_path']).read_text())
    current_code = sha(root/'mechanism_reasoning.py')
    require(plan['contract']['code_sha256'] == current_code, 'Adopted topology uses a different frozen reasoning contract')
    value = {'format':'adopted-verified-mechanism-topology', 'version':1, 'original_program':original,
             'proposal':proposal_ref, 'reasoning_code_sha256':current_code, 'new_api_calls':0}
    with locked(root):
        record = local(root, root/'data/mechanism-pipelines'/original['sha256']/'adopted-topology.json', exists=False)
        if record.exists():
            require(json.loads(record.read_text()) == value, 'A different topology adoption already exists; no silent replacement')
        else:
            write(record, value)
    return outcome('adopted_verified_mechanism_topology', artifacts=[ref(record, root), *topology_artifacts(proposal_path, root=root)],
                   original_program=original, proposal=proposal_ref, has_gaps=False, new_api_calls=0)


def topology(program_path, *, root=ROOT, allow_paid_api=False, task_budget_plan=BUDGET_PLAN,
             max_images=16, max_output_tokens=10000,overnight_task_budget_plan=None):
    """Prepare offline or make at most one explicitly authorized Muse attempt.

    An interrupted, changed, invalid or failed attempt never triggers a new paid
    send, including after an orchestrator/source-code upgrade or process crash.
    """
    import mechanism_reasoning as reasoning
    import object_program
    root = Path(root).resolve(); program_path = local(root, program_path)
    configuration({'mechanism_pipeline_mode': MODE, 'mechanism_task_budget_plan': task_budget_plan,
                   'mechanism_max_images': max_images, 'mechanism_max_output_tokens': max_output_tokens,
                   'overnight_task_budget_plan':overnight_task_budget_plan}, root=root)
    scoped_client=None
    if task_budget_plan!=BUDGET_PLAN:
        import meta_api
        from mechanism_pass_budget import scoped_client_class
        # An outer-plan string alone cannot enable the new scope from a direct
        # CLI. Only an already wrapped child receives this runtime base class.
        scoped_client=scoped_client_class(meta_api.MetaClient,root=root)
    original = ref(program_path, root)
    with locked(root):
        adoption = local(root, root/'data/mechanism-pipelines'/original['sha256']/'adopted-topology.json', exists=False)
        if adoption.exists():
            try:
                saved = json.loads(adoption.read_text())
                require(saved.get('format') == 'adopted-verified-mechanism-topology' and saved.get('version') == 1 and
                        saved['original_program'] == original and saved['reasoning_code_sha256'] == sha(root/'mechanism_reasoning.py'),
                        'Adopted mechanism contract changed')
                checked(saved['proposal'], root)
                object_program.load_proposals(saved['proposal']['path'], original['path'], root=root)
            except (OSError, ValueError, KeyError, TypeError):
                return outcome('pending_adopted_topology_review_no_automatic_resend', artifacts=[ref(adoption, root)],
                               original_program=original, has_gaps=True, new_api_calls=0)
            return outcome('reused_adopted_verified_mechanism_topology', artifacts=[ref(adoption, root), *topology_artifacts(root/saved['proposal']['path'], root=root)],
                           original_program=original, proposal=saved['proposal'], has_gaps=False, new_api_calls=0)
        folder = root/'data/mechanism-pipelines'/original['sha256']/'topology'
        folder_rel = folder.relative_to(root).as_posix()
        if not folder.exists():
            if folder.parent.exists():
                return outcome('pending_namespace_review_no_automatic_resend', original_program=original,
                               has_gaps=True, new_api_calls=0)
            reasoning.prepare(original['path'], folder_rel, root=root, max_images=max_images, max_output_tokens=max_output_tokens)
        artifacts = [ref(p, root) for p in sorted(folder.rglob('*')) if p.is_file()]
        pending = dict(prepared=folder_rel, original_program=original, new_api_calls=0, has_gaps=True)
        # A partial preparation is retained. It is never silently replaced.
        try:
            saved = json.loads((folder/'plan.json').read_text())
            expected = reasoning.plan(original['path'], root=root, max_images=max_images, max_output_tokens=max_output_tokens)
            require(saved == {k:v for k,v in expected.items() if k not in {'prompt','schema','provider_schema'}}, 'Prepared mechanism contract changed')
            require(sha(folder/'prompt.txt') == expected['contract']['prompt_sha256'], 'Prepared mechanism prompt changed')
            require(json.loads((folder/'schema.json').read_text()) == expected['schema'] and
                    json.loads((folder/'provider-schema.json').read_text()) == expected['provider_schema'], 'Prepared mechanism schema changed')
        except (OSError, ValueError, KeyError, TypeError):
            return outcome('pending_topology_contract_review_no_automatic_resend', artifacts=artifacts, **pending)
        proposal_path = folder/'attempt/proposals.json'
        if proposal_path.is_file():
            try:
                object_program.load_proposals(proposal_path.relative_to(root).as_posix(), original['path'], root=root)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return outcome('pending_invalid_saved_topology_no_automatic_resend', artifacts=artifacts, diagnostic=str(exc)[:500], **pending)
            return outcome('reused_verified_mechanism_topology', artifacts=artifacts, proposal=ref(proposal_path, root),
                           prepared=folder_rel, original_program=original, has_gaps=False, new_api_calls=0)
        if (folder/'attempt').exists():
            return outcome('pending_prior_mechanism_attempt_no_automatic_resend', artifacts=artifacts, **pending)
        if not allow_paid_api:
            return outcome('pending_explicit_mechanism_run', artifacts=artifacts, **pending)
        # The reviewed adapter persists its attempt before transport and retains
        # reservations on uncertain outcomes. All three spending guards apply.
        reasoning.run_prepared(folder_rel, root=root, allow_paid_api=True, task_budget_plan=task_budget_plan,
            **({'client':scoped_client(timeout=300)}if scoped_client is not None else{}))
        object_program.load_proposals(proposal_path.relative_to(root).as_posix(), original['path'], root=root)
        artifacts = [ref(p, root) for p in sorted(folder.rglob('*')) if p.is_file()]
        return outcome('mechanism_topology_generated', artifacts=artifacts, proposal=ref(proposal_path, root),
                       prepared=folder_rel, original_program=original, has_gaps=False, new_api_calls=1)


def constrained_scene(program_path, proposal_ref, *, root=ROOT, metadata_paths=None, part_ledger_path=None, held_out_sample_indices=()):
    """Build a private independent estimate, then replace attached motion by FK."""
    import object_program
    import sam3d_common_scene as common
    import sam3d_mechanisms as mechanisms
    root = Path(root).resolve(); program_path = local(root, program_path)
    require_supported_segmentation(program_path,root=root)
    proposal_path = checked(proposal_ref, root)
    proposals = object_program.load_proposals(proposal_ref['path'], program_path.relative_to(root).as_posix(), root=root)
    program = json.loads(program_path.read_text())
    if part_ledger_path is not None:
        import part_ledger
        metadata=part_ledger.metadata_paths(part_ledger_path,program_path=program_path,root=root)
        if metadata_paths is not None:require([local(root,p)for p in metadata_paths]==metadata,'Explicit scene metadata differs from authoritative part ledger')
    else:metadata=list(metadata_paths)if metadata_paths is not None else common.metadata_for_program(program,root)
    require(metadata, 'No source-bound native SAM geometry is available')
    codes = [{'path': name, 'sha256': sha(root/name)} for name in CODE]
    identity = digest({'program': ref(program_path, root), 'proposal': proposal_ref, 'code': codes,
                       'metadata': [ref(p, root) for p in metadata],'part_ledger':ref(part_ledger_path,root) if part_ledger_path is not None else None,'held_out_sample_indices':sorted(held_out_sample_indices)})
    folder = root/'data/mechanism-scenes'/identity
    with locked(root):
        baseline = common.build(program_path, folder/'private-common', root=root, metadata_paths=metadata,part_ledger_path=part_ledger_path,held_out_sample_indices=held_out_sample_indices)
        # Deliberately NO publish_registry call for the independent estimate.
        fitted = mechanisms.build(root/baseline['manifest']['path'], proposal_path, folder/'constrained', root=root)
        scene_path = checked(fitted['manifest'], root)
        scene = json.loads(scene_path.read_text()); mechanics = scene.get('mechanics', {})
        require(scene.get('original_program') == ref(program_path, root) and scene['source']['sha256'] == program['source']['sha256'],
                'Constrained scene source/program mismatch')
        require(mechanics.get('free_floating_child_fallback') is False and mechanics.get('physical_accuracy_verified') is False,
                'Independent child fallback is forbidden')
        expected = {j['id'] for j in proposals['joints'] if j['attachment'] == 'persistent'}
        represented = {j['id'] for j in [*mechanics.get('joints', []), *mechanics.get('unresolved_children', [])]}
        require(expected == represented, 'A persistent connection disappeared during fitting')
        unresolved = mechanics.get('unresolved_children', [])
        unknown_samples = unknown_joint_samples(mechanics.get('joints', []))
        artifacts = [ref(p, root) for p in sorted(folder.rglob('*')) if p.is_file()]
        return outcome('constrained_scene_with_unknown_connections' if unresolved else 'estimated_constrained_scene_ready',
                       artifacts=artifacts, manifest=fitted['manifest'], private_scene=baseline['manifest'],
                       mechanics=fitted['mechanics'], has_gaps=bool(unresolved) or unknown_samples > 0, new_api_calls=0,
                       original_program=ref(program_path, root), unresolved_connections=len(unresolved),
                       unknown_joint_samples=unknown_samples,
                       mechanism_fit_reused=fitted.get('reused_verified_cache') is True,
                       private_common_reused=baseline.get('status')=='reused_verified_estimated_scene')


def export_scene(scene_ref, *, root=ROOT, publish=True):
    """Export and verify a separate program before optional local publication."""
    import object_program
    root = Path(root).resolve(); path = checked(scene_ref, root)
    identity = digest({'scene': scene_ref, 'exporter_sha256': sha(root/'object_program.py')})
    target = root/'data/object-programs'/('auto-'+identity)/'object-program.json'
    with locked(root):
        if target.is_file():
            document = object_program.read_json(target)
            object_program.validate(document, root=root)
            require(document['constrained_scene'] == scene_ref, 'Saved descriptive export belongs to a different scene')
            artifact = object_program.ref(target, root)
        else:
            artifact = object_program.build(scene_ref['path'], target.relative_to(root).as_posix(), root=root)
            document = object_program.read_json(checked(artifact, root))
            object_program.validate(document, root=root)
        if publish:
            from scene_replay_publication import publish_constrained_result
            publish_constrained_result({'artifact': artifact, 'document': document}, document['evidence_program']['path'], root=root)
        unresolved = document.get('unresolved_connections', [])
        unknown_samples = unknown_joint_samples(document.get('joints', []))
        return outcome('descriptive_object_program_ready', artifacts=[ref(p, root) for p in sorted(target.parent.rglob('*')) if p.is_file()],
                       object_program=artifact, manifest=scene_ref, has_gaps=bool(unresolved) or unknown_samples > 0, new_api_calls=0,
                       unknown_joint_samples=unknown_samples,
                       indexed=bool(publish), unresolved_connections=len(unresolved))
