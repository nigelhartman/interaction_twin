"""Versioned model-only discovery of separately moving rigid components.

Category names, seeds, observation intervals and actions all come from the
hosted model. These are proposals, not geometry or verified physical identities.
"""
from pathlib import Path
from decimal import Decimal
import re
import time

from meta_api import MetaClient
from meta_localization import localize_discovery
from scripts import stage2_meta_reason as base
import jsonschema


def prompt_text(rows):
    return base.prompt_text(rows) + (
        '\nRIGID-COMPONENT DISCOVERY v1: The downstream task reconstructs each rigid moving '
        'part independently. Inspect changes across the sequence before choosing tracklets. '
        'If a containing assembly has rigid pieces that visibly move relative to one another, '
        'propose separate tracklets for those pieces and avoid a duplicate whole-assembly track. '
        'Do not merge an open cover with its lower holder or include removable contents in a '
        'holder seed. This is a generic rule, not an assertion that these mechanisms exist. '
        'Choose up to six useful tracklets, prioritizing a visible holder, independently moving '
        'parts and handled removable objects. Deformable regions must be described as deformable '
        'or uncertain, never assumed rigid. A newly reappearing identical object can be a new '
        'uncertain tracklet rather than an asserted identity continuation. Prefer one or two '
        'clearly visible seeds per tracklet, including an unobstructed empty holder when shown. '
        'A closed assembly is a poor seed for a specific subpart when its boundary is uncertain; '
        'prefer a later clearly separated view. Omit a doubtful seed instead of adding an early seed '
        'just to cover the start of the recording. Keep the track interval distinct from its seed time. '
        'Use the original image coordinates and actual sample metadata. State uncertainties '
        'about scope, occlusion and shape explicitly; do not invent pivots, metric scale or contact. '
        'Each tracklet must contain at most ONE seed for each sample_index. Never return '
        'two alternative boxes for the same tracklet and sample; choose another clearly '
        'supported sample or omit the doubtful seed. Check that all sample/time pairs '
        'and action participants satisfy the requested contract before returning.'
    )


def discover(case, run, *, model='muse-spark-1.3', max_output_tokens=16000, timeout=300, client=None,
             allow_contract_repair=True):
    case = Path(case).resolve()
    rows = base.read(case / 'frames/index.json')['frames']
    source = base.read(case / 'manifest.json')['source']
    if not rows or len(rows) > 120:
        raise ValueError('Rigid discovery requires one to 120 sampled source frames')
    out = case / 'reasoning' / run
    out.mkdir(parents=True, exist_ok=False)
    (out / 'implementation.py').write_bytes(Path(__file__).read_bytes())
    paths = base.make_sheets(case, rows, out)
    prompt = prompt_text(rows)
    (out / 'prompt.txt').write_text(prompt)
    record = {'status': 'running', 'provider': 'meta_api', 'model': model,
              'role': 'rigid_component_visual_discovery_v1',
              'inference_location': 'hosted_Meta_Model_API',
              'source_sha256': source['sha256'], 'frame_index_sha256': base.sha(case / 'frames/index.json'),
              'prompt_sha256': base.sha(out / 'prompt.txt'), 'schema_sha256': base.sha(base.SCHEMA),
              'implementation_sha256': base.sha(__file__), 'human_reviewed': False,
              'manual_object_category_hints': False, 'manual_seed_boxes_used': False,
              'input_images': [{'file': str(p.relative_to(case)), 'sha256': base.sha(p)} for p in paths]}
    base.write(out / 'metadata.json', record)
    started = time.monotonic()
    client = client or MetaClient(timeout=timeout)
    try:
        initial, receipt = client.json_result(prompt, base.read(base.SCHEMA), paths,
            output=out, model=model, max_output_tokens=max_output_tokens)
        base.write(out / 'initial-discovery.json', initial)
        repair_cost = Decimal('0')
        try:
            base.validate_discovery(initial, rows)
        except (ValueError, jsonschema.ValidationError):
            if not allow_contract_repair:
                raise
            import discovery_repair as repair
            repair.write_new(out / 'initial-metadata-before-repair.json', record)
            prepared = repair.prepare(case, run,
                '.private/discovery-repairs/' + receipt['request_id'] + '-v1', root=base.ROOT)
            corrected = repair.execute(prepared['prepared'], prepared['plan_ref']['sha256'], root=base.ROOT, client=client)
            initial = corrected['result']
            base.write(out / 'corrected-discovery.json', initial)
            record['contract_repair'] = {'report_ref': corrected['report_ref'],
                'initial_request_resent': False, 'maximum_repair_requests': 1,
                'automatic_retry': False, 'manual_seed_boxes_used': False}
            repair_cost = Decimal(corrected['report']['estimated_cost_usd'])
        localized, localization = localize_discovery(initial, rows, case,
            out / 'full-frame-localization-v1', client, model, max_output_tokens)
        base.validate_discovery(localized, rows)
        record.update(status='complete_candidate_proposals', result=localized,
            usage=receipt['usage'], request_receipt='meta-request.json',
            estimated_cost_usd=str(Decimal(receipt['estimated_cost_usd']) + repair_cost + Decimal(localization['estimated_cost_usd'])),
            localization={'report': 'full-frame-localization-v1/report.json',
                          'report_sha256': base.sha(out / 'full-frame-localization-v1/report.json'),
                          'model_calls': localization['model_calls'], 'manual_boxes_used': False},
            wall_time_s=round(time.monotonic() - started, 3))
        base.write(out / 'discovery.json', record)
        config = base.tracking_config(localized, rows, case.name, base.sha(out / 'discovery.json'))
        config['component_policy'] = 'rigid_component_visual_discovery_v1'
        destination = base.ROOT / 'annotations' / f'new-{case.name}-{run}.json'
        if destination.exists():
            raise ValueError('Preserve existing annotation records')
        base.write(destination, config)
        record['tracking_config'] = str(destination.relative_to(base.ROOT))
        base.write(out / 'metadata.json', {k: v for k, v in record.items() if k != 'result'})
        return record
    except Exception as exc:
        record.update(status='failed_preserved', error_type=type(exc).__name__,
                      automatic_retry=False, wall_time_s=round(time.monotonic() - started, 3))
        base.write(out / 'metadata.json', {k: v for k, v in record.items() if k != 'result'})
        raise


def recover(case, run, *, prepared, authorized_plan_sha256, timeout=300, client=None):
    """Continue one known-accounted invalid initial response without resending it.

    Recovery writes a fresh discovery run only. Its correction and subsequent
    full-frame localization remain model-only and go through existing guards.
    Unknown/repeated correction intents cannot be retried under another run.
    """
    import discovery_repair as repair
    case = Path(case).resolve()
    if not isinstance(run, str) or re.fullmatch(r'[a-z0-9][a-z0-9-]{0,95}', run) is None:
        raise ValueError('Recovery run must be a local slug')
    _, plan, _ = repair.verified_plan(prepared, authorized_plan_sha256, root=base.ROOT)
    if plan['context']['case'] != case.relative_to(base.ROOT).as_posix():
        raise ValueError('Discovery recovery belongs to another source case')
    if run == plan['context']['initial_run']:
        raise ValueError('Recovery must preserve the original discovery run')
    out = case / 'reasoning' / run
    out.mkdir(parents=True, exist_ok=False)
    (out / 'implementation.py').write_bytes(Path(__file__).read_bytes())
    prompt = repair.prompt_for(plan['context'])
    (out / 'prompt.txt').write_text(prompt)
    record = {'status': 'running', 'provider': 'meta_api', 'model': plan['model'],
        'role': 'rigid_component_visual_discovery_v1', 'inference_location': 'hosted_Meta_Model_API',
        'source_sha256': plan['context']['source_sha256'],
        'frame_index_sha256': base.sha(case / 'frames/index.json'), 'schema_sha256': base.sha(base.SCHEMA),
        'prompt_sha256': base.sha(out / 'prompt.txt'), 'implementation_sha256': base.sha(__file__),
        'human_reviewed': False, 'manual_object_category_hints': False, 'manual_seed_boxes_used': False,
        'input_images': [{'file': str((base.ROOT / item['path']).relative_to(case)), 'sha256': item['sha256']}
                         for item in plan['context']['image_inputs']],
        'recovery': {'initial_run': plan['context']['initial_run'],
                     'initial_receipt_ref': plan['context']['initial_receipt_ref'],
                     'repair_plan_ref': repair.ref(base.ROOT / prepared / 'plan.json', base.ROOT),
                     'initial_request_resent': False, 'maximum_repair_requests': 1, 'automatic_retry': False}}
    base.write(out / 'metadata.json', record)
    started = time.monotonic()
    try:
        if client is None:
            client = repair.client_for_recovery(root=base.ROOT, timeout=timeout)
        corrected = repair.execute(prepared, authorized_plan_sha256, root=base.ROOT, client=client)
        initial = corrected['result']
        base.write(out / 'initial-discovery.json', initial)
        rows = base.read(case / 'frames/index.json')['frames']
        base.validate_discovery(initial, rows)
        localized, localization = localize_discovery(initial, rows, case,
            out / 'full-frame-localization-v1', client, plan['model'], plan['max_output_tokens'])
        base.validate_discovery(localized, rows)
        new_cost = Decimal(corrected['report']['estimated_cost_usd']) + Decimal(localization['estimated_cost_usd'])
        record.update(status='complete_candidate_proposals', result=localized,
            usage=corrected['report']['usage'], request_receipt=corrected['report']['request_receipt_ref']['path'],
            contract_repair={'report_ref': corrected['report_ref'], 'initial_request_resent': False,
                             'maximum_repair_requests': 1, 'automatic_retry': False, 'manual_seed_boxes_used': False},
            estimated_cost_usd=str(Decimal(plan['context']['initial_estimated_cost_usd']) + new_cost),
            new_estimated_cost_usd=str(new_cost),
            localization={'report': 'full-frame-localization-v1/report.json',
                          'report_sha256': base.sha(out / 'full-frame-localization-v1/report.json'),
                          'model_calls': localization['model_calls'], 'manual_boxes_used': False},
            wall_time_s=round(time.monotonic() - started, 3))
        base.write(out / 'discovery.json', record)
        config = base.tracking_config(localized, rows, case.name, base.sha(out / 'discovery.json'))
        config['component_policy'] = 'rigid_component_visual_discovery_v1'
        destination = base.ROOT / 'annotations' / f'new-{case.name}-{run}.json'
        if destination.exists():
            raise ValueError('Preserve existing annotation records')
        base.write(destination, config)
        record['tracking_config'] = str(destination.relative_to(base.ROOT))
        base.write(out / 'metadata.json', {k: v for k, v in record.items() if k != 'result'})
        return record
    except BaseException as exc:
        record.update(status='failed_preserved', error_type=type(exc).__name__, automatic_retry=False,
                      wall_time_s=round(time.monotonic() - started, 3))
        base.write(out / 'metadata.json', {k: v for k, v in record.items() if k != 'result'})
        raise
