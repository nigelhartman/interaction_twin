"""Additive pre-transport provenance correction for one unsent SAM request.

The old attempt reached only prompt validation, before _run/request callbacks.
No provider response or uncertain request is replayed. Original packets remain.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json

import charger_tracking_recovery_20261103_v1 as C
import meta_sam_api as SAM

ROOT = Path(__file__).resolve().parent
OUTPUT = C.OUTPUT + '/sam-correction-v2'
PROMPT_ORIGIN = 'model_generated_fresh_Muse_adapter_body_localization'


def prior_proof(root=ROOT):
    root = Path(root).resolve(); old = C.local(C.OUTPUT, root); plan = C.read(old/'sam-plan.json')
    C.require(C.ref(old/'sam-plan.json', root)['sha256'] == '27e92f8b3601c055d1131ed6523928c42ce29717cb729404b5c6b6150ff700ce', 'Exact unsent failed plan required')
    intent = C.read(old/'sam-attempt/intent.json')
    C.require(intent == {'plan': C.ref(old/'sam-plan.json', root), 'maximum_new_requests': 1, 'automatic_retry': False}, 'Exact prior intent required')
    C.require(sorted(p.name for p in (old/'sam-attempt').iterdir()) == ['intent.json'], 'Any prior transport/accounting/result artifact forbids this correction')
    try:
        SAM._model_box_prompt(plan['objects'], width=plan['width'], height=plan['height'], frame_count=len(plan['frame_sources']), prompt_origin=plan['prompt_origin'], media='video')
    except SAM.SamAPIError as error:
        C.require(str(error) == 'Box prompts require explicit model-generated provenance', 'Only exact known local provenance rejection is eligible')
    else:
        raise ValueError('Original invalid prompt no longer fails before request transport')
    # This formatter/parser preflight never constructs a client or invokes _run.
    prompt, normalized = SAM._model_box_prompt(plan['objects'], width=plan['width'], height=plan['height'], frame_count=len(plan['frame_sources']), prompt_origin=PROMPT_ORIGIN, media='video')
    return {'old_plan': C.ref(old/'sam-plan.json', root), 'old_intent': C.ref(old/'sam-attempt/intent.json', root),
        'failure_phase': 'local_box_provenance_validation_before_run_or_request', 'old_accounting_or_result_exists': False,
        'original_error': 'Box prompts require explicit model-generated provenance', 'new_prompt_sha256': C.digest(prompt),
        'new_normalized_prompts': normalized, 'network_or_model_client_constructed_during_preflight': False}


def plan(root=ROOT):
    old = C.local(C.OUTPUT, root); original = C.read(old/'sam-plan.json'); proof = prior_proof(root)
    original = dict(original, prompt_origin=PROMPT_ORIGIN, format='interaction-twin-charger-sam-pretransport-correction',
        correction_implementation=C.ref(Path(__file__), root), pretransport_failure_proof=proof,
        original_prepared_payload_not_sent=True, maximum_new_requests=1, automatic_retries=False)
    for key in ['program', 'localized_by', 'localization_result', 'video', 'encoding']:
        C.require(C.ref(original[key]['path'], root) == original[key], 'Original source/localization/video changed')
    return original


def prepare(root=ROOT):
    value = plan(root); folder = C.local(OUTPUT, root); folder.mkdir(exist_ok=False)
    C.write_new(folder/'plan.json', value)
    return C.ref(folder/'plan.json', root)


def run(authorized_plan_sha256, *, allow_paid_api=False, root=ROOT, model_client=None):
    root = Path(root).resolve(); folder = C.local(OUTPUT, root); path = folder/'plan.json'; value = C.read(path)
    C.require(value == plan(root), 'Exact corrected source packet changed')
    C.require(allow_paid_api and C.ref(path, root)['sha256'] == authorized_plan_sha256, 'Explicit exact corrected plan authorization required')
    attempt = folder/'attempt'; attempt.mkdir(exist_ok=False)
    C.write_new(attempt/'intent.json', {'plan': C.ref(path, root), 'maximum_new_requests': 1, 'automatic_retry': False})
    response = SAM.segment_video_boxes(C.local(value['video']['path'], root), value['objects'], attempt/'result',
        (model_client or C.client(root)).sam_callback(attempt/'accounting', video_frames=len(value['frame_sources'])),
        width=value['width'], height=value['height'], frame_sources=value['frame_sources'], prompt_origin=value['prompt_origin'], propagation_direction='both')
    C.write_new(attempt/'complete.json', {'plan': C.ref(path, root), 'result': C.ref(attempt/'result/metadata.json', root),
        'request': C.ref(attempt/'accounting/meta-request.json', root), 'instance_count': len(response['instances']),
        'new_api_calls': 1, 'automatic_publication': False, 'native_geometry_modified': False, 'identity_gap_merge': False,
        'prior_unsent_attempt_preserved': value['pretransport_failure_proof']})
    return C.ref(attempt/'complete.json', root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--authorized-plan-sha256'); parser.add_argument('--allow-paid-api', action='store_true'); args = parser.parse_args()
    result = prepare() if args.action == 'prepare' else run(args.authorized_plan_sha256, allow_paid_api=args.allow_paid_api)
    print(json.dumps(result), flush=True)
