#!/usr/bin/env python3
"""Attested display-only automation provenance, preserving all model originals."""
from pathlib import Path
import copy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import geometry_intelligence as gi
from scripts.stage3_reconstruction_review import build_review


def main():
    base = ROOT / 'data/moving-parts-reconstruction-20261031/jabra-parts-triposr-v1'
    if (base / 'display-manifest.json').exists():
        raise ValueError('Display derivative already exists; preserve prior records')
    originals = {str(p.relative_to(ROOT)): gi.sha256(p) for p in base.rglob('*') if p.is_file()}
    generator = ROOT / 'scripts/stage3_reconstruction_review.py'
    runtime = ROOT / 'reconstruction_observer.js'
    code_hashes = {str(p.relative_to(ROOT)): gi.sha256(p) for p in [Path(__file__), generator, runtime]}
    manifest = gi.read(base / 'manifest.json')
    localized_path = ROOT / 'data/moving-parts-sam31-20261031/jabra-localized-parts-v2/manifest.json'
    localized = gi.read(localized_path)
    derived = copy.deepcopy(manifest)
    derivations = []
    for component in derived['components']:
        proposal_id = component['proposal_id']
        directory = base / proposal_id
        original_metadata_path = directory / 'metadata.json'
        original_metadata = gi.read(original_metadata_path)
        source = original_metadata['source']
        localized_proposal = next(p for p in localized['proposals'] if p['id'] == proposal_id)
        frame = localized_proposal['frames'][0]
        native_path = gi.local_path(ROOT, frame['native_metadata_url'])
        if gi.sha256(native_path) != frame['native_metadata_sha256']:
            raise ValueError('Exact native SAM metadata identity changed')
        native = gi.read(native_path)
        provenance = native['input_provenance']
        if provenance['manual_seed_boxes_used'] is not False or provenance['reference_mask_passed_to_model'] is not False:
            raise ValueError('Native mask has no verified model-only prompting declaration')
        if provenance['prompt_origin'] != 'model_generated_fullframe_visual_box_localization':
            raise ValueError('Unexpected native prompt provenance')
        if original_metadata['manual_geometry_fallback'] is not False or original_metadata['manual_pose_used'] is not False:
            raise ValueError('Reconstruction did not exclude manual geometry or poses')
        if source['original_sam31_mask_sha256'] != frame['instances'][0]['mask_sha256']:
            raise ValueError('Source mask identity differs from native model candidate')
        if source['original_source_image_sha256'] != frame['image_sha256'] or provenance['image_sha256'] != frame['image_sha256']:
            raise ValueError('Source full-frame identity differs')
        receipt_path = gi.local_path(ROOT, localized['localization']['receipt_url'])
        receipt = gi.read(receipt_path)
        if receipt['provider'] != 'meta_api' or receipt['model'] != 'muse-spark-1.3' or receipt['status'] != 'complete_accounted':
            raise ValueError('Visual localization is not bound to actual accounted model inference')
        exact_localized_row = next(r for r in localized['localization']['records'] if r['component_id'] == proposal_id)
        if exact_localized_row != localized_proposal['localization'] or exact_localized_row['visibility'] != 'visible':
            raise ValueError('Model localization trace differs from part prompt')
        review_metadata = copy.deepcopy(original_metadata)
        review_metadata['mask_provenance'] = {
            'provider': 'Meta Model API', 'model': native['model'],
            'prompt_origin': provenance['prompt_origin'],
            'manual_seed_boxes_used': False, 'manual_intervention': False,
            'manual_geometry_used': False, 'manual_pose_used': False,
            'audit_status': 'not_model_audited',
            'visual_localization_provider': 'Meta Model API', 'visual_localization_model': receipt['model'],
            'visual_localization_receipt_url': localized['localization']['receipt_url'],
            'visual_localization_receipt_sha256': gi.sha256(receipt_path),
            'source_native_sam_metadata_url': frame['native_metadata_url'],
            'source_native_sam_metadata_sha256': frame['native_metadata_sha256'],
            'physical_identity_or_geometry_verified': False,
        }
        attestation = {
            'purpose': 'Expose exact existing model-only mask provenance to unchanged generic review renderer.',
            'original_metadata_url': '/files/' + str(original_metadata_path.relative_to(ROOT)),
            'original_metadata_sha256': gi.sha256(original_metadata_path),
            'localized_manifest_url': '/files/' + str(localized_path.relative_to(ROOT)),
            'localized_manifest_sha256': gi.sha256(localized_path),
            'fields_added': ['mask_provenance', 'display_derivation'],
            'manual_intervention_derivation': 'false derives from exact native SAM input_provenance.manual_seed_boxes_used=false and model-generated box prompt origin, matched to the actual accounted Muse Spark localization; reconstruction also records manual_geometry_fallback=false and manual_pose_used=false.',
            'geometry_changed': False, 'appearance_changed': False, 'inference_executed': False,
            'code_hashes': code_hashes,
        }
        review_metadata['display_derivation'] = attestation
        display_metadata_path = directory / 'metadata-review-v2.json'
        if display_metadata_path.exists() or (directory / 'review-v2').exists():
            raise ValueError('Fresh part display derivative required')
        gi.write(display_metadata_path, review_metadata)
        build_review(display_metadata_path, directory / 'review-v2')
        component['original_review_url'] = component['review_url']
        component['original_preview_url'] = component['preview_url']
        component['review_url'] = '/files/' + str((directory / 'review-v2/index.html').relative_to(ROOT))
        component['preview_url'] = '/files/' + str((directory / 'review-v2/preview.png').relative_to(ROOT))
        component['review_metadata_url'] = '/files/' + str(display_metadata_path.relative_to(ROOT))
        component['review_metadata_sha256'] = gi.sha256(display_metadata_path)
        # metadata_url remains the unchanged original learned-asset metadata.
        derivations.append(dict(attestation, proposal_id=proposal_id,
                                display_metadata_sha256=gi.sha256(display_metadata_path)))
    for relative, expected in originals.items():
        if gi.sha256(ROOT / relative) != expected:
            raise ValueError('An original learned asset/review artifact changed')
    for relative, expected in code_hashes.items():
        if gi.sha256(ROOT / relative) != expected:
            raise ValueError('Display generator/runtime changed')
    derived['display_derivation'] = {
        'source_manifest_url': '/files/' + str((base / 'manifest.json').relative_to(ROOT)),
        'source_manifest_sha256': originals[str((base / 'manifest.json').relative_to(ROOT))],
        'geometry_changed': False, 'new_api_calls': 0, 'new_learned_inference': False,
        'code_hashes': code_hashes, 'attestation_url': '/files/' + str((base / 'display-attestation.json').relative_to(ROOT))}
    gi.write(base / 'display-manifest.json', derived)
    gi.write(base / 'display-attestation.json', {
        'status': 'verified_display_only_automation_provenance',
        'display_manifest_sha256': gi.sha256(base / 'display-manifest.json'),
        'all_original_artifact_hashes_unchanged': True, 'original_artifact_hashes': originals,
        'unchanged_generator_and_runtime': True, 'derivations': derivations,
        'glb_geometry_and_colour_bytes_unchanged': True, 'new_api_calls': 0,
        'model_or_gpu_inference_executed': False})
    print({'status': 'display_derivatives_generated_originals_unchanged',
           'manifest': '/files/' + str((base / 'display-manifest.json').relative_to(ROOT))})


if __name__ == '__main__':
    main()
