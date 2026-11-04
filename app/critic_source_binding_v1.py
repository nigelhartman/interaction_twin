"""Publication-time source/scene/image/payload binding for saved visual critics.

This verifies evidence lineage, not the truth of a model's verdict. It performs
no transport and does not reinterpret model text as paths or approval. One exact
historical thin wrapper is linked explicitly to its immutable original packet.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parent
LEGACY_REVIEW_SHA = '652a9d2391c4991402e94493f8d1620557dbfa5b15d4c4f10f281414fb078e44'
LEGACY_RESULT_SHA = 'da2432b3a8ac74ec4ff1538a45eba407ea5ff5532f73a6f317f33c294fb9053a'
LEGACY_RECEIPT_SHA = '4ed0c468b75f3128812e0073407cf9b8914ca558107bff689d679e7d65bbdcde'
LEGACY_PLAN = {'path': 'output/assembly-quality-20261104/multicase-review-v1/plan.json',
               'sha256': '683328e4eee3ebf3ede15b2b9a79e96e2563c0e5a32c610867082aa42260893d',
               'bytes': 22519}
LEGACY_SCOPE = 'output/assembly-quality-20261104/multicase-review-v1/'


def require(value, message):
    if not value:
        raise ValueError(message)


def _pointer(value):
    require(isinstance(value, dict) and set(value) == {'path', 'sha256', 'bytes'},
            'Exact critic artifact reference required')
    require(type(value['bytes']) is int and 0 < value['bytes'] <= 64 * 1024 * 1024
            and isinstance(value['sha256'], str) and re.fullmatch('[a-f0-9]{64}', value['sha256']),
            'Bounded critic artifact hash/length required')
    return value


def _reader(root):
    root = Path(root).resolve()
    def read(pointer, *, legacy=False, raw=False):
        _pointer(pointer)
        relative = pointer['path']
        require(isinstance(relative, str) and str(PurePosixPath(relative)) == relative
                and re.fullmatch('[A-Za-z0-9_./-]+', relative)
                and not any(p in {'.', '..', ''} or p.startswith('.') for p in relative.split('/')),
                'Ordinary critic artifact path required')
        require(relative.startswith('data/') or legacy and relative.startswith(LEGACY_SCOPE),
                'Critic artifact is outside its exact admitted packet')
        path = root / relative
        require(path.resolve() == path and path.is_file()
                and not any(p.is_symlink() for p in [path, *path.parents]),
                'Confined ordinary critic artifact required')
        require(path.stat().st_size == pointer['bytes'], 'Critic artifact length changed')
        value = path.read_bytes()
        require(hashlib.sha256(value).hexdigest() == pointer['sha256'], 'Critic artifact bytes changed')
        return value if raw else json.loads(value)
    return read


def _frame(program, index, timestamp, pointer, read):
    evidence = next((r for r in program['evidence'] if r['source_index'] == index), None)
    require(evidence is not None and evidence['timestamp_s'] == timestamp,
            'Critic source sample/PTS does not match the native evidence')
    require(pointer['path'] == evidence['frame_path'] and pointer['sha256'] == evidence['frame_sha256'],
            'Critic photograph belongs to another source frame')
    read(pointer, raw=True)
    return evidence


def _scene(scene, document):
    require(scene['original_program'] == document['evidence_program']
            and scene['source']['sha256'] == document['source']['sha256'],
            'Critic scene belongs to another source program')


def image_records(images, receipt):
    require(isinstance(images, list) and 1 <= len(images) <= 6,
            'Bounded actual critic image packet required')
    records = receipt.get('image_inputs')
    require(isinstance(records, list) and len(records) == len(images),
            'Accounted critic image count differs')
    for image, record in zip(images, records):
        require(record['source_sha256'] == image['sha256']
                and record['file'] == PurePosixPath(image['path']).name,
                'Accounted critic image order/hash differs from the bound packet')


def validate_files(document, *, root=ROOT):
    root = Path(root).resolve()
    read = _reader(root)
    program = read(document['evidence_program'])
    baseline = read(document['baseline_scene_ref'])
    _scene(baseline, document)
    results = []
    for review_ref in document['visual_review_refs']:
        review = read(review_ref)
        receipt = read(review['receipt'])
        model_result = read(review['model_result'])
        require(receipt.get('status') == 'complete_accounted' and receipt.get('model') == 'muse-spark-1.3'
                and review.get('automatic_approval') is False
                and model_result.get('physical_accuracy_verified') is False,
                'Actual accounted non-certifying image critic required')
        legacy = False
        packet_ref = review.get('plan')
        if packet_ref is None:
            require(review_ref['sha256'] == LEGACY_REVIEW_SHA
                    and review['receipt']['sha256'] == LEGACY_RECEIPT_SHA
                    and review['model_result']['sha256'] == LEGACY_RESULT_SHA,
                    'Thin critic wrapper lacks an explicit known packet binding')
            packet_ref, legacy = LEGACY_PLAN, True
        packet = read(packet_ref, legacy=legacy)
        images = packet['images']
        image_records(images, receipt)
        for image in images:
            read({k: image[k] for k in ['path', 'sha256', 'bytes']}, legacy=legacy, raw=True)
        require(packet.get('max_output_tokens') == 16000
                and packet.get('maximum_new_requests') == 1
                and packet.get('automatic_retries') is False,
                'Critic request scope differs from its once-only packet')
        source_indices = []
        form = packet['format']
        if form == 'interaction-twin-jabra-earbud-source-orientation-plan':
            context = packet['context']
            require(context['source_scene'] == document['baseline_scene_ref'],
                    'Orientation critic compared another baseline scene')
            part = next((p for p in baseline['components'] if p['id'] == context['component_id']), None)
            require(part is not None and part['asset'] == context['native_asset'],
                    'Orientation critic compared another native component')
            for row in context['source_samples']:
                evidence = _frame(program, row['source_sample_index'], row['timestamp_s'], row['source_rgb'], read)
                pose = next(r for r in part['poses'] if r['source_sample_index'] == row['source_sample_index'])
                require(pose['camera_from_native_mesh'] == row['baseline_pose'],
                        'Orientation critic baseline pose changed')
                observation = next(o for o in evidence['observations'] if o['track_id'] == part['id'])
                require(observation['mask_path'] == row['source_mask']['path']
                        and observation['mask_sha256'] == row['source_mask']['sha256']
                        and row['source_crop'] in images, 'Orientation source mask/crop identity differs')
                read(row['source_mask'], raw=True)
                source_indices.append(row['source_sample_index'])
        elif form == 'interaction-twin-multicase-visual-comparison-plan':
            comparisons = packet['context']['comparisons']
            matching = [r for r in comparisons if r['scene'] == document['baseline_scene_ref']]
            require(len(matching) == 1, 'Critic packet has no unique matching baseline comparison')
            comparison = matching[0]
            require(comparison['scene_program'] == document['evidence_program']
                    and comparison['source_crop'] in images and comparison['render'] in images,
                    'Source/render critic comparison belongs to another program')
            _frame(program, comparison['source_sample_index'], comparison['timestamp_s'], comparison['source_frame'], read)
            source_indices.append(comparison['source_sample_index'])
        elif form == 'interaction-twin-assembly-image-critic-plan':
            manifest = read(packet['manifest'])
            require(manifest['source_sha256'] == document['source']['sha256']
                    and manifest['baseline_scene_ref'] == document['baseline_scene_ref']
                    and manifest['images'] == images,
                    'Assembly critic manifest source/baseline/images differ')
            render_scene = read(manifest.get('candidate_scene_ref', manifest['baseline_scene_ref']))
            _scene(render_scene, document)
            parts = {p['id']: p for p in render_scene['components']}
            for row in manifest['renders']:
                source = next(i for i in images if i['kind'] == 'real_source_rgb'
                              and i['sample_index'] == row['source_sample_index'])
                require(row['source_rgb'] == {k: source[k] for k in ['path', 'sha256', 'bytes']},
                        'Assembly critic render cites a different photograph')
                _frame(program, source['sample_index'], source['timestamp_s'], row['source_rgb'], read)
                require(any({k: i[k] for k in ['path', 'sha256', 'bytes']} == row['render'] for i in images),
                        'Assembly critic render was not actually in the image packet')
                for rendered in row['parts']:
                    part = parts[rendered['id']]
                    pose = next(p for p in part['poses'] if p['source_sample_index'] == source['sample_index'])
                    require(part['asset'] == rendered['asset']
                            and pose['camera_from_native_mesh'] == rendered['matrix'],
                            'Assembly critic render asset/pose differs from its bound scene')
                source_indices.append(source['sample_index'])
            require(receipt['payload_sha256'] == packet['wire_provenance']['payload_sha256']
                    and receipt['image_inputs'] == packet['wire_provenance']['image_inputs'],
                    'Assembly critic accounted wire differs')
        else:
            raise ValueError('Unrecognized visual critic packet format')
        # Reproduce the exact image encoding/prompt/schema payload, without
        # constructing a client, reading credentials or performing transport.
        import source_observation_refinement_v3 as T
        if form == 'interaction-twin-assembly-image-critic-plan':
            folder = root / Path(packet_ref['path']).parent
            def sibling(name, maximum):
                path = folder / name
                require(path.resolve() == path and path.is_file()
                        and not any(p.is_symlink() for p in [path, *path.parents])
                        and 0 < path.stat().st_size <= maximum,
                        'Ordinary bounded critic prompt/schema file required')
                return path.read_bytes()
            prompt = sibling('prompt.txt', 300000).decode()
            provider = json.loads(sibling('provider-schema.json', 2 * 1024 * 1024))
            require(hashlib.sha256(prompt.encode()).hexdigest() == packet['prompt_sha256']
                    and T.digest(provider) == packet['provider_schema_sha256'],
                    'Critic prompt/provider schema changed')
        else:
            prompt, provider = packet['prompt'], packet['schema']
            require(packet['model'] == 'muse-spark-1.3', 'Critic packet model differs')
        wire = T.wire(prompt, provider, images, root=Path(root).resolve())
        require(wire['payload_sha256'] == receipt['payload_sha256']
                and wire['image_inputs'] == receipt['image_inputs'],
                'Actual critic wire cannot be reproduced from the claimed packet')
        results.append({'review_ref': review_ref, 'packet_ref': packet_ref,
                        'historical_explicit_packet_binding': legacy,
                        'source_sample_indices': source_indices,
                        'source_and_baseline_bound': True, 'accounted_payload_reproduced': True,
                        'critic_is_not_automatic_approval': True,
                        'physical_accuracy_verified': False})
    return results
