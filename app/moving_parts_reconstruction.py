"""Two independent offline TripoSR/MPS previews from genuine SAM3.1 masks.

These are single-image model priors, not observed mathematical reconstruction,
SAM3D, common assembly, recovered articulation, metric geometry or robot assets.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

import geometry_intelligence as gi
from scripts import stage3_reconstruct_local as existing


INPUT_MANIFEST = 'data/moving-parts-sam31-20261031/jabra-localized-parts-v2/manifest.json'
PARTS = ['case_base', 'case_lid']
RESOLUTION = 96
SEED = 42


def load_parts(root):
    path = gi.local_path(root, INPUT_MANIFEST)
    manifest = gi.read(path)
    if manifest['status'] != 'complete_localized_candidates' or manifest['requests_attempted'] != 3:
        raise ValueError('Expected the genuine completed visual-localization/SAM candidates')
    components = []
    for proposal_id in PARTS:
        proposal = next(p for p in manifest['proposals'] if p['id'] == proposal_id)
        if len(proposal['frames']) != 1:
            raise ValueError('Exactly one actual source-frame mask per part is supported')
        frame = proposal['frames'][0]
        if frame['instance_count'] != 1 or len(frame['instances']) != 1:
            raise ValueError('A unique native mask is needed; no automatic multi-instance choice')
        if frame['evidence_id'] != 'jabra::source_000023' or frame['timestamp_s'] != 22.001933:
            raise ValueError('Unexpected source frame for this bounded local test')
        instance = frame['instances'][0]
        image, mask = gi.local_path(root, frame['image_url']), gi.local_path(root, instance['mask_url'])
        if gi.sha256(image) != frame['image_sha256'] or gi.sha256(mask) != instance['mask_sha256']:
            raise ValueError('Source SAM/RGB bytes changed')
        components.append({'proposal_id': proposal_id, 'label': proposal['label'], 'source_frame': frame,
                           'image_path': str(image.relative_to(root)), 'mask_path': str(mask.relative_to(root)),
                           'image_sha256': frame['image_sha256'], 'mask_sha256': instance['mask_sha256'],
                           'source_video': manifest['case_source'], 'source_case': manifest['case_id'],
                           'model_localization': proposal['localization'], 'box_transform': proposal['box_transform']})
    return components


def prepare_input(image_path, mask_path):
    with Image.open(image_path) as opened:
        rgb = np.array(opened.convert('RGB'))
    with Image.open(mask_path) as opened:
        mask = np.array(opened)
    if mask.ndim != 2 or mask.shape != rgb.shape[:2] or not set(np.unique(mask)).issubset({0, 255}):
        raise ValueError('A source-size binary SAM mask is required')
    mask = mask > 0
    if not mask.any() or mask.all():
        raise ValueError('Nonempty visible component and background are required')
    image, preprocessing = existing.preprocess(rgb, mask, foreground_ratio=.85)
    before = image.size
    if max(before) > 512:
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    preprocessing.update(model_input_size=list(image.size), pre_cap_size=list(before),
                         deterministic_cap_max_side=512, input_resize_scale=image.width / before[0],
                         mask_holes_inpainted=False, manual_pixel_or_geometry_edits=False,
                         source_to_square_transform='source pixel minus bbox top-left plus recorded crop_offset_xy_in_model_input; optional uniform max512 scale')
    return rgb, mask, image, preprocessing


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root / 'data') or output.exists():
        raise ValueError('Fresh data output directory is required')
    components = load_parts(root)
    output.mkdir(parents=True)
    plan = {'format': 'moving-parts-triposr-plan', 'version': 1, 'components': components,
            'source_manifest': INPUT_MANIFEST, 'source_manifest_sha256': gi.sha256(root / INPUT_MANIFEST),
            'device': 'mps', 'offline': True, 'learned_inferences_maximum': 2,
            'resolution': RESOLUTION, 'seed': SEED, 'manual_geometry_or_poses': False,
            'helper_sha256': gi.sha256(existing.__file__),
            'review_generator_sha256': gi.sha256(root / 'scripts/stage3_reconstruction_review.py'),
            'review_runtime_sha256': gi.sha256(root / 'reconstruction_observer.js')}
    gi.write(output / 'plan.json', plan)
    (output / 'implementation.py').write_bytes(Path(__file__).read_bytes())
    (output / 'implementation.py').chmod(0o444)
    metadata = {'status': 'prepared_not_inferred', 'plan_sha256': gi.sha256(output / 'plan.json'),
                'implementation_sha256': gi.sha256(__file__), 'new_api_calls': 0}
    gi.write(output / 'metadata.json', metadata)
    return metadata


def run(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    record = gi.read(output / 'metadata.json')
    if record['status'] != 'prepared_not_inferred' or (output / 'manifest.json').exists():
        raise ValueError('Already attempted; preserve prior run and no model retry')
    if gi.sha256(__file__) != record['implementation_sha256'] or gi.sha256(output / 'plan.json') != record['plan_sha256']:
        raise ValueError('Frozen reconstruction plan/code changed')
    plan = gi.read(output / 'plan.json')
    for path, expected in [(root / INPUT_MANIFEST, plan['source_manifest_sha256']),
                           (Path(existing.__file__), plan['helper_sha256']),
                           (root / 'scripts/stage3_reconstruction_review.py', plan['review_generator_sha256']),
                           (root / 'reconstruction_observer.js', plan['review_runtime_sha256'])]:
        if gi.sha256(path) != expected:
            raise ValueError('Source input or reviewed helper changed')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('CPU learned operator fallback is forbidden')
    os.environ['HF_HOME'] = str(root / 'models/hf-cache')
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    import trimesh
    from skimage.measure import marching_cubes
    from scripts.stage3_reconstruction_review import build_review
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no CPU learned inference fallback')
    provenance = existing.model_provenance()
    sys.path.insert(0, str(root / 'models/triposr-code'))
    from tsr.system import TSR
    model = TSR.from_pretrained(str(root / 'models/triposr-checkpoint'), 'config.yaml', 'model.ckpt').to('mps').eval()
    model.renderer.set_chunk_size(8192)
    if any(p.device.type != 'mps' for p in model.parameters()):
        raise RuntimeError('All learned parameters must be on MPS')
    manifest = {'format': 'moving-parts-local-reconstruction-review', 'version': 1, 'status': 'running',
                'title': 'Independent moving-part model previews',
                'caption': 'Local TripoSR model priors · unvalidated single-image geometry',
                'source_manifest_url': '/files/' + INPUT_MANIFEST, 'model_provider': 'Stability AI / Tripo AI · local TripoSR',
                'components': [], 'failures': [], 'learned_inferences_attempted': 0, 'new_api_calls': 0,
                'limitations': ['Every component has independent native normalized coordinates, no common assembly or source camera pose.',
                                'Hidden surfaces and colours are model predictions, not observed reconstruction or SAM3D.',
                                'No hinge axis, motion, metric scale, contact, insertion or simulation validity is established.',
                                'Original full native geometry is preserved with no mesh repair, primitive replacement or component selection.']}
    record['status'] = 'running_two_bounded_local_inferences'
    gi.write(output / 'metadata.json', record)
    gi.write(output / 'manifest.json', manifest)
    for component in plan['components']:
        directory = output / component['proposal_id']
        directory.mkdir()
        started = time.monotonic()
        metadata = {'schema_version': '0.1', 'status': 'running', 'object_id': component['proposal_id'], 'backend': 'TripoSR',
                    'provider': 'Stability AI / Tripo AI · offline local model', 'sam3d_output': False,
                    'geometry_status': 'single_image_model_prior_unvalidated', 'model_provenance': provenance,
                    'runtime': {'device': 'mps', 'torch': torch.__version__, 'offline': True, 'cpu_operator_fallback_enabled': False},
                    'learned_inference_executed': False, 'manual_geometry_fallback': False, 'manual_pose_used': False,
                    'seed': SEED, 'density_grid_resolution': RESOLUTION, 'density_isovalue': 25.,
                    'appearance_provenance': 'TripoSR decoder-predicted native vertex colours; hidden colours unobserved',
                    'appearance': 'model-predicted vertex colours, not original photograph UV textures',
                    'coordinates': 'Unchanged native TripoSR normalized canonical coordinates; independent per-part gauge',
                    'metric_scale_verified': False, 'camera_pose_estimated': False,
                    'physical_assembly_established': False, 'articulation_established': False,
                    'validation_status': 'unaccepted_single_image_model_comparison',
                    'source': {'image_path': 'source-rgb.png', 'mask_path': 'source-mask.png',
                               'timestamp_s': component['source_frame']['timestamp_s'], 'sample_index': 23,
                               'time_domain': 'source_video_presentation_seconds', 'video_sha256': component['source_video']['sha256'],
                               'original_source_image_url': component['source_frame']['image_url'],
                               'original_source_image_sha256': component['image_sha256'],
                               'original_sam31_mask_url': component['source_frame']['instances'][0]['mask_url'],
                               'original_sam31_mask_sha256': component['mask_sha256'],
                               'mask_provenance': {'provider': 'Meta Model API', 'model': 'sam-3.1',
                                                   'prompt_origin': 'model_generated_fullframe_visual_box_localization',
                                                   'manual_seed_boxes_used': False, 'audit_status': 'not_model_audited'}},
                    'limitations': manifest['limitations']}
        gi.write(directory / 'metadata.json', metadata)
        try:
            image_path, mask_path = root / component['image_path'], root / component['mask_path']
            if gi.sha256(image_path) != component['image_sha256'] or gi.sha256(mask_path) != component['mask_sha256']:
                raise ValueError('Original RGB/SAM bytes changed before inference')
            rgb, mask, model_input, preprocessing = prepare_input(image_path, mask_path)
            Image.fromarray(rgb).save(directory / 'source-rgb.png')
            Image.fromarray(mask.astype(np.uint8) * 255).save(directory / 'source-mask.png')
            model_input.save(directory / 'model-input.png')
            metadata['preprocessing'] = dict(preprocessing, model_input_sha256=gi.sha256(directory / 'model-input.png'))
            metadata['source'].update(image_sha256=gi.sha256(directory / 'source-rgb.png'), mask_sha256=gi.sha256(directory / 'source-mask.png'))
            manifest['learned_inferences_attempted'] += 1
            gi.write(output / 'manifest.json', manifest)
            torch.manual_seed(SEED)
            inference_started = time.monotonic()
            with torch.inference_mode():
                scene = model([model_input], device='mps')
                torch.mps.synchronize()
            if scene.device.type != 'mps':
                raise RuntimeError('Learned scene must remain on MPS')
            metadata['learned_inference_executed'] = True
            metadata['image_to_triplane_seconds'] = time.monotonic() - inference_started
            np.save(directory / 'learned-triplane.npy', scene.detach().cpu().numpy())
            radius = float(model.renderer.cfg.radius)
            axis = torch.linspace(-radius, radius, RESOLUTION)
            grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
            density = []
            with torch.inference_mode():
                for block in grid.split(8192):
                    prediction = model.renderer.query_triplane(model.decoder, block.to('mps'), scene[0])
                    density.append(prediction['density_act'][:, 0].cpu())
            volume = torch.cat(density).reshape((RESOLUTION,) * 3).numpy()
            np.save(directory / 'learned-density.npy', volume)
            vertices, faces, _, _ = marching_cubes(volume, level=25., spacing=(2 * radius / (RESOLUTION - 1),) * 3)
            vertices -= radius
            colours = []
            with torch.inference_mode():
                for block in torch.from_numpy(vertices.copy()).float().split(8192):
                    colours.append(model.renderer.query_triplane(model.decoder, block.to('mps'), scene[0])['color'].cpu().numpy())
            colours = np.concatenate(colours)
            np.savez_compressed(directory / 'learned-surface.npz', vertices=vertices, faces=faces, vertex_colors_float=colours)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=(np.clip(colours, 0, 1) * 255).astype(np.uint8), process=False)
            mesh.export(directory / 'asset.glb')
            metadata.update(status='single_image_model_prior_unvalidated', wall_seconds=time.monotonic() - started,
                            geometry={'vertices': len(vertices), 'faces': len(faces), 'bounds': mesh.bounds.tolist(),
                                      'all_components_preserved': True, 'mesh_processing': False},
                            outputs={'glb_path': 'asset.glb', 'glb_sha256': gi.sha256(directory / 'asset.glb')})
            gi.write(directory / 'metadata.json', metadata)
            review = build_review(directory / 'metadata.json', directory / 'review')
            manifest['components'].append({'proposal_id': component['proposal_id'], 'label': component['label'],
                'model_provider': manifest['model_provider'], 'glb_url': '/files/' + str((directory / 'asset.glb').relative_to(root)),
                'glb_sha256': gi.sha256(directory / 'asset.glb'), 'metadata_url': '/files/' + str((directory / 'metadata.json').relative_to(root)),
                'metadata_sha256': gi.sha256(directory / 'metadata.json'),
                'review_url': '/files/' + str((directory / 'review/index.html').relative_to(root)),
                'preview_url': '/files/' + str((directory / 'review/preview.png').relative_to(root)),
                'source_image_url': component['source_frame']['image_url'],
                'mask_url': component['source_frame']['instances'][0]['mask_url'],
                'geometry_status': 'single_image_model_prior_unvalidated', 'device': 'mps',
                'learned_inference_executed': True, 'sam3d_output': False,
                'source_camera_available': False, 'physical_assembly_established': False,
                'vertices': len(vertices), 'faces': len(faces), 'wall_seconds': metadata['wall_seconds']})
            print({'part': component['proposal_id'], 'status': metadata['status'], 'vertices': len(vertices), 'faces': len(faces)}, flush=True)
            del scene, density, volume, grid, mesh
            torch.mps.empty_cache()
        except Exception as exc:
            metadata.update(status='failed_no_fallback', failure_type=type(exc).__name__, wall_seconds=time.monotonic() - started)
            gi.write(directory / 'metadata.json', metadata)
            manifest['failures'].append({'proposal_id': component['proposal_id'], 'failure_type': type(exc).__name__})
        finally:
            gi.write(output / 'manifest.json', manifest)
    manifest['status'] = 'complete_with_failures' if manifest['failures'] else 'complete_independent_model_previews'
    gi.write(output / 'manifest.json', manifest)
    record.update(status=manifest['status'], manifest_sha256=gi.sha256(output / 'manifest.json'))
    gi.write(output / 'metadata.json', record)
    return manifest
