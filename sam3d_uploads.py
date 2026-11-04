"""Source-bound SAM 3D reconstruction for ordinary uploads.

The initial handoff contains RGB/mask evidence, never an earlier mesh or pose.
The optional cluster bridge runs a bounded job or records why it is pending.
There is deliberately no alternate geometry backend in this adapter.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import initial_sam3d_handoff as handoffs
import sam3d_milestone as native
import upgrade_bundles as U

ROOT = Path(__file__).resolve().parent
PROVIDER = 'sam3d_native'
DEFAULT_PROFILE = 'meta_sam3d_default'


def native_review(model_path, *, root=ROOT):
    """Build/revalidate an immutable source-bound browser page; no inference."""
    root = Path(root).resolve()
    model_path = U._local(model_path, root)
    model = U.read(model_path)
    handoff = model['initial_reconstruction_lineage']['handoff']
    identity = handoff['sha256']
    handoffs.require(isinstance(identity, str) and len(identity) == 64
                     and all(c in '0123456789abcdef' for c in identity), 'Native review handoff hash differs')
    directory = root / 'data/sam3d-upload-reviews' / identity
    if not directory.exists():
        builder = importlib.import_module('scripts.build_initial_sam3d_review_v2')
        builder.build(model_path.parent, directory, root=root)
    build = U.read(U._local(directory / 'review-build.json', root))
    files = native._tree(directory)
    handoffs.require(build.get('format') == 'sam3d-initial-native-review-build'
                     and build.get('version') == 2 and build.get('new_models_or_api_calls') == 0
                     and build.get('native_geometry_modified') is False
                     and build.get('files') == native._records({k: v for k, v in files.items() if k != 'review-build.json'}),
                     'Saved native review inventory changed')
    manifest = U.read(directory / 'review-manifest.json')
    handoffs.require(manifest.get('component_id') == model['object_id']
                     and manifest.get('source', {}).get('video_sha256') == model['source']['video_sha256']
                     and manifest.get('native_glb_sha256') == U.sha(model_path.parent / 'asset.glb')
                     and manifest.get('initial_handoff') == handoff
                     and manifest.get('import_receipt', {}).get('sha256') == U.sha(model_path.parent / 'import.json'),
                     'Saved native review source/import binding changed')
    return {'review': handoffs.reference(directory / 'index.html', root),
            'review_status': 'verified_native_review_ready', 'review_manifest': handoffs.reference(directory / 'review-manifest.json', root)}


def reconstruct_component(component, bundle, destination, *, cancel_event=None, root=ROOT):
    root = Path(root).resolve()
    bundle = native._directory(bundle, root)
    destination = native._directory(destination, root, exists=False)
    selection = U._local(component['selection'], root)
    identity = hashlib.sha256((component['track_id'] + '\n' + U.sha(selection)
                               + '\n' + U.sha(bundle / 'input.json')).encode()).hexdigest()
    handoff_dir = root / 'data/initial-sam3d-handoffs' / ('upload-' + identity[:32])
    if not handoff_dir.exists():
        handoffs.prepare(bundle / 'input.json', selection, handoff_dir, root=root)
    checked = handoffs.inspect(handoff_dir / 'handoff.json', root=root)
    handoffs.require(checked['component_id'] == component['track_id'], 'Upload component/handoff identity differs')
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError('Cancelled before SAM 3D dispatch; frozen input retained')
    # Only this reviewed adapter chooses a transport. A browser supplies no SSH,
    # scheduler command, host, Python executable, model path or shell argument.
    import sam3d_cluster_bridge
    result = sam3d_cluster_bridge.reconstruct(handoff_dir / 'handoff.json', destination,
                                             cancel_event=cancel_event, root=root)
    handoffs.require(isinstance(result, dict) and result.get('status') in
                     {'native_candidate_ready', 'awaiting_gpu_worker', 'failed_no_retry'},
                     'SAM 3D bridge returned an unsupported status')
    result = dict(result, provider=PROVIDER, component_id=component['track_id'],
                  handoff=checked['handoff'], baseline_asset_read=False, old_poses_read=False,
                  source_video_pose_validated=False, simulation_ready=False)
    if result['status'] == 'native_candidate_ready':
        model_path = U._local(result['model'], root)
        native_path = U._local(result['native_asset'], root)
        model = U.read(model_path)
        binding = U.read(handoff_dir / 'source-binding.json')
        lineage = model.get('initial_reconstruction_lineage', {})
        handoffs.require(model.get('sam3d_output') is True and model.get('learned_inference_executed') is True
                         and model.get('object_id') == component['track_id']
                         and all(model.get('source', {}).get(key) == value for key, value in binding['source'].items())
                         and lineage.get('handoff') == checked['handoff']
                         and lineage.get('source_binding_sha256') == U.sha(handoff_dir / 'source-binding.json')
                         and lineage.get('baseline_asset_read') is False and lineage.get('old_poses_read') is False,
                         'Returned native candidate does not match this upload source/component')
        expected = [r for r in model.get('artifact_files', []) if r.get('file') == 'asset.glb']
        handoffs.require(native_path == model_path.parent / 'asset.glb' and len(expected) == 1
                         and U.sha(native_path) == expected[0].get('sha256'), 'Returned native asset changed')
        # A generated part is useful immediately, but its video motion and common
        # assembly are not established by generation or by a matching source hash.
        result.update(has_gaps=True, native_motion_status='fresh_native_alignment_required')
        try:
            result.update(native_review(model_path, root=root))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            # Browser limits must not erase a genuine model or expand budgets.
            result.update(review_status='native_review_unavailable', review_error=str(exc)[:1000])
    else:
        handoffs.require(not result.get('model') and not result.get('native_asset'),
                         'Pending/failed SAM 3D must not expose a substitute mesh')
        result['has_gaps'] = True
    return result
