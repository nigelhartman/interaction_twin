"""Local, TRAIN-only camera assumptions for ordinary video reconstruction.

The learned model predicts intrinsics, not measured calibration. Depth, point
maps and generated geometry never enter this adapter. Existing geometry must
be reconstructed again rather than reprojected with a different camera.
"""
from __future__ import annotations

import hashlib
import copy
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

POLICY = {
    'version': 'moge2-train-camera-prior-v1',
    'model': 'Ruicheng/moge-2-vits-normal', 'device': 'mps',
    'minimum_train_images': 2, 'maximum_train_images': 3, 'maximum_input_long_side': 512, 'num_tokens': 1200,
    'selection': 'first, middle, last of timestamp-ordered TRAIN views; no quality-result selection',
    'pixel_mapping': 'diag(native_width,native_height,1) @ normalized_K; principal_point -= 0.5',
    'minimum_focal_over_long_side': .15, 'maximum_focal_over_long_side': 5.,
    'maximum_relative_focal_spread': .15,
    'maximum_principal_point_spread_over_long_side': .02,
    'aggregation': 'elementwise median of consistent native-pixel TRAIN camera priors',
    'distortion': 'zero_assumed_unverified', 'heldout_images_used': False,
    'depth_or_point_maps_used': False, 'manual_coordinates': False,
    'new_api_calls': 0, 'downloads_allowed': False, 'device_fallback_allowed': False,
    'camera_calibrated': False, 'physical_scale_known': False,
    'physical_accuracy_verified': False, 'consistency_is_accuracy': False,
    'source_orientation': 'exact decoded saved frame orientation; no additional adapter rotation, crop or padding',
    'preprocessing': 'full-frame cv2 INTER_AREA resize, BGR-to-RGB CHW, float32 divided by 255',
}
LOADED_SOURCE = Path(__file__).read_bytes()
LOADED_SOURCE_SHA256 = hashlib.sha256(LOADED_SOURCE).hexdigest()


def describe():
    return copy.deepcopy(POLICY)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_once(path, value):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError('Output files must not traverse symlink ancestors')
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def write_bytes_once(path, content):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError('Output files must not traverse symlink ancestors')
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(content)


def regular_file(path, root):
    path = Path(path)
    root = Path(root).resolve()
    if not path.is_absolute() or not path.is_relative_to(root) or path.resolve() != path or not path.is_file():
        raise ValueError('Only confined regular files without symlink ancestors are accepted')
    return path


def source_file(root, url):
    if not isinstance(url, str) or not url.startswith('/files/data/'):
        raise ValueError('Camera inputs must be saved source images')
    relative = url.removeprefix('/files/')
    if any(p.startswith('.') or not p for p in relative.split('/')) or any(c in relative for c in ['\\', '?', '#', '\0']):
        raise ValueError('Invalid saved source image path')
    return regular_file(Path(root).resolve() / relative, Path(root).resolve() / 'data')


def dimensions(width, height):
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in [width, height]):
        raise ValueError('Positive integer native width and height are required')
    return width, height


def native_intrinsics(normalized, width, height):
    dimensions(width, height)
    k = np.asarray(normalized, float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or not np.array_equal(k[2], [0., 0., 1.]):
        raise ValueError('Expected finite normalized pinhole intrinsics')
    if abs(k[0, 1]) > 1e-10 or abs(k[1, 0]) > 1e-10:
        raise ValueError('Expected native image dimensions and zero skew')
    k = np.diag([width, height, 1.]) @ k
    k[0, 2] -= .5; k[1, 2] -= .5
    long_side = max(width, height)
    if any(not POLICY['minimum_focal_over_long_side'] <= k[i, i] / long_side <= POLICY['maximum_focal_over_long_side'] for i in [0, 1]):
        raise ValueError('Camera prior exceeds its predeclared focal bounds')
    if not -.1 * width <= k[0, 2] <= 1.1 * width or not -.1 * height <= k[1, 2] <= 1.1 * height:
        raise ValueError('Camera prior has an unsupported principal point')
    return k


def select_training(frames):
    if not isinstance(frames, (list, tuple)) or not all(isinstance(frame, dict) for frame in frames):
        raise ValueError('Expected prepared source-frame records')
    training = [f for f in frames if f.get('split') == 'train']
    for frame in training:
        if (not isinstance(frame.get('id'), str) or not 1 <= len(frame['id']) <= 256
                or isinstance(frame.get('timestamp_s'), bool)
                or not isinstance(frame.get('timestamp_s'), (float, int))
                or not np.isfinite(frame['timestamp_s']) or frame['timestamp_s'] < 0):
            raise ValueError('TRAIN frames need explicit finite source time and identity')
    rows = sorted(training, key=lambda f: (f['timestamp_s'], f['id']))
    if len(rows) < POLICY['minimum_train_images']:
        raise ValueError('At least two distinct TRAIN images support a camera-prior consistency check')
    if len({row['id'] for row in rows}) != len(rows) or len({row.get('image_url') for row in rows}) != len(rows):
        raise ValueError('TRAIN image identities must be distinct')
    if len({row['timestamp_s'] for row in rows}) < POLICY['minimum_train_images']:
        raise ValueError('At least two distinct source presentation times are required')
    return [rows[i] for i in sorted({0, len(rows) // 2, len(rows) - 1})]


def combine(cameras, width, height):
    dimensions(width, height)
    values = np.asarray(cameras, float)
    if values.ndim != 3 or values.shape[1:] != (3, 3) or not POLICY['minimum_train_images'] <= len(values) <= POLICY['maximum_train_images']:
        raise ValueError('Camera prior image count is outside its bound')
    if not np.isfinite(values).all():
        raise ValueError('Each native camera must be finite')
    for camera in values:
        normalized = np.diag([1 / width, 1 / height, 1.]) @ camera.copy()
        normalized[0, 2] += .5 / width; normalized[1, 2] += .5 / height
        native_intrinsics(normalized, width, height)
    median = np.median(values, axis=0)
    # Reuse numerical checks with an exact inverse pixel-centre mapping.
    normalized = np.diag([1 / width, 1 / height, 1.]) @ median.copy()
    normalized[0, 2] += .5 / width; normalized[1, 2] += .5 / height
    native_intrinsics(normalized, width, height)
    focal_spread = float(np.max(np.ptp(values[:, [0, 1], [0, 1]], axis=0) / median[[0, 1], [0, 1]]))
    center_spread = float(np.max(np.ptp(values[:, :2, 2], axis=0)) / max(width, height))
    metrics = {'train_image_count': len(values), 'relative_focal_spread': focal_spread,
               'principal_point_spread_over_long_side': center_spread,
               'agreement_measures_physical_accuracy': False}
    if focal_spread > POLICY['maximum_relative_focal_spread'] or center_spread > POLICY['maximum_principal_point_spread_over_long_side']:
        raise ValueError('TRAIN-frame camera priors disagree; no camera assumption is returned')
    return median, metrics


class CameraPrior:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def describe(self):
        return describe()

    @staticmethod
    def _call_started(directory, frame, size):
        if frame.get('split') != 'train':
            raise ValueError('Only explicitly declared TRAIN images may enter camera inference')
        started = list(directory.glob('call-*-start.json'))
        if len(started) >= POLICY['maximum_train_images']:
            raise ValueError('The frozen camera inference call bound was reached')
        number = len(started) + 1
        write_once(directory / f'call-{number}-start.json', {
            'frame_id': frame['id'], 'split': frame['split'], 'input_size': list(size),
            'call_number': number, 'status': 'inference_invocation_start_recorded',
            'count_semantics': 'Conservative call-start receipt written immediately before model.infer; includes failed invocations'})
        print(json.dumps({'phase': 'camera_prior_frame', 'frame_id': frame['id'],
                          'count': number, 'maximum': POLICY['maximum_train_images']}), flush=True)

    def _predict(self, rows, directory, checkpoint):
        """One prepared MPS model, at most three forward calls; no network."""
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
            raise RuntimeError('CPU operator fallback must be disabled')
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
        import socket
        def no_network(*args, **kwargs):
            raise RuntimeError('Network is disabled in local camera inference')
        socket.socket.connect = no_network
        socket.create_connection = no_network
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError('Prepared MPS camera inference is unavailable; no fallback ran')
        from moge.model.v2 import MoGeModel
        import moge.model.v2 as model_code
        implementation = Path(model_code.__file__)
        regular_file(implementation, implementation.parent)
        implementation_bytes = implementation.read_bytes()
        implementation_hash = hashlib.sha256(implementation_bytes).hexdigest()
        write_once(directory / 'model-code.json', {'module_sha256': implementation_hash,
                   'torch_version': str(torch.__version__), 'checkpoint_sha256': sha(checkpoint)})
        write_bytes_once(directory / 'model-implementation.py', implementation_bytes)
        model = None
        predictions = []
        try:
            model = MoGeModel.from_pretrained(str(checkpoint)).to('mps').eval()
            for frame in rows:
                image = frame['image']; height, width = image.shape[:2]
                factor = min(1., POLICY['maximum_input_long_side'] / max(width, height))
                size = (max(1, round(width * factor)), max(1, round(height * factor)))
                resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                tensor = torch.from_numpy(resized[:, :, ::-1].copy()).permute(2, 0, 1).float().to('mps') / 255.
                self._call_started(directory, frame, size)
                with torch.inference_mode():
                    prediction = model.infer(tensor, num_tokens=POLICY['num_tokens'], use_fp16=False, apply_mask=True)
                    torch.mps.synchronize()
                    k = prediction['intrinsics'].cpu().numpy().astype(float)
                predictions.append({'frame_id': frame['id'], 'input_size': list(size), 'normalized_K': k.tolist()})
                write_once(directory / ('prediction-' + str(len(predictions)) + '.json'), predictions[-1])
                del prediction, tensor
        finally:
            if model is not None:
                del model
            torch.mps.empty_cache()
            write_once(directory / 'model-code-integrity.json', {'module_sha256_before': implementation_hash,
                'module_sha256_after': sha(implementation), 'unchanged': sha(implementation) == implementation_hash})
            if sha(implementation) != implementation_hash:
                raise ValueError('Loaded camera model implementation changed during inference')
        return predictions

    def estimate(self, frames, output):
        directory = Path(output) / 'camera-prior'
        if directory.resolve() != directory or not directory.is_relative_to(self.root / 'data'):
            raise ValueError('Use a new camera-prior directory inside project data')
        directory.mkdir(exist_ok=False)
        report = {'format': 'ordinary-video-camera-prior', 'version': 1, 'status': 'unsupported',
                  'policy': describe(), 'K': None, 'camera_calibrated': False, 'physical_scale_known': False,
                  'physical_accuracy_verified': False, 'new_api_calls': 0, 'actual_model_calls': 0,
                  'completed_model_calls': 0,
                  'model_call_count_semantics': 'Conservative immediately-before-inference call-start receipts, including failed invocations'}
        started = time.monotonic()
        snapshots, buffers = {}, []
        try:
            rows = select_training(frames)
            print(json.dumps({'phase': 'camera_prior_start', 'selected_train_images': len(rows),
                              'maximum': POLICY['maximum_train_images']}), flush=True)
            for frame in rows:
                image = frame.get('image')
                if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not all(image.shape[:2]):
                    raise ValueError('Native source images must be nonempty uint8 BGR arrays')
            dimensions = {tuple(f['image'].shape[:2]) for f in rows}
            if len(dimensions) != 1:
                raise ValueError('Mixed native resolutions cannot share one camera prior')
            height, width = next(iter(dimensions))
            bindings = []
            for frame in rows:
                path = source_file(self.root, frame['image_url'])
                if sha(path) != frame['image_sha256']:
                    raise ValueError('Source pixels changed before camera inference')
                decoded = cv2.imread(str(path))
                if decoded is None or not np.array_equal(decoded, frame['image']):
                    raise ValueError('Loaded pixels do not match the saved native source')
                snapshots[path] = frame['image_sha256']
                array_hash = hashlib.sha256(frame['image'].tobytes()).hexdigest()
                buffers.append((frame['id'], frame['image'], array_hash))
                bindings.append({**{k: frame[k] for k in ['id', 'timestamp_s', 'image_url', 'image_sha256', 'split']},
                                 'decoded_bgr_sha256': array_hash, 'native_size': [width, height]})
            checkpoint = self.root / 'models/moge-2-vits-normal/model.pt'
            provenance = checkpoint.with_name('provenance.json')
            regular_file(checkpoint, self.root / 'models')
            regular_file(provenance, self.root / 'models')
            provenance_bytes = provenance.read_bytes()
            provenance_hash = hashlib.sha256(provenance_bytes).hexdigest()
            model_record = json.loads(provenance_bytes)
            if sha(checkpoint) != model_record['checkpoint_sha256']:
                raise ValueError('Prepared checkpoint changed; downloads are disabled')
            if sha(__file__) != LOADED_SOURCE_SHA256:
                raise ValueError('Camera adapter code changed after this process imported it')
            snapshots.update({checkpoint: model_record['checkpoint_sha256'], provenance: provenance_hash,
                              Path(__file__).resolve(): LOADED_SOURCE_SHA256})
            protocol = {'format': 'ordinary-video-camera-prior-protocol', 'version': 1, 'policy': describe(),
                        'input_views': bindings, 'native_size': [width, height],
                        'checkpoint_sha256': model_record['checkpoint_sha256'],
                        'model_provenance_sha256': provenance_hash, 'implementation_sha256': LOADED_SOURCE_SHA256,
                        'frozen_before_inference': True, 'heldout_images_used': False}
            write_once(directory / 'protocol.json', protocol)
            write_bytes_once(directory / 'implementation.py', LOADED_SOURCE)
            write_bytes_once(directory / 'model-provenance.json', provenance_bytes)
            predictions = self._predict(rows, directory, checkpoint)
            report['actual_model_calls'] = len(list(directory.glob('call-*-start.json')))
            report['completed_model_calls'] = len(list(directory.glob('prediction-*.json')))
            if (report['actual_model_calls'] != len(rows) or report['completed_model_calls'] != len(predictions)
                    or report['completed_model_calls'] != len(rows)):
                raise ValueError('Inference receipts do not support the claimed model-call count')
            if [p['frame_id'] for p in predictions] != [f['id'] for f in rows]:
                raise ValueError('Camera predictions do not match the exact TRAIN inputs')
            for index, prediction in enumerate(predictions, 1):
                if json.loads((directory / f'prediction-{index}.json').read_text()) != prediction:
                    raise ValueError('A saved prediction receipt differs from its camera hypothesis')
            for binding in bindings:
                if sha(source_file(self.root, binding['image_url'])) != binding['image_sha256']:
                    raise ValueError('Source pixels changed during camera inference')
            if (sha(checkpoint) != protocol['checkpoint_sha256'] or sha(__file__) != protocol['implementation_sha256']
                    or sha(provenance) != protocol['model_provenance_sha256']):
                raise ValueError('Camera implementation, provenance or prepared checkpoint changed')
            cameras = [native_intrinsics(p['normalized_K'], width, height) for p in predictions]
            k, metrics = combine(cameras, width, height)
            report.update(status='unvalidated_learned_camera_assumption', K=k.tolist(), native_size=[width, height],
                          metrics=metrics, input_views=bindings, predictions=predictions,
                          protocol_sha256=sha(directory / 'protocol.json'),
                          geometry_must_be_reconstructed_with_this_assumption=True,
                          prior_geometry_or_poses_used=False, learned_depth_or_point_maps_used=False)
            return k, {'report_path': str((directory / 'camera-prior.json').relative_to(self.root)),
                       'kind': report['status'], 'policy': describe(),
                       'input_views': bindings, 'K': k.tolist()}
        except Exception as error:
            report['reason'] = str(error)[:500]
            # Partial forward-call receipts survive a later failure.
            report['actual_model_calls'] = len(list(directory.glob('call-*-start.json')))
            report['completed_model_calls'] = len(list(directory.glob('prediction-*.json')))
            raise
        finally:
            checks = []
            for path, expected in snapshots.items():
                try:
                    unchanged = path.resolve() == path and path.is_file() and sha(path) == expected
                except OSError:
                    unchanged = False
                checks.append({'path': str(path.relative_to(self.root)) if path.is_relative_to(self.root) else 'adapter_implementation',
                               'sha256_before': expected, 'unchanged': unchanged})
            buffer_checks = [{'frame_id': frame_id, 'unchanged': hashlib.sha256(image.tobytes()).hexdigest() == expected}
                             for frame_id, image, expected in buffers]
            integrity_ok = all(row['unchanged'] for row in checks + buffer_checks)
            report['input_integrity'] = {'files': checks, 'decoded_source_buffers': buffer_checks, 'unchanged': integrity_ok}
            for name in ['model-code.json', 'model-code-integrity.json']:
                path = directory / name
                if path.is_file() and path.resolve() == path:
                    report[name.removesuffix('.json').replace('-', '_')] = {'path': str(path.relative_to(self.root)), 'sha256': sha(path)}
            if not integrity_ok:
                report.update(status='unsupported', K=None, reason='A pinned source, decoded image, adapter, checkpoint or provenance changed during inference')
            report['elapsed_s'] = time.monotonic() - started
            if directory.resolve() != directory or not directory.is_relative_to(self.root / 'data'):
                raise ValueError('Camera output ancestry changed; no output was written outside its frozen location')
            write_once(directory / 'camera-prior.json', report)
            if not integrity_ok:
                raise ValueError(report['reason'])
