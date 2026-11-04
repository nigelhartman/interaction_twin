#!/usr/bin/env python3
"""CPU-only packaging of a completed pinned adapter run. Never runs a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import zipfile

MAX_FILE = 256 * 1024 * 1024
MAX_TOTAL = 512 * 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def pack(job_path, output, destination):
    job_path, output, destination = Path(job_path), Path(output), Path(destination)
    job = json.loads(job_path.read_bytes())
    if job.get('format') != 'interaction-twin-upgrade-job' or job.get('version') != '1':
        raise ValueError('Unsupported job manifest')
    if destination.exists():
        raise ValueError('Choose a fresh result ZIP; previous evidence is preserved')
    if output.is_symlink() or not output.is_dir():
        raise ValueError('Adapter output must be a real directory')
    backend = job['backend']
    fixed = {'metadata.json', 'source-rgb.png', 'overlay.png'} if backend == 'sam31_image' else {
        'metadata.json', 'object.glb', 'object-splat.ply', 'layout-and-pointmap.npz',
        'inputs/manifest.json', 'inputs/rgb.png', 'inputs/mask.png',
        'inputs/image-source.png', 'inputs/mask-source.png'}
    files = {}
    for path in sorted(output.rglob('*')):
        if path.is_dir():
            continue
        name = path.relative_to(output).as_posix()
        allowed = name in fixed or (backend == 'sam31_image' and re.fullmatch(r'instance-[0-9]{6}-(mask|rgba)\.png', name))
        if not allowed:
            raise ValueError('Unexpected output file; package only one adapter output directory: ' + name)
        if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size > MAX_FILE:
            raise ValueError('Unsafe or oversized output file: ' + name)
        files['raw/' + name] = path.read_bytes()
    if 'raw/metadata.json' not in files or sum(map(len, files.values())) > MAX_TOTAL:
        raise ValueError('Missing metadata or output exceeds package budget')
    metadata = json.loads(files['raw/metadata.json'])
    if metadata.get('learned_inference_executed') is not True:
        raise ValueError('The adapter did not report completed learned inference')
    receipt = {'format': 'interaction-twin-upgrade-result', 'version': '1',
               'job_id': job['job_id'], 'job_sha256': digest(job_path.read_bytes()),
               'backend': backend, 'fixture': False,
               'input_binding': job['input_binding'],
               'files': [{'path': name, 'bytes': len(raw), 'sha256': digest(raw)} for name, raw in files.items()],
               'inference_attestation': 'adapter metadata; not cryptographic proof of execution',
               'physical_accuracy_verified': False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('result.json', encode(receipt))
        for name, raw in files.items():
            archive.writestr(name, raw)
    return {'zip': str(destination.resolve()), 'files': len(files), 'job_id': job['job_id'],
            'learned_inference_attested_not_independently_verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--adapter-output', type=Path, required=True)
    parser.add_argument('--zip', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(pack(args.job, args.adapter_output, args.zip), indent=2))


if __name__ == '__main__':
    main()
