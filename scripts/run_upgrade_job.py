#!/usr/bin/env python3
"""Run a reviewed handoff inside an already provisioned GPU session.

No installation, download, authentication or scheduler submission is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

PINS = {'sam3d_objects': ('scripts/stage3_sam3d.py', 'f91db411c50efee93d8db7aeb323885650f6f722'),
        'sam31_image': ('scripts/sam31_segment.py', 'f1647e8d0c1d280991f4d1b7d974c2704537fe66')}


def command(job_path, mode, repo, config=None, checkpoint_manifest=None, output=None):
    job_path = Path(job_path).resolve()
    if mode not in {'preflight', 'run'}:
        raise ValueError('Choose an explicit preflight or run')
    job = json.loads(job_path.read_bytes())
    if job.get('format') != 'interaction-twin-upgrade-job' or job.get('version') != '1' or job.get('fixture') is not False:
        raise ValueError('Only a reviewed nonfixture handoff is supported')
    backend = job.get('backend')
    if backend not in PINS:
        raise ValueError('Unsupported backend')
    adapter, pin = PINS[backend]
    if job.get('expected_code_commit') != pin:
        raise ValueError('Handoff has an unsupported code pin')
    for record in job['files']:
        relative = PurePosixPath(record['path'])
        if relative.is_absolute() or any(p.startswith('.') for p in relative.parts) or '\\' in str(relative):
            raise ValueError('Unsafe handoff path')
        path = job_path.parent / relative
        if path.is_symlink() or not path.is_file() or path.resolve() != path or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Handoff bytes changed: ' + str(relative))
    destination = Path(output) if output else job_path.parent / ('output' if mode == 'run' else 'preflight.json')
    destination = destination.resolve()
    if destination.exists():
        raise ValueError('Choose a fresh output path; previous evidence is preserved')
    argv = [sys.executable, str(job_path.parent / adapter), mode,
            '--input-manifest', str(job_path.parent / 'input/input.json'),
            '--repo', str(Path(repo).resolve()), '--expected-code-commit', pin,
            '--output', str(destination)]
    if backend == 'sam3d_objects':
        if config is None or checkpoint_manifest is not None:
            raise ValueError('SAM 3D requires --config; use already provisioned pipeline.yaml')
        appearance = job.get('appearance_mode', 'vertex-color')
        if appearance not in {'vertex-color', 'uv-baked'}:
            raise ValueError('Invalid SAM 3D appearance mode')
        saved = job.get('command_argv', [])
        if '--appearance' not in saved or saved[saved.index('--appearance') + 1] != appearance:
            raise ValueError('Saved SAM 3D appearance command differs from the job')
        argv.extend(['--config', str(Path(config).resolve()), '--appearance', appearance])
    else:
        if checkpoint_manifest is None or config is not None:
            raise ValueError('SAM 3.1 requires --checkpoint-manifest')
        saved = job['command_argv']
        text, origin = saved[saved.index('--prompt') + 1], saved[saved.index('--prompt-origin') + 1]
        if not isinstance(text, str) or not 0 < len(text) <= 200 or origin not in {'cached_model_track_label', 'user_text_concept'}:
            raise ValueError('Invalid text concept provenance')
        argv.extend(['--checkpoint-manifest', str(Path(checkpoint_manifest).resolve()), '--prompt', text, '--prompt-origin', origin])
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight', 'run'])
    parser.add_argument('--job', type=Path, default=Path(__file__).resolve().parents[1] / 'job.json')
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--checkpoint-manifest', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    argv = command(args.job, args.mode, args.repo, args.config, args.checkpoint_manifest, args.output)
    result = subprocess.run(argv, cwd=args.job.resolve().parent, check=False)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
