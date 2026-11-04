"""Read-only inventory of explicitly known local runtimes; no installation.

Top-level module discovery and installed distribution metadata only. This script
does not import a simulator, open credentials, contact a service or start Unity.
It optionally cites an already-written synthetic Blender fixture receipt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {'mujoco': 'mujoco', 'pybullet': 'pybullet', 'pydrake': 'drake',
            'pychrono': 'pychrono', 'dm_control': 'dm-control', 'brax': 'brax',
            'warp': 'warp-lang', 'genesis': 'genesis-world', 'sapien': 'sapien',
            'pymunk': 'pymunk', 'pxr': 'usd-core', 'numpy': 'numpy',
            'scipy': 'scipy', 'trimesh': 'trimesh'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture-receipt', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Fresh output required')
    runtimes = [ROOT / env / 'bin/python' for env in ['.venv', '.venv-sam', '.venv-mcp']]
    runtimes.append(Path('/usr/bin/python3'))
    probe = '''import importlib.util, importlib.metadata, json,sys
packages=json.loads(sys.argv[1]); out={}
for module,distribution in packages.items():
    found=importlib.util.find_spec(module) is not None
    try: version=importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError: version=None
    out[module]={'discoverable':found,'distribution':distribution,'version':version}
print(json.dumps({'executable':sys.executable,'python':sys.version,'packages':out}))
'''
    rows = []
    for runtime in runtimes:
        if not runtime.is_file():
            rows.append({'runtime': str(runtime), 'status': 'not_found'})
            continue
        result = subprocess.run([str(runtime), '-c', probe, json.dumps(PACKAGES)],
                                capture_output=True, text=True, timeout=20, check=False)
        row = {'runtime': str(runtime), 'returncode': result.returncode}
        if result.returncode == 0:
            row['result'] = json.loads(result.stdout)
            row['status'] = 'inspected'
        else:
            row['status'] = 'probe_failed'
            row['stderr'] = result.stderr[:1000]
        rows.append(row)
    apps = {'blender': {'path': '/Applications/Blender.app/Contents/MacOS/Blender'},
            'unity_editors': {'path': '/Applications/Unity/Hub/Editor'}}
    for value in apps.values():
        value['exists'] = Path(value['path']).exists()
    if apps['unity_editors']['exists']:
        apps['unity_editors']['installed_directory_names'] = sorted(
            entry.name for entry in Path(apps['unity_editors']['path']).iterdir() if entry.is_dir())
        apps['unity_editors']['executed'] = False
        apps['unity_editors']['physics_capability_verified'] = False
    result = {'format': 'interaction-twin-local-physics-environment', 'version': 1,
              'utc': datetime.now(timezone.utc).isoformat(), 'runtimes': rows,
              'applications': apps,
              'path_executables': {name: shutil.which(name) for name in ['mujoco', 'mjpython', 'pybullet', 'blender']},
              'scope': 'Four explicit Python runtimes and two known installed application locations only; not an exhaustive disk search.',
              'installs': False, 'downloads': False, 'network': False,
              'script_sha256': digest(__file__)}
    if args.fixture_receipt:
        fixture = json.loads(args.fixture_receipt.read_text())
        result['blender_cpu_execution'] = {
            'receipt_path': str(args.fixture_receipt), 'receipt_sha256': digest(args.fixture_receipt),
            'engine': fixture['engine'], 'fixture_status': fixture['status'],
            'real_object_dynamics_validated': False,
            'execution_verified_by_saved_samples': len(fixture.get('samples', [])) > 0,
            'all_fixture_checks_passed': all(fixture['checks'].values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'runtimes': len(rows),
                      'blender_cpu_receipt_bound': bool(args.fixture_receipt)}))


if __name__ == '__main__':
    main()
