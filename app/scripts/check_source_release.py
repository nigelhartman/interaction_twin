#!/usr/bin/env python3
"""Check the public source without recordings, credentials or model inference.

Run from app/ with a Python environment prepared from requirements-review.txt.
An isolated disposable copy exercises the empty build and loopback server. No
uploads or model-processing endpoints are requested. Only the child server
started here is terminated.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.parse import unquote, urlsplit
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]
PYTHON_TESTS = (
    'tests.test_agent_memory',
    'tests.test_admin_workflow',
    'tests.test_library_query',
)
NODE_TESTS = (
    'tests/test_robot_workbench_ui_v2.cjs',
    'tests/test_aligned_scene_contract.cjs',
    'tests/test_aligned_scene_lifecycle.cjs',
    'tests/test_architecture_explainer.cjs',
    'tests/test_static_robot_fallback.cjs',
    'tests/test_static_pages_adapter.cjs',
)
ENTRY_DOCS = ('README.md', 'docs/source-setup.md', 'docs/PUBLIC_RELEASE.md',
              'docs/technical-overview-20261104.md', 'docs/mcp.md', 'docs/pipeline.md')


def run(args, *, cwd=ROOT):
    print('+ ' + ' '.join(str(v) for v in args), flush=True)
    subprocess.run(args, cwd=cwd, check=True, timeout=120)


def documentation_link_smoke():
    repository = ROOT.parent.resolve()
    checked = 0
    for name in ENTRY_DOCS:
        document = repository / name
        text = document.read_text()
        for match in re.finditer(r'!?\[[^\]]+\]\(([^)]+)\)', text):
            target = urlsplit(match.group(1))
            if target.scheme or target.netloc or not target.path:
                continue
            path = (document.parent / unquote(target.path)).resolve()
            if not path.is_relative_to(repository) or not path.exists():
                line = text[:match.start()].count('\n') + 1
                raise RuntimeError(f'Broken local documentation link: {name}:{line}: {target.path}')
            checked += 1
    print(json.dumps({'check': 'release_documentation_links', 'local_links': checked,
                      'external_network_requests': 0}), flush=True)


def empty_workspace_smoke():
    # Hidden directories (including credentials and environments) are excluded
    # by name before copytree traverses them. Saved artifacts are never inputs.
    excluded = shutil.ignore_patterns('.*', 'data', 'output', '__pycache__',
                                      'node_modules', '*.pyc', '*.sqlite*', '*.log')
    with tempfile.TemporaryDirectory(prefix='interaction-twin-source-review-') as temporary:
        isolated = Path(temporary) / 'app'
        shutil.copytree(ROOT, isolated, ignore=excluded)
        run([sys.executable, '-B', 'scripts/build_workspace.py', '--no-replays'], cwd=isolated)
        report = json.loads((isolated / 'workspace/build-report.json').read_text())
        if report['cases'] != 0 or report['errors']:
            raise RuntimeError('Expected an empty source-only catalog with no build errors')
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        command = [sys.executable, '-B', 'scripts/serve_workspace.py',
                   '--host', '127.0.0.1', '--port', str(port)]
        opener = build_opener(ProxyHandler({}))
        with tempfile.TemporaryFile() as log:
            child = subprocess.Popen(command, cwd=isolated, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 15
                while True:
                    if child.poll() is not None:
                        log.seek(0)
                        raise RuntimeError('Isolated source server exited:\n' + log.read().decode(errors='replace'))
                    try:
                        with opener.open(f'http://127.0.0.1:{port}/health', timeout=1) as response:
                            health = json.load(response)
                        break
                    except (URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise RuntimeError('Isolated source server did not become ready')
                        time.sleep(.1)
                with opener.open(f'http://127.0.0.1:{port}/showcase/index.html', timeout=3) as response:
                    if response.status != 200 or b'Interaction Twin' not in response.read():
                        raise RuntimeError('Source landing page was not served')
                print(json.dumps({'check': 'isolated_empty_workspace', 'cases': 0,
                                  'build_errors': [], 'health': health,
                                  'landing_http': 200, 'inference_requested': False}), flush=True)
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=5)


def main():
    if sys.version_info < (3, 12):
        raise SystemExit('Use Python 3.12 or newer; the release is checked with 3.12.')
    if shutil.which('node') is None:
        raise SystemExit('Node.js 22 or newer is needed for the browser-contract fixtures.')
    documentation_link_smoke()
    run([sys.executable, '-B', '-m', 'unittest', *PYTHON_TESTS])
    run(['node', '--test', *NODE_TESTS])
    empty_workspace_smoke()
    print('Source review passed: fixtures, empty catalog and isolated HTTP startup. '
          'No model inference or physical-accuracy claim.', flush=True)


if __name__ == '__main__':
    main()
