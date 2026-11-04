from contextlib import contextmanager, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from scripts import launch_capture_review as L
from scripts.restore_capture_review_runtime_20261104_v3 import restore_spec


def put(root, name, raw):
    path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    return {'path': name, 'sha256': L.sha(raw), 'bytes': len(raw)}


@contextmanager
def fixture_server(*, changed=False):
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder).resolve()
        item = put(root, 'output/review/index.html', b'<html>Frozen private review</html>')
        binary = put(root, 'data/review/model.glb', b'0123456789')
        spec = {'id': 'fixture-review', 'kind': 'batch', 'port': 1025, 'prefix': '/abcdefghijklmnopqrstuvwxyz/',
                'snapshot_ref': item, 'legacy_pid': 1234, 'legacy_command_script': 'output/old_server.py',
                'legacy_snapshot': 'output/review/snapshot.json',
                'routes': {'/abcdefghijklmnopqrstuvwxyz/': item, '/abcdefghijklmnopqrstuvwxyz/model.glb': binary}}
        server = ThreadingHTTPServer(('127.0.0.1', 0), L.handler(spec, root=root))
        spec['port'] = server.server_port
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        if changed:
            (root / item['path']).write_bytes(b'changed')
        try:
            yield root, spec
        finally:
            server.shutdown(); server.server_close(); worker.join(timeout=3)


class CaptureReviewLauncherTests(unittest.TestCase):
    def test_launcher_self_and_configuration_pins_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            put(root, L.CONFIG, (L.ROOT / L.CONFIG).read_bytes())
            pin = root / L.SELF_PIN; pin.write_text('0' * 64 + '\n')
            with self.assertRaisesRegex(ValueError, 'Launcher code changed'):
                L.load_configuration(root)
            pin.write_text(L.sha(Path(L.__file__).read_bytes()) + '\n')
            self.assertEqual(len(L.load_configuration(root)['viewers']), 3)
            (root / L.CONFIG).write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                L.load_configuration(root)

    def test_readiness_never_follows_redirects_or_sends_credentials(self):
        seen = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                seen.append((self.path, dict(self.headers)))
                self.send_response(302); self.send_header('Location', '/elsewhere'); self.send_header('Content-Length', '0'); self.end_headers()
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        try:
            spec = {'port': server.server_port, 'prefix': '/', 'routes': {'/': {'bytes': 2, 'sha256': L.sha(b'ok')}}}
            self.assertFalse(L.http_ready(spec))
            self.assertEqual([s[0] for s in seen], ['/'])
            self.assertFalse(any(k.lower() in {'authorization', 'cookie', 'proxy-authorization'} for _, headers in seen for k in headers))
        finally:
            server.shutdown(); server.server_close(); worker.join(timeout=3)

    def test_same_frozen_http_bytes_reuse_only_exact_owned_process(self):
        with fixture_server() as (root, spec):
            own = {'pid': 9000, 'uid': os.getuid(), 'argv': ['python', str(root / 'scripts/launch_capture_review.py'), '--serve', spec['id']]}
            with patch.object(L, 'listener_identity', return_value=own):
                state = L.preflight(spec, root=root)
            self.assertEqual(state['status'], 'reuse'); self.assertEqual(state['http_route_count'], 2)

    def test_occupied_foreign_port_is_never_killed_even_with_matching_bytes(self):
        with fixture_server() as (root, spec), patch.object(L, 'listener_identity', return_value={
                'pid': 9001, 'uid': os.getuid(), 'argv': ['python', '-m', 'http.server']}), \
                patch.object(L, 'http_ready', side_effect=AssertionError('Foreign process cannot be reused')), \
                patch.object(L.subprocess, 'Popen', side_effect=AssertionError('Must not start a replacement')):
            with self.assertRaisesRegex(ValueError, 'unrelated process'):
                L.preflight(spec, root=root)

    def test_different_uid_or_legacy_pid_or_command_cannot_be_reused(self):
        spec = {'id': 'fixture-review', 'legacy_pid': 1234, 'kind': 'batch', 'legacy_command_script': 'output/old_server.py',
                'legacy_snapshot': 'output/review/snapshot.json'}
        argv = ['python', 'output/old_server.py', 'serve', '--snapshot', 'output/review/snapshot.json', '--port', '0']
        self.assertTrue(L.owned_listener(spec, {'pid': 1234, 'uid': 44, 'argv': argv}, uid=44))
        for identity in [{'pid': 1235, 'uid': 44, 'argv': argv}, {'pid': 1234, 'uid': 45, 'argv': argv},
                         {'pid': 1234, 'uid': 44, 'argv': argv + ['--anything']}]:
            self.assertFalse(L.owned_listener(spec, identity, uid=44))

    def test_changed_served_bytes_fail_readiness_and_no_browser_or_spawn(self):
        with fixture_server(changed=True) as (root, spec):
            own = {'pid': 9000, 'uid': os.getuid(), 'argv': ['python', str(root / 'scripts/launch_capture_review.py'), '--serve', spec['id']]}
            with patch.object(L, 'listener_identity', return_value=own), patch.object(L.subprocess, 'Popen') as spawn, \
                    patch.object(L.webbrowser, 'open') as browser:
                with self.assertRaisesRegex(ValueError, 'HTTP readiness'):
                    L.preflight(spec, root=root)
                spawn.assert_not_called(); browser.assert_not_called()

    def test_no_proxy_or_auth_and_explicit_readonly_host_origin_range_guards(self):
        with fixture_server() as (root, spec), patch.dict(os.environ, {'http_proxy': 'http://127.0.0.1:1', 'HTTP_PROXY': 'http://127.0.0.1:1', 'NO_PROXY': '', 'no_proxy': ''}):
            self.assertTrue(L.http_ready(spec))
            opener = build_opener(ProxyHandler({})); base = f"http://127.0.0.1:{spec['port']}"
            with opener.open(Request(L.url(spec) + 'model.glb', headers={'Range': 'bytes=2-5'})) as response:
                self.assertEqual(response.status, 206); self.assertEqual(response.read(), b'2345')
                self.assertEqual(response.headers['Content-Range'], 'bytes 2-5/10')
            for path, headers, method, status in [(spec['prefix'], {'Host': 'evil.example'}, 'GET', 403),
                    (spec['prefix'], {'Origin': 'null'}, 'GET', 403), (spec['prefix'], {'Origin': 'https://evil.example'}, 'GET', 403),
                    (spec['prefix'] + '?query=1', {}, 'GET', 404), (spec['prefix'] + '%2e%2e/', {}, 'GET', 404),
                    ('/.secrets', {}, 'GET', 404), (spec['prefix'], {}, 'POST', 405)]:
                with self.subTest(path=path, headers=headers, method=method), self.assertRaises(HTTPError) as error:
                    opener.open(Request(base + path, headers=headers, method=method))
                self.assertEqual(error.exception.code, status); error.exception.close()

    def test_all_ports_preflight_before_any_start_or_browser(self):
        specs = [{'id': 'one'}, {'id': 'two'}]
        def preflight(spec, **kwargs):
            if spec['id'] == 'two': raise ValueError('foreign occupied port')
            return {'status': 'free'}
        with patch.object(L.subprocess, 'Popen') as spawn, patch.object(L.webbrowser, 'open') as browser:
            with self.assertRaisesRegex(ValueError, 'foreign'):
                L.reopen(specs, identity_probe=preflight)
            spawn.assert_not_called(); browser.assert_not_called()

    def test_no_open_reuse_writes_only_new_launch_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); put(root, L.CONFIG, b'fixed config')
            spec = {'id': 'fixture-review', 'port': 55555, 'prefix': '/', 'snapshot_ref': {'path': 'output/snapshot.json', 'sha256': 'a' * 64, 'bytes': 2}}
            browser_calls = []
            with redirect_stdout(StringIO()), patch.object(L.subprocess, 'Popen', side_effect=AssertionError('Reuse cannot spawn')):
                receipt = L.reopen([spec], root=root, no_open=True, identity_probe=lambda *a, **k: {'status': 'reuse', 'pid': 10}, browser=browser_calls.append)
            self.assertEqual(browser_calls, []); self.assertFalse(receipt['browser_open_requested'])
            self.assertFalse(receipt['main_app_or_worker_started']); self.assertEqual(receipt['new_api_calls'], 0)
            saved = list((root / L.RECEIPTS).glob('launch-*.json')); self.assertEqual(len(saved), 1)

    def test_started_child_readiness_failure_stops_only_that_child(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            spec = {'id': 'fixture-review', 'port': 55555, 'prefix': '/', 'routes': {}, 'legacy_pid': 1}
            calls = []; child = SimpleNamespace(pid=44, poll=lambda: None, terminate=lambda: calls.append('terminate'), wait=lambda **k: calls.append('wait'))
            identity = {'pid': 44, 'uid': os.getuid(), 'argv': ['python', str(root / 'scripts/launch_capture_review.py'), '--serve', 'fixture-review']}
            with patch.object(L, 'port_free', return_value=False), patch.object(L, 'listener_identity', return_value=identity), \
                    patch.object(L, 'http_ready', return_value=False):
                with self.assertRaisesRegex(ValueError, 'HTTP readiness'):
                    L.reopen([spec], root=root, no_open=True, identity_probe=lambda *a, **k: {'status': 'free'}, spawn=lambda *a, **k: child)
            self.assertEqual(calls, ['terminate', 'wait'])

    def test_exact_ref_detects_symlink_escape_hash_and_length_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); item = put(root, 'data/source.bin', b'original')
            self.assertEqual(L.checked(item, root), b'original')
            (root / item['path']).write_bytes(b'modified')
            with self.assertRaises(ValueError): L.checked(item, root)
            link = root / 'data/link.bin'; link.symlink_to(root / item['path'])
            with self.assertRaises(ValueError): L.checked({**item, 'path': 'data/link.bin'}, root)
            for name in ['../outside', '.secrets/key', 'data/../source.bin', '/data/file', 'data/./file']:
                with self.subTest(name=name), self.assertRaises(ValueError): L.checked({**item, 'path': name}, root)

    def test_pinned_actual_snapshot_code_and_source_bindings_validate_without_writers(self):
        if not (L.ROOT / L.CONFIG).is_file(): self.skipTest('Exact private configuration not included')
        config = json.loads((L.ROOT / L.CONFIG).read_bytes())
        self.assertEqual(L.sha((L.ROOT / L.CONFIG).read_bytes()), L.CONFIG_SHA)
        with patch.object(L.subprocess, 'Popen', side_effect=AssertionError('Preparation cannot spawn')), \
                patch.object(L.webbrowser, 'open', side_effect=AssertionError('Preparation cannot open browser')):
            specs = [L.prepare_spec(restore_spec(spec)) for spec in config['viewers']]
        self.assertEqual([s['port'] for s in specs], [51108, 65139, 50984])
        self.assertEqual([len(s['routes']) for s in specs], [147, 15, 22])


if __name__ == '__main__': unittest.main()
