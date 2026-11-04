"""Historical capture snapshots retain exact bytes through an additive mapping."""
import ast
import copy
import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener
import zipfile

from scripts import launch_capture_review as V1, launch_capture_review_v2 as V2
from scripts import launch_capture_review_v3 as V3, launch_capture_review_v4 as V4
from scripts import restore_capture_review_runtime_20261103 as Old
from scripts import restore_capture_review_runtime_20261104_v2 as A
from scripts import restore_capture_review_runtime_20261104_v3 as C

PRIVATE = (A.ROOT / A.MANIFEST).is_file()


@unittest.skipUnless(PRIVATE, 'Private historical capture archive is not included')
class RuntimeArchiveV2Tests(unittest.TestCase):
    def test_original_launchers_configs_and_snapshots_remain_byte_identical(self):
        pins = json.loads((A.ROOT / 'output/new-captures-20261104/capture-review-runtime-archive-v2/prior-file-pins.json').read_text())
        for name, pointer in pins.items():
            if name != 'Open Capture Review.command':
                with self.subTest(name=name):
                    self.assertEqual(hashlib.sha256(V1.checked(pointer)).hexdigest(), pointer['sha256'])
        for launcher in [V1, V2, V3]:
            raw = launcher.load_configuration()['viewers'][2]
            snapshot = json.loads(launcher.checked(raw['snapshot_ref']))
            with self.subTest(launcher=launcher.__name__), self.assertRaisesRegex(ValueError, 'Frozen review file'):
                launcher.checked(snapshot['routes']['/showcase/aligned-scene.js'])
            selected = C.restore_spec(A.restore_spec(raw))
            prepared = launcher.prepare_spec(selected)
            self.assertEqual(prepared['port'], raw['port'])
            self.assertEqual(prepared['routes']['/manifest.json'], snapshot['routes']['/manifest.json'])

    def test_all_four_sources_change_only_three_paths_with_identical_runtime_bytes(self):
        manifest = json.loads(V1.checked({'path': A.MANIFEST, 'sha256': A.MANIFEST_SHA, 'bytes': A.MANIFEST_BYTES}))
        self.assertEqual(len(manifest['snapshots']), 4)
        for record in manifest['snapshots']:
            source = json.loads(V1.checked(record['source_snapshot_ref']))
            restored = json.loads(V1.checked(record['restored_snapshot_ref']))
            changed = {url for url in source['routes'] if source['routes'][url] != restored['routes'][url]}
            self.assertEqual(changed, A.ALLOWED_URLS)
            expected = copy.deepcopy(source)
            for url in changed:
                old, new = source['routes'][url], restored['routes'][url]
                self.assertEqual(old['sha256'], new['sha256'])
                self.assertEqual(old['bytes'], new['bytes'])
                self.assertTrue(new['path'].startswith(A.ARCHIVE_PREFIX))
                self.assertEqual(hashlib.sha256(V1.checked(new)).hexdigest(), old['sha256'])
                expected['routes'][url] = new
            self.assertEqual(restored, expected)

    def test_exact_previously_verified_zip_members_are_the_archive_source(self):
        manifest = json.loads((A.ROOT / A.MANIFEST).read_text())
        raw = V1.checked(manifest['source_zip'])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), '96e0e6fb814b8063d650e724d6e1f9759440ff0f6529d6b9799383f719b07f5a')
        V1.checked(manifest['source_zip_build_receipt'])
        with zipfile.ZipFile(A.ROOT / manifest['source_zip']['path']) as archive:
            for row in manifest['archive_members']:
                self.assertEqual(archive.read(row['member']), V1.checked(row['extracted_ref']))

    def test_unmapped_or_changed_identity_never_triggers_an_implicit_fallback(self):
        spec = V3.load_configuration()['viewers'][2]
        for key, value in [('id', 'foreign'), ('port', 59001), ('kind', 'batch'), ('prefix', '/abcdefghijklmnopqrstuvwxyz/')]:
            changed = copy.deepcopy(spec)
            changed[key] = value
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, 'viewer identity changed'):
                A.restore_spec(changed)
        batch = V3.load_configuration()['viewers'][0]
        self.assertEqual(A.restore_spec(batch), batch)
        self.assertIsNot(A.restore_spec(batch), batch)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / A.MANIFEST
            path.parent.mkdir(parents=True)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'selection manifest changed'):
                A.restore_spec(spec, root=root)

    def test_v4_default_preserves_http_source_and_process_ownership_boundaries(self):
        config = V4.load_configuration()
        specs = [V4.prepare_spec(C.restore_spec(spec)) for spec in config['viewers']]
        self.assertEqual([s['port'] for s in specs], [52175, 65139, 52154])
        self.assertEqual([len(s['routes']) for s in specs], [147, 15, 23])
        def functions(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(Path(path).read_text()).body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        old, new = functions(V3.__file__), functions(V4.__file__)
        self.assertEqual(set(old), set(new))
        for name in set(old) - {'load_configuration', 'owned_listener', 'reopen'}:
            self.assertEqual(old[name], new[name], name)
        own = {'pid': 777, 'uid': 44, 'argv': ['python', str(V4.ROOT / 'scripts/launch_capture_review_v4.py'), '--serve', specs[2]['id']]}
        self.assertTrue(V4.owned_listener(specs[2], own, uid=44))
        own['argv'][1] = str(V4.ROOT / 'scripts/launch_capture_review_v3.py')
        self.assertFalse(V4.owned_listener(specs[2], own, uid=44))
        # The V4 command remains an exact archived historical artifact; V5 is the current entry point.
        html_manifest = json.loads(V1.checked({'path':C.MANIFEST,'sha256':C.MANIFEST_SHA,'bytes':C.MANIFEST_BYTES}))
        command = V1.checked(html_manifest['prior_command_ref']).decode()
        self.assertIn('launch_capture_review_v4.py', command)
        self.assertIn(hashlib.sha256(Path(V4.__file__).read_bytes()).hexdigest(), command)

    def test_isolated_http_serves_archived_javascript_and_rejects_private_or_mutated_routes(self):
        selection = V4.load_configuration()['viewers'][2]
        snapshot = json.loads(V4.checked(selection['snapshot_ref']))
        spec = {'id': 'runtime-archive-v2-fixture', 'routes': {url: snapshot['routes'][url] for url in A.ALLOWED_URLS}}
        server = ThreadingHTTPServer(('127.0.0.1', 0), V4.handler(spec))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            opener = build_opener(ProxyHandler({}))
            base = f'http://127.0.0.1:{server.server_port}'
            for url, pointer in spec['routes'].items():
                with opener.open(base + url) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.read(), V4.checked(pointer))
                    self.assertEqual(response.headers['X-Interaction-Twin-Review'], spec['id'] + ':' + V4.CONFIG_SHA)
            for route, headers, status in [('/.secrets', {}, 404), ('/admin/', {}, 404),
                                           ('/showcase/aligned-scene.js?x=1', {}, 404),
                                           ('/showcase/aligned-scene.js', {'Origin': 'null'}, 403)]:
                with self.subTest(route=route), self.assertRaises(HTTPError) as error:
                    opener.open(Request(base + route, headers=headers))
                self.assertEqual(error.exception.code, status)
                error.exception.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == '__main__':
    unittest.main()
