"""Small deterministic preview security/range/query contracts; no inference."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from judge_preview import preview_server as P


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.rows = []
        self.asset = 'data/public-example/source/video.mp4'
        self.html = 'data/public-example/review/index.html'
        self.add('preview.html', b'<h1>Read-only preview</h1>')
        self.add(self.asset, b'0123456789')
        self.add(self.html, b'<script src="owner.js"></script>')
        assets = [{k: row[k] for k in ['path', 'sha256', 'bytes']} for row in self.rows if row['path'].startswith('data/')]
        self.add_json('showcase/public/public-assets.json', {'assets': assets})
        record = {'id': 'public', 'title': 'Public twin', 'summary': 'A saved example.',
                  'source': {'sha256': 'a' * 64, 'original_sha256': 'b' * 64},
                  'components': [], 'relationships': [], 'steps': [], 'counts': {}, 'task_classes': []}
        raw = json.dumps({'records': [record]}).encode()
        digest = hashlib.sha256(raw).hexdigest()
        self.catalog = 'showcase/public/library-catalog-' + digest + '.json'
        self.add(self.catalog, raw)
        self.add_json('showcase/public/library-index.json', {'catalog': {'path': Path(self.catalog).name, 'sha256': digest, 'bytes': len(raw)}})
        self.add_json('showcase/public/cases-index.json', {'curated': True, 'cases': [{'id': 'public'}]})
        self.freeze()

    def add(self, relative, raw, http=True):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        self.rows.append({'path': relative, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw), 'http': http})

    def add_json(self, relative, value):
        self.add(relative, json.dumps(value).encode())

    def replace_json(self, relative, value):
        raw = json.dumps(value).encode()
        (self.root / relative).write_bytes(raw)
        row = next(row for row in self.rows if row['path'] == relative)
        row.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))

    def freeze(self):
        (self.root / 'bundle-manifest.json').write_text(json.dumps({'format': P.FORMAT, 'version': 1,
            'read_only': True, 'inference_available': False, 'mcp_runtime_included': False, 'files': self.rows}))

    def test_exact_current_data_and_same_query_contract(self):
        snapshot = P.Snapshot(self.root)
        self.assertEqual(snapshot.route('/files/' + self.asset)[1], b'0123456789')
        self.assertEqual(snapshot.query({'query': ['saved']})['records'][0]['id'], 'public')
        self.assertTrue(snapshot.query({}, facets=True)['curated_only'])
        self.assertEqual(snapshot.query({'query': ['missing']})['total'], 0)

    def test_payload_budget_rejects_oversize_before_reading_any_member(self):
        # Synthetic declared lengths exercise the aggregate ZIP-bomb guard
        # without allocating large files or weakening per-member validation.
        self.rows = [{'path': f'data/fixture/{i}.glb', 'sha256': '0' * 64,
                      'bytes': size, 'http': True}
                     for i, size in enumerate([128 * 1024 * 1024] * 6)]
        self.freeze()
        with patch.object(P, 'checked_bytes', side_effect=RuntimeError('member check reached')) as check:
            with self.assertRaisesRegex(RuntimeError, 'member check reached'):
                P.Snapshot(self.root)
            check.assert_called_once()
        self.rows[-1]['bytes'] += 1
        self.freeze()
        with patch.object(P, 'checked_bytes') as check:
            with self.assertRaisesRegex(ValueError, 'reviewed total byte budget'):
                P.Snapshot(self.root)
            check.assert_not_called()

    def test_owner_private_raw_directory_and_double_encoded_routes_are_absent(self):
        snapshot = P.Snapshot(self.root)
        for route in ['/admin', '/api/upload', '/workspace/index.html', '/.private/memory',
                      '/files/data/../private.json', '/files/data/%2e%2e/private.json',
                      '/files/data/%252e%252e/private.json', '/data/public-example/source/video.mp4',
                      '/preview_server.py', '//outside.invalid/file', '/files/' + self.asset + '/extra']:
            with self.assertRaises(ValueError, msg=route):
                snapshot.route(route)

    def test_source_tampering_after_startup_fails_closed(self):
        snapshot = P.Snapshot(self.root)
        (self.root / self.asset).write_bytes(b'ABCDEFGHIJ')
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            snapshot.route('/files/' + self.asset)

    def test_unapproved_data_cannot_enter_served_manifest(self):
        self.add('data/private-working/model.glb', b'not public')
        self.freeze()
        with self.assertRaisesRegex(ValueError, 'current-curated'):
            P.Snapshot(self.root)

    def test_workspace_owner_page_cannot_become_a_route(self):
        self.add('workspace/index.html', b'owner')
        self.freeze()
        with self.assertRaisesRegex(ValueError, 'Owner/private'):
            P.Snapshot(self.root)

    def test_symlink_and_catalog_pointer_substitution_fail(self):
        path = self.root / self.asset
        path.unlink()
        other = self.root / 'outside-video'
        other.write_bytes(b'0123456789')
        path.symlink_to(other)
        with self.assertRaises(ValueError):
            P.Snapshot(self.root)

    def test_byte_ranges_preserve_exact_bytes_and_reject_multiple_or_empty(self):
        self.assertEqual(P.byte_range('bytes=0-3', 10), (0, 3, True))
        self.assertEqual(P.byte_range('bytes=4-', 10), (4, 9, True))
        self.assertEqual(P.byte_range('bytes=-3', 10), (7, 9, True))
        self.assertEqual(P.byte_range(None, 10), (0, 9, False))
        for value in ['bytes=0-1,3-4', 'bytes=10-', 'bytes=-0', 'bytes=-', 'bytes=6-3']:
            with self.assertRaises(ValueError):
                P.byte_range(value, 10)

    def test_unknown_oversized_or_duplicate_scalar_filters_do_not_expand_scope(self):
        snapshot = P.Snapshot(self.root)
        for value in [{'path': ['data/private']}, {'limit': ['1000']}, {'limit': ['1', '2']}, {'sort': ['accuracy']},
                      {'parts': ['']}, {'query': ['x' * 201]}]:
            with self.assertRaises(ValueError):
                snapshot.query(value)

    def robotics_fixture(self):
        """Transport-only mathematical fixture; never presented as model output."""
        asset = next(r for r in self.rows if r['path'] == self.asset)
        reference = {k: asset[k] for k in ['path', 'sha256', 'bytes']}
        self.add_json('showcase/demos/robotics/robot-kit.json', {'id': 'fixture_kit', 'title': 'Fixture arm'})
        self.add_json('showcase/demos/robotics/jabra/task.json', {
            'status': 'ready', 'twin_id': 'public', 'kit_id': 'fixture_kit',
            'claims': {'authored_rehearsal': True, 'physical_accuracy_verified': False, 'simulation_ready': False},
            'source_binding': {'program_ref': reference, 'object_program_ref': None}, 'objects': [{'asset_ref': reference}]})
        self.add('showcase/demos/robotics/jabra/starter.zip', b'transport-only-zip-fixture')
        by_name = {r['path']: {k: r[k] for k in ['path', 'sha256', 'bytes']} for r in self.rows}
        self.add_json('showcase/demos/robotics/index.json', {
            'format': 'interaction-twin-frozen-robotics-index', 'version': 1,
            'scope': 'authored_kinematic_snapshot', 'inference_available': False,
            'physical_accuracy_verified': False, 'simulation_ready': False,
            'kit_id': 'fixture_kit', 'kit': by_name['showcase/demos/robotics/robot-kit.json'],
            'tasks': [{'twin_id': 'public', 'title': 'Fixture twin', 'instruction': 'Fixture bounded instruction.',
                       'task': by_name['showcase/demos/robotics/jabra/task.json'],
                       'starter': by_name['showcase/demos/robotics/jabra/starter.zip']}]})
        self.freeze()

    def test_frozen_robotics_only_retrieves_declared_kit_baseline_and_starter(self):
        self.robotics_fixture()
        snapshot = P.Snapshot(self.root)
        kit, path = snapshot.robotics_response('/api/robotics/kit', {'kit_id': ['fixture_kit']})
        self.assertTrue(kit['snapshot_mode']); self.assertIsNone(path)
        query = {'kit_id': ['fixture_kit'], 'twin_id': ['public'], 'instruction': ['Fixture bounded instruction.']}
        task, path = snapshot.robotics_response('/api/robotics/plan', query)
        self.assertEqual(task['twin_id'], 'public'); self.assertIsNone(path)
        task, path = snapshot.robotics_response('/api/robotics/bundle', query)
        self.assertIsNone(task); self.assertEqual(path, 'showcase/demos/robotics/jabra/starter.zip')
        self.assertEqual(snapshot.route('/' + path)[1], b'transport-only-zip-fixture')
        for changed in [{**query, 'instruction': ['An invented new server-planned task']},
                        {**query, 'twin_id': ['unapproved']}, {**query, 'path': ['.private/state']},
                        {**query, 'kit_id': ['fixture_kit', 'other']}]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                snapshot.robotics_response('/api/robotics/plan', changed)
        with self.assertRaises(ValueError): snapshot.robotics_response('/api/robotics/execute', query)

    def test_snapshot_without_robotics_does_not_invent_a_planner(self):
        with self.assertRaisesRegex(ValueError, 'No frozen robotics'):
            P.Snapshot(self.root).robotics_response('/api/robotics/kits', {})

    def test_current_brand_demo_and_workbench_scopes_are_narrowly_allowed(self):
        for path in ['showcase/brand/v2/favicon-32.png', 'showcase/robot-workbench.html',
                     'showcase/demos/charge-phone-v1/index.html', 'showcase/demos/charge-phone-v1/trace.json',
                     'showcase/demos/robotics/wallet/starter.zip','showcase/physics-bench.html',P.PHYSICS_STARTER,
                     'showcase/trailers.html', *sorted(P.TRAILER_TRANSCRIPTS)]:
            self.assertTrue(P.static_scope(path), path)
        for path in ['showcase/demos/unknown/index.html', 'showcase/demos/robotics/private/task.json',
                     'showcase/demos/charge-phone-v1/owner.json', 'showcase/credentials.json','showcase/media/unapproved.zip',
                     'showcase/media/trailer-collection-20261104-v1/07-owner-transcript.txt',
                     'showcase/media/trailer-collection-20261104-v1/01-origin.py.txt',
                     'showcase/media/trailer-collection-20261104-v2/01-origin-transcript.txt',
                     'showcase/media/owner-transcript.txt']:
            self.assertFalse(P.static_scope(path), path)

    def test_gallery_and_only_six_exact_transcripts_are_frozen_inert_downloads(self):
        self.add('showcase/trailers.html', b'<h1>Published films</h1>')
        for path in P.TRAILER_TRANSCRIPTS:
            self.add(path, ('Published transcript: ' + Path(path).name).encode())
        self.freeze()
        snapshot = P.Snapshot(self.root)
        self.assertEqual(snapshot.route('/showcase/trailers.html')[1], b'<h1>Published films</h1>')
        for path in P.TRAILER_TRANSCRIPTS:
            relative, raw = snapshot.route('/' + path)
            self.assertEqual(relative, path)
            self.assertTrue(raw.startswith(b'Published transcript: '))
            self.assertEqual(P.mimetypes.guess_type(relative)[0], 'text/plain')
        # A sibling is not enabled merely because a valid transcript exists.
        self.add('showcase/media/trailer-collection-20261104-v1/owner-feedback.txt', b'private')
        self.freeze()
        with self.assertRaisesRegex(ValueError, 'Owner/private'):
            P.Snapshot(self.root)

    def test_private_named_current_data_requires_exact_public_admission_and_never_expands_siblings(self):
        # Some approved camera/wallet JSON keeps its historical private-named
        # path. The existing exact public inventory admits bytes, not folders.
        path = 'data/private-native-programs/fixture/interaction.json'
        raw = json.dumps({'source_sha256': 'a' * 64, 'fixture_only': True}).encode()
        self.add(path, raw)
        assets_path = 'showcase/public/public-assets.json'
        assets = json.loads((self.root / assets_path).read_bytes())
        admitted = {k: self.rows[-1][k] for k in ['path', 'sha256', 'bytes']}
        assets['assets'].append(admitted)
        self.replace_json(assets_path, assets)
        self.freeze()
        snapshot = P.Snapshot(self.root)
        self.assertEqual(snapshot.route('/files/' + path)[1], raw)
        with self.assertRaises(ValueError):
            snapshot.route('/files/data/private-native-programs/fixture/owner.json')
        # Hash-valid manifest bytes that differ from the exact curated ref fail.
        changed = json.dumps({'source_sha256': 'b' * 64, 'fixture_only': True}).encode()
        (self.root / path).write_bytes(changed)
        row = next(row for row in self.rows if row['path'] == path)
        row.update(sha256=hashlib.sha256(changed).hexdigest(), bytes=len(changed))
        self.freeze()
        with self.assertRaisesRegex(ValueError, 'current-curated'):
            P.Snapshot(self.root)

    def test_hash_consistent_frozen_task_cannot_claim_physics_or_reference_unapproved_data(self):
        self.robotics_fixture()
        task_name = 'showcase/demos/robotics/jabra/task.json'
        index_name = 'showcase/demos/robotics/index.json'
        original_task = (self.root / task_name).read_bytes()
        original_index = (self.root / index_name).read_bytes()
        for mode in ['physics', 'foreign']:
            task = json.loads(original_task)
            if mode == 'physics': task['claims']['simulation_ready'] = True
            else: task['objects'][0]['asset_ref']['path'] = 'data/unapproved/source.mp4'
            raw = json.dumps(task).encode(); (self.root / task_name).write_bytes(raw)
            task_ref = {'path': task_name, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
            index = json.loads(original_index); index['tasks'][0]['task'] = task_ref
            index_raw = json.dumps(index).encode(); (self.root / index_name).write_bytes(index_raw)
            for row in self.rows:
                if row['path'] == task_name: row.update(task_ref)
                elif row['path'] == index_name: row.update(sha256=hashlib.sha256(index_raw).hexdigest(), bytes=len(index_raw))
            self.freeze()
            with self.subTest(mode=mode), self.assertRaises(ValueError): P.Snapshot(self.root)


if __name__ == '__main__':
    unittest.main()
