import copy
import importlib.util
import json
import shutil
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import revisions as R
import stage5
from test_stage5 import fixture

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('serve_workspace_reviews', ROOT / 'scripts/serve_workspace.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
CASE = 'test-case--procedure-v1'


def published_fixture(root):
    document = fixture()
    folder = root / 'data/case'
    folder.mkdir(parents=True)
    (root / 'workspace').mkdir()
    (folder / 'source.mp4').write_bytes(b'fixture video bytes')
    (folder / 'proposal.json').write_text('{}')
    document['source'].update(path='data/case/source.mp4', sha256=stage5.sha(folder / 'source.mp4'), original_sha256=stage5.sha(folder / 'source.mp4'))
    document['motion'] = []
    rows = []
    for e in document['evidence']:
        image = folder / (e['id'] + '.png')
        image.write_bytes(('fixture pixels ' + e['id']).encode())
        e.update(frame_path=str(image.relative_to(root)), frame_sha256=stage5.sha(image))
        rows.append({'index': e['source_index'], 'timestamp_s': e['timestamp_s'], 'file': image.name})
    (folder / 'frames.json').write_text(json.dumps({'frames': rows}))
    (folder / 'manifest.json').write_text(json.dumps({'source': {'sha256': document['source']['sha256'], 'video': {'duration_s': 3.}}}))
    document['inputs'] = [{'id': key, 'path': 'data/case/' + name, 'sha256': stage5.sha(folder / name), 'kind': kind} for key, name, kind in [
        ('semantic_proposal', 'proposal.json', 'model_semantic_proposal'), ('frame_index', 'frames.json', 'source_frame_index'), ('manifest', 'manifest.json', 'source_manifest')]]
    source = folder / 'interaction.json'
    source.write_text(json.dumps(document))
    (root / 'workspace/cases.json').write_text(json.dumps({'cases': [{'id': CASE, 'interaction_path': 'data/case/interaction.json', 'original_sha256': stage5.sha(source)}]}))
    stage5.validate_interaction(document, root=root)
    return source, document


def edits(section=None, item=None, decision='accepted', patch_value=None):
    result = {s: {} for s in R.SECTIONS}
    if section:
        result[section][item] = {'decision': decision, 'note': 'Checked the source observation.', 'patch': patch_value or {}}
    return result


class ReviewContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source, self.original = published_fixture(self.root)
        self.original_bytes = self.source.read_bytes()
        self.store = R.Store(self.root)
        self.digest = stage5.sha(self.source)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, changes, parent=None, digest=None):
        return self.store.save(CASE, {'base_sha256': digest or self.digest, 'parent_revision_id': parent, 'edits': changes})

    def test_revision_is_immutable_and_preserves_original_geometry_identity_and_policy(self):
        changes = edits('steps', 'visible_opening', 'edited', {'title': 'Review the visible opening'})
        row = self.save(changes)
        self.assertEqual(self.source.read_bytes(), self.original_bytes)
        self.assertEqual(row['authorship'], 'user_review_via_local_workspace')
        self.assertEqual(row['document']['steps'][0]['title'], 'Review the visible opening')
        for key in ['assets', 'motion', 'source', 'tracklets', 'runtime_policy']:
            self.assertEqual(row['document'][key], self.original[key])
        self.assertTrue(row['document']['reasoning']['human_reviewed'])
        self.assertTrue(row['document']['reasoning']['manual_semantic_story_used'])
        self.assertIn('User-reviewed derivative', row['document']['limitations'][-1])
        self.assertEqual(self.store.get(CASE, row['revision_id']), row)
        self.assertEqual(self.store.list(CASE)['head_revision_id'], row['revision_id'])
        saved = self.root / 'data/user-revisions' / CASE / (row['revision_id'] + '.json')
        self.assertEqual(saved.stat().st_mode & 0o222, 0)
        first_bytes = saved.read_bytes()
        second = self.save(edits('steps', 'visible_opening'), parent=row['revision_id'])
        self.assertNotEqual(second['revision_id'], row['revision_id'])
        self.assertEqual(saved.read_bytes(), first_bytes)

    def test_rejected_point_state_becomes_unknown_without_implying_success(self):
        row = self.save(edits('assertions', 'open_view', 'rejected'))
        self.assertEqual(stage5.state_at(row['document'], 1.)['part_visual_state']['value'], 'unknown')
        self.assertEqual(row['replay']['steps'][0]['postconditions'][0]['result'], 'unknown')
        self.assertFalse(row['replay']['steps'][0]['physical_completion_verified'])
        self.assertEqual(stage5.state_at(row['document'], .5)['part_visual_state']['value'], 'unknown')

    def test_edited_point_state_updates_only_exact_conditions_and_is_human_inference(self):
        row = self.save(edits('assertions', 'closed_view', 'edited', {'value': 'open', 'rationale': 'The edge is visibly open.'}))
        a = row['document']['assertions'][0]
        self.assertEqual(a['status'], 'inferred')
        self.assertEqual(row['document']['steps'][0]['preconditions'][0]['value'], 'open')
        self.assertEqual(stage5.state_at(row['document'], .1)['part_visual_state']['value'], 'unknown')

    def test_unknown_and_existing_candidate_ids_remain_explicit(self):
        row = self.save(edits('assertions', 'closed_view', 'edited', {'value': 'unknown', 'rationale': 'Occluded'}))
        self.assertEqual(row['document']['assertions'][0]['status'], 'unknown')
        self.assertFalse(row['document']['runtime_policy']['physical_identity_confirmed'])
        self.assertIn('unverified', row['document']['tracklets'][0]['identity_status'])
        self.assertIn('not physical simulation', row['guide'])
        self.assertIn('POINT-SAMPLED VISUAL STATES', row['guide'])

    def test_status_only_unknown_cannot_keep_a_known_assertion(self):
        document, _ = R.apply_edits(self.original, edits('assertions', 'closed_view', 'edited', {'status': 'unknown'}), root=self.root)
        self.assertEqual(document['assertions'][0]['value'], 'unknown')
        self.assertEqual(document['assertions'][0]['status'], 'unknown')
        with self.assertRaisesRegex(ValueError, 'Unknown status'):
            R.apply_edits(self.original, edits('assertions', 'closed_view', 'edited', {'status': 'unknown', 'value': 'open'}), root=self.root)

    def test_parent_and_original_hash_prevent_stale_or_parallel_saves(self):
        first = self.save(edits('steps', 'visible_opening'))
        with self.assertRaisesRegex(R.Conflict, 'Another review'):
            self.save(edits('steps', 'visible_opening'))
        with self.assertRaisesRegex(R.Conflict, 'source program changed'):
            self.save(edits('steps', 'visible_opening'), parent=first['revision_id'], digest='f' * 64)
        self.source.write_text(json.dumps(self.original, indent=2))
        with self.assertRaisesRegex(R.Conflict, 'rebuild and reload'):
            self.store.list(CASE)

    def test_edit_contract_rejects_invented_refs_policy_or_illegal_values(self):
        cases = [edits('steps', 'invented'), edits('steps', 'visible_opening', 'edited', {'runtime_policy': {}}),
                 edits('steps', 'visible_opening', 'edited', {'evidence_ids': ['fake']}),
                 edits('relationships', 'candidate_part', 'edited', {'object_track_id': 'part'}),
                 edits('assertions', 'closed_view', 'edited', {'value': 'seated_and_physically_verified'}),
                 edits('assertions', 'closed_view', 'edited', {'status': 'verified'}),
                 edits('steps', 'visible_opening', 'edited', {'title': ''})]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.save(change)
        self.assertEqual(self.store.list(CASE)['revisions'], [])

    def test_actual_changed_evidence_prevents_save(self):
        (self.root / 'data/case/source_000001.png').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.save(edits('steps', 'visible_opening'))

    def test_saved_revision_refuses_changed_evidence_on_reload(self):
        row = self.save(edits('steps', 'visible_opening'))
        (self.root / 'data/case/source_000001.png').write_bytes(b'changed after review')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.store.get(CASE, row['revision_id'])

    def test_saved_guide_and_replay_are_recomputed_before_serving(self):
        row = self.save(edits('steps', 'visible_opening'))
        target = self.root / 'data/user-revisions' / CASE / (row['revision_id'] + '.json')
        target.chmod(0o600)
        for key, value in [('guide', 'Fabricated guide'), ('replay', {'simulation_ready': True})]:
            changed = copy.deepcopy(row); changed[key] = value
            target.write_text(json.dumps(changed))
            with self.subTest(export=key), self.assertRaisesRegex(ValueError, 'exports do not match'):
                self.store.get(CASE, row['revision_id'])

    def test_only_published_data_sources_and_confined_history_are_accessible(self):
        for case in ['../escape', 'not-published', 'x/y']:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.store.list(case)
        published = self.root / 'workspace/cases.json'
        for path in ['/etc/passwd', 'workspace/cases.json', 'data/../workspace/cases.json']:
            published.write_text(json.dumps({'cases': [{'id': CASE, 'interaction_path': path}]}))
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.store.list(CASE)
        (self.root / 'data/user-revisions').symlink_to(self.root / 'workspace')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.store._directory(CASE)

    def test_tampered_revision_or_unbounded_request_is_rejected(self):
        row = self.save(edits('steps', 'visible_opening'))
        target = self.root / 'data/user-revisions' / CASE / (row['revision_id'] + '.json')
        target.chmod(0o600)
        row['document']['summary'] = 'Fabricated replacement'
        target.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            self.store.get(CASE, row['revision_id'])
        with self.assertRaises(ValueError):
            self.save(edits('steps', 'visible_opening', 'edited', {'title': 'x' * 6001}), parent=row['revision_id'])


class ReviewHTTPContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source, _ = published_fixture(self.root)
        self.patcher = patch.object(S, 'ROOT', self.root)
        self.patcher.start()
        self.server = S.WorkspaceServer(('127.0.0.1', 0), S.Handler)
        from tests.owner_http_fixture import OwnerHTTPSession
        self.owner_http=OwnerHTTPSession(self.server);self.addCleanup(self.owner_http.close)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/api/cases/{CASE}/revisions'

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)
        self.patcher.stop(); self.temp.cleanup()

    def request(self, body=None, headers=None):
        request = urllib.request.Request(self.url, data=json.dumps(body).encode() if body is not None else None, headers=headers or {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request) as response:
            return response.status, json.load(response)

    def test_real_local_save_list_read_and_conflict(self):
        _, history = self.request()
        body = {'base_sha256': history['original_sha256'], 'parent_revision_id': None, 'edits': edits('steps', 'visible_opening')}
        code, row = self.request(body)
        self.assertEqual(code, 201)
        with urllib.request.urlopen(self.url + '/' + row['revision_id']) as response:
            self.assertEqual(json.load(response)['document']['steps'][0]['id'], 'visible_opening')
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request(body)
        self.assertEqual(error.exception.code, 409)
        self.assertEqual(len(self.request()[1]['revisions']), 1)

    def test_cross_origin_unknown_case_and_bounded_body(self):
        body = {'base_sha256': stage5.sha(self.source), 'parent_revision_id': None, 'edits': edits('steps', 'visible_opening')}
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request(body, {'Content-Type': 'application/json', 'Origin': 'https://evil.example'})
        self.assertEqual(error.exception.code, 403)
        self.url = self.url.replace(CASE, 'not-published')
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request()
        self.assertEqual(error.exception.code, 404)
        self.url = self.url.replace('not-published', CASE)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request({'padding': 'x' * (R.MAX_BODY + 1)})
        self.assertEqual(error.exception.code, 400)


@unittest.skipUnless(shutil.which('node'), 'Existing Node.js runtime is required')
class BrowserReviewContracts(unittest.TestCase):
    def scenario(self, name):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'fixture.json'
            path.write_text(json.dumps(fixture()))
            result = subprocess.run([shutil.which('node'), str(ROOT / 'tests/workspace_review_harness.cjs'), str(path), name], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_offline_human_review_and_evidence_export_preserve_original(self):
        self.scenario('offline_review')

    def test_pending_save_blocks_edits_and_updates_only_its_own_project(self):
        self.scenario('pending_save')

    def test_conflict_retains_exportable_unsaved_draft(self):
        self.scenario('conflict_keeps_draft')

    def test_unknown_state_and_invalid_evidence_are_not_promoted(self):
        self.scenario('certainty_and_evidence')

    def test_stale_revision_failure_cannot_replace_another_project_status(self):
        self.scenario('stale_revision_failure')


if __name__ == '__main__':
    unittest.main()
