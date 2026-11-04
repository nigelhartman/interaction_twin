"""The local cached-revision path preserves source, lineage and queue isolation."""
import copy
import json
import time
import unittest
from unittest.mock import patch

import laptop_cached_revision_adoption_20261103 as C
from admin_workflow import AdminWorkflow
from object_program import ref
from tests import test_admin_workflow as F


class CachedRevisionAdoptionTests(unittest.TestCase):
    def setUp(self):
        fixture = F.AdminWorkflowTests('test_consent_queue_restart_four_review_roles_and_no_automatic_publication')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root = fixture.root
        self.store = fixture.store
        self.cid, self.base = fixture.ready()
        self.old_manifest = (self.root/self.base['manifest_ref']['path']).read_bytes()
        self.addCleanup(patch.stopall)
        # These tests exercise real evidence/GLB/workflow validation and revision
        # isolation; kinematic numerics have their own unmocked production QA.
        patch('object_program.validate', return_value=True).start()
        def file(name, value):
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
            return ref(path, self.root)
        self.file = file
        interaction = self.base['manifest']['interaction_ref']
        doc = json.loads((self.root/interaction['path']).read_text())
        scene = file('data/local/scene.json', {'original_program': interaction, 'source': doc['source'], 'components': []})
        source_scene = file('data/local/source-scene.json', {'original_program': interaction, 'source': doc['source'], 'components': []})
        mechanics = file('data/local/mechanics.json', {'joints': []})
        proposal = file('data/local/proposal.json', {'joints': []})
        archive = file('data/local/provenance/source.py.txt', b'exact historical implementation')
        companion = file('data/local/object-program.json', {'evidence_program': interaction, 'constrained_scene': scene,
            'source_scene': source_scene, 'mechanics': mechanics, 'proposal': proposal, 'joints': [],
            'unresolved_connections': [], 'component_assets': [], 'physical_accuracy_verified': False, 'simulation_ready': False,
            'implementation_provenance': [{'original_path': 'source.py', 'archived_file': archive}]})
        self.contract = {'candidate_id': self.cid, 'base_revision_id': self.base['id'], 'base_fingerprint': self.base['fingerprint'],
                         'source_ref': self.base['manifest']['source_ref'], 'interaction_ref': interaction,
                         'scene_ref': scene, 'object_program_ref': companion}

    def prepare(self):
        return C.build(root=self.root, contract=self.contract)

    def adopt(self, plan):
        return C.adopt(plan, root=self.root, contract=self.contract)

    def test_local_revision_is_new_private_and_retains_exact_originals_and_unrelated_queue(self):
        unrelated = self.store.submit_upload(run_id='unrelated_queue', input_path=self.fixture.source, processing_consent=True)
        plan = self.prepare()
        self.assertEqual(len(self.store.detail(self.cid)['revisions']), 1)
        with patch.object(AdminWorkflow, 'worker_once', side_effect=AssertionError('Never drain the general queue')):
            result = self.adopt(plan)
        detail = self.store.detail(self.cid)
        self.assertEqual(len(detail['revisions']), 2)
        revision = detail['revisions'][0]
        self.assertEqual(revision['id'], result['revision_id'])
        self.assertEqual(revision['manifest']['base_revision_id'], self.base['id'])
        self.assertEqual(revision['manifest']['source_ref'], self.base['manifest']['source_ref'])
        self.assertEqual(revision['manifest']['interaction_ref'], self.base['manifest']['interaction_ref'])
        self.assertEqual(revision['manifest']['review']['blocking_failures'], 0)
        self.assertEqual(revision['manifest']['review']['hosted_roles'], [])
        self.assertEqual((self.root/self.base['manifest_ref']['path']).read_bytes(), self.old_manifest)
        self.assertEqual(detail['decisions'], [])
        self.assertEqual(self.store.public_index()['records'], [])
        self.assertEqual(next(job for job in self.store.snapshot()['jobs'] if job['id'] == unrelated['job']['id'])['state'], 'queued')
        self.assertIn('data/local/provenance/source.py.txt', [a['path'] for a in revision['manifest']['artifacts']])

    def test_stale_base_cannot_create_a_third_revision(self):
        plan = self.prepare()
        self.adopt(plan)
        with self.assertRaisesRegex(ValueError, 'current base revision changed'):
            self.adopt(plan)
        self.assertEqual(len(self.store.detail(self.cid)['revisions']), 2)

    def test_changed_source_is_rejected_before_job_creation(self):
        plan = self.prepare()
        before = self.store.snapshot()['jobs']
        (self.root/self.contract['source_ref']['path']).write_bytes(b'changed recording')
        with self.assertRaisesRegex(ValueError, 'missing or changed'):
            self.adopt(plan)
        self.assertEqual(before, self.store.snapshot()['jobs'])
        self.assertEqual(len(self.store.detail(self.cid)['revisions']), 1)

    def test_running_other_job_is_not_leased_interrupted_or_finalized(self):
        plan = self.prepare()
        unrelated = self.store.submit_upload(run_id='running_other', input_path=self.fixture.source, processing_consent=True)
        expiry = time.time()+100
        with self.store.db() as db:
            db.execute("UPDATE jobs SET state='running',lease_token='unrelated-lease',lease_until=? WHERE id=?", (expiry, unrelated['job']['id']))
        with self.assertRaisesRegex(ValueError, 'Another owner job is running'):
            self.adopt(plan)
        with self.store.db() as db:
            row = db.execute('SELECT state,lease_token,lease_until FROM jobs WHERE id=?', (unrelated['job']['id'],)).fetchone()
            self.assertEqual(tuple(row), ('running', 'unrelated-lease', expiry))
            self.assertEqual(db.execute("SELECT COUNT(*) FROM jobs WHERE kind='cached_local_revision'").fetchone()[0], 0)

    def test_tampered_plan_and_failed_finalization_cannot_retry(self):
        plan = self.prepare()
        changed = copy.deepcopy(plan)
        changed['result']['new_api_calls'] = 1
        with self.assertRaisesRegex(ValueError, 'Prepared cached revision changed'):
            self.adopt(changed)
        with patch.object(AdminWorkflow, '_finalize_revision', side_effect=ValueError('Fixture failure')) as finalize:
            with self.assertRaisesRegex(ValueError, 'Fixture failure'):
                self.adopt(plan)
            with self.assertRaisesRegex(ValueError, 'already attempted'):
                self.adopt(plan)
            self.assertEqual(finalize.call_count, 1)
        self.assertEqual(self.store.detail(self.cid)['candidate']['current_revision_id'], self.base['id'])


if __name__ == '__main__':
    unittest.main()
