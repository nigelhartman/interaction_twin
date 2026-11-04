"""Exact owner approval/publication gates for declared geometric derivatives.

Small mathematical fixtures replace heavy geometry validation only. Durable
workflow approval, reference hashing, publication routing and public reads are
the real implementation; no production credentials/state or models are used.
"""
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

from admin_workflow import AdminWorkflow
import curated_publication as C
from curated_twin_catalog import CuratedTwinCatalog
from scripts import serve_workspace as S
from tests import test_admin_workflow as F


class DeclaredAssemblyPublicationTests(unittest.TestCase):
    def setUp(self):
        fixture = F.AdminWorkflowTests('test_consent_queue_restart_four_review_roles_and_no_automatic_publication')
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.fixture, self.root, self.store = fixture, fixture.root, fixture.store
        self.row = copy.deepcopy(fixture.fixture.row)
        self.row['scene_ref'] = None
        self.file('workspace/cases-index.json', {'cases': [self.row]})
        self.program = C.ref(self.root / self.row['interaction_path'], self.root)
        self.native = C.ref(self.root / 'data/case/asset.glb', self.root)
        self.metadata = self.file('data/case/metadata.json', {'fixture': 'native-metadata'})
        self.derived_asset = self.file('data/declared/asset.glb', (self.root / self.native['path']).read_bytes())
        component = lambda asset: {'id': 'part', 'asset': asset, 'native_metadata': self.metadata, 'poses': []}
        self.native_scene = self.file('data/declared/native-scene.json', {'original_program': self.program,
            'source': fixture.fixture.doc['source'], 'components': [component(self.native)]})
        self.keys = {k: self.file('data/declared/' + k + '.json', {'fixture': k}) for k in ['source_scene', 'mechanics', 'proposal']}
        self.base_doc = {'format': 'interaction-twin-object-program', 'version': '0.2',
            'evidence_program': self.program, 'source': fixture.fixture.doc['source'],
            **self.keys, 'constrained_scene': self.native_scene,
            'component_assets': [{'component_id': 'part', 'native_glb': self.native, 'native_metadata': self.metadata}],
            'joints': [], 'unresolved_connections': [], 'removable_relationships': [], 'implementation_provenance': []}
        self.base = self.file('data/declared/native-program.json', self.base_doc)
        self.scene = self.file('data/declared/derived-scene.json', {'original_program': self.program,
            'source': fixture.fixture.doc['source'], 'components': [component(self.derived_asset)]})
        self.report = self.file('data/declared/refinement-report.json', {'fixture': 'report',
            'original_scene': self.native_scene, 'original_program': self.program})
        self.audit = self.file('data/declared/source-audit.json', {'fixture': 'audit',
            'policy': {'cavity_facing_ambiguity_gate': 'withhold_bracketed_open_when_parent_cavity_faces_camera_and_child_cavity_faces_away'}})
        self.declared_doc = {'format': 'interaction-twin-declared-derived-assembly', 'version': 1,
            'evidence_program': self.program, 'source': fixture.fixture.doc['source'],
            'base_object_program_ref': self.base, 'refinement_report_ref': self.report,
            'scene_ref': self.scene, 'source_alignment_audit_ref': self.audit,
            'source_audit_summary': {'default_eligible': True},
            'component_assets': [{'component_id': 'part', 'native_glb': self.derived_asset,
                'original_native_glb': self.native, 'native_metadata': self.metadata}],
            'physical_accuracy_verified': False, 'simulation_ready': False}
        self.declared = self.file('data/declared/derived-assembly.json', self.declared_doc)
        self.record = {'id': self.row['id'], 'title': 'Declared mathematical fixture',
            'summary': fixture.fixture.doc['summary'], 'source': fixture.fixture.doc['source'],
            'interaction': self.program, 'components': [{'id': 'part', 'label': 'part',
                'asset': {'track_id': 'part', **self.native, 'metadata': self.metadata}}],
            'steps': [], 'relationships': [], 'sam3d_reviews': [], 'task_classes': []}
        self.addCleanup(patch.stopall)
        # Geometry/schema behavior has its own actual-data suite. These fixtures
        # preserve source/ref/approval routing while avoiding synthetic claims.
        patch('object_program.validate', return_value=None).start()
        patch('refined_assembly_contract.validate', return_value=None).start()
        # The lightweight curated reader now validates the same declared lineage
        # independently. This routing fixture deliberately lacks the real fitted
        # geometry/clock document; that validator has its own real-data suite.
        # Approval fingerprints, approved artifact allowlists and byte/hash reads
        # remain real here, including the reader's confinement callback.
        patch('declared_assembly_view.validate', return_value=None).start()
        patch('scripts.build_robotics_library.build_record', side_effect=lambda *a, **k: copy.deepcopy(self.record)).start()

    def file(self, relative, value):
        path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        return C.ref(path, self.root)

    def approve(self, strategy='declared_source_mask_refit_assembly_adoption_v1', *, omit_binding=False, duplicate=False):
        submitted = self.store.submit_upload(run_id='fixture-explicit-adoption', input_path=self.fixture.source, processing_consent=True)
        execution = {'strategy': strategy, 'derived_assembly_ref': self.declared}
        artifacts = [self.program, self.native, self.metadata, self.base, self.native_scene, self.scene, self.report, self.audit]
        if not omit_binding: artifacts.append(self.declared)
        result = {'status': 'complete_with_gaps', 'run_id': 'fixture-retained-adoption',
            'interaction_ref': self.program, 'object_program_ref': self.base, 'scene_ref': self.native_scene,
            'artifacts': artifacts, 'repair_execution': [execution] * (2 if duplicate else 1),
            'new_api_calls': 0, 'new_gpu_calls': 0}
        done = self.store.worker_once(lambda job: copy.deepcopy(result))
        self.assertEqual(done['status'], 'complete')
        detail = self.store.detail(submitted['candidate']['id']); revision = detail['revisions'][0]
        self.assertEqual(revision['manifest']['review']['blocking_failures'], 0)
        decision = self.store.approve(submitted['candidate']['id'], revision['id'], revision['fingerprint'])
        self.assertFalse(decision['decision']['physical_accuracy_verified'])
        return decision

    def test_only_explicit_exact_approved_strategy_selects_derived_default_and_keeps_native_kinematics(self):
        self.approve()
        selected = C._approved_records(self.root)[0]
        self.assertEqual(selected['aligned']['manifest'], self.scene)
        self.assertEqual(selected['aligned']['derived_assembly'], self.declared)
        self.assertEqual(selected['aligned']['object_program'], self.base)
        self.assertEqual(selected['record']['components'][0]['asset']['sha256'], self.native['sha256'])
        C.build(root=self.root)
        catalog = CuratedTwinCatalog(self.root)
        derived = catalog.derived_assembly(self.row['id'])
        self.assertEqual(derived['artifact'], self.declared)
        self.assertEqual(derived['document']['scene_ref'], self.scene)
        kinematics = catalog.kinematics(self.row['id'])
        self.assertEqual(kinematics['artifact']['sha256'], self.base['sha256'])
        self.assertEqual(kinematics['object_program']['constrained_scene'], self.native_scene)
        for binding in [self.declared, self.report, self.audit, self.base, self.native_scene, self.derived_asset]:
            self.assertTrue(S.public_static(self.root / binding['path'], root=self.root), binding['path'])

    def test_merely_binding_or_writing_a_declaration_does_not_promote_it(self):
        C.build(root=self.root)
        self.assertEqual(json.loads((self.root / 'showcase/public/library-catalog.json').read_text())['records'], [])
        self.assertFalse(S.public_static(self.root / self.declared['path'], root=self.root))
        self.approve(strategy='unrelated_or_unknown_adoption')
        selected = C._approved_records(self.root)[0]
        self.assertNotIn('derived_assembly', selected['record'])
        self.assertEqual(selected['aligned']['manifest'], self.native_scene)

    def test_historical_declaration_without_facing_gate_cannot_become_default(self):
        changed = copy.deepcopy(self.declared_doc)
        self.audit = self.file('data/declared/source-audit.json', {'fixture': 'historical-audit', 'policy': {}})
        changed['source_alignment_audit_ref'] = self.audit
        self.declared = self.file('data/declared/derived-assembly.json', changed)
        self.approve()
        selected = C._approved_records(self.root)[0]
        self.assertNotIn('derived_assembly', selected['record'])
        self.assertEqual(selected['record']['assembly_refinement_status'], 'withheld_facing_ambiguity')
        self.assertEqual(selected['aligned']['manifest'], self.native_scene)

    def test_explicit_declaration_must_be_in_this_exact_approved_artifact_set(self):
        self.approve(omit_binding=True)
        with self.assertRaisesRegex(ValueError, 'not bound by this approval'): C._approved_records(self.root)

    def test_ambiguous_explicit_strategies_are_rejected(self):
        self.approve(duplicate=True)
        with self.assertRaisesRegex(ValueError, 'Ambiguous'): C._approved_records(self.root)

    def test_source_native_base_and_default_eligibility_cannot_be_rebound(self):
        original = copy.deepcopy(self.declared_doc)
        for mode in ['source', 'base', 'eligible', 'program']:
            with self.subTest(mode=mode):
                # A new isolated workflow for each exact approved declaration.
                with self.store.db() as db:
                    for table in ['decisions', 'repair_plans', 'revisions', 'jobs', 'candidates']:
                        db.execute('DELETE FROM ' + table)
                changed = copy.deepcopy(original)
                if mode == 'source': changed['source']['sha256'] = 'f' * 64
                elif mode == 'base': changed['base_object_program_ref'] = self.report
                elif mode == 'program': changed['evidence_program'] = self.report
                else: changed['source_audit_summary']['default_eligible'] = False
                self.declared = self.file('data/declared/derived-assembly.json', changed)
                self.approve()
                with self.assertRaisesRegex(ValueError, 'another source or is ineligible'): C._approved_records(self.root)

    def test_changed_approved_declaration_is_withheld_by_actual_workflow_hash_check(self):
        self.approve()
        (self.root / self.declared['path']).write_text('{"changed":true}')
        self.assertEqual(self.store.public_index()['records'], [])
        self.assertEqual(C._approved_records(self.root), [])

    def test_companion_bound_part_does_not_disappear_when_newer_unapproved_mesh_wins_generic_discovery(self):
        def review(folder, asset, metadata):
            receipt = self.file(folder + '/import-receipt.json', {'import_files': [
                {'path': 'metadata.json', 'sha256': metadata['sha256'], 'bytes': metadata['bytes']}]})
            manifest = self.file(folder + '/review-manifest.json', {'import_receipt': receipt})
            return {'component_id': 'part', 'source_sha256': self.record['source']['sha256'],
                    'native_glb': asset, 'review_manifest': manifest}
        # A real existing native metadata path, copied only for the unrelated
        # future model. Exact metadata identity is the approval anchor.
        old = review('data/case', self.native, self.metadata)
        newer_metadata = self.file('data/newer/metadata.json', {'fixture': 'unapproved newer anchor'})
        newer_asset = self.file('data/newer/asset.glb', b'newer-unapproved-model-fixture')
        newer = review('data/newer', newer_asset, newer_metadata)
        self.record['components'][0]['asset'] = None
        self.record['sam3d_reviews'] = [newer]
        self.approve(strategy='unrelated_or_unknown_adoption')
        # Raw private discovery is insufficient. Only an exact pre-existing
        # curated pin may supply a still-valid historical display transport.
        with patch('sam3d_review_catalog.discover_reviews', return_value=[old, newer]):
            self.assertEqual(C._approved_records(self.root)[0]['record']['sam3d_reviews'], [])
            legacy_record = copy.deepcopy(self.record)
            legacy_record['sam3d_reviews'] = [old, copy.deepcopy(old)]
            self.file('.private/admin-workflow/legacy-showcase.json', {'records': [{'record': legacy_record}],
                'new_human_approval_fabricated': False})
            selected = C._approved_records(self.root)[0]
        self.assertEqual(selected['record']['sam3d_reviews'], [old],
            'Preserve only the exact legacy-pinned transport after companion binding validation')
        for field in ['interaction', 'source']:
            foreign = copy.deepcopy(legacy_record)
            foreign[field]['sha256'] = 'f' * 64
            self.file('.private/admin-workflow/legacy-showcase.json', {'records': [{'record': foreign}]})
            self.assertEqual(C._approved_records(self.root)[0]['record']['sam3d_reviews'], [])

    def test_typed_derivative_lookup_rejects_unknown_ids_and_private_unadmitted_references(self):
        self.approve(); C.build(root=self.root)
        catalog = CuratedTwinCatalog(self.root)
        for value in ['private-id', self.declared['path'], '.private/derived-assembly.json']:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Unknown curated'):
                catalog.derived_assembly(value)
        private = self.file('data/private/working-derived-assembly.json', self.declared_doc)
        path = self.root / 'showcase/public/library-catalog.json'
        document = json.loads(path.read_text()); document['records'][0]['derived_assembly'] = private
        raw = json.dumps(document).encode(); digest = hashlib.sha256(raw).hexdigest()
        self.file('showcase/public/library-catalog-' + digest + '.json', raw)
        pointer = json.loads((self.root / 'showcase/public/library-index.json').read_text())
        pointer['catalog'] = {'path': 'library-catalog-' + digest + '.json', 'sha256': digest, 'bytes': len(raw)}
        self.file('showcase/public/library-index.json', pointer)
        with self.assertRaisesRegex(ValueError, 'published|exact curated'): catalog.derived_assembly(self.row['id'])


class PublicDemoJSONScopeTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def write(self, relative, size=2):
        path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'{}' if size == 2 else b' ' * size)
        return path

    def test_exact_four_demo_jsons_are_bounded_and_neighbor_private_json_remains_unavailable(self):
        self.assertEqual(S.PUBLIC_DEMO_JSON, {'showcase/demos/charge-phone-v1/' + name for name in
            ['task.json', 'trace.json', 'summary.json', 'muse-review.json']})
        for name in S.PUBLIC_DEMO_JSON:
            path = self.write(name)
            self.assertTrue(S.public_static(path, root=self.root))
            self.write(name, 0); self.assertFalse(S.public_static(path, root=self.root))
            self.write(name, 2 * 1024 * 1024 + 1); self.assertFalse(S.public_static(path, root=self.root))
        for name in ['showcase/demos/charge-phone-v1/private.json', 'showcase/demos/charge-phone-v1/task.json.tmp',
                     'showcase/demos/charge-phone-v1/nested/task.json', 'showcase/demos/other/task.json',
                     'showcase/demos/charge-phone-v1/model-prompt.json', 'data/private/task.json']:
            self.assertFalse(S.public_static(self.write(name), root=self.root), name)


if __name__ == '__main__': unittest.main()
