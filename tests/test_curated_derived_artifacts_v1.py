"""Exact public inventory boundaries; synthetic fixtures do not certify geometry."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from curated_derived_artifacts_v1 import (CuratedDerivedArtifacts, exact_reference,
                                         GLB_MAX_BYTES, OTHER_MAX_BYTES)
from curated_twin_catalog import CuratedTwinCatalog
from twin_catalog import TwinCatalog


class DerivedArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.assets = {}
        self.twin_id = 'synthetic-curated--procedure'
        self.program = self.write('data/fixture/program.json', {'synthetic': 'not actual footage'})
        self.scene = self.write('data/fixture/scene.json', {'synthetic': 'not actual geometry'})
        self.asset = self.write('data/fixture/native.glb', b'glTF\x00\xffsynthetic-binary')
        self.metadata = self.write('data/fixture/metadata.json', {})
        self.report = self.write('data/fixture/report.json', {})
        self.audit = self.write('data/fixture/audit.json', {})
        self.source = {'sha256': 'a' * 64, 'path': 'data/fixture/source.mp4'}
        self.document = {'format': 'interaction-twin-declared-geometry-derivative', 'version': 1,
            'source': self.source, 'evidence_program': self.program,
            'base_object_program_ref': self.program, 'scene_ref': self.scene,
            'baseline_scene_ref': self.scene, 'refinement_report_ref': self.report,
            'source_alignment_audit_ref': self.audit,
            'source_audit_summary': {'default_eligible': True},
            'component_assets': [{'component_id': 'part', 'native_glb': self.asset,
                'original_native_glb': self.asset, 'native_metadata': self.metadata}]}
        self.record = {'id': self.twin_id, 'interaction': self.program, 'source': self.source,
                       'curation': {'status': 'owner_approved'}}
        self.records = [self.record]
        self.pin_document(); self.publish()
        self.catalog = CuratedTwinCatalog(self.root)
        self.reader = CuratedDerivedArtifacts(self.catalog)
        # Only geometry validation is replaced: these are intentionally tiny
        # non-GLTF transport fixtures. Actual snapshots are exercised by SDK QA.
        self.shape = patch('declared_assembly_view.validate', return_value={})
        self.shape.start(); self.addCleanup(self.shape.stop)

    def write(self, relative, value, *, admit=True):
        raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode()
        path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        reference = {'path': relative, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        if admit: self.assets[relative] = reference
        return reference

    def pin_document(self):
        self.record['derived_assembly'] = self.write('data/fixture/derived.json', self.document)

    def publish(self):
        value = {'format': 'interaction-twin-object-program-library', 'version': 1, 'records': self.records}
        raw = json.dumps(value).encode(); digest = hashlib.sha256(raw).hexdigest()
        filename = 'library-catalog-' + digest + '.json'
        self.write('showcase/public/' + filename, raw, admit=False)
        self.write('showcase/public/library-index.json', {'format': 'interaction-twin-library-index',
            'version': 1, 'catalog': {'path': filename, 'sha256': digest, 'bytes': len(raw)}}, admit=False)
        self.write('showcase/public/cases-index.json', {'curated': True, 'cases': [
            {'id': r['id'], 'interaction_path': r['interaction']['path'],
             'original_sha256': r['interaction']['sha256']} for r in self.records]}, admit=False)
        self.write('showcase/public/public-assets.json', {'format': 'interaction-twin-public-assets',
            'version': 1, 'assets': list(self.assets.values())}, admit=False)

    def appearance(self):
        previous = self.write('data/fixture/previous.json', self.document)
        patch_ref = self.write('data/fixture/dark.itcp', struct.pack('<4sIII', b'ITCP', 1, 3, 1) +
                               struct.pack('<IHHBBB', 1, 2, 3, 10, 20, 30))
        probe = self.write('data/fixture/probes.u32', struct.pack('<II', 1, 2))
        rgb = self.write('data/fixture/observation.jpg', b'\xff\xd8synthetic_RGB\xff\xd9')
        mask = self.write('data/fixture/observation.png', b'\x89PNG\x00synthetic_mask')
        unknown_rgb = self.write('data/fixture/unknown.jpg', b'\xff\xd8synthetic_unknown')
        unknown_mask = self.write('data/fixture/unknown.png', b'\x89PNGsynthetic_unknown')
        self.appearance_doc = {'format': 'interaction-twin-source-appearance-states', 'version': 1,
            'previous_derived_assembly_ref': previous, 'scene_ref': self.scene, 'source': self.source,
            'evidence_program': self.program, 'component_id': 'part', 'baseline_asset_ref': self.asset,
            'probe_native_vertex_ids_ref': probe,
            'keyframes': [{'source_rgb_ref': rgb, 'source_mask_ref': mask, 'patch_ref': patch_ref}],
            'observations': [{'status': 'source_observed_appearance_estimate', 'source_rgb_ref': rgb,
                              'source_mask_ref': mask}, {'status': 'unknown', 'source_rgb_ref': unknown_rgb,
                                                        'source_mask_ref': unknown_mask}]}
        self.document['source_appearance_states_ref'] = self.write('data/fixture/appearance.json', self.appearance_doc)
        self.pin_document(); self.publish()
        return [previous, patch_ref, probe, rgb, mask, unknown_rgb, unknown_mask]

    def nonrigid(self):
        trace = self.write('data/fixture/trace.json', {'format': 'interaction-twin-cable-trace-2d',
            'version': 1, 'program_ref': self.program, 'source_sha256': self.source['sha256'],
            'samples': [{'frame_ref': {'path': 'data/PRIVATE-do-not-follow.jpg'}}]})
        self.nonrigid_doc = {'format': 'interaction-twin-nonrigid-observations', 'version': 1,
            'evidence_program': self.program, 'source': self.source, 'trace_ref': trace}
        self.document['nonrigid_observations_ref'] = self.write('data/fixture/nonrigid.json', self.nonrigid_doc)
        self.pin_document(); self.publish()
        return trace

    def listing(self): return self.reader.list_artifacts(self.twin_id)
    def entry(self, reference):
        return next(a for a in self.listing()['artifacts'] if a['path'] == reference['path'])

    def test_exact_native_bytes_and_deterministic_ids_deduplicate_roles(self):
        listing = self.listing(); item = self.entry(self.asset)
        self.assertEqual(listing, self.listing())
        self.assertEqual(len(listing['artifacts']), 7)
        self.assertEqual(self.reader.artifact(self.twin_id, item['artifact_id']), b'glTF\x00\xffsynthetic-binary')
        self.assertEqual(item['mime_type'], 'model/gltf-binary')
        self.assertEqual(item['roles'], ['component:part:native_glb', 'component:part:original_native_glb'])
        self.assertFalse(listing['physical_accuracy_verified'])
        self.assertTrue(item['uri'].endswith('/' + item['artifact_id']))

    def test_typed_binary_patch_probe_and_unknown_source_refs_remain_exact(self):
        references = self.appearance()
        for reference in references:
            item = self.entry(reference)
            self.assertEqual(self.reader.artifact(self.twin_id, item['artifact_id']),
                             (self.root / reference['path']).read_bytes())
        self.assertIn('appearance:observation:source_rgb_ref', self.entry(references[-2])['roles'])

    def test_nonrigid_trace_is_typed_without_unbounded_recursive_source_discovery(self):
        trace = self.nonrigid(); self.assertEqual(json.loads(self.reader.artifact(
            self.twin_id, self.entry(trace)['artifact_id']))['program_ref'], self.program)
        self.assertFalse(any('PRIVATE' in a['path'] for a in self.listing()['artifacts']))

    def test_arbitrary_nested_provider_refs_never_become_resource_authority(self):
        private = self.write('data/private/provider.txt', b'private-not-admitted', admit=False)
        public_but_unrelated = self.write('data/other/source.jpg', b'not-this-twin')
        self.document['untrusted_model_text'] = {'provider_ref': private, 'artifact': public_but_unrelated}
        self.pin_document(); self.publish()
        paths = {a['path'] for a in self.listing()['artifacts']}
        self.assertNotIn(private['path'], paths); self.assertNotIn(public_but_unrelated['path'], paths)

    def test_private_owner_catalog_and_unknown_or_path_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'curated public'):
            CuratedDerivedArtifacts(TwinCatalog(self.root))
        for value in ['private-unapproved--procedure', '../data/fixture', self.twin_id + '/file']:
            with self.subTest(value=value), self.assertRaises(ValueError): self.reader.list_artifacts(value)
        for value in ['../../.secrets/key', self.asset['path'], 'A' * 64, 'a' * 63]:
            with self.subTest(value=value), self.assertRaises(ValueError): self.reader.artifact(self.twin_id, value)

    def test_same_asset_cannot_reuse_another_curated_twin_token(self):
        token = self.entry(self.asset)['artifact_id']
        second = copy.deepcopy(self.record); second['id'] = 'second-curated--procedure'
        self.records.append(second); self.publish()
        other = self.reader.list_artifacts(second['id'])
        self.assertNotEqual(token, next(a['artifact_id'] for a in other['artifacts'] if a['path'] == self.asset['path']))
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.reader.artifact(second['id'], token)

    def test_old_approval_tokens_fail_after_current_declaration_changes(self):
        token = self.entry(self.asset)['artifact_id']
        self.document['revision'] = 'new-current-approval'; self.pin_document(); self.publish()
        with self.assertRaisesRegex(ValueError, 'stale'): self.reader.artifact(self.twin_id, token)

    def test_changed_bytes_or_removed_public_admission_fail_after_listing(self):
        token = self.entry(self.asset)['artifact_id']; path = self.root / self.asset['path']; original = path.read_bytes()
        path.write_bytes(original[:-1] + b'!')
        with self.assertRaisesRegex(ValueError, 'bytes changed'): self.reader.artifact(self.twin_id, token)
        path.write_bytes(original); self.assets.pop(self.asset['path']); self.publish()
        with self.assertRaisesRegex(ValueError, 'not published'): self.reader.artifact(self.twin_id, token)

    def test_declaration_switch_during_read_fails_closed(self):
        token = self.entry(self.asset)['artifact_id']; original_read = self.catalog._bytes
        def race(ref, *, maximum):
            raw = original_read(ref, maximum=maximum)
            if ref == self.asset:
                self.document['revision'] = 'concurrent-approval'; self.pin_document(); self.publish()
            return raw
        with patch.object(self.catalog, '_bytes', side_effect=race), self.assertRaisesRegex(ValueError, 'changed during'):
            self.reader.artifact(self.twin_id, token)

    def test_missing_derivative_is_explicit_unavailable_without_native_fallback(self):
        self.record.pop('derived_assembly'); self.publish()
        self.assertEqual(self.listing()['artifacts'], [])
        self.assertEqual(self.listing()['status'], 'unavailable')

    def test_supported_legacy_declaration_has_no_invented_baseline_ref(self):
        self.document['format'] = 'interaction-twin-declared-derived-assembly'
        self.document.pop('baseline_scene_ref')
        self.audit = self.write(self.audit['path'], {'policy': {'cavity_facing_ambiguity_gate':
            'withhold_bracketed_open_when_parent_cavity_faces_camera_and_child_cavity_faces_away'}})
        self.document['source_alignment_audit_ref'] = self.audit
        self.pin_document(); self.publish()
        self.assertEqual(self.listing()['status'], 'available')
        self.assertNotIn('baseline_scene_ref', {r for a in self.listing()['artifacts'] for r in a['roles']})

    def test_cross_source_appearance_or_forged_previous_descriptor_rejects(self):
        self.appearance()
        for mutation in ['source', 'previous']:
            value = copy.deepcopy(self.appearance_doc)
            if mutation == 'source': value['source'] = {**self.source, 'sha256': 'b' * 64}
            else: value['previous_derived_assembly_ref'] = self.write('data/fixture/wrong-previous.json', {})
            self.document['source_appearance_states_ref'] = self.write('data/fixture/appearance.json', value)
            self.pin_document(); self.publish()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.listing()

    def test_cross_source_trace_and_unrecognized_typed_format_reject(self):
        trace = self.nonrigid()
        for value in [{'format': 'unknown', 'version': 1},
                      {'format': 'interaction-twin-cable-trace-2d', 'version': 1,
                       'program_ref': self.program, 'source_sha256': 'b' * 64}]:
            self.nonrigid_doc['trace_ref'] = self.write(trace['path'], value)
            self.document['nonrigid_observations_ref'] = self.write('data/fixture/nonrigid.json', self.nonrigid_doc)
            self.pin_document(); self.publish()
            with self.assertRaisesRegex(ValueError, 'another source or format'): self.listing()

    def test_missing_typed_appearance_refs_are_not_silently_skipped(self):
        references = self.appearance()
        for reference in references:
            saved = self.assets.pop(reference['path']); self.publish()
            try:
                with self.subTest(path=reference['path']), self.assertRaises(ValueError): self.listing()
            finally: self.assets[reference['path']] = saved; self.publish()

    def test_item_cap_is_fail_closed_instead_of_silent_truncation(self):
        self.document['visual_review_refs'] = [self.write(f'data/fixture/review-{i}.json', {}) for i in range(129)]
        self.pin_document(); self.publish()
        with self.assertRaisesRegex(ValueError, 'item budget'): self.listing()

    def test_size_and_path_guards_reject_before_file_read(self):
        for reference in [{**self.asset, 'bytes': GLB_MAX_BYTES + 1},
                          {**self.report, 'bytes': OTHER_MAX_BYTES + 1},
                          {**self.report, 'bytes': True}, {**self.report, 'path': 'data/../private/x'},
                          {**self.report, 'path': 'data//x'}, {**self.report, 'path': 'data/.hidden/x'},
                          {**self.report, 'extra': 'untyped'}, {**self.report, 'path': '/data/x'}]:
            with self.subTest(reference=reference), self.assertRaises(ValueError): exact_reference(reference)

    def test_lightweight_module_imports_without_site_or_CV_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        code = ('import sys;sys.path.insert(0,' + repr(str(root)) + ');'
                'import curated_derived_artifacts_v1;'
                'assert not any(n.split(".")[0] in {"numpy","cv2","PIL","scipy","torch"} for n in sys.modules)')
        result = subprocess.run([sys.executable, '-I', '-S', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__': unittest.main()
