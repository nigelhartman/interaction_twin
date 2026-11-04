import base64
import copy
import json
from pathlib import Path
import tempfile
import unittest

from reconstruction_catalog import ReconstructionCatalog, INDEX, sha

try:
    from mcp import Client as SDKClient
except ImportError:
    SDKClient = None


class ReconstructionCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.program_hash = 'a' * 64
        self.source_hash = 'b' * 64
        self.document = {'source': {'sha256': self.source_hash, 'duration_s': 3.},
                         'tracklets': [{'id': 'holder'}],
                         'evidence': [{'id': 'source_1', 'timestamp_s': 1., 'frame_sha256': ''}],
                         'steps': [{'id': 'open', 'title': 'Open holder', 'participant_track_ids': ['holder'],
                                    'evidence_ids': ['source_1']}]}
        self.glb = self.artifact('data/case/model.glb', b'glTFexplicit-contract-fixture', 'model/gltf-binary')
        self.image = self.artifact('data/case/source.jpg', b'\xff\xd8explicit-source-fixture', 'image/jpeg')
        self.mask = self.artifact('data/case/mask.png', b'\x89PNG\r\n\x1a\nexplicit-mask-fixture', 'image/png')
        self.quality = self.artifact('data/case/quality.json', {'scope': 'unmeasured_shape'}, 'application/json')
        self.document['evidence'][0]['frame_sha256'] = self.image['sha256']
        self.descriptor = {'format': 'interaction-reconstruction-version', 'version': 1, 'version_id': 'holder-observed-v1',
                           'label': 'Contract fixture', 'interaction_path': 'data/case/interaction.json',
                           'interaction_sha256': self.program_hash, 'source_video_sha256': self.source_hash,
                           'component_id': 'holder', 'kind': 'observed', 'gauge': {'metric': False, 'units': 'arbitrary'},
                           'physical_accuracy_verified': False, 'simulation_ready': False, 'acceptance': 'unreviewed',
                           'artifacts': {'glb': self.glb, 'quality': self.quality},
                           'source_evidence': [{'id': 'source_1', 'program_evidence_id': 'source_1', 'timestamp_s': 1.,
                              'source_video_sha256': self.source_hash, 'image_path': self.image['path'], 'image_sha256': self.image['sha256'],
                              'mask_path': self.mask['path'], 'mask_sha256': self.mask['sha256']}]}
        self.publish()
        self.reader = ReconstructionCatalog(self.root)

    def file(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else (json.dumps(data) + '\n').encode())
        return path

    def artifact(self, relative, data, mime):
        path = self.file(relative, data)
        return {'path': relative, 'sha256': sha(path), 'mime': mime}

    def publish(self):
        path = self.file('data/versions/holder.json', self.descriptor)
        self.file(INDEX, {'format': 'interaction-reconstruction-catalog', 'version': 1, 'versions': [
            {'version_id': self.descriptor['version_id'], 'interaction_path': self.descriptor['interaction_path'],
             'descriptor_path': str(path.relative_to(self.root)), 'descriptor_sha256': sha(path)}]})

    def context(self):
        return {'interaction_path': 'data/case/interaction.json', 'document': self.document, 'program_sha256': self.program_hash}

    def test_source_bound_diagnostic_has_no_accepted_or_physics_promotion(self):
        result = self.reader.list(twin_id='fixture', **self.context())
        self.assertEqual(len(result['versions']), 1)
        self.assertEqual(self.reader.list(twin_id='fixture', selection='accepted', **self.context())['versions'], [])
        self.assertFalse(result['policy']['diagnostic_versions_replace_production_assets'])
        desc = self.reader.describe('holder-observed-v1', twin_id='fixture', **self.context())
        self.assertEqual(desc['related_procedure'][0]['step_id'], 'open')
        self.assertEqual(desc['artifacts']['source-mask-0']['sha256'], self.mask['sha256'])
        self.assertEqual(self.reader.artifact('holder-observed-v1', 'glb', **self.context())[1], b'glTFexplicit-contract-fixture')

    def test_descriptor_and_artifact_changes_fail_closed(self):
        self.file('data/versions/holder.json', {'changed': True})
        with self.assertRaisesRegex(ValueError, 'descriptor changed'):
            self.reader.load('holder-observed-v1', **self.context())
        self.publish()
        self.file('data/case/model.glb', b'glTFchanged')
        with self.assertRaisesRegex(ValueError, 'identity or type changed'):
            self.reader.load('holder-observed-v1', **self.context())

    def test_program_source_component_and_metric_claim_mismatch_rejected(self):
        for key, bad in [('interaction_sha256', 'c' * 64), ('source_video_sha256', 'd' * 64),
                         ('component_id', 'invented'), ('physical_accuracy_verified', True), ('simulation_ready', True)]:
            original = copy.deepcopy(self.descriptor)
            self.descriptor[key] = bad
            self.publish()
            with self.assertRaises(ValueError):
                self.reader.load('holder-observed-v1', **self.context())
            self.descriptor = original
        self.descriptor['gauge']['metric'] = True
        self.publish()
        with self.assertRaisesRegex(ValueError, 'physical'):
            self.reader.load('holder-observed-v1', **self.context())

    def test_confined_paths_hidden_symlinks_and_arbitrary_resources_rejected(self):
        hidden = self.file('data/.hidden-test-file.json', b'not-a-real-secret')
        outside = self.root / 'outside-test.json'
        outside.write_bytes(b'not-a-real-secret')
        (self.root / 'data/link.json').symlink_to(outside)
        for value in ['.secrets/example', 'data/../outside-test.json', 'data/.hidden-test-file.json',
                      'data/link.json', '/etc/passwd', 'data/case/source.jpg?x=1']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.reader.path(value)
        with self.assertRaises(ValueError):
            self.reader.artifact('holder-observed-v1', '../../outside-test.json', **self.context())

    def test_exact_program_source_evidence_identity_required(self):
        self.descriptor['source_evidence'][0]['timestamp_s'] = 2.
        self.publish()
        with self.assertRaisesRegex(ValueError, 'cited program evidence'):
            self.reader.load('holder-observed-v1', **self.context())

    def test_source_resource_alias_cannot_point_to_other_declared_pixels(self):
        self.descriptor['artifacts']['source-image-0'] = self.artifact('data/case/other.jpg', b'\xff\xd8other-test-pixels', 'image/jpeg')
        self.publish()
        with self.assertRaisesRegex(ValueError, 'source image alias'):
            self.reader.describe('holder-observed-v1', twin_id='fixture', **self.context())

    def test_quality_report_cannot_certify_unverified_physics(self):
        self.descriptor['artifacts']['quality'] = self.artifact('data/case/unsafe-quality.json', {'physics_ready': True}, 'application/json')
        self.publish()
        with self.assertRaisesRegex(ValueError, 'physical certification'):
            self.reader.load('holder-observed-v1', **self.context())

    def test_acceptance_requires_pinned_exact_human_inspection_record(self):
        self.descriptor['acceptance'] = 'accepted_for_inspection'
        self.publish()
        with self.assertRaises(ValueError):
            self.reader.load('holder-observed-v1', **self.context())
        self.descriptor['acceptance_record'] = self.artifact('data/case/acceptance.json', {
            'version_id': 'holder-observed-v1', 'glb_sha256': self.glb['sha256'],
            'interaction_sha256': self.program_hash, 'human_confirmed': True, 'status': 'accepted_for_inspection'}, 'application/json')
        self.publish()
        result = self.reader.list(twin_id='fixture', selection='accepted', **self.context())
        self.assertEqual(len(result['versions']), 1)
        self.assertFalse(result['versions'][0]['simulation_ready'])

    def test_proposed_subpart_requires_actual_parent_and_advisory_binding(self):
        self.descriptor.update(component_id='lid', parent_track_id='holder')
        self.publish()
        with self.assertRaises(ValueError):
            self.reader.load('holder-observed-v1', **self.context())
        self.descriptor['proposal_advisory'] = self.artifact('data/case/proposal.json', {
            'proposed_components': [{'id': 'lid', 'parent_track_id': 'holder', 'identity_status': 'model_proposal'}]}, 'application/json')
        self.publish()
        result = self.reader.describe('holder-observed-v1', twin_id='fixture', **self.context())
        self.assertEqual(result['component_binding'], 'model_proposed_subpart_of_existing_tracklet')
        self.assertEqual(result['related_procedure'][0]['step_id'], 'open')


@unittest.skipUnless(SDKClient is not None, 'Official MCP SDK is isolated in .venv-mcp; actual transport checks run there.')
class GenuineReconstructionMCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_sdk_retrieves_diagnostic_part_source_and_report(self):
        from mcp import Client
        from scripts.serve_twin_mcp import create_server
        from twin_catalog import TwinCatalog
        catalog = TwinCatalog()
        twin = 'flow-meta-video-jabra-20261030-v1--flow-procedure-scope-normalized-v1'
        async with Client(create_server(catalog), read_timeout_seconds=90) as client:
            result = await client.call_tool('list_reconstructions', {'twin_id': twin, 'selection': 'diagnostic'})
            self.assertFalse(result.is_error)
            data = result.structured_content or json.loads(result.content[0].text)
            self.assertEqual({r['component_id'] for r in data['versions']}, {'case_base', 'case_lid'})
            result = await client.call_tool('get_reconstruction', {'twin_id': twin, 'version_id': 'case-lid-triposr-v1'})
            desc = result.structured_content or json.loads(result.content[0].text)
            self.assertEqual(desc['parent_track_id'], 'charging_case')
            self.assertFalse(desc['simulation_ready'])
            for key in ['glb', 'source-image-0', 'source-mask-0', 'quality']:
                resource = await client.read_resource(desc['artifacts'][key]['uri'])
                self.assertEqual(resource.contents[0].mime_type, 'application/octet-stream')
                data = base64.b64decode(resource.contents[0].blob, validate=True)
                import hashlib
                self.assertEqual(hashlib.sha256(data).hexdigest(), desc['artifacts'][key]['sha256'])
            wrong = await client.call_tool('get_reconstruction', {'twin_id': twin, 'version_id': 'camera-body-stereo-v1'})
            self.assertTrue(wrong.is_error)


if __name__ == '__main__':
    unittest.main()
