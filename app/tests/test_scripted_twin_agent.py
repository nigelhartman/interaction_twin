"""Fail-closed scripted MCP consumer checks; no models, real media or network."""
import base64
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('scripted_twin_demo_test', ROOT/'scripts/demo_twin_agent.py')
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
spec = importlib.util.spec_from_file_location('scripted_twin_procedure_fixture', ROOT/'tests/test_stage5.py')
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)


class ToolResult:
    def __init__(self, value, is_error=False):
        self.structured_content = value
        self.is_error = is_error
        self.content = []

    def model_dump(self, **kwargs):
        return {'structuredContent': self.structured_content, 'isError': self.is_error}


class FakeMCP:
    def __init__(self):
        self.id = 'test-twin--procedure'
        self.doc = F.fixture(); self.doc['motion'] = []
        self.glb = b'glTF'+struct.pack('<II', 2, 24)+struct.pack('<II', 4, 0x4e4f534a)+b'{}  '
        self.jpeg = b'\xff\xd8fake-source-frame\xff\xd9'
        self.doc['assets'] = [{'id': 'asset_1', 'track_id': 'part', 'path': 'data/test/part.glb',
                               'sha256': D.sha(self.glb), 'backend': 'test', 'geometry_status': 'unreviewed_model_candidate',
                               'physical_scale_known': False}]
        for e in self.doc['evidence']: e['frame_sha256'] = D.sha(self.jpeg)
        self.program = json.dumps(self.doc, indent=2).encode()
        self.program_hash = D.sha(self.program)
        self.calls = []
        self.mode = None

    async def list_tools(self):
        self.calls.append(('tools/list',))
        names = D.TOOLS if self.mode != 'extra_tool' else D.TOOLS | {'execute_shell'}
        if self.mode == 'motion_tools': names = D.TOOLS | D.MOTION_TOOLS
        if self.mode == 'library_tools': names = D.TOOLS | D.MOTION_TOOLS | D.DISCOVERY_TOOLS
        if self.mode == 'robot_tools': names = D.TOOLS | D.MOTION_TOOLS | D.DISCOVERY_TOOLS | D.ROBOT_TOOLS
        if self.mode == 'partial_robot_tools': names = D.TOOLS | D.MOTION_TOOLS | D.DISCOVERY_TOOLS | {'get_robot_kit'}
        if self.mode == 'partial_motion_tools': names = D.TOOLS | {'list_motion_experiments'}
        annotations = SimpleNamespace(read_only_hint=True, destructive_hint=False, open_world_hint=False)
        tools = [SimpleNamespace(name=n, annotations=annotations) for n in sorted(names)]
        if self.mode == 'writable_tool': tools[0].annotations = SimpleNamespace(read_only_hint=False)
        return SimpleNamespace(tools=tools, model_dump=lambda **kwargs: {'tools': sorted(names)})

    async def call_tool(self, name, args):
        self.calls.append(('tool', name, args))
        if name == 'search_twins': return ToolResult({'twins': [{'twin_id': self.id}]})
        if name == 'describe_twin':
            return ToolResult({'twin_id': self.id, 'title': 'Synthetic contract fixture', 'program_sha256': self.program_hash,
                               'assets': [{'track_id': 'part', 'sha256': self.doc['assets'][0]['sha256']}],
                               'policy': {'simulation_ready': False, 'physical_accuracy_verified': False}})
        if name == 'get_procedure':
            return ToolResult({'twin_id': self.id, 'program_sha256': self.program_hash, 'steps': self.doc['steps'],
                               'evidence': self.doc['evidence'][:2]})
        if name == 'query_state':
            t = args['timestamp_s']
            value = 'closed' if t == 0. else 'open' if t == 1. else 'unknown'
            result = {'twin_id': self.id if self.mode != 'wrong_state_twin' else 'another--twin',
                      'program_sha256': self.program_hash, 'timestamp_s': t, 'conditional_poses': [],
                      'states': {'part_visual_state': {'value': value, 'status': 'unknown' if value == 'unknown' else 'visual_model_proposal',
                                                        'evidence_ids': [] if value == 'unknown' else [f'source_{int(t)+1:06d}']}}}
            if self.mode == 'persistent_state' and value == 'unknown':
                result['states']['part_visual_state'] = {'value': 'closed', 'status': 'visual_model_proposal', 'evidence_ids': ['source_000001']}
            if self.mode == 'invented_pose' and value == 'unknown': result['conditional_poses'] = [{'track_id': 'part'}]
            return ToolResult(result)
        if name == 'get_asset':
            uri = f'twin://{self.id}/assets/part' if self.mode != 'arbitrary_uri' else 'file:///etc/passwd'
            return ToolResult({'track_id': 'part', 'sha256': self.doc['assets'][0]['sha256'], 'uri': uri,
                               'local_file_uri': 'file:///must-never-be-read.glb'})
        if name == 'get_evidence':
            e = next(e for e in self.doc['evidence'] if e['id'] == args['evidence_id'])
            return ToolResult({'twin_id': self.id, 'evidence': e, 'frame_uri': f"twin://{self.id}/evidence/{e['id']}"})
        if name == 'get_simulation_manifest':
            return ToolResult({'twin_id': self.id, 'program_sha256': self.program_hash,
                               'conditional_motion_samples': self.doc['motion'], 'demonstration_steps': self.doc['steps'],
                               'policy': {'simulation_ready': self.mode == 'physics_claim'}})
        raise AssertionError('Unexpected tool requested: '+name)

    async def read_resource(self, uri):
        self.calls.append(('resource', uri))
        if uri.endswith('/program'):
            content = SimpleNamespace(uri=uri, mime_type='application/json', text=self.program.decode())
        elif '/assets/' in uri:
            data = self.glb if self.mode != 'bad_glb_hash' else self.glb+b'changed'
            content = SimpleNamespace(uri=uri, mime_type='model/gltf-binary', blob=base64.b64encode(data).decode())
        elif '/evidence/' in uri:
            data = self.jpeg if self.mode != 'bad_jpeg_hash' else b'changed'
            content = SimpleNamespace(uri=uri, mime_type='image/jpeg', blob=base64.b64encode(data).decode())
        else: raise AssertionError('Unexpected resource requested: '+uri)
        return SimpleNamespace(contents=[content])


class ScriptedTwinAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_execution_receipt_stays_outside_manifest_bounded_package(self):
        client = FakeMCP()
        collected = await D.collect_twin(client, twin_id=client.id)
        calls = []
        def builder(output, **kwargs):
            calls.append(kwargs)
            output.mkdir()
            (output/'manifest.json').write_text('{}')
            return {'directory': str(output), 'simulation_ready': False}
        with tempfile.TemporaryDirectory() as temporary, patch('portable_twin.build_package', side_effect=builder):
            output = Path(temporary)/'package'
            result = D.write_package(collected, output)
            receipt_path = Path(result['scripted_receipt'])
            self.assertEqual(receipt_path.parent, output.parent)
            self.assertNotIn('scripted-agent-receipt.json', [p.name for p in output.iterdir()])
            self.assertEqual(calls[0]['asset_bytes'], collected['asset_bytes'])
            self.assertEqual(calls[0]['program_bytes_kind'], 'mcp_json_serialization')
            self.assertEqual(calls[0]['provenance']['scripted_receipt_sha256'], D.sha(receipt_path.read_bytes()))
            with self.assertRaisesRegex(D.DemoError, 'existing demos'):
                D.write_package(collected, output)

    async def test_collects_only_declared_mcp_bytes_without_source_file_reads(self):
        client = FakeMCP()
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('Consumer cannot read source media directly')), \
             patch.object(Path, 'read_text', side_effect=AssertionError('Consumer cannot read source program directly')):
            result = await D.collect_twin(client, twin_id=client.id)
        self.assertEqual(set(result['asset_bytes']), {'part'})
        self.assertEqual(set(result['evidence_bytes']), {'source_000001', 'source_000002'})
        self.assertEqual(result['program_bytes'], client.program)
        self.assertEqual(result['receipt']['planning'], 'deterministic_scripted_workflow_no_LLM')
        self.assertFalse(result['receipt']['model_inference'])
        self.assertFalse(result['receipt']['original_program_file_bytes_included'])
        self.assertEqual(len(result['receipt']['calls']), len(client.calls))
        resources = [c[1] for c in client.calls if c[0] == 'resource']
        self.assertEqual(set(resources), {f'twin://{client.id}/program', f'twin://{client.id}/assets/part',
                                         f'twin://{client.id}/evidence/source_000001', f'twin://{client.id}/evidence/source_000002'})
        self.assertFalse(any('file:' in r or 'source_000003' in r for r in resources))

    async def test_unknown_twin_fails_before_describe_or_resources(self):
        client = FakeMCP()
        with self.assertRaisesRegex(D.DemoError, 'not discovered'):
            await D.collect_twin(client, twin_id='unknown--procedure')
        self.assertEqual(client.calls, [('tools/list',), ('tool', 'search_twins', {'query': '', 'limit': 25})])

    async def test_unexpected_or_writable_mcp_tools_fail_before_discovery(self):
        for mode in ['extra_tool', 'writable_tool', 'partial_motion_tools', 'partial_robot_tools']:
            client = FakeMCP(); client.mode = mode
            with self.subTest(mode=mode), self.assertRaisesRegex(D.DemoError, 'tool|tools'):
                await D.collect_twin(client, twin_id=client.id)
            self.assertEqual(client.calls, [('tools/list',)])

    async def test_exact_versioned_motion_extension_is_compatible(self):
        client=FakeMCP();client.mode='motion_tools'
        result=await D.collect_twin(client,twin_id=client.id)
        self.assertEqual(result['receipt']['calls'][0]['tools'],sorted(D.TOOLS|D.MOTION_TOOLS))

    async def test_readonly_library_discovery_extension_is_compatible(self):
        client=FakeMCP();client.mode='library_tools'
        result=await D.collect_twin(client,twin_id=client.id)
        self.assertEqual(result['receipt']['calls'][0]['tools'], sorted(D.TOOLS|D.MOTION_TOOLS|D.DISCOVERY_TOOLS))

    async def test_exact_readonly_robot_extension_is_compatible_without_using_new_tools(self):
        client=FakeMCP();client.mode='robot_tools'
        result=await D.collect_twin(client,twin_id=client.id)
        self.assertEqual(result['receipt']['calls'][0]['tools'],sorted(D.TOOLS|D.MOTION_TOOLS|D.DISCOVERY_TOOLS|D.ROBOT_TOOLS))
        self.assertFalse(any(c[0]=='tool'and c[1]in D.ROBOT_TOOLS for c in client.calls))

    async def test_hash_mismatch_never_silently_substitutes_local_files(self):
        for mode in ['bad_glb_hash', 'bad_jpeg_hash']:
            client = FakeMCP(); client.mode = mode
            with self.subTest(mode=mode), patch.object(Path, 'read_bytes', side_effect=AssertionError('No fallback reads')), \
                 self.assertRaisesRegex(D.DemoError, 'hash or header mismatch'):
                await D.collect_twin(client, twin_id=client.id)

    async def test_unknown_gaps_cannot_persist_state_or_invent_motion(self):
        for mode in ['persistent_state', 'invented_pose']:
            client = FakeMCP(); client.mode = mode
            with self.subTest(mode=mode), self.assertRaisesRegex(D.DemoError, 'State response'):
                await D.collect_twin(client, twin_id=client.id)

    async def test_response_from_another_twin_is_rejected(self):
        client = FakeMCP(); client.mode = 'wrong_state_twin'
        with self.assertRaisesRegex(D.DemoError, 'different twin or program'):
            await D.collect_twin(client, twin_id=client.id)

    async def test_server_supplied_arbitrary_uri_is_never_requested(self):
        client = FakeMCP(); client.mode = 'arbitrary_uri'
        with self.assertRaisesRegex(D.DemoError, 'Asset descriptor'):
            await D.collect_twin(client, twin_id=client.id)
        self.assertFalse(any(c[0] == 'resource' and c[1].startswith('file:') for c in client.calls))

    async def test_unsupported_physics_claim_is_rejected(self):
        client = FakeMCP(); client.mode = 'physics_claim'
        with self.assertRaisesRegex(D.DemoError, 'claims physics readiness'):
            await D.collect_twin(client, twin_id=client.id)

    async def test_download_budget_and_identifier_guards_precede_calls(self):
        for kwargs in [{'twin_id': '../.secrets/meta.env'}, {'twin_id': 'ok', 'max_evidence': 0},
                       {'twin_id': 'ok', 'max_evidence': True}, {'twin_id': 'ok', 'max_evidence': 13}]:
            client = FakeMCP()
            with self.subTest(kwargs=kwargs), self.assertRaises(D.DemoError): await D.collect_twin(client, **kwargs)
            self.assertEqual(client.calls, [])

    async def test_duplicate_asset_ids_fail_before_resource_download(self):
        client = FakeMCP(); client.doc['assets'].append(copy.deepcopy(client.doc['assets'][0]))
        client.program = json.dumps(client.doc).encode()
        with self.assertRaisesRegex(D.DemoError, 'Duplicate declared'):
            await D.collect_twin(client, twin_id=client.id)
        self.assertFalse(any(c[0] == 'resource' and '/assets/' in c[1] for c in client.calls))


if __name__ == '__main__':
    unittest.main()
