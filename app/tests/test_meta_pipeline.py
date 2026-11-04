"""Offline provider-routing, consent and budget-view regression checks."""
import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pipeline as P

spec = importlib.util.spec_from_file_location('meta_pipeline_workspace', ROOT / 'scripts/serve_workspace.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


class MetaPipelineContracts(unittest.TestCase):
    def test_meta_profiles_have_explicit_bounded_providers(self):
        for name in ['meta_low', 'meta_smoke', 'meta_sam31_low', 'meta_sam31_video_low']:
            p = P.load_profile(name)
            self.assertEqual(p['reasoning_provider'], 'meta_api')
            self.assertEqual(p['reasoning_model'], 'muse-spark-1.3')
            self.assertEqual(p['max_output_tokens'], 16000)
            self.assertIn('$100', p['disclosure'])
        self.assertEqual(P.load_profile('meta_sam31_low')['segmentation_provider'], 'meta_sam31_api')
        self.assertEqual(P.load_profile('meta_sam31_video_low')['segmentation_mode'], 'video_boxes')
        self.assertGreaterEqual(P.load_profile('meta_low')['max_duration_s'], 74.240833)

    def test_unknown_providers_models_and_unbounded_outputs_fail_closed(self):
        original = P.load_profile('meta_low')
        for field, value in [('reasoning_provider', 'arbitrary_shell'), ('segmentation_provider', 'silent_fallback'),
                             ('reasoning_model', '../unreviewed'), ('max_output_tokens', 1000000),
                             ('max_output_tokens', True)]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                P.validate_providers(dict(original, **{field: value}))

    def test_fixed_cli_dispatch_and_no_fallback(self):
        p = P.load_profile('meta_low')
        for phase, expected in [('discover', 'stage2_meta_reason.py'), ('audit', 'stage2_meta_reason.py'),
                                ('procedure_proposal', 'stage5_meta_propose.py')]:
            script, flags = P.reasoning_dispatch(p, phase)
            self.assertEqual(script, expected)
            self.assertEqual(flags, ['--model', 'muse-spark-1.3', '--max-output-tokens', '16000'])
        self.assertEqual(P.reasoning_dispatch(P.load_profile('local_low'), 'discover'), ('stage2_discover.py', []))
        with self.assertRaises(ValueError):
            P.reasoning_dispatch(p, 'execute_user_command')

    def test_provider_dependencies_never_include_key_or_ledger(self):
        p = P.load_profile('meta_sam31_low')
        for segmentation in [False, True]:
            dependencies = P.provider_dependencies(p, segmentation=segmentation)
            self.assertTrue(all(path.suffix == '.py' and path.parent == ROOT for path in dependencies))
            self.assertFalse(any('.secrets' in str(path) for path in dependencies))

    def test_each_meta_pipeline_routes_all_three_semantic_calls_and_sam_variant(self):
        for profile in ['meta_low', 'meta_sam31_low', 'meta_sam31_video_low']:
            with self.subTest(profile=profile), tempfile.TemporaryDirectory(dir=ROOT / 'data') as temp:
                base = Path(temp)
                case = base / 'case'
                case.mkdir()
                config = case / 'annotations.json'
                P.write(config, {'tracks': []})
                runner = P.Runner('provider-fixture', run_root=base / 'runs')
                runner.profile = P.load_profile(profile)
                runner.state['provenance'] = {}
                calls = []

                def command(argv, timeout):
                    script = Path(argv[1]).name
                    args = list(map(str, argv[2:]))
                    calls.append((script, args))
                    if script == 'stage2_meta_reason.py' and args[0] == 'discover':
                        out = case / 'reasoning' / args[args.index('--run') + 1]
                        P.write(out / 'metadata.json', {'tracking_config': P.rel(config)})
                    elif script in {'stage2_meta_segment.py', 'stage2_meta_video_segment.py', 'stage2_track.py'}:
                        (case / 'segmentation' / args[args.index('--run') + 1]).mkdir(parents=True)
                    elif script == 'stage2_meta_reason.py' and args[0] == 'audit':
                        (case / 'segmentation' / args[args.index('--run') + 1] / 'model-audit-v1').mkdir()
                    elif script == 'stage5_meta_propose.py':
                        P.write(Path(args[args.index('--out') + 1]) / 'proposal.json', {'fixture': True})

                def phase(name, dependencies, action, **kwargs):
                    if name == 'prepare':
                        return {'case': P.rel(case)}
                    if name in {'discover', 'segment', 'audit', 'procedure_proposal'}:
                        return action(1)[0]
                    if name == 'select':
                        return {'components': [], 'selections': []}
                    return {}

                with patch.object(runner, 'phase', side_effect=phase), patch.object(runner, 'command', side_effect=command):
                    runner._infer(case / 'input.mp4', {'sha256': 'offline-fixture'})
                semantics = [(script, args[0]) for script, args in calls
                             if script in {'stage2_meta_reason.py', 'stage5_meta_propose.py'} and args[0] != 'review']
                self.assertEqual(semantics, [('stage2_meta_reason.py', 'discover'),
                                             ('stage2_meta_reason.py', 'audit'), ('stage5_meta_propose.py', 'propose')])
                self.assertFalse(any(script == 'stage5_compile.py' and args[0] == 'propose' for script, args in calls))
                sam_api = profile in {'meta_sam31_low', 'meta_sam31_video_low'}
                self.assertEqual(any(script == 'stage2_meta_segment.py' for script, _ in calls), profile == 'meta_sam31_low')
                self.assertEqual(any(script == 'stage2_meta_video_segment.py' for script, _ in calls), profile == 'meta_sam31_video_low')
                self.assertEqual(any(script == 'stage2_track.py' for script, _ in calls), not sam_api)
                self.assertEqual(runner.state['provenance']['reasoning'], 'hosted_Meta_Model_API')
                self.assertEqual(runner.state['provenance']['segmentation'], 'SAM3.1_Meta_API' if sam_api else 'SAM2.1Tiny_MPS')
                self.assertEqual(runner.state['provenance']['reconstruction'], 'TripoSR_MPS_non_Meta_comparison')

    def test_upload_consent_identifies_actual_hosted_provider(self):
        handler = S.Handler.__new__(S.Handler)
        for name, provider in [('meta_low', 'Meta Model API'), ('meta_sam31_low', 'SAM 3.1'), ('local_low', 'OpenAI/Codex')]:
            with self.subTest(profile=name), self.assertRaisesRegex(ValueError, provider.replace('/', '.')):
                handler.start_run({'profile': name, 'case_id': 'unused'})

    def test_budget_status_is_read_only_sanitized_estimate_not_credit_balance(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'data') as temp:
            base = Path(temp)
            before = list(base.iterdir())
            empty = S.meta_budget_status(base)
            self.assertEqual(empty['status'], 'not_started')
            self.assertEqual(list(base.iterdir()), before)
            self.assertFalse(empty['actual_meta_credit_balance_verified'])
            ledger = base / '.secrets/meta-budget.sqlite3'
            ledger.parent.mkdir()
            ledger.write_bytes(b'unit-fixture-only')
            read_calls = []

            def status(path):
                read_calls.append(path)
                return {'settled_usd': '0.020000', 'reserved_usd': '0.100000', 'cap_usd': '100.000000',
                        'remaining_estimated_usd': '99.880000', 'scope': 'local_ledger_only_not_account_balance',
                        'unexpected_private_field': 'never expose'}

            with patch.dict(sys.modules, {'meta_budget': types.SimpleNamespace(read_budget_status=status)}):
                result = S.meta_budget_status(base)
            self.assertEqual(read_calls, [ledger])
            self.assertEqual(result['status'], 'available')
            self.assertEqual(result['settled_usd'], '0.020000')
            self.assertNotIn('unexpected_private_field', result)
            self.assertEqual(ledger.read_bytes(), b'unit-fixture-only')
            self.assertFalse(result['actual_meta_credit_balance_verified'])


if __name__ == '__main__':
    unittest.main()
