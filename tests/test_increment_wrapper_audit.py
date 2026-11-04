"""Independent no-network checks for the AFK increment admission wrapper."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from increment_budget import IncrementMetaClient
from meta_api import MetaClient, MetaError
import pipeline

ROOT = Path(__file__).resolve().parents[1]


class IncrementWrapperAuditTests(unittest.TestCase):
    def test_real_runpy_imports_get_stricter_client_before_stage_imports(self):
        # Each real wrapper/stage import runs in a fresh interpreter, matching
        # production subprocess semantics rather than relying on module caches.
        for script in sorted({"discover_rigid_components.py", "stage2_meta_reason.py",
                              "stage2_meta_video_segment.py", "stage2_meta_segment.py",
                              "stage3_meta_select_track_views.py", "stage5_meta_propose.py"}):
            code = r'''
import importlib.util, pathlib, runpy, sys
root=pathlib.Path(sys.argv[1]); script=sys.argv[2]
sys.path.insert(0,str(root/'scripts'));sys.path.insert(0,str(root))
spec=importlib.util.spec_from_file_location('budget_wrapper',root/'scripts/run_budgeted_stage.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
original=runpy.run_path
seen=[]
def inspect_import(path,run_name):
    namespace=original(path,run_name='__independent_import_audit__')
    import meta_api
    from increment_budget import IncrementMetaClient
    assert meta_api.MetaClient is IncrementMetaClient
    if 'MetaClient' in namespace: assert namespace['MetaClient'] is IncrementMetaClient
    if script=='discover_rigid_components.py':
        assert namespace['rigid_discovery'].MetaClient is IncrementMetaClient
        assert namespace['rigid_discovery'].base.MetaClient is IncrementMetaClient
    seen.append(path)
    return namespace
w.runpy.run_path=inspect_import
sys.argv=['wrapper',script,'--help']
w.main()
assert len(seen)==1
'''
            with self.subTest(script=script):
                completed = subprocess.run([sys.executable, '-c', code, str(ROOT), script],
                                           cwd=ROOT, capture_output=True, text=True, timeout=30)
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_exhausted_increment_rejects_before_original_request_or_transport(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = Path(temporary)/'plan.json'
            plan.write_text(json.dumps({'budget_baseline': {'settled_usd': '1', 'reserved_usd': '3'},
                                        'additional_estimated_api_ceiling_usd': '20'}))
            client = object.__new__(IncrementMetaClient)
            client.increment_plan = plan
            client.ledger = Mock()
            client.ledger.status.return_value = {'settled_usd': '20.01', 'reserved_usd': '3',
                                                 'per_request_cap_usd': '2'}
            client.transport = Mock(side_effect=AssertionError('Network is forbidden'))
            with patch.object(MetaClient, 'request', side_effect=AssertionError('Must reject first')) as base:
                with self.assertRaisesRegex(MetaError, 'additional_increment_estimate_limit'):
                    client.request({'model':'sam-3.1'}, output='unused', kind='sam')
            base.assert_not_called()
            client.transport.assert_not_called()
            client.ledger.reserve.assert_not_called()

    def test_admitted_request_delegates_original_arguments_without_ledger_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = Path(temporary)/'plan.json'
            plan.write_text(json.dumps({'budget_baseline': {'settled_usd': '1', 'reserved_usd': '3'},
                                        'additional_estimated_api_ceiling_usd': '20'}))
            client = object.__new__(IncrementMetaClient)
            client.increment_plan = plan
            client.ledger = Mock()
            client.ledger.status.return_value = {'settled_usd': '1.5', 'reserved_usd': '3',
                                                 'per_request_cap_usd': '2'}
            payload = {'model':'sam-3.1'}
            with patch.object(MetaClient, 'request', return_value=('sentinel', {})) as base:
                self.assertEqual(client.request(payload,output='unused',kind='sam'), ('sentinel',{}))
            base.assert_called_once_with(payload,output='unused',kind='sam')
            self.assertEqual([call[0] for call in client.ledger.method_calls], ['status'])

    def test_scoped_dependencies_pin_envelope_implementation_and_plan(self):
        profile = pipeline.load_profile('meta_sam31_parts')
        for segmentation in (False,True):
            names = {Path(p).name for p in pipeline.provider_dependencies(profile,segmentation=segmentation)}
            self.assertTrue({'increment_budget.py','run_budgeted_stage.py','plan.json'} <= names)

    def test_wrapper_rejects_unreviewed_script_before_import(self):
        completed = subprocess.run([sys.executable,str(ROOT/'scripts/run_budgeted_stage.py'),
                                    '../meta_api.py'],cwd=ROOT,capture_output=True,text=True,timeout=20)
        self.assertEqual(completed.returncode,2)
        self.assertIn('invalid choice', completed.stderr)


if __name__ == '__main__':
    unittest.main()
