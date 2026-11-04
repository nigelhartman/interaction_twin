"""Local injected-client tests. Never contact HF or load actual credentials."""
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sam3d_config_provision', ROOT.parent/'hpc/sam3d-v2/provision_configs.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


class ConfigProvisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.runtime = Path(self.temp.name).resolve()/'runtime'; self.runtime.mkdir()
        self.api = Mock()
        self.api.whoami.return_value = {'DO_NOT_PRINT_AUTH_TEST_SECRET':'identity'}
        self.api.model_info.return_value = SimpleNamespace(sha='a'*40)
        self.factory = Mock(return_value=self.api)
        self.snapshot = Mock(side_effect=self.write_configs)

    def write_configs(self, **kwargs):
        directory = Path(kwargs['local_dir'])/'checkpoints'; directory.mkdir()
        (directory/'pipeline.yaml').write_text('ss_generator_config_path: generator.yaml\nss_generator_ckpt_path: "${workspace_dir}/model.ckpt"\noptional_config_path: null\n')
        (directory/'generator.yaml').write_text('_target_: os.system\narchitecture: synthetic_software_fixture\n')

    def run_provision(self):
        return P.provision(self.runtime,api_factory=self.factory,snapshot_fn=self.snapshot)

    def test_fixed_endpoint_normal_auth_pin_yaml_only_and_exclusive_receipt(self):
        report = self.run_provision()
        self.factory.assert_called_once_with(endpoint='https://huggingface.co',token=True)
        self.api.whoami.assert_called_once_with(token=True)
        self.api.model_info.assert_called_once_with(repo_id='facebook/sam-3d-objects',revision='main',token=True)
        kwargs = self.snapshot.call_args.kwargs
        self.assertEqual(kwargs['revision'],'a'*40)
        self.assertEqual(kwargs['allow_patterns'],['checkpoints/*.yaml'])
        self.assertEqual(kwargs['max_workers'],1)
        self.assertIs(kwargs['token'],True)
        self.assertEqual(kwargs['endpoint'],'https://huggingface.co')
        self.assertEqual(report['status'],'configs_only_ready_for_weight_review')
        self.assertEqual(report['referenced_weight_paths_not_downloaded'],['checkpoints/model.ckpt'])
        self.assertFalse(report['checkpoint_weights_downloaded']); self.assertFalse(report['model_inference_executed'])
        receipt = self.runtime/'weights'/('a'*40)/'config-provision.json'
        self.assertEqual(json.loads(receipt.read_text()),report)
        self.assertNotIn('DO_NOT_PRINT_AUTH_TEST_SECRET',receipt.read_text())
        self.assertFalse(list(self.runtime.rglob('*.ckpt')))
        with self.assertRaises(FileExistsError): self.run_provision()
        self.assertEqual(self.snapshot.call_count,1)

    def test_missing_reference_is_reported_without_more_downloads(self):
        def missing(**kwargs):
            path=Path(kwargs['local_dir'])/'checkpoints';path.mkdir()
            (path/'pipeline.yaml').write_text('ss_generator_config_path: missing.yaml\n')
        self.snapshot.side_effect = missing
        report = self.run_provision()
        self.assertEqual(report['status'],'configs_only_closure_requires_review')
        self.assertEqual(report['missing_config_paths'],['checkpoints/missing.yaml'])
        self.assertEqual(self.snapshot.call_count,1)

    def test_bad_revision_prevents_snapshot_and_destination(self):
        self.api.model_info.return_value = SimpleNamespace(sha='main')
        with self.assertRaises(ValueError): self.run_provision()
        self.snapshot.assert_not_called();self.assertFalse((self.runtime/'weights').exists())

    def test_missing_auth_prevents_model_info_and_download(self):
        self.api.whoami.side_effect=RuntimeError('SYNTHETIC_SECRET_NEVER_LOG')
        with self.assertRaises(RuntimeError):self.run_provision()
        self.api.model_info.assert_not_called();self.snapshot.assert_not_called()

    def test_yaml_python_tag_is_not_executed(self):
        def unsafe(**kwargs):
            path=Path(kwargs['local_dir'])/'checkpoints';path.mkdir()
            (path/'pipeline.yaml').write_text('value: !!python/object/apply:os.system ["must not execute"]\n')
        self.snapshot.side_effect=unsafe
        import yaml
        with self.assertRaises(yaml.YAMLError):self.run_provision()

    def test_external_and_resolver_paths_refuse_without_fetch(self):
        for value in ('../credentials','/tmp/secret','https://example.com/a','${arbitrary:resolver}/a'):
            with self.subTest(value=value),self.assertRaises(ValueError):P.resolve_ref(value)

    def test_normal_workspace_prefix_is_safe_artifact_reference(self):
        self.assertEqual(P.resolve_ref('${workspace_dir}/model.ckpt'),'checkpoints/model.ckpt')

    def test_unsafe_runtime_symlink_refuses_before_auth(self):
        link=self.runtime.parent/'linked-runtime';link.symlink_to(self.runtime,target_is_directory=True)
        with self.assertRaises(ValueError):P.provision(link,api_factory=self.factory,snapshot_fn=self.snapshot)
        self.factory.assert_not_called()

    def test_error_sanitizer_never_stringifies_or_returns_auth_body(self):
        class SyntheticFailure(Exception):
            response=SimpleNamespace(status_code=403,text='DO_NOT_PRINT_AUTH_TEST_SECRET')
            def __str__(self):raise AssertionError('Exception string must not be read')
        self.assertEqual(P.safe_error(SyntheticFailure()),{'error_class':'SyntheticFailure','http_status':403})

    def test_main_suppresses_sdk_output_and_only_prints_class_status_on_error(self):
        class SyntheticFailure(Exception):
            response=SimpleNamespace(status_code=401,text='DO_NOT_PRINT_AUTH_TEST_SECRET')
        def fail(*args,**kwargs):
            print('DO_NOT_PRINT_AUTH_TEST_SECRET')
            raise SyntheticFailure('DO_NOT_PRINT_AUTH_TEST_SECRET')
        output=io.StringIO()
        with patch.object(P,'provision',side_effect=fail),patch('sys.stdout',output):
            code=P.main(['--runtime-root',str(self.runtime)])
        self.assertEqual(code,2)
        self.assertEqual(json.loads(output.getvalue()),{'error_class':'SyntheticFailure','http_status':401})
        self.assertNotIn('DO_NOT_PRINT',output.getvalue())

    def test_exclusive_json_preserves_previous_receipt(self):
        path=self.runtime/'receipt.json';P.exclusive_json(path,{'status':'first'})
        with self.assertRaises(FileExistsError):P.exclusive_json(path,{'status':'changed'})
        self.assertEqual(json.loads(path.read_text()),{'status':'first'})


if __name__ == '__main__':unittest.main()
