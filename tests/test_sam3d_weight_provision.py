"""Mocked HF/weight fixtures only: no network, actual weights or credentials."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from tests.test_sam3d_config_provision import P as C

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sam3d_weight_provision',ROOT.parent/'hpc/sam3d-v2/provision_weights.py')
W=importlib.util.module_from_spec(spec)
with patch.dict(sys.modules,{'provision_configs':C}):spec.loader.exec_module(W)


class WeightProvisionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name).resolve()/'runtime/weights'/W.PIN
        checkpoint=self.directory/'checkpoints';checkpoint.mkdir(parents=True)
        (checkpoint/'pipeline.yaml').write_text(''.join('weight'+str(i)+'_ckpt_path: '+Path(name).name+'\n' for i,name in enumerate(W.WEIGHTS)))
        current=C.inspect_configs(self.directory)
        self.config={'format':'sam3d-config-only-provision','version':1,'model_repository':C.REPO,'model_revision':W.PIN,
             'provisioner_sha256':W.sha_stream(Path(C.__file__)),
             'status':'configs_only_ready_for_weight_review','destination':str(self.directory),
             'checkpoint_weights_downloaded':False,'model_inference_executed':False,**current}
        self.receipt=self.directory/'config-provision.json';C.exclusive_json(self.receipt,self.config)
        self.raw={name:('TEMPORARY SYNTHETIC WEIGHT CONTRACT: '+name).encode() for name in W.WEIGHTS}
        self.rows=[SimpleNamespace(rfilename=name,size=len(raw),lfs=SimpleNamespace(size=len(raw),sha256=W.hashlib.sha256(raw).hexdigest())) for name,raw in self.raw.items()]
        self.api=Mock();self.api.model_info.return_value=SimpleNamespace(sha=W.PIN,siblings=self.rows)
        self.factory=Mock(return_value=self.api)
        self.snapshot=Mock(side_effect=self.download)
        self.disk=Mock(return_value=SimpleNamespace(free=100*2**30))

    def download(self,**kwargs):
        for name,raw in self.raw.items():(Path(kwargs['local_dir'])/name).write_bytes(raw)

    def run_provision(self):
        return W.provision(self.receipt,api_factory=self.factory,snapshot_fn=self.snapshot,disk_usage_fn=self.disk)

    def test_same_pin_exact_six_official_sizes_hashes_and_configs_preserved(self):
        before=(self.directory/'checkpoints/pipeline.yaml').read_bytes()
        report=self.run_provision()
        self.factory.assert_called_once_with(endpoint=C.ENDPOINT,token=True)
        self.api.model_info.assert_called_once_with(repo_id=C.REPO,revision=W.PIN,files_metadata=True,token=True)
        args=self.snapshot.call_args.kwargs
        self.assertEqual(args['allow_patterns'],list(W.WEIGHTS));self.assertEqual(args['revision'],W.PIN)
        self.assertEqual(args['max_workers'],1);self.assertIs(args['token'],True)
        self.assertEqual(args['endpoint'],'https://huggingface.co')
        self.assertEqual(report['status'],'weights_verified_aux_not_ready')
        self.assertTrue(report['official_lfs_hashes_verified']);self.assertFalse(report['model_inference_executed'])
        self.assertFalse(report['auxiliary_models_verified']);self.assertFalse(report['runtime_readiness_established'])
        self.assertEqual((self.directory/'checkpoints/pipeline.yaml').read_bytes(),before)
        self.assertEqual(json.loads((self.directory/'weights-provision.json').read_text()),report)
        for name,raw in self.raw.items():self.assertEqual((self.directory/name).read_bytes(),raw)

    def test_changed_config_rejected_before_any_auth_metadata_or_download(self):
        (self.directory/'checkpoints/pipeline.yaml').write_text('changed: true\n')
        with self.assertRaisesRegex(ValueError,'ConfigFilesOrClosureChanged'):self.run_provision()
        self.factory.assert_not_called();self.snapshot.assert_not_called()

    def test_existing_weight_path_preserved_and_no_model_info_called(self):
        path=self.directory/W.WEIGHTS[0];path.write_bytes(b'preserved')
        with self.assertRaises(FileExistsError):self.run_provision()
        self.assertEqual(path.read_bytes(),b'preserved');self.factory.assert_not_called()

    def test_insufficient_disk_stops_before_snapshot(self):
        size=sum(len(raw) for raw in self.raw.values())
        self.disk.return_value=SimpleNamespace(free=size+W.DISK_MARGIN-1)
        with self.assertRaises(OSError):self.run_provision()
        self.snapshot.assert_not_called();self.assertFalse((self.directory/'weights-provision.json').exists())

    def test_exact_disk_margin_boundary_allowed(self):
        self.disk.return_value=SimpleNamespace(free=sum(len(raw) for raw in self.raw.values())+W.DISK_MARGIN)
        self.assertEqual(self.run_provision()['required_disk_margin_bytes'],20*2**30)

    def test_missing_lfs_sha_size_or_metadata_never_downloads(self):
        for field,value in (('sha256',None),('sha256','not-an-oid'),('size',0),('size',True)):
            previous=getattr(self.rows[0].lfs,field);setattr(self.rows[0].lfs,field,value)
            with self.subTest(field=field,value=value),self.assertRaisesRegex(ValueError,'LFSHashOrSize'):self.run_provision()
            setattr(self.rows[0].lfs,field,previous)
        self.snapshot.assert_not_called()

    def test_official_lfs_dict_shape_also_supported(self):
        row=self.rows[0];row.lfs={'size':row.size,'sha256':W.hashlib.sha256(self.raw[row.rfilename]).hexdigest()}
        self.assertEqual(self.run_provision()['weight_files'][0]['sha256'],row.lfs['sha256'])

    def test_wrong_official_revision_refuses(self):
        self.api.model_info.return_value.sha='b'*40
        with self.assertRaisesRegex(ValueError,'OfficialModelPinDiffers'):self.run_provision()
        self.snapshot.assert_not_called()

    def test_missing_or_duplicate_official_file_refuses(self):
        self.rows.append(self.rows[0])
        with self.assertRaisesRegex(ValueError,'MissingOrDuplicate'):self.run_provision()
        self.snapshot.assert_not_called()

    def test_tampered_download_preserves_partial_but_emits_no_success_receipt(self):
        def bad(**kwargs):
            self.download(**kwargs);(self.directory/W.WEIGHTS[0]).write_bytes(b'incorrect')
        self.snapshot.side_effect=bad
        with self.assertRaisesRegex(ValueError,'OfficialLFSHashDiffers'):self.run_provision()
        self.assertFalse((self.directory/'weights-provision.json').exists())
        self.assertEqual((self.directory/W.WEIGHTS[0]).read_bytes(),b'incorrect')

    def test_config_or_receipt_cannot_change_during_download(self):
        def bad(**kwargs):
            self.download(**kwargs);self.receipt.write_text('{}')
        self.snapshot.side_effect=bad
        with self.assertRaisesRegex(ValueError,'ConfigReceiptChanged'):self.run_provision()
        self.assertFalse((self.directory/'weights-provision.json').exists())

    def test_symlink_weight_path_refuses_before_network(self):
        (self.directory/W.WEIGHTS[0]).symlink_to(self.directory/'absent')
        with self.assertRaisesRegex(ValueError,'Symlink'):self.run_provision()
        self.factory.assert_not_called()

    def test_only_actual_six_declared_weights_allowed(self):
        self.config['referenced_weight_paths_not_downloaded'].append('checkpoints/unrelated.ckpt')
        self.receipt.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError,'BindingDiffers'):self.run_provision()
        self.factory.assert_not_called()

    def test_cli_errors_expose_class_status_only(self):
        class Failure(Exception):
            response=SimpleNamespace(status_code=403,text='SYNTHETIC_SECRET')
            def __str__(self):raise AssertionError('Never stringify')
        def fail(*args,**kwargs):print('SYNTHETIC_SECRET');raise Failure()
        output=io.StringIO()
        with patch.object(W,'provision',side_effect=fail),patch('sys.stdout',output):
            self.assertEqual(W.main(['--config-receipt',str(self.receipt)]),2)
        self.assertEqual(json.loads(output.getvalue()),{'error_class':'Failure','http_status':403})


if __name__=='__main__':unittest.main()
