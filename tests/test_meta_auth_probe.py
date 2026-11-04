import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from urllib.error import HTTPError,URLError
from scripts import meta_auth_probe as probe

FAKE='fake-test-only-token|with-visible-symbols'


class MetaAuthProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.private=self.root/'.secrets';self.private.mkdir(mode=0o700)
        self.keyfile=self.private/'meta.env';self.keyfile.write_text('# A data-only comment\nMODEL_API_KEY='+FAKE+'\n');self.keyfile.chmod(0o600)
        self.receipt=self.private/'meta-auth-attempt.json'

    def call(self,*args):
        stdout=io.StringIO()
        with patch.object(probe,'ROOT',self.root),contextlib.redirect_stdout(stdout):code=probe.main(list(args))
        self.assertNotIn(FAKE,stdout.getvalue())
        return code,json.loads(stdout.getvalue())

    def test_default_never_networks_or_reserves_attempt(self):
        with patch.object(probe,'request_once') as request:
            code,result=self.call()
        self.assertEqual(code,0);self.assertEqual(result['status'],'local_key_ready_no_request')
        request.assert_not_called();self.assertFalse(self.receipt.exists())

    def test_attempt_is_durable_before_only_request_and_cannot_repeat(self):
        def request(key):
            self.assertEqual(key,FAKE);self.assertTrue(self.receipt.is_file())
            self.assertEqual(json.loads(self.receipt.read_text())['status'],'attempt_reserved')
            return {'status':'http_ok','http_status':200,'model_count':2}
        with patch.object(probe,'request_once',side_effect=request) as network:
            self.assertEqual(self.call('--auth-once')[0],0)
            self.assertEqual(self.call('--auth-once')[1]['status'],'attempt_already_consumed')
        self.assertEqual(network.call_count,1);self.assertEqual(self.receipt.stat().st_mode&0o777,0o600)
        self.assertNotIn(FAKE,self.receipt.read_text())

    def test_failed_attempt_stays_consumed_and_error_secret_not_logged(self):
        with patch.object(probe,'request_once',side_effect=URLError(FAKE)) as network:
            self.assertEqual(self.call('--auth-once')[1]['status'],'request_failed')
            self.assertEqual(self.call('--auth-once')[1]['status'],'attempt_already_consumed')
        self.assertEqual(network.call_count,1);self.assertNotIn(FAKE,self.receipt.read_text())

    def test_empty_or_malformed_files_do_not_consume_attempt(self):
        for content in ['', 'MODEL_API_KEY=\n','MODEL_API_KEY=x\nMODEL_API_KEY=y\n',
                        'MODEL_API_KEY=x\nOTHER=y\n','MODEL_API_KEY=x\rINJECT=y',
                        'MODEL_API_KEY=x\x00y','MODEL_API_KEY=x\ty','MODEL_API_KEY=x y']:
            with self.subTest(content=repr(content)),patch.object(probe,'request_once') as network:
                self.keyfile.write_text(content);self.assertEqual(self.call('--auth-once')[0],2)
                network.assert_not_called();self.assertFalse(self.receipt.exists())

    def test_public_file_and_symlinks_are_rejected(self):
        self.keyfile.chmod(0o640)
        with self.assertRaises(probe.ProbeError):probe.probe(self.root,True)
        self.keyfile.chmod(0o600);saved=self.private/'copy.env';self.keyfile.rename(saved);self.keyfile.symlink_to(saved)
        with self.assertRaises(probe.ProbeError):probe.probe(self.root,True)
        self.assertFalse(self.receipt.exists())

    def test_symlink_directory_is_rejected(self):
        actual=self.root/'actual';self.private.rename(actual);self.private.symlink_to(actual,target_is_directory=True)
        with self.assertRaises(probe.ProbeError):probe.probe(self.root,True)

    def test_transport_fixed_get_proxy_disabled_and_bounded_read(self):
        response=Mock();response.status=200;response.read.return_value=b'{"data":[{},{}]}'
        response.__enter__=Mock(return_value=response);response.__exit__=Mock(return_value=False)
        opener=Mock();opener.open.return_value=response
        with patch.object(probe,'build_opener',return_value=opener) as build:
            result=probe.request_once(FAKE)
        request=opener.open.call_args.args[0]
        self.assertEqual(request.full_url,'https://api.meta.ai/v1/models');self.assertEqual(request.get_method(),'GET')
        self.assertIsNone(request.data);self.assertEqual(opener.open.call_count,1)
        self.assertEqual(opener.open.call_args.kwargs,{'timeout':10})
        self.assertEqual(build.call_args.args[0].proxies,{})
        self.assertIsInstance(build.call_args.args[1],probe.NoRedirect)
        response.read.assert_called_once_with(probe.MAX_BYTES+1);self.assertEqual(result['model_count'],2)

    def test_redirect_and_malicious_error_body_never_trigger_second_request(self):
        for code in [302,401]:
            body=io.BytesIO(FAKE.encode());error=HTTPError(probe.ENDPOINT,code,FAKE,{'X-Key':FAKE},body)
            opener=Mock();opener.open.side_effect=error
            with patch.object(probe,'build_opener',return_value=opener):result=probe.request_once(FAKE)
            self.assertEqual(opener.open.call_count,1);self.assertEqual(body.tell(),0)
            self.assertNotIn(FAKE,json.dumps(result));self.assertEqual(result['http_status'],code)
        self.assertIsNone(probe.NoRedirect().redirect_request(None,None,302,'',{},'https://elsewhere.invalid'))

    def test_bounded_response_and_receipt_have_no_provider_text(self):
        response=Mock();response.status=200;response.read.return_value=b'x'*(probe.MAX_BYTES+1)
        response.__enter__=Mock(return_value=response);response.__exit__=Mock(return_value=False)
        opener=Mock();opener.open.return_value=response
        with patch.object(probe,'build_opener',return_value=opener):
            self.assertEqual(self.call('--auth-once')[1]['status'],'response_too_large')
        self.assertEqual(set(json.loads(self.receipt.read_text())),{'status','http_status','model_count'})

    def test_unrecognized_arguments_never_echo_accidentally_pasted_key(self):
        self.assertEqual(self.call('--force',FAKE)[1]['status'],'invalid_arguments')

if __name__=='__main__':unittest.main()
