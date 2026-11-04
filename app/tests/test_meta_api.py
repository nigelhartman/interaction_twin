"""Offline fixed-origin transport and real-SQLite admission integration tests."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

import meta_api as api

FAKE = 'fake-visible-key-value-not-a-real-key'


def payload():
    return {'model':api.SPARK.model, 'messages':[{'role':'user','content':'test-only'}],
            'max_completion_tokens':16000, 'stream':False, 'store':False}


def document(**updates):
    data={'id':FAKE, 'usage':{'prompt_tokens':100,'completion_tokens':20,
                            'prompt_tokens_details':{'cached_tokens':30},
                            'completion_tokens_details':{'reasoning_tokens':10}},
          'choices':[{'finish_reason':'stop','message':{'content':'{"ok":true}'}}]}
    data.update(updates)
    return data


class MetaTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.sequence=0

    def client(self, transport, name=None):
        self.sequence+=1
        return api.MetaClient(ledger_path=self.root/((name or str(self.sequence))+'.sqlite3'),transport=transport)

    def test_fixed_transport_origin_no_proxy_redirects_and_no_key_output(self):
        response=Mock();response.status=200;response.read.return_value=json.dumps(document()).encode()
        response.__enter__=Mock(return_value=response);response.__exit__=Mock(return_value=False)
        opener=Mock();opener.open.return_value=response
        fd=os.open(self.root,os.O_RDONLY)
        with patch.object(api,'private_directory',return_value=fd),patch.object(api,'load_key',return_value=FAKE),patch.object(api,'build_opener',return_value=opener) as build:
            reply=api.fixed_http('/v1/chat/completions',payload())
        request=opener.open.call_args.args[0]
        self.assertEqual(request.full_url,'https://api.meta.ai/v1/chat/completions')
        self.assertEqual(request.get_header('Authorization'),'Bearer '+FAKE)
        self.assertEqual(request.get_method(),'POST');self.assertEqual(opener.open.call_count,1)
        self.assertEqual(build.call_args.args[0].proxies,{})
        self.assertIsInstance(build.call_args.args[1],api.NoRedirect)
        response.read.assert_called_once_with(api.MAX_REPLY+1)
        self.assertEqual(reply['http_status'],200)
        with patch.object(api,'load_key') as load:
            with self.assertRaisesRegex(api.MetaError,'endpoint_not_allowed'):
                api.fixed_http('https://untrusted.invalid',payload())
            load.assert_not_called()

    def test_error_bodies_are_not_read_or_persisted_and_no_retry(self):
        for code in [302,400,401,500]:
            with self.subTest(code=code):
                body=io.BytesIO(json.dumps({'error':{'type':FAKE,'code':FAKE,'param':FAKE,'message':FAKE}}).encode())
                error=HTTPError(api.ORIGIN,code,FAKE,{'Secret':FAKE},body)
                opener=Mock();opener.open.side_effect=error
                fd=os.open(self.root,os.O_RDONLY)
                with patch.object(api,'private_directory',return_value=fd),patch.object(api,'load_key',return_value=FAKE),patch.object(api,'build_opener',return_value=opener):
                    reply=api.fixed_http('/v1/chat/completions',payload())
                self.assertEqual(body.tell(),0);self.assertEqual(opener.open.call_count,1)
                self.assertNotIn(FAKE,json.dumps(reply))

    def test_transport_error_details_are_sanitized(self):
        opener=Mock();opener.open.side_effect=URLError(FAKE)
        fd=os.open(self.root,os.O_RDONLY)
        with patch.object(api,'private_directory',return_value=fd),patch.object(api,'load_key',return_value=FAKE),patch.object(api,'build_opener',return_value=opener):
            with self.assertRaises(api.MetaError) as caught:api.fixed_http('/v1/chat/completions',payload())
        self.assertNotIn(FAKE,str(caught.exception));self.assertEqual(opener.open.call_count,1)

    def test_reservation_precedes_transport_and_actual_accounting_redacts_ids(self):
        def transport(path,body,timeout):
            self.assertEqual(client.ledger.status()['reserved_usd'],'1.378720')
            self.assertEqual(client.ledger.status()['settled_usd'],'0.000000')
            return {'http_status':200,'document':document()}
        client=self.client(transport)
        _,receipt=client.request(payload(),output=self.root/'result')
        self.assertEqual(receipt['estimated_cost_usd'],'0.000177')
        self.assertEqual(client.ledger.status()['reserved_usd'],'0.000000')
        self.assertEqual(client.ledger.status()['settled_usd'],'0.000177')
        data=(self.root/'result/meta-request.json').read_text()
        self.assertNotIn(FAKE,data);self.assertNotIn('test-only',data)
        self.assertEqual((self.root/'result/meta-request.json').stat().st_mode&0o777,0o600)
        with self.assertRaises(api.MetaError):client.request(payload(),output=self.root/'result')
        self.assertEqual(client.ledger.status()['request_count'],1)

    def test_concurrent_same_output_can_only_send_once(self):
        calls=[];lock=threading.Lock();barrier=threading.Barrier(8)
        def transport(*args):
            with lock:calls.append(1)
            return {'http_status':200,'document':document()}
        client=self.client(transport)
        def worker(index):
            barrier.wait()
            try:client.request(payload(),output=self.root/'same');return True
            except api.MetaError:return False
        with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(worker,range(8)))
        self.assertEqual(sum(results),1);self.assertEqual(len(calls),1)
        self.assertEqual(client.ledger.status()['request_count'],1)

    def test_malformed_replies_and_usage_all_hold_and_block(self):
        bad=[None,{}, {'http_status':'200'}, {'http_status':True}, {'http_status':200,'document':None},
             {'http_status':200,'document':{}},
             {'http_status':200,'document':document(usage={'prompt_tokens':1,'completion_tokens':2,'prompt_tokens_details':['bad']})},
             {'http_status':200,'document':document(usage={'prompt_tokens':1,'completion_tokens':2,'completion_tokens_details':'bad'})},
             {'http_status':200,'document':document(usage={'prompt_tokens':-1,'completion_tokens':2})}]
        for index,reply in enumerate(bad):
            with self.subTest(index=index):
                transport=Mock(return_value=reply);client=self.client(transport)
                with self.assertRaises(api.MetaError):client.request(payload(),output=self.root/('bad'+str(index)))
                self.assertTrue(client.ledger.status()['blocked'])
                self.assertEqual(client.ledger.status()['reserved_usd'],'1.378720')
                transport.assert_called_once()
                with self.assertRaises(api.MetaError):client.request(payload(),output=self.root/('again'+str(index)))
                self.assertEqual(transport.call_count,1)

    def test_known_rejections_release_with_status_without_error_text(self):
        for code in [400,401,402,403,404,413,415,422,429]:
            with self.subTest(code=code):
                transport=Mock(return_value={'http_status':code,'error':{'code':FAKE},'document':None})
                client=self.client(transport);folder=self.root/('http'+str(code))
                with self.assertRaisesRegex(api.MetaError,'provider_rejected_'+str(code)):
                    client.request(payload(),output=folder)
                self.assertFalse(client.ledger.status()['blocked'])
                self.assertEqual(client.ledger.status()['reserved_usd'],'0.000000')
                self.assertEqual(client.ledger.status()['settled_usd'],'0.000000')
                self.assertNotIn(FAKE,(folder/'meta-request.json').read_text())
                transport.assert_called_once()

    def test_server_errors_redirects_and_timeout_never_release_or_retry(self):
        transports=[Mock(return_value={'http_status':500}),Mock(return_value={'http_status':302}),Mock(side_effect=TimeoutError(FAKE))]
        for index,transport in enumerate(transports):
            client=self.client(transport);folder=self.root/('uncertain'+str(index))
            with self.assertRaises(api.MetaError) as caught:client.request(payload(),output=folder)
            self.assertNotIn(FAKE,str(caught.exception));self.assertNotIn(FAKE,(folder/'meta-request.json').read_text())
            self.assertTrue(client.ledger.status()['blocked'])
            self.assertEqual(client.ledger.status()['reserved_usd'],'1.378720')
            transport.assert_called_once()

    def test_keyboard_interrupt_retains_hold_and_blocks_new_work(self):
        client=self.client(Mock(side_effect=KeyboardInterrupt()))
        with self.assertRaises(KeyboardInterrupt):client.request(payload(),output=self.root/'cancel')
        self.assertTrue(client.ledger.status()['blocked'])
        self.assertEqual(client.ledger.status()['reserved_usd'],'1.378720')

    def test_pretransport_receipt_failure_releases_known_no_charge(self):
        transport=Mock();client=self.client(transport)
        with patch.object(api,'write',side_effect=OSError(FAKE)):
            with self.assertRaises(api.MetaError) as caught:client.request(payload(),output=self.root/'disk-before')
        self.assertNotIn(FAKE,str(caught.exception));transport.assert_not_called()
        self.assertFalse(client.ledger.status()['blocked'])
        self.assertEqual(client.ledger.status()['reserved_usd'],'0.000000')

    def test_posttransport_receipt_failure_preserves_accounted_cost(self):
        transport=Mock(return_value={'http_status':200,'document':document()});client=self.client(transport)
        original=api.write;calls=[]
        def write(path,body):
            calls.append(1)
            if len(calls)>1:raise OSError(FAKE)
            return original(path,body)
        with patch.object(api,'write',side_effect=write):
            with self.assertRaisesRegex(api.MetaError,'receipt_write_failed_cost_accounted'):
                client.request(payload(),output=self.root/'disk-after')
        self.assertEqual(client.ledger.status()['settled_usd'],'0.000177')
        self.assertEqual(client.ledger.status()['reserved_usd'],'0.000000')
        transport.assert_called_once()

    def test_settlement_exception_is_uncertain_and_no_arbitrary_text(self):
        client=self.client(Mock(return_value={'http_status':200,'document':document()}))
        with patch.object(client.ledger,'settle',side_effect=RuntimeError(FAKE)):
            with self.assertRaisesRegex(api.MetaError,'accounting_error_charge_uncertain'):
                client.request(payload(),output=self.root/'settle')
        self.assertTrue(client.ledger.status()['blocked'])
        self.assertEqual(client.ledger.status()['reserved_usd'],'1.378720')
        self.assertNotIn(FAKE,(self.root/'settle/meta-request.json').read_text())

    def test_unexpected_actual_overrun_recorded_and_blocks(self):
        client=self.client(Mock(return_value={'http_status':200,'document':document(usage={'prompt_tokens':2_000_000,'completion_tokens':1})}))
        _,receipt=client.request(payload(),output=self.root/'overrun')
        self.assertEqual(receipt['estimated_cost_usd'],'2.500005')
        self.assertTrue(client.ledger.status()['blocked'])
        self.assertEqual(client.ledger.status()['reserved_usd'],'0.000000')

    def test_unpriced_options_streaming_tools_storage_and_multiple_outputs_fail_before_transport(self):
        transport=Mock();client=self.client(transport)
        bad=[{'n':2},{'n':True},{'tools':[{}]},{'store':True},{'background':True},{'stream':True},
             {'model':'other'},{'max_completion_tokens':24001},{'max_completion_tokens':True},{'max_tokens':100000}]
        for index,updates in enumerate(bad):
            with self.assertRaises(api.MetaError):client.request(dict(payload(),**updates),output=self.root/('denied'+str(index)))
        transport.assert_not_called();self.assertEqual(client.ledger.status()['request_count'],0)

    def test_sam_minimum_frames_and_malformed_usage_are_accounted(self):
        response={'status':'completed','usage':{'video_frames_processed':50}}
        client=self.client(Mock(return_value={'http_status':200,'document':response}))
        _,receipt=client.request({'model':api.SAM.model,'stream':False,'input':[]},output=self.root/'sam',kind='sam',video_frames=10)
        self.assertEqual(receipt['estimate_usd'],'0.010000')
        self.assertEqual(receipt['estimated_cost_usd'],'0.010000')
        bad=self.client(Mock(return_value={'http_status':200,'document':{'status':'completed','usage':{'video_frames_processed':10**20}}}))
        with self.assertRaises(api.MetaError):bad.request({'model':api.SAM.model,'input':[]},output=self.root/'sam-bad',kind='sam',video_frames=10)
        self.assertTrue(bad.ledger.status()['blocked'])

    def test_sam_success_without_usage_preserves_masks_and_blocks_budget(self):
        raw = {'status':'completed','model':api.SAM.model,
               'output':[{'type':'message','content':[{'type':'output_text','text':'saved-mask-output'}]}]}
        transport=Mock(return_value={'http_status':200,'document':raw})
        client=self.client(transport);folder=self.root/'sam-missing-usage'
        with self.assertRaisesRegex(api.MetaError,'usage_missing_charge_uncertain'):
            client.request({'model':api.SAM.model,'input':[]},output=folder,kind='sam',images=1)
        saved=folder/'sam-response.pending.json'
        self.assertEqual(json.loads(saved.read_text()),raw)
        self.assertEqual(saved.stat().st_mode&0o777,0o600)
        self.assertTrue(client.ledger.status()['blocked'])
        self.assertEqual(client.ledger.status()['reserved_usd'],'0.002500')
        transport.assert_called_once()
        # HTTP errors and text generations never create a raw SAM sidecar.
        rejected=self.client(Mock(return_value={'http_status':400,'error':{'message':FAKE}}))
        with self.assertRaises(api.MetaError):
            rejected.request({'model':api.SAM.model,'input':[]},output=self.root/'sam-reject',kind='sam',images=1)
        self.assertFalse((self.root/'sam-reject/sam-response.pending.json').exists())
        text=self.client(Mock(return_value={'http_status':200,'document':document()}))
        text.request(payload(),output=self.root/'text-only')
        self.assertFalse((self.root/'text-only/sam-response.pending.json').exists())

    def test_invalid_schema_is_rejected_before_request_and_response_failure_is_sanitized(self):
        try:import PIL, jsonschema
        except ImportError:self.skipTest('optional image/schema dependencies not installed')
        client=self.client(Mock())
        with patch.object(client,'request') as request:
            with self.assertRaisesRegex(api.MetaError,'invalid_output_schema_before_transport'):
                client.json_result('test',{'type':'not-a-type'},[],output=self.root/'schema')
            request.assert_not_called()
        schema={'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False}
        client=self.client(Mock(return_value={'http_status':200,'document':document(choices=[{'finish_reason':'stop','message':{'content':json.dumps({'ok':FAKE})}}])}))
        with self.assertRaisesRegex(api.MetaError,'structured_output_schema_invalid') as caught:
            client.json_result('test',schema,[],output=self.root/'schema-response')
        self.assertNotIn(FAKE,str(caught.exception));self.assertFalse((self.root/'schema-response/response.pending.json').exists())
        self.assertEqual(client.ledger.status()['settled_usd'],'0.000177')


if __name__=='__main__':unittest.main()
