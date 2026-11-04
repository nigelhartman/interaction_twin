"""Actual workspace owner gate + private studio integration. No credentials/API."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import admin_studio_http as H
from marketing_studio import MarketingStudio
from marketing_sources import ROOT,file_ref
from tests import test_admin_http_security as security
from tests.test_marketing_studio import Sources,NoMusic,FakeClient

class AdminStudioHTTPTests(unittest.TestCase):
    setUpSecurity=security.AdminHTTPSecurityTests.setUp
    request=security.AdminHTTPSecurityTests.request
    owner=security.AdminHTTPSecurityTests.owner
    def setUp(self):
        self.setUpSecurity();(self.root/'config').mkdir()
        shutil.copy(ROOT/'config/marketing-brand.json',self.root/'config/marketing-brand.json');shutil.copy(ROOT/'showcase/mark.svg',self.root/'showcase/mark.svg')
        (self.root/'sample.mp4').write_bytes(b'PRIVATE-APPROVED-MARKETING-SOURCE')
        self.studio=MarketingStudio(self.root,sources=Sources(self.root),music=NoMusic());self.server.marketing_studio=self.studio
        self.queued=[];self.submit=patch.object(H,'_submit',side_effect=lambda server,work:self.queued.append(work))
        self.submit.start();self.addCleanup(self.submit.stop)
        self.addCleanup(lambda:getattr(self.server,'marketing_executor',SimpleNamespace(shutdown=lambda **_:None)).shutdown(wait=True,cancel_futures=True))
    def create(self):
        _,calls,_=self.request(H.BASE+'/campaigns','POST',{'title':'HTTP fixture','source_ids':['capture'],'music_id':None},headers=self.owner())
        self.assertEqual(calls[0][0],201);return calls[0][1]
    def fake_render(self,rev,fmt,out,*,root,progress):
        out.mkdir();result={}
        for key,name in [('video','film.mp4'),('poster','poster.jpg'),('captions','captions.vtt'),('manifest','manifest.json')]:
            p=out/name;p.write_bytes(('private '+key).encode());result[key]=file_ref(p,root)
        return result|{'format_id':fmt,'duration_s':30,'audio':False}
    def test_all_new_reads_are_private_before_any_adapter_work(self):
        routes=[H.BASE+'/sources',H.BASE+'/campaigns',H.BASE+'/music','/api/admin/operations',H.BASE+'/sources/capture/video',H.BASE+'/music/project_original/audio',H.BASE+'/campaigns/campaign_'+'a'*64+'/revisions/revision_'+'b'*64+'/video']
        with patch.object(H,'_state',side_effect=AssertionError('Unauthenticated route initialized studio')):
            for route in routes:
                with self.subTest(route=route):
                    _,calls,_=self.request(route);self.assertEqual(calls[0][0],401)
        self.assertEqual(self.queued,[])
    def test_owner_gets_never_queue_models_ssh_or_export_and_list_is_array(self):
        d=self.create();self.queued.clear();cid=d['campaign']['id']
        with patch.object(self.studio,'run_next',side_effect=AssertionError('GET rendered')),patch.object(self.studio,'propose_with_muse',side_effect=AssertionError('GET reasoned')),patch.object(self.studio,'export_revision',side_effect=AssertionError('GET exported')),patch('admin_operations._query_gpu',side_effect=AssertionError('GET used SSH')),patch('meta_api.MetaClient.__init__',side_effect=AssertionError('GET loaded modelclient')):
            for route in [H.BASE+'/sources',H.BASE+'/campaigns',H.BASE+'/campaigns/'+cid,'/api/admin/operations']:
                _,calls,_=self.request(route,headers={'Cookie':self.cookie});self.assertEqual(calls[0][0],200)
                if route==H.BASE+'/campaigns':self.assertIsInstance(calls[0][1]['campaigns'],list)
        self.assertEqual(self.queued,[])
    def test_mutation_origin_csrf_and_raw_upload_checked_before_body(self):
        raw=b'PRIVATE AUDIO FIXTURE'
        for headers in [{'Origin':'http://127.0.0.1:8765'},self.owner(Origin='https://foreign.invalid'),{'Cookie':self.cookie,'Origin':'http://127.0.0.1:8765'},self.owner(**{'X-CSRF-Token':'invalid'})]:
            with self.subTest(headers=list(headers)),patch('marketing_music.MusicStore',side_effect=AssertionError('Unauthorized upload created store')):
                h,calls,_=self.request(H.BASE+'/music/upload','POST',raw,headers={**headers,'Content-Type':'audio/wav','X-Filename':'fixture.wav'})
                self.assertIn(calls[0][0],{401,403});self.assertEqual(h.rfile.tell(),0)
        for route in [H.BASE+'/campaigns','/api/admin/operations/gpu/refresh']:
            _,calls,_=self.request(route,'POST',{},headers=self.owner(Origin='https://foreign.invalid'));self.assertEqual(calls[0][0],403)
        self.assertEqual(self.queued,[])
    def test_explicit_authenticated_audio_upload_only_passes_exact_bytes(self):
        with patch('marketing_music.MusicStore')as cls:
            cls.return_value.upload.return_value={'upload_id':'audio_'+'a'*64,'registered_for_rendering':False}
            _,calls,_=self.request(H.BASE+'/music/upload','POST',b'fixture-audio',headers=self.owner(**{'Content-Type':'audio/wav','X-Filename':'fixture.wav'}))
        self.assertEqual(calls[0][0],201);cls.return_value.upload.assert_called_once_with(b'fixture-audio','fixture.wav');self.assertEqual(self.queued,[])
    def test_strict_mutation_fieldsets_preserve_exact_lineage(self):
        d=self.create();cid=d['campaign']['id'];rid=d['latest_revision']['id'];base=H.BASE+'/campaigns/'+cid
        payloads=[(H.BASE+'/campaigns',{'title':'Bad','source_ids':['capture'],'execute':'not allowed'}),(base+'/generate',{'base_revision_id':rid,'anything':True}),(base+'/feedback',{'revision_id':rid,'text':'Hi','actor':'another'}),(base+'/approve',{'revision_id':rid,'autoapprove':True}),(base+'/export',{'revision_id':rid,'path':'../../out'}),(base+'/muse',{'base_revision_id':rid,'allow_paid_api':True,'prompt':'override'}),('/api/admin/operations/gpu/refresh',{'shell':'anything'})]
        before=len(self.queued)
        for route,body in payloads:
            with self.subTest(route=route):
                _,calls,_=self.request(route,'POST',body,headers=self.owner());self.assertEqual(calls[0][0],400)
        self.assertEqual(len(self.queued),before);self.assertEqual(len(self.studio.detail(cid)['revisions']),1)
        _,fb,_=self.request(base+'/feedback','POST',{'revision_id':rid,'text':'Less text','preferences':{'caption_density':'minimal'}},headers=self.owner())
        _,new,_=self.request(base+'/generate','POST',{'base_revision_id':rid,'feedback_id':fb[0][1]['id']},headers=self.owner());self.assertEqual(new[0][0],202)
        value=new[0][1]['latest_revision'];self.assertEqual(value['base_revision_id'],rid);self.assertEqual(value['feedback_id'],fb[0][1]['id'])
    def test_exact_asset_auth_download_gate_and_get_never_creates_zip(self):
        d=self.create();cid=d['campaign']['id'];rid=d['latest_revision']['id'];base=H.BASE+'/campaigns/'+cid
        download=base+'/revisions/'+rid+'/download';video=base+'/revisions/'+rid+'/video'
        _,calls,_=self.request(video,headers={'Cookie':self.cookie});self.assertEqual(calls[0][0],404)
        with patch('marketing_render.render',side_effect=self.fake_render):self.studio.run_job(d['jobs'][0]['id'])
        h,_,sent=self.request(video,headers={'Cookie':self.cookie});self.assertEqual(sent[0],('status',200));self.assertEqual(h.wfile.getvalue(),b'private video')
        _,calls,_=self.request(video);self.assertEqual(calls[0][0],401)
        _,calls,_=self.request(download,headers={'Cookie':self.cookie});self.assertEqual(calls[0][0],404)
        _,calls,_=self.request(base+'/approve','POST',{'revision_id':rid},headers=self.owner());self.assertEqual(calls[0][0],200)
        _,calls,_=self.request(download,headers={'Cookie':self.cookie});self.assertEqual(calls[0][0],404)
        _,calls,_=self.request(base+'/export','POST',{'revision_id':rid},headers=self.owner());self.assertEqual(calls[0][0],201)
        h,_,sent=self.request(download,headers={'Cookie':self.cookie});self.assertEqual(sent[0],('status',200));self.assertTrue(h.wfile.getvalue().startswith(b'PK'))
        self.assertTrue(any(k=='Content-Disposition'and'attachment'in v for k,v in sent))
        _,calls,_=self.request(download);self.assertEqual(calls[0][0],401)
    def test_async_muse_exact_identity_is_idempotent_before_any_new_preview(self):
        d=self.create();cid=d['campaign']['id'];rid=d['latest_revision']['id'];base=H.BASE+'/campaigns/'+cid
        body={'base_revision_id':rid,'allow_paid_api':True};self.queued.clear();previews=[]
        def preview(*args,**kwargs):
            previews.append((args,kwargs))
            if len(previews)>1:raise ValueError('Same original attempt now in progress; do not resend')
            return {'status':'prepared_preview_only'}
        with patch.object(self.studio,'propose_with_muse',side_effect=preview),patch('overnight_budget.OvernightMetaClient',side_effect=AssertionError('Scheduling must not construct a paidclient')):
            _,first,_=self.request(base+'/muse','POST',body,headers=self.owner());self.assertEqual(first[0][0],202)
            _,again,_=self.request(base+'/muse','POST',body,headers=self.owner());self.assertEqual(again[0][0],202)
        self.assertEqual(first[0][1]['ai_task']['id'],again[0][1]['ai_task']['id']);self.assertEqual(len(previews),1);self.assertEqual(len(self.queued),1)
        self.assertEqual(previews[0][0],(cid,rid));self.assertIs(previews[0][1]['allow_paid_api'],False)
    def test_historical_unresolved_muse_attempt_never_gets_resubmitted(self):
        d=self.create();cid=d['campaign']['id'];rid=d['latest_revision']['id'];self.queued.clear()
        with patch.object(self.studio,'propose_with_muse',side_effect=ValueError('Retained unknown attempt; no resend')),patch('overnight_budget.OvernightMetaClient',side_effect=AssertionError('No paidclient')):
            _,calls,_=self.request(H.BASE+'/campaigns/'+cid+'/muse','POST',{'base_revision_id':rid,'allow_paid_api':True},headers=self.owner())
        self.assertEqual(calls[0][0],400);self.assertEqual(self.queued,[])
    def test_async_worker_keeps_exact_base_and_feedback_even_if_newer_revision_exists(self):
        d=self.create();cid=d['campaign']['id'];rid=d['latest_revision']['id'];self.queued.clear()
        fb=self.studio.add_feedback(cid,rid,'Use the agent section.');calls=[];client=object()
        def proposer(*args,**kwargs):
            calls.append((args,kwargs))
            return self.studio.detail(cid)if kwargs['allow_paid_api']else{'status':'prepared_preview_only'}
        with patch.object(self.studio,'propose_with_muse',side_effect=proposer),patch('overnight_budget.OvernightMetaClient',return_value=client),patch.object(H,'_render')as drain:
            _,response,_=self.request(H.BASE+'/campaigns/'+cid+'/muse','POST',{'base_revision_id':rid,'feedback_id':fb['id'],'allow_paid_api':True},headers=self.owner())
            self.assertEqual(response[0][0],202);self.assertEqual(len(self.queued),1)
            newer=self.studio.generate(cid,base_revision_id=rid,feedback_id=fb['id'])['latest_revision']['id']
            self.queued[0]();drain.assert_called_once_with(self.studio)
        self.assertEqual(calls[-1][0],(cid,rid));self.assertEqual(calls[-1][1]['feedback_id'],fb['id']);self.assertIs(calls[-1][1]['client'],client)
        task=next(iter(self.server.marketing_tasks.values()));self.assertEqual(task['status'],'complete');self.assertEqual(task['base_revision_id'],rid);self.assertEqual(task['revision_id'],newer)
    def test_completed_durable_muse_result_restores_exact_task_after_empty_memory_without_dispatch(self):
        d=self.create();cid=d['campaign']['id'];rid=d['latest_revision']['id'];fb=self.studio.add_feedback(cid,rid,'Keep the evidence and agent handoff.')
        proposal={'storyboard':d['latest_revision']['storyboard'],'copy_spec':{'hook_id':'agents','body_fact_ids':['agents','connections']},'creative_notes':['Stored completed fixture']}
        client=FakeClient([proposal,{'verdict':'usable','issues':[]}])
        completed=self.studio.propose_with_muse(cid,rid,feedback_id=fb['id'],client=client,allow_paid_api=True)
        generated=completed['latest_revision']['id'];self.assertEqual(len(client.calls),2)
        # A later presentation draft must not be mistaken for the exact output
        # of the original completed model action when its task cache is gone.
        refreshed=self.studio.refresh_presentation(cid,generated)['latest_revision']['id'];self.assertNotEqual(refreshed,generated)
        self.server.marketing_tasks.clear();self.queued.clear()
        body={'base_revision_id':rid,'feedback_id':fb['id'],'allow_paid_api':True};route=H.BASE+'/campaigns/'+cid
        expected='edit_'+hashlib.sha256(json.dumps([cid,rid,fb['id']],separators=(',',':')).encode()).hexdigest()
        with patch.object(H,'_submit',side_effect=AssertionError('Completed action queued newwork')),patch('overnight_budget.OvernightMetaClient',side_effect=AssertionError('Completed action constructed APIclient')),patch.object(self.studio,'run_next',side_effect=AssertionError('Read/restoration rendered')):
            _,response,_=self.request(route+'/muse','POST',body,headers=self.owner());self.assertEqual(response[0][0],202)
            task=response[0][1]['ai_task'];self.assertEqual(task['id'],expected);self.assertEqual(task['status'],'complete');self.assertTrue(task['restored_from_saved_result'])
            self.assertEqual(task['base_revision_id'],rid);self.assertEqual(task['feedback_id'],fb['id']);self.assertEqual(task['revision_id'],generated)
            _,read,_=self.request(route,headers={'Cookie':self.cookie});self.assertEqual(read[0][0],200);self.assertEqual(read[0][1]['ai_task'],task)
            _,again,_=self.request(route+'/muse','POST',body,headers=self.owner());self.assertEqual(again[0][0],202);self.assertEqual(again[0][1]['ai_task']['id'],expected)
        self.assertEqual(self.queued,[]);self.assertEqual(len(client.calls),2)

if __name__=='__main__':unittest.main()
