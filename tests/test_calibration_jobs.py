"""Local calibration admission/supervision tests. Fake children do no inference."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
import calibration_jobs as J

PROJECT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('calibration_http_contract_server',PROJECT/'scripts/serve_workspace.py')
SERVER=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(SERVER)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name).resolve()
        self.source=self.root/'data/uploads/upload-0123456789abcdef/board clip.mov';self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b'Synthetic process-contract fixture, never decoded')
        for name in J.CODE:
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# Synthetic pinned child source; never executed\n')
        self.body={'input_path':self.source.relative_to(self.root).as_posix(),'inner_rows':6,'inner_cols':9,'square_size_mm':None}

    def tearDown(self):self.temporary.cleanup()
    def runner(self):return J.CalibrationRunner(J.canonical_request(self.body,root=self.root),root=self.root)
    def report(self,runner,*,passed=False):
        runner.result.mkdir(exist_ok=True)
        report={'format':'checkerboard-camera-calibration','version':1,'status':'candidate_passed_checks' if passed else 'insufficient_evidence',
            'source':{'kind':'video','path':self.body['input_path'],'sha256':J.sha(self.source)},'image_size':[1280,960],
            'physical_accuracy_verified':False,'object_metric_scale_established':False,'new_api_calls':0,
            'camera_record':None,'diagnostics':[{'code':'synthetic_fixture_no_board'}]}
        if passed:
            provenance={}
            for key,name in [('protocol','protocol.json'),('detections','detections.json'),('implementation','implementation.py')]:
                p=runner.result/name;p.write_text('Synthetic evidence artifact, never real detections')
                provenance[key]={'path':p.relative_to(self.root).as_posix(),'sha256':J.sha(p)}
            report['provenance']=provenance
            camera={'format':'explicit-checkerboard-camera-record','version':1,'physical_accuracy_verified':False,
                'object_metric_scale_established':False,'metric_scale_verified':False,'heldout_checks_passed':True,
                'source':copy.deepcopy(report['source']),'board':{'inner_rows':6,'inner_cols':9,'square_size_mm':None,'board_pose_output_units':'grid_units','metric_scale_verified':False},
                'image_size':[1280,960],'K':[[900.,0.,640.],[0.,910.,480.],[0.,0.,1.]],
                'distortion_model':'opencv_brown_k1_k2_p1_p2_fixed_k3_zero','distortion_coefficients':[0.,0.,0.,0.,0.],
                'provenance':provenance}
            p=runner.result/'camera-calibration.json';p.write_text(json.dumps(camera));report['camera_record']={'path':p.relative_to(self.root).as_posix(),'sha256':J.sha(p)}
        (runner.result/'report.json').write_text(json.dumps(report));return report
    def finished_process(self,code=0):
        class Process:
            def wait(self):return code
            def poll(self):return code
        return Process()
    def run_with_report(self,runner,*,passed=False,mutate=None):
        def launch(argv,log):
            self.report(runner,passed=passed)
            if mutate:mutate(runner)
            return self.finished_process()
        with patch.object(runner,'_launch',side_effect=launch) as mocked:runner.execute()
        return mocked


class AdmissionTests(Fixture):
    def test_uploaded_only_exact_fields_bounded_board_and_optional_units(self):
        canonical=J.canonical_request(self.body,root=self.root)
        self.assertEqual(canonical['source_sha256'],J.sha(self.source));self.assertIsNone(canonical['square_size_mm'])
        for body in [dict(self.body,input_path='/etc/passwd'),dict(self.body,input_path='data/uploads/upload-0123456789abcdef/../board.mov'),
            dict(self.body,input_path='https://example.invalid/video.mov'),dict(self.body,input_path='data/other.mov'),
            dict(self.body,gpu=True),dict(self.body,inner_rows=True),dict(self.body,inner_cols=17),dict(self.body,inner_rows=16,inner_cols=16),
            dict(self.body,square_size_mm=True),dict(self.body,square_size_mm=float('nan')),dict(self.body,square_size_mm=0)]:
            with self.subTest(body=body),self.assertRaises(ValueError):J.canonical_request(body,root=self.root)
        measured=J.canonical_request(dict(self.body,square_size_mm=24.5),root=self.root)
        self.assertEqual(measured['square_size_mm'],24.5)

    def test_source_code_output_and_lock_ancestor_symlinks_rejected(self):
        other=self.root/'data/other.mov';other.write_bytes(b'outside fixture')
        self.source.unlink();self.source.symlink_to(other)
        with self.assertRaises(ValueError):J.canonical_request(self.body,root=self.root)
        self.source.unlink();self.source.write_bytes(b'fixture restored')
        runner=self.runner();lockdir=self.root/'data/flow-runs';outside=self.root/'elsewhere';outside.mkdir();lockdir.symlink_to(outside,target_is_directory=True)
        with patch.object(runner,'_launch') as launch:runner.execute()
        launch.assert_not_called();self.assertEqual(runner.state['status'],'failed');self.assertFalse((outside/'.model-worker.lock').exists())
        lockdir.unlink()
        base=self.root/J.BASE
        for p in base.iterdir():
            if p.is_dir():
                import shutil;shutil.rmtree(p)
        base.rmdir();base.symlink_to(outside,target_is_directory=True)
        with self.assertRaises(ValueError):self.runner()


class SupervisionTests(Fixture):
    def test_expected_insufficient_evidence_is_completed_not_retried(self):
        runner=self.runner()
        def launch(argv,log):
            self.assertEqual(argv[:3],[str(self.root/'.venv-sam/bin/python'),'-B',str(self.root/'scripts/calibrate_camera.py')])
            self.assertEqual(argv[argv.index('--input')+1],self.body['input_path'])
            self.assertEqual(argv[argv.index('--out')+1],runner.result.relative_to(self.root).as_posix())
            self.assertNotIn('--square-size-mm',argv)
            self.report(runner);return self.finished_process()
        with patch.object(runner,'_launch',side_effect=launch) as mocked:runner.execute()
        mocked.assert_called_once();self.assertEqual(runner.state['status'],'completed')
        self.assertEqual(runner.state['outcome'],'insufficient_evidence');self.assertIsNone(runner.state['camera_record_url'])
        self.assertTrue(runner.state['result_url'].endswith('/report.json'));self.assertEqual(J.POLICY['maximum_runtime_s'],120)
        self.assertFalse(runner.state['policy']['automatic_retry'])
        self.assertTrue((runner.directory/'calibration_core.py').exists())

    def test_passing_download_is_source_numeric_and_evidence_bound(self):
        runner=self.runner();self.run_with_report(runner,passed=True).assert_called_once()
        self.assertEqual(runner.state['status'],'completed');self.assertTrue(runner.state['camera_record_url'].endswith('/camera-calibration.json'))
        report=json.loads((runner.result/'report.json').read_text());camera=json.loads((runner.result/'camera-calibration.json').read_text())
        for mutate in [lambda c:c['K'][0].__setitem__(0,float('nan')),lambda c:c['K'][2].__setitem__(2,2),
            lambda c:c.__setitem__('image_size',[640,480]),lambda c:c['source'].__setitem__('sha256','0'*64),
            lambda c:c['board'].__setitem__('board_pose_output_units','mm'),lambda c:c.__setitem__('physical_accuracy_verified',True),
            lambda c:c.__setitem__('distortion_coefficients',[0,0,0,0]),lambda c:c['provenance']['protocol'].__setitem__('path','/etc/passwd')]:
            bad=copy.deepcopy(camera);mutate(bad)
            with self.subTest(camera=bad),self.assertRaises(ValueError):J.validate_camera(bad,report,runner.request,root=self.root,result=runner.result)
        p=runner.result/'detections.json';p.write_text('corrupted')
        with self.assertRaises(ValueError):J.validate_camera(camera,report,runner.request,root=self.root,result=runner.result)

    def test_wrong_source_failure_and_passing_record_absence_not_published(self):
        for kind in ['other_source','missing_camera','bad_claim']:
            runner=self.runner()
            def mutate(r):
                p=r.result/'report.json';v=json.loads(p.read_text())
                if kind=='other_source':v['source']['sha256']='0'*64
                elif kind=='missing_camera':v['status']='candidate_passed_checks'
                else:v['physical_accuracy_verified']=True
                p.write_text(json.dumps(v))
            self.run_with_report(runner,mutate=mutate)
            self.assertEqual(runner.state['status'],'failed');self.assertIsNone(runner.state['result_url']);self.assertIsNone(runner.state['camera_record_url'])
            self.assertTrue((runner.result/'report.json').exists())

    def test_mutation_before_and_during_child_is_preserved_without_retry(self):
        for stage in ['before','during_source','during_code']:
            runner=self.runner()
            if stage=='before':self.source.write_bytes(self.source.read_bytes()+b' changed')
            def launch(argv,log):
                self.report(runner)
                target=self.source if stage=='during_source' else self.root/J.CODE[0]
                target.write_bytes(target.read_bytes()+b' changed');return self.finished_process()
            with patch.object(runner,'_launch',side_effect=launch) as mocked:runner.execute()
            self.assertEqual(mocked.call_count,0 if stage=='before' else 1);self.assertEqual(runner.state['status'],'failed')
            self.assertIsNone(runner.state['camera_record_url'])

    def test_cancel_before_start_does_not_launch_and_preserves_new_attempt(self):
        runner=self.runner();runner.cancel()
        with patch.object(runner,'_launch') as mocked:runner.execute()
        mocked.assert_not_called();self.assertEqual(runner.state['status'],'cancelled')
        previous=(runner.directory/'job.json').read_bytes();new=self.runner()
        self.assertNotEqual(new.id,runner.id);self.assertEqual((runner.directory/'job.json').read_bytes(),previous)

    def test_shared_file_lock_prevents_an_external_worker_collision(self):
        runner=self.runner();path=self.root/'data/flow-runs/.model-worker.lock';path.parent.mkdir(parents=True)
        with path.open('a+') as locked:
            J.fcntl.flock(locked.fileno(),J.fcntl.LOCK_EX|J.fcntl.LOCK_NB)
            with patch.object(runner,'_launch') as launch:runner.execute()
        launch.assert_not_called();self.assertEqual(runner.state['status'],'failed')
        self.assertIn('shared worker lock',runner.state['error'])

    def test_active_cancel_preserves_partial_files_and_only_terminates_child(self):
        runner=self.runner();started=threading.Event();stopped=threading.Event()
        class Process:
            code=None
            def wait(self):started.set();stopped.wait(3);return self.code
            def poll(self):return self.code
            def terminate(self):self.code=-15;stopped.set()
            def kill(self):self.code=-9;stopped.set()
        def launch(argv,log):runner.result.mkdir();(runner.result/'partial.json').write_text('{"fixture":true}');return Process()
        with patch.object(runner,'_launch',side_effect=launch) as mocked:
            thread=threading.Thread(target=runner.execute);thread.start();self.assertTrue(started.wait(2));runner.cancel();thread.join(2)
        self.assertFalse(thread.is_alive());mocked.assert_called_once();self.assertEqual(runner.state['status'],'cancelled')
        self.assertTrue((runner.result/'partial.json').exists());self.assertIsNone(runner.state['result_url'])

    def test_timeout_terminates_worker_without_retry(self):
        runner=self.runner();stopped=threading.Event()
        class Process:
            code=None
            def wait(self):stopped.wait(2);return self.code
            def poll(self):return self.code
            def terminate(self):self.code=-15;stopped.set()
            def kill(self):self.code=-9;stopped.set()
        with patch.object(runner,'_launch',return_value=Process()) as mocked,patch.dict(J.POLICY,maximum_runtime_s=.02):runner.execute()
        mocked.assert_called_once();self.assertEqual(runner.state['status'],'timed_out');self.assertIsNone(runner.state['result_url'])

    def test_failed_child_preserves_log_and_does_not_fallback(self):
        runner=self.runner()
        with patch.object(runner,'_launch',return_value=self.finished_process(2)) as mocked:runner.execute()
        mocked.assert_called_once();self.assertEqual(runner.state['status'],'failed');self.assertTrue((runner.directory/'worker.log').exists())

    def test_cancel_during_result_validation_never_publishes_camera_link(self):
        runner=self.runner();original=J.validate_camera
        def validate(*args,**kwargs):original(*args,**kwargs);runner.cancel()
        with patch.object(J,'validate_camera',side_effect=validate):self.run_with_report(runner,passed=True)
        self.assertEqual(runner.state['status'],'cancelled');self.assertIsNone(runner.state['camera_record_url']);self.assertIsNone(runner.state['result_url'])
        self.assertTrue((runner.result/'camera-calibration.json').exists())

    def test_process_transport_is_fixed_local_shell_free_and_strips_inherited_environment(self):
        runner=self.runner()
        with (runner.directory/'test.log').open('wb') as log,patch.object(J.subprocess,'Popen') as process:
            runner._launch(['/fixed/python','/fixed/script','--input','local clip.mov'],log)
        process.assert_called_once();kwargs=process.call_args.kwargs
        self.assertIs(kwargs['shell'],False);self.assertEqual(kwargs['cwd'],self.root)
        self.assertEqual(set(kwargs['env']),{'PATH','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'})
        self.assertEqual(kwargs['stdin'],J.subprocess.DEVNULL)

    def test_stale_active_read_is_interrupted_without_mutation(self):
        runner=self.runner();runner.state['status']='running';runner.save();before=(runner.directory/'job.json').read_bytes()
        read=J.record(runner.id,root=self.root);self.assertEqual(read['status'],'interrupted');self.assertEqual(read['saved_status'],'running')
        self.assertEqual(J.record(runner.id,root=self.root,active_job_id=runner.id)['status'],'running')
        self.assertEqual((runner.directory/'job.json').read_bytes(),before)
        with self.assertRaises(ValueError):J.record('../escape',root=self.root)


class HTTPContracts(Fixture):
    def setUp(self):
        super().setUp();self.root_patch=patch.object(SERVER,'ROOT',self.root);self.root_patch.start()
        self.server=SERVER.WorkspaceServer(('127.0.0.1',0),SERVER.Handler)
        from tests.owner_http_fixture import OwnerHTTPSession
        self.owner_http=OwnerHTTPSession(self.server);self.addCleanup(self.owner_http.close)
        self.worker=threading.Thread(target=self.server.serve_forever,daemon=True);self.worker.start()
        self.base='http://127.0.0.1:'+str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.worker.join(2)
        self.root_patch.stop();super().tearDown()

    def request(self,path,body=None,headers=None):
        request=urllib.request.Request(self.base+path,data=None if body is None else json.dumps(body).encode(),
            headers=headers or {'Content-Type':'application/json','Origin':self.base})
        with urllib.request.urlopen(request,timeout=5) as response:return response.status,json.load(response)

    def test_origin_host_json_and_confinement_guards_cover_calibration_route(self):
        for headers,body,code in [
            ({'Content-Type':'application/json','Origin':'https://attacker.invalid'},self.body,403),
            ({'Content-Type':'application/json','Host':'attacker.invalid'},self.body,403),
            ({'Content-Type':'text/plain'},self.body,400),
            ({'Content-Type':'application/json'},dict(self.body,input_path='/etc/passwd'),400)]:
            with self.subTest(headers=headers),self.assertRaises(urllib.error.HTTPError) as caught:self.request('/api/calibration/jobs',body,headers)
            self.assertEqual(caught.exception.code,code)
        self.assertEqual(self.request('/api/calibration/jobs')[1]['jobs'],[])

    def test_shared_worker_slot_status_and_cancel_use_actual_routes(self):
        class Busy:
            id='other-worker';state={'status':'running'}
        self.server.active=Busy()
        with self.assertRaises(urllib.error.HTTPError) as caught:self.request('/api/calibration/jobs',self.body)
        self.assertEqual(caught.exception.code,409);self.server.active=None
        with patch.object(J.CalibrationRunner,'execute',return_value=None) as execute:
            code,created=self.request('/api/calibration/jobs',self.body)
        self.assertEqual(code,202);execute.assert_called_once()
        self.assertEqual(self.request(created['status_url'])[1]['status'],'queued')
        with self.assertRaises(urllib.error.HTTPError) as caught:self.request('/api/calibration/jobs',self.body)
        self.assertEqual(caught.exception.code,409)
        with self.assertRaises(urllib.error.HTTPError) as caught:self.request(created['status_url']+'/cancel',{'unaccepted':True})
        self.assertEqual(caught.exception.code,400)
        self.assertEqual(self.request(created['status_url']+'/cancel',{})[0],202)
        self.assertTrue(self.server.active.cancelled.is_set())

if __name__=='__main__':unittest.main()
