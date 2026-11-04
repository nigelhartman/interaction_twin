import unittest
from contextlib import contextmanager, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import patch
from scripts.launch_workspace import owned_response,owned_server,workspace_url,launch_url,choose_port,main


@contextmanager
def probe_server(*,redirect=False,index=b'prepared public page',health_header='InteractionTwinLocal/0.1',index_header='InteractionTwinLocal/0.1'):
    """A genuine HTTP boundary: workspace needs login; public evidence does not."""
    seen=[]
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path,dict(self.headers)))
            if self.path=='/health':
                status,body,header=200,json.dumps({'status':'ready','host':'127.0.0.1','single_worker':True,'cluster_connected':False}).encode(),health_header
            elif self.path=='/showcase/index.html':status,body,header=(302 if redirect else 200),index,index_header
            else:status,body,header=401,b'Owner login required',index_header
            self.send_response_only(status);self.send_header('Server',header);self.send_header('Content-Length',str(len(body)))
            if status==302:self.send_header('Location','/workspace/index.html')
            self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);(root/'showcase').mkdir();(root/'showcase/index.html').write_bytes(b'prepared public page')
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        try:yield server.server_port,root,seen
        finally:server.shutdown();server.server_close();worker.join(timeout=2)


class WorkspaceLaunchTests(unittest.TestCase):
    def health(self):
        return {'status':'ready','host':'127.0.0.1','single_worker':True,'cluster_connected':False}

    def test_only_matching_product_and_exact_build_are_reused(self):
        health=self.health()
        self.assertTrue(owned_response(health,'InteractionTwinLocal/0.1 Python/3.12',b'local app',b'local app'))
        self.assertFalse(owned_response(health,'SimpleHTTP/0.6',b'local app',b'local app'))
        self.assertFalse(owned_response(health,'InteractionTwinLocal/0.1',b'other checkout',b'local app'))
        health['cluster_connected']=True
        self.assertFalse(owned_response(health,'InteractionTwinLocal/0.1',b'local app',b'local app'))

    def test_only_loopback_unprivileged_numeric_urls(self):
        self.assertEqual(workspace_url(8765),'http://127.0.0.1:8765/workspace/index.html')
        for port in [80,65536,'8765',True]:
            with self.assertRaises(ValueError):workspace_url(port)

    def test_showcase_is_default_and_explicit_workspace_stays_protected(self):
        self.assertEqual(launch_url(8765),'http://127.0.0.1:8765/showcase/index.html')
        self.assertEqual(launch_url(8765,workspace=True),workspace_url(8765))
        for port in [80,65536,'8765',True]:
            with self.assertRaises(ValueError):launch_url(port)

    def test_launcher_cli_opens_showcase_or_explicit_workspace_without_new_server(self):
        report={'python':'/prepared/python','python_present':True,'ffmpeg':'/prepared/ffmpeg','ffprobe':'/prepared/ffprobe','codex':'/prepared/codex'}
        for argv,expected in [([],launch_url(8766)),(['--workspace'],workspace_url(8766))]:
            with self.subTest(argv=argv),patch('scripts.launch_workspace.runtime_report',return_value=report), \
                 patch('scripts.launch_workspace.subprocess.run',side_effect=AssertionError('Normal reopen must not rebuild artifacts')) as build, \
                 patch('scripts.launch_workspace.choose_port',return_value=(8766,True)) as choose, \
                 patch('scripts.launch_workspace.webbrowser.open') as browser, \
                 patch('scripts.launch_workspace.subprocess.Popen',side_effect=AssertionError('An owned server must be reused')), \
                 redirect_stdout(StringIO()):
                self.assertEqual(main(argv),0)
                browser.assert_called_once_with(expected)
                choose.assert_called_once_with(8766)
                build.assert_not_called()

    def test_public_probe_reuses_app_despite_workspace_401_without_auth_or_proxy(self):
        with probe_server() as (port,root,seen),patch.dict('os.environ',{'http_proxy':'http://127.0.0.1:1','HTTP_PROXY':'http://127.0.0.1:1','no_proxy':'','NO_PROXY':''}):
            self.assertTrue(owned_server(port,root))
            self.assertEqual([row[0]for row in seen],['/health','/showcase/index.html'])
            self.assertTrue(all(not any(k.lower()in{'authorization','cookie','proxy-authorization'}for k in headers)for _,headers in seen))
            with patch('scripts.launch_workspace.free_port',return_value=False):self.assertEqual(choose_port(port,root),(port,True))
            self.assertNotIn('/workspace/index.html',[row[0]for row in seen])

    def test_probe_rejects_redirect_foreign_headers_or_changed_public_bytes(self):
        for kwargs in [{'redirect':True},{'health_header':'SimpleHTTP/0.6'},{'index_header':'SimpleHTTP/0.6'},{'index':b'foreign checkout'}]:
            with self.subTest(kwargs=kwargs),probe_server(**kwargs) as (port,root,seen):
                self.assertFalse(owned_server(port,root));self.assertNotIn('/workspace/index.html',[row[0]for row in seen])

    def test_fresh_server_start_does_not_rebuild_or_run_processing(self):
        report={'python':'/prepared/python','python_present':True,'ffmpeg':None,'ffprobe':None,'codex':None}
        child=SimpleNamespace(poll=lambda:None,wait=lambda:0)
        with patch('scripts.launch_workspace.runtime_report',return_value=report), \
             patch('scripts.launch_workspace.subprocess.run',side_effect=AssertionError('No implicit build or inference')) as build, \
             patch('scripts.launch_workspace.choose_port',return_value=(8766,False)), \
             patch('scripts.launch_workspace.owned_server',return_value=True), \
             patch('scripts.launch_workspace.subprocess.Popen',return_value=child) as popen, \
             redirect_stdout(StringIO()):
            self.assertEqual(main(['--no-browser']),0);build.assert_not_called();self.assertEqual(popen.call_count,1)
            command=popen.call_args.args[0];self.assertEqual(Path(command[1]).name,'serve_workspace.py');self.assertEqual(command[-4:],['--host','127.0.0.1','--port','8766'])

    def test_explicit_rebuild_is_opt_in_and_failure_stops_before_server_start(self):
        report={'python':'/prepared/python','python_present':True,'ffmpeg':None,'ffprobe':None,'codex':None}
        for code in [0,7]:
            with self.subTest(code=code),patch('scripts.launch_workspace.runtime_report',return_value=report), \
                 patch('scripts.launch_workspace.subprocess.run',return_value=SimpleNamespace(returncode=code)) as build, \
                 patch('scripts.launch_workspace.choose_port',return_value=(8766,True)) as choose, \
                 patch('scripts.launch_workspace.subprocess.Popen',side_effect=AssertionError('No new server')),redirect_stdout(StringIO()):
                self.assertEqual(main(['--no-browser','--rebuild-workspace']),code);self.assertEqual(build.call_count,1)
                self.assertEqual(Path(build.call_args.args[0][1]).name,'build_workspace.py')
                self.assertEqual(choose.call_count,0 if code else 1)

    @patch('scripts.launch_workspace.free_port',return_value=True)
    @patch('scripts.launch_workspace.owned_server',side_effect=lambda p,r:p==8767)
    def test_reuses_existing_owned_app_before_choosing_an_earlier_free_port(self,owned,free):
        self.assertEqual(choose_port(8765),(8767,True))

    @patch('scripts.launch_workspace.free_port',side_effect=lambda p:p==8766)
    @patch('scripts.launch_workspace.owned_server',return_value=False)
    def test_busy_unrelated_port_is_left_untouched(self,owned,free):
        self.assertEqual(choose_port(8765),(8766,False))

if __name__=='__main__':unittest.main()
