#!/usr/bin/env python3
"""Launch the prepared local workspace without installation or remote access."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
import webbrowser

ROOT=Path(__file__).resolve().parents[1]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, newurl):
        return None


def workspace_url(port):
    if type(port) is not int or not 1024<=port<=65535:
        raise ValueError('Use an unprivileged numeric local port')
    return f'http://127.0.0.1:{port}/workspace/index.html'


def launch_url(port,workspace=False,admin=False):
    verified=workspace_url(port)
    if admin:return verified.replace('/workspace/index.html','/admin/')
    return verified if workspace else verified.replace('/workspace/index.html','/showcase/index.html')


def owned_response(health,server_header,index_bytes,local_index):
    """Recognize this exact product build, never a generic HTTP health page."""
    return (isinstance(health,dict) and health.get('status')=='ready'
            and health.get('host')=='127.0.0.1' and health.get('single_worker') is True
            and health.get('cluster_connected') is False
            and server_header.startswith('InteractionTwinLocal/')
            and hashlib.sha256(index_bytes).digest()==hashlib.sha256(local_index).digest())


def owned_server(port,root=ROOT):
    """No proxy, redirects, credentials, environment files or non-loopback I/O."""
    opener=build_opener(ProxyHandler({}),NoRedirect())
    try:
        public_url=launch_url(port)
        base=f'http://127.0.0.1:{port}'
        with opener.open(Request(base+'/health'),timeout=.6) as response:
            header=response.headers.get('Server','');health=json.loads(response.read(4097))
        with opener.open(Request(public_url),timeout=.6) as response:
            if not response.headers.get('Server','').startswith('InteractionTwinLocal/'):
                return False
            index=response.read(4*1024*1024+1)
        local=(Path(root)/'showcase/index.html').read_bytes()
        return owned_response(health,header,index,local)
    except HTTPError as exc:
        exc.close()
        return False
    except (OSError,ValueError,URLError):
        return False


def free_port(port):
    try:
        with socket.socket() as listener:listener.bind(('127.0.0.1',port))
        return True
    except OSError:return False


def choose_port(start=8766,root=ROOT):
    workspace_url(start)
    free=None
    for port in range(start,min(start+20,65536)):
        if owned_server(port,root):return port,True
        if free is None and free_port(port):free=port
    if free is not None:return free,False
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));return listener.getsockname()[1],False


def runtime_report(root=ROOT):
    root=Path(root)
    python=root/'.venv-sam/bin/python'
    known_dirs=['/opt/homebrew/bin','/usr/local/bin','/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS']
    search_path=os.pathsep.join(known_dirs+[os.environ.get('PATH','')])
    checks={'python':str(python),'python_present':python.is_file(),
            'ffmpeg':shutil.which('ffmpeg',path=search_path),'ffprobe':shutil.which('ffprobe',path=search_path),
            'codex':shutil.which('codex',path=search_path),'credentials_inspected':False,
            'downloads_or_installs':False,'remote_access':False}
    return checks


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8766)
    parser.add_argument('--no-browser',action='store_true',help='Print URL without opening a browser')
    parser.add_argument('--workspace',action='store_true',help='Open the protected workspace instead of the project showcase; owner login is required')
    parser.add_argument('--admin',action='store_true',help='Open the private owner studio')
    parser.add_argument('--rebuild-workspace',action='store_true',help='Explicitly rebuild local workspace review artifacts before opening; no model API calls')
    parser.add_argument('--check',action='store_true',help='Print prerequisite presence; do not start a server')
    args=parser.parse_args(argv)
    if args.admin and args.workspace:parser.error('Choose the private studio or workspace')
    try:workspace_url(args.port)
    except ValueError as exc:parser.error(str(exc))
    checks=runtime_report()
    print('Interaction Twin · local workspace',flush=True)
    print(json.dumps(checks,indent=2),flush=True)
    if args.check:return 0 if checks['python_present'] else 2
    if not checks['python_present']:
        print('Prepared .venv-sam/bin/python is missing. No installation was attempted.',file=sys.stderr);return 2
    missing=[name for name in ['ffmpeg','ffprobe','codex'] if not checks[name]]
    if missing:
        print('Fresh processing prerequisites not on PATH: '+', '.join(missing)+'. Existing evidence can still be inspected. Nothing will be installed automatically.',flush=True)
    python=checks['python']
    task_env=os.environ.copy()
    public_dirs=list(dict.fromkeys(str(Path(checks[name]).parent) for name in ['ffmpeg','ffprobe','codex'] if checks[name]))
    task_env['PATH']=os.pathsep.join(public_dirs+[task_env.get('PATH','')])
    if args.rebuild_workspace:
        print('Explicit local workspace rebuild requested; saved review artifacts may be updated. No model API calls are requested.',flush=True)
        build=subprocess.run([python,str(ROOT/'scripts/build_workspace.py')],cwd=ROOT,env=task_env,check=False)
        if build.returncode:
            print('Workspace build failed; see diagnostics above. No server was started.',file=sys.stderr);return build.returncode
    port,reuse=choose_port(args.port)
    url=launch_url(port,workspace=args.workspace,admin=args.admin)
    if args.workspace:
        print('The workspace is owner-protected. Sign in at '+launch_url(port).replace('/showcase/index.html','/admin/')+' before opening it.',flush=True)
    if reuse:
        print('This workspace is already running: '+url,flush=True)
        if not args.no_browser:webbrowser.open(url)
        return 0
    print('Keep this Terminal open while using the app. Ctrl-C stops its server and active run.',flush=True)
    print('Uploads and working revisions remain owner-only; authorized processing can use hosted models and the configured GPU cluster.',flush=True)
    child=subprocess.Popen([python,str(ROOT/'scripts/serve_workspace.py'),'--host','127.0.0.1','--port',str(port)],cwd=ROOT,env=task_env)
    try:
        deadline=time.monotonic()+15
        while time.monotonic()<deadline and child.poll() is None:
            if owned_server(port):
                print('Ready: '+url,flush=True)
                if not args.no_browser:webbrowser.open(url)
                break
            time.sleep(.15)
        else:
            if child.poll() is not None:return child.returncode
            print('Server did not become ready; stopping only this launcher’s child process.',file=sys.stderr)
            child.terminate();child.wait(timeout=20);return 2
        return child.wait()
    except KeyboardInterrupt:
        if child.poll() is None:
            # The foreground Terminal usually delivered SIGINT to both processes.
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        print('\nLocal workspace stopped.',flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
