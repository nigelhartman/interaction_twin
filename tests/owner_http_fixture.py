"""Synthetic owner sessions for legacy HTTP contract tests, never production.

The real authentication module issues the cookie/CSRF pair in a temporary
credential namespace. Requests keep explicit hostile Origin/Host fields, so the
original boundary tests still exercise their actual checks.
"""
from pathlib import Path
import tempfile
import urllib.request
from unittest.mock import patch
from urllib.parse import urlparse
import admin_auth


class OwnerHTTPSession:
    def __init__(self,server):
        self.temp=tempfile.TemporaryDirectory()
        auth=admin_auth.AdminAuth(Path(self.temp.name).resolve())
        admin_auth.configure_password(auth.root,'Synthetic HTTP test owner password')
        anonymous,token=auth.status()
        status,token=auth.login('Synthetic HTTP test owner password',f'{admin_auth.COOKIE}={token}',anonymous['csrf_token'])
        server.admin_auth=auth
        base=f'http://127.0.0.1:{server.server_port}'
        owner_headers={'Origin':base,'Cookie':f'{admin_auth.COOKIE}={token}','X-CSRF-Token':status['csrf_token']}
        original=urllib.request.urlopen
        def request(value,*args,**kwargs):
            req=urllib.request.Request(value) if isinstance(value,str) else value
            if urlparse(req.full_url).netloc==urlparse(base).netloc:
                for key,value in owner_headers.items():
                    if not req.has_header(key):req.add_header(key,value)
            return original(req,*args,**kwargs)
        self.patcher=patch.object(urllib.request,'urlopen',side_effect=request)
        self.patcher.start()

    def close(self):
        self.patcher.stop();self.temp.cleanup()
