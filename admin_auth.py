"""Single-owner local administration with scrypt and opaque server-side sessions.

No default password, credential discovery, external identity service or API call.
Only this module reads its named admin credential files; it never reads Meta keys.
"""
from __future__ import annotations
from collections import deque
import base64
import hashlib
import hmac
from http.cookies import SimpleCookie, CookieError
import json
import os
from pathlib import Path
import re
import secrets
import stat
import threading
import time

COOKIE='interaction_twin_admin'
TOKEN=re.compile(r'[A-Za-z0-9_-]{43}\Z')
SESSION_SECONDS=12*60*60
BOOTSTRAP_SECONDS=24*60*60
SCRYPT={'n':32768,'r':8,'p':1,'dklen':32,'maxmem':64*1024*1024}


class AuthError(ValueError):
    def __init__(self,message='Administrator authentication required',status=401):
        super().__init__(message);self.status=status


def _folder(root):
    root=Path(root).resolve();folder=root/'.secrets'
    if folder.is_symlink() or folder.resolve()!=folder:
        raise AuthError('Unsafe admin credential directory',503)
    folder.mkdir(mode=0o700,exist_ok=True)
    if not folder.is_dir() or folder.stat().st_uid!=os.getuid():
        raise AuthError('Unsafe admin credential directory',503)
    return folder


def _path(root,name):
    if name not in {'admin-auth.json','admin-bootstrap.json','admin-setup-code'}:
        raise AuthError('Unknown admin credential artifact',503)
    path=_folder(root)/name
    if path.is_symlink():raise AuthError('Unsafe admin credential artifact',503)
    return path


def _read(root,name):
    path=_path(root,name)
    try:fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    except FileNotFoundError:return None
    with os.fdopen(fd,'rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o077 or info.st_size>16384:
            raise AuthError('Admin credential permissions require owner-only access',503)
        try:return json.loads(stream.read())
        except (ValueError,UnicodeError):raise AuthError('Admin credential configuration is invalid',503) from None


def _new(root,name,value,*,replace=False):
    path=_path(root,name)
    data=(json.dumps(value,separators=(',',':'))+'\n').encode() if isinstance(value,dict) else value
    if not replace:
        descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(descriptor,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    else:
        temporary=_folder(root)/('.admin-'+secrets.token_hex(12)+'.tmp')
        descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        try:
            with os.fdopen(descriptor,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
            temporary.replace(path)
        finally:temporary.unlink(missing_ok=True)


def _password(password):
    if not isinstance(password,str) or not 12<=len(password)<=1024 or len(password.encode())>4096:
        raise AuthError('Choose a password with at least 12 characters',400)
    return password.encode()


def credential(password):
    password=_password(password);salt=secrets.token_bytes(32)
    value=hashlib.scrypt(password,salt=salt,**SCRYPT)
    return {'format':'interaction-twin-owner','version':1,'owner':'nigel','credential_id':secrets.token_hex(16),
            'algorithm':'scrypt','n':SCRYPT['n'],'r':SCRYPT['r'],'p':SCRYPT['p'],
            'salt':base64.b64encode(salt).decode(),'password_hash':base64.b64encode(value).decode()}


def _credentials(root):
    value=_read(root,'admin-auth.json')
    if value is None:return None
    if (not isinstance(value,dict) or value.get('format')!='interaction-twin-owner' or value.get('version')!=1 or
        value.get('owner')!='nigel' or value.get('algorithm')!='scrypt' or
        any(value.get(k)!=SCRYPT[k] for k in ['n','r','p']) or
        not re.fullmatch('[a-f0-9]{32}',value.get('credential_id',''))):
        raise AuthError('Admin credential configuration is invalid',503)
    try:
        salt=base64.b64decode(value['salt'],validate=True);hashed=base64.b64decode(value['password_hash'],validate=True)
        if len(salt)!=32 or len(hashed)!=32:raise ValueError()
    except (ValueError,KeyError,TypeError):raise AuthError('Admin credential configuration is invalid',503) from None
    return value


def configure_password(root,password,*,replace=False):
    value=credential(password)
    _new(root,'admin-auth.json',value,replace=replace)
    for name in ['admin-bootstrap.json','admin-setup-code']:_path(root,name).unlink(missing_ok=True)


def create_bootstrap(root,*,replace=False,now=None):
    """Write the owner-copy code without returning or printing its contents."""
    if _credentials(root) is not None:raise AuthError('Owner is already configured; use local password reset',409)
    now=time.time() if now is None else now
    code=secrets.token_urlsafe(32)
    _new(root,'admin-setup-code',(code+'\n').encode(),replace=replace)
    metadata={'format':'interaction-twin-owner-bootstrap','version':1,
              'code_sha256':hashlib.sha256(code.encode()).hexdigest(),'expires_at':now+BOOTSTRAP_SECONDS}
    _new(root,'admin-bootstrap.json',metadata,replace=replace)
    return _path(root,'admin-setup-code')


def cookie_token(header):
    if not isinstance(header,str) or len(header)>4096:return None
    try:
        cookie=SimpleCookie();cookie.load(header);token=cookie[COOKIE].value if COOKIE in cookie else None
    except CookieError:return None
    return token if isinstance(token,str) and TOKEN.fullmatch(token) else None


def cookie_header(token,*,clear=False):
    return f'{COOKIE}={"" if clear else token}; Path=/; Max-Age={0 if clear else SESSION_SECONDS}; HttpOnly; SameSite=Strict'


class AdminAuth:
    def __init__(self,root,*,clock=time.time):
        self.root=Path(root).resolve();self.clock=clock;self.lock=threading.RLock();self.sessions={};self.failures=deque()

    def configured(self):return _credentials(self.root) is not None

    def _prune(self):
        now=self.clock()
        for key in list(self.sessions):
            if self.sessions[key]['expires_at']<=now:del self.sessions[key]
        while self.failures and self.failures[0]<now-900:self.failures.popleft()

    def _key(self,token):return hashlib.sha256(token.encode()).hexdigest()

    def _session(self,header):
        self._prune();token=cookie_token(header)
        return self.sessions.get(self._key(token)) if token else None

    def _issue(self,*,owner=False):
        self._prune()
        if len(self.sessions)>=256:
            oldest=min(self.sessions,key=lambda key:self.sessions[key]['expires_at']);del self.sessions[oldest]
        token,csrf=secrets.token_urlsafe(32),secrets.token_urlsafe(32)
        creds=_credentials(self.root) if owner else None
        self.sessions[self._key(token)]={'csrf':csrf,'owner':'nigel' if owner else None,
            'credential_id':creds['credential_id'] if creds else None,'expires_at':self.clock()+(SESSION_SECONDS if owner else 1200)}
        return token,self.sessions[self._key(token)]

    def _owner(self,session):
        if not session or session['owner']!='nigel':return False
        creds=_credentials(self.root)
        return bool(creds and session['credential_id']==creds['credential_id'])

    def status(self,header=None):
        with self.lock:
            session=self._session(header);token=None
            if session is None:token,session=self._issue()
            metadata=_read(self.root,'admin-bootstrap.json')
            ready=bool(metadata and metadata.get('expires_at',0)>self.clock())
            return {'configured':self.configured(),'authenticated':self._owner(session),
                    'setup_available':ready and not self.configured(),'csrf_token':session['csrf']},token

    def authenticated(self,header):
        with self.lock:return self._owner(self._session(header))

    def require(self,header,csrf=None,*,mutation=False,owner=True):
        with self.lock:
            session=self._session(header)
            if owner and not self._owner(session):raise AuthError()
            if not session:raise AuthError('Start a same-origin administrator session',403)
            if mutation and (not isinstance(csrf,str) or not hmac.compare_digest(session['csrf'],csrf)):
                raise AuthError('Administrator CSRF token is invalid',403)
            return 'nigel' if owner else None

    def _rate(self):
        self._prune()
        if len(self.failures)>=10:raise AuthError('Too many sign-in attempts; try again in 15 minutes',429)

    def login(self,password,header,csrf):
        with self.lock:
            self.require(header,csrf,mutation=True,owner=False);self._rate();value=_credentials(self.root)
            try:raw=_password(password)
            except AuthError:raw=b''
            if value is None:raise AuthError('Configure the owner locally before signing in',409)
            hashed=hashlib.scrypt(raw,salt=base64.b64decode(value['salt']),**SCRYPT)
            if not hmac.compare_digest(hashed,base64.b64decode(value['password_hash'])):
                self.failures.append(self.clock());raise AuthError('Incorrect password',401)
            old=cookie_token(header)
            if old:self.sessions.pop(self._key(old),None)
            self.failures.clear();token,_=self._issue(owner=True)
            return self.status(cookie_header(token))[0],token

    def setup(self,code,password,header,csrf):
        with self.lock:
            self.require(header,csrf,mutation=True,owner=False);self._rate()
            if self.configured():raise AuthError('Owner is already configured',409)
            metadata=_read(self.root,'admin-bootstrap.json')
            valid=isinstance(code,str) and TOKEN.fullmatch(code) and metadata and metadata.get('expires_at',0)>self.clock()
            if not valid or not hmac.compare_digest(hashlib.sha256(code.encode()).hexdigest(),metadata.get('code_sha256','')):
                self.failures.append(self.clock());raise AuthError('Setup code is invalid or expired',403)
            configure_password(self.root,password)
            old=cookie_token(header)
            if old:self.sessions.pop(self._key(old),None)
            token,_=self._issue(owner=True)
            return self.status(cookie_header(token))[0],token

    def logout(self,header,csrf):
        with self.lock:
            self.require(header,csrf,mutation=True)
            token=cookie_token(header)
            self.sessions.pop(self._key(token),None)
