#!/usr/bin/env python3
"""Offline key check by default; explicitly permit one durable GET /v1/models."""
from __future__ import annotations
import json
import os
from pathlib import Path
import stat
import sys
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler,ProxyHandler,Request,build_opener

ROOT=Path(__file__).resolve().parents[1]
ENDPOINT='https://api.meta.ai/v1/models'
MAX_BYTES=256*1024


class ProbeError(Exception):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        return None


def private_directory(root):
    try:
        fd=os.open(Path(root)/'.secrets',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        info=os.fstat(fd)
        if info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)&0o077:
            os.close(fd);raise ProbeError('unsafe_secret_directory')
        return fd
    except OSError:
        raise ProbeError('secret_directory_unavailable') from None


def load_key(directory):
    try:
        fd=os.open('meta.env',os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory)
        with os.fdopen(fd,'rb') as stream:
            info=os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)&0o077:
                raise ProbeError('unsafe_credential_file')
            raw=stream.read(8193)
    except OSError:
        raise ProbeError('credential_file_unavailable') from None
    if not raw:raise ProbeError('credential_file_empty')
    if len(raw)>8192:raise ProbeError('invalid_credential_format')
    try:text=raw.decode('ascii').replace('\r\n','\n')
    except UnicodeError:raise ProbeError('invalid_credential_format') from None
    if any(ord(c)<32 and c!='\n' or ord(c)==127 for c in text):
        raise ProbeError('invalid_credential_format')
    lines=[line for line in text.split('\n') if line.strip() and not line.lstrip(' ').startswith('#')]
    if len(lines)!=1:raise ProbeError('invalid_credential_format')
    name,separator,key=lines[0].partition('=')
    if name.strip()!='MODEL_API_KEY' or separator!='=' or not key:
        raise ProbeError('invalid_credential_format')
    if any(not 33<=ord(c)<=126 for c in key) or key[0] in {'"',"'"}:
        raise ProbeError('invalid_credential_format')
    return key


def write_receipt(fd,result):
    data=(json.dumps(result,separators=(',',':'))+'\n').encode()
    os.lseek(fd,0,os.SEEK_SET);os.ftruncate(fd,0)
    with os.fdopen(os.dup(fd),'wb',closefd=True) as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())


def request_once(key):
    """Exactly one urllib open; proxies/redirects disabled, normal TLS retained."""
    opener=build_opener(ProxyHandler({}),NoRedirect())
    request=Request(ENDPOINT,headers={'Authorization':'Bearer '+key,'Accept':'application/json'},method='GET')
    try:
        with opener.open(request,timeout=10) as response:
            status=response.status
            if not 200<=status<300:
                return {'status':'http_error','http_status':status,'model_count':None}
            body=response.read(MAX_BYTES+1)
    except HTTPError as exc:
        code=exc.code if type(exc.code) is int and 100<=exc.code<=599 else None
        # Never read, print or retain provider error bodies or headers.
        return {'status':'redirect_refused' if code in {301,302,303,307,308} else 'http_error',
                'http_status':code,'model_count':None}
    if len(body)>MAX_BYTES:return {'status':'response_too_large','http_status':status,'model_count':None}
    try:
        document=json.loads(body)
        models=document.get('data') if isinstance(document,dict) else None
        count=len(models) if isinstance(models,list) else None
    except (ValueError,UnicodeError,RecursionError):count=None
    return {'status':'http_ok','http_status':status,'model_count':count}


def probe(root=ROOT,auth_once=False):
    directory=private_directory(root)
    try:
        key=load_key(directory)
        if not auth_once:return {'status':'local_key_ready_no_request','model_count':None}
        try:
            receipt=os.open('meta-auth-attempt.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=directory)
        except FileExistsError:return {'status':'attempt_already_consumed','model_count':None}
        except OSError:raise ProbeError('attempt_reservation_failed') from None
        try:
            # Reservation is durable BEFORE any network request, even on failure.
            write_receipt(receipt,{'status':'attempt_reserved','model_count':None});os.fsync(directory)
            try:result=request_once(key)
            except Exception:result={'status':'request_failed','model_count':None}
            write_receipt(receipt,result)
            return result
        finally:os.close(receipt)
    except OSError:
        raise ProbeError('local_receipt_io_failed') from None
    finally:os.close(directory)


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if args==['--help']:
        print('Usage: meta_auth_probe.py [--auth-once]\nDefault: local format/permission check only. --auth-once permits one durable GET /v1/models. No retries or generation.')
        return 0
    if args not in ([],['--auth-once']):
        print(json.dumps({'status':'invalid_arguments','model_count':None}));return 2
    try:result=probe(ROOT,auth_once=bool(args))
    except ProbeError as exc:result={'status':str(exc),'model_count':None}
    print(json.dumps(result))
    return 0 if result['status'] in {'local_key_ready_no_request','http_ok'} else 2


if __name__=='__main__':raise SystemExit(main())
