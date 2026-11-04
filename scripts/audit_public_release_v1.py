#!/usr/bin/env python3
"""Read only the outgoing release, never local credential stores. Report no values."""
from __future__ import annotations
import argparse, hashlib, io, json, re
from pathlib import Path
from zipfile import ZipFile

TEXT={'.py','.js','.json','.html','.css','.md','.txt','.svg','.yml','.yaml','.toml','.sh','.command','.vtt'}
PATTERNS={
 'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
 'github_token':re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
 'aws_access_key':re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
 'secret_token':re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b'),
 'literal_credential':re.compile(r'''(?im)(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret)\s*["']?\s*[:=]\s*["']([A-Za-z0-9_./+=-]{28,})["']''')}
def scan_text(name,raw):
    try:text=raw.decode('utf-8')
    except UnicodeDecodeError:return []
    out=[]
    for kind,pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            candidate=match.group(1)if match.lastindex else match.group()
            # Literal test fixtures are not credentials; retain their location.
            if candidate.startswith(('test-','fake-','dummy-','example-','not-a-')):continue
            out.append({'file':name,'line':text.count('\n',0,match.start())+1,'kind':kind})
    return out
def audit(root):
    root=root.resolve();findings=[];files=[]
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if '.git' in rel.parts:continue
        if p.is_symlink():findings.append({'file':str(rel),'kind':'symlink'});continue
        if not p.is_file():continue
        if any(x in {'.secrets','.ssh','.aws','__pycache__','node_modules'}for x in rel.parts)or p.name.startswith('.env')or p.suffix in {'.pem','.key','.p12','.sqlite','.sqlite3'}:
            findings.append({'file':str(rel),'kind':'forbidden_private_file'});continue
        raw=p.read_bytes();files.append({'path':rel.as_posix(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
        if p.suffix in TEXT:findings+=scan_text(rel.as_posix(),raw)
        if p.suffix=='.zip':
            with ZipFile(io.BytesIO(raw))as z:
                assert len(z.infolist())<10000 and sum(i.file_size for i in z.infolist())<256*1024*1024
                for i in z.infolist():
                    name=Path(i.filename)
                    if any(x.startswith('.env')or x in {'.secrets','.ssh','.aws'}for x in name.parts):findings.append({'file':str(rel)+'!'+i.filename,'kind':'private_archive_member'})
                    if name.suffix in TEXT:findings+=scan_text(str(rel)+'!'+i.filename,z.read(i))
    return {'status':'PASS'if not findings else'FAIL','files_checked':len(files),'bytes_checked':sum(r['bytes']for r in files),'findings':findings,'inventory':files,'scope':'outgoing_release_only; credential_stores_not_read; heuristic_scan_not_a_security_guarantee'}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--report',type=Path,required=True);a=p.parse_args();r=audit(a.root);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k!='inventory'},indent=2));raise SystemExit(0 if r['status']=='PASS'else 1)
