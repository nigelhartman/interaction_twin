#!/usr/bin/env python3
"""Copy an explicit reviewed metadata allowlist into the existing data route.

No server access is expanded. Original documents/manifests remain byte-identical;
new versions require a fresh directory and links must deliberately select it.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'data/submission-review-20261101'
MAX_BYTES = 4_000_000
ARTIFACTS = (
    ('docs/submission-demo-20261101.md', 'submission-demo.md', 'text/markdown'),
    ('docs/submission-draft-20261101.md', 'submission-draft.md', 'text/markdown'),
    ('docs/positioning-review-20261101.md', 'positioning-review.md', 'text/markdown'),
    ('output/overnight-session-20261031/overnight-trailer-20261031-v2-source-manifest.json', 'film-source-manifest.json', 'application/json'),
    ('output/overnight-session-20261031/overnight-trailer-20261031-v2-verification.json', 'film-verification.json', 'application/json'),
    ('output/submission-walkthrough-20261101-v1/render-v2/source-manifest.json', 'walkthrough-source-manifest.json', 'application/json'),
    ('output/submission-walkthrough-20261101-v1/render-v2/verification.json', 'walkthrough-verification.json', 'application/json'),
    ('output/submission-walkthrough-20261101-v1/render-v2/captions.json', 'walkthrough-captions.json', 'application/json'),
)

def artifact_set(walkthrough='v2'):
    require(walkthrough in {'v2','v3'}, 'Unknown reviewed walkthrough source set')
    return tuple((source.replace('/render-v2/', '/render-'+walkthrough+'/'),name,mime)
                 for source,name,mime in ARTIFACTS)

def require(value, message):
    if not value: raise ValueError(message)

def sha(data): return hashlib.sha256(data).hexdigest()

def confined(root, relative, *, exists=True):
    require(isinstance(relative, str) and '\\' not in relative and '\x00' not in relative, 'Invalid artifact path')
    parts=Path(relative).parts
    require(parts and not Path(relative).is_absolute() and all(p not in {'.','..'} and not p.startswith('.') for p in parts), 'Hidden or traversal path rejected')
    p=root/relative
    require(p.resolve()==p and p.is_relative_to(root), 'Symlinked or escaping artifact rejected')
    if exists: require(p.is_file(), 'Missing allowlisted artifact')
    return p

def strict_json(data):
    return json.loads(data, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-finite JSON rejected')))

def build(*, root=ROOT, version='v1', walkthrough='v2'):
    root=Path(root).resolve()
    require(isinstance(version,str) and re.fullmatch(r'v[1-9][0-9]{0,3}',version), 'Choose a bounded fresh version ID')
    relative=PREFIX+'/'+version;out=confined(root,relative,exists=False)
    require(not out.exists(), 'Existing review version is immutable; choose a new version')
    prepared=[]
    sources=artifact_set(walkthrough)
    for source,name,mime in sources:
        require(Path(name).name==name and name not in {'index.json','README.txt'}, 'Invalid fixed destination')
        p=confined(root,source);data=p.read_bytes()
        require(0<len(data)<=MAX_BYTES and b'\x00' not in data, 'Unexpected metadata size or binary content')
        text=data.decode('utf-8')
        if mime=='application/json': strict_json(text)
        else: require(mime=='text/markdown' and source.endswith('.md'), 'Unsupported allowlisted type')
        prepared.append((source,name,mime,data))
    out.mkdir(parents=True)
    rows=[]
    for source,name,mime,data in prepared:
        with (out/name).open('xb') as stream: stream.write(data)
        copied=relative+'/'+name
        rows.append({'original_path':source,'original_sha256':sha(data),'served_path':copied,
                     'served_url':'/files/'+copied,'served_sha256':sha(data),'bytes':len(data),'mime_type':mime,
                     'transformation':'none_byte_exact_copy'})
    note=('Selected local review documents and provenance only.\n'
          'Original files are unchanged; index.json binds every copy to its original SHA-256.\n'
          'No broad output/docs route was enabled. This folder is local/private; nothing was published.\n'
          'Paths inside copied documents/manifests retain original workspace meaning and are not rewritten.\n'
          'Referenced source recordings and some build evidence are intentionally not copied here.\n'
          'Video, transcript and VTT stay in their existing showcase/media routes.\n')
    (out/'README.txt').write_text(note)
    index={'format':'selected-submission-review-artifacts','version':1,'id':version,'walkthrough_render_version':walkthrough,
           'created_utc':datetime.now(timezone.utc).isoformat(),'source_builder_sha256':sha(Path(__file__).read_bytes()),
           'original_files_modified':False,'broad_static_route_expanded':False,'published':False,
           'artifact_count':len(rows),'artifacts':rows,'limitations':['Private local review only.','Embedded original paths are preserved metadata, not a promise every dependency is HTTP served.']}
    (out/'index.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
    verify(out,root=root)
    return index

def verify(directory, *, root=ROOT):
    root=Path(root).resolve();directory=Path(directory).resolve()
    require(directory.is_relative_to(root/PREFIX), 'Unknown review directory')
    index=strict_json((directory/'index.json').read_text())
    sources=artifact_set(index.get('walkthrough_render_version','v2'))
    require(index['format']=='selected-submission-review-artifacts' and index['artifact_count']==len(sources), 'Unexpected review index')
    expected={(source,name,mime) for source,name,mime in sources};actual=set()
    for row in index['artifacts']:
        source=confined(root,row['original_path']);target=confined(root,row['served_path'])
        require(target.parent==directory, 'Review copy escapes its version')
        require(row['served_url']=='/files/'+row['served_path'], 'Unexpected served route')
        data=source.read_bytes();copy=target.read_bytes()
        require(data==copy and sha(data)==row['original_sha256']==row['served_sha256'] and len(data)==row['bytes'], 'Original/copy hash or byte identity changed')
        actual.add((row['original_path'],target.name,row['mime_type']))
    require(actual==expected and len(index['artifacts'])==len(expected), 'Allowlist changed or duplicated')
    return index

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',default='v1')
    parser.add_argument('--walkthrough',choices=['v2','v3'],default='v2')
    args=parser.parse_args();result=build(version=args.version,walkthrough=args.walkthrough)
    print(json.dumps({'status':'copied_and_verified','artifacts':result['artifact_count'],
                      'index_url':'/files/'+PREFIX+'/'+args.version+'/index.json'}))
