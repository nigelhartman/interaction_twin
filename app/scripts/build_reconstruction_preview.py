#!/usr/bin/env python3
"""Build a saved, no-inference preview of the three reconstruction MCP tools."""
from datetime import datetime,timezone
import copy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from twin_catalog import TwinCatalog


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def build(root=ROOT):
    root=Path(root).resolve();catalog=TwinCatalog(root)
    cases=[]
    for entry in catalog.entries():
        listing=catalog.reconstructions(entry['id'])
        if not listing['versions']:continue
        versions=[]
        for version in listing['versions']:
            vid=version['version_id']
            quality=catalog.reconstruction_report(entry['id'],vid)
            raw_quality=(json.dumps(quality,indent=2,allow_nan=False)+'\n').encode()
            cache=root/'data/reconstruction-tools-preview-20261031/responses'/(vid+'-quality.json')
            cache.parent.mkdir(parents=True,exist_ok=True)
            if cache.exists() and cache.read_bytes()!=raw_quality:
                raise ValueError('Preserve immutable quality response; publish a new version')
            if not cache.exists():cache.write_bytes(raw_quality)
            excerpt=copy.deepcopy(quality)
            omitted=[]
            if len(raw_quality)>150000 and 'frames' in excerpt['report']:
                del excerpt['report']['frames']
                omitted=['report.frames']
            versions.append({'id':vid,'descriptor':catalog.reconstruction(entry['id'],vid),
                             'quality':excerpt,'quality_preview_omitted_fields':omitted,
                             'quality_response_artifact':{'path':str(cache.relative_to(root)),
                                 'sha256':sha(cache),'bytes':len(raw_quality)}})
        cases.append({'id':entry['id'],'list_response':listing,'versions':versions})
    directory=root/'data/reconstruction-agent-use-case-20261031/case-lid-inspection-v1'
    manifest=read(directory/'package-manifest.json')
    for record in manifest['files']:
        path=(directory/record['path']).resolve()
        if not path.is_relative_to(directory) or sha(path)!=record['sha256']:
            raise ValueError('Actual MCP inspection package member identity changed')
    task,trace=read(directory/'inspection-task.json'),read(directory/'mcp-trace.json')
    if (trace['model_inference'] is not False or trace['simulation_executed'] is not False or
            task['simulation_ready'] is not False or manifest['all_resources_retrieved_through_mcp'] is not True):
        raise ValueError('Unexpected inference/execution claim in scripted inspection package')
    archive=directory.with_suffix('.zip')
    verification=read(directory.parent/'verification.json')
    if verification.get('actual_mcp_calls')!=len(trace['calls']) or verification.get('archive_member_hashes_verified') is not True:
        raise ValueError('Original scripted SDK verification record does not match')
    with zipfile.ZipFile(archive) as bundle:
        expected={r['path']:r['sha256'] for r in manifest['files']}
        expected['package-manifest.json']=sha(directory/'package-manifest.json')
        if set(bundle.namelist())!=set(expected):
            raise ValueError('Inspection ZIP members changed')
        if any(hashlib.sha256(bundle.read(name)).hexdigest()!=digest for name,digest in expected.items()):
            raise ValueError('Inspection ZIP member pixels or artifact bytes changed')
    # Keep the verified original run immutable; the preview only links its bytes.
    story={'task':task,'manifest':manifest,'trace_summary':{
        'call_count':len(trace['calls']),'transport':trace['transport'],
        'planning':trace['planning'],'negotiated_protocol':trace['negotiated_protocol'],
        'model_inference':False,'simulation_executed':False,
        'calls':[{'sequence':r['sequence'],'method':r['method'],'name':r.get('name'),
                  'is_error':r.get('is_error',False)} for r in trace['calls']]},
        'links':{name:str((directory/name).relative_to(root)) for name in
                 ['inspection-task.json','mcp-trace.json','package-manifest.json','evidence/source.jpg','evidence/component-mask.png']},
        'zip_path':str(archive.relative_to(root)),'zip_sha256':sha(archive),
        'task_sha256':sha(directory/'inspection-task.json'),'trace_sha256':sha(directory/'mcp-trace.json')}
    result={'format':'saved-reconstruction-tools-preview','version':1,
            'generated_at':datetime.now(timezone.utc).isoformat(),
            'source_index_sha256':sha(root/'data/reconstruction-catalog-20261031/index.json'),
            'new_model_calls':0,'new_mcp_calls':0,'browser_connects_to_mcp':False,
            'cases':cases,'scripted_inspection':story}
    raw=json.dumps(result,indent=2,allow_nan=False)+'\n'
    (root/'workspace/reconstruction-preview.json').write_text(raw)
    (root/'workspace/reconstruction-preview.js').write_text('window.TWIN_RECONSTRUCTION_PREVIEW = '+raw.rstrip().replace('<','\\u003c')+';\n')
    return result


if __name__=='__main__':
    result=build();print(json.dumps({'cases':len(result['cases']),
        'versions':sum(len(c['versions']) for c in result['cases']),
        'actual_recorded_mcp_calls':result['scripted_inspection']['trace_summary']['call_count'],
        'new_model_calls':0,'new_mcp_calls':0}))
