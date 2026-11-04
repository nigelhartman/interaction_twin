#!/usr/bin/env python3
"""Validate/import a bounded private MCP motion-inspection ZIP without executing it."""
import argparse,json,stat,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from portable_twin import inspect_package,safe_name
from rehearse_motion_handoff import verify_motion,sha,require,encoded
from motion_catalog import strict_json


def archive_bytes(path):
    require(Path(path).stat().st_size<=96*1024*1024,'Archive compressed size budget exceeded')
    with zipfile.ZipFile(path) as z:
        infos=z.infolist();require(0<len(infos)<=160 and len({i.filename for i in infos})==len(infos),'Duplicate or unbounded bundle files')
        total=0;files={}
        for info in infos:
            name=safe_name(info.filename);mode=info.external_attr>>16
            require(not info.is_dir() and not stat.S_ISLNK(mode) and (not stat.S_IFMT(mode) or stat.S_ISREG(mode)),'Only regular declared ZIP members')
            require(info.flag_bits&1==0 and info.file_size<=32*1024*1024,'Encrypted or oversized ZIP member')
            total+=info.file_size;require(total<=96*1024*1024,'Uncompressed bundle size budget exceeded')
            data=z.read(info);require(len(data)==info.file_size,'Truncated bundle member');files[name]=data
    return files


def validate(path,extract_to):
    files=archive_bytes(path);require('bundle-manifest.json' in files,'Missing bundle manifest')
    manifest=strict_json(files['bundle-manifest.json'])
    require(manifest.get('format')=='interaction-twin-motion-inspection-bundle' and manifest.get('version')==1,'Unknown bundle format')
    declared=manifest['files'];require(set(files)==set(declared)|{'bundle-manifest.json'},'Undeclared or missing bundle member')
    for name,binding in declared.items():
        safe_name(name);require(sha(files[name])==binding['sha256'] and len(files[name])==binding['bytes'],'Bundle member hash/length mismatch: '+name)
    target=Path(extract_to).absolute();require(not target.exists() and target.resolve()==target,'Use a fresh nonsymlink extraction directory')
    target.mkdir(parents=True)
    for name,data in files.items():
        p=target/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as stream:stream.write(data)
    directory=inspect_package(target/'portable');nested=inspect_package(target/'portable.zip',extract_to=target/'relocated-portable')
    relocated=inspect_package(target/'relocated-portable')
    portable=strict_json(files['portable/manifest.json']);document=strict_json(files['portable/'+portable['program']['path']])
    asset_bytes={a['track_id']:files['portable/'+a['path']] for a in portable['assets']}
    collected={'twin_id':manifest['twin_id'],'document':document,'source_program_sha256':manifest['source_program_sha256'],'asset_bytes':asset_bytes}
    descriptor=strict_json(files['motion/descriptor.json'])
    downloads={'raw':files['motion/raw.json'],'normalized':files['motion/normalized.json'],'source-image-0':files['motion/source.jpg'],'source-mask-0':files['motion/source-mask.png']}
    _,_,evidence=verify_motion(descriptor,downloads,collected)
    trace=strict_json(files['sdk-trace.json']);task=strict_json(files['inspection-task.json']);simulation=strict_json(files['simulation-handoff.json'])
    require(trace['twin_id']==manifest['twin_id']==descriptor['twin_id'] and trace['source_program_sha256']==manifest['source_program_sha256'],'Trace identity differs')
    require(len(trace['calls'])==manifest['scripted_calls'] and [r['sequence'] for r in trace['calls']]==list(range(1,len(trace['calls'])+1)),'Trace calls missing/reordered')
    require(set(trace['assets_verified'])==set(asset_bytes) and all(trace['assets_verified'][k]==sha(v) for k,v in asset_bytes.items()),'Native assets differ from trace')
    require(trace['source_video_included'] is False and trace['model_inference'] is False and trace['simulation_executed'] is False,'Trace execution scope changed')
    require(task['experiment_id']==manifest['experiment_id']==descriptor['experiment_id'] and task['simulation_ready'] is False and task['no_LLM'] is True,'Inspection task differs or claims execution')
    require(simulation['program_sha256']==manifest['source_program_sha256'] and simulation['policy']['simulation_ready'] is False,'Simulation handoff claims physical readiness')
    methods={(r['method'],r.get('name')) for r in trace['calls']}
    require({('tools/call','list_motion_experiments'),('tools/call','get_motion_experiment')}<=methods,'Trace lacks genuine motion-tool responses')
    reads={r['uri']:r for r in trace['calls'] if r['method']=='resources/read'}
    for key,data in downloads.items():
        receipt=reads.get(descriptor['artifacts'][key]['uri']);require(receipt is not None and receipt['decoded_sha256']==sha(data),'Artifact not bound to actual SDK resource receipt')
    return {'format':'independent-relocated-motion-handoff-import','version':1,'status':'passed',
        'source_zip_sha256':sha(Path(path).read_bytes()),'extracted_to':str(target),'twin_id':manifest['twin_id'],
        'experiment_id':descriptor['experiment_id'],'native_assets_verified':len(asset_bytes),'files_verified':len(files),
        'actual_sdk_calls':len(trace['calls']),'exact_source_sample':evidence['source_index'],'timestamp_s':evidence['timestamp_s'],
        'portable_directory':directory,'nested_portable_import':nested,'relocated_portable_reinspection':relocated,
        'source_video_included':False,'model_inference':False,'simulation_executed':False,'simulation_ready':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('archive',type=Path);parser.add_argument('--extract-to',type=Path,required=True);parser.add_argument('--report',type=Path)
    args=parser.parse_args();result=validate(args.archive,args.extract_to)
    if args.report:
        with args.report.open('xb') as stream:stream.write(encoded(result))
    print(json.dumps(result,indent=2))
