#!/usr/bin/env python3
"""Actual read-only MCP → portable twin + separate motion inspection bundle.

Only MCP-returned bytes enter the package. No original document paths are opened
by this client. Output is private/local; source video is deliberately omitted.
"""
import argparse,asyncio,json,sys,zipfile
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from demo_twin_agent import RecordedMCP,collect_twin,write_package,require,sha,hash_value,identifier,verify_state
from motion_catalog import strict_json
from portable_twin import inspect_package,safe_name

DEFAULT_TWIN='flow-rigid-parts-jabra-20261031-v1--flow-procedure-1'
DEFAULT_EXPERIMENT='jabra-case-base-3fps-v3'


def encoded(value):return (json.dumps(value,indent=2,allow_nan=False,ensure_ascii=False)+'\n').encode()


def verify_motion(descriptor,downloads,collected):
    """Independent consumer boundary: exact source/native asset and saved poses."""
    doc=collected['document'];twin=collected['twin_id'];experiment=identifier(descriptor['experiment_id'])
    require(descriptor['twin_id']==twin and descriptor['program_sha256']==collected['source_program_sha256']
        and descriptor['source_video_sha256']==doc['source']['sha256'],'Motion source/program differs from portable twin')
    policy=descriptor['policy']
    for key in ['model_inference','pose_synthesis','pose_interpolation','physical_pose_accuracy_verified','simulation_ready','common_metric_assembly_established']:
        require(policy.get(key) is False,'Unsafe motion policy: '+key)
    require(policy.get('read_only') is True and policy.get('independent_episode_gauges') is True,'Missing independent-gauge read-only boundary')
    component=descriptor['component_id'];asset=next((a for a in doc['assets'] if a['track_id']==component),None)
    require(asset is not None and descriptor['native_mesh_sha256']==asset['sha256']==sha(collected['asset_bytes'][component]),'Motion does not use the same unchanged native GLB')
    require(set(downloads)=={'raw','normalized','source-image-0','source-mask-0'},'Unexpected motion download surface')
    for key,data in downloads.items():
        binding=descriptor['artifacts'][key]
        require(binding['uri']==f'twin://{twin}/motion/{experiment}/artifacts/{key}' and binding['resource_mime_type']=='application/octet-stream','Unexpected motion resource URI')
        require(sha(data)==hash_value(binding['sha256']),'Motion artifact hash mismatch')
    raw=strict_json(downloads['raw']);view=strict_json(downloads['normalized'])
    require(view==descriptor['motion'] and view['input_sha256']==sha(downloads['raw']),'Frozen raw/normalized motion identity differs')
    require(raw['mesh_sha256']==descriptor['native_mesh_sha256'] and raw['interaction_sha256']==collected['source_program_sha256'],'Raw motion asset/source changed')
    source=descriptor['inspection_source'];evidence=next((e for e in doc['evidence'] if e['id']==source['program_evidence_id']),None)
    require(evidence is not None and evidence['source_index']==source['sample_index'] and evidence['timestamp_s']==source['timestamp_s'],'Motion source is not an exact program sample')
    observation=next((o for o in evidence['observations'] if o['track_id']==component),None)
    require(observation is not None and sha(downloads['source-image-0'])==evidence['frame_sha256']
        and sha(downloads['source-mask-0'])==observation['mask_sha256'],'Motion pixels differ from registered source/mask')
    require(downloads['source-image-0'][:2]==b'\xff\xd8' and downloads['source-mask-0'][:8]==b'\x89PNG\r\n\x1a\n','Source image/mask header mismatch')
    return raw,view,evidence


async def collect(client,twin_id,experiment_id):
    collected=await collect_twin(client,twin_id=twin_id,max_evidence=6)
    mcp=RecordedMCP(client);mcp.receipts=collected['receipt']['calls']
    listing=await mcp.tool('list_motion_experiments',{'twin_id':twin_id})
    require(listing['program_sha256']==collected['source_program_sha256'] and any(r['experiment_id']==experiment_id for r in listing['experiments']),'Experiment was not discovered for this exact program')
    descriptor=await mcp.tool('get_motion_experiment',{'twin_id':twin_id,'experiment_id':experiment_id})
    descriptor_bytes=await mcp.resource(f'twin://{twin_id}/motion/{experiment_id}','application/json',maximum_bytes=4*1024*1024)
    require(strict_json(descriptor_bytes)==descriptor,'MCP motion tool/resource descriptors differ')
    downloads={}
    for key in ['raw','normalized','source-image-0','source-mask-0']:
        binding=descriptor['artifacts'][key]
        require(binding['uri']==f'twin://{twin_id}/motion/{experiment_id}/artifacts/{key}','Arbitrary motion resource URI rejected')
        downloads[key]=await mcp.resource(binding['uri'],'application/octet-stream',maximum_bytes=4*1024*1024 if key in {'raw','normalized'} else 8*1024*1024)
    raw,view,evidence=verify_motion(descriptor,downloads,collected)
    exact=await mcp.tool('get_evidence',{'twin_id':twin_id,'evidence_id':evidence['id']})
    require(exact['evidence']==evidence and exact['frame_uri']==f"twin://{twin_id}/evidence/{evidence['id']}",'Evidence tool differs from motion source')
    image=await mcp.resource(exact['frame_uri'],'image/jpeg',maximum_bytes=8*1024*1024)
    require(image==downloads['source-image-0'],'Motion and ordinary evidence resources differ')
    collected['evidence_bytes'][evidence['id']]=image
    collected['receipt']['evidence_frames_verified'][evidence['id']]=sha(image)
    state=await mcp.tool('query_state',{'twin_id':twin_id,'timestamp_s':evidence['timestamp_s']})
    verify_state(collected['document'],state,evidence['timestamp_s'])
    simulation=await mcp.tool('get_simulation_manifest',{'twin_id':twin_id})
    require(simulation['program_sha256']==collected['source_program_sha256'] and simulation['policy']['simulation_ready'] is False,'Unsafe simulator handoff')
    task={'format':'source-linked-motion-inspection-task','version':1,'purpose':'Inspect saved component alignment and source evidence; not robot execution',
        'twin_id':twin_id,'program_sha256':collected['source_program_sha256'],'experiment_id':experiment_id,'component_id':descriptor['component_id'],
        'inspection_source':descriptor['inspection_source'],'matched_comparison':descriptor['matched_comparison'],
        'reserved_display_counts':view['comparison_counts'],'exact_source_state':state,
        'all_procedure_steps':collected['document']['steps'],'simulation_ready':False,'no_LLM':True,
        'limitations':['Generated native model is an unreviewed single-image prior, not measured shape.',
            'Saved poses are conditional image checks in independent episode gauges.',
            'No continuous motion, common scale/assembly, joint/contact recovery or simulator execution.',
            'Portable viewer replays the source program; separate motion files preserve the new experiment.']}
    collected['receipt']['motion_experiment']={'experiment_id':experiment_id,'component_id':descriptor['component_id'],
        'raw_sha256':sha(downloads['raw']),'normalized_sha256':sha(downloads['normalized']),
        'source_sample_index':evidence['source_index'],'timestamp_s':evidence['timestamp_s']}
    return collected,descriptor,downloads,task,simulation


def write(collected,descriptor,downloads,task,simulation,output):
    output=Path(output).absolute();require(not output.exists() and not Path(str(output)+'.zip').exists(),'Use a fresh handoff output; old runs stay immutable')
    require(output.is_relative_to(ROOT/'data') and output.resolve()==output,'Handoff output must be a fresh nonsymlink data directory')
    output.mkdir(parents=True)
    result=write_package(collected,output/'portable')
    (output/'motion').mkdir()
    for name,data in [('raw.json',downloads['raw']),('normalized.json',downloads['normalized']),('source.jpg',downloads['source-image-0']),('source-mask.png',downloads['source-mask-0']),('descriptor.json',encoded(descriptor))]:
        (output/'motion'/name).write_bytes(data)
    for name,obj in [('inspection-task.json',task),('simulation-handoff.json',simulation),('sdk-trace.json',collected['receipt'])]:
        (output/name).write_bytes(encoded(obj))
    (output/'README.txt').write_text('LOCAL / PRIVATE REVIEW PACKAGE\n\nOpen portable/index.html after extracting this archive. This is an offline source-program viewer.\nThe separately saved motion/ files preserve the new same-source interval experiment; they do not replace the program or synthesize continuous motion.\n\nAll3 native GLBs and cited JPEGs were retrieved through actual read-only local MCP SDK calls. No LLM, API, model inference, robot or simulator was run by this client.\nNo complete source video is included. One explicit component mask is included in motion/source-mask.png.\n\nThe assets are rough, unreviewed, single-image model priors. Scale and physical shape remain unknown. Separate episodes/components have independent coordinates. Missing evidence stays unavailable. This is an inspection handoff, not a physics-ready twin.\n\nFiles are hash-pinned in bundle-manifest.json. Standard portable.zip can be independently validated with scripts/inspect_twin_package.py.\n')
    original_check=inspect_package(output/'portable');zip_check=inspect_package(output/'portable.zip')
    files={p.relative_to(output).as_posix():{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in sorted(output.rglob('*')) if p.is_file()}
    manifest={'format':'interaction-twin-motion-inspection-bundle','version':1,'created_at':datetime.now(timezone.utc).isoformat(),
        'twin_id':collected['twin_id'],'source_program_sha256':collected['source_program_sha256'],'experiment_id':descriptor['experiment_id'],
        'files':files,'scripted_calls':len(collected['receipt']['calls']),'native_assets':len(collected['asset_bytes']),
        'cited_jpegs':len(collected['evidence_bytes']),'steps':len(collected['document']['steps']),
        'source_video_included':False,'motion_mask_included':True,'simulation_ready':False,'model_inference':False,
        'portable_directory_verified':original_check,'portable_zip_verified':zip_check}
    (output/'bundle-manifest.json').write_bytes(encoded(manifest))
    archive=Path(str(output)+'.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():z.write(p,safe_name(p.relative_to(output).as_posix()))
    preview={'format':'fresh-source-linked-mcp-handoff-preview','version':1,'twin_id':collected['twin_id'],
        'program_sha256':collected['source_program_sha256'],'experiment_id':descriptor['experiment_id'],'component_id':descriptor['component_id'],
        'call_count':len(collected['receipt']['calls']),'assets':len(collected['asset_bytes']),'evidence_frames':len(collected['evidence_bytes']),
        'steps':len(collected['document']['steps']),'zip_path':archive.relative_to(ROOT).as_posix(),'zip_sha256':sha(archive.read_bytes()),
        'portable_zip_path':(output/'portable.zip').relative_to(ROOT).as_posix(),'portable_zip_sha256':sha((output/'portable.zip').read_bytes()),
        'task_path':(output/'inspection-task.json').relative_to(ROOT).as_posix(),'trace_path':(output/'sdk-trace.json').relative_to(ROOT).as_posix(),
        'bundle_manifest_path':(output/'bundle-manifest.json').relative_to(ROOT).as_posix(),'bundle_manifest_sha256':sha((output/'bundle-manifest.json').read_bytes()),
        'source_image_path':(output/'motion/source.jpg').relative_to(ROOT).as_posix(),'source_mask_path':(output/'motion/source-mask.png').relative_to(ROOT).as_posix(),
        'source_evidence_id':descriptor['inspection_source']['program_evidence_id'],'source_timestamp_s':descriptor['inspection_source']['timestamp_s'],
        'source_sample_index':descriptor['inspection_source']['sample_index'],'reserved_comparison':descriptor['motion']['comparison_counts'],
        'policy':{'no_LLM':True,'model_inference':False,'simulation_executed':False,'simulation_ready':False,'private_local_only':True,'full_source_video_included':False}}
    preview_path=Path(str(output)+'-preview.json');require(not preview_path.exists(),'Preserve prior preview')
    preview_path.write_bytes(encoded(preview))
    return preview|{'preview_path':preview_path.relative_to(ROOT).as_posix()}


async def run(args):
    from mcp import Client,StdioServerParameters
    params=StdioServerParameters(command=str(ROOT/'.venv-mcp/bin/python'),args=[str(ROOT/'scripts/serve_twin_mcp.py')],cwd=ROOT)
    async with Client(params,mode='auto',read_timeout_seconds=90) as client:
        values=await collect(client,identifier(args.twin_id),identifier(args.experiment_id))
    print(json.dumps(write(*values,ROOT/args.output),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--twin-id',default=DEFAULT_TWIN);parser.add_argument('--experiment-id',default=DEFAULT_EXPERIMENT)
    parser.add_argument('--output',default='data/mcp-fresh-jabra-motion-handoff-20261101-v1')
    asyncio.run(run(parser.parse_args()))
