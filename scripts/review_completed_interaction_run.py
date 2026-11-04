#!/usr/bin/env python3
"""Read-only completed pipeline receipt/semantic/child-result consistency check."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from stage5 import validate_interaction
from observed_correspondence_review import review


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def run(run_path,output):
    run_path=Path(run_path).resolve();output=Path(output).resolve()
    if not run_path.is_relative_to(ROOT/'data') or not output.is_relative_to(ROOT/'data') or output.exists():
        raise ValueError('Require a saved run and a fresh project data output')
    output.mkdir(parents=True);(output/'implementation.py').write_bytes(Path(__file__).read_bytes())
    state=read(run_path);program_path=ROOT/state['artifacts']['interaction'];program=read(program_path)
    validation=validate_interaction(program,root=ROOT,verify_files=True)
    failures=[];parent_count=0;child_count=0;children=[];motion=[]
    def verify(path,expected,scope):
        path=Path(path).resolve()
        if not any(path.is_relative_to(ROOT/name) for name in ['data','workspace','annotations']):
            raise ValueError('Only saved data/workspace/annotation receipts are reviewed')
        if not path.is_file() or digest(path)!=expected:failures.append({'scope':scope,'path':str(path.relative_to(ROOT))})
    for phase in state['phases']:
        for item in phase.get('files',[]):
            verify(ROOT/item['path'],item['sha256'],'parent_phase_'+phase['name']);parent_count+=1
    orchestrator=state['provenance']['orchestrator_source'];verify(ROOT/orchestrator['path'],orchestrator['sha256'],'frozen_parent_orchestrator')
    for metadata_path in state['artifacts']['motion']:
        pack=read(ROOT/metadata_path)
        counts=Counter(f['status'] for f in pack['frames'])
        motion.append({'component_id':pack['object_id'],'experiment_path':metadata_path,'sha256':digest(ROOT/metadata_path),
            'frame_count':pack['frame_count'],'supported_nonreference_frames':pack['supported_nonreference_frames'],
            'unsupported_nonreference_frames':pack['unsupported_nonreference_frames'],'reference_frames':pack['reference_frames'],
            'frame_status_counts':dict(counts),'usable_physical_pose_recovered':pack['usable_physical_pose_recovered'],
            'physical_pose_accuracy_measured':pack['physical_pose_accuracy_measured'],
            'source_matches_program':pack['provenance']['source_video_sha256']==program['source']['sha256']})
    for item in state['artifacts'].get('observed_reconstructions',[]):
        job_path=ROOT/item['job_url'].removeprefix('/files/');job=read(job_path)
        workflow_path=ROOT/item['workflow_manifest_url'].removeprefix('/files/');workflow=read(workflow_path)
        if job['interaction_sha256']!=digest(program_path) or job['interaction_path']!=str(program_path.relative_to(ROOT)) or job['track_id'] not in {t['id'] for t in program['tracklets']}:
            failures.append({'scope':'child_program_binding','id':job['id']})
        for stage in workflow['stages']:
            for path,expected in stage.get('artifact_sha256',{}).items():
                verify(ROOT/path,expected,'child_'+job['id']+'_'+stage['stage']);child_count+=1
        sparse=ROOT/item['result_url'].removeprefix('/files/')
        correspondence=review(sparse.parent,root=ROOT)
        if not correspondence['passed']:failures.append({'scope':'child_correspondence_integrity','id':job['id'],'errors':correspondence['errors']})
        children.append({'id':job['id'],'track_id':job['track_id'],'status':job['status'],
            'workflow_status':workflow['status'],'stages':[{'stage':s['stage'],'status':s['status']} for s in workflow['stages']],
            'program_sha256':job['interaction_sha256'],'result_url':item['result_url'],'metrics':job['metrics'],
            'exports':workflow.get('exports',{}),'correspondence_review':correspondence})
    missing=sorted({t['id'] for t in program['tracklets']}-{a['track_id'] for a in program['assets']})
    report={'format':'interaction-twin-completed-run-consistency-review','version':1,
        'passed':bool(validation) and not failures,'failures':failures,'run_id':state['id'],'status':state['status'],
        'program_path':str(program_path.relative_to(ROOT)),'program_sha256':digest(program_path),
        'program_schema_and_all_source_asset_mask_pts_motion_joins_valid':validation,
        'parent_phase_artifact_hashes_checked':parent_count,'child_stage_artifact_hashes_checked':child_count,
        'frozen_parent_orchestrator':orchestrator,'current_later_orchestrator_code_assumed_to_have_run':False,
        'tracklet_count':len(program['tracklets']),'native_asset_count':len(program['assets']),
        'components_without_native_asset':missing,'procedure_step_count':len(program['steps']),'assertion_count':len(program['assertions']),
        'motion':motion,'observed_children':children,
        'interpretation':['A completed phase records computation completion. Sparse conditional motion and model-generated native assets do not establish a continuous recovered physical interaction.',
            'The native model assets are separate from observed geometry. Unsupported dense/completion/export children remain unsupported and do not replace native assets.',
            'Physical identity, intrinsics, metric scale, common assembly, contacts and simulator readiness remain unverified.'],
        'new_api_calls':0,'new_model_inference_calls':0,'manual_geometry_or_poses':False,
        'host_timestamp_utc':datetime.now(timezone.utc).isoformat(),'client_task_date':'2026-10-31',
        'implementation_sha256':digest(__file__)}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();report=run(ROOT/a.run,ROOT/a.output)
    print(json.dumps({k:report[k] for k in ['passed','failures','status','parent_phase_artifact_hashes_checked','child_stage_artifact_hashes_checked','components_without_native_asset']}))
