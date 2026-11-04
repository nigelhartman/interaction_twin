#!/usr/bin/env python3
"""Explicit owner-delegated adoption of two already generated SAM scene drafts.

No inference, networking, credentials, geometry fitting or worker dispatch. The
exact reviewed revisions and existing source recordings are pinned below.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUTPUT=ROOT/'output/connected-library-20261104-v1'
TARGETS=[
    ('camera','candidate_dfb49d89d0064c27a778e0f8eb70a66f',
     'revision_764b89c4e80e4e4b8b010822fbf4fffd',
     '106c3fec649e0a6b58bb7c10f0231259cbb204d4e49942c3e77e2725b75d677c'),
    ('wallet','candidate_8c153ad1b14947f3921aa8e641dfb59f',
     'revision_4639152d3b824999a1e259978d1471b3',
     'c27d403bdc2b72e57783a0c5b980244ec085df19f31c7e096325fd4943aa06fd')]

def main():
    import curated_publication as C
    from admin_operations import budget_and_usage
    from admin_workflow import AdminWorkflow
    from scripts import build_workspace as W
    import object_program
    OUTPUT.mkdir(exist_ok=True)
    receipt=OUTPUT/'adoption.json'
    if receipt.exists():raise ValueError('Already recorded; inspect the preserved receipt')
    workflow=AdminWorkflow(ROOT);before=budget_and_usage(ROOT)[0]
    entries=[]
    for name,candidate,revision,fingerprint in TARGETS:
        detail=workflow.detail(candidate)
        if detail['candidate']['current_revision_id']!=revision:raise ValueError('Revision changed')
        row=next(r for r in detail['revisions']if r['id']==revision)
        if row['fingerprint']!=fingerprint:raise ValueError('Fingerprint changed')
        result=row['manifest'].get('result',row['manifest']);program=result['interaction_ref']
        companion=object_program.read_json(object_program.checked_ref(result['object_program_ref'],ROOT))
        object_program.validate(companion,root=ROOT)
        if companion['evidence_program']!=program:raise ValueError('Companion source differs')
        entries.append((name,detail,row,result))
    old_catalog=C.ref(ROOT/'showcase/public/library-catalog.json',ROOT)
    if not (OUTPUT/'previous-public-catalog.json').exists():
        (OUTPUT/'previous-public-catalog.json').write_bytes((ROOT/old_catalog['path']).read_bytes())
    registry=ROOT/'workspace/registered-cases.json';old_registry=registry.read_bytes()
    if not (OUTPUT/'previous-registered-cases.json').exists():
        (OUTPUT/'previous-registered-cases.json').write_bytes(old_registry)
    registered=json.loads(old_registry)
    for name,detail,row,result in entries:
        path=result['interaction_ref']['path']
        if not any(r['interaction']==path for r in registered['cases']):
            registered['cases'].append({'interaction':path,'title':name.capitalize()+' · source-linked SAM 3D scene'})
    C.atomic(registry,json.dumps(registered,indent=2)+'\n')
    indexed=C.read(ROOT/'workspace/cases-index.json')['cases']
    if not all(any(r['interaction_path']==result['interaction_ref']['path'] and r['original_sha256']==result['interaction_ref']['sha256']for r in indexed)for _,_,_,result in entries):
        W.build(publish_replays=False)
    indexed=C.read(ROOT/'workspace/cases-index.json')['cases']
    for _,_,_,result in entries:
        if not any(r['interaction_path']==result['interaction_ref']['path'] and r['original_sha256']==result['interaction_ref']['sha256']for r in indexed):
            raise ValueError('Exact program was not indexed; no approval')
    approvals=[]
    for name,detail,row,result in entries:
        prior=next((d for d in detail['decisions']if d['action']=='approve'and d['revision_id']==row['id']),None)
        decision=prior or workflow.approve(detail['candidate']['id'],row['id'],row['fingerprint'])
        approvals.append({'case':name,'authorization':'Owner requested connected scenes for all examples; delegated local research-draft curation. No physical acceptance or owner visual inspection is claimed.',
            'reused_exact_approval':prior is not None,'decision':decision})
    # Import the separately verified cadence-only camera derivative as a new
    # immutable revision. Its poses, geometry and unknown barriers are unchanged.
    from admin_pipeline import FixedExecutor
    recipe=C.read(ROOT/'output/connected-scene-audit-20261104-v1/camera-cadence-fixed-import-recipe-v1.json')
    run='run-admin-camera-cadence-20261104-v1'
    previous=next((r for r in workflow.snapshot()['candidates']if r['run_id']==run),None)
    if previous is None:
        if any(j['state']in {'queued','running'}for j in workflow.snapshot()['jobs']):
            raise ValueError('Keep unrelated jobs untouched; owner queue must be idle')
        recipe['result']['run_id']=run
        destination=ROOT/'.private/admin-imports'/(run+'.json')
        with destination.open('x')as f:json.dump(recipe,f,indent=2);f.write('\n')
        destination.chmod(0o600)
        submission=workflow.submit_upload(run_id=run,input_path=recipe['source_ref']['path'],
            source_ref=recipe['source_ref'],processing_consent=True,object_family='hinged_assemblies')
        outcome=workflow.worker_once(FixedExecutor(root=ROOT))
        if outcome['job_id']!=submission['job']['id']or outcome['status']!='complete':
            raise ValueError('Exact local cadence import did not complete')
        previous=submission['candidate']
    detail=workflow.detail(previous['id']);row=detail['revisions'][0]
    if row['manifest']['object_program_ref']!=recipe['result']['object_program_ref']:
        raise ValueError('Cadence import selected another companion')
    prior=next((d for d in detail['decisions']if d['action']=='approve'and d['revision_id']==row['id']),None)
    approvals.append({'case':'camera_cadence','authorization':'Owner delegated connected replay; bounded adjacent-source display interpolation only. No fresh physical or visual certification.',
        'reused_exact_approval':prior is not None,'decision':prior or workflow.approve(previous['id'],row['id'],row['fingerprint'])})
    labels=C.read(ROOT/'config/library-presentation.json')
    labels['recordings']['42bc62511fe63ac7cbe6c0e32f017d89b349706be6447c23f41ad9795ad0f43f']['note']='SAM 3D components and connected source replay. The door has seven supported hinge samples; opening angles and physical contact remain estimates.'
    labels['recordings']['1546de58a21b52eb30020d5fafc2b5ca6c88a7c22322419999f5d64f72011cf1']['note']='SAM 3D wallet and coin with source-linked replay. Flexible flap motion remains withheld; no rigid leather hinge or physical fit is claimed.'
    C.atomic(ROOT/'config/library-presentation.json',json.dumps(labels,indent=2)+'\n')
    result=C.build(root=ROOT)
    after=budget_and_usage(ROOT)[0]
    if before!=after:raise ValueError('A local publication must not change the API ledger')
    catalog=C.read(ROOT/'showcase/public/library-catalog.json')
    scenes=C.read(ROOT/'showcase/public/aligned-scenes.json')['scenes']
    coverage=[]
    for record in catalog['records']:
        scene=next(s for s in scenes if s['source_sha256']==record['source']['sha256'])
        doc=C.read(ROOT/scene['manifest']['path'])
        if doc['original_program']!=record['interaction']:raise ValueError('Scene/program mismatch')
        coverage.append({'id':record['id'],'source_sha256':record['source']['sha256'],'manifest':scene['manifest'],'components':len(doc['components'])})
    value={'format':'interaction-twin-explicit-connected-library-adoption','version':1,'automatic_inference':False,
        'authorization':'Owner asked to make every example have a connected scene; adopts exact cached SAM 3D research drafts.',
        'approvals':approvals,'previous_catalog':old_catalog,'publication':result,'coverage':coverage,
        'new_api_calls':0,'new_gpu_calls':0,'geometry_or_poses_modified':False,
        'physical_accuracy_verified':False,'simulation_ready':False,'budget_unchanged':True}
    with receipt.open('x')as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps({'receipt':C.ref(receipt,ROOT),'publication':result,'connected_recordings':len(coverage)},indent=2))

if __name__=='__main__':main()
