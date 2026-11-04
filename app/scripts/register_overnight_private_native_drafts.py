#!/usr/bin/env python3
"""Adopt two verified native drafts into the owner desk without model calls.

These are explicit source-bound imports, not fresh automatic runs or approvals.
Only deterministic structural reviews run here; hosted/visual review is absent.
"""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    from admin_workflow import AdminWorkflow
    from admin_pipeline import FixedExecutor
    from object_program import ref
    from review_agents import verified_ref
    from overnight_budget import budget_status
    out=ROOT/'output/overnight-pipeline-20261102'
    receipt=out/'private-native-drafts-registration-v1.json'
    assert not receipt.exists(),'Registration already recorded; do not duplicate'
    workflow=AdminWorkflow(ROOT)
    with workflow.db()as db:
        assert not db.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running')").fetchone(),'Owner queue must be idle'
    before=budget_status(root=ROOT)
    submissions=[]
    for case in ['camera','wallet']:
        recipe_path=out/'camera-wallet-native-scenes-v1'/f'{case}-admin-import-recipe.json'
        recipe=json.loads(recipe_path.read_text())
        verified_ref(ROOT,recipe['source_ref'])
        for reference in recipe['result']['artifacts']:verified_ref(ROOT,reference)
        run=recipe['result']['run_id']
        active=ROOT/'.private/admin-imports'/(run+'.json')
        assert not active.exists(),'Exact import intent already exists'
        with active.open('x')as stream:json.dump(recipe,stream,indent=2);stream.write('\n')
        active.chmod(0o600)
        submission=workflow.submit_upload(run_id=run,input_path=recipe['source_ref']['path'],
            source_ref=recipe['source_ref'],processing_consent=True,
            object_family='hinged_assemblies'if case=='camera'else'flexible_or_uncertain')
        submissions.append({'case':case,'recipe_ref':ref(recipe_path,ROOT),'submission':submission})
    executor=FixedExecutor(root=ROOT)
    outcomes=[]
    for _ in submissions:
        outcomes.append(workflow.worker_once(executor))
    after=budget_status(root=ROOT)
    assert before==after,'This structural-only import must not spend or contact a model'
    result={'format':'private-native-draft-registration','version':1,'submissions':submissions,
        'outcomes':outcomes,'new_api_calls':0,'new_gpu_calls':0,
        'hosted_reviews_requested':False,'human_approval':False,'automatic_publication':False,
        'budget':after}
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'private_candidates':[x['submission']['candidate']['id']for x in submissions],
        'new_api_calls':0,'automatic_publication':False}))

if __name__=='__main__':main()
