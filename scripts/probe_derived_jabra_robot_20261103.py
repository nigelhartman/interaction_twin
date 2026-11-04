#!/usr/bin/env python3
"""Offline exact proposed Jabra task probe; no approval or public side effects."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from curated_twin_catalog import CuratedTwinCatalog
import refined_assembly_contract as D
from robot_task_program import plan_robot_task,validate_robot_task


def probe(output,root=ROOT):
    root=Path(root).resolve();output=Path(output)
    if not output.is_absolute():output=root/output
    if output.resolve()!=output or not output.is_relative_to(root/'output')or output.exists():
        raise ValueError('A fresh ordinary output receipt path is required')
    adoption=json.loads((root/'output/submission-finalization-20261103/jabra/cached-adoption-plan-v2.json').read_text())
    reference=adoption['contract']['derived_assembly_ref'];refs={x['path']:x for x in adoption['result']['artifacts']}
    document=D.read(reference,root);D.validate(document,root=root)
    class DiagnosticCatalog(CuratedTwinCatalog):
        def _record(self,twin_id):
            record,row=super()._record(twin_id);record['derived_assembly']=copy.deepcopy(reference)
            # Publication may be between native fallback and revised approval.
            # Only the exact frozen same-source legacy A review is admitted to
            # this explicitly private diagnostic, never a different candidate.
            if not any(x['component_id']=='earbud_a'for x in record.get('sam3d_reviews',[])):
                legacy=json.loads((root/'.private/admin-workflow/legacy-showcase.json').read_text())
                old=next(x['record']for x in legacy['records']if x['record']['interaction']==record['interaction']and x['record']['source']['sha256']==record['source']['sha256'])
                native=next(x for x in old['sam3d_reviews']if x['component_id']=='earbud_a')
                expected=next(x for x in document['component_assets']if x['component_id']=='earbud_a')['original_native_glb']
                if any(native['native_glb'][key]!=expected[key]for key in ['sha256','bytes']):raise ValueError('Frozen legacy A is not the exact original derivative input')
                D.O.checked_ref(native['native_glb'],root,16*1024*1024)
                refs[native['native_glb']['path']]=copy.deepcopy(native['native_glb'])
                record['sam3d_reviews'].append(copy.deepcopy(native))
            return record,row
        def derived_assembly(self,twin_id):
            return{'artifact':copy.deepcopy(reference),'document':copy.deepcopy(document)}
        def kinematics(self,twin_id):
            self._record(twin_id)
            # The separate native baseline was already fully verified by
            # D.validate. Keep the probe independent of publication timing.
            base=document['base_object_program_ref']
            return{'artifact':copy.deepcopy(base),'object_program':D.read(base,root)}
        def _checked(self,item,*,maximum=None):
            if item.get('path')in refs:
                known=refs[item['path']]
                if any(item.get(k)!=known[k]for k in ['path','sha256','bytes']):raise ValueError('Diagnostic reference changed')
                return D.O.checked_ref(known,root,maximum or 32*1024*1024)
            return super()._checked(item,maximum=maximum)
    catalog=DiagnosticCatalog(root);twin_id='flow-sam3d-default-e2e-20261102-v1--flow-procedure-1'
    instructions=['Place the case on the table, open the lid, take out both earbuds and place them on the tray, then close it.',
                  'Open the lid and remove earbud_a to the tray.',
                  'Return earbud_a from the tray to the open case, then close it.',
                  'Place the case and open the lid.']
    plans=[]
    for text in instructions:
        plan=plan_robot_task(twin_id,text,root=root,catalog=catalog)
        validation=validate_robot_task(plan,root=root,catalog=catalog)
        if not validation['valid']:raise ValueError('The derived authored task is not valid: '+json.dumps(validation['errors']))
        plans.append({'text':text,'plan':plan,'validation':validation})
    result={'format':'interaction-twin-offline-derived-task-probe','version':1,'diagnostic_only_not_public':True,
            'source_bound_derivative_ref':reference,'plans':plans,'new_api_calls':0,'new_gpu_calls':0}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);args=parser.parse_args()
    result=probe(args.output)
    print(json.dumps({'output':str(args.output),'plans':len(result['plans']),'actions':[len(x['plan']['actions'])for x in result['plans']],
                      'all_valid':all(x['validation']['valid']for x in result['plans'])}))
