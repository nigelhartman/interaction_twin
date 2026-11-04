#!/usr/bin/env python3
"""Freeze a small local display contract from a completed genuine assessment."""
from pathlib import Path,PurePosixPath
from decimal import Decimal
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mcp_assessment as A

def artifact(path):
 p=Path(path);A.require(p.resolve()==p and p.is_file() and p.is_relative_to(ROOT/'data'),'Expected genuine confined data artifact')
 return {'path':str(p.relative_to(ROOT)),'sha256':A.sha(p),'bytes':p.stat().st_size}

def build():
 directory=ROOT/'data/model-directed-mcp-assessment-20261101/meta-run-v3'
 target=directory.parent/'meta-run-v3-preview.json';A.require(not target.exists(),'Preserve the frozen preview; make a new version for changes')
 report=A.read(directory/'assessment.json');manifest=A.read(directory/'artifact-manifest.json');protocol=A.read(directory/'protocol.json')
 A.require(report['status']==manifest['status']=='complete_fallible_assessment' and report['is_genuine_model_directed'] is True and manifest['model_plan_is_assistant_authored'] is False,'A genuine complete assessment is required')
 A.verify_pins(protocol,ROOT);A.require(report['protocol_sha256']==A.sha(directory/'protocol.json'),'Protocol identity changed')
 for name,item in manifest['files'].items():
  rel=PurePosixPath(name);A.require(not rel.is_absolute() and '..' not in rel.parts and not any(p.startswith('.') for p in rel.parts),'Unsafe saved artifact')
  p=directory/name;A.require(p.resolve()==p and p.is_file() and p.stat().st_size==item['bytes'] and A.sha(p)==item['sha256'],'Saved assessment artifact changed')
 receipts=report['tool_receipts'];A.require(2<=len(report['rounds'])<=4 and len(receipts)==manifest['actual_sdk_tool_calls']<=8,'Unexpected genuine round/tool count')
 final=A.read(directory/f"round-{len(report['rounds']):02d}/model-plan.json")
 A.validate_action(final,protocol,tool_count=len(receipts),round_number=len(report['rounds']),receipts=receipts)
 A.require(final['assessment']==report['final_assessment'],'Model claim strings were changed')
 costs=[];total=Decimal('0');tools=[]
 for row in report['rounds']:
  p=directory/f"round-{row['round']:02d}/meta-request.json";receipt=A.read(p)
  A.require(receipt['status']=='complete_accounted' and receipt['http_status']==200 and receipt['image_inputs']==[] and receipt['model']==report['policy']['model'],'Unexpected provider receipt')
  amount=Decimal(receipt['estimated_cost_usd']);A.require(amount.is_finite() and 0<=amount<=2,'Unexpected reported-usage estimate');total+=amount
  costs.append({'round':row['round'],'request_id':receipt['request_id'],'receipt':artifact(p),'usage':receipt['usage'],
   'reported_usage_estimated_usd':str(amount),'reservation_estimate_usd':receipt['estimate_usd']})
 for row in receipts:
  name=row['receipt_id'];p=directory/name/'sdk-receipt.json';sdk=A.read(p)
  A.require(sdk==row['sdk_receipt'] and sdk['is_error'] is False and sdk['name']==row['tool'],'SDK receipt changed')
  tools.append({'receipt_id':name,'tool':row['tool'],'arguments':row['arguments'],'returned_entity_ids':row['returned_entity_ids'],
   'sdk_receipt':artifact(p),'structured_result':artifact(directory/name/'structured-result.json'),'pruned_result':artifact(directory/name/'pruned-result.json')})
 raw=A.read(ROOT/A.RAW);source=next(r for e in raw['episodes'] for r in e['frames'] if r['sample_index']==53)
 image=ROOT/source['source_image_url'].removeprefix('/files/');mask=ROOT/source['mask_url'].removeprefix('/files/')
 A.require(A.sha(image)==source['source_image_sha256'] and A.sha(mask)==source['mask_sha256'],'Selected source bytes changed')
 failures=[]
 for name in ['meta-run-v1','meta-run-v2']:
  p=directory.parent/name/'assessment.json';old=A.read(p)
  failures.append({'run_id':name,'status':old['status'],'error':old.get('error'),'assessment':artifact(p),
   'artifact_manifest':artifact(p.parent/'artifact-manifest.json'),'planning_rounds_with_returned_plans':len(old['rounds']),
   'successful_pruned_tool_calls':len(old['tool_receipts'])})
 preview={'format':'model-directed-mcp-assessment-preview','version':1,'run_id':'meta-run-v3','status':report['status'],
  'twin_id':report['twin_id'],'component_id':'case_base','experiment_id':report['experiment_id'],
  'program_sha256':report['program_sha256'],'motion_raw_sha256':report['motion_raw_sha256'],
  'model':report['policy']['model'],'planning_rounds':len(report['rounds']),'actual_read_only_tool_calls':len(receipts),
  'assessment_path':str((directory/'assessment.json').relative_to(ROOT)),'assessment_sha256':A.sha(directory/'assessment.json'),
  'manifest_path':str((directory/'artifact-manifest.json').relative_to(ROOT)),'manifest_sha256':A.sha(directory/'artifact-manifest.json'),
  'protocol':artifact(directory/'protocol.json'),'source_program':artifact(ROOT/A.PROGRAM),'source_motion':artifact(ROOT/A.RAW),
  'native_asset':protocol['native_asset'],'source_video_sha256':protocol['source_video']['sha256'],
  'source_evidence':{'evidence_id':'source_000053','sample_index':53,'timestamp_s':source['timestamp_s'],'image':artifact(image),'mask':artifact(mask)},
  'final_assessment':report['final_assessment'],'actual_tool_call_order':tools,'model_plans':[
   {'round':r['round'],'action':r['action'],'plan':artifact(directory/f"round-{r['round']:02d}/model-plan.json"),
    'prompt':artifact(directory/f"round-{r['round']:02d}/prompt.txt")} for r in report['rounds']],
  'cost_sources':costs,'reported_usage_estimated_total_usd':str(total),
  'cost_is_not_live_account_balance_or_final_invoice':True,'prior_preserved_failures':failures,
  'policy':{'genuine_model_directed':True,'fallible_model_assessment':True,'verbatim_model_claims':True,
   'pruned_text_metadata_only':True,'model_received_rgb':False,'model_received_binary_assets':False,
   'citation_validation':'actual receipt/entity identity only; not semantic certification',
   'read_only_tools':True,'simulation_ready':False,'physical_accuracy_verified':False,'geometry_or_pose_changed':False,
   'universal_secret_scrubber_claimed':False,'private_local_only':True,'publicly_published':False}}
 A.put(target,preview);print(json.dumps({'path':str(target.relative_to(ROOT)),'sha256':A.sha(target),'bytes':target.stat().st_size,'rounds':len(report['rounds']),'tool_calls':len(receipts),'reported_usage_estimated_total_usd':str(total)}))

if __name__=='__main__':build()
