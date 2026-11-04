"""Offline, exact alias normalization of an already-accounted visual response.

This is an explicitly labelled local derivative, never a new Meta response.
Only source aliases supplied in the immutable packet may change; verdicts,
observations, claims and proposals remain byte-for-value unchanged. The original
judge's semantic validation still applies, including contradictory verdicts.
"""
from __future__ import annotations
import copy,json
from pathlib import Path

import jsonschema
import generation_visual_judge as J
import semantic_asset_companion as S

ROOT=J.ROOT
FORMAT='interaction-twin-generation-visual-alias-derivative'


def _require(condition,message):
    if not condition:raise ValueError(message)


def aliases(plan):
    """Return only a bijection between exact supplied same-component samples."""
    ctx=plan['context'];samples=[ctx['anchor'],*ctx['candidate_anchors']]
    mapping={};canonical=set();candidate_aliases={}
    for index,sample in enumerate(samples):
        identifier=sample['id'];alias=sample['evidence_id']
        _require(identifier==f'{ctx["component_id"]}:{sample["source_index"]}' and
                 isinstance(alias,str) and alias and alias!='none' and not alias.startswith('orbit:'),
                 'Alias must name an exact supplied same-component source sample')
        _require(alias not in mapping and identifier not in canonical,'Source aliases must be bijective')
        mapping[alias]=identifier;canonical.add(identifier)
        if index:candidate_aliases[alias]=identifier
    _require(not (set(mapping)&canonical),'Source aliases must not collide with canonical IDs')
    return mapping,candidate_aliases


def normalize_result(raw,plan):
    """No inference or client construction; no semantic repair of the response."""
    jsonschema.validate(raw,J.schema());mapping,candidate_aliases=aliases(plan)
    _require(raw['component_id']==plan['context']['component_id'],'Response names another component')
    result=copy.deepcopy(raw);edits=[]
    canonical=set(mapping.values());allowed=canonical|{f'orbit:{i}'for i in range(4)}
    def resolve(value,lookup,accepted,pointer):
        new=lookup.get(value,value)
        _require(new in accepted,'Response contains unsupplied evidence; alias normalization cannot invent it')
        if new!=value:edits.append({'json_pointer':pointer,'from':value,'to':new})
        return new
    for collection in ['evidence','issues']:
        for i,item in enumerate(result[collection]):
            item['evidence_ids']=[resolve(v,mapping,allowed,f'/{collection}/{i}/evidence_ids/{j}')
                                  for j,v in enumerate(item['evidence_ids'])]
    candidates=set(candidate_aliases.values())|{'none'}
    for i,item in enumerate(result['anchor_suggestions']):
        item['candidate_evidence_id']=resolve(item['candidate_evidence_id'],candidate_aliases,candidates,
                                              f'/anchor_suggestions/{i}/candidate_evidence_id')
    try:J.validate_result(result,plan)
    except (ValueError,jsonschema.ValidationError) as error:
        validation={'status':'rejected','reason':str(error)[:1200]}
    else:validation={'status':'valid_fallible_offline_derivative','reason':None}
    return result,edits,validation


def _source(packet,root):
    folder=J._directory(packet,root);plan,_,_=J.verify_plan(folder,root=root)
    raw_ref=S.file_ref(folder/'attempt/model-result.json',root,128000)
    pending_ref=S.file_ref(folder/'attempt/request/response.pending.json',root,128000)
    receipt_ref=S.file_ref(folder/'attempt/request/meta-request.json',root,256000)
    raw=S._json(S.checked(raw_ref,root,128000));pending=S._json(S.checked(pending_ref,root,128000))
    receipt=S._json(S.checked(receipt_ref,root,256000))
    _require(raw==pending,'Raw response differs from the original saved provider parser output')
    _require(receipt.get('status')=='complete_accounted' and receipt.get('provider')=='meta_api' and
             receipt.get('model')==J.MODEL and receipt.get('automatic_retries')is False and
             receipt.get('payload_sha256')==plan['wire_provenance']['payload_sha256'] and
             receipt.get('image_inputs')==plan['wire_provenance']['encoded_image_inputs'] and
             isinstance(receipt.get('estimated_cost_usd'),str),'Exact accounted original image/payload receipt required')
    return plan,raw,receipt,{'plan_ref':S.file_ref(folder/'plan.json',root),
         'raw_result_ref':raw_ref,'provider_parser_output_ref':pending_ref,'request_receipt_ref':receipt_ref}


def build(packet,output,*,root=ROOT):
    root=Path(root).resolve();plan,raw,receipt,refs=_source(packet,root)
    result,edits,validation=normalize_result(raw,plan)
    folder=J._directory(output,root,new=True);folder.mkdir(parents=True,mode=0o700)
    J._write(folder/'implementation.py',Path(__file__).read_bytes())
    J._write(folder/'normalized-result.json',result)
    mapping,_=aliases(plan)
    proof={'format':FORMAT,'version':1,**refs,'request_key':plan['request_key'],
        'component_id':plan['context']['component_id'],'source_program_ref':plan['context']['program_ref'],
        'asset_ref':plan['context']['native']['asset'],'supplied_source_aliases':mapping,
        'normalization_edits':edits,'normalized_result_ref':S.file_ref(folder/'normalized-result.json',root),
        'validation':validation,'estimated_original_cost_usd':receipt['estimated_cost_usd'],
        'request_id':receipt.get('request_id'),'implementation_ref':S.file_ref(folder/'implementation.py',root),
        'original_judge_implementation_ref':plan['implementation_ref'],
        'origin':'deterministic_offline_alias_derivative_not_raw_Meta_response',
        'semantic_content_modified':False,'raw_response_modified':False,'original_accounting_modified':False,
        'human_review_required':True,'new_api_calls':0,'automatic_retry':False,**J.FLAGS,
        'saved_agent_qa_comparison':{'agent_observation':plan['saved_agent_qa'],
            'raw_model_verdict':raw['verdict'],'derivative_validation_status':validation['status'],
            'agreement_is_not_accuracy_measurement':True,'same_issue_requires_owner_inspection':True}}
    J._write(folder/'normalization-proof.json',proof)
    verify(folder/'normalization-proof.json',root=root)
    return S.file_ref(folder/'normalization-proof.json',root)


def verify(path,*,root=ROOT):
    root=Path(root).resolve();proof=S._json(S.checked(S.file_ref(path,root,128000),root,128000))
    _require(proof.get('format')==FORMAT and proof.get('version')==1,'Unknown alias derivative proof')
    _require(S.checked(proof['implementation_ref'],root)==Path(__file__).read_bytes() and
             S.checked(proof['original_judge_implementation_ref'],root)==Path(J.__file__).read_bytes(),
             'Offline normalization implementation changed')
    S.checked(proof['plan_ref'],root,4*1024*1024)
    plan,raw,receipt,refs=_source(Path(proof['plan_ref']['path']).parent,root)
    _require(proof['original_judge_implementation_ref']==plan['implementation_ref'],'Original judge archive binding changed')
    _require(all(proof[k]==v for k,v in refs.items()),'Original response references changed')
    normalized,edits,validation=normalize_result(raw,plan)
    _require(S._json(S.checked(proof['normalized_result_ref'],root,128000))==normalized and
             proof['normalization_edits']==edits and proof['validation']==validation and
             proof['supplied_source_aliases']==aliases(plan)[0],
             'Derivative is not the exact permitted source-alias transformation')
    _require(proof['request_key']==plan['request_key'] and proof['component_id']==plan['context']['component_id'] and
             proof['source_program_ref']==plan['context']['program_ref'] and proof['asset_ref']==plan['context']['native']['asset'] and
             proof['estimated_original_cost_usd']==receipt['estimated_cost_usd'] and proof['request_id']==receipt.get('request_id'),
             'Derivative source or original accounting binding changed')
    _require(proof['origin']=='deterministic_offline_alias_derivative_not_raw_Meta_response' and
             all(proof.get(k)is False for k in ['semantic_content_modified','raw_response_modified','original_accounting_modified','automatic_retry']) and
             all(proof.get(k)is v for k,v in J.FLAGS.items()) and proof['human_review_required']is True and
             type(proof['new_api_calls'])is int and proof['new_api_calls']==0,'Derivative provenance/claim policy changed')
    _require(proof['saved_agent_qa_comparison']=={'agent_observation':plan['saved_agent_qa'],
        'raw_model_verdict':raw['verdict'],'derivative_validation_status':validation['status'],
        'agreement_is_not_accuracy_measurement':True,'same_issue_requires_owner_inspection':True},'Saved QA comparison changed')
    return {'status':validation['status'],'validation_reason':validation['reason'],
        'component_id':proof['component_id'],'verdict':normalized['verdict'],
        'human_review_required':True,'new_api_calls':0,'raw_response_modified':False}
