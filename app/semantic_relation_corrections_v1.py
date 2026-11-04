"""Explicit source-reviewed semantic overlays; original model programs stay exact.

A selected overlay expires when its source/program/approved revision/scene changes.
It grants no geometry, fitting, seating, physical execution or human-review claim.
"""
from __future__ import annotations
import copy,hashlib,json,re
from pathlib import Path

FORMAT='interaction-twin-source-reviewed-relation-corrections'
CONFIG='config/semantic-relation-displays.json'
POLICY={'original_program_preserved':True,'geometry_or_poses_modified':False,'physical_accuracy_verified':False,'simulation_ready':False,'human_reviewed':False,'fresh_Muse_review':False,'authorization':'owner_requested_autonomous_source_review','expires_on_approved_revision_change':True}
def require(value,message):
    if not value:raise ValueError(message)
def reference(value):
    require(isinstance(value,dict)and set(value)=={'path','sha256','bytes'}and isinstance(value['path'],str)and re.fullmatch(r'data/[A-Za-z0-9_./-]+',value['path'])and all(p and not p.startswith('.')for p in value['path'].split('/'))and isinstance(value.get('sha256'),str)and re.fullmatch(r'[a-f0-9]{64}',value['sha256'])and type(value['bytes'])is int and 0<value['bytes']<=64*1024*1024,'Ordinary exact semantic artifact required')
    return value
def read_exact(root,ref):
    reference(ref);p=Path(root)/ref['path'];require(p.resolve()==p and not any(q.is_symlink()for q in [p,*p.parents])and p.is_file(),'Semantic artifact escaped or unavailable');raw=p.read_bytes();require(len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256'],'Semantic artifact bytes changed');return raw
def selections(root):
    p=Path(root)/CONFIG
    if not p.exists():return {}
    require(p.resolve()==p and not any(q.is_symlink()for q in [p,*p.parents])and p.is_file()and 0<p.stat().st_size<=32768,'Invalid semantic selection file')
    d=json.loads(p.read_text());require(set(d)=={'version','recordings'}and d['version']==1 and isinstance(d['recordings'],dict)and len(d['recordings'])<=100,'Invalid semantic selections')
    for source,item in d['recordings'].items():require(re.fullmatch('[a-f0-9]{64}',source)and isinstance(item,dict)and set(item)=={'descriptor'},'Explicit semantic source selection required');reference(item['descriptor'])
    return d['recordings']
def revision_binding(document):
    """Validate the pinned header before deciding whether its revision expired."""
    d=document
    require(isinstance(d,dict)and d.get('format')==FORMAT and type(d.get('version'))is int and d['version']==1,'Unknown semantic correction format')
    for k in ['evidence_program','source','scene_ref','review_ref']:reference(d[k])
    binding=d.get('approved_revision_binding')
    require(isinstance(binding,dict)and set(binding)=={'revision_id','fingerprint'}and isinstance(binding['revision_id'],str)and re.fullmatch('revision_[a-f0-9]{32}',binding['revision_id'])and isinstance(binding['fingerprint'],str)and re.fullmatch('[a-f0-9]{64}',binding['fingerprint']),'Exact semantic approved revision binding required')
    require(d.get('policy')==POLICY,'Semantic correction overstates its review or capabilities')
    return binding
def validate(document,record,program,scene_ref,read):
    d=document
    binding=revision_binding(d)
    require(d['evidence_program']==record['interaction']and d['source']['path']==record['source']['path']and d['source']['sha256']==record['source']['sha256']and d['source']['sha256']==program['source']['sha256']and d['scene_ref']==scene_ref,'Semantic correction source/program/scene differs')
    require(binding=={k:record['curation'][k]for k in ['revision_id','fingerprint']},'Semantic correction approved revision changed')
    require(isinstance(d.get('corrections'),list)and 1<=len(d['corrections'])<=8,'Bounded explicit semantic corrections required')
    parts={c['id']for c in record['components']};old={r['id']:r for r in program['relationships']};seen=set()
    for c in d['corrections']:
        require(set(c)=={'relationship_id','original','replacement','inference','limitations'}and c['relationship_id']not in seen,'Invalid semantic correction operation');seen.add(c['relationship_id']);r=c['original'];n=c['replacement'];require(r==old.get(c['relationship_id'])and isinstance(n,dict)and set(n)==set(r),'Original relationship or replacement fields differ')
        changed={k for k in r if r[k]!=n[k]};require(changed and changed<={'object_track_id','rationale','limitations'}and n['id']==r['id']and n['subject_track_id']==r['subject_track_id']and n['predicate']==r['predicate']and n['status']==r['status']and n['evidence_ids']==r['evidence_ids']and n['object_track_id']in parts and n['object_track_id']!=n['subject_track_id'],'Correction may only revise a candidate host and its explanation')
        require(isinstance(n['rationale'],str)and 0<len(n['rationale'])<=2000 and isinstance(n['limitations'],list)and 1<=len(n['limitations'])<=32 and all(isinstance(v,str)and 0<len(v)<=1000 for v in n['limitations']),'Replacement rationale and limitations must be bounded source-schema text')
        require(set(r['limitations'])<=set(n['limitations']),'Original relationship limitations must be preserved')
        require(c['inference']=='source_reviewed_candidate_not_verified_attachment'and isinstance(c['limitations'],list)and 1<=len(c['limitations'])<=8 and all(isinstance(v,str)and 0<len(v)<=500 for v in c['limitations']),'Semantic inference must retain uncertainty')
    review=json.loads(read(d['review_ref']));require(review.get('format')=='interaction-twin-source-relation-review'and review.get('version')==1 and review.get('evidence_program')==d['evidence_program']and review.get('source_sha256')==d['source']['sha256']and review.get('human_reviewed')is False and review.get('model_API_called')is False and review.get('method')=='direct_pixel_inspection_independent_of_original_Muse_proposal','Missing independent source review')
    evidence={e['id']:e for e in program['evidence']};rows=review.get('frames');require(isinstance(rows,list)and 2<=len(rows)<=12 and len({r['evidence_id']for r in rows})==len(rows),'Bounded independent source frames required')
    needed={e for c in d['corrections']for e in c['original']['evidence_ids']};require(needed<={r['evidence_id']for r in rows},'Review must inspect every original relation citation')
    for row in rows:
        original=evidence.get(row['evidence_id']);require(original is not None and row['source_sample_index']==original['source_index']and row['timestamp_s']==original['timestamp_s']and row['frame']['path']==original['frame_path']and row['frame']['sha256']==original['frame_sha256'],'Semantic review frame differs from original evidence');reference(row['frame']);read(row['frame']);require(isinstance(row.get('finding'),str)and 0<len(row['finding'])<=1000,'Explicit source finding required')
    return review
def project(record,document,artifact):
    reference(artifact);result=copy.deepcopy(record);corrections={c['relationship_id']:c for c in document['corrections']}
    require(not result.get('semantic_correction'),'Semantic correction already selected')
    for i,row in enumerate(result['relationships']):
        c=corrections.get(row['id'])
        if not c:continue
        require({k:row[k]for k in c['original']}==c['original'],'Projected relationship differs from pinned original')
        result['relationships'][i]={**copy.deepcopy(c['replacement']),'citations':copy.deepcopy(row['citations']),'semantic_correction':{'artifact':copy.deepcopy(artifact),'original_object_track_id':c['original']['object_track_id'],'inference':c['inference'],'human_reviewed':False,'physical_accuracy_verified':False}}
    require(all(any(r['id']==key for r in result['relationships'])for key in corrections),'Correction relation missing from catalog')
    result['semantic_correction']=copy.deepcopy(artifact)
    return result
def artifacts(document,review):
    return [document['review_ref'],document['evidence_program'],document['source'],document['scene_ref'],*[r['frame']for r in review['frames']]]
