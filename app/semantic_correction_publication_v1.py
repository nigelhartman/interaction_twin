"""CAS admission of one explicit semantic overlay; never reapprove geometry.

Caller must keep the ordinary publisher idle. No source, pose, native asset,
scene, case, original program or owner approval is replaced by this operation.
"""
from __future__ import annotations
import copy,hashlib,json,os,tempfile
from pathlib import Path
import incremental_observed_publication_v1 as P
import semantic_relation_corrections_v1 as S
from curated_twin_catalog import CuratedTwinCatalog

ROOT=Path(__file__).resolve().parent
def capture(root=ROOT):
    root=Path(root).resolve();baseline=P.capture_baseline(root);path=root/S.CONFIG
    return {'snapshot':baseline,'semantic_selection':P._reference(path,root)if path.exists()else None}
def selection_unchanged(root,expected):
    path=root/S.CONFIG;old=expected['semantic_selection']
    if old is None:S.require(not path.exists(),'Semantic selector changed')
    else:S.require(P._reference(P._path(root,S.CONFIG),root)==old,'Semantic selector changed')
def prepare(*,expected,descriptor,root=ROOT):
    root=Path(root).resolve();raw=P._assert_snapshot(root,expected['snapshot']);selection_unchanged(root,expected)
    catalog=json.loads(raw['catalog']);original=copy.deepcopy(catalog);index=json.loads(raw['index']);registry=json.loads(raw['aligned']);assets=json.loads(raw['assets']);by_path={r['path']:r for r in assets['assets']}
    snapshot=CuratedTwinCatalog(root);records,_=snapshot._snapshot();S.require(records==catalog['records'],'CAS baseline differs from current curated approvals')
    d=json.loads(S.read_exact(root,descriptor));matches=[r for r in catalog['records']if r['source']['sha256']==d['source']['sha256']];S.require(len(matches)==1,'Exactly one approved source required');record=matches[0]
    scenes=[r for r in registry['scenes']if r['source_sha256']==record['source']['sha256']];S.require(len(scenes)==1,'Exactly one current scene required')
    program=json.loads(snapshot._bytes(record['interaction'],maximum=8*1024*1024));review=S.validate(d,record,program,scenes[0]['manifest'],lambda r:S.read_exact(root,r));corrected=S.project(record,d,descriptor)
    from library_query import derive_index
    corrected['search_index']=derive_index(corrected);catalog['records'][catalog['records'].index(record)]=corrected
    restored=copy.deepcopy(corrected);restored.pop('semantic_correction');restored['relationships']=copy.deepcopy(record['relationships']);restored['search_index']=copy.deepcopy(record['search_index']);S.require(restored==record,'Semantic overlay changed other approved fields')
    identities={}
    for relative,r in by_path.items():snapshot._checked(r);identities[relative]=P._fingerprint(P._path(root,relative))
    for r in [descriptor,*S.artifacts(d,review)]:
        S.read_exact(root,r);S.require(r['path']not in by_path or by_path[r['path']]==r,'Semantic admission cannot replace saved bytes');by_path[r['path']]=r;identities[r['path']]=P._fingerprint(P._path(root,r['path']))
    selections=S.selections(root);S.require(d['source']['sha256']not in selections,'Semantic source already selected');selections[d['source']['sha256']]={'descriptor':descriptor}
    assets['assets']=sorted(by_path.values(),key=lambda r:r['path']);content=P.encoded(catalog);digest=hashlib.sha256(content).hexdigest();name='library-catalog-'+digest+'.json';index['catalog']={'path':name,'sha256':digest,'bytes':len(content)}
    outputs={'showcase/public/public-assets.json':P.encoded(assets),'showcase/public/'+name:content,'showcase/public/library-catalog.json':content,'showcase/public/library-index.js':b'window.INTERACTION_LIBRARY_INDEX='+P.encoded(index)+b';\n','showcase/public/library-index.json':P.encoded(index),S.CONFIG:P.encoded({'version':1,'recordings':selections})}
    P._assert_snapshot(root,expected['snapshot']);selection_unchanged(root,expected);P._verify_fingerprints(root,identities)
    return {'expected':copy.deepcopy(expected),'outputs':outputs,'identities':identities,'receipt':{'format':'interaction-twin-semantic-correction-adoption','version':1,'status':'dry_run_validated','baseline':expected,'correction':descriptor,'catalog':{'path':'showcase/public/'+name,'sha256':digest,'bytes':len(content)},'approved_revision_binding':d['approved_revision_binding'],'original_program_assets_poses_scenes_cases_approvals_preserved':True,'human_reviewed':False,'geometry_reapproved':False,'physical_accuracy_verified':False,'new_API_calls':0,'new_GPU_calls':0,'relations_changed':[c['relationship_id']for c in d['corrections']]}}
def commit(*,expected,descriptor,root=ROOT):
    root=Path(root).resolve();plan=prepare(expected=expected,descriptor=descriptor,root=root)
    with tempfile.TemporaryDirectory(dir=root/'showcase/public',prefix='.semantic-stage-')as folder:
        staged=[]
        for i,(relative,raw)in enumerate(plan['outputs'].items()):
            target=root/relative;S.require(not target.is_symlink()and target.parent.resolve()in{root/'showcase/public',root/'config'},'Semantic output escaped fixed directories');p=Path(folder)/str(i);p.write_bytes(raw);staged.append((p,target))
        P._assert_snapshot(root,expected['snapshot']);selection_unchanged(root,expected);P._verify_fingerprints(root,plan['identities'])
        # All immutable data was admitted first; the catalog pointer advances
        # last. Selector persistence is independent of geometry owner approval.
        staged.sort(key=lambda pair:pair[1].name=='library-index.json')
        for p,target in staged:target.parent.mkdir(exist_ok=True);os.replace(p,target)
    return {**plan['receipt'],'status':'committed'}
