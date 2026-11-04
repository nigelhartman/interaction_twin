"""Add source-bound estimated motion without replacing approved observations.

Preparation is read-only. Explicit commit requires the normal publisher idle;
all original pointers/artifacts are rechecked before index-last replacement.
"""
from pathlib import Path
import copy,hashlib,json,os,re,tempfile
from curated_twin_catalog import CuratedTwinCatalog
import motion_hypothesis_contract_v1 as T
from incremental_observed_revision_v2 import (ROOT,FIXED,Prepared,require,encoded,
 _path,_reference,_read,_fingerprint,_verify_fingerprints,_assert_snapshot,capture_baseline)

def historical_catalog(document,root):
 """Check the immutable capture, never the subsequently mutable alias."""
 r=document['catalog_ref'];relative='showcase/public/library-catalog-'+r['sha256']+'.json'
 return _read(root,{'path':relative,'sha256':r['sha256'],'bytes':r['bytes']})

def validate_selection(document,record,scene,read_bytes):
 checked=T.validate(document,read_bytes)
 require(document['twin_id']==record['id']and document['evidence_program_ref']==record['interaction']and
         document['baseline_scene_ref']==scene['manifest']and document['source_ref']['sha256']==record['source']['sha256']and
         document['source_ref']['path']==record['source']['path'],'Motion differs from exact current approved recording/program/scene')
 return checked

def prepare(*,expected,selections,root=ROOT):
 root=Path(root).resolve();expected=copy.deepcopy(expected);raw=_assert_snapshot(root,expected);catalog=json.loads(raw['catalog']);aligned=json.loads(raw['aligned']);asset_doc=json.loads(raw['assets'])
 require(encoded(catalog)==raw['catalog']and encoded(aligned)==raw['aligned'],'Canonical current catalog/scene bytes required')
 snapshot=CuratedTwinCatalog(root);records,_=snapshot._snapshot();require(records==catalog['records'],'Baseline is not the exact current curated approval')
 require(aligned.get('format')=='sam3d-aligned-scene-index'and aligned.get('version')==1 and isinstance(aligned.get('scenes'),list),'Current aligned registry required')
 require(isinstance(selections,dict)and 0<len(selections)<=1000,'Explicit bounded motion selections required')
 old=copy.deepcopy(catalog);old_aligned=copy.deepcopy(aligned);old_assets=snapshot._assets();assets=dict(old_assets);identities={};additions=[];proofs=[]
 for relative,r in old_assets.items():
  p=_path(root,relative);before=_fingerprint(p);snapshot._checked(r);require(_fingerprint(p)==before,'Old approved artifact changed');identities[relative]=before
 def exact(r,maximum=T.MAX_JSON):
  T.reference(r,maximum);p=_path(root,r['path']);before=_fingerprint(p);body=p.read_bytes();require(_fingerprint(p)==before and len(body)==r['bytes']and hashlib.sha256(body).hexdigest()==r['sha256'],'Motion artifact changed: '+r['path']);identities[r['path']]=before;return body
 for source,selection in selections.items():
  require(isinstance(source,str)and re.fullmatch('[a-f0-9]{64}',source)and isinstance(selection,dict)and set(selection)=={'descriptor'},'Explicit exact motion selection required')
  matches=[r for r in catalog['records']if r['source']['sha256']==source];rows=[s for s in aligned['scenes']if s.get('source_sha256')==source];require(len(matches)==len(rows)==1,'Motion must name one current approved recording and connected scene');record=matches[0];scene=rows[0]
  descriptor=T.reference(selection['descriptor'],8*1024*1024);document=json.loads(exact(descriptor,maximum=8*1024*1024));checked=validate_selection(document,record,scene,exact)
  for ref in [record['interaction'],scene['manifest'],document['source_ref'],*[c['asset']for c in checked['source_scene']['components']]]:
   require(old_assets.get(ref['path'])==ref,'Motion cannot introduce a different original source/program/scene/native asset')
  historical_path='showcase/public/library-catalog-'+document['catalog_ref']['sha256']+'.json';historical_file=_path(root,historical_path);historical_before=_fingerprint(historical_file);historical=historical_catalog(document,root);require(_fingerprint(historical_file)==historical_before,'Historical catalog changed during validation');identities[historical_path]=historical_before
  admitted={descriptor['path']:descriptor}
  for ref in checked['artifacts']:
   T.reference(ref);exact(ref,maximum=256*1024*1024);require(ref['path']not in admitted or admitted[ref['path']]==ref,'Conflicting motion dependency');admitted[ref['path']]=ref
  for ref in admitted.values():require(ref['path']not in assets or assets[ref['path']]==ref,'Existing approved artifact cannot be replaced');assets[ref['path']]=ref
  for target in [record,scene]:require('motion_hypothesis'not in target or target['motion_hypothesis']==descriptor,'Cannot silently replace an existing motion selection');target['motion_hypothesis']=copy.deepcopy(descriptor)
  proofs.append({'descriptor':descriptor,'typed_artifact_count':len(admitted),'original_unknowns_preserved':checked['original_unknowns_preserved'],'physical_accuracy_verified':checked['physical_accuracy_verified'],'historical_catalog_fingerprint_checked':{'path':historical_path,'sha256':hashlib.sha256(historical).hexdigest(),'bytes':len(historical)},'source_conditioned_predictions':sum(p['status']=='estimated'for g in document['groups']for e in g['episodes']for p in e['predictions']),'fresh_numeric_source_contract_checked':True,'independent_physical_tracking_truth_established':False})
  additions.append({'twin_id':record['id'],'source_sha256':source,'motion_hypothesis':descriptor})
 def stripped(doc,key):
  value=copy.deepcopy(doc)
  for row in value[key]:row.pop('motion_hypothesis',None)
  return value
 require(stripped(catalog,'records')==stripped(old,'records')and stripped(aligned,'scenes')==stripped(old_aligned,'scenes'),'Original observations/models/programs/approvals or another extension changed')
 require(len(assets)<=20000,'Existing public artifact cap exceeded');asset_doc['assets']=sorted(assets.values(),key=lambda r:r['path']);catalog_raw=encoded(catalog);digest=hashlib.sha256(catalog_raw).hexdigest();name='library-catalog-'+digest+'.json';index=json.loads(raw['index']);index['catalog']={'path':name,'sha256':digest,'bytes':len(catalog_raw)}
 outputs={'showcase/public/public-assets.json':encoded(asset_doc),'showcase/public/'+name:catalog_raw,'showcase/public/aligned-scenes.json':encoded(aligned),'showcase/public/library-catalog.json':catalog_raw,'showcase/public/library-index.js':b'window.INTERACTION_LIBRARY_INDEX='+encoded(index)+b';\n','showcase/public/library-index.json':encoded(index)}
 receipt={'format':'interaction-twin-reviewed-motion-display-addition','version':1,'status':'dry_run_validated','baseline':expected,'catalog':{'path':'showcase/public/'+name,'sha256':digest,'bytes':len(catalog_raw)},'helper':_reference(Path(__file__).resolve(),ROOT),'CAS_utility':_reference(ROOT/'incremental_observed_revision_v2.py',ROOT),'motion_contract':_reference(ROOT/'motion_hypothesis_contract_v1.py',ROOT),'additions':additions,'fresh_motion_contract_proofs':proofs,'old_artifacts_checked':len(old_assets),'old_artifact_bytes_checked':sum(r['bytes']for r in old_assets.values()),'new_artifact_paths':sorted(set(assets)-set(old_assets)),'original_source_model_program_approval_and_extension_fields_unchanged':True,'original_unknowns_preserved':True,'old_cases_bytes_unchanged':True,'old_native_raster_geometry_recomputed':False,'no_private_discovery':True,'physical_accuracy_verified':False,'new_API_calls':0,'new_GPU_calls':0}
 _assert_snapshot(root,expected);_verify_fingerprints(root,identities);return Prepared(expected,outputs,receipt,identities)

def commit(*,expected,selections,root=ROOT):
 """Explicit mutation only; caller keeps all other publication writers idle."""
 root=Path(root).resolve();plan=prepare(expected=expected,selections=selections,root=root);public=root/'showcase/public';_assert_snapshot(root,plan.expected);_verify_fingerprints(root,plan.identities)
 with tempfile.TemporaryDirectory(dir=public,prefix='.motion-stage-')as folder:
  staged=[]
  for i,(relative,body)in enumerate(plan.outputs.items()):
   target=root/relative;require(not target.is_symlink()and target.parent.resolve()==public,'Motion publication output escaped public directory');p=Path(folder)/str(i);p.write_bytes(body);staged.append((p,target))
  _assert_snapshot(root,plan.expected);_verify_fingerprints(root,plan.identities)
  for p,target in staged:os.replace(p,target)
 result=copy.deepcopy(plan.receipt);result['status']='committed';result['committed_index']={'path':'showcase/public/library-index.json','sha256':hashlib.sha256(plan.outputs['showcase/public/library-index.json']).hexdigest(),'bytes':len(plan.outputs['showcase/public/library-index.json'])};return result
