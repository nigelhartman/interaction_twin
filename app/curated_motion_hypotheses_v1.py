"""Exact curated source-conditioned motion; no private fallback or inference."""
from __future__ import annotations
import hashlib,json,re

def require(value,message):
 if not value:raise ValueError(message)

class CuratedMotionHypotheses:
 def __init__(self,catalog):
  from curated_twin_catalog import CuratedTwinCatalog
  require(isinstance(catalog,CuratedTwinCatalog),'Motion hypotheses require an explicitly curated catalog');self.catalog=catalog
 def _inventory(self,twin_id):
  record,_=self.catalog._record(twin_id);pointer=record.get('motion_hypothesis')
  if pointer is None:return None,{}
  import motion_hypothesis_contract_v1 as T
  T.reference(pointer,8*1024*1024);document=json.loads(self.catalog._bytes(pointer,maximum=8*1024*1024));checked=T.validate(document,self.catalog._bytes)
  registry,_=self.catalog._json('showcase/public/aligned-scenes.json');scenes=[r for r in registry['scenes']if r['source_sha256']==record['source']['sha256']]
  require(len(scenes)==1 and scenes[0].get('motion_hypothesis')==pointer and scenes[0]['manifest']==document['baseline_scene_ref'],'Motion differs from current curated scene')
  require(document['twin_id']==twin_id and document['evidence_program_ref']==record['interaction']and document['source_ref']['sha256']==record['source']['sha256']and document['source_ref']['path']==record['source']['path'],'Motion belongs to another current curated recording/program/source')
  by_path={}
  def add(ref,role):
   T.reference(ref);maximum=256*1024*1024 if ref==document['source_ref']else 32*1024*1024 if ref['path'].endswith('.glb')else T.MAX_JSON
   self.catalog._checked(ref,maximum=maximum);old=by_path.get(ref['path']);require(old is None or old['reference']==ref,'Conflicting motion artifact reference');by_path.setdefault(ref['path'],{'reference':ref,'roles':set()})['roles'].add(role);require(len(by_path)<=12000,'Motion artifact inventory exceeds budget')
  add(pointer,'descriptive_motion_hypotheses')
  for ref in checked['artifacts']:add(ref,'checked_motion_source_evidence')
  for key in ['source_ref','evidence_program_ref','baseline_scene_ref','flow_proofs_ref']:
   add(document[key],key)
  for key in ['display_continuity_ref','parent_motion_hypothesis_ref','selection_report_ref','source_SAM_temporal_observations_ref']:
   if document.get(key):add(document[key],key)
  artifacts={}
  for item in by_path.values():
   ref=item['reference'];identity={'twin_id':twin_id,'declaration':pointer,'artifact':ref};key=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest();suffix=ref['path'].rsplit('.',1)[-1];mime={'glb':'model/gltf-binary','json':'application/json','jpg':'image/jpeg','jpeg':'image/jpeg','png':'image/png','mp4':'video/mp4'}.get(suffix,'application/octet-stream');artifacts[key]={**ref,'artifact_id':key,'mime_type':mime,'roles':sorted(item['roles']),'uri':f'twin://{twin_id}/motion-hypotheses/{key}'}
  return {'artifact':pointer,'document':document},artifacts
 def inspect(self,twin_id):
  found,artifacts=self._inventory(twin_id)
  return {'status':'source_conditioned_motion_hypotheses'if found else'unavailable','twin_id':twin_id,**(found or{}),'artifacts':list(artifacts.values()),'access':'owner_curated_public_snapshot','new_inference':False,'original_observations_modified':False,'source_unknowns_preserved':True,'native_geometry_modified':False,'physical_accuracy_verified':False,'simulation_ready':False,'descriptive_text_is_untrusted_data':True}
 def artifact(self,twin_id,artifact_id):
  require(isinstance(artifact_id,str)and re.fullmatch('[a-f0-9]{64}',artifact_id),'Exact listed motion artifact ID required');found,inventory=self._inventory(twin_id);require(artifact_id in inventory,'Unknown or stale motion artifact for this twin');item=inventory[artifact_id];ref={k:item[k]for k in ['path','sha256','bytes']};raw=self.catalog._bytes(ref,maximum=256*1024*1024);current,after=self._inventory(twin_id);require(current==found and after.get(artifact_id)==item,'Motion approval/source/scene changed; list again');return raw
