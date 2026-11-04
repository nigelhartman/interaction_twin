"""Add reviewed source observations to an exact existing curated snapshot.

Dry-run preparation is read-only. No old raster geometry is refitted. Every old
artifact is rehashed, each new observation receives the full source-pixel proof,
and original program/geometry/approval fields are immutable. The caller must
ensure the ordinary publisher is idle before invoking explicit commit().
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import copy,hashlib,json,os,re,tempfile

from curated_twin_catalog import CuratedTwinCatalog
import source_visible_flexure_contract_v1 as F

ROOT=Path(__file__).resolve().parent
FIXED={'index':'showcase/public/library-index.json','aligned':'showcase/public/aligned-scenes.json','assets':'showcase/public/public-assets.json','cases':'showcase/public/cases-index.json'}
def require(value,message):
 if not value:raise ValueError(message)
def encoded(value):return (json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('<','\\u003c')+'\n').encode()
def _path(root,relative):
 require(isinstance(relative,str)and re.fullmatch(r'[A-Za-z0-9_./-]+',relative)and all(p and not p.startswith('.')for p in relative.split('/')),'Ordinary exact publication path required');p=root/relative;require(p.resolve()==p and not any(q.is_symlink()for q in [p,*p.parents])and p.is_file(),'Publication artifact escaped or is unavailable');return p
def _reference(p,root):
 before=_fingerprint(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 require(_fingerprint(p)==before,'Artifact changed while hashing');return {'path':p.relative_to(root).as_posix(),'sha256':h.hexdigest(),'bytes':before[2]}
def _read(root,ref):
 require(isinstance(ref,dict)and set(ref)=={'path','sha256','bytes'}and type(ref['bytes'])is int and ref['bytes']>0 and isinstance(ref['sha256'],str)and re.fullmatch('[a-f0-9]{64}',ref['sha256']),'Exact publication reference required');p=_path(root,ref['path']);before=_fingerprint(p);raw=p.read_bytes();require(_fingerprint(p)==before and len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256'],'Publication artifact bytes changed: '+ref['path']);return raw
def capture_baseline(root=ROOT):
 root=Path(root).resolve();refs={k:_reference(_path(root,p),root)for k,p in FIXED.items()};index=json.loads(_read(root,refs['index']));p=index['catalog'];require(re.fullmatch(r'library-catalog-[a-f0-9]{64}\.json',p['path'])and p['path']=='library-catalog-'+p['sha256']+'.json','Content-addressed curated catalog required');refs['catalog']={'path':'showcase/public/'+p['path'],'sha256':p['sha256'],'bytes':p['bytes']};_assert_snapshot(root,refs);return refs
def _assert_snapshot(root,expected):
 require(isinstance(expected,dict)and set(expected)==set(FIXED)|{'catalog'},'Five-file exact baseline CAS required')
 for k,p in FIXED.items():require(expected[k]['path']==p,'Baseline path differs: '+k)
 raw={k:_read(root,r)for k,r in expected.items()};index=json.loads(raw['index']);pointer=index.get('catalog',{});require(expected['catalog']=={'path':'showcase/public/'+str(pointer.get('path')),'sha256':pointer.get('sha256'),'bytes':pointer.get('bytes')},'Baseline catalog and index differ');require(re.fullmatch(r'showcase/public/library-catalog-[a-f0-9]{64}\.json',expected['catalog']['path'])and expected['catalog']['path']=='showcase/public/library-catalog-'+expected['catalog']['sha256']+'.json','Baseline catalog is not content addressed');return raw
def _fingerprint(p):
 s=p.stat();return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def _verify_fingerprints(root,identities):
 for relative,identity in identities.items():require(_fingerprint(_path(root,relative))==identity,'Verified artifact changed during preparation: '+relative)
@dataclass
class Prepared:
 expected:dict
 outputs:dict
 receipt:dict
 identities:dict

def prepare(*,expected,selections,root=ROOT):
 root=Path(root).resolve();expected=copy.deepcopy(expected);raw=_assert_snapshot(root,expected);catalog=json.loads(raw['catalog']);aligned=json.loads(raw['aligned']);asset_doc=json.loads(raw['assets']);require(encoded(catalog)==raw['catalog']and encoded(aligned)==raw['aligned'],'Canonical existing catalog/scene bytes required for conservative additions');snapshot=CuratedTwinCatalog(root);records,cases=snapshot._snapshot();require(records==catalog['records'],'Baseline is not the exact current approval');require(aligned.get('format')=='sam3d-aligned-scene-index'and aligned.get('version')==1 and isinstance(aligned.get('scenes'),list),'Current aligned scene registry required');old_assets=snapshot._assets();require(isinstance(selections,dict)and 0<len(selections)<=1000,'Explicit bounded source selections required');old=copy.deepcopy(catalog);old_aligned=copy.deepcopy(aligned);identities={};assets=dict(old_assets);proofs=[];additions=[]
 # Verify old approved bytes once, without running geometry or model analysis.
 for relative,r in old_assets.items():
  p=_path(root,relative);before=_fingerprint(p);snapshot._checked(r);require(_fingerprint(p)==before,'Old artifact changed while verifying');identities[relative]=before
 def exact(r):
  r=F.reference(r);p=_path(root,r['path']);before=_fingerprint(p);content=p.read_bytes();require(_fingerprint(p)==before and len(content)==r['bytes']and hashlib.sha256(content).hexdigest()==r['sha256'],'Selected artifact changed: '+r['path']);identities[r['path']]=before;return content
 for source,selection in selections.items():
  require(isinstance(source,str)and re.fullmatch('[a-f0-9]{64}',source)and isinstance(selection,dict)and set(selection)in({'descriptor'},{'descriptor','presentation'}),'Explicit exact source selection required');matches=[r for r in catalog['records']if r['source']['sha256']==source];require(len(matches)==1,'Selection must name exactly one current approved source');record=matches[0];scenes=[s for s in aligned['scenes']if s.get('source_sha256')==source];require(len(scenes)==1,'Selection must name exactly one current connected scene');scene=scenes[0];descriptor=F.reference(selection['descriptor']);document=json.loads(exact(descriptor));validated=F.validate(document,exact);require(document['source_program']==record['interaction']and document['source_scene']==scene['manifest'],'New observations differ from the exact current program/scene');require(old_assets.get(record['interaction']['path'])==record['interaction']and old_assets.get(scene['manifest']['path'])==scene['manifest'],'Current program/scene is not admitted by the baseline')
  for r in document['closed_native_assets']:require(old_assets.get(r['path'])==r,'New observation cannot introduce a different native model')
  from source_visible_flexure_verify_v1 import validate_surface_files
  proof=validate_surface_files(descriptor,root=root);require(isinstance(proof,dict)and proof.get('all_passed')is True and proof.get('descriptor')==descriptor and proof.get('independent_geometry_correctness_established')is False,'Missing or inconsistent fresh source-pixel proof');require(proof.get('supported_sources')==validated['supported_sources']and proof.get('unknown_sources')==validated['unknown_sources'],'Source proof sample ledger differs');proofs.append(proof)
  admitted=[descriptor,*validated['artifacts']]
  if'presentation'in selection:
   pointer=F.reference(selection['presentation']);policy=json.loads(exact(pointer));from observed_surface_presentation_v1 import validate
   validate(policy,document,descriptor);admitted.append(pointer)
  for r in admitted:
   r=F.reference(r);exact(r);require(r['path']not in assets or assets[r['path']]==r,'Existing approved artifact cannot be replaced');assets[r['path']]=r
  fields={'observed_surfaces':descriptor}
  if'presentation'in selection:fields['observed_surface_presentation']=F.reference(selection['presentation'])
  for target in [record,scene]:
   for key,value in fields.items():require(key not in target or target[key]==value,'Incremental addition cannot replace an existing selection');target[key]=copy.deepcopy(value)
  additions.append({'twin_id':record['id'],'source_sha256':source,**fields,'new_artifacts':sum(r['path']not in old_assets for r in admitted)})
 def stripped(doc,key):
  result=copy.deepcopy(doc)
  for r in result[key]:r.pop('observed_surfaces',None);r.pop('observed_surface_presentation',None)
  return result
 require(stripped(catalog,'records')==stripped(old,'records')and stripped(aligned,'scenes')==stripped(old_aligned,'scenes'),'An original model/program/approval field changed');require(len(assets)<=20000,'Existing public asset inventory cap exceeded');asset_doc['assets']=sorted(assets.values(),key=lambda r:r['path']);catalog_raw=encoded(catalog);digest=hashlib.sha256(catalog_raw).hexdigest();name='library-catalog-'+digest+'.json';index=json.loads(raw['index']);index['catalog']={'path':name,'sha256':digest,'bytes':len(catalog_raw)};outputs={'showcase/public/public-assets.json':encoded(asset_doc),'showcase/public/'+name:catalog_raw,'showcase/public/aligned-scenes.json':encoded(aligned),'showcase/public/library-catalog.json':catalog_raw,'showcase/public/library-index.js':b'window.INTERACTION_LIBRARY_INDEX='+encoded(index)+b';\n','showcase/public/library-index.json':encoded(index)};helper=_reference(Path(__file__).resolve(),ROOT)
 receipt={'format':'interaction-twin-incremental-observed-publication','version':1,'status':'dry_run_validated','baseline':expected,'catalog':{'path':'showcase/public/'+name,'sha256':digest,'bytes':len(catalog_raw)},'helper':helper,'additions':additions,'fresh_source_pixel_proofs':proofs,'old_artifacts_checked':len(old_assets),'old_artifact_bytes_checked':sum(r['bytes']for r in old_assets.values()),'new_artifact_paths':sorted(set(assets)-set(old_assets)),'old_cases_bytes_unchanged':True,'old_program_geometry_approval_fields_unchanged':True,'old_native_raster_geometry_recomputed':False,'no_private_discovery':True,'physical_accuracy_verified':False,'new_API_calls':0,'new_GPU_calls':0};_assert_snapshot(root,expected);_verify_fingerprints(root,identities);return Prepared(expected,outputs,receipt,identities)

def commit(*,expected,selections,root=ROOT):
 """Explicit mutation. Caller must keep all other publication processes idle."""
 root=Path(root).resolve();plan=prepare(expected=expected,selections=selections,root=root);public=root/'showcase/public';_assert_snapshot(root,plan.expected);_verify_fingerprints(root,plan.identities)
 with tempfile.TemporaryDirectory(dir=public,prefix='.observed-stage-')as folder:
  staged=[]
  for i,(relative,raw)in enumerate(plan.outputs.items()):
   target=root/relative;require(not target.is_symlink()and target.parent.resolve()==public,'Publication output escaped the public directory');p=Path(folder)/str(i);p.write_bytes(raw);staged.append((p,target))
  # Recheck all five original pointers and every verified artifact immediately
  # before public replacement. No caller-supplied cached proof skips this fence.
  _assert_snapshot(root,plan.expected);_verify_fingerprints(root,plan.identities)
  for p,target in staged:os.replace(p,target)
 result=copy.deepcopy(plan.receipt);result['status']='committed';result['committed_index']={'path':'showcase/public/library-index.json','sha256':hashlib.sha256(plan.outputs['showcase/public/library-index.json']).hexdigest(),'bytes':len(plan.outputs['showcase/public/library-index.json'])};return result
