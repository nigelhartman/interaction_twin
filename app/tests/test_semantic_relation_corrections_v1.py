import copy,hashlib,json,shutil,tempfile,unittest
from pathlib import Path
import semantic_relation_corrections_v1 as S
import semantic_correction_publication_v1 as P
from curated_twin_catalog import CuratedTwinCatalog
from curated_derived_artifacts_v1 import CuratedDerivedArtifacts
ROOT=Path(__file__).resolve().parents[1]
if not (ROOT/'data/semantic-relation-corrections/laptop-20261104-v1/descriptor.json').is_file():
    raise unittest.SkipTest('Saved source-review assets are distributed separately from the source-only checkout')
def read(path):return json.loads(Path(path).read_text())
def ref(path,root=ROOT):
    b=Path(path).read_bytes();return {'path':Path(path).relative_to(root).as_posix(),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
ARTIFACT=ref(ROOT/'data/semantic-relation-corrections/laptop-20261104-v1/descriptor.json')
DOC=read(ROOT/ARTIFACT['path']);CATALOG=read(ROOT/'showcase/public/library-catalog.json');RECORD=copy.deepcopy(next(r for r in CATALOG['records']if r['source']['sha256']==DOC['source']['sha256']));PROGRAM=read(ROOT/RECORD['interaction']['path']);REVIEW=read(ROOT/DOC['review_ref']['path']);SCENE=next(r for r in read(ROOT/'showcase/public/aligned-scenes.json')['scenes']if r['source_sha256']==RECORD['source']['sha256'])
# Build the pre-overlay fixture from the exact original program so this suite
# exercises initial adoption even after the local real catalog has adopted it.
RECORD.pop('semantic_correction',None)
original_relationships={r['id']:r for r in PROGRAM['relationships']}
RECORD['relationships']=[{**copy.deepcopy(original_relationships[r['id']]),'citations':copy.deepcopy(r['citations'])}for r in RECORD['relationships']]
from library_query import derive_index
RECORD['search_index']=derive_index(RECORD)

class SemanticCorrectionTests(unittest.TestCase):
    def validate(self,d=None,r=None,p=None):return S.validate(d or DOC,r or RECORD,p or PROGRAM,SCENE['manifest'],lambda r:S.read_exact(ROOT,r))
    def test_exact_original_and_pixels_authorize_only_inferred_base_host(self):
        self.validate();original=copy.deepcopy(RECORD);corrected=S.project(RECORD,DOC,ARTIFACT);before=next(r for r in PROGRAM['relationships']if r['id']=='r_cable_to_laptop');after=next(r for r in corrected['relationships']if r['id']==before['id']);self.assertEqual(before['object_track_id'],'laptop_lid');self.assertEqual(after['object_track_id'],'laptop_base');self.assertEqual(after['predicate'],before['predicate']);self.assertEqual(after['status'],'inferred');self.assertEqual(after['evidence_ids'],before['evidence_ids']);self.assertFalse(after['semantic_correction']['human_reviewed']);self.assertIn('successful reconnection remain unverified',after['rationale']);self.assertEqual(RECORD,original);self.assertEqual([r['source_sample_index']for r in REVIEW['frames']],[55,58,65,66])
    def test_revision_source_scene_original_relation_and_claim_tampering_reject(self):
        for change in [lambda d:d['approved_revision_binding'].update(revision_id='foreign'),lambda d:d['scene_ref'].update(bytes=1),lambda d:d['source'].update(sha256='0'*64),lambda d:d['corrections'][0]['original'].update(object_track_id='laptop_base'),lambda d:d['corrections'][0]['replacement'].update(predicate='fixed_to'),lambda d:d['corrections'][0]['replacement'].update(object_track_id='invented_part'),lambda d:d['policy'].update(human_reviewed=True),lambda d:d['policy'].update(physical_accuracy_verified=True)]:
            d=copy.deepcopy(DOC);change(d)
            with self.assertRaises(ValueError):self.validate(d)
    def test_new_revision_and_changed_original_program_expire_correction(self):
        r=copy.deepcopy(RECORD);r['curation']['fingerprint']='0'*64
        with self.assertRaisesRegex(ValueError,'revision changed'):self.validate(r=r)
        p=copy.deepcopy(PROGRAM);p['relationships'][1]['rationale']='changed'
        with self.assertRaisesRegex(ValueError,'Original relationship'):self.validate(p=p)
    def test_replacement_text_must_remain_bounded_source_schema_fields(self):
        for key,bad in [('rationale',{}),('rationale',''),('rationale','x'*2001),('limitations','uncertain'),('limitations',[]),('limitations',[{}]),('limitations',['']),('limitations',['x'*1001]),('limitations',['x']*33)]:
            with self.subTest(key=key,bad=bad):
                d=copy.deepcopy(DOC);d['corrections'][0]['replacement'][key]=bad
                with self.assertRaisesRegex(ValueError,'bounded source-schema text'):self.validate(d)
    def test_host_correction_cannot_erase_source_uncertainty_or_malformed_revision_binding(self):
        d=copy.deepcopy(DOC);d['corrections'][0]['replacement']['limitations']=['New note without the original occlusion caveats']
        with self.assertRaisesRegex(ValueError,'limitations must be preserved'):self.validate(d)
        for value in ['revision_new','revision_'+'A'*32,'other_'+'a'*32,'revision_'+'a'*33]:
            d=copy.deepcopy(DOC);d['approved_revision_binding']['revision_id']=value
            with self.assertRaisesRegex(ValueError,'Exact semantic approved revision binding'):self.validate(d)
    def fixture(self):
        temp=tempfile.TemporaryDirectory();root=Path(temp.name).resolve();public=root/'showcase/public';public.mkdir(parents=True);(root/'config').mkdir()
        pointers=[ARTIFACT,*S.artifacts(DOC,REVIEW)]
        for pointer in pointers:
            dest=root/pointer['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/pointer['path'],dest)
        catalog=copy.deepcopy(CATALOG);catalog['records']=[copy.deepcopy(RECORD)];catalog['summary']={'recordings':1,'native_asset_drafts':RECORD['counts']['assets'],'components':len(RECORD['components']),'steps':len(RECORD['steps']),'eligible_saved_versions':1}
        raw=P.P.encoded(catalog);digest=hashlib.sha256(raw).hexdigest();name='library-catalog-'+digest+'.json';(public/name).write_bytes(raw);(public/'library-catalog.json').write_bytes(raw)
        index={'format':'interaction-twin-library-index','version':1,'catalog':{'path':name,'sha256':digest,'bytes':len(raw)},'summary':catalog['summary']};(public/'library-index.json').write_bytes(P.P.encoded(index))
        row=next(c for c in read(ROOT/'showcase/public/cases-index.json')['cases']if c['id']==RECORD['id']);(public/'cases-index.json').write_bytes(P.P.encoded({'curated':True,'cases':[row]}));(public/'aligned-scenes.json').write_bytes(P.P.encoded({'format':'sam3d-aligned-scene-index','version':1,'scenes':[SCENE]}));assets={r['path']:r for r in pointers if r not in [ARTIFACT,DOC['review_ref']]};(public/'public-assets.json').write_bytes(P.P.encoded({'format':'interaction-twin-public-assets','version':1,'assets':list(assets.values())}));return temp,root
    def test_CAS_commit_changes_only_effective_relation_projection_and_preserves_original_bytes(self):
        temp,root=self.fixture()
        with temp:
            before={r['path']:(root/r['path']).read_bytes()for r in S.artifacts(DOC,REVIEW)};expected=P.capture(root);plan=P.prepare(expected=expected,descriptor=ARTIFACT,root=root);self.assertEqual(plan['receipt']['status'],'dry_run_validated');self.assertNotIn('semantic_correction',read(root/'showcase/public/library-catalog.json')['records'][0]);receipt=P.commit(expected=expected,descriptor=ARTIFACT,root=root);self.assertEqual(receipt['status'],'committed');current=read(root/'showcase/public/library-catalog.json')['records'][0];self.assertEqual(current['semantic_correction'],ARTIFACT);self.assertEqual(current['curation'],RECORD['curation']);self.assertEqual(current['interaction'],RECORD['interaction']);self.assertEqual(current['components'],RECORD['components']);self.assertEqual(current['steps'],RECORD['steps']);self.assertEqual(read(root/'showcase/public/aligned-scenes.json')['scenes'],[SCENE]);self.assertEqual(S.selections(root)[DOC['source']['sha256']]['descriptor'],ARTIFACT)
            for path,raw in before.items():self.assertEqual((root/path).read_bytes(),raw)
            with self.assertRaises(ValueError):P.commit(expected=expected,descriptor=ARTIFACT,root=root)
    def test_stale_catalog_and_selector_CAS_reject_before_mutation(self):
        for kind in ['snapshot','selector']:
            temp,root=self.fixture()
            with temp:
                expected=P.capture(root);before=(root/'showcase/public/library-catalog.json').read_bytes()
                if kind=='snapshot':(root/'showcase/public/cases-index.json').write_bytes(b'{}\n')
                else:(root/S.CONFIG).write_text('{}')
                with self.assertRaises(ValueError):P.commit(expected=expected,descriptor=ARTIFACT,root=root)
                self.assertEqual((root/'showcase/public/library-catalog.json').read_bytes(),before)

class SemanticMCPTests(unittest.TestCase):
    def catalog(self):
        corrected=S.project(RECORD,DOC,ARTIFACT);base=CuratedTwinCatalog(ROOT);refs={r['path']:r for r in [ARTIFACT,*S.artifacts(DOC,REVIEW)]};row=next(r for r in base.entries()if r['id']==RECORD['id'])
        class Snapshot(CuratedTwinCatalog):
            def _snapshot(self):return [corrected],[row]
            def _assets(self):return {**base._assets(),**refs}
        return Snapshot(ROOT),corrected
    def test_description_handoff_and_listed_correction_resource_agree(self):
        c,record=self.catalog();description=c.describe(RECORD['id']);handoff=c.simulation_manifest(RECORD['id']);self.assertEqual(description['relationships'],record['relationships']);self.assertEqual(handoff['candidate_relationships'],record['relationships']);self.assertEqual(next(r for r in description['original_relationships']if r['id']=='r_cable_to_laptop')['object_track_id'],'laptop_lid');self.assertEqual(description['semantic_correction']['artifact'],ARTIFACT);inventory=CuratedDerivedArtifacts(c);listed=inventory.list_artifacts(RECORD['id']);item=next(r for r in listed['artifacts']if 'source_reviewed_semantic_correction'in r['roles']);self.assertEqual(inventory.artifact(RECORD['id'],item['artifact_id']),(ROOT/ARTIFACT['path']).read_bytes())
        with self.assertRaises(ValueError):inventory.artifact(RECORD['id'],'0'*64)
    def test_effective_catalog_text_tampering_and_revision_change_are_rejected(self):
        c,record=self.catalog();record['relationships'][1]['object_track_id']='laptop_lid'
        with self.assertRaisesRegex(ValueError,'projection changed'):c.semantic_correction(RECORD['id'])
        c,record=self.catalog();record['curation']['revision_id']='new_revision'
        with self.assertRaisesRegex(ValueError,'revision changed'):c.semantic_correction(RECORD['id'])
    def test_nested_handoff_and_kinematics_keep_original_artifacts_with_explicit_effective_host(self):
        c,record=self.catalog();description=c.describe(RECORD['id']);handoff=c.simulation_manifest(RECORD['id']);kinematics=c.kinematics(RECORD['id']);self.assertEqual(handoff['preparation_draft']['candidate_relationships'],record['relationships']);self.assertEqual(handoff['preparation_draft']['semantic_correction'],ARTIFACT);original=read(ROOT/SCENE['object_program']['path']);self.assertEqual(kinematics['object_program'],original);self.assertEqual(kinematics['effective_candidate_relationships'],record['relationships']);self.assertEqual(kinematics['semantic_correction']['artifact'],ARTIFACT)
        edges=description['kinematics']['removable_relationships'];self.assertTrue(all(e.get('parent_id')!='laptop_lid'for e in edges if e.get('child_id')=='power_cable'));self.assertEqual(description['kinematics']['semantic_correction']['artifact'],ARTIFACT)
    def test_stale_semantic_ids_and_concurrent_selection_switch_cannot_authorize_resource_bytes(self):
        c,record=self.catalog();inventory=CuratedDerivedArtifacts(c);item=next(r for r in inventory.list_artifacts(RECORD['id'])['artifacts']if 'source_reviewed_semantic_correction'in r['roles']);record.pop('semantic_correction')
        with self.assertRaisesRegex(ValueError,'Unknown or stale'):inventory.artifact(RECORD['id'],item['artifact_id'])
        c,record=self.catalog();inventory=CuratedDerivedArtifacts(c);item=next(r for r in inventory.list_artifacts(RECORD['id'])['artifacts']if 'source_reviewed_semantic_correction'in r['roles']);original=c._bytes;reads=0
        def switched(reference,*,maximum):
            nonlocal reads
            raw=original(reference,maximum=maximum)
            if reference==ARTIFACT:
                reads+=1
                if reads==2:record.pop('semantic_correction')
            return raw
        c._bytes=switched
        with self.assertRaisesRegex(ValueError,'semantic correction changed during resource read'):inventory.artifact(RECORD['id'],item['artifact_id'])
    def test_old_correction_resource_expires_after_a_new_approved_revision(self):
        c,record=self.catalog();inventory=CuratedDerivedArtifacts(c);item=next(r for r in inventory.list_artifacts(RECORD['id'])['artifacts']if 'source_reviewed_semantic_correction'in r['roles']);record['curation']['revision_id']='revision_new_owner_approval';record['curation']['fingerprint']='1'*64
        with self.assertRaisesRegex(ValueError,'revision changed'):inventory.artifact(RECORD['id'],item['artifact_id'])

if __name__=='__main__':unittest.main()
