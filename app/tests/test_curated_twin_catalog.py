"""MCP privacy uses exact curated revisions, never newer same-source files."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from curated_twin_catalog import CuratedTwinCatalog
from twin_catalog import TwinCatalog
from tests import test_twin_catalog as owner_fixtures


class CuratedTwinCatalogTests(unittest.TestCase):
    def setUp(self):
        self.fixture=owner_fixtures.TwinCatalogTests(methodName='runTest');self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root;self.id=self.fixture.id;self.doc=self.fixture.doc
        self.row=copy.deepcopy(self.fixture.row);self.row['curated']=True
        self.record={'id':self.id,'title':'Curated test component','summary':self.doc['summary'],
            'interaction':self.ref(self.row['interaction_path']),'source':copy.deepcopy(self.doc['source']),
            'duration_s':self.doc['source']['duration_s'],'counts':{'assets':1,'components':1,'steps':1},
            'components':[{**t,'asset':next((a for a in self.doc['assets']if a['track_id']==t['id']),None)}for t in self.doc['tracklets']],
            'steps':copy.deepcopy(self.doc['steps']),'relationships':copy.deepcopy(self.doc['relationships']),
            'task_classes':['hinged_assemblies'],'sam3d_reviews':[],
            'curation':{'status':'owner_approved','human_approval_recorded':True,'human_guidance_used':True}}
        self.public_assets={}
        for relative in [self.row['interaction_path'],self.doc['source']['path'],*[a['path']for a in self.doc['assets']],
                         *[e['frame_path']for e in self.doc['evidence']],
                         *[o['mask_path']for e in self.doc['evidence']for o in e['observations']]]:
            self.admit(relative)
        self.publish();self.catalog=CuratedTwinCatalog(self.root)

    def ref(self,relative):
        p=self.root/relative
        return {'path':relative,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
    def file(self,relative,value):
        self.fixture.file(relative,value);return self.ref(relative)
    def admit(self,relative):self.public_assets[relative]=self.ref(relative)
    def publish(self):
        catalog={'format':'interaction-twin-object-program-library','version':1,'records':[self.record]}
        raw=json.dumps(catalog).encode();sha=hashlib.sha256(raw).hexdigest();name='library-catalog-'+sha+'.json'
        self.file('showcase/public/'+name,raw)
        self.file('showcase/public/library-index.json',{'format':'interaction-twin-library-index','version':1,'catalog':{'path':name,'sha256':sha,'bytes':len(raw)}})
        self.file('showcase/public/cases-index.json',{'curated':True,'cases':[self.row]})
        self.file('showcase/public/public-assets.json',{'format':'interaction-twin-public-assets','version':1,'assets':list(self.public_assets.values())})
        if not (self.root/'showcase/public/aligned-scenes.json').exists():
            self.file('showcase/public/aligned-scenes.json',{'format':'sam3d-aligned-scene-index','version':1,'scenes':[]})

    def test_search_facets_and_all_known_id_routes_exclude_pending_same_source_programs(self):
        private=copy.deepcopy(self.fixture.row);private['id']='newer-pending--program';private['title']='PRIVATE_NEWER_SAME_SOURCE'
        self.fixture.file('workspace/cases.json',{'cases':[self.fixture.row,private]})
        self.assertTrue(any(r['id']==private['id']for r in TwinCatalog(self.root).entries()))
        with patch('sam3d_review_catalog.discover_reviews',side_effect=AssertionError('No working-candidate discovery')):
            found=self.catalog.search('',25,automatic_only=False,group_by='variant')
            self.assertEqual([r['id']for r in found['twins']],[self.id])
            self.assertEqual(found['access'],'owner_curated_public_snapshot')
            self.assertEqual(self.catalog.library_facets()['facets'],found['facets'])
            self.assertTrue(found['twins'][0]['curation']['human_guidance_used'])
        for name,args in [('load',()),('describe',()),('procedure',()),('state',(0.,)),('asset',('part',)),
                          ('evidence',('source_000001',)),('generated_candidate',('part',)),('kinematics',()),
                          ('reconstructions',()),('motion_experiments',()),('reconstruction',('v1',)),('motion_experiment',('x',))]:
            with self.subTest(method=name),self.assertRaisesRegex(ValueError,'Unknown curated'):
                getattr(self.catalog,name)(private['id'],*args)

    def test_only_approved_exact_assets_and_frames_are_resources(self):
        _,doc,_=self.catalog.load(self.id)
        self.assertEqual(self.catalog.asset(self.id,'part',binary=True),b'glTFcontract-test')
        self.assertEqual(self.catalog.evidence(self.id,'source_000001',binary=True),b'\xff\xd8contract-jpeg\xff\xd9')
        private=self.file('data/pending/asset.glb',b'glTFnot-approved')
        for path in [private['path'],self.doc['inputs'][0]['path'],'.private/agent-memory/feedback.json','.secrets/meta.env']:
            with self.subTest(path=path),self.assertRaisesRegex(ValueError,'not published'):self.catalog.data_path(path)
        asset=self.root/doc['assets'][0]['path'];asset.write_bytes(b'glTFtampered-original')
        with self.assertRaises(ValueError):self.catalog.asset(self.id,'part',binary=True)
        self.assertNotIn('local_file_uri',self.catalog.asset_descriptor(self.id,doc['assets'][0]))

    def test_generated_candidate_is_exact_frozen_review_not_newer_private_candidate(self):
        approved=self.fixture.candidate_record();self.record['sam3d_reviews']=[approved]
        for field in ['native_glb','source_image','source_mask','review_manifest']:self.admit(approved[field]['path'])
        private=copy.deepcopy(approved);private['native_glb']=self.file('data/private/newer.glb',b'glTFPRIVATE-new-candidate')
        self.publish()
        with patch('sam3d_review_catalog.reviews_for_record',return_value=[private]) as scan:
            desc=self.catalog.generated_candidate(self.id,'part')
            self.assertEqual(desc['artifacts']['asset']['sha256'],approved['native_glb']['sha256'])
            self.assertEqual(self.catalog.generated_candidate_artifact(self.id,'part','asset'),b'glTFnew-candidate')
            scan.assert_not_called()
        self.assertNotIn('PRIVATE',json.dumps(desc));self.assertFalse(desc['source_video_alignment_verified'])

    def test_kinematics_uses_exact_curated_companion_never_private_discovery(self):
        keys={k:self.file('data/curated/'+k+'.json',{'type':k})for k in ['source_scene','constrained_scene','mechanics','proposal']}
        obj={'format':'interaction-twin-object-program','version':'0.2','evidence_program':self.record['interaction'],
             **keys,'joints':[],'removable_relationships':[],'unresolved_connections':[]}
        companion=self.file('data/curated/object-program.json',obj)
        for ref in [companion,*keys.values()]:self.admit(ref['path'])
        self.publish()
        self.file('showcase/public/aligned-scenes.json',{'format':'sam3d-aligned-scene-index','version':1,'scenes':[
            {'source_sha256':self.doc['source']['sha256'],'manifest':keys['constrained_scene'],'object_program':companion}]})
        with patch('object_program.discover',side_effect=AssertionError('Never scan newer private companions')),patch('object_program.validate',return_value=True):
            result=self.catalog.kinematics(self.id)
            self.assertEqual(result['artifact']['sha256'],companion['sha256'])
            self.assertEqual(json.loads(self.catalog.kinematics_artifact(self.id)),obj)
            self.assertEqual(json.loads(self.catalog.kinematics_artifact(self.id,'mechanics')),{'type':'mechanics'})
            changed=copy.deepcopy(obj);changed['evidence_program']['sha256']='f'*64
            bad=self.file('data/curated/wrong-companion.json',changed);self.admit(bad['path']);self.publish()
            self.file('showcase/public/aligned-scenes.json',{'format':'sam3d-aligned-scene-index','version':1,'scenes':[
                {'source_sha256':self.doc['source']['sha256'],'manifest':keys['constrained_scene'],'object_program':bad}]})
            with self.assertRaisesRegex(ValueError,'different approved'):self.catalog.kinematics(self.id)

    def test_private_analyses_and_diagnostic_versions_are_not_exposed(self):
        with patch('task_intelligence.cached_analyses',side_effect=AssertionError('No private analysis read')):
            described=self.catalog.describe(self.id)
            self.assertEqual(described['task_brief']['status'],'not_available')
            self.assertTrue(described['curation']['human_guidance_used'])
        self.assertEqual(self.catalog.reconstructions(self.id)['versions'],[])
        self.assertEqual(self.catalog.motion_experiments(self.id)['experiments'],[])
        for name in ['reconstruction','reconstruction_report','reconstruction_artifact','motion_experiment','motion_artifact']:
            with self.subTest(method=name),self.assertRaisesRegex(ValueError,'Private diagnostic'):
                getattr(self.catalog,name)(self.id,'private-version')

    def test_missing_or_mismatched_snapshot_never_falls_back_to_owner_catalog(self):
        (self.root/'showcase/public/library-index.json').unlink()
        with self.assertRaisesRegex(ValueError,'no private catalog fallback'):self.catalog.search()
        self.publish();wrong=copy.deepcopy(self.row);wrong['original_sha256']='f'*64
        self.file('showcase/public/cases-index.json',{'curated':True,'cases':[wrong]})
        with self.assertRaisesRegex(ValueError,'program binding'):self.catalog.describe(self.id)


if __name__=='__main__':unittest.main()
