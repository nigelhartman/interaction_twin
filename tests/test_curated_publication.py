"""Publication isolation: explicit revisions, frozen seeds and exact artifacts."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import curated_publication as C

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('curation_policy_server',ROOT/'scripts/serve_workspace.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)


class CuratedPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.video=self.file('data/approved/video.mp4',b'approved source video fixture')
        self.asset=self.file('data/approved/part.glb',b'glTFapproved-fixture')
        self.pending_asset=self.file('data/pending/part.glb',b'glTFUNAPPROVED-newer-fixture')
        self.private=self.file('data/private/model-prompt.txt',b'PRIVATE_MODEL_PROMPT_NOT_A_PUBLIC_ASSET')
        source={**self.video,'original_sha256':self.video['sha256'],'duration_s':1.,'time_domain':'source_video_presentation_seconds'}
        self.program={'source':source,'evidence':[],'assets':[self.asset],'tracklets':[{'id':'part'}],
            'inputs':[self.private],'relationships':[],'steps':[]}
        self.interaction=self.file('data/approved/program/interaction.json',self.program)
        self.metadata=self.file('data/approved/metadata.json',{'source_anchor':'approved frame'})
        receipt=self.file('data/approved/import-receipt.json',{'import_files':[{'path':'metadata.json',
            'sha256':self.metadata['sha256'],'bytes':self.metadata['bytes']}]})
        review_manifest=self.file('data/approved/review/review-manifest.json',{'import_receipt':receipt})
        self.record={'id':'approved--program','title':'Approved object','source':source,'interaction':self.interaction,
            'components':[{'id':'part','asset':{'track_id':'part',**self.asset,'metadata':self.metadata}}],'steps':[],
            'sam3d_reviews':[{'component_id':'part','native_glb':self.asset,'review_manifest':review_manifest}],
            'other_saved_versions':[],'initial_reconstruction_inputs':[],'cached_insights':[]}
        from library_query import derive_index
        self.record['search_index']=derive_index(self.record)
        self.case={'id':self.record['id'],'interaction_path':self.interaction['path'],
                   'original_sha256':self.interaction['sha256'],'interaction':self.program,
                   'scene_ref':None,'exports':{'private_run':'data/private/worker-log.txt'}}
        self.file('workspace/cases-index.json',{'cases':[self.case]})

    def file(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value if isinstance(value,bytes) else json.dumps(value).encode())
        return C.ref(path,self.root)

    def seed(self):
        self.file('.private/admin-workflow/legacy-showcase.json',{'records':[{'record':self.record,'case':self.case,'aligned':None}],
            'new_human_approval_fabricated':False})

    def native_review(self, folder, component_id, raw):
        asset=self.file(f'data/{folder}/part.glb',raw)
        metadata=self.file(f'data/{folder}/metadata.json',{'source_anchor':folder})
        receipt=self.file(f'data/{folder}/import-receipt.json',{'import_files':[{'path':'metadata.json',
            'sha256':metadata['sha256'],'bytes':metadata['bytes']}]})
        # Current native review manifests pin receipt path/hash, without size.
        manifest=self.file(f'data/{folder}/review/review-manifest.json',{
            'import_receipt':{key:receipt[key] for key in ['path','sha256']}})
        binding={'component_id':component_id,'native_glb':asset,'native_metadata':metadata}
        review={'component_id':component_id,'native_glb':asset,'review_manifest':manifest}
        return binding,review

    def approved_with_companion(self, dynamic, bindings):
        companion=self.file('data/approved/object-program.json',{'evidence_program':self.interaction,
            'component_assets':bindings,'constrained_scene':{'fixture':'scene'}})
        revision=self.file('.private/admin-workflow/revisions/approved.json',{'manifest':{'result':{
            'interaction_ref':self.interaction,'object_program_ref':companion}}})
        approval={'manifest_ref':revision,'revision_id':'revision-fixture','fingerprint':'approved-fingerprint'}
        # The scene/schema verifier is independently tested. Keep actual ref,
        # native import and source-anchor verification active in this test.
        with patch('admin_workflow.AdminWorkflow.public_index',return_value={'records':[approval]}),\
             patch('twin_catalog.TwinCatalog.load',return_value=({},self.program,self.interaction['sha256'])),\
             patch('scripts.build_robotics_library.build_record',return_value=dynamic),\
             patch('object_program.validate',return_value=None):
            return C._approved_records(self.root)[0]['record']

    def test_articulated_companion_keeps_other_approved_source_components_inspectable(self):
        cable,cable_review=self.native_review('approved-cable','cable',b'glTFapproved cable')
        dynamic=copy.deepcopy(self.record)
        dynamic['components'].append({'id':'cable','asset':{'track_id':'cable',**cable['native_glb'],
            'metadata':cable['native_metadata']}})
        dynamic['sam3d_reviews'].append(cable_review)
        # A same-byte review from another source anchor must still be rejected.
        _,foreign_review=self.native_review('foreign-cable-anchor','cable',b'glTFapproved cable')
        dynamic['sam3d_reviews'].append(foreign_review)
        binding={'component_id':'part','native_glb':self.asset,'native_metadata':self.metadata}
        record=self.approved_with_companion(dynamic,[binding])
        self.assertEqual(record['sam3d_reviews'],self.record['sam3d_reviews']+[cable_review])

    def test_companion_replacement_wins_over_source_asset_for_same_component(self):
        replacement,replacement_review=self.native_review('approved-replacement','part',b'glTFreplacement geometry')
        cable,cable_review=self.native_review('approved-cable','cable',b'glTFapproved cable')
        dynamic=copy.deepcopy(self.record)
        dynamic['components'].append({'id':'cable','asset':{'track_id':'cable',**cable['native_glb'],
            'metadata':cable['native_metadata']}})
        dynamic['sam3d_reviews'] += [replacement_review,cable_review]
        record=self.approved_with_companion(dynamic,[replacement])
        self.assertEqual(record['sam3d_reviews'],[replacement_review,cable_review])
        self.assertNotIn(self.record['sam3d_reviews'][0],record['sam3d_reviews'])

    def test_review_receipt_hash_and_optional_size_are_always_enforced(self):
        binding,review=self.native_review('receipt-forms','part',b'glTFsource bound')
        self.assertEqual(C.approved_reviews(self.root,[review],[binding]),[review])
        manifest=C.read(self.root/review['review_manifest']['path'])
        receipt=manifest['import_receipt']
        for changes in [{'sha256':'0'*64},{'bytes':1},{'extra':'unexpected'}]:
            with self.subTest(changes=changes):
                changed=copy.deepcopy(manifest);changed['import_receipt'].update(changes)
                wrong=copy.deepcopy(review)
                wrong['review_manifest']=self.file('data/receipt-forms/review/changed-manifest.json',changed)
                self.assertEqual(C.approved_reviews(self.root,[wrong],[binding]),[])

    def test_editorial_labels_are_source_bound_and_leave_model_title_unchanged(self):
        self.file('config/library-presentation.json',{'version':1,'recordings':{
            self.video['sha256']:{'title':'A concise public title','note':'Generated draft; appearance needs review.'},
            'a'*64:{'title':'Another recording','note':'Unrelated editorial note.'}}})
        binding={'component_id':'part','native_glb':self.asset,'native_metadata':self.metadata}
        record=self.approved_with_companion(copy.deepcopy(self.record),[binding])
        self.assertEqual(record['public_title'],'A concise public title')
        self.assertEqual(record['public_note'],'Generated draft; appearance needs review.')
        self.assertEqual(record['title'],self.record['title'])
        self.assertEqual(record['search_index']['text'],self.record['search_index']['text']+' a concise public title')
        self.assertTrue(record['curation']['human_approval_recorded'])

    def test_editorial_labels_do_not_change_frozen_legacy_examples(self):
        self.seed()
        self.file('config/library-presentation.json',{'version':1,'recordings':{
            self.video['sha256']:{'title':'Should not change legacy title','note':'Only approved records receive labels.'}}})
        with patch.object(C,'_approved_records',return_value=[]):C.build(root=self.root)
        record=json.loads((self.root/'showcase/public/library-catalog.json').read_text())['records'][0]
        self.assertEqual(record['title'],self.record['title'])
        self.assertNotIn('public_title',record)
        self.assertNotIn('public_note',record)

    def test_approved_source_revision_preserves_public_url_and_exact_new_program(self):
        self.seed()
        with patch.object(C,'_approved_records',return_value=[]):C.build(root=self.root)
        changed=copy.deepcopy(self.program)
        changed['summary']='An explicitly approved new procedure'
        pointer=self.file('data/approved/revision2/interaction.json',changed)
        record=copy.deepcopy(self.record)
        record.update(id='new-working--revision',interaction=pointer)
        record['scene_case']={'id':record['id'],'original_sha256':pointer['sha256'],'scene_ref':None}
        case={**self.case,'id':record['id'],'interaction_path':pointer['path'],
              'original_sha256':pointer['sha256'],'interaction':changed}
        approved={'record':record,'case':case,'aligned':None}
        with patch.object(C,'_approved_records',return_value=[approved]):C.build(root=self.root)
        catalog=C.read(self.root/'showcase/public/library-catalog.json')
        cases=C.read(self.root/'showcase/public/cases-index.json')['cases']
        self.assertEqual(len(catalog['records']),1)
        self.assertEqual(catalog['records'][0]['id'],self.record['id'])
        self.assertEqual(catalog['records'][0]['scene_case']['id'],self.record['id'])
        self.assertEqual(cases[0]['id'],self.record['id'])
        self.assertEqual(catalog['records'][0]['interaction'],pointer)
        self.assertEqual(cases[0]['original_sha256'],pointer['sha256'])
        self.assertEqual(cases[0]['interaction_path'],pointer['path'])

    def test_changed_previous_catalog_cannot_reassign_public_identities(self):
        self.seed()
        with patch.object(C,'_approved_records',return_value=[]):C.build(root=self.root)
        index=C.read(self.root/'showcase/public/library-index.json')
        (self.root/'showcase/public'/index['catalog']['path']).write_text('{}')
        with patch.object(C,'_approved_records',return_value=[]),self.assertRaisesRegex(ValueError,'Previous public catalog changed'):
            C.build(root=self.root)

    def test_connected_scene_policy_keeps_incomplete_revision_out_of_library(self):
        self.seed()
        with patch.object(C,'_approved_records',return_value=[]):C.build(root=self.root)
        index=(self.root/'showcase/public/library-index.json').read_bytes()
        self.file('config/library-display-policy.json',{'version':1,'require_connected_scene':True})
        with patch.object(C,'_approved_records',return_value=[]),self.assertRaisesRegex(ValueError,'Connected scene required'):
            C.build(root=self.root)
        self.assertEqual((self.root/'showcase/public/library-index.json').read_bytes(),index)

    def test_presentation_config_rejects_unbounded_or_malformed_metadata(self):
        valid={'version':1,'recordings':{self.video['sha256']:{'title':'Draft','note':'Needs review.'}}}
        invalid=[]
        value=copy.deepcopy(valid);value['unknown']='private-feedback.json';invalid.append(value)
        value=copy.deepcopy(valid);value['version']=True;invalid.append(value)
        value=copy.deepcopy(valid);value['recordings']['invalid']=value['recordings'].pop(self.video['sha256']);invalid.append(value)
        for field,bad in [('title',121*'x'),('note',401*'x'),('title',[]),('note','line\ncontrol'),('note','')]:
            value=copy.deepcopy(valid);value['recordings'][self.video['sha256']][field]=bad;invalid.append(value)
        value=copy.deepcopy(valid);value['recordings'][self.video['sha256']]['feedback']='Private';invalid.append(value)
        for value in invalid:
            with self.subTest(value=value):
                self.file('config/library-presentation.json',value)
                with self.assertRaises(ValueError):C.presentation_metadata(self.root)
        self.file('config/library-presentation.json',b' '*65537)
        with self.assertRaisesRegex(ValueError,'64 KiB'):C.presentation_metadata(self.root)
        self.file('config/library-presentation.json',b'{"version":1,"version":1,"recordings":{}}')
        with self.assertRaisesRegex(ValueError,'Duplicate'):C.presentation_metadata(self.root)

    def test_presentation_config_rejects_symlink(self):
        target=self.file('data/editorial-labels.json',{'version':1,'recordings':{}})
        config=self.root/'config/library-presentation.json';config.parent.mkdir()
        config.symlink_to(self.root/target['path'])
        with self.assertRaisesRegex(ValueError,'symlink'):C.presentation_metadata(self.root)

    def test_newer_unapproved_same_source_model_is_never_grafted_into_approved_revision(self):
        pending_program=copy.deepcopy(self.program);pending_program['assets']=[self.pending_asset]
        pending_ref=self.file('data/pending/program/interaction.json',pending_program)
        pending_case={**self.case,'id':'pending--program','interaction_path':pending_ref['path'],'original_sha256':pending_ref['sha256']}
        self.file('workspace/cases-index.json',{'cases':[self.case,pending_case]})
        revision=self.file('.private/admin-workflow/revisions/approved.json',{'manifest':{'result':{'interaction_ref':self.interaction},
            'human_guidance_used':True,'private_feedback':'Never publish this feedback text'}})
        approval={'manifest_ref':revision,'revision_id':'revision-fixture','fingerprint':'approved-fingerprint'}
        dynamic=copy.deepcopy(self.record)
        dynamic['sam3d_reviews'].append({'component_id':'part','native_glb':self.pending_asset})
        dynamic['other_saved_versions']=[{'path':pending_ref['path']}]
        dynamic['cached_insights']=[{'path':self.private['path']}]
        with patch('admin_workflow.AdminWorkflow.public_index',return_value={'records':[approval]}),\
             patch('twin_catalog.TwinCatalog.load',return_value=({},self.program,self.interaction['sha256'])) as load,\
             patch('scripts.build_robotics_library.build_record',return_value=dynamic):
            result=C._approved_records(self.root)
        load.assert_called_once_with(self.case['id'])
        reviews=result[0]['record']['sam3d_reviews']
        self.assertEqual([r['native_glb']['sha256'] for r in reviews],[self.asset['sha256']])
        self.assertEqual(result[0]['record']['other_saved_versions'],[])
        self.assertEqual(result[0]['record']['cached_insights'],[])
        self.assertTrue(result[0]['record']['curation']['human_approval_recorded'])
        self.assertTrue(result[0]['record']['curation']['human_guidance_used'])
        self.assertNotIn('Never publish this feedback text',json.dumps(result))

    def test_byte_identical_model_from_other_anchor_cannot_replace_an_approved_review(self):
        metadata=self.file('data/unapproved-anchor/metadata.json',{'source_anchor':'different frame and model context'})
        receipt=self.file('data/unapproved-anchor/import-receipt.json',{'import_files':[{'path':'metadata.json',
            'sha256':metadata['sha256'],'bytes':metadata['bytes']}]})
        manifest=self.file('data/unapproved-anchor/review/review-manifest.json',{'import_receipt':receipt})
        graft={'component_id':'part','native_glb':self.asset,'review_manifest':manifest}
        binding={'component_id':'part','native_glb':self.asset,'native_metadata':self.metadata}
        self.assertEqual(C.approved_reviews(self.root,[graft],[binding]),[])
        self.assertEqual(C.approved_reviews(self.root,self.record['sam3d_reviews'],[binding]),self.record['sam3d_reviews'])
        # A reused hash cannot relabel a different physical component either.
        wrong=copy.deepcopy(self.record['sam3d_reviews'][0]);wrong['component_id']='other_part'
        self.assertEqual(C.approved_reviews(self.root,[wrong],[binding]),[])

    def test_frozen_existing_example_ignores_live_catalog_and_admits_only_exact_assets(self):
        self.seed()
        with patch.object(C,'_approved_records',return_value=[]):
            first=C.build(root=self.root)
            # A newer internal catalog with the same source cannot promote itself.
            changed=copy.deepcopy(self.record);changed['components'][0]['asset']=self.pending_asset
            self.file('showcase/library-catalog.json',{'records':[changed]})
            second=C.build(root=self.root)
        self.assertEqual(first['catalog_sha256'],second['catalog_sha256'])
        manifest=json.loads((self.root/'showcase/public/public-assets.json').read_text())
        paths={a['path'] for a in manifest['assets']}
        self.assertIn(self.asset['path'],paths);self.assertIn(self.video['path'],paths)
        self.assertNotIn(self.pending_asset['path'],paths);self.assertNotIn(self.private['path'],paths)
        self.assertTrue(S.public_static(self.root/self.asset['path'],root=self.root))
        self.assertFalse(S.public_static(self.root/self.pending_asset['path'],root=self.root))
        self.assertFalse(S.public_static(self.root/self.private['path'],root=self.root))
        public_case=json.loads((self.root/'showcase/public/cases-index.json').read_text())['cases'][0]
        self.assertEqual(public_case['exports'],{});self.assertTrue(public_case['curated'])

    def test_changed_approved_bytes_are_refused_before_new_pointer_publication(self):
        self.seed()
        with patch.object(C,'_approved_records',return_value=[]):C.build(root=self.root)
        pointer=self.root/'showcase/public/library-index.json';before=pointer.read_bytes()
        (self.root/self.asset['path']).write_bytes(b'glTFchanged-model-fixture')
        with patch.object(C,'_approved_records',return_value=[]),self.assertRaisesRegex(ValueError,'artifact changed'):
            C.build(root=self.root)
        self.assertEqual(pointer.read_bytes(),before)
        self.assertFalse(S.public_static(self.root/self.asset['path'],root=self.root))

    def test_copied_browser_scene_is_source_bound_and_traversal_cannot_publish(self):
        raw=b'{"objects":[],"label":"approved scene fixture"}';digest=hashlib.sha256(raw).hexdigest()
        path=self.root/'workspace/scenes'/f'{digest}.json';path.parent.mkdir(parents=True);path.write_bytes(raw)
        self.case['scene_ref']={'path':'scenes/'+digest+'.json','sha256':digest,'bytes':len(raw)};self.seed()
        with patch.object(C,'_approved_records',return_value=[]):C.build(root=self.root)
        copied=self.root/'showcase/public/scenes'/f'{digest}.json'
        self.assertEqual(copied.read_bytes(),raw);self.assertTrue(S.public_static(copied,root=self.root))
        self.case['scene_ref']['path']='../../.private/owner-data.json';self.seed()
        with patch.object(C,'_approved_records',return_value=[]),self.assertRaisesRegex(ValueError,'curated scene path'):
            C.build(root=self.root)
        self.assertFalse((self.root/'showcase/.private/owner-data.json').exists())


if __name__=='__main__':unittest.main()
