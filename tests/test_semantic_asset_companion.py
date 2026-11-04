"""Exact source grounding, logical features, deferred geometry and no dedup claims."""
import concurrent.futures,copy,json,shutil,unittest
from pathlib import Path
import semantic_asset_companion as S
from tests import test_twin_catalog as fixtures

class SemanticCompanionTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.TwinCatalogTests(methodName='runTest');self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root;self.program='data/case/procedure/interaction.json';self.output='output/semantic-review'
        d=self.fixture.doc;d['tracklets'][0]['label']='USB-C plug';d['tracklets'][1]['label']='phone USB-C port'
        for identifier,label in [('charger','charger'),('phone','phone'),('cable','flexible cable'),('laptop','laptop'),('button','power button'),('cap','screw cap'),('bottle','water bottle'),('receiver','cup')]:
            track=copy.deepcopy(d['tracklets'][0]);track.update(id=identifier,label=label,candidate_entity_ids=[identifier+'_candidate']);d['tracklets'].append(track)
        for identifier,predicate,rationale in [('thread','removable_from','Cap rotates until screw thread disengages.'),('tether','attached_to','The open cap remains attached by a flexible tether and retaining ring.')]:
            d['relationships'].append({'id':identifier,'subject_track_id':'cap','object_track_id':'bottle','predicate':predicate,'status':'visual_model_proposal','rationale':rationale,'limitations':['Thread pitch and tether geometry unknown.'],'evidence_ids':['source_000002','source_000003']})
        self.fixture.publish()
    def file(self,path,value):
        self.fixture.file(path,value);return S.file_ref(path,self.root)
    def entity(self,identifier,kind,track,attributes=None,**extra):
        return{'id':identifier,'kind':kind,'track_ids':[track],'feature_label':identifier,'attributes':attributes or{},'evidence_ids':[],**extra}
    def context(self,program=None,*,entities=None,relations=(),origin='explicit_owner_context'):
        value=S.make_owner_context(program or self.program,entities=entities or[self.entity('owner_charger','charger','charger',owner_reuse_key='charger_from_my_desk')],relations=relations,note='Owner supplied recording context; exact component binding is a proposal.',root=self.root,origin=origin)
        return S.write_context(value,self.output,root=self.root)
    def companion(self,program=None,context=None):
        value=S.build(program or self.program,owner_context_path=context['path']if context else None,root=self.root)
        return S.write_companion(value,self.output,root=self.root)
    def second_program(self):
        shutil.copytree(self.root/'data/case',self.root/'data/case-two')
        def rewrite(v):
            if isinstance(v,str):return v.replace('data/case/','data/case-two/')
            if isinstance(v,list):return[rewrite(x)for x in v]
            if isinstance(v,dict):return{k:rewrite(x)for k,x in v.items()}
            return v
        d=rewrite(copy.deepcopy(self.fixture.doc));d['case_id']='second_recording'
        source=self.file(d['source']['path'],b'ANOTHER DISTINCT CAPTURE');d['source'].update(sha256=source['sha256'],original_sha256=source['sha256'])
        manifest=next(i for i in d['inputs']if i['kind']=='source_manifest');m=json.loads((self.root/manifest['path']).read_text());m['source']['sha256']=source['sha256'];manifest['sha256']=self.file(manifest['path'],m)['sha256']
        path='data/case-two/procedure/interaction.json';self.file(path,d);return path
    def test_program_features_are_hypotheses_with_exact_pointer_provenance_and_deferred_geometry(self):
        value=S.build(self.program,root=self.root);kinds={e['kind']for e in value['entities']}
        self.assertTrue({'connector_endpoint','receptacle','power_button','threaded_cap','flexible_cable','vessel'}<=kinds)
        for e in value['entities']:
            self.assertEqual(e['provenance']['kind'],'program_label_classifier');self.assertFalse(e['provenance']['classification_is_observation']);self.assertIsNone(e['geometry']['metric_pose']);self.assertFalse(e['physical_identity_verified'])
        cable=next(e for e in value['entities']if e['kind']=='flexible_cable');self.assertEqual(cable['attributes']['deformability'],'nonrigid_hypothesis');self.assertIsNone(cable['geometry']['cable_centerline'])
        port=next(e for e in value['entities']if e['kind']=='receptacle');self.assertEqual(port['geometry']['state'],'deferred');self.assertIsNone(port['geometry']['port_frame'])
        self.assertFalse(value['claims']['certified_compatibility']);self.assertFalse(value['claims']['public_or_owner_approved'])
    def test_unscrewing_does_not_invent_a_free_cap_or_fixed_tether_hinge(self):
        value=S.build(self.program,root=self.root);closure=next(r for r in value['relations']if r['kind']=='threaded_closure');tether=next(r for r in value['relations']if r['kind']=='retained_tether')
        self.assertEqual(closure['motion_contract']['attachment'],'thread_disengaged_but_flexible_tether_retained_hypothesis')
        self.assertEqual(closure['motion_contract']['stages'],['rotate_while_engaged','thread_disengagement','retained_by_flexible_tether'])
        self.assertNotIn('free_body',closure['motion_contract']['stages']);self.assertIsNone(tether['motion_contract']['joint_axis']);self.assertIsNone(closure['motion_contract']['pitch_m_per_turn'])
    def test_negated_uncertain_or_limitation_only_tethers_never_establish_retention(self):
        for status,rationale in [('unknown','The cap remains attached by a tether.'),('inferred','There is no tether visible.'),('visual_model_proposal','Cap connection unclear; tether not resolved.')]:
            relation=next(r for r in self.fixture.doc['relationships']if r['id']=='tether');relation.update(status=status,rationale=rationale);self.fixture.publish()
            with self.subTest(status=status,rationale=rationale):
                value=S.build(self.program,root=self.root);self.assertFalse(any(r['kind']=='retained_tether'for r in value['relations']))
                self.assertEqual(next(r for r in value['relations']if r['kind']=='threaded_closure')['motion_contract']['attachment'],'conditional_thread_engagement_retention_unknown')
    def test_logical_button_pour_and_endpoint_features_do_not_need_manual_poses_or_new_masks(self):
        entities=[self.entity('button_feature','power_button','laptop',{'activation':'press','effect':'request_power_change'}),self.entity('computer','laptop','laptop'),self.entity('source_bottle','vessel','bottle',{'contents':'owner_reports_water'}),self.entity('receiving_cup','vessel','receiver'),self.entity('plug_tip','connector_endpoint','cable',{'standard':'usb_c','role':'plug'}),self.entity('phone_port','receptacle','phone',{'standard':'usb_c','role':'port'})]
        relations=[{'id':i,'kind':k,'subject_id':a,'object_id':b,'evidence_ids':[]}for i,k,a,b in [('control','controls','button_feature','computer'),('pour','pour_transfer','source_bottle','receiving_cup'),('connect','connection_candidate','plug_tip','phone_port')]]
        context=self.context(entities=entities,relations=relations);value=S.load(self.companion(context=context)['path'],self.root)
        owned=[e for e in value['entities']if e['provenance']['kind']=='explicit_owner_context'];self.assertEqual(len(owned),6)
        self.assertTrue(all(not e['provenance']['component_binding_owner_confirmed']for e in owned));self.assertTrue(all(e['geometry']['port_frame']is None for e in owned));self.assertFalse(value['claims']['water_physics_recovered'])
        self.assertIn('fluid_physics',next(r for r in value['relations']if r['kind']=='pour_transfer')['missing_requirements'])
    def test_owner_free_body_claim_conflicts_with_source_tether_but_never_releases_cap(self):
        context=self.context(entities=[self.entity('owner_cap','threaded_cap','cap',{'sequence':'rotate_then_disengage','retention':'free_body_hypothesis'}),self.entity('owner_bottle','vessel','bottle')],relations=[{'id':'unscrew','kind':'threaded_closure','subject_id':'owner_cap','object_id':'owner_bottle','evidence_ids':[]}])
        value=S.load(self.companion(context=context)['path'],self.root);self.assertTrue(value['conflicts']);self.assertTrue(all(not c['free_body_motion_authorized']for c in value['conflicts']))
    def test_generic_cap_typing_never_drops_actual_model_tether_and_plug_is_not_whole_cable(self):
        next(t for t in self.fixture.doc['tracklets']if t['id']=='cap')['label']='blue cap'
        self.fixture.doc['tracklets'][0]['label']='cable connector plug';self.fixture.publish()
        context=self.context(entities=[self.entity('typed_cap','threaded_cap','cap',{'sequence':'rotate_then_disengage','retention':'unknown'}),self.entity('typed_bottle','vessel','bottle')])
        value=S.load(self.companion(context=context)['path'],self.root)
        self.assertTrue(any(r['kind']=='retained_tether'and'feature_typing_context_refs'in r['provenance']for r in value['relations']))
        plug=next(e for e in value['entities']if e['local_id']=='part');self.assertEqual(plug['kind'],'connector_endpoint');self.assertEqual(plug['attributes']['standard'],'unknown')
    def test_charger_reuse_stays_two_entities_and_usb_c_candidate_is_not_certified(self):
        second=self.second_program();a=self.companion(context=self.context());b=self.companion(second,self.context(second))
        catalog=S.build_catalog([a['path'],b['path']],root=self.root);self.assertEqual(len(catalog['records']),2);self.assertEqual(len(catalog['possible_reuse']),1)
        reuse=catalog['possible_reuse'][0];self.assertIsNone(reuse['canonical_asset_id']);self.assertFalse(reuse['identity_verified']);self.assertFalse(reuse['deduplication_performed']);self.assertEqual(len(reuse['entity_ids']),2)
        self.assertTrue(catalog['connection_candidates']);self.assertTrue(all(not r['certified_compatible']and r['mating_pose']is None for r in catalog['connection_candidates']))
        reference=S.write_catalog(catalog,self.output,root=self.root);loaded=S.load_catalog(reference['path'],self.root);query=S.query_catalog(loaded,kind='receptacle',standard='usb_c',geometry_state='deferred',limit=1)
        self.assertEqual(query['total'],2);self.assertTrue(query['has_more']);self.assertFalse(query['public_library_access'])
    def test_matching_model_labels_alone_never_merge_or_prove_the_same_charger(self):
        second=self.second_program();a=self.companion();b=self.companion(second);catalog=S.build_catalog([a['path'],b['path']],root=self.root)
        self.assertEqual(catalog['possible_reuse'],[]);self.assertEqual(len([e for e in catalog['entities']if e['kind']=='charger']),2)
    def test_offscreen_owner_charger_reference_does_not_invent_a_visible_track_or_duplicate_mesh(self):
        second=self.second_program();a=self.companion(context=self.context())
        entity=self.entity('owner_charger','charger','charger',owner_reuse_key='charger_from_my_desk');entity.update(track_ids=[],presence='owner_reported_offscreen')
        b=self.companion(second,self.context(second,entities=[entity]));catalog=S.build_catalog([a['path'],b['path']],root=self.root)
        reference=next(e for e in catalog['entities']if e['presence']=='owner_reported_offscreen')
        self.assertEqual(reference['track_ids'],[]);self.assertEqual(reference['geometry']['track_assets'],[]);self.assertTrue(reference['geometry']['reference_only_no_new_asset']);self.assertEqual(len(catalog['possible_reuse']),1)
    def test_context_literal_and_batch_source_are_bound_independently_of_operator_mapping(self):
        note='Owner reports USB-C charging.';source=S.file_ref(self.fixture.doc['source']['path'],self.root)
        batch=self.file('output/batch.json',{'format':'interaction-twin-gpu-priority-capture-batch','captures':[{'source_ref':source,'owner_context':note}]})
        value=S.make_owner_context(self.program,note=note,entities=[self.entity('charger','charger','charger')],root=self.root,context_source_ref=batch,context_source_pointer='/captures/0/owner_context')
        self.assertEqual(S.write_context(value,self.output,root=self.root)['bytes']>0,True)
        with self.assertRaisesRegex(ValueError,'literal source'):S.make_owner_context(self.program,note='Different note',root=self.root,context_source_ref=batch,context_source_pointer='/captures/0/owner_context')
        second=self.second_program()
        with self.assertRaisesRegex(ValueError,'another source recording'):S.make_owner_context(second,note=note,root=self.root,context_source_ref=batch,context_source_pointer='/captures/0/owner_context')
    def test_feedback_is_append_only_exact_revision_bound_and_does_not_train_or_promote(self):
        context=self.context();original=self.companion(context=context);previous=S.load(original['path'],self.root);charger=next(e for e in previous['entities']if e['local_id']=='owner_charger')
        feedback={'format':'interaction-twin-semantic-owner-feedback','version':1,'companion_ref':original,'program_ref':previous['program_ref'],'source_sha256':previous['source_sha256'],'reviews':[{'id':charger['id'],'disposition':'owner_rejects','comment':'This is a different charger.'}],'note':'Owner identity feedback, not model training.','untrusted_owner_belief':True,'training_performed':False}
        ref=self.file('output/feedback.json',feedback);changed=S.append_feedback(original['path'],ref['path'],root=self.root);saved=S.write_companion(changed,self.output,root=self.root);loaded=S.load(saved['path'],self.root)
        self.assertEqual(loaded['supersedes_companion_ref'],original);self.assertEqual(S.load(original['path'],self.root),previous);self.assertEqual(next(e for e in loaded['entities']if e['id']==charger['id'])['owner_review'],'owner_rejects')
        self.assertEqual(loaded['claims'],S.CLAIMS)
        feedback['companion_ref']={'path':saved['path'],'sha256':'0'*64,'bytes':saved['bytes']};self.file('output/feedback.json',feedback)
        with self.assertRaises(ValueError):S.append_feedback(original['path'],'output/feedback.json',root=self.root)
        with self.assertRaises(ValueError):S.load(saved['path'],self.root)
    def test_context_rejects_foreign_tracks_metric_geometry_wrong_source_and_unsupported_physics(self):
        for patch in [{'track_ids':['foreign']},{'attributes':{'port_pose':[0,0,0]}},{'attributes':{'standard':'certified_usb_c'}},{'attributes':{'standard':[]}}]:
            entity=self.entity('plug','connector_endpoint','part',{'standard':'usb_c'});entity.update(patch)
            with self.subTest(patch=patch),self.assertRaises(ValueError):self.context(entities=[entity])
        context=self.context();value=json.loads(S.checked(context,self.root));value['source_sha256']='0'*64;self.file('output/foreign-context.json',value)
        with self.assertRaises(ValueError):S.build(self.program,owner_context_path='output/foreign-context.json',root=self.root)
        with self.assertRaises(ValueError):self.context(origin='source_review_annotation')
        path=self.root/'output/duplicate.json';path.write_text('{"format":1,"format":2}')
        with self.assertRaisesRegex(ValueError,'Duplicate'):S.build(self.program,owner_context_path='output/duplicate.json',root=self.root)
    def test_immutable_concurrent_writes_and_source_or_descriptor_tampering_fail_closed(self):
        value=S.build(self.program,root=self.root)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:refs=list(pool.map(lambda _:S.write_companion(value,self.output,root=self.root),range(2)))
        self.assertEqual(refs[0],refs[1]);self.assertEqual((self.root/refs[0]['path']).stat().st_mode&0o777,0o600)
        changed=copy.deepcopy(value);changed['claims']['certified_compatibility']=True
        with self.assertRaises(ValueError):S.write_companion(changed,self.output,root=self.root)
        self.file('data/case/frames/1.jpg',b'changed source pixels')
        with self.assertRaises(ValueError):S.load(refs[0]['path'],self.root)

if __name__=='__main__':unittest.main()
