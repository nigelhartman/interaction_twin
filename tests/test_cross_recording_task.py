"""Typed source grounding, curation and unknown-physics boundaries for task composition."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cross_recording_task import ConnectionTaskCatalog,CONFIG_PATH,INSTRUCTION
from curated_twin_catalog import CuratedTwinCatalog
ROOT=Path(__file__).resolve().parents[1]
PHONE='flow-capture-phone-20261103-v1--flow-procedure-1'
CHARGER='flow-capture-charger-20261103-v1--flow-procedure-1'


class CrossRecordingTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real=CuratedTwinCatalog(ROOT)
        cls.context=json.loads((ROOT/CONFIG_PATH).read_text())
        cls.docs={tid:cls.real.load(tid) for tid in [PHONE,CHARGER]}

    def local(self,edit=None):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name);(root/'config').mkdir()
        context=copy.deepcopy(self.context)
        if edit:edit(context)
        (root/CONFIG_PATH).write_text(json.dumps(context))
        return ConnectionTaskCatalog(root,catalog=self.real)

    def test_actual_curated_draft_binds_two_sources_and_exact_native_assets(self):
        result=ConnectionTaskCatalog(ROOT).prepare(PHONE,CHARGER)
        self.assertEqual(result['status'],'draft_requires_validation')
        self.assertEqual([s['role']for s in result['sources']],['phone','charger'])
        for binding in result['sources']:
            _,doc,digest=self.docs[binding['twin_id']]
            self.assertEqual(binding['program_sha256'],digest)
            self.assertEqual({a['sha256']for a in binding['assets']},{a['sha256']for a in doc['assets']})
        self.assertEqual(result['robot_actions'],[])
        self.assertEqual(len(result['task_draft']['steps']),5)
        self.assertTrue(all(c['status']=='candidate_requires_validation'and c['electrical_compatibility']=='unknown'and c['physical_fit_verified']is False for c in result['connection_candidates']))
        cross=next(c for c in result['connection_candidates']if c['cross_recording'])
        self.assertEqual(cross['physical_instance_identity'],'unknown')
        self.assertFalse(cross['observed_pair_in_one_recording'])
        self.assertEqual(result['claims']['new_api_calls'],0)

    def test_connector_class_is_owner_context_not_prose_or_title_inference(self):
        result=self.local().interfaces(PHONE)
        self.assertEqual(result['owner_context']['status'],'owner_reported_not_independently_measured')
        self.assertTrue(all(i['connector_family']=='usb_c'and i['type_attribution']=='owner_capture_context'for i in result['interfaces']))
        # Removing the explicit typed record yields unknown, despite unchanged
        # phone title, contact prose, geometry and saved removable relation.
        c=self.local(lambda x:x['recordings'].pop(0))
        self.assertEqual(c.interfaces(PHONE)['interfaces'],[])
        with self.assertRaisesRegex(ValueError,'Required typed'):c.prepare(PHONE,CHARGER)

    def test_changed_approved_revision_or_original_source_refuses_registry(self):
        for field in ['program_sha256','original_source_sha256']:
            with self.subTest(field=field):
                c=self.local(lambda x:x['recordings'][0].__setitem__(field,'0'*64))
                with self.assertRaisesRegex(ValueError,'different approved revision'):c.interfaces(PHONE)

    def test_invalid_component_step_or_outside_evidence_cannot_be_laundered(self):
        for field,value in [('component_id','adapter_early'),('step_ids',['invented_step']),('evidence_ids',['source_000004'])]:
            with self.subTest(field=field):
                c=self.local(lambda x:x['recordings'][0]['interfaces'][0].__setitem__(field,value))
                with self.assertRaises(ValueError):c.interfaces(PHONE)

    def test_step_participant_must_match_even_with_real_step_and_evidence(self):
        c=self.local(lambda x:x['recordings'][0]['interfaces'][0].__setitem__('step_ids',['s4_bring_cable']))
        with self.assertRaisesRegex(ValueError,'outside its recorded steps'):c.interfaces(PHONE)
        c=self.local(lambda x:x['recordings'][0]['interfaces'][1].update(step_ids=['s1_hold_front'],evidence_ids=['source_000004']))
        with self.assertRaisesRegex(ValueError,'participant step'):c.interfaces(PHONE)

    def test_unsupported_goal_distinct_sources_and_ids_are_rejected(self):
        c=self.local()
        for ids,goal in [((PHONE,CHARGER),'Execute the real robot now.'),((PHONE,PHONE),INSTRUCTION),(('../private',CHARGER),INSTRUCTION)]:
            with self.subTest(ids=ids,goal=goal):
                with self.assertRaises(ValueError):c.prepare(*ids,goal)

    def test_private_catalog_never_allowed(self):
        with self.assertRaisesRegex(ValueError,'owner-curated'):ConnectionTaskCatalog(ROOT,catalog=object())
        private='flow-capture-bottle-20261103-v1--flow-procedure-1'
        with self.assertRaisesRegex(ValueError,'Unknown curated twin'):ConnectionTaskCatalog(ROOT).interfaces(private)

    def test_unsupported_family_and_schema_fields_rejected(self):
        for edit in [lambda x:x['recordings'][0]['interfaces'][0].update(connector_family='anything'),
                     lambda x:x['recordings'][0]['interfaces'][0].update(execute='shell'),
                     lambda x:x['recordings'][0]['owner_context'].update(status='measured')]:
            c=self.local(edit)
            with self.assertRaises(ValueError):c.interfaces(PHONE)

    def test_duplicate_interface_and_duplicate_json_keys_rejected(self):
        c=self.local(lambda x:x['recordings'][0]['interfaces'].append(copy.deepcopy(x['recordings'][0]['interfaces'][0])))
        with self.assertRaisesRegex(ValueError,'Duplicate interface'):c.interfaces(PHONE)
        c=self.local();(c.root/CONFIG_PATH).write_text('{"format":"x","format":"y"}')
        with self.assertRaisesRegex(ValueError,'Duplicate connection-context field'):c.interfaces(PHONE)

    def test_context_symlink_not_followed(self):
        c=self.local();path=c.root/CONFIG_PATH;path.unlink();path.symlink_to(ROOT/CONFIG_PATH)
        with self.assertRaisesRegex(ValueError,'unavailable'):c.interfaces(PHONE)

    def test_source_steps_are_exact_and_do_not_claim_charging_succeeded(self):
        result=self.local().prepare(PHONE,CHARGER)
        for step in result['task_draft']['steps']:
            for source in step['source_steps']:
                _,doc,digest=self.docs[source['twin_id']]
                actual=next(s for s in doc['steps']if s['id']==source['step_id'])
                self.assertEqual(source['evidence_ids'],actual['evidence_ids'])
                self.assertEqual(source['start_s'],actual['start_s']);self.assertEqual(source['end_s'],actual['end_s'])
        self.assertEqual(result['task_draft']['steps'][-1]['source_steps'],[])
        self.assertEqual(result['task_draft']['steps'][-1]['completion_status'],'charging_not_observed_or_verified')
        self.assertFalse(result['claims']['charging_success_verified'])
        self.assertFalse(result['claims']['simulation_ready'])
        self.assertIn('flexible_cable_model',{r['id']for r in result['missing_requirements']})


if __name__=='__main__':unittest.main()
