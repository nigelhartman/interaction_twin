"""Meaningful discovery checks: source families, filters and uncertainty."""
import copy
import unittest

from library_query import derive_index, query_records


def record(identifier, source, *, label='Camera battery', part='battery', action='remove_battery',
           title='Remove battery', relation='removable_from', backend='TripoSR', motion=0):
    asset = {'backend': backend} if backend else None
    return {'id': identifier, 'title': label, 'summary': 'A component is removed from a receptacle.',
            'source': {'sha256': source, 'original_sha256': source, 'duration_s': 10},
            'components': [{'id': part, 'label': label, 'asset': asset}],
            'steps': [{'id': action, 'title': title, 'description': 'Demonstrated visible handling'}],
            'relationships': [{'predicate': relation}], 'supported_motion_samples': motion,
            'counts': {'components': 1, 'assets': int(bool(asset)), 'steps': 1}}


class LibraryQueryTests(unittest.TestCase):
    def setUp(self):
        self.old = record('demo-20261030--procedure', 'a' * 64)
        self.new = record('demo-20261102--procedure', 'a' * 64)
        self.wallet = record('wallet-20261101--procedure', 'b' * 64, label='Gold coin', part='coin',
                             action='insert_coin', title='Insert coin', relation='contained_in', backend=None)
        self.rows = [self.old, self.wallet, self.new]

    def test_source_group_dedupes_versions_before_count_filter_and_paging(self):
        result = query_records(self.rows, limit=1)
        self.assertEqual(result['total'], 2)
        self.assertEqual(result['records'][0]['id'], self.new['id'])
        self.assertEqual(result['records'][0]['saved_variant_ids'], [self.new['id'], self.old['id']])
        self.assertEqual(result['facets']['parts'], [{'value': 'battery', 'label': 'Camera battery', 'count': 1},
                                                    {'value': 'coin', 'label': 'Gold coin', 'count': 1}])
        self.assertTrue(result['has_more'])
        second = query_records(self.rows, limit=1, offset=result['next_offset'])
        self.assertEqual(second['records'][0]['id'], self.wallet['id'])
        self.assertFalse(second['has_more'])
        self.assertIsNone(second['next_offset'])
        self.assertEqual(query_records(self.rows, group_by='variant')['total'], 3)

    def test_exact_saved_filters_or_within_field_and_between_fields(self):
        result = query_records(self.rows, 'BATTERY receptacle', parts=['coin', 'BATTERY'],
                               actions=['remove_battery'], relations=['removable_from'], backends=['TripoSR'])
        self.assertEqual([r['id'] for r in result['records']], [self.new['id']])
        self.assertEqual(query_records(self.rows, parts=['battery'], actions=['insert_coin'])['total'], 0)
        self.assertEqual(query_records(self.rows, actions=['remove'])['total'], 0)
        self.assertEqual(query_records(self.rows, 'insert coin')['records'][0]['id'], self.wallet['id'])
        self.assertEqual(result['facets'], query_records(self.rows)['facets'])

    def test_candidate_backends_require_exact_video_component_and_unselected_binding(self):
        candidates = [{'component_id': 'coin', 'source_sha256': 'b' * 64, 'selected': False, 'backend': 'SAM 3D Objects'}]
        self.wallet['sam3d_reviews'] = candidates
        result = query_records(self.rows, backends=['sam3d'])
        self.assertEqual([r['id'] for r in result['records']], [self.wallet['id']])
        index = result['records'][0]['search_index']
        self.assertEqual(index['candidate_backends'], ['sam3d'])
        self.assertEqual(index['selected_backends'], [])
        self.assertEqual(index['readiness'], ['geometry_draft'])
        for change in [{'source_sha256': 'c' * 64}, {'component_id': 'different'}, {'selected': True}]:
            with self.subTest(change=change):
                self.wallet['sam3d_reviews'] = [dict(candidates[0], **change)]
                self.assertEqual(query_records(self.rows, backends=['sam3d'])['total'], 0)

    def test_readiness_never_trusts_saved_physics_claim_or_loose_index(self):
        self.wallet['simulation_ready'] = True
        self.wallet['search_index'] = {'backends': ['sam3d'], 'readiness': ['physics_ready']}
        index = derive_index(self.wallet)
        self.assertEqual(index['readiness'], ['semantic_only'])
        self.assertFalse(index['simulation_ready'])
        self.assertEqual(query_records(self.rows, readiness=['physics_ready'])['total'], 0)
        self.assertEqual(query_records(self.rows, readiness=['semantic_only'])['total'], 1)
        self.new['supported_motion_samples'] = 3
        self.assertEqual(query_records(self.rows, readiness=['motion_draft'])['total'], 1)

    def test_native_candidate_never_inherits_historical_triposr_motion_readiness(self):
        self.new['supported_motion_samples'] = 17
        self.new['sam3d_reviews'] = [{'component_id': 'battery', 'source_sha256': 'a' * 64,
                                     'selected': False, 'backend': 'SAM 3D Objects'}]
        index = derive_index(self.new)
        self.assertEqual(index['readiness'], ['geometry_draft'])
        self.assertEqual(index['selected_motion_samples'], 17)
        self.assertEqual(index['native_supported_motion_samples'], 0)
        self.assertEqual(index['native_motion_status'], 'candidate_not_aligned')
        self.assertEqual(query_records(self.rows, backends=['sam3d'], readiness=['motion_draft'])['total'], 0)

    def test_native_motion_requires_supported_rows_for_selected_sam_component(self):
        self.new['components'][0]['asset']['backend'] = 'SAM 3D Objects'
        self.new['supported_motion_samples'] = 17
        self.new['motion'] = [{'track_id': 'other_part', 'status': 'supported_conditional_motion_candidate'},
                              {'track_id': 'battery', 'status': 'withheld_insufficient_evidence'}]
        self.assertEqual(derive_index(self.new)['readiness'], ['geometry_draft'])
        self.new['motion'].append({'track_id': 'battery', 'status': 'supported_conditional_motion_candidate'})
        index = derive_index(self.new)
        self.assertEqual(index['readiness'], ['motion_draft'])
        self.assertEqual(index['native_supported_motion_samples'], 1)
        self.assertEqual(index['native_motion_component_ids'], ['battery'])
        self.assertFalse(index['candidate_motion_inherited'])

    def test_native_motion_summary_for_human_catalog_stays_bound_to_native_components(self):
        self.new['native_supported_motion_samples'] = 4
        self.new['native_motion_component_ids'] = ['battery']
        self.assertEqual(derive_index(self.new)['native_supported_motion_samples'], 0)
        self.new['components'][0]['asset']['backend'] = 'SAM 3D Objects'
        self.assertEqual(derive_index(self.new)['native_supported_motion_samples'], 4)
        self.new['native_motion_component_ids'] = ['other_part']
        self.assertEqual(derive_index(self.new)['native_supported_motion_samples'], 0)

    def test_typed_task_classes_share_filter_or_and_facets_without_label_inference(self):
        self.new['task_classes']=['hinged_assemblies','removable_components']
        self.wallet['title']='Amazing sliding hinged phone connector'
        self.assertEqual(derive_index(self.wallet)['task_classes'],[])
        result=query_records(self.rows,task_classes=['HINGED_ASSEMBLIES','sliding_mechanisms'],parts=['battery'])
        self.assertEqual([r['id'] for r in result['records']],[self.new['id']])
        self.assertEqual(result['facets']['task_classes'],[
            {'value':'hinged_assemblies','label':'Hinged assemblies','count':1},
            {'value':'removable_components','label':'Removable components','count':1}])
        self.assertIn('task_classes',result['query_contract']['filters'])
        self.assertEqual(query_records(self.rows,task_classes=['connectors_and_insertion'])['total'],0)
        self.assertFalse(result['simulation_ready'])
        self.new['task_classes']=['physics_ready']
        self.assertEqual(derive_index(self.new)['task_classes'],[])

    def test_sort_ties_pagination_are_stable_and_input_unmodified(self):
        original = copy.deepcopy(self.rows)
        desc = query_records(self.rows, sort='title', direction='desc')
        self.assertEqual([r['id'] for r in desc['records']], [self.wallet['id'], self.new['id']])
        latest = query_records(self.rows, sort='version', direction='desc')
        self.assertEqual(latest['records'][0]['id'], self.new['id'])
        self.assertEqual(self.rows, original)
        desc['records'][0]['summary'] = 'Client mutation'
        self.assertEqual(self.rows, original)

    def test_unicode_normalization_matches_browser_lowercase_contract(self):
        self.new['title'] = 'Straße camera'
        self.assertEqual(query_records(self.rows, 'straße')['total'], 1)
        self.assertEqual(query_records(self.rows, 'STRASSE')['total'], 0)

    def test_display_title_is_searchable_without_replacing_model_title_or_inferring_classes(self):
        self.new['public_title']='USB-C charger + cable'
        self.new['public_note']='Editorial review note should not affect search.'
        before=derive_index(self.new)
        for query in ['usb-c charger','camera battery']:
            self.assertEqual([r['id'] for r in query_records(self.rows,query)['records']],[self.new['id']])
        self.assertEqual(query_records(self.rows,'editorial review')['total'],0)
        self.assertEqual(before['task_classes'],[])
        self.assertEqual(self.new['title'],'Camera battery')
        self.assertEqual([r['id'] for r in query_records(self.rows,sort='title')['records']],
                         [self.wallet['id'],self.new['id']])
        self.assertEqual([r['id'] for r in query_records(self.rows,sort='title',direction='desc')['records']],
                         [self.new['id'],self.wallet['id']])

    def test_bounded_unknown_and_malformed_inputs_rejected(self):
        for args in [{'limit': True}, {'offset': -1}, {'offset': 10001}, {'sort': 'physical_accuracy'},
                     {'parts': 'coin'}, {'task_classes':'hinged_assemblies'}, {'actions': ['']}, {'backends': ['x'] * 26}, {'group_by': 'all_files'},
                     {'direction': 'up'}, {'query': 'x' * 201}]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                query_records(self.rows, **args)


if __name__ == '__main__': unittest.main()
