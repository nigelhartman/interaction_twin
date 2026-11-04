"""Pure admission tests: no source frames, native rendering or hinge fitting."""
import copy,unittest
import numpy as np

from scripts import refit_laptop_appearance_hinge_20261103_v1 as R


def fixture():
    result=[]
    for identifier in ['laptop_base','laptop_lid']:
        result.append({'id':identifier,'primary_ranking':{'selected_id':'pca_flip'},
            'pca_hypotheses':[{'id':'pca_flip','rotation':np.diag([1.,-1.,-1.]).tolist()}],
            'frame_results':[{'source_sample_index':index,'ranking':{
                'selected_id':'pca_flip','top_two_margin':.04,
                'ranked':[{'id':'pca_flip','median_mask_iou':.8}]}}for index in R.F.SAMPLES]})
    return result


class AppearanceRefitGateTests(unittest.TestCase):
    def test_two_parts_need_same_predeclared_winner_at_three_distinct_frames(self):
        components=fixture()
        for component in components:
            for row in component['frame_results'][:2]:row['ranking']['top_two_margin']=.01
        before=copy.deepcopy(components);support,rotations=R.qualifying_support(components)
        self.assertEqual(set(rotations),{'laptop_base','laptop_lid'})
        for row in support.values():self.assertEqual(row['qualified_predeclared_samples'],R.F.SAMPLES[2:])
        self.assertEqual(components,before)

    def test_duplicating_a_good_sample_cannot_supply_a_vote(self):
        components=fixture();components[0]['frame_results'][1]=copy.deepcopy(components[0]['frame_results'][0])
        with self.assertRaisesRegex(ValueError,'predeclared sample'):R.qualifying_support(components)

    def test_missing_extra_or_reordered_samples_are_not_the_frozen_basis(self):
        for variant in ['missing','extra','reordered']:
            components=fixture();rows=components[0]['frame_results']
            if variant=='missing':rows.pop()
            elif variant=='extra':rows.append(copy.deepcopy(rows[-1]))
            else:rows[0],rows[1]=rows[1],rows[0]
            with self.subTest(variant=variant),self.assertRaisesRegex(ValueError,'predeclared sample'):
                R.qualifying_support(components)

    def test_one_weak_part_prevents_combined_refit_admission(self):
        components=fixture()
        for row in components[1]['frame_results'][:3]:row['ranking']['selected_id']=None
        support,rotations=R.qualifying_support(components)
        self.assertEqual(set(rotations),{'laptop_base'});self.assertNotEqual(len(rotations),len(components))
        self.assertEqual(len(support['laptop_lid']['qualified_predeclared_samples']),2)

    def test_ties_and_poor_silhouette_overlap_do_not_count_as_consensus(self):
        for failure in ['tie','poor_iou','different_winner']:
            components=fixture()
            for row in components[0]['frame_results'][:3]:
                if failure=='tie':row['ranking']['top_two_margin']=.024999
                elif failure=='poor_iou':row['ranking']['ranked'][0]['median_mask_iou']=.599999
                else:row['ranking']['selected_id']='another_hypothesis'
            with self.subTest(failure=failure):
                support,rotations=R.qualifying_support(components)
                self.assertNotIn('laptop_base',rotations)
                self.assertEqual(len(support['laptop_base']['qualified_predeclared_samples']),2)

    def test_only_exact_laptop_base_and_lid_ids_accepted(self):
        for identifiers in [('laptop_base','power_cable'),('laptop_base','laptop_base')]:
            components=fixture()
            for component,identifier in zip(components,identifiers):component['id']=identifier
            with self.subTest(ids=identifiers),self.assertRaisesRegex(ValueError,'Exact laptop base/lid'):
                R.qualifying_support(components)

    def test_reflections_and_scale_cannot_hide_in_a_qualifying_hypothesis(self):
        for transform in [np.diag([-1.,1.,1.]),np.diag([1.01,1.,1.])]:
            components=fixture();components[0]['pca_hypotheses'][0]['rotation']=transform.tolist()
            with self.assertRaisesRegex(ValueError,'proper rigid'):R.qualifying_support(components)

    def test_no_primary_selected_hypothesis_cannot_be_replaced_by_majority(self):
        components=fixture();components[0]['primary_ranking']['selected_id']=None
        support,rotations=R.qualifying_support(components)
        self.assertNotIn('laptop_base',rotations)
        self.assertEqual(support['laptop_base']['qualified_predeclared_samples'],[])


if __name__=='__main__':unittest.main()
