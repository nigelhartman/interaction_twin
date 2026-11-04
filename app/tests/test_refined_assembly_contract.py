"""The derived branch cannot relabel edits as native or bypass unknowns/gates."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import refined_assembly_contract as D


class SourceMaskPolicyTests(unittest.TestCase):
    def row(self, child=.8, union=.8, original=.8, retained=True):
        return {'mask_iou': {'child': child, 'union': union, 'parent': .7},
                'original_child_convex_mask_iou': original, 'retained': retained,
                'facing_ambiguity_rejected':False}

    def test_per_sample_low_agreement_or_regression_is_withheld(self):
        self.assertTrue(D.eligible_row(self.row()))
        for row in [self.row(child=.54), self.row(union=.59), self.row(child=.7, original=.9)]:
            self.assertFalse(D.eligible_row(row))

    def test_too_few_retained_samples_cannot_select_default_even_with_perfect_fit(self):
        self.assertFalse(D.summarize([self.row() for _ in range(11)], 20)['default_eligible'])
        rows = [self.row(retained=i<12) for i in range(30)]
        self.assertFalse(D.summarize(rows, 30)['default_eligible'])
        self.assertTrue(D.summarize([self.row() for _ in range(12)], 20)['default_eligible'])

    def test_projected_mesh_is_not_clipped_to_the_evidence_mask(self):
        camera = {'K': [[10,0,10],[0,10,10],[0,0,1]],'width':20,'height':20}
        vertices = np.array([[-.8,-.8,2],[.8,-.8,2],[.8,.8,2],[-.8,.8,2]])
        result = D.hull_mask(vertices, np.eye(4), camera)
        evidence = np.zeros((20,20), bool); evidence[9:12,9:12] = True
        self.assertGreater(np.count_nonzero(result & ~evidence), 40)
        with self.assertRaisesRegex(ValueError, 'near plane'):
            D.project(vertices*np.array([1,1,-1]),np.eye(4),camera['K'])

    def test_open_state_bracketing_with_away_facing_cavity_withholds_without_inventing_pose(self):
        bracket=D.state_bracket(23,{21:'open',28:'open',101:'closed'})
        self.assertEqual(bracket,{'state':'open','source_sample_indices':[21,28]})
        self.assertTrue(D.facing_ambiguity(bracket,.68,-.25))
        self.assertFalse(D.facing_ambiguity(D.state_bracket(101,{94:'open',101:'closed'}),.68,-.85))
        self.assertFalse(D.facing_ambiguity(bracket,.68,.65))
        row=self.row();row['facing_ambiguity_rejected']=True
        self.assertFalse(D.eligible_row(row),'A good silhouette cannot establish the correct visible cavity face.')


class ActualDerivedContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.path = cls.root/'data/assembly-refinements/jabra-source-refit-20261103-v2/derived-assembly.json'
        if not cls.path.is_file():
            raise unittest.SkipTest('Actual cached derivative has not been generated')
        cls.document = json.loads(cls.path.read_text())

    def test_actual_fixed_geometry_mask_scores_and_facing_ambiguous_unknowns_recompute(self):
        value = D.validate(copy.deepcopy(self.document), root=self.root)
        self.assertTrue(value['source_audit_summary']['default_eligible'])
        self.assertEqual(value['source_audit_summary']['original_unknown_samples'],23)
        self.assertEqual(value['source_audit_summary']['total_unknown_samples'],52)
        self.assertEqual(value['source_audit_summary']['retained_samples'],49)
        self.assertFalse(value['claims']['physical_accuracy_verified'])
        original = D.read(value['base_object_program_ref'],self.root)
        self.assertNotEqual(value['component_assets'][1]['native_glb'],original['component_assets'][1]['native_glb'])
        self.assertEqual(value['component_assets'][1]['original_native_glb'],original['component_assets'][1]['native_glb'])

    def test_physical_upgrade_source_changes_and_native_relabelling_fail(self):
        for edit in [lambda d:d.update(physical_accuracy_verified=True),
                     lambda d:d['claims'].update(declared_derivative_geometry_used=False)]:
            changed=copy.deepcopy(self.document);edit(changed)
            with self.assertRaises(ValueError):D.validate(changed,root=self.root)
        changed=copy.deepcopy(self.document);changed['component_assets'][1]['geometry_origin']='unchanged_SAM_3D_inference'
        with self.assertRaisesRegex(ValueError,'descriptive bindings'):
            D.validate(changed,root=self.root)

    def test_saved_quality_scores_cannot_be_laundered_into_visibility(self):
        real_read = D.read
        audit_path=self.document['source_alignment_audit_ref']['path']
        def tampered(reference,root=D.ROOT,maximum=32*1024*1024):
            value=real_read(reference,root,maximum)
            if reference['path']==audit_path:
                first=next(r for r in value['rows'] if not r['retained']);first['retained']=True
            return value
        with patch.object(D,'read',side_effect=tampered):
            with self.assertRaisesRegex(ValueError,'quality gate'):
                D.validate(copy.deepcopy(self.document),root=self.root)

    def test_fit_parameter_bounds_must_be_recomputed_from_source_extent(self):
        real_read = D.read
        audit_path=self.document['source_alignment_audit_ref']['path']
        def tampered(reference,root=D.ROOT,maximum=32*1024*1024):
            value=real_read(reference,root,maximum)
            if reference['path']==audit_path:
                value['rows'][0]['parameter_bounds'][0]=[-100,100]
            return value
        with patch.object(D,'read',side_effect=tampered):
            with self.assertRaisesRegex(ValueError,'budget/bounds'):
                D.validate(copy.deepcopy(self.document),root=self.root)


if __name__=='__main__':unittest.main()
