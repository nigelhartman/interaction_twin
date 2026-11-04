"""A fresh semantic audit cannot weaken numeric, identity or component gates."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('audit_extension',ROOT/'scripts/stage4_audit_extension.py')
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)


def row(sample=1,**kwargs):
    value={'sample_index':sample,'timestamp_s':1.,'status':'unsupported_candidate_projection_retained',
           'reasons':['source_mask_not_plausible_in_model_audit'],'mask_model_audit_verdict':'unassessed',
           'candidate_rotation':[[1,0,0],[0,1,0],[0,0,1]],'candidate_translation':[0,0,2],
           'heldout_median_px':2.5,'usable_pose_exported':False}
    value.update(kwargs)
    return value


def check(sample=1,**kwargs):
    value={'track_id':'part','sample_index':sample,'mask_verdict':'plausible_visible_mask',
           'single_component_visible':True,'component_boundary_plausible':True,
           'other_components_inside_mask':False,'reason':'Visible mask isolates requested component.'}
    value.update(kwargs)
    return value


class AuditExtensionTests(unittest.TestCase):
    def test_only_unassessed_sole_failure_is_eligible(self):
        rows=[row(),row(2,reasons=['source_mask_not_plausible_in_model_audit','fewer_than_6_disjoint_heldout_points']),
              row(3,mask_model_audit_verdict='reject_wrong_or_lost_mask'),row(4,status='unsupported'),
              row(5,usable_pose_exported=True)]
        self.assertEqual([r['sample_index'] for r in extension.eligible_rows({'frames':rows})],[1])

    def test_success_preserves_numerics_and_original_experiment(self):
        source={'object_id':'part','frames':[row()]};before=copy.deepcopy(source)
        result=extension.regate(source,[check()],'audit.json','hash')
        self.assertEqual(source,before)
        out=result['frames'][0]
        self.assertEqual(out['status'],'supported_conditional_motion_candidate')
        self.assertEqual(out['candidate_rotation'],source['frames'][0]['candidate_rotation'])
        self.assertEqual(out['candidate_translation'],source['frames'][0]['candidate_translation'])
        self.assertEqual(out['heldout_median_px'],2.5)
        self.assertEqual(out['audit_extension']['previous_reasons'],source['frames'][0]['reasons'])

    def test_uncertainty_scope_failure_or_prior_veto_cannot_upgrade(self):
        for proposal in [check(mask_verdict='uncertain'),check(other_components_inside_mask=True),
                         check(single_component_visible=False),check(component_boundary_plausible=False)]:
            result=extension.regate({'object_id':'part','frames':[row()]},[proposal],'a','h')
            self.assertEqual(result['frames'][0]['status'],'unsupported_candidate_projection_retained')
        original=row(model_scope_audit={'single_component_visible':True,'component_boundary_plausible':True,'other_components_inside_mask':True})
        result=extension.regate({'object_id':'part','frames':[original]},[check()],'a','h')
        self.assertEqual(result['frames'][0]['status'],'unsupported_candidate_projection_retained')

    def test_exact_pair_set_and_complete_observations_are_required(self):
        for proposals in [[],[check(),check()],[check(sample=2)]]:
            with self.assertRaises(ValueError):extension.validate_pairs({'summary':'Test','observations':proposals},[('part',1)])
        with self.assertRaises(ValueError):extension.regate({'object_id':'part','frames':[row()]},[],'a','h')
        extension.validate_pairs({'summary':'Test','observations':[check()]},[('part',1)])


if __name__=='__main__':
    unittest.main()
