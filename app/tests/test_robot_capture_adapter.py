"""Private measured geometry and authored staging retain exact source identity."""
import copy,json,tempfile,unittest
from pathlib import Path
from robot_presentation_capture import adapter,ROOT
from marketing_sources import file_ref

BASE='output/brand-trailer-20261103/robot-capture-v2/baseline-task.json'
REPORT='data/assembly-refinements/jabra-rim-fit-20261103-final-v1/report.json'

class CaptureAdapterTests(unittest.TestCase):
    def test_actual_adapter_binds_derived_lid_and_measured_containment_without_mutating_sources(self):
        before={p:file_ref(p)for p in [BASE,REPORT]};value=adapter(BASE,REPORT)
        self.assertEqual(value['baseline_task_ref'],before[BASE]);self.assertEqual(value['refinement_report_ref'],before[REPORT])
        self.assertEqual([r['id']for r in value['render_asset_overrides']],['case_lid'])
        self.assertEqual(value['actuation']['asset_sha256'],value['render_asset_overrides'][0]['asset_ref']['sha256'])
        self.assertFalse(value['physical_accuracy_verified']);self.assertEqual(value['new_model_calls'],0)
        for row in value['contained_staging']:
            self.assertAlmostEqual(row['offset'][1]+row['display_half_height']+row['clearance_display_units'],row['opening_rim_offset_y'])
            self.assertGreater(row['offset'][1]-row['display_half_height'],row['body_bottom_offset_y']+row['clearance_display_units'])
            self.assertFalse(row['physical_seating_verified'])
        self.assertEqual({p:file_ref(p)for p in before},before)
    def test_foreign_original_mesh_or_physics_claim_cannot_enter_the_adapter(self):
        report=json.loads((ROOT/REPORT).read_text())
        with tempfile.TemporaryDirectory(dir=ROOT/'output')as temporary:
            path=Path(temporary)/'report.json'
            for mutate in [lambda d:d.update(physical_accuracy_verified=True),lambda d:d['assets'][0]['original_asset'].update(sha256='0'*64)]:
                changed=copy.deepcopy(report);mutate(changed);path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):adapter(BASE,path)
    def test_oversize_contained_geometry_fails_instead_of_claiming_seating(self):
        baseline=json.loads((ROOT/BASE).read_text());next(o for o in baseline['objects']if o['id']=='earbud_a')['display_max_dimension']=1.
        with tempfile.TemporaryDirectory(dir=ROOT/'output')as temporary:
            path=Path(temporary)/'baseline.json';path.write_text(json.dumps(baseline))
            with self.assertRaisesRegex(ValueError,'does not fit'):adapter(path,REPORT)

if __name__=='__main__':unittest.main()
