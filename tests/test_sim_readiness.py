import copy
import json
import tempfile
import unittest
from pathlib import Path

import sim_readiness as S


def fixture():
    return {"format": "interaction-twin-evidence-procedure", "version": "0.1", "case_id": "generic",
            "source": {"path": "source.mp4", "sha256": "0" * 64},
            "tracklets": [{"id": "part", "label": "Part", "identity_status": "candidate_unverified"},
                          {"id": "holder", "label": "Holder", "identity_status": "candidate_unverified"}],
            "assets": [{"id": "asset_part", "track_id": "part", "path": "not-opened.glb",
                        "sha256": "1" * 64, "backend": "TripoSR", "geometry_status": "unreviewed_model_candidate",
                        "physical_scale_known": False}],
            "evidence": [{"id": "e1", "timestamp_s": 1.0}],
            "relationships": [{"id": "r1", "subject_track_id": "part", "predicate": "removable_from",
                               "object_track_id": "holder", "evidence_ids": ["e1"], "status": "inferred"}],
            "assertions": [],
            "steps": [{"id": "s1", "participant_track_ids": ["part", "holder"], "evidence_ids": ["e1"]}],
            "motion": [{"track_id": "part", "physical_pose_verified": False, "timestamp_s": 1.0,
                        "status": "supported_conditional_motion_candidate",
                        "camera_from_native_mesh": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]]},
                       {"track_id": "holder", "physical_pose_verified": False, "timestamp_s": 1.0,
                        "status": "unsupported", "camera_from_native_mesh": None}],
            "runtime_policy": {"simulation_ready": False, "physical_identity_confirmed": False,
                               "state_between_evidence": "unknown", "pose_between_evidence": "absent",
                               "pose_interpolation": False}}


class SimulationReadinessTests(unittest.TestCase):
    def test_available_evidence_is_separate_from_all_blocking_physical_unknowns(self):
        report = S.audit_simulation_readiness(fixture())
        checks = {c["id"]: c for c in report["checks"]}
        for key in ("source_evidence", "visual_assets", "conditional_motion", "task_specification"):
            self.assertEqual(checks[key]["status"], "available")
            self.assertFalse(checks[key]["blocking"])
        for key in ("metric_scale", "shared_frame", "collision_geometry", "mass_inertia", "joints",
                    "grasps", "target_frames", "robot_ik", "contact", "seating", "simulation_validation"):
            self.assertEqual(checks[key]["status"], "unknown")
            self.assertTrue(checks[key]["blocking"])
        self.assertFalse(report["simulation_ready"])
        self.assertFalse(report["simulation_runnable"])
        self.assertEqual(report["inventory"]["conditional_pose_samples"], 1)

    def test_draft_keeps_physical_fields_null_and_source_semantics_unchanged(self):
        document = fixture()
        before = copy.deepcopy(document)
        draft = S.build_simulation_draft(document)
        self.assertEqual(document, before)
        self.assertEqual(draft["candidate_relationships"], document["relationships"])
        self.assertEqual(draft["conditional_camera_motion"], document["motion"])
        self.assertFalse(draft["physical_execution_authorized"])
        for field in ("robot", "controller", "success_criteria", "validation_results", "world_frame"):
            self.assertIsNone(draft[field])
        for component in draft["components"]:
            for field in ("mass_kg", "inertia_kg_m2", "meters_per_native_unit", "world_from_body",
                          "collision_meshes", "grasp_frames", "receptacle_frames", "joint_definitions"):
                self.assertIsNone(component[field])

    def test_empty_inventory_stays_unknown_and_does_not_fabricate_geometry(self):
        document = fixture()
        document["assets"] = []
        document["motion"] = []
        document["evidence"] = []
        document["steps"] = []
        document["relationships"] = []
        report = S.audit_simulation_readiness(document)
        self.assertEqual(report["inventory"]["visual_assets"], 0)
        self.assertTrue(all(c["status"] == "unknown" for c in report["checks"]))
        self.assertEqual(S.build_simulation_draft(document)["components"][0]["visual_assets"], [])

    def test_no_source_paths_are_opened_or_interpreted_as_code(self):
        document = fixture()
        document["assets"][0]["path"] = "../../.secrets/do-not-read"
        document["tracklets"][0]["label"] = "$(do_not_execute)"
        draft = S.build_simulation_draft(document)
        self.assertEqual(draft["components"][0]["visual_assets"][0]["path"], "../../.secrets/do-not-read")
        self.assertEqual(draft["components"][0]["label"], "$(do_not_execute)")
        self.assertFalse(S.audit_simulation_readiness(document)["verification"]["artifact_hashes_verified"])

    def test_physical_promotions_and_gap_interpolation_are_rejected(self):
        mutations = [("policy", "simulation_ready"), ("policy", "physical_identity_confirmed"),
                     ("policy", "pose_interpolation"), ("asset", "physical_scale_known"),
                     ("motion", "physical_pose_verified")]
        for group, field in mutations:
            with self.subTest(field=field):
                document = fixture()
                target = document["runtime_policy"] if group == "policy" else document[group + "s" if group == "asset" else group][0]
                target[field] = True
                with self.assertRaises(ValueError):
                    S.audit_simulation_readiness(document)

    def test_invalid_links_duplicates_and_nonfinite_transforms_fail_closed(self):
        mutations = [lambda d: d["assets"][0].update(track_id="missing"),
                     lambda d: d["relationships"][0].update(evidence_ids=["missing"]),
                     lambda d: d["steps"][0].update(participant_track_ids=["missing"]),
                     lambda d: d["tracklets"].append(copy.deepcopy(d["tracklets"][0])),
                     lambda d: d["motion"][0].update(camera_from_native_mesh=[[1]]),
                     lambda d: d["motion"][0]["camera_from_native_mesh"][0].__setitem__(0, float("nan")),
                     lambda d: d.update(version="future")]
        for mutation in mutations:
            document = fixture()
            mutation(document)
            with self.assertRaises(ValueError):
                S.audit_simulation_readiness(document)

    def test_cli_writes_valid_reports_without_a_simulator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "interaction.json"
            source.write_text(json.dumps(fixture()))
            report, draft = root / "reports/readiness.json", root / "reports/draft.json"
            self.assertEqual(S.main([str(source), "--output", str(report), "--draft-output", str(draft)]), 0)
            self.assertFalse(json.loads(report.read_text())["simulation_ready"])
            self.assertFalse(json.loads(draft.read_text())["simulation_runnable"])

    def test_cli_cannot_overwrite_source_or_write_two_outputs_to_one_path(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "interaction.json"
            original = json.dumps(fixture())
            source.write_text(original)
            for args in ([str(source), "--output", str(source)],
                         [str(source), "--output", str(Path(temp) / "same.json"),
                          "--draft-output", str(Path(temp) / "same.json")]):
                with self.assertRaises(SystemExit) as error:
                    S.main(args)
                self.assertEqual(error.exception.code, 2)
            self.assertEqual(source.read_text(), original)


if __name__ == "__main__":
    unittest.main()
