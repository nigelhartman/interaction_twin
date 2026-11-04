"""Check source-time provenance and conservative identity in the full-scene export."""
import hashlib
import json
import unittest
from pathlib import Path

from scripts.stage4_full_prepare import registry

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "data/camera-battery-full-scene"


class FullSceneContractTests(unittest.TestCase):
    def test_unresolved_tracklets_do_not_add_physical_objects(self):
        data = registry("test_source")
        self.assertEqual(len({row["id"] for row in data["objects"]}), 4)
        batteries = {row["id"] for row in data["objects"] if row["kind"] == "battery"}
        self.assertEqual(batteries, {"battery_00001", "battery_00002"})
        ambiguous = [row for row in data["observation_tracks"] if row["physical_object_id"] is None]
        self.assertEqual(len(ambiguous), 6)
        for row in ambiguous:
            self.assertEqual(set(row["candidate_physical_ids"]), batteries)
            self.assertIn("?", row["display_label"])

    @unittest.skipUnless((CASE / "observations.json").is_file(), "Local source export unavailable")
    def test_native_observations_match_exact_source_pts(self):
        data = json.loads((CASE / "observations.json").read_text())
        indexes = {}
        for frame in data["frames"]:
            for observation in frame["objects"] + frame["withheld"]:
                case_id = observation["source_case"]
                if case_id not in indexes:
                    rows = json.loads((ROOT / "data" / case_id / "frames/index.json").read_text())["frames"]
                    indexes[case_id] = {row["index"]: row for row in rows}
                source = indexes[case_id][observation["source_sample_index"]]
                self.assertAlmostEqual(frame["timestamp_s"], source["timestamp_s"], places=6)
                if "mask_file" in observation:
                    self.assertTrue((CASE / observation["mask_file"]).is_file())
                if observation["physical_object_id"] is None:
                    self.assertIn("battery_track_", observation["id"])
                    self.assertEqual(set(observation["candidate_physical_ids"]), {"battery_00001", "battery_00002"})

    @unittest.skipUnless((CASE / "manifest.json").is_file(), "Local source export unavailable")
    def test_full_source_time_coverage_and_images_are_preserved(self):
        data = json.loads((CASE / "frames/index.json").read_text())["frames"]
        manifest = json.loads((CASE / "manifest.json").read_text())
        times = [frame["timestamp_s"] for frame in data]
        self.assertEqual(times[0], 0)
        self.assertEqual(times, sorted(set(times)))
        self.assertLess(manifest["source"]["video"]["duration_s"] - times[-1], 1.01)
        self.assertEqual([f["index"] for f in data], list(range(1, len(data) + 1)))
        for frame in data:
            digest = hashlib.sha256((CASE / frame["file"]).read_bytes()).digest()
            self.assertIn(digest, [hashlib.sha256((ROOT / row["file"]).read_bytes()).digest() for row in frame["source_samples"]])


if __name__ == "__main__":
    unittest.main()
