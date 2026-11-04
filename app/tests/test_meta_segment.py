"""Offline segmentation-wrapper contracts. Synthetic masks, no hosted calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from scripts import stage2_meta_segment as wrapper


def rectangle(x=3, y=4):
    mask = np.zeros((30, 40), dtype=bool)
    mask[y:y + 8, x:x + 8] = True
    return mask


class FakeClient:
    def __init__(self):
        self.requests = []

    def sam_callback(self, output, *, images=0, video_frames=0):
        self.requests.append({"output": Path(output), "images": images, "video_frames": video_frames})
        return lambda payload: {"synthetic_transport_fixture": True}


class MetaSegmentContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.case = Path(self.temp.name) / "fixture-case"
        (self.case / "frames").mkdir(parents=True)
        self.rows = []
        for index in range(1, 5):
            path = self.case / "frames" / f"{index:06d}.jpg"
            Image.fromarray(np.zeros((30, 40, 3), dtype=np.uint8)).save(path)
            self.rows.append({"index": index, "file": str(path.relative_to(self.case)),
                              "timestamp_s": (index - 1) * 1.000083, "width": 40, "height": 30})
        wrapper.write(self.case / "frames/index.json", {"time_domain": wrapper.sam.TIME_DOMAIN, "frames": self.rows})
        wrapper.write(self.case / "manifest.json", {"source": {"sha256": "a" * 64}})
        self.track = {"id": "part_a", "label": "earbud", "description": "A model-described component",
                      "color": "#428DE0", "sample_range": [1, 4], "candidate_entity_ids": ["part_a_candidate"],
                      "identity_status": "model_proposed_tracklet_physical_identity_unverified",
                      "prompts": [{"sample_index": 2, "box_xyxy": [3, 4, 11, 12],
                                   "origin": "Meta_API_model_proposal", "source_timestamp_s": 1.000083}],
                      "withhold_ranges": [], "uncertainties": ["Software fixture, not real identity"]}
        self.cfg = {"schema_version": "0.3", "seed_origin": "Meta_API_model_proposal",
                    "manual_seed_boxes_used": False, "manual_geometry_used": False,
                    "human_confirmed": False, "tracks": [self.track]}
        self.config = self.case / "model-config.json"
        self.save_config()
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def save_config(self):
        self.config.write_text(json.dumps(self.cfg, separators=(",", ":")) + "\n")

    def segmenter(self, detections=None, fail_index=None):
        def fake(input_path, phrase, output, request, *, prompt_origin):
            manifest = wrapper.load(input_path)
            index = manifest["source"]["sample_index"]
            self.calls.append((index, phrase, output, manifest, prompt_origin))
            request({"fixture_payload": True})
            if index == fail_index:
                raise RuntimeError("synthetic transport failure")
            masks = [rectangle()] if detections is None else detections[index]
            output.mkdir()
            records = []
            for number, mask in enumerate(masks):
                path = output / f"instance-{number:06d}.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(path)
                records.append({"instance_id": str(number), "mask": path.name, "confidence": 0.8,
                                "source": manifest["source"]})
            result = {"instances": records, "synthetic_fixture": True}
            wrapper.write(output / "metadata.json", result)
            return result
        return fake

    def test_full_standard_export_exact_pts_all_native_instances_and_config_bytes(self):
        client = FakeClient()
        metadata = wrapper.run_case(self.case, self.config, "sam-api-test", client=client,
                                    segmenter=self.segmenter())
        out = self.case / "segmentation/sam-api-test"
        self.assertEqual((out / "annotations.json").read_bytes(), self.config.read_bytes())
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["model"], "sam-3.1")
        self.assertEqual(metadata["device"], "hosted_Meta_Model_API")
        self.assertIsNone(metadata["checkpoint_sha256"])
        self.assertFalse(metadata["sam_video_propagation_executed"])
        self.assertFalse(metadata["old_sam2_audits_reused"])
        self.assertEqual(metadata["candidate_visible_count"], 4)
        self.assertEqual(metadata["hosted_image_request_count"], 4)
        frames = wrapper.load(out / "tracks.json")["tracks"][0]["frames"]
        self.assertEqual([f["timestamp_s"] for f in frames], [r["timestamp_s"] for r in self.rows])
        self.assertTrue(all(f["status"] == "candidate_visible" for f in frames))
        for frame in frames:
            mask = np.asarray(Image.open(out / frame["raw_mask_file"])) > 0
            self.assertTrue(np.array_equal(mask, rectangle()))
            self.assertEqual(frame["raw_area_pixels"], 64)
            self.assertEqual(frame["raw_bbox_xyxy"], [3, 4, 11, 12])
            self.assertEqual(frame["api_instance"]["instance_id_scope"], "independent_image_response_only_not_temporal_identity")
            self.assertTrue((out / frame["api_instance"]["mask_file"]).is_file())
        for request, call in zip(client.requests, self.calls):
            self.assertEqual(request["images"], 1)
            self.assertNotEqual(request["output"], call[2])
            self.assertEqual(request["output"].name, "accounting")
            self.assertEqual(call[2].name, "result")

    def test_ambiguous_association_is_withheld_and_no_gap_reacquisition(self):
        detections = {1: [rectangle()], 2: [rectangle()], 3: [rectangle(), rectangle()], 4: [rectangle()]}
        metadata = wrapper.run_case(self.case, self.config, "ambiguous-test", client=FakeClient(),
                                    segmenter=self.segmenter(detections))
        self.assertEqual(metadata["native_instance_count"], 5)
        out = self.case / "segmentation/ambiguous-test"
        frames = wrapper.load(out / "tracks.json")["tracks"][0]["frames"]
        self.assertTrue(frames[2]["association"]["ambiguous"])
        self.assertEqual(frames[2]["status"], "withheld_due_to_occlusion_or_ambiguity")
        self.assertTrue(frames[3]["association"]["no_adjacent_supported_association"])
        self.assertEqual(frames[3]["status"], "withheld_due_to_occlusion_or_ambiguity")
        self.assertEqual(frames[2]["raw_area_pixels"], 64)  # Candidate retained, never promoted.

    def test_seed_box_selects_spatial_component_without_manual_object_hints(self):
        candidates = [{"mask": rectangle(22, 18), "record": {"confidence": 0.99}},
                      {"mask": rectangle(), "record": {"confidence": 0.1}}]
        selected, association = wrapper.choose(candidates, seeds=[[3, 4, 11, 12]])
        self.assertIs(selected, candidates[1])
        self.assertTrue(association["accepted"])
        self.assertFalse(association["physical_identity_verified"])

    def test_shared_concept_frame_cache_does_not_merge_tracklet_identity(self):
        other = copy.deepcopy(self.track)
        other.update(id="part_b", candidate_entity_ids=["part_b_candidate"])
        self.cfg["tracks"].append(other)
        self.save_config()
        metadata = wrapper.run_case(self.case, self.config, "cache-test", client=FakeClient(), segmenter=self.segmenter())
        self.assertEqual(metadata["hosted_image_request_count"], 4)
        tracks = wrapper.load(self.case / "segmentation/cache-test/tracks.json")["tracks"]
        self.assertEqual([t["candidate_entity_ids"] for t in tracks], [["part_a_candidate"], ["part_b_candidate"]])
        self.assertTrue(all(t["physical_identity_verified"] is False for t in tracks))

    def test_zero_matches_has_zero_mask_no_invented_object(self):
        wrapper.run_case(self.case, self.config, "empty-test", client=FakeClient(),
                         segmenter=self.segmenter({i: [] for i in range(1, 5)}))
        out = self.case / "segmentation/empty-test"
        frames = wrapper.load(out / "tracks.json")["tracks"][0]["frames"]
        self.assertTrue(all(f["status"] == "no_mask" and f["raw_area_pixels"] == 0 for f in frames))
        self.assertTrue(all(f["api_instance"] is None for f in frames))

    def test_manual_seeds_bad_pts_and_source_escapes_fail_before_requests(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["manual_seed_boxes_used"] = True
        wrapper.write(self.config, cfg)
        with self.assertRaisesRegex(ValueError, "automatic"):
            wrapper.run_case(self.case, self.config, "bad-manual", client=FakeClient(), segmenter=self.segmenter())
        self.assertEqual(self.calls, [])
        self.save_config()
        self.cfg["tracks"][0]["prompts"][0]["source_timestamp_s"] = 50.0
        self.save_config()
        with self.assertRaisesRegex(ValueError, "PTS"):
            wrapper.run_case(self.case, self.config, "bad-pts", client=FakeClient(), segmenter=self.segmenter())
        self.assertEqual(self.calls, [])

    def test_failed_attempt_preserves_failed_metadata_without_retry(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic transport failure"):
            wrapper.run_case(self.case, self.config, "failure-test", client=FakeClient(),
                             segmenter=self.segmenter(fail_index=2))
        out = self.case / "segmentation/failure-test"
        metadata = wrapper.load(out / "metadata.json")
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(metadata["hosted_image_request_count"], 2)
        self.assertEqual([item[0] for item in self.calls], [1, 2])
        self.assertFalse((out / "tracks.json").exists())
        with self.assertRaises(FileExistsError):
            wrapper.run_case(self.case, self.config, "failure-test", client=FakeClient(), segmenter=self.segmenter())

    def test_explicit_withholding_keeps_raw_pixels(self):
        self.track["withhold_ranges"] = [[3, 3]]
        self.save_config()
        wrapper.run_case(self.case, self.config, "withhold-test", client=FakeClient(), segmenter=self.segmenter())
        frame = wrapper.load(self.case / "segmentation/withhold-test/tracks.json")["tracks"][0]["frames"][2]
        self.assertEqual(frame["status"], "withheld_due_to_occlusion_or_ambiguity")
        self.assertEqual(frame["raw_area_pixels"], 64)

    def test_zero_tracks_completes_without_client_parser_or_model_calls(self):
        self.cfg["tracks"] = []
        self.save_config()
        with mock.patch("meta_api.MetaClient", side_effect=AssertionError("No client for zero tracks")) as client:
            with mock.patch.object(wrapper.sam, "parser_preflight", side_effect=AssertionError("No parser for zero tracks")) as parser:
                with mock.patch.object(wrapper.sam, "segment_image", side_effect=AssertionError("No inference for zero tracks")) as segmenter:
                    metadata = wrapper.run_case(self.case, self.config, "zero-tracks")
        client.assert_not_called()
        parser.assert_not_called()
        segmenter.assert_not_called()
        out = self.case / "segmentation/zero-tracks"
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["outcome"], "completed_zero_model_proposed_tracks")
        self.assertFalse(metadata["learned_inference_executed"])
        self.assertFalse(metadata["model_call_made"])
        self.assertEqual(metadata["hosted_image_request_count"], 0)
        self.assertEqual(metadata["estimated_cost_usd"], "0.000000")
        self.assertEqual(wrapper.load(out / "tracks.json")["tracks"], [])
        self.assertEqual((out / "annotations.json").read_bytes(), self.config.read_bytes())
        self.assertFalse((out / "api").exists())


if __name__ == "__main__":
    unittest.main()
