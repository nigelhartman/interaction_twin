"""Offline contracts with synthetic responses; no API keys or model requests.

The tiny 5x5 payload is the official parser's published software example, not a
mask inferred from our recordings. All test outputs live in temporary folders.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

import meta_sam_api as sam


PARSER_PRESENT = importlib.util.find_spec("meta_sam_parser") is not None


def source(index=1, pts=12.001056):
    return {"video_sha256": "a" * 64, "sample_index": index,
            "timestamp_s": pts, "time_domain": sam.TIME_DOMAIN}


def line(frame=0, instance=7, confidence=True):
    c = ";c=0.75" if confidence else ""
    return (f"<{frame}f>{instance}<|box;x1=2;y1=3;x2=6;y2=7;w=10;h=12{c}|>"
            f"<|mask;x=0;y=0{c};data=5,5,!!!!!(QO(0lu8?|>")


def response(text=None):
    return {"id": "synthetic-response-fixture", "model": sam.MODEL, "status": "completed",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "output": [{"type": "message", "status": "completed", "content": [
                {"type": "output_text", "text": line() if text is None else text}]}]}


class SamPayloadTests(unittest.TestCase):
    def test_payload_is_single_concept_single_media_nonstreaming(self):
        part = {"type": "input_image", "image_url": "data:image/png;base64,example"}
        payload = sam.build_payload("  earbud  ", part)
        self.assertEqual(payload["model"], "sam-3.1")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["metadata"], {"mask_encoding": "one_bit", "include_confidence": "true"})
        self.assertEqual(payload["input"][0]["content"][0]["text"], "earbud")
        self.assertEqual(payload["input"][0]["content"][1], part)
        self.assertNotIn("api_key", payload)

    def test_invalid_or_multi_concept_prompts_rejected(self):
        for value in ("", "earbud, battery", "battery; camera", "left\nright", "<0f>0", "a" * 161):
            with self.subTest(value=value), self.assertRaises(sam.SamAPIError):
                sam.build_payload(value, {})

    def test_documented_pricing_minimum_and_no_token_pricing(self):
        self.assertEqual(str(sam.estimate_cost("image")), "0.0025")
        self.assertEqual(str(sam.estimate_cost("video", frame_count=1)), "0.0100")
        self.assertEqual(str(sam.estimate_cost("video", frame_count=1000)), "0.2000")
        with self.assertRaises(sam.SamAPIError):
            sam.estimate_cost("image", frame_count=2)
        with self.assertRaises(sam.SamAPIError):
            sam.estimate_cost("video", frame_count=True)

    def test_source_pts_requires_explicit_correct_time_domain(self):
        for changed in ({"timestamp_s": float("nan")}, {"sample_index": True},
                        {"timestamp_s": -1}, {"video_sha256": "x"}, {"time_domain": "frame_index"}):
            with self.subTest(changed=changed), self.assertRaises(sam.SamAPIError):
                sam._source({**source(), **changed})

    def test_missing_parser_stops_before_request_or_output(self):
        callback = mock.Mock()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "out"
            with mock.patch.object(sam, "parser_preflight", side_effect=sam.SamAPIError("missing parser")):
                with self.assertRaises(sam.SamAPIError):
                    sam._run({}, request=callback, output=out, sources=[source()], width=10,
                             height=12, media="image", provenance={})
            callback.assert_not_called()
            self.assertFalse(out.exists())


@unittest.skipUnless(PARSER_PRESENT, "Install official meta-sam-parser==0.1.0 for codec integration contracts")
class SamDecoderTests(unittest.TestCase):
    def test_official_decoder_reconstructs_full_source_pixels_and_pts(self):
        result, masks = sam.decode_response(response(), width=10, height=12, media="image", sources=[source()])
        self.assertEqual(len(masks), 1)
        self.assertEqual(masks[0].shape, (12, 10))
        self.assertEqual(masks[0].dtype, bool)
        self.assertFalse(masks[0][:3].any())
        self.assertFalse(masks[0][:, :2].any())
        item = result["instances"][0]
        self.assertEqual(item["instance_id"], "7")
        self.assertEqual(item["source"], source())
        self.assertEqual(item["confidence"], 0.75)
        self.assertEqual(item["box_xyxy_half_open"], [2, 3, 7, 8])
        self.assertFalse(result["physical_accuracy_measured"])

    def test_zero_matches_valid_missing_usage_stays_unknown(self):
        fixture = response("")
        del fixture["usage"]
        result, masks = sam.decode_response(fixture, width=10, height=12, media="image", sources=[source()])
        self.assertEqual(result["status"], "completed_zero_matches")
        self.assertEqual(masks, [])
        self.assertEqual(result["usage_status"], "unknown_do_not_release_budget_reservation")

    def test_video_sparse_out_of_order_frames_use_explicit_original_pts(self):
        sources = [source(1, 1.00002), source(5, 1.20145), source(9, 1.50001)]
        result, masks = sam.decode_response(response(line(2, 7) + "\n" + line(0, 7)),
                                           width=10, height=12, media="video", sources=sources)
        self.assertEqual([item["frame_index"] for item in result["instances"]], [2, 0])
        self.assertEqual([item["source"]["timestamp_s"] for item in result["instances"]], [1.50001, 1.00002])
        self.assertEqual(len(masks), 2)  # No mask invented for frame 1.

    def test_unknown_video_frame_and_duplicate_object_frame_rejected(self):
        for text in (line(3), line() + "\n" + line()):
            with self.subTest(text=text), self.assertRaises(sam.SamAPIError):
                sam.decode_response(response(text), width=10, height=12, media="video", sources=[source()])

    def test_refusal_failure_wrong_model_and_ambiguous_lanes_rejected(self):
        for fixture in (dict(response(), status="incomplete"), dict(response(), model="sam2"),
                        dict(response(), output=[]), dict(response(), output=[{"type": "message", "content": [
                            {"type": "refusal", "refusal": "fixture"}]}])):
            with self.subTest(fixture=fixture), self.assertRaises(sam.SamAPIError):
                sam.decode_response(fixture, width=10, height=12, media="image", sources=[source()])

    def test_malformed_mask_wrong_source_dimensions_and_no_resize(self):
        for text in (line().replace("w=10", "w=11"), line().replace("x2=6", "x2=7"),
                     line().replace("!!!!!(QO(0lu8?", "!bad"), "this is not a segmentation result"):
            with self.subTest(text=text), self.assertRaises(sam.SamAPIError):
                sam.decode_response(response(text), width=10, height=12, media="image", sources=[source()])

    def test_optional_confidence_is_unknown_not_zero(self):
        result, _ = sam.decode_response(response(line(confidence=False)), width=10, height=12,
                                        media="image", sources=[source()])
        self.assertIsNone(result["instances"][0]["confidence"])

    def test_image_adapter_preserves_source_no_reference_mask_as_prompt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.fromarray(np.zeros((12, 10, 3), dtype=np.uint8)).save(root / "image.png")
            manifest = {"object_id": "component_a", "image": "image.png", "image_sha256": sam.sha256(root / "image.png"),
                        "source": source()}
            (root / "input.json").write_text(json.dumps(manifest))
            callback = mock.Mock(return_value=response())
            result = sam.segment_image(root / "input.json", "earbud", root / "out", callback,
                                       prompt_origin="synthetic fixture concept")
            callback.assert_called_once()
            payload = callback.call_args.args[0]
            self.assertTrue(payload["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,"))
            self.assertFalse(result["input_provenance"]["reference_mask_passed_to_model"])
            self.assertFalse(result["input_provenance"]["inherited_reference_mask_audit_reused"])
            self.assertTrue((root / "out" / "instance-000001-mask.png").is_file())
            saved = json.loads((root / "out" / "metadata.json").read_text())
            self.assertEqual(saved["instances"][0]["source"], source())
            with self.assertRaises(FileExistsError):
                sam.segment_image(root / "input.json", "earbud", root / "out", callback, prompt_origin="fixture")
            callback.assert_called_once()

    def test_failed_transport_is_one_attempt_never_retried(self):
        callback = mock.Mock(side_effect=RuntimeError("synthetic network failure"))
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "synthetic network failure"):
                sam._run(sam.build_payload("earbud", {}), request=callback,
                         output=Path(folder) / "out", sources=[source()], width=10,
                         height=12, media="image", provenance={})
        callback.assert_called_once()

    def test_model_box_prompt_rounding_and_ids_use_official_builder(self):
        text, normalized = sam._model_box_prompt(
            [{"id": 7, "box_xyxy": [1.2, 2.4, 7.2, 8.1]}], width=10, height=12,
            frame_count=1, prompt_origin="meta_api_model_proposal", media="image")
        self.assertEqual(text, "<0f>7<|box;x1=1;y1=2;x2=7;y2=8;w=10;h=12|>")
        self.assertEqual(normalized[0]["model_box_xyxy_half_open"], [1.2, 2.4, 7.2, 8.1])
        self.assertEqual(normalized[0]["rounded_box_xyxy_half_open"], [1, 2, 8, 9])
        self.assertEqual(normalized[0]["wire_box_xyxy_inclusive"], [1, 2, 7, 8])

    def test_model_box_image_sends_no_concept_and_preserves_original_pts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.fromarray(np.zeros((12, 10, 3), dtype=np.uint8)).save(root / "image.png")
            manifest = {"object_id": "component_a", "image": "image.png", "image_sha256": sam.sha256(root / "image.png"),
                        "source": source()}
            (root / "input.json").write_text(json.dumps(manifest))
            callback = mock.Mock(return_value=response())
            result = sam.segment_image_boxes(
                root / "input.json", [{"id": 7, "box_xyxy": [2, 3, 7, 8]}], root / "out", callback,
                prompt_origin="meta_api_model_proposal")
            callback.assert_called_once()
            self.assertEqual(callback.call_args.args[0]["input"][0]["content"][0]["text"],
                             "<0f>7<|box;x1=2;y1=3;x2=6;y2=7;w=10;h=12|>")
            self.assertNotIn("concept", result["request"])
            self.assertEqual(result["request"]["prompt_type"], "model_generated_box_prompts")
            self.assertEqual(result["instances"][0]["instance_id"], "7")
            self.assertEqual(result["instances"][0]["source"], source())
            self.assertFalse(result["input_provenance"]["manual_seed_boxes_used"])

    def test_box_input_rejects_bad_coordinates_ids_provenance_and_duplicate_frames(self):
        invalid = [[], [{"id": True, "box_xyxy": [1, 1, 3, 3]}],
                   [{"id": -1, "box_xyxy": [1, 1, 3, 3]}],
                   [{"id": 1, "box_xyxy": [-0.1, 1, 3, 3]}],
                   [{"id": 1, "box_xyxy": [1, 1, 10.1, 3]}],
                   [{"id": 1, "box_xyxy": [1, 1, float("nan"), 3]}],
                   [{"id": 1, "frame_index": 1, "box_xyxy": [1, 1, 3, 3]}],
                   [{"id": 1, "box_xyxy": [1, 1, 3, 3]}] * 2,
                   [{"id": i, "box_xyxy": [1, 1, 3, 3]} for i in range(17)]]
        for objects in invalid:
            with self.subTest(objects=objects), self.assertRaises(sam.SamAPIError):
                sam._model_box_prompt(objects, width=10, height=12, frame_count=1,
                                      prompt_origin="meta_api_model_proposal", media="image")
        for origin in ("", "assistant_annotation", "manual_model_annotation"):
            with self.subTest(origin=origin), self.assertRaises(sam.SamAPIError):
                sam._model_box_prompt([{"id": 1, "box_xyxy": [1, 1, 3, 3]}], width=10, height=12,
                                      frame_count=1, prompt_origin=origin, media="image")

    def test_box_video_groups_same_id_and_maps_original_pts_without_offset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "fixture.mp4"
            video.write_bytes(b"synthetic-container-placeholder-not-live-video")
            sources = [source(3, 10.1), source(5, 10.7), source(9, 11.3)]
            callback = mock.Mock(return_value=response(line(2, 7) + "\n" + line(0, 7)))
            result = sam.segment_video_boxes(
                video, [{"id": 7, "frame_index": 2, "box_xyxy": [2, 3, 7, 8]},
                        {"id": 7, "frame_index": 0, "box_xyxy": [2, 3, 7, 8]}],
                root / "out", callback, width=10, height=12, frame_sources=sources,
                prompt_origin="meta_api_model_proposal", file_id="file-existing-fixture")
            payload = callback.call_args.args[0]
            self.assertEqual(payload["input"][0]["content"][1], {"type": "input_video", "file_id": "file-existing-fixture"})
            self.assertEqual(payload["metadata"]["propagation_direction"], "both")
            self.assertTrue(payload["input"][0]["content"][0]["text"].startswith("<0f>7"))
            self.assertIn("<2f>7", payload["input"][0]["content"][0]["text"])
            self.assertEqual([r["source"]["timestamp_s"] for r in result["instances"]], [11.3, 10.1])
            self.assertEqual([r["instance_id"] for r in result["instances"]], ["7", "7"])

    def test_box_video_invalid_direction_and_unprompted_return_ids_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "fixture.mp4"
            video.write_bytes(b"synthetic-container-placeholder-not-live-video")
            callback = mock.Mock(return_value=response())
            with self.assertRaisesRegex(sam.SamAPIError, "Backward"):
                sam.segment_video_boxes(video, [{"id": 7, "box_xyxy": [2, 3, 7, 8]}], root / "backward", callback,
                                        width=10, height=12, frame_sources=[source()], prompt_origin="meta_api_model_proposal",
                                        propagation_direction="backward")
            callback.assert_not_called()
            with self.assertRaisesRegex(sam.SamAPIError, "not present"):
                sam.segment_video_boxes(video, [{"id": 8, "box_xyxy": [2, 3, 7, 8]}], root / "wrong-id", callback,
                                        width=10, height=12, frame_sources=[source()], prompt_origin="meta_api_model_proposal")
            callback.assert_called_once()
            self.assertTrue((root / "wrong-id" / "response.json").is_file())
            self.assertFalse((root / "wrong-id" / "metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
