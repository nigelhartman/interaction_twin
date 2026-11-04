"""Data provenance and fail-closed SAM3D export tests, without model inference."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from scripts.stage3_sam3d import (
    TIME_DOMAIN, checkpoint_provenance, export_model_outputs,
    preflight, read_inputs, sha256,
)


class ReconstructionContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.image = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
        self.mask = np.zeros((8, 10), np.uint8)
        self.mask[2:7, 3:8] = 255
        Image.fromarray(self.image).save(self.directory / "rgb.png")
        Image.fromarray(self.mask).save(self.directory / "mask.png")
        self.manifest = {
            "schema_version": "0.1", "object_id": "battery_00001",
            "image": "rgb.png", "mask": "mask.png",
            "source": {"video_sha256": "a" * 64, "sample_index": 16,
                       "timestamp_s": 40.006633, "time_domain": TIME_DOMAIN},
            "mask_provenance": {"backend": "SAM 2.1 Tiny",
                                "manual_intervention": True,
                                "prompt_origin": "assistant annotated test fixture"},
        }
        self.path = self.directory / "input.json"
        self.save_manifest()

    def tearDown(self):
        self.temporary.cleanup()

    def save_manifest(self):
        self.path.write_text(json.dumps(self.manifest))

    def test_rgb_pts_hashes_and_assistance_are_preserved(self):
        record, image, mask = read_inputs(self.path)
        np.testing.assert_array_equal(image, self.image)
        np.testing.assert_array_equal(mask, self.mask > 0)
        self.assertEqual(record["manifest"]["source"]["timestamp_s"], 40.006633)
        self.assertTrue(record["manifest"]["mask_provenance"]["manual_intervention"])
        self.assertEqual(record["files"]["image"]["sha256"], sha256(self.directory / "rgb.png"))

    def test_assistance_must_be_explicit_and_binary(self):
        for invalid in (None, "false", 0):
            self.manifest["mask_provenance"]["manual_intervention"] = invalid
            self.save_manifest()
            with self.assertRaisesRegex(ValueError, "explicitly true or false"):
                read_inputs(self.path)

    def test_rejects_resized_soft_or_empty_masks(self):
        for invalid in (np.zeros((8, 9), np.uint8), np.full((8, 10), 127, np.uint8), np.zeros((8, 10), np.uint8)):
            Image.fromarray(invalid).save(self.directory / "mask.png")
            with self.assertRaises(ValueError):
                read_inputs(self.path)

    def test_hash_mismatch_and_invalid_timestamps_are_rejected(self):
        self.manifest["image_sha256"] = "0" * 64
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "SHA256"):
            read_inputs(self.path)
        del self.manifest["image_sha256"]
        for invalid in (True, float("nan"), -1, "40.006633"):
            self.manifest["source"]["timestamp_s"] = invalid
            self.save_manifest()
            with self.assertRaisesRegex(ValueError, "original video PTS"):
                read_inputs(self.path)

    def test_missing_runtime_stays_preflight_without_model_result(self):
        args = argparse.Namespace(command="preflight", input_manifest=self.path,
                                  repo=self.directory / "absent-repo", config=None,
                                  expected_code_commit=None, device="cuda")
        report, image, mask = preflight(args)
        self.assertEqual(report["status"], "prerequisites_missing")
        self.assertFalse(report["learned_inference_executed"])
        self.assertNotIn("outputs", report)
        self.assertTrue(any("checkpoint" in text for text in report["errors"]))
        self.assertIsNotNone(image)

    def test_checkpoint_inventory_hashes_local_references_and_rejects_missing(self):
        config = self.directory / "pipeline.yaml"
        config.write_text("ss_generator_ckpt_path: real.safetensors\nss_generator_config_path: generator.yaml\n")
        (self.directory / "generator.yaml").write_text("width: 64\n")
        (self.directory / "real.safetensors").write_bytes(b"synthetic inventory fixture, not model weights")
        record, errors = checkpoint_provenance(config, hash_weights=True)
        self.assertEqual(errors, [])
        self.assertEqual(record["checkpoint_count"], 1)
        weights = next(row for row in record["files"] if row["path"].endswith("real.safetensors"))
        self.assertEqual(weights["sha256"], sha256(self.directory / "real.safetensors"))
        (self.directory / "real.safetensors").unlink()
        _, errors = checkpoint_provenance(config)
        self.assertTrue(any("missing" in text for text in errors))

    def test_safe_workspace_prefix_resolves_without_running_hydra_resolvers(self):
        config = self.directory / "pipeline.yaml"
        config.write_text('ss_generator_ckpt_path: ${workspace_dir}/real.safetensors\nss_generator_config_path: ${workspace_dir}/generator.yaml\n')
        (self.directory / "real.safetensors").write_bytes(b"synthetic weight inventory fixture")
        (self.directory / "generator.yaml").write_text("width: 64\n")
        record, errors = checkpoint_provenance(config)
        self.assertEqual(errors, [])
        self.assertEqual(record["checkpoint_count"], 1)
        config.write_text('ss_generator_ckpt_path: ${oc.env:HOME}/secret.ckpt\n')
        _, errors = checkpoint_provenance(config)
        self.assertTrue(any("only a literal ${workspace_dir}" in error for error in errors))

    def test_missing_model_outputs_never_create_primitive_fallback(self):
        for output in ({}, {"glb": object()}, {"gs": object()}):
            with self.assertRaisesRegex(RuntimeError, "no fallback"):
                export_model_outputs(output, self.directory)
        self.assertFalse((self.directory / "object.glb").exists())

    def test_model_export_preserves_native_layout_and_discloses_vertex_colour(self):
        # Deliberate lightweight export doubles test the data boundary, not SAM3D quality.
        class MeshDouble:
            def export(self, path):
                Path(path).write_bytes(b"synthetic mesh export test fixture")
        class SplatDouble:
            def save_ply(self, path):
                Path(path).write_bytes(b"synthetic splat export test fixture")
        result = {"glb": MeshDouble(), "gs": SplatDouble(),
                  "rotation": np.array([[1., 0, 0, 0]]),
                  "translation": np.array([[.1, -.2, 1.4]]),
                  "scale": np.array([[.3, .3, .3]])}
        record = export_model_outputs(result, self.directory)
        saved = np.load(self.directory / "layout-and-pointmap.npz")
        np.testing.assert_array_equal(saved["translation"], result["translation"])
        self.assertIn("UV texture baking disabled", record["appearance"])
        self.assertFalse(record["manual_geometry_fallback"])
        self.assertFalse(record["metric_scale_verified"])
        self.assertEqual(len(record["files"]), 3)

    def test_uv_mode_requires_a_real_texture_and_does_not_claim_video_projection(self):
        class MeshDouble:
            visual = SimpleNamespace(kind="vertex", uv=None, material=None)
            def export(self, path):
                Path(path).write_bytes(b"synthetic textured export test fixture")
        class SplatDouble:
            def save_ply(self, path):
                Path(path).write_bytes(b"synthetic splat export test fixture")
        mesh = MeshDouble()
        result = {"glb": mesh, "gs": SplatDouble(), "rotation": np.array([[1., 0, 0, 0]]),
                  "translation": np.zeros((1, 3)), "scale": np.ones((1, 3))}
        with self.assertRaisesRegex(RuntimeError, "no UVs"):
            export_model_outputs(result, self.directory, "uv-baked")
        mesh.visual = SimpleNamespace(kind="texture", uv=np.zeros((3, 2)),
                                      material=SimpleNamespace(baseColorTexture=Image.new("RGB", (2, 2), "red")))
        record = export_model_outputs(result, self.directory, "uv-baked")
        self.assertEqual(record["appearance_mode"], "uv-baked")
        self.assertFalse(record["direct_original_video_texture_projection"])
        self.assertIn("100 synthetic views", record["appearance"])


if __name__ == "__main__":
    unittest.main()
