"""Known-answer convention tests; no checkpoints, CUDA, or inference needed."""
import ast
import hashlib
import math
import os
from pathlib import Path
import typing
import unittest
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from geometry import project_points
from sam3d_coordinates import (
    CAMERA_AXES, DECODED_SCALE_SEMANTICS, PYTORCH3D_TO_CV,
    RAW_TO_GLB_ROW, SAM3D_COMMIT, native_layout_to_cv,
    quaternion_wxyz_to_matrix, transform_glb_points,
)


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "models/sam-3d-objects"


def convert(q=(1., 0., 0., 0.), t=(0., 0., 0.), s=(1., 1., 1.), **overrides):
    parameters = dict(rotation_wxyz=q, translation=t, scale_xyz=s,
                      source_commit=SAM3D_COMMIT,
                      scale_semantics=DECODED_SCALE_SEMANTICS)
    parameters.update(overrides)
    return native_layout_to_cv(**parameters)


class SAM3DCoordinateTests(unittest.TestCase):
    def test_identity_layout_has_explicit_glb_basis_and_cv_signs(self):
        # Raw +X -> GLB +X -> CV -X; raw +Y -> GLB -Z -> CV -Y;
        # raw +Z -> GLB +Y -> CV +Z. This is not an identity GLB pose.
        raw = np.eye(3)
        glb = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        result = convert(t=(.1, -.2, 3.))
        np.testing.assert_array_equal(raw @ RAW_TO_GLB_ROW, glb)
        np.testing.assert_allclose(transform_glb_points(glb, result.camera_from_glb),
                                   [[-1.1, .2, 3.], [-.1, -.8, 3.], [-.1, .2, 4.]])

    def test_positive_quaternion_uses_official_row_rotation_direction(self):
        # Conventional +90-degree column quaternion maps +X to +Y.
        # Official row application maps raw +X to -Y, then CV flips to +Y.
        q = [math.sqrt(.5), 0., 0., math.sqrt(.5)]
        rq = quaternion_wxyz_to_matrix(q)
        np.testing.assert_allclose(rq @ [1., 0., 0.], [0., 1., 0.], atol=1e-14)
        result = convert(q=q)
        np.testing.assert_allclose(transform_glb_points([[1., 0., 0.]], result.camera_from_glb),
                                   [[0., 1., 0.]], atol=1e-14)

    def test_random_glb_similarity_matches_independent_scipy_and_pixel_projection(self):
        rng = np.random.default_rng(4709)
        raw = rng.uniform(-.4, .4, (60, 3))
        k = np.array([[730., 0., 479.5], [0., 710., 269.5], [0., 0., 1.]])
        for _ in range(32):
            scipy_rotation = Rotation.random(random_state=rng)
            q_xyzw = scipy_rotation.as_quat()
            q_wxyz = q_xyzw[[3, 0, 1, 2]]
            s = float(rng.uniform(.05, 1.2))
            t = [rng.uniform(-.4, .4), rng.uniform(-.4, .4), 3.]
            # Inverse column action equals the official row action by R(q).
            official_p3d = scipy_rotation.inv().apply(raw * s) + t
            expected_cv = official_p3d * [-1., -1., 1.]
            result = convert(q=q_wxyz, t=t, s=[s]*3)
            actual_cv = transform_glb_points(raw @ RAW_TO_GLB_ROW, result.camera_from_glb)
            np.testing.assert_allclose(actual_cv, expected_cv, atol=2e-15)
            np.testing.assert_allclose(project_points(actual_cv, k), project_points(expected_cv, k), atol=2e-12)
            np.testing.assert_allclose(result.rotation_cv_from_glb.T @ result.rotation_cv_from_glb, np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(result.rotation_cv_from_glb), 1.)

    def test_similarity_contains_scale_once_and_round_trips(self):
        raw = np.array([[.2, -.4, .8], [-.3, .5, .7]])
        glb = raw @ RAW_TO_GLB_ROW
        result = convert(q=[.8, .2, -.1, .4], t=[.1, -.2, 2.], s=[.3]*3)
        np.testing.assert_allclose(np.linalg.norm(result.camera_from_glb[:3, :3], axis=0), .3)
        transformed = transform_glb_points(glb, result.camera_from_glb)
        recovered = transform_glb_points(transformed, np.linalg.inv(result.camera_from_glb))
        np.testing.assert_allclose(recovered, glb, atol=2e-15)
        np.testing.assert_array_equal(result.translation_cv, [-.1, .2, 2.])

    def test_sign_and_nonunit_quaternions_do_not_change_conversion(self):
        q = np.array([.8, .2, -.1, .4])
        expected = convert(q=q).camera_from_glb
        for multiple in (-1., 3.5, 1e200, 1e-200):
            np.testing.assert_allclose(convert(q=q*multiple).camera_from_glb, expected, atol=1e-14)
        np.testing.assert_allclose(convert(q=[q], t=[[0., 0., 0.]], s=[[1., 1., 1.]]).camera_from_glb, expected)

    def test_ambiguous_anisotropic_nonpositive_and_nonfinite_scales_fail(self):
        for invalid in (1., [1.], [[1.]], [1., 2., 1.], [0., 0., 0.], [-1., -1., -1.],
                        [np.nan]*3, [np.inf]*3, np.ones((2, 3))):
            with self.subTest(scale=str(invalid)), self.assertRaises(ValueError):
                convert(s=invalid)

    def test_unaudited_source_and_raw_scale_semantics_fail(self):
        with self.assertRaisesRegex(ValueError, "Unaudited"):
            convert(source_commit="main")
        for invalid in (None, "log_scale", "Gaussian ellipsoid scale", ""):
            with self.assertRaisesRegex(ValueError, "decoded linear"):
                convert(scale_semantics=invalid)

    def test_invalid_quaternion_translation_and_point_inputs_fail(self):
        for invalid in ([0., 0., 0., 0.], [1., 0., 0.], [1., np.nan, 0., 0.], np.ones((2, 4))):
            with self.assertRaises(ValueError):
                convert(q=invalid)
        for invalid in ([1., 2.], np.ones((2, 3)), [0., 0., np.inf]):
            with self.assertRaises(ValueError):
                convert(t=invalid)
        with self.assertRaises(ValueError):
            transform_glb_points([[1., np.nan, 0.]], np.eye(4))
        wrong = np.eye(4); wrong[3, 0] = 1.
        with self.assertRaises(ValueError):
            transform_glb_points([[1., 0., 0.]], wrong)

    def test_metadata_does_not_promote_synthetic_math_to_a_real_pose(self):
        record = convert().metadata()
        self.assertEqual(record["camera_axes"], CAMERA_AXES)
        self.assertFalse(record["metric_scale_verified"])
        self.assertFalse(record["source_video_pose_validated"])
        self.assertFalse(record["real_sam3d_inference_validated"])
        self.assertNotIn("convention_verified", record)
        self.assertEqual(record["scale_units"], "native_model_predicted_scene_units")

    @unittest.skipUnless(REPO.is_dir(), "Optional pinned SAM 3D source checkout absent")
    def test_glb_axis_expression_matches_actual_pinned_exporter(self):
        path = REPO / "sam3d_objects/model/backbone/tdfy_dit/utils/postprocessing_utils.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "7fdd03ebb0ac45ee41e0a35408d7e068b568793e677de4e485af0f9faf812b72")
        function = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "to_glb")
        expression = next(node.value for node in function.body
                          if isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp)
                          and isinstance(node.value.op, ast.MatMult))
        raw = np.arange(18.).reshape(6, 3)
        official_glb = eval(compile(ast.Expression(expression), str(path), "eval"), {"np": np, "vertices": raw})
        np.testing.assert_array_equal(official_glb, raw @ RAW_TO_GLB_ROW)

    @unittest.skipUnless(os.environ.get("SAM3D_PYTORCH3D_REFERENCE_DIR"),
                         "Optional pinned PyTorch3D math sources not provided")
    def test_equivalence_with_actual_pinned_torch_helpers_on_cpu(self):
        # Execute genuine pure-tensor source after stripping imports. Only unused
        # device/compiled-op dependencies are replaced; transformation classes,
        # quaternion conversion and SAM3D compose_transform are the real code.
        import torch
        directory = Path(os.environ["SAM3D_PYTORCH3D_REFERENCE_DIR"])
        namespace = dict(torch=torch, math=math, warnings=warnings, os=os,
                         Optional=typing.Optional, Union=typing.Union, List=typing.List,
                         Device=typing.Union[str, torch.device],
                         make_device=lambda device: torch.device(device),
                         get_device=lambda value, device=None: torch.device(device or "cpu"),
                         _safe_det_3x3=torch.linalg.det)
        hashes = {"rotation_conversions.py": "b819672f569e489dbe73e9ff79743822f6199ff799d07399a4624c9fc0537dc0",
                  "transform3d.py": "b8fdd5c9aa5dcc89b41994984472d6117cc9ccf49f8b225334197f2a373dfd84"}
        for filename, expected_hash in hashes.items():
            path = directory / filename
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected_hash)
            tree = ast.parse(path.read_text())
            tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
            exec(compile(tree, str(path), "exec"), namespace)
        helper = REPO / "sam3d_objects/data/dataset/tdfy/transforms_3d.py"
        self.assertEqual(hashlib.sha256(helper.read_bytes()).hexdigest(),
                         "496d4e39974c188bd56464d4d303b22fe3e20c6d544563b67e365bf6f9cba6fb")
        tree = ast.parse(helper.read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "compose_transform"]
        exec(compile(tree, str(helper), "exec"), namespace)
        rng = np.random.default_rng(554)
        raw = rng.uniform(-1., 1., (80, 3))
        for _ in range(40):
            q = rng.normal(size=4)
            t = rng.uniform(-.5, .5, 3); t[2] += 4.
            s = np.repeat(rng.uniform(.01, 1.5), 3)
            tq, tt, ts = (torch.tensor(value[None], dtype=torch.float64) for value in (q, t, s))
            rq = namespace["quaternion_to_matrix"](tq)
            official = namespace["compose_transform"](ts, rq, tt)
            expected_p3d = official.transform_points(torch.tensor(raw[None], dtype=torch.float64))[0].numpy()
            result = convert(q=q, t=t, s=s)
            actual = transform_glb_points(raw @ RAW_TO_GLB_ROW, result.camera_from_glb)
            np.testing.assert_allclose(actual, expected_p3d @ PYTORCH3D_TO_CV, atol=3e-15)


if __name__ == "__main__":
    unittest.main()
