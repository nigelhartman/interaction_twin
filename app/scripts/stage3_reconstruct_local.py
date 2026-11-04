#!/usr/bin/env python3
"""Genuine offline TripoSR comparison on the algorithmic rank-one SAM input.

This is a non-Meta diagnostic, not completion of the SAM 3D Objects milestone.
No manual geometry, pose, camera corners, shape class or texture replacement is
used. Image-to-triplane and density/colour queries execute on Apple MPS; the
explicit CPU marching-cubes export is geometry processing, not model fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def selected_input(selection_path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    selection = read(selection_path)
    if selection.get("status") != "ranked_inputs_not_reconstruction":
        raise ValueError("Expected the automatic view-selector output")
    if selection.get("selection_method", {}).get("agent_visual_choice") is not False:
        raise ValueError("Input choice must be algorithmic, not an agent visual selection")
    candidate = selection["selected"][0]
    if candidate["selection_rank"] != 1:
        raise ValueError("Only the algorithmic first-ranked view is permitted in this comparison")
    files = {}
    for role in ("rgb", "mask"):
        path = (selection_path.parent / candidate["exports"][role]).resolve()
        if sha256(path) != candidate["exports"][f"{role}_sha256"]:
            raise ValueError(f"Selected {role} hash no longer matches the selection")
        files[role] = str(path)
    rgb = np.asarray(Image.open(files["rgb"]).convert("RGB"))
    mask_raw = np.asarray(Image.open(files["mask"]))
    if mask_raw.ndim != 2 or mask_raw.shape != rgb.shape[:2]:
        raise ValueError("RGB/mask must be single, identical full-frame pixel coordinates")
    if not set(np.unique(mask_raw).tolist()).issubset({0, 1, 255}):
        raise ValueError("Selected mask must be binary")
    mask = mask_raw > 0
    if not mask.any() or mask.all():
        raise ValueError("Mask must contain a nonempty object and background")
    return {"selection": selection, "candidate": candidate, "files": files}, rgb, mask


def preprocess(rgb: np.ndarray, mask: np.ndarray, foreground_ratio: float = .85) -> tuple[Image.Image, dict]:
    ys, xs = np.where(mask)
    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    x0, y0, x1, y1 = box
    crop = rgb[y0:y1, x0:x1]
    alpha = mask[y0:y1, x0:x1, None]
    side = math.ceil(max(crop.shape[:2]) / foreground_ratio)
    canvas = np.full((side, side, 3), 128, np.uint8)
    oy, ox = (side-crop.shape[0])//2, (side-crop.shape[1])//2
    canvas[oy:oy+crop.shape[0], ox:ox+crop.shape[1]] = np.where(alpha, crop, 128)
    return Image.fromarray(canvas), {
        "source_bbox_xyxy": box, "foreground_ratio": foreground_ratio,
        "gray_background_rgb": [128, 128, 128], "square_input_side": side,
        "crop_offset_xy_in_model_input": [ox, oy], "source_object_pixels_modified": False,
        "inpainting": False, "missing_occluded_pixels": "gray background, not completed in the input",
        "upstream_preprocessor": "TripoSR ImagePreprocessor resizes the square RGB to checkpoint cond_image_size",
    }


def model_provenance() -> dict:
    checkpoint = ROOT / "models/triposr-checkpoint"
    repo = ROOT / "models/triposr-code"
    provenance = read(checkpoint / "provenance.json")
    actual = sha256(checkpoint / "model.ckpt")
    if actual != provenance["checkpoint_sha256"]:
        raise ValueError("Existing TripoSR checkpoint SHA256 differs from recorded provenance")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != provenance["code_commit"]:
        raise ValueError("Existing TripoSR code differs from recorded pinned commit")
    provenance.update({
        "checkpoint_sha256_verified": True, "code_commit_verified": True,
        "config_sha256": sha256(checkpoint / "config.yaml"),
        "system_py_sha256": sha256(repo / "tsr/system.py"),
        "renderer_py_sha256": sha256(repo / "tsr/models/nerf_renderer.py"),
        "isosurface_py_sha256": sha256(repo / "tsr/models/isosurface.py"),
        "local_patch": "Optional torchmcubes import only; actual density extracted with explicit CPU scikit-image marching cubes.",
        "local_patch_file": "patches/triposr-optional-native-mesher.patch",
        "learned_weights_modified": False, "checkpoint_download_performed": False,
    })
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "data/camera-battery-full-scene/reconstruction-input-v2/selection.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/camera-battery-full-scene/reconstruction-triposr-camera-v1")
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 32 <= args.resolution <= 256:
        parser.error("--resolution must be between 32 and 256")
    if args.output.exists():
        raise FileExistsError("Output exists; use a fresh run name to preserve the evidence")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("MPS operator fallback must remain disabled")
    # No network calls or credential reads are needed: all weights and DINO
    # configuration were provisioned by the earlier local reconstruction test.
    os.environ["HF_HOME"] = str(ROOT / "models/hf-cache")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import trimesh
    from skimage.measure import marching_cubes
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable; refusing to use CPU learned inference")
    torch.manual_seed(args.seed)
    context, rgb, mask = selected_input(args.selection.resolve())
    provenance = model_provenance()
    candidate, selection = context["candidate"], context["selection"]
    model_input, preprocessing = preprocess(rgb, mask)
    out = args.output.resolve()
    out.mkdir(parents=True)
    started = time.monotonic()
    metadata = {
        "schema_version": "0.1", "status": "running", "object_id": selection["object_id"],
        "backend": "TripoSR", "provider": "Stability AI / Tripo AI non-Meta comparison",
        "sam3d_output": False, "sam3d_milestone_complete": False,
        "learned_inference_executed": False, "model_provenance": provenance,
        "runtime": {"device": "mps", "torch": torch.__version__, "cpu_operator_fallback_enabled": False,
                    "offline": True, "recommended_mps_memory_gib": torch.mps.recommended_max_memory()/2**30},
        "source": {"image_path": "source-rgb.png", "mask_path": "source-mask.png",
                   "sample_index": candidate["sample_index"], "timestamp_s": candidate["timestamp_s"],
                   "time_domain": "source_video_presentation_seconds", "video_sha256": selection["source"]["sha256"],
                   "selection_manifest_path": str(args.selection.resolve()), "selection_manifest_sha256": sha256(args.selection),
                   "selection_rank": 1, "selection_score": candidate["score"],
                   "selection_agent_visual_choice": False, "mask_scope": candidate["mask_scope"],
                   "mask_provenance": candidate["mask_provenance"], "width": rgb.shape[1], "height": rgb.shape[0]},
        "preprocessing": preprocessing, "seed": args.seed, "density_grid_resolution": args.resolution,
        "density_isovalue": 25.0, "geometry_reconstructed_by_model": True,
        "manual_geometry_fallback": False, "manual_pose_used": False,
        "appearance": "TripoSR decoder-predicted vertex colours; not UV photography or direct source texture baking",
        "hidden_surfaces": "single-view model inference, not observed or measured",
        "metric_scale_verified": False, "camera_pose_estimated": False,
        "coordinates": "Unchanged native TripoSR normalized canonical asset coordinates; not video camera coordinates",
        "validation_status": "pending_model_run_and_visual_review",
        "limitations": ["Non-Meta comparison; does not replace the planned SAM 3D Objects experiment.",
                        "SAM segmentation and its actual prompt provider are disclosed in source.mask_provenance.",
                        "Only one algorithmically selected view; hidden shape and colours are model predictions.",
                        "The input mask may include attached components or boundary leakage; visual audits are fallible.",
                        "No video pose, metric scale, physical contact or articulation is established."],
    }
    write(out / "metadata.json", metadata)
    try:
        Image.fromarray(rgb).save(out / "source-rgb.png")
        Image.fromarray(mask.astype(np.uint8)*255).save(out / "source-mask.png")
        rgba = Image.fromarray(rgb).convert("RGBA")
        rgba.putalpha(Image.fromarray(mask.astype(np.uint8)*255))
        rgba.save(out / "source-rgba.png")
        model_input.save(out / "model-input.png")
        shutil.copy2(args.selection, out / "selection-source.json")
        metadata["source"]["image_sha256"] = sha256(out / "source-rgb.png")
        metadata["source"]["mask_sha256"] = sha256(out / "source-mask.png")
        metadata["preprocessing"]["model_input_sha256"] = sha256(out / "model-input.png")
        write(out / "metadata.json", metadata)
        sys.path.insert(0, str(ROOT / "models/triposr-code"))
        from tsr.system import TSR
        load_started = time.monotonic()
        model = TSR.from_pretrained(str(ROOT / "models/triposr-checkpoint"), "config.yaml", "model.ckpt").to("mps").eval()
        model.renderer.set_chunk_size(8192)
        if any(p.device.type != "mps" for p in model.parameters()):
            raise RuntimeError("Model parameters did not all load on MPS")
        torch.mps.synchronize()
        metadata["model_load_seconds"] = time.monotonic() - load_started
        inference_started = time.monotonic()
        with torch.inference_mode():
            scene = model([model_input], device="mps")
            torch.mps.synchronize()
        if scene.device.type != "mps":
            raise RuntimeError("Learned scene was not generated on MPS")
        metadata["learned_inference_executed"] = True
        metadata["image_to_triplane_seconds"] = time.monotonic() - inference_started
        np.save(out / "learned-triplane.npy", scene.detach().cpu().numpy())
        print(f"Actual TripoSR triplane generated on MPS in {metadata['image_to_triplane_seconds']:.3f}s", flush=True)
        density_started = time.monotonic()
        radius = float(model.renderer.cfg.radius)
        axis = torch.linspace(-radius, radius, args.resolution)
        grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
        density = []
        with torch.inference_mode():
            for block in grid.split(8192):
                prediction = model.renderer.query_triplane(model.decoder, block.to("mps"), scene[0])
                density.append(prediction["density_act"][:, 0].cpu())
        volume = torch.cat(density).reshape((args.resolution,)*3).numpy()
        np.save(out / "learned-density.npy", volume)
        metadata["density_query_seconds"] = time.monotonic() - density_started
        metadata["learned_density_range"] = [float(volume.min()), float(volume.max())]
        mesh_started = time.monotonic()
        vertices, faces, normals, values = marching_cubes(volume, level=25.0, spacing=(2*radius/(args.resolution-1),)*3)
        vertices -= radius
        colour_started = time.monotonic()
        colors = []
        with torch.inference_mode():
            for block in torch.from_numpy(vertices.copy()).float().split(8192):
                colors.append(model.renderer.query_triplane(model.decoder, block.to("mps"), scene[0])["color"].cpu().numpy())
        colors = np.concatenate(colors)
        metadata["colour_query_seconds"] = time.monotonic() - colour_started
        np.savez_compressed(out / "learned-surface.npz", vertices=vertices, faces=faces, vertex_colors_float=colors,
                            marching_cubes_normals=normals, marching_cubes_values=values)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces,
                               vertex_colors=(np.clip(colors, 0, 1)*255).astype(np.uint8), process=False)
        for extension in ("glb", "obj", "ply"):
            mesh.export(out / f"asset.{extension}")
        components = mesh.split(only_watertight=False)
        largest = max(components, key=lambda component: component.area)
        largest.export(out / "asset-largest-component.glb")
        metadata["geometry_export_seconds"] = time.monotonic() - mesh_started
        metadata["geometry"] = {
            "vertices": len(mesh.vertices), "faces": len(mesh.faces), "connected_components": len(components),
            "watertight": bool(mesh.is_watertight), "winding_consistent": bool(mesh.is_winding_consistent),
            "bounds": mesh.bounds.tolist(), "extents": mesh.extents.tolist(), "all_components_preserved": True,
            "mesh_export_processing": "unprocessed original marching-cubes vertices/faces; no geometric repair or primitive replacement",
            "largest_component_derivative": {"vertices": len(largest.vertices), "faces": len(largest.faces),
                                             "fraction_of_raw_area": float(largest.area / mesh.area)},
        }
        metadata["outputs"] = {
            "glb_path": "asset.glb", "obj_path": "asset.obj", "ply_path": "asset.ply",
            "raw_learned_density_path": "learned-density.npy", "raw_learned_triplane_path": "learned-triplane.npy",
            "raw_learned_surface_path": "learned-surface.npz", "largest_component_glb_path": "asset-largest-component.glb",
            "vertex_colour_source": "learned TripoSR decoder, not direct original-pixel UV texture",
        }
        outputs = []
        for path in sorted(out.iterdir()):
            if path.is_file() and path.name != "metadata.json":
                outputs.append({"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
        metadata["artifact_files"] = outputs
        metadata.update(status="candidate_generated_pending_visual_review", validation_status="unreviewed_model_candidate",
                        wall_seconds=time.monotonic()-started)
        write(out / "metadata.json", metadata)
        print(json.dumps({"metadata": str(out / "metadata.json"), "status": metadata["status"],
                          "geometry": metadata["geometry"], "wall_seconds": metadata["wall_seconds"]}, indent=2))
    except Exception as error:
        metadata.update(status="failed_no_fallback", error=f"{type(error).__name__}: {error}", wall_seconds=time.monotonic()-started)
        write(out / "metadata.json", metadata)
        raise


if __name__ == "__main__":
    main()
