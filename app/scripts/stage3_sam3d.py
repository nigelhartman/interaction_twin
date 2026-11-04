#!/usr/bin/env python3
"""Pinned, fail-closed adapter for official SAM 3D Objects single-object inference.

`preflight` never imports the upstream repository, downloads checkpoints, reads
credentials or runs learned inference. `run` requires a clean official checkout
at an explicitly supplied commit and an already provisioned CUDA environment.
Inputs are RGB plus the visible-object mask, not a hand-built completion. The
official notebook API exports a vertex-coloured GLB, Gaussian splat and layout;
it does not bake a photographic UV texture in this configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

OFFICIAL_REPO = "https://github.com/facebookresearch/sam-3d-objects"
OFFICIAL_MODEL = "facebook/sam-3d-objects"
TIME_DOMAIN = "source_video_presentation_seconds"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
WEIGHT_SUFFIXES = {".ckpt", ".pt", ".pth", ".safetensors", ".bin"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_inputs(manifest_path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Validate timestamp/assistance provenance and preserve full-frame pixels."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Input manifest must be an object")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,95}", str(manifest.get("object_id", ""))):
        raise ValueError("object_id must be a readable stable candidate ID")
    source = manifest.get("source", {})
    if source.get("time_domain") != TIME_DOMAIN:
        raise ValueError(f"source.time_domain must be {TIME_DOMAIN}")
    timestamp = source.get("timestamp_s")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("source.timestamp_s must be the finite original video PTS")
    index = source.get("sample_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 1:
        raise ValueError("source.sample_index must be a positive integer")
    if not SHA256.fullmatch(str(source.get("video_sha256", ""))):
        raise ValueError("source.video_sha256 must identify the original video")
    provenance = manifest.get("mask_provenance", {})
    if not isinstance(provenance.get("manual_intervention"), bool):
        raise ValueError("mask_provenance.manual_intervention must be explicitly true or false")
    if not isinstance(provenance.get("backend"), str) or not provenance["backend"].strip():
        raise ValueError("mask_provenance.backend is required")
    if not isinstance(provenance.get("prompt_origin"), str) or not provenance["prompt_origin"].strip():
        raise ValueError("mask_provenance.prompt_origin is required")
    resolved = {}
    for role in ("image", "mask"):
        if not isinstance(manifest.get(role), str) or not manifest[role]:
            raise ValueError(f"{role} path is required")
        path = (manifest_path.parent / manifest[role]).resolve()
        if not path.is_file():
            raise ValueError(f"Missing {role}: {path}")
        actual_hash = sha256(path)
        expected_hash = manifest.get(f"{role}_sha256")
        if expected_hash is not None and actual_hash != expected_hash:
            raise ValueError(f"{role} SHA256 does not match the input manifest")
        resolved[role] = {"path": str(path), "sha256": actual_hash, "bytes": path.stat().st_size}
    image = np.asarray(Image.open(resolved["image"]["path"]).convert("RGB"))
    mask_image = np.asarray(Image.open(resolved["mask"]["path"]))
    if mask_image.ndim != 2:
        raise ValueError("Mask must be single-channel binary data; RGB/alpha masks are ambiguous")
    if mask_image.shape != image.shape[:2]:
        raise ValueError("Mask and RGB must share the same full-frame pixel coordinates")
    if not set(np.unique(mask_image).tolist()).issubset({0, 1, 255, False, True}):
        raise ValueError("Mask must contain binary 0/1 or 0/255 values")
    mask = mask_image > 0
    if not mask.any() or mask.all():
        raise ValueError("Mask must identify a nonempty object distinct from background")
    record = {
        "manifest": manifest,
        "manifest_sha256": sha256(manifest_path),
        "files": resolved,
        "width": image.shape[1],
        "height": image.shape[0],
        "mask_pixels": int(mask.sum()),
        "pixel_policy": "unchanged full-frame RGB; binary visible-object mask; no crop or inpainting",
    }
    return record, image, mask


def git_read(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, text=True, capture_output=True).stdout.strip()


def code_provenance(repo: Path, expected_commit: str | None) -> tuple[dict, list[str]]:
    errors = []
    record = {"path": str(repo), "expected_commit": expected_commit, "official_repository": OFFICIAL_REPO}
    if not COMMIT.fullmatch(expected_commit or ""):
        errors.append("Supply --expected-code-commit as a full, reviewed 40-character official commit SHA")
    try:
        record["commit"] = git_read(repo, "rev-parse", "HEAD")
        origin = git_read(repo, "remote", "get-url", "origin")
        official_origin = origin.rstrip("/").removesuffix(".git") == OFFICIAL_REPO
        record["origin"] = OFFICIAL_REPO if official_origin else "non-official origin (not recorded)"
        if not official_origin:
            errors.append("Repository origin must be the public official HTTPS SAM 3D Objects repository")
        if record["commit"] != expected_commit:
            errors.append("Repository HEAD differs from --expected-code-commit")
        tracked_changes = git_read(repo, "status", "--porcelain", "--untracked-files=no")
        record["tracked_changes"] = tracked_changes
        if tracked_changes:
            errors.append("Official checkout has tracked modifications; review and record a clean pinned checkout")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        record["available"] = False
        errors.append(f"No inspectable official Git checkout at {repo} ({type(exc).__name__})")
    api = repo / "notebook/inference.py"
    if api.is_file():
        record["notebook_inference_sha256"] = sha256(api)
    else:
        errors.append("Official notebook/inference.py is missing")
    return record, errors


def checkpoint_provenance(config_path: Path | None, hash_weights: bool = False) -> tuple[dict, list[str]]:
    """Inspect safe YAML and local config references without Hydra instantiation."""
    errors = []
    record = {"pipeline_config": str(config_path) if config_path else None, "files": [], "weights_hashed": hash_weights}
    if config_path is None or not config_path.is_file():
        return record, ["Provision approved SAM 3D Objects checkpoint files and pass their local --config pipeline.yaml; gated HF access may be required"]
    try:
        import yaml
    except ImportError:
        return record, ["PyYAML is required for non-executing checkpoint/config preflight"]
    workspace = config_path.parent
    pending = [config_path]
    visited = set()
    found_weights = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        if not current.is_file():
            errors.append(f"Referenced configuration/checkpoint is missing: {current}")
            continue
        entry = {"path": str(current), "bytes": current.stat().st_size}
        if current.suffix.lower() in WEIGHT_SUFFIXES:
            found_weights.add(current)
            if hash_weights:
                entry["sha256"] = sha256(current)
        else:
            entry["sha256"] = sha256(current)
        record["files"].append(entry)
        if current.suffix.lower() not in {".yaml", ".yml"}:
            continue
        try:
            document = yaml.safe_load(current.read_text())
        except (ValueError, yaml.YAMLError) as exc:
            errors.append(f"Unable to safely parse {current}: {type(exc).__name__}")
            continue
        stack = [("", document)]
        while stack:
            key, value = stack.pop()
            if isinstance(value, dict):
                stack.extend((str(k), v) for k, v in value.items())
            elif isinstance(value, list):
                stack.extend((key, v) for v in value)
            elif isinstance(value, str) and key.endswith(("_ckpt_path", "_config_path")):
                # Inference.__init__ sets this exact field to config_file.parent.
                # Resolve only that literal prefix; never invoke OmegaConf resolvers
                # (which may run arbitrary registered functions) in preflight.
                if value == "${workspace_dir}":
                    value = str(workspace)
                elif value.startswith("${workspace_dir}/"):
                    value = str(workspace / value[len("${workspace_dir}/"):])
                if "${" in value:
                    errors.append(f"Unresolved path interpolation in {current}: {key}; only a literal ${{workspace_dir}} prefix is supported in safe preflight")
                    continue
                candidate = Path(value)
                pending.append(candidate.resolve() if candidate.is_absolute() else (workspace / candidate).resolve())
    if not found_weights:
        errors.append("No local checkpoint weights were resolved from pipeline config; inference remains disabled")
    record["checkpoint_count"] = len(found_weights)
    return record, errors


def runtime_provenance(device: str = "cuda") -> tuple[dict, list[str]]:
    errors = []
    record = {"platform": platform.system(), "machine": platform.machine(), "python": platform.python_version(), "device_requested": device}
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        errors.append("Official runtime requires Linux 64-bit x86; this adapter will not silently switch to MPS or CPU")
    if device != "cuda":
        errors.append("Only the official CUDA runtime is supported by this adapter")
    if not os.environ.get("CONDA_PREFIX"):
        errors.append("Activate the official SAM3D conda environment (notebook/inference.py requires CONDA_PREFIX)")
    dependencies = ["torch", "torchvision", "omegaconf", "hydra", "pytorch3d", "kaolin", "spconv", "utils3d", "moge", "trimesh", "cv2", "gradio", "seaborn"]
    record["dependency_presence"] = {name: importlib.util.find_spec(name) is not None for name in dependencies}
    missing = [name for name, present in record["dependency_presence"].items() if not present]
    if missing:
        errors.append("Missing official runtime dependencies: " + ", ".join(missing))
    try:
        import torch
        record["torch"] = torch.__version__
        record["torch_cuda"] = torch.version.cuda
        record["cuda_available"] = torch.cuda.is_available()
        if not record["cuda_available"]:
            errors.append("No CUDA GPU visible in this Python process; run inside the allocated GPU job")
        else:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            record["gpu_name"] = props.name
            record["gpu_memory_gib"] = props.total_memory / 2**30
            if record["gpu_memory_gib"] < 31:
                errors.append("Official SAM3D requirements specify at least 32 GB GPU memory")
    except ImportError:
        errors.append("PyTorch is not installed in the selected Python runtime")
    return record, errors


def preflight(args: argparse.Namespace) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    errors = []
    try:
        inputs, image, mask = read_inputs(args.input_manifest)
    except (ValueError, OSError, KeyError) as exc:
        inputs, image, mask = {"manifest_path": str(args.input_manifest)}, None, None
        errors.append(f"Invalid reconstruction input: {exc}")
    code, code_errors = code_provenance(args.repo.resolve(), args.expected_code_commit)
    weights, weight_errors = checkpoint_provenance(args.config.resolve() if args.config else None, hash_weights=args.command == "run")
    runtime, runtime_errors = runtime_provenance(args.device)
    errors.extend(code_errors + weight_errors + runtime_errors)
    return {
        "schema_version": "0.1",
        "status": "ready" if not errors else "prerequisites_missing",
        "backend": "Meta SAM 3D Objects",
        "model_repository": OFFICIAL_MODEL,
        "learned_inference_executed": False,
        "inputs": inputs,
        "code": code,
        "checkpoint": weights,
        "runtime": runtime,
        "errors": errors,
        "credential_policy": "No credential reads or checkpoint downloads; pre-provision gated checkpoints in the runtime",
        "official_setup": OFFICIAL_REPO + "/blob/main/doc/setup.md",
    }, image, mask


def numeric_output(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def export_model_outputs(output: dict, destination: Path, appearance: str = "vertex-color") -> dict:
    """Export genuine model return values. Never substitute a procedural asset."""
    mesh = output.get("glb")
    splat = output.get("gs")
    if mesh is None or splat is None:
        raise RuntimeError("Official result must contain both glb mesh and gs splat; no fallback geometry is allowed")
    if appearance not in {"vertex-color", "uv-baked"}:
        raise ValueError("Unknown model appearance mode")
    if appearance == "uv-baked":
        visual = getattr(mesh, "visual", None)
        material = getattr(visual, "material", None)
        uv = getattr(visual, "uv", None)
        texture = getattr(material, "baseColorTexture", None)
        if texture is None:
            texture = getattr(material, "image", None)
        if getattr(visual, "kind", None) != "texture" or uv is None or texture is None:
            raise RuntimeError("UV baking was requested but the model result has no UVs and colour texture; refusing a silent vertex-colour fallback")
    mesh.export(str(destination / "object.glb"))
    splat.save_ply(str(destination / "object-splat.ply"))
    layout = {}
    for key in ("rotation", "translation", "scale", "pointmap", "pointmap_colors"):
        if key in output:
            layout[key] = numeric_output(output[key])
    for key in ("rotation", "translation", "scale"):
        if key not in layout or not np.isfinite(layout[key]).all():
            raise RuntimeError(f"Model layout has missing or non-finite {key}")
    np.savez_compressed(destination / "layout-and-pointmap.npz", **layout)
    export_records = []
    for name in ("object.glb", "object-splat.ply", "layout-and-pointmap.npz"):
        path = destination / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Model export missing or empty: {name}")
        export_records.append({"file": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return {
        "files": export_records,
        "layout_fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in layout.items()},
        "appearance": (
            "official notebook defaults: model-generated vertex colour and coloured Gaussian splat; UV texture baking disabled"
            if appearance == "vertex-color" else
            "official pipeline UV baking: model-generated Gaussian appearance rendered from 100 synthetic views and baked into a UV colour texture; not direct original-video pixel projection"
        ),
        "appearance_mode": appearance,
        "direct_original_video_texture_projection": False,
        "coordinates": "unaltered SAM3D native asset/layout conventions; not yet converted to project OpenCV camera coordinates",
        "geometry_reconstructed_by_model": True,
        "manual_geometry_fallback": False,
        "hidden_surfaces": "model-inferred, not observed or measured",
        "metric_scale_verified": False,
    }


def run(args: argparse.Namespace, report: dict, image: np.ndarray, mask: np.ndarray) -> int:
    destination = args.output.resolve()
    if destination.exists():
        raise ValueError("Output directory already exists; use a fresh run name to preserve evidence")
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    metadata = dict(report)
    metadata.update(status="running", seed=args.seed, learned_inference_executed=False, validation_status="not_yet_reviewed", appearance_requested=args.appearance)
    write_json(destination / "metadata.json", metadata)
    try:
        inputs_dir = destination / "inputs"
        inputs_dir.mkdir()
        shutil.copy2(args.input_manifest, inputs_dir / "manifest.json")
        # Preserve original source files as well as the exact arrays passed to SAM3D.
        for role in ("image", "mask"):
            path = Path(report["inputs"]["files"][role]["path"])
            shutil.copy2(path, inputs_dir / (role + "-source" + path.suffix))
        Image.fromarray(image).save(inputs_dir / "rgb.png")
        Image.fromarray(mask.astype(np.uint8) * 255).save(inputs_dir / "mask.png")
        metadata["model_input_sha256"] = {name: sha256(inputs_dir / name) for name in ("rgb.png", "mask.png")}
        metadata["model_input_array_sha256"] = {"rgb_uint8": hashlib.sha256(image.tobytes()).hexdigest(), "mask_bool": hashlib.sha256(mask.tobytes()).hexdigest()}
        metadata["offline_hub_policy"] = True
        write_json(destination / "metadata.json", metadata)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        import torch
        repo = args.repo.resolve()
        config_path = args.config.resolve()
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(repo / "notebook"))
        from inference import Inference
        if Path(sys.modules["inference"].__file__).resolve() != repo / "notebook/inference.py":
            raise RuntimeError("Unexpected inference module location")
        previous_cwd = Path.cwd()
        try:
            os.chdir(repo)
            model = Inference(str(config_path), compile=False)
            if model._pipeline.device.type != "cuda":
                raise RuntimeError("Upstream config did not initialize CUDA; refusing a silent device fallback")
            torch.cuda.synchronize()
            inference_started = time.monotonic()
            metadata["learned_inference_attempted"] = True
            write_json(destination / "metadata.json", metadata)
            if args.appearance == "uv-baked":
                # Same official pipeline with an explicit documented option change.
                # This is a slower experimental mode; there is no fallback on failure.
                rgba = model.merge_mask_to_rgba(image, mask)
                output = model._pipeline.run(
                    rgba, None, args.seed, stage1_only=False,
                    with_mesh_postprocess=False, with_texture_baking=True,
                    with_layout_postprocess=False, use_vertex_color=False,
                    stage1_inference_steps=None, pointmap=None,
                )
            else:
                output = model(image, mask, seed=args.seed)
            torch.cuda.synchronize()
            metadata["learned_inference_executed"] = True
            metadata["inference_seconds"] = time.monotonic() - inference_started
            metadata["outputs"] = export_model_outputs(output, destination, args.appearance)
        finally:
            os.chdir(previous_cwd)
        metadata.update(status="candidate_generated_pending_visual_review", wall_seconds=time.monotonic() - started)
        write_json(destination / "metadata.json", metadata)
        print(json.dumps({"status": metadata["status"], "metadata": str(destination / "metadata.json")}, indent=2))
        return 0
    except Exception as exc:
        metadata.update(status="failed_no_fallback", error=f"{type(exc).__name__}: {exc}", wall_seconds=time.monotonic() - started)
        write_json(destination / "metadata.json", metadata)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--expected-code-commit")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--appearance", choices=("vertex-color", "uv-baked"), default="vertex-color",
                        help="Official notebook vertex colour by default; experimental UV baking uses model-generated synthetic views, not direct original-video textures")
    parser.add_argument("--output", type=Path, help="Fresh output directory for run; optional JSON report file for preflight")
    args = parser.parse_args(argv)
    if args.command == "run" and args.output is None:
        parser.error("run requires --output with a fresh run directory")
    report, image, mask = preflight(args)
    if args.command == "preflight" or report["errors"]:
        if args.command == "preflight" and args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if args.output.exists():
                parser.error("Preflight report already exists; use a fresh filename")
            write_json(args.output, report)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if not report["errors"] else 2
    return run(args, report, image, mask)


if __name__ == "__main__":
    raise SystemExit(main())
