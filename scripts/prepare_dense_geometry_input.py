#!/usr/bin/env python3
"""Prepare a bounded camera-only observed-geometry input using local SAM 2.

The sole segmentation prompt is a saved SAM 3.1 mask, never manual coordinates.
This diagnostic input is not a published interaction program or a reconstruction.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image
import torch
import sam2
from sam2.build_sam import build_sam2_video_predictor


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "data/flow-meta-video-camera-20261030-v1/flow-procedure-1/interaction.json"
CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
START = Decimal("61.061")
END = Decimal("66.066")
FPS = Decimal("8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def verified(path_text: str, expected: str) -> Path:
    path = (ROOT / path_text).resolve()
    path.relative_to(ROOT)
    if sha256(path) != expected:
        raise ValueError(f"Source artifact hash mismatch: {path_text}")
    return path


def command(args: list[str]) -> str:
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/observed-geometry-experiment-20261031/dense-camera-input-v1")
    parser.add_argument("--start", default=str(START), help="Exact source PTS of a saved SAM 3.1 seed")
    parser.add_argument("--end", default=str(END), help="Bounded ending source PTS")
    parser.add_argument("--fps", default=str(FPS), help="Requested sample rate; actual source PTS are retained")
    parser.add_argument("--max-frames", type=int, default=45)
    args = parser.parse_args()
    start, end, fps = Decimal(args.start), Decimal(args.end), Decimal(args.fps)
    if not (start.is_finite() and end.is_finite() and fps.is_finite() and 0 <= start < end and 0 < fps <= 12 and 2 <= args.max_frames <= 55):
        parser.error("Require a finite positive interval, 0<fps<=12 and 2<=max-frames<=55")
    if os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("CPU operator fallback is enabled; this run requires MPS without fallback")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no CPU fallback, downloads or installs attempted")
    checkpoint = ROOT / "models/sam2.1_hiera_tiny.pt"
    if not checkpoint.is_file():
        raise RuntimeError("Existing SAM 2.1 Tiny checkpoint unavailable; no download attempted")
    out = (ROOT / args.output).resolve()
    out.relative_to(ROOT / "data/observed-geometry-experiment-20261031")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__).resolve(), out / "preparation-script.py")
    started = time.monotonic()
    metadata = {
        "format": "dense_observed_geometry_input_preparation", "version": 1,
        "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "sam2.1_hiera_tiny", "provider": "meta_sam2_local",
        "device": "mps", "torch_version": torch.__version__,
        "cpu_storage_offload": True, "cpu_operator_fallback_enabled": False,
        "checkpoint_path": relative(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "sam2_package": str(Path(sam2.__file__).resolve()), "config": CONFIG,
        "script_sha256": sha256(Path(__file__).resolve()),
        "sampling": {"requested_start_s": str(start), "requested_end_s": str(end), "requested_fps": str(fps), "maximum_frames": args.max_frames, "selection": "nearest actual decoded source PTS to each 1/requested_fps target, within bounded interval; no interpolation"},
        "no_manual_geometry_or_poses": True, "no_api_calls": True,
        "no_model_downloads": True, "no_predicted_depth": True,
        "accuracy_measured": False, "published_interaction_program": False,
    }
    write(out / "metadata.json", metadata)
    try:
        doc = json.loads(PROGRAM.read_text())
        source = verified(doc["source"]["path"], doc["source"]["sha256"])
        manifest_path = source.parents[1] / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        original = verified(relative(source.parents[1] / manifest["source"]["original_file"]), doc["source"]["original_sha256"])
        seed_evidence = next(ev for ev in doc["evidence"] if Decimal(str(ev["timestamp_s"])) == start)
        seed_observation = next(ob for ob in seed_evidence["observations"] if ob["track_id"] == "camera_body")
        if seed_observation["audit_status"] not in {"plausible_visible_mask", "not_model_audited"}:
            raise ValueError("Requested saved SAM 3.1 seed has an uncertain or rejected audit label")
        seed_frame = verified(seed_evidence["frame_path"], seed_evidence["frame_sha256"])
        seed_mask_path = verified(seed_observation["mask_path"], seed_observation["mask_sha256"])
        seed_mask = np.asarray(Image.open(seed_mask_path).convert("L")) > 0
        if seed_mask.shape != (540, 960) or not seed_mask.any():
            raise ValueError("Seed must be a nonempty original-coordinate 960x540 mask")
        probe_args = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_entries", "frame=best_effort_timestamp_time,width,height", "-of", "json", str(source)]
        decoded = json.loads(command(probe_args))["frames"]
        available = [(index, Decimal(frame["best_effort_timestamp_time"])) for index, frame in enumerate(decoded) if start <= Decimal(frame["best_effort_timestamp_time"]) <= end]
        chosen = []
        targets = []
        step = 0
        while start + Decimal(step) / fps <= end:
            target = start + Decimal(step) / fps
            native, pts = min(available, key=lambda item: abs(item[1]-target))
            if native not in [row[0] for row in chosen]:
                chosen.append((native, pts))
                targets.append(str(target))
            step += 1
        if not 2 <= len(chosen) <= args.max_frames or chosen[0][1] != start:
            raise ValueError("Bounded sampling did not yield an exact seed frame within the declared maximum samples")
        if any((decoded[index]["width"], decoded[index]["height"]) != (960, 540) for index, _ in chosen):
            raise ValueError("Source frames must retain 960x540 coordinates")
        frames = out / "frames"
        masks = out / "masks"
        frames.mkdir()
        masks.mkdir()
        select = "+".join(f"eq(n\\,{native})" for native, _ in chosen)
        extract_args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-i", str(source), "-map", "0:v:0", "-vf", f"select={select}", "-fps_mode", "passthrough", "-q:v", "2", str(frames / "%06d.jpg")]
        command(extract_args)
        extracted = sorted(frames.glob("*.jpg"))
        if len(extracted) != len(chosen):
            raise RuntimeError("Decoded frame count differs from exact source-PTS selection")
        # Retain the exact previously audited seed JPEG; no recompression at seed.
        shutil.copyfile(seed_frame, extracted[0])
        for image_path in extracted:
            with Image.open(image_path) as image:
                if image.size != (960, 540):
                    raise ValueError("Extracted frame changed coordinate dimensions")
        metadata.update({
            "source_program": {"path": relative(PROGRAM), "sha256": sha256(PROGRAM)},
            "source_video": {"path": relative(source), "sha256": sha256(source)},
            "original_video": {"path": relative(original), "sha256": sha256(original)},
            "seed": {"origin": "saved_hosted_meta_sam3.1_camera_body_mask", "evidence": seed_evidence, "observation": seed_observation, "frame_used_byte_identical": True, "manual_points_or_box": False, "sam_api": "predictor.add_new_mask(frame_idx=0,obj_id=1,mask=seed_binary)"},
            "sampling": {**metadata["sampling"], "actual_frame_count": len(chosen), "source_native_decoded_frame_indices_zero_based": [row[0] for row in chosen], "source_pts_s_exact_strings": [str(row[1]) for row in chosen], "target_pts_s_exact_strings": targets, "source_frame_size_wh": [960, 540]},
            "extraction": {"probe_command": probe_args, "ffmpeg_command": extract_args, "ffmpeg_version": command(["ffmpeg", "-version"]).splitlines()[0], "seed_frame": "exact saved seed JPEG copied byte-for-byte; all later frames are ffmpeg JPEG q2 of actual decoded source indices"},
            "runtime_code": [
                {"path": str(path), "sha256": sha256(path)}
                for path in [Path(sam2.__file__).parent / "sam2_video_predictor.py", Path(sam2.__file__).parent / "build_sam.py", Path(sam2.__file__).parent / CONFIG]
            ],
        })
        write(out / "metadata.json", metadata)
        print(f"Prepared {len(chosen)} exact-PTS RGB frames; loading SAM 2.1 Tiny on MPS", flush=True)
        predictor = build_sam2_video_predictor(CONFIG, str(checkpoint), device="mps", apply_postprocessing=False)
        mask_records = {}
        inference_started = time.monotonic()
        with torch.inference_mode():
            state = predictor.init_state(str(frames), offload_video_to_cpu=True, offload_state_to_cpu=True)
            predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=seed_mask)
            for index, ids, logits in predictor.propagate_in_video(state, start_frame_idx=0):
                object_index = list(ids).index(1)
                mask = (logits[object_index, 0] > 0).cpu().numpy()
                if mask.shape != (540, 960):
                    raise ValueError("SAM output changed original coordinate dimensions")
                mask_path = masks / f"{index+1:06d}.png"
                Image.fromarray(mask.astype(np.uint8)*255).save(mask_path)
                mask_records[index] = {"path": relative(mask_path), "sha256": sha256(mask_path), "area_pixels": int(mask.sum()), "origin": "local_meta_sam2.1_tiny_mps_mask_prompt_output" if index == 0 else "local_meta_sam2.1_tiny_mps_forward_video_propagation", "seed_identity_unverified": True}
                if len(mask_records) % 5 == 0 or len(mask_records) == len(chosen):
                    write(out / "progress.json", {"completed_masks": len(mask_records), "expected": len(chosen), "wall_time_s": round(time.monotonic()-started, 3)})
                    print(f"SAM 2.1 MPS {len(mask_records)}/{len(chosen)} masks; {time.monotonic()-started:.1f}s", flush=True)
            torch.mps.synchronize()
        if set(mask_records) != set(range(len(chosen))):
            raise RuntimeError("SAM did not return every bounded frame")
        evidence = []
        for index, ((native, pts), frame_path) in enumerate(zip(chosen, extracted)):
            record = mask_records[index]
            evidence.append({
                "id": f"dense_source_{index+1:06d}", "source_index": index+1, "timestamp_s": float(pts),
                "source_native_decoded_frame_index_zero_based": native,
                "frame_path": relative(frame_path), "frame_sha256": sha256(frame_path),
                "observations": [{"track_id": "camera_body", "mask_path": record["path"], "mask_sha256": record["sha256"], "audit_status": "not_model_audited", "gate_status": "candidate_geometric_gate_only", "mask_origin": record["origin"], "area_pixels": record["area_pixels"]}],
            })
        track = next(track for track in doc["tracklets"] if track["id"] == "camera_body")
        geometry_input = {
            "format": "observed_geometry_input", "version": 1, "case_id": f"observed-geometry-20261031-{out.name}",
            "source": doc["source"],
            "tracklets": [{**track, "source_sample_range": [1, len(chosen)], "limitations": track.get("limitations", []) + ["Dense SAM2 propagation initialized only from a hosted SAM3.1 mask; propagated masks have not been visually or model audited."]}],
            "evidence": evidence,
            "provenance": {"metadata_path": relative(out / "metadata.json"), "not_a_full_interaction_program": True, "no_pose_depth_or_mesh_input": True, "mask_generator": "local Meta SAM 2.1 Tiny / MPS", "seed_generator": "saved hosted Meta SAM 3.1 video mask", "no_human_geometry_points_boxes_or_poses": True},
        }
        write(out / "geometry-input.json", geometry_input)
        metadata.update(status="complete", actual_frame_count=len(evidence), mask_count=len(mask_records), nonempty_masks=sum(record["area_pixels"] > 0 for record in mask_records.values()), inference_and_mask_output_s=round(time.monotonic()-inference_started, 3), wall_time_s=round(time.monotonic()-started, 3), geometry_input={"path": relative(out / "geometry-input.json"), "sha256": sha256(out / "geometry-input.json")})
        write(out / "metadata.json", metadata)
        print(json.dumps({key: metadata[key] for key in ["status", "actual_frame_count", "mask_count", "nonempty_masks", "wall_time_s", "geometry_input"]}, indent=2), flush=True)
    except Exception as exc:
        metadata.update(status="failed", error=str(exc), wall_time_s=round(time.monotonic()-started, 3))
        write(out / "metadata.json", metadata)
        raise


if __name__ == "__main__":
    main()
