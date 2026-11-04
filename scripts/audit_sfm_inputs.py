#!/usr/bin/env python3
"""Audit only observed RGB/SAM masks for a sparse geometry experiment.

No generated geometry, prior poses, depth predictions, model execution or network
access is used. Counts and ranking are diagnostics, not reconstruction success.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROGRAMS = [
    "data/flow-meta-video-camera-20261030-v1/flow-procedure-1/interaction.json",
    "data/flow-meta-video-wallet-20261030-v1/flow-procedure-1/interaction.json",
    "data/flow-meta-video-jabra-20261030-v1/flow-procedure-scope-normalized-v1/interaction.json",
    "data/flow-meta-dense-jabra-20261030-v1/flow-procedure-1/interaction.json",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_artifact(rel: str, expected: str) -> tuple[Path, str]:
    path = (ROOT / rel).resolve()
    path.relative_to(ROOT)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Artifact hash mismatch: {rel}")
    return path, actual


def numeric_summary(values: list[float]) -> dict:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def rigid_risks(track: dict) -> list[str]:
    """Only label-based warnings; never a replacement for observing rigidity."""
    ident = track["id"].lower()
    label = track.get("label", "").lower()
    limits = " ".join(track.get("limitations", [])).lower()
    risks = []
    if ident in {"wallet", "pouch_flap"} or "flexible" in limits or "folding" in limits:
        risks.append("model_description_indicates_nonrigid_or_folding_target")
    if ident == "batteries" or "one vs two" in limits:
        risks.append("mask_may_group_multiple_independently_moving_objects")
    if "single containing object" in limits or ident == "charging_case":
        risks.append("mask_may_include_articulating_base_and_lid")
    if "case_body" == ident and "closed versus open" in limits:
        risks.append("closed_open_mask_scope_may_change_or_include_lid")
    if "coin" in label:
        risks.append("thin_near_planar_geometry_depth_may_be_degenerate")
    return risks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", action="append", help="Repository-relative interaction.json; repeatable")
    parser.add_argument("--output", default="output/observed-geometry-input-audit-20261031.json")
    parser.add_argument("--contact-dir", default="output/observed-geometry-input-contact-sheets-20261031")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.025, edgeThreshold=10)
    gray_cache: dict[str, np.ndarray] = {}
    feature_cache: dict[tuple[str, str], dict] = {}
    output = {
        "format": "observed_geometry_input_audit", "version": 1,
        "created_date": "2026-10-31", "opencv_version": cv2.__version__,
        "inputs_policy": {
            "allowed": ["exact procedure-cited source frames", "exact procedure-cited SAM masks", "saved tracklet labels and audit labels"],
            "unused": ["generated mesh geometry", "previous poses", "predicted depth", "LLM inference", "API calls", "model downloads"],
            "feature_method": "SIFT on full source grayscale with a 3 px eroded mask; conservative features additionally require distance to mask boundary >= max(3, 1.5 * keypoint size)",
            "sift_parameters": {"nfeatures": 3000, "contrastThreshold": 0.025, "edgeThreshold": 10},
            "ranking": "sort by label-level rigid suitability, then views with >=12 conservative features, then median conservative features; ranking does not establish parallax, rigidity, correct identity, coverage or 3D success",
        },
        "programs": [], "tracks": [], "ranking": [],
        "limitations": [
            "A SAM mask and its plausibility audit can be wrong or include fingers/background.",
            "Feature count alone does not establish correct correspondence, sufficient parallax or a rigid component.",
            "All physical identities remain those of the saved model-proposed unverified tracklets.",
            "No appearance from hidden surfaces can be measured without observations.",
            "No relative scale is metric unless calibration evidence is supplied.",
        ],
    }
    for program_rel in args.program or DEFAULT_PROGRAMS:
        program_path = (ROOT / program_rel).resolve()
        program_path.relative_to(ROOT)
        doc = json.loads(program_path.read_text())
        output["programs"].append({"path": program_rel, "sha256": sha256(program_path), "case_id": doc["case_id"], "source": doc["source"], "evidence_samples": len(doc["evidence"])})
        by_track = {track["id"]: {"case_id": doc["case_id"], "program_path": program_rel, "tracklet": track, "rigid_assumption_warnings": rigid_risks(track), "observations": []} for track in doc["tracklets"]}
        for ev in doc["evidence"]:
            frame_path, frame_hash = resolve_artifact(ev["frame_path"], ev["frame_sha256"])
            if frame_hash not in gray_cache:
                image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"Cannot read frame: {ev['frame_path']}")
                gray_cache[frame_hash] = image
            gray = gray_cache[frame_hash]
            for ob in ev.get("observations", []):
                mask_path, mask_hash = resolve_artifact(ob["mask_path"], ob["mask_sha256"])
                cache_key = (frame_hash, mask_hash)
                if cache_key not in feature_cache:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is None or mask.shape != gray.shape:
                        raise ValueError(f"Mask/frame dimension mismatch: {ob['mask_path']}")
                    binary = (mask > 0).astype(np.uint8) * 255
                    area = int(np.count_nonzero(binary))
                    eroded = cv2.erode(binary, np.ones((7, 7), np.uint8))
                    keypoints, _ = sift.detectAndCompute(gray, eroded)
                    distance = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                    safe = [kp for kp in keypoints if float(distance[min(gray.shape[0]-1, max(0, round(kp.pt[1]))), min(gray.shape[1]-1, max(0, round(kp.pt[0])))]) >= max(3.0, 1.5 * kp.size)]
                    ys, xs = np.nonzero(binary)
                    bbox = [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1] if area else None
                    feature_cache[cache_key] = {
                        "frame_size_wh": [gray.shape[1], gray.shape[0]], "mask_area_px": area,
                        "mask_fraction": area / gray.size, "mask_bbox_xyxy_exclusive": bbox,
                        "sift_centers_in_eroded_mask": len(keypoints), "conservative_sift_count": len(safe),
                        "conservative_keypoints_xy": [[round(kp.pt[0], 3), round(kp.pt[1], 3)] for kp in safe],
                        "masked_laplacian_variance": float(np.var(cv2.Laplacian(gray, cv2.CV_32F)[binary > 0])) if area else None,
                    }
                by_track[ob["track_id"]]["observations"].append({
                    "evidence_id": ev["id"], "source_index": ev["source_index"], "timestamp_s": ev["timestamp_s"],
                    "frame_path": ev["frame_path"], "frame_sha256": frame_hash,
                    "mask_path": ob["mask_path"], "mask_sha256": mask_hash,
                    "gate_status": ob.get("gate_status"), "audit_status": ob.get("audit_status"),
                    **feature_cache[cache_key],
                })
        for rec in by_track.values():
            observations = rec["observations"]
            rec["summary"] = {
                "observation_count": len(observations), "nonempty_masks": sum(ob["mask_area_px"] > 0 for ob in observations),
                "frames_with_at_least_12_conservative_features": sum(ob["conservative_sift_count"] >= 12 for ob in observations),
                "frames_with_at_least_30_conservative_features": sum(ob["conservative_sift_count"] >= 30 for ob in observations),
                "conservative_sift_count": numeric_summary([ob["conservative_sift_count"] for ob in observations]),
                "mask_area_px": numeric_summary([ob["mask_area_px"] for ob in observations]),
                "audit_label_counts": dict(Counter(ob["audit_status"] for ob in observations)),
            }
            output["tracks"].append(rec)
            print(f"{doc['case_id']} / {rec['tracklet']['id']}: {rec['summary']['frames_with_at_least_12_conservative_features']}/{len(observations)} views with >=12 conservative features; median {rec['summary']['conservative_sift_count']['median']}", file=sys.stderr)
    ordered = sorted(output["tracks"], key=lambda rec: (
        -len(rec["rigid_assumption_warnings"]),
        rec["summary"]["frames_with_at_least_12_conservative_features"],
        rec["summary"]["conservative_sift_count"]["median"] or 0,
    ), reverse=True)
    contact_dir = ROOT / args.contact_dir
    contact_dir.mkdir(parents=True, exist_ok=True)
    for rank, rec in enumerate(ordered, start=1):
        output["ranking"].append({"rank": rank, "case_id": rec["case_id"], "track_id": rec["tracklet"]["id"], "program_path": rec["program_path"], "summary": rec["summary"], "rigid_assumption_warnings": rec["rigid_assumption_warnings"]})
        # Make automatic top-feature-frame sheets for the first six candidates.
        if rank > 6:
            continue
        top = sorted(rec["observations"], key=lambda ob: ob["conservative_sift_count"], reverse=True)[:6]
        if not top:
            continue
        cell_w, cell_h = 320, 300
        sheet = Image.new("RGB", (3*cell_w, 2*cell_h+70), "#f3f0e8")
        draw = ImageDraw.Draw(sheet)
        title = f"{rec['case_id']} / {rec['tracklet']['id']}"
        draw.text((12, 10), title, fill="#162724")
        draw.text((12, 30), "Automatic top SIFT frames | yellow: observed mask | cyan: safe features", fill="#162724")
        draw.text((12, 48), "Candidates only; identity, rigidity and 3D shape are unverified.", fill="#162724")
        for index, ob in enumerate(top):
            rgb = np.asarray(Image.open(ROOT / ob["frame_path"]).convert("RGB")).copy()
            mask = np.asarray(Image.open(ROOT / ob["mask_path"]).convert("L")) > 0
            rgb[mask] = (rgb[mask].astype(float)*0.65 + np.array([255, 207, 35])*0.35).astype(np.uint8)
            for x, y in ob["conservative_keypoints_xy"]:
                cv2.circle(rgb, (round(x), round(y)), 3, (0, 255, 255), 1, cv2.LINE_AA)
            thumb = Image.fromarray(rgb)
            thumb.thumbnail((cell_w-12, cell_h-48))
            x0, y0 = (index % 3)*cell_w, 70+(index // 3)*cell_h
            sheet.paste(thumb, (x0+(cell_w-thumb.width)//2, y0))
            draw.text((x0+8, y0+cell_h-44), f"sample {ob['source_index']} | {ob['timestamp_s']:.3f}s | SIFT {ob['conservative_sift_count']}", fill="#162724")
            draw.text((x0+8, y0+cell_h-26), str(ob["audit_status"])[:43], fill="#162724")
        filename = f"{rank:02d}-{rec['case_id']}-{rec['tracklet']['id']}.jpg"
        path = contact_dir / filename
        sheet.save(path, quality=90)
        rec["contact_sheet"] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
    target = ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(target.relative_to(ROOT)), "program_count": len(output["programs"]), "track_count": len(output["tracks"]), "top_candidates": output["ranking"][:4]}, indent=2))


if __name__ == "__main__":
    main()
