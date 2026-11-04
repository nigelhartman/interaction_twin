#!/usr/bin/env python3
"""Choose reconstruction inputs from observed RGB + SAM masks, without mesh priors.

This is an automatic *view selection* step, not automatic object discovery: the
input masks can have assistant-provided SAM prompts and corrections. No pose,
hand-built geometry, corner annotation or semantic visual judgment is consumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_metrics(rgb: np.ndarray, mask: np.ndarray) -> dict:
    """Image-only quality descriptors; none establishes completeness or accuracy."""
    if rgb.shape[:2] != mask.shape:
        raise ValueError("RGB and mask must have identical dimensions; no silent resizing")
    mask = mask.astype(bool)
    area = int(mask.sum())
    if not area:
        raise ValueError("Cannot score an empty object mask")
    h, w = mask.shape
    ys, xs = np.where(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    bbox_area = (x1 - x0) * (y1 - y0)
    edge_band = np.zeros_like(mask)
    edge_band[:3] = edge_band[-3:] = True
    edge_band[:, :3] = edge_band[:, -3:] = True
    boundary = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    edge_contact = float((boundary & edge_band).sum() / max(1, boundary.sum()))
    touched_sides = sum((x0 <= 2, y0 <= 2, x1 >= w - 2, y1 >= h - 2))
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    # Erosion avoids treating the high-contrast cutout boundary as texture detail.
    sharpness = float(np.var(lap[eroded])) if eroded.sum() >= 16 else 0.0
    return {
        "area_pixels": area,
        "area_fraction": area / (w * h),
        "bbox_xyxy": [x0, y0, x1, y1],
        "bbox_mask_coverage": area / bbox_area,
        "edge_contact_fraction": edge_contact,
        "frame_sides_touched": int(touched_sides),
        "interior_laplacian_variance": sharpness,
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
    }


def temporal_consistency(candidate: dict, neighbors: list[dict]) -> float:
    """Area/coverage stability among nearby observations with the same mask scope.

    Different sampling densities and cover-vs-body scopes must not be compared as
    if they were object growth. This does not infer whether an object is occluded.
    """
    values = []
    for other in neighbors:
        if other is candidate or other["mask_scope"] != candidate["mask_scope"]:
            continue
        dt = abs(other["timestamp_s"] - candidate["timestamp_s"])
        if dt > 2.1 or dt < 0.04:
            continue
        a, b = candidate["metrics"], other["metrics"]
        values.append((dt, math.exp(-abs(math.log(a["area_pixels"] / b["area_pixels"]))) *
                       math.exp(-abs(a["bbox_mask_coverage"] - b["bbox_mask_coverage"]))))
    # One nearest observation each side would be preferable to density weighting;
    # nearest two gives a bounded local diagnostic even at the video ends.
    return float(np.mean([v for _, v in sorted(values)[:2]])) if values else 0.5


def score_candidates(candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    area_p90 = max(1, float(np.percentile([c["metrics"]["area_pixels"] for c in candidates], 90)))
    sharp_p90 = max(1, float(np.percentile([c["metrics"]["interior_laplacian_variance"] for c in candidates], 90)))
    for c in candidates:
        m = c["metrics"]
        terms = {
            "area": min(1.0, math.sqrt(m["area_pixels"] / area_p90)),
            "sharpness": min(1.0, math.log1p(m["interior_laplacian_variance"]) / math.log1p(sharp_p90)),
            "unclipped": max(0.0, 1.0 - 2 * m["edge_contact_fraction"] - 0.25 * m["frame_sides_touched"]),
            "temporal_consistency": temporal_consistency(c, candidates),
            "bbox_mask_coverage": float(m["bbox_mask_coverage"]),
        }
        base = sum(terms[k] * w for k, w in {
            "area": 0.25, "sharpness": 0.25, "unclipped": 0.20,
            "temporal_consistency": 0.20, "bbox_mask_coverage": 0.10,
        }.items())
        # Hand proximity is only a candidate signal, and not present everywhere.
        # Unknown is retained as unknown; no assertion of occlusion-free content.
        proximity = c.get("occlusion_candidate", {}).get("hand_proximity_fraction")
        penalty = 0.05 * proximity if proximity is not None else 0.0
        c["score_components"] = terms
        c["occlusion_candidate_penalty"] = penalty
        c["score"] = base - penalty
    return sorted(candidates, key=lambda c: (-c["score"], c["timestamp_s"], c["sample_index"]))


def select_diverse(ranked: list[dict], count: int, min_interval_s: float = 1.0) -> list[dict]:
    """Avoid spending five reconstruction attempts on adjacent near-duplicate PTS."""
    result = []
    for c in ranked:
        if all(abs(c["timestamp_s"] - p["timestamp_s"]) >= min_interval_s for p in result):
            result.append(c)
            if len(result) == count:
                return result
    return result


def relative(path: Path, base: Path) -> str:
    import os
    return os.path.relpath(path.resolve(), base.resolve())


def load_hand_candidates(scene: Path, source_hash: str) -> dict:
    """Read raw hand-union candidates only, never assistant thumb corridors."""
    result = {}
    for frames_path in scene.parent.glob("*/occlusion/*/frames.json"):
        meta_path = frames_path.parent / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("source_sha256") != source_hash:
            continue
        for frame in json.loads(frames_path.read_text())["frames"]:
            if frame.get("raw_hand_union"):
                result[round(frame["timestamp_s"], 6)] = {
                    "file": frames_path.parent / frame["raw_hand_union"],
                    "metadata_file": meta_path,
                    "meaning": meta.get("mask_meaning"),
                    "prompt_origin": meta.get("prompt_origin", "unknown"),
                }
    return result


def discover_mask_provenance(mask_path: Path, scene: Path) -> dict:
    for parent in mask_path.parents:
        meta = parent / "metadata.json"
        if meta.exists():
            data = json.loads(meta.read_text())
            result = {
                "metadata_file": relative(meta, scene), "metadata_sha256": sha256(meta),
                "model": data.get("model"), "device": data.get("device"),
                "checkpoint_sha256": data.get("checkpoint_sha256"),
                "prompt_origin": data.get("seed_origin", data.get("prompt_origin", "assistant_visual_annotation")),
                "human_confirmed": data.get("human_confirmed", False),
                "prompts": data.get("prompts"), "accuracy_measured": data.get("accuracy_measured", False),
            }
            annotation = parent / "annotations.json"
            if annotation.exists():
                result["annotations_file"] = relative(annotation, scene)
                result["annotations_sha256"] = sha256(annotation)
            return result
        if parent == scene.parent:
            break
    return {"prompt_origin": "assistant_visual_annotation_or_correction_from_existing_pipeline", "human_confirmed": False}


def collect_candidates(scene: Path, object_id: str) -> tuple[list[dict], dict]:
    manifest = json.loads((scene / "manifest.json").read_text())
    registry = json.loads((scene / "registry.json").read_text())
    entry = next((o for o in registry["objects"] if o["id"] == object_id), None)
    if entry is None:
        raise ValueError(f"Object ID {object_id!r} is not a registered physical candidate")
    index = json.loads((scene / "frames/index.json").read_text())
    by_index = {f["index"]: f for f in index["frames"]}
    observations = json.loads((scene / "observations.json").read_text())
    hands = load_hand_candidates(scene, manifest["source"]["sha256"])
    candidates = []
    for frame in observations["frames"]:
        source = by_index[frame["sample_index"]]
        if abs(source["timestamp_s"] - frame["timestamp_s"]) > 1e-6:
            raise ValueError("RGB and observation PTS disagree")
        for obj in frame["objects"]:
            if obj["id"] != object_id or not obj.get("mask_file"):
                continue
            image_path, mask_path = scene / source["file"], (scene / obj["mask_file"]).resolve()
            rgb = np.array(Image.open(image_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L")) > 0
            if rgb.shape[:2] != (540, 960):
                raise ValueError("Milestone expects native existing 960x540 sampled RGB")
            if not mask.any():
                continue
            c = {
                "sample_index": frame["sample_index"], "timestamp_s": frame["timestamp_s"],
                "source_case": obj["source_case"], "source_sample_index": obj["source_sample_index"],
                "source_track_id": obj["source_track_id"], "object_id": object_id,
                "physical_identity_status": obj.get("identity_status"),
                "image_file": relative(image_path, scene), "mask_file": relative(mask_path, scene),
                "image_sha256": sha256(image_path), "mask_sha256": sha256(mask_path),
                "mask_scope": obj.get("scope", "unknown"), "model_provider": obj.get("provider"),
                "segmentation_accuracy_measured": False,
                "mask_provenance": discover_mask_provenance(mask_path, scene),
                "metrics": mask_metrics(rgb, mask), "heuristic_flags": obj.get("heuristic_flags", []),
                "occlusion_candidate": {"status": "not_assessed", "hand_proximity_fraction": None},
            }
            hand = hands.get(round(frame["timestamp_s"], 6))
            if hand is not None:
                hand_mask = np.array(Image.open(hand["file"]).convert("L")) > 0
                if hand_mask.shape != mask.shape:
                    raise ValueError("Hand/RGB dimensions disagree")
                boundary = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                near_hand = cv2.dilate(hand_mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
                c["occlusion_candidate"] = {
                    "status": "raw_sam_hand_boundary_proximity_candidate",
                    "hand_proximity_fraction": float((boundary & near_hand).sum() / max(1, boundary.sum())),
                    "hand_union_file": relative(hand["file"], scene),
                    "hand_metadata_file": relative(hand["metadata_file"], scene),
                    "prompt_origin": hand["prompt_origin"],
                    "limitation": "Hand proximity is not foreground depth ordering or hidden-object completeness.",
                }
            candidates.append(c)
    return candidates, {"manifest": manifest, "registry_object": entry}


def export_selected(scene: Path, output: Path, selected: list[dict]) -> list[dict]:
    result = []
    for rank, candidate in enumerate(selected, 1):
        c = dict(candidate)
        name = f"rank-{rank:02d}-sample-{c['sample_index']:06d}"
        directory = output / name
        directory.mkdir()
        image_path, mask_path = scene / c["image_file"], scene / c["mask_file"]
        rgb = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L").point(lambda x: 255 if x else 0)
        rgb.save(directory / "image.png")
        mask.save(directory / "mask.png")
        rgba = rgb.convert("RGBA")
        rgba.putalpha(mask)
        rgba.save(directory / "rgba.png")
        x0, y0, x1, y1 = c["metrics"]["bbox_xyxy"]
        pad = max(8, round(0.08 * max(x1 - x0, y1 - y0)))
        crop = [max(0, x0-pad), max(0, y0-pad), min(rgb.width, x1+pad), min(rgb.height, y1+pad)]
        rgba.crop(crop).save(directory / "rgba-crop.png")
        c["selection_rank"] = rank
        c["exports"] = {"directory": name, "rgb": f"{name}/image.png", "mask": f"{name}/mask.png",
                        "rgba": f"{name}/rgba.png", "rgba_crop": f"{name}/rgba-crop.png",
                        "crop_xyxy_in_original": crop, "rgb_mask_dimensions": [rgb.width, rgb.height],
                        "background_invented": False, "resized": False}
        c["exports"]["rgb_sha256"] = sha256(directory / "image.png")
        c["exports"]["mask_sha256"] = sha256(directory / "mask.png")
        result.append(c)
    return result


def contact_sheet(output: Path, selected: list[dict]) -> None:
    width, tile_w, tile_h = 1000, 500, 350
    sheet = Image.new("RGB", (width, tile_h * math.ceil(len(selected)/2)), "#18202d")
    draw = ImageDraw.Draw(sheet)
    for i, c in enumerate(selected):
        x, y = (i % 2) * tile_w, (i // 2) * tile_h
        image = Image.open(output / c["exports"]["rgba_crop"]).convert("RGBA")
        image.thumbnail((480, 280))
        sheet.paste(image, (x+(tile_w-image.width)//2, y+48), image)
        draw.text((x+12, y+9), f"#{i+1} sample {c['sample_index']} | {c['timestamp_s']:.3f}s | score {c['score']:.3f}", fill="white")
        draw.text((x+12, y+29), c["mask_scope"][:65], fill="#a7b5cf")
        status = c["occlusion_candidate"]["status"]
        draw.text((x+12, y+330), f"Hand occlusion: {status[:55]}", fill="#ffce83")
    sheet.save(output / "selected-contact-sheet.jpg", quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="camera-battery-full-scene")
    parser.add_argument("--object-id", default="camera_00001")
    parser.add_argument("--output", default="reconstruction-input-v1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-interval-s", type=float, default=1.0)
    args = parser.parse_args()
    if args.top_k < 1 or args.min_interval_s < 0:
        parser.error("--top-k must be positive and --min-interval-s nonnegative")
    scene = ROOT / "data" / args.scene
    output = scene / args.output
    if output.exists():
        raise FileExistsError(f"Preserve prior exports; choose a fresh --output: {output}")
    candidates, context = collect_candidates(scene, args.object_id)
    ranked = score_candidates(candidates)
    selected = select_diverse(ranked, args.top_k, args.min_interval_s)
    if not selected:
        raise ValueError("No usable RGB/mask pairs exist for the requested candidate ID")
    output.mkdir(parents=True)
    selected = export_selected(scene, output, selected)
    source_path = scene / context["manifest"]["source"]["file"]
    source_verified = source_path.is_file() and sha256(source_path) == context["manifest"]["source"]["sha256"]
    if not source_verified:
        raise ValueError("Original source SHA256 could not be verified")
    result = {
        "schema_version": "0.1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ranked_inputs_not_reconstruction", "object_id": args.object_id,
        "object_choice_origin": "explicit_user_scope_component_id_not_automatically_discovered",
        "source": context["manifest"]["source"], "registry_object": context["registry_object"],
        "source_sha256_verified_against_local_video": source_verified,
        "input_contract": {"rgb_mask_same_pts": True, "dimensions": [960, 540],
                           "time_domain": "source_video_presentation_seconds", "resize_performed": False},
        "selection_method": {
            "type": "deterministic_image_and_mask_quality_heuristics", "agent_visual_choice": False,
            "reads_geometry_or_corner_annotations": False, "candidate_count": len(candidates),
            "weights": {"relative_area": .25, "relative_interior_sharpness": .25, "unclipped": .20,
                        "temporal_area_and_coverage_consistency": .20, "bbox_mask_coverage": .10},
            "optional_raw_hand_proximity_max_penalty": .05,
            "minimum_selected_timestamp_interval_s": args.min_interval_s,
            "score_is_probability_or_accuracy": False,
            "selection_does_not_establish": ["complete_visible_object", "occlusion_free_view", "correct_SAM_mask", "novel_view_diversity"],
        },
        "assistance_disclosure": {
            "existing_sam_prompts_and_corrections": "assistant_visual_annotations_recorded_in_mask_provenance",
            "this_selection_uses_no_new_assistant_annotation": True,
            "fully_automatic_from_unseen_video_proven": False,
            "limitations": ["Existing SAM candidates retain segmentation and component-scope uncertainty.",
                            "Camera coarse masks include the attached cover; dense component masks may have occluded boundaries.",
                            "Hand occlusion is unassessed for most timestamps; missing masks are not evidence of no hand.",
                            "Sharpness/size/coverage can rank the wrong view; reconstruction must still be evaluated.",
                            "Battery identity is limited to the outward tracklet; no identity is merged through occlusion."],
        },
        "selected": selected, "all_ranked_candidates": ranked,
        "artifacts": {"contact_sheet": "selected-contact-sheet.jpg"},
    }
    (output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    contact_sheet(output, selected)
    print(json.dumps({"selection_file": str(output / "selection.json"), "candidates": len(candidates),
                      "selected": [{k: c[k] for k in ("selection_rank", "sample_index", "timestamp_s", "score", "mask_scope")} for c in selected]}, indent=2))


if __name__ == "__main__":
    main()
