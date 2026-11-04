#!/usr/bin/env python3
"""Bounded automatic SAM3.1 image masks with explicit uncertain association.

This is per-frame hosted segmentation plus deterministic association, not SAM
video propagation or verified physical identity. Original model boxes/config
and every returned native instance are retained. No geometry/pose is authored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import meta_sam_api as sam

SLUG = re.compile(r"[a-z0-9][a-z0-9-]*\Z")
TRACK = re.compile(r"[a-z][a-z0-9_]*\Z")
MAX_TRACKS = 8
MAX_TRACK_FRAMES = 600


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def confined(path, directory):
    path, directory = Path(path).resolve(), Path(directory).resolve()
    if path != directory and directory not in path.parents:
        raise ValueError("Input path escapes its case")
    return path


def validate_config(cfg, rows):
    if cfg.get("manual_seed_boxes_used") is not False or cfg.get("manual_geometry_used", False):
        raise ValueError("Only explicitly automatic model-proposed configurations are accepted")
    origin = str(cfg.get("seed_origin", "")).lower()
    if "model" not in origin or not any(name in origin for name in ["meta", "codex", "openai"]):
        raise ValueError("An explicit model seed origin is required; no assistant/manual seeds")
    if not isinstance(cfg.get("tracks"), list) or not 0 <= len(cfg["tracks"]) <= MAX_TRACKS:
        raise ValueError("Zero to eight model-proposed tracklets are required")
    if [row.get("index") for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Frame rows must retain contiguous one-based source sampling indices")
    seen, count = set(), 0
    for track in cfg["tracks"]:
        tid = track.get("id", "")
        if not TRACK.fullmatch(tid) or tid in seen:
            raise ValueError("Invalid or duplicate model track ID")
        seen.add(tid)
        lo, hi = track["sample_range"]
        if type(lo) is not int or type(hi) is not int or not 1 <= lo <= hi <= len(rows):
            raise ValueError("Invalid tracklet source sampling range")
        count += hi - lo + 1
        if not track.get("candidate_entity_ids") or not track.get("identity_status"):
            raise ValueError("Candidate IDs and unresolved identity status are required")
        concept(track)
        if not track.get("prompts"):
            raise ValueError("A model-proposed box seed is required")
        for prompt in track["prompts"]:
            index = prompt.get("sample_index")
            if type(index) is not int or not lo <= index <= hi:
                raise ValueError("Model seed is outside its tracklet range")
            row = rows[index - 1]
            box = prompt.get("box_xyxy")
            if not isinstance(box, list) or len(box) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in box):
                raise ValueError("A finite model-proposed native-pixel box is required")
            x0, y0, x1, y1 = box
            if not 0 <= x0 < x1 <= row["width"] or not 0 <= y0 < y1 <= row["height"]:
                raise ValueError("Model seed box lies outside native source pixels")
            if "source_timestamp_s" in prompt and abs(prompt["source_timestamp_s"] - row["timestamp_s"]) > 1e-7:
                raise ValueError("Model seed PTS differs from original frame PTS")
            if "model" not in str(prompt.get("origin", cfg.get("seed_origin", ""))).lower():
                raise ValueError("Model seed prompt origin is required")
        for low, high in track.get("withhold_ranges", []):
            if not lo <= low <= high <= hi:
                raise ValueError("Invalid explicit withholding range")
    if count > MAX_TRACK_FRAMES:
        raise ValueError("Plan exceeds 600 track-frame requests; review a larger bounded plan explicitly")
    return count


def concept(track):
    """Use only a model-authored noun phrase; never inject object names by ID."""
    value = track.get("sam_concept", track.get("label"))
    return sam._concept(value)


def bbox(mask):
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)] if len(xs) else None


def box_iou(a, b):
    if a is None or b is None:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0.0


def choose(candidates, *, seeds=None, previous=None):
    """Conservative image-space proposal association; no 3D or hidden fill."""
    if not candidates:
        return None, {"status": "no_mask", "method": "no_returned_instances", "accepted": False}
    scores = []
    for index, item in enumerate(candidates):
        mask = item["mask"]
        area, bounds = int(mask.sum()), bbox(mask)
        if not area:
            continue
        if seeds:
            overlap = max(box_iou(bounds, seed) for seed in seeds)
            score = overlap
            eligible = overlap >= 0.08
            details = {"seed_box_iou": round(overlap, 6)}
        elif previous is not None:
            prior = previous["mask"]
            prior_area = int(prior.sum())
            intersection = int(np.logical_and(mask, prior).sum())
            union = int(np.logical_or(mask, prior).sum())
            overlap = intersection / union if union else 0
            old_xy, new_xy = np.argwhere(prior).mean(axis=0), np.argwhere(mask).mean(axis=0)
            distance = float(np.linalg.norm(new_xy - old_xy))
            diagonal = math.hypot(*mask.shape)
            ratio = area / prior_area if prior_area else 0
            proximity = max(0.0, 1.0 - distance / max(1.0, diagonal * 0.18))
            score = 0.75 * overlap + 0.25 * proximity
            eligible = (0.18 <= ratio <= 5.0 and distance <= diagonal * 0.18
                        and (overlap >= 0.02 or distance <= max(12, math.sqrt(prior_area) * 1.5)))
            details = {"previous_mask_iou": round(overlap, 6), "centroid_distance_pixels": round(distance, 4),
                       "area_ratio": round(ratio, 6)}
        else:
            # Preserve a diagnostic proposal but never reacquire after a gap by
            # silently treating a per-image instance ordinal as physical identity.
            score = float(item["record"].get("confidence") or 0)
            eligible, details = False, {"no_adjacent_supported_association": True}
        scores.append((score, index, eligible, details))
    if not scores:
        return None, {"status": "no_mask", "method": "only_empty_returned_masks", "accepted": False}
    scores.sort(key=lambda row: (-row[0], row[1]))
    score, index, eligible, details = scores[0]
    margin = score - scores[1][0] if len(scores) > 1 else None
    ambiguous = margin is not None and margin < 0.06
    accepted = bool(eligible and not ambiguous)
    return candidates[index], {"status": "candidate_visible" if accepted else "withheld_due_to_occlusion_or_ambiguity",
                               "accepted": accepted, "method": "model_seed_box_iou" if seeds else "adjacent_mask_iou_centroid",
                               "score": round(score, 6), "runner_up_margin": None if margin is None else round(margin, 6),
                               "ambiguous": ambiguous, "candidate_count": len(candidates), **details,
                               "physical_identity_verified": False}


def associate_track(track, detections):
    """Start on model seed; traverse forwards/backwards without bridging gaps."""
    lo, hi = track["sample_range"]
    seeds = {}
    for prompt in track["prompts"]:
        seeds.setdefault(prompt["sample_index"], []).append(prompt["box_xyxy"])
    anchor = min(seeds)
    selected = {}

    def visit(index, previous):
        proposal, association = choose(detections[index], seeds=seeds.get(index), previous=previous)
        explicitly_withheld = any(a <= index <= b for a, b in track.get("withhold_ranges", []))
        if explicitly_withheld:
            association.update(status="withheld_due_to_occlusion_or_ambiguity", accepted=False,
                               explicit_model_withhold_range=True)
        selected[index] = (proposal, association)
        return proposal if association["accepted"] else None

    state = visit(anchor, None)
    anchor_state = state
    for index in range(anchor + 1, hi + 1):
        state = visit(index, state)
    state = anchor_state
    for index in range(anchor - 1, lo - 1, -1):
        state = visit(index, state)
    return selected


def run_case(case, config_path, run, *, client=None, segmenter=None):
    """Client injection permits offline contract tests; CLI alone enables API use."""
    case = Path(case).resolve()
    if not SLUG.fullmatch(case.name) or not SLUG.fullmatch(run):
        raise ValueError("Invalid case/run slug")
    frame_document = load(case / "frames/index.json")
    rows = frame_document["frames"]
    cfg, manifest = load(config_path), load(case / "manifest.json")
    count = validate_config(cfg, rows)
    source_hash = manifest["source"]["sha256"]
    if not sam.HEX64.fullmatch(source_hash):
        raise ValueError("Source SHA256 is required")
    if frame_document.get("time_domain") != sam.TIME_DOMAIN:
        raise ValueError("Frame index must use original video presentation timestamps")
    for row in rows:
        sam._source({"video_sha256": source_hash, "sample_index": row["index"],
                     "timestamp_s": row["timestamp_s"], "time_domain": sam.TIME_DOMAIN})
        sam._dimensions(row["width"], row["height"])
        image_path = confined(case / row["file"], case)
        with Image.open(image_path) as image:
            if image.size != (row["width"], row["height"]) or image.getexif().get(274, 1) not in (None, 1):
                raise ValueError("Frame pixels/orientation differ from original source coordinate record")
    if count:
        if segmenter is None:
            sam.parser_preflight()
            segmenter = sam.segment_image
        if client is None:
            from meta_api import MetaClient
            client = MetaClient()
    out = case / "segmentation" / run
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, out / "annotations.json")  # Exact original bytes/prompts.
    started = time.monotonic()
    metadata = {"status": "running", "model": "sam-3.1", "device": "hosted_Meta_Model_API",
                "provider": "Meta Model API", "source_sha256": source_hash,
                "checkpoint_sha256": None, "checkpoint_identity": "provider_managed_not_disclosed",
                "human_confirmed": False, "seed_origin": cfg["seed_origin"],
                "manual_seed_boxes_used": False, "manual_geometry_used": False,
                "scope": "automatic_per_frame_image_segmentation_with_deterministic_2D_association",
                "sam_video_propagation_executed": False, "accuracy_measured": False,
                "maximum_track_frame_requests": count, "estimated_maximum_image_cost_usd": str(sam.estimate_cost("image") * count),
                "association_policy": "model seed box IoU; adjacent mask IoU/centroid; ambiguity withheld; no gap bridging",
                "old_sam2_audits_reused": False, "config_sha256": sha(config_path)}
    write(out / "metadata.json", metadata)
    if count == 0:
        write(out / "tracks.json", {"tracks": [],
                                     "outside_track_ranges": "not_assessed_not_evidence_of_absence"})
        metadata.update(status="complete", outcome="completed_zero_model_proposed_tracks",
                        track_count=0, mask_count=0, hosted_image_request_count=0,
                        native_instance_count=0, candidate_visible_count=0,
                        learned_inference_executed=False, model_call_made=False,
                        estimated_cost_usd="0.000000", wall_time_s=round(time.monotonic() - started, 3),
                        review_status="no_model_proposed_tracks_no_masks_to_audit")
        write(out / "metadata.json", metadata)
        return metadata
    cache, all_tracks, calls = {}, [], 0
    try:
        for track in cfg["tracks"]:
            track_started = time.monotonic()
            tid, phrase = track["id"], concept(track)
            lo, hi = track["sample_range"]
            detections = {}
            for index in range(lo, hi + 1):
                row = rows[index - 1]
                key = (index, phrase)
                if key not in cache:
                    calls += 1
                    key_slug = f"request-{calls:06d}"
                    request_dir = out / "api" / key_slug
                    request_dir.mkdir(parents=True)
                    source = {"video_sha256": source_hash, "sample_index": row["index"],
                              "timestamp_s": row["timestamp_s"], "time_domain": sam.TIME_DOMAIN,
                              "source_case": case.name, "source_sample_index": row["index"]}
                    input_manifest = request_dir / "input.json"
                    image_path = confined(case / row["file"], case)
                    write(input_manifest, {"schema_version": "0.1", "object_id": tid,
                                           "image": str(image_path), "image_sha256": sha(image_path), "source": source})
                    # Receipt path and decoder path are distinct; decoder owns
                    # a fresh folder and never overwrites an attempted request.
                    result_dir = request_dir / "result"
                    result = segmenter(input_manifest, phrase, result_dir,
                                       client.sam_callback(request_dir / "accounting", images=1),
                                       prompt_origin=cfg["seed_origin"] + "; automatic SAM concept")
                    native = []
                    for record in result["instances"]:
                        path = confined(result_dir / record["mask"], result_dir)
                        mask = np.asarray(Image.open(path).convert("L")) > 0
                        if mask.shape != (row["height"], row["width"]):
                            raise ValueError("Returned SAM mask coordinates changed")
                        native.append({"record": record, "mask": mask, "path": path,
                                       "request_directory": str(request_dir.relative_to(out))})
                    cache[key] = native
                    print(f"SAM3.1 image {calls}/{count}: sample {index}, concept {phrase!r}, {len(native)} returned instance(s)", flush=True)
                detections[index] = cache[key]
            associations = associate_track(track, detections)
            records = []
            dest = out / "raw_masks" / tid
            dest.mkdir(parents=True)
            for index in range(lo, hi + 1):
                row = rows[index - 1]
                item, association = associations[index]
                path = dest / f"{index:06d}.png"
                if item is None:
                    mask = np.zeros((row["height"], row["width"]), dtype=bool)
                    Image.fromarray(mask.astype(np.uint8)).save(path)
                    response_instance = None
                else:
                    mask = item["mask"]
                    shutil.copyfile(item["path"], path)
                    response_instance = {"instance_id": item["record"]["instance_id"],
                                         "mask_file": str(item["path"].relative_to(out)),
                                         "request_directory": item["request_directory"],
                                         "instance_id_scope": "independent_image_response_only_not_temporal_identity"}
                records.append({"sample_index": row["index"], "timestamp_s": row["timestamp_s"],
                                "raw_mask_file": str(path.relative_to(out)), "raw_area_pixels": int(mask.sum()),
                                "raw_bbox_xyxy": bbox(mask), "status": association["status"],
                                "identity_status": track["identity_status"], "source_frame_sha256": sha(case / row["file"]),
                                "raw_mask_sha256": sha(path), "association": association,
                                "api_instance": response_instance, "all_instances_preserved": True})
            record = {"track_id": tid, "candidate_entity_ids": track["candidate_entity_ids"],
                      "sample_range": [lo, hi], "wall_time_s": round(time.monotonic() - track_started, 3),
                      "frames": records,
                      "model_concept": phrase, "physical_identity_verified": False}
            write(out / "tracks" / f"{tid}.json", record)
            all_tracks.append(record)
        write(out / "tracks.json", {"tracks": all_tracks,
                                     "outside_track_ranges": "not_assessed_not_evidence_of_absence"})
        metadata.update(status="complete", wall_time_s=round(time.monotonic() - started, 3),
                        track_count=len(all_tracks), mask_count=sum(len(x["frames"]) for x in all_tracks),
                        hosted_image_request_count=calls, native_instance_count=sum(len(x) for x in cache.values()),
                        candidate_visible_count=sum(f["status"] == "candidate_visible" for x in all_tracks for f in x["frames"]),
                        review_status="pending_fresh_Meta_mask_audit")
        write(out / "metadata.json", metadata)
        return metadata
    except Exception as exc:
        metadata.update(status="failed", error_type=type(exc).__name__,
                        hosted_image_request_count=calls, wall_time_s=round(time.monotonic() - started, 3),
                        retry_policy="no_automatic_retry_preserve_all_request_receipts")
        write(out / "metadata.json", metadata)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", default="flow-components-1")
    args = parser.parse_args()
    if not SLUG.fullmatch(args.case) or not SLUG.fullmatch(args.run):
        parser.error("Invalid case/run slug")
    result = run_case(ROOT / "data" / args.case, args.config.resolve(), args.run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
