#!/usr/bin/env python3
"""Prepare a full-source timeline and conservative object IDs from real prior masks.

This is a reuse/export step. It neither runs new inference nor transfers a mask
to another timestamp. Tracklet IDs do not imply additional physical batteries.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "data/camera-battery-full-scene"
SOURCE_CASES = ["camera-battery", "camera-battery-action-5fps", "camera-battery-withdrawal-15fps"]
OPTIONAL_SOURCE_CASES = ["camera-battery-insertion-10fps", "camera-battery-second-insertion-10fps"]
TRACK_BINDINGS = {
    "camera": ("camera_00001", "camera_00001", ["camera_00001"]),
    "cover": ("cover_00001", "cover_00001", ["cover_00001"]),
    "out_left": ("battery_00001", "battery_00001", ["battery_00001"]),
    "out_right": ("battery_00002", "battery_00002", ["battery_00002"]),
    "table_front": ("battery_track_00003", None, ["battery_00001", "battery_00002"]),
    "table_back": ("battery_track_00004", None, ["battery_00001", "battery_00002"]),
    "in_first": ("battery_track_00005", None, ["battery_00001", "battery_00002"]),
    "visible_end": ("battery_track_00006", None, ["battery_00001", "battery_00002"]),
    "in_first_dense": ("battery_track_00007", None, ["battery_00001", "battery_00002"]),
    "in_second_dense": ("battery_track_00008", None, ["battery_00001", "battery_00002"]),
}


def read(p):
    return json.loads(Path(p).read_text())


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2) + "\n")


def key(t):
    return round(float(t), 6)


def relative(p):
    return os.path.relpath(p, CASE)


def registry(source_hash):
    objects = [
        {"id": "camera_00001", "label": "Camera", "kind": "camera", "color": "#79aaff",
         "prior_entity_id": "camera_body", "identity_status": "single_camera_candidate", "human_confirmed": False,
         "limitation": "Coarse full-video SAM masks include attached cover; dense masks are visible body candidates."},
        {"id": "cover_00001", "label": "Battery cover", "kind": "cover", "color": "#c5a3ff",
         "prior_entity_id": "compartment_cover", "parent_candidate_id": "camera_00001",
         "identity_status": "attached_component_candidate", "human_confirmed": False},
        {"id": "battery_00001", "label": "Outward battery A", "kind": "battery", "color": "#42e0c2",
         "prior_entity_id": "battery_candidate_01", "source_track_id": "out_left",
         "identity_status": "distinct_outward_candidate_only", "human_confirmed": False,
         "limitation": "Persistent identity after hand occlusion is unknown."},
        {"id": "battery_00002", "label": "Outward battery B", "kind": "battery", "color": "#ffa94f",
         "prior_entity_id": "battery_candidate_02", "source_track_id": "out_right",
         "identity_status": "distinct_outward_candidate_only", "human_confirmed": False,
         "limitation": "Persistent identity after hand occlusion is unknown."},
    ]
    tracks = []
    for track_id, (identity, physical_id, candidates) in TRACK_BINDINGS.items():
        tracks.append({"id": identity, "source_track_id": track_id,
                       "physical_object_id": physical_id, "candidate_physical_ids": candidates,
                       "display_label": identity if physical_id else identity + " ?",
                       "identity_status": "candidate_continuity_within_tracklet" if physical_id else "cross_occlusion_identity_unresolved"})
    return {"schema_version": "0.4", "source_sha256": source_hash,
            "id_policy": "Stable candidate IDs within observed continuity; never silently merge identities through occlusion.",
            "objects": objects, "observation_tracks": tracks,
            "distinct_battery_candidate_count": 2,
            "limitations": ["Later battery track IDs are observations, not additional physical batteries.",
                            "No completed seating, exchange, charge, polarity or second insertion is established.",
                            "Not assessed or withheld does not mean absent.",
                            "All SAM masks remain candidates; ground-truth segmentation accuracy is unavailable."]}


def prepare_frames():
    by_time = {}
    source_cases = SOURCE_CASES + [c for c in OPTIONAL_SOURCE_CASES if (ROOT / "data" / c / "frames/index.json").is_file()]
    for case_id in source_cases:
        source_case = ROOT / "data" / case_id
        for row in read(source_case / "frames/index.json")["frames"]:
            t = key(row["timestamp_s"])
            entry = by_time.setdefault(t, {"timestamp_s": row["timestamp_s"], "source_samples": []})
            entry["source_samples"].append({"case_id": case_id, "index": row["index"],
                "file": str((source_case / row["file"]).relative_to(ROOT)), "timestamp_s": row["timestamp_s"]})
    frames = []
    for index, (_, entry) in enumerate(sorted(by_time.items()), 1):
        # Prefer denser original extraction when exactly the same source PTS occurs.
        selected = entry["source_samples"][-1]
        source = ROOT / selected["file"]
        # Stable filenames survive addition of denser samples. Old numbered
        # exports remain intact for readers that already loaded the old index.
        destination = CASE / "frames" / f"pts-{int(round(entry['timestamp_s'] * 1e6)):09d}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
                raise ValueError(f"Existing sampled image differs: {destination}")
        else:
            os.link(source, destination)
        with Image.open(source) as image:
            width, height = image.size
        if (width, height) != (960, 540):
            raise ValueError("Full-scene source images must share the existing 960x540 coordinate system")
        frames.append({"index": index, "file": str(destination.relative_to(CASE)),
                       "timestamp_s": entry["timestamp_s"], "width": width, "height": height,
                       "source_samples": entry["source_samples"]})
    write(CASE / "frames/index.json", {"schema_version": "0.4",
        "time_domain": "source_video_presentation_seconds", "frames": frames,
        "sampling": "Union of existing 1fps full source, 5fps action, 15fps withdrawal, and any available 10fps insertion samples; exact source PTS retained."})
    return frames


def make_observation(rec, track, seg_dir, source_case, source_index, bbox_field="bbox_xyxy", mask_field="visible_mask_file"):
    identity, physical_id, candidates = TRACK_BINDINGS[track]
    box = rec.get(bbox_field)
    return {"id": identity, "physical_object_id": physical_id,
            "candidate_physical_ids": candidates, "display_label": identity if physical_id else identity + " ?",
            "bbox_xyxy": box, "label_anchor_xy": None if box is None else [(box[0] + box[2]) / 2, box[1]],
            "mask_file": relative(seg_dir / rec[mask_field]),
            "status": "candidate_visible", "identity_status": rec.get("identity_status", "candidate_continuity_within_tracklet"),
            "source_case": source_case, "source_sample_index": source_index, "source_track_id": track,
            "provider": "meta_sam2.1_hiera_tiny_local_mps_previous_inference", "accuracy_measured": False}


def prepare_observations(frames):
    native = {}
    # Full-source masks are assembly candidates, including the attached cover.
    coarse_case = "camera-battery"
    seg = ROOT / "data" / coarse_case / "segmentation/camera-full-1fps"
    for rec in read(seg / "frames.json")["frames"]:
        obs = make_observation(rec, "camera", seg, coarse_case, rec["sample_index"], mask_field="mask_file")
        obs["scope"] = "camera_assembly_includes_attached_cover"
        obs["heuristic_flags"] = rec.get("heuristic_flags", [])
        native.setdefault(key(rec["timestamp_s"]), {"objects": {}, "withheld": []})["objects"][obs["id"]] = obs
    action_case = "camera-battery-action-5fps"
    seg = ROOT / "data" / action_case / "segmentation/components-v2"
    for rec in read(seg / "frames.json")["frames"]:
        target = native.setdefault(key(rec["timestamp_s"]), {"objects": {}, "withheld": []})
        for component in rec["components"]:
            track = component["track_id"]
            obs = make_observation(component, track, seg, action_case, rec["sample_index"])
            obs["scope"] = "resolved_visible_component_candidate"
            target["objects"][obs["id"]] = obs
        for track, reason in rec.get("withheld", {}).items():
            identity, physical_id, candidates = TRACK_BINDINGS[track]
            target["withheld"].append({"id": identity, "physical_object_id": physical_id,
                "candidate_physical_ids": candidates, "source_track_id": track,
                "source_case": action_case, "source_sample_index": rec["sample_index"], "reason": reason})
    # Independent new insertion tracklets never acquire the outward physical
    # IDs merely by being similar cylinders at a later time.
    for case_id, track_id, binding in [
        ("camera-battery-insertion-10fps", "in_first", "in_first_dense"),
        ("camera-battery-second-insertion-10fps", "in_second", "in_second_dense")]:
        seg = ROOT / "data" / case_id / "segmentation/battery-v1"
        track_file = seg / "tracks" / f"{track_id}.json"
        metadata_file = seg / "metadata.json"
        if not track_file.is_file() or not metadata_file.is_file() or read(metadata_file).get("status") != "complete":
            continue
        for rec in read(track_file)["frames"]:
            target = native.setdefault(key(rec["timestamp_s"]), {"objects": {}, "withheld": []})
            if rec["status"] == "candidate_visible":
                obs = make_observation(rec, binding, seg, case_id, rec["sample_index"],
                    bbox_field="raw_bbox_xyxy", mask_field="raw_mask_file")
                obs["source_track_id"] = track_id
                obs["scope"] = "new_raw_insertion_candidate_not_peer_resolved"
                obs["identity_status"] = "cross_occlusion_identity_unresolved"
                target["objects"][obs["id"]] = obs
            else:
                identity, physical_id, candidates = TRACK_BINDINGS[binding]
                target["withheld"].append({"id": identity, "physical_object_id": physical_id,
                    "candidate_physical_ids": candidates, "source_track_id": track_id,
                    "source_case": case_id, "source_sample_index": rec["sample_index"], "reason": [rec["status"]]})
    pilot_case = "camera-battery-withdrawal-15fps"
    seg = ROOT / "data" / pilot_case / "segmentation/battery-v1"
    for rec in read(seg / "frames.json")["frames"]:
        target = native.setdefault(key(rec["timestamp_s"]), {"objects": {}, "withheld": []})
        for component in rec["components"]:
            obs = make_observation(component, "out_left", seg, pilot_case, rec["sample_index"])
            obs["scope"] = "raw_single_part_candidate_not_peer_resolved"
            target["objects"][obs["id"]] = obs
    output = []
    all_ids = ["camera_00001", "cover_00001", "battery_00001", "battery_00002"]
    for frame in frames:
        rec = native.get(key(frame["timestamp_s"]), {"objects": {}, "withheld": []})
        objects = list(rec["objects"].values())
        for obj in objects:
            if not (CASE / obj["mask_file"]).is_file():
                raise FileNotFoundError(obj["mask_file"])
        assessed = {obj["physical_object_id"] for obj in objects if obj["physical_object_id"]}
        output.append({"sample_index": frame["index"], "timestamp_s": frame["timestamp_s"],
            "objects": objects, "withheld": rec["withheld"],
            "not_assessed_physical_ids": [identity for identity in all_ids if identity not in assessed]})
    write(CASE / "observations.json", {"schema_version": "0.4", "frames": output,
        "policy": "Only native-timestamp masks from previous actual SAM inference. No nearest-frame mask copying, new model inference, or inferred absence.",
        "interpolated_observations": False,
        "timeline_interpretation": [
            {"start_s": 0, "end_s": 34, "description": "Camera handling; coarse 1fps assembly mask candidates."},
            {"start_s": 34, "end_s": 38.8, "description": "Open compartment; emerging batteries partly occluded."},
            {"start_s": 39.005633, "end_s": 42.409033, "description": "Two outward battery candidates; denser first-battery pilot."},
            {"start_s": 44.6, "end_s": 47.8, "description": "Table candidates; identity correspondence unresolved."},
            {"start_s": 47.4, "end_s": 49.82, "description": "First inward handling candidate, substantially occluded."},
            {"start_s": 50.0, "end_s": 53.5, "description": "Visible end/hand approach; second insertion and seating unverified."},
            {"start_s": 54, "end_s": 58.5, "description": "Cover manipulation/closing visible; no dense component masks here."},
            {"start_s": 59, "end_s": 74.240833, "description": "Camera handling and rear display; coarse camera assembly masks."}]})
    return output


def make_contact_sheets(frames, observations):
    colors = {"camera_00001": "#79aaff", "cover_00001": "#c5a3ff", "battery_00001": "#42e0c2", "battery_00002": "#ffa94f"}
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default(size=15)
    contacts = []
    # Coarse source samples summarize the entire video; action/track samples remain in the manifest.
    selected = [(f, o) for f, o in zip(frames, observations)
                if any(s["case_id"] == "camera-battery" for s in f["source_samples"])]
    for begin in range(0, len(selected), 24):
        group = selected[begin:begin + 24]
        sheet = Image.new("RGB", (1304, 46 + math.ceil(len(group) / 4) * 218), "#edf2f5")
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 12), "Full source / native SAM mask label anchors / no inferred cross-occlusion identities", font=font, fill="#173244")
        for j, (frame, observation) in enumerate(group):
            image = Image.open(CASE / frame["file"]).convert("RGB")
            d = ImageDraw.Draw(image)
            for obj in observation["objects"]:
                if obj["bbox_xyxy"]:
                    x0, y0, x1, y1 = obj["bbox_xyxy"]
                    color = colors.get(obj["id"], "#ffe05c")
                    d.rectangle((x0, y0, x1, y1), outline=color, width=3)
                    d.text((x0, max(0, y0-19)), obj["display_label"], fill=color, font=font, stroke_width=1, stroke_fill="#15232c")
            thumb = ImageOps.contain(image, (308, 174))
            x, y = 12 + j % 4 * 320, 46 + j // 4 * 218
            sheet.paste(thumb, (x, y))
            draw.text((x, y + 179), f"#{frame['index']:03d} / source {frame['timestamp_s']:.3f}s", font=font, fill="#173244")
        path = CASE / f"contact-sheet-{begin//24+1:02d}.jpg"
        sheet.save(path, quality=94)
        contacts.append(path.name)
    return contacts


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    source = read(ROOT / "data/camera-battery/manifest.json")
    frames = prepare_frames()
    write(CASE / "registry.json", registry(source["source"]["sha256"]))
    observations = prepare_observations(frames)
    contacts = make_contact_sheets(frames, observations)
    write(CASE / "manifest.json", {"schema_version": "0.4", "case_id": CASE.name,
        "source": {**source["source"], "file": "../camera-battery/source/video.mp4"},
        "sampling": {"method": "union_of_existing_native_pts_samples", "full_video_coarse_fps": 1,
                     "action_fps": 5, "withdrawal_fps": 15, "insertion_fps": 10, "interpolation": False},
        "frame_count": len(frames), "image_width": 960, "image_height": 540,
        "artifacts": {"frames": "frames/index.json", "registry": "registry.json", "observations": "observations.json", "contact_sheets": contacts},
        "limitations": ["Full-duration video with mixed-density sampled evidence, not dense 30fps masks or validated 6D poses.",
                        "No new learned-model inference performed by this preparation export."]})
    print(json.dumps({"case": str(CASE), "frames": len(frames), "native_observations": sum(len(f["objects"]) for f in observations),
                      "source_duration_s": source["source"]["video"]["duration_s"]}, indent=2))


if __name__ == "__main__":
    main()
