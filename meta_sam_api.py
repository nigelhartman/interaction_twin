"""Bounded hosted SAM 3.1 adapter with caller-owned transport and spending.

There are no credential, environment, network, retry or model-download operations
here. ``request(payload)`` must be supplied by the application's budget guard.
The official Meta parser is required before any request is dispatched.
"""
from __future__ import annotations

import base64
from dataclasses import asdict
from decimal import Decimal
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
from typing import Callable, Mapping

import numpy as np
from PIL import Image

MODEL = "sam-3.1"
ENDPOINT = "https://api.meta.ai/v1/responses"
PARSER_VERSION = "0.1.0"
TIME_DOMAIN = "source_video_presentation_seconds"
MAX_MEDIA_BYTES = 20 * 1024 * 1024  # Local allowance, not a claimed API limit.
MAX_FRAME_PIXELS = 4096 * 4096
MAX_OUTPUT_CHARS = 100 * 1024 * 1024
MAX_OBJECT_RECORDS = 20000
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
Request = Callable[[dict], dict]


class SamAPIError(ValueError):
    """Invalid input/output; never silently substitutes a local SAM result."""


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parser_preflight() -> dict:
    try:
        import meta_sam_parser
        version = importlib.metadata.version("meta-sam-parser")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise SamAPIError("Install the official meta-sam-parser==0.1.0 before paid requests") from exc
    if version != PARSER_VERSION:
        raise SamAPIError(f"Expected reviewed meta-sam-parser=={PARSER_VERSION}; found {version}")
    return {"package": "meta-sam-parser", "version": version,
            "network_accessed": False, "learned_inference_executed": False}


def estimate_cost(media: str, *, frame_count: int = 1) -> Decimal:
    """Standard USD published 2026-10-30; video has 50-frame minimum/call."""
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
        raise SamAPIError("frame_count must be a positive integer")
    if media == "image":
        if frame_count != 1:
            raise SamAPIError("An image request has exactly one image")
        return Decimal("0.0025")
    if media == "video":
        return Decimal(max(50, frame_count)) * Decimal("0.0002")
    raise SamAPIError("Unsupported media kind")


def _concept(concept: str) -> str:
    if not isinstance(concept, str) or not concept.strip() or len(concept) > 160:
        raise SamAPIError("Provide one short noun phrase, at most 160 characters")
    if any(char in concept for char in ["\n", "\r", ",", ";", "+", "<", ">"]):
        raise SamAPIError("Use one plain concept per call; lists and token prompts are separate contracts")
    return concept.strip()


def _dimensions(width: int, height: int) -> None:
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (width, height)):
        raise SamAPIError("Source width/height must be positive integers")
    if width * height > MAX_FRAME_PIXELS:
        raise SamAPIError("Source frame exceeds the local bounded pixel allowance")


def _source(source: Mapping) -> dict:
    source = dict(source)
    if not HEX64.fullmatch(source.get("video_sha256", "")):
        raise SamAPIError("Original source video SHA256 is required")
    pts = source.get("timestamp_s")
    if isinstance(pts, bool) or not isinstance(pts, (int, float)) or not math.isfinite(pts) or pts < 0:
        raise SamAPIError("Original finite nonnegative source PTS is required")
    sample = source.get("sample_index")
    if isinstance(sample, bool) or not isinstance(sample, int) or sample < 1:
        raise SamAPIError("Positive original sample_index is required")
    if source.get("time_domain") != TIME_DOMAIN:
        raise SamAPIError("Source PTS time_domain is required; API frame index is not a source timestamp")
    return source


def _data_url(path: Path, media: str) -> str:
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp", ".mp4": "video/mp4", ".mov": "video/quicktime"}.get(path.suffix.lower())
    if mime is None or not mime.startswith(media + "/"):
        raise SamAPIError("Unsupported bounded media container")
    if not 0 < path.stat().st_size <= MAX_MEDIA_BYTES:
        raise SamAPIError("Media exceeds the local bounded upload allowance")
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_payload(concept: str, media_part: dict) -> dict:
    """Non-streaming is documented for images and videos; no hidden upload."""
    return {"model": MODEL, "stream": False,
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": _concept(concept)}, dict(media_part)]}],
            "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"}}


def _output_text(response: dict) -> str:
    if not isinstance(response, dict) or response.get("status") != "completed":
        raise SamAPIError("SAM response did not complete; preserve it and do not promote partial masks")
    if response.get("model", MODEL) != MODEL:
        raise SamAPIError("Response identifies an unexpected model")
    lanes = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        if item.get("status", "completed") != "completed":
            raise SamAPIError("SAM output message did not complete")
        for part in item.get("content", []):
            if part.get("type") == "refusal":
                raise SamAPIError("SAM refused this request")
            if part.get("type") == "output_text":
                lanes.append(part.get("text"))
    # Raw REST Responses JSON normally has output[].content[].text. An empty
    # output list plus explicit empty output_text is also a valid zero match.
    if not lanes and response.get("output_text") == "":
        lanes = [""]
    if len(lanes) != 1 or not isinstance(lanes[0], str):
        raise SamAPIError("Expected exactly one finalized SAM output_text lane")
    if len(lanes[0]) > MAX_OUTPUT_CHARS:
        raise SamAPIError("SAM output exceeds the local bounded response allowance")
    return lanes[0]


def decode_response(response: dict, *, width: int, height: int, media: str,
                    sources: list[dict]) -> tuple[dict, list[np.ndarray]]:
    """Decode genuine wire output; masks are source-size binary arrays.

    ``sources[N]`` is the explicit original-PTS mapping for submitted frame N.
    Missing video lines remain absent: no interpolation, identity merge or fill.
    """
    parser_preflight()
    import meta_sam_parser as official
    _dimensions(width, height)
    if media not in ("image", "video") or not sources or (media == "image" and len(sources) != 1):
        raise SamAPIError("Provide explicit source mapping for each submitted frame")
    sources = [_source(source) for source in sources]
    raw = _output_text(response)
    if raw.count("<|mask;") > MAX_OBJECT_RECORDS:
        raise SamAPIError("Too many object records for bounded decoding")
    # The public parser normalizes boxes but drops wire w/h. Verify these first,
    # and cap declared raster area before its decoder allocates buffers.
    for token in re.findall(r"<\|box;([^|]*)\|>", raw):
        fields = dict(field.split("=", 1) for field in token.split(";") if "=" in field)
        if fields.get("w") != str(width) or fields.get("h") != str(height):
            raise SamAPIError("Returned box source dimensions differ from submitted media")
    for mask_h, mask_w in re.findall(r";data=(\d+),(\d+),", raw):
        if not 0 < int(mask_h) * int(mask_w) <= width * height:
            raise SamAPIError("Returned mask raster exceeds source area")
    factory = official.image_segmentation_format if media == "image" else official.video_segmentation_format
    parser = factory().create_parser()
    parser.push(raw)
    parsed = parser.finish(official.CompletedOutcome()).result
    diagnostics = [asdict(item) for item in parsed.diagnostics]
    if diagnostics or any(record.kind == "text" for record in parsed.records):
        raise SamAPIError("SAM parser reported malformed/unknown output; preserve raw response for review")
    masks, instances, seen = [], [], set()
    for record in parsed.records:
        if record.kind != "mask":
            continue
        frame = 0 if record.frame is None else record.frame.frame_index
        if frame < 0 or frame >= len(sources):
            raise SamAPIError("API frame index has no original source PTS mapping")
        identity = (frame, record.object_id)
        if identity in seen:
            raise SamAPIError("Duplicate object/frame output requires explicit revision review")
        seen.add(identity)
        left, top, right, bottom = [int(getattr(record.bounds, key)) for key in ("left", "top", "right", "bottom")]
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise SamAPIError("Returned mask bounds exceed source pixels")
        if (record.mask.width, record.mask.height) != (right-left, bottom-top):
            raise SamAPIError("Mask raster differs from pixel box; no implicit resizing is allowed")
        raster = np.frombuffer(official.decode_mask_to_raster(record.mask), dtype=np.uint8)
        full = np.zeros((height, width), dtype=bool)
        full[top:bottom, left:right] = raster.reshape(record.mask.height, record.mask.width) > 0
        masks.append(full)
        instances.append({"frame_index": frame, "instance_id": str(record.object_id),
                          "identity_scope": "within_response_visible_track_not_verified_physical_identity",
                          "source": sources[frame], "box_xyxy_half_open": [left, top, right, bottom],
                          "mask_encoding": record.mask.encoding, "confidence": record.confidence,
                          "mask_foreground_pixels": int(full.sum()), "mask_raster_shape_hw": [height, width]})
    usage = response.get("usage")
    if not isinstance(usage, dict):
        usage = {}  # Unknown usage stays explicit, never becomes zero billed cost.
    result = {"schema_version": "1.0", "status": "completed" if instances else "completed_zero_matches",
              "backend": "Meta Model API SAM 3.1", "model": response.get("model", MODEL),
              "response_id": response.get("id"), "provider": "Meta Model API hosted inference",
              "media": media, "width": width, "height": height,
              "parser": {"package": "meta-sam-parser", "version": PARSER_VERSION},
              "learned_inference_executed": True, "usage": usage,
              "usage_status": "reported" if usage else "unknown_do_not_release_budget_reservation",
              "frame_sources": sources, "instances": instances,
              "diagnostics": diagnostics, "physical_accuracy_measured": False,
              "limitations": ["Segmentation is 2D, not 3D reconstruction or contact evidence.",
                              "Object identity can fail at entry, exit or complete occlusion.",
                              "No older SAM2 mask audit is applied to these new masks."]}
    return result, masks


def _run(payload: dict, *, request: Request, output: Path, sources: list[dict],
         width: int, height: int, media: str, provenance: dict) -> dict:
    parser_preflight()  # Must fail before calling a paid transport.
    if not callable(request):
        raise SamAPIError("A caller-owned budget-guarded request callback is required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    response = request(payload)  # Exactly one attempt. The caller records all cost.
    (output / "response.json").write_text(json.dumps(response, indent=2) + "\n")
    result, masks = decode_response(response, width=width, height=height, media=media, sources=sources)
    if provenance.get("prompt_type") == "model_generated_box_prompts":
        allowed_ids = {str(obj["id"]) for obj in provenance["box_prompts"]}
        if any(instance["instance_id"] not in allowed_ids for instance in result["instances"]):
            raise SamAPIError("Returned box-prompt object id was not present in the model prompt")
    result["input_provenance"] = provenance
    result["request"] = {"endpoint": ENDPOINT, "model": MODEL,
                         "concept": payload["input"][0]["content"][0]["text"],
                         "metadata": payload["metadata"], "stream": False,
                         "payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                         "transport_and_budget_owned_by_caller": True}
    if provenance.get("prompt_type") == "model_generated_box_prompts":
        result["request"]["object_prompt_text"] = result["request"].pop("concept")
        result["request"]["prompt_type"] = "model_generated_box_prompts"
    result["raw_response_sha256"] = sha256(output / "response.json")
    for index, (record, mask) in enumerate(zip(result["instances"], masks), 1):
        path = output / f"instance-{index:06d}-mask.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
        record.update(mask=path.name, mask_sha256=sha256(path))
    (output / "metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def segment_image(input_manifest: Path, concept: str, output: Path, request: Request,
                  *, prompt_origin: str) -> dict:
    """Send one portable source anchor; old mask is comparison-only, not a seed."""
    from scripts.sam31_segment import read_input
    inputs, rgb, _reference = read_input(Path(input_manifest))
    source = _source(inputs["manifest"]["source"])
    if not isinstance(prompt_origin, str) or not prompt_origin.strip():
        raise SamAPIError("Prompt provenance is required")
    _dimensions(inputs["width"], inputs["height"])
    path = Path(inputs["files"]["image"]["path"])
    payload = build_payload(concept, {"type": "input_image", "image_url": _data_url(path, "image")})
    provenance = {"manifest_sha256": inputs["manifest_sha256"], "image_sha256": sha256(path),
                  "prompt_origin": prompt_origin, "reference_mask_passed_to_model": False,
                  "inherited_reference_mask_audit_reused": False, "manual_seed_boxes_used": False}
    return _run(payload, request=request, output=output, sources=[source], width=rgb.shape[1],
                height=rgb.shape[0], media="image", provenance=provenance)


def _model_box_prompt(objects: list[dict], *, width: int, height: int,
                      frame_count: int, prompt_origin: str, media: str) -> tuple[str, list[dict]]:
    """Format caller-supplied model boxes; never creates or adjusts object poses.

    Model coordinates are continuous, half-open native pixels. Outward rounding
    includes the full proposed region: floor left/top, ceil right/bottom. Meta's
    official builder converts rounded half-open right/bottom to inclusive wire
    x2/y2 by subtracting one. There is no hidden clipping or crop conversion.
    """
    parser_preflight()
    import meta_sam_parser as official
    _dimensions(width, height)
    if not isinstance(prompt_origin, str) or "model" not in prompt_origin.lower():
        raise SamAPIError("Box prompts require explicit model-generated provenance")
    if any(word in prompt_origin.lower() for word in ["manual", "hand_annotation", "assistant_annotation"]):
        raise SamAPIError("Manual or assistant box annotations are not supported by this automatic adapter")
    if media not in ("image", "video") or type(frame_count) is not int or not 1 <= frame_count <= 5000:
        raise SamAPIError("Invalid bounded image/video frame count")
    if not isinstance(objects, list) or not 1 <= len(objects) <= 128:
        raise SamAPIError("Provide 1 to 128 model box observations, at most 16 distinct objects")
    groups, normalized, seen = {}, [], set()
    for obj in objects:
        if not isinstance(obj, dict):
            raise SamAPIError("Each model box observation must be an object")
        identifier = obj.get("id")
        if type(identifier) is not int or identifier < 0 or identifier > (2**53 - 1):
            raise SamAPIError("Object id must be a nonnegative safe integer")
        frame = obj.get("frame_index", 0)
        if type(frame) is not int or not 0 <= frame < frame_count or (media == "image" and frame != 0):
            raise SamAPIError("Box prompt frame must map to an actual submitted source frame")
        if (identifier, frame) in seen:
            raise SamAPIError("Duplicate object/frame box prompt")
        seen.add((identifier, frame))
        box = obj.get("box_xyxy")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise SamAPIError("Model box must have four native half-open coordinates")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in box):
            raise SamAPIError("Model box coordinates must be finite numbers")
        left, top, right, bottom = box
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise SamAPIError("Model box lies outside source pixels; no implicit clipping")
        rounded = [math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom)]
        groups.setdefault(identifier, []).append(official.FramePrompt(
            frame=frame, box=official.PromptBox(*rounded)))
        normalized.append({"id": identifier, "frame_index": frame,
                           "model_box_xyxy_half_open": list(box),
                           "rounded_box_xyxy_half_open": rounded,
                           "wire_box_xyxy_inclusive": [rounded[0], rounded[1], rounded[2]-1, rounded[3]-1],
                           "rounding_policy": "outward_floor_left_top_ceil_right_bottom_no_clipping"})
    if len(groups) > 16:
        raise SamAPIError("At most 16 distinct objects can be prompted in one SAM request")
    text = official.build_object_prompt(
        size=(width, height),
        objects=[official.PromptObject(id=identifier, prompts=prompts)
                 for identifier, prompts in groups.items()])
    return text, normalized


def _box_payload(prompt: str, media_part: dict) -> dict:
    # Box/point grammar is a separate prompt type. It must not be mixed with a
    # noun phrase; the concept-only builder intentionally still rejects tokens.
    return {"model": MODEL, "stream": False,
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt}, dict(media_part)]}],
            "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"}}


def segment_image_boxes(input_manifest: Path, objects: list[dict], output: Path,
                        request: Request, *, prompt_origin: str) -> dict:
    """Segment model-proposed image boxes with their caller-selected numeric ids.

    ``objects`` contains ``id`` and native half-open ``box_xyxy`` for each
    automatically proposed component. Existing masks are never used as seeds.
    """
    from scripts.sam31_segment import read_input
    inputs, rgb, _reference = read_input(Path(input_manifest))
    source = _source(inputs["manifest"]["source"])
    prompt, normalized = _model_box_prompt(objects, width=inputs["width"], height=inputs["height"],
                                           frame_count=1, prompt_origin=prompt_origin, media="image")
    path = Path(inputs["files"]["image"]["path"])
    payload = _box_payload(prompt, {"type": "input_image", "image_url": _data_url(path, "image")})
    provenance = {"manifest_sha256": inputs["manifest_sha256"], "image_sha256": sha256(path),
                  "prompt_origin": prompt_origin, "prompt_type": "model_generated_box_prompts",
                  "box_prompts": normalized, "reference_mask_passed_to_model": False,
                  "inherited_reference_mask_audit_reused": False, "manual_seed_boxes_used": False,
                  "model_box_accuracy_verified": False}
    return _run(payload, request=request, output=output, sources=[source], width=rgb.shape[1],
                height=rgb.shape[0], media="image", provenance=provenance)


def segment_video(video_path: Path, concept: str, output: Path, request: Request,
                  *, width: int, height: int, frame_sources: list[dict],
                  prompt_origin: str, file_id: str | None = None) -> dict:
    """One bounded derivative; source mapping must be prepared by timestamp-aware extraction.

    Supports existing uploaded file IDs or a bounded data URL. Does not perform
    Files API uploads, video resampling, codec conversion or inferred PTS joins.
    Caller must verify the container's displayed dimensions/frame order first.
    """
    _dimensions(width, height)
    if not frame_sources or len(frame_sources) > 5000:
        raise SamAPIError("Provide between 1 and 5000 explicit frame-to-source records")
    sources = [_source(source) for source in frame_sources]
    if not isinstance(prompt_origin, str) or not prompt_origin.strip():
        raise SamAPIError("Prompt provenance is required")
    if any(item["video_sha256"] != sources[0]["video_sha256"] for item in sources):
        raise SamAPIError("One video request must map to one original source recording")
    if any(b["timestamp_s"] <= a["timestamp_s"] for a, b in zip(sources, sources[1:])):
        raise SamAPIError("Submitted video mapping must preserve strictly increasing original PTS")
    video_path = Path(video_path)
    if file_id is not None:
        if not re.fullmatch(r"file-[A-Za-z0-9_-]+", file_id):
            raise SamAPIError("Invalid pre-uploaded Files API identifier")
        part = {"type": "input_video", "file_id": file_id}
    else:
        part = {"type": "input_video", "video_url": _data_url(video_path, "video")}
    provenance = {"submitted_video_sha256": sha256(video_path), "prompt_origin": prompt_origin,
                  "frame_count": len(sources), "original_video_sha256": sources[0]["video_sha256"],
                  "frame_mapping_policy": "explicit submitted frame index to original source PTS; no interpolation",
                  "reference_mask_passed_to_model": False, "manual_seed_boxes_used": False}
    return _run(build_payload(concept, part), request=request, output=output, sources=sources,
                width=width, height=height, media="video", provenance=provenance)


def segment_video_boxes(video_path: Path, objects: list[dict], output: Path, request: Request,
                        *, width: int, height: int, frame_sources: list[dict], prompt_origin: str,
                        file_id: str | None = None, propagation_direction: str = "both") -> dict:
    """Track model-proposed boxes at explicit submitted-frame indices.

    Repeating an id on different frames supplies additional model observations;
    repeating an id/frame is rejected. Frame indices refer to the submitted
    encoded clip and map through ``frame_sources`` to original video PTS.
    """
    _dimensions(width, height)
    if not frame_sources or len(frame_sources) > 5000:
        raise SamAPIError("Provide between 1 and 5000 explicit frame-to-source records")
    sources = [_source(source) for source in frame_sources]
    if any(item["video_sha256"] != sources[0]["video_sha256"] for item in sources):
        raise SamAPIError("One video request must map to one original source recording")
    if any(b["timestamp_s"] <= a["timestamp_s"] for a, b in zip(sources, sources[1:])):
        raise SamAPIError("Submitted video mapping must preserve strictly increasing original PTS")
    prompt, normalized = _model_box_prompt(objects, width=width, height=height, frame_count=len(sources),
                                           prompt_origin=prompt_origin, media="video")
    if propagation_direction not in {"both", "forward", "backward"}:
        raise SamAPIError("Unsupported SAM propagation direction")
    if propagation_direction == "backward" and min(obj["frame_index"] for obj in normalized) == 0:
        raise SamAPIError("Backward tracking requires the earliest model box after frame zero")
    video_path = Path(video_path)
    if file_id is not None:
        if not re.fullmatch(r"file-[A-Za-z0-9_-]+", file_id):
            raise SamAPIError("Invalid pre-uploaded Files API identifier")
        part = {"type": "input_video", "file_id": file_id}
    else:
        part = {"type": "input_video", "video_url": _data_url(video_path, "video")}
    payload = _box_payload(prompt, part)
    payload["metadata"]["propagation_direction"] = propagation_direction
    provenance = {"submitted_video_sha256": sha256(video_path), "prompt_origin": prompt_origin,
                  "prompt_type": "model_generated_box_prompts", "box_prompts": normalized,
                  "propagation_direction": propagation_direction, "frame_count": len(sources),
                  "original_video_sha256": sources[0]["video_sha256"],
                  "frame_mapping_policy": "explicit submitted frame index to original source PTS; no interpolation",
                  "reference_mask_passed_to_model": False, "manual_seed_boxes_used": False,
                  "model_box_accuracy_verified": False}
    return _run(payload, request=request, output=output, sources=sources,
                width=width, height=height, media="video", provenance=provenance)
