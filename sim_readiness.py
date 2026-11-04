"""Inspect an evidence procedure for simulation preparation, without simulation.

This module reads no referenced assets, makes no network calls, imports no
simulator, and cannot certify an asset for physical use. The v0.1 evidence
contract deliberately does not contain verified physical parameters. A browser
preview or a candidate camera pose never fills those gaps.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any


FORMAT = "interaction-twin-simulation-readiness"
VERSION = "0.1"


def _finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite number in interaction document")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


def _items(document: dict, key: str) -> list[dict]:
    value = document.get(key)
    if not isinstance(value, list) or not all(isinstance(i, dict) for i in value):
        raise ValueError(f"{key} must be an array of objects")
    return value


def _unique(items: list[dict], name: str) -> dict[str, dict]:
    result = {}
    for item in items:
        key = item.get("id")
        if not isinstance(key, str) or not key or key in result:
            raise ValueError(f"Missing or duplicate {name} ID")
        result[key] = item
    return result


def _check_input(document: dict) -> tuple[dict, dict]:
    """Fail closed on unsafe readiness claims; not a replacement for stage5 validation."""
    if not isinstance(document, dict):
        raise ValueError("Interaction document must be an object")
    _finite(document)
    if document.get("format") != "interaction-twin-evidence-procedure" or document.get("version") != "0.1":
        raise ValueError("Unsupported evidence procedure format or version")
    if not isinstance(document.get("case_id"), str) or not document["case_id"]:
        raise ValueError("case_id is required")
    tracks = _unique(_items(document, "tracklets"), "track")
    evidence = _unique(_items(document, "evidence"), "evidence")
    _unique(_items(document, "assets"), "asset")
    for key in ("relationships", "steps", "assertions", "motion"):
        _items(document, key)
    policy = document.get("runtime_policy", {})
    expected = {"simulation_ready": False, "physical_identity_confirmed": False,
                "state_between_evidence": "unknown", "pose_between_evidence": "absent",
                "pose_interpolation": False}
    if any(policy.get(key) != value for key, value in expected.items()):
        raise ValueError("Evidence-only runtime policy required; physical readiness cannot be promoted")
    for asset in document["assets"]:
        if asset.get("track_id") not in tracks:
            raise ValueError("Asset references an unknown track")
        if asset.get("physical_scale_known") is not False:
            raise ValueError("v0.1 assets must retain unknown physical scale")
    for row in document["motion"]:
        if row.get("track_id") not in tracks:
            raise ValueError("Motion references an unknown track")
        if row.get("physical_pose_verified") is not False:
            raise ValueError("v0.1 conditional motion cannot claim verified physical pose")
        matrix = row.get("camera_from_native_mesh")
        if matrix is not None:
            if (not isinstance(matrix, list) or len(matrix) != 4 or
                    not all(isinstance(r, list) and len(r) == 4 for r in matrix) or
                    not all(type(x) in (int, float) for r in matrix for x in r)):
                raise ValueError("Conditional pose must be a finite 4x4 numeric matrix or null")
    for relation in document["relationships"]:
        if relation.get("subject_track_id") not in tracks or relation.get("object_track_id") not in tracks:
            raise ValueError("Relationship references an unknown track")
        if any(ref not in evidence for ref in relation.get("evidence_ids", [])):
            raise ValueError("Relationship references unknown evidence")
    for step in document["steps"]:
        if any(ref not in tracks for ref in step.get("participant_track_ids", [])):
            raise ValueError("Step references an unknown track")
        if any(ref not in evidence for ref in step.get("evidence_ids", [])):
            raise ValueError("Step references unknown evidence")
    return tracks, evidence


def audit_simulation_readiness(document: dict) -> dict:
    """Return a deterministic evidence inventory and blocking physical unknowns.

    Call stage5.validate_interaction at the ingestion boundary when full schema,
    reference, timestamp and artifact-hash verification is required. This audit
    performs structural safety checks only and never opens paths in the JSON.
    """
    tracks, evidence = _check_input(document)
    assets = document["assets"]
    motion = document["motion"]
    pose_rows = [r for r in motion if r.get("camera_from_native_mesh") is not None]
    relationships = document["relationships"]
    checks = []

    def add(key: str, title: str, available: bool, detail: str, next_action: str,
            refs: list[str] | None = None, blocking: bool = True) -> None:
        checks.append({"id": key, "title": title, "status": "available" if available else "unknown",
                       "blocking": blocking and not available, "detail": detail,
                       "next_action": next_action, "evidence_ids": list(refs or [])})

    add("source_evidence", "Timestamped source evidence", bool(evidence),
        f"{len(evidence)} evidence records; these are observations and fallible model interpretations.",
        "Verify source hashes and inspect the evidence supporting each task transition.",
        list(evidence), False)
    add("visual_assets", "Generated visual assets", bool(assets),
        f"{len(assets)} asset references in native model units; this auditor does not open or validate meshes.",
        "Inspect shape, separate moving parts, verify assets and preserve visual-to-collision provenance.",
        blocking=False)
    add("conditional_motion", "Conditional camera-space pose candidates", bool(pose_rows),
        f"{len(pose_rows)} non-null conditional poses among {len(motion)} motion records; not a metric world trajectory.",
        "Verify common metric scale, transforms and identity; retain absent poses through unsupported intervals.",
        blocking=False)
    add("task_specification", "Evidence-linked task steps", bool(document["steps"]),
        f"{len(document['steps'])} demonstrated steps; pre/postconditions describe visual evidence, not robot success.",
        "Translate selected steps into reviewed simulator goals, reset states and measurable success/failure criteria.",
        blocking=False)
    requirements = [
        ("physical_identity", "Physical component identity", "Tracklets and generated assets are candidate identities, including possible swaps through occlusion.",
         "Verify which physical object and cavity each track represents before associating robot actions."),
        ("metric_scale", "Metric geometry and dimensions", "The evidence contract explicitly marks every asset's physical scale unknown.",
         "Provide a measured dimension or calibrated acquisition, store meters-per-native-unit and uncertainty per asset."),
        ("shared_frame", "Calibrated common world and robot frames", "Per-part camera-from-native transforms do not establish one shared physical coordinate system.",
         "Record axis conventions, calibrated intrinsics/extrinsics, fixed asset scale and world/robot/target transforms."),
        ("collision_geometry", "Collision geometry and cavity clearance", "No reviewed simulator collision representation is present. A convex hull may close a receptacle's opening.",
         "Build separately versioned collision geometry, preserve the real cavity, and validate clearance and nonpenetration."),
        ("mass_inertia", "Mass, center of mass and inertia", "No measured or validated mass distribution is available.",
         "Measure or explicitly estimate mass/material assumptions, compute a valid inertia tensor in a documented body frame, then validate."),
        ("joints", "Mechanical joints, axes and limits", "Semantic relations are proposals, not physical joint definitions; a removable object must not be automatically welded.",
         "Determine whether each mechanism is rigid, articulated or deformable; validate joint anchors, axes, limits and passive forces where applicable."),
        ("grasps", "Grasp and pregrasp frames", "The video shows human handling, not a verified robot grasp or end-effector pose.",
         "Generate robot-specific candidate grasps; check jaw width, approach clearance, contact normals and lift stability."),
        ("target_frames", "Receptacle and insertion frames", f"{len(relationships)} candidate relations do not locate a calibrated cavity mouth, approach axis or seated pose.",
         "Define object-relative entry/preinsert/seated frames, clearance and alignment tolerances with explicit evidence or measured input."),
        ("robot_ik", "Robot model, IK and collision-free reachability", "No robot articulation, actuator limits or verified reachable motion is attached to this procedure.",
         "Pin a licensed robot model and controller; solve IK and check self/environment collisions and joint limits for all waypoints."),
        ("contact", "Contact, friction and compliance", "Video evidence does not identify reliable friction, contact stiffness, magnetic seating or latch dynamics.",
         "Choose and record a contact model, validate on simple drop/grasp tests, and use compliance/force limits for contact-rich insertion."),
        ("seating", "Insertion and seating verification", "A visual model state such as inside does not prove full seating, electrical contact, charging or latch engagement.",
         "Define geometric and contact/functional success criteria; check release stability and distinguish occluded, blocked and seated outcomes."),
        ("simulation_validation", "Executed simulator validation", "No physics simulator or policy training has been run by this readiness auditor.",
         "Pin simulator/backend/version, test resets, passive settling, grasp/lift and contact insertion, and record seeds, metrics and failures."),
    ]
    for key, title, detail, action in requirements:
        add(key, title, False, detail, action)
    blocking = [c["id"] for c in checks if c["blocking"]]
    return {"format": FORMAT, "version": VERSION, "case_id": document["case_id"],
            "simulation_ready": False, "simulation_runnable": False,
            "mode": "evidence_inventory_and_preparation_only", "checks": checks,
            "blocking_requirement_ids": blocking,
            "inventory": {"tracklets": len(tracks), "visual_assets": len(assets),
                          "evidence_samples": len(evidence), "task_steps": len(document["steps"]),
                          "candidate_relationships": len(relationships),
                          "conditional_pose_samples": len(pose_rows), "motion_records": len(motion)},
            "verification": {"structural_safety_checked": True, "full_contract_validated": False,
                             "artifact_hashes_verified": False, "physical_parameters_verified": False,
                             "simulator_executed": False},
            "limitations": ["A readiness inventory is not a physics validation or a robot controller.",
                            "Missing values stay unknown; no masses, scales, joints, poses or tolerances are fabricated.",
                            "A Three.js robot animation is a presentation mockup and cannot clear these requirements."]}


def build_simulation_draft(document: dict) -> dict:
    """Create a non-runnable handoff with explicit null physical fields."""
    report = audit_simulation_readiness(document)
    assets_by_track: dict[str, list] = {}
    for asset in document["assets"]:
        assets_by_track.setdefault(asset["track_id"], []).append({
            key: copy.deepcopy(asset.get(key)) for key in ("id", "path", "sha256", "backend", "geometry_status")})
    components = [{"track_id": t["id"], "label": t.get("label", t["id"]),
                   "identity_status": t.get("identity_status"),
                   "visual_assets": assets_by_track.get(t["id"], []),
                   "meters_per_native_unit": None, "world_from_body": None,
                   "collision_meshes": None, "mass_kg": None, "center_of_mass_m": None,
                   "inertia_kg_m2": None, "contact_material": None,
                   "joint_definitions": None, "grasp_frames": None,
                   "receptacle_frames": None, "physical_evidence": None}
                  for t in document["tracklets"]]
    return {"format": "interaction-twin-simulation-draft", "version": VERSION,
            "case_id": document["case_id"], "simulation_runnable": False,
            "physical_execution_authorized": False,
            "purpose": "Non-runnable asset and task preparation; all physical parameters require verification.",
            "source": copy.deepcopy(document.get("source", {})),
            "target_simulator": None, "simulator_version": None, "physics_backend": None,
            "world_frame": None, "robot": None, "controller": None,
            "components": components, "candidate_relationships": copy.deepcopy(document["relationships"]),
            "demonstrated_steps": copy.deepcopy(document["steps"]),
            "conditional_camera_motion": copy.deepcopy(document["motion"]),
            "runtime_policy": copy.deepcopy(document["runtime_policy"]),
            "success_criteria": None, "failure_criteria": None, "validation_results": None,
            "blocking_requirement_ids": report["blocking_requirement_ids"],
            "limitations": report["limitations"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interaction", type=Path, help="Evidence interaction.json; referenced paths are never opened")
    parser.add_argument("--output", type=Path, help="Write readiness report here instead of stdout")
    parser.add_argument("--draft-output", type=Path, help="Also write a non-runnable simulator preparation manifest")
    args = parser.parse_args(argv)
    try:
        destinations = [p.resolve() for p in (args.output, args.draft_output) if p is not None]
        if args.interaction.resolve() in destinations or len(destinations) != len(set(destinations)):
            raise ValueError("Report/draft destinations must be distinct and must not overwrite the input")
        document = json.loads(args.interaction.read_text(encoding="utf-8"))
        report = audit_simulation_readiness(document)
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        else:
            print(encoded, end="")
        if args.draft_output:
            args.draft_output.parent.mkdir(parents=True, exist_ok=True)
            args.draft_output.write_text(json.dumps(build_simulation_draft(document), indent=2,
                                                    allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"Readiness audit failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
