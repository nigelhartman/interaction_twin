"""Read-only, source-bound cross-recording connection task drafts.

Connector types come from attributed owner declarations bound to exact approved
program/source hashes. They are never inferred from names, free prose or meshes.
Observed contact steps remain fallible saved-model proposals. No coordinates,
robot actions, electrical specifications or physical success are synthesized.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote

from curated_twin_catalog import CuratedTwinCatalog

ROOT = Path(__file__).resolve().parent
VERSION = "cross-recording-task-1"
CONFIG_PATH = "config/connection-interface-context.json"
ID = re.compile(r"[a-z][a-z0-9_-]{0,179}\Z")
HASH = re.compile(r"[a-f0-9]{64}\Z")
FAMILIES = {"usb_c", "eu_mains_unspecified"}
ROLES = {"plug", "receptacle"}
LOCATIONS = {"device_port", "device_end", "adapter_port", "adapter_end", "free_end", "mains_prongs", "wall_socket"}
INSTRUCTION = "Prepare a task to charge this phone."
MISSING = [
    {"id":"physical_connector_fit", "description":"Measure connector and port geometry, insertion axis, clearance and seating; USB-C labels alone do not establish fit."},
    {"id":"electrical_specifications", "description":"Verify adapter output, cable capability, phone requirements and power negotiation; charging is not verified by visible contact."},
    {"id":"cable_identity_and_ends", "description":"Confirm the cable selected from the charger recording has the intended free end; physical identity across recordings is unknown."},
    {"id":"flexible_cable_model", "description":"Replace the static cable shape with a validated deformable model or measured routing; do not rigidly transfer its whole mesh."},
    {"id":"metric_robot_and_scene", "description":"Calibrate scale, camera/world frames, robot reach, grasp, contact and collision geometry before robot planning."},
    {"id":"mains_setup_validation", "description":"An exact socket/prong standard, electrical state and approved execution setup are unavailable; the mains step stays blocked."},
    {"id":"independent_success_check", "description":"Use independent seating and charging-state checks; this draft contains no executable robot actions or demonstrated cross-recording success."},
]


def _reject(message):
    raise ValueError(message)


def _identifier(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        _reject("Invalid declared identifier")
    return value


def _text(value, maximum=800):
    if not isinstance(value, str) or not 0 < len(value) <= maximum or any(ord(c)<32 or ord(c)==127 for c in value):
        _reject("Invalid bounded descriptive text")
    return value


def _fields(value, required):
    if not isinstance(value, dict) or set(value) != set(required):
        _reject("Unexpected connection-context fields")


def _ids(value, maximum):
    if not isinstance(value,list) or not 0<len(value)<=maximum or len(set(value))!=len(value):
        _reject("Invalid bounded source-reference list")
    return [_identifier(i) for i in value]


def _encoded(value):
    return json.dumps(value,sort_keys=True,separators=(",", ":"),ensure_ascii=False,allow_nan=False).encode()


class ConnectionTaskCatalog:
    """No work-catalog fallback; every read uses the current curated source bytes."""
    def __init__(self, root=ROOT, *, catalog=None):
        self.root=Path(root).resolve()
        self.catalog=catalog if catalog is not None else CuratedTwinCatalog(self.root)
        if getattr(self.catalog,"curated",False) is not True:
            _reject("Connection tasks require the owner-curated library")

    def _context(self):
        path=self.root/CONFIG_PATH
        if (path.resolve()!=path or any(p.is_symlink() for p in [path,*path.parents]) or not path.is_file()
                or not 0<path.stat().st_size<=65536):
            _reject("Connection context is unavailable")
        raw=path.read_bytes()
        if len(raw)>65536:_reject("Connection context exceeds its size budget")
        def unique(items):
            result={}
            for k,v in items:
                if k in result:_reject("Duplicate connection-context field")
                result[k]=v
            return result
        value=json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _: _reject("Invalid connection JSON"))
        _fields(value,{"format","version","recordings"})
        if value["format"]!="interaction-twin-connection-context" or type(value["version"])is not int or value["version"]!=1:
            _reject("Invalid connection-context version")
        rows=value["recordings"]
        if not isinstance(rows,list) or len(rows)>100:_reject("Invalid connection-context record count")
        seen=set()
        for row in rows:
            _fields(row,{"twin_id","program_sha256","original_source_sha256","owner_context","interfaces"})
            key=_identifier(row["twin_id"])
            if key in seen:_reject("Duplicate connection-context recording")
            seen.add(key)
            if any(not isinstance(row[n],str) or not HASH.fullmatch(row[n]) for n in ["program_sha256","original_source_sha256"]):
                _reject("Invalid source hash in connection context")
            owner=row["owner_context"]
            _fields(owner,{"id","attribution","statement","status"})
            _identifier(owner["id"]);_text(owner["attribution"]);_text(owner["statement"])
            if owner["status"]!="owner_reported_not_independently_measured":_reject("Owner context is not a measured observation")
            interfaces=row["interfaces"]
            if not isinstance(interfaces,list) or not 1<=len(interfaces)<=16:_reject("Invalid interface count")
            interface_ids=set()
            for interface in interfaces:
                _fields(interface,{"id","component_id","role","connector_family","location","step_ids","evidence_ids"})
                iid=_identifier(interface["id"]);_identifier(interface["component_id"])
                if iid in interface_ids:_reject("Duplicate interface identifier")
                interface_ids.add(iid)
                if interface["role"] not in ROLES or interface["connector_family"] not in FAMILIES or interface["location"] not in LOCATIONS:
                    _reject("Unsupported typed connector declaration")
                _ids(interface["step_ids"],8);_ids(interface["evidence_ids"],16)
        return rows,hashlib.sha256(raw).hexdigest()

    def interfaces(self,twin_id):
        _identifier(twin_id)
        _,document,digest=self.catalog.load(twin_id)  # Validates approval and actual source/program bytes first.
        rows,context_hash=self._context()
        row=next((r for r in rows if r["twin_id"]==twin_id),None)
        base={"format":"interaction-twin-connection-interfaces","version":1,"twin_id":twin_id,
              "program_sha256":digest,"source_sha256":document["source"]["sha256"],
              "original_source_sha256":document["source"]["original_sha256"],
              "context_registry_sha256":context_hash,"interfaces":[],"owner_context":None,
              "extraction_method":"validated_typed_owner_context_plus_saved_step_evidence_no_title_inference",
              "physical_accuracy_verified":False,"simulation_ready":False,"robot_execution_available":False,"new_api_calls":0}
        if row is None:return base|{"status":"no_typed_interfaces_declared"}
        if row["program_sha256"]!=digest or row["original_source_sha256"]!=document["source"]["original_sha256"]:
            _reject("Connection declaration belongs to a different approved revision")
        tracks={r["id"] for r in document["tracklets"]}
        steps={r["id"]:r for r in document["steps"]}
        evidence={r["id"]:r for r in document["evidence"]}
        result=[]
        for declaration in row["interfaces"]:
            component=declaration["component_id"]
            if component not in tracks:_reject("Connection interface references an undeclared component")
            selected=[]
            for sid in declaration["step_ids"]:
                step=steps.get(sid)
                if step is None or component not in step["participant_track_ids"]:
                    _reject("Connection interface lacks a source-linked participant step")
                selected.append(step)
            cited={eid for s in selected for eid in s["evidence_ids"]}
            if not set(declaration["evidence_ids"])<=cited or not set(declaration["evidence_ids"])<=set(evidence):
                _reject("Connection interface cites evidence outside its recorded steps")
            item=copy.deepcopy(declaration)
            item.update(twin_id=twin_id,program_sha256=digest,type_attribution="owner_capture_context",
                owner_context_id=row["owner_context"]["id"],observed_contact_status="saved_model_proposal_not_verified_seating",
                pose=None,metric_dimensions=None,electrical_specifications=None,physical_fit_verified=False,
                evidence=[{"id":eid,"timestamp_s":evidence[eid]["timestamp_s"],"frame_sha256":evidence[eid]["frame_sha256"],
                    "uri":f"twin://{twin_id}/evidence/{eid}",
                    "frame_url":"/files/"+quote(evidence[eid]["frame_path"],safe="/"),
                    "video_url":"/files/"+quote(document["source"]["path"],safe="/")+"#t="+str(evidence[eid]["timestamp_s"])}
                    for eid in declaration["evidence_ids"]])
            result.append(item)
        return base|{"status":"typed_candidates_with_attributed_context","owner_context":copy.deepcopy(row["owner_context"]),"interfaces":result}

    def prepare(self,phone_twin_id,charger_twin_id,instruction=INSTRUCTION):
        _text(instruction,500)
        normalized=re.sub(r"[^a-z0-9 ]","",instruction.lower()).strip()
        if normalized not in {"prepare a task to charge this phone","prepare a task to charge the phone","prepare a task to charge my phone"}:
            _reject("This bounded compiler supports only 'Prepare a task to charge this phone'. Other goals need a separate typed planner.")
        if phone_twin_id==charger_twin_id:_reject("Choose two distinct approved recordings")
        phone=self.interfaces(phone_twin_id);charger=self.interfaces(charger_twin_id)
        by_id={"phone":{i["id"]:i for i in phone["interfaces"]},"charger":{i["id"]:i for i in charger["interfaces"]}}
        required={"phone":{"phone_port","phone_video_plug"},"charger":{"adapter_port","cable_adapter_end","cable_free_end","adapter_mains_input","wall_socket"}}
        if any(not required[role]<=set(by_id[role]) for role in required):
            _reject("Required typed phone, cable, adapter or socket interface declarations are unavailable")
        def link(role,iid):
            interface=by_id[role][iid]
            return {"twin_id":interface["twin_id"],"interface_id":iid,"component_id":interface["component_id"]}
        def candidate(cid,title,left,right,cross):
            a=by_id[left[0]][left[1]];b=by_id[right[0]][right[1]]
            if a["role"]==b["role"] or a["connector_family"]!=b["connector_family"]:
                _reject("Typed connector family or role does not form a candidate pair")
            return {"id":cid,"title":title,"interfaces":[link(*left),link(*right)],"connector_family":a["connector_family"],
                "status":"candidate_requires_validation","type_basis":"attributed_owner_capture_context",
                "cross_recording":cross,"physical_instance_identity":"unknown" if cross else "within_recording_tracklet_candidate",
                "physical_fit_verified":False,"electrical_compatibility":"unknown","charging_success":"unknown",
                "observed_pair_in_one_recording":not cross,
                "evidence":[{"twin_id":i["twin_id"],"program_sha256":i["program_sha256"],"step_ids":i["step_ids"],"evidence_ids":i["evidence_ids"]} for i in [a,b]]}
        connections=[
            candidate("cable_to_adapter","Cable → adapter",("charger","cable_adapter_end"),("charger","adapter_port"),False),
            candidate("cable_to_phone","Cable from recording A → phone from recording B",("charger","cable_free_end"),("phone","phone_port"),True),
            candidate("adapter_to_socket","Adapter → socket",("charger","adapter_mains_input"),("charger","wall_socket"),False)]
        bindings=[]
        for role,i in [("phone",phone),("charger",charger)]:
            _,doc,digest=self.catalog.load(i["twin_id"])
            assets=[]
            for asset in doc["assets"]:
                descriptor=self.catalog.asset(i["twin_id"],asset["track_id"])
                assets.append({k:descriptor[k] for k in ["track_id","sha256","uri"]})
            bindings.append({"role":role,"twin_id":i["twin_id"],"program_sha256":digest,
                "program_uri":f"twin://{i['twin_id']}/program","source":copy.deepcopy(doc["source"]),"assets":assets,
                "library_url":"/showcase/library.html?case="+quote(i["twin_id"],safe=""),
                "interfaces":i["interfaces"],"owner_context":i["owner_context"]})
        def source_step(role,sid):
            _,doc,digest=self.catalog.load(phone_twin_id if role=="phone" else charger_twin_id)
            step=next(s for s in doc["steps"] if s["id"]==sid)
            return {"twin_id":phone_twin_id if role=="phone" else charger_twin_id,"program_sha256":digest,
                "step_id":sid,"start_s":step["start_s"],"end_s":step["end_s"],"evidence_ids":step["evidence_ids"]}
        steps=[
            {"id":"inspect_parts","title":"Retrieve the phone, cable and adapter","kind":"inspection","depends_on":[],"connection_id":None,
             "source_steps":[source_step("phone","s3_show_edges"),source_step("charger","step_display_cable_end")],
             "completion_status":"artifact_retrieval_only_no_physical_success"},
            {"id":"prepare_adapter_end","title":"Reuse the demonstrated cable-to-adapter step","kind":"source_step_template","depends_on":["inspect_parts"],"connection_id":"cable_to_adapter",
             "source_steps":[source_step("charger","step_join_cable_adapter")],"completion_status":"blocked_until_fit_and_electrical_validation"},
            {"id":"prepare_phone_end","title":"Propose the other cable end for the phone port","kind":"cross_recording_composition_candidate","depends_on":["prepare_adapter_end"],"connection_id":"cable_to_phone",
             "source_steps":[source_step("charger","step_display_cable_end"),source_step("phone","s4_bring_cable"),source_step("phone","s5_contact_edge")],
             "completion_status":"not_demonstrated_across_recordings"},
            {"id":"prepare_power_setup","title":"Keep the demonstrated socket step as a blocked setup reference","kind":"source_step_template","depends_on":["prepare_phone_end"],"connection_id":"adapter_to_socket",
             "source_steps":[source_step("charger","step_insert_outlet")],"completion_status":"blocked_until_mains_setup_validation"},
            {"id":"verify_charging","title":"Require independent seating and charging confirmation","kind":"missing_success_check","depends_on":["prepare_power_setup"],"connection_id":None,
             "source_steps":[],"completion_status":"charging_not_observed_or_verified"},
        ]
        result={"format":"interaction-twin-cross-recording-task","version":1,"compiler_version":VERSION,
            "instruction":instruction,"intent":"prepare_charge_phone","status":"draft_requires_validation",
            "planning_method":"deterministic_typed_composition_not_live_llm","sources":bindings,"connection_candidates":connections,
            "task_draft":{"sequence_kind":"evidence_linked_templates_not_robot_commands","steps":steps},
            "missing_requirements":copy.deepcopy(MISSING),"robot_actions":[],
            "claims":{"cross_recording_composition_proposed":True,"same_physical_cable_verified":False,"physical_accuracy_verified":False,
                "electrical_compatibility_verified":False,"charging_success_verified":False,"simulation_ready":False,
                "robot_execution_available":False,"new_api_calls":0,"inference_performed":False},
            "context_registry_sha256":phone["context_registry_sha256"],
            "limitations":["Owner-declared connector types are attributed context, not a visual measurement or specification check.",
                "Source steps describe visible model proposals; hidden contact and seating remain unresolved.",
                "The two recordings use independent inferred coordinates; their meshes were not merged or rigidly aligned.",
                "Earlier and later adapter tracklets remain distinct; this task uses adapter_late without asserting re-identification.",
                "This task draft is data for an agent to inspect, not permission or an interface to execute a real robot."]}
        result["draft_sha256"]=hashlib.sha256(_encoded(result)).hexdigest()
        return result
