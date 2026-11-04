#!/usr/bin/env python3
"""CPU-only private fit of the exact accounted source-only hinge revision."""
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import laptop_hinge_relocalization_v2 as L
import laptop_hinge_omission_recovery_v1 as R
import mechanism_pipeline as P
import object_program as O
import sam3d_common_scene as S

AUTHORIZED_PLAN = '672ba9548203bae26b3c52ccc3d742091ebea7c5fa0dc915d4ac023745f719b7'


def build():
    doc, _, folder = L.verified_plan()
    R.require(L.M.sha(folder / 'plan.json') == AUTHORIZED_PLAN, 'Exact root-reviewed packet required')
    proposal_path = folder / 'attempt/proposals.json'
    proposal = O.load_proposals(proposal_path.relative_to(ROOT).as_posix(), R.PROGRAM, root=ROOT)
    receipt_path = folder / 'attempt/request/meta-request.json'
    receipt = R.read(receipt_path)
    R.require(receipt.get('status') == 'complete_accounted' and receipt.get('payload_sha256') == doc['wire_provenance']['payload_sha256']
              and receipt.get('image_inputs') == doc['wire_provenance']['image_inputs'], 'Exact accounted revised source request required')
    original_run = ROOT / 'data/flow-runs/run-capture-laptop-20261103-v1/run.json'
    original_run_ref = R.ref(original_run, ROOT)
    output = folder / 'local-fit-v1'
    R.require(not output.exists(), 'Preserve earlier local fit versions')
    output.mkdir()
    if not all(j['joint_type'] == 'revolute' and j['attachment'] == 'persistent' and j['hinge_hints'] for j in proposal['joints']):
        result = {'status': 'revised_source_hinge_unknown_no_fit', 'proposal': R.ref(proposal_path, ROOT),
                  'new_api_calls': 0, 'new_gpu_calls': 0, 'physical_accuracy_verified': False, 'published': False}
    else:
        # Exact native paths from the verified previous private fit, never a
        # global latest-preview lookup or an alternative generated mesh.
        old = R.read(ROOT / R.OUTPUT / 'private-scene-result.json')
        old_scene = P.checked(old['scene']['private_scene'], ROOT)
        native_source = R.read(old_scene)
        metadata = [ROOT / p for p in native_source['model_metadata_paths']]
        R.require({R.read(p)['object_id'] for p in metadata} == {'laptop_base', 'laptop_lid'}, 'Exact two retained native parts required')
        for item in native_source['inputs']:
            R.require(S.ref(item['path'], ROOT) == item, 'Previous native source/geometry bytes changed')
        scene = P.constrained_scene(R.PROGRAM, R.ref(proposal_path, ROOT), root=ROOT, metadata_paths=metadata)
        descriptive = P.export_scene(scene['manifest'], root=ROOT, publish=False)
        result = {'status': scene['status'], 'proposal': R.ref(proposal_path, ROOT), 'scene': scene,
                  'descriptive_export': descriptive, 'scope': 'source_only_relocalized_native_base_and_lid_no_cable',
                  'geometry_unchanged': True, 'manual_joint_coordinates_used': False,
                  'new_api_calls': 0, 'new_gpu_calls': 0, 'published': False, 'physical_accuracy_verified': False}
    R.require(R.ref(original_run, ROOT) == original_run_ref, 'Original main run changed concurrently')
    result.update(original_main_run=original_run_ref, original_main_run_changed=False,
                  source_accounting_receipt=R.ref(receipt_path, ROOT))
    R.write_new(output / 'result.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(build()), flush=True)
