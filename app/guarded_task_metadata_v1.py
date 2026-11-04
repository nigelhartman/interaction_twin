"""Proposed conservative collection metadata for explicit nonrigid guards.

Pure derivation only; no source/scene/catalog/state mutation or inference.
Source-backed withholding is uncertainty, never a recovered flexible mechanism.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from task_taxonomy import derive_task_classes


def require(condition, message):
    if not condition:
        raise ValueError(message)


def derive(interaction, *, object_program=None, mechanism_proposals=None, scene=None):
    guard = (scene or {}).get('mechanics', {}).get('rigidity_feasibility')
    if guard is None:
        return derive_task_classes(interaction, object_program=object_program, mechanism_proposals=mechanism_proposals)
    mechanics = scene['mechanics']
    require(object_program is not None and object_program['evidence_program'] == scene['original_program'] == guard['interaction_ref'],
            'Nonrigid metadata requires the exact original program binding')
    require(guard['source_sha256'] == scene['source']['sha256'] == interaction['source']['sha256'] == object_program['source']['sha256'] and
            guard['proposal_ref'] == mechanics['proposal'] == object_program['proposal'], 'Nonrigid source or proposal differs')
    require(object_program['unresolved_connections'] == mechanics['unresolved_children'], 'Unresolved topology differs')
    for name, expected in [('rigidity_established', False), ('flexible_motion_recovered', False), ('native_geometry_modified', False),
                           ('unaffected_parent_and_removable_poses_preserved', True), ('physical_accuracy_verified', False)]:
        require(guard.get(name) is expected, 'Unsupported nonrigid guard claim: ' + name)
    unresolved = {edge['id']: edge for edge in mechanics['unresolved_children']}
    blocked = guard['withheld_joint_ids']
    require(isinstance(blocked, list) and 0 < len(blocked) <= 32 and len(blocked) == len(set(blocked)) and
            set(blocked) <= unresolved.keys() and len(guard['withheld_component_ids']) == len(blocked) and
            set(guard['withheld_component_ids']) == {unresolved[j]['child_id'] for j in blocked}, 'Withheld joint/component inventory differs')
    require(not set(blocked).intersection(j['id'] for j in mechanics['joints']), 'Withheld child still has a resolved rigid joint')
    direct = set()
    for finding in guard['findings']:
        edge = unresolved.get(finding['joint_id'])
        require(edge and edge['id'] in blocked and edge['parent_id'] == finding['parent_id'] and edge['child_id'] == finding['child_id'] and
                edge['type'] == finding['proposal_joint_type'] and edge['reason'] == 'rigid_motion_withheld_due_to_explicit_source_nonrigidity' and
                finding['reason'] == 'explicit_nonrigidity_in_persistent_joint_proposal' and finding['measured_nonrigidity_verified'] is False,
                'Explicit nonrigidity finding does not match the withheld edge')
        require(finding['joint_id'] not in direct and finding['declarations'], 'Duplicate or absent nonrigid declaration')
        direct.add(finding['joint_id'])
        for d in finding['declarations']:
            normalized = ' '.join(d['assertion_clause'].lower().split())
            require(normalized == d['normalized_assertion'] and d['assertion_clause'] in [clause.strip() for clause in re.split(r'[;,\.\n]+', d['source_limitation'])] and
                    re.fullmatch(r'(?:(?:the )?(?:child|part|component|moving part|moving component) (?:is|remains) )?(?:not rigid|non-rigid|nonrigid|deformable)', normalized),
                    'A complete positive source nonrigidity clause is required')
    require(direct, 'No explicit nonrigidity declarations')
    by_child = {unresolved[j]['child_id']: unresolved[j] for j in blocked}
    parts = {component['id']: component for component in scene['components']}
    for joint_id in blocked:
        edge = unresolved[joint_id]
        require(parts[edge['child_id']]['poses'] and all(row['status'] == 'unknown' and row['camera_from_native_mesh'] is None and
                row.get('joint_id') == joint_id and row.get('method') == 'source_nonrigidity_feasibility_withholding'
                for row in parts[edge['child_id']]['poses']), 'Nonrigid child motion must remain withheld')
        cursor, seen = edge, set()
        while cursor['id'] not in direct:
            require(cursor['reason'] == 'rigid_motion_withheld_due_to_unsupported_ancestor' and cursor['id'] not in seen and
                    cursor['parent_id'] in by_child, 'Unproven nonrigid ancestor chain')
            seen.add(cursor['id'])
            cursor = by_child[cursor['parent_id']]
    revised = copy.deepcopy(object_program)
    # Retain the unresolved relation itself. Its original proposed revolute type
    # is historical evidence, not a discoverable functioning hinge capability.
    revised['unresolved_connections'] = [{**edge, 'type': 'unknown'} if edge['id'] in blocked else edge
                                       for edge in revised['unresolved_connections']]
    proposals = copy.deepcopy(mechanism_proposals)
    if proposals is not None:
        require(proposals.get('interaction_sha256') == guard['interaction_ref']['sha256'] and
                proposals.get('source_sha256') == guard['source_sha256'], 'Proposal classification belongs to another source')
        proposals['joints'] = [joint for joint in proposals['joints'] if joint['id'] not in blocked]
    return derive_task_classes(interaction, object_program=revised, mechanism_proposals=proposals)

