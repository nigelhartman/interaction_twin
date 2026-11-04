"""Predeclared source-sample exclusions for motion-fit diagnostics, no inference.

These holdouts test temporal prediction conditional on the selected native meshes
and the semantic topology. They do not establish unseen-object generalization,
metric calibration, independently correct segmentation or physical simulation.
"""
from __future__ import annotations
import re

FORMAT='interaction-twin-motion-evaluation-protocol'
VERSION=1
HEX=re.compile(r'[a-f0-9]{64}\Z')


def make_protocol(source_sha256, sample_indices, held_out_sample_indices=(), *, protected_sample_indices=()):
    rows=list(sample_indices); held=list(held_out_sample_indices); protected=list(protected_sample_indices)
    if not HEX.fullmatch(str(source_sha256)) or not rows or len(rows)>1200 or any(type(i)is not int or i<1 for i in rows) or len(set(rows))!=len(rows):
        raise ValueError('Bounded unique source samples and exact source hash required')
    if any(type(i)is not int for i in held+protected) or len(set(held))!=len(held) or not set(held+protected)<=set(rows):
        raise ValueError('Evaluation exclusions must name unique saved source samples')
    if set(held)&set(protected):raise ValueError('Native reconstruction anchors cannot be motion holdouts')
    if held and (min(rows) in held or max(rows) in held):raise ValueError('Motion holdouts require surrounding source samples')
    return {'format':FORMAT,'version':VERSION,'source_sha256':source_sha256,
        'mode':'predeclared_mask_holdout' if held else 'retrospective_diagnostic',
        'split_unit':'source_sample_index','fit_sample_indices':sorted(set(rows)-set(held)),
        'held_out_sample_indices':sorted(held),'protected_native_anchor_indices':sorted(set(protected)),
        'held_out_masks_used_for_pose_or_joint_fit':False if held else None,
        'held_out_prediction':'bounded_neighbor_LERP_SLERP_and_joint_FK_no_target_mask' if held else None,
        'scope':'motion prediction conditional on fixed generated geometry and semantic topology; not independent physical validation',
        'no_independent_physical_validation':True}


def choose_holdouts(sample_indices, *, protected_sample_indices=(), stride=5):
    rows=list(sample_indices)
    if type(stride)is not int or not 3<=stride<=20:raise ValueError('Holdout stride must be between3 and20')
    if not rows or any(type(i)is not int or i<1 for i in rows) or len(set(rows))!=len(rows):raise ValueError('Unique positive source samples required')
    protected=set(protected_sample_indices)|{min(rows),max(rows)}
    return [i for i in sorted(rows) if i%stride==0 and i not in protected]
