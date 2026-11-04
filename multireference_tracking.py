"""Conservative selection and merging of independently initialized pose branches.

This module contains no category-specific geometry or manually chosen pixels.
Branches are alternative reference-conditioned observations, not reacquired IDs.
"""
from __future__ import annotations
import math
import numpy as np


def evaluation_frame(timestamp_s, tolerance=.02):
    """Reserve native samples nearest whole seconds before reference selection."""
    return abs(float(timestamp_s)-round(float(timestamp_s))) <= tolerance


def contiguous_nonempty_segment(nonempty, anchor):
    """An empty observation splits the learned sequence; never invent its crop."""
    flags=np.asarray(nonempty,bool)
    if not 0<=anchor<len(flags) or not flags[anchor]:
        raise ValueError('Reference must have an observed nonempty mask')
    start=end=anchor
    while start>0 and flags[start-1]:start-=1
    while end+1<len(flags) and flags[end+1]:end+=1
    return list(range(start,end+1))


def choose_references(candidates, old_reference_time, maximum=2, spacing_s=2.):
    """Image-only quality order; evaluation timestamps never select new refs."""
    if maximum < 0 or maximum > 3:
        raise ValueError('At most three bounded additional automatic references')
    eligible=[]
    for row in candidates:
        if row.get('scope_safe') is not True or row.get('mask_verdict')!='plausible_visible_mask':
            continue
        if evaluation_frame(row['timestamp_s']):
            continue
        if not math.isfinite(row['quality_score']):
            continue
        if abs(row['timestamp_s']-old_reference_time)<spacing_s:
            continue
        eligible.append(row)
    eligible.sort(key=lambda r:(-r['quality_score'],r['timestamp_s']))
    chosen=[]
    for row in eligible:
        if all(abs(row['timestamp_s']-r['timestamp_s'])>=spacing_s for r in chosen):
            chosen.append(row)
        if len(chosen)==maximum:
            break
    return chosen if maximum else []


def scope_veto(scope):
    if scope is None:
        return []
    if not scope['single_component_visible'] or not scope['component_boundary_plausible'] or scope['other_components_inside_mask']:
        return ['component_scope_model_rejected_mask']
    return []


def rotation_distance(a,b):
    delta=np.asarray(a)@np.asarray(b).T
    return float(np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2,-1,1))))


def merge_branches(branches):
    """Select nearest reference by time; withhold disagreeing supported branches.

    No averaging, interpolation, residual ranking or hidden transition is used.
    A rejected competing branch does not veto a passing one: only independent
    passing candidates can expose a disagreement. References never count as
    supported evaluation samples. Every rejected candidate remains upstream.
    """
    if not branches:
        raise ValueError('Need actual branch experiments')
    mesh={b['provenance']['mesh_sha256'] for b in branches}
    video={b['provenance']['source_video_sha256'] for b in branches}
    if len(mesh)!=1 or len(video)!=1:
        raise ValueError('Branches require the same frozen mesh and source video')
    ks=[np.asarray(b['intrinsics']['matrix']) for b in branches]
    if not all(np.allclose(ks[0],k) for k in ks[1:]):
        raise ValueError('A common camera matrix is required')
    rows={}
    for i,branch in enumerate(branches):
        for rec in branch['frames']:
            rows.setdefault(rec['sample_index'],[]).append((i,branch,rec))
    result=[]
    for sample,items in sorted(rows.items()):
        times={r['timestamp_s'] for _,_,r in items}
        if len(times)!=1:
            raise ValueError('A sample ID cannot mix source presentation timestamps')
        rec={'sample_index':sample,'timestamp_s':next(iter(times)), 'status':'unsupported', 'reasons':[],
             'branch_candidates':[{'branch_id':b['branch_id'],'status':r['status'],'reasons':r['reasons']} for _,b,r in items],
             'usable_pose_exported':False,'physical_pose_accuracy_measured':False,
             'evaluation_timestamp_reserved_before_reference_selection':evaluation_frame(next(iter(times)))}
        # Retain an inspection link even when no pose can be exported. This is
        # only an RGB/raw-projection diagnostic, never an accepted transform.
        _,diagnostic_branch,diagnostic=min(items,key=lambda x:(abs(x[2]['timestamp_s']-x[1]['reference_timestamp_s']),x[1]['branch_id']))
        if diagnostic.get('overlay_file'):
            rec.update(diagnostic_branch_id=diagnostic_branch['branch_id'],diagnostic_overlay_file=diagnostic['overlay_file'])
        passing=[x for x in items if x[2]['status']=='supported_conditional_motion_candidate']
        refs=[x for x in items if x[2]['status'].startswith('reference_')]
        conflicts=[]
        for a in range(len(passing)):
            for b in range(a+1,len(passing)):
                ia,ba,ra=passing[a];ib,bb,rb=passing[b]
                angle=rotation_distance(ra['candidate_rotation'],rb['candidate_rotation'])
                ta,tb=np.asarray(ra['candidate_translation']),np.asarray(rb['candidate_translation'])
                relative=float(np.linalg.norm(ta-tb)/max(np.linalg.norm(ta),np.linalg.norm(tb),1e-9))
                if angle>35 or relative>.25:
                    conflicts.append({'branches':[ba['branch_id'],bb['branch_id']], 'rotation_difference_deg':angle,'relative_translation_difference':relative})
        if conflicts:
            rec.update(status='unsupported_independent_reference_branches_disagree',reasons=['independent_reference_branch_disagreement'],branch_conflicts=conflicts)
        elif passing or refs:
            # Selection criterion is declared before observation outcomes.
            _,branch,chosen=min(passing or refs,key=lambda x:(abs(x[2]['timestamp_s']-x[1]['reference_timestamp_s']),x[1]['branch_id']))
            rec.update({k:v for k,v in chosen.items() if k not in {'sample_index','timestamp_s','status','reasons'}})
            rec.update(status=chosen['status'],reasons=chosen['reasons'],selected_branch_id=branch['branch_id'],
                       reference_initialization_independently_verified=False,unverified_orientation_branch=True)
            # A time used by any reference must not count as held-out temporal validation.
            if refs:
                rec['evaluation_timestamp_reserved_before_reference_selection']=False
        else:
            rec['reasons']=sorted({reason for _,_,r in items for reason in r['reasons']}) or ['no_passing_branch']
        result.append(rec)
    return result
