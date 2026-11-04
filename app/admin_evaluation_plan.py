"""Frozen private comparison basis for one automatic candidate lineage.

A reviewer-isolated final set does not become a motion or geometry holdout.
This module never refits, resegments, selects views, sends images or moves a split.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from agent_memory import digest, private_directory
from admin_memory_context import private_path
from object_program import ref

FORMAT='interaction-twin-private-evaluation-plan'
MAX_BYTES=160000


class EvaluationBasisChanged(ValueError):
    pass


def basis(scene,program,*,max_frames=12):
    from native_scene_audit import select_samples
    if scene['source']['sha256']!=program['source']['sha256']:
        raise ValueError('Evaluation source/program differs')
    components=scene.get('components',[])
    if not components:raise ValueError('An evaluation basis needs native scene samples')
    rows=components[0]['poses']
    samples=[{'source_index':row['source_sample_index'],'timestamp_s':row['timestamp_s']}for row in rows]
    if any([{'source_index':row['source_sample_index'],'timestamp_s':row['timestamp_s']}for row in c['poses']]!=samples for c in components):
        raise ValueError('Native parts have inconsistent evaluation sample clocks')
    indices=[s['source_index']for s in samples]
    if indices!=sorted(set(indices)):raise ValueError('Evaluation samples must be unique and ordered')
    chosen=select_samples(scene,max_frames)
    dev=[index for index,role in chosen if role=='development'];final=[index for index,role in chosen if role=='final_test']
    if not dev or not final or set(dev)&set(final):raise ValueError('Explicit disjoint development/final sample sets required')
    expected=sorted(t['id']for t in program['tracklets'])
    if not expected or len(expected)!=len(set(expected)):raise ValueError('Exact unique expected component inventory required')
    protocol=scene.get('evaluation_protocol')or{}
    mode=protocol.get('mode','retrospective_diagnostic')
    return {'source_sha256':program['source']['sha256'],
        'sample_index_universe_sha256':digest(indices),'sample_timing_sha256':digest(samples),
        'source_sample_indices':indices,'source_sample_times':samples,
        'development_source_indices':dev,'final_test_source_indices':final,'expected_component_ids':expected,
        'motion_fit_mode':mode,'motion_holdout_indices':sorted(protocol.get('held_out_sample_indices',[])),
        'final_test_scope':'reviewer isolation only; retrospective motion fit'if mode!='predeclared_mask_holdout'else
            'motion prediction conditional on generated geometry and topology; no independent physical validation',
        'max_audit_frames':max_frames,'physical_accuracy_verified':False,'final_test_scores_used_for_repair':False}


def load(reference,*,root):
    if not isinstance(reference,dict)or set(reference)!={'path','sha256','bytes'}:raise ValueError('Exact evaluation-plan reference required')
    path=private_path(reference['path'],root=root)
    if not path.is_relative_to(Path(root).resolve()/'.private/admin-evaluation-plans'):
        raise ValueError('Evaluation plan is outside its private namespace')
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb')as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode)or info.st_mode&0o077 or info.st_size>MAX_BYTES:raise ValueError('Bounded owner-only evaluation plan required')
        raw=stream.read(MAX_BYTES+1)
    if len(raw)!=reference['bytes']or hashlib.sha256(raw).hexdigest()!=reference['sha256']:
        raise ValueError('Frozen evaluation plan changed')
    value=json.loads(raw)
    if(value.get('format')!=FORMAT or value.get('version')!=1 or value.get('automatic_split_changes')is not False or
       value.get('basis_sha256')!=digest(value.get('basis'))):raise ValueError('Invalid frozen evaluation basis')
    return value


def get_or_create(job,scene,program,*,root,max_frames=12):
    root=Path(root).resolve();candidate=job.get('candidate_id')
    if not isinstance(candidate,str)or not re.fullmatch('candidate_[a-f0-9]{32}',candidate):
        raise ValueError('A stable private candidate lineage is required')
    if job['source_ref']['sha256']not in{program['source']['sha256'],program['source'].get('original_sha256')}:
        raise ValueError('Evaluation upload and normalized evidence source differ')
    current=basis(scene,program,max_frames=max_frames)
    folder=private_directory(root,'admin-evaluation-plans')/candidate
    if folder.is_symlink():raise ValueError('Evaluation lineage cannot use a symlink')
    folder.mkdir(mode=0o700,exist_ok=True);path=folder/'plan.json'
    if path.exists():
        reference=job.get('evaluation_plan_ref')
        if reference is None and job.get('improvement_round',0)!=0:
            raise EvaluationBasisChanged('Repair lineage lacks the first-round frozen evaluation reference')
        reference=reference or ref(path,root)
        if reference['path']!=path.relative_to(root).as_posix():raise EvaluationBasisChanged('Evaluation companion belongs to another candidate')
        saved=load(reference,root=root)
        if saved.get('candidate_id')!=candidate or saved['basis']!=current:
            raise EvaluationBasisChanged('Frozen evaluation sample universe, split, timing or component inventory differs; no resplit')
        return reference,saved
    # Only round zero may establish a new basis. A lost/deleted first-round plan
    # cannot be regenerated around a later repaired result.
    if job.get('improvement_round',0)!=0:
        raise EvaluationBasisChanged('Initial evaluation plan unavailable; no repair-dependent split selection')
    value={'format':FORMAT,'version':1,'candidate_id':candidate,'basis':current,'basis_sha256':digest(current),
        'initial_program_ref':scene['original_program'],'initial_scene_ref':job.get('initial_scene_ref'),
        'automatic_split_changes':False,'automatic_approval':False,'automatic_publication':False}
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    try:
        descriptor=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    except FileExistsError:
        return get_or_create(job,scene,program,root=root,max_frames=max_frames)
    with os.fdopen(descriptor,'wb')as stream:stream.write(raw)
    reference=ref(path,root);return reference,load(reference,root=root)


def verify_audit(reference,audit,*,root):
    saved=load(reference,root=root);expected=saved['basis']
    if audit['source']['sha256']!=expected['source_sha256']:
        raise EvaluationBasisChanged('Audit source differs from the frozen comparison basis')
    actual={role:sorted({r['source_index']for r in audit['samples']if r['role']==role})for role in ['development','final_test']}
    if(actual['development']!=sorted(expected['development_source_indices'])or actual['final_test']!=sorted(expected['final_test_source_indices'])or
       any(r['component_id']not in expected['expected_component_ids']for r in audit['samples'])):
        raise EvaluationBasisChanged('Audit moved development/final samples or changed the component basis')
    return saved


def comparison_key(reference,*,root):
    saved=load(reference,root=root)
    return digest({'evaluation_plan_sha256':reference['sha256'],'expected_component_ids':saved['basis']['expected_component_ids']})
