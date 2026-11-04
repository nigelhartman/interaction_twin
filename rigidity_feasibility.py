"""Private conservative withholding after an explicit source nonrigidity claim.

This postprocessor does not classify objects, fit flexible motion, modify a mesh
or accept a joint. It preserves the proposal and base scene and only removes
unsupported rigid child motion. No inference, transport or publication exists.
"""
from __future__ import annotations
import copy,json,re
from pathlib import Path
import native_scene_audit as A
import object_program as O
import sam3d_common_scene as S
import upgrade_bundles as U

ROOT=Path(__file__).resolve().parent
FORMAT='interaction-twin-source-rigidity-feasibility'
CODE=['rigidity_feasibility.py','native_scene_audit.py','sam3d_common_scene.py','sam3d_mechanisms.py','object_program.py']
DECLARATION=re.compile(r'(?:(?:the )?(?:child|part|component|moving part|moving component) (?:is|remains) )?(not rigid|non-rigid|nonrigid|deformable)\Z')


def require(value,message):
    if not value:raise ValueError(message)


def declarations(proposals):
    """Positive complete clauses only, scoped to persistent proposal limitations.

    Other prose is intentionally unsupported, including object/material names,
    substring matches, “may be” and negated deformability. This guard is narrow;
    no finding must never be interpreted as proof of rigidity.
    """
    findings=[]
    require(isinstance(proposals.get('joints'),list)and len(proposals['joints'])<=24,'Bounded source joint proposals required')
    for joint in proposals['joints']:
        if joint.get('attachment')!='persistent':continue
        limitations=joint.get('limitations',[]);require(isinstance(limitations,list)and len(limitations)<=24,'Bounded proposal limitations required')
        matches=[]
        for index,limitation in enumerate(limitations):
            require(isinstance(limitation,str)and len(limitation)<=4000,'Bounded source limitation text required')
            for clause in re.split(r'[;,\.\n]+',limitation):
                normalized=' '.join(clause.strip().casefold().split())
                if DECLARATION.fullmatch(normalized):
                    matches.append({'limitation_index':index,'source_limitation':limitation,'assertion_clause':clause.strip(),'normalized_assertion':normalized})
        if matches:findings.append({'joint_id':joint['id'],'parent_id':joint['parent_id'],'child_id':joint['child_id'],'proposal_joint_type':joint['joint_type'],
            'reason':'explicit_nonrigidity_in_persistent_joint_proposal','declarations':matches,'measured_nonrigidity_verified':False})
    return findings


def apply(scene,proposals):
    """Pure copy; retain inspectable assets, parents, other poses and raw proposal."""
    findings=declarations(proposals);result=copy.deepcopy(scene);mechanics=result['mechanics'];parts={c['id']:c for c in result['components']}
    persistent=[j for j in proposals['joints']if j['attachment']=='persistent'];ids=[j['id']for j in persistent];children=[j['child_id']for j in persistent]
    require(len(ids)==len(set(ids))and len(children)==len(set(children)),'Unique persistent source joint/child required')
    require(all(j['parent_id']in parts and j['child_id']in parts and j['parent_id']!=j['child_id']for j in persistent),'Known distinct source components required')
    blocked={f['joint_id']for f in findings};blocked_children={f['child_id']for f in findings}
    while True:
        dependent=[j for j in persistent if j['parent_id']in blocked_children and j['id']not in blocked]
        if not dependent:break
        blocked.update(j['id']for j in dependent);blocked_children.update(j['child_id']for j in dependent)
    if not blocked:return result,{'findings':[],'withheld_joint_ids':[],'withheld_component_ids':[],'rigidity_established':False}
    fits={j['id']:j for j in mechanics['joints']};unresolved={j['id']:j for j in mechanics['unresolved_children']}
    require(set(fits).isdisjoint(unresolved)and set(fits)|set(unresolved)==set(ids),'Base mechanics must resolve or withhold every persistent proposal')
    direct={f['joint_id']:f for f in findings}
    for joint in persistent:
        if joint['id']not in blocked:continue
        if joint['id']in direct:
            reason='rigid_motion_withheld_due_to_explicit_source_nonrigidity'
            diagnostic='Persistent proposal declares: '+'; '.join(d['assertion_clause']for d in direct[joint['id']]['declarations'])
        else:
            reason='rigid_motion_withheld_due_to_unsupported_ancestor'
            diagnostic='Persistent parent motion is withheld by an explicit source-bound nonrigidity declaration.'
        unresolved[joint['id']]={'id':joint['id'],'parent_id':joint['parent_id'],'child_id':joint['child_id'],'type':joint['joint_type'],'status':'unknown','reason':reason,'diagnostic':diagnostic[:500]}
        for row in parts[joint['child_id']]['poses']:
            row.update(status='unknown',camera_from_native_mesh=None,reason=reason,method='source_nonrigidity_feasibility_withholding',joint_id=joint['id'],accepted_motion_evidence=False)
    mechanics['joints']=[j for j in mechanics['joints']if j['id']not in blocked]
    mechanics['unresolved_children']=[unresolved[j['id']]for j in persistent if j['id']in unresolved]
    mechanics['structural_connectivity_verified']=False
    result['limitations']=list(result.get('limitations',[]))+['Rigid child motion is withheld when its persistent proposal explicitly declares nonrigidity. Flexible articulation remains unresolved.']
    return result,{'findings':findings,'withheld_joint_ids':[j['id']for j in persistent if j['id']in blocked],
                   'withheld_component_ids':[j['child_id']for j in persistent if j['id']in blocked],
                   'rigidity_established':False,'flexible_motion_recovered':False,'native_geometry_modified':False,'unaffected_parent_and_removable_poses_preserved':True}


def build(scene_path,output,*,root=ROOT):
    root=Path(root).resolve();path,scene,program,_=A._checked_scene(scene_path,root);mechanics_path=U._local(path.parent/'mechanics.json',root);mechanics=U.read(mechanics_path)
    require(scene.get('mechanics')==mechanics,'Source mechanics companion changed')
    proposal_path=A.checked(mechanics['proposal'],root);proposals=O.load_proposals(mechanics['proposal']['path'],scene['original_program']['path'],root=root)
    require(proposals['interaction_sha256']==scene['original_program']['sha256']and proposals['source_sha256']==scene['source']['sha256'],'Rigidity proposal belongs to another source program')
    for reference in scene.get('inputs',[]):A.checked(reference,root)
    original_scene=S.ref(path,root);original_mechanics=S.ref(mechanics_path,root);proposal_ref=S.ref(proposal_path,root)
    corrected,assessment=apply(scene,proposals);out=Path(output);out=out if out.is_absolute()else root/out
    require(out.is_relative_to(root/'data')and out.resolve()==out and not any(p.is_symlink()for p in [out,*out.parents])and not out.exists(),'Fresh ordinary private rigidity output required')
    require(not out.is_relative_to(path.parent)and not out.is_relative_to(proposal_path.parent),'Preserve original scene/proposal evidence trees')
    out.mkdir(parents=True);snapshots=[]
    for name in CODE:
        target=out/'implementation'/name;target.parent.mkdir(exist_ok=True);target.write_bytes((root/name).read_bytes());snapshots.append(S.ref(target,root))
    assessment={**assessment,'proposal_ref':proposal_ref,'source_scene_ref':original_scene,'source_mechanics_ref':original_mechanics,
        'interaction_ref':scene['original_program'],'source_sha256':scene['source']['sha256'],'implementation_snapshots':snapshots,
        'declaration_policy':'whole positive clause only; no object/material name guessing; absence is not proof of rigidity',
        'new_api_calls':0,'new_gpu_calls':0,'automatic_approval':False,'published':False,'physical_accuracy_verified':False}
    receipt={'format':FORMAT,'version':1,'assessment':assessment,'status':'unsupported_rigid_child_motion_withheld'if assessment['findings']else'no_explicit_nonrigidity_declaration',
        'scene_ref':original_scene,'mechanics_ref':original_mechanics}
    if assessment['findings']:
        guarded=corrected['mechanics'];previous_implementations=[mechanics['implementation'],*mechanics.get('additional_implementations',[]),S.source_ref(root/'native_scene_audit.py',root)]
        guarded['implementation']=S.source_ref(root/'rigidity_feasibility.py',root)
        unique={}
        for reference in previous_implementations:
            require(reference['path']not in unique or unique[reference['path']]==reference,'Conflicting original implementation references');unique[reference['path']]=reference
        guarded['additional_implementations']=list(unique.values())
        guarded['rigidity_feasibility']=assessment
        corrected['inputs']=list({r['path']:r for r in [*scene.get('inputs',[]),original_scene,original_mechanics,proposal_ref]}.values())
        S.write(out/'mechanics.json',guarded);S.write(out/'scene.json',corrected);receipt.update(scene_ref=S.ref(out/'scene.json',root),mechanics_ref=S.ref(out/'mechanics.json',root))
    S.write(out/'feasibility.json',receipt)
    require(S.ref(path,root)==original_scene and S.ref(mechanics_path,root)==original_mechanics and S.ref(proposal_path,root)==proposal_ref,'Source evidence changed during withholding')
    return {'status':receipt['status'],'assessment_ref':S.ref(out/'feasibility.json',root),'scene_ref':receipt['scene_ref'],'mechanics_ref':receipt['mechanics_ref'],
            'withheld_joint_ids':assessment['withheld_joint_ids'],'withheld_component_ids':assessment['withheld_component_ids'],'new_api_calls':0,'new_gpu_calls':0,'published':False}
