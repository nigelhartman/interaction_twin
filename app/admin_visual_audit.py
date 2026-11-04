"""Fixed post-compile native audit → visual critic → measured repair proposal.

Inert until the root-reviewed immutable activation receipt matches this code.
Public requests cannot supply packet paths, scores, cost limits or executable text.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from agent_memory import digest, private_directory, write_immutable
from admin_muse_reviews import artifact_binding
from review_agents import verified_ref
from object_program import ref

ROOT=Path(__file__).resolve().parent
ACTIVATION='output/overnight-pipeline-20261102/visual-automation-activation-v4.json'
CODE=('admin_visual_audit.py','admin_execution_policy.py','admin_evaluation_plan.py','admin_machine_context.py','admin_memory_context.py',
      'admin_workflow.py','admin_pipeline.py','admin_http.py','scripts/run_admin_stage.py',
      'native_visual_review.py','native_visual_recovery.py','native_visual_recovery_v2.py','native_scene_audit.py','native_audit_protocol.py',
      'native_appearance.py','alignment_render.py','sam3d_common_scene.py','sam3d_mechanisms.py','part_ledger.py','overnight_budget.py',
      'pipeline.py','mechanism_pipeline.py','rigidity_feasibility.py','admin_muse_reviews.py',
      'mechanism_pass_budget.py','scripts/run_mechanism_pipeline.py','scripts/run_overnight_stage.py',
      'profiles/meta_sam3d_default.json','output/overnight-pipeline-20261102/mechanism-subtask-budget-v1.json')


def prior_critic(manifest,*,root):
    """Code/output changes cannot bypass an uncertain exact-state visual send."""
    from admin_memory_context import private_path
    from native_visual_recovery_v2 import verify_any_report,recovery_reports_for_binding
    recovered=[row for row in recovery_reports_for_binding(manifest,root=root)if row['report'].get('role')=='critic']
    if recovered:
        row=recovered[0]
        return (private_path(row['report_ref']['path'],root=root),row['report']),False
    folder=root/'.private/native-visual-reviews'
    if not folder.exists():return None,False
    plans=list(folder.glob('*/plan.json'))
    if len(plans)>1000:raise ValueError('Private visual inspection budget exceeded')
    unknown=False;completed=[]
    for path in sorted(plans):
        path=private_path(path,root=root)
        if path.stat().st_size>4*1024*1024:raise ValueError('Bounded visual plan required')
        plan=json.loads(path.read_text());context=plan.get('context',{})
        if plan.get('role')!='critic' or any(context.get(k)!=manifest.get(k)for k in ['interaction_ref','scene_ref']):continue
        report_path=path.parent/'attempt/report.json'
        if report_path.exists():
            try:
                report=json.loads(private_path(report_path,root=root).read_text());verify_any_report(report,root=root)
                completed.append((report_path,report))
            except Exception:unknown=True
        elif (path.parent/'attempt').exists()or(root/'.private/native-visual-attempts'/str(plan.get('attempt_namespace','missing'))).exists():
            unknown=True
    if completed:return completed[0],False
    return None,unknown


def claim_dispatch(manifest,prepared_ref,*,root):
    """Exclusive per exact program/scene claim, independent of audit path/code."""
    key=digest({k:manifest[k]for k in ['interaction_ref','scene_ref']})
    base=private_directory(root,'admin-visual-dispatch');folder=base/key
    try:folder.mkdir(mode=0o700)
    except FileExistsError:raise ValueError('Exact-state visual dispatch already claimed; no automatic paid resend')from None
    value={'format':'interaction-twin-exact-state-visual-intent','version':1,
        'binding':{k:manifest[k]for k in ['interaction_ref','scene_ref']},'prepared_plan_ref':prepared_ref,'automatic_retry':False}
    raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
    descriptor=os.open(folder/'intent.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(descriptor,'wb')as stream:stream.write(raw)


def activation_template(*,root=ROOT):
    root=Path(root).resolve()
    return {'format':'interaction-twin-fixed-visual-automation-activation','version':1,'enabled':True,
        'authorization_basis':'explicit_user_approved_overnight_pipeline',
        'maximum_total_rounds':3,'maximum_repairs':2,'automatic_approval':False,'automatic_publication':False,
        'allowed_inputs':['source_rgb','native_mesh_render','sam_masks'],
        'implementation_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()for name in CODE}}


def activation(*,root=ROOT):
    root=Path(root).resolve();path=root/ACTIVATION
    if not path.is_file():return None
    if path.resolve()!=path or path.is_symlink()or path.stat().st_size>16000:raise ValueError('Invalid visual automation activation')
    value=json.loads(path.read_text())
    if value!=activation_template(root=root):raise ValueError('Visual automation code/policy changed; root review required')
    return ref(path,root)


def local_assessment(audit,scene,program,critic,*,evaluation_plan_ref=None,root=ROOT):
    """Development measurements only; model confidence never ranks revisions.

    A safe repair is a bounded experiment under fixed adapters, not physical
    acceptance. Ambiguous/missing comparisons stop instead of inventing scores.
    """
    rows=[r for r in audit['samples']if r['role']=='development']
    usable=[r for r in rows if r['status']in {'diagnostic_available','needs_visual_review'}]
    values=[r['metrics']['mask_iou']for r in usable]
    if not values or any(isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)or not 0<=v<=1 for v in values):return None
    expected={t['id']for t in program['tracklets']}
    scene_ids={c['id']for c in scene['components']}
    assets={a['track_id']:a for a in program.get('assets',[])}
    missing=expected-scene_ids
    inconsistent={c['id']for c in scene['components']if c['id']not in assets or assets[c['id']]['sha256']!=c['asset']['sha256']}
    frame_ids=sorted({r['source_index']for r in rows})
    if not frame_ids or not expected:return None
    coverage=len(usable)/(len(frame_ids)*len(expected))
    if coverage>1:raise ValueError('Duplicate development comparisons cannot inflate coverage')
    targets=sorted({p['target']for p in critic.get('repair_proposals',[])if p['target']in {'reasoning','tracking','reconstruction','mechanics'}})
    error=1-statistics.median(values)
    # Compilation coverage is a reasoning/compile repair; it does not purchase
    # another model when an existing native component merely lacks a program ref.
    if inconsistent:targets=['reasoning']
    elif missing:targets=['reconstruction']
    supported=bool(missing or inconsistent or(error>.25 and targets))
    safe=supported and len(usable)>=3 and coverage>=.35
    if evaluation_plan_ref is not None:
        from admin_evaluation_plan import verify_audit,comparison_key
        verify_audit(evaluation_plan_ref,audit,root=root)
        key=comparison_key(evaluation_plan_ref,root=root)
    else:key=digest({'source_sha256':audit['source']['sha256'],'development_source_indices':frame_ids})
    return {'measurement_kind':'deterministic_local_metrics',
        'comparison_key':key,
        'metrics':{'blocking_issues':len(inconsistent),'missing_components':len(missing),
            'alignment_error':error,'coverage':coverage},
        'repair_targets':targets,'automatic_repair_safe':safe,
        'scope':'development native triangle/mask diagnostics only; unknown hand occlusion and model fallibility retained',
        'physical_accuracy_verified':False,'final_test_evidence_used':False}


def build_reviewer(job,*,root=ROOT):
    """Trusted worker adapter. Builds exact packets only after local activation."""
    root=Path(root).resolve()
    def review(manifest):
        active=activation(root=root)
        if active is None:return {'status':'unavailable','diagnostic':'Visual automation awaits reviewed activation'}
        from admin_execution_policy import validate_policy
        validate_policy(job['improvement_policy'],root)
        lease=job.get('execution_lease');guard=lease.guard if lease else None
        if guard:guard('prepare_native_visual_audit')
        if not manifest.get('scene_ref'):
            return {'status':'unavailable','diagnostic':'No exact connected native scene to review'}
        source=verified_ref(root,manifest['source_ref'])
        scene_path=verified_ref(root,manifest['scene_ref'],8*1024*1024)
        program_path=verified_ref(root,manifest['interaction_ref'],8*1024*1024)
        scene=json.loads(scene_path.read_text());program=json.loads(program_path.read_text())
        if scene.get('original_program')!=manifest['interaction_ref'] or manifest['source_ref']['sha256']not in{program['source']['sha256'],program['source'].get('original_sha256')}:
            raise ValueError('Native review source/program/scene binding differs')
        from admin_evaluation_plan import get_or_create,verify_audit,EvaluationBasisChanged
        try:
            evaluation_ref,evaluation=get_or_create(dict(job,initial_scene_ref=manifest['scene_ref']),scene,program,root=root,max_frames=12)
        except EvaluationBasisChanged:
            return {'status':'unavailable','diagnostic':'Frozen evaluation basis changed; no resplit or visual dispatch'}
        previous,unknown=prior_critic(manifest,root=root)
        if unknown:return {'status':'unavailable','diagnostic':'Prior exact-state visual attempt is incomplete or uncertain; no automatic resend'}
        from native_scene_audit import build_audit,validate_manifest
        identity=digest({'binding':artifact_binding(manifest),'activation':active})
        output='data/native-render-audits/admin-'+identity
        built=build_audit(scene_path,output,root=root,max_frames=12,max_dimension=384)
        audit=validate_manifest(root/built['manifest']['path'],root=root)
        try:verify_audit(evaluation_ref,audit,root=root)
        except EvaluationBasisChanged:
            return {'status':'unavailable','diagnostic':'Native audit differs from frozen development/final basis; no visual dispatch'}
        from native_visual_review import prepare,build_admin_reviewer
        from native_visual_recovery_v2 import verify_any_report,recover,MISMATCH
        if previous:
            previous_path,previous_report=previous
            from admin_memory_context import private_path
            prior_plan=json.loads(private_path(previous_report['plan_ref']['path'],root=root).read_text())
            if any(row['source_index']not in evaluation['basis']['development_source_indices']for row in prior_plan['context']['samples']):
                return {'status':'unavailable','diagnostic':'Preserved critic uses another evaluation split; no reuse or paid resend'}
            stable_key=digest({k:manifest[k]for k in ['interaction_ref','scene_ref']})
            if not(private_directory(root,'admin-visual-dispatch')/stable_key).exists():
                claim_dispatch(manifest,previous_report['plan_ref'],root=root)
            result={'status':'complete','artifact_binding':artifact_binding(manifest),'report_ref':ref(previous_path,root),
                'issues':copy.deepcopy(previous_report['issues']),'repair_targets':[], 'raw_critic_reused':True}
        else:
            prepared='.private/native-visual-reviews/admin-'+identity+'-critic'
            folder=root/prepared
            if not folder.exists():
                prepare(built['manifest']['path'],prepared,root=root,role='critic',round_number=job.get('improvement_round',0)+1)
            authorized=ref(folder/'plan.json',root)
            claim_dispatch(manifest,authorized,root=root)
            fixed=dict(job,authorized_visual_review={'prepared':prepared,'authorized_plan_sha256':authorized['sha256']})
            if guard:guard('native_visual_critic')
            request_fence=(lambda:guard('native_visual_request'))if guard else None
            try:
                result=build_admin_reviewer(fixed,root=root,lease_guard=request_fence)(manifest)
            except ValueError as error:
                # A paid, fully accounted response may contain a narrowly
                # inconsistent action/target pair. The offline adapter can only
                # omit that advice; it cannot retry inference or retarget it.
                if str(error)!=MISMATCH:raise
                recovered_folder='.private/native-visual-recoveries-v2/admin-'+identity
                report=recover(prepared,recovered_folder,root=root)
                if guard:guard('native_visual_offline_recovery')
                result={'status':'complete','artifact_binding':artifact_binding(manifest),
                    'report_ref':ref(root/recovered_folder/'report.json',root),
                    'issues':copy.deepcopy(report['issues']),'repair_targets':[],
                    'raw_critic_recovered_offline':True}
        if result.get('status')!='complete':return result
        from admin_memory_context import private_path
        raw_path=private_path(result['report_ref']['path'],root=root)
        raw=raw_path.read_bytes()
        if len(raw)!=result['report_ref']['bytes']or hashlib.sha256(raw).hexdigest()!=result['report_ref']['sha256']:
            raise ValueError('Raw visual critic report changed')
        critic=json.loads(raw);verify_any_report(critic,root=root)
        assessment=local_assessment(audit,scene,program,critic,evaluation_plan_ref=evaluation_ref,root=root)
        if assessment is None:return dict(result,automatic_repair_safe=False)
        measured={'format':'interaction-twin-source-bound-measured-visual-review','version':2,
            'artifact_binding':artifact_binding(manifest),'raw_critic_ref':result['report_ref'],
            'audit_ref':built['manifest'],'activation_ref':active,'evaluation_plan_ref':evaluation_ref,'improvement_assessment':assessment,
            'issues':copy.deepcopy(critic['issues']),'repair_proposals':copy.deepcopy(critic['repair_proposals']),
            'human_feedback_authored':False,'automatic_approval':False,'automatic_publication':False,
            'final_test_evidence_used':False,'physical_accuracy_verified':False}
        folder=private_directory(root,'measured-visual-reviews');name=write_immutable(folder,'measured',measured)
        measured_ref=ref(folder/(name+'.json'),root)
        return dict(result,report_ref=measured_ref,improvement_assessment=assessment,
            raw_critic_ref=result['report_ref'],automatic_repair_safe=assessment['automatic_repair_safe'])
    return review
