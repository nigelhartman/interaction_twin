"""Frozen local multi-input candidates, not multiview 3D fusion or physical truth."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import fcntl
import html

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'scripts'))
from stage3_select_views import mask_metrics, score_candidates, select_diverse, export_selected

PARTS = ['container_base', 'container_lid', 'small_object_a']
BASELINES = {
 'container_base': 'data/jabra-local-v2/motion-container_base-v1/experiment.json',
 'container_lid': 'data/jabra-local-v2/motion-container_lid-v1/experiment.json',
 'small_object_a': 'data/recording-jabra-5fps-v1/alignment/generated-earbud-crop-motion-v1/experiment.json',
}
POLICY = {'global_timestamp_partition': 'floor(source_pts_seconds / 3) modulo 3: 0 selection, 1 validation, 2 final_test',
          'min_selection_interval_s': 1., 'views_per_part': 3, 'mesh_resolution': 96,
          'search_width': 128, 'refine_count': 2, 'max_evaluations': 60, 'appearance_weight': .25,
          'scale_policy': 'one unchanged native mesh scale=1 per candidate across every timestamp; no shared metric gauge',
          'camera_policy': 'fixed generic 60-degree horizontal-FOV intrinsics per candidate; no per-frame camera refit',
          'comparison': 'internal source-frame diagnostic; full-clip CoTracker context is not held-out generalization',
          'winner_order': ['validation supported count descending', 'validation supported median IoU descending',
                           'validation supported median held-out residual ascending', 'candidate ID ascending'],
          'gate_policy': 'unchanged stage4_track_generated.pose_gate; no post-hoc relaxation',
          'not_multiview_fusion': True, 'physical_accuracy_measured': False}

def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def rel(path):return str(Path(path).resolve().relative_to(ROOT))
def split_name(timestamp):return ['selection','validation','final_test'][math.floor(timestamp/3)%3]

def cached_candidates(case, run, tid):
    """Verify transferred model prompts and cached audit pixels without new reasoning."""
    import numpy as np
    from PIL import Image
    seg=case/'segmentation'/run;manifest=read(case/'manifest.json');cfg=read(seg/'annotations.json');meta=read(seg/'metadata.json')
    if cfg.get('manual_seed_boxes_used') is not False or meta.get('status')!='complete':raise ValueError('Model-generated completed segmentation required')
    origin=Path(manifest['source']['original_source_case']).resolve()
    if not origin.is_relative_to(ROOT/'data'):raise ValueError('Original source case outside data')
    discoveries=[p for p in (origin/'reasoning').glob('*/discovery.json') if sha(p)==cfg['discovery_record_sha256']]
    if len(discoveries)!=1:raise ValueError('Transferred prompts lack exact original discovery hash')
    transfer=read(case/'model-transferred-annotations.json')
    if sha(case/'model-transferred-annotations.json')!=sha(seg/'annotations.json'):raise ValueError('Segmentation prompts differ from saved deterministic transfer')
    sources=[p for p in (origin/'segmentation').glob('*/annotations.json') if sha(p)==transfer['source_annotations_sha256']]
    if len(sources)!=1:raise ValueError('Transferred prompts lack original annotation hash')
    spec=next(row for row in cfg['tracks'] if row['id']==tid)
    audit=read(seg/'model-audit-v1/audit.json')
    if audit['source_sha256']!=manifest['source']['sha256'] or audit['source_track_index_sha256']!=sha(seg/'tracks.json'):raise ValueError('Audit source/index changed')
    audited={(r['sample_index'],r['track_id']) for r in audit['result']['observations'] if r['verdict']=='plausible_visible_mask'}
    evidence={(r['sample_index'],r['track_id']):r for r in read(seg/'model-audit-v1/evidence.json')}
    geometric={(r['sample_index'],t['track_id']) for r in read(seg/'frames.json')['frames'] for t in r['tracks'] if t.get('status')=='candidate_geometric_gate_only'}
    frames={r['index']:r for r in read(case/'frames/index.json')['frames']}
    track=next(row for row in read(seg/'tracks.json')['tracks'] if row['track_id']==tid)
    provenance={'prompt_origin':'model_generated_prompt','manual_seed_boxes_used':False,'model':meta['model'],'device':meta['device'],
                'checkpoint_sha256':meta['checkpoint_sha256'],'prompts':spec['prompts'],
                'annotations_file':os.path.relpath(seg/'annotations.json',case),'annotations_sha256':sha(seg/'annotations.json'),
                'discovery_file':os.path.relpath(discoveries[0],case),'discovery_sha256':sha(discoveries[0]),
                'mask_plausibility_audit_file':os.path.relpath(seg/'model-audit-v1/audit.json',case),
                'mask_plausibility_audit_sha256':sha(seg/'model-audit-v1/audit.json'),'human_confirmed':False,'accuracy_measured':False}
    candidates=[]
    for observation in track['frames']:
        index=observation['sample_index'];pair=(index,tid)
        if pair not in audited or pair not in geometric or observation.get('status')!='candidate_visible':continue
        row=frames[index];image=case/row['file'];mask=seg/observation['raw_mask_file'];ev=evidence[pair]
        if abs(row['timestamp_s']-observation['timestamp_s'])>1e-6 or ev['timestamp_s']!=row['timestamp_s']:raise ValueError('Source PTS changed')
        if sha(image)!=ev['source_frame_sha256'] or sha(mask)!=ev['raw_mask_sha256']:raise ValueError('Audited pixels changed')
        rgb=np.asarray(Image.open(image).convert('RGB'));binary=np.asarray(Image.open(mask).convert('L'))>0;metrics=mask_metrics(rgb,binary)
        x0,y0,x1,y1=metrics['bbox_xyxy']
        if metrics['area_pixels']<400 or min(x1-x0,y1-y0)<16:continue
        candidates.append({'sample_index':index,'timestamp_s':row['timestamp_s'],'source_case':case.name,'source_sample_index':index,
          'source_track_id':tid,'object_id':tid,'physical_identity_status':spec['identity_status'],'image_file':row['file'],
          'mask_file':os.path.relpath(mask,case),'image_sha256':sha(image),'mask_sha256':sha(mask),
          'mask_scope':spec['description'],'model_provider':meta['model'],'mask_provenance':provenance,
          'metrics':metrics,'occlusion_candidate':{'status':'not_assessed','hand_proximity_fraction':None},'segmentation_accuracy_measured':False})
    return candidates,manifest,spec

def score_motion(pack, sample_indices):
    rows={r['sample_index']:r for r in (pack or {}).get('frames',[])}
    supported=[];references=0;missing=0
    for sample in sample_indices:
        row=rows.get(sample)
        if row is None:missing+=1;continue
        if row['status'].startswith('reference_'):references+=1;continue
        if row['status']=='supported_conditional_motion_candidate':supported.append(row)
    ious=sorted(r['raw_full_projection_iou_with_sam'] for r in supported)
    errors=sorted(r['heldout_median_px'] for r in supported)
    median=lambda values:float(__import__('statistics').median(values)) if values else None
    return {'denominator':len(sample_indices),'sample_indices':sample_indices,'supported':len(supported),'missing':missing,'reference_excluded':references,
            'unsupported':len(sample_indices)-missing-references-len(supported),'coverage':len(supported)/len(sample_indices) if sample_indices else None,
            'median_supported_iou':median(ious),'median_supported_heldout_px':median(errors)}

def select_winner(candidates):
    eligible=[r for r in candidates if r.get('status')=='complete' and r.get('validation',{}).get('denominator',0)>0]
    if not eligible:return None
    return min(eligible,key=lambda r:(-r['validation']['supported'],-(r['validation']['median_supported_iou'] or 0),
                                    r['validation']['median_supported_heldout_px'] if r['validation']['median_supported_heldout_px'] is not None else float('inf'),r['id']))['id']

def prepare(output, case, run):
    if output.exists():raise FileExistsError('Fresh experiment required; use --resume to verify saved plan')
    output.mkdir(parents=True)
    frames=read(case/'frames/index.json')['frames'];manifest=read(case/'manifest.json');seg=case/'segmentation'/run/'annotations.json'
    plan={'schema_version':'0.1','created_at':now(),'status':'planned_before_inference','source_case':rel(case),'segmentation_run':run,
          'source_sha256':manifest['source']['sha256'],'policy':POLICY,'splits':{name:[{'sample_index':r['index'],'timestamp_s':r['timestamp_s']} for r in frames if split_name(r['timestamp_s'])==name] for name in ['selection','validation','final_test']},
          'code_inputs':[{'path':rel(p),'sha256':sha(p)} for p in [Path(__file__),ROOT/'scripts/run_local_candidates.py',ROOT/'scripts/stage3_reconstruct_local.py',ROOT/'scripts/stage4_generated_pose.py',ROOT/'scripts/stage4_track_generated.py',ROOT/'scripts/stage4_generated_observer.py']],
          'input_receipts':[{'path':rel(p),'sha256':sha(p)} for p in [case/'manifest.json',case/'frames/index.json',seg,seg.parent/'tracks.json',seg.parent/'model-audit-v1/audit.json']],
          'parts':[],'limitations':['Independent image-to-3D candidates, not multi-view fusion.','Cached mask audits are fallible model judgments, not physical accuracy labels.','Full-clip tracking observes scoring frames; these are internal source-frame diagnostics.','No manual geometry, poses, seed boxes or forced insertion; no metric scale or contact validation.']}
    for tid in PARTS:
        candidates,source,spec=cached_candidates(case,run,tid)
        parent=score_candidates([copy.deepcopy(r) for r in candidates if split_name(r['timestamp_s'])=='selection'])
        chosen=select_diverse(parent,3,1.)
        baseline_path=ROOT/BASELINES[tid];baseline=read(baseline_path);metadata=read(baseline['provenance']['model_metadata_file'])
        excluded={metadata['source']['timestamp_s'],*[r['timestamp_s'] for r in baseline['frames'] if r['status'].startswith('reference_')],*[r['timestamp_s'] for r in chosen]}
        pools={name:[r['sample_index'] for r in candidates if split_name(r['timestamp_s'])==name and all(abs(r['timestamp_s']-t)>1e-6 for t in excluded)] for name in ['validation','final_test']}
        part={'track_id':tid,'mask_scope':spec['description'],'candidate_entity_ids':spec['candidate_entity_ids'],'identity_status':spec['identity_status'],
              'eligible_inputs':[{k:r[k] for k in ['sample_index','timestamp_s','image_sha256','mask_sha256']} for r in candidates],
              'selection_pool':parent,'validation_indices':pools['validation'],'final_test_indices':pools['final_test'],
              'baseline':{'experiment':rel(baseline_path),'sha256':sha(baseline_path),'model_metadata':rel(metadata_path:=Path(baseline['provenance']['model_metadata_file'])),
                          'model_metadata_sha256':sha(metadata_path),'resolution':metadata.get('density_grid_resolution'),'excluded_generation_reference_timestamps':sorted(excluded)},'candidates':[]}
        for number,candidate in enumerate(chosen,1):
            directory=output/tid/f'view-{number:02d}';directory.mkdir(parents=True)
            original_rank=next(i+1 for i,r in enumerate(parent) if r['sample_index']==candidate['sample_index'])
            exports=export_selected(case,directory,[candidate])
            selection={'schema_version':'0.1','status':'ranked_inputs_not_reconstruction','object_id':tid,'source':source['source'],
                       'selection_method':{'type':'single_candidate_subordinate_export_of_frozen_selection_pool','agent_visual_choice':False,
                                          'parent_plan':rel(output/'plan.json'),'original_parent_quality_rank':original_rank,'parent_pool_hash':hashlib.sha256(json.dumps(parent,sort_keys=True).encode()).hexdigest()},
                       'selected':exports,'original_candidate_parent_quality_rank':original_rank,
                       'limitations':['Rank one is the sole input of this subordinate candidate, not a claim that it topped the global view ranking.']}
            write(directory/'selection.json',selection)
            part['candidates'].append({'id':tid+f'-view-{number:02d}','sample_index':candidate['sample_index'],'timestamp_s':candidate['timestamp_s'],
                                       'original_quality_rank':original_rank,'selection':rel(directory/'selection.json'),'selection_sha256':sha(directory/'selection.json'),'directory':rel(directory),'status':'pending'})
        part['baseline']['validation']=score_motion(baseline,pools['validation'])
        plan['parts'].append(part)
    write(output/'plan.json',plan)
    write(output/'run.json',{'started_at':now(),'status':'prepared','plan_sha256':sha(output/'plan.json'),'parts':copy.deepcopy(plan['parts'])})
    return plan

def verify_plan(output, plan, state):
    """Only an explicit, hash-bound orchestration migration may replace wrapper code."""
    if sha(output/'plan.json') != state['plan_sha256']:
        raise ValueError('Frozen plan changed')
    migrations=read(output/'orchestration-migration.json') if (output/'orchestration-migration.json').exists() else None
    for row in plan['code_inputs']+plan['input_receipts']:
        if sha(ROOT/row['path']) == row['sha256']:continue
        if (row['path']=='local_candidates.py' and migrations
            and migrations['plan_sha256']==state['plan_sha256']
            and migrations['original_sha256']==row['sha256']
            and sha(ROOT/migrations['original_snapshot'])==row['sha256']
            and sha(ROOT/'local_candidates.py')==migrations['replacement_sha256']
            and migrations['policy_or_model_changes'] is False):continue
        raise ValueError('Frozen code/input changed: '+row['path'])

def complete_model(path, selection_sha):
    """A completed receipt and every output hash survive a teardown failure."""
    if not path.is_file():return False
    try:
        value=read(path)
        if value['status']!='candidate_generated_pending_visual_review' or not value['learned_inference_executed']:return False
        if value['source']['selection_manifest_sha256']!=selection_sha:return False
        if value['density_grid_resolution']!=96 or value['runtime']['device']!='mps':return False
        artifacts=value['artifact_files']
        if not artifacts or value['outputs']['glb_path'] not in {r['file'] for r in artifacts}:return False
        return all((path.parent/r['file']).is_file() and (path.parent/r['file']).stat().st_size==r['bytes'] and sha(path.parent/r['file'])==r['sha256'] for r in artifacts)
    except (KeyError,ValueError,OSError):return False

def complete_pose(path, model):
    if not path.is_file():return False
    try:
        value=read(path);metadata=read(model)
        return (value['inputs']['manifest_sha256']==sha(model)
                and value['inputs']['mesh_sha256']==sha(model.parent/metadata['outputs']['glb_path'])
                and value['source_video_sha256']==metadata['source']['video_sha256']
                and 'candidate' in value and value['status']!='running')
    except (KeyError,ValueError,OSError):return False

def complete_motion(path, reference):
    if not path.is_file():return False
    try:
        value=read(path)
        return (value['status']=='complete_conditional_tracking_experiment'
                and value['provenance']['reference_sha256']==sha(reference)
                and value['frame_count']==len(value['frames'])
                and (path.parent/'tracks.npz').is_file() and (path.parent/'point-support.json').is_file())
    except (KeyError,ValueError,OSError):return False

def fresh_stage(directory, name):
    """Never overwrite an interrupted or unsuccessful stage directory."""
    path=directory/name
    if not path.exists():return path
    number=2
    while (directory/f'{name}-attempt-{number:02d}').exists():number+=1
    return directory/f'{name}-attempt-{number:02d}'

def run(output, case, run_name='automatic-components-v1', max_runtime_s=1800, resume=False):
    plan=read(output/'plan.json') if resume else prepare(output,case,run_name)
    state=read(output/'run.json');verify_plan(output,plan,state)
    with (output/'worker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A candidate worker already holds the experiment lock')
        return _run_locked(output,plan,state,run_name,max_runtime_s)

def _run_locked(output, plan, state, run_name, max_runtime_s):
    started=time.monotonic();deadline=started+max_runtime_s
    invocation={'started_at':now(),'pid':os.getpid(),'budget_seconds':max_runtime_s,'wrapper_sha256':sha(__file__)}
    state.setdefault('invocations',[]).append(invocation);state['status']='running';write(output/'run.json',state)
    python=ROOT/'.venv-sam/bin/python'
    def command(script,args,directory):
        log=fresh_stage(directory,Path(script).stem+'.log');begin=time.monotonic()
        record={'script':script,'started_at':now(),'log':rel(log)}
        try:
            with log.open('wb') as stream:
                proc=subprocess.run([str(python),str(ROOT/'scripts'/script),*[str(x) for x in args]],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,timeout=max(1,min(600,deadline-time.monotonic())))
            record['returncode']=proc.returncode
        except subprocess.TimeoutExpired:record['timed_out']=True
        finally:record.update(seconds=time.monotonic()-begin,log_sha256=sha(log))
        return record
    for part in state['parts']:
        for candidate in part['candidates']:
            if candidate['status']=='complete' or candidate.get('terminal_failure'):continue
            if time.monotonic()>=deadline:
                candidate['status']='not_attempted_runtime_budget';write(output/'run.json',state);continue
            directory=ROOT/candidate['directory']
            candidate.setdefault('prior_attempt_receipts',[]).append({k:copy.deepcopy(v) for k,v in candidate.items() if k not in {'prior_attempt_receipts'}})
            candidate.update(status='running',resumed_at=now(),commands=[]);write(output/'run.json',state)
            try:
                if sha(ROOT/candidate['selection'])!=candidate['selection_sha256']:raise ValueError('Frozen selected input changed')
                model=(ROOT/candidate['model']) if candidate.get('model') else directory/'model/metadata.json'
                if complete_model(model,candidate['selection_sha256']):
                    candidate.setdefault('recovered_stages',[]).append({'stage':'model','path':rel(model),'sha256':sha(model),'validated_at':now(),'reason':'Complete model receipt and all artifact hashes verified; preserved prior process failure separately'})
                else:
                    target=fresh_stage(directory,'model');candidate['commands'].append(command('stage3_reconstruct_local.py',['--selection',ROOT/candidate['selection'],'--output',target,'--resolution',96],directory));model=target/'metadata.json'
                candidate['model']=rel(model);write(output/'run.json',state)
                if not complete_model(model,candidate['selection_sha256']):raise ValueError('No hash-complete generated model; command receipt retained')
                if candidate['commands'] and candidate['commands'][-1].get('returncode') not in (None,0):candidate['commands'][-1]['complete_artifacts_recovered_despite_process_exit']=True
                pose=(ROOT/candidate['reference']) if candidate.get('reference') else directory/'pose/experiment.json'
                if complete_pose(pose,model):
                    candidate.setdefault('recovered_stages',[]).append({'stage':'pose','path':rel(pose),'sha256':sha(pose),'validated_at':now()})
                else:
                    target=fresh_stage(directory,'pose');candidate['commands'].append(command('stage4_generated_pose.py',['--manifest',model,'--out',target,'--search-width',128,'--refine-count',2,'--max-evaluations',60,'--appearance-weight',.25],directory));pose=target/'experiment.json'
                candidate['reference']=rel(pose);write(output/'run.json',state)
                if not complete_pose(pose,model):raise ValueError('No hash-complete reference pose')
                motion=(ROOT/candidate['experiment']) if candidate.get('experiment') else directory/'motion/experiment.json'
                if not complete_motion(motion,pose):
                    target=fresh_stage(directory,'motion');candidate['commands'].append(command('stage4_track_generated.py',['--reference',pose,'--segmentation-run',run_name,'--output',target],directory));motion=target/'experiment.json'
                candidate['experiment']=rel(motion);write(output/'run.json',state)
                if not complete_motion(motion,pose):raise ValueError('No complete conditional motion: '+str(read(motion).get('status') if motion.is_file() else 'missing receipt'))
                candidate['validation']=score_motion(read(motion),part['validation_indices'])
                observer=motion.parent/'observer-3d.html'
                if not observer.exists():candidate['commands'].append(command('stage4_generated_observer.py',['--experiment',motion,'--out',observer],directory))
                candidate.update(status='complete',experiment_sha256=sha(motion),observer=rel(observer) if observer.exists() else None,physical_pose_verified=False,completed_at=now())
                candidate.pop('error',None)
            except Exception as exc:
                candidate.update(status='failed',error=f'{type(exc).__name__}: {exc}',completed_at=now(),terminal_failure=True)
            finally:write(output/'run.json',state)
        # No winner/test scores are exposed until every declared candidate terminates.
        if any(c['status'] not in {'complete','failed'} for c in part['candidates']):continue
        lockpath=output/f"{part['track_id']}-winner-lock.json"
        winner=select_winner(part['candidates'])
        if lockpath.exists():
            prior=read(lockpath)
            if prior['selected_candidate_id']!=winner:raise ValueError('Locked validation winner would change')
            part.update(prior)
        else:
            part['selected_candidate_id']=winner
            part['selection_status']='validation_support_underdetermined' if winner is None or max((r.get('validation',{}).get('supported',0) for r in part['candidates']),default=0)<2 else 'candidate_selected_by_internal_validation_only'
            part['winner_locked_at']=now();write(lockpath,{k:part[k] for k in ['track_id','selected_candidate_id','selection_status','winner_locked_at']})
        write(output/'run.json',state)
        part['baseline']['final_test']=score_motion(read(ROOT/part['baseline']['experiment']),part['final_test_indices'])
        for candidate in part['candidates']:
            if candidate['status']=='complete':candidate['final_test']=score_motion(read(ROOT/candidate['experiment']),part['final_test_indices'])
        write(output/'run.json',state)
    invocation.update(completed_at=now(),wall_seconds=time.monotonic()-started)
    pending=any(c['status'] not in {'complete','failed'} for p in state['parts'] for c in p['candidates'])
    state.update(status='paused_runtime_budget' if pending else ('complete_with_gaps' if any(c['status']!='complete' for p in state['parts'] for c in p['candidates']) else 'complete'),completed_at=now(),wall_seconds=sum(r.get('wall_seconds',0) for r in state['invocations']))
    write(output/'run.json',state)
    if not pending:build_report(output,state)
    return state

def build_report(output,state):
    report={'schema_version':'0.1','status':state['status'],'created_at':now(),'plan_sha256':state['plan_sha256'],'evaluation_kind':'internal source-frame diagnostics, not physical pose accuracy or held-out generalization','physical_accuracy_measured':False,'not_multiview_fusion':True,'parts':[],'limitations':read(output/'plan.json')['limitations'],'recovery_invocations':state.get('invocations',[])}
    sections=[]
    for part in state['parts']:
        winner=next((c for c in part['candidates'] if c['id']==part.get('selected_candidate_id')),None)
        baseline=part['baseline'];baseval=baseline['validation']
        # Frozen validation ordering is also the baseline-retention gate; final test never chooses deployment.
        ranks=lambda s:(s['supported'],s['median_supported_iou'] or 0,-(s['median_supported_heldout_px'] if s['median_supported_heldout_px'] is not None else float('inf')))
        replace=bool(winner and ranks(winner['validation'])>ranks(baseval))
        decision='candidate_validation_better_physical_unverified' if replace else 'retain_preserved_baseline'
        chosen=winner if replace else None
        row={'track_id':part['track_id'],'selection_status':part['selection_status'],'validation_winner':part.get('selected_candidate_id'),'decision':decision,'adopted_experiment':chosen['experiment'] if chosen else baseline['experiment'],'adopted_model_metadata':chosen['model'] if chosen else baseline['model_metadata'],'baseline':baseline,'winner':winner,'candidates':part['candidates'],'winner_lock':rel(output/f"{part['track_id']}-winner-lock.json")}
        if winner:
            common=part['final_test_indices']
            row['final_common_source_samples']={'sample_indices':common,'denominator':len(common),'baseline_supported':baseline['final_test']['supported'],'validation_winner_supported':winner['final_test']['supported'],'delta':winner['final_test']['supported']-baseline['final_test']['supported']}
        report['parts'].append(row)
        esc=html.escape;link=lambda path,label:f'<a href="{esc(os.path.relpath(ROOT/path,output),quote=True)}">{esc(label)}</a>'
        cards=[]
        for c in part['candidates']:
            v=c.get('validation',{});t=c.get('final_test',{});selection=read(ROOT/c['selection']);preview=Path(c['selection']).parent/selection['selected'][0]['exports']['rgb']
            links=[link(str(preview),'source RGB'),link(c['selection'],'selection receipt')]
            if c.get('model') and (ROOT/c['model']).is_file():links += [link(c['model'],'model metadata'),link(str(Path(c['model']).parent/'asset.glb'),'GLB')]
            if c.get('reference') and (ROOT/c['reference']).is_file():links.append(link(str(Path(c['reference']).parent/'index.html'),'reference fit'))
            if c.get('observer'):links.append(link(c['observer'],'motion + 3D observer'))
            cards.append(f'<article><h3>{esc(c["id"])} · {esc(c["status"])}</h3><p>Source {c["timestamp_s"]:.3f}s · original quality rank {c["original_quality_rank"]}</p><p>Validation {v.get("supported","—")}/{v.get("denominator","—")} · final {t.get("supported","—")}/{t.get("denominator","—")}</p><p>{" · ".join(links)}</p><small>{esc(c.get("error","Independent TripoSR candidate; hidden geometry and appearance are predicted."))}</small></article>')
        summary=f'Baseline validation {baseval["supported"]}/{baseval["denominator"]}; final {baseline["final_test"]["supported"]}/{baseline["final_test"]["denominator"]}.'
        sections.append(f'<section><h2>{esc(part["track_id"])}</h2><p>{esc(summary)} Selected by validation: {esc(part.get("selected_candidate_id") or "none")}. {esc(decision)}.</p><div class="cards">{"".join(cards)}</div></section>')
    write(output/'report.json',report)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Local view candidate study</title><style>body{margin:0;background:#101922;color:#e7f1f5;font:16px/1.5 system-ui;padding:32px;max-width:1400px}h1{font-size:34px}p,small{color:#b8cad5}a{color:#70dbc5}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}article{padding:18px;background:#1b2b38;border:1px solid #3a5260;border-radius:12px}section{margin:35px 0}.note{padding:18px;border:1px solid #b88f4c;background:#3b3222;border-radius:12px}</style><h1>Three saved views × three Jabra components</h1><p>Nine independent image-to-3D candidates, refit and tracked with unchanged gates. No multi-view shape fusion.</p><div class="note">All numbers are internal source-frame diagnostics. The same learned masks and full-clip tracker contribute to fitting. Camera, geometry, scale and physical pose remain unverified. Sparse validation pools for the case parts do not establish robust quality gains. Unsupported intervals stay absent.</div>'+''.join(sections)+'<p><a href="report.json">Machine-readable report</a> · <a href="plan.json">Frozen pre-inference plan</a> · <a href="run.json">Full execution and recovery receipts</a></p>')
    return report
