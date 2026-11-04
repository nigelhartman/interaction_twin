"""Pure polling summaries; full immutable/progress records stay available.

This is a display projection, not a new provenance or physical-quality verdict.
It reads no files, runs no models and never mutates the supplied run record.
"""
import math
import re

ID=re.compile(r'[a-z0-9][a-z0-9-]{0,179}\Z')
TEXT_FIELDS={'id','status','phase','created_at','updated_at','started_at','completed_at','error'}
PHASE_FIELDS={'name','status','started_at','completed_at','error','attempt','elapsed_s','reused'}
CHILD_FIELDS={'id','track_id','status','workflow_status','phase','result_status','viewer_url',
              'workflow_manifest_url','mask_provider','result_kind','label'}


def _fields(record,names):
    if not isinstance(record,dict):return {}
    result={}
    for key in sorted(names):
        value=record.get(key)
        if value is None:
            if key in record:result[key]=None
        elif isinstance(value,str):result[key]=value[:4000 if key=='error' else 2000]
        elif isinstance(value,bool):result[key]=value
        elif isinstance(value,(int,float)) and math.isfinite(value):result[key]=value
    return result


def compact_run_summary(record):
    if not isinstance(record,dict) or not isinstance(record.get('id'),str) or not ID.fullmatch(record['id']):
        raise ValueError('Run summary requires a valid local run identifier')
    summary=_fields(record,TEXT_FIELDS)
    value=record.get('progress',0)
    summary['progress']=max(0,min(1,value)) if isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value) else 0
    profile=record.get('profile')
    summary['profile']=_fields(profile,{'id','title','mode'}) if isinstance(profile,dict) else profile[:120] if isinstance(profile,str) else None
    summary['request']=_fields(record.get('request'),{'profile','case_id'})
    summary['source']=_fields(record.get('source'),{'sha256','bytes','width','height','duration_s','frame_count'})
    phases=record.get('phases',[])
    summary['phases']=[_fields(p,PHASE_FIELDS) for p in phases[:100] if isinstance(p,dict)] if isinstance(phases,list) else []
    artifacts=record.get('artifacts')
    summary['artifacts']=_fields(artifacts,{'case','interaction','viewer','observed_reconstruction_report','constrained_scene','object_program'})
    if isinstance(artifacts,dict):
        children=artifacts.get('observed_reconstructions',[])
        if isinstance(children,list):
            summary['artifacts']['observed_reconstructions']=[_fields(c,CHILD_FIELDS)for c in children[:100] if isinstance(c,dict)]
    summary.update(_fields(record,{'logs_url','observed_child_job_id','observed_geometry_has_gaps'}))
    if isinstance(record.get('observed_child_status'),dict):
        summary['observed_child_status']=_fields(record['observed_child_status'],CHILD_FIELDS)
    if isinstance(record.get('sam3d_reconstruction_status'), dict):
        native = record['sam3d_reconstruction_status']
        summary['sam3d_reconstruction_status'] = _fields(native, {
            'provider', 'generated_components', 'native_motion_status', 'old_poses_reused',
            'fallback_backend_used', 'physical_accuracy_verified'})
        pending = native.get('pending_component_ids', [])
        summary['sam3d_reconstruction_status']['pending_component_ids'] = [value for value in pending[:3]
            if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,159}', value)] if isinstance(pending, list) else []
    if isinstance(record.get('mechanism_pipeline_status'),dict):
        summary['mechanism_pipeline_status']=_fields(record['mechanism_pipeline_status'],{
            'topology','scene','object_program','independent_scene_fallback','physical_accuracy_verified','simulation_ready'})
    summary['fullrecord_url']='/api/runs/'+record['id']
    summary['summary_scope']='polling_display_only_full_record_available'
    return summary
