"""Cadence-derived editorial repetition, separate from .18s observations.

This never infers/interpolates/currently tracks geometry. Unknown ledger rows
remain barriers. The strict source-observation query is unchanged.
"""
from pathlib import Path
import hashlib,json,math
FORMAT='interaction-twin-observed-surface-presentation'
def require(value,message):
 if not value:raise ValueError(message)
def reference(p,root):
 p=Path(p);raw=p.read_bytes();return {'path':p.relative_to(root).as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
def derive(descriptor,descriptor_ref):
 d=descriptor;require(d.get('format')=='interaction-twin-source-visible-flexure-series'and d.get('kind')=='source_observed_monocular_depth_prior','Source-observation descriptor required');require(d['policy']['maximum_source_time_error_s']==.18,'Strict source evidence query must remain .18s');rows=d['samples'];out=[];duration=d['source']['duration_s'];require(len(rows)>1,'Adjacent source cadence required')
 for i,r in enumerate(rows):
  t=r['timestamp_s'];left=t-rows[i-1]['timestamp_s']if i else rows[1]['timestamp_s']-t;right=rows[i+1]['timestamp_s']-t if i+1<len(rows)else t-rows[i-1]['timestamp_s'];require(left>0 and right>0,'Strict source time order required')
  out.append({'source_sample_index':r['source_sample_index'],'observation_timestamp_s':t,'status':r['status'],'left_actual_source_gap_s':left,'right_actual_source_gap_s':right,'display_window_start_s':max(0.,t-min(.55,left/2)),'display_window_end_s':min(duration,t+min(.55,right/2)),'visible_component_ids':r.get('visible_component_ids',[])if r['status']=='source_observed_prior'else[]})
 return {'format':FORMAT,'version':1,'source_observations':descriptor_ref,'source_sha256':d['source']['sha256'],'source_duration_s':duration,'mode':'editorial_source_sample_repeat','strict_observation_query_maximum_source_time_error_s':.18,'display_repeat_cap_s':.55,'bound_derivation':'each_side_minimum_of0.55s_and_half_actual_adjacent_source_gap','nearest_row_includes_unknowns':True,'search_past_unknown_rows':False,'geometry_interpolated':False,'current_geometry':False,'new_tracking_or_geometry_recovered':False,'physical_accuracy_verified':False,'show_estimated_required':True,'required_visible_label':'Observed surface · {observation_timestamp_s}s','required_return_fields':['observation_timestamp_s','source_time_error_s','presentation_mode','interpolated','current_geometry'],'samples':out}
def validate(policy,descriptor,descriptor_ref):
 require(policy==derive(descriptor,descriptor_ref),'Editorial policy differs from exact source cadence or uncertainty fence');return policy
def at(policy,descriptor,time_s,show_estimated=True):
 if not show_estimated or isinstance(time_s,bool)or not isinstance(time_s,(int,float))or not math.isfinite(time_s)or not 0<=time_s<=policy['source_duration_s']:return None
 rows=policy['samples'];nearest=min(rows,key=lambda r:(abs(r['observation_timestamp_s']-time_s),r['observation_timestamp_s']))
 if nearest['status']!='source_observed_prior'or not nearest['display_window_start_s']-1e-12<=time_s<=nearest['display_window_end_s']+1e-12:return None
 original=next(r for r in descriptor['samples']if r['source_sample_index']==nearest['source_sample_index']);require(original['status']=='source_observed_prior','Editorial sample points to unknown observation')
 return {**original,'observation_timestamp_s':nearest['observation_timestamp_s'],'source_time_error_s':abs(time_s-nearest['observation_timestamp_s']),'requested_video_time_s':time_s,'presentation_mode':'editorial_source_sample_repeat','representation':'source_visible_2.5D_depth_prior','interpolated':False,'current_geometry':False,'new_tracking_or_geometry_recovered':False}
