"""Fixed private semantic follow-on for the four owner captures of 2026-11-03.

Reads completed run states only. This is deliberately not a discovery service,
admin publisher, model runner, or generic path-taking API.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

import cable_trace_2d as C
import semantic_asset_companion as S

ROOT = Path(__file__).resolve().parent
BATCH_PATH = 'output/new-captures-20261103/batch-manifest.json'
RUNS = {
    'charger':'run-capture-charger-20261103-v1',
    'phone':'run-capture-phone-20261103-v1',
    'laptop':'run-capture-laptop-20261103-v1',
    'bottle':'run-capture-bottle-consistency-20261103-v2',
}
DEFAULT_OUTPUT = 'output/new-captures-20261103/semantic-review-v1'
REUSE_KEY = 'owner_20261103_charger_cross_recording_candidate'


def _whole_cable(track):
    """A plug-only mask must not become an apparent whole-cable deformation."""
    label = track['label'].lower().replace('_',' ')
    return (bool(re.search(r'\b(?:cable|cord)\b',label)) and
            not re.search(r'\b(?:connector|plug|tip|endpoint|end)\b',label))


def _matching(tracks, pattern, *, exclude=None):
    return [t for t in tracks if re.search(pattern,t['label'],re.I) and
            (exclude is None or not re.search(exclude,t['label'],re.I))]


def _owner_features(kind, program):
    """Narrow fixed owner-statement projections; component mappings stay fallible."""
    tracks = program['tracklets']
    entities, relations, warnings = [], [], []

    def feature(identifier, feature_kind, track, label, attributes=None, *, reuse=False, offscreen=False):
        entity={'id':identifier,'kind':feature_kind,'track_ids':[] if offscreen else [track['id']],
                'feature_label':label,'attributes':attributes or {},'evidence_ids':[]}
        if reuse:entity['owner_reuse_key']=REUSE_KEY
        if offscreen:entity['presence']='owner_reported_offscreen'
        entities.append(entity)
        return identifier

    def relation(identifier, relation_kind, subject, object_):
        relations.append({'id':identifier,'kind':relation_kind,'subject_id':subject,
                          'object_id':object_,'evidence_ids':[]})

    def one(candidates, role):
        if len(candidates)==1:return candidates[0]
        warnings.append({'kind':'semantic_host_unresolved','role':role,
                         'candidate_track_ids':[t['id']for t in candidates],
                         'reason':'No unique actual saved track; no host or feature geometry invented.'})
        return None

    if kind=='charger':
        chargers=_matching(tracks,r'\bcharger\b|\bpower[ _-]+adapter\b')
        charger_ports=[]
        for index,track in enumerate(chargers):
            host=feature(f'owner_charger_{index}','charger',track,'Owner-described charger tracklet; physical continuity remains unverified',reuse=True)
            port=feature(f'owner_charger_port_{index}','receptacle',track,'Owner-reported USB-C charger port; location unresolved',{'standard':'usb_c','role':'port'})
            relation(f'owner_charger_port_of_{index}','feature_of',port,host)
            charger_ports.append(port)
        if not chargers:warnings.append({'kind':'charger_not_visible_as_saved_track','reason':'No charger mesh or visible identity invented.'})
        cables=[t for t in tracks if _whole_cable(t)]
        for index,track in enumerate(cables):
            cable=feature(f'owner_flexible_cable_{index}','flexible_cable',track,'Owner-described flexible USB-C cable',{'deformability':'nonrigid_hypothesis'})
            for end in ['a','b']:
                plug=feature(f'owner_cable_{index}_end_{end}','connector_endpoint',track,
                    f'Owner-reported USB-C cable end {end.upper()}; no spatial or cross-frame endpoint assignment',{'standard':'usb_c','role':'plug'})
                relation(f'owner_cable_{index}_end_{end}_of','feature_of',plug,cable)
                for port_index,port in enumerate(charger_ports):
                    relation(f'owner_connection_candidate_{index}_{end}_{port_index}','connection_candidate',plug,port)
        if len(chargers)>1:warnings.append({'kind':'multiple_charger_tracklets_unmerged','track_ids':[t['id']for t in chargers],
            'reason':'A shared owner reuse key is only a cross-recording candidate; early/late tracklets are not proven identical.'})
    elif kind=='phone':
        phone=one(_matching(tracks,r'\b(?:smartphone|phone)\b'),'phone')
        port=None
        if phone:
            host=feature('owner_phone','phone',phone,'Owner-described phone')
            port=feature('owner_phone_port','receptacle',phone,'Owner-reported phone USB-C port; no metric pose',{'standard':'usb_c','role':'port'})
            relation('owner_phone_port_of','feature_of',port,host)
        for index,track in enumerate(_matching(tracks,r'\b(?:plug|connector)\b')):
            plug=feature(f'owner_phone_plug_{index}','connector_endpoint',track,'Owner-reported USB-C plug; saved mask covers connector, not the whole cable',{'standard':'usb_c','role':'plug'})
            if port:relation(f'owner_phone_connection_{index}','connection_candidate',plug,port)
        chargers=_matching(tracks,r'\bcharger\b|\bpower[ _-]+adapter\b')
        if chargers:
            for index,track in enumerate(chargers):feature(f'owner_phone_charger_{index}','charger',track,'Owner-suggested same charger; identity unverified',reuse=True)
        else:feature('owner_reused_charger_reference','charger',None,'Owner-suggested reused charger, not visible as a saved phone-recording track',reuse=True,offscreen=True)
    elif kind=='laptop':
        bases=_matching(tracks,r'\b(?:laptop|keyboard)\b',exclude=r'\b(?:lid|display|screen|cable|plug|connector)\b')
        base=one(bases,'laptop keyboard/base')
        port=None
        if base:
            host=feature('owner_laptop','laptop',base,'Owner-described laptop host')
            button=feature('owner_laptop_power_button','power_button',base,'Owner-described power button; region and response unverified',{'activation':'press','effect':'request_power_change'})
            relation('owner_laptop_power_button_of','feature_of',button,host)
            relation('owner_laptop_button_control','controls',button,host)
            port=feature('owner_laptop_power_port','receptacle',base,'Owner-described charger port; connector standard and position unknown',{'standard':'unknown','role':'port'})
            relation('owner_laptop_port_of','feature_of',port,host)
        for index,track in enumerate([t for t in tracks if _whole_cable(t)]):
            cable=feature(f'owner_laptop_cable_{index}','flexible_cable',track,'Owner-described charger cable; deformable 3D representation deferred',{'deformability':'unknown'})
            plug=feature(f'owner_laptop_cable_plug_{index}','connector_endpoint',track,'Logical charger-cable endpoint; no measured endpoint identity or pose',{'standard':'unknown','role':'plug'})
            relation(f'owner_laptop_plug_of_{index}','feature_of',plug,cable)
            if port:relation(f'owner_laptop_connection_{index}','connection_candidate',plug,port)
    elif kind=='bottle':
        bottle=one(_matching(tracks,r'\bbottle\b',exclude=r'\b(?:cap|lid|ring|neck|upper|lower|base|top|bottom)\b'),'whole bottle body')
        cap=one(_matching(tracks,r'\bcap\b',exclude=r'\b(?:ring|tether)\b'),'bottle cap')
        receiver=one(_matching(tracks,r'\b(?:drinking glass|glass|cup|tumbler)\b'),'receiving drinking vessel')
        source=feature('owner_source_bottle','vessel',bottle,'Owner-described water bottle; liquid amount and physics unknown',{'contents':'owner_reports_water'})if bottle else None
        if receiver:
            target=feature('owner_receiving_vessel','vessel',receiver,'Owner-described receiving drinking glass',{'contents':'unknown'})
            if source:relation('owner_pour_relation','pour_transfer',source,target)
        if cap:
            closure=feature('owner_screw_cap','threaded_cap',cap,'Owner-described screw cap; thread disengagement does not establish free-body removal',{'sequence':'rotate_then_disengage','retention':'unknown'})
            if source:relation('owner_threaded_closure','threaded_closure',closure,source)
        warnings.append({'kind':'owner_detaches_word_not_free_body_proof',
            'reason':'Literal owner context is preserved. Only corrected saved model evidence may establish a retained tether; no free cap or rigid hinge is inferred.'})
    else:raise ValueError('Unknown fixed capture profile')
    return entities,relations,warnings


def _inputs(capture_ids,root):
    batch_ref=S.file_ref(BATCH_PATH,root,128000)
    batch=S._json(S.checked(batch_ref,root,128000))
    if batch.get('format')!='interaction-twin-gpu-priority-capture-batch' or not isinstance(batch.get('captures'),list):
        raise ValueError('Exact fixed owner capture batch required')
    rows=[]
    for kind in capture_ids:
        matches=[(index,c)for index,c in enumerate(batch['captures'])if c.get('id')==kind]
        if len(matches)!=1:raise ValueError('Exactly one fixed source recording per capture required')
        index,capture=matches[0]
        if capture.get('owner_context_is_observation')is not False:raise ValueError('Owner context must remain separately unverified')
        S.checked(capture['source_ref'],root,512*1024*1024)
        run_id=RUNS[kind]
        state_path=f'data/flow-runs/{run_id}/run.json'
        state_ref=S.file_ref(state_path,root,8*1024*1024)
        state=S._json(S.checked(state_ref,root))
        if state.get('id')!=run_id or state.get('status')not in{'complete','complete_with_gaps'}:
            raise ValueError(f'{run_id} has not completed; no partial pipeline adoption')
        state_source=state.get('source')
        if (not isinstance(state_source,dict)or state_source.get('sha256')!=capture['source_ref']['sha256']or
            type(state_source.get('bytes'))is not int or state_source['bytes']!=capture['source_ref']['bytes']):
            raise ValueError('Completed run state source differs from its exact owner capture path, SHA or byte count')
        try:state_source_ref=S.file_ref(state_source.get('path'),root,512*1024*1024)
        except(ValueError,OSError):raise ValueError('Completed run state source path is unavailable or outside the ordinary project artifacts')from None
        if state_source_ref!=capture['source_ref']:raise ValueError('Completed run state source differs from its exact owner capture path, SHA or byte count')
        interaction=state.get('artifacts',{}).get('interaction')
        if not isinstance(interaction,str)or not interaction.startswith('data/flow-capture-')or not interaction.endswith('/interaction.json'):
            raise ValueError('Completed run must provide its exact ordinary interaction artifact')
        if kind=='bottle'and not interaction.startswith('data/flow-capture-bottle-consistency-20261103-v2/'):
            raise ValueError('The obsolete split-body bottle branch is never an allowed input')
        if not interaction.startswith('data/'+run_id.replace('run-capture-','flow-capture-',1)+'/'):
            raise ValueError('Run interaction must remain inside its exact fixed capture lineage')
        reference,program=S._program(interaction,root)
        if (state.get('artifacts',{}).get('case')!=program['case_id'] or
            capture['source_ref']['sha256']not in{program['source']['sha256'],program['source'].get('original_sha256')}):
            raise ValueError('Run, program and owner batch must bind the same exact source and case')
        rows.append({'capture_id':kind,'run_id':run_id,'run_state_ref':state_ref,'run_status':state['status'],
                     'program_ref':reference,'program':program,'owner_note':capture['owner_context'],
                     'owner_context_pointer':f'/captures/{index}/owner_context'})
    return batch_ref,rows


def build_batch(*,capture_ids=None,output_dir=DEFAULT_OUTPUT,trace_cables=True,root=ROOT):
    """One bounded invocation; any incomplete run fails before output writes."""
    root=Path(root).resolve()
    capture_ids=list(RUNS)if capture_ids is None else list(capture_ids)
    if (not capture_ids or len(capture_ids)>4 or any(not isinstance(k,str)or k not in RUNS for k in capture_ids)or
        len(set(capture_ids))!=len(capture_ids)or type(trace_cables)is not bool):raise ValueError('Choose unique fixed capture IDs only')
    capture_ids=[kind for kind in RUNS if kind in capture_ids]
    batch_ref,rows=_inputs(capture_ids,root)
    results=[]
    for row in rows:
        kind=row['capture_id'];program=row['program'];directory=Path(output_dir)/kind
        entities,relations,warnings=_owner_features(kind,program)
        context=S.make_owner_context(row['program_ref']['path'],entities=entities,relations=relations,note=row['owner_note'],root=root,
            context_source_ref=batch_ref,context_source_pointer=row['owner_context_pointer'])
        context_ref=S.write_context(context,directory,root=root)
        semantic=S.build(row['program_ref']['path'],owner_context_path=context_ref['path'],root=root)
        semantic_ref=S.write_companion(semantic,directory,root=root)
        traces=[];skipped=[]
        for track in program['tracklets']:
            if _whole_cable(track):
                if trace_cables:
                    trace=C.build(row['program_ref']['path'],track['id'],root=root)
                    trace_ref=C.write(trace,directory,root=root)
                    counts={}
                    for sample in trace['samples']:counts[sample['status']]=counts.get(sample['status'],0)+1
                    traces.append({'track_id':track['id'],'trace_ref':trace_ref,'source_program_ref':row['program_ref'],
                                   'coordinate_scope':'2d_saved_mask_pixels_no_depth_or_physical_endpoint_identity','samples_by_status':counts})
                else:skipped.append({'track_id':track['id'],'reason':'Optional local tracing explicitly disabled.'})
            elif re.search(r'\b(?:cable|cord|connector|plug)\b',track['label'],re.I):
                skipped.append({'track_id':track['id'],'reason':'Saved track is not an explicit whole-cable label; a connector-only mask is not a deforming cable.'})
        results.append({k:row[k]for k in['capture_id','run_id','run_state_ref','run_status','program_ref']}|
            {'owner_context_ref':context_ref,'semantic_companion_ref':semantic_ref,'visible_2d_cable_traces':traces,
             'untraced_track_reasons':skipped,'review_warnings':warnings,
             'native_3d_cable_deformation':'deferred_static_SAM3D_appearance_does_not_measure_deformation'})
    catalog=S.build_catalog([r['semantic_companion_ref']['path']for r in results],root=root)
    catalog_ref=S.write_catalog(catalog,output_dir,root=root)
    # A completed run or batch changing during preparation invalidates adoption.
    S.checked(batch_ref,root,128000)
    for row in rows:S.checked(row['run_state_ref'],root)
    manifest={'format':'interaction-twin-private-capture-semantic-batch','version':1,'batch_ref':batch_ref,
              'records':results,'relation_catalog_ref':catalog_ref,
              'scope':'private_additive_owner_review_candidates_not_public_library',
              'claims':copy.deepcopy(S.CLAIMS),'new_model_calls':0,'database_changed':False,
              'existing_pipeline_outputs_changed':False,'publication_performed':False,
              'bottle_input_policy':'Only corrected consistency-v2 run; obsolete split-body run excluded.',
              'whole_cable_trace_policy':'Exact explicit whole-cable labels only; no plug-only masks or inferred invisible spans.'}
    manifest_ref=S._write(manifest,output_dir,root,'semantic-batch')
    return manifest_ref,manifest
