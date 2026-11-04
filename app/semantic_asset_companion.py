"""Private semantic features and cross-recording candidates, never physical truth.

This additive descriptor never changes an evidence program, mesh, mechanism,
curation decision or API route. Owner annotations remain separately sourced
beliefs. No language model, generated code, shell command or robot is run.
"""
from __future__ import annotations
import copy,hashlib,json,os,re,tempfile
from itertools import combinations
from pathlib import Path
from agent_memory import digest
from stage5 import validate_interaction

ROOT=Path(__file__).resolve().parent
FORMAT='interaction-twin-semantic-asset-companion'
ID=re.compile(r'[A-Za-z][A-Za-z0-9_-]{0,79}\Z')
ATTRIBUTES={
 'charger':{},'phone':{},'laptop':{},'container':{},
 'connector_endpoint':{'standard':{'usb_c','unknown'},'role':{'plug','unknown'}},
 'receptacle':{'standard':{'usb_c','unknown'},'role':{'port','unknown'}},
 'power_button':{'activation':{'press','unknown'},'effect':{'request_power_change','unknown'}},
 'threaded_cap':{'sequence':{'rotate_then_disengage','unknown'},'retention':{'tethered_hypothesis','free_body_hypothesis','unknown'}},
 'vessel':{'contents':{'owner_reports_water','unknown'}},
 'flexible_cable':{'deformability':{'nonrigid_hypothesis','unknown'}},
}
RELATIONS={'feature_of','connection_candidate','controls','threaded_closure','retained_tether','pour_transfer','deformable_component_of'}
CLAIMS={'physical_identity_verified':False,'certified_compatibility':False,'metric_port_poses_available':False,
 'physical_accuracy_verified':False,'water_physics_recovered':False,'cable_deformation_recovered':False,
 'automatic_asset_deduplication':False,'training_performed':False,'public_or_owner_approved':False,'new_model_calls':0}

def _keys(value,required,optional=()):
    if not isinstance(value,dict)or not set(required)<=set(value)or not set(value)<=set(required)|set(optional):raise ValueError('Unknown or missing semantic descriptor fields')

def _text(value,maximum=1000):
    if not isinstance(value,str)or not 0<len(value)<=maximum or'\x00'in value:raise ValueError('Bounded plain text required')
    return value

def _json(raw):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('Duplicate semantic JSON field')
            result[key]=value
        return result
    def constant(_value):raise ValueError('Nonfinite semantic JSON value')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)

def _ordinary(path,root,maximum):
    if not isinstance(path,(str,Path)):raise ValueError('Ordinary project artifact path required')
    root=Path(root).resolve();path=Path(path)
    if not path.is_absolute():path=root/path
    if (not path.is_relative_to(root)or path.resolve()!=path or any(p.is_symlink()for p in[path,*path.parents])or
        any(part.startswith('.')for part in path.relative_to(root).parts)or not path.is_file()or not 0<path.stat().st_size<=maximum):raise ValueError('Bounded ordinary project artifact required')
    return path

def file_ref(path,root=ROOT,maximum=512*1024*1024):
    root=Path(root).resolve();path=_ordinary(path,root,maximum);h=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return{'path':path.relative_to(root).as_posix(),'sha256':h.hexdigest(),'bytes':path.stat().st_size}

def checked(reference,root=ROOT,maximum=8*1024*1024):
    _keys(reference,['path','sha256','bytes'])
    if (not isinstance(reference['path'],str)or Path(reference['path']).is_absolute()or'\\'in reference['path']or
        type(reference['bytes'])is not int or not 0<reference['bytes']<=maximum or
        not isinstance(reference['sha256'],str)or not re.fullmatch('[a-f0-9]{64}',reference['sha256'])):raise ValueError('Exact bounded semantic artifact reference required')
    path=_ordinary(reference['path'],root,maximum)
    if file_ref(path,root,maximum)!=reference:raise ValueError('Semantic input bytes changed')
    raw=path.read_bytes()
    if len(raw)!=reference['bytes']or hashlib.sha256(raw).hexdigest()!=reference['sha256']:raise ValueError('Semantic input changed during read')
    return raw

def _program(path,root):
    reference=file_ref(path,root,8*1024*1024);document=_json(checked(reference,root))
    if not isinstance(document,dict):raise ValueError('A typed evidence procedure is required')
    if any(not isinstance(document.get(k),list)or len(document[k])>maximum for k,maximum in [('tracklets',128),('assets',128),('steps',1024),('evidence',10000)]):raise ValueError('Bounded semantic program inspection required')
    validate_interaction(document,root=root,verify_files=True)
    # The original validator verifies content; additionally require ordinary
    # nonsymlink paths for every source/frame/mask/asset this companion cites.
    for item in [document['source'],*document['inputs'],*document['assets']]:_ordinary(item['path'],root,512*1024*1024)
    for frame in document['evidence']:
        _ordinary(frame['frame_path'],root,16*1024*1024)
        for observation in frame['observations']:_ordinary(observation['mask_path'],root,16*1024*1024)
    return reference,document

def make_owner_context(program_path,*,entities=(),relations=(),note,root=ROOT,origin='explicit_owner_context',context_source_ref=None,context_source_pointer=None):
    reference,program=_program(program_path,root)
    value={'format':'interaction-twin-owner-semantic-context','version':1,'program_ref':reference,
        'source_sha256':program['source']['sha256'],'note':_text(note,4000),'encoded_by':'operator',
        'origin':origin,'component_binding_owner_confirmed':False,
        'entities':copy.deepcopy(list(entities)),'relations':copy.deepcopy(list(relations)),
        'untrusted_annotation':True,'physical_truth_verified':False}
    if context_source_ref is not None or context_source_pointer is not None:value.update(context_source_ref=context_source_ref,context_source_pointer=context_source_pointer)
    _context(value,reference,program,root);return value

def _context(value,reference,program,root=ROOT):
    _keys(value,['format','version','program_ref','source_sha256','note','encoded_by','origin','component_binding_owner_confirmed','entities','relations','untrusted_annotation','physical_truth_verified'],['context_source_ref','context_source_pointer'])
    if (value['format']!='interaction-twin-owner-semantic-context'or type(value['version'])is not int or value['version']!=1 or
        value['program_ref']!=reference or value['source_sha256']!=program['source']['sha256']or value['encoded_by']!='operator'or
        not isinstance(value['origin'],str)or value['origin']not in{'explicit_owner_context','source_review_annotation'}or value['component_binding_owner_confirmed']is not False or
        value['untrusted_annotation']is not True or value['physical_truth_verified']is not False):raise ValueError('Owner context must retain its exact source and unverified provenance')
    _text(value['note'],4000);tracks={t['id']for t in program['tracklets']};evidence={e['id']for e in program['evidence']}
    if 'context_source_ref'in value or'context_source_pointer'in value:
        pointer=value.get('context_source_pointer');document=_json(checked(value.get('context_source_ref'),root,128000));source_document=document
        if not isinstance(pointer,str)or not re.fullmatch(r'(?:/[A-Za-z0-9_-]+){1,8}',pointer):raise ValueError('A bounded literal context source pointer is required')
        try:
            for key in pointer.split('/')[1:]:document=document[int(key)]if isinstance(document,list)and key.isdecimal()else document[key]
        except(KeyError,IndexError,TypeError,ValueError):raise ValueError('Context source pointer is unavailable')
        if document!=value['note']:raise ValueError('Context note differs from its literal source')
        if isinstance(source_document,dict)and source_document.get('format')=='interaction-twin-gpu-priority-capture-batch':
            if not re.fullmatch(r'/captures/[0-9]+/owner_context',pointer):raise ValueError('Choose an exact capture owner-context field')
            capture=source_document['captures'][int(pointer.split('/')[2])];original=capture['source_ref'];checked(original,root,512*1024*1024)
            if original['sha256']not in{program['source']['sha256'],program['source'].get('original_sha256')}:raise ValueError('Capture owner context belongs to another source recording')
    if not isinstance(value['entities'],list)or len(value['entities'])>32 or not isinstance(value['relations'],list)or len(value['relations'])>64:raise ValueError('Bounded semantic context required')
    ids=set()
    for entity in value['entities']:
        _keys(entity,['id','kind','track_ids','feature_label','attributes','evidence_ids'],['owner_reuse_key','presence'])
        if not isinstance(entity['id'],str)or not ID.fullmatch(entity['id'])or entity['id']in ids or not isinstance(entity['kind'],str)or entity['kind']not in ATTRIBUTES:raise ValueError('Known unique semantic entity required')
        ids.add(entity['id']);_text(entity['feature_label'],200)
        presence=entity.get('presence','track_bound')
        if not isinstance(presence,str)or presence not in{'track_bound','owner_reported_offscreen'}:raise ValueError('Unknown semantic feature presence')
        if presence=='owner_reported_offscreen'and(value['origin']!='explicit_owner_context'or entity['kind']!='charger'or entity['track_ids']or entity['evidence_ids']or'owner_reuse_key'not in entity):raise ValueError('Offscreen charger references need an owner reuse candidate, without tracks or geometry')
        for key,allowed,maximum in [('track_ids',tracks,8),('evidence_ids',evidence,128)]:
            if not isinstance(entity[key],list)or len(entity[key])>maximum or any(not isinstance(v,str)for v in entity[key])or len(set(entity[key]))!=len(entity[key])or not set(entity[key])<=allowed or(key=='track_ids'and not entity[key]and presence!='owner_reported_offscreen'):raise ValueError('Semantic feature references unknown tracks or evidence')
        attributes=entity['attributes']
        if not isinstance(attributes,dict)or not set(attributes)<=set(ATTRIBUTES[entity['kind']])or any(not isinstance(v,str)or v not in ATTRIBUTES[entity['kind']][k]for k,v in attributes.items()):raise ValueError('Unknown semantic attribute; metric poses and certified standards are not accepted')
        if value['origin']=='source_review_annotation'and not entity['evidence_ids']:raise ValueError('Source review annotations require exact program evidence IDs')
        if 'owner_reuse_key'in entity and(entity['kind']!='charger'or not isinstance(entity['owner_reuse_key'],str)or not ID.fullmatch(entity['owner_reuse_key'])):raise ValueError('Only a bounded owner-suggested charger reuse key is supported')
    relation_ids=set();by_id={e['id']:e for e in value['entities']}
    for relation in value['relations']:
        _keys(relation,['id','kind','subject_id','object_id','evidence_ids'])
        if (not isinstance(relation['id'],str)or not ID.fullmatch(relation['id'])or relation['id']in relation_ids or not isinstance(relation['kind'],str)or relation['kind']not in RELATIONS or
            not isinstance(relation['subject_id'],str)or not isinstance(relation['object_id'],str)or
            relation['subject_id']not in ids or relation['object_id']not in ids or relation['subject_id']==relation['object_id']):raise ValueError('Known unique semantic relation endpoints required')
        relation_ids.add(relation['id']);ev=relation['evidence_ids']
        if not isinstance(ev,list)or len(ev)>128 or any(not isinstance(v,str)for v in ev)or len(set(ev))!=len(ev)or not set(ev)<=evidence:raise ValueError('Unknown semantic relation evidence')
        if value['origin']=='source_review_annotation'and not ev:raise ValueError('Source review relations require exact program evidence IDs')
        a,b=by_id[relation['subject_id']]['kind'],by_id[relation['object_id']]['kind'];kind=relation['kind']
        valid=(kind=='feature_of'or kind=='connection_candidate'and a=='connector_endpoint'and b=='receptacle'or
            kind=='controls'and a=='power_button'and b=='laptop'or kind in{'threaded_closure','retained_tether'}and a=='threaded_cap'and b in{'vessel','container'}or
            kind=='pour_transfer'and a==b=='vessel'or kind=='deformable_component_of'and a=='flexible_cable')
        if not valid:raise ValueError('Semantic relation endpoint types differ')

def _label_kind(label):
    label=label.lower().replace('–','-')
    if re.search(r'\busb[ -]?c\b',label):
        if re.search(r'\b(port|socket|receptacle)\b',label):return'receptacle',{'standard':'usb_c','role':'port'}
        if re.search(r'\b(plug|connector|end)\b',label):return'connector_endpoint',{'standard':'usb_c','role':'plug'}
    rules=[(r'\bpower[ _-]+button\b','power_button',{'activation':'press','effect':'unknown'}),
        (r'\b(?:screw|threaded)[ _-]+(?:cap|lid)\b','threaded_cap',{'sequence':'rotate_then_disengage','retention':'unknown'}),
        (r'\b(?:cable[ _-]+connector|connector[ _-]+plug|plug[ _-]+tip)\b','connector_endpoint',{'standard':'unknown','role':'plug'}),
        (r'\bcharger\b|\bpower[ _-]+adapter\b','charger',{}),(r'\blaptop\b','laptop',{}),(r'\bphone\b','phone',{}),
        (r'\b(?:cable|cord)\b','flexible_cable',{'deformability':'unknown'}),(r'\b(?:bottle|cup|mug|pitcher|jug|vessel)\b','vessel',{'contents':'unknown'})]
    result=next(((kind,attributes)for pattern,kind,attributes in rules if re.search(pattern,label)),None)
    if result and result[0]=='flexible_cable'and re.search(r'\b(flexible|deformable|bending)\b',label):result=('flexible_cable',{'deformability':'nonrigid_hypothesis'})
    return result

def _positive_tether(relation):
    text=relation['rationale']
    if relation['status']not in{'visual_model_proposal','inferred'}:return False
    if re.search(r'\b(?:no|without|unknown|unclear)\b.{0,40}\b(?:tether|retaining ring)\b|\bnot\b.{0,25}\b(?:attached|connected|retained)\b',text,re.I):return False
    return bool(re.search(r'\b(?:remains|retained|connected|attached)\b.{0,100}\b(?:tether|retaining ring)\b|\b(?:tether|retaining ring)\b.{0,60}\b(?:retains|connects|attached)\b',text,re.I))

def _entity(reference,program,local_id,kind,label,tracks,attributes,provenance,evidence_ids,reuse_key=None,presence='track_bound'):
    assets=[]
    for asset in program['assets']:
        if asset['track_id']in tracks:assets.append({'track_id':asset['track_id'],'path':asset['path'],'sha256':asset['sha256'],'status':asset['geometry_status']})
    return{'id':'entity_'+digest([reference['sha256'],provenance,local_id])[:24],'local_id':local_id,'kind':kind,'label':label,
        'track_ids':tracks,'attributes':copy.deepcopy(attributes),'provenance':provenance,'evidence_ids':evidence_ids,
        'presence':presence,'geometry':{'state':'draft_track_asset_available'if assets else'deferred','track_assets':assets,
            'asset_scope':'host_or_component_track_mesh_not_verified_feature_geometry','metric_pose':None,'port_frame':None,
            'cable_centerline':None,'water_mesh_or_particles':None,'thread_pitch':None,'physical_scale_known':False,
            'reference_only_no_new_asset':presence=='owner_reported_offscreen'},
        'owner_reuse_key':reuse_key,'status':'semantic_hypothesis','owner_review':'not_reviewed','physical_identity_verified':False}

def _relation(kind,subject,object_,provenance,evidence_ids,local_id):
    requirements={'connection_candidate':['endpoint_geometry','clearance','connector_and_electrical_compatibility_review'],
        'controls':['button_region','observed_response_review'],'threaded_closure':['thread_disengagement_evidence','retention_or_tether_evidence','thread_geometry_and_pitch','contact_review'],
        'retained_tether':['tether_evidence','deformable_tether_representation','contact_review'],
        'pour_transfer':['source_and_receiver_roles','liquid_state_and_volume','fluid_physics'],
        'deformable_component_of':['deformable_representation','curve_or_material_evidence'],'feature_of':['feature_localization']}[kind]
    return{'id':'relation_'+digest([kind,subject,object_,provenance,local_id])[:24],'local_id':local_id,'kind':kind,'subject_id':subject,'object_id':object_,
        'provenance':provenance,'evidence_ids':evidence_ids,'status':'unverified_candidate','owner_review':'not_reviewed',
        'motion_contract':{'attachment':'conditional_thread_engagement_retention_unknown'if kind=='threaded_closure'else'flexible_tether_retained_hypothesis'if kind=='retained_tether'else'unknown',
            'stages':['rotate_while_engaged','thread_disengagement','retention_requires_evidence']if kind=='threaded_closure'else['thread_disengaged','retained_by_flexible_tether']if kind=='retained_tether'else[],
            'joint_axis':None,'joint_origin':None,'pitch_m_per_turn':None,'release_angle_rad':None,'fluid_physics':None},
        'missing_requirements':requirements,'certified_compatible':False,'physical_accuracy_verified':False}

def _program_relations(program,entities,relations):
    by_track={}
    for entity in entities:
        for track in entity['track_ids']:by_track.setdefault(track,[]).append(entity)
    for index,relation in enumerate(program['relationships']):
        if relation['status']=='unknown':continue
        for a in by_track.get(relation['subject_track_id'],[]):
            for b in by_track.get(relation['object_track_id'],[]):
                kind=None
                if relation['predicate']in{'attached_to','component_of'}and a['kind']=='connector_endpoint'and b['kind']=='receptacle':kind='connection_candidate'
                if relation['predicate']=='removable_from'and a['kind']=='threaded_cap'and b['kind']in{'container','vessel'}:kind='threaded_closure'
                if relation['predicate']=='attached_to'and a['kind']=='threaded_cap'and b['kind']in{'container','vessel'}and _positive_tether(relation):kind='retained_tether'
                if not kind:continue
                provenance={'kind':'program_relationship','json_pointer':f'/relationships/{index}','original_relation_id':relation['id'],'original_status':relation['status']}
                typed_contexts=[e['provenance']['context_ref']for e in[a,b]if'context_ref'in e['provenance']]
                if typed_contexts:provenance.update(feature_typing_context_refs=typed_contexts,feature_typing_owner_confirmed=False)
                relations.append(_relation(kind,a['id'],b['id'],provenance,relation['evidence_ids'],relation['id']))

def build(program_path,*,owner_context_path=None,root=ROOT):
    root=Path(root).resolve();reference,program=_program(program_path,root);entities=[];relations=[];context_ref=None
    tracks={t['id']:t for t in program['tracklets']};automatic={}
    for index,track in enumerate(program['tracklets']):
        classified=_label_kind(track['label'])
        if not classified:continue
        kind,attrs=classified;ev=[e['id']for e in program['evidence']if any(o['track_id']==track['id']for o in e['observations'])]
        origin={'kind':'program_label_classifier','json_pointer':f'/tracklets/{index}/label','literal':track['label'],'classification_is_observation':False}
        entity=_entity(reference,program,track['id'],kind,track['label'],[track['id']],attrs,origin,ev);entities.append(entity);automatic[track['id']]=entity
    if owner_context_path is not None:
        context_ref=file_ref(owner_context_path,root,128000);context=_json(checked(context_ref,root,128000));_context(context,reference,program,root);mapped={}
        for index,item in enumerate(context['entities']):
            origin={'kind':context['origin'],'context_ref':context_ref,'json_pointer':f'/entities/{index}','component_binding_owner_confirmed':False,'observed_fact':False}
            entity=_entity(reference,program,item['id'],item['kind'],item['feature_label'],item['track_ids'],item['attributes'],origin,item['evidence_ids'],item.get('owner_reuse_key'),item.get('presence','track_bound'));entities.append(entity);mapped[item['id']]=entity['id']
        for index,item in enumerate(context['relations']):relations.append(_relation(item['kind'],mapped[item['subject_id']],mapped[item['object_id']],{'kind':context['origin'],'context_ref':context_ref,'json_pointer':f'/relations/{index}','observed_fact':False},item['evidence_ids'],item['id']))
    _program_relations(program,entities,relations)
    by_id={e['id']:e for e in entities};conflicts=[]
    for closure in [r for r in relations if r['kind']=='threaded_closure']:
        for tether in [r for r in relations if r['kind']=='retained_tether']:
            if set(by_id[closure['subject_id']]['track_ids'])==set(by_id[tether['subject_id']]['track_ids'])and set(by_id[closure['object_id']]['track_ids'])==set(by_id[tether['object_id']]['track_ids']):
                closure['motion_contract']['attachment']='thread_disengaged_but_flexible_tether_retained_hypothesis'
                closure['motion_contract']['stages']=['rotate_while_engaged','thread_disengagement','retained_by_flexible_tether']
                closure['motion_contract']['retention_relation_id']=tether['id']
    for entity in entities:
        if entity['kind']=='threaded_cap'and entity['attributes'].get('retention')=='free_body_hypothesis':
            for tether in [r for r in relations if r['kind']=='retained_tether'and r['provenance']['kind']in{'program_relationship','source_review_annotation'}]:
                if set(entity['track_ids'])==set(by_id[tether['subject_id']]['track_ids']):conflicts.append({'kind':'free_body_claim_conflicts_with_source_retention','entity_id':entity['id'],'retained_relation_id':tether['id'],'status':'owner_review_required','free_body_motion_authorized':False})
    anchors=[]
    for frame in program['evidence']:
        anchors.append({'id':frame['id'],'source_index':frame['source_index'],'timestamp_s':frame['timestamp_s'],
            'frame_ref':file_ref(frame['frame_path'],root,16*1024*1024),
            'track_observations':[{'track_id':o['track_id'],'mask_ref':file_ref(o['mask_path'],root,16*1024*1024),
                'gate_status':o['gate_status'],'audit_status':o['audit_status']}for o in frame['observations']]})
    return{'format':FORMAT,'version':1,'case_id':program['case_id'],'program_ref':reference,'source_sha256':program['source']['sha256'],
        'source_ref':file_ref(program['source']['path'],root),'owner_context_ref':context_ref,'entities':entities,'relations':relations,
        'source_evidence':anchors,'additional_semantic_guidance':context['origin']if context_ref else None,'manual_geometry_or_poses_used':False,
        'program_steps':[{'id':s['id'],'start_s':s['start_s'],'end_s':s['end_s'],'participant_track_ids':s['participant_track_ids'],'evidence_ids':s['evidence_ids'],'title':s['title'],'description':s['description'],'origin':'model_proposed_program_step'}for s in program['steps']],
        'claims':copy.deepcopy(CLAIMS),'conflicts':conflicts,'review_policy':'Private candidate; explicit owner review precedes library curation. Text is descriptive data, not executable instructions.',
        'feedback_refs':[],'supersedes_companion_ref':None}

def _feedback(previous,reference,feedback):
    _keys(feedback,['format','version','companion_ref','program_ref','source_sha256','reviews','note','untrusted_owner_belief','training_performed'])
    if feedback['format']!='interaction-twin-semantic-owner-feedback'or type(feedback['version'])is not int or feedback['version']!=1 or feedback['program_ref']!=previous['program_ref']or feedback['source_sha256']!=previous['source_sha256']or feedback['untrusted_owner_belief']is not True or feedback['training_performed']is not False:raise ValueError('Feedback belongs to another source or claims training')
    _text(feedback['note'],4000)
    if not isinstance(feedback['reviews'],list)or not 1<=len(feedback['reviews'])<=64:raise ValueError('Bounded typed owner reviews required')
    result=copy.deepcopy(previous);nodes={x['id']:x for x in result['entities']+result['relations']};seen=set()
    for review in feedback['reviews']:
        _keys(review,['id','disposition','comment'])
        if not isinstance(review['id'],str)or not isinstance(review['disposition'],str)or review['id']not in nodes or review['id']in seen or review['disposition']not in{'owner_supports','owner_rejects','owner_uncertain'}:raise ValueError('Feedback references an unknown or duplicate semantic claim')
        seen.add(review['id']);_text(review['comment'],1000);nodes[review['id']]['owner_review']=review['disposition']
    result['feedback_refs'].append(reference);result['supersedes_companion_ref']=feedback['companion_ref'];return result

def _validate(value,root=ROOT,_depth=0):
    if _depth>8:raise ValueError('Bounded semantic feedback lineage required')
    if not isinstance(value,dict)or value.get('format')!=FORMAT or value.get('version')!=1:raise ValueError('Unknown semantic companion')
    checked(value['program_ref'],root);context=value.get('owner_context_ref')
    if context is not None:checked(context,root,128000)
    if value.get('supersedes_companion_ref')is not None:
        prior_ref=value['supersedes_companion_ref'];checked(prior_ref,root);prior=load(prior_ref['path'],root,_depth+1)
        if len(value['feedback_refs'])!=len(prior['feedback_refs'])+1 or value['feedback_refs'][:-1]!=prior['feedback_refs']:raise ValueError('Semantic feedback lineage changed')
        feedback_ref=value['feedback_refs'][-1];feedback=_json(checked(feedback_ref,root,128000))
        if feedback['companion_ref']!=prior_ref:raise ValueError('Feedback must bind the exact previous semantic revision')
        expected=_feedback(prior,feedback_ref,feedback)
    else:expected=build(value['program_ref']['path'],owner_context_path=context['path']if context else None,root=root)
    if value!=expected:raise ValueError('Semantic descriptor differs from source-bound derivation')
    return value

def load(path,root=ROOT,_depth=0):
    reference=file_ref(path,root);value=_json(checked(reference,root));return _validate(value,root,_depth)

def append_feedback(companion_path,feedback_path,*,root=ROOT):
    previous=load(companion_path,root);ref=file_ref(companion_path,root);feedback_ref=file_ref(feedback_path,root,128000);feedback=_json(checked(feedback_ref,root,128000))
    if feedback.get('companion_ref')!=ref:raise ValueError('Feedback must bind the exact previous semantic revision')
    if len(previous['feedback_refs'])>=8:raise ValueError('Bounded semantic feedback lineage required')
    return _feedback(previous,feedback_ref,feedback)

def _write(value,output_dir,root,prefix):
    root=Path(root).resolve();folder=Path(output_dir)
    if not folder.is_absolute():folder=root/folder
    if not(folder.is_relative_to(root/'data')or folder.is_relative_to(root/'output'))or folder.resolve()!=folder or any(p.is_symlink()for p in[folder,*folder.parents])or any(p.startswith('.')for p in folder.relative_to(root).parts):raise ValueError('Ordinary private-review data/output directory required')
    folder.mkdir(parents=True,exist_ok=True);raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    if len(raw)>8*1024*1024:raise ValueError('Bounded semantic companion required')
    path=folder/(prefix+'-'+digest(value)+'.json');fd,name=tempfile.mkstemp(prefix='.semantic-',dir=folder)
    try:
        with os.fdopen(fd,'wb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        try:os.link(name,path)
        except FileExistsError:
            if path.is_symlink()or path.read_bytes()!=raw:raise ValueError('Immutable semantic output changed')
        path.chmod(0o600)
    finally:
        if os.path.exists(name):os.unlink(name)
    return file_ref(path,root)

def write_companion(value,output_dir,*,root=ROOT):
    _validate(value,root);return _write(value,output_dir,root,'semantic')

def write_context(value,output_dir,*,root=ROOT):
    reference,program=_program(value['program_ref']['path'],root);_context(value,reference,program,root)
    return _write(value,output_dir,root,'semantic-context')

def build_catalog(companion_paths,*,root=ROOT):
    if not isinstance(companion_paths,(list,tuple))or not 1<=len(companion_paths)<=32:raise ValueError('Supply one to32 explicit private companion paths')
    records=[];entities=[];seen=set()
    for path in companion_paths:
        ref=file_ref(path,root);value=load(path,root)
        if value['program_ref']['sha256']in seen:raise ValueError('Choose one explicit semantic revision per exact program')
        seen.add(value['program_ref']['sha256']);records.append({'companion_ref':ref,'program_ref':value['program_ref'],'source_sha256':value['source_sha256'],'case_id':value['case_id']})
        entities.extend({**e,'source_sha256':value['source_sha256'],'companion_ref':ref}for e in value['entities'])
    if len(entities)>1024:raise ValueError('Bounded cross-asset entity inventory required')
    reuse=[];connections=[];reuse_total=0;connection_total=0
    eligible=[e for e in entities if e['owner_review']!='owner_rejects']
    for a,b in combinations(sorted(eligible,key=lambda e:e['id']),2):
        if a['source_sha256']==b['source_sha256']:continue
        if a['kind']==b['kind']=='charger'and a['owner_reuse_key']and a['owner_reuse_key']==b['owner_reuse_key']:
            reuse_total+=1
            if len(reuse)<128:reuse.append({'kind':'possible_reuse','entity_ids':[a['id'],b['id']],'owner_reuse_key':a['owner_reuse_key'],'basis':'separately_source_bound_owner_suggestion',
                'status':'candidate_requires_owner_identity_review','canonical_asset_id':None,'identity_verified':False,'deduplication_performed':False})
        endpoint,port=(a,b)if a['kind']=='connector_endpoint'else(b,a)
        if endpoint['kind']=='connector_endpoint'and port['kind']=='receptacle'and endpoint['attributes'].get('standard')==port['attributes'].get('standard')=='usb_c':
            connection_total+=1
            if len(connections)<128:connections.append({'kind':'cross_asset_connection_candidate','endpoint_id':endpoint['id'],'port_id':port['id'],
                'basis':'matching_unverified_semantic_USB_C_claims','certified_compatible':False,'mating_pose':None,'electrical_compatibility_verified':False})
    return{'format':'interaction-twin-private-semantic-relation-catalog','version':1,'records':records,'entities':entities,'possible_reuse':reuse,
        'connection_candidates':connections,'connection_candidate_limit':128,'possible_reuse_total':reuse_total,'connection_candidate_total':connection_total,
        'possible_reuse_has_more':reuse_total>len(reuse),'connection_candidates_have_more':connection_total>len(connections),'claims':copy.deepcopy(CLAIMS),
        'scope':'explicit_private_candidates_not_curated_public_library','discovery_performed':False,'public_endpoints_changed':False}

def load_catalog(path,root=ROOT):
    reference=file_ref(path,root);value=_json(checked(reference,root))
    if not isinstance(value,dict)or value.get('format')!='interaction-twin-private-semantic-relation-catalog'or value.get('version')!=1:raise ValueError('Unknown private semantic catalog')
    refs=[r['companion_ref']for r in value['records']]
    for ref in refs:checked(ref,root)
    expected=build_catalog([r['path']for r in refs],root=root)
    if value!=expected:raise ValueError('Private semantic catalog differs from exact companion inputs')
    return value

def write_catalog(value,output_dir,*,root=ROOT):
    refs=[r['companion_ref']for r in value['records']]
    for ref in refs:checked(ref,root)
    if value!=build_catalog([r['path']for r in refs],root=root):raise ValueError('Catalog must match its exact explicit companions')
    return _write(value,output_dir,root,'semantic-catalog')

def query_catalog(catalog,*,kind=None,standard=None,geometry_state=None,include_rejected=False,limit=32,offset=0):
    if catalog.get('format')!='interaction-twin-private-semantic-relation-catalog'or catalog.get('version')!=1:raise ValueError('Load an exact private semantic catalog first')
    if (kind is not None and(not isinstance(kind,str)or kind not in ATTRIBUTES)or standard is not None and(not isinstance(standard,str)or standard not in{'usb_c','unknown'})or geometry_state is not None and(not isinstance(geometry_state,str)or geometry_state not in{'draft_track_asset_available','deferred'})or type(include_rejected)is not bool or type(limit)is not int or not 1<=limit<=128 or type(offset)is not int or not 0<=offset<=1024):raise ValueError('Unknown bounded semantic query')
    rows=[e for e in catalog['entities']if(kind is None or e['kind']==kind)and(standard is None or e['attributes'].get('standard')==standard)and(geometry_state is None or e['geometry']['state']==geometry_state)and(include_rejected or e['owner_review']!='owner_rejects')]
    rows.sort(key=lambda e:(e['kind'],e['label'].casefold(),e['id']))
    return{'entities':copy.deepcopy(rows[offset:offset+limit]),'total':len(rows),'offset':offset,'limit':limit,'has_more':offset+limit<len(rows),'certified_compatibility':False,'public_library_access':False}
