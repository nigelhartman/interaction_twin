"""Separate, source-bound SAM hand observations for private alignment diagnostics.

Preparation never instantiates a client. Dispatch is exactly five original RGB
images, needs the exact prepared-plan SHA, uses OvernightMetaClient, and has no
retry path. Hands are possible 2D occluders, never assets, geometry or proof of
depth/contact. Original masks, native renders, programs and scenes stay intact.
"""
from __future__ import annotations
import hashlib,json,math,os,re
from pathlib import Path
from decimal import Decimal
import cv2
import numpy as np
from PIL import Image,ImageDraw
import meta_sam_api as SAM
import native_scene_audit as A
from overnight_budget import PLAN as BUDGET_PLAN,OvernightMetaClient,plan as budget_plan

ROOT=Path(__file__).resolve().parent
FORMAT='interaction-twin-hand-occlusion-evidence'
PREFIX='.private/hand-occlusion/'
ATTEMPTS='.private/hand-occlusion-attempts/'
SHA=re.compile('[a-f0-9]{64}\\Z')
CODE=['hand_occlusion_evidence.py','meta_sam_api.py','native_scene_audit.py','native_appearance.py','alignment_render.py',
      'sam3d_common_scene.py','sam3d_mechanisms.py','overnight_budget.py','increment_budget.py','meta_api.py','meta_budget.py',
      'scripts/sam31_segment.py','scripts/meta_auth_probe.py']
POLICY={'concept':'hand','image_count':5,'training_frame_count':1,'development_frame_count':4,
        'maximum_instances':8,'minimum_instance_confidence':.75,'minimum_mask_pixels':32,
        'maximum_hand_frame_fraction':.60,'maximum_target_conflict_fraction':.05,
        'maximum_render_occlusion_fraction':.60,'mask_geometry':'original_full_source_raster_no_dilation_or_hole_fill',
        'possible_occluder_only':True,'hand_depth_verified':False,'automatic_retry':False,
        'automatic_geometry_updates':False,'automatic_publication':False,'final_test_evidence_used':False}


def require(value,message):
    if not value:raise ValueError(message)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def local(root,path,*,prefix='data/',exists=True,maximum=256*1024*1024):
    root=Path(root).resolve();path=Path(path);path=path if path.is_absolute()else root/path
    try:relative=path.relative_to(root).as_posix()
    except ValueError:raise ValueError('Hand evidence path escapes project')from None
    require(relative.startswith(prefix)and'..'not in path.parts and path.resolve()==path and
        not any(p.is_symlink()for p in [path,*path.parents]),'Hand evidence path is not confined')
    if exists:require(path.is_file()and 0<path.stat().st_size<=maximum,'Bounded existing hand evidence file required')
    return path


def ref(path,root):
    path=Path(path);return {'path':path.relative_to(root).as_posix(),'sha256':sha(path),'bytes':path.stat().st_size}


def checked(reference,root,*,prefix='data/',maximum=256*1024*1024):
    require(isinstance(reference,dict)and set(reference)=={'path','sha256','bytes'}and
        isinstance(reference['sha256'],str)and SHA.fullmatch(reference['sha256'])and
        type(reference['bytes'])is int and 0<reference['bytes']<=maximum,'Exact hand evidence reference required')
    path=local(root,reference['path'],prefix=prefix,maximum=maximum)
    require(ref(path,root)==reference,'Hand evidence bytes changed');return path


def write_new(path,value):
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(descriptor,'wb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())


def read(path):
    value=json.loads(Path(path).read_text());json.dumps(value,allow_nan=False);return value


def binary(path,shape):
    with Image.open(path)as image:pixels=np.asarray(image)
    require(pixels.ndim==2 and pixels.shape==tuple(shape)and set(np.unique(pixels))<={0,1,255},'Exact binary source-raster mask required')
    return pixels>0


def context_for(audit_path,training_samples,development_samples,*,root=ROOT):
    root=Path(root).resolve();audit_path=local(root,audit_path,prefix='data/native-render-audits/',maximum=8*1024*1024)
    audit=A.validate_manifest(audit_path,root=root);scene=read(checked(audit['scene_ref'],root));program=read(checked(audit['interaction_ref'],root))
    train=list(training_samples);dev=list(development_samples)
    require(len(train)==1 and len(dev)==4 and all(type(i)is int for i in train+dev)and len(set(train+dev))==5,'Exactly one training and four distinct development frames required')
    held=set(scene.get('evaluation_protocol',{}).get('held_out_sample_indices',[]));allowed={r['source_index']for r in audit['samples']if r['role']=='development'}
    require(set(train+dev)<=allowed and not set(train+dev)&held,'Final-test/heldout frames cannot enter hand evidence')
    source=program['source'];video=local(root,source['path'],maximum=1024**3);require(sha(video)==source['sha256'],'Original source video changed')
    height,width=scene['source']['height'],scene['source']['width'];evidence={r['source_index']:r for r in program['evidence']};parts={p['id']for p in scene['components']};frames=[]
    require(parts and all(A.COMPONENT_ID.fullmatch(p)for p in parts),'Safe source component identifiers required')
    for index in train+dev:
        row=evidence[index];rgb_path=local(root,row['frame_path'],maximum=SAM.MAX_MEDIA_BYTES);rgb_ref=ref(rgb_path,root);require(rgb_ref['sha256']==row['frame_sha256'],'RGB differs from original source evidence')
        with Image.open(rgb_path)as image:
            require(image.size==(width,height)and image.getexif().get(274,1)in(None,1),'Original RGB dimensions/orientation changed');image.load()
        source_row=SAM._source({'video_sha256':source['sha256'],'sample_index':index,'timestamp_s':row['timestamp_s'],'time_domain':SAM.TIME_DOMAIN})
        objects=[]
        for observation in row['observations']:
            if observation['track_id']not in parts:continue
            mask=local(root,observation['mask_path'],maximum=16*1024*1024);mask_ref=ref(mask,root);require(mask_ref['sha256']==observation['mask_sha256'],'Original object mask changed')
            binary(mask,(height,width))
            usable=observation['gate_status']=='candidate_geometric_gate_only'and not str(observation.get('audit_status','')).startswith('reject')
            objects.append({'component_id':observation['track_id'],'mask_ref':mask_ref,'source_gate_status':observation['gate_status'],
                            'source_audit_status':observation.get('audit_status'),'eligible_observation':usable})
        wire=SAM.build_payload('hand',{'type':'input_image','image_url':SAM._data_url(rgb_path,'image')})
        frames.append({'source':source_row,'role':'training'if index in train else'development','rgb_ref':rgb_ref,'width':width,'height':height,
                       'object_observations':objects,'wire_payload_sha256':digest(wire),
                       'attempt_namespace':digest({'source':source_row,'rgb':rgb_ref,'concept':'hand','purpose':'possible_visual_occluder'})})
    return {'audit_ref':ref(audit_path,root),'scene_ref':audit['scene_ref'],'interaction_ref':audit['interaction_ref'],
            'source':source,'source_video_ref':ref(video,root),'frames':frames,'final_test_evidence_used':False}


def prepare(audit_path,output,*,training_samples,development_samples,root=ROOT):
    root=Path(root).resolve();SAM.parser_preflight();budget=budget_plan(root);context=context_for(audit_path,training_samples,development_samples,root=root)
    folder=local(root,output,prefix=PREFIX,exists=False);require(not folder.exists(),'Fresh hand evidence preparation required')
    code=[]
    for name in CODE:
        path=root/name;require(path.is_file()and not path.is_symlink(),'Hand evidence implementation unavailable');code.append(ref(path,root))
    plan={'format':FORMAT+'-plan','version':1,'policy':POLICY,'context':context,'implementation':code,'parser':SAM.parser_preflight(),
          'task_budget_plan':ref(root/BUDGET_PLAN,root),'task_estimated_ceiling_usd':budget['additional_estimated_api_ceiling_usd'],
          'estimated_cost_usd':format(SAM.estimate_cost('image')*5,'.6f'),'maximum_paid_requests':5,'automatic_retry':False,
          'physical_accuracy_verified':False,'simulation_ready':False,'published':False}
    plan['request_key']=digest(plan);folder.mkdir(mode=0o700,parents=True,exist_ok=False);(folder/'implementation').mkdir(mode=0o700)
    for item in code:
        destination=folder/'implementation'/item['path'];destination.parent.mkdir(mode=0o700,parents=True,exist_ok=True);destination.write_bytes((root/item['path']).read_bytes())
        require(sha(destination)==item['sha256'],'Implementation changed while freezing hand plan')
    for frame in context['frames']:
        directory=folder/f"source-{frame['source']['sample_index']:06d}";directory.mkdir(mode=0o700)
        manifest={'schema_version':'0.1','object_id':'hand_possible_occluder','image':os.path.relpath(root/frame['rgb_ref']['path'],directory),
                  'image_sha256':frame['rgb_ref']['sha256'],'source':frame['source']}
        write_new(directory/'input.json',manifest)
    write_new(folder/'plan.json',plan)
    return {'status':'prepared_offline_no_upload','prepared':folder.relative_to(root).as_posix(),'plan_ref':ref(folder/'plan.json',root),
            'estimated_cost_usd':plan['estimated_cost_usd'],'maximum_paid_requests':5,'source_indices':[f['source']['sample_index']for f in context['frames']],
            'new_api_calls':0,'final_test_evidence_used':False}


def verified_plan(folder,*,root=ROOT):
    root=Path(root).resolve();folder=local(root,folder,prefix=PREFIX,exists=False);plan=read(local(root,folder/'plan.json',prefix=PREFIX,maximum=8*1024*1024));body=dict(plan);key=body.pop('request_key')
    require(digest(body)==key and plan['format']==FORMAT+'-plan'and plan['version']==1 and plan['policy']==POLICY,'Prepared hand plan changed')
    require(plan['automatic_retry']is False and plan['maximum_paid_requests']==5 and plan['estimated_cost_usd']==format(SAM.estimate_cost('image')*5,'.6f'),'Hand request/spending contract changed')
    require(plan['implementation']==[ref(root/name,root)for name in CODE]and plan['parser']==SAM.parser_preflight(),'Prepared hand implementation/parser changed')
    for item in plan['implementation']:
        require(sha(local(root,folder/'implementation'/item['path'],prefix=PREFIX,maximum=500000))==item['sha256'],'Frozen hand implementation changed')
    checked(plan['task_budget_plan'],root,prefix='output/',maximum=128000);budget_plan(root)
    frames=plan['context']['frames'];train=[f['source']['sample_index']for f in frames if f['role']=='training'];dev=[f['source']['sample_index']for f in frames if f['role']=='development']
    require(context_for(plan['context']['audit_ref']['path'],train,dev,root=root)==plan['context'],'Prepared hand source evidence changed')
    for frame in frames:
        directory=folder/f"source-{frame['source']['sample_index']:06d}";manifest=read(local(root,directory/'input.json',prefix=PREFIX,maximum=16000))
        expected={'schema_version':'0.1','object_id':'hand_possible_occluder','image':os.path.relpath(root/frame['rgb_ref']['path'],directory),
                  'image_sha256':frame['rgb_ref']['sha256'],'source':frame['source']}
        require(manifest==expected,'Portable hand RGB input changed')
    return folder,plan


def gate_instance(mask,confidence):
    mask=np.asarray(mask);require(mask.ndim==2 and mask.dtype==bool,'Decoded binary hand mask required');area=int(mask.sum());fraction=area/mask.size;reasons=[]
    if type(confidence)not in(int,float)or not math.isfinite(confidence)or not 0<=confidence<=1:reasons.append('confidence_missing_or_invalid')
    elif confidence<POLICY['minimum_instance_confidence']:reasons.append('confidence_below_declared_threshold')
    if area<POLICY['minimum_mask_pixels']:reasons.append('insufficient_visible_hand_pixels')
    if fraction>POLICY['maximum_hand_frame_fraction']:reasons.append('hand_mask_covers_too_much_of_source_frame')
    return {'eligible_possible_occluder':not reasons,'reasons':reasons,'foreground_pixels':area,'frame_fraction':fraction,'confidence':confidence}


def hand_unions(masks,gates,shape):
    raw=np.zeros(shape,bool);eligible=raw.copy()
    require(len(masks)==len(gates)<=POLICY['maximum_instances'],'Bounded hand instance union required')
    for mask,gate in zip(masks,gates):
        require(mask.shape==tuple(shape),'Hand union source shape changed');raw|=mask
        if gate['eligible_possible_occluder']:eligible|=mask
    fraction=float(eligible.mean());reasons=[]
    if fraction>POLICY['maximum_hand_frame_fraction']:reasons.append('combined_hand_masks_cover_too_much_of_source_frame');eligible[:]=False
    if masks and not any(g['eligible_possible_occluder']for g in gates):reasons.append('no_confident_hand_instance')
    return raw,eligible,{'eligible_possible_occluder':not reasons,'reasons':reasons,'combined_frame_fraction_before_gate':fraction,'zero_matches':not masks}


def accounted_metrics(target,render,hand,*,hand_eligible=True,source_eligible=True):
    target=np.asarray(target,bool);render=np.asarray(render,bool);hand=np.asarray(hand,bool);require(target.shape==render.shape==hand.shape,'Hand/object/render raster mismatch')
    area=int(target.sum());rendered=int(render.sum());conflicts=int(np.count_nonzero(hand&target));conflict_fraction=conflicts/area if area else None
    possible=hand&~target;occluded=int(np.count_nonzero(render&possible));occluded_fraction=occluded/rendered if rendered else None;reasons=[]
    if not hand_eligible:reasons.append('hand_evidence_not_eligible')
    if not source_eligible:reasons.append('source_object_observation_withheld')
    if area<4 or rendered<4:reasons.append('insufficient_source_or_render_pixels')
    if conflict_fraction is not None and conflict_fraction>POLICY['maximum_target_conflict_fraction']:reasons.append('hand_target_segmentation_conflict')
    if occluded_fraction is not None and occluded_fraction>POLICY['maximum_render_occlusion_fraction']:reasons.append('render_too_occluded_for_alignment_selection')
    applied=not reasons;adjusted=render&~possible if applied else render.copy()
    def ratio(a,b):return a/b if b else None
    intersection=int(np.count_nonzero(target&render));adjusted_intersection=int(np.count_nonzero(target&adjusted))
    return {'raw_mask_iou':ratio(intersection,int(np.count_nonzero(target|render))),
            'hand_accounted_mask_iou':ratio(adjusted_intersection,int(np.count_nonzero(target|adjusted))),
            'raw_source_coverage':ratio(intersection,area),'raw_render_precision':ratio(intersection,rendered),
            'hand_accounted_render_precision':ratio(adjusted_intersection,int(adjusted.sum())),
            'hand_target_conflict_pixels':conflicts,'hand_target_conflict_fraction':conflict_fraction,
            'possible_hand_render_only_pixels':occluded,'possible_hand_render_fraction':occluded_fraction,
            'discounted_render_pixels':occluded if applied else 0,'discount_applied':bool(applied and occluded),'accounting_eligible':applied,'gate_reasons':reasons,
            'hand_in_front_verified':False,'raw_native_render_preserved':True}


def _save_mask(path,mask):
    require(not path.exists(),'Immutable hand mask already exists');Image.fromarray(np.asarray(mask,np.uint8)*255).save(path)


def validate_evidence(path,*,root=ROOT):
    """Independently re-decode original paid response; hashes alone do not prove masks."""
    root=Path(root).resolve();path=local(root,path,prefix=PREFIX,maximum=8*1024*1024);document=read(path)
    require(document['format']==FORMAT and document['version']==1 and document['policy']==POLICY,'Unsupported hand evidence contract')
    plan_path=checked(document['plan_ref'],root,prefix=PREFIX,maximum=8*1024*1024);_,plan=verified_plan(plan_path.parent,root=root)
    require(document['request_key']==plan['request_key']and document['interaction_ref']==plan['context']['interaction_ref']and
            document['scene_ref']==plan['context']['scene_ref']and document['source_sha256']==plan['context']['source']['sha256'],'Hand evidence uses another source program')
    require(all(document[k]is False for k in ['asset_identity_created','native_geometry_or_poses_modified','automatic_approval','published','final_test_evidence_used','physical_accuracy_verified']),'Hand observations cannot promote geometry or physics')
    require(len(document['frames'])==5,'All five exact hand observations required')
    for frame,expected in zip(document['frames'],plan['context']['frames']):
        require(frame['source']==expected['source']and frame['role']==expected['role']and frame['rgb_ref']==expected['rgb_ref'],'Hand observation source/time changed')
        metadata=read(checked(frame['sam_metadata_ref'],root,prefix=PREFIX,maximum=8*1024*1024));response_path=checked(frame['sam_response_ref'],root,prefix=PREFIX,maximum=32*1024*1024)
        require(metadata['raw_response_sha256']==sha(response_path),'Hand metadata uses another provider response');response=read(response_path)
        accounted=checked(frame['accounted_sam_response_ref'],root,prefix=PREFIX,maximum=32*1024*1024)
        require(read(accounted)==response,'Hand decoding is not the accounted provider response')
        raw=SAM._output_text(response);require(len(raw)<=32*1024*1024 and raw.count('<|mask;')<=POLICY['maximum_instances'],'Bounded hand decoder required')
        decoded,masks=SAM.decode_response(response,width=expected['width'],height=expected['height'],media='image',sources=[expected['source']])
        require(len(frame['instances'])==len(metadata['instances'])==len(masks),'Hand instance inventory differs from provider response')
        gates=[]
        for instance,saved,original,mask in zip(frame['instances'],metadata['instances'],decoded['instances'],masks):
            require(all(saved[k]==v for k,v in original.items())and instance['instance_id']==original['instance_id']and instance['source']==expected['source'],'Hand instance differs from original SAM decoding')
            mask_path=checked(instance['mask_ref'],root,prefix=PREFIX,maximum=16*1024*1024)
            require(mask_path.name==saved['mask']and sha(mask_path)==saved['mask_sha256']and np.array_equal(binary(mask_path,mask.shape),mask),'Saved hand mask differs from original SAM response')
            gate=gate_instance(mask,original['confidence']);require(instance['gate']==gate,'Hand confidence/mask gate changed');gates.append(gate)
        raw_union,eligible_union,gate=hand_unions(masks,gates,(expected['height'],expected['width']))
        require(frame['union_gate']==gate and np.array_equal(binary(checked(frame['raw_union_ref'],root,prefix=PREFIX),raw_union.shape),raw_union)and
                np.array_equal(binary(checked(frame['eligible_union_ref'],root,prefix=PREFIX),eligible_union.shape),eligible_union),'Hand union/gating changed')
        receipt=read(checked(frame['request_receipt_ref'],root,prefix=PREFIX,maximum=256000))
        require(receipt.get('status')=='complete_accounted'and receipt.get('kind')=='sam'and receipt.get('model')==SAM.MODEL and
                receipt.get('payload_sha256')==expected['wire_payload_sha256']and receipt.get('automatic_retries')is False,'Hand observation lacks exact accounted request provenance')
    return document


def _comparison_panel(rgb,target,render,hand,metrics):
    h,w=target.shape;panes=[]
    image=rgb.copy();image[hand]=(image[hand]*.35+np.array([215,70,190])*.65).astype(np.uint8);panes.append(image)
    image=np.full_like(rgb,235);image[target]=[45,110,230];image[render]=[70,85,105];image[target&render]=[55,160,130];panes.append(image)
    image=np.full_like(rgb,235);image[target&~render]=[55,110,235];image[render&~target]=[225,95,65];image[hand&render&~target]=[225,180,55];panes.append(image)
    canvas=Image.new('RGB',(w*3,h+42),'white');draw=ImageDraw.Draw(canvas)
    for n,(pane,title)in enumerate(zip(panes,['Original RGB + possible hand','SAM blue / native gray / overlap green','Raw residual; possible hand yellow'])):
        canvas.paste(Image.fromarray(pane),(n*w,42));draw.text((n*w+8,7),title,fill=(30,40,55))
    return canvas


def build_comparison(plan,evidence,output,*,root=ROOT):
    root=Path(root).resolve();out=local(root,output,prefix=PREFIX,exists=False);require(not out.exists(),'Fresh private hand comparison required');out.mkdir(mode=0o700)
    context=plan['context'];audit=read(checked(context['audit_ref'],root));by_frame={r['source']['sample_index']:r for r in evidence['frames']};frame_context={f['source']['sample_index']:f for f in context['frames']};samples=[]
    for row in audit['samples']:
        index=row['source_index']
        if index not in by_frame:continue
        require(row['role']=='development','Final-test comparison is excluded');item=by_frame[index];frame=frame_context[index];render_path=checked(row['render_mask_ref'],root)
        with Image.open(render_path)as image:shape=(image.height,image.width)
        render=binary(render_path,shape);target=np.zeros(shape,bool);observed=next((o for o in frame['object_observations']if o['component_id']==row['component_id']),None)
        if observed:
            full=binary(checked(observed['mask_ref'],root),(frame['height'],frame['width']));target=cv2.resize(full.astype(np.uint8),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)>0
        full_hand=binary(checked(item['eligible_union_ref'],root,prefix=PREFIX),(frame['height'],frame['width']));hand=cv2.resize(full_hand.astype(np.uint8),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)>0
        source_eligible=bool(observed and observed['eligible_observation']and not row['status'].startswith('insufficient'))
        metrics=accounted_metrics(target,render,hand,hand_eligible=item['union_gate']['eligible_possible_occluder'],source_eligible=source_eligible)
        with Image.open(checked(frame['rgb_ref'],root))as image:rgb=np.asarray(image.convert('RGB'))
        rgb=cv2.resize(rgb,(shape[1],shape[0]),interpolation=cv2.INTER_AREA);panel=out/(row['sample_id'].replace(':','-')+'-hand-comparison.png');_comparison_panel(rgb,target,render,hand,metrics).save(panel)
        samples.append({'sample_id':row['sample_id'],'source_index':index,'timestamp_s':row['timestamp_s'],'component_id':row['component_id'],'role':frame['role'],
                       'source_rgb_ref':frame['rgb_ref'],'source_mask_ref':observed['mask_ref']if observed else None,'original_native_render_mask_ref':row['render_mask_ref'],
                       'original_native_comparison_panel_ref':row['comparison_panel_ref'],'hand_union_ref':item['eligible_union_ref'],'comparison_panel_ref':ref(panel,root),'metrics':metrics,
                       'source_observation_eligible':source_eligible,'status':'possible_occlusion_diagnostic_only'if metrics['discount_applied']else'no_possible_hand_overlap'if metrics['accounting_eligible']else'insufficient_for_hand_accounting'})
    development=[r for r in samples if r['role']=='development'and r['source_observation_eligible']and r['metrics']['raw_mask_iou']is not None];document={'format':FORMAT+'-comparison','version':1,
        'audit_ref':context['audit_ref'],'scene_ref':context['scene_ref'],'interaction_ref':context['interaction_ref'],'source_sha256':context['source']['sha256'],'samples':samples,
        'development_summary':{'component_samples':len(development),'discounted_samples':sum(r['metrics']['discount_applied']for r in development),
            'raw_mean_iou':float(np.mean([r['metrics']['raw_mask_iou']for r in development]))if development else None,
            'hand_accounted_mean_iou':float(np.mean([r['metrics']['hand_accounted_mask_iou']for r in development]))if development else None},
        'native_geometry_or_poses_modified':False,'automatic_approval':False,'physical_accuracy_verified':False,'final_test_evidence_used':False,
        'limitations':['2D hand segmentation does not establish front/back depth or contact.','Discounted scores are possible-occlusion diagnostics, never independent geometry validation.','No pose correction, asset generation or publication is performed.']}
    write_new(out/'comparison.json',document);return ref(out/'comparison.json',root)


def run_prepared(prepared,*,authorized_plan_sha256=None,allow_paid_api=False,root=ROOT,client=None,dispatch_guard=None):
    root=Path(root).resolve();folder,plan=verified_plan(prepared,root=root)
    require(allow_paid_api is True and authorized_plan_sha256==sha(folder/'plan.json'),'Exact prepared hand plan authorization required before upload')
    if client is not None:require(isinstance(client,OvernightMetaClient),'Only OvernightMetaClient may dispatch hand requests')
    if dispatch_guard is not None:dispatch_guard()
    # All slots are claimed before any client exists; a partial/uncertain batch
    # is retained and cannot be silently replayed via a new output directory.
    namespace=local(root,ATTEMPTS,prefix='.private/',exists=False);namespace.mkdir(mode=0o700,parents=True,exist_ok=True)
    for frame in plan['context']['frames']:
        slot=namespace/frame['attempt_namespace']
        try:slot.mkdir(mode=0o700)
        except FileExistsError:raise ValueError('Hand source frame already attempted; no automatic retry')from None
        write_new(slot/'intent.json',{'plan_ref':ref(folder/'plan.json',root),'source':frame['source'],'automatic_retry':False})
    attempt=folder/'attempt';attempt.mkdir(mode=0o700);evidence={'format':FORMAT,'version':1,'plan_ref':ref(folder/'plan.json',root),'request_key':plan['request_key'],'policy':POLICY,
        'source_sha256':plan['context']['source']['sha256'],'interaction_ref':plan['context']['interaction_ref'],'scene_ref':plan['context']['scene_ref'],'frames':[],
        'asset_identity_created':False,'native_geometry_or_poses_modified':False,'automatic_approval':False,'published':False,'final_test_evidence_used':False,'physical_accuracy_verified':False}
    try:
        if client is None:client=OvernightMetaClient(root=root,timeout=240,dispatch_guard=dispatch_guard)
        require(isinstance(client,OvernightMetaClient),'Only OvernightMetaClient may dispatch hand requests')
        for frame in plan['context']['frames']:
            if dispatch_guard is not None:dispatch_guard()
            index=frame['source']['sample_index'];directory=attempt/f'source-{index:06d}';directory.mkdir(mode=0o700);callback=client.sam_callback(directory/'accounting',images=1)
            def request(payload):
                require(digest(payload)==frame['wire_payload_sha256'],'Hand wire payload changed before dispatch');document=callback(payload)
                raw=SAM._output_text(document);require(len(raw)<=32*1024*1024 and raw.count('<|mask;')<=POLICY['maximum_instances'],'Too many hand instances for bounded decoding')
                return document
            result=SAM.segment_image(folder/f'source-{index:06d}'/'input.json','hand',directory/'sam',request,prompt_origin='automatic_original_source_frame_hand_occluder_observation')
            instances=[];masks=[];gates=[]
            for instance in result['instances']:
                mask_path=directory/'sam'/instance['mask'];mask=binary(mask_path,(frame['height'],frame['width']));gate=gate_instance(mask,instance['confidence']);masks.append(mask);gates.append(gate)
                instances.append({'instance_id':instance['instance_id'],'source':instance['source'],'mask_ref':ref(mask_path,root),'gate':gate})
            raw_union,eligible_union,union_gate=hand_unions(masks,gates,(frame['height'],frame['width']))
            raw_path=directory/'raw-hand-union.png';eligible_path=directory/'eligible-hand-union.png';_save_mask(raw_path,raw_union);_save_mask(eligible_path,eligible_union)
            receipt=read(local(root,directory/'accounting/meta-request.json',prefix=PREFIX,maximum=256000));require(receipt.get('status')=='complete_accounted'and receipt.get('kind')=='sam'and receipt.get('payload_sha256')==frame['wire_payload_sha256'],'Hand request lacks exact accounted SAM provenance')
            frame_evidence={'source':frame['source'],'role':frame['role'],'rgb_ref':frame['rgb_ref'],'instances':instances,
                'raw_union_ref':ref(raw_path,root),'eligible_union_ref':ref(eligible_path,root),'union_gate':union_gate,'sam_metadata_ref':ref(directory/'sam/metadata.json',root),
                'sam_response_ref':ref(directory/'sam/response.json',root),'accounted_sam_response_ref':ref(directory/'accounting/sam-response.pending.json',root),
                'request_receipt_ref':ref(directory/'accounting/meta-request.json',root)}
            write_new(directory/'frame-evidence.json',frame_evidence);evidence['frames'].append(frame_evidence)
        write_new(attempt/'hand-evidence.json',evidence);validate_evidence(attempt/'hand-evidence.json',root=root);comparison=build_comparison(plan,evidence,attempt/'comparison',root=root)
        report={'status':'private_hand_observation_comparison_ready','plan_ref':ref(folder/'plan.json',root),'evidence_ref':ref(attempt/'hand-evidence.json',root),'comparison_ref':comparison,
                'new_api_calls':5,'estimated_cost_usd':plan['estimated_cost_usd'],'automatic_approval':False,'published':False,'final_test_evidence_used':False}
        write_new(attempt/'report.json',report);return report
    except BaseException as error:
        write_new(attempt/'failure.json',{'status':'preserved_no_automatic_retry','error_type':type(error).__name__,'completed_frame_observations':len(evidence['frames'])})
        raise
