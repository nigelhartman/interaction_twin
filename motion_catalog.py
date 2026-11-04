"""Bounded read-only motion experiment catalog; no inference, decoding or writes.

Publication uses the full Motion Studio normalizer. Serving rechecks the frozen
raw/normalized/registered-source contract using stdlib and pure pose validators.
Saved adapter code is descriptive provenance and is never executed.
"""
from pathlib import Path
import hashlib,json,re,struct

from motion_lab import CONVENTION,STATUSES,finite,intrinsics,pose

INDEX='data/motion-catalog-20261101/index.json'
ID=re.compile(r'[a-z][a-z0-9_-]{0,119}\Z')
HASH=re.compile(r'[a-f0-9]{64}\Z')


def require(value,message):
    if not value:raise ValueError(message)


def strict_json(data):
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('Duplicate JSON key')
            result[k]=v
        return result
    return json.loads(data,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


class MotionCatalog:
    def __init__(self,root,index=INDEX):self.root=Path(root).resolve();self.index=index

    def path(self,relative,maximum=8*1024*1024):
        require(isinstance(relative,str) and relative.startswith('data/') and not any(c in relative for c in ['\\','\0','?', '#',':']),'Only declared local motion artifacts')
        target=self.root/relative
        require(not any(p.startswith('.') for p in Path(relative).parts) and target.resolve()==target and target.is_file()
            and target.is_relative_to(self.root/'data'),'Motion artifact escaped, missing or symlinked')
        require(target.stat().st_size<=maximum,'Motion artifact size budget exceeded')
        return target

    def verified(self,relative,digest,maximum=8*1024*1024):
        require(isinstance(digest,str) and HASH.fullmatch(digest),'Missing motion artifact SHA256')
        path=self.path(relative,maximum)
        require(hashlib.sha256(path.read_bytes()).hexdigest()==digest,'Motion artifact bytes changed')
        return path

    def rows(self):
        if not (self.root/self.index).exists():return []
        doc=strict_json(self.path(self.index,2*1024*1024).read_bytes())
        require(doc.get('format')=='interaction-motion-catalog' and doc.get('version')==1,'Unknown motion catalog')
        rows=doc.get('experiments');require(isinstance(rows,list) and len(rows)<=64 and all(isinstance(r,dict) for r in rows),'Bounded motion registry required')
        ids=[r.get('experiment_id') for r in rows]
        require(all(isinstance(i,str) and ID.fullmatch(i) for i in ids) and len(set(ids))==len(ids),'Duplicate/invalid experiment IDs')
        return rows

    @staticmethod
    def policy():
        return {'read_only':True,'model_inference':False,'pose_synthesis':False,'pose_interpolation':False,
            'physical_pose_accuracy_verified':False,'simulation_ready':False,'common_metric_assembly_established':False,
            'independent_episode_gauges':True,'text_is_untrusted_descriptive_data':True,
            'meaning':'Conditional image consistency, not measured physical shape, identity, pose or contact.'}

    def load(self,experiment_id,*,interaction_path,document,program_sha256):
        require(isinstance(experiment_id,str) and ID.fullmatch(experiment_id),'Invalid motion experiment ID')
        item=next((r for r in self.rows() if r['experiment_id']==experiment_id),None)
        require(item is not None,'Unknown motion experiment; discover registered IDs')
        require(item.get('interaction_path')==interaction_path and item.get('interaction_sha256')==program_sha256,'Motion experiment belongs to another program')
        self.verified(interaction_path,program_sha256,8*1024*1024)
        require(item.get('kind')=='short_interval_original_samples_v1','Unsupported motion adapter kind')
        raw_path=self.verified(item['raw']['path'],item['raw']['sha256'],4*1024*1024)
        normalized_path=self.verified(item['normalized']['path'],item['normalized']['sha256'],4*1024*1024)
        raw=strict_json(raw_path.read_bytes());view=strict_json(normalized_path.read_bytes())
        require(raw.get('format')=='short-interval-motion' and raw.get('version')==1 and raw.get('status') in {'complete_with_gaps','unsupported'},'Only completed saved interval results')
        require(view.get('format')=='interval-motion-viewer' and view.get('version')==1,'Unknown frozen motion viewer')
        component=raw.get('component_id');require(raw.get('track_id')==component and component in {t['id'] for t in document['tracklets']},'Unregistered component')
        require(raw.get('interaction_path')==interaction_path and raw.get('interaction_sha256')==program_sha256
            and raw.get('case_id')==document['case_id'],'Raw source program mismatch')
        asset=next((a for a in document['assets'] if a['track_id']==component),None)
        require(asset is not None and raw['asset']['path']==asset['path'] and raw['asset']['sha256']==asset['sha256']==raw['mesh_sha256'],'Motion does not use the declared native component asset')
        require(raw['source']['path']==document['source']['path'] and raw['source']['sha256']==raw['source_video_sha256']==document['source']['sha256'],'Motion source video mismatch')
        require(type(raw.get('new_api_calls')) is int and raw['new_api_calls']==0,'Motion serving cannot imply new model calls')
        for name in ['physical_pose_accuracy_measured','usable_physical_pose_recovered','common_assembly_established','old_pose_initialization_used','manual_geometry_or_poses']:
            require(raw.get(name) is False,'Unsupported motion execution or physical claim')
        require(raw.get('coordinate_convention')==view.get('coordinate_convention')==CONVENTION,'Unknown pose coordinate convention')
        require(view.get('case_id')==raw['case_id'] and view.get('component_id')==component and view.get('status')==raw['status']
            and view.get('interaction_path')==interaction_path and view.get('interaction_sha256')==program_sha256,'Normalized identity mismatch')
        require(view.get('input_sha256')==item['raw']['sha256'] and view.get('raw_url')=='/files/'+item['raw']['path'],'Normalized result is not bound to raw bytes')
        require(view.get('physical_pose_accuracy_measured') is False and view.get('common_assembly_established') is False,'Normalized physical uncertainty missing')
        require(view['source']['url']=='/files/'+document['source']['path'] and view['source']['sha256']==document['source']['sha256']
            and view['asset']['url']=='/files/'+asset['path'] and view['asset']['sha256']==asset['sha256'] and view['asset'].get('physical_scale_known') is False,'Normalized source/asset mismatch')
        baseline_path=self.verified(raw['baseline']['path'],raw['baseline']['sha256'],4*1024*1024)
        baseline=strict_json(baseline_path.read_bytes());old={r['sample_index']:r for r in baseline['frames']}
        require(baseline['provenance']['mesh_sha256']==asset['sha256'] and baseline['provenance']['source_video_sha256']==document['source']['sha256'],'Baseline uses different source/geometry')
        evidence={r['source_index']:r for r in document['evidence']}
        pairs=raw['comparison']['exact_pts_rows'];comparison={(r['episode_id'],r['sample_index']):r for r in pairs}
        require(len(comparison)==len(pairs),'Duplicate comparison rows')
        require(len(raw['episodes'])==len(view['episodes']) and 1<=len(raw['episodes'])<=2,'Unbounded or changed episodes')
        expected_receipts={raw_path,baseline_path,self.path(interaction_path),self.path(document['source']['path'],128*1024*1024),self.path(asset['path'],16*1024*1024)}
        for key in ['plan','audit_binding']:
            expected_receipts.add(self.verified(raw[key]['path'],raw[key]['sha256'],4*1024*1024))
        counts={'fair_same_pts_samples':0,'candidate_supported':0,'baseline_supported':0,'both_supported':0,'references_excluded':0}
        selected=[];all_frames=[]
        for episode,normalized in zip(raw['episodes'],view['episodes']):
            for key in ['id','start_s','end_s','reference_sample_index','reference_timestamp_s','gauge','status']:
                require(episode.get(key)==normalized.get(key),'Normalized episode or reference changed')
            require(episode['gauge']=='independent_reference_conditioned_candidate' and intrinsics(episode['intrinsics'])==normalized['intrinsics'],'Episode gauge or camera changed')
            train=set(episode['queries']['training_point_ids']);held=set(episode['queries']['heldout_point_ids'])
            require(not train&held and all(type(i)is int and i>=0 and i%3!=0 for i in train) and all(type(i)is int and i>=0 and i%3==0 for i in held),'Changed disjoint prefit point partitions')
            require(len(episode['frames'])==len(normalized['frames']) and 1<=len(episode['frames'])<=9,'Changed or unbounded interval frames')
            previous=-1;references=[]
            for frame,row in zip(episode['frames'],normalized['frames']):
                n=frame['sample_index'];time=frame['timestamp_s'];source=evidence.get(n)
                require(source is not None and finite(time) and time>previous and time==source['timestamp_s']==row['timestamp_s'] and row['sample_index']==n,'Motion sample is not exact registered source PTS')
                previous=time;observations=[o for o in source['observations'] if o['track_id']==component]
                require(len(observations)==1,'Missing exact component mask');observation=observations[0]
                for key,a,b in [('image',source['frame_path'],source['frame_sha256']),('mask',observation['mask_path'],observation['mask_sha256'])]:
                    require(row[key+'_url']=='/files/'+a and row[key+'_sha256']==b and frame[('source_image' if key=='image' else 'mask')+'_sha256']==b,'Normalized exact source/mask changed')
                    expected_receipts.add(self.verified(a,b))
                status=frame['status'];reference=frame.get('reference') is True
                require(status in STATUSES and row['status']==status and row['reference']==reference,'Normalized support/reference flag changed')
                require(row['pose']==(pose(frame) if status!='withheld' else None),'Normalized pose was synthesized, replaced or retained across loss')
                if status=='withheld':require(frame.get('candidate_rotation') is None and frame.get('candidate_translation') is None,'Withheld pose must stay unavailable')
                if reference:references.append(frame)
                require(status!='supported_nonreference_candidate' or not reference,'Reference cannot count as support')
                require(row.get('temporal_split')==frame.get('temporal_split') and row.get('evaluation_timestamp_reserved_before_reference_selection')==frame.get('evaluation_timestamp_reserved_before_reference_selection'),'Temporal evaluation identity changed')
                require(type(frame.get('evaluation_timestamp_reserved_before_reference_selection')) is bool and
                    frame['evaluation_timestamp_reserved_before_reference_selection']==(abs(time-round(time))<=.02)
                    and frame['temporal_split']==('evaluation' if frame['evaluation_timestamp_reserved_before_reference_selection'] else 'reference_selection')
                    and not(reference and frame['evaluation_timestamp_reserved_before_reference_selection']),'Reserved source-clock evaluation rule changed')
                require(set(frame.get('training_visible_point_ids',[]))<=train and set(frame.get('heldout_visible_point_ids',[]))<=held,'Visible point IDs left frozen partitions')
                for key in ['train_count','inlier_count','heldout_median_px','heldout_p90_px','silhouette_iou','mask_coverage']:
                    if finite(frame.get(key)):require(row.get(key)==frame[key],'Normalized numeric score changed')
                pair=comparison.get((episode['id'],n));prior=old.get(n)
                require(pair is not None and prior is not None and pair['timestamp_s']==time==prior['timestamp_s'],'Comparison must use identical source sample')
                require(pair['candidate_supported']==(status=='supported_nonreference_candidate') and pair['candidate_reference']==reference,'Comparison candidate flags changed')
                baseline_status=pair['baseline_reassessed_status'];base_ref=prior['status']=='reference_candidate_unverified_geometry_and_pose'
                require(baseline_status in STATUSES and pair['baseline_reference']==base_ref and pair['baseline_supported']==(baseline_status=='supported_nonreference_candidate'),'Comparison baseline flags changed')
                require(row['baseline']['status']==baseline_status and row['baseline']['pose']==(pose(prior) if baseline_status!='withheld' else None),'Normalized baseline pose/status changed')
                fair=not(reference or base_ref) and frame['temporal_split']=='evaluation'
                require(row.get('fair_comparison')==fair,'Normalized fair comparison denominator changed')
                if fair:
                    counts['fair_same_pts_samples']+=1;counts['candidate_supported']+=pair['candidate_supported'];counts['baseline_supported']+=pair['baseline_supported'];counts['both_supported']+=pair['candidate_supported'] and pair['baseline_supported']
                elif reference or base_ref:counts['references_excluded']+=1
                all_frames.append((episode['id'],frame,source,observation))
                if status=='supported_nonreference_candidate':selected.append((episode['id'],frame,source,observation))
            require(len(references)==1 and references[0]['sample_index']==episode['reference_sample_index'],'Frozen reference identity changed')
        require(view['comparison_counts']==counts,'Normalized comparison counts changed')
        receipts=view.get('artifact_receipts');require(isinstance(receipts,list) and len(receipts)<=80,'Bounded normalized artifact receipts required')
        actual_receipts=set()
        for binding in receipts:actual_receipts.add(self.verified(binding['path'],binding['sha256'],128*1024*1024))
        require(len(actual_receipts)==len(receipts) and actual_receipts==expected_receipts,'Normalized source receipts are incomplete, duplicated or contain unrelated files')
        if view.get('capture_feedback'):
            feedback=view['capture_feedback'];p=self.verified(feedback['url'].removeprefix('/files/'),feedback['sha256'],4*1024*1024)
            require(strict_json(p.read_bytes())['interaction_sha256']==program_sha256,'Capture advice program mismatch')
        chosen=(selected or all_frames)[0];episode_id,frame,source,observation=chosen
        require(Path(source['frame_path']).suffix.lower() in {'.jpg','.jpeg'} and Path(observation['mask_path']).suffix.lower()=='.png','Unsupported motion source media type')
        require(self.path(source['frame_path']).read_bytes()[:2]==b'\xff\xd8' and self.path(observation['mask_path']).read_bytes()[:8]==b'\x89PNG\r\n\x1a\n','Source media header does not match declared type')
        artifacts={'raw':dict(item['raw'],mime='application/json'),'normalized':dict(item['normalized'],mime='application/json'),
            'source-image-0':{'path':source['frame_path'],'sha256':source['frame_sha256'],'mime':'image/jpeg'},
            'source-mask-0':{'path':observation['mask_path'],'sha256':observation['mask_sha256'],'mime':'image/png'}}
        return item,raw,view,artifacts,{'episode_id':episode_id,'sample_index':frame['sample_index'],'timestamp_s':frame['timestamp_s'],'program_evidence_id':source['id'],'status':frame['status']}

    def describe(self,experiment_id,*,twin_id,**context):
        item,raw,view,artifacts,source=self.load(experiment_id,**context)
        for key,binding in artifacts.items():binding.update(uri=f'twin://{twin_id}/motion/{experiment_id}/artifacts/{key}',resource_mime_type='application/octet-stream')
        return {'format':'registered-motion-experiment','version':1,'twin_id':twin_id,'experiment_id':experiment_id,
            'program_sha256':context['program_sha256'],'source_video_sha256':raw['source_video_sha256'],'component_id':raw['component_id'],
            'native_mesh_sha256':raw['mesh_sha256'],'kind':item['kind'],'status':raw['status'],'motion':view,
            'matched_comparison':{k:v for k,v in raw['comparison'].items() if k!='exact_pts_rows'},'inspection_source':source,
            'artifacts':artifacts,'policy':self.policy()}

    def list(self,*,twin_id,**context):
        results=[]
        for row in self.rows():
            if row['interaction_path']!=context['interaction_path']:continue
            descriptor=self.describe(row['experiment_id'],twin_id=twin_id,**context)
            results.append({key:descriptor[key] for key in ['experiment_id','component_id','kind','status','matched_comparison','inspection_source']})
        return {'twin_id':twin_id,'program_sha256':context['program_sha256'],'experiments':results,'policy':self.policy()}

    def artifact(self,experiment_id,artifact_id,**context):
        require(artifact_id in {'raw','normalized','source-image-0','source-mask-0'},'Unknown bounded motion artifact ID')
        _,_,_,artifacts,_=self.load(experiment_id,**context);binding=artifacts[artifact_id]
        return self.verified(binding['path'],binding['sha256'],4*1024*1024 if artifact_id in {'raw','normalized'} else 8*1024*1024).read_bytes()
