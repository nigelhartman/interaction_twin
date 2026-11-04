"""Declared project marketing evidence. No directory scans or arbitrary media URLs."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from agent_memory import digest,private_directory

ROOT=Path(__file__).resolve().parent
JABRA='flow-sam3d-default-e2e-20261102-v1--flow-procedure-1'
WALLET='flow-rigid-parts-wallet-20261031-v1--flow-procedure-1'
CAMERA='flow-meta-video-camera-20261030-v1--flow-procedure-1'
CHARGER='flow-capture-charger-20261103-v1--flow-procedure-1'
LAPTOP='flow-capture-laptop-20261103-v1--flow-procedure-1'
EXPLAINER_CHAPTERS=[
 {'id':'brand','label':'Brand opening','start_s':0.,'end_s':3.,'fact_ids':[],'headline_ids':[],'evidence_kind':'brand'},
 {'id':'capture','label':'Original Jabra capture','start_s':3.,'end_s':8.,'fact_ids':['source'],'headline_ids':['source'],'evidence_kind':'source_video'},
 {'id':'masks','label':'Actual SAM 3.1 component masks','start_s':8.,'end_s':14.,'fact_ids':['masks'],'headline_ids':['masks'],'evidence_kind':'segmentation'},
 {'id':'parts','label':'Separate native SAM 3D meshes','start_s':14.,'end_s':21.,'fact_ids':['parts'],'headline_ids':['parts'],'evidence_kind':'generated_meshes'},
 {'id':'reasoning','label':'Muse component and relationship reasoning','start_s':21.,'end_s':27.,'fact_ids':['program'],'headline_ids':['program'],'evidence_kind':'reasoning'},
 {'id':'connected','label':'Estimated connected joint replay','start_s':27.,'end_s':35.,'fact_ids':['connections'],'headline_ids':['connections'],'evidence_kind':'connected_scene'},
 {'id':'program','label':'Descriptive program and joint fields','start_s':35.,'end_s':41.,'fact_ids':['program'],'headline_ids':['program'],'evidence_kind':'object_program'},
 {'id':'library','label':'Curated twin library','start_s':41.,'end_s':46.,'fact_ids':['library'],'headline_ids':['library'],'evidence_kind':'library'},
 {'id':'robot','label':'Authored robot workbench task','start_s':46.,'end_s':52.,'fact_ids':['robotics_limit'],'headline_ids':['robotics'],'evidence_kind':'robot_preview'},
 {'id':'cta','label':'Closing call to action','start_s':52.,'end_s':55.,'fact_ids':[],'headline_ids':[],'evidence_kind':'cta'},
]
EXPLAINER_45S_CHAPTERS=[
 {'id':'brand','label':'Brand opening','start_s':0.,'end_s':59/24,'fact_ids':[],'headline_ids':[],'evidence_kind':'brand'},
 {'id':'capture','label':'Original Jabra capture','start_s':59/24,'end_s':157/24,'fact_ids':['source'],'headline_ids':['source'],'evidence_kind':'source_video'},
 {'id':'masks','label':'Actual SAM 3.1 component masks','start_s':157/24,'end_s':275/24,'fact_ids':['masks'],'headline_ids':['masks'],'evidence_kind':'segmentation'},
 {'id':'parts','label':'Separate native SAM 3D meshes','start_s':275/24,'end_s':412/24,'fact_ids':['parts'],'headline_ids':['parts'],'evidence_kind':'generated_meshes'},
 {'id':'reasoning','label':'Muse component and relationship reasoning','start_s':412/24,'end_s':530/24,'fact_ids':['program'],'headline_ids':['program'],'evidence_kind':'reasoning'},
 {'id':'connected','label':'Estimated connected joint replay','start_s':530/24,'end_s':687/24,'fact_ids':['connections'],'headline_ids':['connections'],'evidence_kind':'connected_scene'},
 {'id':'program','label':'Descriptive program and joint fields','start_s':687/24,'end_s':805/24,'fact_ids':['program'],'headline_ids':['program'],'evidence_kind':'object_program'},
 {'id':'library','label':'Curated twin library','start_s':805/24,'end_s':903/24,'fact_ids':['library'],'headline_ids':['library'],'evidence_kind':'library'},
 {'id':'robot','label':'Authored robot workbench task','start_s':903/24,'end_s':1021/24,'fact_ids':['robotics_limit'],'headline_ids':['robotics'],'evidence_kind':'robot_preview'},
 {'id':'cta','label':'Closing call to action','start_s':1021/24,'end_s':45.,'fact_ids':[],'headline_ids':[],'evidence_kind':'cta'},
]
EXPLAINER_REFINED45_CHAPTERS=[
 {'id':'brand','label':'Brand opening','start_s':0.,'end_s':48/24,'fact_ids':[],'headline_ids':[],'evidence_kind':'brand'},
 {'id':'capture','label':'Original Jabra capture','start_s':48/24,'end_s':120/24,'fact_ids':['source'],'headline_ids':['source'],'evidence_kind':'source_video'},
 {'id':'masks','label':'Actual SAM 3.1 component masks','start_s':120/24,'end_s':216/24,'fact_ids':['masks'],'headline_ids':['masks'],'evidence_kind':'segmentation'},
 {'id':'parts','label':'Three native meshes plus one documented geometric lid derivative','start_s':216/24,'end_s':336/24,'fact_ids':['parts'],'headline_ids':['parts'],'evidence_kind':'generated_meshes'},
 {'id':'reasoning','label':'Muse relationships and estimated geometric joints','start_s':336/24,'end_s':432/24,'fact_ids':['program'],'headline_ids':['program'],'evidence_kind':'reasoning'},
 {'id':'connected','label':'Refined estimated connected joint replay','start_s':432/24,'end_s':576/24,'fact_ids':['connections'],'headline_ids':['connections'],'evidence_kind':'connected_scene'},
 {'id':'program','label':'Refined descriptive companion and source evidence','start_s':576/24,'end_s':672/24,'fact_ids':['program'],'headline_ids':['program'],'evidence_kind':'object_program'},
 {'id':'library','label':'Curated twin library and read-only agent tools','start_s':672/24,'end_s':768/24,'fact_ids':['library'],'headline_ids':['library'],'evidence_kind':'library'},
 {'id':'robot','label':'Complete authored robot rehearsal and released goal','start_s':768/24,'end_s':1008/24,'fact_ids':['robotics_limit'],'headline_ids':['robotics'],'evidence_kind':'robot_preview'},
 {'id':'cta','label':'Closing call to action','start_s':1008/24,'end_s':45.,'fact_ids':[],'headline_ids':[],'evidence_kind':'cta'},
]
DYNAMIC_V2_CHAPTERS=[
 {'id':'capture','label':'Real demonstration and source-bound twin','start_s':0.,'end_s':4.,'fact_ids':['source'],'headline_ids':['source'],'evidence_kind':'source_video'},
 {'id':'masks','label':'Actual SAM 3.1 component masks','start_s':4.,'end_s':8.,'fact_ids':['masks'],'headline_ids':['masks'],'evidence_kind':'segmentation'},
 {'id':'shape_motion','label':'Generated parts and estimated connected motion','start_s':8.,'end_s':16.,'fact_ids':['connections','parts','program'],'headline_ids':['connections','parts','program'],'evidence_kind':'connected_scene'},
 {'id':'program','label':'Readable descriptive object program','start_s':16.,'end_s':20.,'fact_ids':['program'],'headline_ids':['program'],'evidence_kind':'object_program'},
 {'id':'library','label':'Curated twin library','start_s':20.,'end_s':25.,'fact_ids':['library'],'headline_ids':['library'],'evidence_kind':'library'},
 {'id':'agents','label':'Read-only agent library query','start_s':25.,'end_s':29.,'fact_ids':['agents'],'headline_ids':['agents'],'evidence_kind':'library'},
 {'id':'robot','label':'Complete authored robot rehearsal and both released goals','start_s':29.,'end_s':39.,'fact_ids':['robotics_limit'],'headline_ids':['robotics'],'evidence_kind':'robot_preview'},
 {'id':'physics','label':'Actual NVIDIA Newton CPU gravity and contact','start_s':39.,'end_s':43.,'fact_ids':['physics'],'headline_ids':['physics'],'evidence_kind':'physics'},
 {'id':'brand','label':'Closing brand','start_s':43.,'end_s':45.,'fact_ids':[],'headline_ids':[],'evidence_kind':'cta'},
]
DECLARED={
 'pipeline_demo':{'label':'Pipeline · actual SAM masks, parts and object program','kind':'pipeline_film','path':'showcase/media/interaction-twin-cobalt-20261103-v2-web.mp4','sha256':'ef8eabc7a7496e17102c87c500227b6e9f6c3513a4558461cbb69da94506dfe4','duration_s':55.,'pack_ids':[JABRA],'rights':'Approved original project film; original procedural soundtrack is replaced in new renders.','limitation':'Generated geometry and connected motion are estimates.'},
 'robot_workbench':{'label':'Robot workbench · real browser capture','kind':'robot_preview','path':'showcase/media/robot-workbench-jabra-20261103-v1.mp4','sha256':'6b1d1b1b100c2247ff5a80b933340c34a8214ef697e3a1984bd1639beea41417','duration_s':23.523,'pack_ids':[JABRA],'rights':'Approved original browser recording.','limitation':'Authored robot staging and position cues; no physical grasp, contact or robot learning.'},
 'jabra_capture':{'label':'Earbuds and charging case · own capture','kind':'source_video','path':'data/flow-sam3d-default-e2e-20261102-v1/source/video.mp4','sha256':'7474ebd9d06c55b63fc073f632e71e5125353ace2538fa7bcf632cfdc1c3d0f2','duration_s':33.636278,'pack_ids':[JABRA],'rights':'Owner-supplied recording, approved project media.','limitation':'This is the observed demonstration, not a simulation.'},
 'wallet_capture':{'label':'Wallet, flap and coin · own capture','kind':'source_video','path':'data/flow-rigid-parts-wallet-20261031-v1/source/video.mp4','sha256':'b153e7ce79f140c861764c832ba3def149f5596ab4fe23bd3d671af913d5c021','duration_s':33.1029,'pack_ids':[WALLET],'rights':'Owner-supplied recording, approved project media.','limitation':'The wallet flap is deformable; physical articulation is not recovered.'},
 'pipeline_explainer':{'label':'Pipeline explainer · Cobalt Motion · 55 seconds','kind':'pipeline_film','path':'showcase/media/interaction-twin-explainer-20261103-v4.mp4','sha256':'cc554d2e559d4b95678f2bce343210d1ea3aa816c022443301459ff78a636995','duration_s':55.,'pack_ids':[JABRA],
    'rights':'Original project film, owner-authorized for project marketing; original procedural soundtrack is replaced in new cuts.',
    'limitation':'Four native Jabra meshes and estimated joint replay; mating surfaces and physical accuracy are unverified. Robot motion is authored.',
    'chapters':EXPLAINER_CHAPTERS,'chapters_document':{'path':'output/brand-trailer-20261103/renders/v4/storyboard.json','sha256':'4aae7b1afdde2a91fc7531d2ac67e311df9ddc613bc46662799d9e541224b0de','bytes':2530}},
 'pipeline_explainer_45s':{'label':'Pipeline explainer · Cobalt Motion · 45 seconds','kind':'pipeline_film','path':'showcase/media/interaction-twin-explainer-20261103-v5-45s.mp4','sha256':'1f803994111e144ad2402f4a90ebe01907ddb250b512203f294e04a71c56aac7','duration_s':45.,'pack_ids':[JABRA],
    'rights':'Original project film, owner-authorized for project marketing; original procedural soundtrack is replaced in new cuts.',
    'limitation':'Four native Jabra meshes and estimated joint replay; mating surfaces and physical accuracy are unverified. Robot motion is authored.',
    'chapters':EXPLAINER_45S_CHAPTERS,'chapters_document_version':5,
    'chapters_document':{'path':'output/brand-trailer-20261103/renders/v5-final/storyboard.json','sha256':'d1d373a4af75d57ed11aec771f5c68f63210b5f029204fdb945c506e77b5777c','bytes':3742}},
 'pipeline_explainer_refined45':{'label':'Pipeline explainer · refined assembly · 45 seconds','kind':'pipeline_film','path':'showcase/media/interaction-twin-explainer-20261103-v6-refined45.mp4','sha256':'3545dcb03d2c5101fa3c986e32122617fda205635c9b026aba2ae7bab3526627','duration_s':45.,'pack_ids':[JABRA],
    'rights':'Original project film, owner-authorized for project marketing; original procedural soundtrack is replaced in new cuts.',
    'limitation':'Three original native Jabra meshes plus one documented geometrically refined lid. Refined connected FK and source angles are estimates; physical geometry, scale and contact remain unverified. The complete robot rehearsal is authored.',
    'chapters':EXPLAINER_REFINED45_CHAPTERS,'chapters_document_version':6,
    'chapters_document':{'path':'output/brand-trailer-20261103/renders/v6/storyboard.json','sha256':'77c6c48a1fa1f3e256c58bcc8203e57558ecdac8ea01c7ce00e9ee0aca6e8177','bytes':9965}},
 'pipeline_dynamic_45s_v2':{'label':'Project film · Cobalt Motion · 45 seconds','kind':'pipeline_film','path':'showcase/media/interaction-twin-dynamic-20261104-v2.mp4','sha256':'13d48ac20b5a617b53a74efaf49ee1e5372a775028b539cf6ee73b619ea65dbf','duration_s':45.,'pack_ids':[JABRA,WALLET,CHARGER,LAPTOP],
    'rights':'Owner-authorized original project footage and browser captures, original procedural music and local synthetic narration. No external publication performed.',
    'limitation':'Source-bound SAM 3D components and declared geometric refinements use estimated connections. The complete robot task is authored kinematics, not a trained policy or validated grasp. The separate Newton CPU experiment uses assumed scale/materials and convex proxies; physical accuracy remains unverified.',
    'chapters':DYNAMIC_V2_CHAPTERS,'chapters_document_version':7,
    'chapters_document':{'path':'output/presentation-polish-20261104-v2/dynamic-v2-storyboard.json','sha256':'ca2583660a8ea36eeb71da11093d4b47d86bcbcff6070331138cb5e0b4e70d20','bytes':4344}},
 'pipeline_dynamic_45s_v3':{'label':'Project film · Cobalt Motion V3 · 45 seconds','kind':'pipeline_film','path':'showcase/media/interaction-twin-dynamic-20261104-v3.mp4','sha256':'6b590c0aa40550208b475f405979696dfb3e865486ad184ec1c8fb401826b404','duration_s':45.,'pack_ids':[JABRA,WALLET,CHARGER,LAPTOP],
    'rights':'Owner-authorized original project footage and browser captures, original procedural music and local synthetic narration. No external publication performed.',
    'limitation':'Source-bound SAM 3D components and declared geometric refinements use estimated connections. The complete robot task is authored kinematics, not a trained policy or validated grasp. The separate Newton CPU experiment uses assumed scale/materials and convex proxies; physical accuracy remains unverified.',
    'chapters':DYNAMIC_V2_CHAPTERS,'chapters_document_version':8,
    'chapters_document':{'path':'output/presentation-polish-20261104-v3/dynamic-v3-storyboard.json','sha256':'c58e1e5b16a9129958189c951cdaf15d3b9ce3ff3059a98c6f5a6cc81bd856a9','bytes':5197}},
}
FACTS={
 'source':'Inspect the original demonstration and its evidence.',
 'masks':'SAM 3.1 separates and tracks moving components.',
 'parts':'SAM 3D Objects generates separate native meshes.',
 'program':'Muse proposes source-linked steps and component relationships.',
 'connections':'An object program keeps parts, estimated connections and evidence together.',
 'library':'Browse a curated library of descriptive digital twins.',
 'agents':'Agents can search the library through read-only tools and local MCP.',
 'robotics':'Author editable robot tasks in a joint-limited browser workbench.',
 'robotics_limit':'Robot motion is authored; physical grasp and contact are unverified.',
 'physics':'A saved NVIDIA Newton CPU experiment demonstrates gravity and contact with assumed physical parameters.',
 'geometry_limit':'Generated geometry, physical scale and simulation remain unverified.',
}
HEADLINES={
 'source':'Show the real interaction.', 'masks':'Find the moving parts.',
 'parts':'Separate parts. More useful models.', 'program':'Keep how it works.',
 'connections':'Keep the connection.', 'library':'A library, built from demonstrations.',
 'agents':'For people. For their agents.', 'robotics':'Start a robot task.',
 'physics':'Test gravity and contact.',
}
LIMITATION='Geometry, motion and scale are estimates; robot motion is authored, with grasp and contact unverified.'
CTA='Explore the object library.'
AUTH_FOLDER='.private/marketing-source-authorizations'
PACK_SOURCE_ID=re.compile(r'pack_[a-f0-9]{64}\Z')
MAX_VIDEO=512*1024*1024
BRAND_SELECTOR='config/marketing-brand-default.json'
BRAND_REGISTRY={
 'cobalt-motion-v1':{'path':'config/marketing-brand.json','sha256':'38054bcf257b8ac604052a12d2239b811fba1ffb61af954e6b45960740c96b50',
    'version':1,'logo_path':'showcase/mark.svg','logo_sha256':'3c238f78cb357bda3f7df4d6e3fb1fb46d5fbff6e5cfa1b9ac7e2e3e6c3dc992'},
 'cobalt-motion-v2':{'path':'config/marketing-brand-v2.json','sha256':'8841754da81a29293eda7a7b3019dc9433b0a3b549ee19214d6443e32737b649',
    'version':2,'logo_path':'showcase/brand/v2/mark.svg','logo_sha256':'b0d9aa3ef641d8222266a3236c30f6f316ec2932ad06926447273a8afbfc5c48',
    'logo_raster_path':'showcase/brand/v2/mark-1024.png','logo_raster_sha256':'881e92e291fe72ee49d58a25f70caf65ab1912e39262647bd6a999ceff8afb68'},
}

def rendered_text(storyboard,sources,preferences=None):
    """Exact compositor-owned copy shown to critics, renderers and reviewers."""
    preferences=preferences or {};rows=[]
    for entry in storyboard:
        sentence=HEADLINES[entry['headline_id']]if preferences.get('caption_density')=='minimal'else FACTS[entry['fact_id']]
        rows.append({'headline':HEADLINES[entry['headline_id']],'caption':sentence,'limitation':LIMITATION,
            'source_id':entry['source_id'],'source_limitation':sources[entry['source_id']]['limitation']})
    return {'protocol':2,'intro':{'product_name':'Interaction Twin','slogan':'Show it once. Keep how it works.','duration_s':2.},
        'scenes':rows,'outro':{'product_name':'Interaction Twin','slogan':'Show it once. Keep how it works.','cta':CTA,'duration_s':2.},
        'mandatory_limitation':LIMITATION,'footer':'SOURCE-LINKED EVIDENCE · ESTIMATES, NOT PHYSICS'}

def file_ref(path,root=ROOT):
    root=Path(root).resolve();path=Path(path)
    if not path.is_absolute():path=root/path
    if (not path.is_relative_to(root) or path.resolve()!=path or
        any(p.is_symlink() for p in [path,*path.parents]) or not path.is_file()):
        raise ValueError('Ordinary project media required')
    h=hashlib.sha256()
    with path.open('rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return {'path':path.relative_to(root).as_posix(),'sha256':h.hexdigest(),'bytes':path.stat().st_size}

def checked(ref,root=ROOT):
    path=Path(root).resolve()/ref['path']
    if file_ref(path,root)!=ref:raise ValueError('Marketing evidence bytes changed')
    return path

class MarketingSources:
    def __init__(self,root=ROOT):self.root=Path(root).resolve()
    def _pack_binding(self,case_id,catalog=None):
        """Resolve only the exact public program's pinned public source video."""
        from curated_twin_catalog import CuratedTwinCatalog
        if not isinstance(case_id,str)or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,199}',case_id):raise ValueError('Choose an exact curated library pack ID')
        catalog=catalog or CuratedTwinCatalog(self.root)
        record,_=catalog._record(case_id)
        program_ref={k:record['interaction'][k]for k in ['path','sha256','bytes']}
        program=json.loads(catalog._bytes(program_ref,maximum=8*1024*1024))
        source=record.get('source',{});observed=program.get('source',{})
        if any(source.get(k)!=observed.get(k)for k in ['path','sha256']):raise ValueError('Curated pack and program source differ')
        declared=catalog._assets().get(source.get('path'))
        if declared is None or declared.get('sha256')!=source.get('sha256'):raise ValueError('No exact public source video is available for this pack')
        video={k:declared[k]for k in ['path','sha256','bytes']}
        path=catalog._checked(video,maximum=MAX_VIDEO)
        if path.suffix.lower()not in {'.mp4','.mov','.m4v','.webm'}:raise ValueError('The pinned source is not a supported video container')
        duration=source.get('duration_s')
        if type(duration)not in {int,float}or not math.isfinite(duration)or not 0<duration<=3600:raise ValueError('A bounded source video duration is required')
        if observed.get('duration_s')!=duration:raise ValueError('Curated pack source duration differs from its program')
        title=record.get('title')
        if not isinstance(title,str)or not 0<len(title)<=500:raise ValueError('A bounded curated pack label is required')
        return {'format':'interaction-twin-marketing-source-authorization','version':1,'case_id':case_id,
            'curated_record_sha256':digest(record),'program_ref':program_ref,'source_video_ref':video,
            'duration_s':duration,'label':title,'rights_acknowledged':True,
            'rights_origin':'authenticated_owner_acknowledgement_not_legal_verification','publication_performed':False}

    def _authorization(self,identifier,catalog=None):
        if not isinstance(identifier,str)or not PACK_SOURCE_ID.fullmatch(identifier):raise ValueError('Choose a declared or authorized marketing source ID')
        path=self.root/AUTH_FOLDER/(identifier+'.json')
        if path.resolve()!=path or any(p.is_symlink()for p in [path,*path.parents])or not path.is_file()or not 0<path.stat().st_size<=16000:raise ValueError('Saved pack media authorization is unavailable')
        record=json.loads(path.read_text())
        if not isinstance(record,dict)or identifier!='pack_'+digest(record):raise ValueError('Immutable pack media authorization changed')
        current=self._pack_binding(record.get('case_id'),catalog)
        if record!=current:raise ValueError('Pack media authorization belongs to another curated revision')
        return record

    def authorize_pack(self,case_id,*,rights_acknowledged=True):
        """Explicit authenticated owner action; GET never calls this method.

        This acknowledgement grants no external platform, copyright or physical
        correctness claim. A new curated revision requires a fresh exact record.
        """
        if rights_acknowledged is not True:raise ValueError('Acknowledge rights to use this exact source video for marketing')
        value=self._pack_binding(case_id);identifier='pack_'+digest(value)
        folder=private_directory(self.root,'marketing-source-authorizations');path=folder/(identifier+'.json')
        raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
        fd,name=tempfile.mkstemp(prefix='.authorization-',suffix='.tmp',dir=folder)
        try:
            with os.fdopen(fd,'wb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
            try:os.link(name,path)
            except FileExistsError:
                if path.is_symlink()or path.stat().st_size!=len(raw)or path.read_bytes()!=raw:raise ValueError('Immutable pack media authorization changed')
            path.chmod(0o600)
        finally:
            if os.path.exists(name):os.unlink(name)
        return self.get(identifier)

    @staticmethod
    def _authorized_source(identifier,record):
        return {'id':identifier,'label':record['label']+' · authorized pack footage','kind':'source_video',
            'duration_s':record['duration_s'],'pack_ids':[record['case_id']],
            'rights':'Owner acknowledged rights to use this exact curated source video for marketing; not independently verified.',
            'limitation':'Observed source demonstration; generated geometry, motion and physical accuracy remain unverified.',
            'file':record['source_video_ref'],'preview_url':'/api/admin/marketing/sources/'+identifier+'/video',
            'authorization':{'case_id':record['case_id'],'curated_record_sha256':record['curated_record_sha256'],
                'program_ref':record['program_ref'],'rights_acknowledged':True,'rights_origin':record['rights_origin'],
                'automatic_authorization':False,'claim_free_guaranteed':False}}

    def get(self,identifier):
        if identifier not in DECLARED:return self._authorized_source(identifier,self._authorization(identifier))
        item=copy.deepcopy(DECLARED[identifier]);ref=file_ref(item['path'],self.root)
        if ref['sha256']!=item.pop('sha256'):raise ValueError('Declared marketing source changed')
        if 'chapters_document'in item:
            reference=item['chapters_document']
            if not 0<reference['bytes']<=64000:raise ValueError('Bounded frozen chapter document required')
            raw=checked(reference,self.root).read_bytes()
            if len(raw)!=reference['bytes']or hashlib.sha256(raw).hexdigest()!=reference['sha256']:raise ValueError('Frozen chapter document changed during read')
            document=json.loads(raw);chapters=document.get('chapters')
            if document.get('format')!='interaction-twin-brand-explainer-storyboard'or document.get('version')!=item.get('chapters_document_version',2)or not isinstance(chapters,list)or len(chapters)!=len(item['chapters']):raise ValueError('Frozen chapter document differs from the registered film')
            for chapter,observed in zip(item['chapters'],chapters):
                if any(chapter[k]!=observed.get(k)for k in ['start_s','end_s']):raise ValueError('Registered chapter timing differs from the exact film storyboard')
                for field,maximum in [('title',200),('scope',1000)]:
                    text=observed.get(field)
                    if not isinstance(text,str)or not 0<len(text)<=maximum:raise ValueError('Bounded frozen chapter copy required')
                    chapter['editorial_'+field]=text
            item['chapters_sha256']=digest(item['chapters'])
        item.pop('path');return {'id':identifier,**item,'file':ref,'preview_url':'/api/admin/marketing/sources/'+identifier+'/video'}
    def list(self):
        from curated_twin_catalog import CuratedTwinCatalog
        catalog=CuratedTwinCatalog(self.root);packs=[];authorized=[]
        folder=self.root/AUTH_FOLDER
        if folder.exists():
            if folder.resolve()!=folder or any(p.is_symlink()for p in [folder,*folder.parents]):raise ValueError('Ordinary private authorization directory required')
            paths=sorted(folder.glob('pack_*.json'))
            if len(paths)>1000:raise ValueError('Bounded pack media authorization inspection required')
            for path in paths:
                try:authorized.append(self._authorized_source(path.stem,self._authorization(path.stem,catalog)))
                except (ValueError,OSError,KeyError,TypeError):continue
        records,_=catalog._snapshot()
        for record in sorted(records,key=lambda row:(row.get('title','').casefold(),row['id'])):
            catalog._checked(record['interaction']);source_ids=[k for k,v in DECLARED.items()if record['id']in v['pack_ids']]
            source_ids.extend(item['id']for item in authorized if record['id']in item['pack_ids'])
            has_authorization=record['id']in {JABRA,WALLET}or bool([item for item in authorized if record['id']in item['pack_ids']])
            source_ref=None;reason=None
            try:source_ref=self._pack_binding(record['id'],catalog)['source_video_ref']
            except (ValueError,OSError,KeyError,TypeError):reason='The exact public source video is unavailable or no longer matches the curated program.'
            packs.append({'id':record['id'],'label':record['title'],'summary':record.get('summary',''),
                'counts':record['counts'],'program_ref':record['interaction'],'source_ids':source_ids,
                'marketing_media_authorized':has_authorization,'public_source_video_ref':source_ref,
                'can_authorize':not has_authorization and source_ref is not None,
                'source_authorization_status':'authorized'if has_authorization else'needs_rights_acknowledgement'if source_ref else'source_unavailable',
                'source_authorization_reason':reason,
                'note':'Owner-authorized project media.'if has_authorization else'Explicit rights acknowledgement is required before using this source video for marketing.'})
        rows=[]
        for identifier in DECLARED:
            try:rows.append(self.get(identifier))
            except (ValueError,OSError):continue
        rows.extend(authorized)
        return {'sources':rows,'packs':packs,'facts':FACTS,'headlines':HEADLINES,
            'new_api_calls':0,'scope':'private_declared_marketing_media','publication_performed':False}

def _registered_brand(style_id,root=ROOT):
    """Load only fixed approved paths/hashes; a saved snapshot cannot choose IO."""
    if not isinstance(style_id,str)or style_id not in BRAND_REGISTRY:raise ValueError('Unknown approved brand style')
    root=Path(root).resolve();entry=BRAND_REGISTRY[style_id];path=root/entry['path'];ref=file_ref(path,root)
    if ref['bytes']>32000 or ref['sha256']!=entry['sha256']:raise ValueError('Approved brand guide changed')
    raw=path.read_bytes()
    if len(raw)!=ref['bytes']or hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('Approved brand guide changed during read')
    value=json.loads(raw)
    if (not isinstance(value,dict)or value.get('format')!='interaction-twin-marketing-brand'or type(value.get('version'))is not int
        or value['version']!=entry['version']or value.get('style_id')!=style_id or value.get('logo_path')!=entry['logo_path']):raise ValueError('Approved brand guide identity differs')
    logo=file_ref(entry['logo_path'],root)
    if logo['sha256']!=entry['logo_sha256']or logo['bytes']>64000:raise ValueError('Approved logo changed')
    result={'document':value,'reference':ref,'logo':logo}
    if 'logo_raster_path'in entry:
        if value.get('logo_raster_path')!=entry['logo_raster_path']:raise ValueError('Approved brand raster identity differs')
        raster=file_ref(entry['logo_raster_path'],root)
        if raster['sha256']!=entry['logo_raster_sha256']or raster['bytes']>8*1024*1024:raise ValueError('Approved logo raster changed')
        result['logo_raster']=raster
    return result


def validated_brand(snapshot,root=ROOT):
    """Validate the saved style, independent of today's future-campaign default."""
    if not isinstance(snapshot,dict)or not isinstance(snapshot.get('document'),dict):raise ValueError('An exact saved brand snapshot is required')
    expected=_registered_brand(snapshot['document'].get('style_id'),root)
    if snapshot!=expected:raise ValueError('Saved brand document or artifact bindings changed')
    return expected


def brand(root=ROOT):
    root=Path(root).resolve();selector=root/BRAND_SELECTOR
    style_id='cobalt-motion-v1'
    if selector.exists()or selector.is_symlink():
        reference=file_ref(selector,root)
        if reference['bytes']>4096:raise ValueError('Bounded default brand selector required')
        value=json.loads(selector.read_text())
        if (not isinstance(value,dict)or set(value)!={'format','version','brand_id'}or value.get('format')!='interaction-twin-marketing-brand-default'
            or type(value.get('version'))is not int or value['version']!=1):raise ValueError('Unknown default brand selector')
        style_id=value['brand_id']
    return _registered_brand(style_id,root)
