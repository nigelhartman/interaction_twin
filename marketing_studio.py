"""Private campaign studio. Immutable creative revisions; local rendering only.

Feedback is untrusted editing guidance. Approval releases an export, never a
publication. All render inputs are declared IDs and exact hashed project bytes.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
import zipfile

from agent_memory import digest,private_directory,write_immutable
from marketing_sources import ROOT,MarketingSources,FACTS,HEADLINES,brand,checked,file_ref,rendered_text,LIMITATION,CTA
from marketing_music import MusicStore

FORMATS={'landscape':(1280,720),'portrait':(720,1280),'square':(720,720)}
GOALS={'new_twin','pack','pipeline','robotics'}
COPY_HOOKS={'overview':'Show it once. Keep how it works.','robotics':'From a filmed demonstration to an editable robot task.','agents':'Give your agent more than a mesh.','evidence':'Inspect the model. Keep its evidence.'}
PREFERENCES={'pace':{'fast','balanced','slow'},'tone':{'direct','technical','founder'},
    'caption_density':{'minimal','standard'},'focus':{'robotics','agents','library','overview'},'order':{'source_first','result_first'}}
ID=re.compile(r'(campaign|revision|feedback|job|decision|revoke)_[a-f0-9]{64}\Z')

def now():return datetime.now(timezone.utc).isoformat()
def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def bounded_text(value,maximum,*,empty=False):
    if not isinstance(value,str)or len(value)>maximum or(not empty and not value.strip())or '\x00'in value:raise ValueError('Bounded plain text required')
    return value.strip()

def source_chapters(source):
    """Bounded editorial evidence windows, never inferred from a filename."""
    if 'chapters'not in source:return []
    chapters=source['chapters']
    if not isinstance(chapters,list)or not 1<=len(chapters)<=32:raise ValueError('Bounded source chapters required')
    result=[];previous=0.;ids=set()
    for chapter in chapters:
        if not isinstance(chapter,dict)or not {'id','start_s','end_s','fact_ids','headline_ids','evidence_kind'}<=set(chapter):raise ValueError('Typed source chapter fields required')
        if not isinstance(chapter['id'],str)or not re.fullmatch('[a-z][a-z0-9_]{0,39}',chapter['id'])or chapter['id']in ids:raise ValueError('Unique bounded source chapter IDs required')
        start,end=chapter['start_s'],chapter['end_s']
        if any(type(t)not in {int,float}or not math.isfinite(t)for t in [start,end])or not previous<=start<end<=source['duration_s']+.001:raise ValueError('Source chapters must be ordered and within the video')
        for field,allowed in [('fact_ids',FACTS),('headline_ids',HEADLINES)]:
            values=chapter[field]
            if not isinstance(values,list)or len(values)>12 or any(not isinstance(k,str)or k not in allowed for k in values)or len(set(values))!=len(values):raise ValueError('Source chapter copy IDs must be declared')
        if not isinstance(chapter['evidence_kind'],str)or chapter['evidence_kind']not in {'brand','source_video','segmentation','generated_meshes','reasoning','connected_scene','object_program','library','robot_preview','physics','cta'}:raise ValueError('Source chapter evidence kind is unsupported')
        ids.add(chapter['id']);previous=end;result.append(chapter)
    return result

def validate_storyboard(rows,sources,duration):
    if not isinstance(rows,list)or not 1<=len(rows)<=8:raise ValueError('Choose one to eight storyboard scenes')
    total=4.
    for row in rows:
        if not isinstance(row,dict)or set(row)!={'source_id','start_s','duration_s','headline_id','fact_id','transition'}:raise ValueError('Unknown storyboard fields')
        src=sources.get(row['source_id'])
        if not src:raise ValueError('Storyboard source is outside this campaign')
        if row['headline_id']not in HEADLINES or row['fact_id']not in FACTS or row['transition']not in {'crossfade','cobalt_wipe','cut'}:raise ValueError('Choose a grounded headline, fact and transition')
        for key in ['start_s','duration_s']:
            if type(row[key])not in {int,float}or not 0<=row[key]<=60:raise ValueError('Finite bounded scene timing required')
        if row['duration_s']<1 or row['start_s']+row['duration_s']>src['duration_s']+.001:raise ValueError('Scene exceeds declared source footage')
        if row['fact_id']=='robotics'and src['kind']!='robot_preview':raise ValueError('Robot capability needs the actual workbench source')
        chapters=source_chapters(src)
        if (row['fact_id']=='physics'or row['headline_id']=='physics')and not any(c['evidence_kind']=='physics'and row['start_s']>=c['start_s']-1e-6 and row['start_s']+row['duration_s']<=c['end_s']+1e-6 and row['fact_id']in c['fact_ids']and row['headline_id']in c['headline_ids']for c in chapters):
            raise ValueError('Physics copy requires the exact registered physics evidence window')
        if chapters and not any(row['start_s']>=c['start_s']-1e-6 and row['start_s']+row['duration_s']<=c['end_s']+1e-6 and row['fact_id']in c['fact_ids']and row['headline_id']in c['headline_ids']for c in chapters):
            raise ValueError('Scene timing and copy must stay within the matching source chapter')
        total+=row['duration_s']
    # Crossfades overlap scenes, including opening/closing.
    total-=.25*(len(rows)+1)
    if not 8<=total<=60 or abs(total-duration)>.02:raise ValueError('Storyboard timing must include two-second intro/outro and quarter-second transitions')
    return rows

def chapter_storyboard(source_id,source,duration,preferences):
    """Allocate a requested cut across only matching frozen evidence windows."""
    candidates=[c for c in source_chapters(source)if c['fact_ids']and c['headline_ids']and c['end_s']-c['start_s']>=1]
    priorities=['source','masks','parts','connections','program','library','robotics_limit','agents']
    focus={'library':'library','agents':'library','robotics':'robotics_limit'}.get(preferences.get('focus'))
    if focus:priorities=[focus]+[k for k in priorities if k!=focus]
    ranked=sorted(candidates,key=lambda c:(min((priorities.index(k)for k in c['fact_ids']if k in priorities),default=len(priorities)),c['start_s']))
    # A second chapter with the same first fact is useful for the longer cut,
    # after each distinct semantic stage has had a chance to appear.
    unique=[];duplicates=[];seen=set()
    for chapter in ranked:
        key=chapter['fact_ids'][0]
        (duplicates if key in seen else unique).append(chapter);seen.add(key)
    ranked=unique+duplicates
    count=4 if duration<=15 else 6 if duration<=30 else 8
    if preferences.get('pace')=='fast':count+=1
    elif preferences.get('pace')=='slow'or preferences.get('caption_density')=='minimal':count=max(3,count-1)
    count=min(count,len(ranked),8)
    while count and sum(c['end_s']-c['start_s']for c in ranked[:count])+1e-6<duration-4+.25*(count+1):
        if count>=min(len(ranked),8):raise ValueError('Selected chapter evidence cannot fill this cut without mismatched footage')
        count+=1
    chosen=sorted(ranked[:count],key=lambda c:c['start_s'])
    if not chosen:raise ValueError('The selected source has no reusable evidence chapters')
    if preferences.get('order')=='result_first':
        first=next((c for c in chosen if (focus or'connections')in c['fact_ids']),chosen[-1]);chosen=[first]+[c for c in chosen if c is not first]
    remaining=duration-4+.25*(len(chosen)+1);active=list(range(len(chosen)));allocated={}
    while active:
        share=remaining/len(active);saturated=[i for i in active if chosen[i]['end_s']-chosen[i]['start_s']<share]
        if not saturated:
            for i in active:allocated[i]=share
            break
        for i in saturated:
            allocated[i]=chosen[i]['end_s']-chosen[i]['start_s'];remaining-=allocated[i];active.remove(i)
    rows=[]
    for i,chapter in enumerate(chosen):
        length=round(allocated[i],6);start=round(chapter['start_s']+(chapter['end_s']-chapter['start_s']-length)/2,6)
        rows.append({'source_id':source_id,'start_s':start,'duration_s':length,'headline_id':chapter['headline_ids'][0],'fact_id':chapter['fact_ids'][0],'transition':'crossfade'})
    rows[-1]['duration_s']+=duration-(4+sum(row['duration_s']for row in rows)-.25*(len(rows)+1))
    # Decimal output rounding may put a saturated 24fps chapter a few
    # microseconds outside its exact window. Preserve already-valid recipes;
    # otherwise clamp only that bounded rounding error and move the same time
    # into real headroom in another selected chapter. Never widen evidence.
    drift=sum(max(0.,chapter['start_s']-row['start_s'])+max(0.,row['start_s']+row['duration_s']-chapter['end_s'])for row,chapter in zip(rows,chosen))
    if drift>1e-10:
        if drift>1e-5:raise ValueError('Chapter rounding exceeds its bounded precision repair')
        original=sum(row['duration_s']for row in rows)
        for row,chapter in zip(rows,chosen):
            start=max(row['start_s'],chapter['start_s']);end=min(row['start_s']+row['duration_s'],chapter['end_s'])
            row['start_s']=start;row['duration_s']=end-start
        remaining=original-sum(row['duration_s']for row in rows)
        for row,chapter in zip(rows,chosen):
            if remaining<=1e-10:break
            headroom=max(0.,chapter['end_s']-row['start_s']-row['duration_s']);transfer=min(remaining,headroom)
            row['duration_s']+=transfer;remaining-=transfer
        if remaining>1e-10:raise ValueError('Chapter rounding has no evidence headroom')
    return validate_storyboard(rows,{source_id:source},duration)

def default_storyboard(sources,goal,duration,preferences=None):
    choices=list(sources);rows=[];preferences=preferences or {}
    film=next((sid for sid in ['pipeline_dynamic_45s_v3','pipeline_dynamic_45s_v2','pipeline_explainer_refined45','pipeline_explainer_45s','pipeline_explainer','pipeline_demo']if sid in sources and sources[sid]['kind']=='pipeline_film'),None)or next((sid for sid,source in sources.items()if source['kind']=='pipeline_film'),None)
    if film and sources[film].get('chapters'):
        chapter_preferences={**preferences,**({'focus':'robotics'}if goal=='robotics'else{})}
        return chapter_storyboard(film,sources[film],duration,chapter_preferences)
    robot='robot_workbench'in sources
    capture=next((k for k,v in sources.items()if v['kind']=='source_video'),None)
    if goal=='robotics'and robot:
        templates=[(capture or(film if film else choices[0]),1,'source','source'),('robot_workbench',0,'robotics','robotics'),('robot_workbench',9,'robotics','robotics_limit')]
    elif film:
        templates=[(capture or film,0,'source','source'),(film,5,'masks','masks'),(film,13,'connections','connections'),(film,21,'program','program')]
        if robot:templates.append(('robot_workbench',0,'robotics','robotics'))
    else:
        templates=[(choices[0],1,'source','source'),(choices[0],10,'connections','connections'),(choices[-1],20,'library','library')]
    if preferences.get('focus')=='robotics'and robot:
        templates=[('robot_workbench',0,'robotics','robotics')]+[r for r in templates if r[0]!='robot_workbench']
    elif preferences.get('focus')in {'library','agents'}and film:
        key=preferences['focus'];templates=[(film,29 if key=='library' else 36,key,key)]+templates
    if preferences.get('order')=='result_first':templates=templates[1:]+templates[:1]
    if preferences.get('pace')=='slow'or preferences.get('caption_density')=='minimal':templates=templates[:3]
    elif preferences.get('pace')=='fast'and len(templates)<6:templates=templates+[templates[0]]
    count=len(templates);each=(duration-4+.25*(count+1))/count
    for sid,start,head,fact in templates:
        source=sources[sid];start=min(float(start),source['duration_s']-each)
        rows.append({'source_id':sid,'start_s':max(0,round(start,6)),'duration_s':round(each,6),'headline_id':head,'fact_id':fact,'transition':'crossfade'})
    rows[-1]['duration_s']+=duration-(4+sum(r['duration_s']for r in rows)-.25*(len(rows)+1))
    return validate_storyboard(rows,sources,duration)

def social_copy(goal,source_ids,preferences=None,copy_spec=None):
    # All copy is selected from declared statements; preferences cannot invent facts.
    robotic=goal=='robotics'and'robot_workbench'in source_ids
    first='From a filmed demonstration to an editable robot task.'if robotic else'Show it once. Keep how it works.'
    tone=(preferences or {}).get('tone')
    if tone=='technical':first='Video evidence, native meshes and a source-linked object program.'
    elif tone=='founder':first='We are building reusable object descriptions from filmed interactions.'
    body='Native 3D parts, estimated connections and source-linked descriptions in one inspectable object program.'
    limitation=LIMITATION
    if copy_spec is not None:
        if not isinstance(copy_spec,dict)or set(copy_spec)!={'hook_id','body_fact_ids'}or copy_spec['hook_id']not in COPY_HOOKS or not isinstance(copy_spec['body_fact_ids'],list)or not 1<=len(copy_spec['body_fact_ids'])<=4 or any(not isinstance(k,str)or k not in FACTS for k in copy_spec['body_fact_ids']):raise ValueError('Choose approved copy hook and fact IDs')
        if ('robotics'in copy_spec['body_fact_ids']or copy_spec['hook_id']=='robotics')and'robot_workbench'not in source_ids:raise ValueError('Robot copy requires the actual workbench source')
        if 'physics'in copy_spec['body_fact_ids']and not {'pipeline_dynamic_45s_v2','pipeline_dynamic_45s_v3'}.intersection(source_ids):raise ValueError('Physics copy requires the registered Newton source')
        first=COPY_HOOKS[copy_spec['hook_id']];sentences=[FACTS[k]for k in dict.fromkeys(copy_spec['body_fact_ids'])]
        body=' '.join(sentences);robotic=robotic or'robotics'in copy_spec['body_fact_ids']or copy_spec['hook_id']=='robotics'
        x=first
        for sentence in sentences:
            if len(x+' '+sentence+' '+limitation+' '+CTA)<=280:x+=' '+sentence
        x+=' '+limitation+' '+CTA
        if len(x)>280:raise ValueError('Mandatory social copy exceeds X character limit')
        return {'x':x,'linkedin':first+'\n\n'+body+'\n\n'+limitation+'\n\n'+CTA,'alt':'Interaction Twin project demonstration. '+body+' '+limitation,'captions':sentences+[limitation,CTA],'fact_ids':list(copy_spec['body_fact_ids']),'generated_from':'muse_selected_declared_fact_ids_v2','mandatory_limitation':limitation,'cta':CTA,'publication_performed':False}
    x=first
    if len(x+' '+body+' '+limitation+' '+CTA)<=280:x+=' '+body
    x+=' '+limitation+' '+CTA
    if len(x)>280:raise ValueError('Mandatory social copy exceeds X character limit')
    return {'x':x,
        'linkedin':first+'\n\nInteraction Twin turns captured interactions into descriptive digital twins: separate native meshes, component relationships, source-linked steps and evidence. People and agents can inspect a curated library.\n\n'+limitation+'\n\n'+CTA,
        'alt':'Interaction Twin project demonstration. '+body+' '+limitation,
        'captions':[FACTS['source'],FACTS['connections'],FACTS['robotics']if robotic else FACTS['library'],limitation,CTA],
        'fact_ids':['source','parts','connections','robotics'if robotic else'library'],
        'generated_from':'declared_project_fact_templates_v2','mandatory_limitation':limitation,'cta':CTA,'publication_performed':False}

class MarketingStudio:
    def __init__(self,root=ROOT,*,sources=None,music=None):
        self.root=Path(root).resolve();self.folder=private_directory(self.root,'marketing-studio')
        self.records=self.folder/'records';self.records.mkdir(exist_ok=True,mode=0o700)
        self.renders=self.folder/'renders';self.renders.mkdir(exist_ok=True,mode=0o700)
        if self.records.is_symlink()or self.renders.is_symlink():raise ValueError('Ordinary private studio directories required')
        self.db_path=self.folder/'studio.sqlite3'
        if self.db_path.is_symlink():raise ValueError('Ordinary studio index required')
        self.source_store=sources or MarketingSources(self.root);self.music=music or MusicStore(self.root)
        with self._db()as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS campaigns(id TEXT PRIMARY KEY,created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS revisions(id TEXT PRIMARY KEY,campaign_id TEXT NOT NULL,created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS feedback(id TEXT PRIMARY KEY,campaign_id TEXT NOT NULL,revision_id TEXT NOT NULL,created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS revocations(id TEXT PRIMARY KEY,feedback_id TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,campaign_id TEXT NOT NULL,revision_id TEXT NOT NULL,format_id TEXT NOT NULL,status TEXT NOT NULL,progress REAL NOT NULL DEFAULT 0,error TEXT,token TEXT,heartbeat TEXT,result TEXT,created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS approvals(revision_id TEXT PRIMARY KEY,decision_id TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS exports(revision_id TEXT PRIMARY KEY,reference TEXT NOT NULL);''')
        self.db_path.chmod(0o600)
    @contextmanager
    def _db(self):
        db=sqlite3.connect(self.db_path,timeout=15);db.row_factory=sqlite3.Row
        try:yield db;db.commit()
        except BaseException:db.rollback();raise
        finally:db.close()
    def _write(self,prefix,value):return write_immutable(self.records,prefix,value)
    def _read(self,identifier,prefix):
        if not isinstance(identifier,str)or not ID.fullmatch(identifier)or not identifier.startswith(prefix+'_'):raise ValueError('Unknown studio record ID')
        path=self.records/(identifier+'.json')
        if path.is_symlink()or not path.is_file()or path.stat().st_size>512000:raise ValueError('Studio record unavailable')
        value=json.loads(path.read_text())
        if identifier!=prefix+'_'+digest(value):raise ValueError('Immutable studio record changed')
        return value
    def sources(self):return self.source_store.list()|{'formats':[{'id':k,'width':v[0],'height':v[1]}for k,v in FORMATS.items()],'goals':sorted(GOALS),'brand':brand(self.root),'music':self.music.list()['music']}
    def list_campaigns(self):
        with self._db()as db:ids=[r['id']for r in db.execute('SELECT id FROM campaigns ORDER BY created DESC LIMIT 200')]
        return {'campaigns':[{'id':i,**self._read(i,'campaign')}for i in ids],'new_api_calls':0}
    def _campaign(self,identifier):
        value=self._read(identifier,'campaign')
        with self._db()as db:
            if not db.execute('SELECT 1 FROM campaigns WHERE id=?',(identifier,)).fetchone():raise ValueError('Unknown campaign')
        return value
    def _revision(self,campaign_id,revision_id):
        self._campaign(campaign_id);value=self._read(revision_id,'revision')
        if value['campaign_id']!=campaign_id:raise ValueError('Revision belongs to another campaign')
        with self._db()as db:
            if not db.execute('SELECT 1 FROM revisions WHERE id=? AND campaign_id=?',(revision_id,campaign_id)).fetchone():raise ValueError('Unknown campaign revision')
        return value
    def _preferences(self,campaign_id):
        with self._db()as db:
            rows=db.execute('SELECT id FROM feedback ORDER BY created,id').fetchall()
            revoked={r['feedback_id']for r in db.execute('SELECT feedback_id FROM revocations')}
        found={}
        for r in rows:
            item=self._read(r['id'],'feedback')
            if r['id']not in revoked and(item['scope']=='brand'or item['campaign_id']==campaign_id):found.update(item['preferences'])
        return found
    def feedback_memory(self,campaign_id):
        """Bounded, revocable editing notes. Never facts or trained weights."""
        self._campaign(campaign_id)
        with self._db()as db:
            rows=db.execute('SELECT id FROM feedback ORDER BY created DESC,id DESC LIMIT 1000').fetchall()
            revoked={r['feedback_id']for r in db.execute('SELECT feedback_id FROM revocations')}
        result=[]
        for row in rows:
            item=self._read(row['id'],'feedback')
            if row['id']not in revoked and(item['scope']=='brand'or item['campaign_id']==campaign_id):result.append({'id':row['id'],**item})
            if len(result)>=8:break
        return result
    def create_campaign(self,title,*,source_ids,goal='pipeline',format_id='landscape',duration_s=30,brief='',music_id='project_original'):
        title=bounded_text(title,120);brief=bounded_text(brief,2000,empty=True)
        if not isinstance(goal,str)or goal not in GOALS or not isinstance(format_id,str)or format_id not in FORMATS or type(duration_s)not in {int,float}or duration_s not in {15,30,45}:raise ValueError('Choose a supported goal, format and15/30/45-second duration')
        if not isinstance(source_ids,list)or not 1<=len(source_ids)<=8 or any(not isinstance(s,str)for s in source_ids)or len(set(source_ids))!=len(source_ids):raise ValueError('Choose one to eight declared source IDs')
        sources={sid:self.source_store.get(sid)for sid in source_ids}
        if goal=='robotics'and'robot_workbench'not in sources:raise ValueError('Robotics campaign needs its actual workbench recording')
        score=self.music.describe(music_id)if music_id is not None else None
        branding=brand(self.root);story=default_storyboard(sources,goal,duration_s)
        value={'format':'interaction-twin-marketing-campaign','version':1,'created_utc':now(),'nonce':secrets.token_hex(8),'title':title,'goal':goal,'source_ids':source_ids,'sources':sources,'brief':brief,'brief_untrusted':True,'format_id':format_id,'duration_s':duration_s,'music_id':music_id,'brand':branding,'music':score,'publication_performed':False}
        cid=self._write('campaign',value)
        with self._db()as db:db.execute('INSERT INTO campaigns VALUES(?,?)',(cid,value['created_utc']))
        self._new_revision(cid,base=None,feedback=None,storyboard=story,format_id=format_id,duration_s=duration_s,music_id=music_id)
        return self.detail(cid)
    def _new_revision(self,cid,*,base,feedback,storyboard,format_id,duration_s,music_id,agent_proof=None,copy_spec=None,presentation_only=False):
        campaign=self._campaign(cid)
        if type(presentation_only)is not bool:raise ValueError('Explicit internal presentation mode required')
        if format_id not in FORMATS or duration_s not in {15,30,45}:raise ValueError('Choose a supported render format/duration')
        prior=self._revision(cid,base)if base else None
        if presentation_only:
            if prior is None or feedback is not None:raise ValueError('Presentation refresh needs an exact base and cannot consume feedback')
            supplied={'storyboard':storyboard,'format_id':format_id,'duration_s':duration_s,'music_id':music_id,'agent_proof':agent_proof,'copy_spec':copy_spec}
            expected={key:prior.get(key)for key in supplied}
            if digest(supplied)!=digest(expected):raise ValueError('Presentation refresh cannot change creative inputs')
            if digest(prior['sources'])!=digest(campaign['sources'])or digest(prior['brand'])!=digest(campaign['brand']):raise ValueError('Presentation source and brand bindings differ from the campaign')
            validate_storyboard(storyboard,prior['sources'],duration_s)
            for source in prior['sources'].values():checked(source['file'],self.root)
            checked(prior['brand']['reference'],self.root);checked(prior['brand']['logo'],self.root)
            if prior['music']is not None:checked(prior['music']['file'],self.root)
            # Clone every creative field, including model proof and the exact
            # preferences/copy that the critic saw. Current feedback is not read.
            value=copy.deepcopy(prior)
            creative_keys=['storyboard','copy','copy_spec','render_text','preferences','format_id','duration_s','music_id','music','brand','sources','agent_proof']
            creative={key:prior.get(key)for key in creative_keys}
            value.update(created_utc=now(),base_revision_id=base,feedback_id=None,kind='presentation_refresh',
                renderer_version=2,repair_round=prior['repair_round'],human_approval_required=True,
                publication_performed=False,presentation_source_revision_ref=file_ref(self.records/(base+'.json'),self.root),
                creative_content_sha256=digest(creative))
            if digest({key:value.get(key)for key in creative_keys})!=digest(creative):raise ValueError('Presentation refresh changed creative content')
        else:
            preferences=self._preferences(cid);count=prior['repair_round']+1 if prior else 0
            if count>2:raise ValueError('This lineage has reached two repair rounds; create another campaign for a new brief')
            validate_storyboard(storyboard,campaign['sources'],duration_s)
            music=self.music.describe(music_id)if music_id is not None else None
            value={'format':'interaction-twin-marketing-revision','version':1,'campaign_id':cid,'created_utc':now(),'base_revision_id':base,'feedback_id':feedback,'repair_round':count,'storyboard':storyboard,'copy':social_copy(campaign['goal'],campaign['source_ids'],preferences,copy_spec),'render_text':rendered_text(storyboard,campaign['sources'],preferences),'copy_spec':copy_spec,'preferences':preferences,'format_id':format_id,'duration_s':duration_s,'music_id':music_id,'music':music,'brand':campaign['brand'],'sources':campaign['sources'],'agent_proof':agent_proof,'human_approval_required':True,'publication_performed':False}
        with self._db()as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT count(*) FROM revisions WHERE campaign_id=?',(cid,)).fetchone()[0]>=64:raise ValueError('Campaign revision budget reached')
            rid=self._write('revision',value)
            db.execute('INSERT INTO revisions VALUES(?,?,?)',(rid,cid,value['created_utc']))
        self.queue_render(cid,rid,format_id)
        return rid
    def refresh_presentation(self,campaign_id,revision_id):
        """Local editorial rerender only; preserve creative repair allowance."""
        prior=self._revision(campaign_id,revision_id)
        self._new_revision(campaign_id,base=revision_id,feedback=None,storyboard=prior['storyboard'],
            format_id=prior['format_id'],duration_s=prior['duration_s'],music_id=prior['music_id'],
            agent_proof=prior.get('agent_proof'),copy_spec=prior.get('copy_spec'),presentation_only=True)
        return self.detail(campaign_id)
    def generate(self,campaign_id,*,base_revision_id,feedback_id=None,format_id=None,duration_s=None,music_id=None):
        prior=self._revision(campaign_id,base_revision_id);campaign=self._campaign(campaign_id)
        if feedback_id:
            fb=self._read(feedback_id,'feedback')
            if fb['campaign_id']!=campaign_id or fb['revision_id']!=base_revision_id:raise ValueError('Feedback belongs to another revision')
        duration=duration_s or prior['duration_s'];preferences=self._preferences(campaign_id)
        story=default_storyboard(campaign['sources'],campaign['goal'],duration,preferences)
        self._new_revision(campaign_id,base=base_revision_id,feedback=feedback_id,storyboard=story,format_id=format_id or prior['format_id'],duration_s=duration,music_id=music_id if music_id is not None else prior['music_id'])
        return self.detail(campaign_id)
    def add_feedback(self,campaign_id,revision_id,text,scope='campaign',preferences=None):
        self._revision(campaign_id,revision_id);text=bounded_text(text,2000)
        preferences=preferences or {}
        if not isinstance(scope,str)or scope not in {'campaign','brand'}or not isinstance(preferences,dict)or set(preferences)-set(PREFERENCES)or any(not isinstance(v,str)or v not in PREFERENCES[k]for k,v in preferences.items()):raise ValueError('Choose known scoped editing preferences')
        value={'campaign_id':campaign_id,'revision_id':revision_id,'text':text,'scope':scope,'preferences':preferences,'created_utc':now(),'untrusted_annotation':True,'observed_fact':False,'training_performed':False}
        fid=self._write('feedback',value)
        with self._db()as db:db.execute('INSERT INTO feedback VALUES(?,?,?,?)',(fid,campaign_id,revision_id,value['created_utc']))
        return {'id':fid,**value}
    def revoke_feedback(self,campaign_id,feedback_id):
        self._campaign(campaign_id);fb=self._read(feedback_id,'feedback')
        if fb['campaign_id']!=campaign_id:raise ValueError('Feedback belongs to another campaign')
        value={'feedback_id':feedback_id,'created_utc':now(),'actor':'owner'};rid=self._write('revoke',value)
        with self._db()as db:db.execute('INSERT INTO revocations VALUES(?,?)',(rid,feedback_id))
        return {'id':rid,**value}
    def queue_render(self,campaign_id,revision_id,format_id):
        revision=self._revision(campaign_id,revision_id)
        if format_id not in FORMATS:raise ValueError('Unknown render format')
        value={'campaign_id':campaign_id,'revision_id':revision_id,'format_id':format_id,'revision_sha256':file_ref(self.records/(revision_id+'.json'),self.root)['sha256'],'renderer_version':2}
        jid=self._write('job',value)
        with self._db()as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM approvals WHERE revision_id=?',(revision_id,)).fetchone()and not db.execute('SELECT 1 FROM jobs WHERE id=?',(jid,)).fetchone():raise ValueError('Approved render set is frozen; create another revision for another format')
            db.execute('INSERT OR IGNORE INTO jobs(id,campaign_id,revision_id,format_id,status,created) VALUES(?,?,?,?,?,?)',(jid,campaign_id,revision_id,format_id,'planned',now()))
        return self.job(jid)
    def retry_render(self,campaign_id,revision_id,format_id=None):
        revision=self._revision(campaign_id,revision_id);format_id=format_id or revision['format_id']
        if format_id not in FORMATS:raise ValueError('Unknown render format')
        with self._db()as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM approvals WHERE revision_id=?',(revision_id,)).fetchone():raise ValueError('Approved render set is frozen')
            rows=db.execute('SELECT status FROM jobs WHERE revision_id=? AND format_id=?',(revision_id,format_id)).fetchall()
            if not rows or any(r['status']!='error'for r in rows)or len(rows)>=3:raise ValueError('Only a failed local render may be explicitly retried, at most twice')
            value={'campaign_id':campaign_id,'revision_id':revision_id,'format_id':format_id,'revision_sha256':file_ref(self.records/(revision_id+'.json'),self.root)['sha256'],'renderer_version':2,'explicit_owner_retry':len(rows)}
            jid=self._write('job',value);db.execute('INSERT INTO jobs(id,campaign_id,revision_id,format_id,status,created) VALUES(?,?,?,?,?,?)',(jid,campaign_id,revision_id,format_id,'planned',now()))
        return self.job(jid)
    def job(self,job_id):
        self._read(job_id,'job')
        with self._db()as db:row=db.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone()
        if not row:raise ValueError('Unknown render job')
        value=dict(row);value.pop('token',None);value['result']=json.loads(value['result'])if value['result']else None
        return value
    def run_next(self):
        self.recover_interrupted_jobs()
        with self._db()as db:row=db.execute("SELECT id FROM jobs WHERE status='planned' ORDER BY created,id LIMIT 1").fetchone()
        return self.run_job(row['id'])if row else None
    def recover_interrupted_jobs(self):
        """No retry: a free kernel lock fences abandoned local render claims."""
        fd=os.open(self.folder/'renderer.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'a+')as stream:
            try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return 0
            with self._db()as db:return db.execute("UPDATE jobs SET status='error',error='Local render worker was interrupted; explicit retry creates a new preserved attempt.',token=NULL WHERE status='rendering'").rowcount
    def run_job(self,job_id):
        request=self._read(job_id,'job');revision=self._revision(request['campaign_id'],request['revision_id'])
        if file_ref(self.records/(request['revision_id']+'.json'),self.root)['sha256']!=request['revision_sha256']:raise ValueError('Render revision changed')
        lock=self.folder/'renderer.lock'
        fd=os.open(lock,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'a+')as stream:
            try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return self.job(job_id)
            token=secrets.token_hex(24)
            with self._db()as db:
                # The exclusive kernel lock proves no legitimate renderer is
                # running. A claim left by a killed process is terminal, never
                # silently retried into its existing output directory.
                db.execute("UPDATE jobs SET status='error',error='Local render worker was interrupted; explicit retry creates a new preserved attempt.',token=NULL WHERE status='rendering'")
                changed=db.execute("UPDATE jobs SET status='rendering',token=?,heartbeat=?,error=NULL WHERE id=? AND status='planned'",(token,now(),job_id)).rowcount
            if not changed:return self.job(job_id)
            def progress(value):
                with self._db()as db:
                    if not db.execute("UPDATE jobs SET heartbeat=?,progress=? WHERE id=? AND token=? AND status='rendering'",(now(),value,job_id,token)).rowcount:raise ValueError('Render worker lost its claim')
            try:
                from marketing_render import render
                result=render(revision,request['format_id'],self.renders/job_id,root=self.root,progress=progress)
                with self._db()as db:
                    if not db.execute("UPDATE jobs SET status='review_ready',progress=1,result=?,token=NULL WHERE id=? AND token=? AND status='rendering'",(json.dumps(result),job_id,token)).rowcount:raise ValueError('Render worker lost its finalization claim')
            except Exception as exc:
                # No retry: partial output and immutable inputs stay available for diagnosis.
                with self._db()as db:db.execute("UPDATE jobs SET status='error',error=?,token=NULL WHERE id=? AND token=?",(type(exc).__name__+': '+str(exc)[:500],job_id,token))
            return self.job(job_id)
    def detail(self,campaign_id):
        campaign=self._campaign(campaign_id)
        with self._db()as db:
            rows=db.execute('SELECT id FROM revisions WHERE campaign_id=? ORDER BY created,id',(campaign_id,)).fetchall()
            fids=[r['id']for r in db.execute('SELECT id FROM feedback WHERE campaign_id=? ORDER BY created',(campaign_id,))]
            revoked={r['feedback_id']for r in db.execute('SELECT feedback_id FROM revocations')}
            approvals={r['revision_id']:r['decision_id']for r in db.execute('SELECT * FROM approvals')}
            exports={r['revision_id']:json.loads(r['reference'])for r in db.execute('SELECT * FROM exports')}
        revisions=[];jobs=[]
        for r in rows:
            rid=r['id'];value=self._revision(campaign_id,rid)
            with self._db()as db:jids=[r['id']for r in db.execute('SELECT id FROM jobs WHERE revision_id=? ORDER BY created',(rid,))]
            rjobs=[self.job(j)for j in jids];jobs+=rjobs
            ready=next((j for j in reversed(rjobs)if j['status']=='review_ready'),None)
            revisions.append({'id':rid,**value,'jobs':rjobs,'status':ready['status']if ready else(rjobs[-1]['status']if rjobs else'planned'),'preview_ref':ready['result']['video']if ready else None,'approved':rid in approvals,'decision_id':approvals.get(rid),'export_ref':exports.get(rid)})
        return {'campaign':{'id':campaign_id,**campaign},'revisions':revisions,'latest_revision':revisions[-1]if revisions else None,'jobs':jobs,'feedback':[{'id':i,**self._read(i,'feedback'),'revoked':i in revoked}for i in fids],'feedback_memory':self.feedback_memory(campaign_id),'preferences':self._preferences(campaign_id),'ai_attempts':self.muse_attempts(campaign_id),'new_api_calls':0,'publication_performed':False}
    def muse_attempts(self,campaign_id):
        self._campaign(campaign_id);folder=self.folder/'muse'
        if not folder.exists():return []
        if folder.is_symlink():raise ValueError('Ordinary private AI attempts required')
        paths=sorted(folder.glob('*/packet.json'))
        if len(paths)>1000:raise ValueError('Marketing attempt inspection budget reached')
        result=[]
        for path in paths:
            if path.is_symlink()or path.parent.is_symlink()or path.stat().st_size>256000:raise ValueError('Bounded AI packet required')
            packet=json.loads(path.read_text())
            if packet.get('campaign_id')!=campaign_id:continue
            feedback=packet.get('untrusted_feedback');feedback_id=None
            if feedback is not None:
                if not isinstance(feedback,dict):raise ValueError('Bounded feedback binding required')
                feedback_id='feedback_'+digest(feedback)
                if self._read(feedback_id,'feedback')!=feedback:raise ValueError('Stored AI feedback binding changed')
            created=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()
            state={'id':path.parent.name,'base_revision_id':packet['base_revision_id'],'feedback_id':feedback_id,
                'created_utc':created,'time_origin':'packet_file_timestamp_approximate','status':'running','text_only':True,'new_media_uploads':0}
            for name,status in [('rejected.json','needs_owner_direction'),('failure.json','error'),('complete.json','complete')]:
                p=path.parent/name
                if p.exists():
                    if p.is_symlink()or p.stat().st_size>64000:raise ValueError('Bounded AI attempt result required')
                    value=json.loads(p.read_text());state.update(status=status,**{k:value[k]for k in ['error','revision_id','critique']if k in value})
            critic=path.parent/'critique.json'
            if critic.is_file()and not critic.is_symlink()and critic.stat().st_size<64000:state['critique']=json.loads(critic.read_text())['critique']
            proposal=path.parent/'proposal.json'
            if proposal.is_file()and not proposal.is_symlink()and proposal.stat().st_size<64000:state['creative_notes']=json.loads(proposal.read_text())['proposal'].get('creative_notes',[])
            cycles=[]
            for number in [1,2]:
                cycle=path.parent/('cycle-'+str(number))
                if not cycle.exists():continue
                if cycle.is_symlink()or not cycle.is_dir():raise ValueError('Ordinary bounded AI cycle required')
                row={'cycle_number':number,'status':'started'}
                for name,field in [('proposal.json','proposal'),('critique.json','critique')]:
                    entry=cycle/name
                    if not entry.exists():continue
                    if entry.is_symlink()or entry.stat().st_size>64000:raise ValueError('Bounded immutable AI cycle record required')
                    value=json.loads(entry.read_text())[field]
                    if field=='critique':row.update(status='critic_complete',critique=value);state['critique']=value
                    else:row['creative_notes']=value.get('creative_notes',[]);state['creative_notes']=row['creative_notes']
                cycles.append(row)
            state.update(protocol=packet.get('version',1),cycles=cycles,cycle_count=len(cycles),maximum_editor_critic_cycles=packet.get('maximum_editor_critic_cycles',1))
            result.append(state)
        return sorted(result,key=lambda row:(row['created_utc'],row['id']))
    def approve_revision(self,campaign_id,revision_id,actor='owner'):
        revision=self._revision(campaign_id,revision_id)
        if actor!='owner':raise ValueError('Only authenticated owner approval is supported')
        with self._db()as db:
            prior=db.execute('SELECT decision_id FROM approvals WHERE revision_id=?',(revision_id,)).fetchone()
            rows=db.execute("SELECT id,format_id,result FROM jobs WHERE revision_id=? AND status='review_ready' ORDER BY id",(revision_id,)).fetchall()
            active=db.execute("SELECT 1 FROM jobs WHERE revision_id=? AND status IN ('planned','rendering')",(revision_id,)).fetchone()
        if prior:
            old=self._read(prior['decision_id'],'decision');checked(old['revision_ref'],self.root)
            for job in old['approved_renders']:
                for key in ['video','poster','captions','manifest']:checked(job['result'][key],self.root)
            return {'id':prior['decision_id'],**old}
        if active:raise ValueError('Wait for every queued format before approving the render set')
        if not rows:raise ValueError('Render and inspect this revision before approval')
        approved=[]
        for row in rows:
            result=json.loads(row['result'])
            for key in ['video','poster','captions','manifest']:checked(result[key],self.root)
            approved.append({'job_id':row['id'],'format_id':row['format_id'],'result':result})
        value={'campaign_id':campaign_id,'revision_id':revision_id,'revision_ref':file_ref(self.records/(revision_id+'.json'),self.root),'approved_renders':approved,'actor':actor,'created_utc':now(),'approval_scope':'exact_render_set_download_only','publication_authorized':False}
        decision=self._write('decision',value)
        with self._db()as db:
            db.execute('BEGIN IMMEDIATE')
            current=db.execute("SELECT id FROM jobs WHERE revision_id=? AND status='review_ready' ORDER BY id",(revision_id,)).fetchall()
            if [r['id']for r in current]!=[r['job_id']for r in approved]or db.execute("SELECT 1 FROM jobs WHERE revision_id=? AND status IN ('planned','rendering')",(revision_id,)).fetchone():raise ValueError('Render set changed; inspect it again before approving')
            db.execute('INSERT OR IGNORE INTO approvals VALUES(?,?)',(revision_id,decision))
        return {'id':decision,**value}
    def export_revision(self,campaign_id,revision_id):
        revision=self._revision(campaign_id,revision_id)
        with self._db()as db:
            decision=db.execute('SELECT decision_id FROM approvals WHERE revision_id=?',(revision_id,)).fetchone()
            prior=db.execute('SELECT reference FROM exports WHERE revision_id=?',(revision_id,)).fetchone()
        if not decision:raise ValueError('Owner approval is required before export')
        approval=self._read(decision['decision_id'],'decision');checked(approval['revision_ref'],self.root)
        if prior:
            ref=json.loads(prior['reference']);checked(ref,self.root);return ref
        output=self.renders/('export-'+revision_id+'.zip');files={}
        for job in approval['approved_renders']:
            result=job['result']
            for key in ['video','poster','captions','manifest']:
                path=checked(result[key],self.root);files[job['format_id']+'/'+path.name]=path.read_bytes()
        files['social-copy.json']=encoded(revision['copy']);files['storyboard.json']=encoded(revision['storyboard']);files['brand.json']=encoded(revision['brand']);files['approval.json']=encoded(approval);files['source-provenance.json']=encoded({'sources':revision['sources'],'music':revision['music'],'physical_accuracy_verified':False,'publication_performed':False})
        files['mark.svg']=checked(revision['brand']['logo'],self.root).read_bytes()
        if sum(len(b)for b in files.values())>300*1024*1024:raise ValueError('Export exceeds bounded package size')
        with output.open('xb')as stream:
            with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED)as archive:
                for name,data in sorted(files.items()):archive.writestr(name,data)
        output.chmod(0o600);ref=file_ref(output,self.root)
        with self._db()as db:db.execute('INSERT INTO exports VALUES(?,?)',(revision_id,json.dumps(ref)))
        return ref
    def asset(self,campaign_id,revision_id,kind='video',format_id=None):
        """Authenticated-route resolver. GET never starts work or creates export."""
        self._revision(campaign_id,revision_id)
        if kind=='download':
            with self._db()as db:row=db.execute('SELECT reference FROM exports WHERE revision_id=?',(revision_id,)).fetchone()
            if not row:raise ValueError('Approve and export this revision first')
            return checked(json.loads(row['reference']),self.root)
        if kind not in {'video','poster','captions','manifest'}or(format_id is not None and format_id not in FORMATS):raise ValueError('Unknown marketing artifact')
        with self._db()as db:rows=db.execute("SELECT format_id,result FROM jobs WHERE revision_id=? AND status='review_ready' ORDER BY created DESC",(revision_id,)).fetchall()
        row=next((r for r in rows if format_id is None or r['format_id']==format_id),None)
        if not row:raise ValueError('Requested render is not ready')
        return checked(json.loads(row['result'])[kind],self.root)
    def propose_with_muse(self,campaign_id,base_revision_id,*,feedback_id=None,client=None,allow_paid_api=False):
        from marketing_muse import propose
        return propose(self,campaign_id,base_revision_id,feedback_id=feedback_id,client=client,allow_paid_api=allow_paid_api)
