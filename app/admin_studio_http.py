"""Owner-only marketing and operations routes.

The workspace handler authenticates cookies, same Origin and CSRF before entry.
GET never queues rendering, calls a model, queries SSH or creates an export.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import threading
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE='/api/admin/marketing'
CID=r'campaign_[a-f0-9]{64}'
RID=r'revision_[a-f0-9]{64}'
_INIT=threading.Lock()


def _fields(body,required=(),optional=()):
    if not isinstance(body,dict)or not set(required)<=set(body)or set(body)-set(required)-set(optional):
        raise ValueError('Unexpected or missing studio fields')


def _state(server,root):
    with _INIT:
        if not hasattr(server,'marketing_lock'):
            server.marketing_lock=threading.Lock()
            server.marketing_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='marketing-worker')
            server.marketing_pending=0
            server.marketing_tasks={}
        if not hasattr(server,'marketing_studio'):
            from marketing_studio import MarketingStudio
            server.marketing_studio=MarketingStudio(root=root)
    return server.marketing_studio


def _submit(server,work):
    with server.marketing_lock:
        if server.marketing_pending>=8:raise ValueError('The private studio queue is full; wait for its current jobs')
        server.marketing_pending+=1
    def run():
        try:return work()
        finally:
            with server.marketing_lock:server.marketing_pending-=1
    try:return server.marketing_executor.submit(run)
    except BaseException:
        with server.marketing_lock:server.marketing_pending-=1
        raise


def _render(studio):
    # Each explicit mutation can drain a bounded number of persisted local jobs.
    for _ in range(8):
        if studio.run_next()is None:break


def _project_detail(studio,cid):
    from marketing_sources import file_ref
    value=studio.detail(cid)
    for revision in value['revisions']:
        revision['revision_ref']=file_ref(studio.records/(revision['id']+'.json'),studio.root)
        prefix=BASE+'/campaigns/'+cid+'/revisions/'+revision['id']
        if revision.get('preview_ref'):revision['preview_url']=prefix+'/video'
        if revision.get('export_ref'):revision['download_url']=prefix+'/download'
    # Preserve the actual object shared by latest_revision after decoration.
    if value['revisions']:value['latest_revision']=value['revisions'][-1]
    return value


def detail(studio,cid,server):
    value=_project_detail(studio,cid)
    with server.marketing_lock:
        tasks=[dict(v)for v in server.marketing_tasks.values()if v['campaign_id']==cid]
    value['ai_task']=tasks[-1]if tasks else None
    return value


def _muse(server,studio,cid,body):
    _fields(body,('base_revision_id','allow_paid_api'),('feedback_id',))
    if body['allow_paid_api']is not True:raise ValueError('Explicit guarded Muse action required')
    rid=body['base_revision_id'];fid=body.get('feedback_id')
    identity=hashlib.sha256(json.dumps([cid,rid,fid],separators=(',',':')).encode()).hexdigest()
    taskid='edit_'+identity
    with server.marketing_lock:
        prior=server.marketing_tasks.get(taskid)
        if prior:return detail_without_task_lock(studio,cid,server,prior)
    # Validate the exact packet before scheduling, without admitting API usage.
    preview=studio.propose_with_muse(cid,rid,feedback_id=fid,allow_paid_api=False)
    if preview.get('status')!='prepared_preview_only':
        completed=next((a for a in reversed(preview.get('ai_attempts',[]))if a.get('base_revision_id')==rid and a.get('feedback_id')==fid and a.get('status')=='complete'),None)
        if completed:
            with server.marketing_lock:
                server.marketing_tasks[taskid]={'id':taskid,'campaign_id':cid,'base_revision_id':rid,'feedback_id':fid,'status':'complete','revision_id':completed['revision_id'],'text_only':True,'new_media_uploads':0,'restored_from_saved_result':True}
        return detail(studio,cid,server)
    with server.marketing_lock:
        prior=server.marketing_tasks.get(taskid)
        if prior:return detail_without_task_lock(studio,cid,server,prior)
        task={'id':taskid,'campaign_id':cid,'base_revision_id':rid,'feedback_id':fid,'status':'queued','text_only':True,'new_media_uploads':0}
        server.marketing_tasks[taskid]=task
    def work():
        with server.marketing_lock:task['status']='running'
        try:
            from overnight_budget import OvernightMetaClient
            result=studio.propose_with_muse(cid,rid,feedback_id=fid,client=OvernightMetaClient(root=studio.root),allow_paid_api=True)
            nextid=result['latest_revision']['id']
            with server.marketing_lock:task.update(status='complete',revision_id=nextid)
            _render(studio)
        except Exception as exc:
            with server.marketing_lock:task.update(status='error',error=type(exc).__name__+': '+str(exc)[:500])
    try:_submit(server,work)
    except BaseException:
        with server.marketing_lock:server.marketing_tasks.pop(taskid,None)
        raise
    return detail(studio,cid,server)


def detail_without_task_lock(studio,cid,server,task):
    # Called only while the task lock is held; avoid re-entering it.
    value=_project_detail(studio,cid)
    value['ai_task']=dict(task)
    return value


def upload(handler,length,*,root=ROOT):
    """Called only after the main handler's owner/Origin/CSRF checks."""
    from marketing_music import MusicStore,MAX_UPLOAD
    if not 0<length<=MAX_UPLOAD:return handler.json(413,{'error':'Choose an audio file up to25MiB'})
    kind=handler.headers.get_content_type()
    if not(kind.startswith('audio/')or kind=='application/octet-stream'):
        return handler.json(400,{'error':'Choose a supported audio file'})
    handler.connection.settimeout(30)
    data=handler.rfile.read(length)
    if len(data)!=length:raise ValueError('The audio upload ended early')
    return handler.json(201,MusicStore(root).upload(data,handler.headers.get('X-Filename','')))


def dispatch(handler,route,method,body=None,*,root=ROOT):
    if route=='/api/admin/operations'and method=='GET':
        from admin_operations import snapshot
        handler.json(200,snapshot(root=root));return True
    if route=='/api/admin/operations/gpu/refresh'and method=='POST':
        _fields(body)
        from admin_operations import refresh_gpu
        handler.json(200,refresh_gpu(root=root));return True
    if not route.startswith(BASE+'/'):return False
    studio=_state(handler.server,root)
    if method=='GET':
        if route==BASE+'/sources':handler.json(200,studio.sources());return True
        if route==BASE+'/campaigns':handler.json(200,studio.list_campaigns());return True
        if route==BASE+'/music':handler.json(200,studio.music.list());return True
        campaign=re.fullmatch(BASE+'/campaigns/('+CID+')',route)
        if campaign:handler.json(200,detail(studio,campaign[1],handler.server));return True
        source=re.fullmatch(BASE+r'/sources/([a-z_]{1,40}|pack_[a-f0-9]{64})/video',route)
        music=re.fullmatch(BASE+r'/music/(project_original|music_[a-f0-9]{64})/audio',route)
        artifact=re.fullmatch(BASE+'/campaigns/('+CID+')/revisions/('+RID+')/(video|poster|captions|manifest|download)',route)
        if source:
            from marketing_sources import checked
            path=checked(studio.source_store.get(source[1])['file'],root)
        elif music:path=studio.music.file(music[1])
        elif artifact:
            path=studio.asset(artifact[1],artifact[2],artifact[3])
            if artifact[3]=='download':handler.extra_response_headers=[('Content-Disposition','attachment; filename="Interaction-Twin-Marketing.zip"')]
        else:return False
        handler.serve_file(path,head=getattr(handler,'command','GET')=='HEAD');return True
    if method!='POST':return False
    if route==BASE+'/sources':
        _fields(body,('case_id','rights_acknowledged'))
        handler.json(201,{'source':studio.source_store.authorize_pack(body['case_id'],rights_acknowledged=body['rights_acknowledged'])});return True
    if route==BASE+'/music':handler.json(201,studio.music.register(body));return True
    if route==BASE+'/campaigns':
        _fields(body,('title','source_ids'),('goal','format_id','duration_s','brief','music_id'))
        value=studio.create_campaign(**body);cid=value['campaign']['id']
        _submit(handler.server,lambda:_render(studio))
        handler.json(201,detail(studio,cid,handler.server));return True
    action=re.fullmatch(BASE+'/campaigns/('+CID+')/(generate|feedback|approve|export|render|muse|revoke-feedback)',route)
    if not action:return False
    cid,verb=action.groups()
    if verb=='feedback':
        _fields(body,('revision_id','text'),('scope','preferences'))
        handler.json(201,studio.add_feedback(cid,**body));return True
    if verb=='revoke-feedback':
        _fields(body,('feedback_id',));handler.json(200,studio.revoke_feedback(cid,body['feedback_id']));return True
    if verb=='generate':
        _fields(body,('base_revision_id',),('feedback_id','format_id','duration_s','music_id'))
        studio.generate(cid,**body);_submit(handler.server,lambda:_render(studio))
        handler.json(202,detail(studio,cid,handler.server));return True
    if verb=='render':
        _fields(body,('revision_id',),('format_id',));job=studio.retry_render(cid,**body)
        _submit(handler.server,lambda:_render(studio))
        handler.json(202,job);return True
    if verb=='muse':handler.json(202,_muse(handler.server,studio,cid,body));return True
    _fields(body,('revision_id',));rid=body['revision_id']
    if verb=='approve':handler.json(200,studio.approve_revision(cid,rid));return True
    if verb=='export':
        reference=studio.export_revision(cid,rid)
        handler.json(201,{'reference':reference,'download_url':BASE+'/campaigns/'+cid+'/revisions/'+rid+'/download','publication_performed':False});return True
    return False
