"""Exact public GET adapter for curated authored robot kits/tasks.

The workspace server owns HTTP transport. This adapter has no model calls,
owner authentication, writes, robot execution or private-catalog fallback.
"""
from pathlib import Path
import json
import re
import threading
from urllib.parse import parse_qs, urlencode
from robot_task_catalog import RobotTaskCatalog, encoded, FIXED_FILES

ROOT=Path(__file__).resolve().parent
_CACHE={}
_LOCK=threading.RLock()

def _catalog(root):
    root=Path(root).resolve()
    with _LOCK:
        if root not in _CACHE:
            if len(_CACHE)>=8:_CACHE.pop(next(iter(_CACHE)))
            _CACHE[root]=RobotTaskCatalog(root)
        return _CACHE[root]

def _query(query,allowed):
    if isinstance(query,str):
        if len(query.encode())>8192:raise ValueError('Robot request URL exceeds its bound')
        parsed=parse_qs(query,keep_blank_values=True,strict_parsing=True,max_num_fields=8)
    elif isinstance(query,dict):
        parsed={k:(v if isinstance(v,list)else[v])for k,v in query.items()}
        if len(urlencode(parsed,doseq=True).encode())>8192:raise ValueError('Robot request URL exceeds its bound')
    else:raise ValueError('Invalid robot request query')
    if set(parsed)-set(allowed)or any(len(v)!=1 or not isinstance(v[0],str)or not v[0]for v in parsed.values()):
        raise ValueError('Unknown, duplicate or empty robot request field')
    return {k:v[0]for k,v in parsed.items()}

def response(path,query,root=ROOT):
    """Return None or (HTTP status, exact bytes, MIME, headers). GET only."""
    if not isinstance(path,str)or not path.startswith('/api/robotics/'):return None
    # Do not claim arbitrary /api/ routes: the owner's ordinary gate remains intact.
    simple={'/api/robotics/kits','/api/robotics/kit','/api/robotics/plan','/api/robotics/bundle'}
    kit_match=re.fullmatch(r'/api/robotics/kits/([A-Za-z0-9][A-Za-z0-9_-]{0,79})(?:/artifacts/([A-Za-z0-9][A-Za-z0-9_-]{0,79}))?',path)
    bundle_match=re.fullmatch(r'/api/robotics/bundles/([a-f0-9]{64})',path)
    if path not in simple and not kit_match and not bundle_match:return None
    headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'}
    try:
        catalog=_catalog(root);mime='application/json'
        if path=='/api/robotics/kits':
            _query(query,[]);body=encoded(catalog.list_kits())
        elif path=='/api/robotics/kit':
            q=_query(query,['kit_id']);body=encoded(catalog.get_kit(q.get('kit_id')))
        elif path in {'/api/robotics/plan','/api/robotics/bundle'}:
            q=_query(query,['twin_id','kit_id','instruction'])
            if not q.get('twin_id')or not q.get('instruction'):raise ValueError('Exact curated twin_id and bounded instruction are required')
            if len(q['twin_id'])>200 or len(q['instruction'])>2000:raise ValueError('Robot request field exceeds its bound')
            task=catalog.plan(q['twin_id'],q['instruction'],q.get('kit_id'))
            if path.endswith('/plan'):body=encoded(task)
            else:
                artifact=catalog.export_bundle(task);body=catalog.bundle_artifact(artifact['bundle_id']);mime='application/zip'
                headers['Content-Disposition']='attachment; filename="Interaction-Twin-Robot-Starter.zip"'
                headers['X-Content-SHA256']=artifact['artifact']['sha256']
        elif kit_match:
            _query(query,[]);kit_id,artifact_id=kit_match.groups()
            if artifact_id is None:body=encoded(catalog.get_kit(kit_id))
            else:
                body=catalog.kit_artifact(kit_id,artifact_id)
                mime='application/json'if artifact_id in {'kit','schema'}else FIXED_FILES[artifact_id][1]
        else:
            _query(query,[]);body=catalog.bundle_artifact(bundle_match.group(1));mime='application/zip'
            headers['Content-Disposition']='attachment; filename="Interaction-Twin-Robot-Starter.zip"'
        return 200,body,mime,headers
    except (ValueError,KeyError,TypeError,OSError,RecursionError):
        # Paths, private case IDs and implementation details never enter errors.
        return 400,b'{"error":"Robot request unavailable or invalid. Use a declared kit and current curated twin, and resolve unsupported requirements before export."}', 'application/json',headers
