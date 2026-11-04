"""Source-bound correction of discoverable task metadata, never mesh or pose edits."""
import copy
import hashlib
import json
from pathlib import Path

import curated_publication as C
from curated_twin_catalog import CuratedTwinCatalog
from guarded_task_metadata_v1 import derive
from library_query import derive_index
import object_program as O

def refresh(*,root=C.ROOT,output):
    root=Path(root).resolve();output=Path(output)
    C.require(not output.exists(),'Preserve prior collection refresh receipt')
    public=root/'showcase/public';index=C.read(public/'library-index.json')
    previous=C.ref(public/index['catalog']['path'],root)
    C.require(previous['sha256']==index['catalog']['sha256'] and previous['bytes']==index['catalog']['bytes'],'Catalog pointer differs')
    snapshot=CuratedTwinCatalog(root);approved,_=snapshot._snapshot()
    document=C.read(root/previous['path']);old=copy.deepcopy(document)
    C.require(document['records']==approved,'Catalog differs from exact approved snapshot')
    scene_ref=C.ref(public/'aligned-scenes.json',root);scenes=C.read(public/'aligned-scenes.json')['scenes'];proofs=[]
    for record in document['records']:
        row=next(s for s in scenes if s['source_sha256']==record['source']['sha256'])
        scene=C.read(O.checked_ref(row['manifest'],root))
        if scene.get('mechanics',{}).get('rigidity_feasibility'):
            program=C.read(O.checked_ref(record['interaction'],root))
            companion=C.read(O.checked_ref(row['object_program'],root));O.validate(companion,root=root)
            C.require(companion['evidence_program']==record['interaction'],'Classification procedure differs')
            before=record['task_classes'];record['task_classes']=derive(program,object_program=companion,scene=scene)
            proofs.append({'id':record['id'],'source_sha256':record['source']['sha256'],'program':record['interaction'],'scene':row['manifest'],'companion':row['object_program'],'before':before,'after':record['task_classes']})
        record['search_index']=derive_index(record)
    def stripped(value):
        rows=copy.deepcopy(value['records'])
        for r in rows:r.pop('task_classes',None);r.pop('search_index',None)
        return rows
    C.require(stripped(document)==stripped(old),'Metadata refresh attempted a geometry, scene, procedure or approval change')
    C.require(C.ref(public/'aligned-scenes.json',root)==scene_ref,'Scenes changed during refresh')
    document['provenance']['collection_metadata_refresh']={'version':1,'previous_catalog':previous,'implementation':C.ref(__file__,root),'classifier':C.ref(root/'guarded_task_metadata_v1.py',root),'scope':'Exact typed source-nonrigidity guard and deterministic query metadata only; originals and approval unchanged.'}
    raw=C.encoded(document);digest=hashlib.sha256(raw.encode()).hexdigest();name='library-catalog-'+digest+'.json'
    updated={**index,'catalog':{'path':name,'sha256':digest,'bytes':len(raw.encode())}}
    C.atomic(public/name,raw);C.atomic(public/'library-catalog.json',raw)
    C.atomic(public/'library-index.json',C.encoded(updated));C.atomic(public/'library-index.js','window.INTERACTION_LIBRARY_INDEX='+C.encoded(updated)+';\n')
    result={'format':'interaction-twin-conservative-collection-metadata-refresh','version':1,'previous_catalog':previous,'catalog':C.ref(public/name,root),'unchanged_scene_index':scene_ref,'changes':proofs,'geometry_poses_programs_approvals_unchanged':True,'new_api_calls':0,'new_gpu_calls':0}
    with output.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    return result
