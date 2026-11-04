"""Admit exact display transports for already approved native component bytes.

This cannot change a selected program, scene, native geometry, pose or approval.
It verifies transport/import/metadata/source identity, then publishes only that
bounded presentation change. Existing expensive geometry audits remain intact.
"""
import copy
import hashlib
import json
from pathlib import Path

import curated_publication as C
from curated_twin_catalog import CuratedTwinCatalog
import object_program as O
import sam3d_review_catalog as R


def refresh(*,root=C.ROOT,output):
    root=Path(root).resolve();output=Path(output)
    if output.exists():raise ValueError('Display refresh already recorded')
    public=root/'showcase/public'
    snapshot=CuratedTwinCatalog(root);records,cases=snapshot._snapshot()
    index=C.read(public/'library-index.json');previous=C.ref(public/index['catalog']['path'],root)
    document=C.read(root/previous['path']);old=copy.deepcopy(document)
    scenes=C.read(public/'aligned-scenes.json')['scenes']
    old_scenes=C.ref(public/'aligned-scenes.json',root)
    admitted=snapshot._assets();discovered=R.discover_reviews(root)
    proofs=[]
    for record in document['records']:
        case=next(c for c in cases if c['id']==record['id'])
        if (record['scene_case']['original_sha256']!=record['interaction']['sha256'] or
                case['original_sha256']!=record['interaction']['sha256'] or
                case['interaction_path']!=record['interaction']['path'] or
                record['scene_case'].get('scene_ref')!=case.get('scene_ref')):
            raise ValueError('Public scene identity does not bind the exact approved procedure')
        record['scene_case']['id']=record['id']
        aligned=next(s for s in scenes if s['source_sha256']==record['source']['sha256'])
        companion=O.read_json(O.checked_ref(aligned['object_program'],root));O.validate(companion,root=root)
        if companion['evidence_program']!=record['interaction']:raise ValueError('Companion revision differs')
        source=[{'component_id':c['id'],'native_glb':{k:c['asset'][k]for k in ['path','sha256','bytes']},
                 'native_metadata':c['asset']['metadata']}for c in record['components']if c.get('asset')and c['asset'].get('metadata')]
        ids={b['component_id']for b in companion['component_assets']}
        bindings=companion['component_assets']+[b for b in source if b['component_id']not in ids]
        reviews=C.approved_reviews(root,record.get('sam3d_reviews',[])+discovered,bindings)
        reviews=R.match_reviews(reviews,record)
        # No missing transport may be papered over by a matching label or a
        # same-byte model imported from a different source anchor.
        for binding in bindings:
            if not any(r['component_id']==binding['component_id']and
                       r['source_sha256']==record['source']['sha256']and
                       r['native_glb']['sha256']==binding['native_glb']['sha256']and
                       r['native_glb']['bytes']==binding['native_glb']['bytes']for r in reviews):
                raise ValueError('Missing exact native display: '+binding['component_id'])
        for review in reviews:
            for field in ['review_page','source_image','source_mask','native_glb','review_manifest']:
                pointer=review[field];actual=C.ref(C.local(root,pointer['path']),root)
                if actual!=pointer:raise ValueError('Native display changed during admission')
                if pointer['path']in admitted and admitted[pointer['path']]!=pointer:
                    raise ValueError('Existing approved artifact binding changed')
                admitted[pointer['path']]=pointer
        proofs.append({'id':record['id'],'old_displays':len(record.get('sam3d_reviews',[])),
                       'new_displays':len(reviews),'native_metadata_bound_components':[b['component_id']for b in bindings]})
        record['sam3d_reviews']=reviews
    def stripped(doc):
        rows=copy.deepcopy(doc['records'])
        for record in rows:
            record.pop('sam3d_reviews',None);record['scene_case']['id']=record['id']
        return rows
    if stripped(document)!=stripped(old):raise ValueError('Display refresh attempted a semantic or model change')
    if C.ref(public/'aligned-scenes.json',root)!=old_scenes:raise ValueError('Connected scene changed during refresh')
    document['provenance']['native_display_refresh']={'version':2,'previous_catalog':previous,
        'implementation_sha256':C.sha(__file__),'scope':'Exact approved native bytes/source-anchor display and stable public scene IDs only; geometry, poses, procedure hashes and approval unchanged.'}
    raw=C.encoded(document);digest=hashlib.sha256(raw.encode()).hexdigest()
    name='library-catalog-'+digest+'.json'
    pointer={'path':name,'sha256':digest,'bytes':len(raw.encode())}
    public_index={**index,'catalog':pointer}
    C.atomic(public/'public-assets.json',C.encoded({'format':'interaction-twin-public-assets','version':1,'assets':sorted(admitted.values(),key=lambda r:r['path'])}))
    C.atomic(public/name,raw);C.atomic(public/'library-catalog.json',raw)
    C.atomic(public/'library-index.json',C.encoded(public_index))
    C.atomic(public/'library-index.js','window.INTERACTION_LIBRARY_INDEX='+C.encoded(public_index)+';\n')
    result={'format':'interaction-twin-exact-native-display-refresh','version':1,'previous_catalog':previous,
        'catalog':C.ref(public/name,root),'unchanged_scenes':old_scenes,'records':proofs,
        'api_calls':0,'gpu_calls':0,'geometry_or_poses_changed':False,'program_or_approval_changed':False,
        'admitted_assets':len(admitted),'native_displays':sum(p['new_displays']for p in proofs)}
    with output.open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    return result


if __name__=='__main__':
    print(json.dumps(refresh(output=C.ROOT/'output/connected-library-20261104-v1/native-display-refresh.json'),indent=2))
