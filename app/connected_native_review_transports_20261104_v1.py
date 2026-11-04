"""Build four exact saved-native display transports; no inference or promotion."""
from datetime import datetime,timezone
import json
from pathlib import Path

import sam3d_uploads
import sam3d_review_catalog

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'output/connected-library-20261104-v1/native-displays'
JOBS=[
    ('pouch_flap','f258e82403de4102f7524e783fb8c8752e3b57c6ecb67cf483b08b611a30acbd'),
    ('camera','ae496b69bee0a25afa39ecf4039a870b2277496af373a0d7bfdabdf5e0d68e27'),
    ('door','b14488cd91ba211a04448b9b0a32a88d00df37ac02e1fbe27cb5f36c68a9dcdf'),
    ('batteries','05308a1721dc7c1eecd2baee1daee27595a0cda10080d9d00115afb9fbe21c4b'),
]


def ref(path):
    import hashlib
    return {'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}


def write(path,value):
    with path.open('x')as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def build():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    metadata=[{'role':role,'metadata':ref(ROOT/f'data/sam3d-upload-jobs/{job}/candidate/metadata.json')}for role,job in JOBS]
    write(OUTPUT/'plan.json',{'format':'interaction-twin-native-display-transport-plan','version':1,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'implementation':ref(Path(__file__)),
        'inputs':metadata,'executor':'sam3d_uploads.native_review','new_inference_or_API_calls':0,
        'allowed_new_transport_root':'data/sam3d-upload-reviews','catalog_or_approvals_modified':False,
        'caps_or_lineage_checks_modified':False})
    results=[]
    for item in metadata:
        path=ROOT/item['metadata']['path']
        if ref(path)!=item['metadata']:raise ValueError('Exact metadata input changed')
        try:
            result=sam3d_uploads.native_review(path,root=ROOT)
            results.append({**item,'status':'verified','result':result})
            print('Verified native display',item['role'],result['review']['path'],flush=True)
        except (ValueError,OSError,KeyError,TypeError)as error:
            results.append({**item,'status':'failed','error_type':type(error).__name__,'reason':str(error)[:1000]})
            print('Native display failed',item['role'],type(error).__name__,flush=True)
        write(OUTPUT/f'{item["role"]}-receipt.json',results[-1])
    discovered=sam3d_review_catalog.discover_reviews(ROOT)
    write(OUTPUT/'discovered-reviews.json',{'format':'interaction-twin-verified-native-display-discovery','version':1,
        'reviews':discovered,'count':len(discovered),'new_API_or_GPU_calls':0})
    record={'format':'interaction-twin-native-display-transport-completion','version':1,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'plan':ref(OUTPUT/'plan.json'),
        'results':results,'verified_count':sum(r['status']=='verified'for r in results),
        'discovered_count':len(discovered),'discovered':ref(OUTPUT/'discovered-reviews.json'),
        'new_models_API_GPU_calls':0,'catalog_or_approvals_modified':False,
        'caps_or_lineage_checks_changed':False,'publication_changed':False}
    write(OUTPUT/'complete.json',record)
    return {'complete':ref(OUTPUT/'complete.json'),'verified_count':record['verified_count'],'discovered_count':record['discovered_count']}


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
