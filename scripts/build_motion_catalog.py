#!/usr/bin/env python3
"""Register explicit hash-bound Motion Studio index records; no inference."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from twin_catalog import TwinCatalog
from motion_catalog import MotionCatalog,INDEX,strict_json,require
from motion_lab import normalize


def build(root=ROOT,source_index='data/motion-lab-20261031/index.json',output=INDEX):
    root=Path(root).resolve();catalog=TwinCatalog(root);reader=MotionCatalog(root)
    target=root/output
    require(output.startswith('data/') and target.resolve()==target and not target.exists(),'Use a fresh explicit data registry')
    source=reader.path(source_index,2*1024*1024);index=strict_json(source.read_bytes())
    require(index.get('format')=='interval-motion-lab-index' and index.get('version')==1,'Unknown Motion Studio source index')
    rows=[];omitted=[]
    for item in index['runs']:
        normalized_path=reader.verified(item['viewer_url'].removeprefix('/files/'),item['viewer_sha256'],4*1024*1024)
        view=strict_json(normalized_path.read_bytes());raw_path=reader.verified(view['raw_url'].removeprefix('/files/'),view['input_sha256'],4*1024*1024)
        raw=strict_json(raw_path.read_bytes())
        if raw.get('format')!='short-interval-motion':
            omitted.append({'viewer_url':item['viewer_url'],'reason':'Separate native sampling adapter is not enabled in this registry version.'});continue
        candidates=[e for e in catalog.entries() if e['interaction_path']==raw['interaction_path']]
        require(len(candidates)==1,'Motion requires one exact registered publication')
        twin_id=candidates[0]['id'];entry,program,digest=catalog.load(twin_id)
        feedback=view.get('capture_feedback',{}).get('url')
        normalized=normalize(root,raw_path,feedback)
        for key in ['format','version','case_id','component_id','label','status','raw_url','input_sha256','source','asset','coordinate_convention','episodes','baseline','comparison_counts','interaction_path','interaction_sha256','baseline_reassessment_policy','capture_feedback']:
            require(view.get(key)==normalized.get(key),'Frozen viewer differs from current strict source/pose adapter: '+key)
        experiment_id=raw_path.parent.name
        row={'experiment_id':experiment_id,'twin_id':twin_id,'interaction_path':entry['interaction_path'],'interaction_sha256':digest,
            'component_id':raw['component_id'],'kind':'short_interval_original_samples_v1',
            'raw':{'path':raw_path.relative_to(root).as_posix(),'sha256':view['input_sha256']},
            'normalized':{'path':normalized_path.relative_to(root).as_posix(),'sha256':item['viewer_sha256']}}
        rows.append(row)
    require(rows and len(rows)<=64,'No bounded original-source motion records')
    target.parent.mkdir(parents=True,exist_ok=True)
    snapshot=target.parent/'motion-lab-at-publication.py';require(not snapshot.exists(),'Preserve existing registry snapshots')
    shutil.copyfile(root/'motion_lab.py',snapshot)
    doc={'format':'interaction-motion-catalog','version':1,'experiments':rows,
        'publication_source_index':{'path':source_index,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()},
        'strict_build_adapter':{'path':snapshot.relative_to(root).as_posix(),'sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest()},
        'omitted':omitted,'policy':reader.policy(),'new_model_calls':0,'api_calls':0}
    with target.open('x') as stream:json.dump(doc,stream,indent=2,allow_nan=False);stream.write('\n')
    final=MotionCatalog(root,output)
    for row in rows:
        _,program,digest=catalog.load(row['twin_id'])
        final.load(row['experiment_id'],interaction_path=row['interaction_path'],document=program,program_sha256=digest)
    return {'path':output,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'experiments':len(rows),'omitted':omitted}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-index',default='data/motion-lab-20261031/index.json');parser.add_argument('--output',default=INDEX);args=parser.parse_args()
    print(json.dumps(build(source_index=args.source_index,output=args.output)),flush=True)
