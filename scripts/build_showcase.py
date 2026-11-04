#!/usr/bin/env python3
"""Package the local research showcase from verified, existing media only."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
MEDIA=['trailer.mp4','poster.jpg','mesh-loop.mp4','mesh-loop.webm','assets-triptych.jpg',
       'source-jabra.jpg','mask-jabra.jpg','source-wallet.jpg','mesh-base.png','mesh-lid.png',
       'mesh-earbud.png','source-manifest.json','storyboard.jpg']
GRAPHICS=['pipeline.svg','pipeline.png','evidence-program.svg','evidence-program.png']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def build():
    out=ROOT/'showcase/media';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for folder,names in [('showcase-media',MEDIA),('showcase-graphics',GRAPHICS)]:
        for name in names:
            source=ROOT/'output'/folder/name
            if not source.is_file():raise FileNotFoundError(source)
            target=out/name
            if not target.exists() or sha(target)!=sha(source):shutil.copy2(source,target)
            rows.append({'source':str(source.relative_to(ROOT)),'path':str(target.relative_to(ROOT)),
                         'sha256':sha(target),'bytes':target.stat().st_size})
    # Keep full input provenance alongside the page; do not copy source video,
    # checkpoints or account material into the showcase media package.
    figure_manifest=ROOT/'output/showcase-graphics/manifest.json'
    shutil.copy2(figure_manifest,out/'graphics-provenance.json')
    report={'status':'local_showcase_ready','published':False,'files':rows,
            'sample_payload':'workspace/cases.js','source_videos_served_by_local_evidence_route':True,
            'new_inference_performed':False,'physical_accuracy_claimed':False}
    (ROOT/'showcase/build-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'media_files':len(rows),
                      'bytes':sum(r['bytes'] for r in rows),'published':False}))
    return report

if __name__=='__main__':build()
