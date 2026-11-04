"""Additive private viewer precision repair; geometry/evidence remain exact.

Preserves the original v1 bundle and archive. No timeline snapping, pose hold,
interpolation tolerance, server start, publication, or model operation occurs.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import phone_source_regeneration_20261104_v1 as P

ROOT=P.ROOT
BASE=P.OUTPUT/'private-full-review-v1'
OUTPUT=P.OUTPUT/'private-full-review-v1-1'


def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('Exact versioned UI patch context differs')
    return text.replace(old,new,1)


def build():
    prior=json.loads((BASE/'manifest.json').read_text())
    for item in prior['files']:
        path=BASE/item['local']
        if path.stat().st_size!=item['bytes']or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Original v1 review bundle differs')
    if OUTPUT.exists():raise FileExistsError('Preserve earlier precision review output')
    shutil.copytree(BASE,OUTPUT)
    html=(OUTPUT/'index.html').read_text()
    html=replace_once(html,'step="0.001"','step="0.000001"')
    html=replace_once(html,'<button id="previous">← Sample</button><button id="next">Sample →</button>',
        '<button id="previous">← Sample</button><button id="next">Sample →</button>'
        '<label class="sample-picker">Source sample <select id="sample-picker" aria-label="Jump to an exact saved source sample"><option value="">Between samples</option></select></label>')
    (OUTPUT/'index.html').write_text(html)
    css=(OUTPUT/'review.css').read_text()
    css+='\n.sample-picker{display:flex;align-items:center;gap:7px;font-size:11px;color:#62718a}.sample-picker select{max-width:235px;border:1px solid #ccd5e8;border-radius:7px;background:white;color:#24314c;padding:8px;font:inherit}@media(max-width:750px){.sample-picker{flex-basis:100%}.sample-picker select{flex:1;max-width:none}}\n'
    (OUTPUT/'review.css').write_text(css)
    js=(OUTPUT/'review.js').read_text()
    js=replace_once(js,"$('timeline').value=String(time);",
        "$('timeline').value=String(time);$('clock').title='Exact source clock: '+time.toFixed(6)+' s';"
        "const exactSample=manifestScene.components.find(c=>c.id==='phone_blue').poses.find(p=>Number.isSafeInteger(p.source_sample_index)&&Math.abs(p.timestamp_s-time)<=1e-6);"
        "$('sample-picker').value=exactSample?String(exactSample.source_sample_index):'';")
    js=replace_once(js,"const samples=candidate.components.find(c=>c.id==='phone_blue').poses.filter(r=>Number.isSafeInteger(r.source_sample_index));",
        "const samples=candidate.components.find(c=>c.id==='phone_blue').poses.filter(r=>Number.isSafeInteger(r.source_sample_index));"
        "for(const sample of samples){const option=document.createElement('option');option.value=String(sample.source_sample_index);option.textContent='Sample '+sample.source_sample_index+' · '+sample.timestamp_s.toFixed(6)+' s · '+sample.status;$('sample-picker').append(option);}"
        "$('sample-picker').onchange=()=>{const sample=samples.find(p=>String(p.source_sample_index)===$('sample-picker').value);if(sample)seek(sample.timestamp_s);};")
    (OUTPUT/'review.js').write_text(js)
    server=(OUTPUT/'serve-review.py').read_text()
    server=replace_once(server,'default=8871','default=8872')
    (OUTPUT/'serve-review.py').write_text(server)
    readme=(OUTPUT/'README.txt').read_text().replace('8871','8872')
    readme+='\nPRECISION UI V1.1\nThe slider steps in one microsecond, preserving the six-decimal source PTS. Previous/next controls and the source-sample chooser seek the exact saved timestamp. The visible clock uses three decimals; its title shows six. No input snapping, pose-hold extension, geometry, pose, evidence, or interpolation threshold was changed. Unknown neighbors and rotation barriers remain unknown.\n'
    (OUTPUT/'README.txt').write_text(readme)
    changed={'index.html','review.css','review.js','serve-review.py','README.txt'}
    originals={r['local']:r for r in prior['files']}
    for item in prior['files']:
        if item['local']not in changed:
            if (OUTPUT/item['local']).read_bytes()!=(BASE/item['local']).read_bytes():
                raise ValueError('A non-UI private asset changed')
    changes={'format':'interaction-twin-private-phone-review-precision-change','version':'1.1',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'implementation':P.ref(Path(__file__)),
        'previous_bundle_manifest':P.ref(BASE/'manifest.json'),
        'changed_files':sorted(changed),'range_step_seconds':.000001,
        'timeline_snap_or_pose_hold_extension_seconds':0,
        'source_sample_chooser_seeks_exact_saved_PTS':True,'geometry_poses_evidence_unchanged':True,
        'production_or_previous_bundle_changed':False,'server_started':False}
    (OUTPUT/'precision-change.json').write_text(json.dumps(changes,indent=2)+'\n')
    files=[]
    for path in sorted(OUTPUT.rglob('*')):
        if not path.is_file()or path.name=='manifest.json':continue
        local=path.relative_to(OUTPUT).as_posix();row=P.ref(path)
        files.append({'local':local,'sha256':row['sha256'],'bytes':row['bytes'],
            **({'original':originals[local]['original']}if local in originals and local not in changed and 'original'in originals[local]else{})})
    manifest={**prior,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'implementation':P.ref(Path(__file__)),'ui_version':'1.1','files':files,
        'previous_bundle_manifest':P.ref(BASE/'manifest.json'),'precision_change':P.ref(OUTPUT/'precision-change.json')}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    compile(server,str(OUTPUT/'serve-review.py'),'exec')
    archive=P.OUTPUT/'Interaction-Twin-private-phone-review-v1-1.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3)as z:
        for path in sorted(OUTPUT.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(OUTPUT))
    receipt={'format':'interaction-twin-private-phone-review-precision-build','version':'1.1',
        'previous_bundle_manifest':P.ref(BASE/'manifest.json'),'manifest':P.ref(OUTPUT/'manifest.json'),
        'archive':P.ref(archive),'precision_change':P.ref(OUTPUT/'precision-change.json'),
        'server_started':False,'geometry_poses_evidence_unchanged':True,
        'runner_command':['python3',str(OUTPUT/'serve-review.py'),'--port','8872'],
        'default_source_time_s':17.493789,'range_step_s':.000001,'pose_hold_or_time_snap_extension_s':0,
        'previous_bundle_and_server_untouched':True}
    P.save(P.OUTPUT/'private-full-review-v1-1-build.json',receipt)
    return receipt


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
