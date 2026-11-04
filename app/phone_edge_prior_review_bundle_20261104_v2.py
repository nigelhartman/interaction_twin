"""Additive private three-way phone review bundle; never start its server."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import phone_source_regeneration_20261104_v1 as P

ROOT=P.ROOT
BASE=P.OUTPUT/'private-full-review-v1-1'
REPORT=ROOT/'output/submission-polish-20261104-v1/phone-edge-prior-v4'
SCENE=ROOT/'data/geometry-quality-20261104/phone-multiview-edge-prior-v4/private-scene'
OUTPUT=REPORT/'private-v2'


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Versioned private UI patch context differs')
    return text.replace(old,new,1)


def build():
    prior=json.loads((BASE/'manifest.json').read_text())
    for row in prior['files']:
        source=BASE/row['local'];assert source.stat().st_size==row['bytes']and hashlib.sha256(source.read_bytes()).hexdigest()==row['sha256']
    if OUTPUT.exists():raise FileExistsError('Preserve earlier review output')
    shutil.copytree(BASE,OUTPUT)
    additions={
        'data/edge-private-scene.json':SCENE/'scene.json',
        'data/edge-private-declaration.json':SCENE/'declared-source-reconstruction.json',
        'data/edge-private-quality-audit.json':SCENE/'source-quality-audit.json',
        'data/edge-private-report.json':SCENE/'report.json',
        'data/edge-prior-derivation.json':SCENE.parent/'derivation.json',
        'data/edge-fresh-tests.json':REPORT/'fresh-tests-v1/report.json',
        'data/edge-fresh-comparison.png':REPORT/'fresh-tests-v1/comparison.png',
        'data/edge-development-tests.json':REPORT/'development-results.json',
        'data/edge-connector-projection.json':REPORT/'connector-projection-diagnostic.json',
        'data/edge-source-training-comparison.png':REPORT/'source-training-comparison.png',
        'assets/edge-phone.glb':SCENE.parent/'asset.glb',
    }
    for local,source in additions.items():
        target=OUTPUT/local;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    html=(OUTPUT/'index.html').read_text()
    html=once(html,'<h1>One phone. A cleaner shape.</h1>','<h1>Three hypotheses. One source video.</h1>')
    html=once(html,'A source-derived 2.5D phone, built automatically from existing video and SAM masks.',
        'Compare the preserved SAM 3D phone, a source prism, and a thinner automatic edge prior. This is a private research review.')
    html=once(html,'<section class="comparison">',
        '<nav class="model-switch" aria-label="Phone reconstruction hypothesis"><a data-model="original" href="?model=original">01 Original SAM 3D</a><a data-model="source" href="?model=source">02 Source prism</a><a data-model="edge" href="?model=edge">03 Edge prior · experimental</a></nav><section class="comparison">')
    html=once(html,'<a href="data/descriptive-sidecar.json" download>','<a id="description-download" href="data/edge-private-declaration.json" download>')
    html=once(html,'<a href="assets/source-phone.glb" download>Download source-derived GLB ↓</a>',
        '<a id="model-download" href="assets/edge-phone.glb" download>Download displayed GLB ↓</a>')
    html=once(html,'<a href="data/new-test-report.json">New frame tests ↗</a>',
        '<a id="test-report-link" href="data/edge-fresh-tests.json">Fresh frame tests ↗</a>')
    html=once(html,'<a href="data/candidate-scene.json">Full scene JSON</a>',
        '<a id="scene-download" href="data/edge-private-scene.json">Displayed scene JSON</a>')
    (OUTPUT/'index.html').write_text(html)
    css=(OUTPUT/'review.css').read_text()+'\n.model-switch{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px}.model-switch a{font-size:12px;border:1px solid #ccd5e8;background:white;border-radius:7px;padding:8px 12px}.model-switch a[aria-current=page]{background:#315eea;border-color:#315eea;color:white}\n'
    (OUTPUT/'review.css').write_text(css)
    js=(OUTPUT/'review.js').read_text()
    js=once(js,"const originalMode=new URLSearchParams(location.search).get('model')==='original';",
        "const selectedMode=new URLSearchParams(location.search).get('model')||'edge';if(!['original','source','edge'].includes(selectedMode))throw Error('Unknown private comparison mode.');const originalMode=selectedMode==='original',edgeMode=selectedMode==='edge';")
    js=once(js,"const [candidate,baseline]=await Promise.all([json('data/candidate-scene.json'),json('data/baseline-scene.json')]);",
        "const [candidate,baseline,edge]=await Promise.all([json('data/candidate-scene.json'),json('data/baseline-scene.json'),json('data/edge-private-scene.json')]);")
    js=once(js,"const manifestScene=originalMode?baseline:candidate;", "const manifestScene=originalMode?baseline:edgeMode?edge:candidate;")
    js=once(js,"if(originalMode){$('model-label').textContent='Original SAM 3D · unchanged';$('compare-original').textContent='View source-derived repair ↗';$('compare-original').href='./';}",
        "$('model-label').textContent=originalMode?'Original SAM 3D · unchanged':edgeMode?'Edge prior · experimental 2.5D':'Source prism · 2.5D';$('compare-original').textContent=originalMode?'Compare source prism ↗':'Compare original SAM 3D ↗';$('compare-original').href=originalMode?'?model=source':'?model=original';document.querySelectorAll('[data-model]').forEach(a=>{if(a.dataset.model===selectedMode)a.setAttribute('aria-current','page');});$('description-download').href=edgeMode?'data/edge-private-declaration.json':'data/descriptive-sidecar.json';$('test-report-link').href=edgeMode?'data/edge-fresh-tests.json':'data/new-test-report.json';$('scene-download').href=originalMode?'data/baseline-scene.json':edgeMode?'data/edge-private-scene.json':'data/candidate-scene.json';const chosenPhone=manifestScene.components.find(c=>c.id==='phone_blue');$('model-download').href=manifest.asset_mapping.find(r=>r.source_path===chosenPhone.asset.path).local;if(edgeMode){$('validation').textContent='The edge prior passes one of two fresh mask-withheld frame tests and two of four earlier development checks. The failed fresh test stays failed. This candidate is private and is not eligible as the public default.';$('unknown-note').textContent='Original unknowns stay unknown. Source 39 and any shape/appearance regressions are withheld. The failed fresh clock adds an unknown motion barrier. Phone/connector scale, port contact, grasping and insertion remain unverified.';}")
    js=once(js,"$('coverage').textContent=originalMode?'Original phone:38/47 saved estimates':'Repair phone:37/47 saved estimates';",
        "$('coverage').textContent=(originalMode?'Original phone:':edgeMode?'Edge-prior phone:':'Source-prism phone:')+manifestScene.components.find(c=>c.id==='phone_blue').poses.filter(p=>Number.isSafeInteger(p.source_sample_index)&&p.status==='estimated').length+'/47 saved estimates';")
    js=once(js,"const samples=candidate.components.find(c=>c.id==='phone_blue').poses.filter(r=>Number.isSafeInteger(r.source_sample_index));",
        "const samples=manifestScene.components.find(c=>c.id==='phone_blue').poses.filter(r=>Number.isSafeInteger(r.source_sample_index));")
    js=once(js,"mode:originalMode?'original':'source_prism'", "mode:selectedMode")
    (OUTPUT/'review.js').write_text(js)
    server=once((OUTPUT/'serve-review.py').read_text(),'default=8872','default=8873')
    (OUTPUT/'serve-review.py').write_text(server);compile(server,str(OUTPUT/'serve-review.py'),'exec')
    (OUTPUT/'README.txt').write_text('PRIVATE THREE-WAY PHONE REVIEW\n\nRun: python3 serve-review.py --port 8873\nOpen: http://127.0.0.1:8873/?model=edge\nOther hypotheses: ?model=source or ?model=original\n\nOnly127.0.0.1 is bound, with the exact copied file allowlist. No server was started by this builder. The prior review8872 and all previous bundles remain untouched.\n\nIMPORTANT RESULTS\nThe edge prior is a new automatic source-derived2.5D hypothesis, not a new SAM3D generation or a native-SAM affine/subset derivative. Its shape uses source masks/blue cross-sections and an unverified thickness prior. It preserves one watertight body and actual source RGB. Camera bump, USB-C recess/contact frame, metric dimensions and material physics are absent.\n\nIt passes2/4 development mask gates after adaptation, and1/2 newly sampled mask-withheld tests. The fresh failure remains failed; a new unknown playback barrier preserves it. Source39 and any per-source projection/appearance regressions are withheld. This is private comparison only, not eligible as public/default reconstruction.\n\nThe source-clock slider steps in1microsecond. The exact source-sample chooser, previous/next, source-camera and orbit controls share the video time. There is no time snapping or hold extension. All three downloadable GLBs and the exact original video are kept unchanged.\n\nThe user recording is for local private review. No external distribution permission is inferred. No network API, GPU job, upload or inference is triggered by this page.\n')
    mapping=[*prior['asset_mapping'],{'source_path':P.ref(SCENE.parent/'asset.glb')['path'],
        'local':'assets/edge-phone.glb','sha256':P.ref(SCENE.parent/'asset.glb')['sha256']}]
    originals={row['local']:row for row in prior['files']};files=[]
    changed={'index.html','review.css','review.js','serve-review.py','README.txt'}
    for path in sorted(OUTPUT.rglob('*')):
        if not path.is_file()or path.name=='manifest.json':continue
        local=path.relative_to(OUTPUT).as_posix();r=P.ref(path)
        row={'local':local,'sha256':r['sha256'],'bytes':r['bytes']}
        if local in additions:row['original']=P.ref(additions[local])
        elif local in originals and local not in changed and 'original'in originals[local]:row['original']=originals[local]['original']
        files.append(row)
    manifest={**prior,'created_at_utc':datetime.now(timezone.utc).isoformat(),'ui_version':'private-v2-three-way',
        'implementation':P.ref(Path(__file__)),'files':files,'asset_mapping':mapping,
        'previous_bundle_manifest':P.ref(BASE/'manifest.json'),'edge_scene':P.ref(SCENE/'scene.json'),
        'private_declaration':P.ref(SCENE/'declared-source-reconstruction.json'),'public_default_eligible':False}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=REPORT/'Interaction-Twin-private-phone-three-way-v2.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3)as z:
        for path in sorted(OUTPUT.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(OUTPUT))
    report={'format':'interaction-twin-private-three-way-phone-review-build','version':2,
        'bundle_manifest':P.ref(OUTPUT/'manifest.json'),'archive':P.ref(archive),
        'private_scene':P.ref(SCENE/'scene.json'),'private_declaration':P.ref(SCENE/'declared-source-reconstruction.json'),
        'file_count':len(files)+1,'server_started':False,'previous_bundles_or_servers_modified':False,
        'public_catalog_assets_or_default_modified':False,'new_API_or_GPU_calls':0,
        'runner_command':['python3',str(OUTPUT/'serve-review.py'),'--port','8873']}
    P.save(REPORT/'private-v2-build.json',report)
    return report


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
