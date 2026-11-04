"""Publish only the bounded physics experiment and a self-contained CPU starter."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from robotics_physics_bench_v1 import FORMAT, OUT, sha, ref, write

def build():
    source=OUT/'replay-native-v3.json'; replay=json.loads(source.read_text())
    if replay['format']!=FORMAT or replay['claims']['real_physics_engine_execution'] is not True:
        raise ValueError('An actual engine result is required')
    e=replay['execution']
    if not e['finite_output'] or e['contact_steps']<2000 or e['minimum_collision_vertex_height_m']<-.0005 or e['maximum_final_linear_speed_m_s']>.01:
        raise ValueError('Physics experiment did not pass bounded contact/settling checks')
    public=ROOT/'showcase/public/physics/newton-contact-20261104-v1.json'
    public.parent.mkdir(parents=True,exist_ok=True)
    public.write_bytes(source.read_bytes())
    (ROOT/'showcase/physics-bench-source.js').write_text('globalThis.PhysicsBenchSource='+json.dumps({'version':1,'replay':ref(public)},separators=(',',':'))+';\n')
    payloads={
        'run_experiment.py':(ROOT/'robotics_physics_bench_v1.py').read_bytes(),
        'input.json':(OUT/'input-native-v2.json').read_bytes(),
        'reference-replay.json':source.read_bytes(),
        'requirements.txt':b'newton==1.7.0\nwarp-lang==1.17.0\nnumpy==2.2.6\n',
        'README.txt':('INTERACTION TWIN / NEWTON CPU CONTACT STARTER\n\n'
            'Three generated components from the curated Interaction Twin library.\n'
            'Actual Newton 1.7 / Warp 1.17 CPU release + contact test.\n\n'
            'RUN (Python 3.12 on macOS Apple Silicon, Linux or Windows):\n'
            '  python3.12 -m venv .venv\n'
            '  source .venv/bin/activate\n'
            '  pip install -r requirements.txt\n'
            '  python run_experiment.py run --input input.json --output replay.json\n\n'
            'The engine runner is self contained; it reads collision hulls from input.json.\n'
            'native/ holds the unchanged generated appearance assets for a renderer.\n'
            'manifest.json pins every file, including original native source paths/hashes.\n\n'
            'ASSUMPTIONS / LIMITS\n'
            'Sizes, density, friction, restitution and release pose are authored assumptions.\n'
            'Metric scale was not measured. Convex proxies fill concavities.\n'
            'No robot grasp, insertion, learned policy or physical accuracy is validated.\n'
            'This is not Isaac Sim and does not certify the twins simulation ready.\n\n'
            'PRIMARY DOCUMENTATION\n'
            'https://developer.nvidia.com/newton-physics\n'
            'https://newton-physics.github.io/newton/1.7.0/guide/installation.html\n'
            'https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html\n').encode()}
    sources=[]
    for o in replay['objects']:
        r=o['asset_ref'];path=ROOT/r['path']
        if sha(path)!=r['sha256'] or path.stat().st_size!=r['bytes']:raise ValueError('Native source changed')
        name='native/'+o['id']+'.glb';payloads[name]=path.read_bytes()
        sources.append({'file':name,'asset_ref':r,'source_program_ref':o['source_program_ref'],'twin_id':o['twin_id']})
    manifest={'format':'interaction-twin-newton-cpu-starter','version':1,
        'source_components':sources,'claims':replay['claims'],
        'files':[{'path':p,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}for p,b in sorted(payloads.items())]}
    payloads['manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    archive=ROOT/'showcase/media/interaction-twin-newton-starter-20261104-v1.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6)as z:
        for p,b in sorted(payloads.items()):z.writestr('Interaction-Twin-Newton/'+p,b)
    if archive.stat().st_size>32*1024*1024:raise ValueError('Starter exceeds exact download guard')
    with zipfile.ZipFile(archive)as z:
        for row in manifest['files']:
            b=z.read('Interaction-Twin-Newton/'+row['path'])
            if hashlib.sha256(b).hexdigest()!=row['sha256']or len(b)!=row['bytes']:raise ValueError('Starter ZIP hash mismatch')
    receipt={'replay':ref(public),'source_script':ref(ROOT/'robotics_physics_bench_v1.py'),
        'starter_zip':ref(archive),'file_count':len(payloads),'native_sources':sources,
        'execution':e,'api_calls':0,'gpu_calls':0,'official_sources':[
            'https://developer.nvidia.com/newton-physics',
            'https://newton-physics.github.io/newton/1.7.0/guide/installation.html',
            'https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html']}
    write(OUT/'publication.json',receipt)
    print(json.dumps(receipt))

if __name__=='__main__':build()
