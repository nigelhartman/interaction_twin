"""Source-driven deformable-articulation research, separate from rigid SAM twins.

The initial packet contains actual source pixels and unchanged native models.
No API call, hand-authored geometric point, pose path or production approval.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,argparse
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/experience-refinement-20261104-v1/wallet'
SCENE=ROOT/'data/rigidity-feasibility/overnight-camera-wallet-20261103-v1/wallet/scene.json'

def ref(path):
    path=Path(path);raw=path.read_bytes()
    return {'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
def save(path,value):
    text=json.dumps(value,indent=2,allow_nan=False)+'\n'
    with Path(path).open('x')as f:f.write(text)
def font(size):
    p=Path('/System/Library/Fonts/Supplemental/Arial.ttf')
    return ImageFont.truetype(str(p),size)if p.exists()else ImageFont.load_default()
def frames(program,indices,path,title):
    width=1600;columns=2;rows=(len(indices)+1)//2;cellh=570
    board=Image.new('RGB',(width,75+rows*cellh+65),'#f6f8fc');draw=ImageDraw.Draw(board)
    draw.text((25,20),title,font=font(27),fill='#172033');refs=[]
    for n,index in enumerate(indices):
        e=next(e for e in program['evidence']if e['source_index']==index)
        source=ROOT/e['frame_path'];assert ref(source)['sha256']==e['frame_sha256']
        rgb=Image.open(source).convert('RGB');mask=np.zeros((rgb.height,rgb.width),bool)
        for o in e['observations']:
            if o['track_id']not in ['wallet_holder','pouch_flap']:continue
            p=ROOT/o['mask_path'];assert ref(p)['sha256']==o['mask_sha256'];mask|=np.asarray(Image.open(p))>0
        y,x=np.nonzero(mask);pad=round(max(np.ptp(x),np.ptp(y))*.16)
        crop=[max(0,int(x.min())-pad),max(0,int(y.min())-pad),min(rgb.width,int(x.max())+1+pad),min(rgb.height,int(y.max())+1+pad)]
        image=rgb.crop(crop);image.thumbnail((750,490))
        left=(n%2)*800;top=75+(n//2)*cellh
        draw.text((left+24,top+8),f'Source {index:03d} · {e["timestamp_s"]:.3f}s',font=font(22),fill='#172033')
        board.paste(image,(left+(800-image.width)//2,top+55+(490-image.height)//2))
        refs.append({'source_index':index,'timestamp_s':e['timestamp_s'],'RGB':ref(source),'crop_xyxy':crop})
    draw.text((25,board.height-45),'Actual source pixels · crop from saved model masks · no generated appearance or geometry.',font=font(21),fill='#536079')
    board.save(path);return refs

def packet():
    OUT.mkdir(parents=True,exist_ok=False)
    scene=json.loads(SCENE.read_text());program_path=ROOT/scene['original_program']['path']
    assert ref(program_path)==scene['original_program'];program=json.loads(program_path.read_text())
    groups=[('01-source-unfolding.png',[1,6,10,14,21],'Recording · closed wallet opens into two leaves'),
      ('02-source-pouch-flap.png',[24,28,29,30],'Recording · coin pouch flap lifts and bends'),
      ('03-source-coin-removal.png',[52,54,58,67],'Recording · open wallet and coin removal'),
      ('04-source-closing.png',[83,88,94,99],'Recording · coin return, pouch flap and wallet closure')]
    source=[]
    for name,indices,title in groups:
        source.append({'image':ref(OUT/name)if (OUT/name).exists()else None,'frames':frames(program,indices,OUT/name,title)})
        source[-1]['image']=ref(OUT/name)
    import generation_visual_judge as V
    natives=[]
    for cid,name in [('wallet_holder','05-native-holder'),('pouch_flap','06-native-flap')]:
        component=next(c for c in scene['components']if c['id']==cid);path=ROOT/component['asset']['path'];assert ref(path)==component['asset']
        folder=OUT/name;folder.mkdir()
        result=V.render_sheet(component['asset'],folder)
        natives.append({'component_id':cid,'asset':component['asset'],'render':result,'image':result['contact_sheet_ref']})
    images=[s['image']for s in source]+[n['image']for n in natives]
    plan={'format':'interaction-twin-source-flexure-research-plan','version':1,
      'created_at_utc':datetime.now(timezone.utc).isoformat(),'implementation':ref(Path(__file__)),
      'baseline_scene':ref(SCENE),'source_program':ref(program_path),'source':scene['source'],
      'method_proposal':['Determine the native closed-shell intrinsic long/short/thin axes from actual geometry; determine visible open two-leaf boundaries from saved SAM masks and RGB line evidence.',
        'Prefer a source-driven two-leaf continuous-skin proxy with a finite-width flexure band and estimated virtual fold frame. The parameter describes flexure, not a verified rigid leather hinge.',
        'Fit only observed leaf silhouettes/RGB against source frames. Preserve missing observations as unknown and hold out selected source frames from fit.',
        'If the closed native SAM asset has no source-visible interior surfaces, use explicitly source-observed leaf/appearance patches with provenance, rather than relabel the closed body as an open reconstruction.',
        'Separate whole-wallet folding from the smaller coin-pouch flap bending, with independent source evidence and uncertainty. Keep coin motion and original native downloads intact.'],
      'no_authored_geometry_points_or_angles':True,'manual_pose_paths':False,
      'native_GLBS_or_program_modified':False,'new_API_or_GPU_calls':0,
      'physical_accuracy_verified':False,'automatic_publication':False}
    save(OUT/'method-proposal.json',plan)
    receipt={'format':'interaction-twin-wallet-six-image-Muse-review-packet','version':1,
      'created_at_utc':datetime.now(timezone.utc).isoformat(),'proposal':ref(OUT/'method-proposal.json'),
      'images':images,'source_boards':source,'native_boards':natives,
      'questions_for_Muse':['What visible folding/leaves and pouch-flap bending are supported by the recording, and which distinct motion rules should be represented?',
        'Does the original native holder show a closed wallet and lack the observed open interior? Identify qualitative failure modes without certifying3D measurements.',
        'Which source-visible boundaries/texture/occlusion cues can support automatic estimation of a virtual fold frame, finite-width flexure band and uncertain configuration state?',
        'Which geometry/appearance must remain explicitly source-observed or unknown rather than hallucinated? Recommend acceptance and failure conditions for a source-driven flexure proxy.'],
      'hosted_review_not_sent':True,'new_API_or_GPU_calls':0,'production_or_publication_changed':False,
      'physical_accuracy_verified':False}
    save(OUT/'six-image-packet.json',receipt)
    print(json.dumps({'packet':ref(OUT/'six-image-packet.json'),'images':images},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['packet']);args=parser.parse_args();packet()
