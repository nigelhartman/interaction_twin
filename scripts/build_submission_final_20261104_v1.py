#!/usr/bin/env python3
"""45 s real-output trailer + 135 s current app/implementation walkthrough.

Preserves the opening encoded video packets. No remote models or new geometry.
The review loop is explicitly illustrative, not a fabricated API recording.
"""
from __future__ import annotations
import hashlib, importlib.util, json, math, subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import build_flagship_trailer_20261104_v1 as F
A=F.A
OUT=ROOT/'output/submission-final-20261104-v1'
MEDIA=ROOT/'showcase/media/submission-final-20261104-v1'
OPENING=ROOT/'showcase/media/flagship-20261104-v1/cut-v2/interaction-twin-flagship-45s.mp4'
W,H,FPS,D=1920,1080,30,135
BEATS=[(0,18,'Explore the library'),(18,36,'A program behind the geometry'),(36,50,'Build. Review. You approve.'),(50,63,'The same library, for your agent'),(63,78,'One task. Two recordings.'),(78,105,'Rehearse the next move'),(105,125,'How it is implemented'),(125,135,'Inspectable today. Physical validation next.')]
CUES=[(.4,17.5,'Now explore the application. The public demo includes five owner-recorded examples. Expand a card to compare its video and connected replay, inspect individual parts, and download original models and descriptive files.'),
(18.3,35.5,'The object program keeps more than shape. It links parts, estimated joint frames, axes, angles, removable relationships and demonstrated steps to source evidence. Unknown physical properties remain explicit.'),
(36.3,49.6,'Muse proposes the twin. Separate review prompts and code checks critique it. Targeted repairs can run for two rounds. Then you approve. Saved feedback guides later runs.'),
(50.3,62.6,'Copy the connection guide into your agent. The local read-only M C P server exposes the same curated library: search, inspect descriptions, retrieve geometry, and follow source evidence.'),
(63.3,77.6,'This saved client made forty-two M C P calls across phone and charger recordings. It checked six native models and nine source images, then prepared a five-step task draft. A separate Muse critique flags unresolved compatibility.'),
(78.3,103.9,'The browser robot makes the handoff visible. This full rehearsal places the case, opens the connected lid, moves both earbuds onto their goals, then closes the case and withdraws. You can edit supported instructions locally and export the typed task. These are authored kinematics, not simulated grasp forces, trained control or physical robot execution.'),
(105.3,124.6,'Python handles processing and review. The object program connects our curated M C P catalog and Three J S viewers. Meta supplies the models. We built orchestration, source checks, approval, exports and task authoring. GitHub Pages serves saved results without API keys.'),
(125.3,134.7,'Next, validate physical scale and contact. Reuse the evidence. Build the next interaction. Interaction Twin.')]

def write(p,v):F.write(p,v)
def graphic(im,p):
    titles=['Build','Review & improve','You approve'];sub=['Muse + Meta perception','Separate prompts + code checks','Exact revision → library']
    active=min(2,int(p*3));d=ImageDraw.Draw(im)
    for i,x in enumerate([75,705,1335]):
        y=380;F.card(im,(x,y,x+510,y+260),fill='#eef2ff'if i==active else'white',line=F.BLUE if i==active else'#dfe3eb')
        F.label(im,(x+35,y+32),f'0{i+1}',26);F.text(im,(x+35,y+100),titles[i],37,F.INK,True);F.text(im,(x+35,y+183),sub[i],21,F.MUTED)
    for x in [617,1247]:F.text(im,(x,478),'→',49,F.BLUE,True)
    # A pulse indicates an illustrated repair path, never an actual API event.
    q=(p*2)%1;d.line([(960,672),(960,735),(330,735),(330,672)],fill='#9aade4',width=4)
    d.ellipse((960-round(q*630)-9,726,960-round(q*630)+9,744),fill=F.BLUE)
    F.label(im,(570,774),'Targeted repair · at most two rounds',24,F.MUTED)
    F.label(im,(75,935),'Illustrated workflow',21,F.MUTED)

def header(im,t,a,title):
    F.header(im,a.old,(t+45)/4);F.headline(im,title)
    F.label(im,(72,984),'WALKTHROUGH · '+f'{int((t+45)//60):02}:{int((t+45)%60):02}',19,F.MUTED)

def shot(im,name,p):
    image=Image.open(OUT/'captures'/name).convert('RGB')
    # Pixel-preserving screenshot with a small declared editorial push-in.
    width,height=image.size;inset=round(min(width,height)*.015*p)
    image=image.crop((inset,inset,width-inset,height-inset))
    F.card(im,(62,220,1858,951));F.fit(im,image,(78,235,1764,700))

def frame(t,a):
    title=next(title for s,e,title in BEATS if s<=t<e);im=Image.new('RGB',(W,H),F.BG);header(im,t,a,title);kind=''
    if t<18:
        shot(im,'static-library.png'if t<9 else'static-downloads.png',(t%9)/9);kind='actual_current_static_app_screenshot'
    elif t<36:
        F.fit(im,a.connected((t-18)/18),(50,245,870,680));F.card(im,(950,258,1858,911));
        lines=[('part','case_lid'),('parent','case_base'),('joint','revolute · estimated'),('axis','[-0.989, 0.017, -0.146]'),('observed range','0 … 2.862 rad · estimated'),('evidence','timestamped source frames'),('physical accuracy','unverified')]
        for i,(k,v)in enumerate(lines):
            F.text(im,(990,296+i*80),k,22,F.MUTED);F.text(im,(1230,296+i*80),v,23,F.BLUE if i==int((t-18)/18*7)else F.INK,True)
        F.label(im,(990,872),'Readable excerpt of saved fields',21,F.MUTED);kind='exact_saved_program_excerpt_with_rounded_axis'
    elif t<50:graphic(im,(t-36)/14);kind='illustrated_bounded_review_workflow'
    elif t<63:shot(im,'static-agents.png',(t-50)/13);kind='actual_current_agents_page'
    elif t<78:
        p=(t-63)/15;F.fit(im,a.models(p),(30,190,1860,665))
        for i,(v,k)in enumerate([('42','saved MCP calls'),('6','native models'),('9','source images'),('5','proposed steps')]):
            x=130+i*445;F.text(im,(x,842),v,62,F.BLUE,True);F.text(im,(x,920),k,25,F.MUTED)
        kind='actual_current_models_and_saved_MCP_trace'
    elif t<105:
        row=a.fresh['robot'];runtime=min(a.robot_receipt['task_duration_s'],t-78)
        pts=runtime/a.robot_receipt['task_duration_s']*row['encoded_last_timestamp_s']
        image=a.decode('extended-robot',F.checked(row['video']),min(row['encoded_last_timestamp_s']-.04,pts),(1280,800)).crop(F.ROBOT_CROP)
        F.fit(im,image,(40,200,1840,745));F.label(im,(75,942),'Full saved rehearsal · 1× task clock · authored kinematics',23,F.MUTED);kind='full_current_robot_task_1x_then_final_hold'
    elif t<125:
        boxes=[(80,'Process','Python · Muse / SAM / SAM 3D'),(710,'Describe','Parts · joints · steps · evidence'),(1340,'Reuse','Library · MCP · Three.js')]
        for i,(x,k,v)in enumerate(boxes):
            F.card(im,(x,300,x+500,650),fill='#eef2ff'if int((t-105)/20*3)==i else'white');F.label(im,(x+32,340),k,36);F.text(im,(x+32,432),v,22,F.MUTED)
            F.text(im,(x+32,518),['pipeline.py','object_program.py','twin_catalog.py'][i],25,F.INK,True,True)
        for x in [620,1250]:F.text(im,(x,437),'→',40,F.BLUE,True)
        F.text(im,(83,727),'Our contribution',31,F.INK,True);F.text(im,(83,790),'Orchestration · source checks · review · approval · exports · task authoring',29,F.MUTED)
        F.label(im,(83,910),'Meta models are upstream work. Public Pages runs no backend.',24,F.MUTED);kind='implementation_explanation'
    else:
        if t<130:
            for i,s in enumerate(['Inspect the source.','Reuse the object program.','Validate the physical task.']):F.text(im,(100,330+i*150),s,53,F.BLUE if i==int((t-125)/5*3)else F.INK,True)
        else:im=A.ending(min(1.99,(t-130)/5*2),a.old)
        kind='limits_and_brand'
    return im,{'film_s':t+45,'kind':kind,'model_call_performed':False,'geometry_modified':False}

def main():
    OUT.mkdir(exist_ok=True);MEDIA.mkdir(exist_ok=True);assert not (MEDIA/'interaction-twin-3min.mp4').exists();audio_dir=OUT/'audio-v4';audio_dir.mkdir(exist_ok=False)
    A.STORIES['submission-tail']={'duration_s':D,'cues':CUES,'voice':'Samantha','music':{'bpm':96,'kind':'documentary','seed':8414}}
    audio,cues=A.make_audio('submission-tail',audio_dir)
    # Retain natural speech speed; fail rather than accelerating a dense script.
    speech=json.loads((audio_dir/'audio.json').read_text());assert all(v['atempo']==1 for v in speech['voice'])
    a=F.Assets(ROOT/'output/flagship-trailer-20261104-v1/intake-v1/manifest.json')
    tail=OUT/'walkthrough-tail.mp4';clock=[];proofs=[]
    cmd=['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-i',str(audio),'-t',str(D),'-c:v','libx264','-threads','4','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(tail)]
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for n in range(D*FPS):
            image,trace=frame(n/FPS,a);clock.append(trace)
            if n in {0,270,600,1140,1410,1650,1980,2400,2700,3000,3240,3510,3780,3990}:
                p=OUT/f'frame-{n:04}.jpg';image.save(p,quality=95);proofs.append({'time_s':n/FPS+45,'file':F.ref(p)})
            process.stdin.write(image.tobytes())
        process.stdin.close();err=process.stderr.read();assert process.wait()==0,err.decode()
    finally:a.close()
    listing=OUT/'concat.txt';listing.write_text(''.join("file '"+str(p).replace("'","'\\''")+"'\n" for p in [OPENING,tail]))
    movie=MEDIA/'interaction-twin-3min.mp4'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-f','concat','-safe','0','-i',str(listing),'-t','180','-c','copy','-movflags','+faststart',str(movie)],check=True)
    probe=A.verify_video(movie,{'duration_s':180})
    first=(OPENING.parent/'captions.vtt').read_text();extension=''.join(f'{len(F.CUES)+i+1}\n{A.V.stamp(r["start_s"]+45)} --> {A.V.stamp(r["end_s"]+45)}\n{r["text"]}\n\n'for i,r in enumerate(cues));(MEDIA/'captions.vtt').write_text(first+'\n'+extension)
    (MEDIA/'transcript.txt').write_text((OPENING.parent/'transcript.txt').read_text()+'\n\n'+'\n\n'.join(r['text']for r in cues)+'\n')
    subprocess.run(['ffmpeg','-nostdin','-v','error','-ss','1','-i',str(movie),'-frames:v','1','-q:v','2',str(MEDIA/'poster.jpg')],check=True)
    A.contact_sheet(proofs,OUT/'contact-sheet.jpg');write(OUT/'clock.json',clock)
    write(MEDIA/'provenance.json',{'format':'interaction-twin-final-submission-film','version':1,'opening':F.ref(OPENING),'opening_encoded_video_packets_preserved':True,'duration_s':180,'movie':F.ref(movie),'captions':F.ref(MEDIA/'captions.vtt'),'probe':probe,'tail_beats':BEATS,'source_captures':a.fresh,'current_catalog':a.catalog_ref,'screenshots':[F.ref(p)for p in sorted((OUT/'captures').glob('*.png'))],'audio':F.ref(audio_dir/'audio.json'),'clock':F.ref(OUT/'clock.json'),'builder':F.ref(Path(__file__)),'new_api_calls':0,'new_gpu_calls':0,'workflow_animation':'illustration_not_live_API_trace','robot_scope':'authored_kinematics_not_physical_control','publication':False})
    print(json.dumps({'movie':F.ref(movie),'probe':probe},indent=2))

if __name__=='__main__':main()
