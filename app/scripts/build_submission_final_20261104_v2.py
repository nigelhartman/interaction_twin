#!/usr/bin/env python3
"""45 s real-output trailer + 135 s current app/implementation walkthrough.

Additive V2: animated review and model/code architecture illustrations. Copies
the exact 45-second opening video stream without re-encoding. No new geometry.
Illustrated activity is never represented as a live model/API execution trace.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import build_flagship_trailer_20261104_v1 as F
A=F.A
OUT=ROOT/'output/submission-final-20261104-v2'
MEDIA=ROOT/'showcase/media/submission-final-20261104-v2'
CAPTURES=ROOT/'output/submission-final-20261104-v1/captures'
OPENING=ROOT/'showcase/media/flagship-20261104-v1/cut-v2/interaction-twin-flagship-45s.mp4'
W,H,FPS,D=1920,1080,30,135
TOTAL=45+D
BEATS=[(0,18,'Explore the library'),(18,36,'A program behind the geometry'),(36,50,'Build. Review. You approve.'),(50,63,'The same library, for your agent'),(63,78,'One task. Two recordings.'),(78,105,'Rehearse the next move'),(105,125,'How Meta models become reusable twins'),(125,135,'Inspectable today. Physical validation next.')]
CUES=[(.4,17.5,'Now explore the application. The public demo includes five owner-recorded examples. Expand a card to compare its video and connected replay, inspect individual parts, and download original models and descriptive files.'),
(18.3,35.5,'The object program keeps more than shape. It links parts, estimated joint frames, axes, angles, removable relationships and demonstrated steps to source evidence. Unknown physical properties remain explicit.'),
(36.3,49.6,'Muse proposes parts and actions. Separate critic prompts use the same Muse model, alongside code checks. Up to two repair rounds can follow, then the owner approves. Feedback becomes guidance for later runs.'),
(50.3,62.6,'Copy the connection guide into your agent. The local read-only M C P server exposes the same curated library: search, inspect descriptions, retrieve geometry, and follow source evidence.'),
(63.3,77.6,'This saved client made forty-two M C P calls across phone and charger recordings. It checked six native models and nine source images, then prepared a five-step task draft. A separate Muse critique flags unresolved compatibility.'),
(78.3,103.9,'The browser robot makes the handoff visible. This full rehearsal places the case, opens the connected lid, moves both earbuds onto their goals, then closes the case and withdraws. You can edit supported instructions locally and export the typed task. These are authored kinematics, not simulated grasp forces, trained control or physical robot execution.'),
(105.3,124.6,'Meta Muse proposes parts and actions. SAM three point one segments them; SAM three D drafts meshes. Our Python code builds source-linked object programs and manages checks, review and approval. Read-only M C P and Three J S viewers make those programs reusable.'),
(125.3,134.7,'Next, validate physical scale and contact. Reuse the evidence. Build the next interaction. Interaction Twin.')]

def write(p,v):F.write(p,v)
REVIEW_SCHEDULE=((0,2.6,'build'),(2.6,4.6,'review'),(4.6,6.2,'repair'),(6.2,7.8,'build'),(7.8,10.2,'review'),(10.2,11.3,'approval_transfer'),(11.3,14,'approve'))
ARCHITECTURE_SCHEDULE=((0,3.5,0),(3.5,7,1),(7,10.5,2),(10.5,15,3),(15,20,4))
ARCHITECTURE_BOXES=((75,285,575,535),(710,285,1210,535),(1345,285,1845,535),(75,650,905,890),(1015,650,1845,890))
ARCHITECTURE_STAGES=('Muse','SAM 3.1','SAM 3D Objects','Object program + source checks','MCP + Three.js viewers')

def bounded(value):return max(0.,min(1.,float(value)))

def path_point(points,progress):
    """Constant-speed editorial pulse along a polyline; no model event clock."""
    lengths=[math.dist(a,b)for a,b in zip(points,points[1:])]
    remaining=bounded(progress)*sum(lengths)
    for a,b,length in zip(points,points[1:],lengths):
        if remaining<=length:
            q=remaining/length if length else 0
            return(a[0]+(b[0]-a[0])*q,a[1]+(b[1]-a[1])*q)
        remaining-=length
    return points[-1]

def flow(im,points,progress=None):
    d=ImageDraw.Draw(im);d.line(points,fill='#c5d0ed',width=4,joint='curve')
    last,before=points[-1],points[-2];angle=math.atan2(last[1]-before[1],last[0]-before[0]);size=11
    d.polygon([last,(last[0]-size*math.cos(angle-.5),last[1]-size*math.sin(angle-.5)),(last[0]-size*math.cos(angle+.5),last[1]-size*math.sin(angle+.5))],fill='#9cafe0')
    if progress is not None:
        x,y=path_point(points,progress);d.ellipse((x-15,y-15,x+15,y+15),fill='#e0e7ff');d.ellipse((x-7,y-7,x+7,y+7),fill=F.BLUE)

def review_state(p):
    seconds=bounded(p)*14
    start,end,phase=next((s,e,k)for s,e,k in REVIEW_SCHEDULE if s<=seconds<e)if seconds<14 else REVIEW_SCHEDULE[-1]
    return{'illustrated_phase':phase,'phase_progress':bounded((seconds-start)/(end-start)),'same_Muse_model_separate_prompts':True,'maximum_repair_rounds':2,'owner_approval_required':True,'live_model_events':False}

def graphic(im,p):
    state=review_state(p);phase=state['illustrated_phase'];q=state['phase_progress']
    titles=['Build','Review & improve','You approve']
    subs=[('Muse proposes parts + actions','A draft, linked to the video'),('Same Muse · separate critic prompts','Alongside deterministic code checks'),('Owner approves the exact revision','Then it enters the curated library')]
    active={'build':{0},'review':{1},'repair':{0,1},'approval_transfer':{2},'approve':{2}}[phase]
    for i,x in enumerate([75,705,1335]):
        y=365;F.card(im,(x,y,x+510,y+285),fill='#eef2ff'if i in active else'white',line=F.BLUE if i in active else'#dfe3eb')
        F.label(im,(x+32,y+30),f'0{i+1}',25);F.text(im,(x+32,y+91),titles[i],36,F.INK,True)
        F.text(im,(x+32,y+171),subs[i][0],21,F.MUTED);F.text(im,(x+32,y+211),subs[i][1],20,F.MUTED)
    forward=(min(1,q*2.6)if phase=='review'else None)
    flow(im,[(585,505),(705,505)],forward)
    flow(im,[(1215,505),(1335,505)],q if phase=='approval_transfer'else None)
    flow(im,[(960,650),(960,735),(330,735),(330,650)],q if phase=='repair'else None)
    F.label(im,(605,783),'Optional targeted repair · up to 2 rounds',25,F.MUTED)
    F.label(im,(75,935),'Illustrated workflow · feedback is retrievable guidance, not weight training',21,F.MUTED)
    return state

def architecture_state(p):
    seconds=bounded(p)*20
    start,end,stage=next((s,e,k)for s,e,k in ARCHITECTURE_SCHEDULE if s<=seconds<e)if seconds<20 else ARCHITECTURE_SCHEDULE[-1]
    return{'illustrated_stage':ARCHITECTURE_STAGES[stage],'stage_index':stage,'stage_progress':bounded((seconds-start)/(end-start)),'Meta_models_are_upstream':True,'MCP_reuse_is_separate_from_processing':True,'live_model_events':False}

def architecture(im,p):
    state=architecture_state(p);active=state['stage_index'];q=state['stage_progress']
    F.label(im,(75,238),'META MODELS',22,F.MUTED)
    rows=[('Muse','Propose parts + actions','Visual reasoning'),('SAM 3.1','Segment source pixels','Source-linked binary masks'),('SAM 3D Objects','Draft component geometry','Native model drafts'),('Object program','Source checks · review · owner approval','Parts · joints · actions · evidence'),('MCP + viewers','Search · inspect · export','Local MCP server · Three.js previews')]
    for i,(box,row)in enumerate(zip(ARCHITECTURE_BOXES,rows)):
        x,y,right,bottom=box;selected=i==active;F.card(im,box,fill='#eef2ff'if selected else'white',line=F.BLUE if selected else'#dfe3eb')
        F.label(im,(x+30,y+27),'META'if i<3 else'OUR IMPLEMENTATION',19,F.BLUE if selected else F.MUTED)
        F.text(im,(x+30,y+76),row[0],35,F.INK,True)
        F.text(im,(x+30,y+137),row[1],24,F.MUTED)
        F.text(im,(x+30,y+180),row[2],21,F.MUTED)
    paths=[[(575,410),(710,410)],[(1210,410),(1345,410)],[(1595,535),(1595,588),(490,588),(490,650)],[(905,770),(1015,770)]]
    for i,points in enumerate(paths,1):flow(im,points,min(1,q*2.2)if active==i else None)
    F.label(im,(75,935),'Architecture illustration · Pages shows saved results; MCP runs locally',21,F.MUTED)
    return state

def header(im,t,a,title):
    # The shared logo renderer expects a 45-second film clock. Map explicitly
    # from this 180-second film instead of hiding the ratio in a magic divisor.
    F.header(im,a.old,F.D*bounded((t+45)/TOTAL));F.headline(im,title)
    F.label(im,(72,984),'WALKTHROUGH · '+f'{int((t+45)//60):02}:{int((t+45)%60):02}',19,F.MUTED)

def shot(im,name,p):
    image=Image.open(CAPTURES/name).convert('RGB')
    # Pixel-preserving screenshot with a small declared editorial push-in.
    width,height=image.size;inset=round(min(width,height)*.015*p)
    image=image.crop((inset,inset,width-inset,height-inset))
    F.card(im,(62,220,1858,951));F.fit(im,image,(78,235,1764,700))

def frame(t,a):
    title=next(title for s,e,title in BEATS if s<=t<e);im=Image.new('RGB',(W,H),F.BG);header(im,t,a,title);kind='';illustration=None
    if t<18:
        shot(im,'static-library.png'if t<9 else'static-downloads.png',(t%9)/9);kind='actual_current_static_app_screenshot'
    elif t<36:
        F.fit(im,a.connected((t-18)/18),(50,245,870,680));F.card(im,(950,258,1858,911));
        lines=[('part','case_lid'),('parent','case_base'),('joint','revolute · estimated'),('axis','[-0.989, 0.017, -0.146]'),('observed range','0 … 2.862 rad · estimated'),('evidence','timestamped source frames'),('physical accuracy','unverified')]
        for i,(k,v)in enumerate(lines):
            F.text(im,(990,296+i*80),k,22,F.MUTED);F.text(im,(1230,296+i*80),v,23,F.BLUE if i==int((t-18)/18*7)else F.INK,True)
        F.label(im,(990,872),'Readable excerpt of saved fields',21,F.MUTED);kind='exact_saved_program_excerpt_with_rounded_axis'
    elif t<50:illustration=graphic(im,(t-36)/14);kind='illustrated_bounded_review_workflow'
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
        illustration=architecture(im,(t-105)/20);kind='illustrated_Meta_models_and_our_source_bound_program'
    else:
        if t<130:
            for i,s in enumerate(['Inspect the source.','Reuse the object program.','Validate the physical task.']):F.text(im,(100,330+i*150),s,53,F.BLUE if i==int((t-125)/5*3)else F.INK,True)
        else:im=A.ending(min(1.99,(t-130)/5*2),a.old)
        kind='limits_and_brand'
    return im,{'film_s':t+45,'kind':kind,'model_call_performed':False,'geometry_modified':False,**({'illustration':illustration}if illustration else{})}

def main():
    global CAPTURES
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--finalize-existing',action='store_true');parser.add_argument('--captures',type=Path,default=CAPTURES,help='Current app screenshot directory (defaults to the preserved V1 captures).');args=parser.parse_args()
    resume=args.finalize_existing;CAPTURES=args.captures.resolve();assert CAPTURES.is_relative_to(ROOT),'Captures must remain within this project.'
    assert all((CAPTURES/name).is_file()for name in ['static-library.png','static-downloads.png','static-agents.png']),'Expected actual app screenshots are missing.'
    OUT.mkdir(exist_ok=True);MEDIA.mkdir(exist_ok=True);assert not (MEDIA/'interaction-twin-3min.mp4').exists();audio_dir=OUT/'audio-v1'
    A.STORIES['submission-tail-v2']={'duration_s':D,'cues':CUES,'voice':'Samantha','music':{'bpm':96,'kind':'documentary','seed':8414}}
    if resume:audio=audio_dir/'mix.wav';cues=[{'start_s':s,'end_s':e,'text':v}for s,e,v in CUES]
    else:audio_dir.mkdir(exist_ok=False);audio,cues=A.make_audio('submission-tail-v2',audio_dir)
    # Retain natural speech speed; fail rather than accelerating a dense script.
    speech=json.loads((audio_dir/'audio.json').read_text());assert all(v['atempo']==1 for v in speech['voice'])
    a=F.Assets(ROOT/'output/flagship-trailer-20261104-v1/intake-v1/manifest.json')
    tail=OUT/'walkthrough-tail.mp4';clock=[];proofs=[]
    cmd=['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-i',str(audio),'-t',str(D),'-c:v','libx264','-threads','4','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(tail)]
    if not resume:
        process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            for n in range(D*FPS):
                image,trace=frame(n/FPS,a);clock.append(trace)
                if n in {0,270,600,1140,1260,1380,1410,1650,1980,2400,2700,3000,3150,3240,3330,3510,3690,3780,3990}:
                    p=OUT/f'frame-{n:04}.jpg';image.save(p,quality=95);proofs.append({'time_s':n/FPS+45,'file':F.ref(p)})
                process.stdin.write(image.tobytes())
            process.stdin.close();err=process.stderr.read();assert process.wait()==0,err.decode()
        finally:a.close()
    else:
        assert A.verify_video(tail,{'duration_s':135})
        proofs=[{'time_s':int(p.stem.split('-')[1])/FPS+45,'file':F.ref(p)}for p in sorted(OUT.glob('frame-*.jpg'))]
        clock=[{'film_s':n/FPS+45,'chapter':next(title for s,e,title in BEATS if s<=n/FPS<e),'model_call_performed':False,'geometry_modified':False}for n in range(D*FPS)];a.close()
    listing=OUT/'concat.txt';listing.write_text(''.join("file '"+str(p).replace("'","'\\''")+"'\n" for p in [OPENING,tail]))
    movie=MEDIA/'interaction-twin-3min.mp4'
    joined=OUT/'concat-mux-v2.mp4'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-f','concat','-safe','0','-i',str(listing),'-t','180','-c','copy',str(joined)],check=True)
    joined_probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(joined)]));offset=next(s for s in joined_probe['streams']if s['codec_type']=='video')['start_time']
    subprocess.run(['ffmpeg','-nostdin','-v','error','-itsoffset',str(-float(offset)),'-i',str(joined),'-map','0:v','-map','0:a','-c:v','copy','-af','asetpts=PTS-STARTPTS,apad=whole_dur=180,atrim=0:180','-c:a','aac','-b:a','192k','-t','180','-movflags','+faststart',str(movie)],check=True)
    probe=A.verify_video(movie,{'duration_s':180})
    first=(OPENING.parent/'captions.vtt').read_text();extension=''.join(f'{len(F.CUES)+i+1}\n{A.V.stamp(r["start_s"]+45)} --> {A.V.stamp(r["end_s"]+45)}\n{r["text"]}\n\n'for i,r in enumerate(cues));(MEDIA/'captions.vtt').write_text(first+'\n'+extension)
    (MEDIA/'transcript.txt').write_text((OPENING.parent/'transcript.txt').read_text()+'\n\n'+'\n\n'.join(r['text']for r in cues)+'\n')
    subprocess.run(['ffmpeg','-nostdin','-v','error','-ss','1','-i',str(movie),'-frames:v','1','-q:v','2',str(MEDIA/'poster.jpg')],check=True)
    A.contact_sheet(proofs,OUT/'contact-sheet.jpg');write(OUT/'clock.json',clock)
    write(MEDIA/'provenance.json',{'format':'interaction-twin-final-submission-film','version':2,'opening':F.ref(OPENING),'opening_video_stream_copied_without_reencoding':True,'duration_s':180,'movie':F.ref(movie),'captions':F.ref(MEDIA/'captions.vtt'),'probe':probe,'tail_beats':BEATS,'source_captures':a.fresh,'current_catalog':a.catalog_ref,'screenshots':[F.ref(p)for p in sorted(CAPTURES.glob('*.png'))],'audio':F.ref(audio_dir/'audio.json'),'clock':F.ref(OUT/'clock.json'),'builder':F.ref(Path(__file__)),'new_api_calls':0,'new_gpu_calls':0,'workflow_animation':{'scope':'illustration_not_live_API_trace','same_Muse_model_separate_critic_prompts':True,'maximum_repair_rounds':2,'owner_approval_required':True,'feedback_memory':'retrievable_guidance_not_weight_training'},'architecture_animation':{'scope':'illustration_not_live_API_trace','Meta_models':['Muse','SAM 3.1','SAM 3D Objects'],'our_implementation':['source_bound_object_program','evidence_and_joint_structure_checks','bounded_review_and_owner_approval','curated_readonly_MCP','Three.js_viewers'],'Pages':'saved_results_no_backend_or_API_keys','MCP':'separate_local_server'},'robot_scope':'authored_kinematics_not_physical_control','publication':False})
    print(json.dumps({'movie':F.ref(movie),'probe':probe},indent=2))

if __name__=='__main__':main()
