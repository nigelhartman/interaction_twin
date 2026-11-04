#!/usr/bin/env python3
"""A closer45s editorial treatment of genuine, source-bound browser captures.

No reconstruction, poses or physics are authored here. The master capture must
provide all four supported component matrices and declared source intervals.
The complete robot task retains both releases and its closed final state.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import time
from PIL import Image, ImageDraw, ImageOps

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('motion_film_v1_helpers',ROOT/'scripts/build_dynamic_trailer_20261104_v1.py')
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
W,H,FPS,DURATION=1920,1080,30,45
BASE=ROOT/'output/dynamic-trailer-20261104-v2'
VOICE=[(0,4,'Film a real interaction.'),
       (4,8,'SAM follows the moving parts.'),
       (8,16,'SAM 3D builds each part. Muse identifies the connections. The pipeline fits how they move.'),
       (16,20,'The result: a 3D twin with a readable object program.'),
       (20,25,'Build a searchable library from everyday recordings.'),
       (25,29,'Your agent can find parts, actions and evidence.'),
       (29,39,'Use these twins to prototype robot tasks. Open the case. Remove the earbuds. Place them on a target.'),
       (39,43,'Then test the parts with NVIDIA Newton physics.'),
       (43,45,'Interaction Twin.')]
PARTS={'case_base','case_lid','earbud_a','earbud_b'}


def intervals(d):
    values=d.get('fully_supported_source_intervals_s')
    if values is None:values=[d.get('fully_supported_source_interval_s')]
    if not isinstance(values,list)or not 1<=len(values)<=8:raise ValueError('Declared supported source intervals required')
    for row in values:
        if not isinstance(row,list)or len(row)!=2 or not all(type(v)in(int,float)and math.isfinite(v)for v in row)or not 0<=row[0]<row[1]:raise ValueError('Finite ordered source interval required')
    return values


def verify_motion(clip,receipt):
    d=V.read(receipt)
    if(d.get('format')!='interaction-twin-dynamic-film-capture'or d.get('version')!=3 or
       d.get('canvas')!=[W,H]or d.get('fps')!=30 or d.get('duration_s')!=37 or d.get('bytes')!=clip.stat().st_size):
        raise ValueError('A complete37s native1080p source-bound capture is required')
    if d.get('physical_accuracy_verified')is not False or d.get('new_api_calls')!=0 or d.get('camera_microphone_capture')is not False:raise ValueError('Unsupported master capture claims')
    if d.get('native_rgb_display_policy')!='historical_uint8_vertex_rgb_interpolation_unchanged':raise ValueError('The verified native RGB display policy is required')
    ranges=intervals(d);trace=d.get('projection_trace');chapters=d.get('chapter_source_intervals')
    if not isinstance(chapters,list)or len(chapters)!=len(V.MASTER_MAP):raise ValueError('The complete chapter/source clock is required')
    for row,(_,_,a,b)in zip(chapters,V.MASTER_MAP):
        if row.get('film_s')!=[a,b]or row.get('source_s')is not None and row['source_s']not in ranges:raise ValueError('Chapter/source intervals changed or overlap a gap')
    if not isinstance(trace,list)or len(trace)<100:raise ValueError('Actual native projection trace required')
    last=-1
    for row in trace:
        t,s=row.get('film_s'),row.get('source_s')
        if not isinstance(t,(int,float))or not math.isfinite(t)or not last<t<=37 or not isinstance(s,(int,float))or not math.isfinite(s)or not any(a-1e-6<=s<=b+1e-6 for a,b in ranges):raise ValueError('Projection trace escapes the exact supported source clock')
        last=t;parts=row.get('parts',[])
        chapter=next((c for c in chapters if c['film_s'][0]<=t<c['film_s'][1]),chapters[-1]if t==37 else None)
        if chapter is None or chapter['source_s']is None:raise ValueError('A projection cannot be assigned to a metadata-only chapter')
        a,b=chapter['film_s'];sa,sb=chapter['source_s'];expected=sa+(t-a)*(sb-sa)/(b-a)
        if abs(s-expected)>1e-5:raise ValueError('The projected source clock differs from its declared chapter mapping')
        if len(parts)!=4 or {p.get('id')for p in parts}!=PARTS:raise ValueError('Every filmed twin needs all four native parts')
        for p in parts:
            m=p.get('camera_from_native_mesh')
            if not isinstance(m,list)or len(m)!=4 or not all(isinstance(r,list)and len(r)==4 and all(type(v)in(int,float)and math.isfinite(v)for v in r)for r in m):raise ValueError('Withheld part poses cannot enter the film')
    pointers=d.get('inputs',[])
    if not all(d.get(k)in pointers for k in['source_ref','scene_ref'])or d.get('source_sha256')!=d['source_ref'].get('sha256'):raise ValueError('Original source and scene must be included as exact capture inputs')
    for pointer in pointers:V.checked(pointer)
    if V.read(V.checked(d['scene_ref']))['source']['sha256']!=d['source_sha256']:raise ValueError('The captured native scene belongs to a different recording')
    return d


def verify_robot(receipt):
    d=V.read(receipt)
    if d.get('format')!='interaction-twin-complete-robot-browser-capture'or d.get('canvas')!=[W,H]or d.get('captured_full_task')is not True:raise ValueError('A genuine complete robot capture is required')
    final_angles=d.get('final_joint_q_rad')
    if not isinstance(final_angles,dict)or not final_angles or not all(type(q)in(int,float)and math.isfinite(q)and abs(q)<1e-7 for q in final_angles.values()):raise ValueError('The robot must reach its closed joint reference')
    if d.get('final_gripper_attached')is not None or d.get('physical_contact_verified')is not False:raise ValueError('Both releases and the closed case must be completed')
    goals=d.get('final_goals',[])
    if {g.get('id')for g in goals}!={'earbud_a','earbud_b'}or len(goals)!=2 or not all(g.get('reached')is True and g.get('released')is True and g.get('goal_error_display_units',1)<1e-7 for g in goals):raise ValueError('Both actual release goals must be reached')
    trace=d.get('trace',[])
    if not trace or not all(r.get('reachable')is True and r.get('ik_residual',1)<.002 for r in trace):raise ValueError('The complete robot trace must remain reachable')
    V.robot_crop(d)
    return d


def prepare_voice(out,source_folder):
    folder=out/'voice';folder.mkdir(exist_ok=True)
    if source_folder:
        import shutil
        for i in range(len(VOICE)):
            p=Path(source_folder)/f'{i:02}-samantha.aiff'
            if not p.is_file()or V.duration(p)<=.1:raise ValueError('Complete matching local narration required')
            if p.resolve()!=(folder/p.name).resolve():shutil.copyfile(p,folder/p.name)
    with ThreadPoolExecutor(max_workers=3)as pool:
        segments=list(pool.map(lambda r:V.voice_one(r,folder),[(i,a,b,t)for i,(a,b,t)in enumerate(VOICE)]))
    args=['ffmpeg','-nostdin','-v','error'];filters=[];labels=[]
    for i,s in enumerate(segments):
        args+=['-i',str(ROOT/s['processed']['path'])]
        delay=round(s['start_s']*1000);filters.append(f'[{i}:a]adelay={delay}|{delay},apad=whole_dur=45[a{i}]');labels.append(f'[a{i}]')
    filters.append(''.join(labels)+f'amix=inputs={len(segments)}:normalize=0,atrim=0:45,alimiter=limit=.89[voice]')
    target=out/'voice-45.wav';V.run(args+['-filter_complex',';'.join(filters),'-map','[voice]','-ar','48000','-ac','2',str(target)])
    V.write(out/'narration.json',{'segments':segments,'duration_s':45,'hosted_speech':False})
    return segments,target


def smooth(a,b,t):
    x=max(0.,min(1.,t));x=x*x*(3-2*x);return a+(b-a)*x


def robot_box(receipt,elapsed):
    full=V.robot_crop(receipt)
    # A final editorial close-up follows completed transfers, never hides a goal.
    close=(full[0],round(full[1]+.29*(full[3]-full[1])),full[2],full[3])
    first_release=next(r['time_s']for r in receipt['trace']if all(g['released']and g['reached']for g in r['goals']))
    display_release=.6+(first_release-1)*8/23.9
    q=(elapsed-max(7.6,display_release+.05))/1.0
    box=tuple(round(smooth(a,b,q))for a,b in zip(full,close))
    for goal in receipt['final_goal_canvas']:
        x,y=goal['canvas'][:2]
        if not box[0]+8<x<box[2]-8 or not box[1]+8<y<box[3]-8:raise ValueError('Editorial close-up cuts a release goal')
    return box


def brand(draw,logo,im):
    art=logo.resize((45,45),Image.Resampling.LANCZOS);im.paste(art,(60,42),art)
    draw.text((120,47),'Interaction Twin',font=V.font(28,True),fill=V.INK)


def robot_frame(picture,elapsed,receipt,logo):
    im=Image.new('RGB',(W,H),picture.getpixel((0,0)));layer=ImageOps.contain(picture.crop(robot_box(receipt,elapsed)),(1170,900),Image.Resampling.LANCZOS)
    im.paste(layer,(65+(1170-layer.width)//2,118+(900-layer.height)//2))
    d=ImageDraw.Draw(im);brand(d,logo,im)
    d.text((1280,196),'ROBOTICS LAB',font=V.font(20,True),fill=V.BLUE)
    d.multiline_text((1275,240),'Prototype\nthe task.',font=V.font(56,True),fill=V.INK,spacing=0)
    phase=V.robot_stage(receipt,elapsed);labels=['Place the case.','Open the lid.','Remove an earbud.','Place on the tray.','Close the case.']
    d.text((1280,434),f'{phase+1:02} / 05',font=V.font(20),fill=V.MUTED)
    d.text((1276,484),labels[phase],font=V.font(39,True),fill=V.BLUE)
    for i in range(5):d.rounded_rectangle((1280+i*61,565,1326+i*61,570),radius=2,fill=V.BLUE if i<=phase else'#d9deea')
    if elapsed>=8.6:
        d.text((1280,640),'Both earbuds placed.',font=V.font(28,True),fill=V.INK)
        d.text((1280,683),'Case closed. Task complete.',font=V.font(24),fill=V.MUTED)
    return im


def newton_box(elapsed):
    # Preserve every initially airborne object. Push closer only near contact.
    return tuple(round(smooth(a,b,(elapsed-2)/1.5))for a,b in zip((0,0,W,H),(200,340,1750,H)))


def newton_frame(picture,elapsed,logo):
    # A bounded editorial push-in. No individual object transforms change.
    im=Image.new('RGB',(W,H),picture.getpixel((0,0)))
    layer=ImageOps.contain(picture.crop(newton_box(elapsed)),(W,H),Image.Resampling.LANCZOS)
    im.paste(layer,((W-layer.width)//2,H-layer.height))
    d=ImageDraw.Draw(im);brand(d,logo,im)
    d.text((60,136),'NVIDIA NEWTON',font=V.font(20,True),fill=V.BLUE)
    d.multiline_text((55,180),'Gravity.\nContact.',font=V.font(63,True),fill=V.INK,spacing=0)
    return im


def end_frame(t,logo):
    im=Image.new('RGB',(W,H),V.BG);layer=Image.new('RGBA',(W,H));d=ImageDraw.Draw(layer)
    progress=min(1.,max(0.,t/.4));ease=1-(1-progress)**3
    art=logo.resize((160,160),Image.Resampling.LANCZOS);art.putalpha(art.getchannel('A').point(lambda a:round(a*progress)))
    im.paste(art,(300,round(415+8*(1-ease))),art)
    d.text((505,round(409+8*(1-ease))),'Interaction Twin',font=V.font(91,True),fill=(*ImageColor(V.INK),round(255*progress)))
    d.text((510,535),'Show it once. Keep how it works.',font=V.font(39),fill=(*ImageColor(V.MUTED),round(255*progress)))
    im.paste(layer,(0,0),layer)
    return im


def ImageColor(value):return tuple(bytes.fromhex(value.lstrip('#')))


def build(motion,receipt,out,voice_source,robot=None,robot_receipt=None):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    motion,receipt=motion.resolve(),receipt.resolve();md=verify_motion(motion,receipt)
    robot=robot.resolve()if robot else V.checked(V.ROBOT_REF)
    robot_receipt=robot_receipt.resolve()if robot_receipt else V.checked(V.ROBOT_JSON)
    rj=verify_robot(robot_receipt)
    if robot.stat().st_size!=rj['bytes']:raise ValueError('Robot capture length changed')
    nr=V.read(V.NEWTON_REPORT)
    if nr.get('claims',{}).get('real_physics_engine_execution')is not True:raise ValueError('Actual Newton execution required')
    logo=Image.open(V.checked(V.LOGO_REF)).convert('RGBA')
    segments,voice=prepare_voice(out,voice_source);cues=V.captions(out,segments)
    masters=V.decode(motion,out/'master-frames');robots=V.decode(robot,out/'robot-frames');physics=V.decode(V.NEWTON,out/'physics-frames')
    if len(masters)<1106 or len(robots)<805 or len(physics)<280:raise ValueError('A source capture is incomplete')
    silent=out/'silent-45.mp4'
    encoder=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(silent)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    proofs=[];proof_indices={round(t*FPS)for t in[0,4.5,8.8,12,15.3,17.8,21.5,26,30,34.5,37.7,38.5,39.1,40.3,42,44.1]}
    try:
        for n in range(FPS*DURATION):
            t=n/FPS
            if t<29:im=V.source_frame(masters,V.master_time(t)).copy()
            elif t<39:im=robot_frame(V.source_frame(robots,V.robot_time(t-29)),t-29,rj,logo)
            elif t<43:im=newton_frame(V.source_frame(physics,.4+(t-39)*1.8/4),t-39,logo)
            else:im=end_frame(t-43,logo)
            if n in proof_indices:
                p=out/f'proof-{n:04}.jpg';im.save(p,quality=94);proofs.append({'time_s':t,'file':V.ref(p)})
            encoder.stdin.write(im.tobytes())
        encoder.stdin.close();stderr=encoder.stderr.read();code=encoder.wait()
        if code:raise ValueError('Video encoder failed: '+stderr.decode()[-2000:])
    except BaseException:
        encoder.kill();raise
    movie=out/'interaction-twin-dynamic-20261104-v2.mp4'
    mix='[1:a]loudnorm=I=-29:TP=-5:LRA=7,afade=t=in:st=0:d=0.8,afade=t=out:st=43:d=2[music];[2:a]asplit=2[voice][sidechain];[music][sidechain]sidechaincompress=threshold=.015:ratio=5:attack=15:release=250[duck];[duck][voice]amix=inputs=2:normalize=0,alimiter=limit=.89,atrim=0:45[a]'
    V.run(['ffmpeg','-nostdin','-v','error','-i',str(silent),'-i',str(V.MUSIC),'-i',str(voice),'-filter_complex',mix,'-map','0:v:0','-map','[a]','-t','45','-c:v','copy','-c:a','aac','-b:a','192k','-ar','48000','-ac','2','-movflags','+faststart',str(movie)])
    probe=json.loads(V.run(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(movie)]).stdout)
    video=next(s for s in probe['streams']if s['codec_type']=='video');audio=next(s for s in probe['streams']if s['codec_type']=='audio')
    if(video['width'],video['height'],int(video['nb_read_frames']),int(audio['sample_rate']),audio['channels'])!=(W,H,1350,48000,2)or abs(float(probe['format']['duration'])-45)>.001:raise ValueError('Final movie dimensions/frame/audio clock differ')
    V.run(['ffmpeg','-nostdin','-v','error','-ss','12','-i',str(movie),'-frames:v','1','-q:v','2',str(out/'poster.jpg')])
    contact_sheet(proofs,out/'contact-sheet.jpg')
    V.write(out/'verification.json',{'format':'interaction-twin-motion-first-trailer','version':2,'movie':V.ref(movie),'poster':V.ref(out/'poster.jpg'),'captions':V.ref(out/'captions.vtt'),'transcript':V.ref(out/'transcript.txt'),'proof_frames':proofs,'render':{'width':W,'height':H,'fps':FPS,'duration_s':45,'frames':1350,'stereo_sample_rate':48000},'source_captures':{'motion':V.ref(motion),'motion_receipt':V.ref(receipt),'robot':V.ref(robot),'robot_receipt':V.ref(robot_receipt),'newton':V.ref(V.NEWTON),'newton_execution':V.ref(V.NEWTON_REPORT)},'source_intervals_s':intervals(md),'chapter_source_intervals':md['chapter_source_intervals'],'native_rgb_display_policy':md['native_rgb_display_policy'],'time_mapping':{'master':V.MASTER_MAP,'robot':'Original complete0–26.9s task; .6s initial,8s whole motion,1.4s completed final hold.','newton':{'display':[39,43],'source':[.4,2.2]},'brand':[43,45]},'editorial_framing':{'robot':'Fixed whole-motion crop; bounded close-up begins after both transfers, retaining both original projected goals.','newton':'Initially whole-stage; bounded push-in near contact, with unchanged original simulated body poses.'},'robot_scope':'Authored kinematic rehearsal; both earbuds released, lid closed, tool withdrawn. No force/contact/grasp validation.','physics_scope':'Actual Newton CPU gravity/contact with assumed scale/materials and convex proxies.','speech':{'provider':'local Apple Samantha','segments':segments},'music':V.ref(V.MUSIC),'music_rights':'Original procedural project composition; no third-party samples.','logo':V.LOGO_REF,'builder':V.ref(Path(__file__)),'elapsed_s':time.monotonic()-started,'new_api_calls':0,'new_gpu_calls':0,'public_media_replaced':False,'published':False,'ffprobe':probe})
    print(json.dumps({'movie':V.ref(movie),'verification':V.ref(out/'verification.json')}))


def contact_sheet(proofs,target):
    sheet=Image.new('RGB',(1440,math.ceil(len(proofs)/3)*290),V.BG);d=ImageDraw.Draw(sheet)
    for i,p in enumerate(proofs):
        art=Image.open(ROOT/p['file']['path']);art.thumbnail((480,270));x=(i%3)*480;y=(i//3)*290;sheet.paste(art,(x,y));d.text((x+10,y+272),f"{p['time_s']:.1f} s",font=V.font(13),fill=V.MUTED)
    sheet.save(target,quality=94)


def storyboard(out):
    out.mkdir(parents=True,exist_ok=False);logo=Image.open(V.checked(V.LOGO_REF)).convert('RGBA');rj=verify_robot(V.checked(V.ROBOT_JSON));proofs=[]
    source=V.BASE/'render-v1';robot_frames=sorted((source/'robot-frames').glob('*.jpg'));physics_frames=sorted((source/'physics-frames').glob('*.jpg'))
    if not robot_frames or not physics_frames:raise ValueError('Explicit previously verified source frames are required for the design study')
    for name,t,im in[('robot-action',34.5,robot_frame(V.source_frame(robot_frames,V.robot_time(5.5)),5.5,rj,logo)),('robot-complete',38.5,robot_frame(V.source_frame(robot_frames,V.robot_time(9.5)),9.5,rj,logo)),('newton',41,newton_frame(V.source_frame(physics_frames,1.3),2,logo)),('brand',44,end_frame(1,logo))]:
        p=out/(name+'.jpg');im.save(p,quality=95);proofs.append({'time_s':t,'file':V.ref(p)})
    contact_sheet(proofs,out/'contact-sheet.jpg');V.write(out/'design-study.json',{'format':'interaction-twin-film-design-study','version':2,'not_final_trailer':True,'uses_previously_verified_existing_captures':True,'new_master_not_captured':True,'public_media_replaced':False,'proof_frames':proofs,'builder':V.ref(Path(__file__))})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--motion',type=Path);p.add_argument('--receipt',type=Path);p.add_argument('--robot',type=Path);p.add_argument('--robot-receipt',type=Path);p.add_argument('--voice-source',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--storyboard',action='store_true');p.add_argument('--prepare-voice',action='store_true');a=p.parse_args()
    if a.storyboard:storyboard(a.out)
    elif a.prepare_voice:a.out.mkdir(parents=True,exist_ok=False);segments,voice=prepare_voice(a.out,a.voice_source);V.captions(a.out,segments)
    elif a.motion and a.receipt and bool(a.robot)==bool(a.robot_receipt):build(a.motion,a.receipt,a.out,a.voice_source,a.robot,a.robot_receipt)
    else:p.error('Supply explicit --motion and --receipt; robot overrides require both files.')
