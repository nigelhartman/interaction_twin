#!/usr/bin/env python3
"""New local-only 54-second research demo, genuine saved outputs + original score.

No inference, external music, geometry edits, public publishing or old-media writes.
Editorial panels/camera orbits are explicitly distinguished from recovered motion.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import wave

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_showcase_media import font as _font,VideoClip
font=lru_cache(maxsize=100)(_font)
OUT=ROOT/'output/overnight-session-20261031'
MEDIA=ROOT/'showcase/media'
PREFIX='overnight-trailer-20261031-v2'
PREP=OUT/(PREFIX+'-preparation.json')
W,H,FPS=1920,1080,24
DURATIONS=[5,7.5,7.5,10,7.5,9,7.5]
BG='#101f2c';INK='#f6f3eb';MUTED='#acc0c8';TEAL='#68e1c9';BLUE='#8bb8ff';AMBER='#e5b25d'
COLORS={'case_base':(20,186,177),'case_lid':(236,140,68),'earbud_a':(119,139,240),'earbud_b':(197,111,208)}
PROGRAM='data/flow-rigid-parts-jabra-20261031-v1/flow-procedure-1/interaction.json'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def local(value):
    if not isinstance(value,str) or any(t in value for t in ['\\','\0','?','#',':']):raise ValueError('Local declared input only')
    path=(ROOT/value.removeprefix('/files/')).resolve()
    if not path.is_relative_to(ROOT) or any(p.startswith('.') for p in path.relative_to(ROOT).parts):raise ValueError('Input escapes visible project artifacts')
    return path
def record(path,kind,**extra):return dict(path=rel(path),sha256=sha(path),kind=kind,**extra)
def fresh_json(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def txt(canvas,xy,text,size=32,color=INK,bold=False,spacing=10):
    ImageDraw.Draw(canvas).multiline_text(xy,text,font=font(size,bold),fill=color,spacing=spacing)
def tag(canvas,xy,text,color=TEAL):
    x,y=xy;width=int(font(20,True).getlength(text))+32
    ImageDraw.Draw(canvas).rounded_rectangle((x,y,x+width,y+40),12,fill=color)
    txt(canvas,(x+16,y+7),text,20,BG,True)
def panel(canvas,image,box,bg='#eff3ed',contain=True):
    x,y,width,height=map(int,box);piece=Image.new('RGB',(width,height),bg)
    content=(ImageOps.contain if contain else ImageOps.fit)(image.convert('RGB'),(width,height),Image.Resampling.LANCZOS)
    piece.paste(content,((width-content.width)//2,(height-content.height)//2))
    mask=Image.new('L',piece.size);ImageDraw.Draw(mask).rounded_rectangle((0,0,width-1,height-1),24,fill=255)
    canvas.paste(piece,(x,y),mask)
def base(kicker,title,subtitle,number):
    canvas=Image.new('RGB',(W,H),BG);draw=ImageDraw.Draw(canvas)
    draw.rounded_rectangle((68,53,84,76),6,fill=TEAL)
    txt(canvas,(101,46),'INTERACTION TWIN',25,bold=True)
    txt(canvas,(72,113),kicker.upper(),21,TEAL,True)
    txt(canvas,(66,154),title,66,bold=True)
    txt(canvas,(72,238),subtitle,29,MUTED)
    draw.line((72,1008,1848,1008),fill='#354855',width=2)
    txt(canvas,(72,1027),'Working prototype · physical scale, assembly and contact remain unverified.',20,MUTED)
    txt(canvas,(1755,1027),number,20,MUTED,True)
    return canvas


def original_score(destination):
    """Original deterministic A-minor instrumental; no samples or existing melody."""
    sr=48000;duration=sum(DURATIONS);count=int(duration*sr);mix=np.zeros((count,2),np.float64)
    rng=np.random.default_rng(3162026);bpm=96.;beat=60/bpm
    # Original progression and melody, explicitly encoded rather than sampled.
    chords=[(45,[57,60,64]),(41,[53,57,60]),(48,[55,60,64]),(43,[55,59,62])]
    def add(start,length,midi,amp,pan=0.,voice='pad'):
        offset=int(start*sr);n=min(int(length*sr),count-offset)
        if n<=0:return
        t=np.arange(n)/sr;frequency=440*2**((midi-69)/12)
        if voice=='pad':
            signal=np.sin(2*np.pi*frequency*t)+.18*np.sin(2*np.pi*frequency*2.003*t)+.1*np.sin(2*np.pi*frequency*.997*t)
            env=np.minimum(t/.2,1)*np.minimum((n/sr-t)/.65,1)
        elif voice=='pluck':
            signal=np.sin(2*np.pi*frequency*t)+.2*np.sin(2*np.pi*frequency*2*t)
            env=(1-np.exp(-t*150))*np.exp(-t*4.)
        else:
            signal=np.sin(2*np.pi*frequency*t)+.08*np.sin(2*np.pi*frequency*2*t)
            env=(1-np.exp(-t*90))*np.exp(-t*2.)
        signal*=np.maximum(env,0)*amp
        angle=(pan+1)*np.pi/4;mix[offset:offset+n,0]+=signal*np.cos(angle);mix[offset:offset+n,1]+=signal*np.sin(angle)
    for bar,start in enumerate(np.arange(0,duration,4*beat)):
        bass,chord=chords[bar%4]
        for k,note in enumerate(chord):add(start,4*beat+.6,note,.09,(-.4,0,.4)[k])
        for k in [0,2]:add(start+k*beat,beat*1.65,bass,.095,voice='bass')
        if bar>=2:
            order=[0,1,2,1,0,2,1,2] if bar%2==0 else [2,0,1,2,1,0,2,1]
            for k,index in enumerate(order):add(start+k*beat/2,beat*.92,chord[index]+12,.058,(-.32,.32)[k%2],voice='pluck')
        if 4<=bar<=17:
            for k in range(4):
                at=int((start+k*beat)*sr);n=min(int(.22*sr),count-at)
                if n<=0:continue
                t=np.arange(n)/sr
                kick=np.sin(2*np.pi*(48*t+35*.035*(1-np.exp(-t/.035))))*np.exp(-t*25)*.115
                mix[at:at+n]+=kick[:,None]
            for k in range(8):
                at=int((start+k*beat/2)*sr);n=min(int(.065*sr),count-at)
                if n<=0:continue
                noise=rng.normal(size=n);high=np.r_[0.,np.diff(noise)]
                hat=high*np.exp(-np.arange(n)/sr*80)*.011
                mix[at:at+n]+=hat[:,None]*np.array([.85,1.])
    time_axis=np.arange(count)/sr
    fade=np.minimum(time_axis/1.5,1)*np.minimum((duration-time_axis)/3.2,1)
    mix*=np.maximum(fade,0)[:,None]
    mix*=.72/max(1e-12,float(np.abs(mix).max()))
    # Leave headroom for AAC reconstruction; the limiter does not clip samples.
    samples=np.rint(np.clip(mix,-1,1)*32767).astype('<i2')
    with wave.open(str(destination),'wb') as stream:
        stream.setnchannels(2);stream.setsampwidth(2);stream.setframerate(sr);stream.writeframes(samples.tobytes())
    return {'composition':'Original deterministic procedural A-minor instrumental; no sampled audio or referenced song.',
        'rights':'No third-party music or samples; composed in this project script for this local trailer.',
        'seed':3162026,'bpm':bpm,'sample_rate_hz':sr,'channels':2,'duration_s':duration,
        'peak_dbfs':float(20*np.log10(np.abs(mix).max())),
        'rms_dbfs':float(20*np.log10(np.sqrt(np.mean(mix*mix)))),
        'fade_in_s':1.5,'fade_out_s':3.2,'score_implementation':record(Path(__file__),'procedural_score_source')}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);MEDIA.mkdir(exist_ok=True)
    if PREP.exists():raise ValueError('Preserve prepared sources; use a new version')
    directory=OUT/(PREFIX+'-assets');directory.mkdir()
    program_path=local(PROGRAM);program=read(program_path)
    if sha(local(program['source']['path']))!=program['source']['sha256']:raise ValueError('Source video changed')
    sequence=[]
    for source in program['evidence']:
        if not 11.99<=source['timestamp_s']<=19.01:continue
        original=local(source['frame_path'])
        if sha(original)!=source['frame_sha256']:raise ValueError('Source RGB changed')
        rgb=np.asarray(Image.open(original).convert('RGB')).copy();masks=[]
        for obs in source['observations']:
            path=local(obs['mask_path'])
            if sha(path)!=obs['mask_sha256']:raise ValueError('Source mask changed')
            mask=np.asarray(Image.open(path).convert('L'))>127;color=np.asarray(COLORS[obs['track_id']])
            rgb[mask]=np.rint(.62*rgb[mask]+.38*color).astype(np.uint8)
            edge=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0;rgb[edge]=color
            masks.append(record(path,'actual_SAM31_video_mask',component_id=obs['track_id']))
        path=directory/f'mask-{source["source_index"]:06d}.jpg';Image.fromarray(rgb).save(path,quality=96)
        sequence.append({'source_pts_s':source['timestamp_s'],'source_index':source['source_index'],
            'source':record(original,'saved_original_RGB'),'masks':masks,'overlay':record(path,'tint_and_edges_only')})
    # Reuse a frozen source-linked render, never rebuild or modify old model media.
    old=read(ROOT/'output/showcase-studio-20261031/source-manifest.json')
    for entry in old['rendered_asset_inputs']:
        if sha(local(entry['path']))!=entry['sha256']:raise ValueError('Saved orbit input changed')
    orbits=[]
    for name in ['observed','inferred']:
        outputs=[]
        for n in range(24):
            source=local(old['camera_orbit_directory'])/f'{name}-{n:03d}.jpg';dest=directory/f'camera-{name}-{n:03d}.jpg'
            shutil.copyfile(source,dest);outputs.append(record(dest,'unchanged_prior_source_linked_inspection_render',original=record(source,'frozen_prior_render')))
        orbits.append({'kind':name,'frames':outputs})
    source_image=local(old['camera_source_image']);camera=directory/'camera-source.jpg';shutil.copyfile(source_image,camera)
    audio=OUT/(PREFIX+'-original-score.wav')
    inherited=read(OUT/'overnight-trailer-20261031-v1-source-manifest.json')['preparation_content']['score']
    original_audio=local(inherited['file']['path'])
    if sha(original_audio)!=inherited['file']['sha256']:raise ValueError('Original score bytes changed')
    shutil.copyfile(original_audio,audio);score={k:v for k,v in inherited.items() if k!='file'}
    score['reused_original_audio']=record(original_audio,'byte_exact_original_v1_score')
    snapshot=OUT/(PREFIX+'-builder-at-prepare.py');shutil.copyfile(Path(__file__),snapshot)
    prep={'format':'overnight-source-linked-trailer-preparation','version':1,
        'program':record(program_path,'actual_published_interaction_program'),
        'source_video':record(local(program['source']['path']),'actual_upright_recorded_source',original_sha256=program['source']['original_sha256']),
        'source_sequence':sequence,'camera_source':record(camera,'exact_copy_of_camera_source',original=record(source_image,'original_camera_RGB')),
        'camera_orbits':orbits,'inherited_source_linked_render_proof':record(ROOT/'output/showcase-studio-20261031/source-manifest.json','full_prior_render_provenance'),
        'geometry_counts':old['geometry_counts'],'score':dict(score,file=record(audio,'original_instrumental_score')),
        'builder':record(snapshot,'builder_at_prepare'),'duration_s':sum(DURATIONS),'chapters_s':DURATIONS,
        'display_policy':{'inspection_camera_orbits_are_editorial':True,'observed_texture':'original RGB source UV',
            'inferred_colour':'amber display overlay; source geometry unchanged','source_masks':'Exact native masks, tinted overlay only, held at actual saved 3fps source timestamps.',
            'unreviewed_native_priors_not_promoted':True,'private_recordings_local_only':True,
            'source_clock_no_synthetic_frames':True,'physical_accuracy_verified':False,'common_assembly_established':False,
            'simulation_ready':False,'new_model_calls':0,'api_calls':0,'mps_used':False,'external_music':False}}
    fresh_json(PREP,prep);print(json.dumps({'status':'prepared','mask_samples':len(sequence),'score':score,'path':rel(PREP)}),flush=True)


def motion_clip():
    from native_appearance import load_native_glb,render_native_mesh
    directory=OUT/(PREFIX+'-motion');directory.mkdir()
    result_path=local('data/interval-motion-20261031/density-lid-comparison-v1/density-12fps/interval-motion.json');result=read(result_path)
    mesh_path=local(result['asset']['path'])
    if sha(mesh_path)!=result['mesh_sha256']:raise ValueError('Native GLB changed')
    mesh=load_native_glb(mesh_path);episode=result['episodes'][0];K=np.asarray(episode['intrinsics']['matrix'])
    boxes=[];sources=[]
    for row in episode['frames']:
        source=local(row['source_image_url']);mask_path=local(row['mask_url'])
        if sha(source)!=row['source_image_sha256'] or sha(mask_path)!=row['mask_sha256']:raise ValueError('Motion source changed')
        mask=np.asarray(Image.open(mask_path).convert('L'))>127;yy,xx=np.nonzero(mask)
        boxes.append([xx.min(),yy.min(),xx.max()+1,yy.max()+1]);sources.append((source,mask_path,mask))
    boxes=np.asarray(boxes);crop=[max(0,int(boxes[:,0].min())-45),max(0,int(boxes[:,1].min())-45),
        min(episode['frames'][0]['width'],int(boxes[:,2].max())+45),min(episode['frames'][0]['height'],int(boxes[:,3].max())+45)]
    rows=[]
    for j,(row,(source,mask_path,mask)) in enumerate(zip(episode['frames'],sources)):
        rgb=np.asarray(Image.open(source).convert('RGB')).copy();color=np.asarray(COLORS['case_lid'])
        rgb[mask]=np.rint(.75*rgb[mask]+.25*color).astype(np.uint8)
        native=np.full_like(rgb,[229,236,226]);available=row['candidate_rotation'] is not None and row['candidate_translation'] is not None
        if available:
            rendered=render_native_mesh(mesh,K,np.asarray(row['candidate_rotation']),np.asarray(row['candidate_translation']),row['width'],row['height'])
            native[rendered['mask']]=rendered['rgb'][rendered['mask']]
        source_output=directory/f'{j:02d}-source.jpg';native_output=directory/f'{j:02d}-native.jpg'
        Image.fromarray(rgb).crop(crop).save(source_output,quality=97);Image.fromarray(native).crop(crop).save(native_output,quality=97)
        rows.append({'sample_index':row['sample_index'],'timestamp_s':row['timestamp_s'],'status':row['status'],
            'pose_available':available,'native_pts_exact_s':row.get('native_pts_exact_s'),'source':record(source,'exact_original_frame'),'mask':record(mask_path,'actual_native_model_mask',mask_origin=row.get('mask_origin','declared saved mask')),
            'source_crop':record(source_output,'fixed_editorial_crop_and_mask_tint'),'native_crop':record(native_output,'exact_saved_candidate_native_projection'),
            'camera_from_native_mesh':{'rotation':row['candidate_rotation'],'translation':row['candidate_translation']}})
    snapshot=directory/'renderer-at-motion-prepare.py';shutil.copyfile(Path(__file__),snapshot)
    manifest={'format':'exact_saved_conditional_motion_media','version':1,'result':record(result_path,'independently_audited_interval_result'),
        'mesh':record(mesh_path,'unchanged_native_TripoSR_GLb'),'episode_id':episode['id'],'source_video_sha256':result['source_video_sha256'],
        'intrinsics':episode['intrinsics'],'fixed_editorial_crop_xyxy_source_pixels':crop,'frames':rows,
        'renderer':record(ROOT/'native_appearance.py','native_renderer'),
        'builder':record(snapshot,'motion_media_renderer_source'),
        'pose_interpolation':False,'native_geometry_modified':False,'physical_pose_accuracy_verified':False,
        'reference_is_unverified':True,'episode_gauge_independent':True,
        'source_sampling_fps':12,'display_playback_speed':0.5,
        'input_domain':'Separate native PTS source extension; original semantic program unchanged',
        'mask_pipeline':'One original SAM3.1 seed; unchanged local SAM2Tiny propagated masks elsewhere',
        'paired_support_result':'Shared non-reference6/6 at both3fps and12fps; no shared-timestamp gain demonstrated',
        'independent_audit':record(ROOT/'output/interval-motion-20261031/density-lid-pair-independent-audit-v1.json','independent_numeric_and_identity_audit'),
        'new_model_calls':0,'api_calls':0,'mps_used':False}
    fresh_json(OUT/(PREFIX+'-motion-preparation.json'),manifest)
    print(json.dumps({'status':'motion_prepared','saved_source_samples':len(rows),'crop':crop}),flush=True)


def load_assets(screenshots):
    prep=read(PREP)
    sequence=prep['source_sequence']
    assets={'prep':prep,'sequence':sequence,'source':[Image.open(local(r['source']['path'])).convert('RGB') for r in sequence],
        'masks':[Image.open(local(r['overlay']['path'])).convert('RGB') for r in sequence],
        'camera_source':Image.open(local(prep['camera_source']['path'])).convert('RGB')}
    for group in prep['camera_orbits']:assets[group['kind']]=[Image.open(local(r['path'])).convert('RGB') for r in group['frames']]
    motion_path=OUT/(PREFIX+'-motion-preparation.json')
    if motion_path.exists():
        motion=read(motion_path);assets['motion']=motion
        assets['motion_source']=[Image.open(local(r['source_crop']['path'])).convert('RGB') for r in motion['frames']]
        assets['motion_native']=[Image.open(local(r['native_crop']['path'])).convert('RGB') for r in motion['frames']]
    program=read(local(PROGRAM));relationship=program['relationships'][0]
    context=next(r for r in program['evidence'] if r['id']==relationship['evidence_ids'][0])
    source=local(context['frame_path'])
    if sha(source)!=context['frame_sha256']:raise ValueError('Context evidence RGB changed')
    pixels=np.asarray(Image.open(source).convert('RGB')).copy();context_masks=[]
    for obs in context['observations']:
        path=local(obs['mask_path'])
        if sha(path)!=obs['mask_sha256']:raise ValueError('Context mask changed')
        mask=np.asarray(Image.open(path).convert('L'))>127;color=np.asarray(COLORS[obs['track_id']])
        pixels[mask]=np.rint(.62*pixels[mask]+.38*color).astype(np.uint8)
        context_masks.append(record(path,'actual_context_SAM31_mask',component_id=obs['track_id']))
    assets['context_mask']=Image.fromarray(pixels)
    assets['context_evidence']={'id':context['id'],'timestamp_s':context['timestamp_s'],
        'source':record(source,'exact_model_cited_context_evidence'),'masks':context_masks}
    assets['screens']={}
    for row in screenshots:
        path=local(row['path'])
        if not path.is_relative_to(ROOT/'output') or not row.get('url') or not row.get('description'):raise ValueError('Use actual screenshot with URL and description')
        if row.get('sha256') and sha(path)!=row['sha256']:raise ValueError('Screenshot hash changed')
        if row['role']=='agents':
            proof=read(local(row['provenance_path']));handoff_path=ROOT/'data/mcp-fresh-jabra-motion-handoff-20261101-v1-preview.json';handoff=read(handoff_path)
            if proof['preview_sha256']!=sha(handoff_path) or proof['source_pts_s']!=handoff['source_timestamp_s']:raise ValueError('Current MCP capture is not bound to this exact handoff')
            claims=proof['claims']
            if (claims['scripted_sdk_calls'],claims['native_glbs'],claims['unique_cited_jpegs'],claims['procedure_steps'])!=(36,3,6,4):raise ValueError('MCP screenshot counts differ')
        screen=Image.open(path).convert('RGB')
        crop=row.get('editorial_crop_xyxy')
        if crop is not None:
            if len(crop)!=4 or any(type(v)is not int for v in crop) or not(0<=crop[0]<crop[2]<=screen.width and 0<=crop[1]<crop[3]<=screen.height):raise ValueError('Editorial screenshot crop must remain inside the unmodified capture')
            screen=screen.crop(crop)
        assets['screens'][row['role']]=screen
    if not {'motion','agents'}<=set(assets['screens']):raise ValueError('Actual Motion Studio and agent screenshots required')
    return assets


def frame(section,u,assets,live=None):
    # These are graphic layouts over genuine source/model/app captures, not UI mockups.
    n=min(len(assets['sequence'])-1,int(u*len(assets['sequence'])))
    if section==0:
        canvas=Image.new('RGB',(W,H),BG)
        txt(canvas,(76,56),'INTERACTION TWIN',27,TEAL,True)
        txt(canvas,(70,237),'Your video\ncontains a\nprocedure.',94,bold=True,spacing=9)
        txt(canvas,(78,640),'Turn it into parts, geometry\nand context an agent can inspect.',34,MUTED,spacing=13)
        tag(canvas,(78,895),'VIDEO → INTERACTION ASSET')
        panel(canvas,live if live is not None else assets['source'][n],(1230,65,560,950))
    elif section==1:
        canvas=base('01 / Meta perception','Recover its parts.','SAM 3.1 masks follow the visible components in the recorded video.','02 / 07')
        panel(canvas,assets['source'][n],(75,320,380,655));panel(canvas,assets['masks'][n],(515,320,380,655))
        txt(canvas,(1030,365),'ONE CAPTURE.\nSEPARATE COMPONENTS.',43,bold=True,spacing=14)
        for j,(key,name) in enumerate([('case_base','Case base'),('case_lid','Case lid'),('earbud_a','Earbud A'),('earbud_b','Earbud B')]):
            ImageDraw.Draw(canvas).ellipse((1035,550+j*73,1056,571+j*73),fill=COLORS[key]);txt(canvas,(1084,541+j*73),name,34)
        tag(canvas,(1030,896),f'SOURCE {assets["sequence"][n]["source_pts_s"]:.3f} s')
    elif section==2:
        canvas=base('02 / Reasoning + structure','Make every claim inspectable.','Muse Spark proposes components, relationships and steps with source evidence.','03 / 07')
        panel(canvas,assets['context_mask'],(74,325,350,630))
        draw=ImageDraw.Draw(canvas);draw.rounded_rectangle((495,325,1845,945),24,fill='#1c3241')
        program=read(local(PROGRAM));relationship=program['relationships'][0];step=program['steps'][0]
        evidence=next(r for r in program['evidence'] if r['id']==relationship['evidence_ids'][0])
        for j,(value,name) in enumerate([(len(program['tracklets']),'component IDs'),(len(program['steps']),'procedure steps'),(len(program['evidence']),'source samples')]):
            txt(canvas,(535+415*j,356),str(value),64,TEAL,True);txt(canvas,(539+415*j,435),name,26,MUTED)
        txt(canvas,(539,536),relationship['subject_track_id'],39,bold=True)
        txt(canvas,(539,595),'→ '+relationship['predicate']+' → '+relationship['object_track_id'],37,TEAL,True)
        txt(canvas,(539,674),'Step: '+step['title'],33)
        txt(canvas,(539,739),f'Evidence: {evidence["id"]} · {evidence["timestamp_s"]:.3f} s',30,MUTED)
        tag(canvas,(539,856),'ACTUAL SAVED MODEL PROPOSALS')
    elif section==3:
        canvas=base('03 / Evidence-aware 3D','Build 3D. Keep uncertainty visible.','Observed patches retain original source RGB. Optional inferred fill stays separate.','04 / 07')
        angle=min(23,int(u*24));inferred=u>.5
        panel(canvas,assets['camera_source'],(75,357,850,510))
        panel(canvas,assets['inferred' if inferred else 'observed'][angle],(995,357,850,510))
        tag(canvas,(95,894),'ORIGINAL SOURCE RGB',BLUE)
        tag(canvas,(1016,894),'+ INFERRED DEPTH PRIOR' if inferred else '5,377 OBSERVED TRIANGLES',AMBER if inferred else BLUE)
        txt(canvas,(1002,949),'Authored inspection-camera orbit · incomplete visible surface',20,MUTED)
    elif section==4:
        canvas=Image.new('RGB',(W,H),BG)
        txt(canvas,(73,47),'04 / MOTION STUDIO',22,TEAL,True)
        txt(canvas,(520,40),'Inspect the recorded motion.',44,bold=True)
        if u<.35 or not assets.get('motion'):
            shot=assets['screens'].get('overview',assets['screens']['motion']) if u<.15 else assets['screens']['motion']
            panel(canvas,shot,(45,136,1830,825),bg=BG)
            txt(canvas,(73,1000),'Actual app capture · source-linked conditional checks · unsupported frames stay visible',23,MUTED)
        else:
            motion_frames=assets['motion']['frames']
            source_time=motion_frames[0]['timestamp_s']+max(0.,u-.35)*DURATIONS[4]*.5
            index=max(j for j,row in enumerate(motion_frames) if row['timestamp_s']<=min(source_time,motion_frames[-1]['timestamp_s']))
            row=assets['motion']['frames'][index]
            txt(canvas,(76,210),'Recorded pixels + exact saved mask',27,MUTED)
            txt(canvas,(996,210),'Native mesh draft + saved camera pose',27,MUTED)
            panel(canvas,assets['motion_source'][index],(74,260,850,590))
            panel(canvas,assets['motion_native'][index],(994,260,850,590),bg='#e5ece2')
            tag(canvas,(95,876),f'EXACT SOURCE · {row["timestamp_s"]:.3f} s')
            tag(canvas,(1015,876),'UNVERIFIED REFERENCE' if row['status']=='reference_unverified' else 'SAVED CONDITIONAL CANDIDATE',AMBER if row['status']=='reference_unverified' else BLUE)
            txt(canvas,(74,1000),'Lid · 12 fps source samples · half-speed inspection · no pose interpolation · unverified physical pose',23,MUTED)
    elif section==5:
        canvas=Image.new('RGB',(W,H),BG)
        txt(canvas,(73,47),'05 / AGENT ACCESS',22,TEAL,True)
        txt(canvas,(520,40),'Give agents the same context.',44,bold=True)
        panel(canvas,assets['screens']['agents'],(45,136,1830,825),bg=BG)
        txt(canvas,(73,1000),'Same Jabra program · 36 SDK calls · 3 native GLBs · 6 cited frames + motion evidence · no LLM or robot',22,MUTED)
    else:
        canvas=Image.new('RGB',(W,H),BG)
        txt(canvas,(75,58),'INTERACTION TWIN',27,TEAL,True)
        txt(canvas,(70,220),'From a video\nto an interaction\nasset.',91,bold=True,spacing=13)
        txt(canvas,(78,650),'Inspect it. Question it. Reuse its context.',34,MUTED)
        for j,(title,color) in enumerate([('PARTS',TEAL),('3D + SOURCES',BLUE),('PROGRAM + MCP',AMBER)]):tag(canvas,(80,767+j*61),title,color)
        panel(canvas,assets['masks'][min(15,len(assets['masks'])-1)],(1290,114,445,790))
        txt(canvas,(1074,941),'SAM 3.1 + Muse Spark',28,TEAL,True)
        txt(canvas,(78,1030),'Local research prototype · complete physical twins and robot training remain future work.',20,MUTED)
    # Subtle authored timeline, separate from any claimed source/object motion.
    start=sum(DURATIONS[:section]);progress=(start+u*DURATIONS[section])/sum(DURATIONS)
    ImageDraw.Draw(canvas).line((0,H-3,int(W*progress),H-3),fill=TEAL,width=3)
    return canvas


def storyboard(screenshots,name):
    assets=load_assets(screenshots);montage=Image.new('RGB',(1440,1620),'#273a46');entries=[]
    for section in range(7):
        image=frame(section,.5,assets);path=OUT/f'{PREFIX}-{name}-chapter-{section+1:02d}.jpg';image.save(path,quality=96)
        montage.paste(image.resize((720,405)),((section%2)*720,(section//2)*405));entries.append(record(path,'editorial_storyboard_actual_inputs'))
    path=OUT/f'{PREFIX}-{name}-storyboard.jpg';montage.save(path,quality=95)
    print(json.dumps({'storyboard':rel(path),'chapters':entries}),flush=True)


def encode(screenshots):
    destination=MEDIA/(PREFIX+'.mp4')
    if destination.exists():raise ValueError('Preserve previous encoded version')
    assets=load_assets(screenshots);prep=assets['prep'];audio=local(prep['score']['file']['path'])
    # Preparation froze the precise historical score source. Resolve the former
    # live source alias to that byte-identical immutable copy in this manifest.
    score_source=prep['score']['score_implementation']
    if sha(local(score_source['path']))!=score_source['sha256']:
        raise ValueError('Frozen original score implementation changed')
    if sha(audio)!=prep['score']['reused_original_audio']['sha256']:
        raise ValueError('V2 must reuse the exact original score')
    captures=[];capture_dir=OUT/(PREFIX+'-screenshots');capture_dir.mkdir()
    for j,row in enumerate(screenshots):
        original=local(row['path']);copy=capture_dir/f'{j:02d}-{original.name}';shutil.copyfile(original,copy)
        captured=dict(row,**record(copy,'actual_unmodified_app_screenshot',original=record(original,'original_CUA_capture')))
        if row.get('provenance_path'):
            proof=local(row['provenance_path']);value=read(proof)
            if value.get('sha256') and value['sha256']!=sha(original):raise ValueError('Capture proof image hash differs')
            frozen=capture_dir/f'{j:02d}-capture-proof.json';shutil.copyfile(proof,frozen)
            captured['capture_proof']=record(frozen,'unaltered_actual_capture_proof')
        captures.append(captured)
    builder=OUT/(PREFIX+'-builder-at-encode.py');shutil.copyfile(Path(__file__),builder)
    command=['ffmpeg','-hide_banner','-loglevel','warning','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0',
        '-i',str(audio),'-map','0:v:0','-map','1:a:0','-c:v','libx264','-preset','fast','-crf','18','-threads','4','-pix_fmt','yuv420p',
        '-c:a','aac','-b:a','192k','-ar','48000','-movflags','+faststart','-t',str(sum(DURATIONS)),str(destination)]
    log=(OUT/(PREFIX+'-encode.log')).open('x');process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
    started=time.monotonic();count=0;chapter_records=[]
    live=VideoClip(local(prep['source_video']['path']),12,DURATIONS[0],size=(540,960))
    try:
        for section,seconds in enumerate(DURATIONS):
            for n in range(round(seconds*FPS)):
                image=frame(section,n/max(1,round(seconds*FPS)-1),assets,live.frame() if section==0 else None)
                process.stdin.write(image.tobytes());count+=1
            if section==0:live.close()
            chapter_records.append({'chapter':section+1,'start_s':sum(DURATIONS[:section]),'duration_s':seconds})
            print(json.dumps({'chapter':section+1,'frames_encoded':count}),flush=True)
        process.stdin.close()
        if process.wait(timeout=180):raise RuntimeError('Encoder failed; preserved log')
    finally:
        if live.process.poll() is None:live.close()
        if process.poll() is None:process.terminate();process.wait(timeout=20)
        log.close()
    manifest={'format':'source-linked-overnight-trailer','version':1,'preparation':record(PREP,'frozen_preparation'),
        'preparation_content':prep,'builder':record(builder,'exact_encoder_source'),'screenshots':captures,
        'chapters':chapter_records,'width':W,'height':H,'fps':FPS,'duration_s':sum(DURATIONS),'frames':count,
        'render_seconds':round(time.monotonic()-started,3),'output':record(destination,'new_H264_AAC_trailer'),
        'source_playback':'First 5s decode actual recorded source 12..17s at 24fps; frame conversion only drops/duplicates. SAM overlays thereafter use exact saved 3fps samples, no invented frames.',
        'source_decode_helper':record(ROOT/'scripts/build_showcase_media.py','existing_FFmpeg_VideoClip_helper_only'),
        'context_evidence':assets['context_evidence'],
        'motion_sequence':assets.get('motion'),
        'current_same_case_MCP_handoff':record(ROOT/'data/mcp-fresh-jabra-motion-handoff-20261101-v1-preview.json','actual_36_call_same_case_handoff'),
        'current_same_case_MCP_trace':record(ROOT/'data/mcp-fresh-jabra-motion-handoff-20261101-v1/sdk-trace.json','actual_current_readonly_SDK_responses'),
        'motion_playback_policy':'12fps recorded native-source samples, replayed at0.5speed with sample hold only; final sample held. No pose or image interpolation. Source and mask were frozen before tracking.',
        'motion_comparison_limit':'Paired lid shared-source support ties6/6 (reserved2/2) at3fps and12fps; higher display density does not establish accuracy.',
        'motion_sequence_preparation':record(OUT/(PREFIX+'-motion-preparation.json'),'frozen_saved_motion_media') if assets.get('motion') else None,
        'new_model_calls':0,'api_calls':0,'mps_used':False,'published':False}
    fresh_json(OUT/(PREFIX+'-source-manifest.json'),manifest)
    print(json.dumps({'status':'encoded','path':rel(destination),'frames':count,'seconds':sum(DURATIONS)}),flush=True)


def verify():
    manifest=read(OUT/(PREFIX+'-source-manifest.json'));checked=[]
    def visit(value):
        if isinstance(value,dict):
            if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
                path=local(value['path'])
                if sha(path)!=value['sha256']:raise ValueError('Provenance changed: '+value['path'])
                checked.append(value['path'])
            for v in value.values():visit(v)
        elif isinstance(value,list):
            for v in value:visit(v)
    visit(manifest)
    path=local(manifest['output']['path'])
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)],text=True))
    video=next(r for r in probe['streams'] if r['codec_type']=='video');audio=next(r for r in probe['streams'] if r['codec_type']=='audio')
    assert (video['width'],video['height'],video['avg_frame_rate'],video['codec_name'])==(1920,1080,'24/1','h264')
    assert audio['codec_name']=='aac' and audio['channels']==2 and int(audio['sample_rate'])==48000
    assert abs(float(probe['format']['duration'])-54)<.1
    subprocess.run(['ffmpeg','-hide_banner','-v','error','-i',str(path),'-f','null','-'],check=True)
    raw=subprocess.check_output(['ffmpeg','-hide_banner','-v','error','-i',str(path),'-vn','-f','f32le','-acodec','pcm_f32le','-'])
    samples=np.frombuffer(raw,dtype='<f4').reshape(-1,2);peak=float(np.abs(samples).max());rms=float(np.sqrt(np.mean(samples*samples)))
    assert 0<peak<1. and .015<rms<.25
    chunks=[samples[n:n+48000*3] for n in range(0,len(samples),48000*3)]
    audio_check={'decoded_samples_per_channel':len(samples),'peak_dbfs':20*np.log10(peak),'rms_dbfs':20*np.log10(rms),
        'three_second_window_rms_dbfs':[float(20*np.log10(max(1e-12,float(np.sqrt(np.mean(x*x)))))) for x in chunks],
        'first_50ms_rms':float(np.sqrt(np.mean(samples[:2400]**2))),'last_50ms_rms':float(np.sqrt(np.mean(samples[-2400:]**2))),
        'non_silent_background_music':True,'decoded_clipping_samples':int((np.abs(samples)>=1).sum())}
    capture=cv2.VideoCapture(str(path));frames=[];montage=Image.new('RGB',(1440,1620),'#273a46')
    for n,chapter in enumerate(manifest['chapters']):
        t=chapter['start_s']+chapter['duration_s']/2;capture.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,pixels=capture.read()
        if not ok:raise ValueError('Encoded chapter frame did not decode')
        image=Image.fromarray(cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB));frame_path=OUT/f'{PREFIX}-decoded-chapter-{n+1:02d}.jpg';image.save(frame_path,quality=96)
        montage.paste(image.resize((720,405)),((n%2)*720,(n//2)*405));frames.append(record(frame_path,'actual_decoded_encoded_frame',timestamp_s=t))
        if n==0:shutil.copyfile(frame_path,MEDIA/(PREFIX+'-poster.jpg'))
    capture.release();montage_path=OUT/(PREFIX+'-verified-storyboard.jpg');montage.save(montage_path,quality=95)
    result={'format':'overnight-trailer-media-verification','status':'passed','provenance_contracts_checked':len(checked),
        'probe':probe,'fully_decoded_video_and_audio':True,'audio':audio_check,'decoded_frames':frames,
        'verified_storyboard':record(montage_path,'actual_encoded_chapter_montage'),'physical_accuracy_claimed':False}
    fresh_json(OUT/(PREFIX+'-verification.json'),result)
    print(json.dumps({'status':'verified','contracts':len(checked),'audio':audio_check,'storyboard':rel(montage_path)}),flush=True)


def fullframe_qa():
    manifest=read(OUT/(PREFIX+'-source-manifest.json'));path=local(manifest['output']['path'])
    capture=cv2.VideoCapture(str(path));count=0;bad=[];previous=None;chapters={};boundaries=np.cumsum([round(s*FPS) for s in DURATIONS])
    while True:
        ok,pixels=capture.read()
        if not ok:break
        if pixels.shape!=(H,W,3):raise ValueError('Encoded frame dimensions changed')
        grey=cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY);thumb=cv2.resize(grey,(192,108)).astype(float)
        std=float(grey.std());mean=float(grey.mean())
        if std<3 or(mean<2 and std<1):bad.append(count)
        chapter=int(np.searchsorted(boundaries,count,side='right'))+1
        stats=chapters.setdefault(str(chapter),{'frames':0,'material_visual_changes':0,'min_luma_std':1000.,'max_thumbnail_rms_change':0.})
        delta=float(np.sqrt(np.mean((thumb-previous)**2))) if previous is not None else 0.
        stats['frames']+=1;stats['material_visual_changes']+=delta>.1;stats['min_luma_std']=min(stats['min_luma_std'],std);stats['max_thumbnail_rms_change']=max(stats['max_thumbnail_rms_change'],delta)
        previous=thumb;count+=1
    capture.release()
    if count!=round(sum(DURATIONS)*FPS) or bad:raise ValueError('Full-frame quantitative checks failed')
    motion=manifest['motion_sequence'];mapping=[];start=round(sum(DURATIONS[:4])*FPS);frames=round(DURATIONS[4]*FPS)
    for n in range(frames):
        u=n/(frames-1)
        if u<.35:continue
        source_time=motion['frames'][0]['timestamp_s']+(u-.35)*DURATIONS[4]*.5
        index=max(j for j,row in enumerate(motion['frames']) if row['timestamp_s']<=min(source_time,motion['frames'][-1]['timestamp_s']))
        row=motion['frames'][index];mapping.append({'output_frame':start+n,'source_sample_index':row['sample_index'],'source_pts_s':row['timestamp_s']})
    if len({r['source_sample_index'] for r in mapping})!=25:raise ValueError('All25 actual native samples must appear')
    result={'format':'full_encoded_trailer_frame_review','version':2,'status':'passed','video':record(path,'actual_encoded_v2'),
        'decoded_frames':count,'expected_frames':round(sum(DURATIONS)*FPS),'dimensions':[W,H],'fps':FPS,
        'quantitative_black_or_uniform_frames':bad,'chapter_metrics':chapters,'motion_output_sample_map':mapping,'unique_native_lid_samples_shown':25,
        'pose_or_image_interpolation':False,'human_review':'All7 decoded chapter midpoints and the25-sample source/native contact were visually reviewed separately. Quantitative all-frame checks are not human inspection of every pixel.',
        'new_model_calls':0,'api_calls':0,'mps_used':False}
    fresh_json(OUT/(PREFIX+'-fullframe-qa.json'),result)
    print(json.dumps({'status':'passed','decoded_frames':count,'unique_native_lid_samples':25,'bad_frames':bad}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','motion-clip','storyboard','encode','verify','fullframe'])
    parser.add_argument('--screenshots');parser.add_argument('--draft-name',default='draft-01');args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='motion-clip':motion_clip()
    elif args.action=='verify':verify()
    elif args.action=='fullframe':fullframe_qa()
    else:
        if not args.screenshots:raise ValueError('Explicit actual screenshot records required')
        screenshots=read(local(args.screenshots))
        if args.action=='storyboard':storyboard(screenshots,args.draft_name)
        else:encode(screenshots)


if __name__=='__main__':main()
