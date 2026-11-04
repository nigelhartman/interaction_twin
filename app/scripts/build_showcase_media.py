#!/usr/bin/env python3
"""Render a source-linked first trailer from actual saved evidence, offline.

No model inference, asset repair, invented poses, hidden insertion or UI mockup.
Native mesh turntables rotate the inspection camera, not a recovered object.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from alignment_render import render_mesh

OUT=ROOT/'output/showcase-media'
W,H,FPS=1920,1080,24
BG='#f4f1e9';INK='#142c35';MUTED='#597078';TEAL='#087e83';CORAL='#e36f51';LINE='#d7dfdb'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def font(n,bold=False):
    return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if bold else '')+'.ttf',n)


def relative(p):return str(Path(p).resolve().relative_to(ROOT))
def contract(p,kind,**extra):return {'path':relative(p),'sha256':sha(p),'kind':kind,**extra}


def fit_image(im,box,background='white',contain=True):
    x,y,w,h=box
    result=Image.new('RGB',(w,h),background)
    thumb=ImageOps.contain(im,(w,h),Image.Resampling.LANCZOS) if contain else ImageOps.fit(im,(w,h),Image.Resampling.LANCZOS)
    if thumb.mode=='RGBA':result.paste(thumb,((w-thumb.width)//2,(h-thumb.height)//2),thumb)
    else:result.paste(thumb,((w-thumb.width)//2,(h-thumb.height)//2))
    return result


def rounded(im,box,radius=30):
    mask=Image.new('L',im.size);ImageDraw.Draw(mask).rounded_rectangle((0,0,im.width-1,im.height-1),radius,fill=255)
    return im,mask


def card(canvas,im,box,background='white',contain=True):
    panel=fit_image(im,box,background,contain);panel,mask=rounded(panel,box)
    canvas.paste(panel,box[:2],mask)


def text(canvas,xy,value,size=44,color=INK,bold=False,spacing=12):
    ImageDraw.Draw(canvas).multiline_text(xy,value,font=font(size,bold),fill=color,spacing=spacing)


def base(kicker,title,subtitle=None,number=None):
    canvas=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(canvas)
    d.rounded_rectangle((76,57,91,79),7,fill=TEAL)
    text(canvas,(108,49),'INTERACTION TWIN',23,bold=True)
    text(canvas,(78,115),kicker.upper(),21,TEAL,True)
    text(canvas,(76,159),title,65,bold=True,spacing=5)
    if subtitle:text(canvas,(78,244),subtitle,25,MUTED)
    d.line((78,1017,1842,1017),fill=LINE,width=2)
    text(canvas,(78,1031),'VIDEO → PARTS → GEOMETRY → MOTION → EVIDENCE',18,MUTED)
    if number:text(canvas,(1755,1031),number,18,MUTED)
    return canvas


def centered_mesh(primitives,size,angle,reference_rotation=None):
    """Painter display of unchanged triangle topology and model vertex colour."""
    vertices=np.concatenate([np.asarray(p['vertices'],float) for p in primitives]);center=(vertices.min(0)+vertices.max(0))/2
    span=float(np.max(vertices.max(0)-vertices.min(0)))
    if reference_rotation is None:rotation=Rotation.from_euler('xyz',[-18,angle,12],degrees=True).as_matrix()
    else:rotation=Rotation.from_euler('y',angle,degrees=True).as_matrix()@np.asarray(reference_rotation)
    image=Image.new('RGBA',size,(0,0,0,0));draw=ImageDraw.Draw(image)
    triangles=[]
    for primitive in primitives:
        v=(np.asarray(primitive['vertices'])-center)@rotation.T;faces=np.asarray(primitive['faces'])
        colors=np.asarray(primitive['colors'])[:,:3] if primitive.get('colors') is not None else np.tile(np.array(primitive['base_color_factor'][:3]),(len(v),1))
        tri=v[faces];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-12)
        light=np.array([-.3,-.4,-.85]);light/=np.linalg.norm(light);shade=.48+.52*np.abs(normal@light)
        rgb=np.clip(colors[faces].mean(1)*shade[:,None]*255,0,255).astype(np.uint8)
        uv=tri[:,:,:2].copy();uv[:,:,0]=uv[:,:,0]/span*size[0]*.75+size[0]/2;uv[:,:,1]=(1 if reference_rotation is not None else -1)*uv[:,:,1]/span*size[0]*.75+size[1]/2
        for i in range(len(tri)):triangles.append((float(tri[i,:,2].mean()),uv[i],tuple(rgb[i])+(255,)))
    for depth,uv,color in sorted(triangles,key=lambda x:-x[0]):draw.polygon([tuple(p) for p in uv],fill=color)
    return image


def source_lookup(case):
    return read(case/'frames/index.json')['frames']


def mask_overlay(rgb,masks):
    image=np.array(rgb.convert('RGB')).copy()
    colors=[np.array([14,194,185]),np.array([235,130,95]),np.array([115,145,248])]
    for mask,color in zip(masks,colors):
        image[mask]=np.rint(.62*image[mask]+.38*color).astype(np.uint8)
        edge=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0;image[edge]=color
    return Image.fromarray(image)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    live_cases_path=ROOT/'workspace/cases.json'
    cases_path=OUT/'workspace-cases-at-capture.json'
    cases_path.write_bytes(live_cases_path.read_bytes())
    pack=read(cases_path)
    jabra=next(c for c in pack['cases'] if c['title']=='Earbuds & charging case');wallet=next(c for c in pack['cases'] if c['title']=='Wallet & coin')
    scene=jabra['scene'];case=ROOT/'data/recording-jabra-5fps-v1';rows=source_lookup(case);seg=case/'segmentation/automatic-components-v1'
    tracks={t['track_id']:{r['sample_index']:r for r in t['frames']} for t in read(seg/'tracks.json')['tracks']}
    chosen=min(rows,key=lambda r:abs(r['timestamp_s']-18.))
    source=case/chosen['file'];rgb=Image.open(source).convert('RGB');rgb.save(OUT/'source-jabra.jpg',quality=96)
    mask_paths=[seg/tracks[tid][chosen['index']]['raw_mask_file'] for tid in ['container_base','container_lid','small_object_a']]
    masks=[np.array(Image.open(p))>127 for p in mask_paths];mask_overlay(rgb,masks).save(OUT/'mask-jabra.jpg',quality=96)
    sequence=OUT/'mask-sequence';sequence.mkdir(exist_ok=True);mask_sequence=[]
    for row in rows:
        if not 11.99<=row['timestamp_s']<=18.01:continue
        paths=[seg/tracks[tid][row['index']]['raw_mask_file'] for tid in ['container_base','container_lid','small_object_a'] if row['index'] in tracks[tid]]
        observed=[np.array(Image.open(p))>127 for p in paths]
        image=mask_overlay(Image.open(case/row['file']),observed);path=sequence/f'{row["index"]:06d}.jpg';image.save(path,quality=94)
        mask_sequence.append({'file':relative(path),'source_pts_s':row['timestamp_s'],'source_frame':contract(case/row['file'],'source_frame'),
           'mask_inputs':[contract(p,'raw_SAM2_mask') for p in paths]})
    ear=masks[-1];yy,xx=np.nonzero(ear);pad=25;box=(max(0,int(xx.min())-pad),max(0,int(yy.min())-pad),min(rgb.width,int(xx.max())+pad+1),min(rgb.height,int(yy.max())+pad+1))
    cutout=rgb.convert('RGBA');cutout.putalpha(Image.fromarray(ear.astype(np.uint8)*255));cutout.crop(box).save(OUT/'mask-earbud-crop.png')
    wallet_case=ROOT/'data/recording-145141-1fps-v3';wr=min(source_lookup(wallet_case),key=lambda r:abs(r['timestamp_s']-21.))
    Image.open(wallet_case/wr['file']).convert('RGB').save(OUT/'source-wallet.jpg',quality=96)
    manifest={'schema_version':'0.1','project':'Interaction Twin','created_by':'offline source-linked media renderer',
       'inputs':[contract(cases_path,'validated_workspace_payload'),contract(source,'sampled_source_image',source_pts_s=chosen['timestamp_s'])]+[contract(p,'raw_SAM2_mask',source_pts_s=chosen['timestamp_s']) for p in mask_paths],
       'source_videos':[contract(ROOT/jabra['video_path'],'upright_SDR_derivative',original_video_sha256=jabra['interaction']['source'].get('original_sha256'),source_duration_s=jabra['duration_s']),contract(ROOT/wallet['video_path'],'upright_SDR_derivative',source_duration_s=wallet['duration_s'])],
       'display_policy':{'no_new_inference':True,'no_manual_geometry_or_pose':True,'native_turntable':'inspection camera rotates around unchanged learned meshes; not recovered physical motion',
          'conditional_alignment':'only actual supported/reference source timestamps; no interpolation across gaps; source camera/relative scale unverified',
          'appearance':'original generated vertex colours; optional display lighting only; not photographic UV texture',
          'source_playback':'native recorded footage decoded with FFmpeg; frame-rate conversion drops/duplicates frames and does not synthesize new imagery',
          'hidden_insertion_or_seating_recovered':False,'audio':'silent; no unlicensed soundtrack'},
       'actual_models':['SAM2.1Tiny MPS','CoTracker3 MPS','TripoSR MPS, non-Meta comparison','MoGe2Small MPS','hosted OpenAI/Codex reasoning temporary Meta substitute'],
       'not_run':['SAM3.1','SAM3DObjects','Meta reasoning API','cluster GPU'],
       'case_counts':{'jabra':jabra['labels'],'wallet':wallet['labels']},'assets':[]}
    manifest['mask_sequence']=mask_sequence
    turn=OUT/'native-turntable';turn.mkdir(exist_ok=True)
    asset_info=[]
    for obj,name in zip(scene['objects'],['base','lid','earbud']):
        p=Path(obj['provenance']['mesh_file'])
        if sha(p)!=obj['provenance']['mesh_sha256']:raise ValueError('A learned asset changed; refuse stale showcase geometry')
        reference=next((r for r in obj['frames'] if r['visible'] and r['reference']),next(r for r in obj['frames'] if r['visible']))
        matrix=np.array(reference['camera_from_native_mesh']);reference_rotation=matrix[:3,:3]/np.cbrt(np.linalg.det(matrix[:3,:3]))
        im=centered_mesh(obj['primitives'],(700,650),0,reference_rotation);im.save(OUT/f'mesh-{name}.png')
        asset_info.append((obj,name,reference_rotation));manifest['assets'].append(contract(p,'untouched_generated_mesh',object_id=obj['object_id'],geometry_modified=False,
           display_initial_rotation_source_pts_s=reference['timestamp_s'],display_initial_rotation_policy='existing automatic unverified reference orientation; centering is display-only'))
    triptych=Image.new('RGB',(1800,700),'#e9eeeb');draw=ImageDraw.Draw(triptych)
    for j,(obj,name,view) in enumerate(asset_info):
        im=centered_mesh(obj['primitives'],(540,550),0,view);triptych.paste(im,(j*600+30,50),im)
        draw.text((j*600+35,620),name.upper()+' · native inspection',font=font(24,True),fill=INK)
    triptych.save(OUT/'assets-triptych.jpg',quality=96)
    for f in range(30):
        frame=Image.new('RGB',(1800,620),'#e9eeeb')
        for j,(obj,name,view) in enumerate(asset_info):
            im=centered_mesh(obj['primitives'],(560,570),f*12,view);frame.paste(im,(j*600+20,8),im)
        frame.save(turn/f'{f:03d}.jpg',quality=94)
    # Frozen scene payload preserves exact matrices for all media generation.
    write(OUT/'scene-input.json',scene);write(OUT/'case-input.json',{'jabra':{k:jabra[k] for k in ['video_path','duration_s','labels']},'wallet':{k:wallet[k] for k in ['video_path','duration_s','labels']}})
    manifest['inputs'].append(contract(OUT/'scene-input.json','frozen_actual_scene_payload'))
    aligned=OUT/'alignment';aligned.mkdir(exist_ok=True)
    timestamps=set.intersection(*[{round(r['timestamp_s'],6) for r in o['frames'] if r['visible']} for o in scene['objects']])
    matches=sorted(timestamps);alignment=[];width,height=360,640;k=np.array(scene['intrinsics']);k[:2]*=2/3
    rowmap={round(r['timestamp_s'],6):r for r in rows}
    for n,t in enumerate(matches):
        depth=np.full((height,width),np.inf);render=np.full((height,width,3),235,np.uint8)
        for obj in scene['objects']:
            rec=next(r for r in obj['frames'] if abs(r['timestamp_s']-t)<1e-6 and r['visible']);matrix=np.array(rec['camera_from_native_mesh']);scale=np.cbrt(np.linalg.det(matrix[:3,:3]));rotation=matrix[:3,:3]/scale
            for primitive in obj['primitives']:
                colors=np.array(primitive['colors'])[:,:3]*255 if primitive.get('colors') else None
                result=render_mesh(primitive['vertices'],primitive['faces'],k,rotation,matrix[:3,3],width,height,scale=scale,vertex_colors=colors)
                use=result['mask']&(result['depth']<depth);depth[use]=result['depth'][use];render[use]=result['rgb'][use]
        frame=rowmap[t];source_im=Image.open(case/frame['file']).convert('RGB').resize((width,height),Image.Resampling.LANCZOS)
        image=Image.new('RGB',(920,680),'#e9eeeb');image.paste(source_im,(30,20));image.paste(Image.fromarray(render),(530,20))
        image.save(aligned/f'{n:03d}.jpg',quality=95);alignment.append({'file':relative(aligned/f'{n:03d}.jpg'),'source_pts_s':frame['timestamp_s'],'sample_index':frame['index'],
           'all_three_component_poses_conditionally_supported_or_reference':True})
    manifest['alignment_frames']=alignment
    manifest['outputs']=[contract(OUT/name,'showcase_asset') for name in ['source-jabra.jpg','mask-jabra.jpg','source-wallet.jpg','mask-earbud-crop.png','assets-triptych.jpg','mesh-base.png','mesh-lid.png','mesh-earbud.png']]
    write(OUT/'source-manifest.json',manifest)
    print(json.dumps({'status':'assets_ready','out':str(OUT),'alignment_frames':len(alignment),'native_inspection_frames':30}),flush=True)


class VideoClip:
    def __init__(self,path,start,duration,size=(540,960)):
        self.size=size
        self.process=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-ss',str(start),'-i',str(path),'-t',str(duration),'-an',
           '-vf',f'fps={FPS},scale={size[0]}:{size[1]}','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        self.last=None
    def frame(self):
        count=self.size[0]*self.size[1]*3;raw=self.process.stdout.read(count)
        if len(raw)==count:self.last=Image.frombytes('RGB',self.size,raw)
        if self.last is None:raise ValueError('Recorded source video produced no frame')
        return self.last
    def close(self):
        self.process.stdout.close();self.process.terminate() if self.process.poll() is None else None
        self.process.wait(timeout=10)


def trailer_frame(section,u,assets,live=None):
    """Typography/card motion is editorial; source and learned geometry stay real."""
    if section==0:
        c=base('An interaction reconstruction prototype','Ordinary video.\nInspectable structure.',number='01 / 07')
        # Custom hero permits larger hook without competing with the image.
        c=Image.new('RGB',(W,H),BG);text(c,(86,72),'INTERACTION TWIN',25,TEAL,True)
        text(c,(84,270),'Ordinary video.\nInspectable\nstructure.',97,bold=True,spacing=9)
        text(c,(88,640),'Turn recorded interactions into parts,\nconditional 3D and an evidence-linked program.',29,MUTED,spacing=12)
        text(c,(88,944),'A working local prototype',23,TEAL,True)
        card(c,live,(1194,103,536,862))
    elif section==1:
        c=base('01 · Perception','Separate the parts.','Actual SAM masks on the recorded objects.',number='02 / 07')
        card(c,live,(102,326,435,650))
        source_t=12+u*6;mi=max([j for j,r in enumerate(assets['mask_meta']) if r['source_pts_s']<=source_t],default=0)
        card(c,assets['mask_seq'][mi],(668,326,435,650))
        ImageDraw.Draw(c).rounded_rectangle((684,941,1083,978),8,fill=INK)
        text(c,(697,950),f'SAM evidence · {assets["mask_meta"][mi]["source_pts_s"]:.3f} s',19,'white',True)
        text(c,(1230,380),'TRACKED PARTS',23,TEAL,True)
        for j,(label,color) in enumerate([('Case base','#0ec2b9'),('Case lid','#eb825f'),('Earbud','#7391f8')]):
            ImageDraw.Draw(c).ellipse((1230,450+j*92,1250,470+j*92),fill=color);text(c,(1273,438+j*92),label,35,bold=True)
        text(c,(1230,793),'Masks sampled at 5 fps.\nVisible pixels only.\nOcclusion remains unknown.',25,MUTED,spacing=10)
    elif section==2:
        c=base('02 · Geometry','Meet the generated parts.','Untouched learned meshes. Separate native inspection.',number='03 / 07')
        card(c,assets['turn'][min(29,int(u*30))],(72,330,1776,580),'#e9eeeb')
        for j,label in enumerate(['CASE BASE','CASE LID','EARBUD']):text(c,(128+600*j,930),label,23,TEAL,True)
        text(c,(1050,966),'Camera orbit · predicted hidden surfaces',20,MUTED)
    elif section==3:
        c=base('03 · Alignment','Keep the evidence attached.','Recorded source beside the actual conditional mesh projection.',number='04 / 07')
        n=min(len(assets['aligned'])-1,int(u*len(assets['aligned'])));card(c,assets['aligned'][n],(100,323,1020,670),'#e9eeeb')
        text(c,(1220,382),'14 SHARED TIMESTAMPS',25,TEAL,True)
        text(c,(1220,446),'One source clock.\nThree components.\nEvery pose inspectable.',37,bold=True,spacing=15)
        text(c,(1220,710),'Camera and relative scale are estimated.\nNo interpolation across unsupported gaps.\nHidden insertion and seating stay unknown.',24,MUTED,spacing=14)
        text(c,(1220,901),f'Source {assets["alignment_meta"][n]["source_pts_s"]:.6f} s',24,TEAL,True)
    elif section==4:
        c=base('04 · Workspace','Inspect every claim.','Video, 3D, component graph, procedure and source evidence.',number='05 / 07')
        n=min(max(0,len(assets['screens'])-2),int(u*max(1,len(assets['screens'])-1)))
        card(c,assets['screens'][n],(88,312,1744,676),'white')
    elif section==5:
        c=base('05 · A complete local flow','Bring your own video.','Upload → process → inspect → export. Real intermediate artifacts.',number='06 / 07')
        card(c,assets['screens'][-1],(88,312,1300,676),'white')
        text(c,(1452,365),'LOCAL MODELS',22,TEAL,True)
        text(c,(1452,420),'SAM\nTripoSR\nCoTracker',32,bold=True,spacing=20)
        text(c,(1452,686),'Hosted reasoning\nthrough Codex',25,MUTED,spacing=12)
        text(c,(1452,830),'One generated component\nper low-setting upload run.',23,MUTED,spacing=10)
    else:
        c=Image.new('RGB',(W,H),INK);text(c,(88,84),'INTERACTION TWIN',26,'#87d6cc',True)
        text(c,(84,266),'See the parts.\nInspect the motion.\nKeep the evidence.',90,'#f4f1e9',True,spacing=11)
        text(c,(88,671),'From recorded video to an inspectable interaction program.',32,'#b8cbd0')
        text(c,(88,849),'LOCAL PROTOTYPE',22,'#87d6cc',True)
        text(c,(88,905),'Generated geometry and conditional poses · physical contact unverified',25,'#b8cbd0')
        text(c,(88,954),'Gated SAM models have not run. Meta reasoning integration pending.',22,'#b8cbd0')
    return c


def encode(screenshots):
    manifest=read(OUT/'source-manifest.json');cases=read(OUT/'case-input.json')
    if not screenshots:raise ValueError('At least one real app screenshot is required; no mock UI fallback')
    for p in screenshots:
        if not p.is_file():raise ValueError('Real screenshot missing: '+str(p))
    assets={'mask':Image.open(OUT/'mask-jabra.jpg').convert('RGB'),
       'mask_meta':manifest['mask_sequence'],'mask_seq':[Image.open(ROOT/r['file']).convert('RGB') for r in manifest['mask_sequence']],
       'turn':[Image.open(p).convert('RGB') for p in sorted((OUT/'native-turntable').glob('*.jpg'))],
       'aligned':[Image.open(ROOT/r['file']).convert('RGB') for r in manifest['alignment_frames']],
       'alignment_meta':manifest['alignment_frames'],'screens':[Image.open(p).convert('RGB') for p in screenshots]}
    if not assets['aligned']:raise ValueError('Actual shared supported poses required for alignment chapter')
    duration=[4,6,6,6,8,6,6];total=sum(duration)
    output=OUT/'trailer.mp4';log=(OUT/'encode.log').open('w')
    command=['ffmpeg','-hide_banner','-loglevel','warning','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-an',
       '-c:v','libx264','-preset','medium','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
    clips={0:VideoClip(ROOT/cases['jabra']['video_path'],14,4),1:VideoClip(ROOT/cases['jabra']['video_path'],12,6)}
    storyboard=[];start=time.monotonic();frame_id=0
    try:
        for section,seconds in enumerate(duration):
            for f in range(seconds*FPS):
                u=f/max(1,seconds*FPS-1);live=clips[section].frame() if section in clips else None
                frame=trailer_frame(section,u,assets,live)
                # The fine progress rule is editorial only, never a model timeline.
                if section not in {0,6}:ImageDraw.Draw(frame).line((78,1001,78+int((W-156)*(frame_id/(total*FPS))),1001),fill=TEAL,width=4)
                if f==seconds*FPS//2:
                    path=OUT/f'storyboard-{section+1:02d}.jpg';frame.save(path,quality=95);storyboard.append(path)
                process.stdin.write(frame.tobytes());frame_id+=1
            if section in clips:clips[section].close()
        process.stdin.close();code=process.wait(timeout=180)
        if code:raise RuntimeError('Trailer encoder failed; inspect encode.log')
    finally:
        for clip in clips.values():
            if clip.process.poll() is None:clip.close()
        if process.poll() is None:process.terminate();process.wait(timeout=20)
        log.close()
    poster=Image.open(OUT/'storyboard-01.jpg');poster.save(OUT/'poster.jpg',quality=96)
    montage=Image.new('RGB',(1280,720*4),'#e1e6e3')
    for j,p in enumerate(storyboard):montage.paste(Image.open(p).resize((640,360)),((j%2)*640,(j//2)*720//2))
    # Compact seven-frame storyboard, not claimed as an app screenshot.
    montage=montage.crop((0,0,1280,1440));montage.save(OUT/'storyboard.jpg',quality=94)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss','10','-i',str(output),'-t','6','-an','-vf','scale=960:540','-c:v','libx264','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'mesh-loop.mp4')],check=True)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(OUT/'mesh-loop.mp4'),'-an','-c:v','libvpx-vp9','-b:v','0','-crf','34',str(OUT/'mesh-loop.webm')],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(output)],text=True))
    manifest['screenshots']=[contract(p,'actual_app_screenshot') for p in screenshots]
    manifest['trailer']={'file':relative(output),'sha256':sha(output),'duration_s':float(probe['format']['duration']),
       'width':W,'height':H,'fps':FPS,'sections':[{'section':i+1,'duration_s':d} for i,d in enumerate(duration)],
       'source_video_intervals_s':[{'section':1,'start_s':14,'end_s':18},{'section':2,'start_s':12,'end_s':18}],
       'alignment_playback':'14 exact supported/reference observation images displayed sequentially with a held image; no synthesized in-between pose',
       'screen_capture_policy':'Actual screenshots; no fake upload animation or successful model status overlay',
       'render_wall_seconds':round(time.monotonic()-start,3)}
    manifest['outputs']=[o for o in manifest['outputs'] if o['kind']!='showcase_media']
    manifest['outputs']+=[contract(OUT/name,'showcase_media') for name in ['trailer.mp4','poster.jpg','mesh-loop.mp4','mesh-loop.webm','storyboard.jpg']]
    write(OUT/'source-manifest.json',manifest);write(OUT/'ffprobe.json',probe)
    print(json.dumps({'status':'complete','trailer':str(output),'duration_s':total,'frames':frame_id,'wall_seconds':time.monotonic()-start}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','encode']);p.add_argument('--screenshot',type=Path,action='append',default=[]);a=p.parse_args()
    prepare() if a.action=='prepare' else encode([p.resolve() for p in a.screenshot])
if __name__=='__main__':main()
