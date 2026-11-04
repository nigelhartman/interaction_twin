#!/usr/bin/env python3
"""Additive45s same-object pipeline film; no inference or geometry changes.

Four actual representations of the same recording remain visible during the
opening. Editorial panel expansion, exact-mask scan reveals and recorded motion
explain the chain. Source masks are overlays on their exact cited RGB frames.
The complete robot/two-goal/closure and multi-recording beats stay unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from PIL import Image, ImageDraw, ImageOps
import build_flagship_trailer_20261104_v1 as F

ROOT=F.ROOT;A=F.A;W,H,FPS,D=F.W,F.H,F.FPS,F.D
OUT=ROOT/'output/flagship-trailer-20261104-v2'
MEDIA=ROOT/'showcase/media/flagship-20261104-v2'
INTAKE=ROOT/'output/flagship-trailer-20261104-v1/intake-v1/manifest.json'
PREVIOUS=ROOT/'output/flagship-trailer-20261104-v1/cut-v2/verification.json'
STAGES=('Video','SAM masks','Native 3D parts','Connected program')
STAGE_SCOPES=('Real recording','Exact source pixels','SAM 3D drafts','Estimated connection')
COLORS=('#3766ed','#dc8a3d','#8d72d8','#24a4c5')
PROOF_TIMES=(0,.75,1.65,2.45,3.4,4.5,5.5,6.7,8.4,10,14,20,25.8,28.5,32.8,36.5,40.4,44)


def stage_weights(t):
    """Smoothly move the spotlight through all four visible representations."""
    centers=(.7,2.7,4.8,7.2)
    if t<=centers[0]:return(1.,0.,0.,0.)
    if t>=centers[-1]:return(0.,0.,0.,1.)
    for i,(left,right)in enumerate(zip(centers,centers[1:])):
        if t<=right:
            q=A.ease((t-left)/(right-left));out=[0.]*4;out[i]=1-q;out[i+1]=q;return tuple(out)
    raise AssertionError('Unreachable stage')


def panel_boxes(t):
    weights=stage_weights(t);x=66.;boxes=[]
    for weight in weights:
        width=315+480*weight;boxes.append((round(x),246,round(x+width),958));x+=width+16
    assert abs(x-1870)<1e-6
    return boxes


class Assets(F.Assets):
    def __init__(self,intake=INTAKE):
        super().__init__(intake)
        self.intake_ref=F.ref(intake)
        connected_receipt=json.loads(F.checked(self.fresh['connected']['receipt']).read_text())
        # The encoded transition may land either side of the capture's cut.
        # Crop the union of BOTH audited chapters, never only the later one.
        self.connected_crop=F.all_bounds_crop([p for row in connected_receipt['trace']for box in row['projected_native_boxes']for p in box['points']],margin=20)
        self.program_ref=self.records['Earbuds']['interaction'];program=json.loads(F.checked(self.program_ref).read_text())
        assert program['source']['sha256']==self.old.source_sha
        self.mask_rows=[];self.mask_images={};self.mask_refs=[]
        for evidence in program['evidence']:
            if evidence['source_index']not in (75,76,77,78):continue
            frame=self.bound(evidence['frame_path'],evidence['frame_sha256'])
            observations=[]
            for observation in evidence['observations']:
                if observation['track_id']not in ('case_base','case_lid','earbud_a','earbud_b'):continue
                observations.append({**observation,'ref':self.bound(observation['mask_path'],observation['mask_sha256'])})
            assert len(observations)==4
            self.mask_rows.append({'sample':evidence['source_index'],'timestamp_s':evidence['timestamp_s'],'frame':frame,'masks':observations})
        assert len(self.mask_rows)==4 and self.mask_rows[0]['timestamp_s']==24.668833
        self.hero_crop=tuple(self.old.md['editorial_source_crops']['hero'])
        prior=json.loads(F.checked(F.ref(PREVIOUS)).read_text());self.audio_ref=prior['audio'];self.audio_provenance_ref=prior['audio_provenance']
        audio_meta=json.loads(F.checked(self.audio_provenance_ref).read_text())
        assert audio_meta['music_original_additive_synthesis']and not audio_meta['third_party_samples']and not audio_meta['hosted_speech']
        assert len(audio_meta['voice'])==len(F.CUES)and all(v['atempo']==1 for v in audio_meta['voice'])
        for row,cue in zip(audio_meta['voice'],F.CUES):assert row['text']==cue[2]
        F.checked(self.audio_ref)

    def bound(self,path,sha):
        pointer=F.ref(ROOT/path);assert pointer['sha256']==sha,'Cited RGB/mask bytes changed';self.mask_refs.append(pointer);return pointer

    def mask_pair(self,p):
        requested=24.68+max(0.,min(1.,p))*.97
        row=min(self.mask_rows,key=lambda r:abs(r['timestamp_s']-requested));key=row['sample']
        if key not in self.mask_images:
            rgb=Image.open(F.checked(row['frame'])).convert('RGB');overlay=rgb.copy()
            for i,mask in enumerate(row['masks']):
                alpha=Image.open(F.checked(mask['ref'])).convert('L');assert alpha.size==rgb.size
                tint=Image.new('RGB',rgb.size,COLORS[i]);layer=Image.blend(rgb,tint,.57)
                overlay.paste(layer,(0,0),alpha)
            self.mask_images[key]=(rgb.crop(self.hero_crop),overlay.crop(self.hero_crop))
        return(*[p.copy()for p in self.mask_images[key]],row)


def fit_layer(picture,width,height):
    layer=Image.new('RGB',(max(1,width),max(1,height)),'#eef0f5')
    art=ImageOps.contain(picture,layer.size,Image.Resampling.LANCZOS)
    layer.paste(art,((width-art.width)//2,(height-art.height)//2));return layer


def frame(t,a):
    if t>=9:
        im,trace=F.frame(t,a);trace['opening_version']=2;return im,trace
    im=Image.new('RGB',(W,H),F.BG);F.header(im,a.old,t)
    F.headline(im,'A video in. An object program out.')
    p=min(.999,t/9);raw,masked,row=a.mask_pair(p)
    pictures=[a.source(p),masked,a.parts(p),a.connected(p)]
    boxes=panel_boxes(t);weights=stage_weights(t);active=max(range(4),key=weights.__getitem__)
    for i,(box,picture)in enumerate(zip(boxes,pictures)):
        x,y,right,bottom=box;w=right-x
        F.label(im,(x+16,207),f'0{i+1}',18,F.BLUE if i==active else F.MUTED)
        F.text(im,(x+51,202),STAGES[i],24,F.INK if i==active else F.MUTED,True)
        F.card(im,box,line=F.BLUE if i==active else'#dfe3ec')
        inner=(x+12,y+12,right-12,bottom-94);iw=inner[2]-inner[0];ih=inner[3]-inner[1]
        layer=fit_layer(picture,iw,ih)
        if i==1:
            # Source and mask overlay have the same cited pixels/clock. The
            # scan is editorial, not a fabricated live segmentation response.
            source_layer=fit_layer(raw,iw,ih);scan=A.ease(min(1.,max(0.,(t-.65)/2.3)));stop=round(ih*scan)
            source_layer.paste(layer.crop((0,0,iw,stop)),(0,0));layer=source_layer
            if 0<stop<ih:ImageDraw.Draw(layer).line((0,stop,iw,stop),fill=F.BLUE,width=4)
        mask=Image.new('L',layer.size,0);ImageDraw.Draw(mask).rounded_rectangle((0,0,iw-1,ih-1),radius=15,fill=255)
        im.paste(layer,(inner[0],inner[1]),mask)
        F.label(im,(x+23,bottom-66),STAGE_SCOPES[i],21,F.BLUE if i==active else F.MUTED)
        if i==2:
            # Exact four-component recording: labels describe the existing
            # independent WebGL views, never an invented assembly transform.
            F.text(im,(x+23,bottom-35),'Case · lid · two earbuds',16,F.MUTED)
        elif i==3:F.text(im,(x+23,bottom-35),'Parts + joints + actions + evidence',16,F.MUTED)
        elif i==1:F.text(im,(x+23,bottom-35),'Saved SAM 3.1 observations',16,F.MUTED)
        else:F.text(im,(x+23,bottom-35),'One owner-recorded earbud case',16,F.MUTED)
        if i<3:
            cx=right+8;ImageDraw.Draw(im).polygon([(cx-4,575),(cx+3,582),(cx-4,589)],fill=F.BLUE if i<active else'#b6c2db')
    F.label(im,(66,984),'Same object. Separate parts. Reusable interaction.',22,F.MUTED)
    return im,{'film_s':t,'kind':'same_object_visible_pipeline','editorial':True,'visible_stages':list(STAGES),'spotlight_weights':weights,'active_stage':STAGES[active],'source_video_s':24.68+p*.97,'mask_source_sample':row['sample'],'mask_source_s':row['timestamp_s'],'mask_overlay_uses_exact_cited_RGB':True,'source_masks_modified':False,'native_geometry_modified':False,'new_poses':False,'panel_boxes':boxes,'all_four_pipeline_stages_visible':True}


def storyboard(folder,a):
    folder.mkdir(parents=True,exist_ok=False);proofs=[]
    for n,t in enumerate(PROOF_TIMES):
        im,trace=frame(t,a);target=folder/f'frame-{n:02}.jpg';im.save(target,quality=95);proofs.append({'time_s':t,'file':F.ref(target),'trace':trace})
    A.contact_sheet(proofs,folder/'storyboard.jpg')
    F.write(folder/'storyboard.json',{'format':'interaction-twin-same-object-flagship-storyboard','version':2,'duration_s':45,'frames':proofs,'same_object_source':F.ref(a.old.source),'exact_RGB_mask_refs':a.mask_refs,'current_capture_intake':a.intake_ref,'opening_story':'Real source → matching saved SAM masks → actual native component views → connected source-bound replay. All four visible from the first frame; spotlight moves through them.','after9s':'Unchanged current robot/full two-goal/closure and multi-recording/MCP story.','new_API_calls':0,'new_GPU_calls':0,'website_modified':False})
    print(json.dumps({'storyboard':F.ref(folder/'storyboard.json'),'contact_sheet':F.ref(folder/'storyboard.jpg')},indent=2),flush=True)


def render(folder,media,a):
    folder.mkdir(parents=True,exist_ok=False);media.mkdir(parents=True,exist_ok=False)
    (folder/'builder-source.py').write_bytes(Path(__file__).read_bytes());movie=media/'interaction-twin-flagship-45s.mp4'
    command=['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-i',str(F.checked(a.audio_ref)),'-map','0:v','-map','1:a','-t','45','-c:v','libx264','-threads','4','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(movie)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE);traces=[];proofs=[];proof_at={round(t*FPS)for t in PROOF_TIMES};started=time.monotonic()
    try:
        for n in range(D*FPS):
            im,trace=frame(n/FPS,a);traces.append(dict(trace,frame=n))
            if n in proof_at:
                target=folder/f'proof-{n:04}.jpg';im.save(target,quality=95);proofs.append({'time_s':n/FPS,'file':F.ref(target)})
            process.stdin.write(im.tobytes())
        process.stdin.close();error=process.stderr.read();assert process.wait()==0,error.decode()[-2000:]
    except BaseException:process.kill();process.wait();raise
    probe=A.verify_video(movie,{'duration_s':D});A.contact_sheet(proofs,folder/'contact-sheet.jpg')
    A.V.run(['ffmpeg','-nostdin','-v','error','-ss','6.7','-i',str(movie),'-frames:v','1','-q:v','2',str(media/'poster.jpg')])
    old_movie=json.loads(PREVIOUS.read_text())['movie'];previous_media=F.checked(old_movie).parent
    for name in ('captions.vtt','transcript.txt'):(media/name).write_bytes((previous_media/name).read_bytes())
    F.write(folder/'editorial-clock.json',{'frames':traces,'native_geometry_modified':False,'robot_full_source_order_preserved':True})
    provenance={'format':'interaction-twin-flagship-film','version':2,'duration_s':45,'fps':30,'canvas':[W,H],'movie':F.ref(movie),'poster':F.ref(media/'poster.jpg'),'captions':F.ref(media/'captions.vtt'),'transcript':F.ref(media/'transcript.txt'),'builder':F.ref(Path(__file__)),'builder_snapshot':F.ref(folder/'builder-source.py'),'previous_film':old_movie,'reused_audio':a.audio_ref,'audio_provenance':a.audio_provenance_ref,'all_speech_atempo':1,'original_procedural_music':True,'external_media_used':False,'current_catalog':a.catalog_ref,'capture_intake':a.intake_ref,'capture_revalidation':a.capture_validation,'source_program':a.program_ref,'same_object_source':F.ref(a.old.source),'exact_RGB_mask_refs':a.mask_refs,'clock':F.ref(folder/'editorial-clock.json'),'contact_sheet':F.ref(folder/'contact-sheet.jpg'),'probe':probe,'new_API_calls':0,'new_GPU_calls':0,'physical_accuracy_verified':False,'robot_scope':'Complete authored kinematic task; both released goals, closed lid, withdrawn tool. Not physical grasp/contact.','agent_scope':'Saved actual MCP retrieval/task draft; not live LLM or charging execution.','opening_scope':'Editorial panel motion and scan reveal; actual source, unchanged saved masks, native-part WebGL capture and connected replay. No generated source/geometry or new joint values.','website_modified':False,'publication':False,'elapsed_s':time.monotonic()-started}
    F.write(media/'provenance.json',provenance);F.write(folder/'verification.json',provenance)
    print(json.dumps({'movie':F.ref(movie),'provenance':F.ref(media/'provenance.json')},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--storyboard',type=Path);parser.add_argument('--render-folder',type=Path);parser.add_argument('--media-folder',type=Path,default=MEDIA);parser.add_argument('--fresh-captures',type=Path,default=INTAKE);args=parser.parse_args()
    if not(args.storyboard or args.render_folder):parser.error('Choose --storyboard or --render-folder.')
    a=Assets(args.fresh_captures.resolve())
    try:
        if args.storyboard:storyboard(args.storyboard.resolve(),a)
        if args.render_folder:render(args.render_folder.resolve(),args.media_folder.resolve(),a)
    finally:a.close()


if __name__=='__main__':main()
