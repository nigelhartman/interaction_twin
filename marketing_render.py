"""Bounded local Cobalt renders. No user commands, paths, networking or inference."""
from __future__ import annotations
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
from functools import lru_cache
from PIL import Image,ImageDraw,ImageFont
from marketing_sources import checked,file_ref,FACTS,HEADLINES,rendered_text,CTA,validated_brand

FPS=24
RENDERER_VERSION=2
SIZES={'landscape':(1280,720),'portrait':(720,1280),'square':(720,720)}
BG='#f7f8fb';INK='#151923';MUTED='#626b7b';BLUE='#3766ed'

@lru_cache(maxsize=64)
def font(size,bold=False):
    return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold'if bold else'')+'.ttf',size)
def wrap(text,face,width):
    lines=[];line=''
    for word in text.split():
        trial=(line+' '+word).strip()
        if line and face.getlength(trial)>width:lines.append(line);line=word
        else:line=trial
    if line:lines.append(line)
    return lines
def text(im,xy,value,size,width,*,bold=False,color=INK):
    face=font(size,bold);draw=ImageDraw.Draw(im);y=xy[1]
    for line in wrap(value,face,width):draw.text((xy[0],y),line,font=face,fill=color);y+=size*1.19
    return y
def load_brand_artwork(snapshot,root):
    """Return the fixed, self-contained RGBA raster for a saved new style.

    Historical vector geometry stays in logo's original branch. No arbitrary
    SVG parser, URL or raster path can be supplied by a campaign or model.
    """
    saved=validated_brand(snapshot,root)
    reference=saved.get('logo_raster')
    if reference is None:return None
    raw=checked(reference,root).read_bytes()
    if len(raw)!=reference['bytes']or hashlib.sha256(raw).hexdigest()!=reference['sha256']:raise ValueError('Approved logo raster changed during read')
    with Image.open(io.BytesIO(raw))as image:
        if image.format!='PNG'or image.mode!='RGBA'or not 0<image.width==image.height<=1024:raise ValueError('Approved logo needs a square RGBA PNG of at most1024pixels')
        image.load()
        if image.getchannel('A').getextrema()[1]==0:raise ValueError('Approved logo raster cannot be empty')
        return image.copy()

def logo(im,x,y,size,opacity=1.,*,artwork=None):
    if artwork is not None:
        layer=artwork.resize((size,size),Image.Resampling.LANCZOS)
        if opacity<1:layer.putalpha(layer.getchannel('A').point(lambda alpha:round(alpha*max(0.,opacity))))
        im.paste(layer,(x,y),layer);return
    # Exact approved mark.svg geometry, rasterized at4x before downsampling.
    k=4;layer=Image.new('RGBA',(size*k,size*k));d=ImageDraw.Draw(layer);s=size*k/48
    def pts(values):return[(round(a*s),round(b*s))for a,b in values]
    d.rounded_rectangle((0,0,size*k-1,size*k-1),radius=11*s,fill=BLUE)
    d.polygon(pts([(10,27),(24,20),(38,27),(24,35)]),fill='#c1d1ff')
    d.polygon(pts([(10,27),(11,36),(24,43),(24,35)]),fill='#eff4ff')
    d.polygon(pts([(38,27),(37,36),(24,43),(24,35)]),fill='#eff4ff')
    d.line(pts([(10,13),(24,6),(38,13),(24,21),(10,13)]),fill='white',width=round(1.6*s),joint='curve')
    d.line(pts([(11,14),(11,19),(24,26),(37,19),(37,14)]),fill='white',width=round(1.6*s),joint='curve')
    for at in range(10,38,5):d.line(pts([(at,24),(min(at+2,38),24)]),fill='#a6bdff',width=round(1.4*s))
    d.ellipse((7.5*s,21.5*s,12.5*s,26.5*s),fill='white')
    layer=layer.resize((size,size),Image.Resampling.LANCZOS)
    if opacity<1:layer.putalpha(layer.getchannel('A').point(lambda alpha:round(alpha*max(0.,opacity))))
    im.paste(layer,(x,y),layer)
def frame(path,size,*,headline=None,caption=None,limitation=None,opening=False,closing=False,source_kind=None,source_size=None,logo_progress=1.,logo_artwork=None):
    w,h=size;im=Image.new('RGB',size,BG);m=round(min(w,h)*.08);d=ImageDraw.Draw(im)
    if opening or closing:
        icon=round(min(w,h)*.15);x=m;y=round(h*.28)
        progress=max(0.,min(1.,logo_progress));smooth=progress*progress*(3-2*progress)
        actual=round(icon*(.94+.06*smooth));opacity=min(1.,progress/.25)
        logo(im,x+(icon-actual)//2,y+(icon-actual)//2,actual,opacity=opacity,artwork=logo_artwork)
        y=text(im,(m,y+icon+30),'Interaction Twin',round(min(w,h)*.066),w-2*m,bold=True)
        text(im,(m,y+18),'Show it once. Keep how it works.',round(min(w,h)*.036),w-2*m,color=MUTED)
        if closing:text(im,(m,round(h*.73)),CTA,round(min(w,h)*.033),w-2*m,bold=True,color=BLUE)
        body=None
    elif source_kind in {'pipeline_film','robot_preview'}:
        # Source already contains the product UI/branding. Never nest that
        # evidence inside another logo, headline, rounded card and tiny video.
        footer_h=70 if w>=1000 else 92;left=round(w*.04);top=h-footer_h
        d.rectangle((0,top,w,h),fill=BG)
        end=text(im,(left,top+10),caption,14,w-2*left,color=INK)
        if limitation:text(im,(left,end+4),limitation,12,w-2*left,color=MUTED)
        body=(0,0,w,top)
    elif w>h and source_size and source_size[0]<source_size[1]*.85:
        # Preserve the complete portrait recording, now at full usable height.
        # The left column carries the short editorial description.
        icon=30;logo(im,m,m,icon,artwork=logo_artwork);text(im,(m+44,m+4),'INTERACTION TWIN',16,round(w*.44)-m-44,bold=True,color=BLUE)
        width=round(w*.43)-m
        end=text(im,(m,round(h*.27)),headline,round(h*.061),width,bold=True)
        end=text(im,(m,end+24),caption,round(h*.029),width,color=MUTED)
        if limitation:text(im,(m,end+20),limitation,round(h*.021),width,color=MUTED)
        x=round(w*.53);body=(x,m,w-m-x,h-2*m)
    else:
        icon=30;logo(im,m,m,icon,artwork=logo_artwork);text(im,(m+44,m+4),'INTERACTION TWIN',16,w-2*m-44,bold=True,color=BLUE)
        top=text(im,(m,m+62),headline,round(min(w,h)*.052),w-2*m,bold=True)
        footer=round(h*.79)if h>w else h-round(min(w,h)*.27)
        d.rounded_rectangle((m,round(top+24),w-m,footer-20),radius=16,fill='white',outline='#dfe3eb')
        body=(m+10,round(top+34),w-2*m-20,footer-20-round(top+34)-10)
        end=text(im,(m,footer),caption,round(min(w,h)*.026),w-2*m,color=MUTED)
        if limitation:text(im,(m,end+8),limitation,round(min(w,h)*.021),w-2*m,color=MUTED)
        if not limitation:text(im,(m,h-m-14),'SOURCE-LINKED EVIDENCE · ESTIMATES, NOT PHYSICS',11,w-2*m,color=MUTED)
    im.save(path);return body

def brand_sequence(folder,size,*,closing=False,logo_artwork=None):
    """Exactly48 actual frames, not a static logo with an animation claim."""
    folder.mkdir(mode=0o700,exist_ok=False)
    for index in range(48):frame(folder/f'{index:04}.png',size,opening=not closing,closing=closing,logo_progress=index/47,logo_artwork=logo_artwork)
    return folder/'%04d.png'
def run(args,log,timeout=240):
    result=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
    with log.open('ab')as out:out.write(result.stdout[-100000:])
    if result.returncode:raise ValueError('Local media renderer failed; private render log retained')
def probe(path):
    raw=subprocess.run(['ffprobe','-v','error','-protocol_whitelist','file,pipe','-show_format','-show_streams','-of','json',str(path)],capture_output=True,check=True,timeout=15).stdout
    if len(raw)>64000:raise ValueError('Bounded video metadata required')
    return json.loads(raw)
def vtt_time(seconds):
    ms=round(seconds*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02}.{ms%1000:03}'

def render(revision,format_id,output,*,root,progress=lambda _:None):
    from marketing_studio import validate_storyboard
    from marketing_music import MusicStore
    root=Path(root).resolve();output=Path(output)
    if format_id not in SIZES or output.resolve()!=output or not output.is_relative_to(root/'.private/marketing-studio/renders')or any(p.is_symlink()for p in [output,*output.parents]):raise ValueError('Fixed private render destination required')
    output.mkdir(mode=0o700,exist_ok=False);w,h=SIZES[format_id];log=output/'render.log'
    validate_storyboard(revision['storyboard'],revision['sources'],revision['duration_s'])
    compositor=rendered_text(revision['storyboard'],revision['sources'],revision.get('preferences',{}))
    if revision.get('render_text',compositor)!=compositor:raise ValueError('Frozen rendered copy differs from the mandatory caption contract')
    logo_artwork=load_brand_artwork(revision['brand'],root)
    for src in revision['sources'].values():checked(src['file'],root)
    music_path=None
    if revision['music_id']is not None:
        store=MusicStore(root);music_path=store.file(revision['music_id'])
        if store.describe(revision['music_id'])!=revision['music']:raise ValueError('Soundtrack provenance changed')
    common=['ffmpeg','-nostdin','-v','error','-y','-filter_complex_threads','1']
    encode=['-an','-r',str(FPS),'-c:v','libx264','-preset','veryfast','-crf','21','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart']
    segments=[];captions=[];inputs=[];source_sizes={}
    for sid,src in revision['sources'].items():
        meta=probe(checked(src['file'],root));stream=next((s for s in meta['streams']if s['codec_type']=='video'),None)
        if stream is None or stream.get('width',0)<=0 or stream.get('height',0)<=0:raise ValueError('Declared source needs a bounded video stream')
        source_sizes[sid]=(stream['width'],stream['height'])
    for index,entry in enumerate([None,*revision['storyboard'],None]):
        opening=index==0;closing=index==len(revision['storyboard'])+1;duration=2. if entry is None else entry['duration_s']
        base=output/f'layout-{index:02}.png';target=output/f'segment-{index:02}.mp4'
        if entry is None:
            pattern=brand_sequence(output/f'brand-{index:02}',(w,h),closing=closing,logo_artwork=logo_artwork)
            run(common+['-framerate',str(FPS),'-i',str(pattern),'-frames:v','48','-t',str(duration)]+encode+[str(target)],log)
        else:
            src=revision['sources'][entry['source_id']];path=checked(src['file'],root)
            words=compositor['scenes'][index-1]
            x,y,bw,bh=frame(base,(w,h),headline=words['headline'],caption=words['caption'],limitation=words['limitation'],source_kind=src['kind'],source_size=source_sizes[entry['source_id']],logo_artwork=logo_artwork)
            filt=f'[1:v]fps={FPS},scale={bw}:{bh}:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1[v];[0:v][v]overlay=x={x}+({bw}-w)/2:y={y}+({bh}-h)/2:shortest=1,format=yuv420p[out]'
            run(common+['-loop','1','-i',str(base),'-ss',str(entry['start_s']),'-t',str(duration),'-i',str(path),'-filter_complex',filt,'-map','[out]','-t',str(duration)]+encode+[str(target)],log)
            layout='branded_evidence_full_frame'if src['kind']in {'pipeline_film','robot_preview'}else'portrait_capture_two_columns'if w>h and source_sizes[entry['source_id']][0]<source_sizes[entry['source_id']][1]*.85 else'capture_card'
            inputs.append({'source_id':entry['source_id'],'file':src['file'],'start_s':entry['start_s'],'duration_s':duration,'fit':'contain','speed':1.0,'layout':layout,'source_dimensions':list(source_sizes[entry['source_id']]),'evidence_box':[x,y,bw,bh],'outer_headline_rendered':layout!='branded_evidence_full_frame','headline_id':entry['headline_id'],'fact_id':entry['fact_id'],'limitation':src['limitation']})
        segments.append((target,duration));progress((index+1)/(len(revision['storyboard'])+4)*.8)
    # Every source becomes the same frame rate/size/timebase before fixed xfade.
    args=common[:];filters=[];elapsed=segments[0][1]
    for i,(path,_)in enumerate(segments):args+=['-i',str(path)];filters.append(f'[{i}:v]settb=AVTB,setpts=PTS-STARTPTS[v{i}]')
    previous='v0'
    for i in range(1,len(segments)):
        transition='fade'
        if i>1:
            requested=revision['storyboard'][min(i-2,len(revision['storyboard'])-1)]['transition']
            transition='wipeleft'if requested=='cobalt_wipe'else'fade'
        expr=":expr='if(gte(P\\,0.5)\\,A\\,B)'"if i>1 and requested=='cut'else''
        if expr:transition='custom'
        filters.append(f'[{previous}][v{i}]xfade=transition={transition}:duration=0.25:offset={elapsed-.25:.6f}{expr}[x{i}]')
        previous=f'x{i}';elapsed+=segments[i][1]-.25
    silent=output/'picture.mp4'
    run(args+['-filter_complex',';'.join(filters),'-map',f'[{previous}]','-t',str(revision['duration_s'])]+encode+[str(silent)],log,timeout=360)
    progress(.9);video=output/('interaction-twin-'+format_id+'.mp4')
    if music_path:
        fade=max(0,revision['duration_s']-1.2)
        audio=f'loudnorm=I=-22:TP=-2:LRA=7,afade=t=in:st=0:d=0.6,afade=t=out:st={fade}:d=1.2,alimiter=limit=0.794'
        run(common+['-i',str(silent),'-stream_loop','-1','-i',str(music_path),'-map','0:v:0','-map','1:a:0','-c:v','copy','-af',audio,'-c:a','aac','-b:a','128k','-t',str(revision['duration_s']),'-movflags','+faststart',str(video)],log)
    else:silent.rename(video)
    metadata=probe(video);streams=metadata['streams'];v=next(s for s in streams if s['codec_type']=='video');duration=float(metadata['format']['duration'])
    if (v['width'],v['height'])!=(w,h)or abs(duration-revision['duration_s'])>.12 or bool(music_path)!=any(s['codec_type']=='audio'for s in streams):raise ValueError('Rendered video does not match the declared format/duration/audio')
    poster=output/'poster.jpg';run(['ffmpeg','-nostdin','-v','error','-i',str(video),'-ss','3','-frames:v','1','-q:v','2',str(poster)],log)
    at=1.75;vtt=['WEBVTT','','intro','00:00:00.000 --> 00:00:01.750','Interaction Twin. Show it once. Keep how it works.','']
    for n,entry in enumerate(revision['storyboard'],1):
        end=min(revision['duration_s']-1.75,at+entry['duration_s'])
        words=compositor['scenes'][n-1]
        vtt.extend([str(n),vtt_time(at)+' --> '+vtt_time(end),words['headline']+' '+words['caption']+' '+words['limitation'],'']);at=end-.25
    vtt.extend(['outro',vtt_time(revision['duration_s']-1.75)+' --> '+vtt_time(revision['duration_s']),CTA,''])
    captions_path=output/'captions.vtt';captions_path.write_text('\n'.join(vtt))
    manifest={'format':'interaction-twin-marketing-render','version':2,'renderer_version':RENDERER_VERSION,'format_id':format_id,'width':w,'height':h,'fps':FPS,'duration_s':duration,'intro_s':2,'outro_s':2,'transition_s':.25,'brand_animation':{'frames_per_opening_or_closing':48,'scale_from':.94,'scale_to':1.,'fade':'first12frames','geometry':'same approved mark.svg'},'sources':inputs,'brand':revision['brand'],'music':revision['music'],'music_mix':{'target_lufs':-22,'normalization':'FFmpeg single-pass loudnorm target; integrated output not independently certified','fade_in_s':.6,'fade_out_s':1.2,'peak_limiter':.794},'logo_rasterization':'Exact approved mark.svg vector coordinates rendered with Pillow at4x; no alternative logo.','copy':revision['copy'],'render_text':compositor,'artifacts':{'video':file_ref(video,root),'poster':file_ref(poster,root),'captions':file_ref(captions_path,root)},'physical_accuracy_verified':False,'simulation_performed':False,'publication_performed':False}
    if logo_artwork is not None:
        manifest.update(version=3,renderer_version=3,logo_rasterization='Approved versioned RGBA PNG; SVG and raster bytes are pinned in the saved brand.')
        manifest['brand_animation']['geometry']='same approved versioned mark raster'
    manifest_path=output/'render-manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    for path in output.rglob('*'):path.chmod(0o700 if path.is_dir() else 0o600)
    progress(.99)
    return {'video':file_ref(video,root),'poster':file_ref(poster,root),'captions':file_ref(captions_path,root),'manifest':file_ref(manifest_path,root),'duration_s':duration,'format_id':format_id,'audio':bool(music_path),'publication_performed':False}
