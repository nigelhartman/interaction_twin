#!/usr/bin/env python3
"""Build original research figures from preserved project evidence and assets.

Only editorial image crops/resizing are performed. No source model, geometry,
mask or pose is changed. Output SVGs embed their actual image pixels, and are
standalone. Optional PNG rendering uses CairoSVG when installed.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import html
import io
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/showcase-graphics'
MEDIA=ROOT/'output/showcase-media'
CASE=ROOT/'data/recording-jabra-5fps-v1'
SEG=CASE/'segmentation/automatic-components-v1'
PROGRAM=ROOT/'data/jabra-stage6-v1/procedure-v1/interaction.json'
COLORS={'ink':'#17283d','sub':'#657487','line':'#dce4ed','teal':'#087d78','purple':'#7258ad','amber':'#b96c16','navy':'#122236','muted':'#8993a1'}


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def esc(value):return html.escape(str(value),quote=True)


class SVG:
    def __init__(self,width,height,title,description):
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc"><title id="title">{esc(title)}</title><desc id="desc">{esc(description)}</desc>',
                    '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#6c8194"/></marker></defs>',
                    '<rect width="100%" height="100%" fill="#fff"/>']
        self.clip=0
    def rect(self,x,y,w,h,fill='#f5f8fb',stroke='none',radius=12,dash=None):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
    def text(self,x,y,value,size=18,color=None,weight=400,anchor='start',font='Arial, Helvetica, sans-serif'):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" font-weight="{weight}" fill="{color or COLORS["ink"]}" text-anchor="{anchor}">{esc(value)}</text>')
    def lines(self,x,y,values,size=18,color=None,weight=400,gap=None):
        for i,value in enumerate(values):self.text(x,y+i*(gap or size*1.5),value,size,color,weight)
    def line(self,x1,y1,x2,y2,color='#b8c6d4',width=2,dash=None,arrow=False):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+(' marker-end="url(#arrow)"' if arrow else '')+'/>')
    def circle(self,x,y,r,fill,stroke='none',width=1):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')
    def image(self,image,x,y,w,h,label,radius=8,contain=False):
        self.clip+=1;identifier=f'clip{self.clip}'
        if isinstance(image,(str,Path)):image=Image.open(image).convert('RGBA')
        else:image=image.convert('RGBA')
        stream=io.BytesIO();image.save(stream,format='PNG')
        url='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()
        self.parts.append(f'<defs><clipPath id="{identifier}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}"/></clipPath></defs>')
        self.parts.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid {"meet" if contain else "slice"}" clip-path="url(#{identifier})" xlink:href="{url}" aria-label="{esc(label)}"/>')
    def save(self,name):
        path=OUT/name;path.write_text(''.join(self.parts)+'</svg>');return path


def crop_source(sample):
    """Automatically bound the displayed illustration by existing raw masks."""
    rows=read(CASE/'frames/index.json')['frames'];row=next(r for r in rows if r['index']==sample)
    source=CASE/row['file'];image=Image.open(source).convert('RGB')
    union=np.zeros((image.height,image.width),bool);inputs=[]
    for track in read(SEG/'tracks.json')['tracks']:
        frame=next((f for f in track['frames'] if f['sample_index']==sample),None)
        if frame:
            mask=SEG/frame['raw_mask_file'];union|=np.asarray(Image.open(mask))>0
            inputs.append({'path':str(mask.relative_to(ROOT)),'sha256':sha(mask)})
    ys,xs=np.nonzero(union)
    if not len(xs):raise ValueError('No raw mask support for editorial source crop')
    pad=70
    box=[max(0,int(xs.min())-pad),max(0,int(ys.min())-pad),min(image.width,int(xs.max())+pad+1),min(image.height,int(ys.max())+pad+1)]
    crop=image.crop(box)
    metadata={'sample_index':sample,'timestamp_s':row['timestamp_s'],'source_frame':str(source.relative_to(ROOT)),
              'source_frame_sha256':sha(source),'editorial_crop_xyxy':box,'crop_used_for_inference':False,'mask_inputs':inputs}
    return crop,metadata


def label(s,x,y,text,color,width):
    s.rect(x,y,width,30,fill='#ffffff',stroke=color,radius=15)
    s.circle(x+14,y+15,4,color)
    s.text(x+25,y+20,text,13,color,600)


def pipeline(doc):
    s=SVG(1800,1050,'Interaction Twin: from captured video to an evidence-linked interaction program',
          'Four phases show an actual Jabra source frame, raw SAM component-mask overlay, genuine learned meshes with conditional pose samples, and the actual candidate relation and procedure. Hidden seating remains unknown. Current local models and prepared future adapters are separated.')
    s.text(48,59,'VIDEO → COMPONENTS → INTERACTION',15,COLORS['teal'],700)
    s.text(48,111,'Recover a reviewable interaction, one piece of evidence at a time.',35,weight=700)
    s.text(48,147,'Actual Jabra recording · automatic model outputs · source-linked states · explicit uncertainty',19,COLORS['sub'])
    xs=[48,486,924,1362];width=390;y=192;bottom=811
    headings=[('01','Capture','Recorded pixels + source time'),('02','Perceive','Component masks + tracklets'),('03','Reconstruct & align','Learned shape + conditional pose'),('04','Compile','Evidence-linked interaction program')]
    for x,(num,title,subtitle) in zip(xs,headings):
        s.rect(x,y,width,bottom-y,'#f5f8fb','#dce4ed',14)
        s.text(x+22,y+36,num,15,COLORS['teal'],700);s.text(x+62,y+38,title,25,weight=700)
        s.text(x+22,y+72,subtitle,16,COLORS['sub'])
    for x in xs[:-1]:s.line(x+width+7,480,x+width+34,480,arrow=True)
    # Exact source pixels and model masks provided by the media builder.
    s.image(MEDIA/'source-jabra.jpg',70,286,346,310,'Actual Jabra source frame at 18.001578 seconds',contain=True)
    s.text(70,626,'18.001578 s',17,COLORS['teal'],700)
    s.lines(70,660,['540 × 960 upright source','169 samples at 5 fps','Original presentation time preserved'],17,COLORS['sub'])
    s.rect(70,757,346,31,'#e6f3ef',radius=6);s.text(85,779,'SOURCE · user-recorded video',13,COLORS['teal'],700)
    s.image(MEDIA/'mask-jabra.jpg',508,286,346,310,'Actual SAM mask overlay at the same source timestamp',contain=True)
    s.text(508,626,'439 raw component masks',17,COLORS['teal'],700)
    s.lines(508,660,['Base · lid · object tracklets','Mask + visibility + source hashes','Occlusion never confirms absence'],17,COLORS['sub'])
    s.rect(508,757,346,31,'#e6f3ef',radius=6);s.text(523,779,'META · SAM 2.1 Tiny + CoTracker3',13,COLORS['teal'],700)
    # Native mesh previews are real generated geometry, without assembly poses.
    for i,(name,title) in enumerate([('base','Base'),('lid','Lid'),('earbud','Earbud')]):
        x=946+i*116;s.rect(x,286,108,151,'#e9edf3',radius=8)
        s.image(MEDIA/f'mesh-{name}.png',x+5,293,98,111,f'Actual unmodified learned {name} mesh in native inspection',contain=True)
        s.text(x+54,423,title,14,COLORS['sub'],600,anchor='middle')
    s.text(946,473,'Conditional motion samples',17,COLORS['ink'],700)
    # Samples, not an invented continuous trajectory.
    tracks=[('Base','container_base',COLORS['teal']),('Lid','container_lid',COLORS['amber']),('Earbud','small_object_a',COLORS['purple'])]
    for j,(name,tid,color) in enumerate(tracks):
        yy=511+j*35;s.text(946,yy+5,name,14,COLORS['sub'])
        supported=[m for m in doc['motion'] if m['track_id']==tid and m['status']=='supported_conditional_motion_candidate']
        for sample in supported:
            xx=1016+sample['timestamp_s']/doc['source']['duration_s']*212
            s.circle(xx,yy,2.4,color)
        s.text(1250,yy+5,str(len(supported)),14,color,700)
    s.text(1016,609,'0 s',10,COLORS['sub'])
    s.text(1122,609,'source time',10,COLORS['sub'],anchor='middle')
    s.text(1228,609,'33.6 s',10,COLORS['sub'],anchor='end')
    s.text(946,628,'14 exact times support all three',17,COLORS['purple'],700)
    s.lines(946,662,['Generated geometry; inferred backs','Fixed relative depth gauge','Scale and contact unverified'],17,COLORS['sub'])
    s.rect(946,757,346,31,'#eeeaf5',radius=6);s.text(961,779,'TripoSR + MoGe · local comparisons',13,COLORS['purple'],700)
    # Actual graph edge, deliberately inferred, not a recovered joint.
    s.rect(1384,287,119,55,'#e6f3ef',COLORS['teal'],8);s.text(1443,320,'lid',17,COLORS['teal'],600,anchor='middle')
    s.line(1509,314,1602,314,COLORS['amber'],2,'6 5',True)
    s.text(1554,290,'inferred',12,COLORS['amber'],600,anchor='middle')
    s.rect(1610,287,120,55,'#e6f3ef',COLORS['teal'],8);s.text(1670,320,'base',17,COLORS['teal'],600,anchor='middle')
    s.text(1384,371,'attached_to · exact hinge unknown',14,COLORS['sub'])
    s.rect(1384,395,346,200,COLORS['navy'],radius=10)
    code=['tracklet: small_object_a','state @ 22.001933 s:','  held_near_left_opening','pose @ 22.001933 s: absent','','seating: unknown']
    for i,line in enumerate(code):s.text(1401,425+i*27,line,15,'#dbebf6',font='Menlo, Consolas, monospace')
    s.text(1384,628,'14 steps · 43 point assertions',17,COLORS['ink'],700)
    s.lines(1384,662,['Click a statement → source evidence','Inspect geometry and pose gaps','Export JSON · text · GLB'],17,COLORS['sub'])
    s.rect(1384,757,346,31,'#e7edf4',radius=6);s.text(1399,779,'Hosted Codex · temporary reasoning',13,COLORS['ink'],700)
    s.line(48,853,1752,853,COLORS['line'],1)
    label(s,48,880,'EXECUTED LOCALLY',COLORS['teal'],202)
    s.text(268,901,'SAM 2.1 Tiny · CoTracker3 · TripoSR · MoGe-2 Small',18,COLORS['ink'])
    s.text(48,957,'Prepared upgrade path',18,COLORS['sub'],700)
    s.line(276,951,330,951,COLORS['muted'],2,'5 5',True)
    s.text(350,957,'SAM 3.1 · SAM 3D Objects · Meta reasoning API',18,COLORS['sub'])
    s.text(350,991,'Runtime / authentication / actual inference pending. These outputs are not claimed.',15,COLORS['sub'])
    s.text(1752,1030,'Figure 1 · actual assets, editorial arrangement',12,COLORS['sub'],anchor='end')
    return s.save('pipeline.svg')


def evidence(doc):
    s=SVG(1800,1150,'What a video establishes, and what remains unknown',
          'Three actual source crops illustrate approach, hand occlusion and a later anonymous shape visible in the left recess. Model visual-state proposals cite exact source samples. The first earbud identity ends before the later contents state. Geometry is generated, physical seating is unknown, and states do not persist between samples.')
    s.text(48,59,'THE REPRESENTATION',15,COLORS['teal'],700)
    s.text(48,111,'A useful twin keeps its evidence and its unknowns together.',35,weight=700)
    label(s,48,149,'RECORDED · pixels + time',COLORS['teal'],263)
    label(s,334,149,'PREDICTED · masks + shape',COLORS['purple'],286)
    label(s,643,149,'PROPOSED · visual state',COLORS['amber'],267)
    label(s,933,149,'UNKNOWN · hidden contact',COLORS['muted'],295)
    columns=[(48,111,'22.001933 s','Approaching the opening','held_near_left_opening'),
             (636,117,'23.202033 s','Fingers obscure the candidate','partly_occluded_at_left_opening'),
             (1224,126,'25.002200 s','A shape is visible inside','anonymous left-recess contents')]
    crops=[]
    for x,sample,time,title,state in columns:
        crop,metadata=crop_source(sample);crops.append(metadata)
        s.rect(x,218,528,458,'#f5f8fb','#dce4ed',12)
        s.image(crop,x+16,234,496,304,f'Actual source sample {sample} at {time}; editorial mask-union crop',contain=True)
        s.text(x+22,568,time,17,COLORS['teal'],700)
        s.text(x+22,602,title,22,weight=700)
        s.text(x+22,636,state,16,COLORS['amber'],600)
    # A visible relationship across a gap does not imply the same physical object.
    s.text(48,724,'One tracklet. A hidden interval. A later contents observation.',23,weight=700)
    s.line(75,783,1715,783,COLORS['line'],3)
    for xx in [175,790,1570]:s.circle(xx,783,9,'#fff',COLORS['teal'],3)
    s.line(800,783,1558,783,COLORS['muted'],2,'6 7')
    s.text(175,819,'object A · visible relation',16,COLORS['sub'],anchor='middle')
    s.text(790,819,'object A · partly occluded',16,COLORS['sub'],anchor='middle')
    s.text(1570,819,'container contents · anonymous',16,COLORS['sub'],anchor='middle')
    s.rect(961,748,388,29,'#fff',radius=0)
    s.text(1155,769,'identity after 24.002111 s is unverified',15,COLORS['muted'],600,anchor='middle')
    s.rect(48,861,1020,225,COLORS['navy'],radius=12)
    s.text(72,900,'EVIDENCE PROGRAM',14,'#80d1c7',700)
    left=['state(object_a, 22.001933) = held_near_left_opening',
          'pose(object_a, 22.001933)  = absent',
          'state(object_a, 22.101933) = unknown',
          'contents(base, 25.002200)  = anonymous_shape']
    for i,line in enumerate(left):s.text(72,935+i*34,line,18,'#dceaf5',font='Menlo, Consolas, monospace')
    s.rect(1100,861,652,225,'#f5f8fb','#dce4ed',12)
    s.text(1124,900,'WHAT IS NOT RECOVERED',14,COLORS['muted'],700)
    s.lines(1124,942,['Hidden insertion path · mechanical seating','Metric contact · exact hinge · physical dynamics'],21,COLORS['ink'],gap=42)
    s.text(1124,1049,'Generated geometry does not turn these into observed facts.',16,COLORS['sub'])
    s.text(48,1128,'Figure 2 · exact-source visual proposals, not physical execution · source crops only; model outputs remain unchanged',14,COLORS['sub'])
    return s.save('evidence-program.svg'),crops


def render_pngs(paths):
    """Render original vectors; explicitly record the local display backend."""
    try:
        import cairosvg
    except ImportError:
        # Quick Look wrongly crops a wide root SVG into its square thumbnail.
        # A square QA-only canvas preserves the exact original drawing scale;
        # cropping the final white canvas restores its real page dimensions.
        heights={'pipeline':1050,'evidence-program':1150}
        with tempfile.TemporaryDirectory(prefix='interaction-twin-figure-qa-') as temp:
            temp=Path(temp);wrappers=[]
            for path in paths:
                height=heights[path.stem]
                wrapper=temp/path.name
                wrapper.write_text(path.read_text().replace(f'width="1800" height="{height}" viewBox="0 0 1800 {height}"',
                                                            'width="1800" height="1800" viewBox="0 0 1800 1800"',1))
                wrappers.append(wrapper)
            subprocess.run(['/usr/bin/qlmanage','-t','-s','1800','-o',str(temp),*[str(p) for p in wrappers]],check=True,capture_output=True,text=True)
            for path in paths:
                source=Image.open(temp/(path.name+'.png'))
                if source.size!=(1800,1800):raise ValueError('Unexpected Quick Look canvas dimensions')
                source.crop((0,0,1800,heights[path.stem])).save(path.with_suffix('.png'))
        return [p.with_suffix('.png') for p in paths],'macOS_Quick_Look_square_canvas_then_exact_page_crop'
    else:
        for path in paths:cairosvg.svg2png(url=str(path),write_to=str(path.with_suffix('.png')))
        return [p.with_suffix('.png') for p in paths],'CairoSVG'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--png',action='store_true',help='Also render PNGs using available CairoSVG or native macOS Quick Look')
    args=parser.parse_args()
    required=[MEDIA/name for name in ['source-jabra.jpg','mask-jabra.jpg','mesh-base.png','mesh-lid.png','mesh-earbud.png']]
    missing=[str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing:raise SystemExit('Actual media assets are required; no illustrative model replacement: '+', '.join(missing))
    OUT.mkdir(parents=True,exist_ok=True)
    doc=read(PROGRAM)
    paths=[pipeline(doc)];second,crops=evidence(doc);paths.append(second)
    renders,backend=render_pngs(paths) if args.png else ([],None)
    manifest={'status':'original_vector_figures_from_actual_artifacts','semantic_program':str(PROGRAM.relative_to(ROOT)),
              'semantic_program_sha256':sha(PROGRAM),'source_video_sha256':doc['source']['sha256'],
              'inputs':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in required],
              'editorial_source_crops':crops,'outputs':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in paths+renders],
              'geometry_or_pose_changed':False,'new_model_inference_performed':False,'physical_completion_verified':False,
              'timeline_policy':'actual supported source-timestamp marks; no invented continuous trajectory',
              'research_novelty_claimed':False,'png_render_backend':backend,
              'media_source_manifest':{'path':str((MEDIA/'source-manifest.json').relative_to(ROOT)),
                                       'sha256':sha(MEDIA/'source-manifest.json')}}
    write=OUT/'manifest.json';write.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'figures':[str(p.relative_to(ROOT)) for p in paths],'png_previews':[str(p.relative_to(ROOT)) for p in renders]},indent=2))


if __name__=='__main__':main()
