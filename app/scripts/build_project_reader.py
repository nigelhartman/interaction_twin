#!/usr/bin/env python3
"""Create the dated project reader. Documentation only; no inference or app changes."""
from pathlib import Path
from xml.sax.saxutils import escape
import hashlib
import json
import math

from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf/Interaction-Twin-Project-Guide.pdf'
ASSETS = ROOT / 'output/pdf-assets/reader-20261029'
QA = json.loads((ROOT / 'output/native-appearance/delivery-qa-20261029.json').read_text())
CPU = QA['actual_cpu_flow']

for name, filename in [('Reader','Arial.ttf'), ('ReaderBold','Arial Bold.ttf'),
                       ('ReaderItalic','Arial Italic.ttf')]:
    pdfmetrics.registerFont(TTFont(name, '/System/Library/Fonts/Supplemental/' + filename))
pdfmetrics.registerFontFamily('Reader', normal='Reader', bold='ReaderBold', italic='ReaderItalic', boldItalic='ReaderBold')

W,H = A4
M = 43
CW = W-2*M
INK = colors.HexColor('#172B2A')
MUTED = colors.HexColor('#526762')
TEAL = colors.HexColor('#2D7161')
MINT = colors.HexColor('#E8F2EA')
PAPER = colors.HexColor('#FAFAF6')
LINE = colors.HexColor('#D5DFD8')
PURPLE = colors.HexColor('#735BA0')
LAVENDER = colors.HexColor('#F0EBF7')
AMBER = colors.HexColor('#9C6924')
CREAM = colors.HexColor('#FBF0D8')
GREY = colors.HexColor('#EDF0EF')

styles = {
    'body': ParagraphStyle('body', fontName='Reader', fontSize=11.5, leading=16.2, textColor=INK, spaceAfter=9),
    'small': ParagraphStyle('small', fontName='Reader', fontSize=9.4, leading=13, textColor=MUTED),
    'caption': ParagraphStyle('caption', fontName='Reader', fontSize=9.3, leading=12.6, textColor=MUTED),
    'h2': ParagraphStyle('h2', fontName='ReaderBold', fontSize=16, leading=20, textColor=INK),
    'table': ParagraphStyle('table', fontName='Reader', fontSize=10.1, leading=13.9, textColor=INK),
    'th': ParagraphStyle('th', fontName='ReaderBold', fontSize=10, leading=13.5, textColor=TEAL),
    'code': ParagraphStyle('code', fontName='Courier', fontSize=9.5, leading=14, textColor=INK),
}

SOURCES = [
 ('Meta SAM 2', 'https://github.com/facebookresearch/sam2'),
 ('Meta SAM 3D Objects', 'https://github.com/facebookresearch/sam-3d-objects'),
 ('Meta SAM 3.1 model card', 'https://huggingface.co/facebook/sam3.1'),
 ('Meta CoTracker', 'https://github.com/facebookresearch/co-tracker'),
 ('TripoSR', 'https://github.com/VAST-AI-Research/TripoSR'),
 ('Microsoft MoGe', 'https://github.com/microsoft/MoGe'),
 ('NVIDIA: SimReady Central announcement, 21 Oct 2026', 'https://huggingface.co/blog/nvidia/simready-central-assets-robotics-ai-factories'),
 ('NVIDIA SimReady Assets dataset', 'https://huggingface.co/datasets/nvidia/simready-assets'),
 ('NVIDIA SimReady Foundation', 'https://github.com/NVIDIA/simready-foundation'),
 ('NVIDIA SimReady Central Library documentation', 'https://docs.omniverse.nvidia.com/simready/latest/simready-central.html'),
 ('NVIDIA: AI-assisted SimReady authoring, 8 Oct 2026', 'https://developer.nvidia.com/blog/5-steps-to-create-simready-assets-for-robotics-with-frontier-ai-models/'),
 ('Polyfork', 'https://polyfork.dev/'),
 ('Polyfork agent documentation', 'https://polyfork.dev/agents'),
 ('Meta Global Hackathon', 'https://dev.meta.ai/events/global-hackathon'),
 ('Cerebral Valley event page', 'https://cerebralvalley.ai/e/meta-ai-developer-hackathon'),
 ('Articulat3D project and paper', 'https://maxwell-zhao.github.io/Articulat3D/'),
]

def ref(n):
    return '<link href="%s" color="#2D7161">[%d]</link>' % (SOURCES[n-1][1],n)

def para(txt, style='body'):
    # Deliberately avoid Unicode dash glyphs in the deliverable.
    assert not any(x in txt for x in ['\u2011','\u2013','\u2014']), txt
    return Paragraph(txt, styles[style])

class Reader:
    def __init__(self):
        self.c = canvas.Canvas(str(OUT), pagesize=A4, pageCompression=1)
        self.c.setTitle('Interaction Twin | Project Guide | 29 October 2026')
        self.c.setAuthor('Nigel Hartman | prepared with Codex')
        self.c.setSubject('Pipeline, models, evidence, robotics roadmap and hackathon presentation')
        self.n=0; self.y=0; self.entries=[]; self.used=[]

    def start(self, label, title, sub='', key=None):
        if self.n:
            self.finish()
            self.c.showPage()
        self.n+=1
        self.c.setFillColor(PAPER); self.c.rect(0,0,W,H,fill=1,stroke=0)
        self.c.setFillColor(TEAL); self.c.setFont('ReaderBold',9)
        self.c.drawString(M,H-36,'INTERACTION TWIN')
        self.c.setFillColor(MUTED); self.c.setFont('Reader',8.5)
        self.c.drawRightString(W-M,H-36,'PROJECT READER / 29 OCT 2026')
        self.c.setStrokeColor(LINE); self.c.line(M,H-48,W-M,H-48)
        self.y=H-75
        self.c.setFillColor(TEAL); self.c.setFont('ReaderBold',9)
        self.c.drawString(M,self.y,label.upper()); self.y-=19
        heading=Paragraph(title,ParagraphStyle('title',fontName='ReaderBold',fontSize=29,leading=32,textColor=INK))
        _,h=heading.wrap(CW,200); heading.drawOn(self.c,M,self.y-h); self.y-=h+12
        if sub:self.p(sub,style='small',after=15)
        if key:
            self.c.bookmarkPage(key);self.c.addOutlineEntry(title,key,level=0)
            self.entries.append((self.n,title,key))

    def finish(self):
        if self.y<49: raise ValueError(f'Overflow on page {self.n}: {self.y}')
        self.used.append({'page':self.n,'bottom_y':round(self.y,2)})
        self.c.setStrokeColor(LINE);self.c.line(M,37,W-M,37)
        self.c.setFont('Reader',8);self.c.setFillColor(MUTED)
        self.c.drawString(M,24,'PRIVATE WORKING GUIDE / ACTUAL RESULTS + LABELLED PLANS')
        self.c.drawRightString(W-M,24,f'{self.n:02d}')

    def p(self, txt, style='body', after=9, x=None, width=None):
        p=para(txt,style);_,h=p.wrap(width or CW,750)
        p.drawOn(self.c,x if x is not None else M,self.y-h)
        self.y-=h+after
        return h

    def h2(self,txt):self.y-=5;self.p(txt,'h2',9)

    def bullet(self,title,txt):
        self.p('<b>'+title+'</b> '+txt,after=10)

    def note(self,title,txt,tone='mint'):
        bg={'mint':MINT,'amber':CREAM,'purple':LAVENDER,'grey':GREY}[tone]
        p=para('<b>'+title+'</b><br/>'+txt);_,h=p.wrap(CW-26,750)
        self.c.setFillColor(bg);self.c.roundRect(M,self.y-h-25,CW,h+25,9,fill=1,stroke=0)
        p.drawOn(self.c,M+13,self.y-h-12);self.y-=h+39

    def image(self,path,maxh=270,caption=None):
        im=Image.open(path); iw,ih=im.size
        h=min(maxh,CW*ih/iw); w=h*iw/ih
        self.c.drawImage(str(path),M+(CW-w)/2,self.y-h,w,h,mask='auto')
        self.y-=h+9
        if caption:self.p(caption,'caption',12)

    def table(self,headers,rows,widths):
        cells=[[para(x,'th') for x in headers]]+[[para(str(x),'table') for x in row] for row in rows]
        t=Table(cells,colWidths=widths,hAlign='LEFT')
        t.setStyle(TableStyle([
            ('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),MINT),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PAPER]),
            ('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),
            ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
            ('LINEBELOW',(0,0),(-1,0),.8,LINE),('LINEBELOW',(0,1),(-1,-1),.4,LINE),
        ]))
        _,h=t.wrap(CW,800);t.drawOn(self.c,M,self.y-h);self.y-=h+14

    def custom(self,height,fn):
        fn(self.c,M,self.y-height,CW,height)
        self.y-=height+13

    def save(self):
        self.finish();self.c.save()
        return self.used

def label(c,x,y,text,size=10,color=INK,bold=False):
    c.setFont('ReaderBold' if bold else 'Reader',size);c.setFillColor(color);c.drawString(x,y,text)

def box(c,x,y,w,h,title,body='',color=MINT,num=None):
    c.setFillColor(color);c.setStrokeColor(LINE);c.roundRect(x,y,w,h,8,fill=1,stroke=1)
    tx=x+12
    if num:
        c.setFillColor(TEAL);c.circle(tx+8,y+h-20,10,fill=1,stroke=0)
        c.setFillColor(colors.white);c.setFont('ReaderBold',9);c.drawCentredString(tx+8,y+h-23,num)
        tx+=27
    p=para('<b>'+title+'</b>','table');_,ph=p.wrap(w-(tx-x)-10,100);p.drawOn(c,tx,y+h-12-ph)
    if body:
        p=para(body,'small');_,ph=p.wrap(w-24,h);p.drawOn(c,x+12,y+h-45-ph)

def arrow(c,x1,y1,x2,y2,color=TEAL):
    c.setStrokeColor(color);c.setFillColor(color);c.setLineWidth(1.6);c.line(x1,y1,x2,y2)
    a=math.atan2(y2-y1,x2-x1); s=5
    p=c.beginPath();p.moveTo(x2,y2)
    p.lineTo(x2-s*math.cos(a-.5),y2-s*math.sin(a-.5))
    p.lineTo(x2-s*math.cos(a+.5),y2-s*math.sin(a+.5));p.close();c.drawPath(p,fill=1,stroke=0)

def pipeline(c,x,y,w,h):
    rows=[('1','INPUT','Video + exact timestamps','WORKS',MINT),
          ('2','PERCEPTION','Parts + SAM masks','LOCAL BASELINE',MINT),
          ('3','GEOMETRY','Generated component meshes','PARTIAL',CREAM),
          ('4','ALIGNMENT','Conditional visible poses','PARTIAL',CREAM),
          ('5','PROCEDURE','Evidence-linked state + steps','DESCRIPTIVE',LAVENDER),
          ('6','PRODUCT','Viewer + package + MCP','WORKS LOCALLY',MINT),
          ('7','VALIDATION','Software checks + final proof','PARTIAL',CREAM)]
    rh=48; gap=8
    for i,(n,t,b,status,col) in enumerate(rows):
        yy=y+h-(i+1)*rh-i*gap
        c.setFillColor(col);c.roundRect(x,yy,w,rh,7,fill=1,stroke=0)
        c.setFillColor(TEAL);c.circle(x+19,yy+24,10,fill=1,stroke=0)
        c.setFillColor(colors.white);c.setFont('ReaderBold',9);c.drawCentredString(x+19,yy+21,n)
        label(c,x+39,yy+29,t,9,TEAL,True);label(c,x+39,yy+12,b,10)
        c.setFillColor(MUTED);c.setFont('ReaderBold',8);c.drawRightString(x+w-12,yy+21,status)
        if i<6:arrow(c,x+19,yy-1,x+19,yy-gap+1)

def reconstruction(c,x,y,w,h):
    cw=(w-36)/3
    for i,(title,col) in enumerate([('IMAGE + MASK',MINT),('3D MODEL',LAVENDER),('NATIVE ASSET',MINT)]):
        box(c,x+i*(cw+18),y,cw,h,title,color=col)
    # Schematic mask; this is deliberately not represented as a model result.
    c.setFillColor(TEAL);c.roundRect(x+28,y+43,90,61,14,fill=1,stroke=0)
    c.setFillColor(PAPER);c.ellipse(x+52,y+58,x+94,y+89,fill=1,stroke=0)
    label(c,x+17,y+18,'Visible pixels only',9)
    mid=x+cw+18
    label(c,mid+15,y+80,'TripoSR now',11,INK,True)
    label(c,mid+15,y+60,'SAM 3D planned',10,PURPLE)
    label(c,mid+15,y+18,'Learned completion',9)
    rx=x+2*(cw+18)
    pts=[(rx+35,y+55),(rx+97,y+43),(rx+118,y+90),(rx+58,y+104),(rx+71,y+76)]
    c.setStrokeColor(TEAL);c.setLineWidth(1)
    for a,b in [(0,1),(1,2),(2,3),(3,0),(0,4),(1,4),(2,4),(3,4)]:
        c.line(*pts[a],*pts[b])
    label(c,rx+15,y+18,'Mesh + appearance',9)
    arrow(c,x+cw+2,y+h/2,x+cw+16,y+h/2)
    arrow(c,mid+cw+2,y+h/2,mid+cw+16,y+h/2)

def pose_figure(c,x,y,w,h):
    cw=(w-24)/2
    box(c,x,y+60,cw,h-60,'REFERENCE VIEW','Raycast visible pixels onto the generated mesh.',MINT)
    box(c,x+cw+24,y+60,cw,h-60,'LATER FRAME','CoTracker follows matching pixels; PnP estimates the pose.',LAVENDER)
    for dx in [0,cw+24]:
        c.setStrokeColor(TEAL);c.roundRect(x+dx+31,y+84,cw-62,47,7,fill=0,stroke=1)
        for j in range(5):
            c.setFillColor(TEAL if dx==0 else PURPLE);c.circle(x+dx+48+j*27,y+98+(j%2)*14,3,fill=1,stroke=0)
    arrow(c,x+cw+2,y+112,x+cw+22,y+112)
    box(c,x,y,w,43,'CHECK: silhouette + visible points + fit residual',color=CREAM)
    arrow(c,x+w/2,y+59,x+w/2,y+45,AMBER)

def coverage(c,x,y,w,h):
    label(c,x,y+h-13,'SUPPORTED NON-REFERENCE TIMESTAMPS',9,TEAL,True)
    for i,(title,n,col) in enumerate([('Preserved baseline',CPU['baseline_supported'],TEAL),('Latest CPU refit',CPU['candidate_supported'],PURPLE)]):
        yy=y+h-57-i*65
        label(c,x,yy+16,title,11,INK,True)
        ww=(w-65)/12
        for j in range(12):
            c.setFillColor(col if j<n else LINE);c.roundRect(x+j*(ww+1),yy-8,ww-3,15,3,fill=1,stroke=0)
        label(c,x+w-48,yy-5,f'{n} / 12',12,col,True)
    label(c,x,y+3,'Same 12 timestamps; each side can withhold different samples.',9,MUTED)

def nvidia_bridge(c,x,y,w,h):
    half=(w-20)/2
    box(c,x,y+h-115,half,115,'OUR CAPTURED OBJECT','Candidate parts, observed motion, source-linked procedure.',MINT)
    box(c,x+half+20,y+h-115,half,115,'SIMREADY LIBRARY','Robot, gripper, workspace and environment assets.',LAVENDER)
    mid=y+61
    arrow(c,x+half/2,y+h-119,x+w/2-30,mid+42)
    arrow(c,x+w-half/2,y+h-119,x+w/2+30,mid+42)
    box(c,x,y+38,w,61,'PROPOSED: OpenUSD + scale + physics + validation',color=CREAM)
    label(c,x+12,y+14,'A future bounded Isaac Sim task, with declared capability checks.',10,MUTED)

def rubric(c,x,y,w,h):
    rows=[('Impact',30,TEAL),('Meta tooling',25,PURPLE),('Technical depth / execution',20,AMBER),('Demo',25,TEAL)]
    for i,(name,n,col) in enumerate(rows):
        yy=y+h-22-i*37
        label(c,x,yy,name,10,INK,True)
        xx=x+183;ww=w-226
        c.setFillColor(LINE);c.roundRect(xx,yy-2,ww,12,3,fill=1,stroke=0)
        c.setFillColor(col);c.roundRect(xx,yy-2,ww*n/30,12,3,fill=1,stroke=0)
        label(c,x+w-32,yy,f'{n}%',10,col,True)

def screenshot_crop():
    p=ASSETS/'workspace-page.jpg'
    im=Image.open(p)
    # Crop only the page chrome and side navigation; source and output remain together.
    im.crop((215,128,1265,1082)).save(ASSETS/'workspace-panels.png')
    captures={
        'showcase.jpg':'http://127.0.0.1:8765/showcase/index.html',
        'sample.jpg':'http://127.0.0.1:8765/showcase/index.html#samples',
        'model-lab.jpg':'http://127.0.0.1:8765/workspace/upgrades.html?case=jabra-stage6-v1--procedure-v1&track=small_object_a#plan-section',
        'agents.jpg':'http://127.0.0.1:8765/workspace/agents.html?twin=jabra-stage6-v1--procedure-v1',
        'workspace-page.jpg':'http://127.0.0.1:8765/workspace/index.html?case=jabra-stage6-v1--procedure-v1&t=17.402',
    }
    manifest={'date':'2026-10-29','method':'Real localhost browser screenshots using CUA; no UI compositing',
              'captures':[{'file':fn,'url':url,'sha256':hashlib.sha256((ASSETS/fn).read_bytes()).hexdigest()} for fn,url in captures.items()],
              'derived_crop':{'file':'workspace-panels.png','source':'workspace-page.jpg','rectangle':[215,128,1265,1082]},
              'diagrams':'Original vector explanations; schematic, not additional model outputs'}
    (ASSETS/'screenshots.json').write_text(json.dumps(manifest,indent=2))

def build():
    OUT.parent.mkdir(parents=True,exist_ok=True);screenshot_crop();d=Reader()

    d.start('The project, explained','Interaction Twin',
            'A private reading guide for Nigel Hartman | Snapshot: 29 October 2026',key='cover')
    d.p('Turn a demonstration into<br/><b>parts, motion and meaning.</b>',after=18)
    d.image(ASSETS/'showcase.jpg',maxh=278,caption='Real local project page, captured for this guide. The page presents a research prototype, not a validated robotics product.')
    d.note('The goal','A reusable interaction asset: generated 3D components, observed movement, candidate relationships and a procedure linked to its source video.')
    d.p('Inside: how the models connect, what works today, what is missing, the NVIDIA opportunity, and a focused route to a compelling hackathon demo.')
    d.p('Read in about 25-35 minutes. For a fast overview, read pages 2, 3, 11, 12 and 19. Screenshots are real; explanatory diagrams are schematic. Plans are labelled throughout.','small')

    d.start('01 / Quick read','What we have. What we need.',key='quick')
    d.note('Working now','A local video-processing app, baseline segmentation and reconstruction, conditional motion, an evidence compiler, review tools, portable exports and a read-only MCP library.')
    d.note('The main unfinished part','Reliable automatic reconstruction and source-consistent 3D alignment. Physical dimensions, exact assembly, contact and hidden insertion are not validated.','amber')
    d.p('<b>Have we succeeded?</b> We have a connected prototype and substantial product infrastructure. We have not yet proved the full digital-twin promise. The next milestone is one convincing, repeatable object-interaction result.')
    d.table(['Find it quickly','Pages'],[
        ('<link href="#pipeline" color="#2D7161">Pipeline and model explanations</link>','3-7'),
        ('<link href="#workspace" color="#2D7161">Actual app and local use</link>','8-10'),
        ('<link href="#results" color="#2D7161">Results, gaps and the access upgrade</link>','11-13'),
        ('<link href="#agents" color="#2D7161">Agent library and NVIDIA simulation path</link>','14-16'),
        ('<link href="#positioning" color="#2D7161">Differentiation, SaaS and winning strategy</link>','17-20'),
        ('<link href="#faq1" color="#2D7161">FAQs, glossary and sources</link>','21-24'),
    ],[CW-80,80])
    d.p('<b>Our strongest sentence:</b> "We turn recorded demonstrations into inspectable interaction assets that people and agents can reuse."')
    d.p('"We automated the entire production of simulation-ready twins" is a future ambition. Today that would overstate both geometry quality and physics validation.','small')

    d.start('02 / The whole system','One pipeline. Seven stages.',
            'The stages connect, but implementation and physical quality are different completion gates.',key='pipeline')
    d.custom(384,pipeline)
    d.p('<b>Two kinds of output travel together.</b> The visual side stores masks, geometry and conditional poses. The semantic side stores component IDs, observed states, candidate relations and steps. Source timestamps join them.')
    d.p('<b>Our current position:</b> Stage 6 works as a local product; Stage 7 software checks and presentation are active. The quality work in Stages 3-4 remains open, and Stage 5 describes evidence rather than executing physics.','small')

    d.start('03 / Perception','SAM finds the pixels.',
            'Segmentation answers "which visible pixels belong to this part?" It does not recover a complete 3D object.',key='sam')
    d.custom(128,lambda c,x,y,w,h: (
        box(c,x,y,w*.30,h,'1 / FIND PARTS','Reasoning proposes object labels and useful moments.',MINT),
        box(c,x+w*.35,y,w*.30,h,'2 / MASK THEM','SAM separates each visible component from the image.',LAVENDER),
        box(c,x+w*.70,y,w*.30,h,'3 / AUDIT','Check masks, identities and occlusion uncertainty.',CREAM),
        arrow(c,x+w*.305,y+65,x+w*.345,y+65),arrow(c,x+w*.655,y+65,x+w*.695,y+65)))
    d.h2('What runs today')
    d.p('Meta <b>SAM 2.1 Tiny</b> runs locally on the Mac through MPS. It produces masks across sampled video frames. Dense Jabra work used 5 fps samples and larger crops for small parts. '+ref(1))
    d.h2('What SAM 3.1 changes')
    d.p('SAM 3.1 is the intended newer perception model. Its upstream model supports image/video segmentation. Our prepared integration currently compares <b>one anchor-image mask</b>; full video propagation and renewed audits still need integration. '+ref(3))
    d.bullet('Why it matters.','A clean mask removes the hand and background from the reconstruction input. A bad mask can make a reconstruction model generate part of a finger or combine two objects.')
    d.bullet('What remains difficult.','Tiny reflective earbuds, nearly identical parts, motion blur and hand occlusion. A mask alone cannot establish that a reappearing object is the same physical item.')
    d.note('Automation boundary','Recent learned-model flows avoid manually designed geometry and hand-set poses. Historical camera/battery pilots did use disclosed assistant seeds or fitted proxies; they remain separate baselines.','grey')

    d.start('04 / Reconstruction','How a mask becomes a 3D asset.',
            'The reconstruction model supplies shape and appearance. SAM supplies the isolation of the object.',key='reconstruct')
    d.custom(150,reconstruction)
    d.p('We automatically select a useful view: enough visible object area, a plausible mask and limited occlusion. The masked image becomes a component input. <b>One selected mesh is then reused across time.</b> We do not create a new mesh at every frame.')
    d.p('<b>Today:</b> TripoSR generated the current learned meshes on the Mac. It is a non-Meta baseline. Several independent view candidates were tested; this is candidate selection, not multi-view fusion. '+ref(5))
    d.p('<b>Next:</b> genuine Meta SAM 3D Objects inference. It predicts complete shape, appearance and scene layout from an image plus mask. Our adapter and import contracts are prepared; no genuine SAM 3D result has run here yet. '+ref(2))
    d.h2('Where do the back and textures come from?')
    d.p('Unseen surfaces are a learned guess. Current appearance is model-generated; it is not a photographic scan of every side. Native colours and supported UV materials are preserved. Optional SAM 3D UV baking uses generated Gaussian appearance, not new observations of the hidden back.')
    d.note('For a useful twin','A convincing outer shell can support inspection. A battery socket or tight insertion needs observed inner geometry, scale and clearance checks. A plausible completion cannot certify fit.','amber')

    d.start('05 / Registration','How 3D aligns to 2D video.',
            'We ask which camera-relative pose makes the generated model project onto the observed object.',key='alignment')
    d.custom(207,pose_figure)
    d.bullet('1. Fit a reference view.','Search orientation, position and one fixed component scale using silhouette and supported appearance. Camera intrinsics are estimated or assumed; they are not a calibration measurement.')
    d.bullet('2. Attach visible points.','Raycasts associate pixels in that reference view with points on the mesh. CoTracker follows the corresponding pixels and estimates visibility in later frames. '+ref(4))
    d.bullet('3. Solve and check.','PnP estimates rotation and translation from those 3D-to-2D correspondences. Fit checks reject weak frames. Unsupported movement remains absent, rather than becoming an invented insertion.')
    d.p('<b>Depth helps, but does not solve everything.</b> MoGe estimates depth/pointmaps and camera information. Earlier tests used it for geometric support and a relative gauge. A visible depth surface is only part of the object; full-to-partial scaled ICP is an optional refinement after a good initial alignment. '+ref(6))
    d.note('The important limit','Current poses are conditional on generated geometry and camera assumptions. Separate component coordinates are not yet a verified common metric assembly.','amber')

    d.start('06 / Reasoning + compiler','The result is more than a mesh.',
            'The distinctive product goal is an inspectable account of what the parts did, with evidence attached.',key='program')
    d.p('Reasoning proposes labels, relationships and steps: a lid appears attached to a base; a separate part approaches an opening; the cover later looks closed. The compiler verifies IDs, references and structure, then stores the proposal with its evidence.')
    d.note('Current reasoning identity','Existing proposals came from hosted OpenAI through the local Codex CLI, or are reused from saved responses. "Local CLI" does not mean local model inference. Meta reasoning integration remains to be built.','purple')
    d.h2('A tiny example from the Jabra program')
    d.note('Readable summary of an actual record',
           '<font name="Courier" size="10">component: container_lid<br/>relation: attached_to container_base<br/>status: inferred<br/>evidence: source_000031, source_000036, ...<br/>limit: exact hinge geometry not established</font>','grey')
    d.p('The saved program also has exact-time states and procedure steps. For example, the left-opening approach is linked to approximately 22.002-23.002 seconds; the later occlusion is recorded separately. It does not assert successful seating.')
    d.bullet('Why explicit uncertainty is useful.','A human or agent can distinguish an observed open-looking cover from an inferred attachment, and an unknown hidden action from verified evidence.')
    d.bullet('What validation means.','A valid JSON program can still describe an incorrect interpretation. Schema checks do not prove semantic truth, physical identity or causality.')

    d.start('07 / Real app screenshot','Inspect the video and the draft.',
            'Actual workspace at 17.402 s. Source video and generated components share the source timeline.',key='workspace')
    d.image(ASSETS/'workspace-panels.png',maxh=427,caption='Real browser capture, cropped to the main workspace panels. Jabra assets were generated by TripoSR. Camera-relative positions and appearance are estimates; their visual arrangement does not establish physical assembly.')
    d.p('<b>What you can do:</b> scrub the recording, inspect conditional pose samples, orbit a component, select a procedure step and jump back to its cited frame. Unsupported intervals stay visible as gaps.')
    d.p('Human corrections are saved as separate revisions. The original automatic output remains available for comparison.','small')

    d.start('08 / Try it locally','From a new video to a saved asset.',key='use')
    d.p('Open the local <link href="http://127.0.0.1:8765/showcase/index.html" color="#2D7161">project page</link> while the processor is running. Use <b>Try your video</b> or <b>New video</b> in the workspace.')
    d.table(['Action','What happens'],[
        ('1. Choose a short video','The processor preserves the source and timestamps. Use a clear, slow interaction with one or a few rigid parts.'),
        ('2. Inspect the run plan','Check sampling and backends. A fresh clip normally needs fresh reasoning; cached reasoning only applies to its matching saved evidence.'),
        ('3. Start the allowed run','Local model stages execute with logs and per-component failures. Hosted reasoning needs the explicit consent shown by the app.'),
        ('4. Inspect the result','Review original frames, native assets, supported motion, states and gaps. Completion of the job does not mean every component succeeded.'),
        ('5. Review or export','Keep a separate correction revision; create a portable package containing geometry, program and cited evidence.'),
    ],[128,CW-128])
    d.p('<b>While access waits:</b> explore saved cases and use Model Lab with cached reasoning. The <b>CPU refit</b> mode reuses the existing mesh; it is not new reconstruction. New arbitrary clips cannot reuse another video\'s reasoning.')
    d.p('The app is local, not publicly hosted. Portable export can omit the full source video; it is off by default. Standalone offline pixels have not been freshly browser-verified because the file-preview policy blocks that route.','small')

    d.start('09 / Models + compute','Which capability runs where?',
            'Models are replaceable. Provider identity and the source evidence stay recorded.',key='models')
    d.table(['Capability','Working baseline','After access / remaining work'],[
        ('Segmentation','Meta SAM 2.1 Tiny<br/>Mac MPS','SAM 3.1 GPU experiment. Current adapter: one anchor mask; video integration pending.'),
        ('3D reconstruction','TripoSR<br/>Mac MPS<br/><i>Non-Meta</i>','SAM 3D Objects on prepared Linux/NVIDIA runtime. No genuine output yet.'),
        ('Point tracking','Meta CoTracker3<br/>MPS or explicit CPU','Higher-resolution tracking if useful; compare matching source times.'),
        ('Depth / camera support','MoGe-2 Small<br/>Earlier MPS experiments','Better geometric consistency if experiments justify it; calibration remains separate.'),
        ('Visual reasoning','Hosted OpenAI via Codex; cached on refits','Implement a Meta provider after credentials, billing and generation authorization.'),
        ('Pose + compilation','Local fitting / PnP; JSON validation','Refit real new model output; preserve unknown intervals and metadata.'),
        ('Viewer / library','Local browser, export, MCP stdio','Public distribution and simulator bridge are separate future steps.'),
    ],[109,166,CW-275])
    d.note('Why the GPU helps','It enables the intended reconstruction/perception runtime and larger model settings. It does not automatically fix an occluded cavity, wrong identity or unmeasured scale. Two GPUs can serve independent component jobs only if memory and runtime constraints permit.','grey')

    d.start('10 / Evidence, not optimism','The latest result did not improve.',
            'Latest CPU refit: 29 October 2026. A useful comparison can reject a new candidate.',key='results')
    d.custom(180,coverage)
    d.table(['Internal diagnostic','Baseline','CPU candidate'],[
        ('Supported / 12 common timestamps','7','6'),
        ('Median supported silhouette IoU','0.875','0.778'),
        ('Median supported reprojection residual','0.37 px','0.75 px'),
    ],[CW-186,93,93])
    d.p('The original remains preferred. These are <b>internal, conditional source-frame diagnostics</b>. Masks enter fitting/gates; temporal context is used. Supported subsets differ (7 versus 6). This is not independent physical pose accuracy.')
    d.p('<b>What the 93-second run proves:</b> the seven downstream workflow phases connect, actual CoTracker ran on 13 frames, and 14 cached steps plus 43 assertions survived. Geometry and reasoning were reused; no new reconstruction was produced in that time.')
    d.note('Other useful numbers','454 tests passed, 1 skipped. Ten saved catalog cases validate, including repeated variants of the same recordings. That is neither ten independent videos nor ten physically verified twins. The original dense Jabra case has 3 learned assets and 14 steps.','grey')

    d.start('11 / Priorities','The missing features that matter.',
            'The best next work improves the central result, rather than expanding the interface.',key='gaps')
    d.table(['Priority','Completion gate'],[
        ('P0 / Recognisable geometry','One genuine SAM 3D component retains native appearance and looks like the filmed part, with visible errors recorded.'),
        ('P0 / Supported motion','Useful visible pose coverage on unchanged source timestamps. No hidden insertion or identity guesses promoted to facts.'),
        ('P0 / Repeatability','A second clear unseen recording works without object-specific mesh code or manually set poses.'),
        ('P1 / Meta reasoning','A bounded, source-linked interpretation from a genuine Meta model, with provider identity and cost recorded.'),
        ('P1 / Professional payoff','One person can prepare an inspectable equipment procedure faster than a declared manual baseline.'),
        ('P2 / Simulation bridge','Metric scale, colliders and an OpenUSD task-specific handoff pass explicit checks. Not required for the first honest descriptive demo.'),
    ],[148,CW-148])
    d.h2('Stretch only after those gates')
    d.p('Multi-view fusion, measured texture baking, stronger camera recovery, multiple interacting objects, mechanism fitting, task-specific physics and a public capability-aware catalog. Add the feature that repairs an observed failure; avoid a second speculative platform.')
    d.note('MVP versus vision','Hackathon MVP: one polished demonstration-to-interaction-asset flow plus a second input. Longer-term vision: validated simulation content and reusable robotics tasks.','mint')

    d.start('12 / The upgrade plan','Change one model. Keep evidence.',
            'Prepared infrastructure reduces integration work. It is not evidence that the pending model has already succeeded.',key='upgrade')
    d.image(ASSETS/'model-lab.jpg',maxh=242,caption='Actual Model Lab screenshot. CPU refit is selected; the other modes and cached-input plan are real app controls. Nothing was inferred for this PDF.')
    d.bullet('1. Start with one genuine SAM 3D result.','Use the prepared image/mask bundle on the authorized runtime. Preserve the original model bytes, coordinate metadata and provenance. Import the result beside the matching handoff.')
    d.bullet('2. Rerun only dependent stages.','Refit and track the component, recompile using the same cached interpretation, then compare exact common timestamps. Keep the baseline when it is stronger.')
    d.bullet('3. Upgrade perception and reasoning separately.','Test SAM 3.1 anchor masks first. Integrate video propagation deliberately. Implement and test a Meta reasoning provider before changing the full pipeline.')
    d.p('<b>Spending control:</b> begin with one bounded, explicitly authorized request, a token/output budget, no automatic retries and a saved response. A local guard can bound our client; it is not an account-wide billing cap. No paid call or model download was made to create this guide.','small')

    d.start('13 / The agent library','Make the asset useful to an agent.',
            'MCP is the distribution and inspection interface. The asset itself carries the useful structure.',key='agents')
    d.image(ASSETS/'agents.jpg',maxh=235,caption='Actual browser catalog. This screen shows saved tool responses. It is not a live LLM conversation or browser MCP connection; the real stdio server is tested separately.')
    d.p('An agent can discover a saved twin, read components and procedures, query exact-time states, fetch geometry and inspect the source evidence behind a claim. The interface is read-only; it cannot run a robot or trigger model generation.')
    d.note('A concrete reuse example','"Find the case-opening step, fetch the case and lid assets, and return the frames that support the attachment claim." The result includes stable IDs, an inferred relation, source references and explicit missing physics.')
    d.p('A scripted client has already used the actual MCP server: 25 calls retrieved 3 hash-verified GLBs and 5 source JPEGs and built a portable package. It used no LLM. This demonstrates retrieval and packaging, not autonomous robot planning.')
    d.p('<b>Next library improvement:</b> capability flags such as visual draft, motion-supported or physics-tested, with exact asset versions and evidence. Public hosting, publishing rights and cloud search remain separate product decisions.','small')

    d.start('14 / NVIDIA connection','Where SimReady fits.',
            'Verified against primary NVIDIA sources on 29 October 2026.',key='nvidia')
    d.p('NVIDIA announced <b>SimReady Central on 21 October 2026</b> with more than 2,000 freely accessible OpenUSD assets. Its official NVIDIA dataset currently lists 2,677; the count changes. We could not verify an exact fixed "2,500" release. '+ref(7)+' '+ref(8))
    d.custom(245,nvidia_bridge)
    d.p('<b>The complementary idea:</b> use a curated robot, gripper and environment, then add the custom object and demonstrated interaction captured by our app. The bridge above is a proposal; no OpenUSD export or Isaac Sim integration exists in the current prototype.')
    d.p('The NVIDIA dataset declares CC-BY-4.0: free access still requires attribution. The wider registry can contain other licenses. Asset readiness depends on its stated simulation profile; a catalog entry is not a promise of every physics capability. '+ref(8)+' '+ref(9)+' '+ref(10))
    d.note('A strong, accurate pitch','"NVIDIA provides a growing library of prepared simulation assets. We are building a route from a filmed custom interaction to an inspectable asset that can eventually join that ecosystem."','mint')

    d.start('15 / Robotics roadmap','A render is not yet a simulator.',
            'Proposed path, not completed functionality. Pick one target task and validate only the claims it requires.',key='simulation')
    d.table(['Step','Why it is necessary'],[
        ('1 / Export OpenUSD','Preserve component geometry/materials and attach IDs, source evidence, version and uncertainty metadata.'),
        ('2 / Establish units','Use a known dimension or reliable specifications; register axes, camera and a shared world. Monocular scale is ambiguous.'),
        ('3 / Add physical properties','Validate collision shapes, rigid bodies, mass/inertia and friction. Add joint axes and limits only where the mechanism needs them.'),
        ('4 / Compose a scene','Use licensed, version-pinned SimReady environment and robot assets. Keep dataset attribution.'),
        ('5 / Run profile checks','Check the selected SimReady requirements and actual runtime behavior; fix invalid properties and dependencies.'),
        ('6 / Test a bounded task','Begin with grasp-and-place, then compare with observations. Tight insertion needs better cavity geometry and contact/clearance validation.'),
    ],[133,CW-133])
    d.p('<b>Useful later:</b> object-specific scene preparation, synthetic perception data, demonstrations for task authoring, and eventually manipulation-policy experiments. Robot learning still needs its own action representation, rewards, sensors and evaluation; our descriptive procedure is not a trained policy.')
    d.p('NVIDIA already demonstrates AI-assisted CAD/OpenUSD preparation, physics configuration, validation and Isaac tasks. AI authoring itself is not new. Our proposed emphasis is the recorded interaction and its evidence-linked component/state structure. '+ref(11))

    d.start('16 / Positioning + SaaS','What is ours to build?',
            'Product differentiation is a hypothesis to demonstrate. Research novelty has not been established.',key='positioning')
    d.table(['Reference','Central focus','Our proposed emphasis'],[
        ('Articulat3D','Monocular articulated reconstruction using 3D tracks, rigid groups and joint-constrained refinement. '+ref(16),'Reusable component relationships, observed states, procedure and source evidence. We have not outperformed its geometry.'),
        ('Polyfork','Program-authored, adjustable 3D assets; typed controls and agent access. '+ref(12)+' '+ref(13),'Derive a descriptive interaction package from recorded activity. We do not recover arbitrary parametric CAD programs.'),
        ('NVIDIA SimReady','Prepared OpenUSD assets, profile-specific simulation checks and agent discovery. '+ref(7)+' '+ref(9),'Supply custom filmed objects/interactions, then validate a future simulator handoff. MCP alone is not novel.'),
    ],[88,197,CW-285])
    d.h2('A plausible first paying customer')
    d.p('<b>Equipment-procedure authors</b> who already record teardown, setup or maintenance demonstrations. Help them produce an interactive, evidence-linked guide with inspectable parts. This can provide value before engineering-grade physics is available.')
    d.p('<b>Test the business claim:</b> ask one author to prepare the same guide manually and with the prototype. Record elapsed time, corrections, missing parts and whether another person can follow it. No revenue, market demand or time savings have yet been measured.')
    d.p('Possible SaaS later: per-video processing plus a private asset/team library. Robotics content preparation is a higher-validation expansion. Start with the workflow that users can trust today.','small')

    d.start('17 / Hackathon strategy','Win through one convincing result.',
            'No project can be guaranteed to win. The most important remaining evidence is visual quality and repeatability.',key='strategy')
    d.custom(159,rubric)
    d.p('Weights above come from your supplied participant-guide snapshot: Impact 30%, Meta tooling 25%, Technical depth/execution 20%, Demo 25%. They were not freshly accessible on the public portal. Confirm the latest participant rules before submission. The public page\'s application-selection criteria are a different rubric. '+ref(14))
    d.bullet('Impact: show a costly task getting easier.','Prepare an equipment procedure, identify the authoring burden and measure one small benefit. A useful output is more persuasive than a long future-use list.')
    d.bullet('Meta tooling: make it essential.','Show actual Meta segmentation/tracking, then genuine SAM 3D and Meta reasoning if access and quality permit. Disclose TripoSR/OpenAI substitutes; do not claim planned models ran.')
    d.bullet('Technical depth: connect the hard parts.','Preserved native assets, source-timed fitting, visibility gates and evidence compilation should produce a reusable result. Software test counts support reliability, not physical truth.')
    d.bullet('Demo: one clear transformation.','Ordinary recording becomes an interactive, queryable asset. Show the source, output, source-linked interaction and agent reuse in three minutes.')

    d.start('18 / Presentation','The three-minute story.',
            'Suggested storyboard. Use real outputs and identify cached/precomputed segments.',key='demo')
    d.table(['Time','Show','What the audience learns'],[
        ('0:00-0:20','The task + raw recording','A real authoring problem and a familiar input.'),
        ('0:20-0:45','Automatic part masks','What the perception models isolate; show the true model names.'),
        ('0:45-1:25','Video beside 3D + motion','A useful visual transformation. Briefly expose a withheld interval.'),
        ('1:25-2:00','Click relation / step / evidence','The result understands a procedure in an inspectable, fallible way.'),
        ('2:00-2:25','Retrieve + export through MCP','The structure is reusable. Label a scripted client as scripted.'),
        ('2:25-2:45','Second recording or fresh run','The workflow is not only a single hand-built scene.'),
        ('2:45-3:00','Measured payoff + next gate','What it saves and what still needs physical validation.'),
    ],[76,165,CW-241])
    d.note('Suggested opening','"A recording shows what happened, but it is hard to reuse. Interaction Twin turns it into parts, observed motion and an evidence-linked procedure that people and agents can inspect."')
    d.p('<b>Finish on the payoff:</b> an asset ready for inspection, reuse and further validation. A future Isaac Sim clip can extend this story only if it actually exists and its task has been checked.')
    d.p('Keep a saved-result fallback and a short real processing excerpt. Never edit a gap into apparent continuous recovery, or present a montage as live inference.','small')

    d.start('19 / Until the deadline','A focused working plan.',
            'Conditional plan from this snapshot. Access and model quality are not guaranteed.',key='schedule')
    d.table(['Date (Berlin)','Main objective'],[
        ('30 Oct','If authorized access is ready: one genuine SAM 3D component; inspect native output and test one bounded Meta reasoning request. Otherwise improve capture/evaluation data.'),
        ('31 Oct-1 Nov','Compare matched evidence, keep stronger originals, test a second clear recording. Narrow the capture assumptions if needed.'),
        ('2 Nov','Measure one authoring payoff. Freeze scope and avoid speculative major integrations.'),
        ('3 Nov','Record the three-minute demo; prepare source/model disclosures, repository and a saved-result fallback.'),
        ('4 Nov','Check portal rules and contribution boundaries. Suggested internal submission buffer: 18:00 Berlin.'),
    ],[111,CW-111])
    d.p('<b>Published deadline:</b> 4 November, 21:00 Eastern = <b>5 November, 03:00 CET in Berlin</b>. Recheck the portal for changes. '+ref(14)+' '+ref(15))
    d.p('Your participant-guide snapshot requires an accessible <b>public repository and three-minute demo video</b>, with the hackathon-built contributions identified. Recheck current rules; keep credentials, checkpoints and private source material out of the public repository.','small')
    d.h2('Record a better input now')
    d.p('Choose a rigid matte object with a removable part. Keep the camera steady, fill the frame, move slowly, pause in clear states and show each part separately from several sides. Let the cavity be visible. Add a known dimension if later metric scale matters.')
    d.p('Ray-Ban hands-free capture is a promising input option, but untested here. Head motion, blur and hands blocking contact can make alignment harder. A clear phone recording can be the stronger first demo; capture quality matters more than the camera brand.','small')

    d.start('20 / FAQ','Questions you will probably ask.',key='faq1')
    faqs=[
        ('Can we understand any YouTube video?', 'No. Cuts, zooms, blur, partial views and tiny parts can defeat both reconstruction and tracking. The MVP should declare a bounded capture setup and preserve source rights for anything distributed.'),
        ('What is the difference between mesh, splat and texture?', 'A mesh is a triangle surface. A splat represents appearance with coloured 3D Gaussian elements. A UV texture maps an image onto a mesh. None alone proves collision or contact accuracy.'),
        ('Why not just use a depth model and scaled ICP?', 'Predicted depth is uncertain and shows only visible surfaces. ICP refines a plausible initial alignment; it can match the wrong side or scale without a good camera/object setup. We combine reference fitting, correspondences and rejection gates.'),
        ('Does the app know that the earbud is charging?', 'No. A visual appearance near a recess does not establish electrical contact or successful seating. Those states remain unknown without supporting evidence or another sensor/test.'),
        ('Does the reasoning model decide everything?', 'It proposes meanings and candidate relationships. Geometry/tracking supply separate observations, the compiler checks references, and the viewer exposes the evidence. Those checks still cannot guarantee factual correctness.'),
        ('Did you, the coding agent, make the result by hand?', 'The recent learned-model route uses automatic view selection and actual reconstruction/tracking models without hand-designed meshes or poses. Earlier pilots had disclosed manual proxies/seeds. Software authorship and model-produced extraction are different roles.'),
    ]
    for q,a in faqs:d.bullet(q,a)

    d.start('21 / FAQ','What success would mean.',key='faq2')
    faqs=[
        ('Is the project finished?', 'The local app works. The central automatic geometry/alignment result is still partial; physical assembly and insertion are unfinished. The latest CPU candidate did not beat the original.'),
        ('Will larger models automatically fix it?', 'They may improve masks, shape and interpretation. They cannot recover evidence that was never visible, remove monocular scale ambiguity by measurement, or prove physics. Every replacement needs a real comparison.'),
        ('Could robots train with it?', 'Eventually, for validated tasks. First we need calibrated units, suitable geometry/collisions, dynamics and actual simulator tests. MCP retrieval today is useful preparation, not robot policy training.'),
        ('Are our digital twins programs like Polyfork?', 'They are descriptive interaction programs: components, states, relations, steps and evidence. They are not recovered CAD factories with reliable geometric parameters, nor executable robot controllers.'),
        ('What is the fairest NVIDIA connection?', 'A proposed bridge from custom captured interactions to the wider OpenUSD/SimReady ecosystem. Their prepared assets can provide the surrounding scene; our captured object still needs a capability-specific validation pass.'),
        ('Do we have a good chance to win?', 'There is a credible story and substantial implementation. A probability is not supportable. The strongest improvement is a beautiful, repeatable source-to-asset result with a clear professional payoff and genuine Meta model use.'),
    ]
    for q,a in faqs:d.bullet(q,a)
    d.note('Our next decisive milestone','One fresh, clean interaction becomes recognisable generated parts, useful visible motion and an evidence-linked procedure, then another person or agent reuses it.','mint')

    d.start('22 / Sources','Primary research and product sources.',
            'Public pages checked on 29 October 2026. Click a title to open the primary source.',key='sources')
    descriptions={
        1:'Current segmentation backend.',2:'Planned reconstruction capability; setup requires its supported GPU runtime.',
        3:'Newer segmentation model; distinguish upstream features from our anchor-only adapter.',
        4:'Actual learned point tracking.',5:'Current non-Meta single-image reconstruction baseline.',6:'Earlier depth/camera-support experiments.',
        7:'Dated announcement: more than 2,000 assets, 21 Oct 2026.',8:'Dynamic count and CC-BY-4.0; 2,677 listed at check time.',
        9:'Profile-specific simulation requirements and validation.',10:'Registry access, licensing and versioned content.',
        11:'Existing AI-assisted authoring overlap; simulation tests are distinct from real dynamics.',
        12:'Official asset-remixing product, not proof of arbitrary reconstruction.',13:'Agent-facing catalogue and asset metadata.',
        14:'Public hackathon timing and Indie themes.',15:'Organizer confirmation of deadline/finals.',16:'Articulated monocular reconstruction and motion constraints.'}
    for n,(title,url) in enumerate(SOURCES,1):
        d.p(f'<b>{n:02d}</b> <link href="{url}" color="#2D7161"><b>{escape(title)}</b></link><br/>{descriptions[n]}','small',after=8)
    d.p('The detailed 30/25/20/25 judging weights come from your supplied participant-guide snapshot, not a newly accessible public final-rules page. Confirm the current guide before submission.','small')

    d.start('23 / Receipts + glossary','How to verify this guide.',
            'Local evidence paths are relative to the interaction-twin project folder. These records are snapshots, not external certifications.',key='receipts')
    d.table(['Evidence','What it supports'],[
        ('output/native-appearance/<br/>delivery-qa-20261029.json','Latest scope, 454 pass / 1 skip, CPU facts, unchanged originals and known limits.'),
        ('output/native-appearance/<br/>full-cpu-validation-20261029.json','Exact 12-timestamp comparison, input/output hashes and preserved downstream artifacts.'),
        ('output/benchmark-local-v4/<br/>report.json','Ten valid saved cases, including variants; not independent physical ground truth.'),
        ('output/mcp/<br/>smoke-ten-catalog-20261029.json','Actual MCP protocol checks against the saved catalogue.'),
        ('docs/pipeline.md<br/>docs/model-upgrades.md<br/>docs/model-handoffs.md','Local processing, cached upgrade modes and prepared GPU exchange.'),
        ('output/pdf-assets/<br/>reader-20261029/screenshots.json','Real screenshot URLs, hashes and the declared workspace crop. Diagrams are explanatory vectors.'),
    ],[219,CW-219])
    d.p('<b>Mask:</b> selected image pixels. <b>Pose:</b> position and orientation. <b>Intrinsics:</b> camera projection settings. <b>PnP:</b> pose from 3D points and their image locations. <b>Gauge:</b> a coordinate/scale convention that may lack physical units.')
    d.p('<b>GLB:</b> portable geometry/material container. <b>OpenUSD:</b> scene description used by the proposed simulation bridge. <b>MCP:</b> a tool/resource interface for clients. <b>SimReady profile:</b> requirements for a specific simulation use, not a universal accuracy badge.')
    d.p('<b>Useful local pages:</b> <link href="http://127.0.0.1:8765/showcase/index.html" color="#2D7161">Project</link> | <link href="http://127.0.0.1:8765/workspace/index.html" color="#2D7161">Workspace</link> | <link href="http://127.0.0.1:8765/workspace/upgrades.html" color="#2D7161">Model Lab</link> | <link href="http://127.0.0.1:8765/workspace/agents.html" color="#2D7161">Agent catalog</link>. These require the local processor on this computer.','small')
    d.p('This document was prepared from current local records, actual app captures and primary public sources. Its preparation used no model inference, paid requests, cluster access or public uploads. Revisit the quality and access statements after the next genuine model run.','small')

    used=d.save()
    (ASSETS/'layout.json').write_text(json.dumps({'pages':len(used),'layout':used,'pdf':str(OUT)},indent=2))
    print(json.dumps({'pdf':str(OUT),'pages':len(used),'bytes':OUT.stat().st_size,'lowest_content_y':min(x['bottom_y'] for x in used)},indent=2))

if __name__=='__main__':build()
