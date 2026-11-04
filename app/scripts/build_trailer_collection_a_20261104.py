#!/usr/bin/env python3
"""Three different edits of genuine Interaction Twin source-bound captures.

Only the editorial canvas, crop, clock and graphics change. Saved reconstruction,
native component coordinates and robot task states are never rewritten. No API,
SSH, credential access or media download is performed by this renderer.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('verified_dynamic_v3', ROOT / 'scripts/build_dynamic_trailer_20261104_v3.py')
OLD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OLD)
V = OLD.V
W, H, FPS = 1920, 1080, 30
PAPER, INK, BLUE, MUTED = '#f7f8fb', '#111725', '#3766ed', '#68758a'
DARK, PALE, LINE = '#0b1222', '#ecf2ff', '#243656'
OUT = ROOT / 'output/trailer-collection-20261104-v1/set-a'
MEDIA = ROOT / 'showcase/media/trailer-collection-20261104-v1'
MASTER = ROOT / 'output/motion-trailer-20261104-v5/interaction-twin-motion-20261104-v5.webm'
MASTER_JSON = MASTER.with_suffix('.json')
CACHE = ROOT / 'output/dynamic-trailer-20261104-v3/render-candidate-v5'
MASTER_FRAMES = CACHE / 'master-frames'
ROBOT_FRAMES = CACHE / 'robot-frames'
ORBIT = ROOT / 'output/trailer-collection-20261104-v1/native-orbits/native-orbits-20261104-v1.webm'
ORBIT_JSON = ORBIT.with_suffix('.json')

STORIES = {
    '01-origin': {
        'title': 'The origin story', 'duration_s': 43,
        'story': 'A human demonstration becomes separate parts, connected motion and a prototype robot task.',
        'look': 'Human-first, warm documentary framing, large authentic footage, restrained narration and spacious piano pulses.',
        'music': {'bpm': 84, 'kind': 'documentary', 'seed': 1104},
        'voice': 'Samantha',
        'cues': [
            (0.2, 6.1, 'Everyday objects already know how to move. We show them once.'),
            (6.3, 11.4, 'SAM separates the parts in the recording.'),
            (11.6, 17.7, 'SAM 3D gives them shape. Muse helps describe how they connect.'),
            (18.0, 24.3, 'Our pipeline fits the movement and keeps its source evidence.'),
            (24.5, 29.0, 'A twin becomes a readable object program.'),
            (29.4, 40.5, 'Then use its parts and connections to prototype a robot task. Open the case. Move both earbuds to the tray. Close the case.'),
            (41.0, 43.0, 'Interaction Twin.'),
        ],
        'chapters': [(0, 6.2, 'A real interaction'), (6.2, 11.5, 'Separate the parts'), (11.5, 18, 'Give them shape'), (18, 24.5, 'Keep the connection'), (24.5, 29.3, 'Describe how it works'), (29.3, 41, 'Prototype the next move'), (41, 43, 'Interaction Twin')],
    },
    '02-blueprint': {
        'title': 'The object program', 'duration_s': 34,
        'story': 'The anatomy of a twin unfolds as a live technical graph: parts, one hinge, source-bound motion, then an agent query.',
        'look': 'Dark cobalt technical cinema, moving native geometry, animated relationship graph and rhythmic sound design; no narration.',
        'music': {'bpm': 112, 'kind': 'blueprint', 'seed': 2104},
        'voice': None,
        'cues': [
            (0, 5.4, 'A shape is only the beginning.'),
            (5.4, 11, 'Find the parts. Keep their identities.'),
            (11, 19, 'Understand the connection: a lid that stays on its hinge.'),
            (19, 26, 'Keep geometry, motion and evidence in one object program.'),
            (26, 32, 'Let an agent find the interaction it needs.'),
            (32, 34, 'Interaction Twin. Show it once. Keep how it works.'),
        ],
        'chapters': [(0, 5.4, 'Beyond shape'), (5.4, 11, 'Part identities'), (11, 19, 'Connection graph'), (19, 26, 'Object program'), (26, 32, 'Agent discovery'), (32, 34, 'Interaction Twin')],
    },
    '03-mission': {
        'title': 'The next move', 'duration_s': 39,
        'story': 'A clear robot mission runs to both released goals and a closed case, before revealing the real demonstration behind it.',
        'look': 'Goal-first robotics field journal, a large uninterrupted task view, live goal checks and a reveal of the human source.',
        'music': {'bpm': 98, 'kind': 'mission', 'seed': 3104},
        'voice': 'Daniel',
        'cues': [
            (0.1, 6.1, 'The task: open the case. Place both earbuds on the tray. Close the case.'),
            (7.2, 10.7, 'The lid stays connected.'),
            (12.8, 17.1, 'Each earbud has its own goal.'),
            (21.0, 26.7, 'Both earbuds placed. Now close the case.'),
            (27.2, 31.6, 'It started with a real human demonstration.'),
            (32.0, 36.5, 'Video, parts, connections and an object program. A starting point for robotics.'),
            (37.0, 39, 'Interaction Twin.'),
        ],
        'chapters': [(0, 2, 'The mission'), (2, 27, 'Complete robot rehearsal'), (27, 32, 'The origin revealed'), (32, 37, 'The reusable twin'), (37, 39, 'Interaction Twin')],
    },
}


def ref(p):
    p = Path(p).resolve()
    return {'path': p.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}


def write(p, value):
    Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def ease(v):
    v = max(0, min(1, v))
    return v * v * (3 - 2 * v)


def font(size, bold=False, mono=False):
    path = '/System/Library/Fonts/Menlo.ttc' if mono else '/System/Library/Fonts/Supplemental/Arial' + (' Bold' if bold else '') + '.ttf'
    return ImageFont.truetype(path, size)


FONTS = {}


def f(size, bold=False, mono=False):
    key = size, bold, mono
    if key not in FONTS:
        FONTS[key] = font(*key)
    return FONTS[key]


def text(d, xy, s, size=30, color=INK, bold=False, mono=False, spacing=7):
    d.multiline_text(xy, s, font=f(size, bold, mono), fill=color, spacing=spacing)


def contain(im, source, box, bg=None, radius=0):
    x, y, w, h = [round(v) for v in box]
    if bg:
        ImageDraw.Draw(im).rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=bg)
    art = ImageOps.contain(source, (w, h), Image.Resampling.LANCZOS)
    if radius:
        card = Image.new('RGB', (w, h), bg or PAPER)
        card.paste(art, ((w-art.width)//2, (h-art.height)//2))
        mask = Image.new('L', (w, h)); ImageDraw.Draw(mask).rounded_rectangle((0, 0, w-1, h-1), radius=radius, fill=255)
        im.paste(card, (x, y), mask)
    else:
        im.paste(art, (x + (w-art.width)//2, y + (h-art.height)//2))


def pill(d, xy, s, *, dark=False, color=BLUE, width=None):
    x, y = xy
    width = width or d.textbbox((0, 0), s, font=f(19, True))[2] + 34
    d.rounded_rectangle((x, y, x+width, y+36), radius=9, fill='#192940' if dark else '#e9efff')
    text(d, (x+17, y+7), s, 19, '#93b2ff' if dark else color, True)


class Assets:
    def __init__(self):
        self.md = OLD.verify_motion(MASTER, MASTER_JSON)
        self.robot_ref = V.checked(V.ROBOT_REF)
        self.robot_json = V.checked(V.ROBOT_JSON)
        self.robot = OLD.verify_robot(self.robot_json)
        self.source = V.checked(self.md['source_ref'])
        self.source_sha = self.md['source_ref']['sha256']
        self.logo = Image.open(V.checked(V.LOGO_REF)).convert('RGBA')
        self.cache = OrderedDict()
        self.frame_counts = {}
        self.reader = None
        self.orbit_reader = None
        self.robot_crop = V.robot_crop(self.robot)
        self.geometry_crop = (590, 280, 1380, 1010)
        # The previous compositor proves this pre-existing caption rectangle is
        # disjoint from ALL conservatively projected native component bounds.
        clearance = json.loads((CACHE/'hero-caption-clearance.json').read_text())
        if clearance['overlaps'] != 0 or clearance['component_bounds_checks'] != 2040:
            raise ValueError('The source caption removal needs its exact geometry-clearance proof')
        orbit = json.loads(ORBIT_JSON.read_text())
        if orbit.get('format') != 'interaction-twin-native-editorial-orbits' or orbit.get('geometry_modified') is not False or orbit.get('physical_accuracy_verified') is not False or orbit.get('video_sha256') != ref(ORBIT)['sha256']:
            raise ValueError('Exact independent native-orbit capture required')
        for asset in orbit['assets']:
            V.checked(asset['asset_ref'])
        for folder, minimum in [(MASTER_FRAMES, 1106), (ROBOT_FRAMES, 805)]:
            count=len(list(folder.glob('*.jpg')))
            if count < minimum:
                raise ValueError('Existing verified decoded frame cache is incomplete')
            self.frame_counts[folder]=count

    def jpg(self, folder, t):
        count = self.frame_counts[folder]
        index = max(1, min(count, round(t * FPS) + 1))
        p = folder / f'{index:05}.jpg'
        if not p.is_file():
            raise ValueError('Missing saved decoded frame: ' + str(p))
        key = str(p)
        if key not in self.cache:
            self.cache[key] = Image.open(p).convert('RGB')
        im = self.cache.pop(key); self.cache[key] = im
        while len(self.cache) > 10:
            self.cache.popitem(last=False)
        return im

    def hero(self, progress):
        t = 9 + 2/FPS + max(0, min(1, progress)) * (9 - 4/FPS)
        image = self.jpg(MASTER_FRAMES, t).copy()
        # Remove the old EDITORIAL caption only, not model pixels. Exact saved
        # native bounds demonstrate >=13.9 px of separation from this rectangle.
        ImageDraw.Draw(image).rectangle((77, 912, 641, 982), fill=PAPER)
        return image.crop(self.geometry_crop), t

    def sam(self, progress):
        t = 4 + 2/FPS + max(0, min(1, progress)) * (5-4/FPS)
        return self.jpg(MASTER_FRAMES, t).crop((63, 310, 893, 990)), t

    def program(self, progress):
        t = 18 + 2/FPS + max(0, min(1, progress)) * (6-4/FPS)
        return self.jpg(MASTER_FRAMES, t).crop((65, 298, 1870, 1000)), t

    def raw_source(self, source_s, crop='intro'):
        # Source-only shots can show any original filmed frame. This does not
        # promote its unavailable reconstruction or invent geometry.
        if self.reader is None or source_s < self.reader.time - .05:
            if self.reader: self.reader.close()
            self.reader = RawReader(self.source, source_s)
        picture = self.reader.get(source_s)
        return picture.crop(tuple(self.md['editorial_source_crops'][crop]))

    def robot_image(self, source_s):
        return self.jpg(ROBOT_FRAMES, source_s).crop(self.robot_crop)

    def orbit(self, source_s):
        if self.orbit_reader is None or source_s < self.orbit_reader.time - .05:
            if self.orbit_reader: self.orbit_reader.close()
            self.orbit_reader = RawReader(ORBIT, source_s, (1080, 1080))
        return self.orbit_reader.get(source_s)

    def robot_state(self, source_s):
        return min(self.robot['trace'], key=lambda row: abs(row['time_s'] - source_s))

    def close(self):
        if self.reader: self.reader.close(); self.reader = None
        if self.orbit_reader: self.orbit_reader.close(); self.orbit_reader = None


class RawReader:
    """Bounded streaming decode of the exact owner source; no disk frame dump."""
    def __init__(self, path, start, dimensions=(540,960)):
        self.start = start; self.index = -1; self.time = start; self.picture = None
        self.width,self.height=dimensions
        self.process = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-ss', str(start), '-i', str(path), '-vf', f'fps=30,scale={self.width}:{self.height}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def get(self, time_s):
        target = max(0, round((time_s-self.start)*FPS))
        while self.index < target:
            raw = self.process.stdout.read(self.width*self.height*3)
            if len(raw) != self.width*self.height*3:
                raise ValueError('Original source decoder ended early')
            self.picture = Image.frombytes('RGB', (self.width,self.height), raw)
            self.index += 1
        self.time = self.start + self.index/FPS
        return self.picture

    def close(self):
        self.process.stdout.close()
        if self.process.poll() is None: self.process.terminate()
        self.process.wait(timeout=10)


def brand(im, assets, *, dark=False, large=False):
    d = ImageDraw.Draw(im)
    n = 48 if large else 40
    logo = assets.logo.resize((n, n), Image.Resampling.LANCZOS)
    im.paste(logo, (68, 49), logo)
    text(d, (68+n+17, 54), 'Interaction Twin', 27 if large else 24, PALE if dark else INK, True)


def ending(t, assets, *, dark=False, mission=False):
    im = Image.new('RGB', (W, H), DARK if dark else PAPER)
    d = ImageDraw.Draw(im); e = ease(t/.6)
    size = 138; logo = assets.logo.resize((size, size), Image.Resampling.LANCZOS)
    im.paste(logo, (310, round(405+(1-e)*15)), logo)
    text(d, (500, round(395+(1-e)*15)), 'Interaction Twin', 88, PALE if dark else INK, True)
    text(d, (506, 522), 'Show it once. Keep how it works.', 38, '#94a6c4' if dark else MUTED)
    return im


def origin_frame(t, a):
    im = Image.new('RGB', (W, H), PAPER); d = ImageDraw.Draw(im); brand(im, a)
    if t < 6.2:
        p = t/6.2; shot = a.raw_source(14.5+p*5.2)
        contain(im, shot, (835, 145, 1000, 842), radius=28, bg='#e7e9ed')
        pill(d, (75, 208), 'IT STARTS IN THE REAL WORLD')
        text(d, (70, 291), 'A moment.\nA movement.\nA way it works.', 75, INK, True, spacing=6)
        text(d, (76, 630), 'Ordinary video.\nExtraordinary reusable detail.', 30, MUTED, spacing=10)
        d.line((75, 900, 75+round(620*ease(p)), 900), fill=BLUE, width=5)
        return im, {'kind':'original_source_video','source_s':14.5+p*5.2}
    if t < 11.5:
        p = (t-6.2)/5.3; shot, clock = a.sam(p)
        contain(im, shot, (85, 235, 1020, 750), radius=28, bg='#e9edf4')
        pill(d, (1190, 285), 'SAM 3.1')
        text(d, (1185, 362), 'The parts\ncome into focus.', 64, INK, True)
        for i, label in enumerate(['Case body', 'Connected lid', 'Two earbuds']):
            y=605+i*76
            d.ellipse((1191, y+8, 1204, y+21), fill=['#5d83ff','#e5ad63','#54b8b0'][i])
            text(d, (1226, y), label, 30, MUTED)
        return im, {'kind':'actual_saved_sam_masks','master_s':clock,'saved_observations_repeated_for_display':True}
    if t < 18:
        p=(t-11.5)/6.5; shot, clock=a.hero(p)
        contain(im, shot, (760, 170, 1080, 830))
        pill(d, (75, 275), 'SAM 3D + MUSE')
        text(d, (70, 354), 'Give shape\na little meaning.', 73, INK, True)
        text(d, (76, 608), 'Reconstruct each part.\nUnderstand the connections.', 30, MUTED)
        return im, {'kind':'source_bound_native_scene','master_s':clock}
    if t < 24.5:
        p=(t-18)/6.5; shot, clock=a.hero(p)
        contain(im, shot, (55, 181, 1070, 840))
        pill(d, (1185, 273), 'CONNECTED MOTION')
        text(d, (1180, 348), 'The lid\nstays a lid.', 70, INK, True)
        text(d, (1186, 580), 'One hinge.\nSeparate removable parts.\nEvidence kept with every twin.', 29, MUTED, spacing=12)
        return im, {'kind':'source_bound_native_scene','master_s':clock}
    if t < 29.3:
        p=(t-24.5)/4.8; shot, clock=a.hero(p)
        contain(im, shot, (1010, 202, 810, 760))
        pill(d, (75, 210), 'THE OBJECT PROGRAM')
        text(d, (70, 286), 'Not just what it is.\nHow it works.', 72, INK, True)
        labels=['Parts','Connections','Actions','Evidence']
        for i,label in enumerate(labels):
            x=77+(i%2)*388; y=578+(i//2)*99
            d.rounded_rectangle((x,y,x+345,y+70),radius=15,fill='white',outline='#dfe5ef')
            text(d,(x+23,y+18),label,29,BLUE,True)
        return im, {'kind':'source_bound_native_scene_plus_editorial_saved_field_labels','master_s':clock}
    if t < 41:
        p=(t-29.3)/11.7; source=min(26.9,p*26.9); row=a.robot_state(source)
        shot=a.robot_image(source)
        im=Image.new('RGB',(W,H),shot.getpixel((0,0)));d=ImageDraw.Draw(im);brand(im,a)
        contain(im,shot,(48,120,1185,940))
        pill(d,(1282,206),'ROBOTICS LAB')
        text(d,(1276,282),'Build the\nnext move.',64,INK,True)
        phase=mission_phase(row)
        text(d,(1282,511),phase[0],31,BLUE,True)
        goal_cards(d,row,(1282,593),compact=True)
        text(d,(1282,913),'Kinematic task preview',22,MUTED)
        return im, {'kind':'complete_authored_robot_task','robot_source_s':source,'goals':row['goals'],'joint_q_rad':row['joint_q_rad'],'physical_contact_verified':False}
    return ending(t-41,a), {'kind':'brand'}


def scan_grid(d,t):
    for x in range(66,W,80): d.line((x,142,x,H-75),fill='#142139',width=1)
    for y in range(145,H-60,80): d.line((66,y,W-66,y),fill='#142139',width=1)
    y=180+int((t*.09%1)*730)
    d.line((66,y,W-66,y),fill='#213658',width=1)


def technical_view(im,shot,box):
    # Preserve native pixels on an intentionally light material-inspection stage.
    # This is a composited viewport, not a relit or recoloured reconstruction.
    contain(im,shot,box,bg=PAPER,radius=24)


def graph(d,p,angle):
    x,y=1260,360
    d.rounded_rectangle((x+45,y,x+565,y+61),radius=13,fill='#152a4a',outline='#36537d')
    text(d,(x+67,y+18),'case_base',23,PALE,True,mono=True)
    if p>.1:
        d.line((x+305,y+61,x+305,y+186),fill='#628df4',width=3)
        pill(d,(x+205,y+100),'REVOLUTE',dark=True,width=194)
        d.rounded_rectangle((x+45,y+186,x+565,y+247),radius=13,fill='#152a4a',outline='#36537d')
        text(d,(x+67,y+204),'case_lid',23,PALE,True,mono=True)
        text(d,(x+83,y+267),f'q estimate  {angle:.3f} rad',23,'#9fbdfd',mono=True)
    if p>.35:
        # A separate dotted containment branch, never a fictitious hinge chain
        # from the lid through the two removable earbuds.
        path=[(x+45,y+30),(x-32,y+30),(x-32,y+410),(x+435,y+410)]
        for start,end in zip(path,path[1:]):
            length=math.dist(start,end);steps=math.ceil(length/16)
            for k in range(0,steps,2):
                aa=k/steps;bb=min(1,(k+1)/steps)
                d.line((tuple(start[i]+(end[i]-start[i])*aa for i in range(2)),tuple(start[i]+(end[i]-start[i])*bb for i in range(2))),fill='#42618e',width=2)
        text(d,(x+48,y+350),'REMOVABLE PARTS',20,'#94a8c8',True)
        for i,label in enumerate(['earbud_a','earbud_b']):
            xx=x+i*292
            d.line((xx+137,y+410,xx+137,y+448),fill='#42618e',width=2)
            d.rounded_rectangle((xx,y+448,xx+272,y+510),radius=13,fill='#132135',outline='#36537d')
            text(d,(xx+24,y+468),label,23,PALE,True,mono=True)


def blueprint_frame(t,a):
    im=Image.new('RGB',(W,H),DARK);d=ImageDraw.Draw(im);scan_grid(d,t);brand(im,a,dark=True)
    if t<5.4:
        p=t/5.4;clock=.05+p*2.83;shot=a.orbit(clock)
        technical_view(im,shot,(760,178,1065,830))
        pill(d,(75,242),'OBJECT / 001',dark=True)
        text(d,(70,324),'A shape is\nonly the\nbeginning.',78,PALE,True,spacing=4)
        text(d,(77,683),'What can it do?',31,'#8ca4ca')
        text(d,(810,931),'case_base / generated native part',20,MUTED,mono=True)
        return im,{'kind':'independent_native_part_editorial_orbit','orbit_source_s':clock,'component':'case_base','recovered_pose':False}
    if t<11:
        p=(t-5.4)/5.6;shot,clock=a.sam(p)
        technical_view(im,shot,(75,235,990,725))
        pill(d,(1178,204),'SEGMENT / IDENTIFY',dark=True)
        text(d,(1173,282),'Find the parts.\nKeep the IDs.',63,PALE,True)
        labels=['case_base','case_lid','earbud_a','earbud_b']
        for i,label in enumerate(labels):
            reveal=ease(p*3-i*.25)
            x=1180+round((1-reveal)*70); y=522+i*88
            if reveal:
                d.rounded_rectangle((x,y,1828,y+65),radius=13,fill='#13233c',outline='#2d4772')
                text(d,(x+26,y+18),label,25,PALE,mono=True)
        return im,{'kind':'actual_saved_sam_masks','master_s':clock,'saved_observations_repeated_for_display':True}
    if t<19:
        p=(t-11)/8;shot,clock=a.hero(p)
        technical_view(im,shot,(75,236,1075,725))
        text(d,(75,153),'UNDERSTAND THE CONNECTION',26,'#9bb8fa',True)
        text(d,(1260,200),'A lid.\nOn its hinge.',61,PALE,True)
        trace=min((r for r in a.md['projection_trace']if r['chapter_index']==2),key=lambda r:abs(r['film_s']-clock))
        angle=trace['joints'][0]['q_rad'];graph(d,p,angle)
        return im,{'kind':'source_bound_native_scene_and_exact_joint_graph','master_s':clock,'joint_q_rad_estimate':angle}
    if t<26:
        p=(t-19)/7;shot,clock=a.hero(p)
        technical_view(im,shot,(1052,229,770,760))
        pill(d,(75,194),'THE OBJECT PROGRAM',dark=True)
        text(d,(70,267),'Geometry has\na readable side.',67,PALE,True)
        code=['parts: [case_base, case_lid,','        earbud_a, earbud_b]','joint: revolute','parent: case_base','child: case_lid','actions: open · remove · return','evidence: video + masks']
        limit=max(1,min(len(code),math.ceil(p*len(code)*1.3)))
        for i,line in enumerate(code[:limit]):text(d,(79,515+i*52),line,27,'#a7c3ff'if i in[0,2,5]else'#b1bed3',mono=True)
        return im,{'kind':'source_bound_native_scene_and_editorial_readable_saved_field_excerpt','master_s':clock,'excerpt_not_verbatim_serialization':True}
    if t<32:
        p=(t-26)/6;shot,clock=a.hero(p)
        technical_view(im,shot,(1015,216,807,774))
        pill(d,(75,205),'ASK YOUR AGENT',dark=True)
        text(d,(70,282),'Find the next\ninteraction.',69,PALE,True)
        query='Find a hinged case with\nremovable earbuds.'
        visible=query[:min(len(query),math.ceil(p*2.7*len(query)))]
        d.rounded_rectangle((75,530,947,692),radius=18,fill='#15233b',outline='#4579f2',width=2)
        text(d,(100,558),visible,31,PALE,mono=True)
        if p>.43:
            text(d,(81,764),'FOUND  /  Earbuds + charging case',28,'#9fc0ff',True)
            text(d,(82,821),'4 parts  ·  connected lid  ·  source evidence',23,'#94a8c8')
            text(d,(82,906),'Search. Inspect. Download. Build.',27,PALE,True)
        return im,{'kind':'source_bound_native_scene_plus_saved_successful_agent_query','master_s':clock,'query_response_ref':'capture.actual_agent_filter_response'}
    return ending(t-32,a,dark=True),{'kind':'brand'}


def mission_phase(row):
    action=row['action_id']; n=int(action.split('_')[-1])if action.startswith('action_')else 0
    goals=sum(g['released']and g['reached']for g in row['goals'])
    if action=='film_final_hold':return 'Mission complete.',4
    if action=='film_withdraw_arm' or n>=36:return 'Close the case.',4
    if goals==2:return 'Both earbuds placed.',3
    if 25<=n<=35:return 'Transfer earbud B.',3
    if 14<=n<=24:return 'Transfer earbud A.',2
    if 12<=n<=13:return 'Open the connected lid.',1
    return 'Place the case.',0


def goal_cards(d,row,origin,compact=False):
    x,y=origin; width=520 if compact else 545; spacing=96 if compact else 109
    for i,g in enumerate(row['goals']):
        reached=g['released']and g['reached']; yy=y+i*spacing
        d.rounded_rectangle((x,yy,x+width,yy+78),radius=14,fill='#eaf0ff'if reached else 'white',outline='#cad7f8'if reached else'#dce3ed')
        text(d,(x+23,yy+24),'A'if i==0 else'B',26,BLUE,True)
        text(d,(x+70,yy+15),'Earbud '+('A'if i==0 else'B'),25,INK,True)
        text(d,(x+70,yy+45),'Placed + released'if reached else'Tray goal pending',19,BLUE if reached else MUTED)
        if reached:
            d.line((x+width-48,yy+41,x+width-40,yy+49,x+width-25,yy+29),fill=BLUE,width=4)


def mission_frame(t,a):
    im=Image.new('RGB',(W,H),'#f3f5f9');d=ImageDraw.Draw(im);brand(im,a)
    if t<27:
        source=0 if t<2 else min(26.9,(t-2)*26.9/25)
        row=a.robot_state(source);phase,index=mission_phase(row)
        shot=a.robot_image(source)
        im=Image.new('RGB',(W,H),shot.getpixel((0,0)));d=ImageDraw.Draw(im);brand(im,a)
        contain(im,shot,(38,120,1212,940))
        pill(d,(1294,184),'TASK / 001')
        if t<2:
            text(d,(1288,261),'Open the case.\nPlace both buds.\nClose the case.',50,INK,True,spacing=11)
        else:
            text(d,(1288,261),'The next\nmove.',67,INK,True)
            text(d,(1294,456),phase,30,BLUE,True)
        goal_cards(d,row,(1294,565))
        d.line((1294,834,1812,834),fill='#d3dceb',width=4)
        d.line((1294,834,1294+round(518*source/26.9),834),fill=BLUE,width=4)
        text(d,(1294,878),f'{source:04.1f} s  /  26.9 s',23,MUTED,mono=True)
        text(d,(1294,948),'Authored kinematic rehearsal',21,MUTED)
        return im,{'kind':'complete_authored_robot_task','robot_source_s':source,'goals':row['goals'],'joint_q_rad':row['joint_q_rad'],'attached':row['attached'],'physical_contact_verified':False}
    if t<32:
        p=(t-27)/5;shot=a.raw_source(15+p*5.2)
        contain(im,shot,(831,154,1015,844),radius=26,bg='#e5e8ec')
        pill(d,(75,242),'THE ORIGIN REVEALED')
        text(d,(70,328),'It started\nwith a person.\nAnd a video.',72,INK,True,spacing=7)
        text(d,(78,684),'A familiar interaction becomes\na reusable starting point.',30,MUTED,spacing=11)
        return im,{'kind':'original_source_video','source_s':15+p*5.2}
    if t<37:
        p=(t-32)/5;shot,clock=a.hero(p)
        contain(im,shot,(841,167,1010,833))
        pill(d,(75,237),'THE REUSABLE TWIN')
        text(d,(70,323),'The parts.\nThe connections.\nThe next task.',70,INK,True,spacing=7)
        text(d,(77,679),'Model + object program + evidence',28,BLUE,True)
        text(d,(77,741),'For agents building robotics prototypes.',27,MUTED)
        return im,{'kind':'source_bound_native_scene','master_s':clock}
    return ending(t-37,a,mission=True),{'kind':'brand'}


RENDERERS={'01-origin':origin_frame,'02-blueprint':blueprint_frame,'03-mission':mission_frame}


def make_music(path,duration,parameters):
    sr=48000; size=round(duration*sr); t=np.arange(size,dtype=np.float64)/sr
    rng=np.random.default_rng(parameters['seed']);audio=np.zeros(size,dtype=np.float64); beat=60/parameters['bpm'];kind=parameters['kind']
    # Original additive synthesis. No samples, imported tracks or external rights.
    chord_sets={'documentary':[[146.83,220,293.66,369.99],[130.81,196,261.63,329.63],[110,164.81,220,293.66],[130.81,196,261.63,349.23]],'blueprint':[[110,164.81,220,329.63],[98,146.83,196,293.66],[123.47,185,246.94,369.99],[82.41,123.47,164.81,246.94]],'mission':[[146.83,220,293.66,440],[130.81,196,261.63,392],[164.81,246.94,329.63,493.88],[110,164.81,220,329.63]]}
    chords=chord_sets[kind]
    for block,start in enumerate(np.arange(0,duration,beat*4)):
        chord=chords[block%4];mask=(t>=start)&(t<start+beat*4);tt=t[mask]-start
        envelope=np.minimum(1,tt/.7)*np.minimum(1,(beat*4-tt)/.7)
        pad=sum(np.sin(2*np.pi*h*tt+.02*np.sin(tt*.6))for h in chord)/len(chord)
        audio[mask]+=pad*envelope*(.075 if kind=='documentary'else .055)
        for note_i in range(8):
            st=start+note_i*beat/2; index=round(st*sr); n=min(size-index,round(beat*1.9*sr))
            if n<=0:continue
            nt=np.arange(n)/sr;hz=chord[(note_i+block)%len(chord)]*(2 if kind=='documentary'else 4)
            decay=np.exp(-nt*(3.5 if kind=='documentary'else 8))
            env=np.minimum(1,nt/.012)*decay
            note=(np.sin(2*np.pi*hz*nt)+.22*np.sin(2*np.pi*hz*2.003*nt))*env
            audio[index:index+n]+=note*(.075 if kind=='documentary'else .045)
    if kind!='documentary':
        for i,st in enumerate(np.arange(0,duration,beat)):
            index=round(st*sr);n=min(size-index,round(.18*sr));tt=np.arange(n)/sr
            kick=np.sin(2*np.pi*(48*tt+30*.04*(1-np.exp(-tt/.04))))*np.exp(-tt*26)
            audio[index:index+n]+=kick*.16
            if i%2:
                index2=round((st+beat/2)*sr);n2=min(size-index2,round(.065*sr))
                if n2>0:
                    tt2=np.arange(n2)/sr;noise=rng.normal(0,1,n2);noise=np.r_[0,np.diff(noise)]
                    audio[index2:index2+n2]+=noise*np.exp(-tt2*90)*.012
    fade=np.minimum(1,t/.9)*np.minimum(1,(duration-t)/1.5); audio*=fade
    left=audio*.82;right=(audio*.80+.018*np.sin(2*np.pi*261.63*t)*np.sin(t*.8))*fade
    pcm=np.stack([left,right],axis=1)
    with wave.open(str(path),'wb')as wf:
        wf.setnchannels(2);wf.setsampwidth(2);wf.setframerate(sr);wf.writeframes(np.int16(np.clip(pcm,-.92,.92)*32767).tobytes())


def make_audio(name,folder):
    story=STORIES[name];duration=story['duration_s'];music=folder/'original-score.wav';make_music(music,duration,story['music'])
    source=[{'start_s':a,'end_s':b,'text':txt}for a,b,txt in story['cues']]
    if story['voice']:
        vocals=[]
        for i,(start,end,words)in enumerate(story['cues']):
            aiff=folder/f'voice-{i:02}.aiff';wav=folder/f'voice-{i:02}.wav'
            V.run(['/usr/bin/say','-v',story['voice'],'-r','173','-o',str(aiff),words])
            raw=V.duration(aiff);available=end-start-.12;tempo=max(1.,raw/available)
            if tempo>1.38:raise ValueError('Narration too fast: '+words)
            V.run(['ffmpeg','-nostdin','-v','error','-i',str(aiff),'-af',f'atempo={tempo:.9f},loudnorm=I=-18:TP=-3:LRA=7','-ar','48000','-ac','2',str(wav)])
            actual=V.duration(wav)
            vocals.append({'start_s':start,'end_s':start+actual,'text':words,'raw':ref(aiff),'processed':ref(wav),'atempo':tempo,'local_voice':story['voice']})
        audio=folder/'mix.wav';args=['ffmpeg','-nostdin','-v','error','-i',str(music)];filters=[];labels=[]
        for i,vocal in enumerate(vocals,1):
            args+=['-i',str(ROOT/vocal['processed']['path'])];filters.append(f'[{i}:a]adelay={round(vocal["start_s"]*1000)}|{round(vocal["start_s"]*1000)},apad=whole_dur={duration}[v{i}]');labels.append(f'[v{i}]')
        filters.append(''.join(labels)+f'amix=inputs={len(labels)}:normalize=0,atrim=0:{duration},asplit=2[voice][side]')
        filters.append('[0:a]volume=.52[music];[music][side]sidechaincompress=threshold=.018:ratio=5:attack=12:release=270[duck];[duck][voice]amix=inputs=2:normalize=0,loudnorm=I=-17:TP=-3:LRA=8[a]')
        V.run(args+['-filter_complex',';'.join(filters),'-map','[a]','-ar','48000','-ac','2',str(audio)])
    else:
        audio=folder/'mix.wav';V.run(['ffmpeg','-nostdin','-v','error','-i',str(music),'-af','loudnorm=I=-19:TP=-3:LRA=8','-ar','48000','-ac','2',str(audio)]);vocals=[]
    write(folder/'audio.json',{'music':ref(music),'mix':ref(audio),'music_original_additive_synthesis':True,'parameters':story['music'],'voice':vocals,'hosted_speech':False,'third_party_samples':False})
    return audio,source


def captions(name,source):
    vtt='WEBVTT\n\n'
    for i,row in enumerate(source,1):
        vtt+=f'{i}\n{V.stamp(row["start_s"])} --> {V.stamp(row["end_s"])}\n{row["text"]}\n\n'
    (MEDIA/(name+'.vtt')).write_text(vtt)
    (MEDIA/(name+'-transcript.txt')).write_text('\n\n'.join(row['text']for row in source)+'\n')


def contact_sheet(proofs,target):
    sheet=Image.new('RGB',(1440,math.ceil(len(proofs)/3)*292),PAPER);d=ImageDraw.Draw(sheet)
    for i,row in enumerate(proofs):
        pic=Image.open(ROOT/row['file']['path']);pic.thumbnail((480,270));x=i%3*480;y=i//3*292;sheet.paste(pic,(x,y));text(d,(x+10,y+272),f'{row["time_s"]:.2f} s',14,MUTED)
    sheet.save(target,quality=94)


def storyboard(name,assets,folder):
    story=STORIES[name];proofs=[]
    times=sorted(set([a+.25 for a,_,_ in story['chapters']]+[story['duration_s']-1.0]))
    if name=='03-mission': times=sorted(set(times+[8,14,20,24.4,26.6]))
    for i,t in enumerate(times):
        picture,trace=RENDERERS[name](t,assets);p=folder/f'design-{i:02}.jpg';picture.save(p,quality=94)
        proofs.append({'time_s':t,'file':ref(p),'trace':trace})
    contact_sheet(proofs,folder/'design-contact-sheet.jpg')
    write(folder/'storyboard.json',{'name':name,'story':story,'frames':proofs})


def verify_video(movie,story):
    probe=json.loads(V.run(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(movie)]).stdout)
    video=next(s for s in probe['streams']if s['codec_type']=='video');audio=next(s for s in probe['streams']if s['codec_type']=='audio')
    if (video['width'],video['height'],int(video['nb_read_frames']),audio['channels'],int(audio['sample_rate']))!=(W,H,FPS*story['duration_s'],2,48000):raise ValueError('Final video clock/dimensions/audio failed')
    if abs(float(probe['format']['duration'])-story['duration_s'])>.001:raise ValueError('Trailer duration failed')
    black=V.run(['ffmpeg','-nostdin','-v','info','-i',str(movie),'-an','-vf','blackdetect=d=.06:pix_th=.05:pic_th=.99','-f','null','-']).stderr
    if 'black_start:' in black:raise ValueError('Unexpected black movie frames')
    return probe


def render(name,assets):
    started=time.monotonic();story=STORIES[name];folder=OUT/name;folder.mkdir(exist_ok=False,parents=True)
    (folder/'builder-source.py').write_bytes(Path(__file__).read_bytes())
    storyboard(name,assets,folder);assets.close()
    audio,cues=make_audio(name,folder);captions(name,cues)
    movie=MEDIA/(name+'.mp4')
    if movie.exists():raise ValueError('Refusing to replace an existing collection trailer')
    encoder=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-i',str(audio),'-map','0:v','-map','1:a','-t',str(story['duration_s']),'-c:v','libx264','-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-ar','48000','-ac','2','-movflags','+faststart',str(movie)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    traces=[];proofs=[];proof_at={round((a+.3)*FPS)for a,b,c in story['chapters']}|{round((story['duration_s']-1)*FPS)}
    if name=='03-mission':proof_at|={8*FPS,14*FPS,20*FPS,round(24.4*FPS),round(26.6*FPS)}
    try:
        for n in range(story['duration_s']*FPS):
            t=n/FPS;im,row=RENDERERS[name](t,assets);row.update(frame=n,film_s=t);traces.append(row)
            # Quiet visual cadence: short white/cobalt exposure, no flashes,
            # glitch, source displacement or transitions across a robot release.
            if n in proof_at:
                p=folder/f'proof-{n:04}.jpg';im.save(p,quality=95);proofs.append({'time_s':t,'file':ref(p)})
            encoder.stdin.write(im.tobytes())
        encoder.stdin.close();err=encoder.stderr.read();code=encoder.wait()
        if code:raise RuntimeError('Encoding failed: '+err.decode()[-2000:])
    except BaseException:
        encoder.kill();encoder.wait();raise
    finally:assets.close()
    probe=verify_video(movie,story)
    poster_time={'01-origin':13.5,'02-blueprint':14.5,'03-mission':20}[name]
    poster=MEDIA/(name+'-poster.jpg')
    V.run(['ffmpeg','-nostdin','-v','error','-ss',str(poster_time),'-i',str(movie),'-frames:v','1','-q:v','2',str(poster)])
    contact_sheet(proofs,folder/'contact-sheet.jpg')
    write(folder/'editorial-clock.json',{'frames':traces,'geometry_poses_modified':False,'source_bound_capture':ref(MASTER_JSON),'master_frame_selection':'round(capture_s*30), actual decoded MediaRecorder frames; nearest original draw-clock bounded by source receipt, not exact per-frame matrix identity','raw_source_frame_selection':'Original source-only RGB, streamed30fps, no unavailable reconstruction promoted','robot_clock':'Complete 0–26.9s task, original order; no task actions omitted'})
    robot_rows=[r for r in traces if r['kind']=='complete_authored_robot_task']
    if robot_rows:
        last=robot_rows[-1]
        if last['robot_source_s']<26.8 or abs(last['joint_q_rad'])>1e-7 or not all(g['released']and g['reached']for g in last['goals']):raise ValueError('Complete two-goal robot ending required')
    provenance={'format':'interaction-twin-trailer-collection-entry','version':1,'id':name,'title':story['title'],'story':story['story'],'art_direction':story['look'],'duration_s':story['duration_s'],'dimensions':[W,H],'fps':FPS,'movie':ref(movie),'poster':ref(poster),'captions':ref(MEDIA/(name+'.vtt')),'transcript':ref(MEDIA/(name+'-transcript.txt')),'chapters':[{'start_s':a,'end_s':b,'title':title}for a,b,title in story['chapters']],'source_captures':{'motion':ref(MASTER),'motion_receipt':ref(MASTER_JSON),'raw_video':ref(assets.source),'robot':ref(assets.robot_ref),'robot_receipt':ref(assets.robot_json),'native_orbits':ref(ORBIT),'native_orbit_receipt':ref(ORBIT_JSON),'caption_clearance':ref(CACHE/'hero-caption-clearance.json')},'audio':ref(folder/'audio.json'),'clock':ref(folder/'editorial-clock.json'),'proof_contact_sheet':ref(folder/'contact-sheet.jpg'),'builder':ref(Path(__file__)),'builder_snapshot':ref(folder/'builder-source.py'),'editorial_changes':['New typography, graphics, composition, crops and time remapping','Removal of prior editorial caption only; exact conservative geometry clearance retained','No source geometry, native textures, part transforms, joint values or task states modified'],'scope':'Research prototype. Source-bound reconstruction estimates; robot footage is an authored kinematic rehearsal, not trained control, validated grasp, contact or hardware execution. Technical graph and readable field excerpt are editorial explanations of the saved program, not another model inference.','source_rights':'Existing owner-recorded Jabra material and original project graphics/music; no new external material','new_api_calls':0,'new_gpu_calls':0,'publication':False,'ffprobe':probe,'elapsed_s':time.monotonic()-started}
    provenance['builder']=provenance['builder_snapshot']
    provenance['builder_working_path_at_render']='scripts/build_trailer_collection_a_20261104.py'
    write(MEDIA/(name+'-provenance.json'),provenance)
    write(folder/'verification.json',provenance)
    print(json.dumps({'movie':ref(movie),'duration_s':story['duration_s'],'provenance':ref(MEDIA/(name+'-provenance.json'))}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--story',choices=list(STORIES)+['all'],default='all');p.add_argument('--storyboard',action='store_true');a=p.parse_args()
    OUT.mkdir(exist_ok=True,parents=True);MEDIA.mkdir(exist_ok=True,parents=True);assets=Assets()
    names=list(STORIES)if a.story=='all'else[a.story]
    try:
        for name in names:
            if a.storyboard:
                folder=OUT/(name+'-design-v1');folder.mkdir(exist_ok=False);storyboard(name,assets,folder);assets.close()
            else:render(name,assets)
    finally:assets.close()


if __name__=='__main__':main()
