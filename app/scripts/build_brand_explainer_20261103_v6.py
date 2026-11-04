#!/usr/bin/env python3
"""A clean45-second refinement edit with a complete authored robot task.

Refined WebGL footage is recorded separately by the visible bounded capture
page. This compositor never changes geometry/poses or hides its provenance.
"""
from __future__ import annotations
import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from marketing_sources import brand,validated_brand,file_ref,checked
from marketing_render import load_brand_artwork

OUT=ROOT/'output/brand-trailer-20261103';W,H,FPS,DURATION=1920,1080,24,45
SOURCE_MASTER={'path':'output/brand-trailer-20261103/renders/v5-final/verification.json','sha256':'04f56a1fc09e806d6a431c30a4c6528a66dc467dfadf714c51aac944d2be0050','bytes':8827}
BG,INK,MUTED,BLUE,LINE='#f7f8fb','#151923','#626b7b','#3766ed','#dfe3eb'
PARTS=['case_base','case_lid','earbud_a','earbud_b']
LABELS={'case_base':'Case base','case_lid':'Hinged lid','earbud_a':'Earbud A','earbud_b':'Earbud B'}
COLORS={'case_base':'#3766ed','case_lid':'#d48d47','earbud_a':'#9179cb','earbud_b':'#409baf'}
SOURCE_CHAPTERS=[
 (0,3,'Show it once. Keep how it works.','A real filmed interaction; the newTwinframe brand.'),
 (3,8,'Start with a demonstration.','Original owner-supplied footage. One case, a lid and two earbuds.'),
 (8,14,'SAM follows the moving parts.','Actual saved SAM3.1 masks. Mask observations are3fps; display repeats samples.'),
 (14,21,'SAM 3D builds the parts.','Refined assembly with estimated joint connections. Explicit derivative provenance; physical geometry remains unverified.'),
 (21,27,'Work out how parts connect.','Muse proposes relationships; geometry fits estimated joints and motion. Review is not physical certification.'),
 (27,35,'Replay the connected interaction.','Estimated joint motion, linked to the recording. Saved constrained FK and bounded interpolation; generated mating surfaces remain unverified.'),
 (35,41,'More than a mesh.','Readable projection of actual component, joint, relationship and evidence fields.'),
 (41,46,'Find it. Inspect it. Reuse it.','Current curated library metadata; read-only discovery and local MCP. NativeGLB plus descriptiveJSON.'),
 (46,52,'Start a reusable robot task.','Actual browser capture. Robot staging and position cues are authored; no physics or trained policy.'),
 (52,55,'Capture once. Reuse the interaction.','3D parts, connections and source evidence. Generated geometry, physical scale, grasp and contact remain unverified.'),
]
CHAPTER_FRAMES=[0,48,120,216,336,432,576,672,768,1008,1080]
CHAPTERS=[(CHAPTER_FRAMES[i]/FPS,CHAPTER_FRAMES[i+1]/FPS,title,scope)for i,(_,_,title,scope)in enumerate(SOURCE_CHAPTERS)]

def source_time(t):
    """Keep each chapter's complete progress at a faster, frame-bound pace."""
    if not math.isfinite(t)or not 0<=t<=DURATION:raise ValueError('Display time must be within the45-second film')
    i=min(len(CHAPTERS)-1,max(i for i,(a,_,_,_)in enumerate(CHAPTERS)if a<=t))
    a,b,_,_=CHAPTERS[i];old_a,old_b,_,_=SOURCE_CHAPTERS[i]
    return old_a+(t-a)/(b-a)*(old_b-old_a)

def display_time(old_t):
    if not math.isfinite(old_t)or not 0<=old_t<=55:raise ValueError('Source editorial time must be within55seconds')
    i=min(len(SOURCE_CHAPTERS)-1,max(i for i,(a,_,_,_)in enumerate(SOURCE_CHAPTERS)if a<=old_t))
    a,b,_,_=CHAPTERS[i];old_a,old_b,_,_=SOURCE_CHAPTERS[i]
    return a+(old_t-old_a)/(old_b-old_a)*(b-a)

def robot_source_time(p,t):
    """Show the complete task, preserving visible closed and released-goal holds."""
    if not math.isfinite(t)or not 0<=t<=10:raise ValueError('Robot chapter time must be within10seconds')
    timing=p['_robot_timing'];first=timing['initial_hold_end_s'];last=timing['final_hold_start_s'];end=timing['duration_s']
    if not all(math.isfinite(x)for x in [first,last,end])or not 0<first<last<end:raise ValueError('Complete robot source timing is unavailable')
    if t<=.6:return first*t/.6
    if t<=8.6:return first+(last-first)*(t-.6)/8
    return last+(end-last)*(t-8.6)/1.4

def robot_editorial_crop(preparation,padding=36):
    """One fixed blank-margin crop enclosing the recorded whole-motion union."""
    camera=preparation['robot_goal_validation']['camera']['recorded_native_bbox_union']
    minimum,maximum=camera['ndc_min'],camera['ndc_max']
    if not 0<=padding<=100 or len(minimum)!=2 or len(maximum)!=2 or not all(math.isfinite(v)for v in [*minimum,*maximum]):raise ValueError('Finite complete robot camera envelope required')
    bounds=[math.floor((minimum[0]+1)*W/2)-padding,math.floor((1-maximum[1])*H/2)-padding,math.ceil((maximum[0]+1)*W/2)+padding,math.ceil((1-minimum[1])*H/2)+padding]
    return [max(0,bounds[0]),max(0,bounds[1]),min(W,bounds[2]),min(H,bounds[3])]

def verify_robot_capture(document,expected_manifest):
    if document.get('format')!='interaction-twin-complete-robot-browser-capture'or document.get('version')!=3 or document.get('captured_full_task')is not True or document.get('manifest')!=expected_manifest:raise ValueError('The exact completed robot capture is required')
    trace=document.get('trace')
    if not isinstance(trace,list)or not 500<=len(trace)<=2048:raise ValueError('Complete bounded robot state trace required')
    baseline=read(verify(expected_manifest['baseline_task_ref']));kit=read(verify(expected_manifest['kit_ref']));refinement=read(verify(expected_manifest['refinement_ref']))
    code=r'''const fs=require('node:fs');globalThis.AlignedSceneContract=require('./showcase/aligned-scene-contract.js');globalThis.RobotWorkbenchController=require('./showcase/robot-controller.js');globalThis.RobotTaskPlan=require('./showcase/robot-task-plan.js');const R=require('./showcase/robot-presentation-v2.js'),C=globalThis.RobotWorkbenchController,d=JSON.parse(fs.readFileSync(0,'utf8')),p=R.build(d.baseline,d.kit,d.refinement),r=d.document;
const eq=(a,b)=>{if(typeof a==='number'||typeof b==='number')return typeof a==='number'&&typeof b==='number'&&Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<1e-7;if(Array.isArray(a)||Array.isArray(b))return Array.isArray(a)&&Array.isArray(b)&&a.length===b.length&&a.every((v,i)=>eq(v,b[i]));if(a&&b&&typeof a==='object'&&typeof b==='object')return Object.keys(b).every(k=>Object.hasOwn(a,k)&&eq(a[k],b[k]));return a===b;};
if(!eq(r.duration_s,p.timeline.duration_s)||!eq(r.chronology,p.chronology)||!eq(r.task_proof,p.proof)||!p.proof.all_original_nonfinal_wait_actions_included||!eq(r.goal_anchors,p.goals))throw Error('Canonical complete task differs');
let previous=-1;for(const row of r.trace){if(!(row.time_s>previous&&row.time_s>=0&&row.time_s<=p.timeline.duration_s))throw Error('Robot trace time differs');previous=row.time_s;const s=C.at(p.timeline,row.time_s),angle=s.object_joints['case_base:j1_case_base_case_lid'],objects=Object.fromEntries(p.goals.map(g=>[g.id,{position:s.objects[g.id].position,parent_id:s.objects[g.id].parent_id||null}]));if(!eq(row.tool,s.tool)||!eq(row.attached,s.attached)||!eq(row.gripper,s.gripper)||!eq(row.joint_q_rad,angle)||!eq(row.objects,objects)||!eq(row.goals,R.goalState(p,s))||row.reachable!==true||s.ik.reachable!==true||!eq(row.ik_residual,s.ik.residual))throw Error('Observed robot state differs from typed replay');}
const end=C.at(p.timeline,p.timeline.duration_s),goals=R.goalState(p,end);if(r.trace[0].time_s>.1||previous<p.timeline.duration_s-.1||r.final_gripper_attached!==null||!eq(r.final_goals,goals)||!goals.every(g=>g.released&&g.reached&&g.goal_error_display_units<1e-6)||!eq(r.final_joint_q_rad,end.object_joints)||Math.abs(end.object_joints['case_base:j1_case_base_case_lid'])>1e-8)throw Error('Final released goals and closed lid were not reached');
if(!eq(r.final_tool,end.tool))throw Error('Final robot withdrawal differs');
if(!Array.isArray(r.final_goal_canvas)||r.final_goal_canvas.length!==2||r.final_goal_canvas.some(g=>!p.goals.some(p=>p.id===g.id)||!Array.isArray(g.canvas)||g.canvas[0]<0||g.canvas[0]>r.canvas[0]||g.canvas[1]<0||g.canvas[1]>r.canvas[1]))throw Error('Final goal is outside the captured canvas');
const f=r.camera_fit;if(!f||f.method!=='full_motion_FK_link_housings_gripper_and_native_object_bbox_union'||f.frequency_hz!==24||f.sampled_states!==Math.ceil(r.duration_s*24)+1||f.all_points_inside!==true||!(f.final_case_width_pixels>=140)||!(f.edge_margin_fraction>=.02&&f.edge_margin_fraction<=.2)||!Array.isArray(f.position)||!Array.isArray(f.target)||f.position.length!==3||f.target.length!==3||![...f.position,...f.target].every(Number.isFinite)||!Array.isArray(f.ndc_min)||!Array.isArray(f.ndc_max)||[...f.ndc_min,...f.ndc_max].some(v=>!Number.isFinite(v)||Math.abs(v)>1-2*f.edge_margin_fraction+1e-8))throw Error('Bounded full-motion camera proof required');
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),sub=(a,b)=>a.map((v,i)=>v-b[i]),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],unit=a=>{const n=Math.hypot(...a);if(n<1e-9)throw Error('Invalid capture view');return a.map(x=>x/n);},direction=unit(sub(f.position,f.target)),right=unit(cross([0,1,0],direction)),up=cross(direction,right),tan=Math.tan(f.vertical_fov_deg*Math.PI/360),aspect=r.canvas[0]/r.canvas[1],project=x=>{const v=sub(x,f.position),depth=-dot(v,direction);if(!(depth>0&&tan>0&&Number.isFinite(tan)))throw Error('Invalid camera projection');return[dot(v,right)/(depth*tan*aspect),dot(v,up)/(depth*tan)];};let landmarkPoints=0;
for(let frame=0;frame<=Math.ceil(r.duration_s*24);frame++){const s=C.at(p.timeline,Math.min(frame/24,r.duration_s));for(const key of['origin','elbow','wrist','tip'])for(let mask=0;mask<8;mask++){const radius=key==='tip'?.12:.095,x=s.ik[key].map((v,i)=>v+((mask&(1<<i))?radius:-radius)),ndc=project(x);if(ndc.some(v=>!Number.isFinite(v)||Math.abs(v)>1-2*f.edge_margin_fraction+1e-7))throw Error('Robot FK leaves the fitted camera envelope');landmarkPoints++;}}
for(const goal of p.goals){const ndc=project(goal.position),canvas=[(ndc[0]+1)*r.canvas[0]/2,(1-ndc[1])*r.canvas[1]/2],saved=r.final_goal_canvas.find(g=>g.id===goal.id);if(!eq(saved.canvas.slice(0,2),canvas))throw Error('Final goal camera projection differs');}
console.log(JSON.stringify({valid:true,scope:'authored_task_state_only',trace_rows:r.trace.length,complete_task:true,final_goals:goals,final_joint_q_rad:end.object_joints,proof:p.proof,camera:{recorded_native_bbox_union:f,independently_verified_FK_landmark_points:landmarkPoints}}));'''
    result=subprocess.run(['node','-e',code],cwd=ROOT,input=json.dumps({'document':document,'baseline':baseline,'kit':kit,'refinement':refinement}),text=True,capture_output=True,timeout=30)
    if result.returncode:raise ValueError('Complete robot capture verification failed: '+result.stderr[-1000:])
    return json.loads(result.stdout)

def prepare_v6():
    target=OUT/'v6-preparation.json'
    if target.exists():load_v6_preparation();print(json.dumps({'preparation_reused':True}));return
    native_input_ref=ref(OUT/'native-capture-v3/inputs.json');native_input=read(verify(native_input_ref))
    robot_input_ref=ref(OUT/'robot-capture-v2/inputs.json');robot_input=read(verify(robot_input_ref))
    capture_server_ref=ref(OUT/'capture-server-v3.json');capture_server=read(verify(capture_server_ref))
    if capture_server.get('format')!='interaction-twin-local-presentation-server'or capture_server.get('routes')!={'/robot-capture-manifest.json':robot_input_ref,'/native-capture-manifest.json':native_input_ref}:raise ValueError('Capture server manifest bindings differ')
    for item in capture_server['files']:verify(item)
    capture_folder=ROOT/'output/trailer-refinement-20261103'
    native_clip=capture_folder/'refined-capture-v1.webm';native_receipt=capture_folder/'refined-capture-v1.json'
    robot_clip=capture_folder/'robot-complete-capture-v3.webm';robot_receipt=capture_folder/'robot-complete-capture-v3.json'
    for path in [native_clip,native_receipt,robot_clip,robot_receipt]:verify(ref(path))
    from native_assembly_refinement_v1 import verify_report
    report=verify_report(native_input['refinement_report'],ROOT);scene=read(verify(native_input['scene']))
    if report['scene_ref']!=native_input['scene']or report['assets']!=native_input['assets']:raise ValueError('Native refinement manifest differs from reproducible report')
    native_document=read(native_receipt)
    if native_document.get('format')!='interaction-twin-refined-browser-capture'or native_document.get('version')!=1 or native_document.get('config',{}).get('scene')!=native_input['scene']or native_document.get('refinement_report')!=native_input['refinement_report']or native_document.get('assets')!=native_input['assets']or native_document.get('original_scene')!=native_input['original_scene']or native_document.get('independent_child_poses_used')is not False or native_document.get('canvas_dimensions')!=[W,H]or native_document.get('bytes')!=native_clip.stat().st_size:raise ValueError('Wrong refined browser capture')
    validate_annotation_trace(native_document,scene)
    robot_document=read(robot_receipt)
    if robot_document.get('canvas')!=[W,H]or robot_document.get('fps')!=FPS or robot_document.get('bytes')!=robot_clip.stat().st_size:raise ValueError('Wrong robot capture video binding')
    robot_validation=verify_robot_capture(robot_document,robot_input)
    robot_adapter=read(verify(robot_input['refinement_ref']))
    if robot_adapter['refinement_report_ref']!=native_input['refinement_report']or robot_adapter['refined_scene_ref']!=native_input['scene']:raise ValueError('Robot and native clips use different refinement assets')
    companion=ref(verify(native_input['scene']).parent/'descriptive-companion.json');description=read(verify(companion))
    if description.get('scene_ref')!=native_input['scene']or description.get('refinement_report')!=native_input['refinement_report']or description.get('joints')!=scene['mechanics']['joints']:raise ValueError('Refined descriptive companion differs')
    folder=OUT/'v6-assets';folder.mkdir()
    native_frames=decode(native_clip,folder/'native-frames',0.,17.)
    native_preparation=folder/'native-capture-preparation.json';write(native_preparation,{'clip':ref(native_clip),'receipt':ref(native_receipt),'frames':native_frames,'scene':native_input['scene'],'refinement_report':native_input['refinement_report'],'scope':'Actual refined browser capture;3original components and1documented mesh derivative, refined estimated FK.'})
    paths=[verify(row)for row in native_frames['frames']];quads=[(0,0,960,540),(960,0,1920,540),(0,540,960,1080),(960,540,1920,1080)]
    framing=folder/'editorial-framing.json';write(framing,{'format':'interaction-twin-fixed-editorial-framing','version':6,'native_capture_preparation':ref(native_preparation),'method':'One padded foreground union for the entire shot, never a moving crop or geometry gap concealment.','independent_native':{key:fixed_union_bbox(paths[:7*FPS],region,padding=18)for key,region in zip(PARTS,quads)},'connected_native':fixed_union_bbox(paths[7*FPS:],(0,0,W,H),padding=55)})
    chronology=robot_document['chronology'];timing={'initial_hold_end_s':chronology['initial_hold_s'][1],'final_hold_start_s':chronology['final_hold_s'][0],'duration_s':robot_document['duration_s']}
    frames=decode(robot_clip,folder/'robot-complete-frames',0.,timing['duration_s'])
    component_provenance={'original_native_components':sum(not a['geometry_modified']for a in report['assets']),'derived_components':sum(a['geometry_modified']for a in report['assets']),'assets':report['assets']}
    write(target,{'format':'interaction-twin-brand-explainer-v6-preparation','version':1,'source_sha256':native_input['source_sha256'],'scene':native_input['scene'],'refinement_report':native_input['refinement_report'],'descriptive_companion':companion,'native_capture_preparation':ref(native_preparation),'editorial_framing':ref(framing),'robot_capture':ref(robot_clip),'robot_capture_receipt':ref(robot_receipt),'robot_frames':frames,'robot_timing':timing,'robot_goal_validation':robot_validation,'geometry_modified':any(a['geometry_modified']for a in report['assets']),'component_provenance':component_provenance,'input_manifests':[native_input_ref,robot_input_ref],'capture_server':capture_server_ref,'new_model_or_API_calls':0})
    load_v6_preparation();print(json.dumps({'prepared':True,'native_frames':len(native_frames['frames']),'robot_frames':len(frames['frames']),'complete_robot_goal_validation':robot_validation['valid']}),flush=True)

def load_v6_preparation():
    path=OUT/'v6-preparation.json'
    if not path.is_file():raise ValueError('Waiting for verified refined geometry and complete browser task captures; no fallback film is rendered')
    p=read(path)
    if p.get('format')!='interaction-twin-brand-explainer-v6-preparation'or p.get('version')!=1:raise ValueError('Invalid V6 preparation')
    for key in ['scene','refinement_report','descriptive_companion','native_capture_preparation','editorial_framing','robot_capture','robot_capture_receipt']:verify(p[key])
    capture_server=read(verify(p['capture_server']))
    for item in capture_server['files']:verify(item)
    if p.get('source_sha256')!='7474ebd9d06c55b63fc073f632e71e5125353ace2538fa7bcf632cfdc1c3d0f2'or not isinstance(p.get('geometry_modified'),bool)or p.get('robot_goal_validation',{}).get('valid')is not True:raise ValueError('Refinement/source/complete robot outcome is not verified')
    report=read(verify(p['refinement_report']))
    if p['component_provenance']['assets']!=report['assets']or p['geometry_modified']!=any(a['geometry_modified']for a in report['assets']):raise ValueError('Refined component inventory differs')
    for item in report['assets']:verify(item['asset']);verify(item['original_asset'])
    for row in p['robot_frames']['frames']:verify(row)
    if p['robot_frames'].get('start_s')!=0 or p['robot_frames'].get('fps')!=24 or abs(p['robot_frames'].get('duration_s',0)-p['robot_timing']['duration_s'])>1/24 or len(p['robot_frames']['frames'])<math.floor(p['robot_timing']['duration_s']*24)-2:raise ValueError('Whole robot capture frames are required')
    scene=read(verify(p['scene']));description=read(verify(p['descriptive_companion']))
    if description.get('format')!='interaction-twin-refined-mechanism-description'or description.get('scene_ref')!=p['scene']or description.get('refinement_report')!=p['refinement_report']or description.get('original_program')!=scene['original_program']or description.get('joints')!=scene['mechanics']['joints']:raise ValueError('Refined descriptive companion is not bound to the displayed joint')
    robot_source_time({'_robot_timing':p['robot_timing']},10)
    return p

def read(path):return json.loads(Path(path).read_text())
def ref(path,**extra):return file_ref(path,ROOT)|extra
def verify(row):return checked({k:row[k]for k in ['path','sha256','bytes']},ROOT)
def write(path,value):
    with path.open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def run(args):subprocess.run(args,check=True,capture_output=True,timeout=180)
@lru_cache(maxsize=80)
def font(size,bold=False,mono=False):
    name='/System/Library/Fonts/Menlo.ttc'if mono else'/System/Library/Fonts/Supplemental/Arial'+(' Bold'if bold else'')+'.ttf'
    return ImageFont.truetype(name,size)
def txt(im,xy,value,size=30,color=INK,bold=False,mono=False,spacing=10):ImageDraw.Draw(im).multiline_text(xy,value,font=font(size,bold,mono),fill=color,spacing=spacing)
def box(im,rect,fill='white',radius=22,outline=LINE):ImageDraw.Draw(im).rounded_rectangle(tuple(map(round,rect)),radius,fill=fill,outline=outline,width=2)
def line(im,pts,color=LINE,width=3):ImageDraw.Draw(im).line(pts,fill=color,width=width,joint='curve')
def ease(p):p=max(0,min(1,p));return p*p*(3-2*p)
@lru_cache(maxsize=12)
def image(relative):return Image.open(ROOT/relative).convert('RGB')
def contain(im,picture,rect,fill=BG):
    x,y,w,h=map(round,rect);layer=Image.new('RGB',(w,h),fill);pic=ImageOps.contain(picture.convert('RGB'),(w,h),Image.Resampling.LANCZOS);layer.paste(pic,((w-pic.width)//2,(h-pic.height)//2));im.paste(layer,(x,y))
def mark(im,prep,x,y,size,alpha=1.):
    art=load_brand_artwork(prep['brand'],ROOT).resize((size,size),Image.Resampling.LANCZOS)
    if alpha<1:art.putalpha(art.getchannel('A').point(lambda a:round(a*alpha)))
    im.paste(art,(round(x),round(y)),art)
def base(prep,kicker,title,sub=None):
    im=Image.new('RGB',(W,H),BG);mark(im,prep,72,40,46);txt(im,(132,48),'Interaction Twin',27,INK,True)
    txt(im,(72,133),kicker.upper(),21,BLUE,True);txt(im,(68,174),title,66,INK,True,spacing=6)
    if sub:txt(im,(72,263),sub,29,MUTED)
    return im
def chip(im,x,y,value,*,color=BLUE,fill='#eef2ff',size=22):
    width=round(font(size,True).getlength(value))+28;box(im,(x,y,x+width,y+42),fill,12,None);txt(im,(x+14,y+8),value,size,color,True)

def decode(path,folder,start,duration):
    folder.mkdir();run(['ffmpeg','-nostdin','-v','error','-ss',str(start),'-i',str(path),'-t',str(duration),'-vf','fps=24','-q:v','2',str(folder/'%04d.jpg')])
    rows=[ref(p)for p in sorted(folder.glob('*.jpg'))]
    if len(rows)<max(1,int(duration*FPS)-2):raise ValueError('Decoded source sequence is shorter than declared')
    return {'start_s':start,'duration_s':duration,'fps':24,'frames':rows,'selection':'FFmpeg nearest-frame24fps display; no generated frames'}



def clip_frame(clip,elapsed):
    n=max(0,min(len(clip['frames'])-1,round(elapsed*clip['fps'])));return image(clip['frames'][n]['path'])
def source_frame(p,clock):
    name='intro'if clock<12 else'closing'if clock>=31 else'interaction';clip=p['clips'][name];return clip_frame(clip,clock-clip['start_s'])
def native_frame(p,t):return clip_frame(p['_native']['frames'],t)
def fixed_union_bbox(paths,region,*,padding=36):
    """Editorial crop of blank render pixels; one fixed box for a whole shot.

    The threshold detects the dark native parts against this capture's fixed
    light background. A union over every decoded frame prevents tracking zoom
    and retains the complete visible silhouette throughout the shot.
    """
    left,top,right,bottom=region;bounds=[]
    if padding<0 or right<=left or bottom<=top:raise ValueError('Invalid editorial crop region')
    for path in paths:
        a=np.asarray(Image.open(path).convert('RGB'))
        if a.shape[:2]!=(H,W):raise ValueError('Expected native1080p capture frame')
        ys,xs=np.where(a[top:bottom,left:right].min(axis=2)<200)
        if len(xs):bounds.append((int(xs.min()+left),int(ys.min()+top),int(xs.max()+left+1),int(ys.max()+top+1)))
    if not bounds:raise ValueError('No native geometry visible in crop region')
    return [max(left,min(b[0]for b in bounds)-padding),max(top,min(b[1]for b in bounds)-padding),min(right,max(b[2]for b in bounds)+padding),min(bottom,max(b[3]for b in bounds)+padding)]
def validate_annotation_trace(document,scene):
    """Verify saved annotation math using the browser's real FK contract.

    No browser, inference or network. The only child process is a fixed Node
    verifier reading the exact declared scene and bounded JSON from stdin.
    """
    trace=document.get('projection_trace')
    if not isinstance(trace,list)or not 200<=len(trace)<=1024:raise ValueError('Bounded complete joint annotation trace required')
    if trace[0].get('capture_time_s',100)>7.15 or trace[-1].get('capture_time_s',0)<16.8:raise ValueError('Joint annotation trace does not span the connected shot')
    code=r'''const fs=require('node:fs'),C=require('./showcase/aligned-scene-contract.js'),d=JSON.parse(fs.readFileSync(0,'utf8')),s=C.validateScene(d.scene,d.document.config.source_sha256),j=s.mechanics.joints.find(j=>j.id===d.document.joint_annotation.joint_id),cam=d.document.capture_camera;
const mul=(m,v)=>m.map(r=>r.reduce((a,x,i)=>a+x*v[i],0)),near=(a,b)=>Array.isArray(a)&&a.length===b.length&&a.every((v,i)=>Number.isFinite(v)&&Math.abs(v-b[i])<1e-5),flip=[[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]],rigidInverse=m=>{const r=[0,1,2].map(i=>[m[0][i],m[1][i],m[2][i],0]);for(let i=0;i<3;i++)r[i][3]=-r[i].slice(0,3).reduce((a,x,k)=>a+x*m[k][3],0);return[...r,[0,0,0,1]];},view=rigidInverse(cam.matrix_world),project=p=>{const v=mul(cam.projection_matrix,mul(view,[...p,1]));return[(v[0]/v[3]+1)*960,(1-v[1]/v[3])*540,v[2]/v[3]];};let previous=-1;
for(const r of d.document.projection_trace){if(!(r.capture_time_s>previous&&r.capture_time_s>=7&&r.capture_time_s<=17.1&&r.joint_id===j.id&&r.annotation_only===true))throw Error('Invalid typed annotation timeline');previous=r.capture_time_s;const q=r.capture_time_s-7,t=q<=6?15+q*5/6:31+(q-6)/4;if(Math.abs(t-r.source_time_s)>1e-6)throw Error('Source clock changed');const state=C.sceneAt(s,t,true),parent=state.parts.find(p=>p.id===j.parent_id),angle=state.joints.find(p=>p.id===j.id);if(!parent.camera_from_native_mesh||angle.q_rad===null||Math.abs(angle.q_rad-r.q_rad)>1e-5)throw Error('Unsupported source joint');const world=C.multiply(C.multiply(flip,parent.camera_from_native_mesh),j.parent_from_joint),pivot=mul(world,[0,0,0,1]).slice(0,3),direction=mul(world,[...j.axis,0]).slice(0,3),len=Math.hypot(...direction),axis=direction.map(x=>x/len),L=d.document.joint_annotation.axis_length_world;if(!Number.isFinite(L)||L<=0)throw Error('Invalid diagnostic axis length');const a=pivot.map((v,i)=>v-axis[i]*L/2),b=pivot.map((v,i)=>v+axis[i]*L/2);if(!near(r.pivot_world,pivot)||!near(r.axis_world,axis)||!near(r.pivot_canvas,project(pivot))||!Array.isArray(r.axis_canvas)||r.axis_canvas.length!==2||!near(r.axis_canvas[0],project(a))||!near(r.axis_canvas[1],project(b)))throw Error('Joint pivot/axis projection differs');}console.log(JSON.stringify({verified:true,rows:d.document.projection_trace.length}));'''
    result=subprocess.run(['node','-e',code],cwd=ROOT,input=json.dumps({'document':document,'scene':scene}),text=True,capture_output=True,timeout=30)
    if result.returncode:raise ValueError('Joint annotation evidence failed: '+result.stderr[-1000:])
    return json.loads(result.stdout)
def connected_picture(p,t):return native_frame(p,t).crop(p['_framing']['connected_native'])
def descriptive_lines(p):
    obj=p['_object_program'];joint=p['_refined_scene']['mechanics']['joints'][0]
    sample=next(row for row in joint['samples']if row['source_sample_index']==21)
    if 'source_000021'not in obj['joints'][0]['evidence_ids']:raise ValueError('Original source evidence citation is absent')
    origin=[joint['parent_from_joint'][i][3]for i in range(3)]
    return ['{',f'  "parent_id": "{joint["parent_id"]}",',f'  "child_id": "{joint["child_id"]}",',
      f'  "type": "{joint["type"]}",',f'  "status": "{joint["status"]}",',
      '  "axis": ['+', '.join(f'{n:.2f}'for n in joint['axis'])+'],',
      '  "joint_origin_parent": ['+', '.join(f'{n:.2f}'for n in origin)+'],',
      f'  "q_rad": {sample["q_rad"]:.2f},',f'  "timestamp_s": {sample["timestamp_s"]:.2f},',
      '  "evidence_ids": ["source_000021"],','  "physical_accuracy_verified": false','}']
def node(im,x,y,title,active=True):
    box(im,(x,y,x+410,y+114),'#eef2ff'if active else'white',18,BLUE if active else LINE);txt(im,(x+24,y+23),title,30,INK,True)
def film_frame(p,t):
    display_t=t
    t=source_time(t)
    if t<3:
        im=Image.new('RGB',(W,H),BG);progress=ease(t/1.8);size=round(150*(.94+.06*progress));mark(im,p,96,205,size,min(1,t/.45))
        txt(im,(92,420),'Interaction Twin',88,INK,True);txt(im,(96,548),'Keep the shape.\nKeep how it works.',51,MUTED,spacing=12)
        contain(im,source_frame(p,1+t),(1050,80,770,910))
    elif t<8:
        im=base(p,'01 / Capture','Start with a demonstration.')
        txt(im,(84,415),'One case.\nA lid.\nTwo removable earbuds.',48,INK,True,spacing=22)
        txt(im,(86,680),'The recording is the evidence.',29,MUTED);contain(im,source_frame(p,1+t-3),(950,215,850,750))
    elif t<14:
        im=base(p,'02 / Segment','SAM follows the moving parts.','SAM 3.1 separates the base, lid and both earbud tracklets.')
        clock=12+(t-8)*.8;row=min(p['masks'],key=lambda r:abs(r['timestamp_s']-clock));contain(im,image(row['mask_overlay']['path']),(80,340,900,620))
        for i,key in enumerate(PARTS):
            y=380+i*130;box(im,(1080,y,1120,y+40),COLORS[key],11,None);txt(im,(1152,y-3),LABELS[key],38,INK,True)
    elif t<21:
        im=base(p,'03 / Reconstruct','SAM 3D builds the parts.','Refined 3D assembly · estimated joint connections.')
        captured=native_frame(p,(display_t-9)/5*7)
        for i,key in enumerate(PARTS):
            x=90+(i%2)*900;y=330+(i//2)*330
            contain(im,captured.crop(p['_framing']['independent_native'][key]),(x,y,840,265))
            chip(im,x+270,y+275,LABELS[key],size=25)
    elif t<27:
        q=t-21;im=base(p,'04 / Reason + review','Work out how parts connect.','Muse proposes relationships; geometry fits estimated joints and motion.')
        center=(940,575);nodes={'case_base':(720,515),'case_lid':(130,365),'earbud_a':(1320,365),'earbud_b':(1320,675)}
        for i,key in enumerate(['case_lid','earbud_a','earbud_b']):
            if q>i*.65:
                x,y=nodes[key];line(im,[(925,570),(x+205,y+57)],BLUE,5)
        for key,(x,y)in nodes.items():node(im,x,y,LABELS[key],q>.3)
        chip(im,300,540,'revolute · estimated',size=23);chip(im,1260,575,'removable',size=23)
        for i,label in enumerate(['Propose relationships','Check against evidence','Owner reviews the result']):
            if q>1.5+i*.6:txt(im,(110+i*600,882),str(i+1).zfill(2)+'  '+label,26,BLUE,True)
    elif t<35:
        q=(display_t-18)/6*10;clock=15+q*5/6 if q<=6 else 31+(q-6)/4
        im=base(p,'05 / Connect','Replay the connected interaction.','Estimated joint motion, linked to the recording.')
        contain(im,source_frame(p,clock),(55,335,570,610));contain(im,connected_picture(p,7+q),(690,315,1190,625))
        chip(im,110,945,'Captured video',size=22);chip(im,1050,945,'Connected scene · estimated',size=22)
    elif t<41:
        im=base(p,'06 / Describe','More than a mesh.','Keep parts, connections, actions and source evidence together.')
        contain(im,connected_picture(p,10),(65,330,820,560));box(im,(920,338,1848,915),'#151923',22,None)
        txt(im,(950,358),'REFINED JOINT · SOURCE EVIDENCE 6.67s',21,'#a6bdff',True)
        for i,value in enumerate(descriptive_lines(p)):txt(im,(950,405+i*42),value,26,'#c1d1ff'if i in [0,11]else'white',mono=True)
        chip(im,92,946,'3D parts · GLB',size=24);chip(im,590,946,'Object program · JSON',size=24);chip(im,1280,946,'Source evidence',size=24)
    elif t<46:
        im=base(p,'07 / Reuse','Find it. Inspect it. Reuse it.','Use it in robotics authoring. Let agents inspect it.')
        cards=[('Earbuds + case','Open · remove · return'),('Wallet + coin','Containment · remove'),('Camera + batteries','Open · exchange')]
        for i,(title,sub)in enumerate(cards):
            y=350+i*168;box(im,(76,y,820,y+140));txt(im,(106,y+24),title,34,INK,True);txt(im,(108,y+82),sub,25,MUTED)
        box(im,(900,350,1848,904),'#151923',22,None);txt(im,(945,395),'READ-ONLY AGENT TOOLS',22,'#a6bdff',True)
        for i,(name,note)in enumerate([('search_twins','Find a useful object'),('get_kinematics','Read its connections'),('get_evidence','Inspect the recording')]):txt(im,(945,470+i*124),name,35,'white',mono=True);txt(im,(945,523+i*124),note,24,'#b9c3d7')
    elif t<52:
        im=base(p,'08 / Build on it','Start a reusable\nrobot task.')
        txt(im,(74,392),'An authored robot rehearsal,\nbuilt with these parts.',29,MUTED,spacing=13)
        picture=clip_frame(p['clips']['robot_complete'],robot_source_time(p,display_t-32))
        contain(im,picture.crop(p['_robot_crop']),(830,82,1040,960))
    else:
        im=Image.new('RGB',(W,H),BG);mark(im,p,100,145,132);txt(im,(96,362),'Capture once.\nReuse the interaction.',86,INK,True,spacing=12)
        txt(im,(102,624),'3D parts. Connections. Evidence.',38,MUTED);txt(im,(104,758),'Explore the object library.',34,BLUE,True)
    start=max(a for a,_,_,_ in SOURCE_CHAPTERS if a<=t);fade=min(1.,(t-start)/.22)if start else min(1.,t/.25)
    return Image.blend(Image.new('RGB',(W,H),BG),im,fade)if fade<1 else im

def encode(name='v6'):
    if not re.fullmatch('[a-z0-9-]{1,40}',name):raise ValueError('Bounded fresh render name required')
    master=read(verify(SOURCE_MASTER))
    for key in ['movie','storyboard','preparation','native_capture_preparation','editorial_framing','builder']:verify(master[key])
    preparation=load_v6_preparation()
    native_preparation=verify(preparation['native_capture_preparation']);framing_path=verify(preparation['editorial_framing'])
    p=read(OUT/'preparation.json');p['_native']=read(native_preparation)
    p['_framing']=read(framing_path);p['_object_program']=read(verify(preparation['descriptive_companion']))
    p['_refined_scene']=read(verify(preparation['scene']));p['_robot_timing']=preparation['robot_timing'];p['clips']['robot_complete']=preparation['robot_frames']
    p['_robot_crop']=robot_editorial_crop(preparation)
    for row in p['inputs']+[p['_native']['clip'],p['_native']['receipt']]:verify(row)
    validated_brand(p['brand'],ROOT)
    for clip in [*p['clips'].values(),p['_native']['frames']]:
        for row in clip['frames']:verify(row)
    target=OUT/'renders'/name;target.mkdir(parents=True,exist_ok=False);shutil.copyfile(__file__,target/'builder-at-encode.py')
    write(target/'robot-editorial-framing.json',{'format':'interaction-twin-fixed-robot-editorial-framing','version':1,'capture':preparation['robot_capture'],'receipt':preparation['robot_capture_receipt'],'rectangle_pixels':p['_robot_crop'],'padding_pixels':36,'method':'Fixed crop of blank margins, enclosing all padded arm FK landmarks and native object boxes throughout the complete task. No geometry, pose, gap or goal is hidden.','camera_proof':preparation['robot_goal_validation']['camera'],'layout':'Two-column typography plus complete robot canvas; no moving crop.'})
    write(target/'storyboard.json',{'format':'interaction-twin-brand-explainer-storyboard','version':6,'duration_s':DURATION,'fps':FPS,
      'chapters':[dict(start_s=a,end_s=b,title=c,scope=d)for a,b,c,d in CHAPTERS],
      'chapter_frame_boundaries':CHAPTER_FRAMES,'source_master':SOURCE_MASTER,'source_editorial_duration_s':55,
      'source_time_jump':{'source_ranges_s':[[15,20],[31,32]],'display_frame':519,'display_time_s':519/FPS},
      'complete_robot_task':{'display_start_s':32,'display_end_s':42,'initial_hold_display_s':.6,'motion_display_s':8,'final_hold_display_s':1.4,'capture':preparation['robot_capture'],'timing':preparation['robot_timing'],'goal_validation':preparation['robot_goal_validation'],'editorial_framing':ref(target/'robot-editorial-framing.json')},
      'retime':'Frame-bound earlier chapters; complete10-second recorded FK excerpt shown in6seconds. Full robot source uses0.6s closed hold,8s whole motion,1.4s final goal hold. Saved frames sampled without synthesizing motion.',
      'producer_asset_kind':'refined_sam3d_assembly','component_provenance':preparation['component_provenance'],
      'removed_drawn_elements':['Intro blue demonstration tagline','All10chapters’ footer rules and tiny annotation notes','Final creator credit'],
      'why':'Keep the shape. Keep how it works. A reusable object program retains parts, connections and source evidence.',
      'formats_shown':['GLB','JSON','source evidence'],'claims':{'physical_accuracy_verified':False,'robot_motion':'authored position-only rehearsal','MCP':'local read-only tools, not hosted service'}})
    picture=target/'picture.mp4';movie=target/'interaction-twin-explainer.mp4';started=time.monotonic()
    proc=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','24','-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','19','-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(picture)],stdin=subprocess.PIPE)
    proofs=[];proof_times=[round(display_time(t)*FPS)/FPS for t in [1.5,5.,10.,17.,24.,30.,38.,43.,49.,53.]]
    proof_times[8]=41.25;task_proofs=[];task_proof_times=[32.25,33.,35.,38.,41.5]
    try:
        for n in range(DURATION*FPS):
            t=n/FPS;im=film_frame(p,t)
            if any(abs(t-x)<.5/FPS for x in proof_times):
                path=target/f'proof-{n:04}.jpg';im.save(path,quality=96);proofs.append(ref(path,film_time_s=t))
            if any(abs(t-x)<.5/FPS for x in task_proof_times):
                path=target/f'task-proof-{n:04}.jpg';im.save(path,quality=96);task_proofs.append(ref(path,film_time_s=t,robot_source_time_s=robot_source_time(p,t-32)))
            proc.stdin.write(np.asarray(im,np.uint8).tobytes())
            if n%(FPS*5)==0:print(json.dumps({'film_s':int(t),'wall_s':round(time.monotonic()-started,1)}),flush=True)
        proc.stdin.close();code=proc.wait(timeout=90)
        if code:raise ValueError('FFmpeg failed')
    except BaseException:proc.kill();proc.wait();raise
    run(['ffmpeg','-nostdin','-v','error','-i',str(picture),'-i',str(verify(p['music'])),'-map','0:v:0','-map','1:a:0','-t','45','-c:v','copy','-af','loudnorm=I=-22:TP=-2:LRA=7,afade=t=in:st=0:d=0.6,afade=t=out:st=43:d=2','-ar','48000','-c:a','aac','-b:a','192k','-movflags','+faststart',str(movie)])
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(movie)]))
    video=next(s for s in probe['streams']if s['codec_type']=='video')
    if(video['width'],video['height'])!=(1920,1080)or abs(float(probe['format']['duration'])-DURATION)>.1 or int(video.get('nb_frames',0))!=DURATION*FPS or not any(s['codec_type']=='audio'for s in probe['streams']):raise ValueError('Final media contract failed')
    sheet=Image.new('RGB',(1600,5*480),BG)
    for i,row in enumerate(proofs):
        x=(i%2)*800;y=(i//2)*480;sheet.paste(image(row['path']).resize((800,450),Image.Resampling.LANCZOS),(x,y));txt(sheet,(x+12,y+454),f'{row["film_time_s"]:.1f}s · {CHAPTERS[i][2]}',17,MUTED)
    sheet.save(target/'proof-sheet.jpg',quality=96);film_frame(p,display_time(30)).save(target/'poster.jpg',quality=96)
    vtt=['WEBVTT',''];srt=[]
    def stamp(t,sep):
        ms=round(t*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02}{sep}{ms%1000:03}'
    for i,(a,b,title,scope)in enumerate(CHAPTERS,1):vtt.extend([stamp(a,'.')+' --> '+stamp(b,'.'),title,scope,'']);srt.extend([str(i),stamp(a,',')+' --> '+stamp(b,','),title,''])
    (target/'captions.vtt').write_text('\n'.join(vtt));(target/'captions.srt').write_text('\n'.join(srt))
    write(target/'verification.json',{'format':'interaction-twin-brand-explainer-verification','version':6,'source_master':SOURCE_MASTER,'movie':ref(movie),'poster':ref(target/'poster.jpg'),'proof_sheet':ref(target/'proof-sheet.jpg'),'proof_frames':proofs,'task_proof_frames':task_proofs,'storyboard':ref(target/'storyboard.json'),'captions':ref(target/'captions.vtt'),'srt':ref(target/'captions.srt'),'ffprobe':probe,'preparation':ref(OUT/'v6-preparation.json'),'native_capture_preparation':ref(native_preparation),'editorial_framing':ref(framing_path),'refined_scene':preparation['scene'],'refinement_report':preparation['refinement_report'],'descriptive_companion':preparation['descriptive_companion'],'robot_capture':preparation['robot_capture'],'builder':ref(target/'builder-at-encode.py'),'elapsed_wall_s':time.monotonic()-started,'new_API_or_model_calls':0,'geometry_modified':preparation['geometry_modified'],'joint_frames_refined':True,'physics_performed':False,'published':False,'landing_changed':False,'music_rights':'Original procedural project instrumental; no external samples or tracks. Original pitch, first45s with clean fade.'})
    (target/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>45-second refined explainer · local review</title><style>body{margin:40px auto;max-width:1200px;background:#f7f8fb;color:#151923;font:18px Arial}video,img{width:100%;border-radius:18px}p{line-height:1.6}a{color:#3766ed}</style><h1>Keep the shape. Keep how it works.</h1><p>Refined45-second edit: actual footage, saved SAM masks, a documented SAM3D assembly derivative, source-bound estimated connections and a complete authored robot rehearsal. No new inference or publication.</p><video controls poster="poster.jpg" src="interaction-twin-explainer.mp4"><track kind="subtitles" src="captions.vtt" srclang="en" label="English"></video><p><a href="interaction-twin-explainer.mp4" download>Download MP4</a> · <a href="verification.json">Exact provenance</a></p><img src="proof-sheet.jpg" alt="Ten proof frames"><p>Geometry, physical scale, grasp and contact remain unverified. Robot task is authored.</p>')
    print(json.dumps({'complete':True,'movie':str(movie),'proof':str(target/'proof-sheet.jpg'),'duration_s':DURATION}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','encode']);parser.add_argument('--name',default='v6');args=parser.parse_args()
    if args.command=='prepare':prepare_v6()
    else:encode(args.name)
if __name__=='__main__':main()
