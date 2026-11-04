#!/usr/bin/env python3
"""Package one portable, offline video / orbitable-3D alignment observer."""
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import trimesh

ROOT = Path(__file__).resolve().parents[1]


def uri(path):
    mime = {'.jpg': 'image/jpeg', '.png': 'image/png', '.mp4': 'video/mp4'}[path.suffix]
    return 'data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode()


def overlays(out, records):
    """Use saved full 3D renders; hand masks only change overlay alpha."""
    result = []
    for row in records:
        name = f'{row["sample_index"]:06d}'
        colors = np.array(Image.open(out / 'mesh_rgb' / (name + '.png')).convert('RGB'))
        full = np.array(Image.open(out / 'full_masks' / (name + '.png'))) > 0
        visible = np.array(Image.open(out / 'visible_masks' / (name + '.png'))) > 0
        # Foreground alpha and the raw projection are independently inspectable.
        rgba = np.zeros((*full.shape, 4), np.uint8)
        rgba[..., :3] = np.clip(.35 * colors + .65 * np.array([30, 217, 203]), 0, 255).astype(np.uint8)
        rgba[..., 3] = visible.astype(np.uint8) * 155
        stream = io.BytesIO()
        Image.fromarray(rgba).save(stream, format='PNG', optimize=True)
        result.append({'fill': 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode(),
                       'outline': uri(out / 'review_frames' / (name + '-outline.png'))})
    return result


DOCUMENT = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Battery / alignment observer</title>
<style>
:root{color-scheme:dark;--bg:#10151d;--panel:#171f2a;--muted:#a0afbf;--line:#2b3948;--teal:#6de0ce}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#edf4fa;font:15px/1.5 system-ui,-apple-system,sans-serif}main{max-width:1600px;margin:auto;padding:30px}header{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-bottom:20px}h1{font-size:28px;letter-spacing:-.7px;margin:0;font-weight:650}header p{color:var(--muted);margin:4px 0 0}.stage{font-size:12px;color:var(--teal);border:1px solid #33574f;border-radius:25px;padding:6px 12px;white-space:nowrap}.views{display:grid;grid-template-columns:1fr 1fr;gap:18px}.panel{border:1px solid var(--line);border-radius:14px;overflow:hidden;background:var(--panel)}.panel-head{display:flex;justify-content:space-between;align-items:center;padding:14px 18px;gap:8px}.panel-head h2{font-size:16px;margin:0}.panel-head span{color:var(--muted);font-size:12px}.viewport{position:relative;background:#111a25;aspect-ratio:16/10;overflow:hidden}.video-box{position:absolute;top:50%;left:0;width:100%;aspect-ratio:16/9;transform:translateY(-50%)}video,#overlay{position:absolute;inset:0;width:100%;height:100%;display:block}#overlay{pointer-events:none}#scene{width:100%;height:100%;display:block;touch-action:none;cursor:grab}#scene:active{cursor:grabbing}.scene-hint{position:absolute;left:15px;bottom:12px;color:#abc0ce;font-size:12px;pointer-events:none;background:#101924b8;border-radius:6px;padding:5px 9px}.panel-tools{padding:12px 16px;min-height:73px;border-top:1px solid var(--line);display:flex;align-items:center;gap:10px;flex-wrap:wrap}.panel-tools label{display:inline-flex;align-items:center;gap:6px;font-size:13px;white-space:nowrap}input{accent-color:var(--teal)}button,select{font:inherit;color:inherit;background:#22303f;border:1px solid #405365;border-radius:7px;padding:8px 12px;cursor:pointer}button:hover{background:#2a3d50}button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--teal);outline-offset:3px}button.active{background:#21483f;border-color:#4a9785;color:#b4ffef}.timeline{margin-top:18px;padding:18px 20px;border:1px solid var(--line);border-radius:12px;background:var(--panel)}.row{display:flex;align-items:center;gap:14px;flex-wrap:wrap}#play{background:var(--teal);color:#102c25;font-weight:650;border-color:var(--teal);min-width:98px}.spacer{flex:1}#time{font-variant-numeric:tabular-nums;color:#d3e4ee;font-size:14px}#scrubber{width:100%;display:block;margin:18px 0 10px;cursor:ew-resize}.ticks{display:flex;justify-content:space-between;color:var(--muted);font-size:12px}.status-row{margin-top:16px;display:flex;align-items:flex-start;gap:10px;min-height:46px}.badge{background:#183f37;color:#9ef2d6;border:1px solid #2d6a59;border-radius:6px;padding:4px 9px;font-size:12px;white-space:nowrap}.badge.unsupported{background:#462428;color:#ffbfc1;border-color:#9c4e54}.badge.reference_initialization{background:#463a1c;color:#f9de9f;border-color:#8f7542}#note{color:var(--muted);font-size:13px;padding-top:3px}.footer{display:grid;grid-template-columns:1fr 1fr;gap:25px;color:var(--muted);font-size:13px;margin-top:18px}.footer p{margin:0}.footer strong{color:#d2e5ec;font-weight:550}a{color:var(--teal)}#error{display:none;padding:18px;background:#51252b;color:#ffd9db;position:absolute;inset:15px;border-radius:8px;z-index:2}small{font-size:12px}.speed{display:inline-flex;align-items:center;gap:8px;color:var(--muted);font-size:13px}#frame-count{color:var(--muted);font-size:13px}
@media(max-width:900px){.views{grid-template-columns:1fr}.viewport{aspect-ratio:16/9}.footer{grid-template-columns:1fr}main{padding:16px}header{align-items:flex-start}.stage{display:none}h1{font-size:25px}.panel-tools{min-height:0}}
</style></head><body><main>
<header><div><h1>Watch the video. Explore its 3D motion.</h1><p>One filmed battery, one fixed-size mesh, synchronized in time.</p></div><span class="stage">STAGE 4 · BATTERY PILOT</span></header>
<div class="views">
<section class="panel"><div class="panel-head"><h2>Video + alignment</h2><span>Original footage · 3D overlay</span></div><div class="viewport"><div class="video-box"><video id="video" preload="auto" muted playsinline loop poster="POSTER_URI" src="VIDEO_URI"></video><canvas id="overlay" width="960" height="540" aria-label="Projected 3D battery overlay"></canvas></div></div><div class="panel-tools"><label><input id="overlay-toggle" type="checkbox" checked>3D overlay</label><label><input id="outline-toggle" type="checkbox" checked>Full outline</label><small style="color:var(--muted)">Thumb remains in front of the mesh.</small></div></section>
<section class="panel"><div class="panel-head"><h2>Interactive 3D scene</h2><span id="mode-label">Orbit view · fixed scene coordinates</span></div><div class="viewport"><canvas id="scene" aria-label="Drag to orbit the moving 3D battery. Scroll to zoom. Shift drag to pan."></canvas><div class="scene-hint">Drag to orbit · scroll to zoom · Shift + drag to pan</div><div id="error" role="alert"></div></div><div class="panel-tools"><button id="orbit-view" class="active">Orbit view</button><button id="camera-view">Match video</button><button id="reset-view" aria-label="Reset the scene view">Reset</button><label><input id="path-toggle" type="checkbox" checked>Motion path</label><label><input id="plane-toggle" type="checkbox">Video plane</label></div></section>
</div>
<section class="timeline"><div class="row"><button id="play">▶ Play</button><label class="speed">Speed <select id="speed" aria-label="Playback speed"><option value="0.25">0.25×</option><option value="0.5" selected>0.5×</option><option value="1">1×</option></select></label><span id="frame-count"></span><span class="spacer"></span><span id="time"></span></div><input id="scrubber" type="range" min="0" max="51" value="0" step="1" aria-label="Video and 3D motion sample"><div class="ticks"><span>Source 39.006 s</span><span>Drag to inspect a moment</span><span>42.409 s</span></div><div class="status-row"><span id="status" class="badge"></span><span id="note"></span></div></section>
<div class="footer"><p><strong>The scene stays fixed while the battery moves.</strong> The small axis belongs to the battery. “Match video” restores the recording-camera projection; orbit to see it from another angle. The optional video plane is a flat visual reference, not reconstructed background geometry.</p><p><strong>Current model: a fitted cylindrical approximation.</strong> Hidden surfaces, physical size and rotation around its axis remain uncertain. Tracking is sampled at about 15 fps. Red samples show unsupported candidates. This HTML contains its video, geometry and overlays and works offline.</p></div>
</main>
<script id="observer-data" type="application/json">DATA_JSON</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('observer-data').textContent);
const $=id=>document.getElementById(id), video=$('video'), overlay=$('overlay'), ctx=overlay.getContext('2d'), canvas=$('scene');
const state={index:0, mode:'orbit', yaw:.588, pitch:-.395, distance:.235, target:[-.0151,-.0045,.3341], drag:null, lastTime:0, seekPending:false};
const baseTarget=state.target.slice(), imageCache=new Map();
const vec={add:(a,b)=>a.map((x,i)=>x+b[i]),sub:(a,b)=>a.map((x,i)=>x-b[i]),scale:(a,s)=>a.map(x=>x*s),dot:(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),cross:(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],norm:a=>{const n=Math.hypot(...a);return a.map(x=>x/n)}};
function multiply(a,b){const c=new Float32Array(16);for(let col=0;col<4;col++)for(let row=0;row<4;row++)for(let k=0;k<4;k++)c[col*4+row]+=a[k*4+row]*b[col*4+k];return c}
function lookAt(eye,target,up){const z=vec.norm(vec.sub(eye,target)),x=vec.norm(vec.cross(up,z)),y=vec.cross(z,x);return new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-vec.dot(x,eye),-vec.dot(y,eye),-vec.dot(z,eye),1])}
function perspective(fov,aspect,near=.002,far=2){const f=1/Math.tan(fov/2),nf=1/(near-far);return new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)*nf,-1,0,0,2*far*near*nf,0])}
function cameraProjection(){const k=data.K,w=960,h=540,n=.002,f=2;return new Float32Array([2*k[0][0]/w,0,0,0,0,2*k[1][1]/h,0,0,1-2*k[0][2]/w,2*k[1][2]/h-1,-(f+n)/(f-n),-1,0,0,-2*f*n/(f-n),0])}
function orbitCamera(){const cp=Math.cos(state.pitch);return vec.add(state.target,[state.distance*Math.sin(state.yaw)*cp,state.distance*Math.sin(state.pitch),-state.distance*Math.cos(state.yaw)*cp])}
function image(uri){if(!imageCache.has(uri)){const img=new Image();img.onload=drawOverlay;img.src=uri;imageCache.set(uri,img)}return imageCache.get(uri)}
function drawOverlay(){ctx.clearRect(0,0,960,540);const frame=data.frames[state.index],fill=image(frame.overlay.fill),outline=image(frame.overlay.outline);if($('overlay-toggle').checked&&fill.complete&&fill.naturalWidth)ctx.drawImage(fill,0,0);if($('outline-toggle').checked&&outline.complete&&outline.naturalWidth)ctx.drawImage(outline,0,0);if(frame.status==='unsupported'){ctx.strokeStyle='#f3777d';ctx.lineWidth=5;ctx.strokeRect(2.5,2.5,955,535)}}

class SceneRenderer{
  constructor(canvas){
    this.gl=canvas.getContext('webgl',{antialias:true,alpha:false});
    if(!this.gl)throw new Error('WebGL is unavailable. Enable hardware acceleration or open this file in a browser with WebGL support.');
    const gl=this.gl;
    this.colorProgram=this.program(`attribute vec3 aPosition;attribute vec3 aColor;uniform mat4 uMVP;varying vec3 vColor;void main(){vColor=aColor;gl_Position=uMVP*vec4(aPosition,1.0);}`,`precision mediump float;varying vec3 vColor;uniform float uOpacity;uniform vec3 uTint;void main(){gl_FragColor=vec4(vColor*uTint,uOpacity);}`);
    this.textureProgram=this.program(`attribute vec3 aPosition;attribute vec2 aUV;uniform mat4 uMVP;varying vec2 vUV;void main(){vUV=aUV;gl_Position=uMVP*vec4(aPosition,1.0);}`,`precision mediump float;varying vec2 vUV;uniform sampler2D uTexture;uniform float uOpacity;void main(){vec4 c=texture2D(uTexture,vUV);gl_FragColor=vec4(c.rgb,uOpacity);}`);
    this.positions=gl.createBuffer();this.colors=gl.createBuffer();this.uvs=gl.createBuffer();this.texture=gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D,this.texture);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,1,1,0,gl.RGB,gl.UNSIGNED_BYTE,new Uint8Array([30,40,53]));
    gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
    this.meshPositions=[];this.meshColors=[];this.world=[];
    this.poster=new Image();this.poster.src=data.poster;this.poster.onload=()=>this.updateTexture(this.poster);
  }
  program(vs,fs){const gl=this.gl;function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s}const p=gl.createProgram();gl.attachShader(p,shader(gl.VERTEX_SHADER,vs));gl.attachShader(p,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));return p}
  attribute(program,name,buffer,values,size){const gl=this.gl,location=gl.getAttribLocation(program,name);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(values),gl.DYNAMIC_DRAW);gl.enableVertexAttribArray(location);gl.vertexAttribPointer(location,size,gl.FLOAT,false,0,0)}
  colored(positions,colors,mvp,mode,opacity=1,tint=[1,1,1]){const gl=this.gl,p=this.colorProgram;gl.useProgram(p);this.attribute(p,'aPosition',this.positions,positions,3);this.attribute(p,'aColor',this.colors,colors,3);gl.uniformMatrix4fv(gl.getUniformLocation(p,'uMVP'),false,mvp);gl.uniform1f(gl.getUniformLocation(p,'uOpacity'),opacity);gl.uniform3fv(gl.getUniformLocation(p,'uTint'),tint);gl.drawArrays(mode,0,positions.length/3)}
  updateMesh(){const r=data.frames[state.index],R=r.rotation,t=r.translation,s=data.scale;this.world=data.mesh.vertices.map(p=>R.map((row,k)=>s*vec.dot(row,p)+t[k]));const positions=[],colors=[],light=vec.norm([-.3,-.6,-1]);for(const face of data.mesh.faces){const a=this.world[face[0]],b=this.world[face[1]],c=this.world[face[2]],n=vec.norm(vec.cross(vec.sub(b,a),vec.sub(c,a))),shade=.5+.5*Math.abs(vec.dot(n,light));for(const j of face){positions.push(...this.world[j]);colors.push(...data.mesh.colors[j].map(x=>x/255*shade))}}this.meshPositions=positions;this.meshColors=colors}
  updateTexture(source){if(!source)return;const gl=this.gl;gl.bindTexture(gl.TEXTURE_2D,this.texture);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,gl.RGB,gl.UNSIGNED_BYTE,source)}
  plane(mvp,opacity){const gl=this.gl,p=this.textureProgram,k=data.K,z=.425;const corner=(u,v)=>[(u-k[0][2])*z/k[0][0],(v-k[1][2])*z/k[1][1],z];const a=corner(0,0),b=corner(960,0),c=corner(960,540),d=corner(0,540);gl.useProgram(p);this.attribute(p,'aPosition',this.positions,[...a,...b,...c,...a,...c,...d],3);this.attribute(p,'aUV',this.uvs,[0,0,1,0,1,1,0,0,1,1,0,1],2);gl.uniformMatrix4fv(gl.getUniformLocation(p,'uMVP'),false,mvp);gl.uniform1f(gl.getUniformLocation(p,'uOpacity'),opacity);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,this.texture);gl.uniform1i(gl.getUniformLocation(p,'uTexture'),0);gl.depthMask(false);gl.drawArrays(gl.TRIANGLES,0,6);gl.depthMask(true)}
  draw(){const gl=this.gl,dpr=Math.min(window.devicePixelRatio||1,2),w=Math.round((canvas.clientWidth||640)*dpr),h=Math.round((canvas.clientHeight||400)*dpr);if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h}gl.viewport(0,0,w,h);gl.clearColor(.062,.094,.137,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);let mvp;
    if(state.mode==='camera'){const rw=Math.min(w,h*960/540),rh=rw*540/960;gl.viewport((w-rw)/2,(h-rh)/2,rw,rh);mvp=multiply(cameraProjection(),lookAt([0,0,0],[0,0,1],[0,-1,0]))}
    else mvp=multiply(perspective(42*Math.PI/180,w/h),lookAt(orbitCamera(),state.target,[0,-1,0]));
    this.lastMVP=mvp;
    const linePositions=[],lineColors=[];const segment=(a,b,color)=>{linePositions.push(...a,...b);lineColors.push(...color,...color)};
    // Grid and paths remain in one recording-camera coordinate system.
    if(state.mode==='orbit')for(let i=0;i<=16;i++){const x=-.16+i*.02,z=.15+i*.02;segment([x,.067,.15],[x,.067,.47],[.16,.23,.29]);segment([-.16,.067,z],[.16,.067,z],[.16,.23,.29])}
    if($('path-toggle').checked){for(let i=1;i<data.frames.length;i++){const a=data.frames[i-1],b=data.frames[i];if(a.status==='unsupported'||b.status==='unsupported')continue;segment(a.translation,b.translation,[.25,.76,.66])}}
    const r=data.frames[state.index],t=r.translation;for(let axis=0;axis<3;axis++){const direction=r.rotation.map(row=>row[axis]*.016);segment(t,vec.add(t,direction),[[.95,.35,.38],[.38,.85,.54],[.38,.62,1]][axis])}
    if($('plane-toggle').checked&&state.mode==='orbit'){const k=data.K,z=.425;const corners=[[0,0],[960,0],[960,540],[0,540]].map(([u,v])=>[(u-k[0][2])*z/k[0][0],(v-k[1][2])*z/k[1][1],z]);for(let i=0;i<4;i++){segment(corners[i],corners[(i+1)%4],[.37,.43,.62]);segment([0,0,0],corners[i],[.20,.26,.37])}}
    this.colored(linePositions,lineColors,mvp,gl.LINES,.85);
    const tint=r.status==='unsupported'?[1.0,.62,.55]:[1,1,1];this.colored(this.meshPositions,this.meshColors,mvp,gl.TRIANGLES,1,tint);
    if($('plane-toggle').checked)this.plane(mvp,state.mode==='camera'?.92:.27);
  }
}
let renderer=null;
try{renderer=new SceneRenderer(canvas);renderer.updateMesh()}catch(error){$('error').style.display='block';$('error').textContent=error.message}
function updateStatus(){const r=data.frames[state.index];$('scrubber').value=state.index;$('frame-count').textContent=`Sample ${state.index+1} / ${data.frames.length}`;$('time').textContent=`Source ${r.timestamp_s.toFixed(3)} s`;$('status').className='badge '+r.status;$('status').textContent=r.status==='unsupported'?'Unsupported candidate':r.status==='reference_initialization'?'Reference frame':'Tracking supported';$('note').textContent=r.status==='unsupported'?r.reasons.join('; '):r.status==='reference_initialization'?'The mesh is initialized here. Rotate the scene to inspect the shape.':`Mesh size stays fixed · held-out tracking error ${r.heldout_reprojection_px.median.toFixed(2)} px · estimated camera and scale`}
function draw(){drawOverlay();if(renderer)renderer.draw()}
function setSample(index,seek=true){state.index=Math.max(0,Math.min(data.frames.length-1,index));if(seek){const relative=Math.max(0,data.frames[state.index].timestamp_s-data.start);state.seekPending=Math.abs(video.currentTime-relative)>1e-5;if(state.seekPending)video.currentTime=relative}updateStatus();if(renderer)renderer.updateMesh();draw()}
function nearestSample(time){let best=0,distance=Infinity;for(let i=0;i<data.frames.length;i++){const d=Math.abs(data.frames[i].timestamp_s-(data.start+time));if(d<distance){distance=d;best=i}}return best}
function resetView(){state.target=baseTarget.slice();state.distance=.235;state.yaw=.588;state.pitch=-.395;state.mode='orbit';viewButtons();draw()}
function viewButtons(){$('orbit-view').className=state.mode==='orbit'?'active':'';$('camera-view').className=state.mode==='camera'?'active':'';$('mode-label').textContent=state.mode==='orbit'?'Orbit view · fixed scene coordinates':'Recording-camera projection'}
function playLabel(){$('play').textContent=video.paused?'▶ Play':'Ⅱ Pause'}
$('play').addEventListener('click',()=>{if(video.paused){video.play().catch(error=>{$('note').textContent='Playback could not start: '+error.message})}else video.pause()});video.addEventListener('play',playLabel);video.addEventListener('pause',playLabel);
$('speed').addEventListener('change',()=>{video.playbackRate=Number($('speed').value)});video.playbackRate=.5;
$('scrubber').addEventListener('input',e=>{video.pause();setSample(Number(e.target.value))});video.addEventListener('seeked',()=>{state.seekPending=false;if(renderer&&video.readyState>=2)renderer.updateTexture(video);draw()});video.addEventListener('loadeddata',()=>{if(renderer)renderer.updateTexture(video);draw()});video.addEventListener('error',()=>{$('note').textContent='The embedded video could not be decoded in this browser.'});
for(const id of['overlay-toggle','outline-toggle','plane-toggle','path-toggle'])$(id).addEventListener('change',draw);
$('camera-view').addEventListener('click',()=>{state.mode='camera';viewButtons();draw()});$('orbit-view').addEventListener('click',()=>{state.mode='orbit';viewButtons();draw()});$('reset-view').addEventListener('click',resetView);
canvas.addEventListener('pointerdown',e=>{if(state.mode!=='orbit')return;state.drag={x:e.clientX,y:e.clientY,pan:e.shiftKey||e.button===2};canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener('pointermove',e=>{if(!state.drag)return;const dx=e.clientX-state.drag.x,dy=e.clientY-state.drag.y;if(state.drag.pan){const eye=orbitCamera(),forward=vec.norm(vec.sub(state.target,eye)),right=vec.norm(vec.cross(forward,[0,-1,0])),up=vec.cross(right,forward),factor=state.distance*.0017;state.target=vec.add(state.target,vec.add(vec.scale(right,-dx*factor),vec.scale(up,dy*factor)))}else{state.yaw-=dx*.007;state.pitch=Math.max(-1.4,Math.min(1.4,state.pitch+dy*.007))}state.drag.x=e.clientX;state.drag.y=e.clientY;draw()});
for(const event of['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>state.drag=null);canvas.addEventListener('contextmenu',e=>e.preventDefault());canvas.addEventListener('wheel',e=>{if(state.mode!=='orbit')return;e.preventDefault();state.distance=Math.max(.055,Math.min(1.2,state.distance*Math.exp(e.deltaY*.001)));draw()},{passive:false});canvas.addEventListener('dblclick',resetView);
window.addEventListener('resize',draw);document.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON'].includes(e.target.tagName))return;if(e.code==='Space'){e.preventDefault();$('play').click()}else if(e.key==='ArrowRight'){video.pause();setSample(state.index+1)}else if(e.key==='ArrowLeft'){video.pause();setSample(state.index-1)}});
function animate(){if(!state.seekPending){const index=nearestSample(video.currentTime||0);if(index!==state.index)setSample(index,false)}if(renderer&&video.readyState>=2&&!video.paused&&$('plane-toggle').checked)renderer.updateTexture(video);if(renderer)renderer.draw();requestAnimationFrame(animate)}
setSample(0,false);requestAnimationFrame(animate);
// Exposed only for local diagnostic checks; no network dependencies.
window.__observer={data,state,setSample,draw,orbitCamera,resetView,renderer,multiply,lookAt,cameraProjection};
</script></body></html>'''


def main():
    out = ROOT / 'data/camera-battery-withdrawal-15fps/alignment/battery-pilot-v2'
    clip = out / 'observer-source.mp4'
    if not clip.exists():
        temporary = out / 'observer-source.pending.mp4'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i',
                        str(ROOT / 'data/camera-battery-withdrawal-15fps/source/video.mp4'),
                        '-vf', 'trim=start_frame=1169:end_frame=1273,setpts=PTS-STARTPTS,scale=960:540',
                        '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(temporary)], check=True)
        temporary.rename(clip)
    report = json.loads((out / 'alignment.json').read_text())
    mesh = trimesh.load(ROOT / 'data/camera-battery-withdrawal-15fps/geometry/battery-fitted-v1/battery.ply', force='mesh')
    rendered = overlays(out, report['frames'])
    frames = [{**row, 'overlay': overlay} for row, overlay in zip(report['frames'], rendered)]
    poster = ROOT / 'data/camera-battery-withdrawal-15fps/geometry/local-battery-v3/inputs/000001.jpg'
    data = {'mesh': {'vertices': mesh.vertices.tolist(), 'faces': mesh.faces.tolist(), 'colors': np.array(mesh.visual.vertex_colors)[:, :3].tolist()},
            'frames': frames, 'K': report['camera']['intrinsics'], 'scale': report['asset_scale'],
            'start': 1169 * 1001 / 30000, 'poster': uri(poster)}
    document = DOCUMENT.replace('DATA_JSON', json.dumps(data, separators=(',', ':'), allow_nan=False).replace('</', '<\\/'))
    document = document.replace('POSTER_URI', data['poster']).replace('VIDEO_URI', uri(out / 'observer-source.mp4'))
    (out / 'observer.html').write_text(document)
    manifest = {'scope': 'simplified Stage 4 observation view; no pose or geometry changes',
                'alignment_run': 'battery-pilot-v2', 'source_start_s': data['start'],
                'video_source': 'source frames 1169..1272 inclusive, trim and scale only; silent 104-frame native-cadence clip',
                'video_sha256': hashlib.sha256((out / 'observer-source.mp4').read_bytes()).hexdigest(),
                'pose_sha256': hashlib.sha256((out / 'poses.npz').read_bytes()).hexdigest(),
                'fixed_orbit_target': [-.0151, -.0045, .3341], 'interpolation': 'nearest source-timestamped sample; no synthetic poses',
                'external_network_dependencies': False, 'media_and_geometry_embedded': True,
                'html_bytes': (out / 'observer.html').stat().st_size,
                'video_plane': 'optional flat reference at nominal camera Z .425; not reconstructed background'}
    (out / 'observer-metadata.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'html': str(out / 'observer.html'), **manifest}, indent=2))


if __name__ == '__main__':
    main()
