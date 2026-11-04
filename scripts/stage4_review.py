#!/usr/bin/env python3
"""Offline evidence review for the fixed-scale battery alignment pilot.

This report consumes real fit outputs.  It never clips a projection to the target
SAM mask.  Its only compositing exclusion is the saved foreground-thumb mask.
"""
import argparse
import html
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text())


def font(size):
    try:
        return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', size)
    except OSError:
        return ImageFont.load_default(size=size)


def metric(record, name, sub=None):
    value = record.get(name)
    if sub:
        value = value.get(sub) if isinstance(value, dict) else None
    return '—' if value is None else f'{value:.2f}'


def render_review_frames(out, report, pack):
    """Save the unclipped boundary, default composite, and labelled movie frames."""
    dest = out / 'review_frames'
    dest.mkdir(exist_ok=True)
    movie = out / 'movie_frames'
    movie.mkdir(exist_ok=True)
    tiles = []
    highlights = [1, 16, 23, 32, 48, 52]
    for row, record in zip(pack['samples'], report['frames']):
        index = row['sample_index']
        name = f'{index:06d}'
        source = np.asarray(Image.open(out / '../../geometry/local-battery-v3' / row['image']).convert('RGB'))
        raw_path = out / 'full_masks' / (name + '.png')
        full = np.asarray(Image.open(raw_path)) > 0 if raw_path.exists() else np.zeros(source.shape[:2], bool)
        visible_path = out / 'visible_masks' / (name + '.png')
        visible = np.asarray(Image.open(visible_path)) > 0 if visible_path.exists() else full
        mesh_path = out / 'mesh_rgb' / (name + '.png')
        mesh = np.asarray(Image.open(mesh_path).convert('RGB')) if mesh_path.exists() else np.zeros_like(source)
        # Full-mesh outline is preserved even where the foreground thumb hides it.
        boundary = cv2.morphologyEx(full.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
        yy, xx = np.indices(full.shape)
        dashed = boundary & ((xx + yy) % 14 < 8)
        raw_outline = np.zeros((*full.shape, 4), np.uint8)
        raw_outline[dashed] = [255, 217, 76, 255]
        Image.fromarray(raw_outline).save(dest / (name + '-outline.png'))
        composite = source.copy()
        tint = .35 * mesh.astype(float) + .65 * np.array([30., 217., 203.])
        composite[visible] = np.clip(.47 * source[visible] + .53 * tint[visible], 0, 255).astype(np.uint8)
        composite[dashed] = [255, 217, 76]
        Image.fromarray(composite).save(dest / (name + '-overlay.jpg'), quality=94)
        status = record['status']
        color = '#b6333c' if status == 'unsupported' else ('#9a6f0c' if status == 'reference_initialization' else '#1b7f73')
        frame = Image.new('RGB', (1920, 676), '#10222e')
        frame.paste(Image.fromarray(source), (0, 73))
        frame.paste(Image.fromarray(composite), (960, 73))
        draw = ImageDraw.Draw(frame)
        draw.text((20, 10), 'Stage 4 / fixed-size battery alignment candidate', font=font(27), fill='white')
        draw.text((20, 43), 'Original source', font=font(19), fill='#adbfca')
        draw.text((980, 43), 'Mesh + foreground thumb; yellow = full raw projection', font=font(19), fill='#adbfca')
        draw.rectangle((0, 613, 1919, 647), fill=color)
        detail = f"Sample {index:02d}/52 | source {row['timestamp_s']:.6f}s | {status.upper()} | held-out median {metric(record, 'heldout_reprojection_px', 'median')} px"
        draw.text((20, 618), detail, font=font(21), fill='white')
        draw.text((20, 651), 'Internal consistency candidate; estimated camera/scale, uncertain axial roll. Not verified 6D pose.', font=font(18), fill='#d2dce2')
        frame.save(movie / (name + '.jpg'), quality=93)
        if index in highlights:
            tile = Image.new('RGB', (640, 421), '#10222e')
            tile.paste(Image.fromarray(source).resize((320, 180)), (0, 50))
            tile.paste(Image.fromarray(composite).resize((320, 180)), (320, 50))
            d = ImageDraw.Draw(tile)
            d.text((14, 10), f"Sample {index:02d} / {row['timestamp_s']:.3f}s", font=font(24), fill='white')
            d.rectangle((0, 230, 639, 268), fill=color)
            d.text((14, 236), status.replace('_', ' ').upper(), font=font(20), fill='white')
            d.text((14, 284), f"Held-out median / p90: {metric(record, 'heldout_reprojection_px', 'median')} / {metric(record, 'heldout_reprojection_px', 'p90')} px", font=font(20), fill='white')
            d.text((14, 317), f"Visible silhouette IoU: {metric(record, 'occlusion_composite_iou_with_visible_sam')}", font=font(20), fill='#c8d7df')
            reasons = record.get('reasons', [])
            note = '; '.join(reasons) if reasons else ('Reference ray assignment; not an accuracy test.' if index == 16 else 'Passes internal pilot gates; no independent pose labels.')
            import textwrap
            for k, line in enumerate(textwrap.wrap(note, 60)[:3]):
                d.text((14, 351 + 21 * k), line, font=font(17), fill='#c8d7df')
            tiles.append(tile)
    sheet = Image.new('RGB', (1920, 904), '#10222e')
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 9), 'Battery alignment pilot / source + mesh overlay / unsupported candidates remain visible', font=font(28), fill='white')
    draw.text((20, 44), 'Yellow outlines are full, unclipped projections. Foreground thumb masks use an assistant-annotated corridor; no target-mask clipping.', font=font(19), fill='#b3c7d3')
    for j, tile in enumerate(tiles):
        sheet.paste(tile, ((j % 3) * 640, 62 + (j // 3) * 421))
    sheet.save(out / 'summary.jpg', quality=95)
    times = [r['timestamp_s'] for r in report['frames']]
    fps = 1 / float(np.median(np.diff(times)))
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-framerate', str(fps), '-i', str(movie / '%06d.jpg'), '-c:v', 'libx264', '-crf', '19', '-pix_fmt', 'yuv420p', '-r', '30', '-movflags', '+faststart', str(out / 'preview.mp4')], check=True)


def plot_evidence(out, report):
    os.environ['MPLCONFIGDIR'] = str(ROOT / '.cache/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = report['frames']
    times = np.array([r['timestamp_s'] for r in records])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for field, label, color in [('median', 'Held-out median', '#1b867a'), ('p90', 'Held-out p90', '#5794be')]:
        values = [np.nan if r['sample_index'] == 16 else r.get('heldout_reprojection_px', {}).get(field, np.nan) for r in records]
        axes[0].plot(times, values, label=label, color=color, lw=2)
    axes[0].axhline(3, ls='--', color='#1b867a', alpha=.6, label='Median gate: 3 px')
    axes[0].axhline(8, ls='--', color='#5794be', alpha=.6, label='P90 gate: 8 px')
    axes[0].set(ylabel='Reprojection error (pixels)', title='Held-out point IDs from the same CoTracker run — internal consistency, not pose ground truth')
    axes[1].plot(times, [r.get('full_projection_iou_with_visible_sam', np.nan) for r in records], color='#b18e32', label='Full projection vs visible SAM')
    axes[1].plot(times, [r.get('occlusion_composite_iou_with_visible_sam', np.nan) for r in records], color='#1b867a', label='Foreground-composited projection vs visible SAM')
    axes[1].axhline(.55, ls='--', color='#627581', alpha=.6, label='Composite IoU gate: .55')
    axes[1].set(xlabel='Source video time (s)', ylabel='Silhouette IoU', ylim=(0, 1.02), title='Visible-mask agreement depends on segmentation and foreground ordering')
    for ax in axes:
        ax.axvline(times[15], ls=':', color='#718492')
        for r in records:
            if r['status'] == 'unsupported':
                ax.axvspan(r['timestamp_s'] - .025, r['timestamp_s'] + .025, color='#d45b62', alpha=.18)
        ax.grid(alpha=.18)
        ax.legend(fontsize=8, loc='upper left')
    fig.savefig(out / 'fit-evidence.png', dpi=160)
    plt.close(fig)


DOCUMENT = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Battery alignment / Stage 4 pilot</title>
<style>*{box-sizing:border-box}body{margin:0;background:#ecf1f4;color:#173244;font:15px system-ui;line-height:1.5}main{max-width:1600px;margin:auto;padding:25px}h1{font-size:30px;margin:0 0 6px}h2{font-size:20px;margin:8px 0}p{margin:9px 0}.notice{padding:16px;background:#dae8ee;border-left:4px solid #3b7d91}.warning{background:#f8e9e8;border-left-color:#c0454d}.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.card{background:white;border:1px solid #cedce4;border-radius:8px;padding:14px;min-width:0}.viewport{display:block;width:100%;height:auto;aspect-ratio:16/9;background:#10222e;object-fit:contain;border-radius:4px}canvas.viewport{height:auto;touch-action:none}#scene{cursor:grab}button,select{font:inherit;padding:7px 12px;border:1px solid #9fb5c1;border-radius:5px;background:white;color:#173244}button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #388cab}input[type=range]{width:100%;accent-color:#27867a}label{display:inline-flex;gap:6px;align-items:center;margin:6px 12px 3px 0}.toolbar{display:flex;align-items:center;gap:12px;margin:15px 0 6px;flex-wrap:wrap}.badge{font-size:13px;padding:5px 9px;border-radius:4px;color:white;background:#287e73}.unsupported{background:#b6333c}.reference_initialization{background:#9a6f0c}.meta{font-size:13px;color:#526e7d}.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:14px 0}.stats>div{background:white;border:1px solid #cfdee5;border-radius:6px;padding:12px}.number{font-size:25px;font-weight:650}.links a{display:inline-block;margin:5px 14px 5px 0}a{color:#176a9b}img.evidence,video{display:block;width:100%;border-radius:6px;margin:15px 0}.details{display:grid;grid-template-columns:1fr 1fr;gap:16px}li{margin:8px 0}#frame-note{min-height:45px}.legend{padding:4px 0;color:#6d5b0f}code{font-size:13px}@media(max-width:1050px){.panels{grid-template-columns:1fr 1fr}.panels>.card:last-child{grid-column:1/-1}.stats{grid-template-columns:repeat(3,1fr)}}@media(max-width:650px){main{padding:14px}.panels,.details{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}}
</style><main><h1>Stage 4 / One battery, one fixed-size mesh</h1><p>Camera-relative tracking over 52 recorded samples. Inspect the evidence and the failure intervals.</p>
<p class="notice"><strong>Limited alignment pilot; Stage 3 remains incomplete.</strong> A cylinder completed from video proportions is fitted to one reference, then tracked with disjoint fitting/checking point IDs. The mesh never changes scale. Camera intrinsics and physical scale are estimated; axial roll, hidden geometry and actual 6D-pose accuracy are unverified.</p>
<div class="stats"><div><div class="number">SUPPORTED/51</div><div class="meta">supported nonreference candidates</div></div><div><div class="number">MEDIAN px</div><div class="meta">median of held-out medians*</div></div><div><div class="number">IOU</div><div class="meta">median composite silhouette IoU*</div></div><div><div class="number">SCALE</div><div class="meta">constant model units / asset unit</div></div><div><div class="number">1 reference</div><div class="meta">initialization excluded from error aggregate</div></div></div>
<div class="toolbar"><button id="previous" aria-label="Previous sample">←</button><button id="play">Play</button><button id="next" aria-label="Next sample">→</button><strong id="time"></strong><span id="status" class="badge"></span></div>
<input id="frame" aria-label="Recorded sample" type="range" min="0" max="51" value="15" step="1"><p id="frame-note"></p>
<div class="panels"><section class="card"><h2>1 / Original recording</h2><img id="source" class="viewport" alt="Original recorded frame"><p class="meta">Exact source timestamps are preserved. Two physical batteries appear; this pilot follows the selected local track candidate only.</p></section>
<section class="card"><h2>2 / Model-to-video projection</h2><canvas id="overlay" class="viewport" aria-label="Mesh over source frame"></canvas><label><input id="mesh-visible" type="checkbox" checked>Mesh overlay</label><label><input id="foreground" type="checkbox" checked>Foreground thumb</label><label><input id="raw-outline" type="checkbox" checked>Full raw projection outline</label><p class="legend">Yellow dashes = entire projected mesh, including hidden regions.</p><p class="meta">The foreground mask is a conservative subset of SAM hand masks inside an assistant-annotated thumb corridor. It supplies an ordering prior, not measured depth. No projection is clipped to the target SAM mask.</p></section>
<section class="card"><h2>3 / Rigid mesh inspection</h2><canvas id="scene" class="viewport" aria-label="Orbitable camera-relative mesh"></canvas><select id="view-mode" aria-label="Mesh viewing mode"><option value="orbit">Orbit around component</option><option value="camera">Recording-camera projection</option></select><button id="reset">Reset view</button><p class="meta" id="view-note">Drag to orbit; wheel to zoom. Orbit mode recentres the component for inspection, retaining each fitted camera-relative orientation. The full camera mode preserves its fitted location in the recording frame.</p></section></div>
<div class="stats"><div><div class="number" id="train">—</div><div class="meta">training inliers / visible</div></div><div><div class="number" id="heldout">—</div><div class="meta">held-out median / p90 pixels</div></div><div><div class="number" id="silhouette">—</div><div class="meta">full / composite IoU</div></div><div><div class="number" id="occluded">—</div><div class="meta">full projection hidden by thumb</div></div><div><div class="number" id="depth">—</div><div class="meta">source-depth support</div></div></div>
<p class="notice warning"><strong>Unsupported samples: UNSUPPORTED.</strong> Their candidate poses remain visible for diagnosis; the viewer never substitutes a previous successful frame. Sample 52 has rejected source depth and motion blur. The reference projection is conditional on ray assignment and is not an accuracy test.</p>
<h2>Fit evidence</h2><img class="evidence" src="fit-evidence.png" alt="Held-out reprojection errors and raw versus foreground-composited silhouette IoU"><p class="meta">*Aggregate values exclude reference sample 16 and rejected-depth sample 52, while retaining the other unsupported candidates. Held-out point IDs come from the same CoTracker run; they are disjoint from training IDs but do not provide independently labelled pose accuracy. Red bands identify unsupported samples.</p>
<div class="details"><section class="card"><h2>What the pilot establishes</h2><ul><li>One unchanged closed asset and one fixed scale can explain much of this short withdrawal sequence.</li><li>Canonical points are assigned by exact mesh-ray intersections at sample 16. The pose solver uses training point IDs; the held-out IDs are used only for the reported check.</li><li>Full projections, foreground composites, rejected candidates and raw metrics are retained for review.</li></ul></section><section class="card"><h2>What remains uncertain</h2><ul>LIMITATIONS</ul></section></div>
<h2>Preview</h2><video controls loop preload="metadata" poster="summary.jpg" src="preview.mp4"></video><p class="meta">Recorded source and candidate overlay, played at source cadence. The red status strip marks unsupported candidates.</p>
<p class="links"><a href="alignment.json">All poses, metrics and gates (JSON)</a><a href="poses.npz">Numeric poses + canonical points (NPZ)</a><a href="../../geometry/battery-fitted-v1/battery.glb">Battery asset (GLB)</a><a href="../../geometry/battery-fitted-v1/metadata.json">Geometry provenance</a><a href="../../occlusion/hands-v1/metadata.json">Hand-mask method</a><a href="../../occlusion/hands-v1/contact-02.jpg">Hand-mask review</a><a href="../../geometry/stage3-local-progress.html">Stage 3 evidence</a></p>
</main><script>
const data=EMBEDDED_DATA;
const el=id=>document.getElementById(id),overlay=el('overlay'),oc=overlay.getContext('2d'),scene=el('scene'),sc=scene.getContext('2d');
let index=15,timer=null,yaw=0,pitch=0,zoom=1,drag=null,loadToken=0;
const cache=new Map();
function image(path){if(cache.has(path))return cache.get(path);const promise=new Promise(resolve=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>resolve(null);img.src=path});cache.set(path,promise);return promise}
function fmt(value,digits=2){return value==null?'—':Number(value).toFixed(digits)}
function drawOverlay(images){overlay.width=960;overlay.height=540;oc.clearRect(0,0,960,540);if(!images.source){oc.fillText('Source image unavailable',20,30);return}oc.drawImage(images.source,0,0);if(el('mesh-visible').checked&&images.mesh&&images.full){const off=document.createElement('canvas');off.width=960;off.height=540;const ctx=off.getContext('2d');ctx.drawImage(images.mesh,0,0);const pixels=ctx.getImageData(0,0,960,540),maskCanvas=document.createElement('canvas');maskCanvas.width=960;maskCanvas.height=540;const mc=maskCanvas.getContext('2d');mc.drawImage(el('foreground').checked&&images.visible?images.visible:images.full,0,0);const mask=mc.getImageData(0,0,960,540).data;for(let j=0;j<pixels.data.length;j+=4){pixels.data[j]=.35*pixels.data[j]+.65*30;pixels.data[j+1]=.35*pixels.data[j+1]+.65*217;pixels.data[j+2]=.35*pixels.data[j+2]+.65*203;pixels.data[j+3]=mask[j]>0?135:0}ctx.putImageData(pixels,0,0);oc.drawImage(off,0,0)}if(el('raw-outline').checked&&images.outline)oc.drawImage(images.outline,0,0);if(data.frames[index].status==='unsupported'){oc.strokeStyle='#df4b57';oc.lineWidth=7;oc.strokeRect(3,3,954,534)}}
let currentImages=null;
async function update(){const token=++loadToken,r=data.frames[index],name=String(r.sample_index).padStart(6,'0');el('frame').value=index;el('time').textContent=`Sample ${r.sample_index}/52 · source ${r.timestamp_s.toFixed(6)} s`;el('status').className='badge '+r.status;el('status').textContent=r.status.replaceAll('_',' ').toUpperCase();el('source').src=r.source;el('train').textContent=`${r.train_inliers??'—'} / ${r.train_visible??'—'}`;el('heldout').textContent=`${fmt(r.heldout_reprojection_px?.median)} / ${fmt(r.heldout_reprojection_px?.p90)}`;el('silhouette').textContent=`${fmt(r.full_projection_iou_with_visible_sam)} / ${fmt(r.occlusion_composite_iou_with_visible_sam)}`;el('occluded').textContent=r.foreground_occluded_fraction==null?'—':fmt(r.foreground_occluded_fraction*100,1)+'%';el('depth').textContent=r.depth_supported?'Available, uncalibrated':'Rejected / unsupported';const reasons=r.reasons??[r.reason].filter(Boolean);el('frame-note').textContent=reasons.length?'Unsupported: '+reasons.join('; '):r.status==='reference_initialization'?'Reference initialization: canonical points are assigned here; reference reprojection is not an accuracy test.':'Passes the internal pilot gates. Segmentation, tracking, symmetry and camera uncertainty still apply.';drawScene();oc.clearRect(0,0,overlay.width,overlay.height);const paths={source:r.source,mesh:`mesh_rgb/${name}.png`,full:`full_masks/${name}.png`,visible:`visible_masks/${name}.png`,outline:`review_frames/${name}-outline.png`};const entries=await Promise.all(Object.entries(paths).map(async([key,path])=>[key,await image(path)]));if(token!==loadToken)return;currentImages=Object.fromEntries(entries);drawOverlay(currentImages)}
function rotate(p){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),x=cy*p[0]+sy*p[2],z=-sy*p[0]+cy*p[2];return[x,cp*p[1]-sp*z,sp*p[1]+cp*z]}
function drawScene(){const w=scene.clientWidth||480,h=scene.clientHeight||270,dpr=window.devicePixelRatio||1;scene.width=w*dpr;scene.height=h*dpr;sc.scale(dpr,dpr);sc.fillStyle='#10222e';sc.fillRect(0,0,w,h);const r=data.frames[index];if(!r.rotation){sc.fillStyle='white';sc.fillText('No pose candidate',20,30);return}const R=r.rotation,t=r.translation,s=data.scale;const points=data.mesh.vertices.map(p=>R.map((row,k)=>s*(row[0]*p[0]+row[1]*p[1]+row[2]*p[2])+t[k]));const mode=el('view-mode').value;let q,projected;if(mode==='camera'){q=points;const K=data.K,scale=Math.min(w/960,h/540),ox=(w-960*scale)/2,oy=(h-540*scale)/2;projected=q.map(p=>[ox+scale*(K[0][0]*p[0]/p[2]+K[0][2]),oy+scale*(K[1][1]*p[1]/p[2]+K[1][2])]);sc.strokeStyle='#456171';sc.strokeRect(ox,oy,960*scale,540*scale)}else{const c=[0,0,0];for(const p of points)for(let k=0;k<3;k++)c[k]+=p[k]/points.length;q=points.map(p=>rotate(p.map((v,k)=>v-c[k])));const factor=Math.min(w,h)*.43*zoom/data.radius;projected=q.map(p=>[w/2+factor*p[0],h/2+factor*p[1]])}const order=data.mesh.faces.map((f,i)=>[f.reduce((v,j)=>v+q[j][2],0),i]).sort((a,b)=>b[0]-a[0]);for(const[_,i]of order){const f=data.mesh.faces[i];if(f.some(j=>!Number.isFinite(projected[j][0])||(mode==='camera'&&q[j][2]<=0)))continue;const a=q[f[0]],b=q[f[1]],c=q[f[2]],u=b.map((v,k)=>v-a[k]),v=c.map((value,k)=>value-a[k]),n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]],length=Math.hypot(...n)||1,shade=.58+.42*Math.abs((-.3*n[0]-.4*n[1]-n[2])/length/Math.sqrt(1.25)),color=[0,0,0];for(const j of f)for(let k=0;k<3;k++)color[k]+=data.mesh.colors[j][k]/3;sc.fillStyle=`rgb(${color.map(v=>Math.round(v*shade)).join(',')})`;sc.beginPath();f.forEach((j,k)=>k?sc.lineTo(...projected[j]):sc.moveTo(...projected[j]));sc.closePath();sc.fill()}sc.fillStyle=r.status==='unsupported'?'#ffb0b8':'#c4dde7';sc.font='12px system-ui';sc.fillText(mode==='camera'?'Fitted recording-camera projection':'Component-centred orbit; camera-relative pose',12,h-12);if(r.status==='unsupported'){sc.strokeStyle='#df4b57';sc.lineWidth=5;sc.strokeRect(2,2,w-4,h-4)}}
function stop(){if(timer)clearInterval(timer);timer=null;el('play').textContent='Play'}
el('frame').addEventListener('input',e=>{stop();index=Number(e.target.value);update()});el('previous').addEventListener('click',()=>{stop();index=Math.max(0,index-1);update()});el('next').addEventListener('click',()=>{stop();index=Math.min(data.frames.length-1,index+1);update()});el('play').addEventListener('click',()=>{if(timer){stop();return}el('play').textContent='Pause';timer=setInterval(()=>{index=(index+1)%data.frames.length;update()},data.frame_ms)});for(const id of['mesh-visible','foreground','raw-outline'])el(id).addEventListener('change',()=>{if(currentImages)drawOverlay(currentImages)});el('view-mode').addEventListener('change',drawScene);el('reset').addEventListener('click',()=>{yaw=0;pitch=0;zoom=1;drawScene()});scene.addEventListener('pointerdown',e=>{if(el('view-mode').value!=='orbit')return;drag=[e.clientX,e.clientY];scene.setPointerCapture(e.pointerId)});scene.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.01;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-drag[1])*.01));drag=[e.clientX,e.clientY];drawScene()});for(const event of['pointerup','pointercancel'])scene.addEventListener(event,()=>drag=null);scene.addEventListener('wheel',e=>{if(el('view-mode').value!=='orbit')return;e.preventDefault();zoom=Math.max(.3,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));drawScene()},{passive:false});window.addEventListener('resize',()=>{drawScene();if(currentImages)drawOverlay(currentImages)});document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;if(e.key==='ArrowRight'){index=Math.min(data.frames.length-1,index+1);update()}if(e.key==='ArrowLeft'){index=Math.max(0,index-1);update()}});update();
</script></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='battery-pilot-v2')
    args = parser.parse_args()
    case = ROOT / 'data/camera-battery-withdrawal-15fps'
    out = case / 'alignment' / args.run
    report = read(out / 'alignment.json')
    pack = read(case / 'geometry/local-battery-v3/experiment.json')
    if len(report['frames']) != len(pack['samples']):
        raise ValueError('Fit records and input samples do not match')
    for a, b in zip(report['frames'], pack['samples']):
        if a['sample_index'] != b['sample_index'] or a['timestamp_s'] != b['timestamp_s']:
            raise ValueError('Fit/source frame indices or timestamps differ')
    render_review_frames(out, report, pack)
    plot_evidence(out, report)
    mesh = trimesh.load(case / 'geometry/battery-fitted-v1/battery.ply', force='mesh')
    frames = []
    for record, row in zip(report['frames'], pack['samples']):
        frames.append({**record, 'source': '../../geometry/local-battery-v3/' + row['image']})
    vertices = np.asarray(mesh.vertices)
    center = vertices.mean(0)
    embedded = {'mesh': {'vertices': np.round(vertices, 8).tolist(), 'faces': np.asarray(mesh.faces).tolist(), 'colors': np.asarray(mesh.visual.vertex_colors)[:, :3].tolist()},
                'frames': frames, 'scale': report['asset_scale'], 'K': report['camera']['intrinsics'],
                'radius': float(np.linalg.norm(vertices - center, axis=1).max() * report['asset_scale']),
                'frame_ms': float(np.median(np.diff([r['timestamp_s'] for r in frames])) * 1000)}
    summary = report['summary']
    replacements = {'SUPPORTED': str(summary['supported_nonreference_frames']),
                    'MEDIAN': f"{summary['nonreference_nonblurred_median_of_heldout_medians_px']:.2f}",
                    'IOU': f"{summary['median_occlusion_composite_iou']:.3f}",
                    'SCALE': f"{report['asset_scale']:.5f}",
                    'UNSUPPORTED': ', '.join(map(str, summary['unsupported_samples'])),
                    'LIMITATIONS': ''.join('<li>' + html.escape(note) + '</li>' for note in report['evaluation_limits']),
                    'EMBEDDED_DATA': json.dumps(embedded, separators=(',', ':'), allow_nan=False).replace('</', r'<\/')}
    document = DOCUMENT
    # Replace long placeholders first: SUPPORTED is a substring of UNSUPPORTED.
    for key in sorted(replacements, key=len, reverse=True):
        document = document.replace(key, replacements[key])
    (out / 'index.html').write_text(document)
    print(json.dumps({'review': str(out / 'index.html'), 'video': str(out / 'preview.mp4'), 'summary': str(out / 'summary.jpg'), 'sample_count': len(frames)}, indent=2))


if __name__ == '__main__':
    main()
