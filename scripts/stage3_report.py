#!/usr/bin/env python3
"""Render a local depth/track diagnostic, including an offline orbitable surface."""
import argparse,json,html,re,subprocess,sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont,ImageOps
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from geometry import lift_pixels,intrinsics_from_hfov

def read(p):return json.loads(p.read_text())
def font(n):
    try:return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',n)
    except OSError:return ImageFont.load_default(size=n)

def surface_image(xyz,colors,yaw,pitch,size=(640,330)):
    image=Image.new('RGB',size,'#142838');d=ImageDraw.Draw(image)
    center=np.median(xyz,axis=0);p=xyz-center
    c,s=np.cos(yaw),np.sin(yaw);p=p@np.array([[c,0,s],[0,1,0],[-s,0,c]]).T
    c,s=np.cos(pitch),np.sin(pitch);p=p@np.array([[1,0,0],[0,c,-s],[0,s,c]]).T
    span=max(np.ptp(xyz,axis=0).max(),1e-8);q=p[:,:2]*min(size)*.82/span+np.array(size)/2
    for i in np.argsort(p[:,2])[::-1]:
        x,y=q[i];color=tuple(int(v) for v in colors[i]);d.ellipse((x-1,y-1,x+1,y+1),fill=color)
    return image

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',default='camera-battery-action-5fps');p.add_argument('--run',default='local-battery-v1');p.add_argument('--motion-dir',default='motion');a=p.parse_args()
    if any(not re.fullmatch(r'[a-z0-9][a-z0-9-]*',s) for s in [a.case,a.run,a.motion_dir]):p.error('Invalid case/run/motion directory')
    case=ROOT/'data'/a.case;out=case/'geometry'/a.run;pack=read(out/'experiment.json');rows=pack['samples'];depth_meta=read(out/'depth/metadata.json');motion=read(out/a.motion_dir/'metadata.json');tracks=np.load(out/a.motion_dir/'tracks.npz')
    if depth_meta['status']!='complete':raise ValueError('Depth incomplete')
    tag='' if a.motion_dir=='motion' else '-'+a.motion_dir.removeprefix('motion-')
    dest=out/('review'+tag);dest.mkdir(exist_ok=False)
    depths=[np.load(out/'depth'/f"{r['sample_index']:06d}.npy") for r in rows]
    masks=[(np.asarray(Image.open(out/r['mask']))>0)&np.isfinite(d)&(d>0) for r,d in zip(rows,depths)]
    depth_records={r['sample_index']:r for r in depth_meta['frames']}
    camera_label='assumed camera' if pack['camera']['origin']=='assumed_horizontal_fov' else 'model-estimated camera'
    lo,hi=np.percentile(np.concatenate([d[m][::10] for d,m in zip(depths,masks)]),[2,98]);clouds=[];timing=[]
    for j,(row,depth,mask) in enumerate(zip(rows,depths,masks)):
        base=np.asarray(Image.open(out/row['image']).convert('RGB'));overlay=base.copy();overlay[mask]=(overlay[mask]*.65+np.array([255,174,67])*.35).astype(np.uint8)
        drawing=Image.fromarray(overlay);d=ImageDraw.Draw(drawing)
        valid=tracks['valid'][j];uv=tracks['uv'][j]
        for k in np.flatnonzero(valid):
            x,y=uv[k];color=tuple(int(v) for v in cv2.applyColorMap(np.array([[k*4%255]],np.uint8),cv2.COLORMAP_TURBO)[0,0][::-1]);d.ellipse((x-3,y-3,x+3,y+3),fill=color,outline='white')
        normalized=np.clip((depth-lo)/max(hi-lo,1e-8),0,1)
        colored=cv2.applyColorMap((normalized*255).astype(np.uint8),cv2.COLORMAP_TURBO)[:,:,::-1].copy()
        colored[~mask]=(base[~mask]*.25).astype(np.uint8)
        ys,xs=np.where(mask);step=max(1,int(np.ceil(len(xs)/1500)));xs=xs[::step];ys=ys[::step]
        xyz=lift_pixels(np.c_[xs,ys],depth,pack['camera']['intrinsics']);colors=base[ys,xs]
        clouds.append({'sample':row['sample_index'],'source_s':row['timestamp_s'],'points':np.round(xyz,6).tolist(),'colors':colors.tolist()})
        canvas=Image.new('RGB',(1280,850),'#edf2f5');d=ImageDraw.Draw(canvas)
        d.text((16,10),f"Stage 3 local test | source {row['timestamp_s']:.3f}s | {int(valid.sum())}/{motion['seed_count']} tracked points",font=font(24),fill='#163044')
        usable=depth_records[row['sample_index']].get('usable_for_3d',True)
        notice='Predicted depth + '+camera_label+'. Uncalibrated; no accepted object pose.' if usable else 'DEPTH CORRECTION WITHHELD: raw diagnostic surface; exclude from 3D fitting.'
        d.text((16,45),notice,font=font(20),fill='#526979' if usable else '#9b3d22')
        canvas.paste(ImageOps.contain(drawing,(640,360)),(0,90));canvas.paste(ImageOps.contain(Image.fromarray(colored),(640,360)),(640,90))
        d.text((16,70),'RGB / SAM / '+motion['backend'],font=font(17),fill='#163044');d.text((656,70),f'Object depth: fixed colour range {lo:.2f}-{hi:.2f} model units',font=font(17),fill='#163044')
        canvas.paste(surface_image(xyz,colors,0,0),(0,490));canvas.paste(surface_image(xyz,colors,1.05,.2),(640,490))
        d.text((16,460),'Visible depth surface / front view',font=font(20),fill='#163044');d.text((656,460),'Same partial surface / oblique view',font=font(20),fill='#163044')
        d.text((16,827),'Surface views are centred independently for shape inspection; they do not show physical motion.',font=font(16),fill='#526979')
        canvas.save(dest/f'{j+1:06d}.jpg',quality=93)
        timing.append({'preview_index':j+1,'sample_index':row['sample_index'],'timestamp_s':row['timestamp_s']})
    fps=1/np.median(np.diff([r['timestamp_s'] for r in rows]));video=out/('local-depth-tracks'+tag+'-preview.mp4')
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-framerate',str(fps),'-i',str(dest/'%06d.jpg'),'-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-r','30','-movflags','+faststart',str(video)],check=True)
    anchor=next(i for i,r in enumerate(rows) if r['sample_index']==motion['anchor_sample']);summary_name='summary'+tag+'.jpg';Image.open(dest/f'{anchor+1:06d}.jpg').save(out/summary_name,quality=95)
    sensitivity=[];row=rows[anchor];ys,xs=np.where(masks[anchor]);uv=np.c_[xs,ys]
    for fov in [45,60,75]:
        xyz=lift_pixels(uv,depths[anchor],intrinsics_from_hfov(row['width'],row['height'],fov));sensitivity.append({'assumed_hfov':fov,'visible_surface_xyz_span_model_units':np.ptp(xyz,axis=0).tolist()})
    quality={'status':'generated_pending_visual_review','frame_count':len(rows),'seed_points':motion['seed_count'],'tracked_point_counts':[r['tracked_points'] for r in motion['frames']],'low_rigid_residual_nonanchor_frames':motion['low_residual_nonanchor_frames'],'nonanchor_frames':motion['evaluated_nonanchor_frames'],'intrinsics_sensitivity':sensitivity,'geometry_reconstruction_completed':False,'full_alignment_completed':False,'metric_accuracy_measured':False,'depth_color_range_fixed_across_frames':[float(lo),float(hi)],'preview_frame_map':timing}
    quality_name='quality'+tag+'.json';(out/quality_name).write_text(json.dumps(quality,indent=2)+'\n')
    e=html.escape;stats=''.join(f'<tr><td>{r["timestamp_s"]:.3f}s</td><td>{r["tracked_points"]}</td><td>{e(r["status"])}</td><td>{r.get("residual_fraction_of_reference_span",float("nan")):.3f}</td></tr>' for r in motion['frames'])
    buttons=''.join(f'<button data-frame="{i}">{r["timestamp_s"]:.2f}s</button>' for i,r in enumerate(rows))
    data=json.dumps(clouds,separators=(',',':'))
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Stage 3 / Local battery experiment</title><style>body{margin:0;background:#eef3f6;color:#183244;font:16px system-ui}main{max-width:1250px;margin:auto;padding:26px}video{width:100%}.notice{background:#dae8ef;padding:16px}canvas{display:block;width:100%;height:420px;background:#142838;touch-action:none;cursor:grab}button{padding:8px;margin:3px;background:white;color:#183244;border:1px solid #a0b6c5;border-radius:4px;cursor:pointer}button:focus-visible{outline:3px solid #2988bd}table{width:100%;border-collapse:collapse}td,th{padding:9px;text-align:left;border-bottom:1px solid #cbd8df}a{color:#245f9e}.scroll{overflow:auto}</style><main><h1>Stage 3 / Local battery experiment</h1>
<p class="notice">Real local depth inference and CPU optical-flow tracks. Depth units and camera intrinsics are uncalibrated. This is a partial visible surface, not a reconstructed complete battery. The procedural OBJ is an unfitted loader fixture only.</p>
<video controls preload="metadata" poster="summary.jpg" src="local-depth-tracks-preview.mp4"></video>
<h2>Inspect the estimated surface</h2><p>Drag to orbit. Use the mouse wheel to zoom. Each surface is centred independently for shape inspection; changing frames does not depict world-space movement. Missing backs stay missing.</p><p id="frame-label"></p><canvas id="cloud" aria-label="Orbitable partial depth surface"></canvas><p>FRAME_BUTTONS</p>
<p><a href="experiment.json">Experiment and coordinate contract</a> · <a href="depth/metadata.json">Depth model and timings</a> · <a href="motion/metadata.json">Point tracking and diagnostic fits</a> · <a href="quality.json">Quality and camera sensitivity</a> · <a href="unit-proxy-battery.obj">Unfitted procedural test proxy</a></p>
<h2>Diagnostic rigidity check</h2><p>We test whether lifted correspondences can be related by one fixed-scale rigid transform. Low residual is an internal consistency check, not accuracy. These transforms are not exported as accepted object poses. Ratios are median residual / reference point-cloud span.</p><div class="scroll"><table><tr><th>Source time</th><th>Tracked points</th><th>Diagnostic result</th><th>Residual ratio</th></tr>STAT_ROWS</table></div>
<h2>What remains</h2><p>Actual Meta SAM 3D geometry, estimated recording-camera calibration, temporally consistent depth and TAPIP3D trajectories remain GPU work. Full model-to-video fitting belongs to Stage 4. No physical identity across occlusion, insertion success or attachment is inferred here.</p></main><script>
const frames=CLOUD_DATA,canvas=document.getElementById('cloud'),ctx=canvas.getContext('2d');let current=ANCHOR,yaw=.8,pitch=.15,zoom=1,drag=null;
function draw(){const w=canvas.clientWidth,h=canvas.clientHeight,dpr=devicePixelRatio||1;canvas.width=w*dpr;canvas.height=h*dpr;ctx.scale(dpr,dpr);ctx.fillStyle='#142838';ctx.fillRect(0,0,w,h);const f=frames[current],points=f.points,c=[0,0,0];for(const p of points)for(let k=0;k<3;k++)c[k]+=p[k]/points.length;let span=0;for(const p of points)span=Math.max(span,Math.hypot(p[0]-c[0],p[1]-c[1],p[2]-c[2]));const scale=Math.min(w,h)*.42*zoom/Math.max(span,1e-8),cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);const q=points.map((p,i)=>{let x=p[0]-c[0],y=p[1]-c[1],z=p[2]-c[2],xx=cy*x+sy*z,zz=-sy*x+cy*z;return [w/2+xx*scale,h/2+(cp*y-sp*zz)*scale,sp*y+cp*zz,i];}).sort((a,b)=>b[2]-a[2]);for(const [x,y,z,i] of q){const color=f.colors[i];ctx.fillStyle=`rgb(${color.join(',')})`;ctx.fillRect(x-1.5,y-1.5,3,3);}document.getElementById('frame-label').textContent=`Source ${f.source_s.toFixed(3)}s / sample ${f.sample} / ${points.length} partial-surface points`;}
canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.01;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-drag[1])*.01));drag=[e.clientX,e.clientY];draw()});canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointercancel',()=>drag=null);canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.2,Math.min(5,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});document.querySelectorAll('[data-frame]').forEach(b=>b.addEventListener('click',()=>{current=Number(b.dataset.frame);draw()}));window.addEventListener('resize',draw);draw();
</script></html>'''.replace('FRAME_BUTTONS',buttons).replace('STAT_ROWS',stats).replace('CLOUD_DATA',data).replace('ANCHOR',str(anchor))
    doc=doc.replace('local-depth-tracks-preview.mp4',video.name).replace('summary.jpg',summary_name).replace('motion/metadata.json',a.motion_dir+'/metadata.json').replace('quality.json',quality_name).replace('CPU optical-flow tracks',html.escape(motion['backend'])+' tracks')
    if (out/'consistency.json').exists():
        doc=doc.replace('<h2>Diagnostic rigidity check</h2>','<p><a href="consistency.json">Depth consistency, disjoint point subsets and withheld frames</a></p><h2>Diagnostic rigidity check</h2>')
    (out/('review'+tag+'.html')).write_text(doc);print(out/('review'+tag+'.html'))

if __name__=='__main__':main()
