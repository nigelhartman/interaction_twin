#!/usr/bin/env python3
"""Clone the exact workbench renderer for one output-only, full-canvas capture.

Only the initial plan transport, fixed editorial camera, viewport and recording
controls differ. The original renderer, controller, rig and saved task stay
unchanged. The generated page is an additive local capture helper, not a demo
default or a public export.
"""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import build_flagship_trailer_20261104_v1 as F

APPEND=r'''
// Additive film capture adapter. No source geometry, joint, plan or IK changes.
const CC_W=1920,CC_H=1080,CC_D=14,CC_FPS=30;
const ccHash=async bytes=>[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(v=>v.toString(16).padStart(2,'0')).join('');
const ccBytes=value=>new TextEncoder().encode(JSON.stringify(value));
const ccPanel=document.createElement('div');ccPanel.id='case-capture-controls';
ccPanel.innerHTML='<button id="cc-record">Record full14s</button> <button data-cc-seek="0">Start</button> <button data-cc-seek="2">Open</button> <button data-cc-seek="8">Placement</button> <button data-cc-seek="13.5">Final</button> <span id="cc-status">Preparing camera…</span> <a id="cc-video" hidden>Video↓</a> <a id="cc-receipt" hidden>Receipt↓</a>';
document.body.append(ccPanel);
function ccClock(t){const value=Math.min(15,Math.max(0,t/14*15));for(let i=1;i<CASE_CAPTURE_KNOTS.length;i++){const[a,x]=CASE_CAPTURE_KNOTS[i-1],[b,y]=CASE_CAPTURE_KNOTS[i];if(value<=b)return x+(y-x)*(value-a)/(b-a);}return timeline.duration_s;}
function ccState(t){const state=C.at(timeline,ccClock(t));renderRobot(state);renderObjects(state);scene.updateMatrixWorld(true);return state;}
function ccGroups(){return[['robot',armRoot],['case_assembly',caseAssembly.root],...nativeRoots.entries()];}
function ccCorners(box){const out=[];for(const x of[box.min.x,box.max.x])for(const y of[box.min.y,box.max.y])for(const z of[box.min.z,box.max.z])out.push(new THREE.Vector3(x,y,z));return out;}
function ccFocusGroups(){return[['tool_and_fingers',robotJoints[4]],['case_assembly',caseAssembly.root],...nativeRoots.entries()];}
function ccProjected(){return ccFocusGroups().map(([id,object])=>({id,points:ccCorners(new THREE.Box3().setFromObject(object,true)).map(p=>{p.project(camera);return[(p.x+1)/2*CC_W,(1-p.y)/2*CC_H,p.z];})}));}
let ccBounds=null,ccRecording=false,ccURLs=[],ccWide=null,ccMedium=null;
function ccFixedView(points){
 const direction=new THREE.Vector3(1.35,1.15,1.7).normalize(),initial=new THREE.Box3().setFromPoints(points).getCenter(new THREE.Vector3());camera.position.copy(initial).add(direction);camera.lookAt(initial);camera.updateMatrixWorld(true);
 const right=new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld,0),up=new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld,1),tan=Math.tan(camera.fov*Math.PI/360),horizontal=tan*camera.aspect*(1-2*110/CC_W),vertical=tan*(1-2*75/CC_H),coords=points.map(p=>[p.dot(right),p.dot(up),p.dot(direction)]),maxDepth=Math.max(...coords.map(p=>p[2]));
 function fitAt(depth){let xl=-Infinity,xh=Infinity,yl=-Infinity,yh=Infinity;for(const[x,y,z]of coords){const d=depth-z;if(d<=.03)return null;xl=Math.max(xl,x-d*horizontal);xh=Math.min(xh,x+d*horizontal);yl=Math.max(yl,y-d*vertical);yh=Math.min(yh,y+d*vertical);}return xl<=xh&&yl<=yh?[(xl+xh)/2,(yl+yh)/2]:null;}
 let lo=maxDepth+.031,hi=lo+1;while(!fitAt(hi)){hi=maxDepth+(hi-maxDepth)*2;if(hi>100)throw Error('Camera fit unbounded.');}for(let i=0;i<60;i++){const mid=(lo+hi)/2;if(fitAt(mid))hi=mid;else lo=mid;}
 const depth=hi+.005,[cx,cy]=fitAt(depth),target=new THREE.Vector3().addScaledVector(right,cx).addScaledVector(up,cy).addScaledVector(direction,initial.dot(direction));
 return{position:target.clone().addScaledVector(direction,depth-initial.dot(direction)),target};
}
function ccCamera(t){const u=Math.min(1,Math.max(0,(t-1)/1.3)),mix=u*u*(3-2*u),position=ccWide.position.clone().lerp(ccMedium.position,mix),target=ccWide.target.clone().lerp(ccMedium.target,mix);camera.position.copy(position);camera.lookAt(target);camera.updateMatrixWorld(true);controls.target.copy(target);}
function ccFit(){
 setPlaying(false);resizeObserver?.disconnect();renderer.setPixelRatio(1);renderer.setSize(CC_W,CC_H,false);camera.aspect=CC_W/CC_H;camera.updateProjectionMatrix();controls.enabled=false;controls.enableDamping=false;
 const union=new THREE.Box3(),whole=[],focus=[];for(let t=0;t<=CC_D+.001;t+=.05){ccState(Math.min(CC_D,t));for(const[,object]of ccGroups()){const box=new THREE.Box3().setFromObject(object,true);union.union(box);whole.push(...ccCorners(box));}for(const[,object]of ccFocusGroups())focus.push(...ccCorners(new THREE.Box3().setFromObject(object,true)));}
 const tray=ccCorners(new THREE.Box3(new THREE.Vector3(-.51,0,.19),new THREE.Vector3(-.19,.02,.51)));whole.push(...tray);focus.push(...tray);ccWide=ccFixedView(whole);ccMedium=ccFixedView(focus);
 ccBounds={world:[union.min.toArray(),union.max.toArray()],wide:{position:ccWide.position.toArray(),target:ccWide.target.toArray()},medium:{position:ccMedium.position.toArray(),target:ccMedium.target.toArray()},full_trajectory_samples:281,fit:'per-object perspective bounds, one-second full-rig establish then smooth1.3s push to case, tray, both earbuds and complete tool/fingers; upper arm intentionally may leave frame after establish',minimum_horizontal_margin_px:96,minimum_vertical_margin_px:65};
 const proofs=[];for(let t=0;t<=CC_D+.001;t+=.05){ccState(Math.min(CC_D,t));ccCamera(t);const boxes=ccProjected();if(boxes.some(b=>b.points.some(p=>p[0]<96||p[0]>CC_W-96||p[1]<65||p[1]>CC_H-65||p[2]<-1||p[2]>1)))throw Error('Full task focus framing failed.');proofs.push({time_s:Math.min(CC_D,t),projected_bounds:boxes,camera_position:camera.position.toArray()});}
 ccBounds.projected_full_trajectory=proofs;ccState(0);ccCamera(0);renderer.render(scene,camera);$('cc-status').textContent='Ready · establishing shot → full task focus';
}
function ccDraw(t){const state=ccState(t);ccCamera(t);const boxes=ccProjected();if(boxes.some(b=>b.points.some(p=>p[0]<96||p[0]>CC_W-96||p[1]<65||p[1]>CC_H-65||p[2]<-1||p[2]>1)))throw Error('Capture bounds changed.');renderer.render(scene,camera);return{film_s:t,captured_task_runtime_s:ccClock(t),state,projected_bounds:boxes,camera_position:camera.position.toArray(),camera_target:controls.target.toArray()};}
function ccLink(id,blob,name){const url=URL.createObjectURL(blob);ccURLs.push(url);$(id).href=url;$(id).download=name;$(id).hidden=false;}
async function ccRecord(){
 if(!ready||ccRecording)throw Error('Wait for the exact source-bound task.');if(await ccHash(ccBytes(plan))!==CASE_CAPTURE_PLAN_SHA)throw Error('Saved task bytes differ from prior capture.');
 const mime=['video/webm;codecs=vp9','video/webm;codecs=vp8'].find(v=>MediaRecorder.isTypeSupported(v));if(!mime)throw Error('Canvas capture unavailable.');ccRecording=true;$('cc-record').disabled=true;$('cc-status').textContent='Recording actual WebGL…';
 const stream=renderer.domElement.captureStream(CC_FPS),recorder=new MediaRecorder(stream,{mimeType:mime,videoBitsPerSecond:14000000}),chunks=[],trace=[];let start=null,failed=false,total=0;
 recorder.ondataavailable=event=>{total+=event.data.size;if(total>64*1024*1024){failed=true;recorder.stop();}else if(event.data.size)chunks.push(event.data);};recorder.onerror=()=>{failed=true;};
 recorder.onstop=async()=>{stream.getTracks().forEach(track=>track.stop());ccRecording=false;$('cc-record').disabled=false;if(failed||trace.at(-1)?.film_s<CC_D-.02){$('cc-status').textContent='Incomplete capture withheld.';return;}
  const final=trace.at(-1).state,goals=plan.goal.component_ids.map(id=>({id,error:Math.hypot(...final.objects[id].position.map((v,i)=>v-timeline.final.objects[id].position[i])),attached:final.attached===id})),closed=Object.values(final.object_joints).every(v=>Math.abs(v)<1e-7);
  if(final.attached!==null||!closed||goals.some(g=>g.error>1e-5||g.attached)){$('cc-status').textContent='Final goal validation failed.';return;}
  const blob=new Blob(chunks,{type:mime}),bytes=await blob.arrayBuffer(),videoRef={path:'output/trailer-rework-20261105-v1/case-capture/fullscreen-case-14s.webm',sha256:await ccHash(bytes),bytes:blob.size};
  const receipt={format:'interaction-twin-fullscreen-case-rehearsal-capture',version:1,video_ref:videoRef,canvas:[CC_W,CC_H],fps:CC_FPS,duration_s:CC_D,encoded_clock_note:'Real-time14s canvas recording; decoder clock is separately checked on intake.',typed_plan:plan,typed_plan_snapshot_JSON_sha256:CASE_CAPTURE_PLAN_SHA,original_workbench_renderer:CASE_CAPTURE_ORIGINAL_JS,editorial_clock_knots:CASE_CAPTURE_KNOTS,prior_capture_receipt:CASE_CAPTURE_PRIOR_RECEIPT,native_asset_refs:plan.objects.map(o=>o.asset_ref),framing:ccBounds,trace,final_goals:goals,final_attached:final.attached,final_closed_reference:closed,both_earbuds_released:true,full_source_action_order_preserved:true,source_geometry_modified:false,authored_kinematic_rehearsal:true,physical_contact_verified:false,simulation_ready:false,new_API_calls:0,new_GPU_calls:0};
  ccLink('cc-video',blob,'fullscreen-case-14s.webm');ccLink('cc-receipt',new Blob([JSON.stringify(receipt,null,2)+'\n'],{type:'application/json'}),'fullscreen-case-14s.json');$('cc-status').textContent='Complete · both placements, closed case, action framed';
 };
 ccDraw(0);recorder.start(200);function step(now){if(recorder.state!=='recording')return;start??=now;const t=Math.min(CC_D,(now-start)/1000);try{trace.push(ccDraw(t));}catch(error){failed=true;$('cc-status').textContent=error.message;recorder.stop();return;}if(t===CC_D)recorder.stop();else requestAnimationFrame(step);}requestAnimationFrame(step);
}
for(const button of document.querySelectorAll('[data-cc-seek]'))button.onclick=()=>{if(!ccRecording)ccDraw(Number(button.dataset.ccSeek));};$('cc-record').onclick=()=>ccRecord().catch(error=>{$('cc-status').textContent=error.message;});
if(!ready)throw Error('Exact workbench failed to initialize.');if(await ccHash(ccBytes(plan))!==CASE_CAPTURE_PLAN_SHA)throw Error('Preserved plan snapshot mismatch.');ccFit();
window.CaseMotionCapture={snapshot:()=>({ready,recording:ccRecording,plan_sha256:CASE_CAPTURE_PLAN_SHA,framing:ccBounds}),seek:t=>ccDraw(t)};
addEventListener('pagehide',()=>{for(const url of ccURLs)URL.revokeObjectURL(url);});
'''

def main():
    prior=ROOT/'output/flagship-trailer-20261104-v1/captures/current-robot-v2/flagship-current-robot-20261104-v1.json'
    doc=json.loads(prior.read_text());original=ROOT/'showcase/robot-workbench.js';source=original.read_text()
    before="initial=await jsonURL('/api/robotics/plan?'+params);"
    assert source.count(before)==1;source=source.replace(before,'initial=copy(CASE_CAPTURE_PLAN);')
    preamble='const CASE_CAPTURE_PLAN='+json.dumps(doc['typed_plan'],separators=(',',':'))+';\n'
    for name,value in [('CASE_CAPTURE_PLAN_SHA',doc['typed_plan_snapshot_JSON_sha256']),('CASE_CAPTURE_KNOTS',F.robot_editorial_knots(doc['typed_plan'])),('CASE_CAPTURE_ORIGINAL_JS',F.ref(original)),('CASE_CAPTURE_PRIOR_RECEIPT',F.ref(prior))]:preamble+='const '+name+'='+json.dumps(value,separators=(',',':'))+';\n'
    js=ROOT/'showcase/case-motion-capture-20261105-v1.js';page=ROOT/'showcase/case-motion-capture-20261105-v1.html'
    assert not js.exists()and not page.exists(),'Preserve previous generated helpers.'
    js.write_text(preamble+source+APPEND)
    html=(ROOT/'showcase/robot-workbench.html').read_text().replace('src="robot-workbench.js"','src="case-motion-capture-20261105-v1.js"')
    style='<style>html,body{margin:0!important;width:1920px!important;height:1080px!important;overflow:hidden!important}#workbench-viewport{position:fixed!important;inset:0!important;width:1920px!important;height:1080px!important;min-height:0!important;border:0!important;z-index:1000}#workbench-canvas{width:min(100vw,1920px)!important;height:auto!important}#workbench-viewport>*:not(canvas){display:none!important}#case-capture-controls{position:fixed;left:18px;bottom:18px;z-index:9999;background:#fff;padding:12px;border-radius:12px;font:14px Arial;box-shadow:0 3px 16px #0002}#case-capture-controls button,#case-capture-controls a{padding:8px;margin-right:8px}#case-capture-controls a[hidden]{display:none}</style>'
    html=html.replace('</head>',style+'</head>');page.write_text(html)
    print(json.dumps({'page':F.ref(page),'script':F.ref(js),'original_renderer':F.ref(original),'prior_plan_sha256':doc['typed_plan_snapshot_JSON_sha256'],'url':'http://127.0.0.1:8766/showcase/case-motion-capture-20261105-v1.html?capture=1&case=flow-sam3d-default-e2e-20261102-v1--flow-procedure-1'},indent=2))
if __name__=='__main__':main()
