/* Record the actual local robot preview. No camera, microphone, upload or model call. */
(function(root){'use strict';
const TYPES=['video/webm;codecs=vp9','video/webm;codecs=vp8','video/mp4','video/webm'];
function supported(env=root){return !!(env.MediaRecorder&&env.HTMLCanvasElement?.prototype?.captureStream&&TYPES.some(t=>env.MediaRecorder.isTypeSupported(t)));}
function selectMime(Recorder){return TYPES.find(t=>Recorder.isTypeSupported(t))||null;}
function create({canvas,getState,startPlayback,stopPlayback,getCaption=()=>'',onStatus=()=>{},onReady=()=>{},env=root}){
 let recorder=null,stream=null,chunks=[],drawing=null,deadline=null,output=null,bytes=0,started=0,disposed=false,stopping=false,invalidated=false;
 const clock=()=>env.performance.now();
 function cleanup(){if(drawing!==null)env.cancelAnimationFrame(drawing);drawing=null;if(deadline!==null)env.clearTimeout(deadline);deadline=null;for(const track of stream?.getTracks()||[])track.stop();stream=null;}
 function stop(){if(!recorder||stopping)return;stopping=true;stopPlayback();if(recorder.state!=='inactive')recorder.stop();else cleanup();}
 function start(){
  if(disposed||recorder)throw Error('A recording is already active or unavailable.');
  const state=getState(),mime=selectMime(env.MediaRecorder);
  if(!mime||!canvas.captureStream)throw Error('This browser cannot record a canvas video.');
  if(!state?.ready||!Number.isFinite(state.duration_s)||state.duration_s<=0||state.duration_s>60)throw Error('Choose a ready task of at most 60 seconds.');
  const target=env.document.createElement('canvas');target.width=1280;target.height=800;
  const ctx=target.getContext('2d');if(!ctx)throw Error('The video canvas is unavailable.');
  stream=target.captureStream(30);chunks=[];bytes=0;stopping=false;invalidated=false;started=clock();
  try{recorder=new env.MediaRecorder(stream,{mimeType:mime,videoBitsPerSecond:4500000});}catch(error){cleanup();throw error;}
  recorder.ondataavailable=event=>{if(invalidated)return;if(event.data?.size){bytes+=event.data.size;if(bytes>40*1024*1024){invalidated=true;chunks=[];onStatus('Recording stopped at the 40 MB clip limit.');stop();return;}chunks.push(event.data);}};
  recorder.onerror=()=>{invalidated=true;chunks=[];onStatus('Recording failed. The task and native assets are unchanged.');stop();};
  recorder.onstop=()=>{const type=recorder.mimeType||mime,elapsed=(clock()-started)/1000;cleanup();recorder=null;stopping=false;if(!disposed&&!invalidated&&chunks.length&&bytes<=40*1024*1024){if(output)env.URL.revokeObjectURL(output);const blob=new env.Blob(chunks,{type});output=env.URL.createObjectURL(blob);onReady({url:output,filename:'interaction-twin-robot-task.'+(type.startsWith('video/mp4')?'mp4':'webm'),mime:type,bytes:blob.size,duration_s:elapsed});onStatus('Task clip ready · '+elapsed.toFixed(1)+'s · '+(blob.size/1024/1024).toFixed(1)+'MB');}chunks=[];};
  function draw(){if(!recorder||stopping)return;const state=getState();ctx.fillStyle='#f7f8fb';ctx.fillRect(0,0,1280,800);ctx.fillStyle='#315fdb';ctx.font='700 14px Arial';ctx.fillText('INTERACTION TWIN / ROBOTICS WORKBENCH',36,34);ctx.fillStyle='#182238';ctx.font='700 25px Arial';ctx.fillText('A captured object. A programmable task.',36,68);
   const ratio=Math.min(1280/canvas.width,630/canvas.height),w=canvas.width*ratio,h=canvas.height*ratio;ctx.drawImage(canvas,(1280-w)/2,88+(630-h)/2,w,h);
   ctx.fillStyle='#fff';ctx.fillRect(0,718,1280,82);ctx.fillStyle='#253450';ctx.font='700 17px Arial';ctx.fillText(String(getCaption()).slice(0,112),36,749);ctx.fillStyle='#66748a';ctx.font='14px Arial';ctx.fillText('Source-bound native parts · authored robot staging · joint-limited IK · no physical validation',36,778);
   if(!state?.ready||state.time_s>=state.duration_s-.015){stop();return;}drawing=env.requestAnimationFrame(draw);
  }
  try{startPlayback();recorder.start(250);onStatus('Recording the actual task preview…');deadline=env.setTimeout(stop,Math.min(65000,(state.duration_s+5)*1000));drawing=env.requestAnimationFrame(draw);}catch(error){cleanup();recorder=null;stopPlayback();throw error;}
 }
 return{start,stop,get active(){return !!recorder;},dispose(){disposed=true;stop();if(output){env.URL.revokeObjectURL(output);output=null;}}};
}
const api=Object.freeze({create,supported,selectMime});root.RobotPreviewRecording=api;if(typeof module!=='undefined'&&module.exports)module.exports=api;
if(!root.document)return;
const button=document.getElementById('record-preview'),link=document.getElementById('record-download'),canvas=document.getElementById('workbench-canvas');
if(!button||!link||!canvas)return;
let session=null,locked=[];
const status=document.createElement('span');status.id='record-status';status.setAttribute('role','status');status.setAttribute('aria-live','polite');button.parentNode.append(status);
function restore(){for(const[node,disabled]of locked)node.disabled=disabled;locked=[];button.textContent='Record task clip';}
button.onclick=()=>{try{if(session?.active){session.stop();return;}const work=root.RobotWorkbench;if(!work?.snapshot().ready)throw Error('Wait for a ready task.');if(!supported(root))throw Error('Canvas video recording is unavailable in this browser.');link.hidden=true;
 locked=Array.from(document.querySelectorAll('#task-form input,#task-form button,[data-prompt],#apply-plan,#task-time,#task-speed,#reset-task,#play-task')).map(node=>[node,node.disabled]);for(const[node]of locked)node.disabled=true;
 session?.dispose();session=create({canvas,getState:()=>work.snapshot(),startPlayback:()=>{work.play(false);work.seek(0);const speed=document.getElementById('task-speed');speed.value='1';work.play(true);},stopPlayback:()=>{work.play(false);restore();},getCaption:()=>document.getElementById('current-step')?.textContent||'Robot task preview',onStatus:text=>{status.textContent=text;},onReady:result=>{link.href=result.url;link.download=result.filename;link.textContent='Download task clip ↓';link.hidden=false;link.dataset.bytes=String(result.bytes);link.dataset.duration=String(result.duration_s);link.dataset.mime=result.mime;}});button.textContent='Stop recording';session.start();
 }catch(error){restore();status.textContent=error.message;}};
addEventListener('pagehide',()=>{session?.dispose();restore();});
})(globalThis);
