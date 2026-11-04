'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const R=require('../showcase/robot-recorder.js');
function fixture({duration=2,constructorError=false}={}){
 const raf=new Map(),timers=new Map(),recorders=[],ready=[],statuses=[],revoked=[];let id=0,time=0,stops=0,starts=0,trackStops=0;
 const state={ready:true,time_s:0,duration_s:duration};
 class Recorder{
  static isTypeSupported(type){return type==='video/webm;codecs=vp8';}
  constructor(stream,options){if(constructorError)throw Error('Encoder unavailable');this.state='inactive';this.mimeType=options.mimeType;recorders.push(this);}
  start(){this.state='recording';}
  stop(){this.state='inactive';this.ondataavailable({data:new Blob(['encoded frames'])});this.onstop();}
 }
 const canvas={width:800,height:480,captureStream:()=>({getTracks:()=>[{stop(){trackStops++;}}]})};
 const ctx={fillRect(){},fillText(){},drawImage(){}};
 const env={MediaRecorder:Recorder,HTMLCanvasElement:{prototype:{captureStream(){}}},Blob,URL:{createObjectURL:()=> 'blob:clip',revokeObjectURL:u=>revoked.push(u)},performance:{now:()=>time},document:{createElement:()=>({...canvas,getContext:()=>ctx})},requestAnimationFrame:fn=>{raf.set(++id,fn);return id;},cancelAnimationFrame:n=>raf.delete(n),setTimeout:fn=>{timers.set(++id,fn);return id;},clearTimeout:n=>timers.delete(n)};
 const session=R.create({canvas,getState:()=>state,startPlayback:()=>{starts++;state.time_s=0;},stopPlayback:()=>stops++,getCaption:()=> 'Open the connected lid',onReady:x=>ready.push(x),onStatus:x=>statuses.push(x),env});
 return{session,state,env,ready,statuses,recorders,revoked,advance:t=>{time=t*1000;state.time_s=t;const entries=[...raf];raf.clear();for(const[,fn]of entries)fn();},counts:()=>({stops,starts,trackStops,raf:raf.size,timers:timers.size})};
}
test('actual canvas preview recording ends at task completion and releases media resources',()=>{const f=fixture();assert.equal(R.supported(f.env),true);f.session.start();assert.equal(f.session.active,true);f.advance(.5);f.advance(2);assert.equal(f.session.active,false);assert.equal(f.ready.length,1);assert.equal(f.ready[0].mime,'video/webm;codecs=vp8');assert.equal(f.ready[0].filename,'interaction-twin-robot-task.webm');assert.ok(f.ready[0].bytes>0);assert.deepEqual(f.counts(),{stops:1,starts:1,trackStops:1,raf:0,timers:0});f.session.dispose();assert.deepEqual(f.revoked,['blob:clip']);});
test('encoder failures and oversize streams never produce a misleading finished clip',()=>{for(const kind of ['error','oversize']){const f=fixture();f.session.start();if(kind==='error')f.recorders[0].onerror();else f.recorders[0].ondataavailable({data:{size:41*1024*1024}});assert.equal(f.ready.length,0);assert.equal(f.counts().trackStops,1);assert.equal(f.session.active,false);}});
test('unsupported duration and constructor failure are bounded without leaving a stream alive',()=>{const long=fixture({duration:61});assert.throws(()=>long.session.start(),/60 seconds/);assert.equal(long.counts().trackStops,0);const broken=fixture({constructorError:true});assert.throws(()=>broken.session.start(),/Encoder unavailable/);assert.equal(broken.counts().trackStops,1);assert.equal(broken.session.active,false);});
test('page disposal cancels recording without publishing a partial artifact',()=>{const f=fixture();f.session.start();f.session.dispose();assert.equal(f.ready.length,0);assert.equal(f.counts().trackStops,1);assert.throws(()=>f.session.start(),/unavailable/);});
