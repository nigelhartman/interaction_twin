import assert from 'node:assert/strict';
import * as THREE from '../workspace/robot-vendor/three.module.js';
import {sourceCameraMatrices,letterbox} from '../workspace/geometry-camera.mjs';
let checked=0,maxError=0;
for(let i=0;i<100;i++){
 const rot=new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(i*.013,-i*.007,i*.003));
 const e=rot.elements,R=[[e[0],e[4],e[8]],[e[1],e[5],e[9]],[e[2],e[6],e[10]]],t=[.1*Math.sin(i),-.2,.3];
 const w=i%2?960:640,h=i%2?540:480,K=[[730+i,i%7,275+i%19],[0,710+i,235-i%13],[0,0,1]];
 const a=sourceCameraMatrices(R,t,K,w,h),camera=new THREE.Camera();camera.matrixAutoUpdate=false;camera.matrix.fromArray(a.world);camera.updateMatrixWorld(true);camera.projectionMatrix.fromArray(a.projection);camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
 for(let j=0;j<100;j++){
  const q=[(j%10-5)*.13,(Math.floor(j/10)-5)*.11,3+j*.015];
  const p=[0,1,2].map(k=>R[0][k]*(q[0]-t[0])+R[1][k]*(q[1]-t[1])+R[2][k]*(q[2]-t[2]));
  const expected=[(K[0][0]*q[0]+K[0][1]*q[1])/q[2]+K[0][2],K[1][1]*q[1]/q[2]+K[1][2]];
  const n=new THREE.Vector3(...p).project(camera),actual=[(n.x+1)*w/2-.5,(1-n.y)*h/2-.5];
  const error=Math.hypot(actual[0]-expected[0],actual[1]-expected[1]);maxError=Math.max(maxError,error);assert.ok(error<1e-9);checked++;
 }
}
for(const [w,h,iw,ih]of[[1200,700,960,540],[390,600,960,540],[600,390,480,640],[100,900,1920,1080]]){
 const r=letterbox(w,h,iw,ih);assert.ok(r.x>=0&&r.y>=0);assert.ok(r.width<=w+1e-9&&r.height<=h+1e-9);assert.ok(Math.abs(r.width/r.height-iw/ih)<1e-12);assert.ok(Math.abs(r.x*2+r.width-w)<1e-9);assert.ok(Math.abs(r.y*2+r.height-h)<1e-9);
}
assert.throws(()=>letterbox(0,300,100,100));assert.throws(()=>sourceCameraMatrices([],[],[],960,540));
console.log(JSON.stringify({checked,max_projection_error_px:maxError,letterbox_cases:4}));
