const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const F=require('../showcase/robot-camera-fit-v1.js'),C=require('../showcase/robot-controller.js');require('../showcase/robot-task-plan.js');const A=require('../showcase/aligned-scene-contract.js'),R=require('../showcase/robot-presentation-v2.js');
const ROOT=path.resolve(__dirname,'..'),read=p=>JSON.parse(fs.readFileSync(path.join(ROOT,p))),directory='output/brand-trailer-20261103/robot-capture-v2/';
function corners(bounds){return Array.from({length:8},(_,mask)=>[0,1,2].map(i=>bounds[(mask>>i)&1][i]));}
function envelope(){
 const r=R.build(read(directory+'baseline-task.json'),read(directory+'robot-kit.json'),read(directory+'refinement-inputs.json')),bounds=Object.fromEntries(read('output/brand-trailer-20261103/robot-capture-v3/native-bounds.json').parts.map(p=>[p.id,p.native_bounds])),points=[],pad=(p,radius)=>points.push(...corners([p.map(v=>v-radius),p.map(v=>v+radius)]));
 const base=r.plan.objects.find(o=>o.role==='assembly_base'),bb=bounds[base.id],center=bb[0].map((v,i)=>(v+bb[1][i])/2),scale=base.display_max_dimension/Math.max(...bb[0].map((v,i)=>bb[1][i]-v)),joint=r.plan.joints[0],at=(m,p)=>m.slice(0,3).map(row=>row[3]+row.slice(0,3).reduce((s,v,i)=>s+v*p[i],0)),world=(p,s)=>p.map((v,i)=>(v-center[i])*scale+s.objects[base.id].position[i]);
 pad(r.timeline.robot.base,.18);
 for(let frame=0;frame<=Math.ceil(r.timeline.duration_s*24);frame++){const s=C.at(r.timeline,Math.min(frame/24,r.timeline.duration_s));for(const key of['origin','elbow','wrist','tip'])pad(s.ik[key],key==='tip'?.12:.095);const q=s.object_joints[base.id+':'+joint.id],child=A.multiply(A.multiply(joint.source_joint.parent_from_joint,A.axisRotation(joint.source_joint.axis,q)),joint.source_joint.joint_from_child_zero);for(const p of corners(bb))points.push(world(p,s));for(const p of corners(bounds[joint.child_id]))points.push(world(at(child,p),s));for(const o of r.plan.objects.filter(o=>o.role==='removable')){const b=bounds[o.id],c=b[0].map((v,i)=>(v+b[1][i])/2),scale=o.display_max_dimension/Math.max(...b[0].map((v,i)=>b[1][i]-v));for(const p of corners(b))points.push(p.map((v,i)=>(v-c[i])*scale+s.objects[o.id].position[i]));}}
 for(const goal of r.goals)pad(goal.position,.09);return{points,presentation:r};
}
test('complete actual robot envelope fits the independent Three.js camera projection with visible goals',async()=>{
 const T=await import('../workspace/robot-vendor/three.module.js'),{points,presentation:r}=envelope(),fit=F.fit(points),camera=new T.PerspectiveCamera(fit.vertical_fov_deg,fit.aspect,fit.near,fit.far);camera.position.fromArray(fit.position);camera.lookAt(...fit.target);camera.updateMatrixWorld(true);
 for(const p of points){const q=new T.Vector3(...p).project(camera);assert.ok(Math.abs(q.x)<=.86+1e-9&&Math.abs(q.y)<=.86+1e-9);assert.ok(q.z>-1&&q.z<1);}
 const pixel=p=>{const v=new T.Vector3(...p).project(camera);return[(v.x+1)*960,(1-v.y)*540];},targets=r.goals.map(g=>pixel(g.position));assert.ok(Math.hypot(...targets[0].map((v,i)=>v-targets[1][i]))>50);
 const old=new T.PerspectiveCamera(36,16/9,.01,20);old.position.set(.85,1.02,1.3);old.lookAt(-.35,.25,.04);old.updateMatrixWorld(true);assert.ok(points.some(p=>Math.abs(new T.Vector3(...p).project(old).y)>1),'Regression fixture must include the observed old top crop.');
});
test('camera framing translates with the scene and rejects invalid or vertical envelopes',()=>{
 const points=[[-1,0,-.4],[1,.8,.2],[.2,1.4,.4]],a=F.fit(points),delta=[.5,.2,-.3],b=F.fit(points.map(p=>p.map((v,i)=>v+delta[i])));assert.ok(a.position.every((v,i)=>Math.abs(v+delta[i]-b.position[i])<1e-12));assert.throws(()=>F.fit([[NaN,0,0]]));assert.throws(()=>F.fit(points,{view_direction:[0,1,0]}));assert.throws(()=>F.fit(points,{edge_margin_fraction:0}));
});
module.exports={envelope};
