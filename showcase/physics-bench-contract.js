/* Strict saved-physics replay contract. Never upgrades the source twin claims. */
(function(root){'use strict';
const fail=m=>{throw Error(m);},finite=n=>typeof n==='number'&&Number.isFinite(n),vec=(v,n)=>Array.isArray(v)&&v.length===n&&v.every(finite);
function validate(d){
 if(d?.format!=='interaction-twin-newton-contact-experiment'||d.version!==1)fail('Unsupported physics experiment.');
 if(d.engine?.name!=='Newton'||d.engine.version!=='1.7.0'||d.engine.device!=='cpu'||d.engine.solver!=='SolverXPBD'||d.engine.steps!==2400||Math.abs(d.engine.timestep_s-1/600)>1e-12)fail('Unknown engine execution.');
 if(d.claims?.real_physics_engine_execution!==true||d.claims.metric_scale_measured!==false||d.claims.material_parameters_measured!==false||d.claims.physical_accuracy_verified!==false||d.claims.robot_grasp_validated!==false||d.claims.simulation_ready!==false)fail('Unsupported physical accuracy claim.');
 if(!Array.isArray(d.objects)||d.objects.length!==3||!Array.isArray(d.frames)||d.frames.length!==241)fail('Incomplete saved physics sequence.');
 const ids=new Set();
 for(const o of d.objects){if(!/^[A-Za-z][A-Za-z0-9_-]{0,80}$/.test(o.id)||ids.has(o.id)||!vec(o.native_center,3)||!finite(o.native_to_assumed_metre_scale)||o.native_to_assumed_metre_scale<=0)fail('Invalid physics object.');ids.add(o.id);const r=o.asset_ref;if(!r||!/^data\/[A-Za-z0-9_./-]+\.glb$/.test(r.path)||r.path.split('/').some(p=>!p||p.startsWith('.'))||!/^([a-f0-9]{64})$/.test(r.sha256)||!Number.isInteger(r.bytes)||r.bytes<=0||r.bytes>32*1024*1024)fail('Unverified native mesh reference.');const c=o.collision_proxy;if(c?.kind!=='deterministic_convex_support_hull'||c.concavities_preserved!==false||!Array.isArray(c.vertices)||c.vertices.length<4||c.vertices.length>1024||!c.vertices.every(v=>vec(v,3))||!Array.isArray(c.triangles)||c.triangles.length>4096||!c.triangles.every(t=>Array.isArray(t)&&t.length===3&&t.every(i=>Number.isInteger(i)&&i>=0&&i<c.vertices.length)))fail('Invalid collision hull.');}
 d.frames.forEach((f,i)=>{if(!finite(f.time_s)||Math.abs(f.time_s-i/60)>1e-9||!Array.isArray(f.poses_xyz_xyzw)||f.poses_xyz_xyzw.length!==3||!f.poses_xyz_xyzw.every(p=>vec(p,7)&&Math.abs(Math.hypot(...p.slice(3))-1)<1e-4)||!Array.isArray(f.velocities)||f.velocities.length!==3||!f.velocities.every(v=>vec(v,6))||!Number.isInteger(f.contact_count)||f.contact_count<0||f.contact_count>256||!Array.isArray(f.contact_points_xyz)||f.contact_points_xyz.length!==f.contact_count||!f.contact_points_xyz.every(v=>vec(v,3)))fail('Invalid saved Newton state.');});
 return d;
}
function at(d,t){if(!finite(t))fail('Finite replay time required.');t=Math.min(4,Math.max(0,t));const index=Math.min(240,Math.floor(t*60));return d.frames[index];}
const api={validate,at};root.PhysicsBenchContract=api;if(typeof module!=='undefined')module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
