/* A fixed perspective view fitted to the complete finite motion envelope. */
(function(root){'use strict';
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),sub=(a,b)=>a.map((v,i)=>v-b[i]),add=(a,b)=>a.map((v,i)=>v+b[i]),scale=(a,n)=>a.map(v=>v*n),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],unit=a=>scale(a,1/Math.hypot(...a));
function fit(points,{vertical_fov_deg=36,aspect=16/9,view_direction=[1.2,.77,1.26],edge_margin_fraction=.07}={}){
 if(!Array.isArray(points)||!points.length||points.length>100000||points.some(p=>!Array.isArray(p)||p.length!==3||p.some(v=>!Number.isFinite(v)))||!(vertical_fov_deg>10&&vertical_fov_deg<100)||!(aspect>.5&&aspect<4)||!(edge_margin_fraction>=.02&&edge_margin_fraction<=.2)||view_direction.length!==3||!view_direction.every(Number.isFinite)||Math.hypot(...view_direction)<1e-9)throw Error('A bounded finite camera envelope is required.');
 const direction=unit(view_direction),right=unit(cross([0,1,0],direction)),up=cross(direction,right);
 if(!right.every(Number.isFinite))throw Error('The capture view cannot be parallel to world up.');
 const minimum=[0,1,2].map(i=>Math.min(...points.map(p=>p[i]))),maximum=[0,1,2].map(i=>Math.max(...points.map(p=>p[i])));let target=minimum.map((v,i)=>(v+maximum[i])/2);
 const xs=points.map(p=>dot(sub(p,target),right)),ys=points.map(p=>dot(sub(p,target),up));target=add(target,add(scale(right,(Math.min(...xs)+Math.max(...xs))/2),scale(up,(Math.min(...ys)+Math.max(...ys))/2)));
 const tangent=Math.tan(vertical_fov_deg*Math.PI/360),limit=1-2*edge_margin_fraction;let distance=.2;
 for(const p of points){const v=sub(p,target),depth=dot(v,direction);distance=Math.max(distance,depth+.05,depth+Math.abs(dot(v,right))/(tangent*aspect*limit),depth+Math.abs(dot(v,up))/(tangent*limit));}
 const position=add(target,scale(direction,distance)),project=p=>{const v=sub(p,target),depth=distance-dot(v,direction);return[dot(v,right)/(depth*tangent*aspect),dot(v,up)/(depth*tangent),depth];},projected=points.map(project),ndc_min=[0,1].map(i=>Math.min(...projected.map(p=>p[i]))),ndc_max=[0,1].map(i=>Math.max(...projected.map(p=>p[i])));
 if([...ndc_min,...ndc_max].some(v=>Math.abs(v)>limit+1e-9))throw Error('The full capture envelope does not fit.');
 return{position,target,vertical_fov_deg,aspect,near:.01,far:Math.max(20,distance+maximum.map((v,i)=>v-minimum[i]).reduce((s,v)=>s+Math.abs(v),0)+2),edge_margin_fraction,
  proof:{sampled_points:points.length,world_min:minimum,world_max:maximum,ndc_min,ndc_max,all_points_inside:true},project};
}
const api=Object.freeze({fit});root.RobotCameraFitV1=api;if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(globalThis);
