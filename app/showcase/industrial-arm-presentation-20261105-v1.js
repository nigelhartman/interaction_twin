/* Authored industrial visual rig. Exact existing five-DOF axes and lengths.
 * Appearance only: no manufacturer, hardware or contact-validation claim. */
import * as THREE from 'three';

export const INDUSTRIAL_ARM_GEOMETRY = Object.freeze({
  version: 1, tool_endpoint_local_x: .14,
  finger_size: [.09,.008,.014], finger_center_x: .095,
  pad_size: [.020,.006,.018], pad_center_x: .138,
  pad_inward_offset: .002, jaw_center_separation_minus_inner_gap: .010,
  palm_size: [.055,.26,.065], palm_center_x: .035,
  authored_display_only: true, hardware_identity: null,
});

export function createIndustrialArm(description, options={}) {
  const r=description;
  if(!r || r.link_lengths?.length!==2 || Math.abs(r.tool_length-.14)>1e-9)
    throw Error('The existing five-DOF .14-tool rig is required.');
  const steel=new THREE.MeshStandardMaterial({color:'#bcc9d7',metalness:.72,roughness:.28});
  const shell=new THREE.MeshStandardMaterial({color:'#dce3eb',metalness:.43,roughness:.29});
  const graphite=new THREE.MeshStandardMaterial({color:'#243246',metalness:.57,roughness:.33});
  const dark=new THREE.MeshStandardMaterial({color:'#101c2b',metalness:.35,roughness:.42});
  const blue=new THREE.MeshStandardMaterial({color:options.accent||'#3766ed',metalness:.28,roughness:.29});
  const rubber=new THREE.MeshStandardMaterial({color:'#344457',metalness:.04,roughness:.83});
  const geometries=[],materials=[steel,shell,graphite,dark,blue,rubber];
  function mesh(g,m,parent,x=0,y=0,z=0){geometries.push(g);const o=new THREE.Mesh(g,m);o.position.set(x,y,z);o.castShadow=true;o.receiveShadow=true;parent.add(o);return o;}
  function box(parent,size,m,at=[0,0,0]){return mesh(new THREE.BoxGeometry(...size),m,parent,...at);}
  function cylinder(parent,radius,length,m,at=[0,0,0],axis='y',segments=40){
    const o=mesh(new THREE.CylinderGeometry(radius,radius,length,segments),m,parent,...at);
    if(axis==='z')o.rotation.x=Math.PI/2;
    if(axis==='x')o.rotation.z=-Math.PI/2;
    return o;
  }
  function bolt(parent,at,axis='z',radius=.007){
    cylinder(parent,radius*1.3,.0025,graphite,at,axis,20);
    const shifted=at.slice(),k={x:0,y:1,z:2}[axis];shifted[k]+=.003;
    cylinder(parent,radius,.006,steel,shifted,axis,6);
  }
  function bearing(parent,radius,width){
    cylinder(parent,radius,width,graphite,[0,0,0],'z');
    for(const sign of[-1,1]){
      cylinder(parent,radius*.91,.009,steel,[0,0,sign*(width/2+.002)],'z');
      cylinder(parent,radius*.69,.012,blue,[0,0,sign*(width/2+.008)],'z');
      cylinder(parent,radius*.34,.014,dark,[0,0,sign*(width/2+.015)],'z');
      if(sign===1)for(let i=0;i<6;i++){
        const a=i*Math.PI/3;bolt(parent,[Math.cos(a)*radius*.80,Math.sin(a)*radius*.80,width/2+.008],'z',radius*.065);
      }
    }
  }
  function link(parent,length,width,depth){
    const shape=new THREE.Shape(),a=.045,b=length-.045;
    shape.moveTo(a,-width*.49);shape.lineTo(b,-width*.34);
    shape.quadraticCurveTo(length-.018,0,b,width*.34);
    shape.lineTo(a,width*.49);shape.quadraticCurveTo(.018,0,a,-width*.49);
    const body=mesh(new THREE.ExtrudeGeometry(shape,{depth,bevelEnabled:true,steps:1,bevelSegments:3,bevelSize:.006,bevelThickness:.005,curveSegments:12}),shell,parent,0,0,-depth/2);
    body.userData.industrial_shell=true;
    for(const sign of[-1,1]){
      box(parent,[length*.55,width*.40,.006],graphite,[length*.49,0,sign*(depth/2+.006)]);
      box(parent,[length*.42,.012,.008],blue,[length*.48,width*.24,sign*(depth/2+.007)]);
      for(const x of[length*.24,length*.72])bolt(parent,[x,-width*.10,sign*(depth/2+.010)],'z',.005);
    }
    // Narrow underside service rail gives the link a mechanical silhouette.
    box(parent,[length*.53,.014,depth*.55],dark,[length*.50,-width*.46,0]);
  }

  const root=new THREE.Group();root.name='authored_industrial_five_dof_arm';root.position.fromArray(r.base);
  cylinder(root,.177,.035,graphite,[0,.0175,0]);
  cylinder(root,.158,.079,shell,[0,.071,0]);
  cylinder(root,.162,.012,steel,[0,.116,0]);
  cylinder(root,.151,.020,blue,[0,.132,0]);
  for(let i=0;i<6;i++){const a=i*Math.PI/3;bolt(root,[Math.cos(a)*.148,.039,Math.sin(a)*.148],'y',.009);}
  const height=r.shoulder_height-.151;
  mesh(new THREE.CylinderGeometry(.082,.108,height,40),shell,root,0,.151+height/2,0);
  cylinder(root,.086,.028,graphite,[0,r.shoulder_height-.015,0]);
  box(root,[.075,height*.46,.012],graphite,[0,.151+height*.46,.097]);
  for(let i=0;i<3;i++)box(root,[.048,.004,.014],steel,[0,.151+height*.36+i*.016,.106]);

  const yaw=new THREE.Group();yaw.position.y=r.shoulder_height;root.add(yaw);
  const shoulder=new THREE.Group();yaw.add(shoulder);bearing(shoulder,.092,.125);
  link(shoulder,r.link_lengths[0],.122,.103);
  const elbow=new THREE.Group();elbow.position.x=r.link_lengths[0];shoulder.add(elbow);bearing(elbow,.078,.110);
  link(elbow,r.link_lengths[1],.100,.083);
  const wrist=new THREE.Group();wrist.position.x=r.link_lengths[1];elbow.add(wrist);bearing(wrist,.054,.083);
  const roll=new THREE.Group();wrist.add(roll);
  cylinder(roll,.047,.025,graphite,[.013,0,0],'x');
  cylinder(roll,.043,.008,steel,[.029,0,0],'x');
  cylinder(roll,.039,.010,blue,[.036,0,0],'x');

  // The exact collision-relevant adapter matches the separately checked grasp.
  const palm=box(roll,INDUSTRIAL_ARM_GEOMETRY.palm_size,graphite,[.035,0,0]);
  const palmCover=box(roll,[.036,.222,.006],shell,[.035,0,.0355]);
  for(const y of[-.105,.105])bolt(roll,[.035,y,.040],'z',.005);
  box(roll,[.018,.09,.007],blue,[.035,0,.040]);
  const fingers=[];
  for(const sign of[-1,1]){
    const group=new THREE.Group();group.userData.sign=sign;group.name=sign<0?'jaw_negative':'jaw_positive';roll.add(group);
    const finger=box(group,INDUSTRIAL_ARM_GEOMETRY.finger_size,steel,[.095,0,0]);
    const pad=box(group,INDUSTRIAL_ARM_GEOMETRY.pad_size,rubber,[.138,-sign*.002,0]);
    group.userData.finger=finger;group.userData.pad=pad;fingers.push(group);
  }
  function setAperture(fullInnerPadGap){
    if(!Number.isFinite(fullInnerPadGap)||fullInnerPadGap<0||fullInnerPadGap>.24)
      throw Error('Bounded authored jaw opening required.');
    const centers=fullInnerPadGap+INDUSTRIAL_ARM_GEOMETRY.jaw_center_separation_minus_inner_gap;
    for(const finger of fingers)finger.position.y=finger.userData.sign*centers/2;
    root.userData.full_inner_pad_gap=fullInnerPadGap;
  }
  setAperture(.11);
  const joints=[yaw,shoulder,elbow,wrist,roll];
  function applyJoints(q){
    if(!Array.isArray(q)||q.length!==5||q.some(v=>!Number.isFinite(v)))throw Error('Five finite joints required.');
    yaw.rotation.y=-q[0];shoulder.rotation.z=q[1];elbow.rotation.z=q[2];wrist.rotation.z=q[3];roll.rotation.x=q[4];
  }
  return {root,joints,fingers,palm,palmCover,setAperture,applyJoints,
    geometry_parameters:INDUSTRIAL_ARM_GEOMETRY,
    dispose(){for(const g of geometries)g.dispose();for(const m of materials)m.dispose();}};
}
