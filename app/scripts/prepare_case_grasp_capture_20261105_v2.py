"""Additive authored grasp study. Original meshes/plans/captures remain intact."""
from pathlib import Path
import hashlib,json,math
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/trailer-rework-20261105-v2/case-grasp'
_TRIANGLE_BOUNDS_CACHE={}

def ref(p):
    p=Path(p);raw=p.read_bytes();return {'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}

def mesh(o):
    s=trimesh.load(ROOT/o['asset_ref']['path'],force='scene',process=False);v=[];f=[];n=0
    for key in s.graph.nodes_geometry:
        t,g=s.graph[key];m=s.geometry[g];p=trimesh.transform_points(m.vertices,t);v.append(p);f.append(m.faces+n);n+=len(p)
    return np.concatenate(v),np.concatenate(f)

def native_world(plan):
    out={};base=next(o for o in plan['objects']if o['id']=='case_base');lo,hi=np.array(next(a['actuation']['parent_native_bounds']for a in plan['actions']if a.get('actuation')));center=(lo+hi)/2;scale=.38/max(hi-lo)
    for o in plan['objects']:
        v,f=mesh(o);low,high=v.min(0),v.max(0)
        if o['id']=='case_lid':
            j=plan['joints'][0]['source_joint'];m=np.array(j['parent_from_joint'])@np.array(j['joint_from_child_zero']);v=trimesh.transform_points(v,m);v=(v-center)*scale+base['initial_transform']['position']
        elif o['id']=='case_base':v=(v-center)*scale+o['initial_transform']['position']
        else:v=Rotation.from_quat(o['initial_transform']['quaternion_xyzw']).apply((v-(low+high)/2)*o['display_max_dimension']/max(high-low))+o['initial_transform']['position']
        out[o['id']]=(v,f)
    return out

def triangle_box_count(triangles,center,half,epsilon=1e-8):
    """Exact separating-axis surface intersection; not solid/contact simulation."""
    lo=np.array(center)-half;hi=np.array(center)+half;key=id(triangles)
    if key not in _TRIANGLE_BOUNDS_CACHE:
        if len(_TRIANGLE_BOUNDS_CACHE)>=2:_TRIANGLE_BOUNDS_CACHE.clear()
        low,high=triangles.min(1),triangles.max(1);_TRIANGLE_BOUNDS_CACHE[key]=(triangles,low,high,low.min(0),high.max(0))
    _,low,high,minimum,maximum=_TRIANGLE_BOUNDS_CACHE[key]
    if np.any(minimum>hi)or np.any(maximum<lo):return 0
    t=triangles[(low<=hi).all(1)&(high>=lo).all(1)]-center
    if not len(t):return 0
    e=np.roll(t,-1,axis=1)-t;axes=[np.broadcast_to(a,(len(t),3))for a in np.eye(3)]
    axes.append(np.cross(e[:,0],e[:,1]))
    axes += [np.cross(edge,a)for edge in [e[:,i]for i in range(3)]for a in np.eye(3)]
    active=np.ones(len(t),dtype=bool)
    for axis in axes:
        proj=np.einsum('nki,ni->nk',t,axis);radius=np.abs(axis)@half
        active &= (proj.min(1)<=radius+epsilon)&(proj.max(1)>=-radius-epsilon)
    return int(active.sum())

def clipped_support(triangles,axis,perp,q,minimum_y):
    t=triangles[(triangles.max(1)[:,1]>=minimum_y)&(triangles.min(1)[:,perp]<=q+.009)&(triangles.max(1)[:,perp]>=q-.009)];points=[]
    for triangle in t:
        poly=list(triangle)
        for coordinate,bound,sign in [(1,minimum_y,1),(perp,q-.009,1),(perp,q+.009,-1)]:
            clipped=[]
            for a,b in zip(poly,poly[1:]+poly[:1]):
                da=sign*(a[coordinate]-bound);db=sign*(b[coordinate]-bound)
                if da>=0:clipped.append(a)
                if (da>=0)!=(db>=0):clipped.append(a+(b-a)*da/(da-db))
            poly=clipped
            if not poly:break
        points.extend(poly)
    points=np.array(points);return float(points[:,axis].min()),float(points[:,axis].max())

def grasp(v,object_position,triangles):
    anchor_y=float(v[:,1].max()-.010);upper=v[v[:,1]>=anchor_y-.008]
    spans=np.ptp(upper,axis=0);axis=0 if spans[0]<=spans[2] else 2;perp=2 if axis==0 else 0
    # Search central source-surface cross sections, using actual support extrema.
    options=[]
    for q in np.linspace(np.quantile(upper[:,perp],.25),np.quantile(upper[:,perp],.75),21):
        band=upper[np.abs(upper[:,perp]-q)<=.009]
        if len(band)<20:continue
        low,high=band[:,axis].min(),band[:,axis].max();options.append((high-low,q,(low+high)/2,low,high,len(band)))
    width,q,c,low,high,count=max(options);low,high=clipped_support(triangles,axis,perp,q,anchor_y-.008);width=high-low;c=(low+high)/2
    anchor=np.array(object_position,dtype=float);anchor[1]=anchor_y;anchor[axis]=c;anchor[perp]=q
    return {'jaw_axis_world':'X'if axis==0 else'Z','jaw_angle_world_rad':0 if axis==0 else math.pi/2,'anchor_initial_world':anchor.tolist(),'offset_from_object_origin':(anchor-object_position).tolist(),'support_width':float(width),'closed_jaw_center_separation':float(width+.0105),'open_jaw_center_separation':float(width+.0285),'surface_pad_clearance_each_side':.00025,'support_band_vertices':count,'support_band_y_min':anchor_y-.008,'support_perpendicular_half_width':.009,'source_surface_support_planes':[float(low),float(high)]}

def boxes(g,aperture):
    a=np.array(g['anchor_initial_world']);axis=0 if g['jaw_axis_world']=='X'else 2;perp=2 if axis==0 else 0;out=[]
    for sign in [-1,1]:
        c=a.copy();c[axis]+=sign*aperture/2;c[1]+=.045;half=np.zeros(3);half[axis]=.004;half[1]=.045;half[perp]=.007;out.append((f'finger_{sign}',c,half))
        c=a.copy();c[axis]+=sign*(aperture/2-.002);c[1]+=.002;half=np.zeros(3);half[axis]=.003;half[1]=.010;half[perp]=.009;out.append((f'pad_{sign}',c,half))
    c=a.copy();c[1]+=.105;half=np.zeros(3);half[axis]=.13;half[1]=.0275;half[perp]=.0325;out.append(('palm',c,half))
    c=a.copy();c[1]+=.14;out.append(('wrist_housing_bbox',c,np.array([.048,.048,.048])))
    return out

def main():
    prior=ROOT/'output/trailer-rework-20261105-v1/case-capture/fullscreen-case-14s.json';old=json.loads(prior.read_text());plan=old['typed_plan'];objects={o['id']:o for o in plan['objects']};geo=native_world(plan);params={}
    closed=np.concatenate([geo[k][0]for k in ['case_base','case_lid']]);closed_triangles=np.concatenate([geo[k][0][geo[k][1]]for k in ['case_base','case_lid']]);params['case_base']=grasp(closed,np.array(objects['case_base']['initial_transform']['position']),closed_triangles)
    for k in ['earbud_a','earbud_b']:params[k]=grasp(geo[k][0],np.array(objects[k]['initial_transform']['position']),geo[k][0][geo[k][1]])
    params['earbud_a'].update(cavity_open_jaw_center_separation=params['earbud_a']['closed_jaw_center_separation']+.0045,preshape_start_height_above_grasp=.10,preshape_complete_height_above_grasp=.05)
    report={'format':'interaction-twin-authored-native-surface-grasp-study','version':2,'units':'authored_scene_units_not_metric_calibration','original_capture':ref(prior),'mesh_assets':[o['asset_ref']for o in plan['objects']],'original_assets_modified':False,'new_model_calls':0,'physical_contact_or_grasp_force_validated':False,'finger_geometry':{'steel_size_local':[.09,.008,.014],'steel_center_local':[.095,0,0],'pad_size_local':[.020,.006,.018],'pad_center_local_x':.138,'pad_inward_offset':.002,'palm_size_local':[.055,.26,.065],'palm_center_local_x':.035},'grasp_parameters':params,'closed_object_surface_checks':{}}
    for k,g in params.items():
        tris=np.concatenate([geo[n][0][geo[n][1]]for n in (['case_base','case_lid']if k=='case_base'else[k])]);rows=[]
        for closure in [0,.25,.5,.75,1]:
            aperture=g['open_jaw_center_separation']*(1-closure)+g['closed_jaw_center_separation']*closure
            rows.append({'closure':closure,'jaw_center_separation':aperture,'triangle_intersections':{name:triangle_box_count(tris,c,h)for name,c,h in boxes(g,aperture)}})
        report['closed_object_surface_checks'][k]=rows
    # Check the real cavity/other component surfaces at the open source joint.
    lid=objects['case_lid'];v,f=mesh(lid);j=plan['joints'][0]['source_joint'];rotation=np.eye(4);rotation[:3,:3]=Rotation.from_rotvec(np.array(j['axis'])*1.8).as_matrix();m=np.array(j['parent_from_joint'])@rotation@np.array(j['joint_from_child_zero']);lo,hi=np.array(plan['actions'][13]['actuation']['parent_native_bounds']);scale=.38/max(hi-lo);opened=(trimesh.transform_points(v,m)-(lo+hi)/2)*scale+objects['case_base']['initial_transform']['position'];report['open_cavity_surface_checks']={}
    for k in ['earbud_a','earbud_b']:
        obstacles={'case_base':geo['case_base'][0][geo['case_base'][1]],'open_lid':opened[f]}
        if k=='earbud_a':obstacles['other_earbud']=geo['earbud_b'][0][geo['earbud_b'][1]]
        g=params[k];rows=[]
        for closure in [0,.25,.5,.75,1]:
            aperture=g.get('cavity_open_jaw_center_separation',g['open_jaw_center_separation'])*(1-closure)+g['closed_jaw_center_separation']*closure
            rows.append({'closure':closure,'obstacle_intersections':{n:{name:triangle_box_count(tris,c,h)for name,c,h in boxes(g,aperture)}for n,tris in obstacles.items()}})
        report['open_cavity_surface_checks'][k]=rows
        report.setdefault('open_cavity_approach_checks',{})[k]=[]
        for fraction in np.linspace(0,1,21):
            g2=dict(g);g2['anchor_initial_world']=g['anchor_initial_world'][:];g2['anchor_initial_world'][1]=.42*(1-fraction)+g['anchor_initial_world'][1]*fraction
            height=g2['anchor_initial_world'][1]-g['anchor_initial_world'][1];u=np.clip((.10-height)/.05,0,1)if k=='earbud_a'else 0;aperture=g['open_jaw_center_separation']*(1-u)+g.get('cavity_open_jaw_center_separation',g['open_jaw_center_separation'])*u
            report['open_cavity_approach_checks'][k].append({'fraction':float(fraction),'tool_y':g2['anchor_initial_world'][1],'jaw_center_separation':float(aperture),'obstacle_intersections':{n:{name:triangle_box_count(tris,c,h)for name,c,h in boxes(g2,aperture)}for n,tris in obstacles.items()}})
    report['lid_outward_cue_surface_checks']=[];report['lid_cue_outward_offset_scene_units']=.015
    native_lid,native_faces=mesh(lid);cue=np.array(plan['actions'][13]['actuation']['anchor_native']);base_position=np.array(objects['case_base']['initial_transform']['position'])
    for q in np.linspace(0,1.8,25):
        rotation=np.eye(4);rotation[:3,:3]=Rotation.from_rotvec(np.array(j['axis'])*q).as_matrix();m=np.array(j['parent_from_joint'])@rotation@np.array(j['joint_from_child_zero']);points=(trimesh.transform_points(native_lid,m)-(lo+hi)/2)*scale+base_position;point=(trimesh.transform_points(cue[None],m)[0]-(lo+hi)/2)*scale+base_position;n=m[:3,1];point+=n/np.linalg.norm(n)*.015;g={'anchor_initial_world':point.tolist(),'jaw_axis_world':'X'}
        combined=np.concatenate([points[native_faces],geo['case_base'][0][geo['case_base'][1]]]);report['lid_outward_cue_surface_checks'].append({'joint_angle_rad':float(q),'tool':point.tolist(),'triangle_intersections':{name:triangle_box_count(combined,c,h)for name,c,h in boxes(g,.085)}})
    OUT.mkdir(parents=True,exist_ok=True);p=OUT/'grasp-study.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
