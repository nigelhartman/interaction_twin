"""Authored V3 B placement against actual staged A/B/case/lid surfaces."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
import prepare_case_grasp_capture_20261105_v2 as S

def main():
    folder=S.ROOT/'output/trailer-rework-20261105-v3/case-B-orientation';binding=json.loads((folder/'candidate-binding.json').read_text());p=binding['candidate_typed_plan'];objects={o['id']:o for o in p['objects']};geo=S.native_world(p);av,af=geo['earbud_a'];ap=np.array(objects['earbud_a']['initial_transform']['position']);bp=np.array(objects['earbud_b']['initial_transform']['position']);c=binding['display_correction'];geo['earbud_b']=((av-ap)*c['world_relative_multiplier']+bp+c['display_offset'],af);lid=objects['case_lid'];native_lid,lid_faces=S.mesh(lid);j=p['joints'][0]['source_joint'];lo,hi=np.array(p['actions'][13]['actuation']['parent_native_bounds']);scale=.38/max(hi-lo);source=json.loads((folder/'B-placement-factory-boxes.json').read_text());rows=[]
    for pose in source['poses']:
        state=pose['objects'];q=pose['object_joints']['case_base:'+p['joints'][0]['id']];r=np.eye(4);r[:3,:3]=Rotation.from_rotvec(np.array(j['axis'])*q).as_matrix();m=np.array(j['parent_from_joint'])@r@np.array(j['joint_from_child_zero']);world_lid=(S.trimesh.transform_points(native_lid,m)-(lo+hi)/2)*scale+state['case_base']['position'];obstacles={'case_lid':world_lid[lid_faces]}
        for id in ['case_base','earbud_a','earbud_b']:
            v,f=geo[id];delta=np.array(state[id]['position'])-objects[id]['initial_transform']['position'];obstacles[id]=(v+delta)[f]
        counts={}
        for id,triangles in obstacles.items():
            counts[id]=[]
            for box in pose['gripper_mesh_boxes']:
                low,high=np.array(box['world_bounds']);count=S.triangle_box_count(triangles,(low+high)/2,(high-low)/2);counts[id].append({'mesh_index':box['index'],'count':count})
        rows.append({'label':pose['label'],'task_runtime_s':pose['task_runtime_s'],'actual_A_goal':state['earbud_a']['position'],'attached':pose['attached'],'gripper':pose['gripper'],'surface_tests':counts})
    total=sum(len(cs)for r in rows for cs in r['surface_tests'].values());nonzero=[{'state':r['label'],'obstacle':id,**c}for r in rows for id,cs in r['surface_tests'].items()for c in cs if c['count']]
    report={'candidate_binding':S.ref(folder/'candidate-binding.json'),'proxy_transform_method':binding['display_correction']['method'],'proxy_world_relative_multiplier':binding['display_correction']['world_relative_multiplier'],'format':'interaction-twin-B-placement-relative-obstacle-check','version':3,'factory_sha256':source['factory_sha256'],'candidate_plan_sha256':source['candidate_plan_sha256'],'states':len(rows),'actual_factory_meshes_per_state':len(source['poses'][0]['gripper_mesh_boxes']),'triangle_vs_mesh_world_box_tests':total,'nonzero':nonzero,'results':rows,'method':'All33 actual frozen factory gripper/wrist mesh bounds vs exact staged native A/B/case/open-lid triangle surfaces at B descent, before release, detachment, opening and retreat. Conservative mesh world boxes, not a full continuous collision or force simulation.','physical_contact_verified':False};out=folder/'B-placement-relative-obstacle-check.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k]for k in ['states','triangle_vs_mesh_world_box_tests','nonzero']},indent=2))

if __name__=='__main__':main()
