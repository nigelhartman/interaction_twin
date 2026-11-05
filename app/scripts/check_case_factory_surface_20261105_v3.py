"""Independent actual-factory conservative shape-vs-native-surface check."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
import prepare_case_grasp_capture_20261105_v2 as S

def main():
    folder=S.ROOT/'output/trailer-rework-20261105-v3/case-B-orientation';binding=json.loads((folder/'candidate-binding.json').read_text());plan=binding['candidate_typed_plan'];objects={o['id']:o for o in plan['objects']};geo=S.native_world(plan); av,af=geo['earbud_a'];ap=np.array(objects['earbud_a']['initial_transform']['position']);bp=np.array(objects['earbud_b']['initial_transform']['position']);c=binding['display_correction'];geo['earbud_b']=((av-ap)*c['world_relative_multiplier']+bp+c['display_offset'],af);d=json.loads((folder/'industrial-factory-critical-boxes.json').read_text());lid=objects['case_lid'];v,f=S.mesh(lid);j=plan['joints'][0]['source_joint'];r=np.eye(4);r[:3,:3]=Rotation.from_rotvec(np.array(j['axis'])*1.8).as_matrix();m=np.array(j['parent_from_joint'])@r@np.array(j['joint_from_child_zero']);lo,hi=np.array(plan['actions'][13]['actuation']['parent_native_bounds']);scale=.38/max(hi-lo);opened=(S.trimesh.transform_points(v,m)-(lo+hi)/2)*scale+objects['case_base']['initial_transform']['position'];results=[];delta=np.array([.35,0,.25]);scenes={}
    for id in ['case_base','earbud_a','earbud_b']:
        if id=='case_base':scene=np.concatenate([v[f]for v,f in geo.values()])
        else:
            names=['case_base',id]+(['earbud_b']if id=='earbud_a'else[]);arrays=[geo[n][0][geo[n][1]]+delta for n in names]+[opened[f]+delta];scene=np.concatenate(arrays)
        scenes[id]=scene
    for pose in d['poses']:
        row={'object_id':pose['object_id'],'closure':pose['closure'],'mesh_box_surface_intersections':[]}
        for box in pose['gripper_mesh_boxes']:
            low,high=np.array(box['world_bounds']);count=S.triangle_box_count(scenes[pose['object_id']],(low+high)/2,(high-low)/2);row['mesh_box_surface_intersections'].append({'index':box['index'],'geometry_type':box['geometry_type'],'count':count})
        results.append(row)
    report={'candidate_binding':S.ref(folder/'candidate-binding.json'),'proxy_transform_method':binding['display_correction']['method'],'proxy_world_relative_multiplier':binding['display_correction']['world_relative_multiplier'],'format':'interaction-twin-industrial-factory-surface-check','version':3,'factory':d['factory'],'actual_factory_meshes_per_pose':len(d['poses'][0]['gripper_mesh_boxes']),'closure_cohorts':[0,.25,.5,.75,1],'conservative_world_AABB_surface_tests':sum(len(r['mesh_box_surface_intersections'])for r in results),'nonzero_surface_test_count':sum(c['count']>0 for r in results for c in r['mesh_box_surface_intersections']),'results':results,'method':'Actual factory geometry-local mesh boxes transformed by source-bound IK and locked-roll poses, tested against exact staged native triangle surfaces. Conservative boxes may reject even when rounded solids clear. Not force/friction/contact simulation.','physical_contact_verified':False};p=folder/'industrial-factory-surface-check.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k]for k in ['factory','actual_factory_meshes_per_pose','conservative_world_AABB_surface_tests','nonzero_surface_test_count']},indent=2));

if __name__=='__main__':main()
