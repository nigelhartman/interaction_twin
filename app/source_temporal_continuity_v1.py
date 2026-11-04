"""Source-checked display segments; no pose, geometry or joint is edited."""
from pathlib import Path
import copy
import math
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

import source_temporal_recovery_v1 as T
from native_appearance import load_native_glb
import motion_hypothesis_contract_v1 as C

LIMITS={'maximum_adjacent_rotation_rad':math.radians(12),'maximum_adjacent_depth_ratio':1.12,
    'maximum_source_relative_center_delta_error':.12,'maximum_adjacent_scale_ratio':1.000001,
    'maximum_interpolation_gap_s':.22}


def normalize_segments(hypothesis_ref,output,*,root=T.ROOT):
    root=Path(root).resolve();out=root/output;T.require(not out.exists(),'New continuity derivative required');out.mkdir(parents=True)
    implementation=T.ref(Path(__file__),root);d=T.read_checked(hypothesis_ref,root);scene=T.read_checked(d['baseline_scene_ref'],root)
    parts={p['id']:p for p in scene['components']};sam_rows={}
    if d.get('source_SAM_temporal_observations_ref'):
        sam=T.read_checked(d['source_SAM_temporal_observations_ref'],root)
        for row in sam['rows']:
            if row['status']=='estimated':
                mask=np.asarray(Image.open(root/row['consensus_mask_ref']['path']))>0;ys,xs=np.nonzero(mask)
                sam_rows[(sam['episode_id'],round(row['timestamp_s'],6))]={'mask_ref':row['consensus_mask_ref'],
                    'centroid_px':[float(xs.mean()),float(ys.mean())],'diagonal_px':float(math.hypot(xs.max()-xs.min(),ys.max()-ys.min()))}
    k=np.asarray(d['camera']['K']);reports=[]
    for group in d['groups']:
        if not group['episodes']:continue
        cid=group['root_component_id'];mesh=load_native_glb(root/parts[cid]['asset']['path']);low=mesh.vertices.min(0);high=mesh.vertices.max(0);center=(low+high)/2
        probes=np.array([[x,y,z]for x in [low[0],center[0],high[0]]for y in [low[1],center[1],high[1]]for z in [low[2],center[2],high[2]]])
        for episode in group['episodes']:
            if not episode['predictions']or episode.get('view_change_break'):continue
            nodes=[{'timestamp_s':episode['anchor_before']['timestamp_s'],'matrix':episode['anchor_before']['camera_from_native_mesh'],'kind':'anchor_before'}]
            nodes += [{'timestamp_s':p['timestamp_s'],'matrix':p['component_poses'].get(cid)if p['status']=='estimated'else None,'kind':'prediction','prediction':p}for p in episode['predictions']]
            nodes += [{'timestamp_s':episode['anchor_after']['timestamp_s'],'matrix':episode['anchor_after']['camera_from_native_mesh'],'kind':'anchor_after'}]
            edges=[];counter=0;nodes[0]['segment_id']=episode['id']+':continuous-0'
            for a,b in zip(nodes,nodes[1:]):
                edge={'from_s':a['timestamp_s'],'to_s':b['timestamp_s'],'passed':False,'source_delta_basis':'individually_verified_RGB_pixel_correspondences'}
                if a['matrix']is None or b['matrix']is None:
                    edge['reason']='unknown_motion_barrier';counter+=1
                else:
                    ra,ta,sa=T.matrix_parts(a['matrix']);rb,tb,sb=T.matrix_parts(b['matrix'])
                    angle=float(Rotation.from_matrix(rb@ra.T).magnitude());depth=float(max(ta[2]/tb[2],tb[2]/ta[2]));scale=float(max(sa/sb,sb/sa))
                    edge.update(rotation_rad=angle,depth_ratio=depth,scale_ratio=scale,source_relative_center_delta_error=None)
                    observed_a=sam_rows.get((episode['id'],round(a['timestamp_s'],6)));observed_b=sam_rows.get((episode['id'],round(b['timestamp_s'],6)))
                    if observed_a and observed_b:
                        ca=T.project(probes,a['matrix'],k).mean(0);cb=T.project(probes,b['matrix'],k).mean(0)
                        error=float(np.linalg.norm((cb-ca)-(np.asarray(observed_b['centroid_px'])-observed_a['centroid_px']))/max(1.,(observed_a['diagonal_px']+observed_b['diagonal_px'])/2))
                        edge.update(source_relative_center_delta_error=error,source_delta_basis='actual_bidirectional_SAM_source_mask_centroid_delta')
                    edge['passed']=bool(angle<=LIMITS['maximum_adjacent_rotation_rad']and depth<=LIMITS['maximum_adjacent_depth_ratio']and scale<=LIMITS['maximum_adjacent_scale_ratio']and b['timestamp_s']-a['timestamp_s']<=LIMITS['maximum_interpolation_gap_s']and(edge['source_relative_center_delta_error']is None or edge['source_relative_center_delta_error']<=LIMITS['maximum_source_relative_center_delta_error']))
                    edge['reason']=None if edge['passed']else'pose_or_source_delta_discontinuity'
                    if not edge['passed']:counter+=1
                b['segment_id']=episode['id']+f':continuous-{counter}'
                edges.append(edge)
            for node in nodes:
                if node.get('prediction')is not None and node['matrix']is not None:node['prediction']['view_segment_id']=node['segment_id']
            episode['anchor_before_view_segment_id']=nodes[0]['segment_id'];episode['anchor_after_view_segment_id']=nodes[-1]['segment_id']
            episode['display_continuity_policy']='interpolate_root_then_fixed_joint_FK_only_between_adjacent_same_checked_segment_including_saved_anchors'
            reports.append({'episode_id':episode['id'],'root_component_id':cid,'native_asset_ref':parts[cid]['asset'],
                'native_probe_points':probes.tolist(),'source_rows':[{'timestamp_s':n['timestamp_s'],**sam_rows[(episode['id'],round(n['timestamp_s'],6))]}for n in nodes if(episode['id'],round(n['timestamp_s'],6))in sam_rows],
                'nodes':[{'timestamp_s':n['timestamp_s'],'segment_id':n['segment_id'],'has_pose':n['matrix']is not None,'kind':n['kind']}for n in nodes],
                'edges':edges,'all_edges_pass':all(e['passed']for e in edges),'pose_or_joint_edits':False})
    report={'format':'interaction-twin-source-checked-display-continuity','version':1,'input_hypothesis_ref':hypothesis_ref,
        'source_ref':d['source_ref'],'evidence_program_ref':d['evidence_program_ref'],'baseline_scene_ref':d['baseline_scene_ref'],
        'limits':LIMITS,'episodes':reports,'source_original_unknowns_preserved':True,'poses_angles_geometry_or_source_masks_edited':False,'physical_accuracy_verified':False}
    T.write_new(out/'continuity-report.json',report)
    d['display_continuity_ref']=T.ref(out/'continuity-report.json',root);d['parent_motion_hypothesis_ref']=hypothesis_ref
    d['additional_implementation_fingerprints']=[*d.get('additional_implementation_fingerprints',[]),implementation]
    T.require(T.ref(Path(__file__),root)==implementation,'Executing continuity helper changed')
    C.validate(d,lambda r,maximum=None:(root/r['path']).read_bytes())
    T.write_new(out/'hypotheses.json',d)
    return {'hypotheses':T.ref(out/'hypotheses.json',root),'continuity':d['display_continuity_ref'],
        'episodes':[{'id':e['episode_id'],'all_edges_pass':e['all_edges_pass'],'failed_edges':sum(not x['passed']for x in e['edges'])}for e in reports]}
