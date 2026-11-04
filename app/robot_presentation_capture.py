"""Prepare a private, source-bound geometric presentation; never curate it.

The original typed planner and native SAM assets stay unchanged. A measured
derivative lid and explicitly authored containment anchors belong to this film
rehearsal only, with no grasp, seating, scale or physical simulation claim.
"""
from __future__ import annotations
import copy,json,math
from pathlib import Path
from portable_twin import decode_glb_bytes
from robot_task_geometry import actuation_descriptor
from scripts.serve_presentation_capture import checked
from marketing_sources import file_ref

ROOT=Path(__file__).resolve().parent

def adapter(baseline_path,report_path,*,root=ROOT):
    root=Path(root).resolve();baseline_ref=file_ref(baseline_path,root);report_ref=file_ref(report_path,root)
    baseline=json.loads(checked(baseline_ref,root));report=json.loads(checked(report_ref,root))
    if (report.get('format')!='interaction-twin-native-assembly-refinement'or report.get('version')!=1 or
        report.get('source_sha256')!=baseline['source_binding']['source_sha256']or report.get('physical_accuracy_verified')is not False or
        report.get('manual_geometry_or_poses_used')is not False):raise ValueError('An exact automatic estimated assembly refinement is required')
    from native_assembly_refinement_v1 import verify_report
    report=verify_report(report_ref,root)
    inventory={o['id']:o for o in baseline['objects']};rows={r['id']:r for r in report['assets']}
    if set(inventory)!=set(rows)or len(rows)!=len(report['assets']):raise ValueError('Refinement component inventory differs')
    for cid,row in rows.items():
        original=row['original_asset'];curated=inventory[cid]['asset_ref']
        if any(original[k]!=curated[k]for k in ['sha256','bytes']):raise ValueError('Refinement has a foreign original component')
        # The curated review stores an immutable copy of the inference GLB.
        # Distinct literal paths are accepted only with an exact byte join.
        if checked(original,root)!=checked(curated,root):raise ValueError('Original and curated mesh copies differ')
        checked(row['asset'],root)
    joint=report['joint'];original=next(j for j in baseline['joints']if j['id']==joint['id'])
    if any(joint[k]!=original[k]for k in ['parent_id','child_id']):raise ValueError('Refined joint identity differs')
    modified=[r['id']for r in rows.values()if r['geometry_modified']]
    if modified!=[joint['child_id']]:raise ValueError('Only the explicitly declared articulated child may be a derived mesh')
    refined=joint['refined'];required={'axis','parent_from_joint','joint_from_child_zero','parent_native_scale','child_native_scale','limits_rad','zero_reference','physical_accuracy_verified'}
    if not required<=set(refined):raise ValueError('Complete refined joint geometry is required')
    source={k:copy.deepcopy(refined[k]if k in refined else v)for k,v in original['source_joint'].items()}
    assets={cid:{'asset_ref':row['asset']}for cid,row in rows.items()}
    class Inputs:
        def _checked(self,ref,maximum):
            if ref not in [r['asset']for r in rows.values()]or ref['bytes']>maximum:raise ValueError('Actuation needs the exact displayed mesh')
            checked(ref,root);return root/ref['path']
        def _bytes(self,ref,maximum):self._checked(ref,maximum);return checked(ref,root)
    cue=actuation_descriptor({**original,'source_joint':source},assets,Inputs())
    parent=inventory[joint['parent_id']]
    if parent['initial_transform']['quaternion_xyzw']!=[0,0,0,1]:raise ValueError('This authored capture needs an upright source container')
    decoded=decode_glb_bytes(checked(rows[joint['parent_id']]['asset'],root));bounds=decoded['mesh_stats']['bounds']
    vertices=[v for p in decoded['primitives']for v in p['vertices']]
    support=report['rim_support']['parent_vertex_ids']
    if not isinstance(support,list)or len(support)<32 or any(type(i)is not int or not 0<=i<len(vertices)for i in support):raise ValueError('Measured native opening-rim support required')
    heights=sorted(vertices[i][1]for i in support);rim_y=heights[int(.05*(len(heights)-1))]
    scale=parent['display_max_dimension']/max(bounds[1][i]-bounds[0][i]for i in range(3))*parent['initial_transform']['uniform_scale'];center_y=(bounds[0][1]+bounds[1][1])/2
    rim_offset=(rim_y-center_y)*scale;bottom_offset=(bounds[0][1]-center_y)*scale;clearance=.004
    staged=[]
    for cid in baseline['goal']['component_ids']:
        obj=inventory[cid];attachment=next(a for a in baseline['attachments']if a['child_id']==cid)
        if attachment['parent_id']!=joint['parent_id']or obj['role']!='removable':raise ValueError('Contained staging needs an existing removable relationship')
        native=decode_glb_bytes(checked(rows[cid]['asset'],root))['mesh_stats']['bounds']
        half=(native[1][1]-native[0][1])/2*obj['display_max_dimension']/max(native[1][i]-native[0][i]for i in range(3))*obj['initial_transform']['uniform_scale']
        offset=attachment['offset'][:];offset[1]=rim_offset-clearance-half
        if offset[1]-half<bottom_offset+clearance:raise ValueError('The displayed component does not fit the bounded container height')
        staged.append({'id':cid,'parent_id':attachment['parent_id'],'offset':offset,'display_half_height':half,
            'opening_rim_offset_y':rim_offset,'body_bottom_offset_y':bottom_offset,'clearance_display_units':clearance,
            'method':'native_measured_opening_rim_lower_5pct_then_bbox_containment_authored_staging','physical_seating_verified':False})
    return{'format':'interaction-twin-robot-presentation-refinement','version':1,'baseline_task_ref':baseline_ref,'refinement_report_ref':report_ref,
        'refined_scene_ref':report['scene_ref'],'source_asset_refs':[{'id':o['id'],'asset_ref':o['asset_ref']}for o in baseline['objects']],
        'render_asset_overrides':[{'id':joint['child_id'],'asset_ref':rows[joint['child_id']]['asset'],'scope':'mesh_derived_geometric_correction'}],
        'joint_id':joint['id'],'parent_id':joint['parent_id'],'child_id':joint['child_id'],'source_joint':source,'closed_q_rad':0,
        'open_q_rad':max(source['limits_rad'][0],min(source['limits_rad'][1],next(a['value']for a in baseline['actions']if a['operation']=='set_joint'and a['value']>0))),
        'actuation':cue,'contained_staging':staged,'physical_accuracy_verified':False,'new_model_calls':0,
        'claims':{'curated_original_task_unchanged':True,'derived_lid_geometry_modified':True,'authored_robot_staging':True,'source_alignment_revalidated':False,'contact_or_grasp_verified':False}}
