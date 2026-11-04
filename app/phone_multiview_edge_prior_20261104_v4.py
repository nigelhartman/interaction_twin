"""Private automatic multi-view phone edge prior and projection diagnostics.

Dimensions derive from exact saved source masks/blue pixels, never authored3D
coordinates. Target development masks never fit geometry, poses or reference
selection. Original source prism, native assets, public scenes and UI stay exact.
"""
from datetime import datetime,timezone
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import ConvexHull
import trimesh

import phone_source_regeneration_20261104_v1 as P
import phone_source_prism_20261104_v1 as G
import phone_source_prism_evaluation_20261104_v1 as E
import phone_source_prism_ray_tracking_20261104_v2 as R
import sam3d_common_scene as S
from native_appearance import load_native_glb,render_native_mesh

ROOT=P.ROOT
OUTPUT=ROOT/'output/submission-polish-20261104-v1/phone-edge-prior-v4'
ASSET_OUTPUT=ROOT/'data/geometry-quality-20261104/phone-multiview-edge-prior-v4'
EDGE=[18,39,12,13]
TRAIN=[10,12,13,35,37,39,46]


def image_panel(sheet,pixels,mask,x,y,w,h,label):
    yy,xx=np.nonzero(mask);padding=15
    crop=[max(0,int(xx.min())-padding),max(0,int(yy.min())-padding),min(pixels.shape[1],int(xx.max())+padding+1),min(pixels.shape[0],int(yy.max())+padding+1)]
    part=Image.fromarray(pixels).crop(crop);scale=min((w-20)/part.width,(h-42)/part.height)
    part=part.resize((max(1,round(part.width*scale)),max(1,round(part.height*scale))),Image.Resampling.LANCZOS)
    sheet.paste(part,(x+(w-part.width)//2,y+35+(h-42-part.height)//2))
    ImageDraw.Draw(sheet).text((x+8,y+8),label,fill='#162139')
    return crop


def build():
    OUTPUT.mkdir(parents=True,exist_ok=False);ASSET_OUTPUT.mkdir(parents=True,exist_ok=False)
    base=json.loads(E.BASE.read_text());k=np.asarray(base['camera']['K'])
    sk=k.copy();sk[0]*=.3;sk[1]*=.3;sk[0,2]+=(.3-1)/2;sk[1,2]+=(.3-1)/2
    geometry=json.loads((G.OUTPUT/'report.json').read_text())
    oldmesh=load_native_glb(G.OUTPUT/'asset.glb')
    unit=json.loads((G.OUTPUT/'full-scene-v1/unit-gauge-poses.json').read_text())
    program=json.loads((P.CASE/'flow-procedure-1/interaction.json').read_text())
    proposal={'format':'interaction-twin-private-phone-multiview-edge-plan','version':4,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'implementation':P.ref(Path(__file__)),
        'original_source_prism':P.ref(G.OUTPUT/'asset.glb'),'original_geometry_report':P.ref(G.OUTPUT/'report.json'),
        'Muse_visual_review':P.ref(ROOT/'output/submission-polish-20261104-v1/phone-review/attempt/result.json'),
        'edge_source_indices':EDGE,'shape_rule':'PCA central60percent cross-sections;blue source pixels within exact SAM mask;median of lowest20percent widths/phone_length across four views',
        'source_colour_rule':{'HSV_h':[85,125],'minimum_saturation':60,'minimum_value':45},
        'geometry_change':'only canonicalZ scale from data-derived thickness;allXY,faces,colours preserved',
        'source_training_pose_indices':TRAIN,'pose_rule':'four proper canonical sign flips plus all-axis automatic local silhouette refit;select by RGB error under mask-IoU floor;no manuallychosenface',
        'development_target_masks_used_for_geometry_or_pose_fit':False,
        'development_target_reference_selection':'frozen_RGB_feature_prism_ray_PnP_confidence_before_mask_read',
        'minimum_mask_IoU':.65,'maximum_baseline_regression':.05,
        'new_API_or_GPU_calls':0,'automatic_publication':False,'physical_accuracy_verified':False}
    P.save(OUTPUT/'selection-plan.json',proposal)
    observations=[];widths=[]
    for index in EDGE:
        rgb,mask,ref=G.frame(index);center,rotation,length,aspect=G.basis(mask)
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        blue=mask&(hsv[:,:,0]>=85)&(hsv[:,:,0]<=125)&(hsv[:,:,1]>=60)&(hsv[:,:,2]>=45)
        yy,xx=np.nonzero(blue);coordinates=(np.column_stack([xx,yy])-center)@rotation/length
        sections=[]
        for lo in np.linspace(-.3,.25,12):
            points=coordinates[(coordinates[:,1]>=lo)&(coordinates[:,1]<lo+.05),0]
            if len(points)<4:continue
            width=float(np.percentile(points,95)-np.percentile(points,5))
            if width<=0:continue
            sections.append({'longitudinal_bin':[float(lo),float(lo+.05)],'width_over_length':width,'observed_blue_pixels':len(points)})
            widths.append({'source_index':index,'width_over_length':width})
        observations.append({'source':ref,'PCA_mask_aspect':aspect,'blue_pixels':int(blue.sum()),'sections':sections})
    ordered=sorted(widths,key=lambda w:w['width_over_length']);count=max(3,int(np.ceil(len(ordered)*.2)))
    lower=ordered[:count];thickness=float(np.median([w['width_over_length']for w in lower]))
    oldthickness=geometry['estimated_edge_thickness_over_length'];factor=thickness/oldthickness
    if not .1<factor<2:raise ValueError('Source prior changed outside a bounded relative envelope; no model created')
    mesh=trimesh.load(G.OUTPUT/'asset.glb',force='mesh',process=False);original_vertices=mesh.vertices.copy();original_faces=mesh.faces.copy();original_colors=mesh.visual.vertex_colors.copy()
    mesh.vertices[:,2]*=factor
    path=ASSET_OUTPUT/'asset.glb';path.write_bytes(mesh.export(file_type='glb'))
    loaded=trimesh.load(path,force='mesh',process=False)
    assert np.array_equal(loaded.faces,original_faces)and np.array_equal(loaded.visual.vertex_colors,original_colors)
    assert np.allclose(loaded.vertices[:,:2],original_vertices[:,:2],atol=1e-7,rtol=0)
    assert loaded.is_watertight and len(loaded.split(only_watertight=False))==1
    prior={'format':'interaction-twin-automatic-multiview-phone-edge-prior','version':4,
        'plan':P.ref(OUTPUT/'selection-plan.json'),'source_observations':observations,
        'all_cross_section_widths':ordered,'selected_lower_envelope':lower,
        'original_thickness_over_length':oldthickness,'new_thickness_over_length':thickness,'Z_scale_factor':factor,
        'asset':P.ref(path),'original_asset':P.ref(G.OUTPUT/'asset.glb'),
        'all_XY_faces_and_source_vertex_RGB_preserved':True,'watertight':True,'components':1,
        'metric_thickness_verified':False,'physical_accuracy_verified':False,
        'limitations':['Lower observed blue widths are a view-dependent case silhouette prior, not measured physical thickness.',
            'Perspective, occlusion, source colour thresholds and noisy SAM boundaries can bias the estimate.',
            'This does not recover camera bump, port recess, bevels, hidden electronics or contact geometry.']}
    P.save(ASSET_OUTPUT/'derivation.json',prior);P.save(OUTPUT/'edge-prior.json',prior)
    candidate=load_native_glb(path);vertices=candidate.vertices[np.linspace(0,len(candidate.vertices)-1,2000).astype(int)]
    trained={};source_results=[]
    sheet=Image.new('RGB',(1200,400*len(TRAIN)),'#f5f7fc')
    for row,index in enumerate(TRAIN):
        rgb,mask,ref=G.frame(index);oldmatrix=np.asarray(unit[str(index)])
        target=cv2.resize(mask.astype(np.uint8),(216,288),interpolation=cv2.INTER_NEAREST)>0
        srgb=cv2.resize(rgb,(216,288),interpolation=cv2.INTER_AREA)
        def measure(which,m):
            rend=render_native_mesh(which,sk,m[:3,:3],m[:3,3],216,288)
            inter=target&rend['mask'];iou=float(inter.sum()/max(1,(target|rend['mask']).sum()))
            inside=cv2.erode(inter.astype(np.uint8),np.ones((3,3),np.uint8))>0
            error=float(np.abs(rend['rgb'].astype(float)-srgb.astype(float))[inside].mean()) if inside.any() else 255.
            return rend,iou,error
        before,beforeiou,beforeerror=measure(oldmesh,oldmatrix)
        fixed,fixediou,fixederror=measure(candidate,oldmatrix)
        options=[]
        for signs in [[1,1,1],[-1,1,-1],[1,-1,-1],[-1,-1,1]]:
            initial=oldmatrix[:3,:3]@np.diag(signs)
            fitted=S.estimate_pose(vertices,initial,1,k,mask,float(oldmatrix[2,3]),search=True)
            m=fitted['matrix'];rend,iou,error=measure(candidate,m)
            if iou>=.65 and iou>=beforeiou-.05:
                options.append({'matrix':m,'render':rend,'IoU':iou,'RGB_MAE':error,'signs':signs,'score':error/255+.25*(1-iou)})
        if options:
            chosen=min(options,key=lambda o:o['score']);trained[str(index)]=chosen['matrix'].tolist()
        else:chosen=None
        source_results.append({'source_index':index,'source':ref,'original_IoU':beforeiou,'original_RGB_MAE':beforeerror,
            'new_geometry_fixed_pose_IoU':fixediou,'new_geometry_fixed_pose_RGB_MAE':fixederror,
            'selected_training_IoU':chosen['IoU']if chosen else None,'selected_training_RGB_MAE':chosen['RGB_MAE']if chosen else None,
            'selected_signs':chosen['signs']if chosen else None,'source_training_mask_used':True,'mask_or_RGB_ground_truth':False})
        image_panel(sheet,rgb,mask,0,row*400,400,400,f'SOURCE{index} | training RGB')
        bp=np.full((288,216,3),[238,241,247],np.uint8);bp[before['mask']]=before['rgb'][before['mask']]
        image_panel(sheet,bp,before['mask'],400,row*400,400,400,f'Prior | IoU {beforeiou:.3f} | RGB {beforeerror:.1f}')
        if chosen:
            cp=np.full((288,216,3),[238,241,247],np.uint8);cp[chosen['render']['mask']]=chosen['render']['rgb'][chosen['render']['mask']]
            image_panel(sheet,cp,chosen['render']['mask'],800,row*400,400,400,f'Edge prior + auto face | IoU {chosen["IoU"]:.3f} | RGB {chosen["RGB_MAE"]:.1f}')
        print('Edge-prior training diagnostic',index,round(beforeiou,3),round(chosen['IoU'],3)if chosen else'unknown',flush=True)
    sheet.save(OUTPUT/'source-training-comparison.png')
    P.save(OUTPUT/'source-training-results.json',source_results);P.save(OUTPUT/'source-training-unit-poses.json',trained)
    tests=json.loads((G.OUTPUT/'new-disjoint-tests-v1/attempt/new-observations.json').read_text())
    planes=np.unique(np.round(ConvexHull(candidate.vertices).equations,10),axis=0)
    predictions=[]
    for item in tests:
        target=np.asarray(Image.open(ROOT/item['target_RGB']['path']).convert('RGB'));options=[]
        for index in [item['reference_source_index'],item['right_source_index']]:
            pose=trained.get(str(index),unit.get(str(index)))
            if pose is None:continue
            rgb,mask,sref=G.frame(index)
            tracked=R.track(rgb,target,mask,np.asarray(pose),k,planes)
            options.append({'source_index':index,'reference':sref,**tracked})
        known=[o for o in options if o['status']=='estimated']
        selected=max(known,key=lambda o:o['confidence_for_reference_selection'])if known else None
        predictions.append({'test_id':item['test_id'],'target_RGB':item['target_RGB'],'reference_options':options,'selected':selected})
    P.save(OUTPUT/'development-predictions-before-target-mask-reads.json',predictions)
    oldreport=json.loads((G.OUTPUT/'new-disjoint-tests-v1/report.json').read_text());old={r['test_id']:r for r in oldreport['results']}
    results=[]
    for item,pred in zip(tests,predictions):
        target=cv2.resize((np.asarray(Image.open(ROOT/item['target_mask']['path']))>0).astype(np.uint8),(216,288),interpolation=cv2.INTER_NEAREST)>0
        selected=pred['selected'];value=None
        if selected is not None:
            m=np.asarray(selected['matrix']);render=render_native_mesh(candidate,sk,m[:3,:3],m[:3,3],216,288)
            value=float((render['mask']&target).sum()/max(1,(render['mask']|target).sum()))
        baseline=old[item['test_id']]['baseline_full_triangle_IoU']
        results.append({'test_id':item['test_id'],'baseline_original_SAM_IoU':baseline,'previous_prism_IoU':old[item['test_id']]['candidate_full_triangle_IoU'],
            'new_edge_prior_IoU':value,'passes_unrelaxed_mask_gate':value is not None and value>=.65 and value>=baseline-.05,
            'target_mask_used_for_fit':False,'fresh_heldout':False})
    P.save(OUTPUT/'development-results.json',results)
    # Projection diagnostic only: source-observed connector and phone masks,
    # compared with each saved rendered placement. No port/contact anchor exists.
    projected=[];scene=json.loads((G.OUTPUT/'full-scene-v1/scene.json').read_text());bycomp={c['id']:c for c in scene['components']}
    plug=load_native_glb(ROOT/bycomp['cable_connector']['asset']['path'])
    for index in [35,37,39]:
        evidence=next(e for e in program['evidence']if e['source_index']==index)
        masks={o['track_id']:np.asarray(Image.open(ROOT/o['mask_path']))>0 for o in evidence['observations']}
        if not all(key in masks for key in ['phone_blue','cable_connector']):continue
        phonepixels=np.column_stack(np.nonzero(masks['phone_blue']))[:,::-1];plugpixels=np.column_stack(np.nonzero(masks['cable_connector']))[:,::-1]
        from scipy.spatial import cKDTree
        sourcegap=float(cKDTree(phonepixels).query(plugpixels)[0].min())
        matrix=np.asarray(trained.get(str(index),unit[str(index)]));pr=render_native_mesh(candidate,k,matrix[:3,:3],matrix[:3,3],720,960)
        pose=next(p for p in bycomp['cable_connector']['poses']if p['source_sample_index']==index)
        m=np.asarray(pose['camera_from_native_mesh']);scale=float(np.cbrt(np.linalg.det(m[:3,:3])))
        cr=render_native_mesh(plug,k,m[:3,:3]/scale,m[:3,3],720,960,scale=scale)
        # Convert phone to the same inherited display gauge before comparing.
        gauge=bycomp['phone_blue']['native_uniform_scale'];pm=matrix.copy();pm[:3,:]*=gauge
        pr=render_native_mesh(candidate,k,pm[:3,:3]/gauge,pm[:3,3],720,960,scale=gauge)
        pp=np.column_stack(np.nonzero(pr['mask']))[:,::-1];cp=np.column_stack(np.nonzero(cr['mask']))[:,::-1]
        gap=float(cKDTree(pp).query(cp)[0].min())if len(cp)else None
        projected.append({'source_index':index,'source_timestamp_s':evidence['timestamp_s'],
            'source_observed_mask_minimum_distance_px':sourcegap,'rendered_mask_minimum_distance_px':gap,
            'source_phone_bbox_wh':np.ptp(phonepixels,axis=0).tolist(),'source_connector_bbox_wh':np.ptp(plugpixels,axis=0).tolist(),
            'rendered_phone_bbox_wh':np.ptp(pp,axis=0).tolist(),'rendered_connector_bbox_wh':np.ptp(cp,axis=0).tolist()if len(cp)else None,
            'phone_connector_relative_metric_scale_verified':False,'port_frame':None,'contact_forced':False,'insertion_verified':False})
    P.save(OUTPUT/'connector-projection-diagnostic.json',projected)
    report={'format':'interaction-twin-private-phone-multiview-edge-review','version':4,
        'plan':P.ref(OUTPUT/'selection-plan.json'),'derivation':P.ref(ASSET_OUTPUT/'derivation.json'),'candidate':P.ref(path),
        'source_training_results':P.ref(OUTPUT/'source-training-results.json'),'source_training_comparison':P.ref(OUTPUT/'source-training-comparison.png'),
        'fitted_source_poses':P.ref(OUTPUT/'source-training-unit-poses.json'),
        'development_prediction_receipt':P.ref(OUTPUT/'development-predictions-before-target-mask-reads.json'),
        'development_results':P.ref(OUTPUT/'development-results.json'),'development_passing_mask_gates':sum(r['passes_unrelaxed_mask_gate']for r in results),
        'connector_projection_diagnostic':P.ref(OUTPUT/'connector-projection-diagnostic.json'),
        'original_thickness_over_length':oldthickness,'new_edge_prior_thickness_over_length':thickness,
        'physical_accuracy_verified':False,'fresh_heldout_validation':False,'new_API_or_GPU_calls':0,
        'public_assets_catalog_or_viewer_modified':False,'automatic_publication':False}
    P.save(OUTPUT/'report.json',report)
    return {'report':P.ref(OUTPUT/'report.json'),'new_thickness':thickness,'original_thickness':oldthickness,
        'development_passing_mask_gates':report['development_passing_mask_gates'],'results':results}


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
