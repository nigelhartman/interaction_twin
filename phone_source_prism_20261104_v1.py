"""Private automatic 2.5D phone hypothesis from real RGB/SAM observations.

This deliberately has a different backend from SAM3D. A source-mask convex
outline is extruded by an observed edge-aspect upper-bound hypothesis. It is
not photogrammetry, metric geometry, a recovered camera, or hidden-part repair.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay, ConvexHull, cKDTree
import trimesh

import phone_source_regeneration_20261104_v1 as P
import generation_visual_judge as V

ROOT=P.ROOT
OUTPUT=ROOT/'data/geometry-quality-20261104/phone-source-prism-v1'


def frame(index):
    program=json.loads((P.CASE/'flow-procedure-1/interaction.json').read_text())
    e=next(e for e in program['evidence']if e['source_index']==index)
    o=next(o for o in e['observations']if o['track_id']=='phone_blue')
    rgb=np.asarray(Image.open(ROOT/e['frame_path']).convert('RGB'))
    mask=np.asarray(Image.open(ROOT/o['mask_path']))>0
    assert P.ref(ROOT/e['frame_path'])['sha256']==e['frame_sha256']
    assert P.ref(ROOT/o['mask_path'])['sha256']==o['mask_sha256']
    return rgb,mask,{'source_index':index,'timestamp_s':e['timestamp_s'],
        'rgb':P.ref(ROOT/e['frame_path']),'mask':P.ref(ROOT/o['mask_path'])}


def basis(mask):
    ys,xs=np.nonzero(mask)
    pixels=np.column_stack([xs,ys]).astype(float)
    center=pixels.mean(axis=0)
    _,vectors=np.linalg.eigh(np.cov(pixels.T))
    long=vectors[:,-1]
    if long[np.argmax(abs(long))]<0:long=-long
    short=np.array([long[1],-long[0]])
    rotation=np.column_stack([short,long])
    local=(pixels-center)@rotation
    low,high=local.min(axis=0),local.max(axis=0)
    middle=(low+high)/2
    center=center+rotation@middle
    length=high[1]-low[1]
    return center,rotation,length,(high[0]-low[0])/length


def colours(rgb,mask,canonical,*,body_width):
    center,rotation,length,width=basis(mask)
    # Affine broad-face RGB projection only. Outside-mask requests are mapped
    # to the nearest observed boundary pixel, never to generated RGB.
    mapped=canonical.copy()
    mapped[:,0]*=width/body_width
    pixels=mapped*length@rotation.T+center
    indices=np.rint(pixels).astype(int)
    valid=(indices[:,0]>=0)&(indices[:,0]<mask.shape[1])&(indices[:,1]>=0)&(indices[:,1]<mask.shape[0])
    inside=np.zeros(len(indices),bool)
    inside[valid]=mask[indices[valid,1],indices[valid,0]]
    ys,xs=np.nonzero(mask)
    nearest=cKDTree(np.column_stack([xs,ys]))
    if not inside.all():
        _,near=nearest.query(pixels[~inside])
        indices[~inside]=np.column_stack([xs[near],ys[near]])
    return rgb[indices[:,1],indices[:,0]],int((~inside).sum())


def build():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    # These development indices and their prior visibility are explicitly
    # fixed before any new geometry or pose evaluation; no unseen-test claim.
    plan={'format':'interaction-twin-source-prism-construction-plan','version':1,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'implementation':P.ref(Path(__file__)),
        'broad_outline_source_index':9,'back_appearance_source_index':9,
        'front_appearance_source_index':4,'edge_aspect_source_index':18,
        'outline':'convex_hull_of_saved_model_mask;no_authored_points',
        'thickness':'min_area_rectangle_short_over_long_edge_mask_aspect;unverified_upper_bound_hypothesis',
        'holes_or_port_depth_generated':False,'camera_pose_authored':False,
        'backend':'automatic_source_silhouette_2_5D_prism_not_SAM3D',
        'new_API_or_GPU_calls':0,'automatic_publication':False,'physical_accuracy_verified':False}
    P.save(OUTPUT/'construction-plan.json',plan)
    back,bmask,bref=frame(9);front,fmask,fref=frame(4);_,emask,eref=frame(18)
    center,rotation,length,width=basis(bmask)
    yy,xx=np.nonzero(bmask)
    hullpx=cv2.convexHull(np.column_stack([xx,yy]).astype(np.float32))[:,0,:]
    hull=((hullpx-center)@rotation)/length
    # Half-pixel source sampling supports appearance without synthesizing detail.
    gridx=np.arange(hull[:,0].min(),hull[:,0].max()+1e-8,1.5/length)
    gridy=np.arange(hull[:,1].min(),hull[:,1].max()+1e-8,1.5/length)
    grid=np.stack(np.meshgrid(gridx,gridy),axis=-1).reshape(-1,2)
    inside=np.array([cv2.pointPolygonTest(hull.astype(np.float32),tuple(p),False)>=0 for p in grid])
    points=np.unique(np.vstack([grid[inside],hull]),axis=0)
    triangles=Delaunay(points).simplices
    boundary=ConvexHull(points).vertices
    ey,ex=np.nonzero(emask)
    rectangle=cv2.minAreaRect(np.column_stack([ex,ey]).astype(np.float32))
    thickness=min(rectangle[1])/max(rectangle[1])
    n=len(points)
    vertices=np.vstack([np.column_stack([points,np.full(n,thickness/2)]),
                        np.column_stack([points,np.full(n,-thickness/2)])])
    # scipy Delaunay in2D uses anticlockwise triangles; front+back opposite.
    faces=[*triangles.tolist(),*(triangles[:,::-1]+n).tolist()]
    for i,a in enumerate(boundary):
        b=boundary[(i+1)%len(boundary)]
        faces.extend([[int(a),int(a+n),int(b+n)],[int(a),int(b+n),int(b)]])
    backcol,backoutside=colours(back,bmask,points,body_width=width)
    frontpoints=points.copy();frontpoints[:,0]*=-1
    frontcol,frontoutside=colours(front,fmask,frontpoints,body_width=width)
    colors=np.column_stack([np.vstack([frontcol,backcol]),np.full(n*2,255,np.uint8)]).astype(np.uint8)
    mesh=trimesh.Trimesh(vertices=vertices,faces=np.asarray(faces),vertex_colors=colors,process=False)
    mesh.fix_normals(multibody=True)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert len(mesh.split(only_watertight=False))==1
    (OUTPUT/'asset.glb').write_bytes(mesh.export(file_type='glb'))
    imported=trimesh.load(OUTPUT/'asset.glb',force='mesh',process=False)
    assert imported.is_watertight and len(imported.split(only_watertight=False))==1
    rendered=OUTPUT/'render';rendered.mkdir()
    render=V.render_sheet(P.ref(OUTPUT/'asset.glb'),rendered)
    report={'format':'interaction-twin-phone-source-prism-private-audit','version':1,
        'construction_plan':P.ref(OUTPUT/'construction-plan.json'),'implementation':P.ref(Path(__file__)),
        'source_inputs':[bref,fref,eref],'candidate':P.ref(OUTPUT/'asset.glb'),'render':render,
        'backend':plan['backend'],'watertight':bool(imported.is_watertight),
        'winding_consistent':bool(imported.is_winding_consistent),'connected_components':len(imported.split(only_watertight=False)),
        'vertices':len(imported.vertices),'triangles':len(imported.faces),
        'source_aspect_width_over_length':float(width),'estimated_edge_thickness_over_length':float(thickness),
        'convex_outline_additional_area_fraction':float((cv2.contourArea(hullpx)-bmask.sum())/bmask.sum()),
        'source_RGB_sampling':{'front_nearest_boundary_samples':frontoutside,'back_nearest_boundary_samples':backoutside,
            'front_back_2D_affine_mapping':True,'unseen_surfaces_not_observed':True,'sides_interpolate_observed_rim_vertex_colours':True},
        'manual_geometry_or_points_used':False,'learned_3D_model_used':False,
        'fresh_geometry_validation_passed':False,'source_pose_fitting_complete':False,
        'automatic_promotion':False,'physical_accuracy_verified':False,
        'limitations':['This is a 2.5D silhouette extrusion with a simple flat-surface prior, not a complete physical phone reconstruction.',
            'Perspective, edge view obliqueness, hand occlusion and model-mask errors bias dimensions; thickness is unverified and nonmetric.',
            'Camera bump, buttons, connector recess and curved bevel depth are not reconstructed.',
            'Front/back source RGB is affinely mapped; reflections are baked observations and unseen surfaces are not recovered.',
            'Watertightness and one component only validate topology, not visual or physical fit.',
            'No old phone pose is reused, no unknown pose is filled, and mandatory failed heldout guards remain binding.']}
    P.save(OUTPUT/'report.json',report)
    return {'report':P.ref(OUTPUT/'report.json'),'candidate':report['candidate'],'render':render,
            'watertight':report['watertight'],'triangles':report['triangles']}


if __name__=='__main__':
    print(json.dumps(build(),indent=2))
