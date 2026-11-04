"""Versioned, offline geometric mating of two independently generated shells.

This is an explicitly declared *derivative*, not a correction to source evidence.
Original GLBs, source clocks, source masks and programs are never overwritten.
The bounded algorithm uses complete peripheral rim samples, opposed opening
planes and a source-cited hinge side. No hand-authored point, pose or mesh is
accepted. The current writer deliberately supports only one identity-node,
indexed triangle mesh; other encodings fail closed rather than lose appearance.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from native_appearance import load_native_glb

ROOT = Path(__file__).resolve().parent
FORMAT = 'interaction-twin-native-assembly-refinement'
RIM_BINS = 128
MAX_DISPLACEMENT_FRACTION = .12
MAX_RADIAL_SCALE_CHANGE = .25


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(path, root=ROOT):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {'path': str(path.relative_to(Path(root).resolve())), 'sha256': digest(raw), 'bytes': len(raw)}


def checked(reference, root=ROOT):
    if not isinstance(reference, dict) or set(reference) != {'path', 'sha256', 'bytes'}:
        raise ValueError('Exact file reference required')
    p = (Path(root) / reference['path']).resolve()
    if not p.is_relative_to(Path(root).resolve()) or not reference['path'].startswith(('data/', 'output/')):
        raise ValueError('Reference outside evidence namespace')
    raw = p.read_bytes()
    if len(raw) != reference['bytes'] or digest(raw) != reference['sha256']:
        raise ValueError('Evidence hash or length changed')
    return raw


def write(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError('Refinement outputs are immutable; choose a fresh directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def unit(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all() or np.linalg.norm(value) < 1e-9:
        raise ValueError('Finite nonzero vector required')
    return value / np.linalg.norm(value)


def opening_frame(vertices, opening_normal):
    """Robust opening-plane orientation and long footprint axis from vertices."""
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 1000 or not np.isfinite(v).all():
        raise ValueError('Resolved finite shell mesh required')
    hint = unit(opening_normal)
    support = v[v @ hint >= np.quantile(v @ hint, .95)]
    normal = np.linalg.svd(support - support.mean(0), full_matrices=False)[2][-1]
    normal *= np.sign(normal @ hint)
    if normal @ hint < .9:
        raise ValueError('Opening support plane contradicts native cavity evidence')
    axis = np.linalg.svd(v[::max(1, len(v)//20000)] - v.mean(0), full_matrices=False)[2][0]
    axis = unit(axis - normal * (axis @ normal))
    # A stable purely geometric sign. Source joint sign is reconciled later.
    if axis[np.argmax(np.abs(axis))] < 0:
        axis *= -1
    across = unit(np.cross(normal, axis))
    basis = np.stack([axis, across, normal], axis=1)
    coordinates = v @ basis
    lo, hi = np.quantile(coordinates[:, :2], [.001, .999], axis=0)
    center = (lo + hi) / 2
    half = (hi-lo)/2
    if min(half) < 1e-5:
        raise ValueError('Degenerate rim footprint')
    return {'basis': basis, 'center': center, 'half': half, 'coordinates': coordinates}


def periodic(values, theta):
    values = np.asarray(values, dtype=float)
    grid = (np.arange(len(values)) + .5) * 2*np.pi/len(values)
    return np.interp(np.mod(theta, 2*np.pi), np.r_[grid[-1]-2*np.pi, grid, grid[0]+2*np.pi],
                     np.r_[values[-1], values, values[0]])


def rim_profile(vertices, frame, bins=RIM_BINS):
    """Sample the full outer opening lip, retaining actual native vertex IDs.

    Per-sector candidates lie in the highest opening lip first, THEN its outer
    radial support. This ordering avoids mistaking a bulging mid-wall vertex
    for the opening at the rounded ends. Extremal heights are trimmed.
    This is a measured outer-lip proxy, not an asserted CAD mating surface.
    """
    q = np.asarray(vertices) @ frame['basis']
    xy = q[:, :2] - frame['center']
    normalized = xy / frame['half']
    theta = np.mod(np.arctan2(normalized[:, 1], normalized[:, 0]), 2*np.pi)
    radius = np.linalg.norm(normalized, axis=1)
    sector = np.floor(theta/(2*np.pi)*bins).astype(int) % bins
    rows = []
    dense_ids = []
    for i in range(bins):
        ids = np.flatnonzero(sector == i)
        if len(ids) < 12:
            raise ValueError('Insufficient whole-rim angular support')
        high = np.quantile(q[ids, 2], .995)
        ids = ids[q[ids, 2] >= high-.015*np.ptp(q[:, 2])]
        ids = ids[radius[ids] >= np.quantile(radius[ids], .65)]
        if len(ids) < 3:
            raise ValueError('Insufficient outer-lip support')
        dense_ids.extend(ids.tolist())
        threshold = np.quantile(q[ids, 2], .9)
        ids = ids[q[ids, 2] >= threshold-.002*np.ptp(q[:, 2])]
        index = int(ids[np.argmax(radius[ids])])
        rows.append({'vertex_id': index, 'point': np.asarray(vertices)[index],
                     'coordinates': q[index], 'theta': float(theta[index])})
    points = np.array([r['point'] for r in rows])
    return {'rows': rows, 'points': points, 'radius': np.linalg.norm((points @ frame['basis'])[:, :2]-frame['center'], axis=1),
            'height': (points @ frame['basis'])[:, 2], 'support_count': bins,
            'dense_vertex_ids':np.unique(dense_ids),'dense_points':np.asarray(vertices)[np.unique(dense_ids)]}


def similarity_closed(parent_frame, child_frame, parent_rim, child_rim, original_closed):
    """Oppose cavity faces; select the ambiguous long-axis sign by prior pose."""
    bp, bc = parent_frame['basis'], child_frame['basis']
    candidates = [bp @ np.diag([sign, -sign, -1.]) @ bc.T for sign in [-1., 1.]]
    r0 = np.asarray(original_closed)[:3, :3]
    u, _, vt = np.linalg.svd(r0)
    r0 = u @ vt
    rotation = min(candidates, key=lambda r: Rotation.from_matrix(r @ r0.T).magnitude())
    # Least-squares uniform footprint scale, used as the honest rigid baseline.
    cp = parent_rim['points'].mean(0)
    cc = child_rim['points'].mean(0)
    hp = np.ptp(parent_rim['points'] @ bp[:, :2], axis=0)
    hc = np.ptp(child_rim['points'] @ bc[:, :2], axis=0)
    scale = float(hp @ hc / (hc @ hc))
    if not .75 <= scale <= 1.25:
        raise ValueError('Independent shell scale mismatch exceeds refinement bound')
    result = np.eye(4)
    result[:3, :3] = rotation * scale
    result[:3, 3] = cp - result[:3, :3] @ cc
    return result


def transform(points, matrix):
    return np.asarray(points) @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]


def rim_metrics(parent_points, child_points, span):
    p, c = np.asarray(parent_points), np.asarray(child_points)
    distances = np.r_[cKDTree(p).query(c)[0], cKDTree(c).query(p)[0]]
    return {'symmetric_mean_distance': float(distances.mean()), 'symmetric_p95_distance': float(np.quantile(distances, .95)),
            'symmetric_max_distance': float(distances.max()), 'p95_fraction_of_parent_width': float(np.quantile(distances, .95)/span),
            'parent_sample_count': len(p), 'child_sample_count': len(c), 'measurement': 'whole_outer_opening_lip_vertex_proxy_not_physical_clearance'}


def derive_warp(parent_vertices, child_vertices, parent_frame, child_frame, parent_rim, child_rim, closed):
    """Bounded, smooth perimeter and lip-height fit; no added triangles/patches."""
    p_basis = parent_frame['basis']
    p = np.asarray(parent_vertices)
    c0 = transform(child_vertices, closed)
    # The parent opening frame is the common, intrinsically derived plane.
    # Match the whole radial contour, not just one selected anchor.
    pc = parent_rim['points'] @ p_basis
    cc = transform(child_rim['points'], closed) @ p_basis
    center = (np.quantile(pc[:, :2], .01, axis=0) + np.quantile(pc[:, :2], .99, axis=0))/2
    p_half = np.ptp(pc[:, :2], axis=0)/2
    c_center = (np.quantile(cc[:, :2], .01, axis=0) + np.quantile(cc[:, :2], .99, axis=0))/2
    c_half = np.ptp(cc[:, :2], axis=0)/2
    # Anisotropic footprint stretch is recorded explicitly as mesh geometry.
    stretch = p_half/c_half
    if np.max(abs(stretch-1)) > MAX_RADIAL_SCALE_CHANGE:
        raise ValueError('Cross-rim stretch exceeds declared bound')
    cq = c0 @ p_basis
    cq[:, :2] = (cq[:, :2]-c_center)*stretch + center
    cc[:, :2] = (cc[:, :2]-c_center)*stretch + center
    def samples(points):
        normalized = (points[:, :2]-center)/p_half
        theta = np.mod(np.arctan2(normalized[:, 1], normalized[:, 0]), 2*np.pi)
        order = np.argsort(theta)
        grid = (np.arange(RIM_BINS)+.5)*2*np.pi/RIM_BINS
        def interp(vals):
            return np.interp(grid,np.r_[theta[order][-1]-2*np.pi,theta[order],theta[order][0]+2*np.pi],np.r_[vals[order][-1],vals[order],vals[order][0]])
        return interp(np.linalg.norm(normalized,axis=1)),interp(points[:,2])
    pr, ph = samples(pc)
    cr, ch = samples(cc)
    radial = gaussian_filter1d(pr/cr, .6, mode='wrap')
    height_delta = gaussian_filter1d(ph-ch, .6, mode='wrap')
    normalized = (cq[:, :2]-center)/p_half
    theta = np.arctan2(normalized[:,1],normalized[:,0])
    radial_scale = periodic(radial,theta)
    if np.max(abs(radial_scale-1)) > MAX_RADIAL_SCALE_CHANGE:
        raise ValueError('Angular rim correction exceeds declared bound')
    cq[:,:2] = (cq[:,:2]-center)*radial_scale[:,None]+center
    # The lid's opening normal points opposite the parent's. Apply the lip
    # displacement near its opening and fade over the full shell thickness.
    child_height = np.asarray(child_vertices) @ child_frame['basis'][:,2]
    lo,hi=np.quantile(child_height,[.02,.98])
    weight=np.clip((child_height-lo)/(hi-lo),0,1)
    weight=weight*weight*(3-2*weight)
    cq[:,2] += periodic(height_delta,theta)*weight
    changed_closed = cq @ p_basis.T
    inv = np.linalg.inv(closed)
    new_vertices = transform(changed_closed,inv)
    width = float(np.ptp(p @ p_basis[:,0]))
    displacement=np.linalg.norm(new_vertices-np.asarray(child_vertices),axis=1)
    if displacement.max()/width > MAX_DISPLACEMENT_FRACTION:
        raise ValueError('Derived mesh displacement exceeds declared bound')
    return new_vertices, {'method':'whole_rim_intrinsic_footprint_stretch_and_smooth_angular_lip_height',
                          'in_plane_scale':stretch.tolist(),'maximum_angular_radial_scale_delta':float(max(abs(radial-1))),
                          'maximum_native_vertex_displacement':float(displacement.max()),'p95_native_vertex_displacement':float(np.quantile(displacement,.95)),
                          'maximum_displacement_fraction_parent_width':float(displacement.max()/width),
                          'vertex_count_unchanged':len(new_vertices),'faces_and_appearance_unchanged':True,
                          'no_connector_or_occlusion_patch_added':True}


def rewrite_positions(raw, vertices):
    """Write a new GLB, preserving every non-position binary byte exactly."""
    if raw[:4] != b'glTF' or struct.unpack_from('<I',raw,4)[0] != 2:
        raise ValueError('GLB2 required')
    jlen,jkind=struct.unpack_from('<II',raw,12)
    if jkind != 0x4e4f534a:
        raise ValueError('GLB JSON first required')
    document=json.loads(raw[20:20+jlen])
    off=20+jlen;blen,bkind=struct.unpack_from('<II',raw,off)
    if bkind!=0x004e4942 or off+8+blen!=len(raw):
        raise ValueError('Exactly one embedded binary chunk required')
    if len(document.get('nodes',[]))!=1 or document['nodes'][0].get('mesh')!=0 or set(document['nodes'][0])-{'name','mesh','extras'}:
        raise ValueError('Only identity single native mesh writer supported')
    primitives=document['meshes'][0]['primitives']
    if len(primitives)!=1 or 'NORMAL' in primitives[0]['attributes']:
        raise ValueError('Writer requires one primitive without stored normals')
    a=document['accessors'][primitives[0]['attributes']['POSITION']]
    bv=document['bufferViews'][a['bufferView']]
    if a['componentType']!=5126 or a['type']!='VEC3' or bv.get('byteStride',12)!=12 or a.get('sparse') or a['count']!=len(vertices):
        raise ValueError('Packed float native positions required')
    data=bytearray(raw[off+8:])
    start=bv.get('byteOffset',0)+a.get('byteOffset',0)
    encoded=np.asarray(vertices,dtype='<f4').tobytes()
    if len(encoded)!=a['count']*12 or start+len(encoded)>len(data):
        raise ValueError('Invalid position allocation')
    original_positions=bytes(data[start:start+len(encoded)])
    data[start:start+len(encoded)]=encoded
    a['min']=np.asarray(vertices,dtype=np.float32).min(0).tolist()
    a['max']=np.asarray(vertices,dtype=np.float32).max(0).tolist()
    document.setdefault('asset',{}).setdefault('extras',{})['interaction_twin_derivative']='mesh_derived_geometric_mating_v1'
    encoded_json=json.dumps(document,separators=(',',':'),allow_nan=False).encode()
    encoded_json+=b' '*((-len(encoded_json))%4)
    total=12+8+len(encoded_json)+8+len(data)
    new=struct.pack('<4sII',b'glTF',2,total)+struct.pack('<II',len(encoded_json),0x4e4f534a)+encoded_json+struct.pack('<II',len(data),0x004e4942)+data
    return new,{'original_position_bytes_sha256':digest(original_positions),'derivative_position_bytes_sha256':digest(encoded),
                'unchanged_nonposition_binary_sha256':digest(bytes(data[:start])+bytes(data[start+len(encoded):])),
                'original_nonposition_binary_sha256':digest(raw[off+8:off+8+start]+raw[off+8+start+len(encoded):]),
                'binary_length_unchanged':len(data)==blen,'topology_color_uv_material_image_bytes_preserved':True}


def hinge_from_rim(parent_frame,parent_rim,old_anchor,old_axis,closed_child_vertices=None):
    """Derive a rear lip tangent from the source-cited hinge side, across its span."""
    q=parent_rim['points']@parent_frame['basis']
    center=parent_frame['center']
    old=np.asarray(old_anchor)@parent_frame['basis']
    side=1. if old[1]-center[1]>=0 else -1.
    across=side*(q[:,1]-center[1])
    # The broad rear straight segment, not a single surface vertex.
    select=(across>=np.quantile(across,.80))&(abs(q[:,0]-center[0])<.60*parent_frame['half'][0])
    points=parent_rim['points'][select]
    if len(points)<12:
        raise ValueError('Rear hinge span has insufficient geometric support')
    matched=None
    if closed_child_vertices is not None:
        matched=np.asarray(closed_child_vertices)[cKDTree(closed_child_vertices).query(points)[1]]
        centers=(points+matched)/2
    else:
        centers=points
    center3=centers.mean(0)
    axis=np.linalg.svd(centers-center3,full_matrices=False)[2][0]
    axis=unit(axis)
    if axis@np.asarray(old_axis)<0:
        axis*=-1
    along=(centers-center3)@axis
    residual=np.linalg.norm(centers-center3-along[:,None]*axis,axis=1)
    ends=np.array([center3+axis*np.quantile(along,.05),center3+axis*np.quantile(along,.95)])
    return center3,axis,{'support_count':len(points),'source_cited_side_anchor_native':np.asarray(old_anchor).tolist(),
                        'axis_segment_endpoints_parent_native':ends.tolist(),'rear_rim_line_p95_distance':float(np.quantile(residual,.95)),
                        'paired_shell_rim_support_used':matched is not None,
                        'method':'source_cited_rear_side_then_whole_paired_native_rim_midpoint_span_PCA'}


def axis_rotation(axis,q):
    result=np.eye(4);result[:3,:3]=Rotation.from_rotvec(unit(axis)*q).as_matrix();return result


def seam_metrics(parent_vertices,child_vertices,closed,pivot,axis,angles,segment):
    """Distances at BOTH geometric hinge endpoints, at every requested angle."""
    p=np.asarray(parent_vertices);c=np.asarray(child_vertices);tree=cKDTree(p)
    endpoints=np.asarray(segment)
    p_nearest=tree.query(endpoints)[1]
    closed_child=transform(c,closed);child_ids=cKDTree(closed_child).query(endpoints)[1]
    rows=[]
    for angle in angles:
        a=np.eye(4);a[:3,3]=pivot
        matrix=a@axis_rotation(axis,angle)@np.linalg.inv(a)@closed
        cp=transform(c[child_ids],matrix)
        distances=np.linalg.norm(cp-p[p_nearest],axis=1)
        rows.append({'q_rad':float(angle),'endpoint_distances':distances.tolist(),
                     'maximum_endpoint_distance':float(max(distances))})
    return {'endpoint_parent_vertex_ids':p_nearest.tolist(),'endpoint_child_vertex_ids':child_ids.tolist(),'samples':rows,
            'maximum_across_angles_and_both_endpoints':float(max(r['maximum_endpoint_distance']for r in rows)),
            'measurement':'fixed_actual_mesh_vertices_near_both_derived_hinge_endpoints_not_single_anchor'}


def triangle_quality(original,derivative,faces):
    a=np.asarray(original)[faces];b=np.asarray(derivative)[faces]
    an=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);bn=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
    la=np.linalg.norm(an,axis=1);lb=np.linalg.norm(bn,axis=1);valid=(la>1e-14)&(lb>1e-14)
    cosine=np.sum(an[valid]*bn[valid],axis=1)/(la[valid]*lb[valid])
    return {'nondegenerate_triangles_checked':int(valid.sum()),'orientation_reversals':int((cosine<=0).sum()),
            'new_degenerate_triangles':int(((la>1e-14)&(lb<=1e-14)).sum()),'minimum_normal_cosine':float(cosine.min()),
            'self_intersection_checked':False}


def run(scene_ref,output,parent_id='case_base',child_id='case_lid',root=ROOT):
    started=time.monotonic();root=Path(root).resolve();output=(root/output).resolve()
    if not output.is_relative_to(root/'data') or output.exists():
        raise ValueError('Fresh data output namespace required')
    scene=json.loads(checked(scene_ref,root));parts={p['id']:p for p in scene['components']}
    joint=next(j for j in scene['mechanics']['joints']if j['parent_id']==parent_id and j['child_id']==child_id)
    prior=joint.get('cavity_mating_prior',{})
    if prior.get('status')!='estimated_geometry_prior':
        raise ValueError('Two resolved native opening faces and source-cited hinge evidence required')
    pr,cr=parts[parent_id]['asset'],parts[child_id]['asset']
    pm,cm=load_native_glb(checked(pr,root)),load_native_glb(checked(cr,root))
    pf=opening_frame(pm.vertices,prior['parent']['normal']);cf=opening_frame(cm.vertices,prior['child']['normal'])
    pp= rim_profile(pm.vertices,pf);cc=rim_profile(cm.vertices,cf)
    original_closed=np.asarray(joint['parent_from_joint'])@np.asarray(joint['joint_from_child_zero'])
    closed=similarity_closed(pf,cf,pp,cc,original_closed)
    new_vertices,warp=derive_warp(pm.vertices,cm.vertices,pf,cf,pp,cc,closed)
    raw,encoding=rewrite_positions(checked(cr,root),new_vertices)
    actual=load_native_glb(raw)
    if not np.array_equal(actual.faces,cm.faces) or not np.array_equal(actual.vertex_colors,cm.vertex_colors):
        raise ValueError('Derivative changed original topology or appearance')
    triangle=triangle_quality(cm.vertices,actual.vertices,cm.faces)
    if triangle['orientation_reversals'] or triangle['new_degenerate_triangles']:
        raise ValueError('Derivative reverses or degenerates native triangles')
    pivot,axis,hinge=hinge_from_rim(pf,pp,joint['parent_anchor_native'],joint['axis'],transform(actual.vertices,closed))
    parent_joint=np.eye(4);parent_joint[:3,3]=pivot
    child_zero=np.linalg.inv(parent_joint)@closed
    refined=copy.deepcopy(joint)
    # Old one-anchor and silhouette diagnostics describe a different transform.
    # They remain byte-preserved in original_scene/report.joint.original only.
    for historical in ('cavity_mating_prior','anchor_selection','fit_diagnostics','shared_parent_refinement'):
        refined.pop(historical,None)
    refined.update(axis=axis.tolist(),parent_from_joint=parent_joint.tolist(),joint_from_child_zero=child_zero.tolist(),
                   parent_native_scale=1.,child_native_scale=float(np.cbrt(np.linalg.det(closed[:3,:3]))),
                   parent_anchor_native=pivot.tolist(),child_anchor_native=transform(pivot[None],np.linalg.inv(closed))[0].tolist(),
                   parent_anchor_vertex_id=None,child_anchor_vertex_id=None,
                   refinement_kind='mesh_derived_geometric_mating_v1',physical_contact_verified=False,
                   zero_state_alignment_verified=False,
                   matrix_convention='parent_native_pose * parent_from_joint * axis_rotation(q) * joint_from_child_zero; refined child mesh native coordinates',
                   limitations=['The transform and derived lid mesh improve geometric mating; original source-video alignment has not been revalidated.',
                                'A fixed whole-rim-derived axis enforces kinematic attachment; physical hinge/contact and metric scale remain unverified.'])
    # Preserve observed angles/source times. A0 is now geometric closure; this
    # is an explicit new gauge, not a remeasurement of video angles.
    for row in refined.get('samples',[]):
        if row.get('q_rad') is not None:row['reason']='original_estimated_angle_in_refined_geometric_closed_gauge';row['accepted_motion_evidence']=False
    width=float(np.ptp(pm.vertices@pf['basis'][:,0]))
    new_rim=actual.vertices[[r['vertex_id']for r in cc['rows']]]
    original_metrics=rim_metrics(pp['points'],transform(cc['points'],original_closed),width)
    similarity_metrics=rim_metrics(pp['points'],transform(cc['points'],closed),width)
    final_metrics=rim_metrics(pp['points'],transform(new_rim,closed),width)
    angles=np.linspace(0,max(joint['limits_rad']),7)
    seam=seam_metrics(pm.vertices,actual.vertices,closed,pivot,axis,angles,hinge['axis_segment_endpoints_parent_native'])
    accepted=(final_metrics['symmetric_p95_distance']<original_metrics['symmetric_p95_distance']*.5
              and final_metrics['p95_fraction_of_parent_width']<.025
              and seam['maximum_across_angles_and_both_endpoints']/width<.015)
    if not accepted:
        raise ValueError('Whole-rim geometric quality gate rejected candidate')
    output.mkdir(parents=True)
    implementation=output/'implementation.py';implementation.write_bytes(Path(__file__).read_bytes())
    asset=output/'case-lid-derived.glb';asset.write_bytes(raw);derived_ref=ref(asset,root)
    result=copy.deepcopy(scene);rp={p['id']:p for p in result['components']};rp[child_id]['asset']=derived_ref
    rp[child_id]['full_mesh_geometry_modified']=True;rp[child_id]['geometry_provenance']={'kind':'mesh_derived_geometric_correction','original_asset':cr}
    rp[child_id]['original_native_uniform_scale']=rp[child_id]['native_uniform_scale']
    rp[child_id]['native_uniform_scale']=rp[parent_id]['native_uniform_scale']*refined['child_native_scale']
    rjoint=next(i for i,j in enumerate(result['mechanics']['joints'])if j['id']==joint['id']);result['mechanics']['joints'][rjoint]=refined
    parent_poses={r['source_sample_index']:r for r in rp[parent_id]['poses']};angles_by={r['source_sample_index']:r for r in refined['samples']}
    for row in rp[child_id]['poses']:
        parent=parent_poses.get(row['source_sample_index'],{});angle=angles_by.get(row['source_sample_index'],{})
        if parent.get('camera_from_native_mesh') is None or angle.get('q_rad') is None:
            row['camera_from_native_mesh']=None;row['status']='unknown';continue
        matrix=np.asarray(parent['camera_from_native_mesh'])@parent_joint@axis_rotation(axis,angle['q_rad'])@child_zero
        row['camera_from_native_mesh']=matrix.tolist();row['reason']='fixed_refined_joint_FK_not_independent_child_pose';row['status']='estimated'
    result['assembly_refinement']={'kind':FORMAT,'version':1,'original_scene':scene_ref,'status':'geometric_derivative_not_source_reconstruction_validation',
                                   'geometry_modified':True,'physical_accuracy_verified':False,'source_alignment_revalidated':False}
    result['physical_accuracy_verified']=False;result['simulation_ready']=False
    result.setdefault('limitations',[]).append('The lid is a declared mesh-derived rim-fitting derivative. Its fit improves geometric mating but is not calibrated physical clearance or a newly verified source-video pose.')
    write(output/'scene.json',result)
    assets=[{'id':p['id'],'original_asset':p['asset'],'asset':rp[p['id']]['asset'],'geometry_modified':p['id']==child_id}for p in scene['components']]
    report={'format':FORMAT,'version':1,'status':'accepted_geometric_derivative','original_scene':scene_ref,'scene_ref':ref(output/'scene.json',root),
            'source_sha256':scene['source']['sha256'],'original_program':scene['original_program'],'assets':assets,
            'joint':{'id':joint['id'],'parent_id':parent_id,'child_id':child_id,'original':joint,'refined':refined,'closed_q_rad':0.,'open_q_rad':max(joint['limits_rad']),
                     'source_sample_times_and_angles_preserved':True,'closed_angle_is_refined_geometric_gauge':True},
            'metrics':{'original':original_metrics,'similarity_only':similarity_metrics,'refined':final_metrics,'hinge_both_ends':seam,
                       'dense_opening_lip_original':rim_metrics(pp['dense_points'],transform(cc['dense_points'],original_closed),width),
                       'dense_opening_lip_refined':rim_metrics(pp['dense_points'],transform(actual.vertices[cc['dense_vertex_ids']],closed),width)},
            'rim_support':{'parent_vertex_ids':[r['vertex_id']for r in pp['rows']],'child_vertex_ids':[r['vertex_id']for r in cc['rows']],
                           'sampling':'128 full-perimeter sectors; trimmed99.5pct opening support first; then outer35pct radial support'},
            'warp':warp,'hinge':hinge,'encoding':encoding,'triangle_quality':triangle,'implementation':ref(implementation,root),
            'physical_accuracy_verified':False,'source_alignment_revalidated':False,'manual_geometry_or_poses_used':False,'new_model_calls':0,
            'limitations':['Independent SAM3D shells were not physically compatible before this explicitly declared derivative.',
                           'Rim distances use measured outer-lip vertex proxies and normalized native units, not metric calibrated tolerances.',
                           'Original source-estimated angles are preserved in a newly defined geometric closure gauge; source alignment is not revalidated.',
                           'No physical hinge/contact, collision simulation, robot grasp or mechanics verification is claimed.'],
            'wall_seconds':time.monotonic()-started}
    write(output/'report.json',report)
    companion={'format':'interaction-twin-refined-mechanism-description','version':1,'status':'geometric_derivative_research_draft',
               'source':scene['source'],'original_program':scene['original_program'],'original_scene':scene_ref,'scene_ref':report['scene_ref'],
               'refinement_report':ref(output/'report.json',root),'assets':assets,'joints':[refined],
               'claims':{'native_originals_preserved':True,'derivative_mesh_geometry_used':True,'physical_simulation_performed':False,'physical_accuracy_verified':False},
               'limitations':report['limitations']}
    write(output/'descriptive-companion.json',companion)
    return {'report':ref(output/'report.json',root),'scene':report['scene_ref'],'description':ref(output/'descriptive-companion.json',root),'accepted':accepted}


def verify_report(reference,root=ROOT):
    """Recompute the actual derivative from original bytes, not model claims."""
    root=Path(root).resolve();report=json.loads(checked(reference,root))
    if report.get('format')!=FORMAT or report.get('version')!=1 or report.get('status')!='accepted_geometric_derivative':
        raise ValueError('Wrong assembly refinement contract')
    if digest(checked(report['implementation'],root))!=digest(Path(__file__).read_bytes()):
        raise ValueError('Versioned assembly implementation differs')
    original=json.loads(checked(report['original_scene'],root));scene=json.loads(checked(report['scene_ref'],root))
    if report.get('source_sha256')!=original['source']['sha256'] or scene['source']!=original['source'] or scene['original_program']!=original['original_program']:
        raise ValueError('Source/program binding changed')
    original_parts={p['id']:p for p in original['components']};parts={p['id']:p for p in scene['components']}
    if set(parts)!=set(original_parts) or len(report['assets'])!=len(parts):
        raise ValueError('Component inventory changed')
    joint=report['joint'];old=next(j for j in original['mechanics']['joints']if j['id']==joint['id'])
    new=next(j for j in scene['mechanics']['joints']if j['id']==joint['id'])
    if old!=joint['original'] or new!=joint['refined'] or joint['closed_q_rad']!=0 or new['limits_rad']!=old['limits_rad']:
        raise ValueError('Joint provenance or angle range changed')
    pid,cid=old['parent_id'],old['child_id']
    for item in report['assets']:
        i=item['id']
        if item['original_asset']!=original_parts[i]['asset'] or item['asset']!=parts[i]['asset'] or item['geometry_modified']!=(i==cid):
            raise ValueError('Asset derivative binding changed')
        checked(item['asset'],root);checked(item['original_asset'],root)
        if i!=cid and parts[i]!=original_parts[i]:
            raise ValueError('Unrelated source component changed')
    pm=load_native_glb(checked(original_parts[pid]['asset'],root));cm=load_native_glb(checked(original_parts[cid]['asset'],root))
    prior=old['cavity_mating_prior'];pf=opening_frame(pm.vertices,prior['parent']['normal']);cf=opening_frame(cm.vertices,prior['child']['normal'])
    pp=rim_profile(pm.vertices,pf);cp=rim_profile(cm.vertices,cf)
    old_closed=np.asarray(old['parent_from_joint'])@np.asarray(old['joint_from_child_zero'])
    closed=similarity_closed(pf,cf,pp,cp,old_closed)
    vertices,warp=derive_warp(pm.vertices,cm.vertices,pf,cf,pp,cp,closed)
    expected,encoding=rewrite_positions(checked(original_parts[cid]['asset'],root),vertices)
    if checked(parts[cid]['asset'],root)!=expected or report['warp']!=warp or report['encoding']!=encoding:
        raise ValueError('Derivative cannot be reproduced from bound native vertices')
    actual=load_native_glb(expected)
    pivot,axis,hinge=hinge_from_rim(pf,pp,old['parent_anchor_native'],old['axis'],transform(actual.vertices,closed))
    if not np.allclose(new['axis'],axis,atol=1e-12) or not np.allclose(np.asarray(new['parent_from_joint'])[:3,3],pivot,atol=1e-12):
        raise ValueError('Joint is not the automatic whole-rim hinge')
    if not np.allclose(np.asarray(new['parent_from_joint'])@np.asarray(new['joint_from_child_zero']),closed,atol=1e-12):
        raise ValueError('Closed pose differs from automatic rim fit')
    if report['hinge']!=hinge or report['triangle_quality']!=triangle_quality(cm.vertices,actual.vertices,cm.faces):
        raise ValueError('Geometric quality evidence changed')
    width=float(np.ptp(pm.vertices@pf['basis'][:,0]));child_points=actual.vertices[[r['vertex_id']for r in cp['rows']]]
    metrics={'original':rim_metrics(pp['points'],transform(cp['points'],old_closed),width),
             'similarity_only':rim_metrics(pp['points'],transform(cp['points'],closed),width),
             'refined':rim_metrics(pp['points'],transform(child_points,closed),width),
             'hinge_both_ends':seam_metrics(pm.vertices,actual.vertices,closed,pivot,axis,np.linspace(0,max(old['limits_rad']),7),hinge['axis_segment_endpoints_parent_native']),
             'dense_opening_lip_original':rim_metrics(pp['dense_points'],transform(cp['dense_points'],old_closed),width),
             'dense_opening_lip_refined':rim_metrics(pp['dense_points'],transform(actual.vertices[cp['dense_vertex_ids']],closed),width)}
    if report['metrics']!=metrics:
        raise ValueError('Rim or hinge metrics cannot be reproduced')
    for key in ('physical_accuracy_verified','source_alignment_revalidated','manual_geometry_or_poses_used'):
        if report.get(key) is not False:raise ValueError('Unverified claim changed')
    if report.get('new_model_calls')!=0 or scene.get('simulation_ready')is not False:
        raise ValueError('Offline/unverified scope changed')
    if [(r['source_sample_index'],r['timestamp_s'],r['q_rad'])for r in new['samples']]!=[(r['source_sample_index'],r['timestamp_s'],r['q_rad'])for r in old['samples']]:
        raise ValueError('Source angle clock changed')
    parent_poses={r['source_sample_index']:r for r in parts[pid]['poses']};angle_rows={r['source_sample_index']:r for r in new['samples']}
    for row in parts[cid]['poses']:
        p=parent_poses[row['source_sample_index']];q=angle_rows[row['source_sample_index']]['q_rad']
        if p['camera_from_native_mesh']is None or q is None:
            if row['camera_from_native_mesh']is not None or row['status']!='unknown':raise ValueError('Unsupported child placement invented')
        else:
            pose=np.asarray(p['camera_from_native_mesh'])@np.asarray(new['parent_from_joint'])@axis_rotation(new['axis'],q)@np.asarray(new['joint_from_child_zero'])
            if not np.allclose(row['camera_from_native_mesh'],pose,atol=1e-12):raise ValueError('Independent child pose substituted')
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(run(ref(ROOT/args.scene),args.output),indent=2))
