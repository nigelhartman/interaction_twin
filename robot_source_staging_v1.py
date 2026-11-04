"""Source-relative visual staging from exact saved component observations.

This strips the video camera gauge, keeps a single parent normalization, and
preserves relative native scale/orientation. It does not infer physical fit.
Different parts may use different, explicitly recorded source samples.
"""
from __future__ import annotations

import itertools
import copy
import hashlib
import json
import math
import threading
from collections import OrderedDict

METHOD = 'source_relative_component_display_v1'
_CACHE = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CAP_CACHE = OrderedDict()


def multiply(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def similarity(matrix):
    if (not isinstance(matrix, list) or len(matrix) != 4 or
        any(not isinstance(row, list) or len(row) != 4 for row in matrix) or
        any(type(v) not in (int, float) or not math.isfinite(v) for row in matrix for v in row) or
        any(abs(matrix[3][i] - (1 if i == 3 else 0)) > 1e-8 for i in range(4))):
        raise ValueError('Finite homogeneous source transform required')
    scales = [math.sqrt(sum(matrix[i][j] ** 2 for i in range(3))) for j in range(3)]
    scale = sum(scales) / 3
    if scale <= 1e-8 or max(abs(s - scale) for s in scales) > scale * 1e-5:
        raise ValueError('Source transform must have one positive uniform scale')
    rotation = [[matrix[i][j] / scale for j in range(3)] for i in range(3)]
    if any(abs(sum(rotation[k][i] * rotation[k][j] for k in range(3)) - (i == j)) > 1e-5
           for i in range(3) for j in range(3)):
        raise ValueError('Source transform rotation is not orthonormal')
    det = sum(rotation[0][i] * (rotation[1][(i+1) % 3] * rotation[2][(i+2) % 3] -
              rotation[1][(i+2) % 3] * rotation[2][(i+1) % 3]) for i in range(3))
    if abs(det - 1) > 1e-5:
        raise ValueError('Reflected source transform cannot define a native rotation')
    return scale, rotation


def inverse(matrix):
    scale, rotation = similarity(matrix)
    result = [[rotation[j][i] / scale for j in range(3)] + [0.] for i in range(3)]
    for i in range(3):
        result[i][3] = -sum(result[i][j] * matrix[j][3] for j in range(3))
    return result + [[0., 0., 0., 1.]]


def transform(matrix, point):
    return [sum(matrix[i][j] * point[j] for j in range(3)) + matrix[i][3] for i in range(3)]


def quaternion(rotation):
    trace = sum(rotation[i][i] for i in range(3))
    if trace > 0:
        s = math.sqrt(trace + 1) * 2
        q = [(rotation[2][1] - rotation[1][2]) / s,
             (rotation[0][2] - rotation[2][0]) / s,
             (rotation[1][0] - rotation[0][1]) / s, s / 4]
    else:
        i = max(range(3), key=lambda k: rotation[k][k]);j = (i + 1) % 3;k = (i + 2) % 3
        s = math.sqrt(max(0, 1 + rotation[i][i] - rotation[j][j] - rotation[k][k])) * 2
        q = [0., 0., 0., (rotation[k][j] - rotation[j][k]) / s]
        q[i] = s / 4;q[j] = (rotation[i][j] + rotation[j][i]) / s;q[k] = (rotation[i][k] + rotation[k][i]) / s
    norm = math.sqrt(sum(v * v for v in q));q = [v / norm for v in q]
    return [-v for v in q] if q[3] < 0 else q


def component_display(parent_bounds, child_bounds, parent_from_child, display_span=.38, vertices=None):
    """Reproduce one native child under the same centered parent scale."""
    scale, rotation = similarity(parent_from_child)
    parent_center = [(a + b) / 2 for a, b in zip(*parent_bounds)]
    child_center = [(a + b) / 2 for a, b in zip(*child_bounds)]
    parent_span = max(parent_bounds[1][i] - parent_bounds[0][i] for i in range(3))
    child_span = max(child_bounds[1][i] - child_bounds[0][i] for i in range(3))
    if min(parent_span, child_span, display_span) <= 1e-8:
        raise ValueError('Nondegenerate native bounds required')
    common_scale = display_span / parent_span
    center = transform(parent_from_child, child_center)
    points = [transform(parent_from_child, point) for point in
              (vertices if vertices is not None else itertools.product(*zip(*child_bounds)))]
    low = [min(point[i] for point in points) for i in range(3)]
    high = [max(point[i] for point in points) for i in range(3)]
    return {'display_max_dimension': common_scale * scale * child_span,
            'quaternion_xyzw': quaternion(rotation),
            'center_offset': [(center[i] - parent_center[i]) * common_scale for i in range(3)],
            'half_height': (high[1] - low[1]) * common_scale / 2,
            'oriented_half_extents': [(high[i] - low[i]) * common_scale / 2 for i in range(3)],
            'lower_support_extents': [(center[i] - low[i]) * common_scale for i in range(3)],
            'upper_support_extents': [(high[i] - center[i]) * common_scale for i in range(3)],
            'parent_from_child': parent_from_child,
            'native_center_in_parent': center,
            'native_aabb_in_parent': [low, high]}


def cap_tilt_from_supported_reference(relative, native_center, native_cap_normal,
                                      reference_relative, reference_cap_normal):
    """Minimal proper visual rotation, retaining the saved horizontal cap yaw.

    The normals must first be identified by a source/native visual review.
    This is an authored display-rest hypothesis, not a recovered source pose.
    """
    scale, rotation = similarity(relative)
    _, reference_rotation = similarity(reference_relative)
    def direction(matrix, normal):
        if (not isinstance(normal, list) or len(normal) != 3 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in normal) or
                abs(sum(v*v for v in normal)-1) > 1e-6):
            raise ValueError('A reviewed unit native cap normal is required')
        return [sum(matrix[i][j]*normal[j] for j in range(3)) for i in range(3)]
    source = direction(rotation, native_cap_normal)
    reference = direction(reference_rotation, reference_cap_normal)
    horizontal = math.hypot(source[0], source[2])
    if reference[1] <= 0 or horizontal < 1e-8:
        raise ValueError('The supported reference must face upward with a source-defined yaw')
    target_height = min(1., reference[1]);target_horizontal = math.sqrt(max(0., 1-target_height**2))
    target = [source[0]/horizontal*target_horizontal, target_height,
              source[2]/horizontal*target_horizontal]
    cross = [source[1]*target[2]-source[2]*target[1],
             source[2]*target[0]-source[0]*target[2],
             source[0]*target[1]-source[1]*target[0]]
    sine = math.sqrt(sum(v*v for v in cross));cosine = max(-1., min(1., sum(a*b for a,b in zip(source,target))))
    angle = math.atan2(sine, cosine)
    if sine < 1e-10:
        delta = [[float(i == j) for j in range(3)] for i in range(3)]
    else:
        x,y,z = [v/sine for v in cross];c = math.cos(angle);s = math.sin(angle);v = 1-c
        delta = [[c+x*x*v,x*y*v-z*s,x*z*v+y*s],
                 [y*x*v+z*s,c+y*y*v,y*z*v-x*s],
                 [z*x*v-y*s,z*y*v+x*s,c+z*z*v]]
    if len(native_center) != 3 or any(not math.isfinite(v) for v in native_center):
        raise ValueError('Finite native center required')
    center = transform(relative, native_center)
    refined = [[sum(delta[i][k]*relative[k][j] for k in range(3)) for j in range(3)] + [0.]
               for i in range(3)] + [[0.,0.,0.,1.]]
    for i in range(3):refined[i][3] = center[i]-sum(refined[i][j]*native_center[j] for j in range(3))
    similarity(refined)
    return refined, {'source_cap_direction_parent':source,
        'reference_cap_direction_parent':reference, 'target_cap_direction_parent':target,
        'minimal_proper_rotation_rad':angle, 'delta_rotation_parent':delta,
        'source_yaw_preserved':True, 'source_uniform_scale':scale,
        'native_center_in_parent_unchanged':True, 'physical_orientation_verified':False}


def apply_reviewed_cap_profile(descriptor, context, profile):
    """Compile a visually reviewed native semantic hint against exact inputs.

    Review fingerprints are nonroutable provenance. Read-only MCP planning does
    not load private worker logs, invoke reasoning or import image/CV packages.
    """
    if not descriptor or not profile:return descriptor
    if (profile.get('format') != 'interaction-twin-compiled-native-cap-profile' or profile.get('version') != 1 or
            profile.get('method') != 'match_supported_reference_tilt_preserve_source_yaw' or
            profile.get('source_sha256') != context['binding']['source_sha256'] or
            profile.get('evidence_program_ref') != context['binding']['program_ref'] or
            profile.get('scene_ref') != descriptor['scene_ref'] or
            profile.get('physical_orientation_verified') is not False):
        raise ValueError('Reviewed visual rest profile differs from the exact source program')
    for name in ['semantic_review_sha256','composition_review_sha256']:
        digest=profile.get(name)
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Exact visual review fingerprints are required')
    confidence=profile.get('semantic_confidence')
    if type(confidence) not in (int,float) or not math.isfinite(confidence) or not 0<confidence<=1:
        raise ValueError('Finite visual semantic confidence required')
    rows={r['component_id']:r for r in descriptor['components']}
    bindings=profile.get('components',[])
    if len(bindings)!=2 or {b['component_id']for b in bindings}!={profile['component_id'],profile['reference_component_id']}:
        raise ValueError('Reviewed cap needs one exact selected part and supported reference')
    for binding in bindings:
        cid=binding['component_id'];row=rows.get(cid)
        if (not row or binding['native_asset_ref'] != context['assets'][cid]['asset_ref'] or
                any(binding[k]!=row[k] for k in ['source_sample_index','parent_source_mask_ref','child_source_mask_ref'])):
            raise ValueError('Native cap review does not match selected supported source masks')
    cue=profile['source_visible_rim_protrusion'];row=rows[profile['component_id']]
    if (cue['parent_mask_ref']!=row['parent_source_mask_ref'] or cue['child_mask_ref']!=row['child_source_mask_ref'] or
            cue['source_sample_index']!=row['source_sample_index']):
        raise ValueError('Native cap protrusion cue belongs to another observation')
    samples=cue.get('parent_contour_samples_xy',[]);box=cue.get('child_box_xyxy',[])
    if len(samples)<4 or len(samples)>2048 or len(box)!=4 or any(type(v)is not int for xy in samples for v in xy) or any(type(v)is not int for v in box) or box[3]<box[1]:
        raise ValueError('Bounded compiled source contour required')
    values=sorted(xy[1]for xy in samples);mid=len(values)//2;median=values[mid]if len(values)%2 else sum(values[mid-1:mid+1])/2
    fraction=max(0.,min(.5,(median-box[1])/(box[3]-box[1]+1)))
    if abs(fraction-cue['bounded_visible_height_fraction'])>1e-12 or abs(median-cue['parent_rim_proxy_y_px'])>1e-12:
        raise ValueError('Compiled source contour fraction differs')
    key=hashlib.sha256(json.dumps([profile,descriptor],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    with _CACHE_LOCK:
        cached=copy.deepcopy(_CAP_CACHE.get(key))
        if cached is not None:_CAP_CACHE.move_to_end(key)
    if cached is not None:return cached
    result=copy.deepcopy(descriptor);target=next(r for r in result['components']if r['component_id']==profile['component_id']);reference=rows[profile['reference_component_id']]
    center=[(a+b)/2 for a,b in zip(*target['native_bounds'])]
    refined,proof=cap_tilt_from_supported_reference(target['parent_from_child'],center,profile['native_cap_normal'],reference['parent_from_child'],profile['reference_native_cap_normal'])
    from portable_twin import decode_glb_bytes
    native=decode_glb_bytes(context['catalog']._bytes(context['assets'][target['component_id']]['asset_ref'],maximum=16*1024*1024))
    vertices=[p for primitive in native['primitives']for p in primitive['vertices']]
    measured=component_display(result['parent_native_bounds'],target['native_bounds'],refined,vertices=vertices)
    original=copy.deepcopy(target);target.update(measured);target['parent_from_child']=original['parent_from_child']
    target['source_quaternion_xyzw']=original['quaternion_xyzw'];target['orientation_derived_parent_from_child']=refined
    target['source_observation_support']={k:original[k]for k in ['half_height','oriented_half_extents','lower_support_extents','upper_support_extents','native_aabb_in_parent']}
    target['orientation_hypothesis']={**proof,'id':'source_matched_cap_tilt','method':profile['method'],
        'native_outer_cap_normal':profile['native_cap_normal'],'reference_component_id':profile['reference_component_id'],
        'semantic_review_sha256':profile['semantic_review_sha256'],'composition_review_sha256':profile['composition_review_sha256'],
        'semantic_confidence':confidence,'authored_visual_rest_not_recovered_source_pose':True}
    target['source_visible_rim_protrusion']=copy.deepcopy(cue)
    result['reviewed_visual_rest_profile']=copy.deepcopy(profile)
    with _CACHE_LOCK:
        _CAP_CACHE[key]=copy.deepcopy(result)
        while len(_CAP_CACHE)>16:_CAP_CACHE.popitem(last=False)
    return result


def _containment_windows(document, child_id):
    variables = {v['id'] for v in document.get('variables', []) if v.get('subject_track_id') == child_id}
    evidence = {e['id']: e for e in document.get('evidence', [])}
    exact = {}
    for row in document.get('assertions', []):
        if row.get('variable_id') in variables and row.get('value') == 'in_case' and row.get('evidence_id') in evidence:
            exact[evidence[row['evidence_id']]['source_index']] = row['evidence_id']
    # A pre-removal state gives an inferred initial containment window, not a
    # new measured pose or a hold in the recorded replay. Every chosen pose
    # below must still be present, supported and geometrically compatible.
    windows = []
    for step in document.get('steps', []):
        for row in step.get('preconditions', []):
            if row.get('variable_id') in variables and row.get('value') == 'in_case' and row.get('evidence_id') in evidence:
                end = step['start_s']
                contrary = [evidence[a['evidence_id']]['timestamp_s'] for a in document.get('assertions', [])
                            if a.get('variable_id') in variables and a.get('value') in {'on_table', 'in_hand', 'unknown'}
                            and a.get('evidence_id') in evidence and evidence[a['evidence_id']]['timestamp_s'] < end]
                if not contrary:
                    windows.append((0., end, row['evidence_id']))
    return exact, windows


def constrain_rest(descriptor, geometry):
    """A reproducible visual packing proposal, not recovered seating/contact.

    Preserve source rotations and the relative size ratio of all removable
    parts. A shared scale reduction is allowed only when their measured native
    support cannot fit the parent's existing measured display envelope. Every
    delta is retained beside the original source estimate.
    """
    if not descriptor or not descriptor['components'] or not geometry:
        return descriptor
    result = copy.deepcopy(descriptor)
    bounds = result['parent_native_bounds'];span = max(bounds[1][i]-bounds[0][i] for i in range(3))
    scale = result['base_display_max_dimension']/span
    low = [(bounds[0][i]-sum(row[i] for row in bounds)/2)*scale for i in range(3)]
    high = [(bounds[1][i]-sum(row[i] for row in bounds)/2)*scale for i in range(3)]
    # Same explicit display clearance as the pre-existing rim-based staging.
    clearance = .004
    floor = geometry['body_bottom_offset_y']+clearance
    rim = geometry['opening_rim_offset_y']-clearance
    rows = sorted(result['components'], key=lambda row:(row['center_offset'][0],row['component_id']))
    width = high[0]-low[0]-2*clearance;depth = high[2]-low[2]-2*clearance;height = rim-floor
    if min(width, depth, height) <= 0:
        raise ValueError('Native parent envelope cannot stage removable parts')
    fractions = {}
    for row in rows:
        cue = row.get('source_visible_rim_protrusion')
        fraction = 0.
        if cue:
            if (cue.get('method') != 'adjacent_parent_mask_top_median_vs_child_top' or
                    cue.get('source_sample_index') != row.get('source_sample_index') or
                    cue.get('parent_mask_ref') != row.get('parent_source_mask_ref') or
                    cue.get('child_mask_ref') != row.get('child_source_mask_ref') or
                    cue.get('physical_seating_verified') is not False):
                raise ValueError('Visual protrusion needs its exact selected source masks')
            fraction = cue.get('bounded_visible_height_fraction')
            if type(fraction) not in (int,float) or not math.isfinite(fraction) or not 0 <= fraction <= .5:
                raise ValueError('Source-visible protrusion fraction is not bounded')
        fractions[row['component_id']] = fraction
    factor = min(1., width/(2*sum(row['oriented_half_extents'][0] for row in rows)),
                 depth/(2*max(row['oriented_half_extents'][2] for row in rows)),
                 height/max((1-fractions[row['component_id']])*2*row['half_height'] for row in rows))
    if not 0 < factor <= 1:
        raise ValueError('Source/native support cannot define bounded visual rest')
    gap = (width-2*factor*sum(row['oriented_half_extents'][0] for row in rows))/(len(rows)+1)
    cursor = low[0]+clearance+gap
    for row in rows:
        raw_center = row['center_offset'][:];raw_span = row['display_max_dimension']
        half = [v*factor for v in row['oriented_half_extents']]
        below = [v*factor for v in row['lower_support_extents']]
        above = [v*factor for v in row['upper_support_extents']]
        row['raw_source_center_offset'] = raw_center
        row['raw_source_display_max_dimension'] = raw_span
        row['raw_source_half_height'] = row['half_height']
        row['display_max_dimension'] *= factor
        row['half_height'] *= factor;row['oriented_half_extents'] = half
        row['lower_support_extents'] = below;row['upper_support_extents'] = above
        fraction = fractions[row['component_id']]
        center_y = (rim+fraction*(below[1]+above[1])-above[1] if fraction > 0 else
                    max(floor+below[1], min(rim-above[1], raw_center[1])))
        row['center_offset'] = [cursor+below[0], center_y,
                                max(low[2]+clearance+below[2], min(high[2]-clearance-above[2], raw_center[2]))]
        cursor += 2*half[0]+gap
        row['visual_rest_translation_delta'] = [row['center_offset'][i]-raw_center[i] for i in range(3)]
        row['source_rotation_unchanged'] = not bool(row.get('orientation_hypothesis'))
    result['visual_rest_constraints'] = {'method':'measured_native_parent_envelope_packing_not_physical_cavity_fit',
        'shared_child_scale_multiplier':factor, 'existing_display_clearance':clearance,
        'floor_offset_y':floor, 'opening_rim_offset_y':rim,
        'parent_display_bounds':[low,high], 'source_rotation_unchanged':all(row['source_rotation_unchanged'] for row in rows),
        'source_visible_child_upper_region_above_rim':any(fractions.values()),
        'all_child_size_ratios_unchanged':True,'parent_or_native_geometry_changed':False,
        'actual_cavity_surface_or_lid_clearance_verified':False,'physical_contact_verified':False}
    return result


def build(context, parent_id, child_ids):
    """Read only the exact current approved scene; never search newer files."""
    derived = context.get('derived_assembly')
    if not derived:
        return None
    reference = derived['document']['scene_ref'];catalog = context['catalog']
    scene = json.loads(catalog._bytes(reference, maximum=6 * 1024 * 1024))
    if scene.get('source', {}).get('sha256') != context['binding']['source_sha256'] or scene.get('original_program') != context['binding']['program_ref']:
        raise ValueError('Source-relative staging requires the exact curated scene/program')
    components = {c['id']: c for c in scene['components']}
    for cid in [parent_id, *child_ids]:
        if cid not in components or components[cid]['asset'] != context['assets'][cid]['asset_ref']:
            raise ValueError('Source-relative staging native inventory differs')
    key = (reference['sha256'], context['binding']['program_ref']['sha256'],
           tuple((cid, context['assets'][cid]['asset_ref']['path'], context['assets'][cid]['asset_ref']['sha256'])
                 for cid in [parent_id, *child_ids]))
    with _CACHE_LOCK:
        cached = copy.deepcopy(_CACHE.get(key))
        if cached is not None:_CACHE.move_to_end(key)
    if cached is not None:
        for row in cached['components']:
            for name in ['parent_source_mask_ref', 'child_source_mask_ref']:
                catalog._checked(row[name], maximum=4 * 1024 * 1024)
        return cached
    from robot_task_geometry import native_bounds
    bounds = {cid: native_bounds(context['assets'][cid]['asset_ref'], catalog) for cid in [parent_id, *child_ids]}
    parent_poses = {p['source_sample_index']: p for p in components[parent_id]['poses']}
    rows = [];unsupported = []
    for cid in child_ids:
        temporary = {index for a in scene.get('temporary_attachments', [])
                     if a['parent_id'] == parent_id and a['child_id'] == cid
                     for index in a['source_sample_indices']}
        exact, windows = _containment_windows(context['doc'], cid)
        candidates = []
        for pose in components[cid]['poses']:
            index = pose['source_sample_index'];parent = parent_poses.get(index)
            if pose['status'] not in {'estimated', 'accepted'} or not parent or parent['status'] not in {'estimated', 'accepted'}:
                continue
            if index in temporary:rank, basis, assertion = 0, 'explicit_saved_temporary_attachment', None
            elif index in exact:rank, basis, assertion = 1, 'exact_semantic_containment_assertion', exact[index]
            else:
                window = next((w for w in windows if w[0] <= pose['timestamp_s'] < w[1]), None)
                if not window:continue
                rank, basis, assertion = 2, 'inferred_initial_containment_window', window[2]
            relative = multiply(inverse(parent['camera_from_native_mesh']), pose['camera_from_native_mesh'])
            display = component_display(bounds[parent_id], bounds[cid], relative)
            center = display['native_center_in_parent'];low, high = display['native_aabb_in_parent'];pb = bounds[parent_id]
            # Reject gauge failures such as a supposedly contained part behind
            # or below the whole case. This is only a broad overlap test.
            if not all(pb[0][i] <= center[i] <= pb[1][i] for i in [0, 2]):continue
            if not low[1] < pb[1][1] or not high[1] > pb[0][1]:continue
            quality = min(parent.get('convex_silhouette_fit_iou', 0), pose.get('convex_silhouette_fit_iou', 0))
            if quality <= 0 or not .02 <= display['display_max_dimension'] <= 1:continue
            for p in [parent, pose]:catalog._checked(p['source_mask'], maximum=4 * 1024 * 1024)
            candidates.append((rank, -quality, index, {**display, 'component_id': cid,
                'source_sample_index': index, 'timestamp_s': pose['timestamp_s'],
                'selection_basis': basis, 'semantic_evidence_id': assertion,
                'parent_source_mask_ref': parent['source_mask'], 'child_source_mask_ref': pose['source_mask'],
                'source_pose_status': 'estimated', 'quality_min_iou': quality,
                'native_bounds': bounds[cid], 'physical_contact_verified': False}))
        selected = None
        if candidates:
            from portable_twin import decode_glb_bytes
            native = decode_glb_bytes(catalog._bytes(context['assets'][cid]['asset_ref'], maximum=16 * 1024 * 1024))
            vertices = [p for primitive in native['primitives'] for p in primitive['vertices']]
            for candidate in sorted(candidates, key=lambda row: row[:3]):
                row = candidate[3]
                measured = component_display(bounds[parent_id], bounds[cid], row['parent_from_child'], vertices=vertices)
                low, high = measured['native_aabb_in_parent']
                if not low[1] < bounds[parent_id][1][1] or not high[1] > bounds[parent_id][0][1]:continue
                selected = {**row, **measured, 'support_bounds_method': 'all_exact_native_vertices_after_source_relative_transform'}
                break
        if selected:rows.append(selected)
        else:unsupported.append(cid)
    result = {'method': METHOD, 'scene_ref': reference, 'parent_id': parent_id,
            'parent_native_bounds': bounds[parent_id], 'base_display_max_dimension': .38,
            'components': rows, 'unsupported_component_ids': unsupported,
            'different_source_samples_may_be_composed': True,
            'physical_contact_verified': False, 'metric_scale_known': False}
    with _CACHE_LOCK:
        _CACHE[key] = copy.deepcopy(result)
        while len(_CACHE) > 16:_CACHE.popitem(last=False)
    return result
