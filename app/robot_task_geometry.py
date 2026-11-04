"""Source-bound bounding-box actuation cues, with no physical grasp claim.

Only the existing bounded portable GLB decoder is used; PIL/trimesh/inference
are not dependencies. The resulting tiny descriptor makes a task portable.
"""
from __future__ import annotations
from collections import OrderedDict
import copy
import math
import threading

KIND = 'authored_native_bbox_surface_cue_v1'
RULE = 'bbox_face_center_farthest_from_joint_axis'
_BOUNDS = OrderedDict()
_BOUNDS_LOCK = threading.Lock()


def native_bounds(reference, catalog):
    catalog._checked(reference, maximum=16*1024*1024)
    key = reference['sha256']
    with _BOUNDS_LOCK:
        if key in _BOUNDS:
            _BOUNDS.move_to_end(key)
            return copy.deepcopy(_BOUNDS[key])
    from portable_twin import decode_glb_bytes
    decoded = decode_glb_bytes(catalog._bytes(reference, maximum=16*1024*1024))
    bounds = decoded['mesh_stats']['bounds']
    if (len(bounds) != 2 or any(len(p) != 3 for p in bounds) or
        any(not math.isfinite(v) for p in bounds for v in p) or
        max(bounds[1][i]-bounds[0][i] for i in range(3)) <= 1e-9):
        raise ValueError('Native bounds cannot define an authored actuation cue')
    with _BOUNDS_LOCK:
        _BOUNDS[key] = copy.deepcopy(bounds)
        while len(_BOUNDS) > 64:_BOUNDS.popitem(last=False)
    return copy.deepcopy(bounds)


def transform(matrix, point):
    return [sum(matrix[i][j]*point[j] for j in range(3))+matrix[i][3] for i in range(3)]


def rotate_axis(point, axis, angle):
    norm = math.sqrt(sum(v*v for v in axis));axis = [v/norm for v in axis]
    c, s = math.cos(angle), math.sin(angle);dot = sum(a*b for a,b in zip(axis, point))
    cross = [axis[1]*point[2]-axis[2]*point[1], axis[2]*point[0]-axis[0]*point[2], axis[0]*point[1]-axis[1]*point[0]]
    return [point[i]*c+cross[i]*s+axis[i]*dot*(1-c) for i in range(3)]


def rotate_quaternion(point, quaternion):
    x, y, z, w = quaternion
    cross = [y*point[2]-z*point[1], z*point[0]-x*point[2], x*point[1]-y*point[0]]
    twice = [2*v for v in cross]
    cross2 = [y*twice[2]-z*twice[1], z*twice[0]-x*twice[2], x*twice[1]-y*twice[0]]
    return [point[i]+w*twice[i]+cross2[i] for i in range(3)]


def actuation_descriptor(joint, assets, catalog):
    source = joint['source_joint'];parent = assets[joint['parent_id']]['asset_ref'];child = assets[joint['child_id']]['asset_ref']
    parent_bounds = native_bounds(parent, catalog);child_bounds = native_bounds(child, catalog)
    center = [(a+b)/2 for a,b in zip(*child_bounds)];axis = source['axis'];norm = math.sqrt(sum(v*v for v in axis));axis = [v/norm for v in axis]
    candidates = []
    for dimension in range(3):
        for side in range(2):
            point = center[:];point[dimension] = child_bounds[side][dimension]
            in_joint = transform(source['joint_from_child_zero'], point)
            dot = sum(a*b for a,b in zip(in_joint, axis))
            distance = math.sqrt(max(0, sum(v*v for v in in_joint)-dot*dot))
            candidates.append((round(distance, 12), point))
    maximum = max(row[0] for row in candidates)
    anchor = next(point for distance,point in candidates if distance == maximum)
    return {'kind': KIND, 'child_id': joint['child_id'], 'asset_sha256': child['sha256'],
            'anchor_native': anchor, 'anchor_rule': RULE, 'parent_native_bounds': parent_bounds,
            'tool_tracking': 'position_only', 'approach_fraction': .25, 'release_fraction': .8, 'sample_count': 41}


def cue_position(descriptor, joint, parent_object, parent_position, angle):
    source = joint['source_joint'];point = transform(source['joint_from_child_zero'], descriptor['anchor_native'])
    point = transform(source['parent_from_joint'], rotate_axis(point, source['axis'], angle))
    bounds = descriptor['parent_native_bounds'];center = [(a+b)/2 for a,b in zip(*bounds)]
    size = max(bounds[1][i]-bounds[0][i] for i in range(3))
    scale = parent_object['display_max_dimension']*parent_object['initial_transform']['uniform_scale']/size
    point = rotate_quaternion([(point[i]-center[i])*scale for i in range(3)], parent_object['initial_transform']['quaternion_xyzw'])
    return [parent_position[i]+point[i] for i in range(3)]


def cue_state(descriptor, joint, parent_object, parent_position, start_tool, start_angle, end_angle, progress):
    """Deterministic approach, position-only lid arc, and return to the cue."""
    start = descriptor['approach_fraction'];release = descriptor['release_fraction']
    smooth = lambda t:t*t*(3-2*t)
    def mix(a,b,t):return [a[i]+(b[i]-a[i])*t for i in range(3)]
    if progress < start:
        t = smooth(progress/start);angle = start_angle
        point = mix(start_tool, cue_position(descriptor,joint,parent_object,parent_position,angle),t);gripper = t
    elif progress < release:
        t = smooth((progress-start)/(release-start));angle = start_angle+(end_angle-start_angle)*t
        point = cue_position(descriptor,joint,parent_object,parent_position,angle);gripper = 1
    else:
        t = smooth((progress-release)/(1-release));angle = end_angle
        point = mix(cue_position(descriptor,joint,parent_object,parent_position,angle),start_tool,t);gripper = 1-t
    return {'position': point, 'angle_rad': angle, 'gripper': gripper}
