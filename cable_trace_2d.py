"""Source-bound 2D cable-mask paths, without invented depth or occlusion filling.

This optional, private companion reads saved evidence only. A skeleton's two
ends are not verified cable connectors; branches and invisible spans remain
unresolved. No model, robot, shell command, or existing pipeline phase runs.
"""
from __future__ import annotations

import copy
import hashlib
import heapq
import io
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy import ndimage
import skimage
from skimage.morphology import skeletonize
from PIL import Image

import semantic_asset_companion as S

ROOT = Path(__file__).resolve().parent
FORMAT = 'interaction-twin-cable-trace-2d'
MAX_NODES = 20000
MAX_POINTS = 4096
CLAIMS = {
    'two_dimensional_only': True,
    'physical_endpoints_identified': False,
    'depth_recovered': False,
    'metric_scale_known': False,
    'three_dimensional_cable_recovered': False,
    'occluded_spans_filled': False,
    'cable_identity_verified': False,
    'material_or_physics_recovered': False,
    'public_or_owner_approved': False,
    'new_model_calls': 0,
}


def _integer(value, minimum, maximum, name):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'Bounded integer {name} required')
    return value


def _line(a, b):
    """Every integer pixel of the source-space line, including both ends."""
    x0, y0 = a
    x1, y1 = b
    dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
    dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy


def _parts(adjacency):
    remaining = set(adjacency)
    groups = []
    for seed in sorted(adjacency):
        if seed not in remaining:
            continue
        remaining.remove(seed)
        group, queue = {seed}, [seed]
        while queue:
            for neighbor in adjacency[queue.pop()]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    group.add(neighbor)
                    queue.append(neighbor)
        groups.append(group)
    return groups


def _subgraph(adjacency, nodes):
    return {node: {other: weight for other, weight in adjacency[node].items() if other in nodes}
            for node in sorted(nodes)}


def _largest(adjacency):
    groups = _parts(adjacency)
    def rank(group):
        length = sum(sum(adjacency[node].values()) for node in group) / 2
        return -length, -len(group), min(group)
    return (_subgraph(adjacency, min(groups, key=rank)) if groups else {}), len(groups)


def _prune(adjacency, maximum_length):
    """Remove only short observed terminal spurs that terminate at a junction."""
    graph = copy.deepcopy(adjacency)
    removed = 0
    for _ in range(32):
        discard = set()
        for endpoint in sorted(node for node in graph if len(graph[node]) == 1):
            chain, previous, current, length = [endpoint], None, endpoint, 0.
            while True:
                next_nodes = [n for n in graph[current] if n != previous]
                if not next_nodes:
                    break
                following = next_nodes[0]
                length += graph[current][following]
                chain.append(following)
                previous, current = current, following
                if len(graph[current]) != 2 or length > maximum_length:
                    break
            if len(graph[current]) >= 3 and length <= maximum_length:
                discard.update(chain[:-1])
        if not discard:
            break
        removed += len(discard)
        graph = _subgraph(graph, set(graph) - discard)
    return graph, removed


def _farthest(graph, start):
    distances, parent, pending = {start: 0.}, {}, [(0., start)]
    while pending:
        distance, node = heapq.heappop(pending)
        if distance != distances[node]:
            continue
        for following, weight in sorted(graph[node].items()):
            proposed = distance + weight
            if proposed < distances.get(following, math.inf):
                distances[following] = proposed
                parent[following] = node
                heapq.heappush(pending, (proposed, following))
    end = min(distances, key=lambda node: (-distances[node], node))
    return end, distances[end], parent


def _path(graph):
    """Exact tree diameter; bounded double-sweep approximation for cyclic masks."""
    edges = sum(map(len, graph.values())) // 2
    is_tree = edges == len(graph) - 1
    start, _, _ = _farthest(graph, min(graph))
    best = None
    for _ in range(1 if is_tree else 3):
        end, distance, parents = _farthest(graph, start)
        ordered, current = [end], end
        while current != start:
            current = parents[current]
            ordered.append(current)
        candidate = (distance, ordered)
        if best is None or distance > best[0]:
            best = candidate
        start = end
    return best[1], best[0], is_tree


def trace_mask(mask, *, max_long_edge=960, prune_spur_px=6):
    """Find an observed continuous principal path in a single binary mask.

    Coordinates are original mask pixels with origin at its top left. Exact
    nearest-pixel downsampling is allowed, but an edge is kept only if every
    source pixel on its segment is foreground. No missing component is joined.
    """
    _integer(max_long_edge, 64, 960, 'max_long_edge')
    _integer(prune_spur_px, 0, 24, 'prune_spur_px')
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.dtype.kind not in 'bui':
        raise ValueError('A two-dimensional binary integer mask is required')
    height, width = mask.shape
    if not 1 <= height <= 4096 or not 1 <= width <= 4096 or height * width > 12000000:
        raise ValueError('Mask dimensions exceed bounded local tracing')
    values = np.unique(mask)
    if not set(map(int, values)) <= {0, 1, 255} or (1 in values and 255 in values):
        raise ValueError('Only an actual binary mask, not logits or probabilities, is accepted')
    original = mask != 0
    foreground = int(original.sum())
    result = {
        'status': 'withheld_insufficient_mask', 'points_px': [],
        'coordinate_system': 'original_binary_mask_pixels_top_left_xy',
        'mask_size_px': [width, height], 'working_size_px': [width, height],
        'source_foreground_pixels': foreground,
        'source_component_count': 0, 'working_component_count': 0,
        'skeleton_nodes': 0, 'pruned_spur_nodes': 0, 'retained_skeleton_nodes': 0,
        'branch_nodes': 0, 'observed_endpoint_count': 0, 'path_length_px': 0.,
        'principal_path_fraction_of_retained_skeleton': None,
        'path_method': None, 'path_is_complete_mask': False,
        'endpoint_identity': 'unassigned_lexicographic_order_not_connector_identity',
        'endpoint_uncertain': True, 'flags': [],
    }
    if foreground < 3:
        result['flags'] = ['empty_or_too_small_mask']
        return result
    if foreground / (width * height) > .45:
        result.update(status='withheld_bulk_or_full_frame_mask', flags=['mask_covers_over_45_percent_of_full_frame'])
        return result
    labels, component_count = ndimage.label(original, structure=np.ones((3, 3), dtype=np.uint8))
    result['source_component_count'] = int(component_count)
    scale = min(1., max_long_edge / max(width, height))
    working_width, working_height = max(1, round(width * scale)), max(1, round(height * scale))
    xs = np.minimum(width - 1, np.floor((np.arange(working_width) + .5) * width / working_width).astype(int))
    ys = np.minimum(height - 1, np.floor((np.arange(working_height) + .5) * height / working_height).astype(int))
    working_labels = labels[ys[:, None], xs[None, :]]
    skeleton = skeletonize(working_labels != 0)
    positions = np.argwhere(skeleton)
    result.update(working_size_px=[working_width, working_height], skeleton_nodes=int(len(positions)))
    if scale < 1:
        result['flags'].append('nearest_pixel_reduction_may_omit_thin_visible_spans')
    if component_count > 1:
        result['flags'].append('disconnected_source_mask_spans_not_joined')
    if len(positions) < 2 or len(positions) > MAX_NODES:
        result.update(status='withheld_skeleton_complexity' if len(positions) > MAX_NODES else 'withheld_insufficient_mask')
        return result
    lookup = {tuple(position): i for i, position in enumerate(positions)}
    source_xy = [(int(xs[x]), int(ys[y])) for y, x in positions]
    adjacency = {i: {} for i in range(len(positions))}
    for node, (y, x) in enumerate(positions):
        for dy, dx in [(0, 1), (1, -1), (1, 0), (1, 1)]:
            following = lookup.get((y + dy, x + dx))
            if following is None or working_labels[y, x] != working_labels[y + dy, x + dx]:
                continue
            # A diagonal beside an existing orthogonal skeleton route would
            # introduce a three-pixel cycle without new observed connectivity.
            if dy and dx and ((y + dy, x) in lookup or (y, x + dx) in lookup):
                continue
            if not all(original[py, px] for px, py in _line(source_xy[node], source_xy[following])):
                continue
            weight = math.hypot(dx, dy)
            adjacency[node][following] = adjacency[following][node] = weight
    selected, group_count = _largest(adjacency)
    result['working_component_count'] = group_count
    if group_count > 1:
        result['flags'].append('multiple_skeleton_components_only_one_continuous_path_selected')
    selected, pruned = _prune(selected, prune_spur_px)
    selected, _ = _largest(selected)
    result.update(pruned_spur_nodes=pruned, retained_skeleton_nodes=len(selected))
    if len(selected) < 2:
        return result
    result['branch_nodes'] = sum(len(neighbors) > 2 for neighbors in selected.values())
    result['observed_endpoint_count'] = sum(len(neighbors) == 1 for neighbors in selected.values())
    ordered, _, is_tree = _path(selected)
    points = [source_xy[node] for node in ordered]
    if points[-1] < points[0]:
        points.reverse()
    if len(points) > MAX_POINTS:
        result.update(status='withheld_path_complexity', flags=result['flags'] + ['path_over_4096_points_not_simplified_or_bridged'])
        return result
    if result['branch_nodes']:
        result['flags'].append('branch_or_self_overlap_endpoint_assignment_ambiguous')
    if not is_tree:
        result['flags'].append('cyclic_projection_approximate_geodesic_not_a_physical_open_cable')
    if any(x <= 1 or y <= 1 or x >= width - 2 or y >= height - 2 for x, y in [points[0], points[-1]]):
        result['flags'].append('path_endpoint_near_frame_edge_may_be_truncated')
    if pruned:
        result['flags'].append('short_terminal_skeleton_spurs_removed_not_missing_cable_filled')
    length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    result.update(status='observed_2d_path_candidate', points_px=[list(p) for p in points],
                  path_length_px=length,
                  principal_path_fraction_of_retained_skeleton=len(ordered) / len(selected),
                  path_method='exact_tree_geodesic_diameter' if is_tree else 'bounded_double_sweep_geodesic_approximation')
    return result


def _image(reference, root, *, binary=False):
    raw = S.checked(reference, root, 16 * 1024 * 1024)
    with Image.open(io.BytesIO(raw)) as image:
        if image.width > 4096 or image.height > 4096 or image.width * image.height > 12000000:
            raise ValueError('Bounded source image dimensions required')
        if binary:
            if image.format != 'PNG' or image.mode not in {'1', 'L', 'P'}:
                raise ValueError('Saved single-channel binary PNG mask required')
            value = np.asarray(image.convert('L'))
            values = set(map(int, np.unique(value)))
            if not values <= {0, 1, 255} or {1, 255} <= values:
                raise ValueError('Mask pixels are not binary')
            return value
        return [image.width, image.height]


def _selected_frames(program, evidence_ids, maximum):
    evidence = program['evidence']
    if evidence_ids is not None:
        if (not isinstance(evidence_ids, (tuple, list)) or not 1 <= len(evidence_ids) <= maximum or
            any(not isinstance(e, str) for e in evidence_ids) or len(set(evidence_ids)) != len(evidence_ids)):
            raise ValueError('Explicit unique bounded evidence IDs required')
        lookup = {e['id']: e for e in evidence}
        if not set(evidence_ids) <= set(lookup):
            raise ValueError('Cable evidence belongs to another program')
        return sorted([lookup[e] for e in evidence_ids], key=lambda row: row['timestamp_s'])
    if len(evidence)>maximum:
        first,last=evidence[0]['timestamp_s'],evidence[-1]['timestamp_s']
        indices=sorted({min(range(len(evidence)),key=lambda index:(abs(evidence[index]['timestamp_s']-(first+i*(last-first)/(maximum-1))),index))
                        for i in range(maximum)})
    else:indices=range(len(evidence))
    return [evidence[i] for i in indices]


def build(program_path, track_id, *, evidence_ids=None, max_frames=90, max_long_edge=960, prune_spur_px=6, root=ROOT):
    """Explicit track selection; no name guessing, inference, tracking or gap fill."""
    _integer(max_frames, 2, 90, 'max_frames')
    _integer(max_long_edge, 64, 960, 'max_long_edge')
    _integer(prune_spur_px, 0, 24, 'prune_spur_px')
    reference, program = S._program(program_path, root)
    track = next((t for t in program['tracklets'] if t['id'] == track_id), None)
    if not isinstance(track_id, str) or track is None:
        raise ValueError('Choose one exact saved track ID for cable-mask interpretation')
    frames = _selected_frames(program, evidence_ids, max_frames)
    samples = []
    for frame in frames:
        item = {
            'evidence_id': frame['id'], 'source_index': frame['source_index'], 'timestamp_s': frame['timestamp_s'],
            'frame_ref': S.file_ref(frame['frame_path'], root, 16 * 1024 * 1024),
            'mask_ref': None, 'gate_status': None, 'audit_status': None,
            'frame_size_px': None, 'trace': None,
            'status': 'withheld_no_saved_track_mask',
            'between_samples': 'absent_no_temporal_interpolation_or_endpoint_identity_assignment',
        }
        observations = [o for o in frame['observations'] if o['track_id'] == track_id]
        if len(observations) > 1:
            raise ValueError('Ambiguous duplicate masks for one saved track and source sample')
        if observations:
            observation = observations[0]
            item.update(mask_ref=S.file_ref(observation['mask_path'], root, 16 * 1024 * 1024),
                        gate_status=observation['gate_status'], audit_status=observation['audit_status'])
            if any(any(word in item[field].lower() for word in ['withheld', 'reject', 'error', 'missing']) for field in ['gate_status', 'audit_status']):
                item['status'] = 'withheld_saved_observation_gate'
            else:
                item['frame_size_px'] = _image(item['frame_ref'], root)
                mask = _image(item['mask_ref'], root, binary=True)
                if item['frame_size_px'] != [mask.shape[1], mask.shape[0]]:
                    item['status'] = 'withheld_frame_mask_dimension_mismatch'
                else:
                    item['trace'] = trace_mask(mask, max_long_edge=max_long_edge, prune_spur_px=prune_spur_px)
                    item['status'] = item['trace']['status']
        samples.append(item)
    return {
        'format': FORMAT, 'version': 1, 'program_ref': reference,
        'source_ref': S.file_ref(program['source']['path'], root), 'source_sha256': program['source']['sha256'],
        'case_id': program['case_id'], 'track_id': track_id, 'track_label': track['label'],
        'track_role': 'operator_selected_cable_hypothesis_not_automatic_identity_confirmation',
        'parameters': {'max_frames': max_frames, 'max_long_edge': max_long_edge, 'prune_spur_px': prune_spur_px,
                       'max_skeleton_nodes': MAX_NODES, 'max_path_points': MAX_POINTS},
        'prune_spur_units': 'working_mask_pixels_only_not_metric_length',
        'selection': {'method': 'explicit_evidence_ids' if evidence_ids is not None else 'deterministic_source_time_quantiles',
                      'evidence_ids': [e['id'] for e in frames], 'program_evidence_count': len(program['evidence'])},
        'implementation': {'algorithm': 'cable_trace_2d_v1', 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           'numpy': np.__version__, 'scipy': scipy.__version__, 'skimage': skimage.__version__},
        'samples': samples, 'claims': copy.deepcopy(CLAIMS),
        'limitations': [
            'A visible SAM silhouette is fallible evidence; this path is not a reconstructed 3D rod.',
            'Masks can include hands, connectors and background; principal skeleton paths can be ambiguous.',
            'Disconnected spans, crossings, loops, border truncation and unknown physical endpoints remain explicit.',
            'No invisible spans, future-frame interpolation, connector orientation, length in metres or material parameters are inferred.',
            'An existing static SAM 3D appearance asset does not provide measured cable deformation.',
        ],
    }


def validate(value, *, root=ROOT):
    if not isinstance(value, dict) or value.get('format') != FORMAT or type(value.get('version')) is not int or value['version'] != 1:
        raise ValueError('Unknown cable trace companion')
    for key in ['program_ref', 'source_ref']:
        S.checked(value.get(key), root, 512 * 1024 * 1024)
    parameters = value.get('parameters')
    if not isinstance(parameters, dict) or set(parameters) != {'max_frames', 'max_long_edge', 'prune_spur_px', 'max_skeleton_nodes', 'max_path_points'}:
        raise ValueError('Unknown cable trace parameters')
    selection = value.get('selection')
    if not isinstance(selection, dict) or selection.get('method') not in {'explicit_evidence_ids', 'deterministic_source_time_quantiles'}:
        raise ValueError('Unknown cable evidence selection')
    expected = build(value['program_ref']['path'], value.get('track_id'),
                     evidence_ids=selection.get('evidence_ids') if selection['method'] == 'explicit_evidence_ids' else None,
                     max_frames=parameters['max_frames'], max_long_edge=parameters['max_long_edge'],
                     prune_spur_px=parameters['prune_spur_px'], root=root)
    if value != expected:
        raise ValueError('Cable path differs from the exact saved source/masks or declared deterministic algorithm')
    return True


def write(value, output_dir, *, root=ROOT):
    validate(value, root=root)
    return S._write(value, output_dir, root, 'cable-trace-2d')


def load(path, *, root=ROOT):
    reference = S.file_ref(path, root, 8 * 1024 * 1024)
    value = S._json(S.checked(reference, root))
    validate(value, root=root)
    return value
