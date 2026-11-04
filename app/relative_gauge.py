"""Same-image depth ratios for a conditional, non-metric component gauge.

The unknown multiplicative depth gauge of a source frame cancels in pairwise
log differences. This does not cancel model shape, camera or depth bias.
"""
from itertools import combinations
import math
import numpy as np
from scipy.optimize import least_squares


def log_depth_measurement(predicted, rendered, support, minimum_pixels=32):
    predicted, rendered, support = np.asarray(predicted), np.asarray(rendered), np.asarray(support, bool)
    if predicted.shape != rendered.shape or support.shape != predicted.shape:
        raise ValueError('Depth and support must refer to the same source pixels')
    valid = support & np.isfinite(predicted) & np.isfinite(rendered) & (predicted > 0) & (rendered > 0)
    if valid.sum() < minimum_pixels:
        raise ValueError('Insufficient common visible depth support')
    values = np.log(predicted[valid] / rendered[valid])
    centre = float(np.median(values))
    return {'log_scale': centre, 'scale': math.exp(centre), 'support_pixels': int(valid.sum()),
            'log_mad': float(np.median(np.abs(values-centre))),
            'ratio_p10': float(np.exp(np.quantile(values,.1))),
            'ratio_p90': float(np.exp(np.quantile(values,.9)))}


def pair_edges(frames):
    pairs, seen = {}, set()
    for frame in frames:
        t = float(frame['timestamp_s'])
        if not math.isfinite(t) or t < 0 or t in seen:
            raise ValueError('Depth frames need unique finite source timestamps')
        seen.add(t)
        objects = {x['object_id']: x for x in frame['objects'] if x.get('status') == 'measured'}
        if len(objects) != sum(x.get('status') == 'measured' for x in frame['objects']):
            raise ValueError('Duplicate depth component')
        for a,b in combinations(sorted(objects),2):
            x,y=objects[a],objects[b]
            value=float(x['log_scale'])-float(y['log_scale'])
            if not math.isfinite(value):
                raise ValueError('Non-finite scale evidence')
            pairs.setdefault((a,b),[]).append({'timestamp_s':t,'log_ratio':value})
    edges=[]
    for (a,b), rows in sorted(pairs.items()):
        rows.sort(key=lambda x:x['timestamp_s'])
        # Whole timestamps, not pixels, are held out before estimating an edge.
        held=[r for i,r in enumerate(rows) if len(rows)>=3 and i%3==2]
        train=[r for r in rows if r not in held]
        fit=float(np.median([r['log_ratio'] for r in train]))
        mad=float(np.median([abs(r['log_ratio']-fit) for r in train]))
        errors=[abs(r['log_ratio']-fit) for r in held]
        accept=mad<=math.log(1.25) and all(e<=math.log(1.35) for e in errors)
        edges.append({'a':a,'b':b,'log_ratio':fit,'ratio':math.exp(fit),
                      'training_timestamps_s':[r['timestamp_s'] for r in train],
                      'heldout_timestamps_s':[r['timestamp_s'] for r in held],
                      'training_log_mad':mad,'heldout_log_errors':errors,
                      'heldout_relative_errors':[math.expm1(e) for e in errors],
                      'status':('temporally_checked_candidate' if held else 'single_or_two_frame_candidate') if accept else 'rejected_temporal_disagreement',
                      'accepted_for_conditional_graph':accept,
                      'independent_physical_validation':False})
    return edges


def solve_gauge(object_ids, edges, anchor, anchor_scale):
    nodes=list(object_ids)
    if len(nodes)!=len(set(nodes)) or anchor not in nodes or not math.isfinite(anchor_scale) or anchor_scale<=0:
        raise ValueError('Unique nodes and a finite positive explicit gauge anchor required')
    adjacency={n:set() for n in nodes}
    accepted=[e for e in edges if e['accepted_for_conditional_graph']]
    for e in accepted:
        if e['a'] not in adjacency or e['b'] not in adjacency or e['a']==e['b']:
            raise ValueError('Edge references unknown or identical components')
        adjacency[e['a']].add(e['b']);adjacency[e['b']].add(e['a'])
    connected={anchor};todo=[anchor]
    while todo:
        for n in adjacency[todo.pop()]-connected:
            connected.add(n);todo.append(n)
    free=sorted(connected-{anchor});index={n:i for i,n in enumerate(free)}
    active=[e for e in accepted if e['a'] in connected and e['b'] in connected]
    base=math.log(anchor_scale)
    def value(n,x):return base if n==anchor else float(x[index[n]])
    def residual(x):return [value(e['a'],x)-value(e['b'],x)-e['log_ratio'] for e in active]
    x=least_squares(residual,np.full(len(free),base),loss='soft_l1',f_scale=math.log(1.15)).x if free else np.empty(0)
    errors=residual(x)
    consistent=all(abs(e)<=math.log(1.35) for e in errors)
    scales={n:math.exp(value(n,x)) for n in sorted(connected)} if consistent else {}
    return {'status':('connected_conditional_pairwise_gauge' if len(connected)==len(nodes) else 'disconnected_conditional_pairwise_gauge') if consistent else 'rejected_graph_inconsistency',
            'anchor_object_id':anchor,'anchor_scale':anchor_scale,'selected_scales':scales,
            'connected_to_anchor':sorted(connected),'unplaced_components':sorted(set(nodes)-connected),
            'edge_log_residuals':errors,'all_components_connected':len(connected)==len(nodes),
            'metric_scale_verified':False,'physical_assembly_verified':False,
            'temporal_stability_verified_for_all_edges':bool(active) and all(e['heldout_timestamps_s'] for e in active),
            'policy':'One constant scale per unchanged mesh; both mesh vertices and pose translation scale together. No geometry, pose or contact adjustment.'}
