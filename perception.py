"""Model-independent Stage 2 contracts and conservative mask composition."""
import numpy as np

def resolve_masks(raw_masks, track_specs, withheld_ids=()):
    """Preserve raw masks elsewhere; child parts override parent, peer overlaps stay unknown."""
    withheld=set(withheld_ids)
    candidates={k:np.asarray(v,dtype=bool).copy() for k,v in raw_masks.items() if k not in withheld}
    if not candidates:return {},None
    shapes={v.shape for v in candidates.values()}
    if len(shapes)!=1:raise ValueError('Mask size mismatch')
    shape=next(iter(shapes));ambiguous=np.zeros(shape,dtype=bool)
    children=[k for k in candidates if k!='camera']
    count=np.zeros(shape,dtype=np.uint8)
    for k in children:count+=candidates[k]
    ambiguous=count>1
    out={k:v.copy() for k,v in candidates.items()}
    for k in children:out[k]&=~ambiguous
    if 'camera' in out:
        out['camera']&=count==0
    return out,ambiguous

def validate_event_result(result, rows, track_ids):
    by_index={r['index']:r for r in rows};lo=rows[0]['timestamp_s'];hi=rows[-1]['timestamp_s']
    ids=set()
    for event in result['events']:
        if event['id'] in ids:raise ValueError('Duplicate event ID')
        ids.add(event['id'])
        if not lo<=event['start_s']<=event['end_s']<=hi:raise ValueError('Event outside sampled interval')
        if not event['participant_track_ids'] or not set(event['participant_track_ids'])<=set(track_ids):raise ValueError('Unknown event participant')
        if not event['evidence_samples'] or any(i not in by_index for i in event['evidence_samples']):raise ValueError('Unknown or missing evidence sample')
        for i in event['evidence_samples']:
            if not event['start_s']-.21<=by_index[i]['timestamp_s']<=event['end_s']+.21:raise ValueError('Evidence does not support event interval')
    for link in result['identity_links']:
        if link['from_track_id'] not in track_ids or link['to_track_id'] not in track_ids:raise ValueError('Unknown identity-link track')
        if link['from_track_id']==link['to_track_id']:raise ValueError('Identity link must connect different tracklets')
        if not link['evidence_samples'] or any(i not in by_index for i in link['evidence_samples']):raise ValueError('Unknown identity evidence')
    return True
