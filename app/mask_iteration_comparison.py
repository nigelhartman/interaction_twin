"""Read-only comparison of exact saved SAM pixels and fallible model audits.

Mask agreement is not accuracy. Empty/withheld masks are counted separately,
never rewarded as a perfect empty-empty match. No inference or geometry runs.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def within(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Comparison input must be an existing confined file')
    return path


def mask_agreement(before, after):
    before, after = np.asarray(before, dtype=bool), np.asarray(after, dtype=bool)
    if before.shape != after.shape or before.ndim != 2:
        raise ValueError('Masks require identical original pixel dimensions')
    a, b = int(before.sum()), int(after.sum())
    both = a > 0 and b > 0
    return {'before_area_pixels': a, 'after_area_pixels': b,
            'state': ('both_nonempty' if both else 'before_only_nonempty' if a else
                      'after_only_nonempty' if b else 'both_empty'),
            'iou_agreement': float(np.count_nonzero(before & after) /
                                   np.count_nonzero(before | after)) if both else None}


def _unique(rows, key):
    result = {}
    for row in rows:
        k = key(row)
        if k in result:
            raise ValueError('Duplicate comparison key')
        result[k] = row
    return result


def compare(case, before_run, after_run, *, root):
    root, case = Path(root).resolve(), Path(case).resolve()
    source_path, index_path = within(root,case/'manifest.json'), within(root,case/'frames/index.json')
    source_record = read(source_path)['source']
    source = source_record['sha256']
    frames = _unique(read(index_path)['frames'], lambda r:r['index'])
    input_files = {source_path,index_path}
    video_path=within(root,case/source_record['file'])
    if sha(video_path)!=source:
        raise ValueError('Original source video bytes changed')
    input_files.add(video_path)
    image_hashes = {}
    def relative(path): return str(Path(path).relative_to(root))
    loaded = []
    for name in (before_run,after_run):
        run = within(root,case/'segmentation'/name/'tracks.json').parent
        paths = {k:within(root,run/k) for k in
                 ('tracks.json','frames.json','annotations.json','metadata.json',
                  'model-audit-v1/audit.json','model-audit-v1/evidence.json')}
        input_files.update(paths.values())
        tracks, metadata, audit = (read(paths[k]) for k in
                                  ('tracks.json','metadata.json','model-audit-v1/audit.json'))
        if (metadata['source_sha256'] != source or audit['source_sha256'] != source or
                audit['status'] != 'complete_candidate_visual_audit' or
                audit['source_track_index_sha256'] != sha(paths['tracks.json']) or
                metadata['config_sha256'] != sha(paths['annotations.json'])):
            raise ValueError('Run/audit/config source identity mismatch')
        raw = _unique((dict(r,track_id=t['track_id']) for t in tracks['tracks'] for r in t['frames']),
                      lambda r:(r['track_id'],r['sample_index']))
        for key,row in raw.items():
            f = frames[row['sample_index']]
            image = within(root,case/f['file'])
            if f['index'] not in image_hashes:
                image_hashes[f['index']]=sha(image)
                with Image.open(image) as source_image:
                    if source_image.size!=(f['width'],f['height']):
                        raise ValueError('Source image/index dimensions differ')
            mask = within(root,run/row['raw_mask_file'])
            if (abs(row['timestamp_s']-f['timestamp_s']) > 1e-7 or
                    row['source_frame_sha256'] != image_hashes[f['index']] or
                    row['raw_mask_sha256'] != sha(mask)):
                raise ValueError('Original source/mask pixels or PTS changed')
            row['_mask_path'],row['_image_path'] = mask,image
        evidence = _unique(read(paths['model-audit-v1/evidence.json']),
                           lambda r:(r['track_id'],r['sample_index']))
        verdicts = _unique(audit['result']['observations'],lambda r:(r['track_id'],r['sample_index']))
        if set(verdicts) != set(evidence):
            raise ValueError('Audit verdict/evidence sets differ')
        for key,row in evidence.items():
            r=raw[key]
            if (row['source_frame_sha256'] != r['source_frame_sha256'] or
                    row['raw_mask_sha256'] != r['raw_mask_sha256'] or
                    abs(row['timestamp_s']-r['timestamp_s']) > 1e-7):
                raise ValueError('Visual audit pixels are not exact run pixels')
        for item in audit['input_images']:
            path=within(root,run/'model-audit-v1'/item['file'])
            if sha(path)!=item['sha256']:
                raise ValueError('Visual audit contact sheet identity changed')
            input_files.add(path)
        gates={(f['sample_index'],t['track_id']): t['status'] for f in read(paths['frames.json'])['frames']
               for t in f['tracks']}
        loaded.append({'name':name,'run':run,'raw':raw,'verdicts':verdicts,'evidence':evidence,
                       'gates':gates,'metadata':metadata,
                       'audit_verdict_counts':dict(Counter(r['verdict'] for r in verdicts.values()))})
    before,after=loaded
    keys=sorted(set(before['raw'])&set(after['raw']))
    rows=[]
    for key in keys:
        a,b=before['raw'][key],after['raw'][key]
        if a['source_frame_sha256'] != b['source_frame_sha256'] or abs(a['timestamp_s']-b['timestamp_s'])>1e-7:
            raise ValueError('Common sample does not have identical source pixels/PTS')
        f=frames[key[1]]
        masks=[np.asarray(Image.open(r['_mask_path']).convert('L'))>0 for r in (a,b)]
        if any(m.shape != (f['height'],f['width']) for m in masks):
            raise ValueError('Mask is not in the original full-frame dimensions')
        metrics=mask_agreement(*masks)
        if [metrics['before_area_pixels'],metrics['after_area_pixels']] != [a['raw_area_pixels'],b['raw_area_pixels']]:
            raise ValueError('Stored mask area does not match native pixels')
        row={'track_id':key[0],'sample_index':key[1],'timestamp_s':a['timestamp_s'],
             'source_image_url':'/files/'+relative(a['_image_path']),
             'source_frame_sha256':a['source_frame_sha256'],**metrics}
        for label,data,r in [('before',before,a),('after',after,b)]:
            row[label]={'mask_url':'/files/'+relative(r['_mask_path']), 'mask_sha256':r['raw_mask_sha256'],
                        'native_returned_instance_preserved':r['native_returned_instance_preserved'],
                        'zero_placeholder_for_missing_or_withheld':r['zero_placeholder_for_missing_or_withheld'],
                        'status':r['status'],'geometric_gate':data['gates'].get((key[1],key[0])),
                        'audit':data['verdicts'].get(key),
                        'audit_crop_xyxy':data['evidence'].get(key,{}).get('crop_xyxy')}
        row['audit_paired'] = key in before['verdicts'] and key in after['verdicts']
        row['audit_crop_identical']=(row['before']['audit_crop_xyxy']==row['after']['audit_crop_xyxy']) if row['audit_paired'] else None
        rows.append(row)
    summaries=[]
    for tid in sorted({r['track_id'] for r in rows}):
        subset=[r for r in rows if r['track_id']==tid]
        paired=[r for r in subset if r['audit_paired']]
        iou=[r['iou_agreement'] for r in subset if r['iou_agreement'] is not None]
        summary={'track_id':tid,'common_exact_source_samples':len(subset),
                 'mask_states':dict(Counter(r['state'] for r in subset)),
                 'both_nonempty_iou_agreement_median':float(np.median(iou)) if iou else None,
                 'paired_model_audit_count':len(paired),
                 'paired_audit_identical_crop_count':sum(r['audit_crop_identical'] for r in paired),
                 'paired_verdict_transitions':dict(Counter(r['before']['audit']['verdict']+' -> '+
                                                          r['after']['audit']['verdict'] for r in paired))}
        for label in ('before','after'):
            summary[label]={'native_returned_observations':sum(r[label]['native_returned_instance_preserved'] for r in subset),
                            'zero_placeholders':sum(r[label]['zero_placeholder_for_missing_or_withheld'] for r in subset),
                            'geometric_gate_candidates':sum(r[label]['geometric_gate']=='candidate_geometric_gate_only' for r in subset),
                            'audited_common_source_samples':sum(r[label]['audit'] is not None for r in subset)}
        summaries.append(summary)
    return {'format':'exact-source-mask-iteration-comparison','version':1,
            'source_video_sha256':source,'source_case':relative(case),
            'before_run':before_run,'after_run':after_run,
            'method_frozen_before_data_comparison':True,
            'implementation_sha256':sha(__file__),
            'matching_protocol':'Same proposed track ID, sample index, PTS (1e-7 s), exact full-frame RGB bytes and native dimensions. No resampling or temporal interpolation.',
            'input_files':[{'path':relative(p),'sha256':sha(p)} for p in sorted(input_files)],
            'unmatched_before_samples':len(set(before['raw'])-set(after['raw'])),
            'unmatched_after_samples':len(set(after['raw'])-set(before['raw'])),
            'common_exact_source_samples':len(rows),
            'audit_counts_unmatched_descriptive_only':{
                'before':before['audit_verdict_counts'],'after':after['audit_verdict_counts']},
            'tracks':summaries,'frames':rows,
            'measured_mask_accuracy':False,'physical_identity_verified':False,
            'limitations':[
                'Common proposed IDs do not establish the same physical component through occlusion.',
                'IoU is agreement between fallible predictions, not accuracy or a quality score.',
                'Empty or withheld masks are unknown. Empty-empty observations receive no IoU score.',
                'Audit sample selection varies by run. Only paired exact-source observations are compared.',
                'Even paired audits may use different mask-dependent crops and context; verdict transitions are descriptive, not independent accuracy measurements.',
                'The second run used feedback from the first model audit; the first audit is development feedback, not a held-out test.',
                'All source frames/masks and audit contact-sheet hashes were checked; no human ground truth exists.',
                'No 3D shape, metric scale, contact, assembly or motion accuracy is evaluated.']}
