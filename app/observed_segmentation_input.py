"""Bind saved segmentation observations into a geometry-only input document.

No inference, manual seed, semantic procedure or physical identity is created.
Fresh source-bound audit verdicts are preserved; unchecked masks stay unchecked.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import observed_geometry as G


def build(root,case_directory,segmentation_directory,annotations_path,output_directory,audit_path=None):
    root=Path(root).resolve();case=Path(case_directory).resolve();seg=Path(segmentation_directory).resolve()
    annotations_path=Path(annotations_path).resolve();output=Path(output_directory).resolve()
    if any(not p.is_relative_to(root/'data') for p in [case,seg,output]) or not annotations_path.is_relative_to(root/'annotations'):
        raise ValueError('Require existing project data and annotation paths')
    if output.exists():raise ValueError('Preserve prior geometry input; use a fresh output')
    manifest=json.loads((case/'manifest.json').read_text());metadata=json.loads((seg/'metadata.json').read_text())
    annotations=json.loads(annotations_path.read_text());tracks=json.loads((seg/'tracks.json').read_text())['tracks']
    index=json.loads((case/'frames/index.json').read_text())['frames']
    if metadata['status']!='complete' or metadata['source_sha256']!=manifest['source']['sha256'] or metadata['config_sha256']!=G.digest(annotations_path):
        raise ValueError('Segmentation source/config identity mismatch')
    if metadata.get('manual_seed_boxes_used') is not False or annotations.get('manual_seed_boxes_used') is not False:
        raise ValueError('Only model-proposed saved segmentation inputs are supported')
    audits={};audit=None
    if audit_path:
        audit_path=Path(audit_path).resolve()
        if not audit_path.is_relative_to(seg):raise ValueError('Audit must belong to this segmentation run')
        audit=json.loads(audit_path.read_text())
        if audit['status']!='complete_candidate_visual_audit' or audit['source_sha256']!=manifest['source']['sha256'] or audit['source_track_index_sha256']!=G.digest(seg/'tracks.json'):
            raise ValueError('Fresh mask audit identity mismatch')
        audits={(r['track_id'],r['sample_index']):r for r in audit['result']['observations']}
    track_by_id={row['track_id']:row for row in tracks}
    specifications={row['id']:row for row in annotations['tracks']}
    if set(track_by_id)!=set(specifications):raise ValueError('Annotation and segmentation track IDs differ')
    frame_by_index={row['index']:row for row in index}
    evidence_by_index={};excluded=[]
    for tid,track in track_by_id.items():
        for observation in track['frames']:
            sample=observation['sample_index'];frame=frame_by_index[sample]
            if abs(frame['timestamp_s']-observation['timestamp_s'])>1e-7:raise ValueError('Source PTS mismatch')
            image=G.safe_file(root,str((case/frame['file']).relative_to(root)),observation['source_frame_sha256'])
            mask=G.safe_file(root,str((seg/observation['raw_mask_file']).relative_to(root)),observation['raw_mask_sha256'])
            verdict=audits.get((tid,sample),{}).get('verdict','not_model_audited')
            if observation.get('status')!='candidate_visible' or observation.get('zero_placeholder_for_missing_or_withheld') or observation.get('association',{}).get('accepted') is not True:
                excluded.append({'track_id':tid,'sample_index':sample,'reason':'source_withheld_or_missing_native_association'})
                continue
            pixels=cv2.imread(str(mask),cv2.IMREAD_GRAYSCALE)
            if pixels is None or pixels.shape!=(frame['height'],frame['width']):raise ValueError('Native mask/source dimensions differ')
            row=evidence_by_index.setdefault(sample,{'id':f'source_{sample:06d}','source_index':sample,
                  'timestamp_s':frame['timestamp_s'],'frame_path':str(image.relative_to(root)),
                  'frame_sha256':G.digest(image),'observations':[]})
            row['observations'].append({'track_id':tid,'mask_path':str(mask.relative_to(root)),
                  'mask_sha256':G.digest(mask),'audit_status':verdict,
                  'gate_status':'candidate_geometric_gate_only','mask_origin':metadata['scope'],
                  'native_instance':observation.get('api_instance'),
                  'physical_identity_verified':False})
    source_path=case/manifest['source']['file']
    if G.digest(source_path)!=manifest['source']['sha256']:raise ValueError('Source video changed')
    tracklets=[{'id':tid,'label':spec['label'],'candidate_entity_ids':spec['candidate_entity_ids'],
                'identity_status':'model_proposed_tracklet_physical_identity_unverified',
                'limitations':['Segmentation audit is visual plausibility only; physical identity and rigidity unverified.']}
               for tid,spec in specifications.items()]
    source={'path':str(source_path.relative_to(root)),'sha256':manifest['source']['sha256'],
            'original_sha256':manifest['source'].get('original_sha256'),
            'duration_s':manifest['source']['video']['duration_s'],'time_domain':'source_video_presentation_seconds'}
    program={'format':'observed_geometry_input','version':1,'case_id':case.name+'--'+seg.name,
             'source':source,'tracklets':tracklets,'evidence':[evidence_by_index[i] for i in sorted(evidence_by_index)],
             'physical_identity_verified':False,'source_masks_are_model_proposals':True}
    inputs=[case/'manifest.json',case/'frames/index.json',seg/'metadata.json',seg/'tracks.json',annotations_path]
    if audit_path:inputs.append(audit_path)
    provenance={'format':'saved_segmentation_geometry_input','version':1,'status':'complete',
                'model':metadata['model'],'provider':metadata['provider'],'device':metadata['device'],
                'source_case':str(case.relative_to(root)),'segmentation_run':str(seg.relative_to(root)),
                'input_sha256':{str(p.relative_to(root)):G.digest(p) for p in inputs},
                'implementation_sha256':G.digest(__file__),'host_timestamp_utc':datetime.now(timezone.utc).isoformat(),
                'client_task_date':'2026-10-31','new_api_calls':0,'learned_inference_calls':0,
                'manual_coordinates_or_geometry':False,'physical_identity_verified':False,
                'included_observations':sum(len(r['observations']) for r in program['evidence']),
                'excluded_observations':excluded,'fresh_visual_audit_bound':audit is not None,
                'audit_verdicts_propagated_without_manual_changes':True}
    output.mkdir(parents=True)
    (output/'geometry-input.json').write_text(json.dumps(program,indent=2)+'\n')
    (output/'metadata.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (output/'implementation.py').write_bytes(Path(__file__).read_bytes())
    return program,provenance
