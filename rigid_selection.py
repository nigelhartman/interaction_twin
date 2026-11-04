"""Attest exact model rigid-component seeds and omission-only derivatives.

No localization, pixel boxes, masks or source records are modified. This adapter
supports the explicitly versioned rigid policy; unknown config variants fail.
"""
from pathlib import Path
import hashlib
import json
import re

from component_seed_review import derive
from scripts.stage2_meta_reason import tracking_config, validate_discovery

POLICY = 'rigid_component_visual_discovery_v1'
DERIVATIVE_ORIGIN = 'meta_api_model_proposal_with_visual_audit_omission_only'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def confined(case, path):
    case, path=Path(case).resolve(),Path(path).resolve()
    if not path.is_relative_to(case) or not path.is_file():
        raise ValueError('Rigid selection input escapes the source case or is missing')
    return path


def _digest(value):
    if not isinstance(value,str) or not re.fullmatch(r'[a-f0-9]{64}',value):
        raise ValueError('Missing exact source/config hash')
    return value


def _sam_receipt(case, run, source):
    config_path=confined(case,run/'annotations.json')
    metadata_path=confined(case,run/'metadata.json')
    cfg,meta=read(config_path),read(metadata_path)
    if (meta.get('status')!='complete' or meta.get('source_sha256')!=source or
            meta.get('config_sha256')!=sha(config_path) or
            meta.get('seed_origin')!=cfg.get('seed_origin')):
        raise ValueError('SAM receipt/config/source identity mismatch')
    if (meta.get('manual_seed_boxes_used') is not False or meta.get('manual_geometry_used') is not False or
            meta.get('sam_video_propagation_executed') is not True or
            meta.get('learned_inference_executed') is not True or
            cfg.get('manual_seed_boxes_used') is not False or cfg.get('manual_geometry_used') is not False):
        raise ValueError('Actual automatic SAM video provenance is required')
    return cfg,meta


def _canonical(case,cfg,source,rows):
    target=_digest(cfg.get('discovery_record_sha256'))
    matches=[confined(case,p) for p in sorted((case/'reasoning').glob('*/discovery.json')) if sha(p)==target]
    if not matches:
        raise ValueError('No hash-matched original rigid model discovery')
    discovery=matches[0]
    record=read(discovery)
    if (record.get('provider')!='meta_api' or record.get('inference_location')!='hosted_Meta_Model_API' or
            record.get('status')!='complete_candidate_proposals' or record.get('source_sha256')!=source or
            record.get('frame_index_sha256')!=sha(case/'frames/index.json') or record.get('role')!=POLICY):
        raise ValueError('Rigid discovery provider/source/frame identity mismatch')
    validate_discovery(record['result'],rows)
    expected=tracking_config(record['result'],rows,read(case/'manifest.json')['case_id'],target)
    expected['component_policy']=POLICY
    if cfg!=expected:
        raise ValueError('Rigid seeds differ from exact canonical model configuration')
    return discovery,record


def validate_config(case,run):
    case,run=Path(case).resolve(),Path(run).resolve()
    source=read(confined(case,case/'manifest.json'))['source']['sha256']
    rows=read(confined(case,case/'frames/index.json'))['frames']
    cfg,meta=_sam_receipt(case,run,source)
    lineage={'adapter':'rigid_component_selection_v1','omission_only_derivative':False,
             'manual_boxes_used':False,'physical_identity_verified':False}
    if cfg.get('seed_origin')=='meta_api_model_proposal':
        discovery,record=_canonical(case,cfg,source,rows)
    elif cfg.get('seed_origin')==DERIVATIVE_ORIGIN:
        review=cfg.get('seed_review',{})
        targets={key:_digest(review.get(key)) for key in
                 ('source_config_sha256','source_tracks_sha256','source_audit_sha256')}
        parents=[]
        for tracks in sorted((case/'segmentation').glob('*/tracks.json')):
            if sha(tracks)!=targets['source_tracks_sha256']:
                continue
            parent=tracks.parent
            config=confined(case,parent/'annotations.json')
            audit=confined(case,parent/'model-audit-v1/audit.json')
            if sha(config)==targets['source_config_sha256'] and sha(audit)==targets['source_audit_sha256']:
                parents.append(parent)
        if not parents:
            raise ValueError('No exact parent SAM/config/audit lineage for omission derivative')
        parent=parents[0]
        parent_cfg,parent_meta=_sam_receipt(case,parent,source)
        if parent_cfg.get('seed_origin')!='meta_api_model_proposal':
            raise ValueError('Only a single canonical-to-omission derivation is supported')
        discovery,record=_canonical(case,parent_cfg,source,rows)
        audit_path=confined(case,parent/'model-audit-v1/audit.json')
        audit=read(audit_path)
        if audit.get('source_sha256')!=source:
            raise ValueError('Parent audit source mismatch')
        expected,changes=derive(parent_cfg,audit,targets['source_tracks_sha256'])
        expected['seed_review']={**targets,'changes':changes,'manual_seed_boxes_used':False,'new_boxes_added':False}
        if cfg!=expected:
            raise ValueError('Omission derivative changes fields or model seed coordinates')
        lineage.update(omission_only_derivative=True,parent_run=str(parent.relative_to(case)),
                       source_config_sha256=targets['source_config_sha256'],
                       source_tracks_sha256=targets['source_tracks_sha256'],
                       source_audit_sha256=targets['source_audit_sha256'],omission_changes=changes)
    else:
        raise ValueError('Unknown rigid seed origin')
    lineage.update(discovery_file=str(discovery.relative_to(case)),discovery_sha256=sha(discovery),
                   discovery_implementation_role=record['role'],configuration_sha256=sha(run/'annotations.json'),
                   segmentation_metadata_sha256=sha(run/'metadata.json'))
    return cfg,meta,discovery,record,lineage
