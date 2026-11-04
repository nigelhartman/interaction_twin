#!/usr/bin/env python3
"""Index existing isolated results; never infer or change reconstructed geometry."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/observed-geometry-experiment-20261031'
EXPERIMENT_BASES=[BASE,ROOT/'data/observed-geometry-calibration-20261031',
                  ROOT/'data/observed-surface-experiment-20261031',ROOT/'data/observed-moving-parts-20261031',
                  ROOT/'data/observed-app-runs-20261031',ROOT/'data/observed-cotracker-experiment-20261101']

def mask_provider(protocol, seen=None):
    """Follow declared experiment provenance, never infer a new model result."""
    seen=set() if seen is None else seen
    name=protocol.get('program_path') or protocol.get('source_program')
    if name:
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT/'data'):
            raise ValueError('Geometry input is outside project data')
        if path.name=='geometry-input.json':
            metadata=json.loads((path.parent/'metadata.json').read_text())
            return 'Local '+metadata['model']+' on '+metadata['device']+' · saved SAM 3.1 mask seed'
        return 'Saved Meta SAM 3.1 video masks'
    source=protocol.get('sparse_input') or protocol.get('source_directory')
    if source:
        path=(ROOT/source/'protocol.json').resolve()
        if not path.is_relative_to(ROOT/'data') or path in seen:
            raise ValueError('Invalid geometry provenance chain')
        seen.add(path)
        return mask_provider(json.loads(path.read_text()),seen)
    return 'Provider not declared in saved protocol'

def main():
    runs=[]
    paths=[p for base in EXPERIMENT_BASES for p in base.glob('*/viewer.json')]
    for path in sorted(paths,key=lambda p:p.stat().st_mtime,reverse=True):
        original=path
        display=path.with_name('viewer-display.json')
        if display.is_file():
            attestation=json.loads(path.with_name('display-provenance.json').read_text())
            for key,file in [('source_viewer_sha256',path),('source_protocol_sha256',path.with_name('protocol.json')),
                             ('display_viewer_sha256',display),('implementation_sha256',path.with_name('display-implementation.py'))]:
                if hashlib.sha256(file.read_bytes()).hexdigest()!=attestation[key]:
                    raise ValueError('Display derivative identity mismatch: '+str(file))
            path=display
        result=json.loads(path.read_text())
        if result.get('inferred_surface') is not None:
            # Completion is selected as a separate layer of its bound surface.
            continue
        protocol=json.loads((path.parent/'protocol.json').read_text())
        runs.append({'id':result['id'],'label':result['title']+' · '+result['id'],
                     'status':result['status'],'viewer_url':'/files/'+str(path.relative_to(ROOT)),
                     'kind':result.get('kind','sparse_geometry'),
                     'original_viewer_url':'/files/'+str(original.relative_to(ROOT)),
                     'source_sparse_status':protocol.get('source_sparse_status'),
                     'stereo_focal_px':protocol.get('K',[[None]])[0][0],
                     'mask_provider':('Local SAM 2.1 Tiny · saved hosted model seed proposals' if path.parent.parent.name=='observed-moving-parts-20261031' else 'Saved source component SAM masks · see source program provenance' if path.parent.parent.name=='observed-app-runs-20261031' else 'Saved SAM masks · explicit learned point-track or SIFT provider in protocol' if path.parent.parent.name=='observed-cotracker-experiment-20261101' else mask_provider(protocol)),
                     'metrics':result.get('metrics',{})})
    local_masks=[]
    for path in sorted(BASE.glob('dense-camera-input-*/metadata.json')):
        metadata=json.loads(path.read_text())
        local_masks.append({'metadata_url':'/files/'+str(path.relative_to(ROOT)),
                            'status':metadata['status'],'model':metadata['model'],'device':metadata['device']})
    summary={'title':'Geometry from observed video pixels','runs':runs,
             'default_run_id':'camera-body-stereo-v1',
             'indexing_api_calls':0,'local_segmentation_runs':local_masks,'indexing_model_calls':0,
             'limitations':['Experiments are separate candidates; original app results are unchanged.',
                            'Sparse points and observed partial patches are not full objects or robot-ready meshes.',
                            'Camera intrinsics, metric scale, physical identity and hidden geometry remain unverified.',
                            'Failures are preserved. No synthetic geometry substitutes for real results.',
                            'Dense diagnostic masks use local SAM 2.1 Tiny seeded by saved SAM 3.1 masks; the geometry uses source pixels only.',
                            'Held-out checks test geometry conditional on saved segmentation, not an unseen end-to-end recording.']}
    extension_paths={'exports':'data/observed-surface-exports-20261031/manifest.json',
                     'completion_exports':'data/observed-surface-exports-20261031/completion-exports.json',
                     'validation':'data/observed-surface-validation-20261031/manifest.json',
                     'moving_parts':'data/observed-moving-parts-20261031/manifest.json',
                     'meta_review':'data/geometry-intelligence-20261031/reconstruction-review-v3/review.json',
                     'sam_parts':'data/moving-parts-sam31-20261031/jabra-localized-parts-v2/manifest.json',
                     'part_assets':'data/moving-parts-reconstruction-20261031/jabra-parts-triposr-v1/display-manifest.json'}
    extension_paths['tracking_comparison']='data/observed-cotracker-experiment-20261101/comparison-v1/comparison.json'
    summary['extensions']={k:'/files/'+v for k,v in extension_paths.items() if (ROOT/v).is_file()}
    summary['completion_candidates']=[]
    for base in ['observed-depth-completion-20261031','observed-surface-completion-20261031']:
        for path in sorted((ROOT/'data'/base).glob('*/viewer.json')):
            d=json.loads(path.read_text())
            if d.get('original_model_prior_url'):
                # An identical-geometry mask-check derivative is provenance,
                # not another reconstruction method to rank in the selector.
                summary['completion_mask_check_url']='/files/'+str((path.parent/'filter-report.json').relative_to(ROOT))
                continue
            summary['completion_candidates'].append({'id':d['id'],'source_run_id':'camera-body-stereo-v1',
                'label':d['title'],'viewer_url':'/files/'+str(path.relative_to(ROOT)),
                'status':d['status'],'metrics':d.get('metrics',{})})
    workflow_exports=[]; workflow_validation=[]; workflow_completions=[]
    for job_path in sorted((ROOT/'data/observed-app-runs-20261031').glob('observed-*/job.json')):
        job=json.loads(job_path.read_text())
        if job.get('workflow') != 'gated-local-v1' and not job.get('workflow_manifest_url'):
            continue
        if job.get('exports',{}).get('observed'):
            workflow_exports.append(job['exports']['observed'])
        if job.get('exports',{}).get('inferred'):
            workflow_completions.append(job['exports']['inferred'])
        surface_id=job['id']+'-dense'
        for stage in job.get('stages',[]):
            result=stage.get('result',{})
            if stage['stage']=='validation' and result.get('report_url'):
                workflow_validation.append({'source_run_id':surface_id,'report_url':result['report_url']})
            if stage['stage']=='completion' and result.get('viewer_url'):
                summary['completion_candidates'].append({'id':job['id']+'-completion','source_run_id':surface_id,
                    'label':'Optional workflow completion','viewer_url':result['viewer_url'],'status':stage['status']})
    summary['workflow_exports']=workflow_exports
    summary['workflow_validation']=workflow_validation
    summary['workflow_completion_exports']=workflow_completions
    BASE.mkdir(parents=True,exist_ok=True)
    (BASE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'runs':len(runs),'summary':str((BASE/'summary.json').relative_to(ROOT))}))

if __name__=='__main__': main()
