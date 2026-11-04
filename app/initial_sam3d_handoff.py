"""Freeze an initial SAM3D source anchor without an existing 3D asset.

Consumes the pipeline's existing selection + portable RGB/mask input. No model,
baseline mesh, pose, credential, remote command or network operation is used.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re

import numpy as np

import upgrade_bundles as U
from scripts.stage3_sam3d import read_inputs

ROOT = Path(__file__).resolve().parent
FORMAT = 'sam3d-initial-reconstruction-handoff'
DEPENDENCIES = ('initial_sam3d_handoff.py','scripts/stage3_sam3d.py','scripts/stage3_reconstruction_bundle.py')
FILES = {'input.json','image.png','mask.png','provenance/selection.json','provenance/case-manifest.json',
         'provenance/frame-index.json','provenance/source-frame.jpg','provenance/source-mask.png',
         'adapter/stage3_sam3d.py','implementation.py','source-binding.json','run-plan.json','README.txt'}


def require(value,message):
    if not value:
        raise ValueError(message)


def reference(path,root):
    path = U._local(path,root)
    return {'path':path.relative_to(root).as_posix(),'sha256':U.sha(path),'bytes':path.stat().st_size}


def inspect_anchor(input_manifest,selection_path,*,root=ROOT):
    root = Path(root).resolve()
    input_path,selection_path = U._local(input_manifest,root),U._local(selection_path,root)
    require(input_path.stat().st_size <= 1024*1024 and selection_path.stat().st_size <= 8*1024*1024,'Source JSON exceeds budget')
    manifest,selection = U.read(input_path),U.read(selection_path)
    require(isinstance(manifest,dict) and isinstance(selection,dict),'Source JSON objects required')
    require(not {'baseline','input_binding','source_binding'} & set(manifest),'Use an original initial RGB/mask input, not an upgrade/baseline bundle')
    require(manifest.get('image') == 'image.png' and manifest.get('mask') == 'mask.png','Initial input must use its own full-frame RGB/mask')
    source,provenance,choice = manifest.get('source'),manifest.get('mask_provenance'),manifest.get('selection')
    require(all(isinstance(v,dict) for v in (source,provenance,choice)),'Source, mask and selection records required')
    require(provenance.get('manual_intervention') is False and provenance.get('prompt_origin') in {'model_generated_prompt','automatic_mask_generation'},'An explicitly automatic source mask is required')
    require(choice.get('selection_sha256') == U.sha(selection_path) and choice.get('agent_visual_selection') is False
            and choice.get('manual_geometry_used') is False and choice.get('manual_camera_corners_used') is False,'Automatic selection binding differs')
    require(selection.get('object_id') == manifest.get('object_id'),'Selection component differs')
    method = selection.get('selection_method')
    require(isinstance(method,dict) and method.get('agent_visual_choice') is False
            and method.get('reads_geometry_or_corner_annotations') is False,'Selection must not use agent geometry or corner choices')
    selected = selection.get('selected')
    require(isinstance(selected,list) and 1 <= len(selected) <= 10 and all(isinstance(v,dict) for v in selected),'Bounded source selection required')
    matches = [v for v in selected if v.get('selection_rank') == choice.get('rank')]
    require(len(matches) == 1,'One exact selected rank is required')
    row = matches[0]
    require(type(choice.get('rank')) is int and choice['rank'] >= 1 and choice.get('score') == row.get('score'),'Selected rank/score differs')
    case_name = row.get('source_case')
    require(isinstance(case_name,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,159}',case_name),'Ordinary source case identifier required')
    case = root/'data'/case_name
    require(selection_path.is_relative_to(case) and input_path.is_relative_to(case),'Selection and portable input must share one source case')
    case_path,index_path = U._local(case/'manifest.json',root),U._local(case/'frames/index.json',root)
    case_record,index = U.read(case_path),U.read(index_path)
    require(isinstance(case_record,dict) and isinstance(index,dict) and case_record.get('source') == selection.get('source'),'Source case/selection identity differs')
    require(source.get('video_sha256') == selection['source'].get('sha256') and source.get('source_case') == case_name,'Portable input source video differs')
    for field in ('sample_index','timestamp_s','source_sample_index'):
        require(source.get(field) == row.get(field),'Source PTS/sample identity differs')
    require(row.get('source_track_id') == row.get('object_id') == manifest['object_id'],'Source component identity differs')
    frames = index.get('frames')
    require(isinstance(frames,list) and all(isinstance(f,dict) for f in frames),'Source frame index required')
    frame_rows = [f for f in frames if f.get('index') == row['source_sample_index']]
    require(len(frame_rows) == 1 and frame_rows[0].get('timestamp_s') == row['timestamp_s']
            and frame_rows[0].get('file') == row.get('image_file') and index.get('time_domain') == U.TIME_DOMAIN,'Source frame index/PTS differs')
    exports = row.get('exports')
    require(isinstance(exports,dict) and exports.get('resized') is False and exports.get('background_invented') is False,'Full native source exports required')
    for role in ('image','mask'):
        local = U._local(input_path.parent/manifest[role],root)
        export_key = 'rgb' if role == 'image' else 'mask'
        exported = U._local(selection_path.parent/exports[export_key],root)
        require(U.sha(local) == manifest.get(role+'_sha256') == exports.get(export_key+'_sha256') == U.sha(exported),'Portable source pixel file differs')
    record,rgb,mask = read_inputs(input_path)
    require(exports.get('rgb_mask_dimensions') == [record['width'],record['height']],'Source input dimensions differ')
    raw_rgb,raw_mask = U._local(case/row['image_file'],root),U._local(case/row['mask_file'],root)
    require(U.sha(raw_rgb) == row.get('image_sha256') and U.sha(raw_mask) == row.get('mask_sha256'),'Original frame/SAM file differs')
    original_rgb,original_mask = U._pixels(raw_rgb,raw_mask)
    require(np.array_equal(original_rgb,rgb) and np.array_equal(original_mask,mask),'Full-frame RGB/mask arrays changed from saved source')
    expected_provenance = dict(row['mask_provenance'],backend=row['model_provider'],prompt_origin=row['mask_provenance']['prompt_origin'],manual_intervention=False,scope=row['mask_scope'])
    require(provenance == expected_provenance,'Portable mask provenance differs from selected source')
    video = U._local(case/selection['source']['file'],root)
    require(U.sha(video) == source['video_sha256'],'Current source video hash differs')
    files = {'input.json':input_path,'image.png':input_path.parent/'image.png','mask.png':input_path.parent/'mask.png',
             'provenance/selection.json':selection_path,'provenance/case-manifest.json':case_path,
             'provenance/frame-index.json':index_path,'provenance/source-frame.jpg':raw_rgb,
             'provenance/source-mask.png':raw_mask}
    dependencies = {}
    for name in DEPENDENCIES:
        p=root/name
        require(p.is_file() and p.resolve() == p and not p.is_symlink(),'Reviewed source adapter unavailable')
        dependencies[name] = U.sha(p)
    binding = {'format':'sam3d-initial-source-anchor','version':1,'component_id':manifest['object_id'],
      'input_manifest':reference(input_path,root),'selection':reference(selection_path,root),'source_video':reference(video,root),
      'source':copy.deepcopy(source),'source_case_manifest':reference(case_path,root),'frame_index':reference(index_path,root),
      'rgb':reference(input_path.parent/'image.png',root),'mask':reference(input_path.parent/'mask.png',root),
      'source_frame':reference(raw_rgb,root),'source_mask':reference(raw_mask,root),
      'decoded_arrays':{'width':record['width'],'height':record['height'],'rgb_sha256':U.digest(rgb.tobytes()),'mask_bool_sha256':U.digest(mask.tobytes())},
      'mask_provenance':copy.deepcopy(provenance),'implementation_sha256':dependencies,
      'baseline_required':False,'baseline_asset_read':False,'old_poses_read':False,'manual_geometry_used':False,
      'physical_accuracy_verified':False,'metric_scale_verified':False,'simulation_ready':False}
    return binding,files


def output_path(output,root):
    output = Path(output)
    if not output.is_absolute():output=root/output
    relative = output.relative_to(root)
    require(len(relative.parts) == 3 and relative.parts[:2] == ('data','initial-sam3d-handoffs')
            and all(not p.startswith('.') and '\\' not in p for p in relative.parts),'Choose a fresh direct child of data/initial-sam3d-handoffs')
    cursor=root
    for part in relative.parts:
        cursor=cursor/part
        require(not cursor.is_symlink(),'Symlink output forbidden')
    require(not output.exists(),'Preserve previous handoff; choose a fresh directory')
    return output


def run_plan(directory):
    return {'format':'sam3d-initial-reconstruction-plan','version':1,'input_manifest':'input.json','input_manifest_sha256':U.sha(directory/'input.json'),
          'adapter':'adapter/stage3_sam3d.py','adapter_sha256':U.sha(directory/'adapter/stage3_sam3d.py'),
          'official_code_commit':U.SAM3D_COMMIT,'device':'cuda','seed':42,'appearance':'vertex-color',
          'maximum_inferences':1,'automatic_retry':False,'baseline_required':False,
          'runtime_status':'not_checked_requires_reviewed_offline_runtime_and_weight_mapping',
          'adapter_preflight_argv_template':['python','adapter/stage3_sam3d.py','preflight','--input-manifest','input.json','--repo','/REPLACE/reviewed-official-source','--config','/REPLACE/reviewed-pipeline.yaml','--expected-code-commit',U.SAM3D_COMMIT,'--device','cuda','--output','/REPLACE/fresh-preflight-report.json'],
          'commands_executed':False,'returned_native_importer_status':'baseline_free_initial_import_not_implemented',
          'simulation_ready':False}


def prepare(input_manifest,selection,output,*,root=ROOT):
    root=Path(root).resolve();directory=output_path(output,root)
    binding,files=inspect_anchor(input_manifest,selection,root=root)
    directory.mkdir(parents=True,exist_ok=False)
    for name,path in files.items():
        target=directory/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    (directory/'adapter').mkdir()
    (directory/'adapter/stage3_sam3d.py').write_bytes((root/'scripts/stage3_sam3d.py').read_bytes())
    (directory/'implementation.py').write_bytes((root/'initial_sam3d_handoff.py').read_bytes())
    U.write(directory/'source-binding.json',binding)
    U.write(directory/'run-plan.json',run_plan(directory))
    (directory/'README.txt').write_text('INITIAL RECONSTRUCTION PREPARED\n\nThis is an initial RGB/SAM source anchor, not a generated 3D asset or a GPU submission.\nNo TripoSR asset, old poses, full video, credentials or weights are included.\nNative full-frame source arrays are preserved.\nActual inference requires a separately reviewed offline runtime, exact local weights and fresh output.\nThe preflight argument array is data and has not been executed. No automatic retry is authorized by this file.\nA baseline-free returned-native importer remains unfinished; do not fabricate upgrade-job lineage.\nNo physical scale, assembly, contact or robot readiness is established.\n')
    require(inspect_anchor(input_manifest,selection,root=root)[0] == binding,'Source anchor changed during preparation')
    inventory=[{'path':p.relative_to(directory).as_posix(),'sha256':U.sha(p),'bytes':p.stat().st_size}
               for p in sorted(directory.rglob('*')) if p.is_file()]
    record={'format':FORMAT,'version':1,'status':'initial_reconstruction_prepared','component_id':binding['component_id'],
            'source_binding_sha256':U.sha(directory/'source-binding.json'),'files':inventory,
            'no_inference':True,'remote_job_submitted':False,'baseline_required':False,'baseline_asset_read':False,
            'old_poses_read':False,'full_video_included':False,'preferred_assets_changed':False,'simulation_ready':False}
    U.write(directory/'handoff.json',record)
    return {'handoff':reference(directory/'handoff.json',root),'status':record['status'],'component_id':record['component_id'],
            'baseline_required':False,'no_inference':True,'download_url':'/files/'+str((directory/'handoff.json').relative_to(root))}


def inspect(handoff,*,root=ROOT):
    root=Path(root).resolve();path=U._local(handoff,root)
    require(path.stat().st_size <= 1024*1024,'Handoff JSON exceeds budget')
    record=U.read(path)
    require(isinstance(record,dict) and record.get('format') == FORMAT and type(record.get('version')) is int
            and record['version'] == 1 and record.get('status') == 'initial_reconstruction_prepared','Initial handoff identity differs')
    for key,value in {'no_inference':True,'remote_job_submitted':False,'baseline_required':False,'baseline_asset_read':False,
                      'old_poses_read':False,'full_video_included':False,'preferred_assets_changed':False,'simulation_ready':False}.items():
        require(record.get(key) is value,'Initial handoff scope differs')
    directory=path.parent
    actual_names=set()
    for artifact in sorted(directory.rglob('*')):
        require(not artifact.is_symlink(),'Symlink handoff artifacts forbidden')
        if artifact.is_file():actual_names.add(artifact.relative_to(directory).as_posix())
    require(path.name == 'handoff.json' and actual_names == FILES | {'handoff.json'},'Unexpected initial handoff file inventory')
    files=[]
    for name in sorted(FILES):
        artifact=U._local(directory/name,root)
        files.append({'path':name,'sha256':U.sha(artifact),'bytes':artifact.stat().st_size})
    require(files == record.get('files'),'Initial handoff inventory/hash differs')
    binding_path=U._local(directory/'source-binding.json',root)
    require(U.sha(binding_path) == record.get('source_binding_sha256'),'Initial source binding differs')
    binding=U.read(binding_path)
    actual,source_files=inspect_anchor(binding['input_manifest']['path'],binding['selection']['path'],root=root)
    require(actual == binding and record['component_id'] == actual['component_id'],'Current source anchor differs')
    for name,source in source_files.items():require((directory/name).read_bytes() == source.read_bytes(),'Portable source copy differs')
    require(U.sha(directory/'adapter/stage3_sam3d.py') == actual['implementation_sha256']['scripts/stage3_sam3d.py']
            and U.sha(directory/'implementation.py') == actual['implementation_sha256']['initial_sam3d_handoff.py'],'Portable adapter/implementation differs')
    require(U.read(directory/'run-plan.json') == run_plan(directory),'Initial offline runtime/command policy differs')
    read_inputs(directory/'input.json')
    return {'status':'verified_initial_reconstruction_prepared','handoff':reference(path,root),'component_id':record['component_id'],
            'source':binding['source'],'baseline_required':False,'no_inference':True,'simulation_ready':False}
