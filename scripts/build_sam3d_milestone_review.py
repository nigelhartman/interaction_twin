#!/usr/bin/env python3
"""Build a native-only review AFTER an actual validated SAM3D import exists.

No inference, network, model selection, original geometry/pose changes or fake
candidate pages. Import attestations are verified, not cryptographic proof.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import sam3d_milestone as M
import upgrade_bundles as U
from portable_twin import decode_glb_bytes
MAX_SCENE_JSON = 64 * 1024 * 1024


def json_bytes(value):
    return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def scene_json_bytes(value):
    """Compact derived geometry only; native GLB bytes/coordinates stay intact."""
    raw=(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n').encode()
    if len(raw)>MAX_SCENE_JSON:
        raise ValueError('Derived native scene exceeds the unchanged browser JSON budget')
    return raw


def verified_import(directory,*,root=ROOT):
    directory=M._directory(directory,root)
    files=M._tree(directory)
    if 'import.json' not in files:
        raise ValueError('An actual validated native import receipt is required')
    receipt=M._object(U._json_bytes(files['import.json']),'Native import receipt')
    if (receipt.get('format')!='sam3d-native-milestone-readiness' or receipt.get('version')!='1'
            or receipt.get('fixture') is not False or receipt.get('status')!='validated_native_candidate_pending_review'
            or receipt.get('import_directory')!=str(directory.relative_to(Path(root).resolve()))):
        raise ValueError('Unsupported or fixture native import receipt')
    if receipt.get('import_files')!=M._records({name:raw for name,raw in files.items() if name!='import.json'}):
        raise ValueError('Native imported file inventory/hash differs')
    for field in ('physical_accuracy_verified','metric_scale_verified','source_video_pose_validated','assembly_verified','contact_verified','simulation_executed','model_selection_changed'):
        if receipt.get(field) is not False:
            raise ValueError('Native import physical/model-selection scope differs')
    bundle=M._object(receipt.get('input_bundle'),'Import input bundle')
    original=M._directory(bundle.get('path',''),root)
    report,source,raw,metadata=M.inspect_output(original,directory/'raw',root=root)
    if U.digest(source['files']['bundle-manifest.json'])!=bundle.get('manifest_sha256'):
        raise ValueError('Frozen input bundle changed')
    for name,payload in source['files'].items():
        if files.get('input-bundle/'+name)!=payload:
            raise ValueError('Imported frozen input copy differs')
    if receipt.get('source_binding')!=report['source_binding'] or receipt.get('native_layout_candidate')!=report['native_layout_candidate'] or receipt.get('fit_contract')!=report['fit_contract']:
        raise ValueError('Import source/native-layout/material binding differs')
    model=M._object(U._json_bytes(files.get('metadata.json',b'null')),'Native model metadata')
    if (model.get('fixture') not in (None,False) or model.get('backend')!='Meta SAM 3D Objects'
            or model.get('sam3d_output') is not True or model.get('learned_inference_executed') is not True
            or model.get('object_id')!=source['binding']['track_id'] or model.get('outputs')!={'glb_path':'asset.glb'}
            or model.get('manual_geometry_fallback') is not False or model.get('manual_pose_used') is not False
            or model.get('camera_pose_estimated') is not False or model.get('metric_scale_verified') is not False):
        raise ValueError('Native candidate metadata provenance differs')
    if files.get('asset.glb')!=raw['object.glb']:
        raise ValueError('Imported candidate GLB differs from raw model export')
    return directory,receipt,source,raw,metadata


def native_scene(decoded,track_id,up,image_size):
    """A new display camera only. No reference poses or geometry transforms."""
    return {'source_glb_sha256':decoded['source_glb_sha256'],
            'scene':{'objects':[{'object_id':track_id,'primitives':decoded['primitives'],
                               'native_scene':decoded['native_scene'],'mesh_stats':decoded['mesh_stats'],
                               'native_up':up,'frames':[],'max_hold_s':0,'source_projection_rows':None}],
                     'scene':decoded['native_scene'],'native_only':True,'timeline':[],
                     'source_projection_rows':None,'image_size':image_size,
                     'physical_assembly_verified':False}}


def camera_only_renderer(source, background='dark'):
    """Derive an isolated display copy; original geometry/renderer stay unchanged."""
    if background not in {'dark', 'neutral'}:
        raise ValueError('Unsupported review display background')
    old_basis="return this.inspection==='native'?{direction:[s*cp,-c*cp,sp],right:[c,s,0],up:[-s*sp,c*sp,cp]}:{direction:[s*cp,-sp,-c*cp],right:[c,0,s],up:[-s*sp,-cp,c*sp]}"
    new_basis="return this.inspection==='native'?(this.object().native_up==='Y'?{direction:[s*cp,sp,c*cp],right:[c,0,-s],up:[-s*sp,cp,-c*sp]}:{direction:[s*cp,-c*cp,sp],right:[c,s,0],up:[-s*sp,c*sp,cp]}):{direction:[s*cp,-sp,-c*cp],right:[c,0,s],up:[-s*sp,-cp,c*sp]}"
    old_up="this.inspection==='native'?[0,0,1]:[0,-1,0]"
    new_up="this.inspection==='native'?(this.object().native_up==='Y'?[0,1,0]:[0,0,1]):[0,-1,0]"
    if source.count(old_basis)!=1 or source.count(old_up)!=1:
        raise ValueError('Shared native renderer changed; review camera-only patch')
    result=source.replace(old_basis,new_basis).replace(old_up,new_up)
    if background == 'neutral':
        old_clear='g.clearColor(.031,.047,.075,1);'
        if result.count(old_clear)!=1:
            raise ValueError('Shared native background changed; review display-only patch')
        result=result.replace(old_clear,'g.clearColor(.91,.93,.90,1);')
    return result


def build(import_directory,output,*,root=ROOT,background='dark'):
    root=Path(root).resolve()
    destination=M._directory(output,root,exists=False)
    if destination.exists():
        raise ValueError('Choose a fresh review version')
    directory,receipt,source,raw,metadata=verified_import(import_directory,root=root)
    if destination.is_relative_to(directory) or destination.is_relative_to(source['directory']):
        raise ValueError('Review must remain separate from frozen evidence trees')
    info=source['binding'];track=info['track_id']
    baseline=decode_glb_bytes(source['baseline_asset'])
    candidate=decode_glb_bytes(raw['object.glb'])
    # All validation/decoding above happens before any page/output is created.
    payloads={'source-rgb.png':source['files']['image.png'],'source-mask.png':source['files']['mask.png'],
              'baseline.glb':source['baseline_asset'],'candidate.glb':raw['object.glb'],
              'baseline-scene.json':scene_json_bytes(native_scene(baseline,track,'Z',[info['source']['width'],info['source']['height']])),
              'candidate-scene.json':scene_json_bytes(native_scene(candidate,track,'Y',[info['source']['width'],info['source']['height']]))}
    renderer_source=(root/'workspace/scene.js').read_text()
    renderer=camera_only_renderer(renderer_source,background).encode()
    manifest={'format':'sam3d-native-review','version':1,'fixture':False,'native_comparison_only':True,
              'binding':{'case_id':info['twin_id'],'track_id':track,'program_sha256':info['base_sha256'],
                         'video_sha256':info['source']['video_sha256'],'sample_index':info['source']['sample_index'],
                         'timestamp_s':info['source']['timestamp_s'],'time_domain':info['source']['time_domain'],
                         'width':info['source']['width'],'height':info['source']['height']},
              'input_bundle_manifest_sha256':U.digest(source['files']['bundle-manifest.json']),
              'import_receipt':{'path':str((directory/'import.json').relative_to(root)),'sha256':U.sha(directory/'import.json')},
              'files':M._records(payloads),
              'baseline':{'backend':source['baseline_metadata'].get('backend'),'asset_sha256':U.digest(source['baseline_asset']),
                          'native_up':'Z','mesh_stats':baseline['mesh_stats'],'appearance':'Preserved generated appearance'},
              'candidate':{'backend':'Meta SAM 3D Objects','asset_sha256':U.digest(raw['object.glb']),
                           'native_up':'Y','mesh_stats':candidate['mesh_stats'],'appearance_mode':metadata['outputs']['appearance_mode'],
                           'appearance':'Model-generated appearance; hidden surfaces inferred','selected':False},
              'renderer':{'shared_source_path':'workspace/scene.js','shared_source_sha256':U.digest(renderer_source.encode()),
                          'review_copy_sha256':U.digest(renderer),'camera_only_patch':'Native display orbit supports per-asset Y-up or recorded Z-up default; no geometry/pose modification',
                          'lighting':'unlit base color','pbr_rendering':False,'splat_rendering':False,
                          'display_background':background,
                          'display_clear_rgba':[.91,.93,.90,1] if background=='neutral' else [.031,.047,.075,1],
                          'native_color_values_modified':False},
              'scope':{'independent_gauges':True,'geometry_bytes_modified':False,'pose_reused':False,
                       'source_video_alignment_established':False,'metric_scale_verified':False,
                       'physical_accuracy_verified':False,'assembly_verified':False,'contact_verified':False,
                       'simulation_executed':False,'preferred_model_changed':False},
              'execution_evidence':'Adapter/import attestations checked; review root launch receipt separately. This page does not prove inference execution or physical accuracy.'}
    manifest_raw=json_bytes(manifest)
    templates=root/'native_review'
    client=(templates/'review.js').read_bytes();style=(templates/'style.css').read_bytes()
    index=(templates/'index.html').read_text().replace('%%MANIFEST_SHA%%',U.digest(manifest_raw)).replace('%%CASE_ID%%',html.escape(info['twin_id'],quote=True)).replace('%%TRACK_ID%%',html.escape(track,quote=True))
    for role,payload in (('RENDERER',renderer),('CLIENT',client),('STYLE',style)):
        index=index.replace('%%'+role+'_INTEGRITY%%','sha256-'+base64.b64encode(bytes.fromhex(U.digest(payload))).decode())
    destination.mkdir(parents=True,exist_ok=False)
    for name,payload in {**payloads,'review-manifest.json':manifest_raw,'scene.js':renderer,'review.js':client,'style.css':style,'index.html':index.encode()}.items():
        (destination/name).write_bytes(payload)
    delivery={'format':'sam3d-native-review-build','version':1,'status':'native_comparison_only_unselected',
              'input_import_sha256':U.sha(directory/'import.json'),'manifest_sha256':U.digest(manifest_raw),
              'builder_sha256':U.sha(Path(__file__)),'files':M._records(M._tree(destination)),
              'no_models_or_api_calls':True,'original_evidence_modified':False,'catalog_or_preferred_model_changed':False}
    U.write(destination/'review-build.json',delivery)
    return {'review':str(destination/'index.html'),'manifest_sha256':delivery['manifest_sha256'],'status':delivery['status']}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--background',choices=('dark','neutral'),default='dark',help='Display-only WebGL clear color; original native colours stay unchanged')
    args=parser.parse_args(argv)
    try:
        value=build(args.import_directory,args.output,background=args.background)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(json.dumps({'status':'native_review_unavailable','error':str(exc)}));return 2
    print(json.dumps(value,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
