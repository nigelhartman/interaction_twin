#!/usr/bin/env python3
"""Native-only review of a real baseline-free initial SAM3D import.

No inference, baseline, geometry changes, source placement, model selection or
physics. The review revalidates actual frozen input and complete returned bytes.
"""
import argparse
import base64
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import initial_sam3d_import as N
import sam3d_milestone as M
import upgrade_bundles as U

spec=importlib.util.spec_from_file_location('shared_sam3d_review_builder',ROOT/'scripts/build_sam3d_milestone_review.py')
SHARED=importlib.util.module_from_spec(spec);spec.loader.exec_module(SHARED)


def verified_initial_import(directory,*,root=ROOT):
    root=Path(root).resolve();directory=M._directory(directory,root);files=M._tree(directory)
    if 'import.json' not in files:raise ValueError('An actual initial native import receipt is required')
    receipt=M._object(U._json_bytes(files['import.json']),'Initial native import receipt')
    if (receipt.get('format')!='sam3d-initial-native-readiness' or type(receipt.get('version')) is not int or receipt['version']!=1
            or receipt.get('fixture') is not False or receipt.get('status')!='validated_initial_native_candidate_pending_review'
            or receipt.get('import_directory')!=directory.relative_to(root).as_posix()):
        raise ValueError('Unsupported or fixture initial import receipt')
    if receipt.get('import_files')!=M._records({k:v for k,v in files.items() if k!='import.json'}):
        raise ValueError('Initial imported inventory/hash differs')
    for key in ['baseline_required','baseline_asset_read','old_poses_read','model_inference_ran_during_import',
                'physical_accuracy_verified','metric_scale_verified','source_video_pose_validated','assembly_verified',
                'contact_verified','simulation_executed','simulation_ready','model_selection_changed','catalog_changed']:
        if receipt.get(key) is not False:raise ValueError('Initial import scope changed')
    handoff=M._object(receipt.get('initial_handoff'),'Initial handoff reference')
    path=U._local(handoff.get('path',''),root)
    if U.sha(path)!=handoff.get('sha256') or path.stat().st_size!=handoff.get('bytes'):raise ValueError('Original initial handoff changed')
    checked,source,native=N.inspect_output(path,directory/'raw',root=root)
    for field in ['initial_handoff','source_binding_sha256','source','component_id','native_layout_candidate','fit_contract']:
        if receipt.get(field)!=checked[field]:raise ValueError('Initial source/native candidate binding differs')
    for name,raw in source['files'].items():
        if files.get('initial-handoff/'+name)!=raw:raise ValueError('Imported initial handoff copy differs')
    if files.get('asset.glb')!=native['files']['object.glb']:raise ValueError('Native GLB differs from the original returned model')
    model=M._object(U._json_bytes(files.get('metadata.json',b'null')),'Initial model metadata')
    if (model.get('fixture') not in (None,False) or model.get('backend')!='Meta SAM 3D Objects'
            or model.get('status')!='imported_initial_native_candidate_pending_alignment'
            or model.get('object_id')!=checked['component_id'] or model.get('outputs')!={'glb_path':'asset.glb'}
            or model.get('sam3d_output') is not True or model.get('learned_inference_executed') is not True
            or model.get('geometry_reconstructed_by_model') is not True
            or any(model.get(k) is not False for k in ['manual_geometry_fallback','manual_pose_used','camera_pose_estimated',
                                                       'metric_scale_verified','physical_accuracy_verified','selected','catalog_changed','simulation_ready'])
            or 'upgrade_lineage' in model or 'milestone_lineage' in model):
        raise ValueError('Initial model provenance or selection scope differs')
    lineage=M._object(model.get('initial_reconstruction_lineage'),'Initial lineage')
    if (lineage.get('handoff')!=handoff or lineage.get('source_binding_sha256')!=checked['source_binding_sha256']
            or lineage.get('raw_asset_sha256')!=U.digest(files['asset.glb'])
            or any(lineage.get(k) is not False for k in ['geometry_bytes_modified','baseline_required','baseline_asset_read','old_poses_read'])):
        raise ValueError('Initial lineage cannot inherit baseline geometry or poses')
    expected_source=copy.deepcopy(source['binding']['source'])
    expected_source.update(image_path='raw/inputs/image-source.png',mask_path='raw/inputs/mask-source.png',
        image_sha256=U.digest(native['files']['inputs/image-source.png']),mask_sha256=U.digest(native['files']['inputs/mask-source.png']),
        width=source['input_record']['width'],height=source['input_record']['height'],mask_provenance=source['binding']['mask_provenance'])
    if model.get('source')!=expected_source:raise ValueError('Initial model source RGB/mask/time differs')
    return directory,receipt,source,native,model


def build(import_directory,output,*,root=ROOT):
    root=Path(root).resolve();output=M._directory(output,root,exists=False)
    if output.exists():raise ValueError('Choose a fresh initial review version')
    directory,receipt,source,native,model=verified_initial_import(import_directory,root=root)
    if output.is_relative_to(directory) or output.is_relative_to(source['directory']):raise ValueError('Review must be separate from frozen evidence')
    record=source['input_record'];track=receipt['component_id'];raw=native['files'];decoded=native['decoded']
    mask=np.asarray(Image.open(io.BytesIO(raw['inputs/mask.png'])))>0
    rgba=np.zeros((*mask.shape,4),np.uint8);rgba[mask]=[42,202,165,112]
    buffer=io.BytesIO();Image.fromarray(rgba).save(buffer,format='PNG')
    payloads={'source-rgb.png':source['files']['image.png'],'source-mask.png':source['files']['mask.png'],
              'mask-overlay.png':buffer.getvalue(),'candidate.glb':raw['object.glb'],
              'candidate-scene.json':SHARED.scene_json_bytes(SHARED.native_scene(decoded,track,'Y',[record['width'],record['height']])),
              'model-metadata.json':(directory/'metadata.json').read_bytes()}
    shared_renderer=(root/'workspace/scene.js').read_text();renderer=SHARED.camera_only_renderer(shared_renderer,'neutral').encode()
    manifest={'format':'sam3d-initial-native-review','version':1,'fixture':False,'native_only':True,
              'component_id':track,'title':track.replace('_',' ').capitalize(),
              'source':copy.deepcopy(source['binding']['source']),'image_size':[record['width'],record['height']],
              'native_glb_sha256':U.digest(raw['object.glb']),'native_up':'Y','mesh_stats':decoded['mesh_stats'],
              'appearance':model['appearance'],'files':M._records(payloads),
              'initial_handoff':receipt['initial_handoff'],
              'import_receipt':{'path':(directory/'import.json').relative_to(root).as_posix(),'sha256':U.sha(directory/'import.json')},
              'scope':{'baseline_required':False,'original_geometry_modified':False,'old_pose_reused':False,
                       'source_video_placement_established':False,'physical_accuracy_verified':False,'metric_scale_verified':False,
                       'assembly_verified':False,'contact_verified':False,'simulation_ready':False,'selected':False,'catalog_changed':False},
              'renderer':{'shared_source_sha256':U.digest(shared_renderer.encode()),'review_copy_sha256':U.digest(renderer),
                          'display_background':'neutral','display_clear_rgba':[.91,.93,.90,1.],
                          'native_up':'Y','geometry_or_material_changed':False,'pbr_rendering':False,'splat_rendering':False},
              'execution_scope':'Genuine-result attestations and source bytes revalidated. Independent root launch evidence is reviewed separately; no physical validity is granted.'}
    manifest_raw=U.encoded(manifest);templates=root/'initial_native_review'
    client=(templates/'review.js').read_bytes();style=(templates/'style.css').read_bytes()
    page=(templates/'index.html').read_text().replace('%%MANIFEST_SHA%%',U.digest(manifest_raw))
    for key,rawbytes in [('SCENE',renderer),('CLIENT',client),('STYLE',style)]:
        page=page.replace('%%'+key+'_INTEGRITY%%','sha256-'+base64.b64encode(bytes.fromhex(U.digest(rawbytes))).decode())
    final={**payloads,'review-manifest.json':manifest_raw,'scene.js':renderer,'review.js':client,'style.css':style,'index.html':page.encode()}
    output.mkdir(parents=True,exist_ok=False)
    for name,rawbytes in final.items():(output/name).write_bytes(rawbytes)
    U.write(output/'review-build.json',{'format':'sam3d-initial-native-review-build','version':1,'files':M._records(final),
        'builder_sha256':U.sha(Path(__file__)),'original_import_sha256':U.sha(directory/'import.json'),
        'new_models_or_api_calls':0,'native_geometry_modified':False,'catalog_changed':False,'physical_accuracy_verified':False})
    return {'status':'initial_native_review_built_unselected','review':(output/'index.html').relative_to(root).as_posix(),
            'component_id':track,'mesh_stats':decoded['mesh_stats'],'scene_bytes':len(payloads['candidate-scene.json'])}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--import-directory',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:result=build(args.import_directory,args.output)
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(json.dumps({'status':'initial_native_review_unavailable','error':str(error)}));raise SystemExit(2)
    print(json.dumps(result,indent=2))
