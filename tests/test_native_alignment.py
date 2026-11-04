"""CPU-only synthetic contract checks, never published as learned results."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import trimesh

import native_alignment as N
from native_appearance import load_native_glb, render_native_mesh

ROOT=Path(__file__).resolve().parents[1]


def script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(directory):
    """Known synthetic UV plane and matching frames; not a reconstruction."""
    case=directory/'case'; model=case/'model'; model.mkdir(parents=True)
    yy,xx=np.mgrid[:32,:32]
    pixels=np.zeros((32,32,3),np.uint8)
    pixels[:,:,0]=np.where((xx//4+yy//4)%2,240,25)
    pixels[:,:,1]=np.where(xx<16,180,35)
    pixels[:,:,2]=np.where(yy<16,220,50)
    material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(pixels),doubleSided=True)
    mesh=trimesh.Trimesh(vertices=[[-.8,-.6,0],[.8,-.6,0],[.8,.6,0],[-.8,.6,0]],
                         faces=[[0,1,2],[0,2,3]],process=False,
                         visual=trimesh.visual.TextureVisuals(uv=[[0,0],[1,0],[1,1],[0,1]],material=material))
    raw=mesh.export(file_type='glb'); (model/'asset.glb').write_bytes(raw)
    native=load_native_glb(raw)
    width,height=96,72; focal=width/(2*np.tan(np.deg2rad(60)/2))
    k=np.array([[focal,0,(width-1)/2],[0,focal,(height-1)/2],[0,0,1.]])
    rendered=render_native_mesh(native,k,np.eye(3),np.array([0.,0.,2.]),width,height)
    frame=rendered['rgb']; mask=(rendered['mask']*255).astype(np.uint8)
    (case/'frames').mkdir(); (case/'source.mp4').write_bytes(b'synthetic-contract-fixture-not-a-video')
    source_sha=sha(case/'source.mp4')
    write(case/'manifest.json',{'source':{'sha256':source_sha,'original_sha256':source_sha},'synthetic_contract_fixture':True})
    frames=[]; track_frames=[]
    seg=case/'segmentation/synthetic'; seg.mkdir(parents=True)
    for index in [1,2]:
        Image.fromarray(frame).save(case/'frames'/f'{index:06d}.png')
        Image.fromarray(mask).save(seg/f'{index:06d}.png')
        frames.append({'index':index,'timestamp_s':float(index-1),'file':f'frames/{index:06d}.png','width':width,'height':height})
        track_frames.append({'sample_index':index,'raw_mask_file':f'{index:06d}.png'})
    write(case/'frames/index.json',{'frames':frames})
    Image.fromarray(frame).save(model/'source-rgb.png'); Image.fromarray(mask).save(model/'source-mask.png')
    selection=case/'selected/selection.json'; write(selection,{'all_ranked_candidates':[],'synthetic_contract_fixture':True})
    metadata={'object_id':'synthetic_uv_fixture','backend':'Synthetic contract fixture (not model output)',
              'appearance':'synthetic embedded checker texture; no source/model reconstruction',
              'synthetic_contract_fixture':True,'outputs':{'glb_path':'asset.glb'},
              'artifact_files':[{'file':'asset.glb','sha256':sha(model/'asset.glb')}],
              'source':{'image_path':'source-rgb.png','mask_path':'source-mask.png',
                        'image_sha256':sha(model/'source-rgb.png'),'mask_sha256':sha(model/'source-mask.png'),
                        'width':width,'height':height,'sample_index':1,'timestamp_s':0.,'video_sha256':source_sha,
                        'selection_manifest_path':str(selection),'selection_manifest_sha256':sha(selection),
                        'mask_provenance':{'prompt_origin':'model_generated_prompt','manual_seed_boxes_used':False,
                                           'prompt_provider':'synthetic contract fixture; no inference'}}}
    write(model/'metadata.json',metadata)
    write(seg/'annotations.json',{'manual_seed_boxes_used':False})
    write(seg/'tracks.json',{'tracks':[{'track_id':'synthetic_uv_fixture','frames':track_frames}]})
    write(seg/'quality.json',{'tracks':{'synthetic_uv_fixture':{'withheld_samples':[]}}})
    write(seg/'model-audit-v1/audit.json',{'result':{'observations':[{'track_id':'synthetic_uv_fixture','sample_index':i,
          'verdict':'plausible_visible_mask'} for i in [1,2]]},'synthetic_contract_fixture':True})
    return case,model,native,raw


class NativeAlignmentTests(unittest.TestCase):
    def test_native_receipt_binds_loader_decoder_geometry_order_and_actual_backend(self):
        with tempfile.TemporaryDirectory() as temp:
            _,_,mesh,_=fixture(Path(temp))
            receipt=N.native_representation_receipt(mesh)
            self.assertEqual(receipt['decoder_source_sha256'],sha(ROOT/'portable_twin.py'))
            self.assertFalse(receipt['vertex_conversion_performed']); self.assertTrue(receipt['native_origin_preserved'])
            self.assertIsNone(mesh.vertex_colors)
            N.verify_native_reference({'native_appearance':receipt},mesh)
            for key in ['flattened_vertices_sha256','flattened_faces_sha256','loader_source_sha256','decoder_source_sha256']:
                changed=copy.deepcopy(receipt); changed[key]='0'*64
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'interpretation changed'):
                    N.verify_native_reference({'native_appearance':changed},mesh)
            with self.assertRaisesRegex(ValueError,'interpretation changed'): N.verify_native_reference({},mesh)
        self.assertEqual(N.native_model_provenance({'backend':'Meta SAM 3D Objects'})['reconstruction_backend'],'Meta SAM 3D Objects')
        self.assertTrue(N.native_model_provenance({'backend':'Meta SAM 3D Objects'})['sam3d_output'])
        self.assertFalse(N.native_model_provenance({'backend':'fixture'})['sam3d_output'])
        with self.assertRaises(ValueError): N.native_model_provenance({})

    def test_cpu_reference_and_mocked_tracking_use_uv_without_vertex_conversion(self):
        pose=script('stage4_native_pose'); track=script('stage4_native_track')
        with tempfile.TemporaryDirectory() as temp:
            case,model,mesh,raw=fixture(Path(temp)); out=case/'native-fit'
            argv=['stage4_native_pose.py','--manifest',str(model/'metadata.json'),'--out',str(out),
                  '--search-width','96','--refine-count','1','--max-evaluations','30','--appearance-weight','.25']
            with patch.object(sys,'argv',argv): pose.main()
            record=json.loads((out/'experiment.json').read_text())
            self.assertEqual((model/'asset.glb').read_bytes(),raw)
            self.assertEqual(record['inputs']['mesh_sha256'],hashlib.sha256(raw).hexdigest())
            self.assertEqual(record['native_appearance'],N.native_representation_receipt(mesh))
            self.assertFalse(record['provenance']['uv_converted_to_vertex_colours'])
            self.assertEqual(record['provenance']['reconstruction_backend'],'Synthetic contract fixture (not model output)')
            self.assertNotIn('backend_label_adapter',record)
            self.assertFalse(record['usable_pose_gate']['accepted']); self.assertFalse(record['physical_scale_known'])
            self.assertIn('portable_twin.py',[r['file'] for r in record['alignment_backend']['project_sources']])
            self.assertIn('Synthetic contract fixture',(out/'index.html').read_text())
            self.assertNotIn('TripoSR',(out/'index.html').read_text())
            self.assertEqual(record['reference_full_resolution_fast_raster_iou'],record['reference_full_resolution_iou'])
            def points(images,seeds,anchor,device,maximum_dimension=384):
                return np.repeat(seeds[None],len(images),axis=0),np.ones((len(images),len(seeds)),bool),{'synthetic_contract_fixture':True,'inference_executed':False}
            motion=case/'native-motion'
            with patch.object(sys,'argv',['stage4_native_track.py','--reference',str(out/'experiment.json'),
                       '--segmentation-run','synthetic','--output',str(motion)]),patch.object(track,'cotracker',side_effect=points) as learned:
                track.main()
            result=json.loads((motion/'experiment.json').read_text())
            self.assertEqual(result['status'],'complete_conditional_tracking_experiment')
            self.assertEqual(result['native_appearance'],record['native_appearance'])
            self.assertFalse(result['provenance']['uv_converted_to_vertex_colours'])
            self.assertEqual(result['provenance']['reconstruction_backend'],record['provenance']['reconstruction_backend'])
            self.assertFalse(result['physical_pose_accuracy_measured']); self.assertEqual(learned.call_count,1)
            self.assertIn('Synthetic contract fixture:',(motion/'observer.html').read_text())
            self.assertEqual((model/'asset.glb').read_bytes(),raw)

    def test_explicit_cpu_device_and_aspect_following_shapes_have_no_fallback(self):
        track=script('stage4_native_track')
        unavailable=SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:False)))
        available=SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:True)))
        with patch.dict('os.environ',{'PYTORCH_ENABLE_MPS_FALLBACK':'1'}):
            self.assertEqual(track.validate_tracking_device('cpu',unavailable),'cpu')
            with self.assertRaisesRegex(RuntimeError,'select cpu explicitly'):
                track.validate_tracking_device('mps',unavailable)
            with self.assertRaisesRegex(RuntimeError,'fallback must remain disabled'):
                track.validate_tracking_device('mps',available)
        with self.assertRaisesRegex(ValueError,'explicit cpu or mps'): track.validate_tracking_device('auto',available)
        self.assertEqual(track.tracking_input_shape((1920,1080,3))[0],(384,216))
        self.assertEqual(track.tracking_input_shape((1080,1920,3))[0],(216,384))
        self.assertEqual(track.tracking_input_shape((720,960,3),128)[0],(96,128))
        shape,receipt=track.tracking_input_shape((715,953,3))
        self.assertLessEqual(max(shape),384); self.assertTrue(all(x%8==0 for x in shape))
        self.assertGreater(receipt['aspect_ratio_relative_error'],0)
        self.assertFalse(receipt['input_aspect_ratio_preserved'])
        self.assertEqual(receipt['source_aspect_ratio_wh'],953/715)
        self.assertEqual(receipt['model_aspect_ratio_wh'],shape[1]/shape[0])
        self.assertTrue(receipt['predictor_outputs_are_original_source_pixels'])
        for shape,maximum in [((0,32),384),((32,32),512),((32,32),130)]:
            with self.assertRaises(ValueError): track.tracking_input_shape(shape,maximum)

    def test_unsupported_alpha_material_is_blocked_before_reference_output(self):
        pose=script('stage4_native_pose')
        with tempfile.TemporaryDirectory() as temp:
            case,model,_,raw=fixture(Path(temp))
            length,kind=struct.unpack_from('<II',raw,12)
            document=json.loads(raw[20:20+length].decode())
            document['materials'][0]['alphaMode']='BLEND'
            body=json.dumps(document,separators=(',',':')).encode(); body+=b' '*((-len(body))%4)
            tail=raw[20+length:]
            changed=struct.pack('<4sII',b'glTF',2,20+len(body)+len(tail))+struct.pack('<II',len(body),kind)+body+tail
            (model/'asset.glb').write_bytes(changed)
            metadata=json.loads((model/'metadata.json').read_text()); metadata['artifact_files'][0]['sha256']=sha(model/'asset.glb')
            write(model/'metadata.json',metadata); output=case/'blocked-native-fit'
            with patch.object(sys,'argv',['stage4_native_pose.py','--manifest',str(model/'metadata.json'),'--out',str(output)]):
                with self.assertRaisesRegex(ValueError,'opaque|OPAQUE|Alpha|alpha'):
                    pose.main()
            self.assertFalse(output.exists())
            self.assertEqual((model/'asset.glb').read_bytes(),changed)

    def test_native_silhouette_uses_same_mask_and_origin_as_final_renderer(self):
        pose=script('stage4_native_pose')
        with tempfile.TemporaryDirectory() as temp:
            _,_,mesh,_=fixture(Path(temp)); k=np.array([[30.,0,24],[0,30.,24],[0,0,1.]])
            r=np.eye(3); origin=np.array([.1,.2,.3]); t=np.array([0.,0.,3.])
            a=pose.silhouette(mesh,r,t,k,(48,48),origin)
            b=render_native_mesh(mesh,k,r,t-r@origin,48,48)['mask']
            self.assertTrue(np.array_equal(a,b))


if __name__=='__main__': unittest.main()
