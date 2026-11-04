"""Synthetic mathematical contracts only; fixtures never enter the real index."""
from copy import deepcopy
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from PIL import Image
from motion_lab import normalize,sha,CONVENTION

ROOT=Path(__file__).resolve().parents[1]


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


class MotionLabContracts(unittest.TestCase):
    def fixture(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name).resolve();data=root/'data';data.mkdir()
        source=data/'source.mp4';source.write_bytes(b'SYNTHETIC unit fixture no real video')
        image=data/'source.jpg';Image.new('RGB',(8,8),'grey').save(image)
        mask=data/'mask.png';Image.new('L',(8,8),255).save(mask)
        document={'asset':{'version':'2.0'},'scenes':[{'nodes':[]}],'scene':0}
        body=json.dumps(document,separators=(',',':')).encode();body+=b' '*((-len(body))%4)
        asset=data/'asset.glb';asset.write_bytes(struct.pack('<4sIIII',b'glTF',2,20+len(body),len(body),0x4e4f534a)+body)
        K={'matrix':[[8,0,3.5],[0,8,3.5],[0,0,1]],'width':8,'height':8,'calibrated':False}
        R=[[1,0,0],[0,1,0],[0,0,1]];t=[0,0,3]
        rawframes=[];frames=[];comparisons=[]
        for i,(time,status,rawstatus)in enumerate([(1,'reference_unverified','reference_candidate_unverified_geometry_and_pose'),(1.5,'supported_nonreference_candidate','unsupported_candidate_projection_retained'),(2,'withheld','unsupported')],1):
            raw={'sample_index':i,'timestamp_s':time,'status':rawstatus,'source_frame_sha256':sha(image),'sam_mask_sha256':sha(mask),'reasons':[]}
            if i<3:raw.update(candidate_rotation=R,candidate_translation=t)
            rawframes.append(raw)
            frame={'sample_index':i,'timestamp_s':time,'status':status,'reference':i==1,'reasons':[],
                   'source_image_url':'/files/data/source.jpg','source_image_sha256':sha(image),'mask_url':'/files/data/mask.png','mask_sha256':sha(mask),
                   'candidate_rotation':R if i<3 else None,'candidate_translation':t if i<3 else None,
                   'training_visible_point_ids':[1,2],'heldout_visible_point_ids':[0],
                   'evaluation_timestamp_reserved_before_reference_selection':i>1,'temporal_split':'evaluation'if i>1 else'reference_selection'}
            frames.append(frame)
            comparisons.append({'episode_id':'episode_01','sample_index':i,'timestamp_s':time,
                'baseline_status':rawstatus,'baseline_reassessed_status':status,'baseline_reassessed_reasons':[],
                'baseline_original_supported':False,'baseline_supported':i==2,'candidate_supported':i==2,
                'baseline_reference':i==1,'candidate_reference':i==1})
        baseline=data/'baseline.json';save(baseline,{'intrinsics':K,'provenance':{'source_video_sha256':sha(source),'mesh_sha256':sha(asset)},'frames':rawframes})
        evidence=[{'source_index':f['sample_index'],'timestamp_s':f['timestamp_s'],'frame_path':'data/source.jpg',
                   'frame_sha256':sha(image),'observations':[{'track_id':'part','mask_path':'data/mask.png','mask_sha256':sha(mask)}]}for f in frames]
        program=data/'interaction.json';save(program,{'case_id':'fixture','tracklets':[{'id':'part'}],'source':{'sha256':sha(source)},'evidence':evidence})
        result={'format':'short-interval-motion','version':1,'status':'complete_with_gaps','case_id':'fixture','component_id':'part',
                'interaction_path':'data/interaction.json','interaction_sha256':sha(program),
                'source':{'path':'data/source.mp4','sha256':sha(source)},'asset':{'path':'data/asset.glb','sha256':sha(asset),'physical_scale_known':False},
                'coordinate_convention':CONVENTION,'physical_pose_accuracy_measured':False,'common_assembly_established':False,
                'baseline':{'path':'data/baseline.json','sha256':sha(baseline)},'baseline_reassessment_policy':{
                    'uses_saved_numeric_candidates_only':True,'uses_same_current_audit_and_original_geometric_gates':True,
                    'new_inference_or_pose_refit':False,'original_outputs_overwritten':False},
                'comparison':{'exact_same_mesh':True,'exact_pts_rows':comparisons},
                'episodes':[{'id':'episode_01','start_s':1,'end_s':2,'reference_sample_index':1,'reference_timestamp_s':1,
                'intrinsics':K,'gauge':'independent_reference_conditioned_candidate','queries':{'training_point_ids':[1,2],'heldout_point_ids':[0,3]},'frames':frames}]}
        result.update(track_id='part',new_api_calls=0,usable_physical_pose_recovered=False,old_pose_initialization_used=False,manual_geometry_or_poses=False,source_video_sha256=result['source']['sha256'],mesh_sha256=result['asset']['sha256'])
        path=data/'interval-motion.json';save(path,result);return root,path,result

    def test_normalizer_exact_flags_no_pose_substitution_and_no_mutations(self):
        root,path,doc=self.fixture();before=path.read_bytes();result=normalize(root,path)
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(result['comparison_counts'],{'fair_same_pts_samples':2,'candidate_supported':1,'baseline_supported':1,'both_supported':1,'references_excluded':1})
        rows=result['episodes'][0]['frames'];self.assertIsNone(rows[2]['pose']);self.assertIsNone(rows[2]['baseline']['pose'])
        self.assertEqual(rows[1]['baseline']['original_status'],'unsupported_candidate_projection_retained')
        self.assertFalse(rows[1]['baseline']['original_supported']);self.assertEqual(rows[1]['baseline']['status'],'supported_nonreference_candidate')

    def test_bad_matrix_source_geometry_claim_or_false_count_fails(self):
        for field in ['rotation','withheld_pose','comparison','source','scale','split','gltf_uri','program','running','failed','boolean_api','temporal','point_partition','registered_frame','baseline_refit','supported_reference']:
            with self.subTest(field=field):
                root,path,doc=self.fixture()
                if field=='rotation':doc['episodes'][0]['frames'][1]['candidate_rotation'][0][0]=2
                if field=='withheld_pose':doc['episodes'][0]['frames'][2]['candidate_rotation']=[[1,0,0],[0,1,0],[0,0,1]]
                if field=='comparison':doc['comparison']['exact_pts_rows'][1]['baseline_supported']=False
                if field=='source':doc['source']['sha256']='0'*64
                if field=='scale':doc['asset']['physical_scale_known']=True
                if field=='split':doc['episodes'][0]['queries']['heldout_point_ids']=[1]
                if field=='program':doc['component_id']='invented_part'
                if field in {'running','failed'}:doc['status']=field
                if field=='boolean_api':doc['new_api_calls']=False
                if field=='temporal':doc['episodes'][0]['frames'][1]['temporal_split']='reference_selection'
                if field=='point_partition':doc['episodes'][0]['frames'][1]['training_visible_point_ids']=[5]
                if field=='registered_frame':doc['episodes'][0]['frames'][1]['sample_index']=50
                if field=='baseline_refit':doc['baseline_reassessment_policy']['new_inference_or_pose_refit']=True
                if field=='supported_reference':doc['episodes'][0]['frames'][0]['status']='supported_nonreference_candidate'
                if field=='gltf_uri':
                    p=root/'data/asset.glb';gl={'asset':{'version':'2.0'},'images':[{'uri':'https://outside.test/file.png'}]};b=json.dumps(gl).encode();b+=b' '*((-len(b))%4);p.write_bytes(struct.pack('<4sIIII',b'glTF',2,20+len(b),len(b),0x4e4f534a)+b);doc['asset']['sha256']=sha(p)
                save(path,doc)
                with self.assertRaises(ValueError):normalize(root,path)

    def test_optional_capture_advice_is_source_bound_and_cannot_admit_poses(self):
        root,path,doc=self.fixture();audit=root/'data/audit.json';save(audit,{'synthetic':True})
        frame=doc['episodes'][0]['frames'][2]
        feedback={'format':'source-capture-feedback','version':3,'program_case_id':'fixture',
            'interaction_path':doc['interaction_path'],'interaction_sha256':doc['interaction_sha256'],
            'source':doc['source'],'episodes':[{'component_id':'part','episode_id':'episode_01',
            'start_s':1,'end_s':2,'reference_sample_index':1,'diagnostics':[{'title':'Synthetic advice','action':'Capture again.'}],
            'supplemental_scope_observations':[{'sample_index':3,'track_id':'part',
            'source_image_sha256':frame['source_image_sha256'],'mask_sha256':frame['mask_sha256'],
            'audit_path':'data/audit.json','audit_sha256':sha(audit),'notes':'Synthetic fallible scope note.'}]}]}
        coach=root/'data/feedback.json';save(coach,feedback)
        before=path.read_bytes();result=normalize(root,path,coach)
        self.assertEqual(path.read_bytes(),before)
        self.assertFalse(result['capture_feedback']['used_for_tracking_selection_or_fitting'])
        self.assertIsNone(result['episodes'][0]['frames'][2]['pose'])
        self.assertEqual(result['comparison_counts']['candidate_supported'],1)
        self.assertNotEqual(result['id'],normalize(root,path)['id'])
        feedback['episodes'][0]['supplemental_scope_observations'][0]['mask_sha256']='0'*64;save(coach,feedback)
        with self.assertRaisesRegex(ValueError,'source/mask identity'):normalize(root,path,coach)

    def test_actual_math_modules_match_pinhole_for_10000_points_and_never_hold_samples(self):
        code=r'''
import assert from 'node:assert/strict';
import * as THREE from './workspace/robot-vendor/three.module.js';
import {sourceCameraMatrices,letterbox} from './workspace/geometry-camera.mjs';
import {exactSample,displayPose,poseMatrix,projectCV} from './workspace/motion-math.mjs';
let seed=17;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296};
const K=[[467.65,2.3,269.5],[0,481.7,479.5],[0,0,1]];let maximum=0;
for(let i=0;i<10000;i++){const axis=new THREE.Vector3(random(),random(),random()).normalize(),q=new THREE.Quaternion().setFromAxisAngle(axis,(random()-.5)*6),m=new THREE.Matrix4().makeRotationFromQuaternion(q),e=m.elements,R=[[e[0],e[4],e[8]],[e[1],e[5],e[9]],[e[2],e[6],e[10]]],t=[(random()-.5)*2,(random()-.5)*2,4+random()*4],p=[random()-.5,random()-.5,random()-.5];
 const expected=projectCV(p,R,t,K),native=new THREE.Vector4(...p,1).applyMatrix4(new THREE.Matrix4().fromArray(poseMatrix(R,t))),camera=sourceCameraMatrices([[1,0,0],[0,1,0],[0,0,1]],[0,0,0],K,540,960),clip=native.applyMatrix4(new THREE.Matrix4().fromArray(camera.world).invert()).applyMatrix4(new THREE.Matrix4().fromArray(camera.projection));const actual=[(clip.x/clip.w+1)*270-.5,(1-clip.y/clip.w)*480-.5];maximum=Math.max(maximum,Math.hypot(actual[0]-expected[0],actual[1]-expected[1]));}
assert(maximum<1e-9);const pose={R:[[1,0,0],[0,1,0],[0,0,1]],t:[0,0,3]},rows=[{timestamp_s:1,status:'supported_nonreference_candidate',pose},{timestamp_s:2,status:'withheld',pose}];assert.equal(exactSample(rows,1.1),null);assert.equal(exactSample(rows,.999),null);assert.equal(exactSample(rows,2.001),null);assert.equal(exactSample(rows,1),rows[0]);assert.equal(displayPose(rows[1]),null);assert.equal(displayPose({status:'reference_unverified',pose},'candidate',false),null);assert.equal(displayPose({status:'reference_unverified',pose}),pose);assert.equal(displayPose(rows[0],'baseline'),null);const box=letterbox(1000,500,540,960);for(const[k,v]of Object.entries({x:359.375,y:0,width:281.25,height:500}))assert(Math.abs(box[k]-v)<1e-9);console.log(JSON.stringify({projected_points:10000,maximum_pixel_residual:maximum,no_interpolation:true}));
'''
        result=subprocess.run(['node','--input-type=module','-e',code],cwd=ROOT,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
