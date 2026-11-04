import copy,hashlib,json,struct,unittest
import source_appearance_states_v1 as A

class SourceAppearanceStatesTest(unittest.TestCase):
    def fixture(self):
        data={}
        def ref(name,value,raw=False):
            b=value if raw else json.dumps(value).encode();r={'path':'data/fixture/'+name,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)};data[r['path']]=b;return r
        source={'sha256':'1'*64,'width':10,'height':10};asset=ref('asset.glb',b'asset',True);rgb=ref('rgb.jpg',b'rgb',True);mask=ref('mask.png',b'mask',True);matrix=[[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]];camera={'width':10,'height':10,'K':[[10,0,5],[0,10,5],[0,0,1]]}
        evidence=[{'source_index':i,'timestamp_s':float(i),'frame_path':rgb['path'],'frame_sha256':rgb['sha256']}for i in range(4)]
        program={'source':source,'evidence':evidence};pr=ref('program.json',program)
        poses=[{'source_sample_index':i,'timestamp_s':float(i),'status':'estimated'if i!=3 else'unknown','source_mask':mask,'camera_from_native_mesh':matrix}for i in range(4)]
        scene={'source':source,'camera':camera,'components':[{'id':'lid','asset':asset,'poses':poses}]};sr=ref('scene.json',scene)
        previous={'scene_ref':sr,'source':source,'evidence_program':pr,'component_assets':[{'component_id':'lid','native_glb':asset}]};prev=ref('previous.json',previous)
        patch=ref('patch.itcp',struct.pack('<4sIII',b'ITCP',1,3,1)+struct.pack('<IHHBBB',1,2,3,10,20,30),True)
        keys=[{'id':k,'source_sample_index':i,'timestamp_s':float(i),'appearance_label':label,'source_rgb_ref':rgb,'source_mask_ref':mask,'camera_from_native_mesh':matrix,'source_camera':camera,'patch_ref':patch,'painted_vertex_count':1,'native_fallback_vertex_count':2}for k,i,label in [('dark',0,'dark'),('lit',2,'lit')]]
        rows=[{'source_sample_index':i,'timestamp_s':float(i),'status':'source_observed_appearance_estimate'if i!=3 else'unknown','keyframe_id':('dark'if i<2 else'lit')if i!=3 else None,'appearance_label':'dark'if i<2 else'lit','source_rgb_ref':rgb,'source_mask_ref':mask}for i in range(4)]
        doc={'format':A.FORMAT,'version':1,'display_policy':A.POLICY,'physical_accuracy_verified':False,'simulation_ready':False,'electronic_state_verified':False,'previous_derived_assembly_ref':prev,'scene_ref':sr,'evidence_program':pr,'source':source,'baseline_asset_ref':asset,'component_id':'lid','vertex_count':3,'keyframes':keys,'observations':rows}
        return doc,lambda r,**kw:data[r['path']]
    def test_valid_source_clock_and_rgb_bindings(self):
        d,read=self.fixture();self.assertEqual(A.validate(d,read)['scene']['source'],d['source'])
    def test_unknown_and_transition_gap_withhold(self):
        d,_=self.fixture();self.assertEqual(A.at(d,0)['id'],'dark');self.assertEqual(A.at(d,1)['id'],'dark');self.assertIsNone(A.at(d,1.2));self.assertIsNone(A.at(d,2.2));self.assertIsNone(A.at(d,3));self.assertIsNone(A.at(d,-.1))
    def test_mutated_bytes_rejected(self):
        d,read=self.fixture();p=d['keyframes'][0]['patch_ref']['path']
        with self.assertRaisesRegex(ValueError,'hash/size'):A.validate(d,lambda r,**kw:read(r)+b'x'if r['path']==p else read(r))
    def test_mismatched_frame_and_pose_rejected(self):
        for mutate in [lambda d:d['keyframes'][0].update(timestamp_s=.1),lambda d:d['observations'][0]['source_rgb_ref'].update(sha256='2'*64),lambda d:d['keyframes'][0]['camera_from_native_mesh'][0].__setitem__(0,2),lambda d:d.update(electronic_state_verified=True)]:
            d,read=self.fixture();mutate(d)
            with self.assertRaises(ValueError):A.validate(d,read)
    def test_duplicate_vertex_and_out_of_range_rejected(self):
        for rows in [[(1,2,3,1,2,3),(1,2,3,4,5,6)],[(3,2,3,1,2,3)]]:
            raw=struct.pack('<4sIII',b'ITCP',1,3,len(rows))+b''.join(struct.pack('<IHHBBB',*r)for r in rows)
            with self.assertRaises(ValueError):A.decode_patch(raw)
    def test_unknown_optional_rgb_cannot_admit_unrelated_source(self):
        d,read=self.fixture();d['observations'][3]['source_rgb_ref']={'path':'data/fixture/unrelated.jpg','sha256':'f'*64,'bytes':10}
        with self.assertRaisesRegex(ValueError,'unknown-row'):A.validate(d,read)

if __name__=='__main__':unittest.main()
