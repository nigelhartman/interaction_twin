import copy,hashlib,io,json,unittest
from pathlib import Path
import numpy as np,trimesh
from PIL import Image
from source_visible_flexure_contract_v1 import ROOT,validate,validate_descriptor
from source_visible_flexure_finalize_v1 import footprint_keep,unlit
from source_visible_texture_v1 import exact_source_texture,visible_components
from source_visible_flexure_verify_v1 import decode_surface
class SourceSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path=ROOT/'data/source-visible-flexure/wallet-20261104-v4/descriptor.json';cls.doc=json.loads(cls.path.read_text())
    def read(self,r):return (ROOT/r['path']).read_bytes()
    def test_real_exact_transport_and_unknowns(self):
        r=validate(self.doc,self.read);self.assertEqual(len(r['supported_sources']),91);self.assertEqual(r['unknown_sources'],[38,54,55,57,58,59,60,67]);self.assertTrue(all(x['path'].startswith('data/')for x in r['artifacts']));self.assertFalse(any('model.pt'in x['path']or'output/'in x['path']for x in r['artifacts']))
    def test_original_unknown_cannot_be_promoted(self):
        d=copy.deepcopy(self.doc);d['samples'][37]=copy.deepcopy(d['samples'][36]);d['samples'][37]['source_sample_index']=38;d['samples'][37]['timestamp_s']=self.doc['samples'][37]['timestamp_s']
        with self.assertRaisesRegex(ValueError,'unknown barrier'):validate(d,self.read)
    def test_clock_and_source_camera_cannot_drift(self):
        for mutation in ['clock','camera']:
            d=copy.deepcopy(self.doc)
            if mutation=='clock':d['samples'][27]['timestamp_s']+=.001
            else:d['camera']['K'][0][0]*=1.01
            with self.assertRaises(ValueError):validate(d,self.read)
    def test_checkpoint_cannot_be_admitted_as_data_route(self):
        d=copy.deepcopy(self.doc);d['samples'][0]['depth_model']['path']='models/moge/model.pt'
        with self.assertRaisesRegex(ValueError,'nonroutable'):validate(d,self.read)
    def test_triangle_pixel_rectangle_cannot_cross_unknown(self):
        xy=np.array([[1,1],[1,5],[5,1],[5,5]]);f=np.array([[0,1,2],[1,3,2]]);m=np.ones((8,8),bool);self.assertTrue(footprint_keep(xy,f,m).all());m[3,3]=False;self.assertFalse(footprint_keep(xy,f,m).any())
    def test_visibility_uses_actual_retained_triangle_not_mask_reference(self):
        xy=np.array([[1,1],[1,3],[3,1]]);f=np.array([[0,1,2]]);host=np.ones((6,6),bool);flap=np.zeros((6,6),bool);flap[5,5]=True
        self.assertEqual(visible_components(xy,f,{'host':host,'flap':flap},['host','flap']),['host'])
    def test_exact_JPEG_UV_preserves_geometry_and_source_RGB_separately(self):
        image=Image.new('RGB',(8,8),(90,110,130));buffer=io.BytesIO();image.save(buffer,format='JPEG');source=buffer.getvalue();xy=np.array([[1,1],[1,5],[5,1]]);v=np.array([[0,0,1],[0,.4,1],[.4,0,1]],float);faces=np.array([[0,1,2]]);colors=np.array([[90,110,130]]*3,np.uint8);base=unlit(trimesh.Trimesh(vertices=v,faces=faces,vertex_colors=colors,process=False).export(file_type='glb'));new=decode_surface(exact_source_texture(base,xy,source,8,8));old=decode_surface(base)
        np.testing.assert_array_equal(new['vertices'],old['vertices']);np.testing.assert_array_equal(new['faces'],old['faces']);np.testing.assert_array_equal(new['colors'],colors);np.testing.assert_array_equal(new['UV'],(xy.astype(np.float32)+.5)/8);self.assertNotIn('COLOR_0',new['attributes']);self.assertIn('_SOURCE_RGB',new['attributes']);gd=new['document'];iv=gd['bufferViews'][gd['images'][0]['bufferView']];self.assertEqual(new['binary'][iv['byteOffset']:iv['byteOffset']+iv['byteLength']],source)
if __name__=='__main__':unittest.main()
