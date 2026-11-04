import itertools,json,struct,unittest
import numpy as np

import declared_geometry_view_v1 as V
import declared_geometry_contract_v1 as D
import planar_assembly_calibration_v1 as C
import native_assembly_refinement_v1 as N
import observed_surface_projection_v1 as P
from native_appearance import load_native_glb
from tests.test_native_appearance import fixture


class PlanarMatingTests(unittest.TestCase):
    def test_lower_rear_pivot_avoids_gap_without_infinite_plane_constraint(self):
        p=np.array(list(itertools.product([-1.,1.],[0.,.04],[0.,1.4])))
        c=np.array(list(itertools.product([-1.1,1.1],[-.65,.65],[-.015,.015])))
        j={'parent_native_scale':1.,'child_native_scale':1.,'parent_anchor_native':[0.,.04,0.],
           'axis':[-1.,0.,0.],'parent_from_joint':np.eye(4).tolist(),'joint_from_child_zero':np.eye(4).tolist()}
        v,a,axis,b,r=C.calibrate(p,c,j,[0.,1.,0.],[0.,0.,1.])
        self.assertGreaterEqual(r['minimum_signed_deck_clearance_native'],-1e-8)
        self.assertLess(r['minimum_infinite_plane_clearance_native'],0.)
        closed=C.transform(v,a@b)@np.asarray(r['parent_frame'])
        pivot=np.asarray(r['pivot_parent_native'])@np.asarray(r['parent_frame'])
        self.assertAlmostEqual(pivot[2],closed[:,2].min(),places=7)
        np.testing.assert_allclose(np.ptp(closed,axis=0)[:2],[2.,1.4],atol=1e-6)


class DerivativeReproductionTests(unittest.TestCase):
    def test_real_glb_affine_reproduction_rejects_undeclared_extra_motion(self):
        raw=fixture([{'vertices':[[0,0,0],[1,0,0],[0,1,0]],'material':0}],materials=[{'doubleSided':True}],images=[])
        matrix=np.eye(4);matrix[0,0]=.94;matrix[1,1]=1.1
        m=load_native_glb(raw);vertices=C.transform(m.vertices,matrix);derived,_=N.rewrite_positions(raw,vertices)
        record={'original_asset':{'id':'old'},'derived_asset':{'id':'new'},'operations':[{'kind':'affine_positions','matrix':matrix.tolist()}]}
        read=lambda ref:raw if ref['id']=='old'else derived
        self.assertTrue(D.reproduce_asset(record,read)['source_geometry_reproduced'])
        record['operations'][0]['matrix'][0][3]=.02
        with self.assertRaisesRegex(ValueError,'positions'):D.reproduce_asset(record,read)

    def test_reader_confines_paths_and_never_accepts_physically_promoted_scope(self):
        for p in ['../secrets','data/../secret','data/.private/a.json','file:///tmp/a.json']:
            with self.assertRaises(ValueError):V.reference({'path':p,'sha256':'0'*64,'bytes':4})
        called=[]
        with self.assertRaisesRegex(ValueError,'physically upgraded'):
            V.validate({'format':V.FORMAT,'version':1,'claims':V.CLAIMS,'physical_accuracy_verified':True,'simulation_ready':False},lambda *a,**k:called.append(a))
        self.assertEqual(called,[])

    def test_observed_vertex_rewrite_preserves_geometry_and_unpainted_alpha(self):
        raw=fixture([{'vertices':[[0,0,1],[1,0,1],[0,1,1]],'colors':[[1.,0.,0.,1.]]*3}],materials=[],images=[])
        document,binary=P.unpack(raw);attribute=document['meshes'][0]['primitives'][0]['attributes']['COLOR_0'];accessor=document['accessors'][attribute];view=document['bufferViews'][accessor['bufferView']]
        accessor.update(componentType=5121,normalized=True);view['byteLength']=12
        start=view.get('byteOffset',0);body=bytearray(binary);body[start:start+12]=bytes([255,0,0,255]*3)
        encoded=json.dumps(document,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%4);size=12+8+len(encoded)+8+len(body)
        raw=struct.pack('<4sII',b'glTF',2,size)+struct.pack('<II',len(encoded),0x4e4f534a)+encoded+struct.pack('<II',len(body),0x004e4942)+body
        before=load_native_glb(raw);colors=before.vertex_colors.copy();colors[1]=[1,2,3];changed=P.rewrite_vertex_colors(raw,colors);after=load_native_glb(changed)
        np.testing.assert_array_equal(before.vertices,after.vertices);np.testing.assert_array_equal(before.faces,after.faces)
        np.testing.assert_array_equal(after.vertex_colors,colors)
        doc,body=P.unpack(changed);a=doc['accessors'][doc['meshes'][0]['primitives'][0]['attributes']['COLOR_0']];v=doc['bufferViews'][a['bufferView']];rgba=np.frombuffer(body[v['byteOffset']:v['byteOffset']+12],np.uint8).reshape(3,4)
        np.testing.assert_array_equal(rgba[:,3],[255,255,255])


if __name__=='__main__':unittest.main()
