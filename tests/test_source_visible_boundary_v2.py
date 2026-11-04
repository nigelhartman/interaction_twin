import copy,hashlib,json,unittest
import numpy as np
from source_visible_boundary_mesh_v1 import conservative_triangle_inside,build
from source_visible_flexure_verify_v1 import covered_pixel_squares_inside,verify_cached_depth
from source_visible_flexure_finalize_v1 import footprint_keep
from source_visible_flexure_contract_v1 import ROOT,validate

def square_intersects_triangle(triangle,x,y):
    """Independent polygon-clipping oracle, including zero-area contact."""
    polygon=[np.asarray(p,float)for p in triangle]
    for axis,bound,lower in [(0,x-.5,True),(0,x+.5,False),(1,y-.5,True),(1,y+.5,False)]:
        result=[]
        if not polygon:return False
        previous=polygon[-1];inside_previous=previous[axis]>=bound if lower else previous[axis]<=bound
        for current in polygon:
            inside=current[axis]>=bound if lower else current[axis]<=bound
            if inside!=inside_previous:
                weight=(bound-previous[axis])/(current[axis]-previous[axis]);result.append(previous+weight*(current-previous))
            if inside:result.append(current)
            previous=current;inside_previous=inside
        polygon=result
    return bool(polygon)

class ConservativePixelProofTests(unittest.TestCase):
    def test_unused_rectangle_corner_is_allowed_only_in_explicit_new_proof(self):
        xy=np.array([[1,1],[1,5],[5,1]]);safe=np.ones((8,8),bool);safe[5,5]=False
        self.assertFalse(footprint_keep(xy,np.array([[0,1,2]]),safe)[0])
        self.assertTrue(conservative_triangle_inside(xy,safe));self.assertTrue(covered_pixel_squares_inside(xy,safe))
    def test_hole_and_edge_touch_are_not_ignored(self):
        xy=np.array([[1,1],[1,5],[5,1]])
        for pixel in [(2,2),(3,3),(1,1),(5,1)]:
            safe=np.ones((8,8),bool);safe[pixel[1],pixel[0]]=False
            self.assertFalse(conservative_triangle_inside(xy,safe));self.assertFalse(covered_pixel_squares_inside(xy,safe))
    def test_both_proofs_match_independent_clipping_for_random_pixels(self):
        rng=np.random.default_rng(701)
        for _ in range(160):
            xy=rng.integers(1,8,(3,2));pixel=rng.integers(0,9,2);safe=np.ones((9,9),bool);safe[pixel[1],pixel[0]]=False;expected=not square_intersects_triangle(xy,*pixel)
            self.assertEqual(conservative_triangle_inside(xy,safe),expected,(xy,pixel));self.assertEqual(covered_pixel_squares_inside(xy,safe),expected,(xy,pixel))
    def test_real_boundary_topology_is_bounded_and_avoids_hole(self):
        safe=np.zeros((80,100),bool);safe[5:75,5:95]=True;safe[28:51,40:62]=False;rgb=np.zeros((80,100,3),np.uint8);y,x=np.mgrid[9:71:5,9:91:5];known=safe[y,x];xy=np.c_[x[known],y[known]];depth=np.ones(len(xy),np.float32);K=np.array([[100,0,50],[0,100,40],[0,0,1]])
        mesh=build(rgb,safe,K,xy,depth,5);self.assertLessEqual(len(mesh['vertices']),1800);self.assertGreater(len(mesh['faces']),0)
        self.assertTrue(all(covered_pixel_squares_inside(mesh['source_xy'][f],safe)for f in mesh['faces']))

class CachedDepthProofTests(unittest.TestCase):
    def setUp(self):
        self.parent={'source_xy':np.array([[2,2],[2,6],[6,2]]),'depth':np.array([1,2,3],np.float32)};self.mapping={'source_xy':np.array([[3,3]]),'depth':np.array([1.75],np.float32),'parent_depth_vertex_ids':np.array([[0,1,2]],np.uint16),'parent_depth_weights':np.array([[.5,.25,.25]]),'depth_sampling_modes':np.array([0],np.uint8)};self.row={'depth_sampling':{'maximum_interpolation_parent_edge_px':10,'maximum_nearest_sampling_distance_px':6}};self.safe=np.ones((10,10),bool)
    def test_exact_barycentric_depth_is_reproducible(self):
        self.assertTrue(verify_cached_depth(self.mapping,self.parent,self.row,self.safe)['exact_cached_depth_weight_derivation'])
    def test_changed_depth_weights_coordinates_and_unknown_rejected(self):
        for key,value in [('depth',np.array([1.76],np.float32)),('parent_depth_weights',np.array([[.6,.2,.2]])),('source_xy',np.array([[4,3]])),('parent_depth_vertex_ids',np.array([[0,1,3]]))]:
            mapping=copy.deepcopy(self.mapping);mapping[key]=value
            with self.assertRaises(ValueError):verify_cached_depth(mapping,self.parent,self.row,self.safe)
        self.safe[4,3]=False
        with self.assertRaisesRegex(ValueError,'unknown'):verify_cached_depth(self.mapping,self.parent,self.row,self.safe)
    def test_nearest_cannot_be_distant_wrong_or_cross_unknown(self):
        m={'source_xy':np.array([[2,1]]),'depth':np.array([1],np.float32),'parent_depth_vertex_ids':np.array([[0,0,0]],np.uint16),'parent_depth_weights':np.array([[1.,0.,0.]]),'depth_sampling_modes':np.array([1],np.uint8)}
        self.assertEqual(verify_cached_depth(m,self.parent,self.row,self.safe)['bounded_nearest_samples'],1)
        m['source_xy']=np.array([[2,6]])
        with self.assertRaisesRegex(ValueError,'not nearest'):verify_cached_depth(m,self.parent,self.row,self.safe)
        m['source_xy']=np.array([[2,1]]);self.safe[1,2]=False
        with self.assertRaisesRegex(ValueError,'unknown'):verify_cached_depth(m,self.parent,self.row,self.safe)

class SavedBoundaryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path=ROOT/'data/source-visible-flexure/wallet-20261104-v5-1/descriptor.json'
        if not cls.path.is_file():raise unittest.SkipTest('Saved source-visible candidate is excluded from source-only package')
        cls.doc=json.loads(cls.path.read_text())
    def read(self,r):return (ROOT/r['path']).read_bytes()
    def test_original_unknowns_and_typed_depth_maps_are_preserved(self):
        out=validate(self.doc,self.read);self.assertEqual(len(out['supported_sources']),91);self.assertEqual(out['unknown_sources'],[38,54,55,57,58,59,60,67]);paths=[r['path']for r in out['artifacts']];self.assertIn(self.doc['cached_depth_parent']['path'],paths);self.assertTrue(any('wallet-20261104-v4/source-'in p and p.endswith('.npz')for p in paths));self.assertFalse(any(p.startswith('models/')or p.startswith('output/')for p in paths))
    def test_parent_descriptor_validation_has_no_unlisted_artifact_reads(self):
        out=validate(self.doc,self.read);allowed={r['path']:r for r in out['artifacts']}
        def fenced_read(r):
            self.assertEqual(allowed.get(r['path']),r,'Unlisted parent dependency was read');return self.read(r)
        result=validate(self.doc,fenced_read);self.assertEqual(result['supported_sources'],out['supported_sources'])
    def test_refined_geometry_cannot_claim_unchanged_parent(self):
        d=copy.deepcopy(self.doc);d['samples'][0]['geometry_unchanged_from_footprint_verified_parent']=True
        with self.assertRaisesRegex(ValueError,'retessellated'):validate(d,self.read)
    def test_unrelated_map_and_parent_unknown_are_rejected(self):
        d=copy.deepcopy(self.doc);d['samples'][0]['cached_depth_parent_map']=d['samples'][1]['cached_depth_parent_map']
        with self.assertRaisesRegex(ValueError,'matching original'):validate(d,self.read)
        d=copy.deepcopy(self.doc);d['samples'][37]=copy.deepcopy(d['samples'][36]);d['samples'][37]['source_sample_index']=38;d['samples'][37]['timestamp_s']=self.doc['samples'][37]['timestamp_s']
        with self.assertRaisesRegex(ValueError,'unknown barrier'):validate(d,self.read)
    def test_proof_downgrade_and_unknown_version_are_rejected(self):
        for version in [1,3,True]:
            d=copy.deepcopy(self.doc);d['source_footprint_proof_version']=version
            with self.assertRaises(ValueError):validate(d,self.read)
    def test_source_camera_and_sampling_bounds_cannot_drift(self):
        for key in ['camera','distance','triangle']:
            d=copy.deepcopy(self.doc)
            if key=='camera':d['camera']['K'][0][0]*=1.01
            elif key=='distance':d['samples'][0]['depth_sampling']['maximum_nearest_sampling_distance_px']+=1
            else:d['policy']['triangle_support']='ignore masks'
            with self.assertRaises(ValueError):validate(d,self.read)

if __name__=='__main__':unittest.main()
