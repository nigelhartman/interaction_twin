import json,struct,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import trimesh
import large_native_evidence_v1 as L
import portable_twin as P

class LargeNativeEvidenceTests(unittest.TestCase):
 def setUp(self):
  self.mesh=trimesh.creation.icosphere(subdivisions=3);self.v=np.asarray(self.mesh.vertices);self.f=np.asarray(self.mesh.faces);self.c=np.column_stack([np.arange(len(self.v))%255,np.zeros((len(self.v),2),int),np.full(len(self.v),255)]).astype('u1')
 def rewrite(self,raw,change):
  n=struct.unpack_from('<I',raw,12)[0];doc=json.loads(raw[20:20+n]);change(doc);b=raw[20+n+8:];js=json.dumps(doc,separators=(',',':')).encode();js+=b' '*((-len(js))%4);return struct.pack('<III',0x46546c67,2,28+len(js)+len(b))+struct.pack('<II',len(js),0x4e4f534a)+js+struct.pack('<II',len(b),0x004e4942)+b
 def test_original_colour_geometry_roundtrip(self):
  raw=L.encode_colour_mesh(self.v,self.f,self.c);v,f,c=L.colour_mesh(raw);self.assertTrue(np.allclose(v,self.v));self.assertTrue(np.array_equal(f,self.f));self.assertTrue(np.array_equal(c,self.c));self.assertEqual(P.decode_glb_bytes(raw)['mesh_stats']['faces'],len(self.f))
 def test_transformed_archive_nodes_rejected(self):
  raw=L.encode_colour_mesh(self.v,self.f,self.c);raw=self.rewrite(raw,lambda d:d['nodes'][0].update(translation=[1,0,0]))
  with self.assertRaisesRegex(ValueError,'transformed'):L.colour_mesh(raw)
 def test_external_buffers_rejected(self):
  raw=L.encode_colour_mesh(self.v,self.f,self.c);raw=self.rewrite(raw,lambda d:d['buffers'][0].update(uri='https://invalid/mesh.bin'))
  with self.assertRaisesRegex(ValueError,'embedded buffer'):L.colour_mesh(raw)
 def test_nonfinite_archive_positions_rejected(self):
  v=self.v.copy();v[0,0]=float('nan');raw=L.encode_colour_mesh(v,self.f,self.c)
  with self.assertRaisesRegex(ValueError,'Nonfinite'):L.colour_mesh(raw)
 def test_out_of_range_indices_rejected(self):
  f=self.f.copy();f[0,0]=len(self.v);raw=L.encode_colour_mesh(self.v,f,self.c)
  with self.assertRaisesRegex(ValueError,'indices'):L.colour_mesh(raw)
 def test_reduction_is_deterministic_and_native_untouched(self):
  mesh=trimesh.creation.icosphere(subdivisions=5);v=np.asarray(mesh.vertices).copy();f=np.asarray(mesh.faces).copy();c=np.tile([70,100,150,255],(len(v),1)).astype('u1');original_v=v.copy();original_f=f.copy();original_c=c.copy()
  a,decoded,proof=L.derive_preview(v,f,c,target_faces=20000);b,_,again=L.derive_preview(v,f,c,target_faces=20000)
  self.assertEqual(a,b);self.assertEqual(proof,again);self.assertLessEqual(decoded['mesh_stats']['faces'],20000);self.assertLessEqual(len(a),P.MAX_ASSET)
  self.assertTrue(np.array_equal(v,original_v));self.assertTrue(np.array_equal(f,original_f));self.assertTrue(np.array_equal(c,original_c))
  _,_,colours=L.colour_mesh(a);self.assertTrue(set(map(tuple,colours)).issubset(set(map(tuple,original_c))))
 def test_original_caps_unchanged(self):
  before=(P.MAX_ASSET,P.MAX_TRIANGLE_INDICES,P.MAX_RENDERED_VERTICES)
  with self.assertRaises(ValueError):P.decode_glb_bytes(b'0'*(P.MAX_ASSET+1))
  self.assertEqual(before,(16*2**20,3_000_000,1_000_000))
 def test_reference_rejects_outside_private_evidence_before_read(self):
  with patch.object(Path,'read_bytes',side_effect=AssertionError('Must not read outside data/output')):
   with self.assertRaisesRegex(ValueError,'Ordinary'):L.reference('/etc/passwd')
 def test_check_detects_tampered_bytes(self):
  with tempfile.TemporaryDirectory(dir=L.ROOT/'output')as tmp:
   p=Path(tmp)/'item.bin';p.write_bytes(b'original');ref=L.reference(p);p.write_bytes(b'changed')
   with self.assertRaisesRegex(ValueError,'changed'):L.checked(ref)

if __name__=='__main__':unittest.main()
