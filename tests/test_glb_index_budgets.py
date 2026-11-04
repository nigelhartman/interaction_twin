"""Role-specific native index bounds and aggregate node expansion, no models."""
import copy
import json
import struct
import unittest
from unittest.mock import patch

import portable_twin as P
from tests.test_portable_twin import glb, pack_glb


def indexed(count=3, component=5125, kind='SCALAR', normalized=False, instances=1, vertex_count=3):
    xyz=[0,0,0,1,0,0,0,1,0]+[.5,.5,.5]*(vertex_count-3)
    positions=struct.pack('<'+'f'*len(xyz),*xyz)
    codes={5120:'b',5121:'B',5122:'h',5123:'H',5125:'I',5126:'f'}
    values=([0,1,2]*((count+2)//3))[:count]
    indices=struct.pack('<'+codes[component]*count,*values)
    raw=positions+indices;length=len(raw);raw+=b'\0'*(-len(raw)%4)
    doc={'asset':{'version':'2.0'},'buffers':[{'byteLength':length}],
         'bufferViews':[{'byteLength':len(positions)},{'byteOffset':len(positions),'byteLength':len(indices)}],
         'accessors':[{'bufferView':0,'componentType':5126,'count':vertex_count,'type':'VEC3'},
                      {'bufferView':1,'componentType':component,'count':count,'type':kind,'normalized':normalized}],
         'meshes':[{'primitives':[{'attributes':{'POSITION':0},'indices':1}]}],
         'nodes':[{'mesh':0} for _ in range(instances)],'scenes':[{'nodes':list(range(instances))}],'scene':0}
    return doc,raw


class GLBIndexBudgetTests(unittest.TestCase):
    def test_index_role_can_exceed_attribute_count_without_changing_semantics(self):
        doc,raw=indexed(12)
        with patch.object(P,'MAX_ACCESSOR_COUNT',3),patch.object(P,'MAX_TRIANGLE_INDICES',12):
            decoded=P.decode_glb_bytes(pack_glb(doc,raw))
        self.assertEqual(decoded['primitives'][0]['faces'],[[0,1,2]]*4)
        self.assertEqual(decoded['mesh_stats']['rendered_vertices'],3)

    def test_exact_index_cap_passes_and_one_triangle_over_fails(self):
        with patch.object(P,'MAX_TRIANGLE_INDICES',9):
            self.assertEqual(P.decode_glb_bytes(pack_glb(*indexed(9)))['mesh_stats']['faces'],3)
            with self.assertRaisesRegex(ValueError,'accessor'):P.decode_glb_bytes(pack_glb(*indexed(12)))

    def test_index_role_remains_unsigned_scalar_and_nonnormalized(self):
        for component,kind,normalized in [(5120,'SCALAR',False),(5122,'SCALAR',False),(5126,'SCALAR',False),
                                           (5125,'VEC3',False),(5125,'SCALAR',True)]:
            with self.subTest(component=component,kind=kind,normalized=normalized),self.assertRaisesRegex(ValueError,'unsigned SCALAR'):
                P.decode_glb_bytes(pack_glb(*indexed(3,component,kind,normalized)))
        for component in [5121,5123,5125]:
            self.assertEqual(P.decode_glb_bytes(pack_glb(*indexed(3,component)))['mesh_stats']['faces'],1)

    def test_aggregate_index_cap_counts_node_instances_before_list_expansion(self):
        with patch.object(P,'MAX_TRIANGLE_INDICES',9):
            self.assertEqual(P.decode_glb_bytes(pack_glb(*indexed(3,instances=3)))['mesh_stats']['faces'],3)
            with self.assertRaisesRegex(ValueError,'expanded scene'):P.decode_glb_bytes(pack_glb(*indexed(3,instances=4)))

    def test_position_and_color_accessors_do_not_get_the_larger_index_budget(self):
        doc,raw=indexed(6)
        with patch.object(P,'MAX_ACCESSOR_COUNT',2),patch.object(P,'MAX_TRIANGLE_INDICES',9):
            with self.assertRaisesRegex(ValueError,'accessor'):P.decode_glb_bytes(pack_glb(doc,raw))
        # COLOR_0 deliberately references the six-element accessor: count guard
        # must reject it as an attribute even though index-role accepts six.
        doc['meshes'][0]['primitives'][0]['attributes']['COLOR_0']=1
        with patch.object(P,'MAX_ACCESSOR_COUNT',3),patch.object(P,'MAX_TRIANGLE_INDICES',9):
            with self.assertRaisesRegex(ValueError,'accessor'):P.decode_glb_bytes(pack_glb(doc,raw))

    def test_existing_small_native_output_is_identical_under_old_and_new_index_caps(self):
        raw=glb(texture=True)
        with patch.object(P,'MAX_TRIANGLE_INDICES',1_500_000):old=P.decode_glb_bytes(raw)
        self.assertEqual(old,P.decode_glb_bytes(raw))

    def test_per_primitive_position_limit_uses_same_unchanged_aggregate_vertex_bound(self):
        with patch.object(P,'MAX_RENDERED_VERTICES',4):
            self.assertEqual(P.decode_glb_bytes(pack_glb(*indexed(vertex_count=4)))['mesh_stats']['rendered_vertices'],4)
            with self.assertRaisesRegex(ValueError,'positions'):P.decode_glb_bytes(pack_glb(*indexed(vertex_count=5)))
        with patch.object(P,'MAX_RENDERED_VERTICES',6):
            self.assertEqual(P.decode_glb_bytes(pack_glb(*indexed(instances=2)))['mesh_stats']['rendered_vertices'],6)
            with self.assertRaisesRegex(ValueError,'expanded scene'):P.decode_glb_bytes(pack_glb(*indexed(instances=3)))


if __name__=='__main__':unittest.main()
