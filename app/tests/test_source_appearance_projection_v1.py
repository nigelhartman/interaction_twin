import unittest
import source_appearance_projection_v1 as A
from tests.test_native_appearance import fixture

class AppearanceNativeScopeTest(unittest.TestCase):
    def test_float_colour_native_cannot_claim_exact_legacy_byte_display(self):
        raw=fixture([{'vertices':[[0,0,1],[1,0,1],[0,1,1]],'colors':[[.2,.3,.4,1.]]*3}],materials=[],images=[])
        with self.assertRaisesRegex(ValueError,'byteRGB|texture/float'):A.native_vertex_scope(raw,3)

    def test_wrong_native_vertex_count_rejected_before_projection(self):
        raw=fixture([{'vertices':[[0,0,1],[1,0,1],[0,1,1]],'colors':[[.2,.3,.4,1.]]*3}],materials=[],images=[])
        with self.assertRaisesRegex(ValueError,'byteRGB'):A.native_vertex_scope(raw,4)

if __name__=='__main__':unittest.main()
