"""Analytic synthetic GLB fixtures, never claimed as learned reconstructions."""
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from PIL import Image

from alignment_render import render_mesh
from native_appearance import (alignment_contract, load_native_glb, render_native_mesh,
                               render_native_mask, ray_native_mesh_points)


def fixture(parts=None, *, materials=None, images=None, samplers=None, nodes=None, changes=None):
    """Build an embedded temporary/test-only GLB from small analytic arrays."""
    parts = parts or [{'vertices': [[0, 0, 1], [8, 0, 1], [0, 8, 1]], 'uv': [[0, 0], [1, 0], [0, 1]], 'material': 0}]
    image = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], np.uint8)
    images = [image] if images is None else images
    materials = ([{'doubleSided': True, 'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}}}] if materials is None else materials)
    document = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}],
                'nodes': nodes or [{'mesh': 0}], 'meshes': [{'primitives': []}],
                'materials': materials, 'bufferViews': [], 'accessors': [],
                'images': [], 'textures': [], 'samplers': samplers or [{'magFilter': 9728, 'minFilter': 9728, 'wrapS': 33071, 'wrapT': 33071}]}
    binary = bytearray()
    def buffer(data):
        while len(binary) % 4:
            binary.append(0)
        index = len(document['bufferViews'])
        document['bufferViews'].append({'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(data)})
        binary.extend(data)
        return index
    def accessor(value, kind, component=5126):
        array = np.asarray(value, dtype='<f4' if component == 5126 else '<u2')
        index = len(document['accessors'])
        document['accessors'].append({'bufferView': buffer(array.tobytes()), 'componentType': component, 'count': len(array), 'type': kind})
        return index
    for part in parts:
        attributes = {'POSITION': accessor(part['vertices'], 'VEC3')}
        if 'uv' in part:
            attributes['TEXCOORD_0'] = accessor(part['uv'], 'VEC2')
        if 'colors' in part:
            attributes['COLOR_0'] = accessor(part['colors'], 'VEC4' if len(part['colors'][0]) == 4 else 'VEC3')
        primitive = {'attributes': attributes, 'indices': accessor(part.get('indices', [[0], [1], [2]]), 'SCALAR', 5123)}
        if 'material' in part:
            primitive['material'] = part['material']
        document['meshes'][0]['primitives'].append(primitive)
    for array in images:
        stream = io.BytesIO(); Image.fromarray(array).save(stream, format='PNG')
        document['images'].append({'bufferView': buffer(stream.getvalue()), 'mimeType': 'image/png'})
        document['textures'].append({'source': len(document['images']) - 1, 'sampler': 0})
    document['buffers'] = [{'byteLength': len(binary)}]
    if changes:
        changes(document)
    data = json.dumps(document, separators=(',', ':')).encode()
    data += b' ' * (-len(data) % 4)
    binary += b'\0' * (-len(binary) % 4)
    chunks = struct.pack('<II', len(data), 0x4E4F534A) + data + struct.pack('<II', len(binary), 0x004E4942) + binary
    return struct.pack('<III', 0x46546C67, 2, 12 + len(chunks)) + chunks


K = np.eye(3)
R = np.eye(3)
T = np.zeros(3)


class NativeAppearanceTests(unittest.TestCase):
    def render(self, raw, width=10, height=10):
        return render_native_mesh(load_native_glb(raw), K, R, T, width, height)

    def test_native_uv_and_original_bytes_preserved(self):
        raw = fixture(); before = hashlib.sha256(raw).hexdigest()
        mesh = load_native_glb(raw)
        self.assertIsNone(mesh.vertex_colors)
        self.assertTrue(mesh.receipt['uv_preserved'])
        self.assertFalse(mesh.receipt['uv_to_vertex_color_conversion'])
        self.assertEqual(mesh.receipt['source_glb_sha256'], before)
        np.testing.assert_array_equal(mesh.appearance.primitives[0].uv, [[0, 0], [1, 0], [0, 1]])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), before)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'native.glb'; path.write_bytes(raw)
            self.assertEqual(load_native_glb(path).receipt, mesh.receipt)
            self.assertEqual(path.read_bytes(), raw)

    def test_known_image_corners_use_gltf_top_left_origin(self):
        result = self.render(fixture())
        np.testing.assert_array_equal(result['rgb'][1, 1], [255, 0, 0])
        np.testing.assert_array_equal(result['rgb'][1, 6], [0, 255, 0])
        np.testing.assert_array_equal(result['rgb'][6, 1], [0, 0, 255])

    def test_perspective_correct_uv_differs_from_affine(self):
        image = np.tile(np.arange(16, dtype=np.uint8)[None, :, None] * 16, (1, 1, 3))
        parts = [{'vertices': [[0, 0, 1], [32, 0, 4], [0, 32, 4]], 'uv': [[0, 0], [1, 0], [0, 1]], 'material': 0}]
        result = self.render(fixture(parts, images=[image]))
        # Pixel2,2 has screen barycentric(.5,.25,.25), perspective u=.1,
        # versus affine u=.25. A16-column nearest texture picks column1, not4.
        np.testing.assert_array_equal(result['rgb'][2, 2], [16, 16, 16])
        self.assertAlmostEqual(float(result['depth'][2, 2]), 1.6, places=6)

    def test_texture_vertex_tint_factor_multiply_in_linear_space(self):
        parts = [{'vertices': [[0, 0, 1], [8, 0, 1], [0, 8, 1]], 'uv': [[.25, .25]] * 3,
                  'colors': [[.5, .5, .5]] * 3, 'material': 0}]
        materials = [{'doubleSided': True, 'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}, 'baseColorFactor': [.5, .5, .5, 1]}}]
        result = self.render(fixture(parts, materials=materials, images=[np.full((1, 1, 3), 128, np.uint8)]))
        np.testing.assert_array_equal(result['rgb'][2, 2], [66, 66, 66])

    def test_bilinear_filter_uses_linear_texture_values(self):
        parts = [{'vertices': [[0, 0, 1], [8, 0, 1], [0, 8, 1]], 'uv': [[.5, .5]] * 3, 'material': 0}]
        sampler = [{'magFilter': 9729, 'minFilter': 9729, 'wrapS': 33071, 'wrapT': 33071}]
        result = self.render(fixture(parts, images=[np.array([[[0, 0, 0], [255, 255, 255]]], np.uint8)], samplers=sampler))
        np.testing.assert_array_equal(result['rgb'][2, 2], [188, 188, 188])

    def test_repeat_clamp_and_mirrored_repeat(self):
        parts = [{'vertices': [[0, 0, 1], [8, 0, 1], [0, 8, 1]], 'uv': [[1.25, .25]] * 3, 'material': 0}]
        for mode, expected in [(10497, [255, 0, 0]), (33071, [0, 255, 0]), (33648, [0, 255, 0])]:
            with self.subTest(mode=mode):
                sampler = [{'magFilter': 9728, 'minFilter': 9728, 'wrapS': mode, 'wrapT': 33071}]
                np.testing.assert_array_equal(self.render(fixture(parts, samplers=sampler))['rgb'][2, 2], expected)

    def test_uv_seams_remain_separate_vertices(self):
        parts = [{'vertices': [[0, 0, 1], [4, 0, 1], [0, 8, 1]], 'uv': [[.25, .25]] * 3, 'material': 0},
                 {'vertices': [[4, 0, 1], [8, 0, 1], [4, 8, 1]], 'uv': [[.75, .25]] * 3, 'material': 0}]
        mesh = load_native_glb(fixture(parts))
        self.assertEqual(len(mesh.vertices), 6)
        result = render_native_mesh(mesh, K, R, T, 10, 10)
        np.testing.assert_array_equal(result['rgb'][1, 1], [255, 0, 0])
        np.testing.assert_array_equal(result['rgb'][1, 5], [0, 255, 0])
        self.assertEqual(result['face_index'][1, 5], 1)

    def test_two_materials_share_zbuffer_and_global_face_ids(self):
        materials = [{'doubleSided': True, 'pbrMetallicRoughness': {'baseColorFactor': [0, 0, 1, 1]}},
                     {'doubleSided': True, 'pbrMetallicRoughness': {'baseColorFactor': [1, 0, 0, 1]}}]
        parts = [{'vertices': [[0, 0, 2], [16, 0, 2], [0, 16, 2]], 'material': 0},
                 {'vertices': [[0, 0, 1], [8, 0, 1], [0, 8, 1]], 'material': 1}]
        result = self.render(fixture(parts, materials=materials, images=[]))
        np.testing.assert_array_equal(result['rgb'][2, 2], [255, 0, 0])
        self.assertEqual(result['face_index'][2, 2], 1)
        self.assertEqual(result['depth'][2, 2], 1)

    def test_scene_transforms_instances_and_native_origin(self):
        nodes = [{'translation': [2, 3, 0], 'children': [1, 2]}, {'mesh': 0, 'translation': [1, 0, 0]}, {'mesh': 0, 'translation': [10, 0, 0]}]
        mesh = load_native_glb(fixture(nodes=nodes))
        self.assertEqual(len(mesh.vertices), 6)
        np.testing.assert_array_equal(mesh.vertices[0], [3, 3, 1])
        np.testing.assert_array_equal(mesh.vertices[3], [12, 3, 1])
        self.assertEqual(mesh.faces.tolist(), [[0, 1, 2], [3, 4, 5]])
        self.assertTrue(mesh.receipt['scene_node_transforms_applied'])

    def test_near_clipping_geometry_and_attributes(self):
        parts = [{'vertices': [[-.5, -.5, -.5], [3, -.5, 1], [-.5, 3, 1]], 'uv': [[0, 0], [1, 0], [0, 1]], 'material': 0}]
        mesh = load_native_glb(fixture(parts))
        native = render_native_mesh(mesh, K, R, T, 10, 10)
        old = render_mesh(mesh.vertices, mesh.faces, K, R, T, 10, 10)
        self.assertGreater(int(native['mask'].sum()), 0)
        np.testing.assert_array_equal(native['mask'], old['mask'])
        np.testing.assert_array_equal(native['depth'], old['depth'])
        self.assertTrue(np.isfinite(native['depth'][native['mask']]).all())

    def test_near_clipped_uv_has_independent_analytic_value(self):
        image = np.tile(np.arange(16, dtype=np.uint8)[None, :, None] * 16, (1, 1, 3))
        parts = [{'vertices': [[-.5, -.5, -.5], [3, -.5, 1], [-.5, 3, 1]],
                  'uv': [[0, 0], [1, 0], [0, 1]], 'material': 0}]
        rendered = self.render(fixture(parts, images=[image]))
        # Ray(2,2,1) intersects at(.2,.2,.1), native barycentric(.6,.2,.2),
        # UV(.2,.2), so a16-column nearest ramp selects column3, RGB48.
        self.assertTrue(rendered['mask'][2, 2])
        self.assertAlmostEqual(float(rendered['depth'][2, 2]), .1, places=7)
        np.testing.assert_array_equal(rendered['rgb'][2, 2], [48, 48, 48])

    def test_mask_only_matches_rgb_geometry_exactly(self):
        mesh = load_native_glb(fixture())
        rendered = render_native_mesh(mesh, K, R, T, 10, 10)
        mask = render_native_mask(mesh, K, R, T, 10, 10)
        self.assertIsNone(mask['rgb'])
        for key in ('mask', 'depth', 'face_index'):
            np.testing.assert_array_equal(mask[key], rendered[key])

    def test_default_culling_and_rays_agree(self):
        materials = [{'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}}}]
        mesh = load_native_glb(fixture(materials=materials))
        self.assertFalse(render_native_mesh(mesh, K, R, T, 10, 10)['mask'].any())
        points, valid = ray_native_mesh_points([[2, 2]], mesh, K, R, T)
        self.assertFalse(valid.any()); self.assertTrue(np.isnan(points).all())
        parts = [{'vertices': [[0, 0, 1], [0, 8, 1], [8, 0, 1]], 'uv': [[0, 0], [0, 1], [1, 0]], 'material': 0}]
        mesh = load_native_glb(fixture(parts, materials=materials))
        self.assertTrue(render_native_mesh(mesh, K, R, T, 10, 10)['mask'][2, 2])
        points, valid = ray_native_mesh_points([[2, 2]], mesh, K, R, T)
        self.assertTrue(valid[0]); np.testing.assert_allclose(points[0], [2, 2, 1])

    def test_negative_node_scale_corrects_native_winding(self):
        materials = [{'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}}}]
        parts = [{'vertices': [[0, 0, 1], [0, 8, 1], [8, 0, 1]], 'uv': [[0, 0], [0, 1], [1, 0]], 'material': 0}]
        mesh = load_native_glb(fixture(parts, materials=materials, nodes=[{'mesh': 0, 'scale': [-1, 1, 1], 'translation': [8, 0, 0]}]))
        self.assertEqual(mesh.receipt['primitive_culling'][0]['node_winding_sign'], -1)
        self.assertTrue(render_native_mesh(mesh, K, R, T, 10, 10)['mask'][2, 6])
        self.assertTrue(ray_native_mesh_points([[6, 2]], mesh, K, R, T)[1][0])

    def test_legacy_vertex_pixels_and_depth_unchanged(self):
        import trimesh
        parts = [{'vertices': [[0, 0, 1], [32, 0, 4], [0, 32, 4]], 'colors': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}]
        raw = fixture(parts, materials=[], images=[])
        mesh = load_native_glb(raw)
        old = trimesh.load(io.BytesIO(raw), file_type='glb', force='mesh', process=False)
        expected = render_mesh(old.vertices, old.faces, K, R, T, 10, 10, vertex_colors=np.asarray(old.visual.vertex_colors)[:, :3])
        self.assertTrue(mesh.receipt['legacy_compatible'])
        actual = render_native_mesh(mesh, K, R, T, 10, 10)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_material_factor_prevents_legacy_shortcut(self):
        parts = [{'vertices': [[0, 0, 1], [8, 0, 1], [0, 8, 1]], 'colors': [[1, 0, 0]] * 3, 'material': 0}]
        materials = [{'doubleSided': True, 'pbrMetallicRoughness': {'baseColorFactor': [.25, 1, 1, 1]}}]
        mesh = load_native_glb(fixture(parts, materials=materials, images=[]))
        self.assertFalse(mesh.receipt['legacy_compatible'])
        self.assertIsNone(mesh.vertex_colors)
        np.testing.assert_array_equal(render_native_mesh(mesh, K, R, T, 10, 10)['rgb'][2, 2], [137, 0, 0])

    def test_alignment_contract_declaration_and_receipt_stability(self):
        raw = fixture()
        first = alignment_contract(raw, 'uv-baked')
        self.assertTrue(first['eligible']); self.assertEqual(first, alignment_contract(raw, 'uv_texture'))
        self.assertFalse(alignment_contract(raw, 'vertex-color')['eligible'])
        self.assertFalse(alignment_contract(raw, 'unknown')['eligible'])
        self.assertNotIn('timestamp', json.dumps(first))

    def test_unsupported_appearance_fails_closed(self):
        changes = {
            'blend': lambda d: d['materials'][0].update(alphaMode='BLEND'),
            'mask': lambda d: d['materials'][0].update(alphaMode='MASK'),
            'factor_alpha': lambda d: d['materials'][0]['pbrMetallicRoughness'].update(baseColorFactor=[1, 1, 1, .5]),
            'texture_transform': lambda d: d['materials'][0]['pbrMetallicRoughness']['baseColorTexture'].update(extensions={'KHR_texture_transform': {'offset': [.2, 0]}}),
            'normal': lambda d: d['materials'][0].update(normalTexture={'index': 0}),
            'roughness_map': lambda d: d['materials'][0]['pbrMetallicRoughness'].update(metallicRoughnessTexture={'index': 0}),
            'emission': lambda d: d['materials'][0].update(emissiveFactor=[1, 0, 0]),
            'alternate_uv': lambda d: d['materials'][0]['pbrMetallicRoughness']['baseColorTexture'].update(texCoord=1),
            'mipmap': lambda d: d['samplers'][0].update(minFilter=9987),
            'mixed_filters': lambda d: d['samplers'][0].update(minFilter=9729),
            'bad_wrap': lambda d: d['samplers'][0].update(wrapS=0),
            'external': lambda d: d['images'][0].update(uri='https://example.invalid/private.png'),
            'extension': lambda d: d.update(extensionsUsed=['KHR_materials_transmission']),
            'index_shape': lambda d: d['accessors'][2].update(type='VEC3', count=1),
            'index_signed': lambda d: d['accessors'][2].update(componentType=5122),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                result = alignment_contract(fixture(changes=change), 'uv-baked')
                self.assertFalse(result['eligible']); self.assertTrue(result['reason'])

    def test_texture_alpha_and_vertex_alpha_rejected(self):
        rgba = np.array([[[255, 0, 0, 100]]], np.uint8)
        self.assertFalse(alignment_contract(fixture(images=[rgba]), 'uv-baked')['eligible'])
        parts = [{'vertices': [[0, 0, 1], [8, 0, 1], [0, 8, 1]], 'colors': [[1, 1, 1, .5]] * 3}]
        self.assertFalse(alignment_contract(fixture(parts, materials=[], images=[]), 'vertex-color')['eligible'])

    def test_decompressed_image_budget(self):
        image = np.zeros((4097, 4097, 3), np.uint8)
        result = alignment_contract(fixture(images=[image]), 'uv-baked')
        self.assertFalse(result['eligible']); self.assertIn('pixel budget', result['reason'])

    def test_missing_uv_and_malformed_file_fail_closed(self):
        parts = [{'vertices': [[0, 0, 1], [8, 0, 1], [0, 8, 1]], 'material': 0}]
        self.assertFalse(alignment_contract(fixture(parts), 'uv-baked')['eligible'])
        self.assertFalse(alignment_contract(b'not a GLB', 'uv-baked')['eligible'])

    def test_invalid_projection_contract_and_render_budget(self):
        mesh = load_native_glb(fixture())
        for kwargs in ({'scale': -1}, {'scale': 0}):
            with self.assertRaises(ValueError):
                render_native_mesh(mesh, K, R, T, 10, 10, **kwargs)
        with self.assertRaises(ValueError):
            render_native_mesh(mesh, K, R, T, 5000, 5000)
        with self.assertRaises(ValueError):
            render_native_mesh(mesh, K, R * 2, T, 10, 10)


if __name__ == '__main__':
    unittest.main()
