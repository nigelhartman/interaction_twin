"""Independent binary/appearance checks for portable observed-patch exports."""
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

import observed_surface_export as export


def unpack_glb(raw):
    """A test-only glTF accessor reader; never calls the exporter's helpers."""
    magic, version, total = struct.unpack_from('<III', raw)
    if (magic, version, total) != (0x46546c67, 2, len(raw)):
        raise ValueError('Bad GLB header')
    chunks, offset = {}, 12
    while offset < len(raw):
        count, kind = struct.unpack_from('<II', raw, offset)
        if count % 4 or offset + 8 + count > len(raw) or kind in chunks:
            raise ValueError('Bad GLB chunk')
        chunks[kind] = raw[offset + 8:offset + 8 + count]
        offset += count + 8
    doc = json.loads(chunks[0x4e4f534a])
    binary = chunks[0x004e4942]

    def accessor(index):
        item = doc['accessors'][index]
        view = doc['bufferViews'][item['bufferView']]
        columns = {'VEC3': 3, 'VEC2': 2, 'SCALAR': 1}[item['type']]
        datatype = {5126: '<f4', 5123: '<u2', 5125: '<u4'}[item['componentType']]
        count = item['count'] * columns
        start = view.get('byteOffset', 0) + item.get('byteOffset', 0)
        result = np.frombuffer(binary, dtype=datatype, count=count, offset=start)
        if result.nbytes > view['byteLength']:
            raise ValueError('Bad accessor length')
        return result.reshape(item['count'], columns)

    primitive = doc['meshes'][0]['primitives'][0]
    positions = accessor(primitive['attributes']['POSITION'])
    uv = accessor(primitive['attributes']['TEXCOORD_0'])
    indices = accessor(primitive['indices']).reshape(-1, 3)
    view = doc['bufferViews'][doc['images'][0]['bufferView']]
    texture = binary[view.get('byteOffset', 0):view.get('byteOffset', 0) + view['byteLength']]
    return doc, positions, uv, indices, texture


class SurfaceExportTests(unittest.TestCase):
    def setUp(self):
        pixels = np.zeros((8, 8, 3), np.uint8)
        pixels[:4, :4] = [240, 10, 15]
        pixels[:4, 4:] = [10, 230, 20]
        pixels[4:, :4] = [15, 20, 225]
        pixels[4:, 4:] = [220, 210, 30]
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, 'JPEG', quality=100, subsampling=0)
        self.jpeg = buffer.getvalue()
        self.vertices = np.array([[0, 0, 4], [0, 1, 4], [1, 0, 4]], float)
        self.triangles = np.array([[0, 1, 2]])
        self.uv = np.array([[.1875, .8125], [.1875, .1875], [.8125, .8125]])

    def glb(self, **changes):
        args = dict(vertices=self.vertices, triangles=self.triangles,
                    display_uv=self.uv, texture=self.jpeg, name='synthetic-test-only',
                    extras={'physical_scale_known': False})
        args.update(changes)
        return export.make_glb(**args)

    def test_actual_accessors_keep_positions_topology_bounds_and_one_sidedness(self):
        raw, conversion = self.glb()
        doc, positions, uv, indices, texture = unpack_glb(raw)
        np.testing.assert_array_equal(positions, self.vertices)
        np.testing.assert_array_equal(indices, self.triangles)
        np.testing.assert_array_equal(doc['accessors'][0]['min'], self.vertices.min(axis=0))
        np.testing.assert_array_equal(doc['accessors'][0]['max'], self.vertices.max(axis=0))
        self.assertFalse(doc['materials'][0]['doubleSided'])
        self.assertEqual(set(doc['meshes'][0]['primitives'][0]['attributes']), {'POSITION', 'TEXCOORD_0'})
        self.assertEqual(doc['nodes'][0], {'name': 'synthetic-test-only', 'mesh': 0})
        self.assertEqual(texture, self.jpeg)
        normal = np.cross(positions[1] - positions[0], positions[2] - positions[0])
        self.assertGreater(np.dot(normal, -positions.mean(axis=0)), 0)
        self.assertEqual(conversion['position_max_float32_error'], 0)

    def test_top_left_uv_selects_correct_asymmetric_source_quadrants(self):
        raw, _ = self.glb()
        doc, positions, uv, indices, texture = unpack_glb(raw)
        image = np.asarray(Image.open(io.BytesIO(texture)))
        # glTF top-left UV convention; select the actual decoded original pixels.
        pixels = image[np.floor(uv[:, 1] * 8).astype(int), np.floor(uv[:, 0] * 8).astype(int)]
        self.assertEqual(np.argmax(pixels, axis=1).tolist(), [0, 2, 1])
        self.assertEqual(doc['materials'][0]['pbrMetallicRoughness']['baseColorFactor'], [1, 1, 1, 1])
        self.assertEqual(doc['extensionsRequired'], ['KHR_materials_unlit'])
        self.assertEqual(doc['materials'][0]['extensions'], {'KHR_materials_unlit': {}})
        self.assertEqual(doc['samplers'][0], {'magFilter': 9729, 'minFilter': 9729, 'wrapS': 33071, 'wrapT': 33071})

    def test_float32_error_is_measured_and_does_not_move_geometry_deliberately(self):
        vertices = self.vertices + .123456789
        raw, conversion = self.glb(vertices=vertices)
        _, decoded, _, _, _ = unpack_glb(raw)
        self.assertAlmostEqual(conversion['position_max_float32_error'], np.max(np.abs(decoded - vertices)))
        self.assertLess(conversion['position_max_float32_error'], 1e-6)
        self.assertEqual(conversion['position_transform'], 'none')

    def test_bad_indices_nonfinite_geometry_uv_or_image_rejected(self):
        for changes in [dict(triangles=[[0, 1, 3]]), dict(triangles=[[-1, 1, 2]]),
                        dict(vertices=[[0, 0, float('nan')]] * 3), dict(display_uv=[[2., 0.]] * 3),
                        dict(texture=b'not an image')]:
            with self.subTest(changes=changes):
                with self.assertRaises((ValueError, OSError)):
                    self.glb(**changes)

    def test_path_traversal_external_hidden_encoded_and_symlink_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ['../oops', 'data/../oops', '/tmp/oops', 'data/%2e%2e/x', 'data/a?x=1',
                         'data/a#x', 'data/a\\x', 'data/.secrets/key', 'data/a\nb']:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    export.local_path(root, name)
            (root / 'data').mkdir()
            (root / 'data' / 'link').symlink_to(Path('/tmp'))
            with self.assertRaises(ValueError):
                export.local_path(root, 'data/link/x')

    def test_texture_hash_corruption_is_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'texture.jpg'
            path.write_bytes(self.jpeg)
            expected = hashlib.sha256(self.jpeg).hexdigest()
            self.assertEqual(export.read_checked(path, expected), self.jpeg)
            path.write_bytes(self.jpeg + b'corruption')
            with self.assertRaises(ValueError):
                export.read_checked(path, expected)

    def package(self, mutation=None, extra_name=None):
        glb, _ = self.glb()
        files = {'observed-patch.glb': glb, 'metadata.json': b'{}', 'README.txt': b'Example',
                 'export-implementation.py': b'# No automatic execution\n'}
        manifest = {'format': 'observed-surface-portable', 'version': 1,
                    'files': {name: {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
                              for name, raw in files.items()}}
        if mutation:
            mutation(files)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            for name, raw in files.items():
                archive.writestr(name, raw)
            archive.writestr('package-manifest.json', json.dumps(manifest))
            if extra_name:
                archive.writestr(extra_name, b'unauthorized')
        return buffer.getvalue()

    def test_zip_corrupt_member_undeclared_member_and_traversal_fail_closed(self):
        self.assertTrue(export.validate_package(self.package())['all_member_hashes_verified'])
        for raw in [self.package(lambda files: files.update({'README.txt': b'Changed'})),
                    self.package(extra_name='unexpected.txt'), self.package(extra_name='../escape')]:
            with self.assertRaises(ValueError):
                export.validate_package(raw)

    def test_original_saved_exports_binary_geometry_texture_and_integrity(self):
        root = Path(__file__).resolve().parents[1]
        exports = root / 'data/observed-surface-exports-20261031'
        if not exports.exists():
            self.skipTest('Actual saved exports have not been built')
        for name in ['camera-body-stereo-v1', 'camera-body-stereo-refined-k-v2']:
            with self.subTest(name=name):
                source = json.loads((root / 'data/observed-surface-experiment-20261031' / name / 'viewer-display.json').read_text())
                surface = source['surface']
                raw = (exports / name / 'observed-patch.glb').read_bytes()
                doc, positions, uv, indices, texture = unpack_glb(raw)
                np.testing.assert_array_equal(positions, np.asarray(surface['vertices'], dtype=np.float32))
                np.testing.assert_array_equal(indices, surface['triangles'])
                expected_uv = np.array(surface['uv']); expected_uv[:, 1] = 1 - expected_uv[:, 1]
                np.testing.assert_array_equal(uv, expected_uv.astype(np.float32))
                texture_original = (root / surface['texture_url'][7:]).read_bytes()
                self.assertEqual(texture, texture_original)
                metadata = json.loads((exports / name / 'metadata.json').read_text())
                self.assertEqual(metadata['glb_sha256'], hashlib.sha256(raw).hexdigest())
                self.assertFalse(metadata['physical_scale_known'])
                self.assertFalse(metadata['complete_object'])
                self.assertFalse(metadata['physics_ready'])
                self.assertEqual(metadata['confidence']['shape_accuracy'], 'unmeasured')
                self.assertTrue(export.validate_package((exports / name / 'portable.zip').read_bytes())['all_member_hashes_verified'])


if __name__ == '__main__':
    unittest.main()
