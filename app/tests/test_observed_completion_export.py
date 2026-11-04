"""Check exported observed/inferred layers with independent binary reads."""
import hashlib
import io
import json
from pathlib import Path
import struct
import unittest

import numpy as np
from PIL import Image

import observed_completion_export as completion
from observed_surface_export import make_glb


def parse(raw):
    magic, version, length = struct.unpack_from('<III', raw)
    if magic != 0x46546c67 or version != 2 or length != len(raw):
        raise ValueError('Bad GLB')
    n, kind = struct.unpack_from('<II', raw, 12)
    if kind != 0x4e4f534a:
        raise ValueError('Missing JSON')
    doc = json.loads(raw[20:20 + n])
    size, kind = struct.unpack_from('<II', raw, 20 + n)
    if kind != 0x004e4942:
        raise ValueError('Missing BIN')
    return doc, raw[28 + n:28 + n + size]


def attributes(document, binary, mesh_index):
    primitive = document['meshes'][mesh_index]['primitives'][0]

    def read(index):
        item = document['accessors'][index]
        view = document['bufferViews'][item['bufferView']]
        columns = {'VEC3': 3, 'VEC2': 2, 'SCALAR': 1}[item['type']]
        dtype = {5126: '<f4', 5123: '<u2', 5125: '<u4'}[item['componentType']]
        return np.frombuffer(binary, dtype=dtype, count=columns * item['count'],
            offset=view.get('byteOffset', 0) + item.get('byteOffset', 0)).reshape(item['count'], columns)

    position = read(primitive['attributes']['POSITION'])
    uv = read(primitive['attributes']['TEXCOORD_0'])
    triangles = read(primitive['indices']).reshape(-1, 3)
    material = document['materials'][primitive['material']]
    texture = document['textures'][material['pbrMetallicRoughness']['baseColorTexture']['index']]
    view = document['bufferViews'][document['images'][texture['source']]['bufferView']]
    image = binary[view.get('byteOffset', 0):view.get('byteOffset', 0) + view['byteLength']]
    return position, uv, triangles, image


class CompletionExportTests(unittest.TestCase):
    def setUp(self):
        buffer = io.BytesIO()
        Image.new('RGB', (8, 8), (180, 70, 20)).save(buffer, 'JPEG')
        self.texture = buffer.getvalue()
        self.vertices = np.array([[0., 0., 4.], [0., 1., 4.], [1., 0., 4.]])
        self.faces = [[0, 1, 2]]
        self.uv = [[.1, .9], [.1, .1], [.9, .9]]

    def piece(self, offset):
        return make_glb(self.vertices + offset, self.faces, self.uv, self.texture,
            name='Synthetic contract fixture only', extras={})[0]

    def provenances(self):
        return [dict(layer_name='Observed surface', provider='fixture-observed',
                     shape_source='fixture-observation', depth_observed=True, physical_confidence='unknown'),
                dict(layer_name='Inferred depth prior', provider='fixture-model',
                     shape_source='fixture-inferred', depth_observed=False, physical_confidence='unknown')]

    def test_two_separate_nodes_materials_accessors_and_geometry_without_transforms(self):
        raw = completion.combine_layers(self.piece(0), self.piece(.25), self.provenances())
        doc, binary = parse(raw)
        self.assertEqual(doc['scenes'][0]['nodes'], [0, 1])
        self.assertEqual([n['name'] for n in doc['nodes']], ['Observed surface', 'Inferred depth prior'])
        self.assertEqual(len(doc['meshes']), 2)
        self.assertEqual(len(doc['materials']), 2)
        for index in [0, 1]:
            position, uv, triangles, texture = attributes(doc, binary, index)
            np.testing.assert_array_equal(position, (self.vertices + index * .25).astype(np.float32))
            np.testing.assert_array_equal(triangles, self.faces)
            self.assertEqual(texture, self.texture)
            self.assertEqual(set(doc['nodes'][index]), {'name', 'mesh', 'extras'})
            p = doc['meshes'][index]['primitives'][0]
            for value in [doc['nodes'][index], doc['meshes'][index], p, doc['materials'][index]]:
                self.assertEqual(value['extras'], self.provenances()[index])
            self.assertFalse(doc['materials'][index]['doubleSided'])
            self.assertEqual(set(p['attributes']), {'POSITION', 'TEXCOORD_0'})

    def test_corrupt_glb_and_hidden_transform_are_rejected(self):
        with self.assertRaises(ValueError):
            completion.combine_layers(b'corrupt', self.piece(0), self.provenances())
        doc, binary = parse(self.piece(0))
        doc['nodes'][0]['scale'] = [2, 2, 2]
        encoded = json.dumps(doc).encode(); encoded += b' ' * (-len(encoded) % 4)
        raw = struct.pack('<III', 0x46546c67, 2, 28 + len(encoded) + len(binary))
        raw += struct.pack('<II', len(encoded), 0x4e4f534a) + encoded
        raw += struct.pack('<II', len(binary), 0x004e4942) + binary
        with self.assertRaises(ValueError):
            completion.combine_layers(raw, self.piece(0), self.provenances())

    def test_inferred_only_provenance_is_on_node_mesh_primitive_material_without_binary_changes(self):
        original = self.piece(.25)
        raw = completion.annotate_single_layer(original, self.provenances()[1])
        before, before_binary = parse(original)
        doc, binary = parse(raw)
        for a, b in zip(attributes(doc, binary, 0)[:3], attributes(before, before_binary, 0)[:3]):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(attributes(doc, binary, 0)[3], self.texture)
        for value in [doc['nodes'][0], doc['meshes'][0], doc['meshes'][0]['primitives'][0], doc['materials'][0]]:
            self.assertEqual(value['extras'], self.provenances()[1])

    def test_actual_exports_preserve_both_geometry_sources_original_texture_and_face_provenance(self):
        root = Path(__file__).resolve().parents[1]
        output = root / 'data/observed-surface-exports-20261031'
        manifest_path = output / 'completion-exports.json'
        if not manifest_path.exists():
            self.skipTest('Saved completion exports have not been built')
        observed = json.loads((root / 'data/observed-surface-experiment-20261031/camera-body-stereo-v1/viewer-display.json').read_text())
        original_raw = (output / 'camera-body-stereo-v1/observed-patch.glb').read_bytes()
        original_doc, original_binary = parse(original_raw)
        original_arrays = attributes(original_doc, original_binary, 0)
        for entry in json.loads(manifest_path.read_text())['exports']:
            with self.subTest(id=entry['id']):
                metadata = json.loads((output / entry['id'] / 'metadata.json').read_text())
                raw = (output / entry['id'] / 'observed-and-inferred.glb').read_bytes()
                doc, binary = parse(raw)
                inferred = json.loads((root / metadata['source_viewer_path']).read_text())['inferred_surface']
                obs = attributes(doc, binary, 0)
                inf = attributes(doc, binary, 1)
                for decoded, original in zip(obs[:3], original_arrays[:3]):
                    np.testing.assert_array_equal(decoded, original)
                np.testing.assert_array_equal(inf[0], np.asarray(inferred['vertices'], np.float32))
                np.testing.assert_array_equal(inf[2], inferred['triangles'])
                expected_uv = np.array(inferred['uv']); expected_uv[:, 1] = 1 - expected_uv[:, 1]
                np.testing.assert_array_equal(inf[1], expected_uv.astype(np.float32))
                self.assertEqual(obs[3], inf[3])
                self.assertEqual(obs[3], original_arrays[3])
                self.assertEqual(metadata['observed_glb_sha256'], hashlib.sha256(original_raw).hexdigest())
                self.assertEqual(metadata['glb_sha256'], hashlib.sha256(raw).hexdigest())
                self.assertEqual(metadata['observed_triangles'], 5377)
                self.assertEqual(metadata['inferred_triangles'], len(inferred['triangles']))
                self.assertFalse(metadata['physical_scale_known'])
                self.assertFalse(metadata['physics_ready'])
                self.assertFalse(metadata['observed_geometry_changed'])
                self.assertEqual(len(doc['nodes']), 2)
                inferred_extras = doc['nodes'][1]['extras']
                self.assertFalse(inferred_extras['depth_observed'])
                self.assertEqual(inferred_extras['every_face_confidence'], 'unknown')
                self.assertEqual(inferred_extras['triangle_count'], len(inferred['triangles']))
                self.assertTrue(inferred_extras['all_faces_inferred'])
                self.assertEqual(inferred_extras['provider'], entry['inferred_provider'])
                for mesh_index in [0, 1]:
                    primitive = doc['meshes'][mesh_index]['primitives'][0]
                    for obj in [doc['nodes'][mesh_index], doc['meshes'][mesh_index], primitive, doc['materials'][primitive['material']]]:
                        self.assertEqual(obj['extras']['physical_confidence'], 'unknown')
                        self.assertFalse(obj['extras']['physical_scale_known'])

    def test_workflow_inferred_only_download_preserves_exact_inferred_source_and_observed_export(self):
        root = Path(__file__).resolve().parents[1]
        records = list((root / 'data/observed-app-runs-20261031').glob('*-workflow/inferred-version.json'))
        if not records:
            self.skipTest('No real completed inferred workflow has been run')
        for record in records:
            version = json.loads(record.read_text())
            with self.subTest(version=version['version_id']):
                asset = version['artifacts']['glb']
                raw = (root / asset['path']).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), asset['sha256'])
                doc, binary = parse(raw)
                self.assertEqual(len(doc['nodes']), 1)
                self.assertEqual(len(doc['meshes']), 1)
                self.assertEqual(doc['nodes'][0]['name'], version['inferred_node_name'])
                # The first frozen workflow exports carry this at the asset
                # level; newer exports additionally pin each primitive/node.
                provenance = doc['nodes'][0].get('extras', doc['asset']['extras'])
                self.assertTrue(provenance['all_faces_inferred'])
                self.assertFalse(provenance['depth_observed'])
                self.assertEqual(provenance['every_face_confidence'], 'unknown')
                combo = version['artifacts']['combined_preview']
                combined = (root / combo['path']).read_bytes()
                self.assertEqual(hashlib.sha256(combined).hexdigest(), combo['sha256'])
                both, both_binary = parse(combined)
                self.assertEqual([n['name'] for n in both['nodes']], ['Observed surface', version['inferred_node_name']])
                actual = attributes(doc, binary, 0)
                expected = attributes(both, both_binary, 1)
                for a, b in zip(actual[:3], expected[:3]):
                    np.testing.assert_array_equal(a, b)
                self.assertEqual(actual[3], expected[3])
                observed_version = json.loads((record.parent / 'reconstruction-version.json').read_text())
                observed = (root / observed_version['artifacts']['glb']['path']).read_bytes()
                original, original_bin = parse(observed)
                for a, b in zip(attributes(both, both_binary, 0)[:3], attributes(original, original_bin, 0)[:3]):
                    np.testing.assert_array_equal(a, b)
                self.assertFalse(version['contains_observed_layer'])
                self.assertFalse(version['physical_accuracy_verified'])
                self.assertFalse(version['simulation_ready'])


if __name__ == '__main__':
    unittest.main()
