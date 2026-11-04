"""Presentation dependency and bounded provenance policy; no snapshot build."""
from pathlib import Path
import copy
import hashlib
import json
import struct
import tempfile
import unittest

from scripts import build_judge_preview as B


class JudgePreviewBuilderTests(unittest.TestCase):
    def test_omissions_are_exact_declared_implementation_types_with_bounded_size(self):
        base = {'path': 'data/object-programs/auto-' + 'a' * 64 + '/provenance/object_program.py.txt', 'bytes': 2000}
        self.assertTrue(B.omitted_provenance(base))
        self.assertFalse(B.omitted_provenance({'path': 'data/model/metadata.json', 'bytes': 20}))
        for row in [{**base, 'path': 'data/other/provenance/object_program.py.txt'},
                    {**base, 'path': base['path'].replace('object_program.py.txt', 'arbitrary.py.txt')},
                    {**base, 'bytes': 512 * 1024 + 1}]:
            with self.subTest(row=row), self.assertRaises(ValueError): B.omitted_provenance(row)
        ancestor=B.INCLUDED_ANCESTOR_SOURCE_TEXT[0]
        self.assertFalse(B.omitted_provenance(ancestor))
        with self.assertRaises(ValueError):B.omitted_provenance({**ancestor,'sha256':'0'*64})

    def fixture(self, root):
        for name in B.STATIC:
            path = root / 'showcase' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('<html></html>' if name.endswith('.html') else 'test')
        for name in B.DEMO_FILES:
            path = root / 'showcase/demos/charge-phone-v1' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('<html></html>' if name.endswith('.html') else 'test')

    def test_html_links_choose_current_film_captions_brand_and_static_demo(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); self.fixture(root)
            (root / 'showcase/index.html').write_text('<link href="brand/v2/mark.svg"><video poster="media/current.jpg"><source src="media/current45.mp4"><track src="media/current45.vtt"></video><a href="https://creator.invalid/">Creator</a><a href="/api/library">API</a>')
            files = B.static_dependencies(root)
            self.assertIn('showcase/media/current45.mp4', files)
            self.assertIn('showcase/media/current45.vtt', files)
            self.assertIn('showcase/brand/v2/mark.svg', files)
            self.assertIn('showcase/demos/charge-phone-v1/muse-review.json', files)
            self.assertFalse(any('cobalt-20261103-v2-web' in f for f in files))

    def test_owner_or_hidden_html_links_fail_before_copy(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); self.fixture(root)
            for link in ['../admin/index.html', '../.secrets/password.txt', '../workspace/index.html']:
                (root / 'showcase/index.html').write_text('<a href="' + link + '">not a preview resource</a>')
                with self.subTest(link=link), self.assertRaises(ValueError): B.static_dependencies(root)

    def test_physics_modules_and_only_exact_reviewed_starter_zip_are_included(self):
        with tempfile.TemporaryDirectory()as name:
            root=Path(name).resolve();self.fixture(root)
            (root/'showcase/physics-bench.html').write_text('<a href="media/interaction-twin-newton-starter-20261104-v1.zip">Newton starter</a>')
            paths=B.static_dependencies(root)
            for path in ['showcase/nonrigid-observations-contract.js','showcase/physics-bench.js','showcase/physics-bench-contract.js','showcase/physics-bench-source.js',B.PHYSICS_STARTER]:self.assertIn(path,paths)
            (root/'showcase/physics-bench.html').write_text('<a href="media/arbitrary.zip">Unreviewed archive</a>')
            with self.assertRaisesRegex(ValueError,'Unknown local presentation asset type'):B.static_dependencies(root)

    def appearance_fixture(self,root,*,change_previous=None,appearance_format='interaction-twin-source-appearance-states'):
        approved={}
        def write(name,value,*,binary=False):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
            raw=value if binary else json.dumps(value,sort_keys=True).encode()
            path.write_bytes(raw)
            reference={'path':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
            approved[name]=reference;return reference
        program=write('data/fixture/program.json',{'format':'synthetic_program_not_real_capture'})
        scene=write('data/fixture/scene.json',{'format':'synthetic_scene_not_real_capture'})
        report=write('data/fixture/refinement.json',{})
        audit=write('data/fixture/source-audit.json',{})
        metadata=write('data/fixture/native-metadata.json',{})
        asset=write('data/fixture/native.glb',b'\x00\xffsynthetic_binary_native_asset_not_real_GLTF',binary=True)
        original={'format':'interaction-twin-declared-geometry-derivative','version':1,
            'base_object_program_ref':program,'evidence_program':program,'scene_ref':scene,
            'baseline_scene_ref':scene,'refinement_report_ref':report,'source_alignment_audit_ref':audit,
            'component_assets':[{'component_id':'lid','native_glb':asset,'original_native_glb':asset,'native_metadata':metadata}]}
        previous=copy.deepcopy(original)
        if change_previous:change_previous(previous)
        previous_ref=write('data/fixture/previous.json',previous)
        probe=write('data/fixture/probes.u32',struct.pack('<II',1,2),binary=True)
        keys=[];binary_refs=[probe];source_refs=[]
        for index,label in [(0,'dark'),(2,'boot')]:
            # Deliberately non-JSON bytes prove the builder keeps typed binary
            # patches/probes opaque instead of decoding arbitrary content.
            patch=write(f'data/fixture/{label}.itcp',struct.pack('<4sIII',b'ITCP',1,3,1)+struct.pack('<IHHBBB',1,2,3,10,20,30),binary=True)
            rgb=write(f'data/fixture/rgb-{index}.jpg',b'\xff\xd8\xfffixture_RGB',binary=True)
            mask=write(f'data/fixture/mask-{index}.png',b'\x89PNG\x00fixture_mask',binary=True)
            keys.append({'id':label,'source_rgb_ref':rgb,'source_mask_ref':mask,'patch_ref':patch})
            binary_refs.append(patch);source_refs.extend([rgb,mask])
        unknown_rgb=write('data/fixture/unknown-rgb.jpg',b'\xff\xd8\xffunknown_RGB',binary=True)
        unknown_mask=write('data/fixture/unknown-mask.png',b'\x89PNG\x00unknown_mask',binary=True)
        appearance={'format':appearance_format,'version':1,'previous_derived_assembly_ref':previous_ref,
            'scene_ref':scene,'evidence_program':program,'baseline_asset_ref':asset,
            'probe_native_vertex_ids_ref':probe,'keyframes':keys,
            'observations':[{'status':'unknown','source_rgb_ref':unknown_rgb,'source_mask_ref':unknown_mask}]}
        appearance_ref=write('data/fixture/appearance.json',appearance)
        derivative=write('data/fixture/derived.json',{**original,'source_appearance_states_ref':appearance_ref})
        index=root/'showcase/public/aligned-scenes.json';index.parent.mkdir(parents=True,exist_ok=True)
        index.write_text(json.dumps({'scenes':[{'manifest':scene,'object_program':program,'derived_assembly':derivative}]}))
        return {'approved':approved,'derivative':derivative,'appearance':appearance_ref,
            'previous':previous_ref,'binary_refs':binary_refs,
            'source_refs':source_refs+[unknown_rgb,unknown_mask],'asset':asset}

    def test_both_new_appearance_runtime_helpers_are_explicit_static_dependencies(self):
        with tempfile.TemporaryDirectory()as name:
            root=Path(name).resolve();self.fixture(root)
            files=B.static_dependencies(root)
            self.assertIn('showcase/native-byte-rgb-display-v1.js',files)
            self.assertIn('showcase/source-appearance-states-v1.js',files)

    def test_typed_appearance_lineage_admits_exact_binary_patches_and_probe_without_JSON_decode(self):
        with tempfile.TemporaryDirectory()as name:
            root=Path(name).resolve();f=self.appearance_fixture(root)
            proof=B.verify_display_lineage(root,f['approved'])
            self.assertEqual(len(proof),1)
            self.assertEqual(proof[0]['document_ref'],f['derivative'])
            self.assertTrue(proof[0]['source_appearance_included'])
            for reference in f['binary_refs']:
                with self.assertRaises((ValueError,UnicodeError)):
                    json.loads((root/reference['path']).read_bytes())

    def test_missing_appearance_patch_probe_previous_or_source_refs_are_never_silently_omitted(self):
        with tempfile.TemporaryDirectory()as name:
            root=Path(name).resolve();f=self.appearance_fixture(root)
            for missing in [f['appearance'],f['previous'],*f['binary_refs'],*f['source_refs']]:
                with self.subTest(path=missing['path']):
                    approved={k:v for k,v in f['approved'].items()if k!=missing['path']}
                    with self.assertRaisesRegex(ValueError,'missing an exact approved public reference'):
                        B.verify_display_lineage(root,approved)

    def test_changed_binary_patch_and_probe_bytes_reject_despite_unchanged_declared_inventory(self):
        with tempfile.TemporaryDirectory()as name:
            root=Path(name).resolve();f=self.appearance_fixture(root)
            for reference in f['binary_refs']:
                path=root/reference['path'];original=path.read_bytes()
                with self.subTest(path=reference['path']):
                    path.write_bytes(original+b'\x00')
                    try:
                        with self.assertRaisesRegex(ValueError,'Pinned source bytes changed'):
                            B.verify_display_lineage(root,f['approved'])
                    finally:path.write_bytes(original)

    def test_source_appearance_requires_exact_prior_descriptor_and_known_sidecar_format(self):
        for mutation in [lambda d:d.update(extra_unreviewed_field=True),
                         lambda d:d['component_assets'][0].update(component_id='unrelated_part')]:
            with tempfile.TemporaryDirectory()as name:
                root=Path(name).resolve();f=self.appearance_fixture(root,change_previous=mutation)
                with self.assertRaisesRegex(ValueError,'changed its prior descriptor'):
                    B.verify_display_lineage(root,f['approved'])
        with tempfile.TemporaryDirectory()as name:
            root=Path(name).resolve();f=self.appearance_fixture(root,appearance_format='unreviewed_appearance_format')
            with self.assertRaisesRegex(ValueError,'Unknown typed source appearance sidecar'):
                B.verify_display_lineage(root,f['approved'])


if __name__ == '__main__': unittest.main()
