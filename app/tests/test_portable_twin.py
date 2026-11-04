"""Portable bytes preserve provenance, native appearance and exact sample gaps."""
import base64
import copy
import importlib.util
import json
from pathlib import Path
import stat
import struct
import subprocess
import tempfile
import unittest
import zipfile

import portable_twin as P

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('portable_fixture', ROOT/'tests/test_stage5.py')
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')


def glb(*, texture=False, external=False):
    binary = struct.pack('<9f3H', 0.,0.,0., 1.,0.,0., 0.,1.,0., 0,1,2)
    binary += b'\0' * (-len(binary) % 4)
    uv_offset = len(binary)
    binary += struct.pack('<6f', 0,0,1,0,0,1)
    color_offset = len(binary)
    binary += struct.pack('<12f', 1,0,0,1,0,1,0,1,0,0,1,1)
    image_offset = len(binary)
    binary += PNG
    length = len(binary)
    binary += b'\0' * (-len(binary) % 4)
    doc = {'asset': {'version':'2.0'}, 'scene':0, 'scenes':[{'nodes':[0]}],
           'nodes':[{'mesh':0,'translation':[2,3,4]}], 'buffers':[{'byteLength':length}],
           'bufferViews':[{'buffer':0,'byteLength':36}, {'buffer':0,'byteOffset':36,'byteLength':6},
                          {'buffer':0,'byteOffset':uv_offset,'byteLength':24}, {'buffer':0,'byteOffset':color_offset,'byteLength':48},
                          {'buffer':0,'byteOffset':image_offset,'byteLength':len(PNG)}],
           'accessors':[{'bufferView':0,'componentType':5126,'count':3,'type':'VEC3'},
                        {'bufferView':1,'componentType':5123,'count':3,'type':'SCALAR'},
                        {'bufferView':2,'componentType':5126,'count':3,'type':'VEC2'},
                        {'bufferView':3,'componentType':5126,'count':3,'type':'VEC4'}],
           'materials':[{'pbrMetallicRoughness':{'baseColorFactor':[.5,.6,.7,1]}}],
           'meshes':[{'primitives':[{'attributes':{'POSITION':0,'COLOR_0':3},'indices':1,'material':0}]}]}
    if texture:
        doc['meshes'][0]['primitives'][0]['attributes']['TEXCOORD_0'] = 2
        doc['materials'][0]['pbrMetallicRoughness']['baseColorTexture'] = {'index':0}
        doc['textures'] = [{'source':0}]
        doc['images'] = [{'uri':'https://forbidden.test/image.png'}] if external else [{'bufferView':4,'mimeType':'image/png'}]
    return pack_glb(doc, binary)


def pack_glb(doc, binary):
    raw = json.dumps(doc,separators=(',',':')).encode()
    raw += b' ' * (-len(raw) % 4)
    return struct.pack('<III',0x46546C67,2,28+len(raw)+len(binary))+struct.pack('<II',len(raw),0x4E4F534A)+raw+struct.pack('<II',len(binary),0x004E4942)+binary


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mesh = glb(texture=True)
        self.doc = fixture_module.fixture()
        self.doc['assets'] = [{'id':'generated_part','track_id':'part','path':'data/source/part.glb',
                              'sha256':P.digest(self.mesh),'backend':'Synthetic test fixture',
                              'geometry_status':'unreviewed_model_candidate','physical_scale_known':False}]
        for e in self.doc['evidence']:
            e['frame_sha256'] = P.digest(PNG)
        self.doc['evidence'][0]['observations'] = [{'track_id':'part','mask_path':'data/source/mask.png',
            'mask_sha256':P.digest(PNG),'gate_status':'supported','audit_status':'not_audited'}]
        self.original = (json.dumps(self.doc, indent=3)+'\n').encode()
        self.args = dict(document=self.doc,program_bytes=self.original,asset_bytes={'part':self.mesh},
                         evidence_bytes={e['id']:PNG for e in self.doc['evidence']},
                         mask_bytes={('source_000001','part'):PNG})

    def build(self, name='bundle', **changes):
        args = {**self.args,**changes}
        return P.build_package(self.root/name,**args)

    def rewrite(self, directory, name, data):
        (directory/name).write_bytes(data)
        manifest = json.loads((directory/'manifest.json').read_bytes())
        row = next(r for r in manifest['files'] if r['path']==name)
        row.update(sha256=P.digest(data),bytes=len(data))
        (directory/'manifest.json').write_bytes(P.encoded(manifest))

    def test_round_trip_preserves_original_bytes_masks_mesh_and_policy(self):
        result = self.build()
        out = Path(result['directory'])
        self.assertEqual((out/'program/original-program.json').read_bytes(),self.original)
        self.assertEqual((out/'assets/part.glb').read_bytes(),self.mesh)
        self.assertEqual(result['counts'],{'assets':1,'included_evidence':3,'included_masks':1,'steps':1})
        for source in [out,Path(result['zip'])]:
            checked = P.inspect_package(source)
            self.assertFalse(checked['simulation_ready'])
            self.assertFalse(checked['video_included'])
        moved = self.root/'relocated'
        P.inspect_package(result['zip'],extract_to=moved)
        self.assertTrue((moved/'index.html').is_file())
        self.assertEqual(P.inspect_package(moved)['manifest_sha256'],result['manifest_sha256'])

    def test_glb_appearance_and_scene_transforms_preserved(self):
        mesh = P.decode_glb_bytes(self.mesh)
        p = mesh['primitives'][0]
        self.assertEqual(p['vertices'][0],[2,3,4])
        self.assertEqual(p['colors'][0],[1,0,0,1])
        self.assertEqual(p['uv'],[[0,0],[1,0],[0,1]])
        self.assertTrue(p['texture'].startswith('data:image/png;base64,'))
        self.assertEqual(p['base_color_factor'],[.5,.6,.7,1])

    def test_mcp_serialization_keeps_original_citation_distinct(self):
        raw = P.encoded(self.doc)
        result = self.build(program_bytes=raw,program_bytes_kind='mcp_json_serialization',source_program_sha256='1'*64)
        manifest = json.loads(Path(result['manifest']).read_bytes())
        self.assertFalse(manifest['program']['original_bytes_included'])
        self.assertEqual(manifest['program']['original_program_sha256'],'1'*64)
        self.assertEqual(manifest['program']['sha256'],P.digest(raw))
        P.inspect_package(result['zip'])

    def test_missing_optional_evidence_is_explicit(self):
        result = self.build(evidence_bytes={},mask_bytes={})
        manifest = json.loads(Path(result['manifest']).read_bytes())
        self.assertTrue(all(not e['included'] and e['path'] is None for e in manifest['evidence']))
        P.inspect_package(result['zip'])

    def test_output_collision_never_overwrites(self):
        result = self.build()
        with self.assertRaises(ValueError):self.build()
        self.assertEqual(P.inspect_package(result['directory'])['assets'],1)

    def test_bad_asset_program_or_evidence_hashes_fail_closed(self):
        for change in [dict(asset_bytes={'part':self.mesh+b'bad'}),dict(program_bytes=b'{}'),
                       dict(source_program_sha256='1'*64),dict(evidence_bytes={'source_000001':PNG+b'bad'}),
                       dict(asset_bytes={}),dict(mask_bytes={('invented','part'):PNG})]:
            with self.subTest(change=list(change)),self.assertRaises(ValueError):self.build(**change)
        self.assertFalse((self.root/'bundle').exists())

    def test_source_video_is_opt_in_and_hash_checked(self):
        self.doc['source']['sha256'] = P.digest(b'fake-test-video')
        self.args['program_bytes'] = P.encoded(self.doc)
        with self.assertRaises(ValueError):self.build(source_video_bytes=b'wrong')
        result = self.build(source_video_bytes=b'fake-test-video')
        self.assertTrue(P.inspect_package(result['zip'])['video_included'])

    def test_package_tampering_extra_files_and_semantic_changes_rejected(self):
        result = self.build()
        out = Path(result['directory'])
        (out/'extra.txt').write_text('unexpected')
        with self.assertRaises(ValueError):P.inspect_package(out)
        (out/'extra.txt').unlink()
        raw = (out/'evidence/source_000001.png').read_bytes()
        (out/'evidence/source_000001.png').write_bytes(raw+b'bad')
        with self.assertRaises(ValueError):P.inspect_package(out)

    def test_rehashed_arbitrary_javascript_or_viewer_data_rejected(self):
        result = self.build()
        out = Path(result['directory'])
        original = (out/'viewer.js').read_bytes()
        self.rewrite(out,'viewer.js',original+b'\nalert(1)')
        with self.assertRaisesRegex(ValueError,'trusted'):P.inspect_package(out)
        self.rewrite(out,'viewer.js',original)
        payload = json.loads((out/'viewer-data.json').read_bytes())
        payload['policy']['simulation_ready'] = True
        self.rewrite(out,'viewer-data.json',P.encoded(payload))
        self.rewrite(out,'viewer-data.js',P.script_data(payload))
        with self.assertRaisesRegex(ValueError,'Viewer data'):P.inspect_package(out)

    def test_rehashed_fake_mesh_primitives_rejected_by_redecode(self):
        result = self.build();out = Path(result['directory'])
        payload = json.loads((out/'viewer-data.json').read_bytes())
        payload['scene']['objects'][0]['primitives'][0]['vertices'][0][0] += 9
        self.rewrite(out,'viewer-data.json',P.encoded(payload))
        self.rewrite(out,'viewer-data.js',P.script_data(payload))
        with self.assertRaisesRegex(ValueError,'Viewer data'):P.inspect_package(out)

    def test_rehashed_false_backend_appearance_and_original_path_citations_rejected(self):
        cases=[('assets','backend','SAM3D_FALSE_TEST_CLAIM'),('assets','original_workspace_path','data/invented/mesh.glb'),
               ('assets','appearance','measured_photographic_texture'),('source','original_workspace_path','data/invented/video.mp4'),
               ('evidence','original_workspace_path','data/invented/frame.png')]
        for index,(section,key,value) in enumerate(cases):
            with self.subTest(section=section,key=key):
                result=self.build(name='forged-'+str(index));out=Path(result['directory'])
                manifest=json.loads((out/'manifest.json').read_bytes())
                target=manifest[section] if section=='source' else manifest[section][0]
                target[key]=value
                payload=json.loads((out/'viewer-data.json').read_bytes())
                (payload[section] if section=='source' else payload[section][0])[key]=value
                for name,data in [('viewer-data.json',P.encoded(payload)),('viewer-data.js',P.script_data(payload))]:
                    (out/name).write_bytes(data)
                    record=next(r for r in manifest['files'] if r['path']==name)
                    record.update(bytes=len(data),sha256=P.digest(data))
                (out/'manifest.json').write_bytes(P.encoded(manifest))
                with self.assertRaisesRegex(ValueError,'citation|identity/units'):P.inspect_package(out)

    def test_untrusted_paths_duplicates_and_symlink_zip_rejected(self):
        for name in ['../escape','/absolute','.hidden','a/../b','a\\b','https:x']:
            with self.subTest(name=name),self.assertRaises(ValueError):P.safe_name(name)
        archive = self.root/'bad.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('index.html',b'one');z.writestr('index.html',b'two')
        with self.assertRaisesRegex(ValueError,'Duplicate'):P.inspect_package(archive)
        archive.unlink()
        with zipfile.ZipFile(archive,'w') as z:
            info = zipfile.ZipInfo('link');info.create_system=3;info.external_attr=(stat.S_IFLNK|0o777)<<16
            z.writestr(info,'target')
        with self.assertRaisesRegex(ValueError,'symlinks'):P.inspect_package(archive)

    def test_directory_symlink_and_import_collision_rejected(self):
        result=self.build();out=Path(result['directory'])
        (out/'link').symlink_to(out/'index.html')
        with self.assertRaisesRegex(ValueError,'symlink'):P.inspect_package(out)
        (out/'link').unlink()
        with self.assertRaises(ValueError):P.inspect_package(result['zip'],extract_to=out)

    def test_external_glb_textures_and_truncated_bytes_rejected(self):
        for raw in [glb(texture=True,external=True),self.mesh[:-4],b'glTFbad']:
            with self.subTest(length=len(raw)),self.assertRaises(ValueError):P.decode_glb_bytes(raw)

    def test_script_data_escapes_html_breakouts(self):
        result = P.script_data({'x':'</script><script>alert(1)</script>\u2028'})
        self.assertNotIn(b'<',result)
        self.assertIn(b'\\u003c',result)

    def test_offline_viewer_uses_exact_states_and_hides_gaps(self):
        source = """
const v=require(process.argv[1]);const s=require(process.argv[2]);const d=JSON.parse(process.argv[3]);
if(v.exactState(d,0)[0].value!=='closed'||v.exactState(d,.5)[0].value!=='unknown')throw Error('state leaked');
if(v.exactEvidence(d,.5)!==null)throw Error('future sample leaked');
const o={frames:[{timestamp_s:1,visible:true,status:'supported_conditional_motion_candidate'}],max_hold_s:0};
if(!s.recordAt(o,1).visible||s.recordAt(o,1.00001).visible)throw Error('pose leaked');
for(const bad of ['../escape','/absolute','https://external']){let threw=false;try{v.path(bad)}catch(e){threw=true}if(!threw)throw Error('unsafe link')}
"""
        result = subprocess.run(['node','-e',source,str(P.TEMPLATES/'viewer.js'),str(P.TEMPLATES/'scene.js'),json.dumps(self.doc)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_actual_shipped_viewer_runtime_controls_and_render_decisions(self):
        holder = copy.deepcopy(self.doc['assets'][0])
        holder.update(id='generated_holder',track_id='holder')
        self.doc['assets'].append(holder)
        self.args['asset_bytes']['holder']=self.mesh
        self.args['program_bytes']=P.encoded(self.doc)
        result = self.build()
        command=['node',str(ROOT/'tests/portable_viewer_runtime.cjs'),
                 str(Path(result['directory'])/'viewer-data.json'),str(P.TEMPLATES)]
        run=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertEqual(json.loads(run.stdout)['status'],'passed')

    def test_actual_viewer_gap_navigation_at_point_one_seconds(self):
        self.doc['assets'].append({**self.doc['assets'][0],'id':'generated_holder','track_id':'holder'})
        self.args['asset_bytes']['holder']=self.mesh
        for e in self.doc['evidence']:e['timestamp_s']/=5
        for m in self.doc['motion']:m['timestamp_s']/=5
        for s in self.doc['steps']:
            s['start_s']/=5;s['end_s']/=5
        self.args['program_bytes']=P.encoded(self.doc)
        result=self.build()
        run=subprocess.run(['node',str(ROOT/'tests/portable_viewer_runtime.cjs'),
                            str(Path(result['directory'])/'viewer-data.json'),str(P.TEMPLATES)],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)

    def test_shipped_html_has_only_local_resources_and_no_network_capabilities(self):
        import re
        html=(P.TEMPLATES/'index.html').read_text()
        self.assertIn("connect-src 'none'",html)
        for target in re.findall(r'(?:src|href)="([^"]+)"',html):
            self.assertEqual(P.safe_name(target),target)
        for name in ['viewer.js','scene.js']:
            source=(P.TEMPLATES/name).read_text()
            self.assertNotRegex(source,r'\b(?:fetch|WebSocket|XMLHttpRequest|importScripts)\s*\(')


if __name__=='__main__':unittest.main()
