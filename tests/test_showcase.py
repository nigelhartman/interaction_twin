"""Independent local showcase link, provenance and packaging regressions."""
import hashlib
from html.parser import HTMLParser
import importlib.util
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs,unquote,urlsplit

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('showcase_builder',ROOT/'scripts/build_showcase.py')
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Links(HTMLParser):
    def __init__(self):super().__init__();self.rows=[]
    def handle_starttag(self,tag,attributes):
        for key,value in attributes:
            if key in {'href','src','poster'} and value:self.rows.append((tag,key,value))


class ShowcaseArtifacts(unittest.TestCase):
    def test_public_page_assets_resolve_without_upload_or_private_project_navigation(self):
        page=ROOT/'showcase/index.html';links=Links();links.feed(page.read_text())
        uploads=[]
        for tag,key,value in links.rows:
            parsed=urlsplit(value)
            if parsed.scheme or parsed.netloc:
                self.assertEqual(tag,'a','Executable/media assets must stay local: '+value)
                continue
            if not parsed.path:continue
            target=((ROOT/unquote(parsed.path).removeprefix('/files/')) if parsed.path.startswith('/files/') else (page.parent/unquote(parsed.path))).resolve()
            self.assertTrue(target.is_relative_to(ROOT),value)
            self.assertTrue(target.is_file(),value)
            if parse_qs(parsed.query).get('new')==['1']:
                uploads.append(target)
                self.assertEqual(target,ROOT/'workspace/index.html')
                self.assertEqual(set(parse_qs(parsed.query)),{'new'})
        self.assertEqual(uploads, [], 'The public showcase must remain browse-only')
        for tag,key,value in links.rows:
            if tag=='a':
                self.assertNotIn('/admin', value)
                self.assertNotIn('../workspace/', value, 'Private project pages must not be public navigation')

    def test_packaged_media_bytes_match_saved_sources_and_report(self):
        report=json.loads((ROOT/'showcase/build-report.json').read_text())
        self.assertFalse(report['published']);self.assertFalse(report['new_inference_performed'])
        self.assertFalse(report['physical_accuracy_claimed']);self.assertTrue(report['files'])
        for row in report['files']:
            with self.subTest(path=row['path']):
                source=ROOT/row['source'];target=ROOT/row['path']
                self.assertTrue(source.is_file());self.assertTrue(target.is_file())
                self.assertEqual(sha(source),row['sha256']);self.assertEqual(sha(target),row['sha256'])
                self.assertEqual(target.stat().st_size,row['bytes'])

    def test_media_provenance_still_matches_actual_video_masks_and_meshes(self):
        manifest=json.loads((ROOT/'showcase/media/source-manifest.json').read_text())
        rows=[]
        def visit(value):
            if isinstance(value,dict):
                name=value.get('path',value.get('file'))
                if name and 'sha256' in value:rows.append((name,value['sha256']))
                for child in value.values():visit(child)
            elif isinstance(value,list):
                for child in value:visit(child)
        visit(manifest)
        self.assertTrue(any(name.endswith('.glb') for name,_ in rows))
        self.assertTrue(any('/raw_masks/' in name for name,_ in rows))
        for name,expected in set(rows):
            with self.subTest(path=name):
                path=(ROOT/name).resolve()
                self.assertTrue(path.is_relative_to(ROOT));self.assertTrue(path.is_file())
                self.assertEqual(sha(path),expected)

    def test_interactive_sample_payloads_have_strict_source_and_asset_lineage(self):
        import stage5
        payload=json.loads((ROOT/'workspace/cases.json').read_text())
        javascript=(ROOT/'workspace/cases.js').read_text().strip()
        self.assertTrue(javascript.startswith('window.TWIN_CASES='))
        embedded=json.loads(javascript[len('window.TWIN_CASES='):].rstrip(';'))
        self.assertEqual(embedded,payload)
        ids=re.findall(r"id:'([^']+)'",(ROOT/'showcase/app.js').read_text())
        self.assertGreaterEqual(len(ids),3)
        cases={c['id']:c for c in payload['cases']}
        for name in ids:
            with self.subTest(sample=name):
                self.assertIn(name,cases)
                doc=cases[name]['interaction'];stage5.validate_interaction(doc)
                self.assertFalse(doc['runtime_policy']['simulation_ready'])
                self.assertEqual(doc['source']['time_domain'],'source_video_presentation_seconds')
                self.assertTrue(doc['evidence'])

    def test_missing_required_media_fails_without_fabricating_a_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for folder,names in [('showcase-media',B.MEDIA),('showcase-graphics',B.GRAPHICS)]:
                for name in names:
                    if name=='mesh-earbud.png':continue
                    path=root/'output'/folder/name;path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(('fixture '+name).encode())
            with patch.object(B,'ROOT',root),self.assertRaises(FileNotFoundError) as error:B.build()
            self.assertIn('mesh-earbud.png',str(error.exception))
            self.assertFalse((root/'showcase/media/mesh-earbud.png').exists())
            self.assertFalse((root/'showcase/build-report.json').exists())

    @unittest.skipUnless(shutil.which('ffprobe'),'Existing FFprobe is required; no installation attempted')
    def test_trailer_packaged_duration_and_geometry_match_its_manifest(self):
        manifest=json.loads((ROOT/'showcase/media/source-manifest.json').read_text())['trailer']
        result=subprocess.run(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(ROOT/'showcase/media/trailer.mp4')],capture_output=True,text=True,check=True,timeout=15)
        actual=json.loads(result.stdout);video=next(s for s in actual['streams'] if s['codec_type']=='video')
        self.assertEqual(video['width'],manifest['width']);self.assertEqual(video['height'],manifest['height'])
        self.assertAlmostEqual(float(actual['format']['duration']),manifest['duration_s'],delta=1/manifest['fps'])
        self.assertFalse(any(s['codec_type']=='audio' for s in actual['streams']))


@unittest.skipUnless(shutil.which('node'),'Existing Node.js is required; no installation attempted')
class ShowcaseControls(unittest.TestCase):
    def scenario(self,name):
        result=subprocess.run([shutil.which('node'),str(ROOT/'tests/showcase_app_harness.cjs'),name],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_public_curated_source_and_time_links_preserve_exact_binding_and_read_only_navigation(self):
        result=subprocess.run([shutil.which('node'),str(ROOT/'tests/test_showcase_public_deeplink.cjs')],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_case_switching_uses_actual_source_bound_sam3d_previews_without_old_scenes(self):
        self.scenario('switch_native')

    def test_pending_native_recording_never_substitutes_its_historical_model(self):
        self.scenario('pending_no_substitute')

    def test_clicked_fresh_chapters_survive_actual_shared_step_boundaries(self):
        self.scenario('shared_step_boundaries')

    def test_missing_sample_preserves_the_current_working_native_preview(self):
        self.scenario('missing_sample')

    def test_catalog_error_removes_the_old_native_preview_without_substitution(self):
        self.scenario('catalog_error')

    def test_stale_catalog_success_cannot_replace_a_newer_recording(self):
        self.scenario('stale_success')

    def test_stale_catalog_error_cannot_replace_a_newer_recording(self):
        self.scenario('stale_error')

    def test_actual_aligned_scene_binding_playhead_handshake_and_reused_mode_switch(self):
        if not (ROOT/'showcase/public/aligned-scenes.json').exists():self.skipTest('Actual curated aligned scene is not present in this checkout')
        self.scenario('aligned_sync')


if __name__=='__main__':unittest.main()
