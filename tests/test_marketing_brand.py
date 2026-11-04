"""Versioned brand selection never rewrites saved campaign or movie evidence."""
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from PIL import Image,ImageDraw

import marketing_sources as S
import marketing_render as R
from marketing_studio import MarketingStudio


class MarketingBrandTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        for relative in ['config/marketing-brand.json','showcase/mark.svg']:
            path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy(S.ROOT/relative,path)
        self.old=S.brand(self.root);self.old_bytes={p:(self.root/p).read_bytes()for p in ['config/marketing-brand.json','showcase/mark.svg']}

    def install_v2(self,*,mode='RGBA',size=64):
        # Deliberately authored software fixture, never a real public brand.
        folder=self.root/'showcase/brand/v2';folder.mkdir(parents=True,exist_ok=True)
        svg=folder/'fixture.svg';svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect width="64" height="64" fill="#ff0080"/></svg>')
        png=folder/'fixture.png';image=Image.new(mode,(size,size),'#ff0080');image.save(png)
        config=copy.deepcopy(self.old['document']);config.update(version=2,style_id='cobalt-motion-v2',logo_path='showcase/brand/v2/fixture.svg',logo_raster_path='showcase/brand/v2/fixture.png')
        path=self.root/'config/marketing-brand-v2.json';path.write_text(json.dumps(config))
        registry=copy.deepcopy(S.BRAND_REGISTRY);registry['cobalt-motion-v2']={'path':'config/marketing-brand-v2.json','sha256':S.file_ref(path,self.root)['sha256'],'version':2,'logo_path':config['logo_path'],'logo_sha256':S.file_ref(svg,self.root)['sha256'],'logo_raster_path':config['logo_raster_path'],'logo_raster_sha256':S.file_ref(png,self.root)['sha256']}
        patched=patch.object(S,'BRAND_REGISTRY',registry);patched.start();self.addCleanup(patched.stop)
        (self.root/S.BRAND_SELECTOR).write_text(json.dumps({'format':'interaction-twin-marketing-brand-default','version':1,'brand_id':'cobalt-motion-v2'}))
        return S.brand(self.root)

    def test_missing_selector_is_exact_legacy_snapshot_and_original_logo_pixels(self):
        self.assertEqual(S.brand(self.root),self.old);self.assertEqual(set(self.old),{'document','reference','logo'})
        expected={30:'f95f901c4bd438628f66ffc674d0842a8d2128cd3f7442a3f45abf08b3b7da31',90:'b9aa8d61d947f653fd2c44f5043058691fd5707b50c2b95daa205a974981482f'}
        for size,sha in expected.items():
            image=Image.new('RGB',(size+20,size+20),R.BG);R.logo(image,10,10,size)
            self.assertEqual(hashlib.sha256(image.tobytes()).hexdigest(),sha)

    def test_new_default_keeps_old_style_valid_and_both_old_files_byte_identical(self):
        new=self.install_v2();self.assertEqual(new['document']['style_id'],'cobalt-motion-v2');self.assertIn('logo_raster',new)
        self.assertEqual(S.validated_brand(self.old,self.root),self.old);self.assertIsNone(R.load_brand_artwork(self.old,self.root))
        artwork=R.load_brand_artwork(new,self.root);self.assertEqual(artwork.mode,'RGBA');self.assertEqual(artwork.size,(64,64))
        for path,original in self.old_bytes.items():self.assertEqual((self.root/path).read_bytes(),original)
        target=Image.new('RGB',(80,80),R.BG);R.logo(target,8,8,64,artwork=artwork)
        self.assertEqual(target.getpixel((40,40)),(255,0,128))

    def test_selector_never_accepts_arbitrary_paths_styles_extra_fields_or_symlinks(self):
        selector=self.root/S.BRAND_SELECTOR
        for value in [{'format':'interaction-twin-marketing-brand-default','version':1,'brand_id':'../../.secrets/meta.env'},
                      {'format':'interaction-twin-marketing-brand-default','version':1,'brand_id':'cobalt-motion-v1','path':'other.json'},
                      {'format':'interaction-twin-marketing-brand-default','version':True,'brand_id':'cobalt-motion-v1'}]:
            selector.write_text(json.dumps(value))
            with self.subTest(value=value),self.assertRaises(ValueError):S.brand(self.root)
        selector.unlink();selector.symlink_to(self.root/'config/marketing-brand.json')
        with self.assertRaises(ValueError):S.brand(self.root)

    def test_saved_document_and_all_three_ref_bindings_are_exact_not_free_form(self):
        new=self.install_v2()
        for change in [('document','slogan','Changed'),('reference','path','.secrets/meta.env'),('logo','sha256','a'*64),('logo_raster','path','private.png')]:
            value=copy.deepcopy(new);value[change[0]][change[1]]=change[2]
            with self.subTest(change=change),self.assertRaises(ValueError):S.validated_brand(value,self.root)
        legacy=copy.deepcopy(self.old);legacy['document']['style_id']='unknown'
        with self.assertRaisesRegex(ValueError,'Unknown approved'):R.load_brand_artwork(legacy,self.root)

    def test_modified_registered_config_or_artwork_and_invalid_png_fail_closed(self):
        new=self.install_v2();png=self.root/new['logo_raster']['path'];original=png.read_bytes();png.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'raster changed'):S.brand(self.root)
        png.write_bytes(original);config=self.root/new['reference']['path'];config.write_text(config.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'guide changed'):S.brand(self.root)
        for mode,size in [('RGB',64),('RGBA',1025)]:
            with self.subTest(mode=mode,size=size):
                invalid=self.install_v2(mode=mode,size=size)
                with self.assertRaisesRegex(ValueError,'square RGBA'):R.load_brand_artwork(invalid,self.root)

    @unittest.skipUnless(shutil.which('ffmpeg')and shutil.which('ffprobe'),'Local FFmpeg required')
    def test_queued_legacy_campaign_really_renders_and_exports_after_new_default(self):
        source=self.root/'software-fixture.mp4'
        subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','color=c=0x3766ed:s=96x64:r=6:d=6','-an','-c:v','libx264','-pix_fmt','yuv420p',str(source)],check=True,capture_output=True,timeout=20)
        root=self.root
        class Sources:
            def get(self,identifier):
                if identifier!='fixture':raise ValueError('Only software fixture')
                return {'id':identifier,'label':'Synthetic test clip','kind':'source_video','duration_s':6.,'file':S.file_ref(source,root),'pack_ids':[],'limitation':'Synthetic software fixture.'}
        studio=MarketingStudio(self.root,sources=Sources());old=studio.create_campaign('Old queued fixture',source_ids=['fixture'],duration_s=15,music_id=None)
        cid=old['campaign']['id'];rid=old['latest_revision']['id'];campaign_file=studio.records/(cid+'.json');revision_file=studio.records/(rid+'.json');saved=(campaign_file.read_bytes(),revision_file.read_bytes())
        new=self.install_v2();fresh=studio.create_campaign('New brand fixture',source_ids=['fixture'],duration_s=15,music_id=None)
        self.assertEqual(fresh['latest_revision']['brand'],new);self.assertEqual(studio.detail(cid)['latest_revision']['brand'],self.old)
        job=studio.run_job(old['jobs'][0]['id']);self.assertEqual(job['status'],'review_ready')
        manifest=json.loads((self.root/job['result']['manifest']['path']).read_text());self.assertEqual(manifest['brand'],self.old);self.assertEqual(manifest['renderer_version'],2)
        movie=(self.root/job['result']['video']['path']).read_bytes();studio.approve_revision(cid,rid);reference=studio.export_revision(cid,rid);archive=(self.root/reference['path']).read_bytes()
        with zipfile.ZipFile(io.BytesIO(archive))as bundle:self.assertEqual(bundle.read('mark.svg'),self.old_bytes['showcase/mark.svg'])
        self.assertEqual(studio.export_revision(cid,rid),reference);self.assertEqual((self.root/reference['path']).read_bytes(),archive);self.assertEqual((self.root/job['result']['video']['path']).read_bytes(),movie)
        self.assertEqual((campaign_file.read_bytes(),revision_file.read_bytes()),saved)
        new_job=studio.run_job(fresh['jobs'][0]['id']);self.assertEqual(new_job['status'],'review_ready')
        new_manifest=json.loads((self.root/new_job['result']['manifest']['path']).read_text())
        self.assertEqual(new_manifest['renderer_version'],3);self.assertEqual(new_manifest['brand'],new)
        self.assertEqual(new_manifest['brand']['logo_raster'],new['logo_raster'])
        studio.approve_revision(fresh['campaign']['id'],fresh['latest_revision']['id'])
        new_export=studio.export_revision(fresh['campaign']['id'],fresh['latest_revision']['id'])
        with zipfile.ZipFile(self.root/new_export['path'])as bundle:
            self.assertEqual(bundle.read('mark.svg'),(self.root/new['logo']['path']).read_bytes())
            self.assertNotEqual(bundle.read('mark.svg'),self.old_bytes['showcase/mark.svg'])


if __name__=='__main__':unittest.main()
