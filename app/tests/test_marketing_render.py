"""Real local FFmpeg contract, using an explicitly synthetic software fixture."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from PIL import Image,ImageChops
from marketing_sources import ROOT,brand,file_ref,LIMITATION,CTA
from marketing_render import render,probe,frame,brand_sequence,BG

@unittest.skipUnless(shutil.which('ffmpeg')and shutil.which('ffprobe'),'Local FFmpeg required')
class MarketingRenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
        (self.root/'showcase').mkdir();(self.root/'config').mkdir();(self.root/'.private/marketing-studio/renders').mkdir(parents=True)
        shutil.copy(ROOT/'showcase/mark.svg',self.root/'showcase/mark.svg');shutil.copy(ROOT/'config/marketing-brand.json',self.root/'config/marketing-brand.json')
        self.clip=self.root/'synthetic.mp4'
        subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','color=c=0x3766ed:s=96x64:r=6:d=6','-an','-c:v','libx264','-pix_fmt','yuv420p',str(self.clip)],check=True,capture_output=True,timeout=20)
        self.revision={'sources':{'fixture':{'file':file_ref(self.clip,self.root),'kind':'source_video','duration_s':6.,'limitation':'Synthetic software fixture, excluded from genuine catalog.'}},'storyboard':[{'source_id':'fixture','start_s':0.,'duration_s':4.5,'headline_id':'source','fact_id':'source','transition':'crossfade'}],'duration_s':8,'brand':brand(self.root),'music_id':None,'music':None,'copy':{'x':'Software fixture'},'preferences':{}}
    def tearDown(self):self.tmp.cleanup()
    def test_real_render_has_declared_dimensions_duration_captions_and_exact_provenance(self):
        out=self.root/'.private/marketing-studio/renders/software-fixture';progress=[]
        result=render(self.revision,'square',out,root=self.root,progress=progress.append)
        info=probe(self.root/result['video']['path']);stream=next(s for s in info['streams']if s['codec_type']=='video')
        self.assertEqual((stream['width'],stream['height']),(720,720));self.assertAlmostEqual(float(info['format']['duration']),8,delta=.1)
        self.assertFalse(any(s['codec_type']=='audio'for s in info['streams']))
        manifest=json.loads((self.root/result['manifest']['path']).read_text())
        self.assertEqual(manifest['sources'][0]['file'],file_ref(self.clip,self.root));self.assertFalse(manifest['publication_performed'])
        captions=(self.root/result['captions']['path']).read_text()
        self.assertIn('WEBVTT',captions);self.assertIn(LIMITATION,captions);self.assertIn(CTA,captions)
        self.assertEqual(manifest['render_text']['mandatory_limitation'],LIMITATION);self.assertGreater(len(progress),2)
    def test_arbitrary_destination_and_changed_source_fail_closed(self):
        with self.assertRaises(ValueError):render(self.revision,'square',self.root/'public',root=self.root)
        self.clip.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'changed'):render(self.revision,'square',self.root/'.private/marketing-studio/renders/changed',root=self.root)
    def test_declared_cut_and_wipe_transitions_really_render(self):
        for transition in ['cut','cobalt_wipe']:
            with self.subTest(transition=transition):
                self.revision['storyboard'][0]['transition']=transition
                result=render(self.revision,'square',self.root/'.private/marketing-studio/renders'/transition,root=self.root)
                self.assertAlmostEqual(result['duration_s'],8,delta=.1)
    def test_branded_evidence_uses_full_frame_without_another_logo_or_card(self):
        for kind in ['pipeline_film','robot_preview']:
            path=self.root/(kind+'.png')
            box=frame(path,(1280,720),headline='Must not be drawn twice',caption='Actual saved project evidence.',limitation=LIMITATION,source_kind=kind,source_size=(1920,1080))
            self.assertEqual(box,(0,0,1280,650))
            im=Image.open(path).convert('RGB')
            # The whole evidence area is a clean compositor surface: duplicate
            # outer logos, headlines and card outlines would change these pixels.
            self.assertIsNone(ImageChops.difference(im.crop((0,0,1280,650)),Image.new('RGB',(1280,650),BG)).getbbox())
            self.assertIsNotNone(ImageChops.difference(im.crop((0,650,1280,720)),Image.new('RGB',(1280,70),BG)).getbbox())
    def test_portrait_capture_in_landscape_has_full_height_right_column(self):
        path=self.root/'two-columns.png';box=frame(path,(1280,720),headline='Show the real interaction.',caption='Inspect the original demonstration.',limitation=LIMITATION,source_kind='source_video',source_size=(720,1280))
        x,y,w,h=box;self.assertGreater(x,1280*.5);self.assertEqual(y,58);self.assertEqual(h,604)
        self.assertLessEqual(x+w,1280-58)
        # A full portrait source is now604px high, not a tiny strip inside the
        # earlier short central card. Its original aspect ratio stays intact.
        source_h=min(h,w*1280/720);self.assertEqual(source_h,604)
    def test_intro_and_outro_have48_distinct_scale_fade_frames(self):
        for closing in [False,True]:
            folder=self.root/('outro'if closing else'intro');pattern=brand_sequence(folder,(320,180),closing=closing)
            self.assertEqual(pattern.name,'%04d.png');self.assertEqual(len(list(folder.glob('*.png'))),48)
            first=Image.open(folder/'0000.png').convert('RGB');middle=Image.open(folder/'0024.png').convert('RGB');last=Image.open(folder/'0047.png').convert('RGB')
            self.assertIsNotNone(ImageChops.difference(first,middle).getbbox());self.assertIsNotNone(ImageChops.difference(middle,last).getbbox())
    def test_real_branded_render_manifest_records_larger_contained_evidence(self):
        self.revision['sources']['fixture']['kind']='pipeline_film'
        result=render(self.revision,'landscape',self.root/'.private/marketing-studio/renders/branded-fixture',root=self.root)
        manifest=json.loads((self.root/result['manifest']['path']).read_text())
        self.assertEqual(manifest['renderer_version'],2);source=manifest['sources'][0]
        self.assertEqual(source['layout'],'branded_evidence_full_frame');self.assertEqual(source['evidence_box'],[0,0,1280,650]);self.assertFalse(source['outer_headline_rendered'])
        self.assertEqual(source['fit'],'contain');self.assertEqual(manifest['brand_animation']['frames_per_opening_or_closing'],48)

if __name__=='__main__':unittest.main()
