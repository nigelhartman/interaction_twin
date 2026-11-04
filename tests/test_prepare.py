import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import twin

class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.previous=twin.ROOT;twin.ROOT=Path(self.tmp.name)
        self.video=twin.ROOT/'fixture.mp4'
        twin.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=320x240:rate=12','-t','3','-c:v','libx264','-pix_fmt','yuv420p',self.video])
    def tearDown(self):
        twin.ROOT=self.previous;self.tmp.cleanup()
    def args(self,**kwargs):
        data=dict(case='fixture',file=str(self.video),url=None,fps=1,max_width=320,start=.5,end=2.6);data.update(kwargs)
        return SimpleNamespace(**data)
    def test_original_pts_survive_interval_sampling(self):
        with contextlib.redirect_stdout(io.StringIO()):twin.prepare(self.args())
        case=twin.case_path('fixture');rows=twin.read_json(case/'frames/index.json')['frames']
        self.assertEqual([r['timestamp_s'] for r in rows],[.5,1.5,2.5])
        self.assertTrue(all((case/r['file']).is_file() for r in rows))
        self.assertTrue((case/'overview.jpg').is_file());self.assertTrue((case/'review.html').is_file())
    def test_idempotence_and_refusal_to_overwrite_changed_settings(self):
        with contextlib.redirect_stdout(io.StringIO()):twin.prepare(self.args())
        manifest=twin.case_path('fixture')/'manifest.json';before=manifest.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):twin.prepare(self.args())
        self.assertEqual(before,manifest.read_bytes())
        with self.assertRaises(ValueError):twin.prepare(self.args(fps=2))
        self.assertEqual(before,manifest.read_bytes())
    def test_reject_rebinding_case_to_another_url(self):
        with contextlib.redirect_stdout(io.StringIO()):twin.prepare(self.args())
        with self.assertRaises(ValueError):twin.prepare(self.args(url='https://example.com/other-video'))
    def test_reject_invalid_rate_and_interval(self):
        for change in [dict(fps=0),dict(start=2,end=1),dict(end=4)]:
            with self.subTest(change=change),self.assertRaises(ValueError):twin.prepare(self.args(**change))
    def test_case_slug_cannot_escape_data_directory(self):
        for value in ['../outside','/tmp/escape','bad name','']:
            with self.subTest(value=value),self.assertRaises(ValueError):twin.case_path(value)

if __name__=='__main__':unittest.main()
