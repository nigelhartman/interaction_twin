"""Small synthetic codec fixture; no owner footage, catalog or model calls."""
import importlib.util,json,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
try:
    import replace_case_film_20261105_v3 as C
except ModuleNotFoundError as e:
    if e.name and e.name.split('.')[0] in {'PIL','numpy'}:
        raise unittest.SkipTest('Optional film environment (Pillow and NumPy) is required.')from e
    raise

class FilmSplice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all(shutil.which(n)for n in('ffmpeg','ffprobe')):raise unittest.SkipTest('FFmpeg is optional in source-only review.')
        cls.temp=tempfile.TemporaryDirectory(prefix='interaction-twin-splice-')
        cls.folder=Path(cls.temp.name);cls.source=cls.folder/'synthetic.mp4'
        C.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','testsrc2=size=160x90:rate=30:duration=3','-f','lavfi','-i','sine=frequency=500:sample_rate=48000:duration=3','-c:v','libx264','-pix_fmt','yuv420p','-g','30','-keyint_min','30','-sc_threshold','0','-c:a','aac','-t','3',str(cls.source)])
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_keyframe_segments_rejoin_without_extra_b_frames(self):
        parts=C.split_video(self.source,self.folder/'parts',[30,60],90)
        joined=C.join_video(parts,[30,30,30],self.folder,'joined')
        self.assertEqual(C.hashes(self.source),C.hashes(joined))
        final=self.folder/'audio-copy.mp4';C.mux_audio(joined,self.source,final,3)
        self.assertEqual(C.audio_packets(self.source),C.audio_packets(final))
        probe=json.loads(C.run(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(final)]).stdout)
        self.assertAlmostEqual(float(probe['format']['duration']),3,places=3)
    def test_split_rejects_nonkeyframe_boundary(self):
        with self.assertRaises(ValueError):C.split_video(self.source,self.folder/'bad-parts',[31,60],90)
    def test_range_proof_detects_noncase_changes(self):
        self.assertEqual(C.equal_ranges(['a','b','c'],['a','x','c'],[(0,1),(2,3)])[1]['decoded_pictures_equal'],1)
        with self.assertRaises(ValueError):C.equal_ranges(['a','b'],['x','b'],[(0,1)])
        with self.assertRaises(ValueError):C.equal_ranges(['a'],['a','b'],[(0,1)])
    def test_output_is_additive_and_scoped(self):
        self.assertEqual(C.output(C.OUT/'new-draft'),C.OUT/'new-draft')
        for path in(C.OUT,ROOT/'showcase/media/overwrite.mp4',self.folder/'outside'):
            with self.assertRaises(ValueError):C.output(path)

if __name__=='__main__':unittest.main()
