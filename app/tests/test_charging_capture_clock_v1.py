"""Decoded timing gate is independent of optional MediaRecorder metadata."""
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/'scripts/bind_charging_capture_20261105_v1.py'
spec=importlib.util.spec_from_file_location('charging_binder',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ClockTests(unittest.TestCase):
    def frames(self):return [{'best_effort_timestamp_time':str(i/30)} for i in range(390)]
    def test_complete_clock_without_format_duration(self):
        duration,pts=module.decoded_clock(self.frames());self.assertAlmostEqual(duration,13);self.assertEqual(len(pts),390)
    def test_missing_tail_rejected(self):
        with self.assertRaises(ValueError):module.decoded_clock(self.frames()[:-20])
    def test_duplicate_or_large_gap_rejected(self):
        frames=self.frames();frames[100]['best_effort_timestamp_time']=frames[99]['best_effort_timestamp_time']
        with self.assertRaises(ValueError):module.decoded_clock(frames)
        frames=self.frames();del frames[100:110]
        with self.assertRaises(ValueError):module.decoded_clock(frames)
    def test_nonfinite_or_too_sparse_rejected(self):
        frames=self.frames();frames[100]['best_effort_timestamp_time']='nan'
        with self.assertRaises(ValueError):module.decoded_clock(frames)
        with self.assertRaises(ValueError):module.decoded_clock(self.frames()[::2])

if __name__=='__main__':unittest.main()
