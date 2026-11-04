"""Compact media provenance and known-413 continuation; no hosted calls."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
C=module('compact_sam','scripts/stage2_meta_video_compact_20261103.py')
R=module('compact_recovery','scripts/recover_capture_compact_sam_20261103.py')


class CompactSAMTests(unittest.TestCase):
    def test_real_cpu_encode_preserves_all_frame_times_dimensions_and_discloses_loss(self):
        with tempfile.TemporaryDirectory()as tmp:
            case=Path(tmp);rows=[];sources=[]
            for n in range(4):
                a=np.zeros((128,96,3),np.uint8);a[:,:,n%3]=np.arange(96,dtype=np.uint8)[None,:]+n*20;a[20:70,10+n*7:55+n*7]=[30,110,210]
                path=case/f'{n}.png';Image.fromarray(a).save(path)
                rows.append({'file':path.name,'width':96,'height':128});sources.append({'timestamp_s':n*.6667,'sample_index':n+1,'video_sha256':'a'*64})
            video,encoding=C.prepare_video(case,rows,case/'encoded',sources)
            self.assertLessEqual(video.stat().st_size,C.MAX_VIDEO_BYTES)
            self.assertEqual((encoding['width'],encoding['height'],encoding['frame_count']),(96,128,4))
            self.assertTrue(encoding['lossy_compression_performed']);self.assertFalse(encoding['lossless_pixels_claimed'])
            self.assertFalse(encoding['resize_performed']);self.assertFalse(encoding['crop_performed'])
            self.assertEqual([r['source']for r in encoding['frames']],sources)
            self.assertEqual([r['submitted_timestamp_s']for r in encoding['frames']],[0,1,2,3])
            self.assertTrue(all(len(r['decoded_yuv420_pixels_sha256'])==64 for r in encoding['frames']))
            self.assertTrue(all(r['psnr_yuv_db']is None or r['psnr_yuv_db']>=30 for r in encoding['frames']))

    def test_exact_guarded_segmentation_argv_is_replaced_and_other_commands_are_unchanged(self):
        runner=object.__new__(R.CompactRecoveryRunner)
        args=[str(ROOT/'.venv-sam/bin/python'),str(ROOT/'scripts/run_overnight_stage.py'),'--task-budget-plan',R.V1.BUDGET,
            'stage2_meta_video_segment.py','--case','source-case','--config','annotations/auto.json','--run','flow-components-2']
        with patch.object(R.P.Runner,'command')as command:
            runner.command(args,300)
            actual=command.call_args.args[0]
            self.assertEqual(actual,[args[0],str(R.ADAPTER),*args[5:],'--allow-paid-api'])
            other=[args[0],'scripts/stage3_select.py'];runner.command(other,300);self.assertEqual(command.call_args.args[0],other)
        with self.assertRaisesRegex(ValueError,'exact reviewed'):runner.command([args[0],'stage2_meta_video_segment.py'],300)

    def test_known_413_requires_matching_released_ledger_and_preserves_original(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(R,'ROOT',Path(tmp)),patch.object(R.V1,'ROOT',Path(tmp)):
            root=Path(tmp);case=root/'data/case';out=case/'segmentation/flow-components-1';account=out/'api/accounting';account.mkdir(parents=True)
            request={'model':'sam-3.1','endpoint':'/v1/responses','kind':'sam','status':'rejected_before_generation','http_status':413,
                'estimated_cost_usd':'0.000000','automatic_retries':False,'request_id':'meta-original'}
            path=account/'meta-request.json';path.write_text(json.dumps(request));before=path.read_bytes()
            encoding=out/'video-input';encoding.mkdir();video=encoding/'original.mp4';video.write_bytes(b'untouched-original-encoding')
            (encoding/'encoding.json').write_text(json.dumps({'video':video.name,'video_sha256':R.P.sha(video),'video_bytes':video.stat().st_size}))
            secret=root/'.secrets';secret.mkdir();db=sqlite3.connect(secret/'meta-budget.sqlite3')
            db.execute('CREATE TABLE requests(request_id,model,endpoint,kind,state,reason,actual)')
            db.execute('INSERT INTO requests VALUES(?,?,?,?,?,?,?)',('meta-original','sam-3.1','/v1/responses','sam','released','provider_http_rejected_before_generation:413',None));db.commit();db.close()
            state={'status':'failed','phases':[{'name':'segment','status':'failed','attempt':1}]}
            self.assertEqual(R.known_413(state,case)['ledger_state'],'released');self.assertEqual(path.read_bytes(),before)
            db=sqlite3.connect(secret/'meta-budget.sqlite3');db.execute("UPDATE requests SET state='uncertain'");db.commit();db.close()
            with self.assertRaisesRegex(ValueError,'released ledger'):R.known_413(state,case)

    def test_uncertain_http_response_cannot_enter_changed_payload_recovery(self):
        with tempfile.TemporaryDirectory()as tmp,patch.object(R,'ROOT',Path(tmp)):
            case=Path(tmp)/'case';p=case/'segmentation/flow-components-1/api/accounting';p.mkdir(parents=True)
            (p/'meta-request.json').write_text(json.dumps({'model':'sam-3.1','endpoint':'/v1/responses','kind':'sam',
                'status':'transport_error_charge_uncertain','http_status':413,'estimated_cost_usd':'0.000000','automatic_retries':False}))
            with self.assertRaisesRegex(ValueError,'known uncharged HTTP413'):
                R.known_413({'status':'failed','phases':[{'name':'segment','status':'failed','attempt':1}]},case)


if __name__=='__main__':unittest.main()
