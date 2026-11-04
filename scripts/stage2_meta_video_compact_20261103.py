#!/usr/bin/env python3
"""Versioned same-coordinate compact SAM video, with explicit lossy provenance.

No resize, crop, new frames, changed source times or manual prompts. The encoded
H264 derivative is smaller, not pixel-identical: decoded hashes and quality
diagnostics disclose that distinction before any explicitly enabled paid call.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stage2_meta_video_segment as S

CRF=18
MAX_VIDEO_BYTES=9*1024*1024
MIN_PSNR_DB=30.


def prepare_video(case,rows,directory,sources):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    width,height=rows[0]['width'],rows[0]['height'];video=directory/'sampled-compact-1fps.mp4';records=[]
    with tempfile.TemporaryDirectory(prefix='compact-encoding-',dir=directory)as temp:
        work=Path(temp)
        for n,(row,source)in enumerate(zip(rows,sources)):
            path=S.confined(case/row['file'],case)
            with Image.open(path)as im:rgb=im.convert('RGB');pixels=np.asarray(rgb);rgb.save(work/f'{n:06d}.png')
            records.append({'submitted_frame_index':n,'submitted_timestamp_s':float(n),'source':source,
                'source_image_file':row['file'],'source_image_sha256':S.sha(path),
                'source_rgb_pixels_sha256':hashlib.sha256(pixels.tobytes()).hexdigest()})
        reference,decoded=work/'reference.yuv',work/'decoded.yuv'
        S.command(['ffmpeg','-hide_banner','-loglevel','error','-framerate','1','-start_number','0','-i',work/'%06d.png',
            '-frames:v',len(rows),'-pix_fmt','yuv420p','-f','rawvideo',reference])
        S.command(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pixel_format','yuv420p',
            '-video_size',f'{width}x{height}','-framerate','1','-i',reference,'-frames:v',len(rows),'-an',
            '-c:v','libx264','-preset','veryfast','-crf',str(CRF),'-pix_fmt','yuv420p','-movflags','+faststart',video])
        if not 0<video.stat().st_size<=MAX_VIDEO_BYTES:raise ValueError('Compact video exceeds9MiB; no paid request allowed')
        probe=json.loads(S.command(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_streams',
            '-show_frames','-show_entries','stream=codec_name,width,height,nb_read_frames,avg_frame_rate:frame=best_effort_timestamp_time','-of','json',video]).stdout)
        streams,frames=probe.get('streams',[]),probe.get('frames',[])
        if (len(streams)!=1 or streams[0].get('codec_name')!='h264'or(streams[0]['width'],streams[0]['height'])!=(width,height)
            or int(streams[0].get('nb_read_frames',-1))!=len(rows)or len(frames)!=len(rows)or streams[0].get('avg_frame_rate')!='1/1'):
            raise ValueError('Compact codec, dimensions, frame count or frame rate differs')
        if any(abs(float(row.get('best_effort_timestamp_time',-1))-n)>1e-6 for n,row in enumerate(frames)):
            raise ValueError('Compact frame order/times differ')
        S.command(['ffmpeg','-hide_banner','-loglevel','error','-i',video,'-frames:v',len(rows),'-pix_fmt','yuv420p','-f','rawvideo',decoded])
        frame_bytes=width*height*3//2
        if reference.stat().st_size!=len(rows)*frame_bytes or decoded.stat().st_size!=reference.stat().st_size:raise ValueError('Compact decoded plane count differs')
        with reference.open('rb')as a,decoded.open('rb')as b:
            for row in records:
                expected,actual=a.read(frame_bytes),b.read(frame_bytes)
                error=np.frombuffer(expected,np.uint8).astype(np.float64)-np.frombuffer(actual,np.uint8).astype(np.float64)
                mse=float(np.mean(error*error));psnr=None if mse==0 else 10*math.log10(255*255/mse)
                if psnr is not None and psnr<MIN_PSNR_DB:raise ValueError('Compact decoded quality is below the reviewed diagnostic floor; no paid call')
                row.update(reference_yuv420_pixels_sha256=hashlib.sha256(expected).hexdigest(),
                    decoded_yuv420_pixels_sha256=hashlib.sha256(actual).hexdigest(),mean_absolute_yuv_error=float(np.mean(np.abs(error))),
                    psnr_yuv_db=psnr,decoded_pixels_equal_reference=expected==actual)
    manifest={'format':'interaction-twin-compact-sam-video','version':1,
        'status':'verified_order_dimensions_and_disclosed_lossy_compression','video':video.name,'video_sha256':S.sha(video),
        'video_bytes':video.stat().st_size,'codec':'h264','codec_settings':f'libx264 veryfast crf={CRF} yuv420p',
        'fps':1,'width':width,'height':height,'frame_count':len(rows),'frames':records,
        'original_pts_replaced_by_encoded_pts':False,'resize_performed':False,'crop_performed':False,
        'lossy_compression_performed':True,'lossless_pixels_claimed':False,'minimum_per_frame_psnr_yuv_db':MIN_PSNR_DB,
        'maximum_inline_video_bytes':MAX_VIDEO_BYTES,'approximate_base64_bytes':4*((video.stat().st_size+2)//3),
        'source_mapping':'Each submitted frame N maps to exactly frames[N].source original PTS. No new observations or interpolation.',
        'colour_provenance':f'Saved RGB converted toYUV420, then H264 CRF{CRF} lossy compressed. Decoded per-frame hashes and errors recorded; pixels are not asserted identical.',
        'implementation_sha256':S.sha(__file__),'mask_accuracy_verified':False}
    S.write(directory/'encoding.json',manifest)
    return video,manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run',required=True);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--allow-paid-api',action='store_true');args=parser.parse_args()
    if not S.SLUG.fullmatch(args.case)or not S.SLUG.fullmatch(args.run):raise ValueError('Bounded case/run required')
    case=ROOT/'data'/args.case
    if args.prepare_only:
        rows_doc=S.load(case/'frames/index.json');rows=rows_doc['frames'];manifest=S.load(case/'manifest.json')
        S.validate_config(S.load(args.config),rows);_,_,sources=S.verify_rows(case,rows,rows_doc,manifest)
        video,encoding=prepare_video(case,rows,case/'segmentation'/args.run/'video-input',sources)
        print(json.dumps({'prepared':True,'video_bytes':video.stat().st_size,'frame_count':len(rows),'encoding':str(video.parent/'encoding.json')}));return
    if not args.allow_paid_api:raise ValueError('Paid compact segmentation requires explicit --allow-paid-api')
    from overnight_budget import OvernightMetaClient
    result=S.run_case(case,args.config,args.run,client=OvernightMetaClient(root=ROOT),video_preparer=prepare_video)
    S.write(case/'segmentation'/args.run/'compact-adapter.json',{'format':'interaction-twin-compact-sam-adapter','version':1,
        'implementation_sha256':S.sha(__file__),'base_stage_sha256':S.sha(S.__file__),'pixels_losslessly_preserved':False,
        'coordinates_and_sample_times_unchanged':True,'budget_guard':'fixed overnight20 +priorincrement +global100 +perrequest',
        'automatic_retry':False})
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
