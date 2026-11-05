"""Bind exact UI-downloaded WebM + receipt; no browser, network or rendering."""
from __future__ import annotations
import argparse, hashlib, json, math, shutil, statistics, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/charging-rehearsal-20261105-v1'

def ref(p: Path):
    raw=p.read_bytes()
    return {'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}

def decoded_clock(frames):
    if not isinstance(frames,list) or not 325<=len(frames)<=2000:raise ValueError('A complete bounded decoded frame ledger is required.')
    timestamps=[float(f['best_effort_timestamp_time']) for f in frames]
    steps=[b-a for a,b in zip(timestamps,timestamps[1:])]
    if not all(map(math.isfinite,timestamps)) or not -.001<=timestamps[0]<=.12 or min(steps)<=0 or max(steps)>.16:raise ValueError('Continuous monotonic decoded timestamps with no excessive gap are required.')
    duration=timestamps[-1]+statistics.median(steps)
    if not 12.8<=duration<=13.3:raise ValueError('Decoded capture duration is outside the complete 13-second task tolerance.')
    return duration,timestamps

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--video',type=Path,required=True);ap.add_argument('--receipt',type=Path,required=True);a=ap.parse_args()
    for p,limit in [(a.video,64*1024*1024),(a.receipt,4*1024*1024)]:
        if not p.is_file() or p.is_symlink() or not 0<p.stat().st_size<=limit:raise ValueError('A bounded regular UI-downloaded capture is required.')
    d=json.loads(a.receipt.read_text());raw=a.video.read_bytes()
    if d['video_ref']!={'path':'output/charging-rehearsal-20261105-v1/charging-rehearsal-v1.webm','sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}:raise ValueError('Downloaded clip differs from its browser receipt.')
    validation=json.loads(subprocess.run(['node',str(ROOT/'scripts/validate_charging_capture_20261105_v1.cjs'),str(a.receipt)],check=True,capture_output=True,text=True,timeout=30).stdout)
    probe=json.loads(subprocess.run(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,duration','-of','json',str(a.video)],check=True,capture_output=True,text=True,timeout=30).stdout)
    streams=probe['streams'];v=[s for s in streams if s.get('codec_type')=='video']
    if len(v)!=1 or v[0].get('width')!=1920 or v[0].get('height')!=1080 or any(s.get('codec_type')=='audio' for s in streams):raise ValueError('Full-size silent canvas recording required.')
    frames=json.loads(subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time,pkt_duration_time','-of','json',str(a.video)],check=True,capture_output=True,text=True,timeout=30).stdout)['frames']
    duration,timestamps=decoded_clock(frames)
    OUT.mkdir(exist_ok=True,parents=True);video=OUT/'charging-rehearsal-v1.webm';original=OUT/'charging-rehearsal-v1.browser-receipt.json';final=OUT/'charging-rehearsal-v1.json'
    for source,target in [(a.video,video),(a.receipt,original)]:
        if target.exists() and target.read_bytes()!=source.read_bytes():raise FileExistsError('Refusing to overwrite a different prior capture.')
        if source.resolve()!=target.resolve():shutil.copyfile(source,target)
    d.update(video_ref=ref(video),video_ref_requires_local_download_binding=False,exact_browser_receipt_ref=ref(original),local_validation=validation,decoded_video_probe=probe,decoded_duration_s=duration,decoded_frame_count=len(timestamps),decoded_first_timestamp_s=timestamps[0],decoded_last_timestamp_s=timestamps[-1],decoded_maximum_frame_gap_s=max(b-a for a,b in zip(timestamps,timestamps[1:])),decoded_duration_method='last decoded frame timestamp plus median frame step; no WebM metadata duration assumed',actual_browser_visual_review_required=True)
    if final.exists():raise FileExistsError('Preserve the prior bound receipt.')
    final.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'capture':ref(video),'receipt':ref(final),'validation':validation},indent=2))

if __name__=='__main__':main()
