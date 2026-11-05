"""Bind the exact local UI-downloaded case rehearsal. No API or rendering."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/trailer-rework-20261105-v2/case-grasp'
def ref(p):
    raw=p.read_bytes();return{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--video',required=True,type=Path);ap.add_argument('--receipt',required=True,type=Path);a=ap.parse_args()
    for p,limit in[(a.video,64*1024*1024),(a.receipt,16*1024*1024)]:
        if not p.is_file()or p.is_symlink()or not 0<p.stat().st_size<=limit:raise ValueError('Bounded regular browser downloads required.')
    d=json.loads(a.receipt.read_text());raw=a.video.read_bytes()
    expected={'path':'output/trailer-rework-20261105-v2/case-grasp/fullscreen-case-grasp-14s.webm','sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    if d['video_ref']!=expected:raise ValueError('Browser video and receipt differ.')
    check=json.loads(subprocess.run(['node',str(ROOT/'scripts/validate_case_motion_capture_20261105_v2.cjs'),str(a.receipt)],check=True,capture_output=True,text=True).stdout)
    probe=json.loads(subprocess.run(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,duration','-of','json',str(a.video)],check=True,capture_output=True,text=True).stdout)
    frames=json.loads(subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time,pkt_duration_time','-of','json',str(a.video)],check=True,capture_output=True,text=True).stdout)['frames']
    timestamps=[float(f['best_effort_timestamp_time'])for f in frames]
    if len(timestamps)<350 or timestamps[0]>.12 or any(b<=a for a,b in zip(timestamps,timestamps[1:])):raise ValueError('Continuous decoded frame timestamps required.')
    import statistics
    frame_step=statistics.median(b-a for a,b in zip(timestamps,timestamps[1:]));duration=timestamps[-1]+frame_step
    if max(b-a for a,b in zip(timestamps,timestamps[1:]))>.16:raise ValueError('Decoded video has an excessive frame gap.')
    video=[s for s in probe['streams']if s['codec_type']=='video']
    if len(video)!=1 or(video[0]['width'],video[0]['height'])!=(1920,1080)or any(s['codec_type']=='audio'for s in probe['streams'])or not 13.8<=duration<=14.3:raise ValueError('Full-size silent complete14s recording required.')
    OUT.mkdir(parents=True,exist_ok=True);movie=OUT/'fullscreen-case-grasp-14s.webm';original=OUT/'fullscreen-case-grasp-14s.browser-receipt.json';receipt=OUT/'fullscreen-case-grasp-14s.json'
    for source,target in[(a.video,movie),(a.receipt,original)]:
        if target.exists()and target.read_bytes()!=source.read_bytes():raise FileExistsError('Preserve the prior capture.')
        if target.resolve()!=source.resolve():shutil.copyfile(source,target)
    d.update(exact_browser_receipt_ref=ref(original),video_ref=ref(movie),local_validation=check,decoded_video_probe=probe,decoded_duration_s=duration,decoded_frame_count=len(timestamps),decoded_first_timestamp_s=timestamps[0],decoded_last_timestamp_s=timestamps[-1],decoded_duration_method='last decoded frame timestamp plus median frame step; no WebM metadata duration assumed',video_ref_requires_local_download_binding=False)
    if receipt.exists():raise FileExistsError('Preserve the prior bound receipt.')
    receipt.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'video':ref(movie),'receipt':ref(receipt),'validation':check},indent=2))
if __name__=='__main__':main()
