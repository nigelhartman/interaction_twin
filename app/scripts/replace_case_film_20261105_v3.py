#!/usr/bin/env python3
"""Replace only the accepted 14-second case shot in additive local film drafts.

Old films, charging pictures, narration and caption files remain unchanged.
The public source contains this editor, not the separately approved private
captures/receipts that it requires. No inference, upload or publication occurs.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, time
from pathlib import Path
import build_flagship_trailer_20261105_v3 as M

ROOT=M.ROOT; F=M.F; FPS=30
OUT=ROOT/'output/trailer-rework-20261105-v3/films'
OLD45=ROOT/'output/trailer-rework-20261105-v2/films/flagship-story-v1/full-quality-draft-45s.mp4'
OLD180=ROOT/'output/trailer-rework-20261105-v2/films/walkthrough-story-v1/interaction-twin-180s-v4-draft.mp4'
OLD45_SHA='5457fd71e7fb88be16a398e5b82d62896e31513318bcb039fd42e58114d5eed5'
OLD180_SHA='e4c08c436c06c48dcefb0d99d8bb47cea8932aac182d06d5fa73717061db0fa8'
CHARGING_SHA='59138541e2c6faadcfd99e6e9e212af8ab1401594a0eb3896121a72b5de082b8'

def run(args):
    return subprocess.run(args,check=True,capture_output=True,text=True)

def output(path):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT) or path==OUT:raise ValueError('Use a new explicit V3 film output directory.')
    return path

def hashes(path):
    raw=run(['ffmpeg','-nostdin','-v','error','-i',str(path),'-map','0:v:0','-an','-f','framemd5','pipe:1']).stdout
    return [r.rsplit(',',1)[1].strip()for r in raw.splitlines()if r and not r.startswith('#')]

def audio_packets(path):
    raw=run(['ffprobe','-v','error','-select_streams','a:0','-show_packets','-show_entries','packet=data_hash','-show_data_hash','sha256','-of','json',str(path)]).stdout
    return [p['data_hash']for p in json.loads(raw)['packets']]

def split_video(source,folder,cuts,total):
    """Split at existing keyframes; segment muxer avoids B-frame overrun cuts."""
    folder.mkdir(parents=True,exist_ok=False)
    if not cuts or cuts!=sorted(set(cuts)) or any(type(n)is not int or not 0<n<total for n in cuts):raise ValueError('Ordered interior frame boundaries required.')
    run(['ffmpeg','-nostdin','-v','error','-i',str(source),'-map','0:v:0','-an','-c:v','copy','-f','segment','-segment_frames',','.join(map(str,cuts)),'-reset_timestamps','1',str(folder/'part-%02d.mp4')])
    parts=sorted(folder.glob('part-*.mp4'));edges=[0,*cuts,total]
    if len(parts)!=len(edges)-1:raise ValueError('Requested split boundary is not an existing keyframe.')
    for p,a,b in zip(parts,edges,edges[1:]):
        if len(hashes(p))!=b-a:raise ValueError('Keyframe split does not preserve exact frame boundaries.')
    return parts

def join_video(parts,counts,folder,name):
    if len(parts)!=len(counts)or any(n<=0 for n in counts):raise ValueError('One positive frame count per segment required.')
    listing=folder/(name+'.concat.txt')
    listing.write_text(''.join("file '"+str(p).replace("'","'\\''")+f"'\nduration {n/FPS:.12f}\n"for p,n in zip(parts,counts)))
    target=folder/(name+'.mp4')
    run(['ffmpeg','-nostdin','-v','error','-f','concat','-safe','0','-i',str(listing),'-map','0:v:0','-an','-c:v','copy',str(target)])
    return target

def encode_pictures(target,pictures,count):
    process=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','30','-i','pipe:0','-an','-frames:v',str(count),'-c:v','libx264','-threads','4','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        seen=0
        for image in pictures:
            if image.size!=(1920,1080):raise ValueError('Full-canvas pictures required.')
            process.stdin.write(image.convert('RGB').tobytes());seen+=1
        if seen!=count:raise ValueError('Exact editorial frame count required.')
        process.stdin.close();error=process.stderr.read()
        if process.wait()!=0:raise RuntimeError(error.decode()[-1500:])
    except BaseException:
        process.kill();process.wait();raise

def mux_audio(video,old,target,duration):
    run(['ffmpeg','-nostdin','-v','error','-i',str(video),'-i',str(old),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','copy','-t',str(duration),'-movflags','+faststart',str(target)])

def equal_ranges(old,new,ranges):
    if len(old)!=len(new):raise ValueError('Film decoded frame count changed.')
    for a,b in ranges:
        if old[a:b]!=new[a:b]:raise ValueError(f'Copied decoded pictures changed at frames {a}:{b}.')
    return [{'frame_range_start_inclusive_end_exclusive':[a,b],'decoded_pictures_equal':b-a}for a,b in ranges]

def build(folder,case_receipt,freeze,review):
    folder=output(folder)
    if F.ref(OLD45)['sha256']!=OLD45_SHA or F.ref(OLD180)['sha256']!=OLD180_SHA:raise ValueError('Preserve the exact owner-liked V2 films.')
    c=json.loads(case_receipt.read_text());case_ref=F.ref(case_receipt)
    if c.get('version')!=3 or c.get('duration_s')!=14 or c.get('canvas')!=[1920,1080]or c.get('local_validation',{}).get('passed')is not True or c.get('video_ref_requires_local_download_binding')is not False:raise ValueError('Locally bound, independently accepted V3 case required.')
    F.checked(c['video_ref']);F.checked(c['exact_browser_receipt_ref'])
    validation=json.loads(run(['node',str(ROOT/'scripts/validate_case_motion_capture_20261105_v3.cjs'),str(case_receipt)]).stdout)
    frozen=json.loads(freeze.read_text())
    for r in frozen['references']:F.checked(r)
    review_ref=F.ref(review)
    old45_receipt=json.loads((OLD45.parent/'draft-receipt.json').read_text());charging=json.loads(F.checked(old45_receipt['charging_capture']).read_text())
    if charging['video_ref']['sha256']!=CHARGING_SHA:raise ValueError('Accepted USB-C capture must remain unchanged.')
    folder.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    snapshot=folder/'case-replacement-editor-source.py';snapshot.write_bytes(Path(__file__).read_bytes())
    # Decode and encode the new case exactly once; reuse those encoded pictures
    # for both films. Only the proven settled final picture may be held.
    a=object.__new__(M.Assets);a.case_capture=c;a.case_capture_ref=case_ref;a.readers={};a.canonical_capture_frames={}
    def case_pictures():
        for n in range(420):
            im,_=a.case_picture(9+n/FPS);M.scope(im);yield im
    case_video=folder/'accepted-case-14s.mp4'
    try:encode_pictures(case_video,case_pictures(),420)
    finally:a.close()
    # The old source has an IDR at 8 1/3s, rather than9s. Its20 remaining
    # pipeline pictures need one re-encode before the new independently coded
    # case shot. Every other non-case video picture is copied exactly.
    reader=M.A.RawReader(OLD45,250/FPS,(1920,1080));boundary=folder/'unchanged-pipeline-boundary-20f.mp4'
    try:encode_pictures(boundary,(reader.get((250+n)/FPS).copy()for n in range(20)),20)
    finally:reader.close()
    old_parts=split_video(OLD45,folder/'previous45-segments',[250,690],1350)
    silent45=join_video([old_parts[0],boundary,case_video,old_parts[2]],[250,20,420,660],folder,'joined45-video')
    movie45=folder/'interaction-twin-45s-case-v3.mp4';mux_audio(silent45,OLD45,movie45,45)
    long_parts=split_video(OLD180,folder/'previous180-segments',[1350,3690,4110],5400)
    silent180=join_video([silent45,long_parts[1],case_video,long_parts[3]],[1350,2340,420,1290],folder,'joined180-video')
    movie180=folder/'interaction-twin-180s-case-v3.mp4';mux_audio(silent180,OLD180,movie180,180)
    old_short,new_short,old_long,new_long=hashes(OLD45),hashes(movie45),hashes(OLD180),hashes(movie180)
    if len(new_short)!=1350 or len(new_long)!=5400 or new_long[:1350]!=new_short:raise ValueError('Exact45/180s frame clocks and copied opening required.')
    copy45=equal_ranges(old_short,new_short,[(0,250),(690,1350)])
    copy180=equal_ranges(old_long,new_long,[(0,250),(690,3690),(4110,5400)])
    if new_short[270:690]!=new_long[3690:4110]:raise ValueError('Same encoded case pictures must appear in both films.')
    for old,new in[(OLD45,movie45),(OLD180,movie180)]:
        if audio_packets(old)!=audio_packets(new):raise ValueError('Narration/music encoded audio packets changed.')
    for label,old in[('45s',OLD45),('180s',OLD180)]:
        for name in('captions.vtt','transcript.txt'):(folder/(label+'-'+name)).write_bytes((old.parent/name).read_bytes())
    for p,d in[(movie45,45),(movie180,180)]:M.A.verify_video(p,{'duration_s':d})
    F.write(folder/'provenance.json',{'format':'interaction-twin-case-only-film-replacement','version':3,'movies':{'45s':F.ref(movie45),'180s':F.ref(movie180)},'previous_movies':[F.ref(OLD45),F.ref(OLD180)],'case_capture':case_ref,'case_validation':validation,'authored_B_display_proxy':c.get('authored_B_display_proxy'),'source_B_reconstruction_repaired':False,'A_and_case_paths_preserved':c.get('A_and_case_paths_preserved'),'case_implementation_freeze':F.ref(freeze),'independent_case_review':review_ref,'charging_capture_unchanged':old45_receipt['charging_capture'],'charging_video_unchanged_sha256':CHARGING_SHA,'normal150_narration_music_audio_packets_unchanged':True,'captions_and_transcripts_byte_copied':True,'copied_picture_checks':{'45s':copy45,'180s':copy180},'new_case_identical_in_both_films':True,'new45_opening_decoded_pictures_equal_in180':1350,'pipeline_boundary_reencoded_from_unchanged_source_frames':[250,270],'case_replacement_film_s':{'45s':[9,23],'180s':[[9,23],[123,137]]},'normalized_new_capture_frames':a.canonical_capture_frames,'case_only_editorial_change':'Case-only authored V3 display/grasp revision. The exact display derivative and retained original-asset lineage are declared in the bound capture; no native reconstruction repair is claimed.','physical_contact_or_hardware_success_verified':False,'builder':F.ref(Path(__file__)),'builder_snapshot':F.ref(snapshot),'new_API_calls':0,'publication':False,'elapsed_s':time.monotonic()-started})
    print(json.dumps({'movies':[F.ref(movie45),F.ref(movie180)],'provenance':F.ref(folder/'provenance.json')},indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--case',type=Path,required=True);p.add_argument('--freeze',type=Path,required=True);p.add_argument('--review',type=Path,required=True);a=p.parse_args();build(a.output.resolve(),a.case.resolve(),a.freeze.resolve(),a.review.resolve())
if __name__=='__main__':main()
