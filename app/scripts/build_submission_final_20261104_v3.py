#!/usr/bin/env python3
"""Join the additive same-object45s opening with the unchanged V2 135s tail.

Both video streams are copied without re-encoding. A decoded-frame comparison
checks the complete opening and complete tail. Only audio timestamps/padding are
normalized for a precise180s delivery. No models, UI capture or geometry run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import build_flagship_trailer_20261104_v1 as F

ROOT=F.ROOT;A=F.A
OPENING=ROOT/'showcase/media/flagship-20261104-v2/interaction-twin-flagship-45s.mp4'
TAIL=ROOT/'output/submission-final-20261104-v2/walkthrough-tail.mp4'
V2=ROOT/'showcase/media/submission-final-20261104-v2/provenance.json'
OUT=ROOT/'output/submission-final-20261104-v3'
MEDIA=ROOT/'showcase/media/submission-final-20261104-v3'


def decoded_video_hashes(path):
    raw=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(path),'-map','0:v:0','-an','-f','framemd5','pipe:1'])
    return[line.rsplit(',',1)[1].strip()for line in raw.decode().splitlines()if line and not line.startswith('#')]


def build(opening=OPENING,tail=TAIL,out=OUT,media=MEDIA):
    for path in (opening,tail,V2):
        assert path.resolve()==path and path.is_relative_to(ROOT)and path.is_file(),'Exact existing project inputs required.'
    prior=json.loads(V2.read_text());F.checked(prior['movie'])
    opening_provenance=json.loads((opening.parent/'provenance.json').read_text())
    assert opening_provenance['version']==2 and opening_provenance['movie']==F.ref(opening)
    assert opening_provenance['all_speech_atempo']==1 and opening_provenance['physical_accuracy_verified']is False
    assert opening_provenance['new_API_calls']==opening_provenance['new_GPU_calls']==0
    audio=json.loads(F.checked(prior['audio']).read_text());assert all(v['atempo']==1 for v in audio['voice'])
    assert A.verify_video(opening,{'duration_s':45})and A.verify_video(tail,{'duration_s':135})
    old_media=F.checked(prior['movie']).parent
    first_caption=(opening.parent/'captions.vtt').read_text().rstrip()
    assert(old_media/'captions.vtt').read_text().startswith(first_caption),'Opening speech/captions changed; regenerate the combined captions explicitly.'
    assert(old_media/'transcript.txt').read_text().startswith((opening.parent/'transcript.txt').read_text().rstrip())
    out.mkdir(parents=True,exist_ok=False);media.mkdir(parents=True,exist_ok=False)
    (out/'builder-source.py').write_bytes(Path(__file__).read_bytes())
    listing=out/'concat.txt';listing.write_text(''.join("file '"+str(p).replace("'","'\\''")+"'\n"for p in(opening,tail)))
    joined=out/'concat-copy.mp4';movie=media/'interaction-twin-3min.mp4'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-f','concat','-safe','0','-i',str(listing),'-t','180','-c','copy',str(joined)],check=True)
    streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(joined)]))['streams']
    offset=float(next(s for s in streams if s['codec_type']=='video')['start_time'])
    subprocess.run(['ffmpeg','-nostdin','-v','error','-itsoffset',str(-offset),'-i',str(joined),'-map','0:v','-map','0:a','-c:v','copy','-af','asetpts=PTS-STARTPTS,apad=whole_dur=180,atrim=0:180','-c:a','aac','-b:a','192k','-t','180','-movflags','+faststart',str(movie)],check=True)
    probe=A.verify_video(movie,{'duration_s':180})
    new_frames=decoded_video_hashes(movie);first=decoded_video_hashes(opening);last=decoded_video_hashes(tail)
    assert len(first)==1350 and len(last)==4050 and len(new_frames)==5400
    assert new_frames[:1350]==first,'Opening decoded pixels differ.'
    assert new_frames[1350:]==last,'V2 tail decoded pixels differ.'
    proof={'format':'interaction-twin-submission-stream-copy-proof','version':1,'opening':F.ref(opening),'tail':F.ref(tail),'movie':F.ref(movie),'opening_frames_equal':1350,'tail_frames_equal':4050,'video_reencoded':False,'frame_hashes_sha256':hashlib.sha256('\n'.join(new_frames).encode()).hexdigest(),'all_model_calls':0}
    F.write(out/'decoded-copy-proof.json',proof)
    for name in('captions.vtt','transcript.txt'):(media/name).write_bytes((old_media/name).read_bytes())
    subprocess.run(['ffmpeg','-nostdin','-v','error','-ss','6.7','-i',str(movie),'-frames:v','1','-q:v','2',str(media/'poster.jpg')],check=True)
    provenance={'format':'interaction-twin-final-submission-film','version':3,'duration_s':180,'fps':30,'canvas':[1920,1080],'movie':F.ref(movie),'poster':F.ref(media/'poster.jpg'),'captions':F.ref(media/'captions.vtt'),'transcript':F.ref(media/'transcript.txt'),'opening':F.ref(opening),'opening_provenance':F.ref(opening.parent/'provenance.json'),'unchanged_V2_tail':F.ref(tail),'previous_V2_provenance':F.ref(V2),'opening_video_stream_copied_without_reencoding':True,'tail_video_stream_copied_without_reencoding':True,'decoded_copy_proof':F.ref(out/'decoded-copy-proof.json'),'builder':F.ref(Path(__file__)),'builder_snapshot':F.ref(out/'builder-source.py'),'probe':probe,'all_speech_atempo':1,'new_api_calls':0,'new_gpu_calls':0,'geometry_modified':False,'tail_scope':'Unchanged V2 current-app walkthrough, same-Muse reviewer-role illustration, Meta/custom-code map, saved MCP trace and complete authored robot rehearsal.','publication':False}
    F.write(media/'provenance.json',provenance);F.write(out/'verification.json',provenance)
    print(json.dumps({'movie':F.ref(movie),'copy_proof':F.ref(out/'decoded-copy-proof.json'),'provenance':F.ref(media/'provenance.json')},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--opening',type=Path,default=OPENING);parser.add_argument('--tail',type=Path,default=TAIL);parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--media',type=Path,default=MEDIA);args=parser.parse_args()
    build(args.opening.resolve(),args.tail.resolve(),args.out.resolve(),args.media.resolve())


if __name__=='__main__':main()
