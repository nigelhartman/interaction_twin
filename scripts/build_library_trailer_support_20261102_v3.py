#!/usr/bin/env python3
"""Create local captions, poster and review page for the genuine 58s trailer."""
import argparse
import json
from pathlib import Path
import shutil
import textwrap

from build_library_trailer_20261102_v3 import CHAPTERS, OUT, ROOT, PREFIX, assets, frame, record, fresh


def timestamp(value, comma=False):
    milliseconds = round(value * 1000)
    h, milliseconds = divmod(milliseconds, 3600000)
    m, milliseconds = divmod(milliseconds, 60000)
    s, milliseconds = divmod(milliseconds, 1000)
    return f'{h:02}:{m:02}:{s:02}' + (',' if comma else '.') + f'{milliseconds:03}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--take', default='v3')
    args = parser.parse_args()
    if not args.take.replace('-', '').isalnum():
        raise ValueError('Simple versioned take required')
    name = PREFIX + '-' + args.take
    directory = OUT / args.take
    video = ROOT / 'showcase/media' / (name + '.mp4')
    if not video.is_file() or not (directory / 'verification.json').is_file():
        raise ValueError('Verify the actual encoded film first')
    vtt, srt, rows, at = ['WEBVTT', ''], [], [], 0
    for j, chapter in enumerate(CHAPTERS):
        # Captions are the exact local spoken script, held through each chapter.
        start, end = at + .35, at + chapter['seconds'] - .25
        caption = '\n'.join(textwrap.wrap(chapter['narration'], width=60))
        vtt += [str(j + 1), timestamp(start) + ' --> ' + timestamp(end), caption, '']
        srt += [str(j + 1), timestamp(start, True) + ' --> ' + timestamp(end, True), caption, '']
        rows.append(dict(start_s=at, duration_s=chapter['seconds'], title=chapter['title'], narration=chapter['narration']))
        at += chapter['seconds']
    vtt_path = ROOT / 'showcase/media' / (name + '.vtt')
    srt_path = ROOT / 'showcase/media' / (name + '.srt')
    with vtt_path.open('x') as stream: stream.write('\n'.join(vtt))
    with srt_path.open('x') as stream: stream.write('\n'.join(srt))
    a = assets()
    poster = frame(3, .03, a)
    poster_path = ROOT / 'showcase/media' / (name + '-designed-poster.jpg')
    poster.save(poster_path, quality=97)
    preview = directory / 'preview'
    preview.mkdir()
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Interaction Twin · 58-second film</title><style>
:root{font-family:Arial,sans-serif;color:#152d24;background:#f4f2e9}*{box-sizing:border-box}body{margin:0}main{max-width:1250px;margin:auto;padding:42px 24px 80px}a{color:inherit;text-underline-offset:4px}header{display:flex;align-items:center;justify-content:space-between;gap:20px;margin-bottom:44px}.brand{font-size:17px;font-weight:800;letter-spacing:.06em;text-decoration:none}.brand span{background:#152d24;color:#dcf3a1;display:inline-grid;place-items:center;width:35px;height:35px;border-radius:10px;margin-right:12px}small{letter-spacing:.12em;font-size:11px;font-weight:700}h1{font-size:clamp(34px,6vw,66px);letter-spacing:-.055em;line-height:1.02;margin:16px 0 22px}p{color:#607065;line-height:1.6;font-size:17px;max-width:760px}video{display:block;width:100%;border-radius:22px;background:#152d24;margin-top:32px;box-shadow:0 20px 65px #152d241a}nav{display:flex;flex-wrap:wrap;gap:10px;margin:24px 0}nav a{border:1px solid #ccd6c7;padding:14px 18px;border-radius:12px;min-height:46px;text-decoration:none;font-size:14px;font-weight:700}.primary{background:#152d24;color:#dcf3a1}details{margin:26px 0;border-top:1px solid #d7dfd3;padding-top:24px}summary{cursor:pointer;min-height:44px;font-weight:700}li{margin:14px 0;color:#607065;line-height:1.5}footer{border-top:1px solid #d7dfd3;padding-top:24px;margin-top:44px;color:#607065;font-size:13px}@media(max-width:600px){main{padding:24px 18px 45px}header{margin-bottom:32px}header>a:last-child{display:none}video{border-radius:12px}nav{display:grid;grid-template-columns:1fr 1fr}nav a{display:flex;align-items:center;justify-content:center;text-align:center;font-size:13px;padding:13px 10px;min-height:48px}h1{letter-spacing:-.04em}}</style>
<main><header><a class="brand" href="/showcase/index.html"><span>↔</span>INTERACTION TWIN</a><a href="/showcase/library.html">Explore the generated library ↗</a></header>
<small>58 SECONDS · ACTUAL SAVED RESULTS · LOCAL REVIEW</small><h1>From a video.<br>To a robot task draft.</h1>
<p>Reuse a demonstration as parts, actions and source-linked evidence. An inspectable starting point for robot task authoring, with a read-only MCP interface for agents.</p>
<video controls playsinline preload="metadata" poster="/showcase/media/NAME-designed-poster.jpg"><source src="/showcase/media/NAME.mp4" type="video/mp4"><track kind="captions" label="English" srclang="en" src="/showcase/media/NAME.vtt"></video>
<nav><a class="primary" href="/showcase/library.html">Explore the library</a><a href="/files/data/sam3d-native-review-20261101/earbud-a-v2/index.html">Inspect the new SAM 3D part</a><a href="/showcase/media/NAME.mp4" download>Download MP4</a><a href="/showcase/media/NAME.srt" download>Download captions</a></nav>
<details><summary>What is real, and what remains unverified?</summary><ul><li>Actual user recording, official Meta SAM 3.1 masks, saved Muse proposals, original TripoSR drafts and the first genuine SAM 3D earbud. Its GPU execution, all returned bytes and unchanged native import were independently reviewed.</li><li>The MCP sequence is an actual saved 36-call local scripted run, not a new live LLM run.</li><li>Robot movement and anchors are an illustrative authoring preview. Physical scale, accurate geometry, common assembly, contact and simulation readiness remain unverified.</li><li>One SAM 3D earbud is shown as an unselected candidate. Other parts keep TripoSR drafts. The source projection uses raw model layout and a camera estimated from the inferred pointmap: no silhouette or pose fitting, and no validated temporal tracking. <a href="/files/data/sam3d-source-projection-20261102/review-v1/index.html">Inspect the projection diagnostic ↗</a></li><li>Local synthetic narration and original procedural instrumental music. Editing this film triggered no additional model/API requests. The film has not been publicly published or submitted.</li></ul></details>
<footer>Working research prototype · editorial camera orbits and chapter transitions · source evidence preserved · estimated playback does not change exports</footer></main></html>'''.replace('NAME', name)
    (preview / 'index.html').write_text(html)
    shutil.copyfile(preview / 'index.html', ROOT / 'showcase/film.html')
    fresh(directory / 'support.json', dict(video=record(video, 'verified_actual_film'), poster=record(poster_path, 'editorial_poster_using_genuine_source_baseline_SAM3D_comparison'),
        captions=[record(vtt_path, 'WebVTT_exact_narration'), record(srt_path, 'SRT_exact_narration')],
        preview=record(preview / 'index.html', 'local_review_page'), storyboard=rows, published=False))
    print(json.dumps({'preview': str(preview / 'index.html'), 'poster': str(poster_path)}))


if __name__ == '__main__': main()
