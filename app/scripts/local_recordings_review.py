#!/usr/bin/env python3
"""Build a private local diagnostic for two recordings and sampled perception.

All JSON is embedded to work from file:// without fetch, a server or external
dependencies. Existing videos, masks and metadata are read without modification.
The output holds the preceding sampled overlay and explicitly reports its age;
it does not fabricate intermediate masks or claim native-rate segmentation.
"""
import argparse
import json
import math
import os
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
LATEST_REVIEWS = [
    ("jabra-local-v2/review/index.html",
     "Improved Jabra · three-component 3D scene", "14 simultaneous source observations. Fixed generated meshes; conditional depth and pose estimates."),
    ("jabra-local-v2/approach-v1/index.html",
     "Approach to the case · measurements and video", "61 earbud samples, 24 reliable paired 2D observations. Final seating remains unknown."),
    ("jabra-local-v2/occlusion-v1/review/index.html",
     "Hands and occlusion · source evidence", "100 local SAM predictions, 16 sampled audits and eight model-proposed interaction windows."),
    ("jabra-local-v2/procedure-v1/index.html",
     "Stage 5 · updated component and procedure replay", "Seven saved semantic steps now reference the stronger motion evidence. Reinitialization stays labelled."),
]
SPECS = [
    ("recording-144929-1fps-v3", "Jabra earbuds & charging case", "Earbuds taken out and put back",
     [("reconstruction-triposr-earbud-v1/review/index.html", "Earbud · local TripoSR experiment"),
      ("reconstruction-triposr-case-v1/review/index.html", "Case · local TripoSR experiment"),
      ("alignment/generated-earbud-v1/index.html", "Earbud · automatic source-frame fit"),
      ("alignment/generated-earbud-motion-v1/observer.html", "Earbud · motion projection evidence"),
      ("alignment/generated-earbud-motion-v1/observer-3d.html", "Earbud · video and orbitable 3D motion")]),
    ("recording-145141-1fps-v3", "Wallet & coin", "Coin taken out and put back",
     [("reconstruction-triposr-round-piece-v1/review/index.html", "Round piece · local TripoSR experiment"),
      ("alignment/generated-coin-v1/index.html", "Coin · automatic source-frame fit")]),
]


def read(path):
    return json.loads(path.read_text())


def latest_reviews(out_dir):
    """Link only existing private artifacts contained in this project's data."""
    data_root = (ROOT / "data").resolve()
    rows, references = [], []
    for relative, title, description in LATEST_REVIEWS:
        target = (data_root / relative).resolve()
        if not target.is_relative_to(data_root):
            raise ValueError("Review path escapes project data")
        if not target.is_file():
            continue
        references.append(target)
        rows.append({"title": title, "description": description,
                     "href": quote(os.path.relpath(target, out_dir), safe="/")})
    return rows, references


def case_payload(case, title, subtitle, reconstruction_specs, out_dir):
    base = ROOT / "data" / case
    references = []

    def href(relative):
        target = (base / relative).resolve()
        if not target.is_relative_to(base.resolve()) or not target.is_file():
            raise ValueError(f"Missing or invalid local reference: {case}/{relative}")
        references.append(target)
        return quote(os.path.relpath(target, out_dir), safe="/")

    manifest = read(base / "manifest.json")
    frame_index = read(base / "frames/index.json")
    summary = read(base / "perception-summary.json")
    if frame_index.get("time_domain") != "source_video_presentation_seconds" or summary.get("source_pts_preserved") is not True:
        raise ValueError(f"{case} does not establish original source PTS")
    segmentation = "segmentation/automatic-components-v1"
    quality = read(base / segmentation / "quality.json")
    annotations = read(base / segmentation / "annotations.json")
    samples = read(base / segmentation / "frames.json")["frames"]
    sample_tracks = {sample["sample_index"]: sample["tracks"] for sample in samples}
    audit = read(base / summary["mask_audit"]["record"])
    audit_lookup = {(row["sample_index"], row["track_id"]): row for row in audit["result"]["observations"]}
    frames = []
    previous = -math.inf
    for frame in frame_index["frames"]:
        timestamp = frame["timestamp_s"]
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp <= previous:
            raise ValueError("Frame PTS must be finite, strictly increasing numbers")
        previous = timestamp
        index = frame["index"]
        tracks = []
        for track in sample_tracks.get(index, []):
            tracks.append({"id": track["track_id"], "status": track["status"],
                           "reasons": track.get("reasons", []),
                           "audit": audit_lookup.get((index, track["track_id"]))})
        frames.append({"sample": index, "timestamp": timestamp,
                       "overlay": href(f"{segmentation}/overlays/{index:06d}.jpg"),
                       "source_frame": href(frame["file"]), "tracks": tracks})
    tracks = []
    ranked_path = base / "automatic-reconstruction-inputs-v1/index.json"
    ranked = read(ranked_path)["tracks"] if ranked_path.exists() else []
    rankings = {row["track_id"]: row for row in ranked}
    for track in annotations["tracks"]:
        track_id = track["id"]
        entry = {"id": track_id, "label": track["label"], "description": track.get("description", ""),
                 "color": track.get("color", "#9ca3af"), "range": track["sample_range"],
                 "uncertainties": track.get("uncertainties", []),
                 "withheld": quality["tracks"].get(track_id, {}).get("withheld_samples", [])}
        if track_id in rankings:
            row = rankings[track_id]
            entry["ranking"] = {"href": href(row["selection"]), "timestamp": row["top_timestamp_s"],
                                "sample": row["top_sample"], "count": row["candidate_count"]}
        tracks.append(entry)
    pts = {frame["sample"]: frame["timestamp"] for frame in frames}
    issues = []
    for row in audit["result"]["observations"]:
        if row["verdict"] != "plausible_visible_mask":
            issues.append({**row, "timestamp": pts[row["sample_index"]]})
    links = [("Perception summary", "perception-summary.json"),
             ("Model discovery + seed proposals", summary["discovery"]["record"]),
             ("Raw visual audit", summary["mask_audit"]["record"]),
             ("Geometric gate / withheld masks", f"{segmentation}/quality.json"),
             ("Original PTS frame index", "frames/index.json"),
             ("Normalization provenance", "normalization.json")]
    if ranked_path.exists():
        links.append(("Ranked reconstruction inputs", "automatic-reconstruction-inputs-v1/index.json"))
    reconstructions = [{"title": label, "href": href(relative)} for relative, label in reconstruction_specs
                       if (base / relative).is_file()]
    result = {
        "id": case, "title": title, "subtitle": subtitle,
        "duration": manifest["source"]["video"]["duration_s"],
        "video": href(manifest["source"]["file"]),
        "frames": frames, "tracks": tracks, "issues": issues,
        "stats": {"frames": len(frames), "raw_masks": summary["segmentation"]["raw_mask_count"],
                  "tracks": len(tracks), "checked": summary["mask_audit"]["checked_observations"],
                  "plausible": summary["mask_audit"]["plausible_candidates"],
                  "rejected": summary["mask_audit"]["rejected_candidates"],
                  "uncertain": summary["mask_audit"]["uncertain_candidates"]},
        "discovery_summary": read(base / summary["discovery"]["record"])["result"]["summary"],
        "models": {"segmentation": summary["segmentation"]["model"],
                   "device": summary["segmentation"]["device"],
                   "reasoning": summary["discovery"]["model"],
                   "reasoning_location": summary["discovery"]["inference_location"]},
        "no_manual_seeds": summary["discovery"]["manual_seed_boxes_used"] is False,
        "no_manual_corrections": summary["discovery"]["manual_corrections_used"] is False,
        "links": [{"title": label, "href": href(relative)} for label, relative in links],
        "reconstructions": reconstructions,
    }
    return result, references


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Your recordings · perception review</title><style>
:root{color-scheme:dark;--bg:#10141b;--card:#19212c;--line:#303d4d;--text:#eef3fa;--muted:#a6b4c6;--accent:#80d6cd;--bad:#ff939d;--warn:#f5c476}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,sans-serif}main{max-width:1480px;margin:auto;padding:32px}h1{margin:4px 0 8px;font-size:34px;letter-spacing:-1px}h2{font-size:19px;margin:0 0 10px}h3{font-size:15px;margin:0 0 8px}p{margin:0 0 12px}.eyebrow{color:var(--accent);font-size:12px;letter-spacing:1.5px;text-transform:uppercase}.muted,small{color:var(--muted)}button,a,input{font:inherit}button{cursor:pointer;border:1px solid var(--line);background:#202b39;color:var(--text);border-radius:9px;padding:8px 13px}button:hover{border-color:var(--accent)}button:focus-visible,a:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:3px}button.active{background:#214c4d;border-color:var(--accent)}a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}nav{display:flex;gap:10px;margin:23px 0}.card{background:var(--card);border:1px solid var(--line);border-radius:15px;padding:20px}.layout{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:18px;align-items:start}.players{display:grid;grid-template-columns:1fr 1fr;gap:16px}.media{width:100%;height:min(59vh,650px);background:#080c12;border-radius:10px;object-fit:contain;display:block}.caption{margin:9px 0;font-size:13px}.pill{display:inline-block;font-size:11px;background:#27313f;border:1px solid var(--line);padding:2px 7px;border-radius:5px;margin-left:6px}.timing{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:16px 0 10px}.timing>div{background:#111a25;padding:10px;border-radius:8px}.timing strong{display:block;font-variant-numeric:tabular-nums;font-size:19px}.timing span{font-size:11px;color:var(--muted)}.controls{display:flex;gap:8px;align-items:center;margin:10px 0}.controls input{flex:1;min-width:60px;accent-color:var(--accent)}.notice{font-size:13px;color:var(--warn);border-left:3px solid var(--warn);padding-left:11px;margin:13px 0 0}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:15px 0}.stat{background:#111a25;padding:10px;border-radius:9px}.stat strong{font-size:23px;display:block}.stat span{font-size:11px;color:var(--muted)}.good{color:var(--accent)}.bad{color:var(--bad)}.warn{color:var(--warn)}.track{padding:12px 0;border-top:1px solid var(--line)}.track:first-child{border-top:0}.trackhead{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.dot{height:9px;width:9px;border-radius:50%;display:inline-block;flex-shrink:0}code{font-size:12px;overflow-wrap:anywhere}.track p{font-size:12px;color:var(--muted);margin:5px 0}.status{font-size:12px;margin:5px 0}.detailsgrid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}.issues button{padding:4px 8px;font-size:12px;margin-right:7px}.issue{border-top:1px solid var(--line);padding:11px 0;font-size:13px}.issue:first-child{border:0}details{margin:12px 0}summary{cursor:pointer;color:var(--text)}.links{display:flex;flex-direction:column;gap:8px;font-size:13px}.links a{overflow-wrap:anywhere}.modelrow{margin-bottom:13px;font-size:13px}.modelrow strong{display:block;margin-bottom:3px}.recon{margin-top:15px;border-top:1px solid var(--line);padding-top:15px}.recon a{display:block;margin:8px 0}footer{font-size:12px;color:var(--muted);margin-top:22px}ul{padding-left:18px}li{margin:4px 0}.empty{color:var(--muted);font-size:13px}.samplelink{font-size:12px;padding:3px 7px}#currentTracks{max-height:370px;overflow:auto}.mediaerror{color:var(--bad);font-size:13px;margin:8px 0}#overlayError{display:none}@media(max-width:1100px){.layout{grid-template-columns:1fr}.sidebar{display:grid;grid-template-columns:1fr 1fr;gap:18px}.sidebar .card{margin:0!important}.media{height:58vh}}@media(max-width:650px){main{padding:17px}h1{font-size:27px}.players,.detailsgrid,.sidebar{grid-template-columns:1fr}.media{height:55vh}.card{padding:15px}.timing{gap:5px}.timing strong{font-size:16px}nav{flex-wrap:wrap}.controls{flex-wrap:wrap}}
</style></head><body><main>
<div class="eyebrow">Interaction Twin · private diagnostic</div><h1>Your two recordings</h1>
<p class="muted">Model-discovered components and real SAM masks, with failures kept visible.</p>
<section class="card" id="latestReviews" style="margin-top:20px" aria-label="Latest experiments"><h2>Latest Jabra experiments</h2><div id="latestLinks" class="detailsgrid" style="margin-top:8px"></div><p class="muted" style="font-size:12px;margin-top:14px">The 1 fps perception review below is the preserved baseline. The links above open the newer tracking, assembly and procedure experiments.</p></section>
<nav id="tabs" aria-label="Choose recording"></nav>
<div class="layout"><section class="card">
<h2 id="caseTitle"></h2><p id="caseSummary" class="muted"></p>
<div class="players"><div><h3>Source video <span class="pill">original timing</span></h3><video id="source" class="media" controls playsinline preload="metadata"></video><div class="caption muted">Upright SDR derivative; original presentation timestamps retained.</div><p id="videoError" class="mediaerror" hidden></p></div>
<div><h3>Sampled SAM overlay <span class="pill">~1 fps</span></h3><img id="overlay" class="media" alt="SAM component masks at the preceding sampled source frame"><div class="caption" id="sampleLabel"></div><p id="overlayError" class="mediaerror">Sample image could not be loaded.</p></div></div>
<div class="timing"><div><strong id="videoTime">0.000 s</strong><span>SOURCE TIME</span></div><div><strong id="sampleTime">—</strong><span>MASK SAMPLE TIME</span></div><div><strong id="sampleAge">—</strong><span>SAMPLE AGE</span></div></div>
<div class="controls"><button id="previous" aria-label="Previous sampled frame">← Sample</button><input id="sampleSlider" type="range" min="0" max="1" step="1" value="0" aria-label="Choose sampled mask"><button id="next" aria-label="Next sampled frame">Sample →</button><button id="atSample">Pause at sample</button></div>
<p class="notice">The right image holds the preceding real sample. Masks are not interpolated or run at the video's native frame rate. Its age shows the time difference.</p>
</section><aside class="sidebar"><section class="card"><h2>What the models found</h2><p class="muted" id="counts"></p><div class="stats" id="stats"></div><p class="muted" style="font-size:12px">Counts are model plausibility judgments on selected observations, not measured mask accuracy. Unchecked observations remain unaudited.</p><h3>At the sampled frame</h3><div id="currentTracks"></div></section>
<section class="card" style="margin-top:18px"><h2>Models & provenance</h2><div id="models"></div><p class="muted" style="font-size:12px">Discovery supplies candidate seed boxes; SAM propagates them. No object-specific manual boxes or corrections were used for these runs. Physical identity through occlusion remains unverified.</p><div class="recon" id="reconstructions"></div></section></aside></div>
<div class="detailsgrid"><section class="card"><h2>Candidate component IDs</h2><p class="muted" style="font-size:13px">These identify observed tracklets, not confirmed physical identities.</p><div id="allTracks"></div></section>
<section class="card"><h2>Failures & uncertain observations</h2><p class="muted" style="font-size:13px">Click a timestamp to inspect the evidence. Rejected masks remain in the raw review; they are not accepted geometry inputs.</p><div id="issues" class="issues"></div><details><summary>Raw records & ranked inputs</summary><div id="links" class="links" style="margin-top:12px"></div></details></section></div>
<footer>Local review of approved perception, geometry and alignment experiments. No simulated seating, hidden contact or usable 3D pose is established by these masks. No SAM 3.1, SAM 3D Objects or Meta reasoning API inference is represented here. Recordings remain private local files.</footer>
</main><script id="reviewData" type="application/json">__PAYLOAD__</script><script id="latestData" type="application/json">__LATEST__</script><script>
'use strict';
const cases=JSON.parse(document.getElementById('reviewData').textContent);
const el=id=>document.getElementById(id),video=el('source'),overlay=el('overlay');
let active=null,activeIndex=-1,lastFrame=-2;
function node(tag,text,className){const result=document.createElement(tag);if(text!==undefined)result.textContent=text;if(className)result.className=className;return result;}
function link(title,href){const result=node('a',title);result.href=href;return result;}
const latest=JSON.parse(document.getElementById('latestData').textContent);el('latestReviews').hidden=!latest.length;for(const item of latest){const row=node('div');row.append(link(item.title,item.href),node('p',item.description,'muted'));el('latestLinks').append(row);}
function precedingSample(frames,t){let lo=0,hi=frames.length;while(lo<hi){const mid=(lo+hi)>>1;if(frames[mid].timestamp<=t)lo=mid+1;else hi=mid;}return lo-1;}
function seek(time){video.pause();video.currentTime=Math.max(0,Math.min(active.duration,time));update(video.currentTime);}
function seekIndex(index){const bounded=Math.max(0,Math.min(active.frames.length-1,index));seek(active.frames[bounded].timestamp);}
function verdict(audit){if(!audit)return ['Not model-audited','muted'];if(audit.verdict==='plausible_visible_mask')return ['Model: plausible','good'];if(audit.verdict==='reject_wrong_or_lost_mask')return ['Model: rejected','bad'];return ['Model: uncertain','warn'];}
function renderCurrent(frame){const host=el('currentTracks');host.replaceChildren();if(!frame.tracks.length){host.append(node('p','No component masks at this sample.','empty'));return;}for(const row of frame.tracks){const definition=active.tracks.find(track=>track.id===row.id);const box=node('div',undefined,'track'),head=node('div',undefined,'trackhead');const dot=node('span',undefined,'dot');dot.style.background=definition?definition.color:'#9ca3af';head.append(dot,node('code',row.id));box.append(head);const [text,style]=verdict(row.audit);box.append(node('div',text,'status '+style));if(row.status.indexOf('withheld')>=0||row.reasons.length)box.append(node('div','Geometric gate: withheld · '+row.reasons.join(', '),'status warn'));if(row.audit)box.append(node('p',row.audit.reason));host.append(box);}}
function update(time){if(!active)return;const t=Number.isFinite(time)?time:0,index=precedingSample(active.frames,t);el('videoTime').textContent=t.toFixed(3)+' s';if(index<0){overlay.removeAttribute('src');el('sampleTime').textContent='—';el('sampleAge').textContent='—';el('sampleLabel').textContent='Before the first sampled frame.';lastFrame=-1;return;}const frame=active.frames[index],age=Math.max(0,t-frame.timestamp);el('sampleTime').textContent=frame.timestamp.toFixed(3)+' s';el('sampleAge').textContent=age.toFixed(3)+' s';el('sampleAge').className=age>1.05?'warn':'';if(index!==lastFrame){overlay.src=frame.overlay;el('sampleLabel').textContent='Sample '+frame.sample+' / '+active.frames.length+' · '+frame.timestamp.toFixed(6)+' s';el('sampleSlider').value=String(index);renderCurrent(frame);lastFrame=index;}}
function renderTracks(){const host=el('allTracks');host.replaceChildren();for(const track of active.tracks){const box=node('div',undefined,'track'),head=node('div',undefined,'trackhead'),dot=node('span',undefined,'dot');dot.style.background=track.color;head.append(dot,node('code',track.id));box.append(head,node('p',track.label+' · samples '+track.range.join('–')),node('p',track.description));if(track.withheld.length)box.append(node('p','Geometric gate withheld '+track.withheld.length+' samples: '+track.withheld.join(', '),'warn'));if(track.ranking){const row=node('p'),jump=node('button','Inspect best view: '+track.ranking.timestamp.toFixed(3)+' s','samplelink');jump.addEventListener('click',()=>seek(track.ranking.timestamp));row.append(jump,document.createTextNode(' · '),link(track.ranking.count+' ranked candidates',track.ranking.href));box.append(row);}if(track.uncertainties.length){const detail=node('details'),list=node('ul');detail.append(node('summary','Identity / visibility limits'));for(const text of track.uncertainties)list.append(node('li',text));detail.append(list);box.append(detail);}host.append(box);}}
function renderIssues(){const host=el('issues');host.replaceChildren();if(!active.issues.length){host.append(node('p','No model-rejected or uncertain observations in the supplied audit subset.','empty'));return;}for(const issue of active.issues){const box=node('div',undefined,'issue'),button=node('button',issue.timestamp.toFixed(3)+' s');button.addEventListener('click',()=>seek(issue.timestamp));box.append(button,node('code',issue.track_id),node('div',issue.verdict==='reject_wrong_or_lost_mask'?'Rejected':'Uncertain',issue.verdict==='reject_wrong_or_lost_mask'?'bad':'warn'),node('p',issue.reason));host.append(box);}}
function selectCase(index){if(activeIndex===index)return;video.pause();activeIndex=index;active=cases[index];lastFrame=-2;el('caseTitle').textContent=active.title;el('caseSummary').textContent=active.discovery_summary;Array.from(el('tabs').children).forEach((button,i)=>{button.classList.toggle('active',i===index);button.setAttribute('aria-pressed',String(i===index));});video.src=active.video;video.poster=active.frames[0].source_frame;video.load();el('videoError').hidden=true;el('sampleSlider').max=String(active.frames.length-1);el('counts').textContent=active.stats.tracks+' candidate tracks · '+active.stats.raw_masks+' raw masks · '+active.stats.frames+' sampled frames';el('stats').replaceChildren();for(const [label,value,style]of [['Checked',active.stats.checked,''],['Plausible',active.stats.plausible,'good'],['Rejected',active.stats.rejected,'bad'],['Uncertain',active.stats.uncertain,'warn']]){const box=node('div',undefined,'stat');box.append(node('strong',String(value),style),node('span',label));el('stats').append(box);}const models=el('models');models.replaceChildren();for(const [title,description]of [['Segmentation','Meta SAM 2.1 Tiny · local Mac MPS'],['Discovery + visual audit',active.models.reasoning+' · hosted OpenAI via local Codex CLI'],['Current substitutes','SAM 3.1 and Meta reasoning API have not run.']]){const row=node('div',undefined,'modelrow');row.append(node('strong',title),node('span',description,'muted'));models.append(row);}el('links').replaceChildren(...active.links.map(row=>link(row.title,row.href)));const recon=el('reconstructions');recon.replaceChildren(node('h3','Generated 3D experiments'));if(active.reconstructions.length){for(const item of active.reconstructions)recon.append(link(item.title,item.href));recon.append(node('p','Open each reviewer for reconstruction quality and provenance. A generated mesh does not establish a usable video pose.','muted'));}else{recon.append(node('p','No reconstruction reviewer available at page build time.','empty'));}renderTracks();renderIssues();update(0);}
cases.forEach((item,index)=>{const button=node('button',item.title);button.addEventListener('click',()=>selectCase(index));el('tabs').append(button);});
el('previous').addEventListener('click',()=>seekIndex(Math.max(0,lastFrame-1)));el('next').addEventListener('click',()=>seekIndex(lastFrame+1));el('atSample').addEventListener('click',()=>seekIndex(Math.max(0,lastFrame)));el('sampleSlider').addEventListener('input',event=>seekIndex(Number(event.target.value)));
video.addEventListener('timeupdate',()=>update(video.currentTime));video.addEventListener('seeking',()=>update(video.currentTime));video.addEventListener('seeked',()=>update(video.currentTime));video.addEventListener('loadedmetadata',()=>update(video.currentTime));video.addEventListener('error',()=>{el('videoError').textContent='The source video could not be loaded in this preview. Sample navigation and audit records remain available.';el('videoError').hidden=false;});overlay.addEventListener('error',()=>{el('overlayError').style.display='block';});overlay.addEventListener('load',()=>{el('overlayError').style.display='none';});
function animate(){if(!video.paused&&!video.ended)update(video.currentTime);requestAnimationFrame(animate);}selectCase(0);requestAnimationFrame(animate);
</script></body></html>'''


def build(out_dir):
    out_dir = out_dir.resolve()
    payload = []
    references = []
    for case, title, subtitle, reconstructions in SPECS:
        entry, paths = case_payload(case, title, subtitle, reconstructions, out_dir)
        payload.append(entry)
        references.extend(paths)
    latest, latest_paths = latest_reviews(out_dir)
    references.extend(latest_paths)
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    latest_encoded = json.dumps(latest, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / "index.html"
    destination.write_text(HTML.replace("__PAYLOAD__", encoded).replace("__LATEST__", latest_encoded))
    return {"output": str(destination), "clips": len(payload),
            "referenced_local_files_checked": len(set(references)),
            "sampled_frames": sum(len(case["frames"]) for case in payload),
            "reconstruction_reviewers_found": sum(len(case["reconstructions"]) for case in payload),
            "latest_experiments_found": len(latest)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/local-recordings-review-v3")
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.out), indent=2))
