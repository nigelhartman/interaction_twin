#!/usr/bin/env python3
"""Render the saved candidate report; CPU-only presentation, no model calls."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import local_candidates as C


def render(output: Path) -> Path:
    output=output.resolve();report=C.read(output/'report.json');plan=C.read(output/'plan.json')
    if C.sha(output/'plan.json')!=report['plan_sha256']:raise ValueError('Report does not match frozen plan')
    esc=html.escape
    label_edits=C.read(output/'presentation-receipt.json').get('observer_heading_edits',[]) if (output/'presentation-receipt.json').exists() else []
    # The preserved generic observer renderer predates case components and has
    # an earbud-only heading. Correct text labels in generated pages only;
    # preserve its original output and all pose/mesh data byte-for-byte.
    for part in report['parts']:
        for candidate in part['candidates']:
            observer=(ROOT/candidate['observer']) if candidate.get('observer') else None
            if observer and observer.is_file():
                original=observer.read_text()
                revised=original.replace('Generated earbud · video and interactive 3D',esc(part['track_id'])+' · video and interactive 3D').replace('Full video beside the moving generated earbud','Video and conditional '+esc(part['track_id'])+' motion')
                if revised!=original:
                    backup=observer.with_name('observer-3d-original-labels.html')
                    if not backup.exists():backup.write_text(original)
                    observer.write_text(revised)
                    label_edits.append({'path':str(observer.relative_to(ROOT)),'original_sha256':C.sha(backup),'revised_sha256':C.sha(observer),'changes':'title and h1 text only; no embedded data or code changed'})
    def local(path):
        resolved=(ROOT/path).resolve()
        if not resolved.is_relative_to(ROOT/'data'):raise ValueError('Report link outside project data')
        if not resolved.is_file():return None
        return esc(os.path.relpath(resolved,output),quote=True)
    def link(path,label):
        url=local(path)
        return f'<a href="{url}">{esc(label)}</a>' if url else f'<span>{esc(label)} unavailable</span>'
    def cell(score):
        if not score:return '—'
        residual=score.get('median_supported_heldout_px');overlap=score.get('median_supported_iou')
        return f'<b>{score["supported"]}/{score["denominator"]}</b><small>IoU {overlap:.3f} · {residual:.2f}px</small>' if overlap is not None and residual is not None else f'<b>{score["supported"]}/{score["denominator"]}</b><small>No supported fit</small>'
    summaries=[];sections=[]
    for part in report['parts']:
        tid=part['track_id'];base=part['baseline'];winner=part.get('winner')
        summaries.append(f'<tr><th><a href="#{esc(tid)}">{esc(tid)}</a></th><td>{cell(base["validation"])}</td><td>{cell(winner.get("validation") if winner else None)}</td><td>{cell(base.get("final_test"))}</td><td>{cell(winner.get("final_test") if winner else None)}</td><td>{"Retain baseline" if part["decision"]=="retain_preserved_baseline" else "Candidate selected on validation"}</td></tr>')
        cards=[]
        for c in part['candidates']:
            selection=C.read(ROOT/c['selection']);crop=Path(c['selection']).parent/selection['selected'][0]['exports']['rgba_crop']
            assets=[]
            if c.get('model') and (ROOT/c['model']).is_file():
                assets=[link(c['model'],'Model receipt'),link(str(Path(c['model']).parent/'asset.glb'),'Original GLB'),link(str(Path(c['model']).parent/'model-input.png'),'Model input')]
            if c.get('reference'):assets.append(link(str(Path(c['reference']).parent/'index.html'),'Reference fit'))
            if c.get('observer'):assets.append(link(c['observer'],'Inspect motion in 3D'))
            cropurl=local(crop);photo=f'<img src="{cropurl}" alt="Automatic saved SAM crop for {esc(c["id"])}">' if cropurl else ''
            selected=c['id']==part.get('validation_winner');badge='<span class="badge">Validation winner</span>' if selected else ''
            cards.append(f'<article>{photo}<div class="content">{badge}<h3>{esc(c["id"])}</h3><p>Source {c["timestamp_s"]:.3f}s · quality rank {c["original_quality_rank"]} in the frozen input pool</p><dl><div><dt>Validation</dt><dd>{cell(c.get("validation"))}</dd></div><div><dt>Final diagnostic</dt><dd>{cell(c.get("final_test"))}</dd></div></dl><p class="status">{esc(c["status"])}</p><nav>{"".join(assets)}</nav><small>{esc(c.get("error","Unmodified learned shape and decoder-predicted colour. Hidden surfaces remain inferred."))}</small></div></article>')
        sections.append(f'<section id="{esc(tid)}"><header><div class="eyebrow">{esc(tid)}</div><h2>{"Retained baseline" if part["decision"]=="retain_preserved_baseline" else "Validation-selected candidate"}</h2><p>{esc(part["selection_status"])}. All candidates keep their own native scale and camera; this gallery does not assemble them into one physically registered scene.</p><p>{link(base["experiment"],"Preserved baseline motion")} · {link(base["model_metadata"],"Preserved baseline model")}</p></header><div class="cards">{"".join(cards)}</div></section>')
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Saved views → candidate twins</title><style>
    :root{color-scheme:dark}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#0a141d;color:#edf5f6;font:15px/1.55 system-ui}main{max-width:1440px;margin:auto;padding:40px 28px 90px}a{color:#7ce4cb;text-decoration:none}a:hover{text-decoration:underline}.hero{padding:28px 0 35px}.eyebrow{color:#83b9c5;font-size:12px;letter-spacing:.13em;text-transform:uppercase}h1{font-size:clamp(33px,5vw,62px);max-width:950px;line-height:1.1;margin:15px 0 22px}h2{font-size:30px;margin:7px 0}h3{font-size:19px;margin:14px 0 8px}p{color:#b6cad4;max-width:1000px}.hero p{font-size:18px}.pipeline{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:28px 0}.pipeline b{border:1px solid #365265;border-radius:9px;background:#172a37;padding:10px 16px}.pipeline span{color:#7395a8}.note{padding:22px;border:1px solid #8d763d;background:#312b1c;border-radius:12px;color:#e7dab8}.table{overflow:auto;border:1px solid #2d4757;border-radius:12px;margin:25px 0}table{border-collapse:collapse;width:100%;min-width:780px;background:#112330}th,td{text-align:left;padding:17px;border-bottom:1px solid #2d4757;vertical-align:top}thead{font-size:12px;text-transform:uppercase;color:#91b1c1}td small,dd small{display:block;color:#a3bdca;font-size:11px}section{margin-top:65px;scroll-margin-top:30px}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px;margin-top:25px}article{background:#132532;border:1px solid #345164;border-radius:14px;overflow:hidden;min-width:0}article>img{display:block;width:100%;height:220px;object-fit:contain;padding:22px;background:radial-gradient(ellipse,#3a4c58,#17232c)}.content{padding:23px}.content p{font-size:13px}.badge{display:inline-block;border:1px solid #458e78;background:#1c5143;color:#a5f3d5;font-size:11px;padding:3px 9px;border-radius:20px}dl{display:flex;gap:22px;margin:20px 0}dt{font-size:11px;color:#9db9c8}dd{margin:4px 0;font-size:22px}nav{display:flex;gap:8px;flex-wrap:wrap;margin:17px 0}nav a{border:1px solid #3c5c6d;border-radius:6px;padding:5px 9px;font-size:11px}.status{font-size:12px;color:#9dcab9}.content>small{display:block;color:#9eb6c3;font-size:11px;overflow-wrap:anywhere}footer{padding-top:45px;color:#8ba8b8;font-size:12px}@media(max-width:850px){main{padding:25px 16px}.cards{grid-template-columns:1fr}.hero{padding-top:10px}article>img{height:200px}}
    </style><main><div class="hero"><div class="eyebrow">Local experiment · saved evidence · no new hosted inference</div><h1>Can a different saved view give us a better interaction asset?</h1><p>Three Jabra components. Three independently reconstructed candidates each. Automatic inputs, untouched learned meshes, fixed conditional cameras and a comparison against preserved baselines.</p><div class="pipeline"><b>Saved SAM crops</b><span>→</span><b>TripoSR · MPS</b><span>→</span><b>Generic pose + CoTracker3</b><span>→</span><b>Validation lock</b><span>→</span><b>Final diagnostic</b></div></div><div class="note"><b>Read these scores as internal diagnostics.</b> Full-clip tracking observes scoring frames, and the same masks influence fitting. Base has only 2 validation / 2 final samples; lid 3 / 3; earbud 24 / 15. Baseline tracking used larger SAM crops; new candidates use the frozen generic full-frame tracker. This is a whole-pipeline comparison, not a causal test of mesh quality. Hidden surfaces, metric scale, physical pose, contact and seating remain unverified. Three single-view meshes are not multi-view reconstruction.</div><div class="table"><table><thead><tr><th>Component</th><th>Baseline validation</th><th>Winner validation</th><th>Baseline final</th><th>Winner final</th><th>Decision from validation</th></tr></thead><tbody>'''+''.join(summaries)+'''</tbody></table></div><p>Each score is supported non-reference source samples / declared denominator. IoU and disjoint-point residual summarize supported samples only; unknown/unsupported samples are retained. Final diagnostics do not choose the winner.</p>'''+''.join(sections)+'''<footer><p><a href="report.json">Report JSON</a> · <a href="plan.json">Frozen plan</a> · <a href="run.json">Execution + recovery receipts</a> · <a href="orchestration-migration.json">Shutdown recovery record</a></p><p>No manual geometry, poses, seed boxes or forced insertion paths were added. Unsupported movement remains absent. Multiple candidates may fail; failed and interrupted artifacts are preserved.</p></footer></main></html>'''
    target=output/'index.html'
    if target.is_file() and not (output/'index-generated-basic.html').exists():(output/'index-generated-basic.html').write_bytes(target.read_bytes())
    target.write_text(document)
    C.write(output/'presentation-receipt.json',{'created_at':C.now(),'report_sha256':C.sha(output/'report.json'),'plan_sha256':report['plan_sha256'],'renderer':str(Path(__file__).relative_to(ROOT)),'renderer_sha256':C.sha(__file__),'html_sha256':C.sha(target),'observer_heading_edits':label_edits,'new_model_inference':False,'geometry_or_poses_modified':False})
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(render(args.output))
