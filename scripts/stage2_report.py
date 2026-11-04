#!/usr/bin/env python3
"""Publish local diagnostic artifacts and a conservative candidate-identity registry."""
import argparse,json,html,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--run',default='components-v1');a=p.parse_args()
 case=ROOT/'data'/a.case;run=case/'segmentation'/a.run;cfg=read(run/'annotations.json');quality=read(run/'quality.json');rows=read(case/'frames/index.json')['frames'];events=read(case/'reasoning'/a.run/'events.json')
 registry={'schema_version':'0.2','source_sha256':read(case/'manifest.json')['source']['sha256'],'identity_policy':'candidate IDs, never automatic cross-occlusion merges','human_confirmed':False,
 'entity_candidates':[{'id':'camera_body','role':'camera body','status':'single_body_candidate'},{'id':'compartment_cover','role':'camera compartment cover','status':'unique_attached_component_candidate'},{'id':'battery_candidate_01','role':'cylindrical battery-like component','status':'distinct_outward_candidate_only'},{'id':'battery_candidate_02','role':'cylindrical battery-like component','status':'distinct_outward_candidate_only'}],
 'tracklet_bindings':[{'track_id':t['id'],'candidate_entity_ids':t['candidate_entity_ids'],'identity_status':t['identity_status'],'source_interval_s':[rows[t['sample_range'][0]-1]['timestamp_s'],rows[t['sample_range'][1]-1]['timestamp_s']]} for t in cfg['tracks']],
 'proposed_identity_links':events['result']['identity_links'],
 'limitations':['Additional tracklets do not establish additional physical batteries.','Table and inward tracklets remain ambiguous between the outward candidates.','Not-assessed and withheld spans do not prove absence.','Camera/cover separation is a 2D mask hypothesis, not a validated articulation model.']}
 write(case/'entities.json',registry)
 e=html.escape;prefix='segmentation/'+a.run+'/'
 track_rows=''
 for t in cfg['tracks']:
  q=quality['tracks'][t['id']]
  track_rows+=f'<tr><td><span class="swatch" style="background:{e(t["color"])}"></span>{e(t["label"])}</td><td>{e(t["id"])}</td><td>{q["candidate_visible_samples"]} / {q["raw_predictions"]}</td><td>{len(q["withheld_samples"])}</td><td>{e(t["identity_status"])}</td><td><a href="{prefix}raw-review-{e(t["id"])}.jpg">Inspect raw predictions</a></td></tr>'
 event_rows=''
 for ev in events['result']['events']:
  nearest=min(rows,key=lambda r:abs(r['timestamp_s']-ev['start_s']))['index']
  refs=' '.join(f'<a href="frames/{i:06d}.jpg">#{i} ({rows[i-1]["timestamp_s"]:.3f}s)</a>' for i in ev['evidence_samples'])
  limits='; '.join(ev['limitations'])
  event_rows+=f'<tr><td><button data-sample="{nearest}">{ev["start_s"]:.3f}-{ev["end_s"]:.3f}s</button><br>{e(ev["status"])}</td><td>{e(ev["description"])}<br><small>{e(limits)}</small></td><td>{refs}</td></tr>'
 concerns=''.join('<li>'+e(x['track_id']+': '+x['description'])+'</li>' for x in events['result']['mask_concerns'])
 notes=''.join('<li>'+e(x)+'</li>' for x in quality.get('review_notes',[]))
 unknown=''.join('<li>'+e(x)+'</li>' for x in events['result']['uncertainties'])
 sheets=''.join(f'<a href="{prefix}{e(s)}"><img src="{prefix}{e(s)}" loading="lazy" alt="Timestamped component-mask contact sheet"></a>' for s in quality['contact_sheets'])
 fps=read(case/'manifest.json')['sampling']['fps']
 doc=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage 2 - Component and event review</title><style>
body{{margin:0;background:#f2f5f7;color:#183244;font:16px system-ui}}main{{max-width:1260px;margin:auto;padding:28px}}video{{width:100%;background:#172e3e}}h1{{margin-bottom:8px}}.notice{{padding:16px;background:#e2edf4}}table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{padding:12px 8px;border-bottom:1px solid #ccd9e0;text-align:left;vertical-align:top}}th{{background:#e4edf2}}small{{color:#536979}}a{{color:#205d9e}}button{{font:inherit;padding:8px;border:1px solid #adbcc6;border-radius:5px;background:white;color:#183244;cursor:pointer}}button:focus-visible{{outline:3px solid #397db8}}.swatch{{display:inline-block;width:13px;height:13px;margin-right:7px}}.scroll{{overflow-x:auto}}.sheets img{{width:100%;margin:10px 0}}@media(max-width:600px){{main{{padding:14px}}}}
</style><main><h1>Stage 2 / Components and events</h1><p>{len(rows)} frames at approximately {fps:g} fps / source {rows[0]['timestamp_s']:.3f}-{rows[-1]['timestamp_s']:.3f}s</p>
<p class="notice">Real SAM 2.1 Tiny masks on MPS, with assistant-authored prompts and withholding decisions. Event reasoning uses the local Codex CLI with hosted OpenAI inference as a temporary Meta substitute. Outputs are candidates, not human-confirmed identities or measured 3D motion.</p>
<video id="preview" controls preload="metadata" poster="{prefix}comparison/000001.jpg" src="{prefix}components-preview.mp4"></video><p>Playback holds each sampled result for 0.2s. The camera, cover and battery tracklets are separate candidates. A question mark means correspondence across occlusion is unresolved. Grey pixels indicate conflicting part masks.</p>
<p><a href="entities.json">Entity/identity registry</a> · <a href="reasoning/{e(a.run)}/events.json">Event evidence</a> · <a href="{prefix}quality.json">Quality report</a> · <a href="{prefix}annotations.json">Seeds and withholding log</a> · <a href="{prefix}frames.json">Per-sample masks</a></p>
<h2>What was observed</h2><p>{e(events['result']['summary'])}</p><h2>Component coverage</h2><p>Counts describe exported candidate samples within each local tracklet, not accuracy. Gaps remain unknown.</p><div class="scroll"><table><tr><th>Target</th><th>Track ID</th><th>Candidate samples / raw predictions</th><th>Withheld</th><th>Identity status</th><th>Raw review</th></tr>{track_rows}</table></div>
<h2>Event proposals</h2><p>Click a source interval to seek the sampled preview. Original evidence frames open separately.</p><div class="scroll"><table><tr><th>Source time / status</th><th>Proposal and limitations</th><th>Evidence</th></tr>{event_rows}</table></div>
<h2>Quality review</h2><ul>{notes}</ul><h3>Reasoning-model mask concerns</h3><ul>{concerns}</ul><h3>Remaining uncertainty</h3><ul>{unknown}</ul>
<h2>All sampled overlays</h2><div class="sheets">{sheets}</div></main><script>document.querySelectorAll('button[data-sample]').forEach(b=>b.addEventListener('click',()=>{{const v=document.getElementById('preview');v.currentTime=(Number(b.dataset.sample)-1)/{fps};v.scrollIntoView({{behavior:'smooth',block:'center'}});}}));</script></html>'''
 (case/'stage2-review.html').write_text(doc);print(case/'stage2-review.html')
if __name__=='__main__':main()
