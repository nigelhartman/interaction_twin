#!/usr/bin/env python3
"""Inspect sampled 2D approach cues without turning occlusion into insertion.

Measures the whole existing source tracklet. Raw area is neither visibility nor
confidence; silhouette overlap is neither contact nor successful seating.
"""
import argparse
import html
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from stage2_discover import read, sha, write


def mask_metrics(moving, containing, reference_area):
    moving, containing = np.asarray(moving,dtype=bool), np.asarray(containing,dtype=bool)
    if moving.shape != containing.shape or moving.ndim != 2:
        raise ValueError('Masks must have the same two-dimensional source coordinates')
    if not math.isfinite(reference_area) or reference_area <= 0:
        raise ValueError('Positive finite reference mask area required')
    area = int(moving.sum())
    result = {'raw_mask_area_pixels':area,'raw_mask_area_over_reference':area/float(reference_area),
              'centroid_distance_over_base_bbox_diagonal':None,'minimum_pixel_center_gap_over_base_bbox_diagonal':None,
              'earbud_mask_fraction_overlapping_base':None,'base_bbox_diagonal_pixels':None}
    if not area or not containing.any():
        return result
    ey,ex = np.nonzero(moving);by,bx = np.nonzero(containing)
    diagonal = float(np.hypot(bx.max()-bx.min()+1,by.max()-by.min()+1))
    result.update(centroid_distance_over_base_bbox_diagonal=float(np.hypot(ex.mean()-bx.mean(),ey.mean()-by.mean())/diagonal),
                  minimum_pixel_center_gap_over_base_bbox_diagonal=float(distance_transform_edt(~containing)[moving].min()/diagonal),
                  earbud_mask_fraction_overlapping_base=float((moving&containing).sum()/area),base_bbox_diagonal_pixels=diagonal)
    return result


def cue_state(moving_verdict,base_verdict,moving_geometric_ok,base_geometric_ok):
    if moving_verdict=='reject_wrong_or_lost_mask' or base_verdict=='reject_wrong_or_lost_mask':
        return 'known_bad_mask_pair_excluded'
    if not moving_geometric_ok or not base_geometric_ok:
        return 'geometric_mask_gate_failed'
    if moving_verdict=='plausible_visible_mask' and base_verdict=='plausible_visible_mask':
        return 'audited_plausible_sampled_2d_cue'
    return 'unassessed_or_uncertain_pair_diagnostic_only'


def merge_verdict(previous,fresh,scope_ok=True):
    """A new positive does not erase an earlier exact-mask rejection/uncertainty."""
    priority={'not_model_audited':0,'unassessed':0,'plausible_visible_mask':1,
              'uncertain':2,'reject_wrong_or_lost_mask':3}
    if previous not in priority or fresh not in priority:
        raise ValueError('Unexpected mask verdict')
    if not scope_ok:
        fresh='reject_wrong_or_lost_mask'
    return max((previous,fresh),key=lambda v:priority[v])


def merge_exact_audits(paths,source_sha,tracks_path,tracks,frames,case,segmentation,verdict):
    sources=[];observations={}
    for path in paths:
        path=Path(path).resolve();record=read(path);evidence_path=path.parent/'evidence.json'
        if record['source_sha256']!=source_sha or record['source_track_index_sha256']!=sha(tracks_path):
            raise ValueError('Fresh audit does not match the exact actual source track index')
        evidence=read(evidence_path)
        keys=[(r['track_id'],r['sample_index']) for r in evidence]
        supplied={(r['track_id'],r['sample_index']):r for r in evidence}
        result=record['result']['observations'];actual=[(r['track_id'],r['sample_index']) for r in result]
        if len(keys)!=len(set(keys)) or len(actual)!=len(set(actual)) or set(keys)!=set(actual):
            raise ValueError('Fresh audit evidence/response pairs differ or duplicate')
        for item in result:
            tid,sample=item['track_id'],item['sample_index'];key=(tid,sample);e=supplied[key]
            if tid not in tracks or sample not in tracks[tid] or sample not in frames:
                raise ValueError('Fresh audit refers to an absent actual mask')
            row=frames[sample];mask=segmentation/tracks[tid][sample]['raw_mask_file']
            if e['image_sha256']!=sha(case/row['file']) or e['mask_sha256']!=sha(mask) or abs(e['timestamp_s']-row['timestamp_s'])>1e-7:
                raise ValueError('Fresh audit pixels/timestamp differ; no verdict transfer allowed')
            scope_ok=item['single_component_visible'] and item['component_boundary_plausible'] and not item['other_components_inside_mask']
            old=verdict.get(key,'not_model_audited');fresh=item['verdict']
            verdict[key]=merge_verdict(old,fresh,scope_ok)
            observations.setdefault(key,[]).append({'audit_file':str(path),'audit_sha256':sha(path),
                                                   'evidence_file':str(evidence_path),'evidence_sha256':sha(evidence_path),
                                                   'source_image_sha256':e['image_sha256'],'source_mask_sha256':e['mask_sha256'],
                                                   'previous_verdict':old,'new_verdict':fresh,'scope_passed':bool(scope_ok),
                                                   'effective_verdict':verdict[key],'reason':item['reason']})
        sources += [path,evidence_path]
    return sources,observations


def semantic_windows(path, object_id):
    """Take model-authored temporal windows unchanged; never select pixel paths."""
    record = read(path)
    windows=[]
    if 'steps' in record:
        for s in record['steps']:
            if object_id in s['participant_track_ids']:
                windows.append({'id':s['id'],'title':s['title'],'first_timestamp_s':s['start_s'],'last_timestamp_s':s['end_s'],
                                'limitations':s['limitations'],'epistemic_status':'existing_model_semantic_proposal'})
    elif 'result' in record and 'events' in record['result']:
        for s in record['result']['events']:
            windows.append({'id':s['id'],'title':s['description'],'first_timestamp_s':s['first_timestamp_s'],
                            'last_timestamp_s':s['last_timestamp_s'],'limitations':s['uncertainties'],
                            'epistemic_status':s['epistemic_status']})
    else:
        raise ValueError('Expected actual compiled model steps or fresh model event record')
    for w in windows:
        if not math.isfinite(w['first_timestamp_s']) or not math.isfinite(w['last_timestamp_s']) or w['first_timestamp_s']>w['last_timestamp_s']:
            raise ValueError('Invalid actual model window')
    return windows


def plot_svg(rows):
    keys=[('minimum_pixel_center_gap_over_base_bbox_diagonal','Minimum 2D pixel gap / base-box diagonal'),
          ('centroid_distance_over_base_bbox_diagonal','Centroid distance / base-box diagonal'),
          ('earbud_mask_fraction_overlapping_base','Earbud-mask fraction overlapping base mask'),
          ('raw_mask_area_over_reference','Raw earbud-mask area / reference mask area')]
    width,height,left,right,top=1000,750,65,25,30
    lo,hi=rows[0]['timestamp_s'],rows[-1]['timestamp_s']
    xx=lambda t:left+(width-left-right)*(t-lo)/(hi-lo)
    parts=[f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Sampled mask-derived two dimensional diagnostics"><rect width="1000" height="750" fill="#101d30"/>']
    for j,(key,label) in enumerate(keys):
        y0=top+j*175
        values=[r['metrics'][key] for r in rows if r['metrics'][key] is not None]
        maximum=max(values,default=1.)*1.12 or 1.
        parts.append(f'<text x="{left}" y="{y0}" fill="#ecf3fb" font-size="15">{html.escape(label)}</text>')
        for fraction in [0,.5,1]:
            y=y0+130-fraction*100
            parts.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#2d3c50"/><text x="8" y="{y+4}" fill="#aab9cb" font-size="12">{fraction*maximum:.2g}</text>')
        for r in rows:
            x=xx(r['timestamp_s']);v=r['metrics'][key];state=r['cue_status'];color='#55dbb0' if state=='audited_plausible_sampled_2d_cue' else '#f2bc64'
            title=html.escape(f"sample {r['sample_index']} / {r['timestamp_s']:.6f}s / {state} / {v}")
            if v is None:
                parts.append(f'<g data-sample="{r["sample_index"]}" tabindex="0"><title>{title}</title><path d="M{x-3:.2f},{y0+143}l6,6m-6,0l6,-6" stroke="#ef7c80" stroke-width="2"/></g>')
            else:
                y=y0+130-(v/maximum)*100
                fill=color if state=='audited_plausible_sampled_2d_cue' else '#101d30'
                parts.append(f'<g data-sample="{r["sample_index"]}" tabindex="0"><title>{title}</title><circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{fill}" stroke="{color}" stroke-width="1.6"/></g>')
        for t in np.linspace(lo,hi,7):
            parts.append(f'<text x="{xx(t):.2f}" y="{y0+165}" text-anchor="middle" fill="#aab9cb" font-size="12">{t:.1f}s</text>')
    parts.append('</svg>')
    return ''.join(parts)


def build(case,motion,semantics,out,object_id='small_object_a',base_id='container_base',additional_audits=()):
    case,motion,semantics,out=[Path(p).resolve() for p in (case,motion,semantics,out)]
    if out.exists():
        raise ValueError('Fresh approach output required')
    manifest=read(case/'manifest.json');pack=read(motion)
    if pack['object_id']!=object_id or pack['provenance']['source_video_sha256']!=manifest['source']['sha256']:
        raise ValueError('Motion/source/object mismatch')
    semantic_record=read(semantics)
    semantic_source=semantic_record.get('source_sha256',semantic_record.get('source',{}).get('sha256'))
    if semantic_source!=manifest['source']['sha256']:
        raise ValueError('Semantic windows refer to another source video')
    segmentation=case/'segmentation/automatic-components-v1'
    tracks_path=segmentation/'tracks.json';tracks={t['track_id']:{r['sample_index']:r for r in t['frames']} for t in read(tracks_path)['tracks']}
    audit_path=segmentation/'model-audit-v1/audit.json';audit=read(audit_path)
    if audit['source_sha256']!=manifest['source']['sha256'] or audit['source_track_index_sha256']!=sha(tracks_path):
        raise ValueError('Model mask audit does not match actual source tracks')
    verdict={(r['track_id'],r['sample_index']):r['verdict'] for r in audit['result']['observations']}
    gates={(t['track_id'],r['sample_index']):t for r in read(segmentation/'frames.json')['frames'] for t in r['tracks']}
    frames={r['index']:r for r in read(case/'frames/index.json')['frames']}
    added_sources,added_observations=merge_exact_audits(additional_audits,manifest['source']['sha256'],tracks_path,tracks,frames,case,segmentation,verdict)
    poses={r['sample_index']:r for r in pack['frames']}
    reference=pack['reference_sample_index'];reference_mask=segmentation/tracks[object_id][reference]['raw_mask_file']
    reference_area=int((np.array(Image.open(reference_mask).convert('L'))>127).sum())
    out.mkdir(parents=True);(out/'overlays').mkdir()
    rows=[]
    for sample,entry in sorted(tracks[object_id].items()):
        source=frames[sample];image_path=case/source['file'];earbud_path=segmentation/entry['raw_mask_file']
        if sample not in tracks[base_id]:
            raise ValueError('Missing contemporaneous containing-component mask')
        base_path=segmentation/tracks[base_id][sample]['raw_mask_file'];pose=poses.get(sample)
        if pose and (pose['source_frame_sha256']!=sha(image_path) or pose['sam_mask_sha256']!=sha(earbud_path) or abs(pose['timestamp_s']-source['timestamp_s'])>1e-7):
            raise ValueError('Motion pixels or timestamps changed')
        earbud=np.array(Image.open(earbud_path).convert('L'))>127;base=np.array(Image.open(base_path).convert('L'))>127
        moving_v=verdict.get((object_id,sample),'not_model_audited');base_v=verdict.get((base_id,sample),'not_model_audited')
        moving_g=gates[(object_id,sample)];base_g=gates[(base_id,sample)]
        moving_ok=bool(earbud.any()) and not moving_g['reasons'];base_ok=bool(base.any()) and not base_g['reasons']
        status=cue_state(moving_v,base_v,moving_ok,base_ok)
        metrics=mask_metrics(earbud,base,reference_area)
        if status in {'known_bad_mask_pair_excluded','geometric_mask_gate_failed'}:
            # The raw earbud area is a diagnostic input count. Pair geometry
            # from a known bad mask is excluded, never treated as evidence.
            for key in ['centroid_distance_over_base_bbox_diagonal','minimum_pixel_center_gap_over_base_bbox_diagonal','earbud_mask_fraction_overlapping_base','base_bbox_diagonal_pixels']:
                metrics[key]=None
            if moving_v=='reject_wrong_or_lost_mask' or not moving_ok:
                metrics['raw_mask_area_over_reference']=None
        rgb=np.array(Image.open(image_path).convert('RGB'));overlay=rgb.copy()
        if base_v!='reject_wrong_or_lost_mask' and base_ok:
            overlay[base]=np.rint(.7*overlay[base]+.3*np.array([235,181,74])).astype(np.uint8)
        if moving_v!='reject_wrong_or_lost_mask' and moving_ok:
            overlay[earbud]=np.rint(.55*overlay[earbud]+.45*np.array([56,220,202])).astype(np.uint8)
        overlay_file=out/'overlays'/f'{sample:06d}.jpg';Image.fromarray(overlay).save(overlay_file,quality=94)
        rows.append({'sample_index':sample,'timestamp_s':source['timestamp_s'],'cue_status':status,
                     'moving_mask_audit_verdict':moving_v,'base_mask_audit_verdict':base_v,
                     'moving_exact_hash_audit_extensions':added_observations.get((object_id,sample),[]),
                     'base_exact_hash_audit_extensions':added_observations.get((base_id,sample),[]),
                     'moving_geometric_gate_passed':moving_ok,'base_geometric_gate_passed':base_ok,
                     'moving_gate_reasons':moving_g['reasons'],'base_gate_reasons':base_g['reasons'],
                     'metrics':metrics,'source_frame_file':str(image_path),'source_frame_sha256':sha(image_path),
                     'moving_mask_file':str(earbud_path),'moving_mask_sha256':sha(earbud_path),
                     'base_mask_file':str(base_path),'base_mask_sha256':sha(base_path),'overlay_file':str(overlay_file),
                     'pose_status':pose['status'] if pose else 'outside_motion_experiment',
                     'physical_identity_across_occlusion':'unverified','successful_seating':'unknown','contact':'unknown'})
    source_path=case/manifest['source']['file']
    if sha(source_path)!=manifest['source']['sha256']:
        raise ValueError('Actual video hash changed')
    windows=semantic_windows(semantics,object_id)
    last_supported=max((r['timestamp_s'] for r in pack['frames'] if r['status'].startswith('supported_')),default=None)
    record={'status':'complete_sampled_2d_approach_diagnostics','object_id':object_id,'containing_component_id':base_id,
            'source_video_file':str(source_path),'source_video_sha256':manifest['source']['sha256'],
            'time_domain':'exact_source_presentation_seconds','source_tracklet_sample_count':len(rows),
            'selection_policy':'Every sample in the actual moving-object SAM tracklet, without selecting a favorable action interval.',
            'reference_sample_index':reference,'reference_timestamp_s':frames[reference]['timestamp_s'],
            'reference_mask_file':str(reference_mask),'reference_mask_sha256':sha(reference_mask),'reference_raw_mask_area_pixels':reference_area,
            'inputs':[{'file':str(p),'sha256':sha(p)} for p in [case/'manifest.json',case/'frames/index.json',tracks_path,segmentation/'frames.json',audit_path,motion,semantics]+added_sources],
            'audit_merge_policy':'Require matching actual video/track index, exact RGB and mask hashes and exact source PTS; retain earlier rejection/uncertainty; a scope veto rejects that pair.',
            'semantic_windows':windows,'rows':rows,'reliable_2d_sample_count':sum(r['cue_status']=='audited_plausible_sampled_2d_cue' for r in rows),
            'last_supported_3d_pose_timestamp_s':last_supported,'final_insertion_path_status':'unsupported_not_filled',
            'successful_seating':'unknown','physical_contact':'unknown','manual_geometry_or_pose_or_mask_changes':False,
            'limitations':['Mask area is not a visibility fraction or confidence: perspective, rotation, SAM error, hand occlusion and scale can all change it.',
                           'Overlap and decreasing 2D distance do not establish 3D approach, entry, contact or seating.',
                           'Only simultaneous model-plausible masks with passing geometric gates are tagged as reliable sampled 2D cues; model plausibility is still fallible.',
                           'Open orange points are unassessed or uncertain diagnostic inputs. Known rejected masks and failed geometric pairs are excluded from pair geometry.',
                           'No interpolation, hidden motion, filled pose paths or final seating inference is generated.']}
    write(out/'measurements.json',record)
    svg=plot_svg(rows);(out/'timeline.svg').write_text(svg)
    display=json.loads(json.dumps(record))
    for r in display['rows']:
        for k in ['source_frame_file','overlay_file']:
            r[k]=os.path.relpath(r[k],out)
    data=json.dumps(display).replace('<','\\u003c')
    video=html.escape(os.path.relpath(source_path,out),quote=True)
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Jabra · approach evidence</title><style>
body{margin:0;background:#081322;color:#e6eff9;font:15px/1.5 system-ui}main{max-width:1350px;margin:auto;padding:26px}h1{font-size:28px}a{color:#87d7ff}.note,.card{background:#132239;border:1px solid #2d435d;padding:16px;border-radius:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}video{width:100%;max-height:560px;background:#000}.image-row{display:flex;gap:8px}.image-row img{width:48%;object-fit:contain;max-height:420px}svg{width:100%;height:auto}svg [data-sample]{cursor:pointer}button,select{background:#233b54;color:white;border:1px solid #56738a;padding:8px;border-radius:5px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.events button{display:block;text-align:left;margin:8px 0;width:100%}.legend{color:#becdde}.good{color:#55dbb0}.unknown{color:#f2bc64}.bad{color:#ef7c80}@media(max-width:850px){.grid{grid-template-columns:1fr}}
</style><main><h1>Approach evidence, with the gaps visible</h1><p>Automatic measurements across the entire 61-sample earbud tracklet. Click a point to seek the full source video and inspect its exact mask evidence.</p><p class="note"><strong>The final insertion path is unsupported. Seating and contact remain unknown.</strong> Raw mask area is not visibility. Silhouette overlap is not physical contact. Dots are independent source samples; no hidden path is filled.</p><section class="grid"><div class="card"><video id="video" controls preload="metadata" src="VIDEO"></video><p id="time"></p><label>Source sample <select id="sample"></select></label><div class="events" id="events"></div></div><div class="card"><h2 id="selected"></h2><p id="status"></p><div class="image-row"><img id="rgb" alt="Exact source frame"><img id="overlay" alt="Automatically derived mask overlay"></div><pre id="details"></pre></div></section><section class="card" style="margin-top:18px"><p class="legend"><span class="good">● Both masks model-plausible and geometric gates pass</span> · <span class="unknown">○ Unassessed/uncertain diagnostic</span> · <span class="bad">× Excluded known bad pair</span>. Base mask amber; earbud mask teal. The model has not measured accuracy.</p>SVG</section><p><a href="measurements.json">Measurements, source hashes and uncertainty</a> · <a href="timeline.svg">Standalone sampled plot</a></p></main><script id="data" type="application/json">DATA</script><script>
const d=JSON.parse(document.getElementById('data').textContent),v=document.getElementById('video'),s=document.getElementById('sample');
function choose(index,seek=true){const r=d.rows.find(x=>x.sample_index===Number(index));if(!r)return;if(seek)v.currentTime=r.timestamp_s;s.value=r.sample_index;document.getElementById('selected').textContent=`Sample ${r.sample_index} · ${r.timestamp_s.toFixed(6)} s`;document.getElementById('status').textContent=r.cue_status.replaceAll('_',' ');document.getElementById('rgb').src=r.source_frame_file;document.getElementById('overlay').src=r.overlay_file;document.getElementById('details').textContent=JSON.stringify({moving_mask_audit:r.moving_mask_audit_verdict,base_mask_audit:r.base_mask_audit_verdict,pose:r.pose_status,measurements:r.metrics,seating:r.successful_seating,contact:r.contact},null,2);}
for(const r of d.rows){const o=document.createElement('option');o.value=r.sample_index;o.textContent=`${r.sample_index} · ${r.timestamp_s.toFixed(3)}s`;s.append(o);}s.onchange=()=>choose(s.value);
for(const p of document.querySelectorAll('[data-sample]')){p.onclick=()=>choose(p.dataset.sample);p.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose(p.dataset.sample);}};}
for(const e of d.semantic_windows){if(e.last_timestamp_s<d.rows[0].timestamp_s||e.first_timestamp_s>d.rows.at(-1).timestamp_s)continue;const b=document.createElement('button');b.textContent=`Model proposal ${e.first_timestamp_s.toFixed(2)}–${e.last_timestamp_s.toFixed(2)}s: ${e.title}`;b.title=e.limitations.join(' ');b.onclick=()=>{v.currentTime=e.first_timestamp_s;const r=d.rows.reduce((a,r)=>Math.abs(r.timestamp_s-e.first_timestamp_s)<Math.abs(a.timestamp_s-e.first_timestamp_s)?r:a);choose(r.sample_index,false);};document.getElementById('events').append(b);}
v.ontimeupdate=()=>{document.getElementById('time').textContent=`Video ${v.currentTime.toFixed(3)}s · exact selected frame ${Number(s.selectedOptions[0]?.textContent.split('·')[1]?.replace('s','')||0).toFixed(3)}s`;};choose(d.reference_sample_index,false);v.addEventListener('loadedmetadata',()=>{v.currentTime=d.reference_timestamp_s;},{once:true});
</script></html>'''.replace('VIDEO',video).replace('SVG',svg).replace('DATA',data)
    (out/'index.html').write_text(document)
    print(json.dumps({'output':str(out),'samples':len(rows),'reliable_2d_samples':record['reliable_2d_sample_count'],
                      'last_supported_3d_pose_s':last_supported,'final_seating':'unknown'}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True);parser.add_argument('--motion',type=Path,required=True)
    parser.add_argument('--semantics',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--object-id',default='small_object_a');parser.add_argument('--base-id',default='container_base')
    parser.add_argument('--additional-audit',type=Path,nargs='*',default=[])
    a=parser.parse_args();build(a.case,a.motion,a.semantics,a.out,a.object_id,a.base_id,a.additional_audit)


if __name__=='__main__':
    main()
