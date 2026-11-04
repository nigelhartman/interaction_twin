#!/usr/bin/env python3
"""Local static owner review of the genuinely unresolved recovered laptop hinge."""
import html
import json
import os
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import laptop_hinge_omission_recovery_v1 as R
import native_scene_audit as A
import object_program as O


def build():
    R.verify()
    folder = ROOT / R.OUTPUT
    result = R.read(folder / 'private-scene-result.json')
    output = folder / 'review'
    if output.exists():
        raise ValueError('Preserve the existing owner review version')
    audits = {name: A.validate_manifest(folder / ('visual-' + name) / 'audit.json', root=ROOT)
              for name in ['constrained', 'unconstrained_diagnostic']}
    # Final-test panels are preserved by the auditor but excluded from this
    # inspection/feedback page and proof sheet; no optimization uses their scores.
    development = {name: [s for s in doc['samples'] if s['role'] == 'development'] for name, doc in audits.items()}
    program_path = ROOT / result['descriptive_export']['object_program']['path']
    O.validate(R.read(program_path), root=ROOT)
    output.mkdir()

    def url(path):
        return html.escape(os.path.relpath(path, output), quote=True)

    sections = []
    proof_images = []
    for name, title, explanation in [
        ('constrained', 'Current constrained result', 'No native surface supported the cited hinge seam. The lid is withheld; no detached or floating fallback is shown.'),
        ('unconstrained_diagnostic', 'Independent pose diagnostic', 'These per-part silhouette estimates are only a diagnostic. They do not establish an assembly or a working hinge.'),
    ]:
        cards = []
        for sample in sorted(development[name], key=lambda s: (s['timestamp_s'], s['component_id'])):
            path = ROOT / sample['comparison_panel_ref']['path']
            label = f"{sample['component_id']} · {sample['timestamp_s']:.3f}s · {sample['status']}"
            cards.append(f'<figure><img src="{url(path)}" alt="{html.escape(label, quote=True)}"><figcaption>{html.escape(label)}</figcaption></figure>')
            if name == 'constrained' or sample['component_id'] == 'laptop_lid':
                proof_images.append((title + ' / ' + label, Image.open(path).convert('RGB')))
        sections.append(f'<section><h2>{title}</h2><p>{explanation}</p><div class="grid">{"".join(cards)}</div></section>')
    links = [
        ('Descriptive object program · JSON', program_path),
        ('Constrained scene · JSON', ROOT / result['scene']['manifest']['path']),
        ('Original Muse response · unchanged', ROOT / R.PREPARED / 'attempt/model-result.json'),
        ('Explicit omission-only derivative', folder / 'offline-omission-only-derivative.json'),
        ('Minus-only + settled-accounting proof', folder / 'recovery.json'),
        ('Excluded cable and private result scope', folder / 'private-scene-result.json'),
    ]
    program = R.read(ROOT / R.PROGRAM)
    video = ROOT / program['source']['path']
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Private laptop hinge review</title><style>
body{{margin:0;background:#f7f8fb;color:#151923;font:16px system-ui,sans-serif}}main{{max-width:1260px;padding:36px;margin:auto}}h1{{font-size:38px;letter-spacing:-1.4px;margin:12px 0}}h2{{margin-top:40px}}p{{max-width:900px;line-height:1.6;color:#515d72}}.tag{{color:#3766ed;font-weight:700;font-size:13px;letter-spacing:.1em}}.notice{{border:1px solid #d7dfed;background:white;border-radius:16px;padding:22px;margin:24px 0}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}figure{{margin:0;background:white;border:1px solid #d7dfed;border-radius:12px;overflow:hidden}}img{{display:block;width:100%;height:auto}}figcaption{{padding:12px;font-size:13px;color:#515d72}}a{{color:#3766ed;display:inline-block;margin:6px 24px 6px 0}}video{{max-height:400px;width:min(100%,420px);background:#151923;border-radius:12px}}code{{font-size:14px}}@media(max-width:720px){{main{{padding:20px}}.grid{{grid-template-columns:1fr}}h1{{font-size:30px}}}}
</style><main><div class="tag">PRIVATE OWNER REVIEW · LOCAL CPU RESULT</div><h1>The hinge hypothesis survived. The geometric fit did not.</h1>
<p>A valid lid-hinge proposal was retained unchanged from an accounted Muse response. The entire cable relation was omitted because its citations do not overlap the base tracklet. This is an offline derivative, with zero new API or GPU calls.</p>
<div class="notice"><strong>Not an articulated laptop yet.</strong><p>The fitter found no native mesh surface supporting the cited hinge seam. Its strict result withholds the lid instead of inventing a pivot. The original failed run and all three generated models remain unchanged; this separate branch contains the base and lid only. The flexible cable is excluded.</p><code>Physical accuracy: unverified · publication: disabled · owner approval: absent</code></div>
<video controls preload="metadata" src="{url(video)}"></video>
{''.join(sections)}<section><h2>Exact artifacts</h2><p>Development diagnostics only. Raw final-test audit artifacts remain separate and were not used for fitting adjustments.</p>{''.join(f'<a href="{url(path)}">{html.escape(label)}</a>' for label,path in links)}</section></main></html>'''
    (output / 'index.html').write_text(page)
    width = 1200
    row_height = 290
    sheet = Image.new('RGB', (width, row_height * ((len(proof_images) + 1) // 2)), '#f7f8fb')
    draw = ImageDraw.Draw(sheet)
    for i, (label, image) in enumerate(proof_images):
        x, y = (i % 2) * 600, (i // 2) * row_height
        image.thumbnail((580, 248))
        sheet.paste(image, (x + 10, y + 32))
        draw.text((x + 10, y + 10), label[:85], fill='#151923')
    sheet.save(output / 'development-proof-sheet.jpg', quality=94)
    receipt = {'format': 'interaction-twin-private-laptop-hinge-owner-review', 'version': 1,
        'recovery': R.ref(folder / 'recovery.json', ROOT), 'scene_result': R.ref(folder / 'private-scene-result.json', ROOT),
        'review': R.ref(output / 'index.html', ROOT), 'proof_sheet': R.ref(output / 'development-proof-sheet.jpg', ROOT),
        'audit_manifests': [R.ref(folder / ('visual-' + name) / 'audit.json', ROOT) for name in audits],
        'displayed_development_sample_ids': {name: [s['sample_id'] for s in rows] for name, rows in development.items()},
        'final_test_panels_inspected': False, 'hinge_fit_succeeded': False, 'original_run_changed': False,
        'new_api_calls': 0, 'new_gpu_calls': 0, 'published': False, 'physical_accuracy_verified': False}
    R.write_new(output / 'review.json', receipt)
    return receipt


if __name__ == '__main__':
    print(json.dumps(build()), flush=True)
