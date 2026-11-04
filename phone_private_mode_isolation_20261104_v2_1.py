"""Additive UI-only isolation fix for the private three-way phone comparison.

No inference, model/scene changes, publication, old-bundle edits or server start.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil, zipfile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output/submission-polish-20261104-v1/phone-edge-prior-v4'
OLD, NEW = OUT / 'private-v2', OUT / 'private-v2-1'

def ref(path):
    raw = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}

def main():
    assert not NEW.exists()
    old_manifest = ref(OLD / 'manifest.json')
    shutil.copytree(OLD, NEW)
    path = NEW / 'review.js'
    script = path.read_text()
    script = script.replace("$('description-download').href=edgeMode?'data/edge-private-declaration.json':'data/descriptive-sidecar.json';$('test-report-link').href=edgeMode?'data/edge-fresh-tests.json':'data/new-test-report.json';", "$('description-download').href=originalMode?'data/baseline-scene.json':edgeMode?'data/edge-private-declaration.json':'data/descriptive-sidecar.json';$('description-download').textContent=originalMode?'Download original saved scene ↓':'Download description ↓';$('test-report-link').href=originalMode||edgeMode?'data/edge-fresh-tests.json':'data/new-test-report.json';$('test-report-link').textContent=originalMode?'Edge-prior comparison tests ↗':edgeMode?'Edge-prior fresh frame tests ↗':'Source-prism frame tests ↗';")
    marker = "$('coverage').textContent="
    assert marker in script
    script = script.replace(marker, "if(originalMode){$('appearance-tag').textContent='Preserved generated appearance';$('geometry-tag').textContent='Original native SAM 3D geometry';$('method-note').textContent='This mode displays the unchanged original SAM 3D phone and its original saved scene. The intersecting-slab defect is preserved for comparison; source-derived reconstruction claims apply only in the other modes.';$('validation').textContent='No repair is applied in this original mode. The linked edge-prior test report is comparison evidence about another candidate, not validation of this original model.';$('unknown-note').textContent='Original saved pose estimates and unknown states are shown unchanged. Camera, metric scale, connector contact, grasping and insertion remain unverified.';}else{$('appearance-tag').textContent='Actual front + back RGB';$('geometry-tag').textContent='One watertight source-derived body';} " + marker)
    path.write_text(script)
    page = NEW / 'index.html'
    html = page.read_text().replace('<span>Actual front + back RGB</span><span>One watertight body</span>', '<span id="appearance-tag">Verifying selected appearance…</span><span id="geometry-tag">Verifying selected geometry…</span>')
    html = html.replace('<p>This phone uses a flat-face', '<p id="method-note">This phone uses a flat-face')
    html = html.replace('>Coverage report</a>', '>Source-prism coverage comparison</a>')
    html = html.replace('>Prior comparison checks</a>', '>Source-prism prior comparison checks</a>')
    page.write_text(html)
    readme = NEW / 'README.txt'
    readme.write_text(readme.read_text() + '\nUI-only revision v2.1: original mode downloads its original scene, labels cross-model tests explicitly, and does not inherit source-prism geometry/validation claims. All copied scene/model/evidence bytes are identical to v2. No server was started.\n')
    manifest = json.loads((NEW / 'manifest.json').read_text())
    manifest['ui_version'] = 'private-v2-1-three-way-mode-isolation'
    manifest['created_at_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['previous_bundle_manifest'] = old_manifest
    manifest['mode_isolation'] = {'original_description': 'exact original saved scene', 'cross_model_tests_explicitly_labelled': True, 'all_scene_geometry_evidence_bytes_unchanged': True}
    manifest['implementation'] = ref(Path(__file__))
    for row in manifest['files']:
        raw = (NEW / row['local']).read_bytes()
        row.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    (NEW / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    changed = {'review.js', 'index.html', 'README.txt', 'manifest.json'}
    assert all((OLD / p.relative_to(NEW)).read_bytes() == p.read_bytes() for p in NEW.rglob('*') if p.is_file() and p.relative_to(NEW).as_posix() not in changed)
    assert ref(OLD / 'manifest.json') == old_manifest
    archive = OUT / 'Interaction-Twin-private-phone-three-way-v2-1.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(NEW.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(NEW).as_posix())
    receipt = {'format': 'interaction-twin-private-phone-ui-mode-isolation-revision', 'version': 1, 'previous_manifest': old_manifest, 'bundle_manifest': ref(NEW / 'manifest.json'), 'archive': ref(archive), 'changed_bundle_files': sorted(changed), 'unchanged_scene_geometry_evidence_bytes': True, 'original_bundle_unchanged': True, 'server_started': False, 'API_or_GPU_calls': 0, 'public_default_eligible': False, 'runner_command': ['python3', str(NEW / 'serve-review.py'), '--port', '8873']}
    (OUT / 'private-v2-1-build.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))

if __name__ == '__main__': main()
