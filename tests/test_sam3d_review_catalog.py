"""The generated library must bind native geometry and source time exactly."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import sam3d_review_catalog as C

ROOT = Path(__file__).resolve().parents[1]
REVIEW = 'data/sam3d-initial-native-review-20261102/jabra-base-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NativeDiscovery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.review = self.root / REVIEW
        self.review.mkdir(parents=True)
        self.manifest = json.loads((ROOT / REVIEW / 'review-manifest.json').read_text())
        self.receipt_path = self.root / self.manifest['import_receipt']['path']
        self.receipt_path.parent.mkdir(parents=True)
        for name in ['review-manifest.json', 'index.html', 'source-rgb.png', 'source-mask.png', 'candidate.glb']:
            shutil.copyfile(ROOT / REVIEW / name, self.review / name)
        for name in ['import.json', 'metadata.json', 'asset.glb']:
            shutil.copyfile(ROOT / self.receipt_path.parent.relative_to(self.root) / name, self.receipt_path.parent / name)

    def rewrite_manifest(self):
        (self.review / 'review-manifest.json').write_text(json.dumps(self.manifest))

    def rewrite_model(self, change):
        path = self.receipt_path.parent / 'metadata.json'
        doc = json.loads(path.read_text()); change(doc); path.write_text(json.dumps(doc))
        receipt = json.loads(self.receipt_path.read_text())
        for ref in receipt['import_files']:
            if ref['path'] == 'metadata.json': ref.update(sha256=sha(path), bytes=path.stat().st_size)
        self.receipt_path.write_text(json.dumps(receipt))
        self.manifest['import_receipt']['sha256'] = sha(self.receipt_path)
        self.rewrite_manifest()

    def test_real_source_and_unchanged_geometry_are_discovered(self):
        rows = C.discover_reviews(self.root)
        self.assertEqual([r['component_id'] for r in rows], ['case_base'])
        self.assertEqual(rows[0]['native_glb']['sha256'], self.manifest['native_glb_sha256'])
        self.assertFalse(rows[0]['selected']); self.assertFalse(rows[0]['simulation_ready'])

    def test_changed_timestamp_is_rejected_even_when_video_matches(self):
        self.manifest['source']['timestamp_s'] += 1
        self.rewrite_manifest()
        self.assertEqual(C.discover_reviews(self.root), [])

    def test_source_bound_candidate_cannot_be_claimed_as_selected(self):
        self.manifest['scope']['selected'] = True
        self.rewrite_manifest()
        self.assertEqual(C.discover_reviews(self.root), [])

    def test_changed_original_asset_invalidates_unchanged_preview_copy(self):
        with (self.receipt_path.parent / 'asset.glb').open('r+b') as stream: stream.write(b'bad!')
        self.assertEqual(C.discover_reviews(self.root), [])

    def test_manual_geometry_cannot_be_made_genuine_by_rehashing_metadata(self):
        self.rewrite_model(lambda d: d.update(manual_geometry_fallback=True))
        self.assertEqual(C.discover_reviews(self.root), [])

    def test_candidate_only_matches_its_exact_recording_and_component(self):
        record = {'source': {'sha256': self.manifest['source']['video_sha256']}, 'components': [{'id': 'case_base'}]}
        self.assertEqual(len(C.reviews_for_record(self.root, record)), 1)
        other = copy.deepcopy(record); other['source']['sha256'] = '0' * 64
        self.assertEqual(C.reviews_for_record(self.root, other), [])
        record['components'] = [{'id': 'case_lid'}]
        self.assertEqual(C.reviews_for_record(self.root, record), [])

    def test_hidden_or_symlink_artifacts_are_rejected(self):
        with self.assertRaises(ValueError): C._file(self.root, 'data/../secrets')
        image = self.review / 'source-rgb.png'; image.unlink(); image.symlink_to(ROOT / REVIEW / 'source-rgb.png')
        self.assertEqual(C.discover_reviews(self.root), [])
