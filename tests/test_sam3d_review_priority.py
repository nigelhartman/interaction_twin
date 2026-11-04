"""Select source-bound native versions by program identity and real import time."""
import copy
import unittest

from sam3d_review_catalog import match_reviews


def candidate(name, mesh, date, *, version=1, source='a' * 64, component='part'):
    return {'component_id': component, 'source_sha256': source,
            'native_glb': {'sha256': mesh}, 'review_page': {'path': 'data/' + name + '/index.html'},
            'imported_at_utc': date, 'display_transport_version': version, 'selected': False}


class NativeReviewPriorityTests(unittest.TestCase):
    def test_exact_program_mesh_wins_over_newer_candidate_without_cross_source_or_part(self):
        old = candidate('sam3d-native-review-old', 'b' * 64, '2026-11-01T12:00:00+00:00')
        newer = candidate('sam3d-initial-native-review-new', 'c' * 64, '2026-11-03T12:00:00+00:00')
        other_source = candidate('z-other-source', 'b' * 64, '2026-11-05T12:00:00+00:00', source='f' * 64)
        other_part = candidate('z-other-part', 'b' * 64, '2026-11-05T12:00:00+00:00', component='missing')
        record = {'source': {'sha256': 'a' * 64}, 'components': [
            {'id': 'part', 'asset': {'sha256': 'b' * 64, 'backend': 'Meta SAM 3D Objects'}}]}
        originals = copy.deepcopy([old, newer, other_source, other_part])
        self.assertEqual(match_reviews([newer, old, other_part, other_source], record), [old])
        self.assertEqual([old, newer, other_source, other_part], originals)

    def test_import_utc_time_then_transport_version_wins_over_lexical_prefix(self):
        record = {'source': {'sha256': 'a' * 64}, 'components': [{'id': 'part', 'asset': None}]}
        old = candidate('z-sam3d-native-review-old', 'b' * 64, '2026-11-01T12:00:00+00:00', version=2)
        newest = candidate('a-sam3d-initial-native-review-new', 'c' * 64, '2026-11-03T12:00:00Z')
        transport_v2 = candidate('a-transport-v2', 'c' * 64, '2026-11-03T13:00:00+01:00', version=2)
        invalid = candidate('zzz-invalid-date', 'd' * 64, 'invalid', version=2)
        naive = candidate('zzz-naive-date', 'e' * 64, '2026-12-30T12:00:00', version=2)
        rows = [naive, transport_v2, old, invalid, newest]
        self.assertEqual(match_reviews(rows, record), [transport_v2])
        self.assertEqual(match_reviews(list(reversed(rows)), record), [transport_v2])


if __name__ == '__main__': unittest.main()
