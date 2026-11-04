"""The public Newton starter is one bounded ZIP, never a general archive route."""
from email.message import Message
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from scripts import serve_workspace as S


class NewtonPublicDownloadBoundaryTests(unittest.TestCase):
    name = 'showcase/media/interaction-twin-newton-starter-20261104-v1.zip'

    def test_exact_allowlist_and_no_archive_extension_expansion(self):
        self.assertEqual(S.PUBLIC_PHYSICS_DOWNLOADS, {self.name})
        self.assertNotIn('.zip', S.PUBLIC_STATIC_EXTENSIONS)
        self.assertTrue(S.public_static(S.ROOT / self.name))
        for name in ['showcase/media/other.zip', self.name + '.bak',
                     self.name.replace('-v1.zip', '-v2.zip'),
                     'showcase/public/physics/other.zip', 'data/other.zip']:
            with self.subTest(name=name):
                self.assertFalse(S.public_static(S.ROOT / name))

    def test_missing_empty_and_oversize_exact_name_are_withheld(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            p = root / self.name
            self.assertFalse(S.public_static(p, root=root))
            p.parent.mkdir(parents=True)
            p.touch()
            self.assertFalse(S.public_static(p, root=root))
            for size, expected in [(1, True), (32 * 1024 * 1024, True),
                                   (32 * 1024 * 1024 + 1, False)]:
                with p.open('wb') as stream:
                    stream.truncate(size)
                self.assertEqual(S.public_static(p, root=root), expected)

    def test_neighbor_archives_and_symlink_substitution_are_withheld(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            p = root / self.name
            p.parent.mkdir(parents=True)
            neighbor = p.with_name('other.zip')
            neighbor.write_bytes(b'PK\x03\x04fixture')
            self.assertFalse(S.public_static(neighbor, root=root))
            p.symlink_to(neighbor.name)
            self.assertFalse(S.public_static(p, root=root))

    def request(self, route, *, method='HEAD', byte_range=None):
        h = S.Handler.__new__(S.Handler)
        h.path = route
        h.headers = Message()
        h.headers['Host'] = '127.0.0.1:8766'
        if byte_range:
            h.headers['Range'] = byte_range
        h.wfile = BytesIO()
        h.server = SimpleNamespace(admin_auth=None, public_asset_cache={})
        sent, errors = [], []
        h.send_response = lambda status: sent.append(('status', status))
        h.send_header = lambda key, value: sent.append((key, value))
        h.end_headers = lambda: None
        h.json = lambda status, value: errors.append((status, value))
        getattr(h, 'do_' + method)()
        return h, sent, errors

    def test_anonymous_head_and_range_serve_exact_zip_bytes_only(self):
        p = S.ROOT / self.name
        h, sent, errors = self.request('/' + self.name)
        self.assertEqual(errors, [])
        self.assertIn(('status', 200), sent)
        self.assertIn(('Content-Type', 'application/zip'), sent)
        self.assertIn(('Content-Length', str(p.stat().st_size)), sent)
        self.assertEqual(h.wfile.getvalue(), b'')
        h, sent, errors = self.request('/' + self.name, method='GET', byte_range='bytes=0-31')
        self.assertEqual(errors, [])
        self.assertIn(('status', 206), sent)
        self.assertIn(('Content-Range', f'bytes 0-31/{p.stat().st_size}'), sent)
        with p.open('rb') as stream:
            self.assertEqual(h.wfile.getvalue(), stream.read(32))
        for route in ['/showcase/media/other.zip', '/showcase/public/physics/other.zip',
                      '/showcase/media/interaction-twin-newton-starter-20261104-v2.zip']:
            _, sent, errors = self.request(route)
            self.assertEqual(sent, [])
            self.assertEqual(errors[0][0], 404)


if __name__ == '__main__':
    unittest.main()
