import hashlib
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer

from scripts import serve_laptop_hinge_review_20261103_v2 as S


class PrivateReviewServer(unittest.TestCase):
    def test_exact_path_hash_and_symlink_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root / 'data').mkdir()
            p = root / 'data/file.json'
            p.write_bytes(b'{}')
            reference = S.R.ref(p, root)
            self.assertEqual(S.checked(reference, root), p)
            p.write_bytes(b'[]')
            with self.assertRaisesRegex(ValueError, 'changed'):
                S.checked(reference, root)
            (root / 'data/link.json').symlink_to(p)
            with self.assertRaises(ValueError):
                S.checked(dict(reference, path='data/link.json'), root)

    def test_loopback_byte_ranges_unknown_files_and_mutations(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root / 'data').mkdir()
            p = root / 'data/video.mp4'
            p.write_bytes(b'0123456789')
            routes = {'/files/data/video.mp4': S.R.ref(p, root)}
            server = ThreadingHTTPServer(('127.0.0.1', 0), S.handler(routes, root=root))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                origin = f'http://127.0.0.1:{server.server_port}'
                with urlopen(Request(origin + '/files/data/video.mp4', headers={'Range': 'bytes=2-5'})) as response:
                    self.assertEqual(response.status, 206)
                    self.assertEqual(response.read(), b'2345')
                    self.assertEqual(response.headers['Content-Range'], 'bytes 2-5/10')
                for headers in [{'Host': 'evil.example'}, {'Origin': 'https://evil.example'}, {'Origin': 'null'}]:
                    with self.subTest(headers=headers), self.assertRaises(HTTPError) as error:
                        urlopen(Request(origin + '/files/data/video.mp4', headers=headers))
                    self.assertEqual(error.exception.code, 403)
                with urlopen(Request(origin + '/files/data/video.mp4', headers={'Origin': origin})) as response:
                    self.assertEqual(response.read(), b'0123456789')
                for path, status in [('/files/.secrets/meta.env', 404), ('/files/data/other.json', 404), ('/files/data/%2e%2e/file', 400)]:
                    with self.subTest(path=path), self.assertRaises(HTTPError) as error:
                        urlopen(origin + path)
                    self.assertEqual(error.exception.code, status)
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(origin + '/files/data/video.mp4', headers={'Range': 'bytes=20-'}))
                self.assertEqual(error.exception.code, 416)
                p.write_bytes(b'abcdefghij')
                with self.assertRaises(HTTPError) as error:
                    urlopen(origin + '/files/data/video.mp4')
                self.assertEqual(error.exception.code, 409)
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(origin + '/files/data/video.mp4', method='POST', data=b''))
                self.assertEqual(error.exception.code, 501)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
