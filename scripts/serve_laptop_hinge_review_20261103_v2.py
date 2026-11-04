#!/usr/bin/env python3
"""Private laptop review with exact files and loopback Host/Origin fencing."""
import argparse
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import serve_laptop_hinge_review_20261103_v1 as V1

checked = V1.checked
preparation = V1.preparation
R = V1.R


def handler(routes, *, root=ROOT):
    Base = V1.handler(routes, root=root)
    class Handler(Base):
        def respond(self, head):
            port = self.server.server_port
            hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
            origins = {'http://' + host for host in hosts}
            if self.headers.get('Host') not in hosts or self.headers.get('Origin') not in {None, *origins}:
                self.send_error(403, 'Loopback owner review only')
                return
            super().respond(head)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--server-id', default='preview2')
    args = parser.parse_args()
    R.require(0 <= args.port <= 65535 and re.fullmatch(r'[a-z0-9-]{1,32}', args.server_id), 'Bounded loopback server identity required')
    manifest, routes = preparation()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(routes))
    receipt = {'format': 'interaction-twin-private-laptop-review-server', 'version': 2,
               'origin': f'http://127.0.0.1:{server.server_port}', 'routes': routes, 'private_manifest': manifest,
               'loopback_host_and_origin_guard': True, 'read_only': True,
               'implementation': R.ref(Path(__file__).resolve(), ROOT), 'base_implementation': R.ref(Path(V1.__file__).resolve(), ROOT),
               'new_api_calls': 0, 'new_gpu_calls': 0, 'published': False}
    destination = ROOT / 'output/new-captures-20261103' / ('laptop-hinge-server-' + args.server_id + '.json')
    R.write_new(destination, receipt)
    print(receipt['origin'], flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
