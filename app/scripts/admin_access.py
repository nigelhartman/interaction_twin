#!/usr/bin/env python3
"""Configure Nigel's private admin access locally; never print a secret."""
import argparse
from getpass import getpass
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import admin_auth


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['bootstrap','set-password','reset-password'])
    parser.add_argument('--replace',action='store_true',help='Replace an unconsumed bootstrap code only')
    args=parser.parse_args(argv)
    try:
        if args.command=='bootstrap':
            path=admin_auth.create_bootstrap(ROOT,replace=args.replace)
            print('One-time setup code saved with owner-only permissions: '+str(path))
            print('Open that file yourself, then use /admin. The code is never printed here.')
        else:
            configured=admin_auth.AdminAuth(ROOT).configured()
            if args.command=='set-password' and configured:raise admin_auth.AuthError('Owner is configured; use reset-password',409)
            if args.command=='reset-password' and not configured:raise admin_auth.AuthError('Owner is not configured; use set-password',409)
            password=getpass('New owner password (at least 12 characters): ')
            if password!=getpass('Confirm password: '):raise admin_auth.AuthError('Passwords did not match',400)
            admin_auth.configure_password(ROOT,password,replace=configured)
            print('Owner password saved as scrypt hash. Existing sessions are invalidated.')
        return 0
    except (admin_auth.AuthError,OSError) as exc:
        print(str(exc),file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
