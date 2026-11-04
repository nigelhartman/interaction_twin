#!/usr/bin/env python3
"""User-run owner setup: hidden Terminal password prompt, then private studio.

This launcher is for the owner to run interactively. It never chooses, prints,
exports or embeds a password, reads setup codes or starts inference.
"""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    import admin_auth
    from scripts.launch_workspace import main as launch
    if not admin_auth.AdminAuth(ROOT).configured():
        print('Create your owner password once. Input is hidden; choose at least 12 characters.',flush=True)
        result=subprocess.run([sys.executable,str(ROOT/'scripts/admin_access.py'),'set-password'],cwd=ROOT,check=False)
        if result.returncode:return result.returncode
    return launch(['--admin'])

if __name__=='__main__':raise SystemExit(main())
